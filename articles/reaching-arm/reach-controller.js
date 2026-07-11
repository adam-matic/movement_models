// reach-controller.js
// ---------------------------------------------------------------------------
// A JavaScript port of the "current best arm" controller from the musclesim
// project (arm2dof/gto_reaching_2dof.py). It drives motornet.js's
// CompliantTendonArm26 (6 muscles, 2 joints) with a three-level, model-free
// Perceptual-Control-Theory cascade — no inverse kinematics, no inverse
// dynamics:
//
//   L3  task POSITION (reach r, point phi), perceived from joint angles by
//       forward kinematics, commands a task FORCE (task-space impedance):
//           Q_r*   = KP_R  *(r*  - r)   - KD_R  * rdot
//           Q_phi* = KP_PHI*(phi*- phi) - KD_PHI* phidot
//   L2  task FORCE (from the 6 GTO tendon-tension sensors): mapped to muscle
//       activation-rate through a fixed distribution matrix W.
//   L1  muscle activation integrator drives the 6 muscles.
//
// The arm is FREE (it moves); the motion supplies the inner force loop with the
// velocity damping it lacks when clamped. Perception is forward kinematics + GTO
// force only, all through transport delays. This continuous-time version keeps
// ring buffers for the sensory delays and ramps the commanded reference toward
// the target with a bounded-rate reference governor, exactly as the deployed
// controller does (a literal step overshoots).
//
// Deployed parameter values are taken verbatim from the musclesim repo:
//   gains best_reaching_2dof.npy, distribution map best_reaching_W_joint_2dof.npy.

import { CompliantTendonArm26 } from '../../lib/motornetjs/index.js';

export const M = 6;                 // muscles
export const DT = 0.002;            // 2 ms control/plant step (Euler)
export const Q1_0 = 0.70, Q2_0 = 1.20;  // start posture (shoulder, elbow) [rad]
export const MUSCLE_NAMES = ['pec', 'deltoid', 'brachiorad', 'tricepslat', 'biceps', 'tricepslong'];

// Deployed default gains (best_reaching_2dof.npy) and constants.
export const DEFAULTS = {
  KP_R: 232.0, KD_R: 41.57,        // reach impedance: N/m, N/(m/s)
  KP_PHI: 12.87, KD_PHI: 2.57,     // point impedance: Nm/rad, Nm/(rad/s)
  LEAK: 0.05,                      // L2 activation leak
  K_CO: 0.012,                     // co-contraction (stiffness) gain
  C_REF: 0.0,                      // co-contraction reference (total tendon force, N)
  PROP_DELAY_MS: 20,               // joint-angle proprioception (Ia)
  GTO_DELAY_MS: 20,                // tendon-force feedback (Ib)
  SPN_DELAY_MS: 20,                // moment-arm (geometry) sensing
};

// Deployed distribution map W (best_reaching_W_joint_2dof.npy), 6x2:
// column 0 weights the Reach error, column 1 the Point error. Row order matches
// MUSCLE_NAMES.
export const W_JOINT = [
  [0.030982, 0.429422],   // pec
  [-0.087651, -0.18036],  // deltoid
  [-0.226026, 0.082392],  // brachiorad
  [0.021621, -0.020362],  // tricepslat
  [-0.053444, 0.477113],  // biceps
  [0.018714, -0.269611],  // tricepslong
];

// Forward kinematics of the 2-link arm, plus the geometry terms the controller
// needs. Returns {r, phi, rdot, phidot, drdq2, alpha_p}.
export function taskFk(q1, q2, q1d, q2d, L1, L2) {
  const c2 = Math.cos(q2), s2 = Math.sin(q2);
  const r = Math.sqrt(L1 * L1 + L2 * L2 + 2 * L1 * L2 * c2);
  const alpha = Math.atan2(L2 * s2, L1 + L2 * c2);
  const phi = q1 + alpha;
  const drdq2 = -L1 * L2 * s2 / r;
  const alpha_p = L2 * (L1 * c2 + L2) / (r * r);
  return { r, phi, rdot: drdq2 * q2d, phidot: q1d + alpha_p * q2d, drdq2, alpha_p };
}

// Geometric inverse kinematics (r, phi) -> (q1, q2), elbow-up branch. Only used
// to convert a clicked Cartesian target into a task reference; the controller
// itself never inverts (r,phi)->q online.
export function taskIk(r, phi, L1, L2) {
  const c2 = Math.min(1, Math.max(-1, (r * r - L1 * L1 - L2 * L2) / (2 * L1 * L2)));
  const q2 = Math.acos(c2);
  const alpha = Math.atan2(L2 * Math.sin(q2), L1 + L2 * Math.cos(q2));
  const q1 = phi - alpha;
  return { q1, q2 };
}

// A small fixed-size ring buffer of number-arrays, prefilled with a copy of the
// initial value so delayed reads during warm-up return the rest state.
class Delay {
  constructor(size, init) {
    this.size = size;
    this.buf = new Array(size);
    for (let i = 0; i < size; i++) this.buf[i] = init.slice();
  }
  push(step, value) { this.buf[step % this.size] = value; }
  get(step, delay) {
    const idx = step - delay;
    return this.buf[(idx < 0 ? 0 : idx) % this.size];
  }
}

export class ReachController {
  constructor(params = {}) {
    this.p = { ...DEFAULTS, W: W_JOINT.map((r) => r.slice()), ...params };
    this.arm = new CompliantTendonArm26({ timestep: DT, integrationMethod: 'euler' });
    this.L1 = this.arm.skeleton.L1;
    this.L2 = this.arm.skeleton.L2;
    this.reset();
  }

  reset() {
    this.arm.reset({ jointState: [Q1_0, Q2_0] });
    this.step = 0;
    this.a = new Array(M).fill(0);          // muscle activations
    const g0 = taskFk(Q1_0, Q2_0, 0, 0, this.L1, this.L2);
    this.r0 = g0.r; this.phi0 = g0.phi;
    // commanded reference (governed) and the target the user asks for
    this.rRef = g0.r; this.phiRef = g0.phi;
    this.rTarget = g0.r; this.phiTarget = g0.phi;
    // delay buffers (max 128 steps = 256 ms headroom)
    const SIZE = 128;
    const geom = this.arm.states.geometry;
    this.qBuf = new Delay(SIZE, [Q1_0, Q2_0, 0, 0]);
    this.fBuf = new Delay(SIZE, new Array(M).fill(0));
    this.rBuf = new Delay(SIZE, [geom[2].slice(), geom[3].slice()].flat());
    // last-recorded diagnostics
    this.rec = { r: g0.r, phi: g0.phi, x: this.arm.states.fingertip[0], y: this.arm.states.fingertip[1],
      QrCmd: 0, QphiCmd: 0, force: new Array(M).fill(0), cocon: 0 };
  }

  setTargetTask(r, phi) { this.rTarget = r; this.phiTarget = phi; }
  setTargetCartesian(x, y) {
    const r = Math.hypot(x, y);
    const phi = Math.atan2(y, x);
    // clamp reach to the reachable annulus
    const rMin = Math.abs(this.L1 - this.L2) + 0.02;
    const rMax = (this.L1 + this.L2) - 0.02;
    this.setTargetTask(Math.min(rMax, Math.max(rMin, r)), phi);
  }

  // Advance one control step (DT). Mirrors run_reach's inner loop, default path.
  advance() {
    const p = this.p;
    const propD = Math.round(p.PROP_DELAY_MS / 1000 / DT);
    const gtoD = Math.round(p.GTO_DELAY_MS / 1000 / DT);
    const spnD = Math.round(p.SPN_DELAY_MS / 1000 / DT);
    const s = this.step;

    // reference governor: ramp the commanded reference toward the target at a
    // bounded rate (a literal step overshoots — see CURRENT_BEST_ARM.md).
    const R_RATE = 0.10;        // m/s   reach ramp rate
    const PHI_RATE = 0.9;       // rad/s point ramp rate
    this.rRef += Math.max(-R_RATE * DT, Math.min(R_RATE * DT, this.rTarget - this.rRef));
    this.phiRef += Math.max(-PHI_RATE * DT, Math.min(PHI_RATE * DT, this.phiTarget - this.phiRef));

    // delayed perception
    const qd = this.qBuf.get(s, propD);
    const Fd = this.fBuf.get(s, gtoD);
    const Rflat = this.rBuf.get(s, spnD);
    const Rd = [Rflat.slice(0, M), Rflat.slice(M, 2 * M)];    // (2,6) moment arms
    const g = taskFk(qd[0], qd[1], qd[2], qd[3], this.L1, this.L2);

    // L3: task-space impedance -> task force command
    const QrCmd = p.KP_R * (this.rRef - g.r) - p.KD_R * g.rdot;
    const QphiCmd = p.KP_PHI * (this.phiRef - g.phi) - p.KD_PHI * g.phidot;

    // L2: perceive joint torque from the GTOs, map UP into task space
    let tau0 = 0, tau1 = 0;
    for (let m = 0; m < M; m++) { tau0 -= Rd[0][m] * Fd[m]; tau1 -= Rd[1][m] * Fd[m]; }
    const den = Math.abs(g.drdq2) > 1e-3 ? g.drdq2 : Math.sign(g.drdq2 + 1e-12) * 1e-3;
    const QrPerc = (-g.alpha_p * tau0 + tau1) / den;
    const QphiPerc = tau0;

    const errQr = QrCmd - QrPerc;
    const errQphi = QphiCmd - QphiPerc;

    // (3) static distribution map: da = W * E_task - LEAK * a  (+ co-contraction)
    const W = p.W;
    let sumF = 0; for (let m = 0; m < M; m++) sumF += Fd[m];
    const coDrive = p.C_REF > 0 ? p.K_CO * (p.C_REF - sumF) : 0;
    for (let m = 0; m < M; m++) {
      let da = W[m][0] * errQr + W[m][1] * errQphi - p.LEAK * this.a[m] + coDrive;
      this.a[m] = Math.min(1, Math.max(0, this.a[m] + da * DT));
    }

    // L1: step the free arm
    this.arm.step(this.a);

    // record & buffer
    const js = this.arm.states.joint;
    const ms = this.arm.states.muscle;
    const force = ms[6].slice();
    const geom = this.arm.states.geometry;
    this.step = s + 1;
    this.qBuf.push(this.step, js.slice(0, 4));
    this.fBuf.push(this.step, force);
    this.rBuf.push(this.step, [geom[2].slice(), geom[3].slice()].flat());

    const gt = taskFk(js[0], js[1], js[2], js[3], this.L1, this.L2);
    const ft = this.arm.states.fingertip;
    let cocon = 0; for (let m = 0; m < M; m++) cocon += force[m];
    this.rec = { r: gt.r, phi: gt.phi, x: ft[0], y: ft[1], QrCmd, QphiCmd, force, cocon };
    return this.rec;
  }

  // Joint angles for rendering.
  get joints() { return this.arm.states.joint; }
  get fingertip() { return this.arm.states.fingertip; }
}
