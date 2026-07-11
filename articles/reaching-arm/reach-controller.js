// reach-controller.js
// ---------------------------------------------------------------------------
// A JavaScript port of the "current best arm" controller from the musclesim
// project (arm2dof/gto_reaching_2dof.py), reduced to a FOUR-muscle arm: the two
// bi-articular muscles (biceps, triceps longus) have been removed, leaving four
// mono-articular muscles — pectoralis and deltoid at the shoulder,
// brachioradialis and triceps lateralis at the elbow. Each joint therefore has a
// single antagonist pair.
//
// It drives motornet.js's compliant-tendon 2-joint arm with a three-level,
// model-free Perceptual-Control-Theory cascade — no inverse kinematics, no
// inverse dynamics:
//
//   L3  task POSITION (reach r, point phi), perceived from joint angles by
//       forward kinematics, commands a task FORCE (task-space impedance):
//           Q_r*   = KP_R  *(r*  - r)   - KD_R  * rdot
//           Q_phi* = KP_PHI*(phi*- phi) - KD_PHI* phidot
//   L2  task FORCE (from the 4 GTO tendon-tension sensors): mapped to muscle
//       activation-rate through a fixed distribution matrix W (now 4x2).
//   L1  muscle activation integrator drives the 4 muscles.
//
// The arm is FREE (it moves); the motion supplies the inner force loop with the
// velocity damping it lacks when clamped. Perception is forward kinematics + GTO
// force only, all through transport delays.
//
// The gains and the 4x2 distribution matrix W below were produced by the
// in-browser evolutionary optimiser (reach-evolve.js), which searches for
// SMOOTH and ACCURATE reaching on this four-muscle arm.

import { CompliantTendonArm26 } from '../../lib/motornetjs/index.js';

export const M = 4;                 // muscles (bi-articular pair removed)
export const DT = 0.002;            // 2 ms control/plant step (Euler)
export const Q1_0 = 0.70, Q2_0 = 1.20;  // start posture (shoulder, elbow) [rad]
export const MUSCLE_NAMES = ['pec', 'deltoid', 'brachiorad', 'tricepslat'];

// A four-muscle compliant-tendon arm. It reuses CompliantTendonArm26's skeleton,
// muscle model and moment-arm polynomials but keeps only the four mono-articular
// muscles (drops biceps and triceps longus, columns 4 and 5).
// Shoulder upper limit. The stock Arm26 caps the shoulder at 135°; a workspace
// analysis (rasterised forward kinematics) shows raising it to 160° enlarges the
// reachable area by ~18% (0.466 -> 0.552 m^2) while both shoulder muscles stay
// inside their active force-length band (pec 0.56, deltoid 1.45 normalized at the
// limit; 180° would push the pectoralis to 0.48, below the ascending limb).
export const SHOULDER_MAX_DEG = 160;
export const ELBOW_MAX_DEG = 155;

export class CompliantTendonArm24 extends CompliantTendonArm26 {
  constructor(opts = {}) {
    const deg = Math.PI / 180;
    if (!opts.posUpperBound) {
      opts = { ...opts, posUpperBound: [SHOULDER_MAX_DEG * deg, ELBOW_MAX_DEG * deg] };
    }
    super(opts);
    this.nMuscles = M;
    this.inputDim = M;
    this.muscleName = ['pectoralis', 'deltoid', 'brachioradialis', 'tricepslat'];
    // rebuild the muscle vector with the four mono-articular columns only
    const params = { max_isometric_force: [838, 1207, 1422, 1549],
      tendon_length: [0.070, 0.070, 0.172, 0.187],
      optimal_muscle_length: [0.134, 0.140, 0.092, 0.093] };
    for (const key of this.muscle.toBuildKeys) {
      if (!(key in params) && key in this.muscle.toBuildDefaults) params[key] = this.muscle.toBuildDefaults[key];
    }
    this.muscle.build(this.dt, params);
    // keep the first four columns of the moment-arm polynomials
    this.a0 = this.a0.slice(0, M);
    this.a1 = [this.a1[0].slice(0, M), this.a1[1].slice(0, M)];
    this.a2 = [this.a2[0].slice(0, M), this.a2[1].slice(0, M)];
  }

  // Same moment-arm polynomial evaluation as RigidTendonArm26, generalised to
  // this.nMuscles instead of a hard-coded 6.
  _getGeometry(jointState) {
    const n = this.nMuscles;
    const pos = [jointState[0] - this.a3[0], jointState[1] - this.a3[1]];
    const vel = [jointState[2], jointState[3]];
    const mtuLen = new Array(n);
    const mtuVel = new Array(n);
    const moment = [new Array(n), new Array(n)];
    for (let m = 0; m < n; m++) {
      let len = this.a0[m];
      let velSum = 0;
      for (let k = 0; k < 2; k++) {
        const ma = pos[k] * this.a2[k][m] * 2 + this.a1[k][m];
        moment[k][m] = ma;
        len += (this.a1[k][m] + pos[k] * this.a2[k][m]) * pos[k];
        velSum += vel[k] * ma;
      }
      mtuLen[m] = len;
      mtuVel[m] = velSum;
    }
    return [mtuLen, mtuVel, ...moment];
  }
}

// Deployed default gains and constants. The four impedance gains and the 4x2
// distribution matrix W (below) come from reach-evolve.js — see EVOLVED_PARAMS.
export const DEFAULTS = {
  KP_R: 131.61, KD_R: 44.71,       // reach impedance: N/m, N/(m/s)
  KP_PHI: 4.54, KD_PHI: 2.27,      // point impedance: Nm/rad, Nm/(rad/s)
  LEAK: 0.05,                      // L2 activation leak
  K_CO: 0.012,                     // co-contraction (stiffness) gain
  C_REF: 0.0,                      // co-contraction reference (total tendon force, N)
  PROP_DELAY_MS: 20,               // joint-angle proprioception (Ia)
  GTO_DELAY_MS: 20,                // tendon-force feedback (Ib)
  SPN_DELAY_MS: 20,                // moment-arm (geometry) sensing
};

// Distribution map W (4x2): column 0 weights the Reach error, column 1 the Point
// error. Row order matches MUSCLE_NAMES. These gains and this W were produced by
// the evolutionary optimiser (reach-evolve.js) searching for smooth, accurate
// reaching across the workspace on the four-muscle arm — workspace-grid settled
// accuracy: mean ~0.5 cm, max ~2 cm.
export const W_JOINT = [
  [0.011479, 0.293859],   // pec
  [-0.094605, -0.231490], // deltoid
  [-0.076504, 0.077893],  // brachiorad
  [0.066731, 0.058669],   // tricepslat
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
    this.arm = new CompliantTendonArm24({ timestep: DT, integrationMethod: 'euler' });
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
    const Rd = [Rflat.slice(0, M), Rflat.slice(M, 2 * M)];    // (2,4) moment arms
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
