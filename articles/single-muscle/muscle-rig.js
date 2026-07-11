// muscle-rig.js
// ---------------------------------------------------------------------------
// A single Hill-type muscle with a compliant tendon, anchored at BOTH ends so
// the whole musculotendon unit (MTU) length is fixed — an isometric rig. The
// muscle fibre (contractile element, CE) can still shorten: as it does, the
// series-elastic tendon stretches by the same amount, so force rises while the
// overall length stays put. This is exactly what happens in a real isometric
// contraction — the fibres shorten a few percent and load the tendon.
//
// Built on motornet.js's CompliantTendonHillMuscle, driven standalone at a fixed
// geometry (constant MTU length, zero MTU velocity).
//
// Two afferent sensors are modelled, each with its own transport delay:
//   • spindle (Ia)  — senses fibre LENGTH and fibre VELOCITY
//   • Golgi tendon organ (Ib) — senses tendon FORCE
//
// A note on units. The library stores "muscle velocity" scaled by vmax (=10·l0),
// so it is 10× the true rate of fibre-length change. We instead report the
// physically honest fibre velocity — the actual time-derivative of the fibre
// length the simulation integrates — computed by finite difference, in mm/s.

import { CompliantTendonHillMuscle } from '../../lib/motornetjs/index.js';

export const DT = 0.001;   // 1 ms fine step for a single muscle

// One representative muscle. optimal fibre length 10 cm, tendon slack 20 cm,
// peak isometric force 500 N. MTU length is set so that at rest the fibre sits at
// its optimal length and the tendon at its slack length (both unloaded).
export const MUSCLE = {
  maxIsoForce: 500,
  optimalFibreLength: 0.10,   // l0 (m)
  tendonSlackLength: 0.20,    // (m)
  normalizedSlackFibre: 1.4,  // passive element slack (× l0)
};
export const MTU_LENGTH = MUSCLE.tendonSlackLength + MUSCLE.optimalFibreLength; // 0.30 m

// A fixed-size ring buffer of scalars, prefilled, for modelling afferent delays.
class Delay {
  constructor(size, init) { this.size = size; this.buf = new Array(size).fill(init); }
  push(step, v) { this.buf[step % this.size] = v; }
  get(step, delay) { const i = step - delay; return this.buf[(i < 0 ? 0 : i) % this.size]; }
}

export class MuscleRig {
  constructor(opts = {}) {
    this.mtu = opts.mtu ?? MTU_LENGTH;
    this.muscle = new CompliantTendonHillMuscle();
    this.muscle.build(DT, {
      max_isometric_force: [MUSCLE.maxIsoForce],
      tendon_length: [MUSCLE.tendonSlackLength],
      optimal_muscle_length: [MUSCLE.optimalFibreLength],
      normalized_slack_muscle_length: [MUSCLE.normalizedSlackFibre],
    });
    this.l0 = this.muscle.l0_ce[0];
    this.geom = [[this.mtu], [0], [0], [0]];   // [mtuLen, mtuVel, moment0, moment1]
    this.reset();
  }

  reset() {
    this.state = this.muscle.getInitialMuscleState(this.geom);
    this.step = 0;
    this.prevLen = this.state[1][0];
    this.vel = 0;                               // true fibre velocity (m/s), shortening < 0
    const SIZE = 256;                           // 256 ms of delay headroom
    this.lenBuf = new Delay(SIZE, this.state[1][0]);
    this.velBuf = new Delay(SIZE, 0);
    this.forceBuf = new Delay(SIZE, this.state[6][0]);
  }

  // Advance one step under the given excitation u ∈ [0,1].
  advance(u) {
    const uu = [Math.max(0, Math.min(1, u))];
    const deriv = this.muscle.ode(uu, this.state);
    this.state = this.muscle.integrate(DT, deriv, this.state, this.geom);
    const len = this.state[1][0];
    // true fibre velocity = d(fibre length)/dt (finite difference), lightly smoothed
    const vRaw = (len - this.prevLen) / DT;
    this.vel += 0.2 * (vRaw - this.vel);         // one-pole smoother (τ≈5 ms)
    this.prevLen = len;
    this.step += 1;
    this.lenBuf.push(this.step, len);
    this.velBuf.push(this.step, this.vel);
    this.forceBuf.push(this.step, this.state[6][0]);
  }

  // ---- true (physical) state --------------------------------------------
  get activation() { return this.state[0][0]; }
  get fibreLength() { return this.state[1][0]; }
  get tendonLength() { return this.mtu - this.state[1][0]; }
  get force() { return this.state[6][0]; }
  get fibreVelocity() { return this.vel; }       // m/s, shortening negative

  // ---- delayed afferent readings ----------------------------------------
  // Spindle (Ia): fibre length & velocity, delayed by `delaySteps`.
  spindle(delaySteps) {
    return {
      length: this.lenBuf.get(this.step, delaySteps),
      velocity: this.velBuf.get(this.step, delaySteps),
    };
  }
  // Golgi tendon organ (Ib): tendon force, delayed by `delaySteps`.
  gto(delaySteps) { return this.forceBuf.get(this.step, delaySteps); }
}

// A muscle-fibre SPEED controller. It regulates the fibre shortening velocity
// (positive = shortening) toward a reference set externally, using the spindle
// Ia velocity as the perceived signal — a proportional-integral output whose
// integral is the excitation command. In isometric conditions the fibre has only
// a few millimetres of travel (the tendon loads up), so a one-way setpoint stalls
// as force rises; a bidirectional reference (shorten / lengthen) drives the fibre
// back and forth within that travel indefinitely.
export class SpeedController {
  constructor(params = {}) {
    this.KP = params.KP ?? 3;          // proportional gain (per mm/s, ×1e-3)
    this.KI = params.KI ?? 18;         // integral gain (per mm/s·s, ×1e-3)
    this.SPINDLE_DELAY_MS = params.SPINDLE_DELAY_MS ?? 15;
    this.GTO_DELAY_MS = params.GTO_DELAY_MS ?? 15;
    this.refShorteningMmps = 0;        // reference: + shortening, − lengthening
    this.reset();
  }
  reset() { this.integ = 0; this.u = 0.02; }

  // Compute the next excitation from the rig's delayed spindle velocity.
  update(rig) {
    const dSteps = Math.round(this.SPINDLE_DELAY_MS / 1000 / DT);
    const vShortMmps = -rig.spindle(dSteps).velocity * 1000;   // mm/s, shortening +
    const err = this.refShorteningMmps - vShortMmps;
    this.integ += err * DT;
    // anti-windup: clamp the integral to the range that keeps u in [0,1]
    this.integ = Math.max(-1 / (this.KI * 1e-3), Math.min(1 / (this.KI * 1e-3), this.integ));
    this.u = Math.max(0, Math.min(1, this.KP * 1e-3 * err + this.KI * 1e-3 * this.integ));
    return this.u;
  }
}
