// Muscle models: a JavaScript port of motornet/muscle.py.
//
// A single Muscle object holds every muscle of an effector in vectorized form.
// State is stored as rows: state[i] is an array over muscles for the i-th state
// variable (the batch dimension of the Python implementation is dropped). The
// geometry state passed in has row 0 = musculotendon length, row 1 =
// musculotendon velocity, rows 2.. = moment arms.

// Broadcast a scalar / length-1 array to a length-n array.
function bc(x, n) {
  if (Array.isArray(x)) {
    if (x.length === 1) return new Array(n).fill(x[0]);
    if (x.length !== n) throw new Error(`expected scalar or length ${n}, got ${x.length}`);
    return x.slice();
  }
  return new Array(n).fill(x);
}

const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);

export class Muscle {
  constructor({ inputDim = 1, outputDim = 1, minActivation = 0, tauActivation = 0.015, tauDeactivation = 0.05 } = {}) {
    this.inputDim = inputDim;
    this.outputDim = outputDim;
    this.minActivation = minActivation;
    this.tauActivation = tauActivation;
    this.tauDeactivation = tauDeactivation;
    this.stateName = [];
    this.nMuscles = null;
    this.dt = null;
    this.built = false;
    // names of build parameters this muscle requires, and their defaults
    this.toBuildKeys = ['max_isometric_force'];
    this.toBuildDefaults = {};
  }

  clipActivationScalar(a) { return clamp(a, this.minActivation, 1); }
  clipActivation(arr) { return arr.map((a) => this.clipActivationScalar(a)); }

  // dActivation/dt, following Thelen (2003) eq. 1-2 (also MuJoCo's muscle model).
  activationOde(action, activation) {
    const a = this.clipActivation(action);
    const act = this.clipActivation(activation);
    return act.map((cur, m) => {
      const tmp = 0.5 + 1.5 * cur;
      const tau = a[m] > cur ? this.tauActivation * tmp : this.tauDeactivation / tmp;
      return (a[m] - cur) / tau;
    });
  }

  // Default ODE: only the activation evolves. Returns a single state-derivative row.
  ode(action, muscleState) {
    return [this.activationOde(action, muscleState[0])];
  }

  build(_timestep, _params) { throw new Error('not implemented'); }
  getInitialMuscleState(_geometryState) { throw new Error('not implemented'); }
  integrate(_dt, _stateDerivative, _muscleState, _geometryState) { throw new Error('not implemented'); }
}

export class ReluMuscle extends Muscle {
  constructor(opts = {}) {
    super(opts);
    this.name = 'ReluMuscle';
    this.stateName = ['activation', 'muscle length', 'muscle velocity', 'force'];
    this.stateDim = this.stateName.length;
    this.toBuildKeys = ['max_isometric_force'];
  }

  build(timestep, params) {
    this.nMuscles = (Array.isArray(params.max_isometric_force) ? params.max_isometric_force : [params.max_isometric_force]).length;
    this.maxIsoForce = bc(params.max_isometric_force, this.nMuscles);
    this.dt = timestep;
    const n = this.nMuscles;
    this.vmax = new Array(n).fill(1);
    this.l0_se = new Array(n).fill(1);
    this.l0_ce = new Array(n).fill(1);
    this.l0_pe = new Array(n).fill(1);
    this.built = true;
  }

  integrate(dt, stateDerivative, muscleState, geometryState) {
    const activation = this.clipActivation(muscleState[0].map((a, m) => a + stateDerivative[0][m] * dt));
    const force = activation.map((a, m) => a * this.maxIsoForce[m]);
    return [activation, geometryState[0].slice(), geometryState[1].slice(), force];
  }

  getInitialMuscleState(geometryState) {
    const n = this.nMuscles;
    const activation0 = new Array(n).fill(this.minActivation);
    const force0 = new Array(n).fill(0);
    return [activation0, geometryState[0].slice(), geometryState[1].slice(), force0];
  }
}

export class MujocoHillMuscle extends Muscle {
  constructor({ minActivation = 0, passiveForces = 1, tauActivation = 0.01, tauDeactivation = 0.04, ...rest } = {}) {
    super({ minActivation, tauActivation, tauDeactivation, ...rest });
    this.name = 'MujocoHillMuscle';
    this.stateName = ['activation', 'muscle length', 'muscle velocity', 'force-length PE', 'force-length CE', 'force-velocity CE', 'force'];
    this.stateDim = this.stateName.length;
    this.passiveForces = passiveForces;
    this.toBuildKeys = ['max_isometric_force', 'tendon_length', 'optimal_muscle_length', 'normalized_slack_muscle_length', 'lmin', 'lmax', 'vmax', 'fvmax'];
    this.toBuildDefaults = { normalized_slack_muscle_length: 1.3, lmin: 0.5, lmax: 1.6, vmax: 1.5, fvmax: 1.2 };
  }

  build(timestep, params) {
    const n = (Array.isArray(params.tendon_length) ? params.tendon_length : [params.tendon_length]).length;
    this.nMuscles = n;
    this.maxIsoForce = bc(params.max_isometric_force, n);
    this.l0_pe = bc(params.normalized_slack_muscle_length, n);
    this.l0_ce = bc(params.optimal_muscle_length, n);
    this.l0_se = bc(params.tendon_length, n);
    this.lmin = bc(params.lmin, n);
    this.lmax = bc(params.lmax, n);
    this.vmax = bc(params.vmax, n);
    this.fvmax = bc(params.fvmax, n);
    this.dt = timestep;
    // derived
    this.b = this.lmax.map((v) => 0.5 * (1 + v));
    this.c = this.fvmax.map((v) => v - 1);
    this.p1 = this.b.map((v) => v - 1);
    this.p2 = this.l0_pe.map((v) => 0.25 * v);
    this.mid = this.lmin.map((v) => 0.5 * (v + 0.95));
    this.built = true;
  }

  _bump(L, mid, lmax, m) {
    const lmin = this.lmin[m];
    const left = 0.5 * (lmin + mid);
    const right = 0.5 * (mid + lmax);
    if (L <= lmin || L >= lmax) return 0;
    let x;
    if (L < left) x = (L - lmin) / (left - lmin);
    else if (L < mid) x = (mid - L) / (mid - left);
    else if (L < right) x = (L - mid) / (right - mid);
    else x = (lmax - L) / (lmax - right);
    const pfivexx = 0.5 * x * x;
    if (L < left) return pfivexx;
    if (L < mid) return 1 - pfivexx;
    if (L < right) return 1 - pfivexx;
    return pfivexx;
  }

  integrate(dt, stateDerivative, muscleState, geometryState) {
    const n = this.nMuscles;
    const activation = this.clipActivation(muscleState[0].map((a, m) => a + stateDerivative[0][m] * dt));
    const rows = [activation, new Array(n), new Array(n), new Array(n), new Array(n), new Array(n), new Array(n)];
    for (let m = 0; m < n; m++) {
      const mtuLen = geometryState[0][m];
      const muscleLen = Math.max(mtuLen - this.l0_se[m], 0.001) / this.l0_ce[m];
      const muscleVel = geometryState[1][m] / this.vmax[m];
      const b = this.b[m];
      const p1 = this.p1[m];
      const p2 = this.p2[m];
      const fvmax = this.fvmax[m];
      const cc = this.c[m];

      let x;
      if (muscleLen <= 1) x = 0;
      else if (muscleLen <= b) x = (muscleLen - 1) / p1;
      else x = (muscleLen - b) / p1;

      let flpe;
      if (muscleLen <= 1) flpe = 0;
      else if (muscleLen <= b) flpe = p2 * x ** 3;
      else flpe = p2 * (1 + 3 * x);

      const flce = this._bump(muscleLen, 1, this.lmax[m], m) + 0.15 * this._bump(muscleLen, this.mid[m], 0.95, m);

      let fvce;
      if (muscleVel <= -1) fvce = 0;
      else if (muscleVel <= 0) fvce = (muscleVel + 1) * (muscleVel + 1);
      else if (muscleVel <= cc) fvce = fvmax - ((cc - muscleVel) * (cc - muscleVel)) / cc;
      else fvce = fvmax;

      const force = (activation[m] * flce * fvce + this.passiveForces * flpe) * this.maxIsoForce[m];
      rows[1][m] = muscleLen * this.l0_ce[m];
      rows[2][m] = muscleVel * this.vmax[m];
      rows[3][m] = flpe;
      rows[4][m] = flce;
      rows[5][m] = fvce;
      rows[6][m] = force;
    }
    return rows;
  }

  getInitialMuscleState(geometryState) {
    const n = this.nMuscles;
    const muscleState = [new Array(n).fill(this.minActivation)];
    const deriv = [new Array(n).fill(0)];
    return this.integrate(this.dt, deriv, muscleState, geometryState);
  }
}

export class RigidTendonHillMuscle extends Muscle {
  constructor({ minActivation = 0.001, ...rest } = {}) {
    super({ minActivation, ...rest });
    this.name = 'RigidTendonHillMuscle';
    this.stateName = ['activation', 'muscle length', 'muscle velocity', 'force-length PE', 'force-length CE', 'force-velocity CE', 'force'];
    this.stateDim = this.stateName.length;

    this.s_as = 0.001;
    this.f_iso_n_den = 0.66 ** 2;
    this.k_se = 1 / 0.04 ** 2;
    this.q_crit = 0.3;
    this.min_flce = 0.01;

    this.toBuildKeys = ['max_isometric_force', 'tendon_length', 'optimal_muscle_length', 'normalized_slack_muscle_length'];
    this.toBuildDefaults = { normalized_slack_muscle_length: 1.4 };
  }

  build(timestep, params) {
    const n = (Array.isArray(params.tendon_length) ? params.tendon_length : [params.tendon_length]).length;
    this.nMuscles = n;
    this.dt = timestep;
    this.maxIsoForce = bc(params.max_isometric_force, n);
    this.l0_ce = bc(params.optimal_muscle_length, n);
    this.l0_se = bc(params.tendon_length, n);
    const nsl = bc(params.normalized_slack_muscle_length, n);
    this.l0_pe = nsl.map((v, m) => v * this.l0_ce[m]);
    this.k_pe = this.l0_pe.map((v, m) => 1 / (1.66 - v / this.l0_ce[m]) ** 2);
    this.musculotendon_slack_len = this.l0_pe.map((v, m) => v + this.l0_se[m]);
    this.vmax = this.l0_ce.map((v) => 10 * v);
    this.built = true;
  }

  integrate(dt, stateDerivative, muscleState, geometryState) {
    const n = this.nMuscles;
    const activation = this.clipActivation(muscleState[0].map((a, m) => a + stateDerivative[0][m] * dt));
    const rows = [activation, new Array(n), new Array(n), new Array(n), new Array(n), new Array(n), new Array(n)];
    for (let m = 0; m < n; m++) {
      const mtuLen = geometryState[0][m];
      const muscleVel = geometryState[1][m];
      const muscleLen = Math.max(mtuLen - this.l0_se[m], 0);
      const muscleStrain = Math.max((muscleLen - this.l0_pe[m]) / this.l0_ce[m], 0);
      const muscleLenN = muscleLen / this.l0_ce[m];
      const muscleVelN = muscleVel / this.vmax[m];
      const a = activation[m];

      const flpe = this.k_pe[m] * muscleStrain ** 2;
      const flce = Math.max(1 + (-(muscleLenN ** 2) + 2 * muscleLenN - 1) / this.f_iso_n_den, this.min_flce);

      const a_rel_st = muscleLenN > 1 ? 0.41 * flce : 0.41;
      const b_rel_st = a < this.q_crit
        ? 5.2 * (1 - 0.9 * ((a - this.q_crit) / (5e-3 - this.q_crit))) ** 2
        : 5.2;
      const dfdvcon0 = (a * (flce + a_rel_st)) / b_rel_st;
      const f_x_a = flce * a;

      const tmp_p_nom = f_x_a * 0.5;
      const tmp_p_den = this.s_as - dfdvcon0 * 2;
      const p1 = -tmp_p_nom / tmp_p_den;
      const p2 = tmp_p_nom ** 2 / tmp_p_den;
      const p3 = -1.5 * f_x_a;

      let nom; let den;
      if (muscleVelN < 0) {
        nom = muscleVelN * a * a_rel_st + f_x_a * b_rel_st;
        den = b_rel_st - muscleVelN;
      } else {
        nom = -p1 * p3 + p1 * this.s_as * muscleVelN + p2 - p3 * muscleVelN + this.s_as * muscleVelN ** 2;
        den = p1 + muscleVelN;
      }
      const activeForce = Math.max(nom / den, 0);
      const force = (activeForce + flpe) * this.maxIsoForce[m];

      rows[1][m] = muscleLen;
      rows[2][m] = muscleVel;
      rows[3][m] = flpe;
      rows[4][m] = flce;
      rows[5][m] = activeForce;
      rows[6][m] = force;
    }
    return rows;
  }

  getInitialMuscleState(geometryState) {
    const n = this.nMuscles;
    return this.integrate(this.dt, [new Array(n).fill(0)], [new Array(n).fill(this.minActivation)], geometryState);
  }
}

export class RigidTendonHillMuscleThelen extends Muscle {
  constructor({ minActivation = 0.001, ...rest } = {}) {
    super({ minActivation, ...rest });
    this.name = 'RigidTendonHillMuscleThelen';
    this.stateName = ['activation', 'muscle length', 'muscle velocity', 'force-length PE', 'force-length CE', 'force-velocity CE', 'force'];
    this.stateDim = this.stateName.length;

    this.pe_k = 5;
    this.pe_1 = this.pe_k / 0.6;
    this.pe_den = Math.exp(this.pe_k) - 1;
    this.ce_gamma = 0.45;
    this.ce_Af = 0.25;
    this.ce_fmlen = 1.4;

    this.toBuildKeys = ['max_isometric_force', 'tendon_length', 'optimal_muscle_length', 'normalized_slack_muscle_length'];
    this.toBuildDefaults = { normalized_slack_muscle_length: 1 };
  }

  build(timestep, params) {
    const n = (Array.isArray(params.tendon_length) ? params.tendon_length : [params.tendon_length]).length;
    this.nMuscles = n;
    this.dt = timestep;
    this.maxIsoForce = bc(params.max_isometric_force, n);
    this.l0_ce = bc(params.optimal_muscle_length, n);
    this.l0_se = bc(params.tendon_length, n);
    const nsl = bc(params.normalized_slack_muscle_length, n);
    this.l0_pe = this.l0_ce.map((v, m) => v * nsl[m]);
    this.musculotendon_slack_len = this.l0_pe.map((v, m) => v + this.l0_se[m]);
    this.vmax = this.l0_ce.map((v) => 10 * v);

    this.ce_0 = this.vmax.map((v) => 3 * v);
    this.ce_1 = this.vmax.map((v) => this.ce_Af * v);
    this.ce_2 = this.vmax.map((v) => 3 * this.ce_Af * v * this.ce_fmlen - 3 * this.ce_Af * v);
    this.ce_3 = 8 * this.ce_Af * this.ce_fmlen + 8 * this.ce_fmlen;
    this.ce_4 = this.vmax.map((v, m) => this.ce_Af * this.ce_fmlen * v - this.ce_1[m]);
    this.ce_5 = 8 * (this.ce_Af + 1);
    this.built = true;
  }

  integrate(dt, stateDerivative, muscleState, geometryState) {
    const n = this.nMuscles;
    const activation = this.clipActivation(muscleState[0].map((a, m) => a + stateDerivative[0][m] * dt));
    const rows = [activation, new Array(n), new Array(n), new Array(n), new Array(n), new Array(n), new Array(n)];
    for (let m = 0; m < n; m++) {
      const mtuLen = geometryState[0][m];
      const muscleLen = Math.max(mtuLen - this.l0_se[m], 0.001);
      const muscleVel = geometryState[1][m];
      const a = activation[m];
      const a3 = a * 3;
      const cond = muscleVel <= 0;
      let nom; let den;
      if (cond) {
        nom = this.ce_Af * (a * this.ce_0[m] + 4 * muscleVel + this.vmax[m]);
        den = a3 * this.ce_1[m] + this.ce_1[m] - 4 * muscleVel;
      } else {
        nom = this.ce_2[m] * a + this.ce_3 * muscleVel + this.ce_4[m];
        den = this.ce_4[m] * a3 + this.ce_5 * muscleVel + this.ce_4[m];
      }
      const fvce = Math.max(nom / den, 0);
      const flpe = Math.max((Math.exp((this.pe_1 * (muscleLen - this.l0_pe[m])) / this.l0_ce[m]) - 1) / this.pe_den, 0);
      const flce = Math.exp(-((muscleLen / this.l0_ce[m] - 1) ** 2) / this.ce_gamma);
      const force = (a * flce * fvce + flpe) * this.maxIsoForce[m];

      rows[1][m] = muscleLen;
      rows[2][m] = muscleVel;
      rows[3][m] = flpe;
      rows[4][m] = flce;
      rows[5][m] = fvce;
      rows[6][m] = force;
    }
    return rows;
  }

  getInitialMuscleState(geometryState) {
    const n = this.nMuscles;
    return this.integrate(this.dt, [new Array(n).fill(0)], [new Array(n).fill(this.minActivation)], geometryState);
  }
}

export class CompliantTendonHillMuscle extends RigidTendonHillMuscle {
  constructor({ minActivation = 0.01, ...rest } = {}) {
    super({ minActivation, ...rest });
    this.name = 'CompliantTendonHillMuscle';
    this.stateName = ['activation', 'muscle length', 'muscle velocity', 'force-length PE', 'force-length SE', 'active force', 'force'];
    this.stateDim = this.stateName.length;
  }

  // Returns the normalized muscle velocity that satisfies the force balance.
  _normalizedMuscleVel(muscleLenN, activation, activeForce) {
    const flce = Math.max(1 + (-(muscleLenN ** 2) + 2 * muscleLenN - 1) / this.f_iso_n_den, this.min_flce);
    const a_rel_st = muscleLenN < 1 ? 0.41 * flce : 0.41;
    const b_rel_st = activation < this.q_crit
      ? 5.2 * (1 - 0.9 * ((activation - this.q_crit) / (5e-3 - this.q_crit))) ** 2
      : 5.2;
    const f_x_a = flce * activation;
    const dfdvcon0 = (f_x_a + activation * a_rel_st) / b_rel_st;

    const p1 = (-f_x_a * 0.5) / (this.s_as - dfdvcon0 * 2);
    const p3 = -1.5 * f_x_a;
    const p2_term = (4 * (f_x_a * 0.5) ** 2 * -this.s_as) / (this.s_as - dfdvcon0 * 2);

    let sqrtTerm = activeForce ** 2 + 2 * activeForce * p1 * this.s_as
      + 2 * activeForce * p3 + p1 ** 2 * this.s_as ** 2 + 2 * p1 * p3 * this.s_as
      + p2_term + p3 ** 2;
    sqrtTerm = Math.max(sqrtTerm, 0);

    let nom; let den;
    if (activeForce < f_x_a) {
      nom = b_rel_st * (activeForce - f_x_a);
      den = activeForce + activation * a_rel_st;
    } else {
      nom = -activeForce + p1 * this.s_as - p3 - Math.sqrt(sqrtTerm);
      den = -2 * this.s_as;
    }
    return nom / den;
  }

  ode(excitation, muscleState) {
    const n = this.nMuscles;
    const dAct = this.activationOde(excitation, muscleState[0]);
    const velN = new Array(n);
    for (let m = 0; m < n; m++) {
      const muscleLenN = muscleState[1][m] / this.l0_ce[m];
      const activeForce = muscleState[5][m];
      velN[m] = this._normalizedMuscleVel(muscleLenN, muscleState[0][m], activeForce);
    }
    return [dAct, velN];
  }

  integrate(dt, stateDerivative, muscleState, geometryState) {
    const n = this.nMuscles;
    const rows = [new Array(n), new Array(n), new Array(n), new Array(n), new Array(n), new Array(n), new Array(n)];
    for (let m = 0; m < n; m++) {
      const muscleLen = muscleState[1][m];
      const muscleLenN = muscleLen / this.l0_ce[m];
      const mtuLen = geometryState[0][m];
      const tendonLen = mtuLen - muscleLen;
      const tendonStrain = Math.max((tendonLen - this.l0_se[m]) / this.l0_se[m], 0);
      const muscleStrain = Math.max((muscleLen - this.l0_pe[m]) / this.l0_ce[m], 0);

      const flse = Math.min(this.k_se * tendonStrain ** 2, 1);
      const flpe = this.k_pe[m] * muscleStrain ** 2;
      const activeForce = Math.max(flse - flpe, 0);

      const dActivation = stateDerivative[0][m];
      const muscleVelN = stateDerivative[1][m];
      const activation = this.clipActivationScalar(muscleState[0][m] + dActivation * dt);
      const newMuscleLen = (muscleLenN + dt * muscleVelN) * this.l0_ce[m];
      const muscleVel = muscleVelN * this.vmax[m];
      const force = flse * this.maxIsoForce[m];

      rows[0][m] = activation;
      rows[1][m] = newMuscleLen;
      rows[2][m] = muscleVel;
      rows[3][m] = flpe;
      rows[4][m] = flse;
      rows[5][m] = activeForce;
      rows[6][m] = force;
    }
    return rows;
  }

  getInitialMuscleState(geometryState) {
    const n = this.nMuscles;
    const activation = new Array(n);
    const muscleLen = new Array(n);
    const dActivation = new Array(n).fill(0);
    const muscleVelN = new Array(n);
    for (let m = 0; m < n; m++) {
      const mtuLen = geometryState[0][m];
      const a0 = this.minActivation;
      const kpe = this.k_pe[m];
      const kse = this.k_se;
      const l0ce = this.l0_ce[m];
      const l0pe = this.l0_pe[m];
      const l0se = this.l0_se[m];
      let mlen;
      if (mtuLen < 0) mlen = -1;
      else if (mtuLen < l0se) mlen = 0.001 * l0ce;
      else if (mtuLen < l0se + l0pe) mlen = mtuLen - l0se;
      else {
        mlen = (kpe * l0pe * l0se ** 2 - kse * l0ce ** 2 * mtuLen + kse * l0ce ** 2 * l0se
          - l0ce * l0se * Math.sqrt(kpe * kse) * (-mtuLen + l0pe + l0se))
          / (kpe * l0se ** 2 - kse * l0ce ** 2);
      }
      const tendonLen = mtuLen - mlen;
      const tendonStrain = Math.max((tendonLen - l0se) / l0se, 0);
      const muscleStrain = Math.max((mlen - l0pe) / l0ce, 0);
      const flse = Math.min(kse * tendonStrain ** 2, 1);
      const flpe = Math.min(kpe * muscleStrain ** 2, 1);
      const activeForce = Math.max(flse - flpe, 0);

      activation[m] = a0;
      muscleLen[m] = mlen;
      muscleVelN[m] = this._normalizedMuscleVel(mlen / l0ce, a0, activeForce);
    }
    return this.integrate(this.dt, [dActivation, muscleVelN], [activation, muscleLen], geometryState);
  }
}
