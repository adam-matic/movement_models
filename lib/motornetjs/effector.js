// Effector: a JavaScript port of motornet/effector.py.
//
// An Effector wraps a Skeleton with one or more muscles, computes muscle
// geometry (lengths, velocities, moment arms) from the joint configuration, and
// integrates the coupled muscle + skeleton dynamics forward in time. Batch size
// is fixed to 1 (single simulation) in this port.

import { PointMass, TwoDofArm, LockedShoulderArm } from './skeleton.js';
import { ReluMuscle, CompliantTendonHillMuscle } from './muscle.js';
import { RNG } from './random.js';

// Apply f element-wise across matching nested arrays (used to combine RK4 slopes).
function mapN(arrays, f) {
  if (Array.isArray(arrays[0])) {
    return arrays[0].map((_, i) => mapN(arrays.map((a) => a[i]), f));
  }
  return f(...arrays);
}

function broadcastBounds(lb, ub, dof) {
  const toArr = (x) => (Array.isArray(x) ? x.slice() : new Array(dof).fill(x));
  let lo = toArr(lb);
  let hi = toArr(ub);
  if (lo.length === 1) lo = new Array(dof).fill(lo[0]);
  if (hi.length === 1) hi = new Array(dof).fill(hi[0]);
  return { lo, hi };
}

export class Effector {
  constructor({
    skeleton, muscle, name = 'Effector', nMinisteps = 1, timestep = 0.01,
    integrationMethod = 'euler', damping = 0,
    posLowerBound = null, posUpperBound = null, velLowerBound = null, velUpperBound = null,
    rng = null,
  }) {
    this.name = name;
    this.damping = damping;
    this.skeleton = skeleton;
    this.dof = skeleton.dof;
    this.spaceDim = skeleton.spaceDim;
    this.stateDim = skeleton.stateDim;
    this.outputDim = skeleton.outputDim;
    this.nMinisteps = nMinisteps;
    this.dt = timestep;
    this.minidt = this.dt / this.nMinisteps;
    this.halfMinidt = this.minidt / 2;
    this.integrationMethod = integrationMethod.toLowerCase();
    this.rng = rng || new RNG();

    const pLB = posLowerBound ?? skeleton.posLowerBound;
    const pUB = posUpperBound ?? skeleton.posUpperBound;
    const vLB = velLowerBound ?? skeleton.velLowerBound;
    const vUB = velUpperBound ?? skeleton.velUpperBound;
    const pos = broadcastBounds(pLB, pUB, this.dof);
    const vel = broadcastBounds(vLB, vUB, this.dof);
    this.posLowerBound = pos.lo;
    this.posUpperBound = pos.hi;
    this.velLowerBound = vel.lo;
    this.velUpperBound = vel.hi;
    this.posRangeBound = this.posLowerBound.map((v, i) => v - this.posUpperBound[i]);

    this.skeleton.build(this.dt, this.posUpperBound, this.posLowerBound, this.velUpperBound, this.velLowerBound);

    this.muscle = muscle;
    this.forceIndex = this.muscle.stateName.indexOf('force');
    this.nMuscles = 0;
    this.inputDim = 0;
    this.muscleName = [];
    this.muscleStateDim = this.muscle.stateDim;
    this.geometryStateDim = 2 + this.dof;
    this.geometryStateName = ['musculotendon length', 'musculotendon velocity'];
    for (let d = 0; d < this.dof; d++) this.geometryStateName.push(`moment for joint ${d}`);

    // muscle-path bookkeeping (used by the generic geometry calculation)
    this._pathFixationBody = []; // [point]
    this._pathCoordinates = Array.from({ length: this.spaceDim }, () => []); // [dim][point]
    this._muscleIndex = []; // [point] -> 1-based muscle id
    this._muscleTransitions = []; // [segment] bool
    this._sectionSplits = []; // segment grouping per muscle
    this._tobuild = {}; // build-param name -> per-muscle array

    if (this.integrationMethod === 'euler') this._integrate = this._euler.bind(this);
    else if (['rk4', 'rungekutta4', 'runge-kutta4', 'runge-kutta-4'].includes(this.integrationMethod)) this._integrate = this._rungekutta4.bind(this);
    else throw new Error(`Unknown integration method: ${integrationMethod}`);

    this.states = { joint: null, cartesian: null, muscle: null, geometry: null, fingertip: null };
  }

  addMuscle({ pathFixationBody, pathCoordinates, name = null, ...params }) {
    // pathCoordinates is points x dims; transpose to dims x points to match storage
    const nPoints = pathFixationBody.length;
    for (let p = 0; p < nPoints; p++) {
      this._pathFixationBody.push(pathFixationBody[p]);
      for (let d = 0; d < this.spaceDim; d++) this._pathCoordinates[d].push(pathCoordinates[p][d]);
    }
    this.nMuscles += 1;
    this.inputDim += this.muscle.inputDim;
    for (let p = 0; p < nPoints; p++) this._muscleIndex.push(this.nMuscles);
    this._recomputeSplits();

    // collect build parameters, applying muscle defaults where omitted
    for (const key of this.muscle.toBuildKeys) {
      if (!this._tobuild[key]) this._tobuild[key] = [];
      if (key in params) this._tobuild[key].push(params[key]);
      else if (key in this.muscle.toBuildDefaults) this._tobuild[key].push(this.muscle.toBuildDefaults[key]);
      else throw new Error(`Missing keyword argument ${key}.`);
    }
    this.muscle.build(this.minidt, this._tobuild);

    this.muscleName.push(name ?? `muscle_${this.nMuscles}`);
  }

  _recomputeSplits() {
    const mi = this._muscleIndex;
    const L = mi.length;
    const transitions = [];
    const nonzero = [];
    for (let i = 0; i < L - 1; i++) {
      const diff = mi[i + 1] - mi[i];
      transitions.push(diff === 1);
      if (diff !== 0) nonzero.push(i);
    }
    const rowSplits = [0, ...nonzero.map((i) => i + 1), L - 1];
    const sectionSplits = [];
    for (let i = 0; i < rowSplits.length - 1; i++) sectionSplits.push(rowSplits[i + 1] - rowSplits[i]);
    this._muscleTransitions = transitions;
    this._sectionSplits = sectionSplits;
  }

  step(action, { endpointLoad = null, jointLoad = null } = {}) {
    const eLoad = endpointLoad ?? new Array(this.spaceDim).fill(0);
    const jLoad = jointLoad ?? new Array(this.dof).fill(0);
    const a = this.muscle.clipActivation(action);
    for (let i = 0; i < this.nMinisteps; i++) this.integrate(a, eLoad, jLoad);
  }

  reset({ batchSize = 1, jointState = null } = {}) {
    const joint0 = this._parseInitialJointState(jointState);
    const geometry0 = this.getGeometry(joint0);
    const muscle0 = this.muscle.getInitialMuscleState(geometry0);
    this._setState({ joint: joint0, muscle: muscle0, geometry: geometry0 });
  }

  getGeometry(jointState) {
    return this._getGeometry(jointState);
  }

  // Generic geometry: build cartesian fixation paths, then reduce segment
  // lengths / velocities / moment arms per muscle.
  _getGeometry(jointState) {
    const { xy, dxydt, dxyddof } = this.skeleton.path2cartesian(this._pathCoordinates, this._pathFixationBody, jointState);
    const L = this._pathFixationBody.length;
    const nSeg = L - 1;
    const sd = this.spaceDim;
    const dof = this.dof;

    const segLen = new Array(nSeg);
    const segVel = new Array(nSeg);
    const segMom = Array.from({ length: dof }, () => new Array(nSeg));
    for (let p = 0; p < nSeg; p++) {
      if (this._muscleTransitions[p]) {
        segLen[p] = 0; segVel[p] = 0;
        for (let k = 0; k < dof; k++) segMom[k][p] = 0;
        continue;
      }
      let lenSq = 0; let vel = 0;
      const dpos = new Array(sd);
      for (let d = 0; d < sd; d++) {
        dpos[d] = xy[d][p + 1] - xy[d][p];
        const dvel = dxydt[d][p + 1] - dxydt[d][p];
        lenSq += dpos[d] * dpos[d];
        vel += dpos[d] * dvel;
      }
      const len = Math.sqrt(lenSq);
      segLen[p] = len;
      segVel[p] = vel / len;
      for (let k = 0; k < dof; k++) {
        let mom = 0;
        for (let d = 0; d < sd; d++) mom += (dxyddof[d][k][p + 1] - dxyddof[d][k][p]) * dpos[d];
        segMom[k][p] = mom / len;
      }
    }

    // sum each muscle's segments
    const nm = this.nMuscles;
    const mtuLen = new Array(nm);
    const mtuVel = new Array(nm);
    const moments = Array.from({ length: dof }, () => new Array(nm));
    let off = 0;
    for (let m = 0; m < nm; m++) {
      const count = this._sectionSplits[m];
      let sl = 0; let sv = 0;
      const sm = new Array(dof).fill(0);
      for (let s = 0; s < count; s++) {
        sl += segLen[off + s];
        sv += segVel[off + s];
        for (let k = 0; k < dof; k++) sm[k] += segMom[k][off + s];
      }
      off += count;
      mtuLen[m] = sl; mtuVel[m] = sv;
      for (let k = 0; k < dof; k++) moments[k][m] = sm[k];
    }
    return [mtuLen, mtuVel, ...moments];
  }

  _setState(states) {
    Object.assign(this.states, states);
    this.states.cartesian = this.joint2cartesian(states.joint);
    this.states.fingertip = this.states.cartesian.slice(0, this.spaceDim);
  }

  integrate(action, endpointLoad, jointLoad) {
    this._integrate(action, endpointLoad, jointLoad);
  }

  _euler(action, endpointLoad, jointLoad) {
    const states0 = this.states;
    const deriv = this.ode(action, states0, endpointLoad, jointLoad);
    const states = this.integrationStep(this.minidt, deriv, states0);
    this._setState(states);
  }

  _rungekutta4(action, endpointLoad, jointLoad) {
    const states0 = this.states;
    const k1 = this.ode(action, states0, endpointLoad, jointLoad);
    let states = this.integrationStep(this.halfMinidt, k1, states0);
    const k2 = this.ode(action, states, endpointLoad, jointLoad);
    states = this.integrationStep(this.halfMinidt, k2, states);
    const k3 = this.ode(action, states, endpointLoad, jointLoad);
    states = this.integrationStep(this.minidt, k3, states);
    const k4 = this.ode(action, states, endpointLoad, jointLoad);
    const k = {
      muscle: mapN([k1.muscle, k2.muscle, k3.muscle, k4.muscle], (a, b, c, d) => (a + 2 * (b + c) + d) / 6),
      joint: mapN([k1.joint, k2.joint, k3.joint, k4.joint], (a, b, c, d) => (a + 2 * (b + c) + d) / 6),
    };
    states = this.integrationStep(this.minidt, k, states0);
    this._setState(states);
  }

  integrationStep(dt, stateDerivative, states) {
    const muscle = this.muscle.integrate(dt, stateDerivative.muscle, states.muscle, states.geometry);
    const joint = this.skeleton.integrate(dt, stateDerivative.joint, states.joint);
    const geometry = this.getGeometry(joint);
    return { muscle, joint, geometry };
  }

  ode(action, states, endpointLoad, jointLoad) {
    const moments = states.geometry.slice(2); // [dof][muscle]
    const forces = states.muscle[this.forceIndex]; // [muscle]
    const jointVel = states.joint.slice(this.dof);
    const generalizedForces = new Array(this.dof);
    for (let k = 0; k < this.dof; k++) {
      let s = 0;
      for (let m = 0; m < this.nMuscles; m++) s += forces[m] * moments[k][m];
      generalizedForces[k] = -s + jointLoad[k] - this.damping * jointVel[k];
    }
    return {
      muscle: this.muscle.ode(action, states.muscle),
      joint: this.skeleton.ode(generalizedForces, states.joint, endpointLoad),
    };
  }

  drawRandomUniformStates(batchSize = 1) {
    const pos = this.posUpperBound.map((ub, i) => this.posRangeBound[i] * this.rng.uniform() + ub);
    const vel = new Array(this.dof).fill(0);
    return [...pos, ...vel];
  }

  _parseInitialJointState(jointState) {
    if (jointState === null) return this.drawRandomUniformStates(1);
    const n = jointState.length;
    if (n === this.stateDim) return this.drawFixedStates(jointState.slice(0, this.dof), jointState.slice(this.dof));
    if (n === this.stateDim / 2) return this.drawFixedStates(jointState, null);
    throw new Error(`Unexpected joint_state length ${n}`);
  }

  drawFixedStates(position, velocity = null) {
    const vel = velocity ?? new Array(this.dof).fill(0);
    if (position.length !== this.dof) throw new Error(`position has ${position.length} entries but dof=${this.dof}`);
    return [...position, ...vel];
  }

  joint2cartesian(jointState) {
    return this.skeleton.joint2cartesian(jointState);
  }
}

export class ReluPointMass24 extends Effector {
  constructor({ timestep = 0.01, maxIsometricForce = 500, mass = 1, ...rest } = {}) {
    super({ skeleton: new PointMass({ spaceDim: 2, mass }), muscle: new ReluMuscle(), timestep, ...rest });
    const f = maxIsometricForce;
    this.addMuscle({ pathFixationBody: [0, 1], pathCoordinates: [[2, 2], [0, 0]], name: 'UpperRight', max_isometric_force: f });
    this.addMuscle({ pathFixationBody: [0, 1], pathCoordinates: [[-2, 2], [0, 0]], name: 'UpperLeft', max_isometric_force: f });
    this.addMuscle({ pathFixationBody: [0, 1], pathCoordinates: [[2, -2], [0, 0]], name: 'LowerRight', max_isometric_force: f });
    this.addMuscle({ pathFixationBody: [0, 1], pathCoordinates: [[-2, -2], [0, 0]], name: 'LowerLeft', max_isometric_force: f });
  }
}

// Shared logic for the constant-moment-arm effectors (Reacher, FreePointMass24).
class ConstantMomentArmEffector extends Effector {
  _configureConstant({ muscleNames, momentArm, muscleKwargs }) {
    this.muscleStateDim = this.muscle.stateDim;
    this.geometryStateDim = 2 + this.dof;
    this.nMuscles = muscleNames.length;
    this.inputDim = this.nMuscles;
    this.muscleName = muscleNames.slice();
    const params = { ...muscleKwargs };
    if (!('max_isometric_force' in params)) params.max_isometric_force = new Array(this.nMuscles).fill(1000);
    for (const key of this.muscle.toBuildKeys) {
      if (!(key in params) && key in this.muscle.toBuildDefaults) params[key] = this.muscle.toBuildDefaults[key];
    }
    this.muscle.build(this.dt, params);
    this._momentArm = momentArm; // [dof][muscle]
  }

  _getGeometry(jointState) {
    const lenVel = [new Array(this.nMuscles).fill(0), new Array(this.nMuscles).fill(0)];
    return [...lenVel, ...this._momentArm.map((row) => row.slice())];
  }
}

export class FreePointMass24 extends ConstantMomentArmEffector {
  constructor({ muscle, skeleton = null, timestep = 0.01, muscleKwargs = {}, ...rest } = {}) {
    const posLowerBound = rest.posLowerBound ?? [-0.6, -0.6];
    const posUpperBound = rest.posUpperBound ?? [0.6, 0.6];
    delete rest.posLowerBound; delete rest.posUpperBound;
    super({ skeleton: skeleton ?? new PointMass({ spaceDim: 2, mass: 1 }), muscle, timestep, posLowerBound, posUpperBound, ...rest });
    this._configureConstant({
      muscleNames: ['r', 'u', 'l', 'd'],
      momentArm: [[-1, 0, 1, 0], [0, -1, 0, 1]],
      muscleKwargs,
    });
  }
}

export class Reacher extends ConstantMomentArmEffector {
  constructor({ muscle, skeleton = null, timestep = 0.01, muscleKwargs = {}, ...rest } = {}) {
    const deg = Math.PI / 180;
    const posLowerBound = rest.posLowerBound ?? [0, 0];
    const posUpperBound = rest.posUpperBound ?? [135 * deg, 155 * deg];
    delete rest.posLowerBound; delete rest.posUpperBound;
    const sk = skeleton ?? new TwoDofArm({ m1: 1.82, m2: 1.43, l1g: 0.135, l2g: 0.165, i1: 0.051, i2: 0.057, l1: 0.309, l2: 0.333 });
    super({ skeleton: sk, muscle, timestep, posLowerBound, posUpperBound, ...rest });
    this._configureConstant({
      muscleNames: ['sf', 'se', 'ef', 'ee'],
      momentArm: [[-1, 1, 0, 0], [0, 0, -1, 1]],
      muscleKwargs,
    });
  }
}

export class RigidTendonArm26 extends Effector {
  constructor({ muscle, skeleton = null, timestep = 0.01, muscleKwargs = {}, ...rest } = {}) {
    const deg = Math.PI / 180;
    const posLowerBound = rest.posLowerBound ?? [0, 0];
    const posUpperBound = rest.posUpperBound ?? [135 * deg, 155 * deg];
    delete rest.posLowerBound; delete rest.posUpperBound;
    const sk = skeleton ?? new TwoDofArm({ m1: 1.82, m2: 1.43, l1g: 0.135, l2g: 0.165, i1: 0.051, i2: 0.057, l1: 0.309, l2: 0.333 });
    super({ skeleton: sk, muscle, timestep, posLowerBound, posUpperBound, ...rest });

    this.muscleStateDim = this.muscle.stateDim;
    this.geometryStateDim = 2 + this.dof;
    this.nMuscles = 6;
    this.inputDim = 6;
    this.muscleName = ['pectoralis', 'deltoid', 'brachioradialis', 'tricepslat', 'biceps', 'tricepslong'];

    const params = { ...muscleKwargs };
    for (const key of this.muscle.toBuildKeys) {
      if (!(key in params) && key in this.muscle.toBuildDefaults) params[key] = this.muscle.toBuildDefaults[key];
    }
    params.max_isometric_force = [838, 1207, 1422, 1549, 414, 603];
    params.tendon_length = [0.039, 0.066, 0.172, 0.187, 0.204, 0.217];
    params.optimal_muscle_length = [0.134, 0.140, 0.092, 0.093, 0.137, 0.127];
    this._buildArmMuscle(params);

    // polynomial moment-arm coefficients (Nijhof & Kouwenhoven 2000)
    this.a0 = [0.151, 0.2322, 0.2859, 0.2355, 0.3329, 0.2989];
    this.a1 = [[-0.03, 0.03, 0, 0, -0.03, 0.03], [0, 0, -0.014, 0.025, -0.016, 0.03]];
    this.a2 = [[0, 0, 0, 0, 0, 0], [0, 0, -4e-3, -2.2e-3, -5.7e-3, -3.2e-3]];
    this.a3 = [Math.PI / 2, 0];
  }

  _buildArmMuscle(params) {
    this.muscle.build(this.dt, params);
  }

  _getGeometry(jointState) {
    const pos = [jointState[0] - this.a3[0], jointState[1] - this.a3[1]];
    const vel = [jointState[2], jointState[3]];
    const mtuLen = new Array(6);
    const mtuVel = new Array(6);
    const moment = [new Array(6), new Array(6)];
    for (let m = 0; m < 6; m++) {
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

export class CompliantTendonArm26 extends RigidTendonArm26 {
  constructor({ timestep = 0.0002, skeleton = null, muscleKwargs = {}, ...rest } = {}) {
    const integrationMethod = rest.integrationMethod ?? 'rk4';
    delete rest.integrationMethod;
    super({ muscle: new CompliantTendonHillMuscle(), skeleton, timestep, integrationMethod, muscleKwargs, ...rest });
    // relaxed tendon lengths and shoulder rest-length for numerical stability
    const params = { ...muscleKwargs };
    for (const key of this.muscle.toBuildKeys) {
      if (!(key in params) && key in this.muscle.toBuildDefaults) params[key] = this.muscle.toBuildDefaults[key];
    }
    params.max_isometric_force = [838, 1207, 1422, 1549, 414, 603];
    params.tendon_length = [0.070, 0.070, 0.172, 0.187, 0.204, 0.217];
    params.optimal_muscle_length = [0.134, 0.140, 0.092, 0.093, 0.137, 0.127];
    this.muscle.build(timestep, params);
    this.a0 = [0.182, 0.2362, 0.2859, 0.2355, 0.3329, 0.2989];
  }
}

// 1-DOF elbow arm with 3 muscles (brachioradialis, tricepslat, biceps).
// The shoulder is locked at a fixed angle via LockedShoulderArm. MTU geometry
// uses the same polynomial moment-arm coefficients as RigidTendonArm26 with the
// constant shoulder contribution folded into a0 at construction time.
export class RigidTendonElbow13 extends Effector {
  constructor({
    muscle, shoulderAngle = Math.PI / 4, skeleton = null,
    timestep = 0.01, muscleKwargs = {}, ...rest
  } = {}) {
    const deg = Math.PI / 180;
    const posLowerBound = rest.posLowerBound ?? [0];
    const posUpperBound = rest.posUpperBound ?? [155 * deg];
    delete rest.posLowerBound; delete rest.posUpperBound;
    const sk = skeleton ?? new LockedShoulderArm({ shoulderAngle });
    super({ skeleton: sk, muscle, timestep, posLowerBound, posUpperBound, ...rest });

    this.muscleStateDim = this.muscle.stateDim;
    this.geometryStateDim = 3; // mtuLen, mtuVel, moment_elbow
    this.nMuscles = 3;
    this.inputDim = 3;
    this.muscleName = ['brachioradialis', 'tricepslat', 'biceps'];

    const params = { ...muscleKwargs };
    for (const key of this.muscle.toBuildKeys) {
      if (!(key in params) && key in this.muscle.toBuildDefaults) params[key] = this.muscle.toBuildDefaults[key];
    }
    // muscles 2, 3, 4 from RigidTendonArm26 (brachioradialis, tricepslat, biceps)
    params.max_isometric_force = [1422, 1549, 414];
    params.tendon_length = [0.172, 0.187, 0.204];
    params.optimal_muscle_length = [0.092, 0.093, 0.137];
    this.muscle.build(this.minidt, params);

    // Moment-arm polynomial coefficients from Arm26 (Nijhof & Kouwenhoven 2000).
    // Shoulder reference offset: a3[0] = PI/2. Elbow reference offset: a3[1] = 0.
    // The shoulder position is fixed, so fold its contribution into a constant a0.
    const shoPos = shoulderAngle - Math.PI / 2;
    const a0_arm26 = [0.2859, 0.2355, 0.3329]; // a0 for muscles 2, 3, 4
    const a1_sho = [0, 0, -0.03]; // a1[0] for muscles 2, 3, 4 (shoulder terms)
    const a2_sho = [0, 0, 0]; // a2[0] for muscles 2, 3, 4
    this._a0 = a0_arm26.map((a0, i) => a0 + (a1_sho[i] + shoPos * a2_sho[i]) * shoPos);
    this._a1 = [-0.014, 0.025, -0.016]; // a1[1] (elbow moment arm linear term)
    this._a2 = [-4e-3, -2.2e-3, -5.7e-3]; // a2[1] (elbow moment arm quadratic term)
  }

  _getGeometry(jointState) {
    const elbPos = jointState[0];
    const elbVel = jointState[1];
    const mtuLen = new Array(3);
    const mtuVel = new Array(3);
    const moment = [new Array(3)];
    for (let m = 0; m < 3; m++) {
      const ma = this._a1[m] + elbPos * this._a2[m] * 2;
      moment[0][m] = ma;
      mtuLen[m] = this._a0[m] + (this._a1[m] + elbPos * this._a2[m]) * elbPos;
      mtuVel[m] = elbVel * ma;
    }
    return [mtuLen, mtuVel, ...moment];
  }
}

// Compliant-tendon variant of RigidTendonElbow13. Uses RK4 and a smaller
// default timestep for numerical stability, matching the CompliantTendonArm26
// convention.
export class CompliantTendonElbow13 extends RigidTendonElbow13 {
  constructor({ shoulderAngle = Math.PI / 4, skeleton = null, timestep = 0.0002, muscleKwargs = {}, ...rest } = {}) {
    const integrationMethod = rest.integrationMethod ?? 'rk4';
    delete rest.integrationMethod;
    super({
      muscle: new CompliantTendonHillMuscle(),
      shoulderAngle, skeleton, timestep, integrationMethod, muscleKwargs, ...rest,
    });
    // rebuild with the compliant timestep (super already built, this overwrites)
    const params = { ...muscleKwargs };
    for (const key of this.muscle.toBuildKeys) {
      if (!(key in params) && key in this.muscle.toBuildDefaults) params[key] = this.muscle.toBuildDefaults[key];
    }
    params.max_isometric_force = [1422, 1549, 414];
    params.tendon_length = [0.172, 0.187, 0.204];
    params.optimal_muscle_length = [0.092, 0.093, 0.137];
    this.muscle.build(timestep, params);
  }
}
