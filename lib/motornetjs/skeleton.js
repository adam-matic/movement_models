// Skeleton dynamics: a JavaScript port of motornet/skeleton.py.
//
// All tensors in the Python library carry a leading batch dimension. This port
// targets interactive, single-simulation use in the browser, so the batch
// dimension is dropped: a joint state is a flat array [pos_0..pos_{dof-1},
// vel_0..vel_{dof-1}], and per-muscle geometry quantities are arrays over
// muscles.

// Element-wise clip of a vector x against (possibly scalar) lower/upper bounds.
function clipVec(x, lb, ub) {
  return x.map((v, i) => {
    const lo = Array.isArray(lb) ? lb[i] : lb;
    const hi = Array.isArray(ub) ? ub[i] : ub;
    return Math.min(Math.max(v, lo), hi);
  });
}

export class Skeleton {
  constructor(dof, spaceDim, opts = {}) {
    this.dof = dof;
    this.spaceDim = spaceDim;
    this.name = opts.name || 'skeleton';
    this.inputDim = opts.inputDim ?? this.dof;
    this.stateDim = opts.stateDim ?? this.dof * 2;
    this.outputDim = opts.outputDim ?? this.stateDim;
    this.geometryStateDim = 2 + this.dof;

    this.posLowerBound = opts.posLowerBound ?? -1;
    this.posUpperBound = opts.posUpperBound ?? +1;
    this.velLowerBound = opts.velLowerBound ?? -1000;
    this.velUpperBound = opts.velUpperBound ?? +1000;

    this.dt = null;
    this.built = false;
  }

  // Called by the Effector wrapper once the timestep and final bounds are known.
  build(timestep, posUpperBound, posLowerBound, velUpperBound, velLowerBound) {
    this.posUpperBound = posUpperBound;
    this.posLowerBound = posLowerBound;
    this.velUpperBound = velUpperBound;
    this.velLowerBound = velLowerBound;
    this.dt = timestep;
    this.built = true;
  }

  clipPosition(pos) {
    return clipVec(pos, this.posLowerBound, this.posUpperBound);
  }

  // Clip velocities to their bounds, then zero out any velocity that would push
  // the joint further past a position bound it already sits on.
  clipVelocity(pos, vel) {
    return vel.map((v, i) => {
      const lo = Array.isArray(this.velLowerBound) ? this.velLowerBound[i] : this.velLowerBound;
      const hi = Array.isArray(this.velUpperBound) ? this.velUpperBound[i] : this.velUpperBound;
      let nv = Math.min(Math.max(v, lo), hi);
      const pLo = Array.isArray(this.posLowerBound) ? this.posLowerBound[i] : this.posLowerBound;
      const pHi = Array.isArray(this.posUpperBound) ? this.posUpperBound[i] : this.posUpperBound;
      if (nv < 0 && pos[i] <= pLo) nv = 0;
      if (nv > 0 && pos[i] >= pHi) nv = 0;
      return nv;
    });
  }

  // Subclasses implement these.
  ode(_inputs, _jointState, _endpointLoad) { throw new Error('not implemented'); }
  integrate(_dt, _stateDerivative, _jointState) { throw new Error('not implemented'); }
  joint2cartesian(_jointState) { throw new Error('not implemented'); }
  path2cartesian(_pathCoordinates, _pathFixationBody, _jointState) { throw new Error('not implemented'); }
}

export class PointMass extends Skeleton {
  constructor({ spaceDim = 2, mass = 1, name = 'point_mass' } = {}) {
    super(spaceDim, spaceDim, { name });
    this.mass = mass;
  }

  ode(inputs, _jointState, endpointLoad) {
    return inputs.map((v, i) => v + endpointLoad[i]);
  }

  integrate(dt, stateDerivative, jointState) {
    const oldPos = jointState.slice(0, this.dof);
    const oldVel = jointState.slice(this.dof);
    let newVel = oldVel.map((v, i) => v + (stateDerivative[i] * dt) / this.mass);
    let newPos = oldPos.map((p, i) => p + oldVel[i] * dt);
    newVel = this.clipVelocity(newPos, newVel);
    newPos = this.clipPosition(newPos);
    return [...newPos, ...newVel];
  }

  joint2cartesian(jointState) {
    return jointState.slice();
  }

  // path2cartesian for a point mass. A fixation body of 0 means the point is
  // anchored in the worldspace; otherwise it rides on the point mass.
  path2cartesian(pathCoordinates, pathFixationBody, jointState) {
    const pos = jointState.slice(0, this.dof);
    const vel = jointState.slice(this.dof);
    const nPoints = pathFixationBody.length;
    const xy = [];
    const dxydt = [];
    const dxyddof = [];
    for (let d = 0; d < this.dof; d++) {
      xy[d] = new Array(nPoints);
      dxydt[d] = new Array(nPoints);
      dxyddof[d] = [];
      for (let k = 0; k < this.dof; k++) dxyddof[d][k] = new Array(nPoints);
      for (let p = 0; p < nPoints; p++) {
        const fixed = pathFixationBody[p] === 0;
        xy[d][p] = (fixed ? 0 : pos[d]) + pathCoordinates[d][p];
        dxydt[d][p] = fixed ? 0 : vel[d];
        for (let k = 0; k < this.dof; k++) {
          dxyddof[d][k][p] = fixed ? 0 : (d === k ? 1 : 0);
        }
      }
    }
    return { xy, dxydt, dxyddof };
  }
}

export class TwoDofArm extends Skeleton {
  constructor({
    name = 'two_dof_arm', m1 = 1.864572, m2 = 1.534315, l1g = 0.180496,
    l2g = 0.181479, i1 = 0.013193, i2 = 0.020062, l1 = 0.309, l2 = 0.26,
    viscosity = 0,
  } = {}) {
    const deg = Math.PI / 180;
    const lb = [-0 * deg, 0 * deg];
    const ub = [140 * deg, 160 * deg];
    super(2, 2, { name, posLowerBound: lb, posUpperBound: ub });

    this.m1 = m1; this.m2 = m2;
    this.L1g = l1g; this.L2g = l2g;
    this.I1 = i1; this.I2 = i2;
    this.L1 = l1; this.L2 = l2;
    this.c_viscosity = viscosity;

    // mass-matrix coefficients (constant + cos(elbow)-modulated parts)
    this.inertia_11_c = m1 * l1g ** 2 + i1 + m2 * (l2g ** 2 + l1 ** 2) + i2;
    this.inertia_12_c = m2 * l2g ** 2 + i2;
    this.inertia_22_c = m2 * l2g ** 2 + i2;
    this.inertia_11_m = 2 * m2 * l1 * l2g;
    this.inertia_12_m = m2 * l1 * l2g;
    this.coriolis_1 = -m2 * l1 * l2g;
    this.coriolis_2 = m2 * l1 * l2g;
  }

  ode(inputs, jointState, endpointLoad) {
    const [pos0, pos1, vel0, vel1] = jointState;
    const posSum = pos0 + pos1;
    const c1 = Math.cos(pos0);
    const c2 = Math.cos(pos1);
    const c12 = Math.cos(posSum);
    const s1 = Math.sin(pos0);
    const s2 = Math.sin(pos1);
    const s12 = Math.sin(posSum);

    // inertia (mass) matrix
    const i00 = this.inertia_11_c + c2 * this.inertia_11_m;
    const i01 = this.inertia_12_c + c2 * this.inertia_12_m;
    const i10 = this.inertia_12_c + c2 * this.inertia_12_m;
    const i11 = this.inertia_22_c + c2 * 0;

    // coriolis (+ optional viscous damping)
    const cor0 = this.coriolis_1 * s2 * (2 * vel0 + vel1) * vel1 + this.c_viscosity * vel0;
    const cor1 = this.coriolis_2 * s2 * vel0 * vel0 + this.c_viscosity * vel1;

    // distribute endpoint load to joint torques via the transpose Jacobian
    const j11 = -this.L1 * s1 - this.L2 * s12;
    const j12 = -this.L2 * s12;
    const j21 = this.L1 * c1 + this.L2 * c12;
    const j22 = this.L2 * c12;
    const tau0 = inputs[0] + (j11 * endpointLoad[0] + j21 * endpointLoad[1]);
    const tau1 = inputs[1] + (j12 * endpointLoad[0] + j22 * endpointLoad[1]);

    const rhs0 = -cor0 + tau0;
    const rhs1 = -cor1 + tau1;

    // acceleration = inertia^{-1} * rhs
    const denom = 1 / (i00 * i11 - i01 * i10);
    const acc0 = denom * (i11 * rhs0 - i01 * rhs1);
    const acc1 = denom * (-i10 * rhs0 + i00 * rhs1);
    return [acc0, acc1];
  }

  integrate(dt, stateDerivative, jointState) {
    const oldPos = jointState.slice(0, 2);
    const oldVel = jointState.slice(2);
    let newVel = oldVel.map((v, i) => v + stateDerivative[i] * dt);
    let newPos = oldPos.map((p, i) => p + oldVel[i] * dt);
    newVel = this.clipVelocity(newPos, newVel);
    newPos = this.clipPosition(newPos);
    return [...newPos, ...newVel];
  }

  joint2cartesian(jointState) {
    const [pos0, pos1, vel0, vel1] = jointState;
    const posSum = pos0 + pos1;
    const c1 = Math.cos(pos0);
    const s1 = Math.sin(pos0);
    const c12 = Math.cos(posSum);
    const s12 = Math.sin(posSum);
    const x = this.L1 * c1 + this.L2 * c12;
    const y = this.L1 * s1 + this.L2 * s12;
    const vx = -(this.L1 * s1 + this.L2 * s12) * vel0 - this.L2 * s12 * vel1;
    const vy = (this.L1 * c1 + this.L2 * c12) * vel0 + this.L2 * c12 * vel1;
    return [x, y, vx, vy];
  }

  // path2cartesian for the two-link arm. Fixation bodies: 0 = worldspace,
  // 1 = upper arm, 2 = forearm. See Sherman, Seth & Delp (2013) for the moment
  // arm derivation.
  path2cartesian(pathCoordinates, pathFixationBody, jointState) {
    const sho = jointState[0];
    const elbWrtSho = jointState[1];
    const elb = elbWrtSho + sho;
    const shoVel = jointState[2];
    const elbVel = jointState[3] + shoVel;
    const elbX = this.L1 * Math.cos(sho);
    const elbY = this.L1 * Math.sin(sho);
    const nPoints = pathFixationBody.length;

    const xy = [new Array(nPoints), new Array(nPoints)];
    const dxydt = [new Array(nPoints), new Array(nPoints)];
    // dxyddof[d][k][p], d over space dim (x,y), k over dof (shoulder, elbow)
    const dxyddof = [
      [new Array(nPoints), new Array(nPoints)],
      [new Array(nPoints), new Array(nPoints)],
    ];

    for (let p = 0; p < nPoints; p++) {
      const body = pathFixationBody[p];
      const ang = body === 0 ? 0 : (body === 1 ? -sho : -elb);
      const ca = Math.cos(ang);
      const sa = Math.sin(ang);
      const cx = pathCoordinates[0][p];
      const cy = pathCoordinates[1][p];

      // derivatives of the fixation point wrt the angle of the bone it sits on
      const dxda = cx * sa - cy * ca;
      const dyda = cx * ca + cy * sa;

      // derivatives wrt each joint angle (shoulder, elbow)
      const dxda1 = (body === 0 ? 0 : dxda) + (body === 2 ? -elbY : 0);
      const dyda1 = (body === 0 ? 0 : dyda) + (body === 2 ? elbX : 0);
      const dxda2 = body === 2 ? dxda : 0;
      const dyda2 = body === 2 ? dyda : 0;

      dxyddof[0][0][p] = dxda1;
      dxyddof[1][0][p] = dyda1;
      dxyddof[0][1][p] = dxda2;
      dxyddof[1][1][p] = dyda2;

      dxydt[0][p] = dxda1 * shoVel + dxda2 * elbVel;
      dxydt[1][p] = dyda1 * shoVel + dyda2 * elbVel;

      const originX = body === 2 ? elbX : 0;
      const originY = body === 2 ? elbY : 0;
      xy[0][p] = dyda + originX;
      xy[1][p] = -dxda + originY;
    }
    return { xy, dxydt, dxyddof };
  }
}

// A single-DOF arm: the shoulder is locked at a fixed angle and only the elbow
// rotates. The forearm inertia is constant (no configuration-dependent mass
// matrix), Coriolis terms vanish because the shoulder velocity is always zero,
// and the effective inertia reduces to I2 + m2*L2g^2.
//
// Elbow angle convention: same as TwoDofArm — measured relative to the upper
// arm, so the absolute forearm angle is shoulderAngle + elbowAngle.
export class LockedShoulderArm extends Skeleton {
  constructor({
    name = 'locked_shoulder_arm',
    shoulderAngle = Math.PI / 4,
    m2 = 1.43, l2 = 0.333, l2g = 0.165, i2 = 0.057, l1 = 0.309,
    viscosity = 0,
  } = {}) {
    const deg = Math.PI / 180;
    super(1, 2, {
      name,
      posLowerBound: [0 * deg],
      posUpperBound: [155 * deg],
      velLowerBound: [-1000],
      velUpperBound: [1000],
    });
    this.shoulderAngle = shoulderAngle;
    this.L1 = l1;
    this.L2 = l2;
    // effective elbow inertia (shoulder locked, so cross-inertia terms drop out)
    this.inertia = m2 * l2g * l2g + i2;
    this.c_viscosity = viscosity;
    // elbow pivot position in world space (constant)
    this._elbX = l1 * Math.cos(shoulderAngle);
    this._elbY = l1 * Math.sin(shoulderAngle);
  }

  // inputs[0] = elbow generalized force (torque from muscles).
  // Returns [elbow_angular_acceleration].
  ode(inputs, jointState, endpointLoad) {
    const elbVel = jointState[1];
    const totalAngle = this.shoulderAngle + jointState[0];
    const s = Math.sin(totalAngle);
    const c = Math.cos(totalAngle);
    // Jacobian column for the elbow DOF: d(hand_xy)/d(elbow_angle)
    const jx = -this.L2 * s;
    const jy = this.L2 * c;
    const tau = inputs[0] + jx * endpointLoad[0] + jy * endpointLoad[1];
    return [(tau - this.c_viscosity * elbVel) / this.inertia];
  }

  integrate(dt, stateDerivative, jointState) {
    let newVel = [jointState[1] + stateDerivative[0] * dt];
    let newPos = [jointState[0] + jointState[1] * dt];
    newVel = this.clipVelocity(newPos, newVel);
    newPos = this.clipPosition(newPos);
    return [...newPos, ...newVel];
  }

  // Returns [x, y, vx, vy] of the fingertip (hand).
  joint2cartesian(jointState) {
    const totalAngle = this.shoulderAngle + jointState[0];
    const elbVel = jointState[1];
    const x = this._elbX + this.L2 * Math.cos(totalAngle);
    const y = this._elbY + this.L2 * Math.sin(totalAngle);
    const vx = -this.L2 * Math.sin(totalAngle) * elbVel;
    const vy = this.L2 * Math.cos(totalAngle) * elbVel;
    return [x, y, vx, vy];
  }

  path2cartesian() {
    throw new Error('LockedShoulderArm: path muscles are not supported; use analytical _getGeometry in the effector.');
  }
}
