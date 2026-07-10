// Environment: a JavaScript port of motornet/environment.py.
//
// Wraps an Effector to produce gym-style observations and to manage the task
// goal, sensory delays, and noise. Batch size is fixed to 1 in this port.

import { RNG } from './random.js';

export class Environment {
  constructor({
    effector, qInit = null, name = 'Env', maxEpDuration = 1,
    actionNoise = 0, obsNoise = 0, actionFrameStacking = 0,
    proprioceptionDelay = null, visionDelay = null,
    proprioceptionNoise = 0, visionNoise = 0, rng = null,
  }) {
    this.name = name;
    this.effector = effector;
    this.rng = rng || effector.rng || new RNG();
    this.effector.rng = this.rng;
    this.dt = effector.dt;
    this.maxEpDuration = maxEpDuration;
    this.elapsed = null;

    this.qInit = qInit;
    this.goal = null;

    this._actionNoise = actionNoise;
    this._obsNoise = obsNoise;
    this.proprioceptionNoise = proprioceptionNoise;
    this.visionNoise = visionNoise;
    this.actionFrameStacking = actionFrameStacking;

    const pDelay = proprioceptionDelay === null ? this.dt : proprioceptionDelay;
    const vDelay = visionDelay === null ? this.dt : visionDelay;
    this.proprioceptionDelay = Math.round(pDelay / this.dt);
    this.visionDelay = Math.round(vDelay / this.dt);

    this.obsBuffer = {
      proprioception: new Array(this.proprioceptionDelay).fill(null),
      vision: new Array(this.visionDelay).fill(null),
      action: new Array(this.actionFrameStacking).fill(null),
    };

    this._buildSpaces();
  }

  get muscle() { return this.effector.muscle; }
  get skeleton() { return this.effector.skeleton; }
  get nMuscles() { return this.effector.nMuscles; }
  get spaceDim() { return this.effector.spaceDim; }
  get states() { return this.effector.states; }

  _getObsSize() {
    const goal = this.skeleton.spaceDim;
    const vision = this.skeleton.spaceDim;
    const proprioception = 2 * this.nMuscles;
    const action = this.nMuscles * this.actionFrameStacking;
    return goal + vision + proprioception + action;
  }

  _buildSpaces() {
    this.actionSpaceSize = this.nMuscles;
    this.observationSpaceSize = this._getObsSize();
    const toArr = (noise, n) => (Array.isArray(noise) ? noise : new Array(n).fill(noise));
    this.actionNoise = toArr(this._actionNoise, this.actionSpaceSize);
    this.obsNoise = toArr(this._obsNoise, this.observationSpaceSize);
  }

  // Normalized muscle length followed by normalized muscle velocity per muscle.
  getProprioception() {
    const mlen = this.states.muscle[1].map((v, m) => v / this.muscle.l0_ce[m]);
    const mvel = this.states.muscle[2].map((v, m) => v / this.muscle.vmax[m]);
    return this.applyNoise([...mlen, ...mvel], new Array(2 * this.nMuscles).fill(this.proprioceptionNoise));
  }

  // Cartesian position of the endpoint (fingertip).
  getVision() {
    return this.applyNoise(this.states.fingertip.slice(), new Array(this.spaceDim).fill(this.visionNoise));
  }

  getObs(action = null, deterministic = false) {
    this.updateObsBuffer(action);
    let obs = [
      ...this.goal,
      ...this.obsBuffer.vision[0],
      ...this.obsBuffer.proprioception[0],
    ];
    for (let i = 0; i < this.actionFrameStacking; i++) obs = obs.concat(this.obsBuffer.action[i]);
    if (!deterministic) obs = this.applyNoise(obs, this.obsNoise);
    return obs;
  }

  updateObsBuffer(action = null) {
    this.obsBuffer.proprioception = [...this.obsBuffer.proprioception.slice(1), this.getProprioception()];
    this.obsBuffer.vision = [...this.obsBuffer.vision.slice(1), this.getVision()];
    if (action !== null && this.actionFrameStacking > 0) {
      this.obsBuffer.action = [...this.obsBuffer.action.slice(1), action.slice()];
    }
  }

  applyNoise(loc, noise) {
    return loc.map((v, i) => v + (noise[i] ? this.rng.normal(0, noise[i]) : 0));
  }

  step(action, { deterministic = false, endpointLoad = null, jointLoad = null } = {}) {
    this.elapsed += this.dt;
    let noisyAction = action;
    if (!deterministic) noisyAction = this.applyNoise(action, this.actionNoise);
    this.effector.step(noisyAction, { endpointLoad, jointLoad });

    const obs = this.getObs(noisyAction);
    const terminated = this.elapsed >= this.maxEpDuration;
    const info = {
      states: this.states,
      action,
      noisyAction,
      goal: this.goal.slice(),
    };
    return { obs, reward: null, terminated, truncated: false, info };
  }

  reset({ batchSize = 1, jointState = null, deterministic = false } = {}) {
    const js = jointState ?? this.qInit;
    this.goal = new Array(this.skeleton.spaceDim).fill(0);
    this.elapsed = 0;
    this.effector.reset({ batchSize, jointState: js });

    const action = new Array(this.actionSpaceSize).fill(0);
    this.obsBuffer.proprioception = this.obsBuffer.proprioception.map(() => this.getProprioception());
    this.obsBuffer.vision = this.obsBuffer.vision.map(() => this.getVision());
    this.obsBuffer.action = this.obsBuffer.action.map(() => action.slice());

    const obs = this.getObs(null, deterministic);
    const info = { states: this.states, action, noisyAction: action, goal: this.goal.slice() };
    return { obs, info };
  }

  joint2cartesian(jointState) {
    return this.effector.joint2cartesian(jointState);
  }
}

export class RandomTargetReach extends Environment {
  constructor(opts) {
    super(opts);
    // target info is noiseless
    for (let i = 0; i < this.skeleton.spaceDim; i++) this.obsNoise[i] = 0;
  }

  reset({ batchSize = 1, jointState = null, deterministic = false } = {}) {
    const js = jointState ?? this.qInit;
    this.effector.reset({ batchSize, jointState: js });

    // random reachable fingertip position as the goal
    const randomJoint = this.effector.drawRandomUniformStates(1);
    this.goal = this.joint2cartesian(randomJoint).slice(0, this.skeleton.spaceDim);
    this.elapsed = 0;

    const action = new Array(this.actionSpaceSize).fill(0);
    this.obsBuffer.proprioception = this.obsBuffer.proprioception.map(() => this.getProprioception());
    this.obsBuffer.vision = this.obsBuffer.vision.map(() => this.getVision());
    this.obsBuffer.action = this.obsBuffer.action.map(() => action.slice());

    const obs = this.getObs(null, deterministic);
    const info = { states: this.states, action, noisyAction: action, goal: this.goal.slice() };
    return { obs, info };
  }
}
