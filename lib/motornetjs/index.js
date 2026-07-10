// motornet.js — a browser/JavaScript port of the MotorNet simulation engine.
//
// This is a faithful forward-simulation port of the Python toolbox
// (https://github.com/OlivierCodol/MotorNet): skeletons, muscles, effectors,
// environments, and a GRU policy for inference. Training (back-propagation
// through the simulation) is not included — train in PyTorch, then load the
// resulting weights here with PolicyGRU.loadWeights to run trained controllers
// in the browser.

export { Skeleton, PointMass, TwoDofArm, LockedShoulderArm } from './skeleton.js';
export {
  Muscle, ReluMuscle, MujocoHillMuscle, RigidTendonHillMuscle,
  RigidTendonHillMuscleThelen, CompliantTendonHillMuscle,
} from './muscle.js';
export {
  Effector, ReluPointMass24, FreePointMass24, Reacher,
  RigidTendonArm26, CompliantTendonArm26,
  RigidTendonElbow13, CompliantTendonElbow13,
} from './effector.js';
export { Environment, RandomTargetReach } from './environment.js';
export { PolicyGRU } from './policy.js';
export { RNG } from './random.js';

// Namespaced bundles mirroring the Python module layout (mn.effector.*, etc.).
import * as skeleton from './skeleton.js';
import * as muscle from './muscle.js';
import * as effector from './effector.js';
import * as environment from './environment.js';
import * as policy from './policy.js';

export { skeleton, muscle, effector, environment, policy };
