// reach-evolve.js
// ---------------------------------------------------------------------------
// An in-browser evolutionary optimiser for the four-muscle reaching controller.
// It searches the four task-space impedance gains and the 4x2 distribution
// matrix W (12 parameters) for SMOOTH and ACCURATE reaching, using the exact
// deployed ReachController in the loop — so whatever it finds drops straight into
// the live simulation.
//
// This is a JavaScript port of the (mu, lambda) hill-climb evolution used in the
// musclesim project (arm2dof/gto_reaching_optimize_2dof.py and
// gto_reaching_joint_tune_w_2dof.py): each generation perturbs the current best
// with shrinking Gaussian noise, evaluates every candidate on a multi-target
// reach, and keeps the best. Evaluation runs the whole controller cascade
// (delays and all), so accuracy is measured on the real closed loop and
// smoothness is measured from the real activation and endpoint traces.

import { ReachController, taskFk, DT, M, Q1_0, Q2_0 } from './reach-controller.js';

// 12-parameter vector layout: 4 gains then the 8 W entries (row-major, 4x2).
export const PARAM_NAMES = ['KP_R', 'KD_R', 'KP_PHI', 'KD_PHI'];
export const N_GAINS = 4;
export const N_PARAMS = N_GAINS + 2 * M;

// search bounds and a per-parameter exploration floor (min perturbation scale)
const LO = [40, 2, 2, 0.2, ...Array(2 * M).fill(-1.5)];
const HI = [800, 200, 80, 20, ...Array(2 * M).fill(1.5)];
const FLOOR = [10, 3, 0.6, 0.1, ...Array(2 * M).fill(0.06)];

// ---- loss weights ----------------------------------------------------------
const T_REACH = 1.8;             // seconds allowed per reach (home -> target)
const NR = Math.round(T_REACH / DT);
const PRERUN = 150;              // settle steps at home before each reach
const SETTLE = Math.round(0.30 / DT);   // last 0.30 s scores settled accuracy
const W_EFF = 0.05;              // control-smoothness: RMS activation rate (per cm)
const W_JERK = 0.4;              // path-smoothness: RMS endpoint acceleration
const W_DIV = 6.0;              // divergence penalty above 1.5 cm settled error
const W_MAX = 0.6;              // weight on the worst target (robustness)

export function paramsToVector(p) {
  return [p.KP_R, p.KD_R, p.KP_PHI, p.KD_PHI, ...p.W.flat()];
}

export function vectorToParams(v) {
  const W = [];
  for (let m = 0; m < M; m++) W.push([v[N_GAINS + 2 * m], v[N_GAINS + 2 * m + 1]]);
  return { KP_R: v[0], KD_R: v[1], KP_PHI: v[2], KD_PHI: v[3], W };
}

// Apply a candidate vector onto a controller's parameter block, in place.
export function applyVector(controller, v) {
  const p = vectorToParams(v);
  controller.p.KP_R = p.KP_R; controller.p.KD_R = p.KD_R;
  controller.p.KP_PHI = p.KP_PHI; controller.p.KD_PHI = p.KD_PHI;
  controller.p.W = p.W;
}

// A grid of reach targets spanning the reliably reachable annulus/arc around the
// home posture. Evaluating over the whole grid (not one trajectory) forces the
// evolved gains to hold accuracy across the workspace, not just near home.
export function buildTargets(r0, phi0) {
  const targets = [];
  for (const r of [0.40, 0.47, 0.54, 0.60]) {
    for (const dphi of [-0.45, -0.20, 0, 0.20, 0.45]) targets.push([r, phi0 + dphi]);
  }
  return targets;
}

// A smaller grid for the interactive in-browser optimiser, so a live "Evolve"
// run finishes in a few seconds.
export function buildTargetsCoarse(r0, phi0) {
  const targets = [];
  for (const r of [0.44, 0.54, 0.60]) {
    for (const dphi of [-0.40, 0, 0.40]) targets.push([r, phi0 + dphi]);
  }
  return targets;
}

// Run one home->target reach with the deployed controller; return its metrics.
function simulateReach(c, v, rt, pt) {
  c.reset();
  applyVector(c, v);
  for (let i = 0; i < PRERUN; i++) c.advance();
  c.setTargetTask(rt, pt);
  const tx = rt * Math.cos(pt), ty = rt * Math.sin(pt);
  const x = new Float64Array(NR), y = new Float64Array(NR);
  const aPrev = c.a.slice();
  let effortSq = 0, jerkSq = 0, errSq = 0, nErr = 0, finite = true;
  for (let i = 0; i < NR; i++) {
    const rec = c.advance();
    x[i] = rec.x; y[i] = rec.y;
    if (!isFinite(rec.x) || !isFinite(rec.y)) { finite = false; break; }
    for (let m = 0; m < M; m++) { const d = (c.a[m] - aPrev[m]) / DT; effortSq += d * d; aPrev[m] = c.a[m]; }
    if (i >= 2) {
      const ax = (x[i] - 2 * x[i - 1] + x[i - 2]) / (DT * DT);
      const ay = (y[i] - 2 * y[i - 1] + y[i - 2]) / (DT * DT);
      jerkSq += ax * ax + ay * ay;
    }
    if (i >= NR - SETTLE) { const dx = rec.x - tx, dy = rec.y - ty; errSq += dx * dx + dy * dy; nErr++; }
  }
  if (!finite) return { errCm: 1e3, effort: 0, jerk: 0, finite: false };
  return {
    errCm: 100 * Math.sqrt(errSq / Math.max(1, nErr)),
    effort: Math.sqrt(effortSq / (NR * M)),
    jerk: Math.sqrt(jerkSq / Math.max(1, NR - 2)),
    finite: true,
  };
}

// Evaluate a candidate vector over the whole target grid: returns { loss, comp }.
export function evaluate(v, r0, phi0, targets) {
  const c = new ReachController();
  let sumErr = 0, sumEff = 0, sumJerk = 0, sumDiv = 0, maxErr = 0, anyBad = false;
  for (const [rt, pt] of targets) {
    const m = simulateReach(c, v, rt, pt);
    if (!m.finite) anyBad = true;
    sumErr += m.errCm; sumEff += m.effort; sumJerk += m.jerk;
    sumDiv += Math.max(0, m.errCm - 1.5);
    maxErr = Math.max(maxErr, m.errCm);
  }
  const n = targets.length;
  const errCm = sumErr / n, effort = sumEff / n, jerk = sumJerk / n, div = sumDiv / n;
  const loss = errCm + W_EFF * effort + W_JERK * jerk + W_DIV * div + W_MAX * maxErr
    + (anyBad ? 1e6 : 0);
  return { loss, comp: { errCm, maxErrCm: maxErr, effort, jerk } };
}

// Simple deterministic Gaussian (Box–Muller) over a seeded LCG, so a given seed
// reproduces the same evolution in the browser and in Node.
function makeRng(seed) {
  let s = (seed >>> 0) || 1;
  const next = () => { s = (1664525 * s + 1013904223) >>> 0; return s / 4294967296; };
  return () => {
    let u = 0, v = 0;
    while (u === 0) u = next();
    while (v === 0) v = next();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
}

const clampVec = (v) => v.map((x, i) => Math.min(HI[i], Math.max(LO[i], x)));

// (mu, lambda) hill-climb. `onGen(gen, best)` is called after each generation
// (for a live progress read-out). Returns { vector, loss, comp }.
export function evolve({ seedVector, gens = 16, pop = 16, seed = 0, onGen = null } = {}) {
  const gauss = makeRng(seed);
  const g0 = taskFk(Q1_0, Q2_0, 0, 0, ...seedArmLengths());
  const r0 = g0.r, phi0 = g0.phi;
  const targets = buildTargets(r0, phi0);

  let best = clampVec(seedVector.slice());
  let { loss: bestLoss, comp: bestComp } = evaluate(best, r0, phi0, targets);
  let scale = 0.35;
  for (let gen = 0; gen < gens; gen++) {
    let genBest = best, genLoss = bestLoss, genComp = bestComp;
    for (let k = 0; k < pop; k++) {
      const cand = (k === 0) ? best.slice()
        : clampVec(best.map((x, i) => x + gauss() * scale * Math.max(Math.abs(x), FLOOR[i])));
      const { loss, comp } = evaluate(cand, r0, phi0, targets);
      if (loss < genLoss) { genBest = cand; genLoss = loss; genComp = comp; }
    }
    if (genLoss < bestLoss) { best = genBest; bestLoss = genLoss; bestComp = genComp; }
    scale *= 0.88;
    if (onGen) onGen(gen, { vector: best, loss: bestLoss, comp: bestComp });
  }
  return { vector: best, loss: bestLoss, comp: bestComp };
}

// Async (browser) variant: seeds from the current deployed parameters and
// polishes them on the coarse grid, awaiting `yieldFn()` between generations so
// the page stays responsive. `onGen(gen, state)` drives the live read-out.
export async function evolveAsync({ seedVector, gens = 10, pop = 12, seed = 0,
  onGen = null, yieldFn = null } = {}) {
  const gauss = makeRng(seed);
  const g0 = taskFk(Q1_0, Q2_0, 0, 0, ...seedArmLengths());
  const r0 = g0.r, phi0 = g0.phi;
  const targets = buildTargetsCoarse(r0, phi0);

  let best = clampVec(seedVector.slice());
  let { loss: bestLoss, comp: bestComp } = evaluate(best, r0, phi0, targets);
  const startLoss = bestLoss;
  let scale = 0.25;                 // start small: polishing an already-good seed
  for (let gen = 0; gen < gens; gen++) {
    let genBest = best, genLoss = bestLoss, genComp = bestComp;
    for (let k = 0; k < pop; k++) {
      const cand = (k === 0) ? best.slice()
        : clampVec(best.map((x, i) => x + gauss() * scale * Math.max(Math.abs(x), FLOOR[i])));
      const { loss, comp } = evaluate(cand, r0, phi0, targets);
      if (loss < genLoss) { genBest = cand; genLoss = loss; genComp = comp; }
    }
    if (genLoss < bestLoss) { best = genBest; bestLoss = genLoss; bestComp = genComp; }
    scale *= 0.9;
    if (onGen) onGen(gen, { gens, vector: best, loss: bestLoss, comp: bestComp, startLoss });
    if (yieldFn) await yieldFn();
  }
  return { vector: best, loss: bestLoss, comp: bestComp, startLoss };
}

// Arm link lengths without holding onto a controller (cheap, one-off).
let _armLens = null;
function seedArmLengths() {
  if (!_armLens) { const c = new ReachController(); _armLens = [c.L1, c.L2]; }
  return _armLens;
}
