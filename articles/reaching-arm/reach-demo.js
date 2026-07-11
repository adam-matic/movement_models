// reach-demo.js — interactive front-end for the model-free reaching controller.
import { ReachController, taskIk, taskFk, MUSCLE_NAMES, DEFAULTS, M, DT } from './reach-controller.js';
import { evolveAsync, paramsToVector, applyVector } from './reach-evolve.js';

const controller = new ReachController();
const L1 = controller.L1, L2 = controller.L2;

// ---- canvas + camera ------------------------------------------------------
const canvas = document.getElementById('arm_canvas');
let camera = null;
function fitCanvas() {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.parentElement.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  canvas.style.width = rect.width + 'px';
  canvas.style.height = rect.height + 'px';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  makeCamera(rect.width, rect.height);
  return [ctx, rect.width, rect.height];
}
function makeCamera(W, H) {
  const xmin = -0.42, xmax = 0.55, ymin = -0.08, ymax = 0.68;
  const margin = 12;
  const s = Math.min((W - 2 * margin) / (xmax - xmin), (H - 2 * margin) / (ymax - ymin));
  const ox = (W - s * (xmax - xmin)) / 2;
  const oy = (H - s * (ymax - ymin)) / 2;
  camera = {
    s,
    toScreen: (x, y) => [ox + (x - xmin) * s, H - (oy + (y - ymin) * s)],
    toWorld: (px, py) => [xmin + (px - ox) / s, ymin + (H - py - oy) / s],
  };
}

// endpoint trail
const trail = [];
const TRAIL_MAX = 260;

function draw(ctx, W, H) {
  ctx.clearRect(0, 0, W, H);

  // reachable annulus (light guide)
  const base = camera.toScreen(0, 0);
  ctx.fillStyle = '#eef2fb';
  ctx.beginPath();
  ctx.arc(base[0], base[1], (L1 + L2) * camera.s, 0, 2 * Math.PI);
  ctx.arc(base[0], base[1], Math.abs(L1 - L2) * camera.s, 0, 2 * Math.PI, true);
  ctx.fill('evenodd');

  // target crosshair (from the commanded task target)
  const tx = controller.rTarget * Math.cos(controller.phiTarget);
  const ty = controller.rTarget * Math.sin(controller.phiTarget);
  const tp = camera.toScreen(tx, ty);
  ctx.strokeStyle = '#e07b00'; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(tp[0], tp[1], 11, 0, 2 * Math.PI); ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(tp[0] - 16, tp[1]); ctx.lineTo(tp[0] + 16, tp[1]);
  ctx.moveTo(tp[0], tp[1] - 16); ctx.lineTo(tp[0], tp[1] + 16); ctx.stroke();

  // endpoint trail
  if (trail.length > 1) {
    ctx.strokeStyle = 'rgba(32,82,187,0.35)'; ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < trail.length; i++) {
      const p = camera.toScreen(trail[i][0], trail[i][1]);
      i === 0 ? ctx.moveTo(p[0], p[1]) : ctx.lineTo(p[0], p[1]);
    }
    ctx.stroke();
  }

  // arm links
  const j = controller.joints;
  const ex = L1 * Math.cos(j[0]), ey = L1 * Math.sin(j[0]);
  const hx = ex + L2 * Math.cos(j[0] + j[1]), hy = ey + L2 * Math.sin(j[0] + j[1]);
  const p0 = camera.toScreen(0, 0), p1 = camera.toScreen(ex, ey), p2 = camera.toScreen(hx, hy);
  ctx.lineCap = 'round';
  ctx.strokeStyle = '#c7cfdd'; ctx.lineWidth = 16;
  ctx.beginPath(); ctx.moveTo(...p0); ctx.lineTo(...p1); ctx.lineTo(...p2); ctx.stroke();
  ctx.strokeStyle = '#2052BB'; ctx.lineWidth = 8;
  ctx.beginPath(); ctx.moveTo(...p0); ctx.lineTo(...p1); ctx.lineTo(...p2); ctx.stroke();
  for (const [p, c, rad] of [[p0, '#8a94a6', 8], [p1, '#dfe4ee', 7], [p2, '#e07b00', 8]]) {
    ctx.fillStyle = c; ctx.beginPath(); ctx.arc(p[0], p[1], rad, 0, 2 * Math.PI); ctx.fill();
  }
}

// ---- muscle bars ----------------------------------------------------------
const barsWrap = document.getElementById('muscle_bars');
const bars = [];
MUSCLE_NAMES.forEach((name) => {
  const row = document.createElement('div'); row.className = 'mbar-row';
  const lab = document.createElement('span'); lab.className = 'mbar-name'; lab.textContent = name;
  const track = document.createElement('div'); track.className = 'mbar-track';
  const fill = document.createElement('div'); fill.className = 'mbar-fill';
  track.appendChild(fill); row.appendChild(lab); row.appendChild(track);
  barsWrap.appendChild(row); bars.push(fill);
});

// ---- parameter sliders ----------------------------------------------------
// [id, param-key, min, max, step, format, scale-to-param]
const SLIDERS = [
  ['kp_r', 'KP_R', 20, 400, 1, (v) => v.toFixed(0)],
  ['kd_r', 'KD_R', 0, 90, 0.5, (v) => v.toFixed(1)],
  ['kp_phi', 'KP_PHI', 1, 40, 0.1, (v) => v.toFixed(1)],
  ['kd_phi', 'KD_PHI', 0, 8, 0.05, (v) => v.toFixed(2)],
  ['leak', 'LEAK', 0, 0.2, 0.005, (v) => v.toFixed(3)],
  ['k_co', 'K_CO', 0, 0.05, 0.001, (v) => v.toFixed(3)],
  ['prop', 'PROP_DELAY_MS', 0, 60, 1, (v) => v.toFixed(0) + ' ms'],
  ['gto', 'GTO_DELAY_MS', 0, 60, 1, (v) => v.toFixed(0) + ' ms'],
  ['spn', 'SPN_DELAY_MS', 0, 60, 1, (v) => v.toFixed(0) + ' ms'],
];
const sliderEls = {};
SLIDERS.forEach(([id, key, , , , fmt]) => {
  const input = document.getElementById('sl_' + id);
  const out = document.getElementById('val_' + id);
  sliderEls[id] = { input, out, key, fmt };
  input.value = controller.p[key];
  out.textContent = fmt(controller.p[key]);
  input.addEventListener('input', () => {
    const v = parseFloat(input.value);
    controller.p[key] = v;
    out.textContent = fmt(v);
  });
});

// ---- reference controls ---------------------------------------------------
const rRefSl = document.getElementById('sl_rref');
const rRefVal = document.getElementById('val_rref');
const phiRefSl = document.getElementById('sl_phiref');
const phiRefVal = document.getElementById('val_phiref');
const cRefSl = document.getElementById('sl_cref');
const cRefVal = document.getElementById('val_cref');

function syncRefSlidersFromTarget() {
  rRefSl.value = controller.rTarget;
  rRefVal.textContent = (controller.rTarget * 100).toFixed(1) + ' cm';
  phiRefSl.value = controller.phiTarget * 180 / Math.PI;
  phiRefVal.textContent = (controller.phiTarget * 180 / Math.PI).toFixed(0) + '°';
}
rRefSl.addEventListener('input', () => {
  controller.setTargetTask(parseFloat(rRefSl.value), controller.phiTarget);
  rRefVal.textContent = (parseFloat(rRefSl.value) * 100).toFixed(1) + ' cm';
});
phiRefSl.addEventListener('input', () => {
  controller.setTargetTask(controller.rTarget, parseFloat(phiRefSl.value) * Math.PI / 180);
  phiRefVal.textContent = parseFloat(phiRefSl.value).toFixed(0) + '°';
});
cRefSl.addEventListener('input', () => {
  controller.p.C_REF = parseFloat(cRefSl.value);
  cRefVal.textContent = parseFloat(cRefSl.value).toFixed(0) + ' N';
});
// initialise reference slider ranges from the reachable workspace
rRefSl.min = (Math.abs(L1 - L2) + 0.02).toFixed(3);
rRefSl.max = (L1 + L2 - 0.02).toFixed(3);
syncRefSlidersFromTarget();
cRefVal.textContent = controller.p.C_REF.toFixed(0) + ' N';

// click / drag on the canvas sets the Cartesian target
let dragging = false;
function setTargetFromEvent(e) {
  const rect = canvas.getBoundingClientRect();
  const px = (e.clientX ?? e.touches[0].clientX) - rect.left;
  const py = (e.clientY ?? e.touches[0].clientY) - rect.top;
  const [x, y] = camera.toWorld(px, py);
  controller.setTargetCartesian(x, y);
  syncRefSlidersFromTarget();
}
canvas.addEventListener('pointerdown', (e) => { dragging = true; setTargetFromEvent(e); });
canvas.addEventListener('pointermove', (e) => { if (dragging) setTargetFromEvent(e); });
window.addEventListener('pointerup', () => { dragging = false; });

// ---- buttons --------------------------------------------------------------
document.getElementById('btn_reset').addEventListener('click', () => {
  controller.reset(); trail.length = 0; syncRefSlidersFromTarget();
});
function refreshSliders() {
  SLIDERS.forEach(([id]) => {
    const el = sliderEls[id];
    el.input.value = controller.p[el.key];
    el.out.textContent = el.fmt(controller.p[el.key]);
  });
}
document.getElementById('btn_defaults').addEventListener('click', () => {
  Object.assign(controller.p, { ...DEFAULTS, W: DEFAULTS_W });
  refreshSliders();
  controller.p.C_REF = 0; cRefSl.value = 0; cRefVal.textContent = '0 N';
  evoStatus.textContent = 'defaults restored';
});
// keep a copy of the deployed evolved W so "Restore defaults" can undo an evolve
const DEFAULTS_W = controller.p.W.map((r) => r.slice());

// ---- evolutionary optimiser (in-browser) ----------------------------------
const evoBtn = document.getElementById('btn_evolve');
const evoStatus = document.getElementById('evo_status');
let evolving = false;
evoBtn.addEventListener('click', async () => {
  if (evolving) return;
  evolving = true;
  evoBtn.disabled = true;
  const seedVector = paramsToVector(controller.p);
  const raf = () => new Promise((res) => requestAnimationFrame(res));
  const res = await evolveAsync({
    seedVector, gens: 10, pop: 12, seed: (Math.random() * 1e9) | 0,
    yieldFn: raf,
    onGen: (gen, st) => {
      evoStatus.textContent =
        `evolving… gen ${gen + 1}/${st.gens} · loss ${st.startLoss.toFixed(2)} → ${st.loss.toFixed(2)} · `
        + `mean ${st.comp.errCm.toFixed(2)} cm, max ${st.comp.maxErrCm.toFixed(2)} cm`;
    },
  });
  applyVector(controller, res.vector);       // deploy the evolved parameters
  refreshSliders();
  evoStatus.textContent =
    `evolved · loss ${res.startLoss.toFixed(2)} → ${res.loss.toFixed(2)} · `
    + `settled mean ${res.comp.errCm.toFixed(2)} cm, max ${res.comp.maxErrCm.toFixed(2)} cm, `
    + `smoothness ${res.comp.jerk.toFixed(2)}`;
  evoBtn.disabled = false;
  evolving = false;
});
let paused = false;
const pauseBtn = document.getElementById('btn_pause');
pauseBtn.addEventListener('click', () => { paused = !paused; pauseBtn.textContent = paused ? '▶ Run' : '⏸ Pause'; });

// ---- diagnostics readout --------------------------------------------------
const diag = document.getElementById('diag');

// ---- main loop ------------------------------------------------------------
let ctx, W, H;
function resize() { [ctx, W, H] = fitCanvas(); }
let acc = 0, last = performance.now();
function frame(now) {
  const dtReal = Math.min((now - last) / 1000, 0.1); last = now;
  if (!paused) {
    acc += dtReal;
    let n = 0;
    while (acc >= DT && n < 60) {   // up to 120 ms of sim per frame
      const rec = controller.advance();
      trail.push([rec.x, rec.y]);
      if (trail.length > TRAIL_MAX) trail.shift();
      acc -= DT; n++;
    }
  }
  draw(ctx, W, H);
  // bars
  controller.a.forEach((a, i) => { bars[i].style.width = (Math.min(1, a) * 100).toFixed(1) + '%'; });
  // diagnostics
  const rec = controller.rec;
  const errCm = Math.hypot(rec.x - controller.rTarget * Math.cos(controller.phiTarget),
    rec.y - controller.rTarget * Math.sin(controller.phiTarget)) * 100;
  diag.innerHTML =
    `reach r = ${(rec.r * 100).toFixed(1)} cm &nbsp;·&nbsp; point φ = ${(rec.phi * 180 / Math.PI).toFixed(1)}° ` +
    `&nbsp;·&nbsp; endpoint error = ${errCm.toFixed(2)} cm &nbsp;·&nbsp; Σ tendon force = ${rec.cocon.toFixed(0)} N`;
  requestAnimationFrame(frame);
}

let resizeTimer;
window.addEventListener('resize', () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(resize, 100); });
resize();
requestAnimationFrame(frame);
