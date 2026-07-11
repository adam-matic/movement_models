// muscle-lab.js — interactive front-end for the isometric single-muscle rig.
import { MuscleRig, SpeedController, DT, MUSCLE, MTU_LENGTH } from './muscle-rig.js';

const rig = new MuscleRig();
const ctl = new SpeedController();
let mode = 'manual';          // 'manual' | 'speed'
let manualU = 0;
let paused = false;

// ---- canvases -------------------------------------------------------------
const rigCanvas = document.getElementById('rig_canvas');
const chartCanvas = document.getElementById('chart_canvas');
function fit(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.parentElement.getBoundingClientRect();
  canvas.width = rect.width * dpr; canvas.height = rect.height * dpr;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return [ctx, rect.width, rect.height];
}
let rigCtx, rigW, rigH, chCtx, chW, chH;
function resize() {
  [rigCtx, rigW, rigH] = fit(rigCanvas);
  [chCtx, chW, chH] = fit(chartCanvas);
}

// ---- rig drawing ----------------------------------------------------------
function drawRig() {
  const ctx = rigCtx, W = rigW, H = rigH;
  ctx.clearRect(0, 0, W, H);
  const padX = 34, midY = H * 0.5;
  const x0 = padX, x1 = W - padX;                 // left wall, right wall (fixed)
  const toX = (m) => x0 + (m / MTU_LENGTH) * (x1 - x0);
  const jx = toX(rig.fibreLength);                // muscle–tendon junction

  // fixed walls (hatched)
  ctx.strokeStyle = '#9aa4b2'; ctx.lineWidth = 2;
  for (const wx of [x0, x1]) {
    ctx.beginPath(); ctx.moveTo(wx, midY - 46); ctx.lineTo(wx, midY + 46); ctx.stroke();
    for (let k = -5; k <= 5; k++) {
      const yy = midY + k * 9; const dir = (wx === x0) ? -1 : 1;
      ctx.beginPath(); ctx.moveTo(wx, yy); ctx.lineTo(wx + dir * 8, yy + 8); ctx.stroke();
    }
  }

  // muscle belly (left): thickness grows with activation
  const th = 12 + 26 * Math.min(1, rig.activation);
  const grad = ctx.createLinearGradient(0, midY - th, 0, midY + th);
  grad.addColorStop(0, '#e06666'); grad.addColorStop(0.5, '#cc2b2b'); grad.addColorStop(1, '#e06666');
  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.moveTo(x0, midY - 8);
  ctx.quadraticCurveTo((x0 + jx) / 2, midY - th, jx, midY - 6);
  ctx.lineTo(jx, midY + 6);
  ctx.quadraticCurveTo((x0 + jx) / 2, midY + th, x0, midY + 8);
  ctx.closePath(); ctx.fill();

  // tendon (right): zig-zag spring from junction to right wall
  ctx.strokeStyle = '#c79a3a'; ctx.lineWidth = 3; ctx.lineJoin = 'round';
  ctx.beginPath(); ctx.moveTo(jx, midY);
  const n = 9, span = x1 - jx;
  for (let i = 0; i <= n; i++) {
    const xx = jx + span * i / n;
    const yy = midY + (i === 0 || i === n ? 0 : (i % 2 ? -9 : 9));
    ctx.lineTo(xx, yy);
  }
  ctx.lineTo(x1, midY); ctx.stroke();

  // junction node
  ctx.fillStyle = '#333'; ctx.beginPath(); ctx.arc(jx, midY, 5, 0, 2 * Math.PI); ctx.fill();

  // sensor tags
  ctx.font = '12px IBM Plex Sans, sans-serif'; ctx.textAlign = 'center';
  ctx.fillStyle = '#cc2b2b'; ctx.fillText('spindle (Ia)', (x0 + jx) / 2, midY - th - 8);
  ctx.fillStyle = '#c79a3a'; ctx.fillText('GTO (Ib)', (jx + x1) / 2, midY - 16);

  // length labels
  ctx.fillStyle = '#888'; ctx.font = '11px IBM Plex Sans, sans-serif';
  ctx.fillText('muscle ' + (rig.fibreLength * 1000).toFixed(0) + ' mm', (x0 + jx) / 2, midY + th + 16);
  ctx.fillText('tendon ' + (rig.tendonLength * 1000).toFixed(0) + ' mm', (jx + x1) / 2, midY + 22);
}

// ---- scrolling charts -----------------------------------------------------
const HIST = 1500;            // samples kept (~ HIST * frameStep of sim time)
const hForce = [], hVel = [], hRef = [];
function pushHist() {
  hForce.push(rig.force); hVel.push(-rig.fibreVelocity * 1000);
  hRef.push(mode === 'speed' ? ctl.refShorteningMmps : NaN);
  if (hForce.length > HIST) { hForce.shift(); hVel.shift(); hRef.shift(); }
}
function drawCharts() {
  const ctx = chCtx, W = chW, H = chH;
  ctx.clearRect(0, 0, W, H);
  const gap = 10, h = (H - gap) / 2, padL = 40, padR = 8;
  const pw = W - padL - padR;
  const n = hForce.length;
  const xAt = (i) => padL + (i / (HIST - 1)) * pw;

  // panel 1: force
  const fMax = Math.max(120, ...hForce) * 1.1;
  drawPanel(ctx, 0, h, padL, pw, 'force (N)', 0, fMax, [
    { data: hForce, color: '#c79a3a', width: 2 },
  ], xAt, n);
  // panel 2: velocity (ref dashed + actual)
  drawPanel(ctx, h + gap, h, padL, pw, 'shortening v (mm/s)', -22, 22, [
    { data: hRef, color: '#e07b00', width: 1.5, dash: [5, 4] },
    { data: hVel, color: '#2052BB', width: 2 },
  ], xAt, n, true);
}
function drawPanel(ctx, y0, h, padL, pw, label, ymin, ymax, series, xAt, n, zeroLine) {
  const toY = (v) => y0 + h - (v - ymin) / (ymax - ymin) * h;
  ctx.strokeStyle = '#e6e9f0'; ctx.lineWidth = 1;
  ctx.strokeRect(padL, y0, pw, h);
  if (zeroLine) {
    ctx.strokeStyle = '#d0d5de'; ctx.beginPath();
    ctx.moveTo(padL, toY(0)); ctx.lineTo(padL + pw, toY(0)); ctx.stroke();
  }
  ctx.fillStyle = '#999'; ctx.font = '11px IBM Plex Sans, sans-serif'; ctx.textAlign = 'left';
  ctx.fillText(label, padL + 4, y0 + 12);
  ctx.textAlign = 'right';
  ctx.fillText(ymax.toFixed(0), padL - 4, y0 + 10);
  ctx.fillText(ymin.toFixed(0), padL - 4, y0 + h - 1);
  for (const s of series) {
    ctx.strokeStyle = s.color; ctx.lineWidth = s.width;
    ctx.setLineDash(s.dash || []);
    ctx.beginPath(); let started = false;
    for (let i = 0; i < n; i++) {
      const v = s.data[i]; if (!Number.isFinite(v)) { started = false; continue; }
      const px = xAt(i), py = toY(Math.max(ymin, Math.min(ymax, v)));
      started ? ctx.lineTo(px, py) : ctx.moveTo(px, py); started = true;
    }
    ctx.stroke(); ctx.setLineDash([]);
  }
}

// ---- readouts -------------------------------------------------------------
const ro = (id) => document.getElementById('ro_' + id);
function updateReadouts(u) {
  ro('u').textContent = u.toFixed(2);
  ro('act').textContent = rig.activation.toFixed(2);
  ro('len').textContent = (rig.fibreLength * 1000).toFixed(1) + ' mm';
  ro('ten').textContent = '+' + ((rig.tendonLength - MUSCLE.tendonSlackLength) * 1000).toFixed(1) + ' mm';
  ro('vel').textContent = (-rig.fibreVelocity * 1000).toFixed(1) + ' mm/s';
  ro('force').textContent = rig.force.toFixed(0) + ' N';
}

// ---- controls -------------------------------------------------------------
const primary = document.getElementById('sl_primary');
const primaryVal = document.getElementById('val_primary');
const primaryNm = document.getElementById('primary_nm');
const gainCtrls = document.getElementById('gain_ctrls');
const btnManual = document.getElementById('mode_manual');
const btnSpeed = document.getElementById('mode_speed');

function setMode(m) {
  mode = m;
  btnManual.classList.toggle('on', m === 'manual');
  btnSpeed.classList.toggle('on', m === 'speed');
  gainCtrls.style.display = m === 'speed' ? '' : 'none';
  if (m === 'manual') {
    primaryNm.textContent = 'Excitation';
    primary.min = 0; primary.max = 1; primary.step = 0.01; primary.value = manualU;
    primaryVal.textContent = manualU.toFixed(2);
  } else {
    primaryNm.textContent = 'Target speed';
    primary.min = -12; primary.max = 12; primary.step = 0.5; primary.value = ctl.refShorteningMmps;
    primaryVal.textContent = ctl.refShorteningMmps.toFixed(1) + ' mm/s';
    // bumpless handover: seed the integrator so the controller starts at the
    // muscle's current activation (no output jump when switching from manual).
    ctl.integ = ctl.KI > 0 ? rig.activation / (ctl.KI * 1e-3) : 0;
    ctl.u = rig.activation;
  }
}
primary.addEventListener('input', () => {
  const v = parseFloat(primary.value);
  if (mode === 'manual') { manualU = v; primaryVal.textContent = v.toFixed(2); }
  else { ctl.refShorteningMmps = v; primaryVal.textContent = v.toFixed(1) + ' mm/s'; }
});
btnManual.addEventListener('click', () => setMode('manual'));
btnSpeed.addEventListener('click', () => setMode('speed'));

const bind = (id, key, fmt, scale = 1) => {
  const el = document.getElementById('sl_' + id), out = document.getElementById('val_' + id);
  el.addEventListener('input', () => { ctl[key] = parseFloat(el.value) * scale; out.textContent = fmt(parseFloat(el.value)); });
};
bind('kp', 'KP', (v) => v.toFixed(1));
bind('ki', 'KI', (v) => v.toFixed(0));
bind('spd', 'SPINDLE_DELAY_MS', (v) => v.toFixed(0) + ' ms');
bind('gtod', 'GTO_DELAY_MS', (v) => v.toFixed(0) + ' ms');

document.getElementById('btn_reset').addEventListener('click', () => {
  rig.reset(); ctl.reset(); manualU = 0;
  hForce.length = hVel.length = hRef.length = 0;
  setMode(mode);
});
const pauseBtn = document.getElementById('btn_pause');
pauseBtn.addEventListener('click', () => { paused = !paused; pauseBtn.textContent = paused ? '▶ Run' : '⏸ Pause'; });

// ---- main loop ------------------------------------------------------------
let acc = 0, last = performance.now(), sinceSample = 0;
function frame(now) {
  const dtReal = Math.min((now - last) / 1000, 0.1); last = now;
  let u = mode === 'manual' ? manualU : ctl.u;
  if (!paused) {
    acc += dtReal;
    let steps = 0;
    while (acc >= DT && steps < 40) {   // up to 40 ms of sim per frame
      u = mode === 'manual' ? manualU : ctl.update(rig);
      rig.advance(u);
      acc -= DT; steps++;
      if (++sinceSample >= 3) { pushHist(); sinceSample = 0; }   // ~3 ms/sample
    }
  }
  drawRig(); drawCharts(); updateReadouts(u);
  requestAnimationFrame(frame);
}

let rt; window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(resize, 100); });
resize();
setMode('manual');
requestAnimationFrame(frame);
