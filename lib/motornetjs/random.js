// Seedable pseudo-random number generator with Gaussian sampling.
//
// motornet (Python) relies on NumPy's Generator for the i.i.d. Gaussian noise
// it injects into actions and observations. The browser has no equivalent, so
// this module provides a small, deterministic generator. Note that a fixed seed
// will NOT reproduce NumPy's exact stream -- it only guarantees reproducibility
// within this JavaScript implementation.

export class RNG {
  constructor(seed = null) {
    // 32-bit state for mulberry32
    this._state = (seed === null ? (Math.random() * 2 ** 32) >>> 0 : seed >>> 0);
    this._spare = null; // cached value for Box-Muller
  }

  // Uniform float in [0, 1).
  uniform() {
    let t = (this._state += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  // Standard normal sample (Box-Muller, with caching of the spare value).
  normal(mean = 0, std = 1) {
    if (this._spare !== null) {
      const v = this._spare;
      this._spare = null;
      return mean + std * v;
    }
    let u1 = 0;
    let u2 = 0;
    while (u1 === 0) u1 = this.uniform(); // avoid log(0)
    u2 = this.uniform();
    const mag = Math.sqrt(-2 * Math.log(u1));
    this._spare = mag * Math.sin(2 * Math.PI * u2);
    return mean + std * mag * Math.cos(2 * Math.PI * u2);
  }
}
