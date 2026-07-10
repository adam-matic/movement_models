// PolicyGRU: a JavaScript port of motornet/policy.py.
//
// A single-layer GRU followed by a sigmoid-activated linear readout. This port
// implements the forward pass for inference (running a trained controller in the
// browser); it does not implement back-propagation/training. Weights trained in
// PyTorch can be loaded with `loadWeights`, using the exact parameter names from
// `torch.nn.GRU` / `torch.nn.Linear`.

const sigmoid = (x) => 1 / (1 + Math.exp(-x));

export class PolicyGRU {
  constructor(inputDim, hiddenDim, outputDim) {
    this.inputDim = inputDim;
    this.hiddenDim = hiddenDim;
    this.outputDim = outputDim;
    this.nLayers = 1;
    // weights are null until loaded; shapes follow PyTorch conventions
    this.weight_ih = null; // (3*hidden, input), gate order [r, z, n]
    this.weight_hh = null; // (3*hidden, hidden)
    this.bias_ih = null; // (3*hidden)
    this.bias_hh = null; // (3*hidden)
    this.fc_weight = null; // (output, hidden)
    this.fc_bias = null; // (output)
  }

  // Load weights from a plain object using PyTorch parameter names, e.g. the
  // dict produced by `{name: p.detach().numpy().tolist() for name, p in policy.named_parameters()}`.
  loadWeights(w) {
    this.weight_ih = w['gru.weight_ih_l0'];
    this.weight_hh = w['gru.weight_hh_l0'];
    this.bias_ih = w['gru.bias_ih_l0'];
    this.bias_hh = w['gru.bias_hh_l0'];
    this.fc_weight = w['fc.weight'];
    this.fc_bias = w['fc.bias'];
    return this;
  }

  initHidden() {
    return new Array(this.hiddenDim).fill(0);
  }

  // One GRU step for a single (unbatched) observation vector x and hidden state
  // h. Returns { u, h } with the sigmoid-gated motor command u and the new
  // hidden state h.
  forward(x, h) {
    const H = this.hiddenDim;
    const newH = new Array(H);
    for (let j = 0; j < H; j++) {
      // pre-activations for reset (r), update (z), new (n) gates
      let ir = this.bias_ih[j]; let hr = this.bias_hh[j];
      let iz = this.bias_ih[H + j]; let hz = this.bias_hh[H + j];
      let inn = this.bias_ih[2 * H + j]; let hn = this.bias_hh[2 * H + j];
      const wr = this.weight_ih[j]; const wz = this.weight_ih[H + j]; const wn = this.weight_ih[2 * H + j];
      for (let k = 0; k < this.inputDim; k++) {
        ir += wr[k] * x[k];
        iz += wz[k] * x[k];
        inn += wn[k] * x[k];
      }
      const ur = this.weight_hh[j]; const uz = this.weight_hh[H + j]; const un = this.weight_hh[2 * H + j];
      for (let k = 0; k < H; k++) {
        hr += ur[k] * h[k];
        hz += uz[k] * h[k];
        hn += un[k] * h[k];
      }
      const r = sigmoid(ir + hr);
      const z = sigmoid(iz + hz);
      const n = Math.tanh(inn + r * hn);
      newH[j] = (1 - z) * n + z * h[j];
    }

    const u = new Array(this.outputDim);
    for (let o = 0; o < this.outputDim; o++) {
      let acc = this.fc_bias[o];
      const row = this.fc_weight[o];
      for (let j = 0; j < H; j++) acc += row[j] * newH[j];
      u[o] = sigmoid(acc);
    }
    return { u, h: newH };
  }
}
