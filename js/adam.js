// adam.js — Adam over named parameter groups; state matches the params' type.

export class Adam {
  /** @param {Record<string, Float32Array|Float64Array>} params @param {Record<string, {lr: number}>} groups */
  constructor(params, groups, { beta1 = 0.9, beta2 = 0.999, eps = 1e-8 } = {}) {
    this.groups = groups;
    this.beta1 = beta1;
    this.beta2 = beta2;
    this.eps = eps;
    this.t = 0;
    this.state = new Map(
      Object.entries(params)
        .filter(([name]) => groups[name])
        .map(([name, values]) => [
          name,
          { m: new values.constructor(values.length), v: new values.constructor(values.length) },
        ]),
    );
  }

  /** @param {Record<string, Float32Array|Float64Array>} params @param {Record<string, Float32Array|Float64Array>} grads */
  step(params, grads) {
    this.t += 1;
    const { beta1, beta2, eps } = this;
    const c1 = 1 - beta1;
    const c2 = 1 - beta2;
    const invBc2 = 1 / (1 - Math.pow(beta2, this.t));
    for (const name of Object.keys(params)) {
      const g = grads[name];
      const p = params[name];
      if (!g || !this.groups[name]) continue;
      const { m, v } = this.state.get(name);
      const a = this.groups[name].lr / (1 - Math.pow(beta1, this.t));
      for (let i = 0; i < p.length; i++) {
        const gi = g[i];
        const mi = m[i] = beta1 * m[i] + c1 * gi;
        const vi = v[i] = beta2 * v[i] + c2 * gi * gi;
        p[i] -= a * mi / (Math.sqrt(vi * invBc2) + eps);
      }
    }
  }
}
