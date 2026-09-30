const OP = {
  LEAF: 0, CONST: 1, ADD: 2, SUB: 3, MUL: 4, DIV: 5,
  ADDC: 6, MULC: 7, NEG: 8, SQRT: 9, SIN: 10, COS: 11,
};

// A traced scalar program replayed without allocations.
export class Tape {
  constructor() {
    this.columns = { op: [], a: [], b: [], c: [], value: [] };
    this.leaves = [];
    this.length = 0;
  }

  push(op, a, b, c, value) {
    const node = this.length++;
    this.columns.op.push(op);
    this.columns.a.push(a);
    this.columns.b.push(b);
    this.columns.c.push(c);
    this.columns.value.push(value);
    return node;
  }

  input(value, group, index) {
    const node = this.push(OP.LEAF, -1, -1, 0, value);
    this.leaves.push({ node, group, index });
    return node;
  }

  const_(value) { return this.push(OP.CONST, -1, -1, value, value); }
  add(a, b) { return this.push(OP.ADD, a, b, 0, this.columns.value[a] + this.columns.value[b]); }
  sub(a, b) { return this.push(OP.SUB, a, b, 0, this.columns.value[a] - this.columns.value[b]); }
  mul(a, b) { return this.push(OP.MUL, a, b, 0, this.columns.value[a] * this.columns.value[b]); }
  div(a, b) { return this.push(OP.DIV, a, b, 0, this.columns.value[a] / this.columns.value[b]); }
  addc(a, c) { return this.push(OP.ADDC, a, -1, c, this.columns.value[a] + c); }
  mulc(a, c) { return this.push(OP.MULC, a, -1, c, this.columns.value[a] * c); }
  neg(a) { return this.push(OP.NEG, a, -1, 0, -this.columns.value[a]); }
  sqrt(a) { return this.push(OP.SQRT, a, -1, 0, Math.sqrt(this.columns.value[a])); }
  sin(a) { return this.push(OP.SIN, a, -1, 0, Math.sin(this.columns.value[a])); }
  cos(a) { return this.push(OP.COS, a, -1, 0, Math.cos(this.columns.value[a])); }

  seal() {
    this.op = Int32Array.from(this.columns.op);
    this.a = Int32Array.from(this.columns.a);
    this.b = Int32Array.from(this.columns.b);
    this.c = Float64Array.from(this.columns.c);
    this.value = Float64Array.from(this.columns.value);
    this.grad = new Float64Array(this.length);
    const groups = new Map();
    for (const leaf of this.leaves) {
      if (!groups.has(leaf.group)) groups.set(leaf.group, { nodes: [], indices: [] });
      groups.get(leaf.group).nodes.push(leaf.node);
      groups.get(leaf.group).indices.push(leaf.index);
    }
    this.runs = [...groups].map(([group, run]) => ({
      group,
      nodes: Int32Array.from(run.nodes),
      indices: Int32Array.from(run.indices),
    }));
    this.columns = null;
    this.leaves = null;
  }

  load(params) {
    for (const { group, nodes, indices } of this.runs) {
      const values = params[group];
      for (let i = 0; i < nodes.length; i++) this.value[nodes[i]] = values[indices[i]];
    }
  }

  forward() {
    const { op, a, b, c, value } = this;
    for (let i = 0; i < this.length; i++) {
      switch (op[i]) {
        case OP.LEAF: case OP.CONST: break;
        case OP.ADD: value[i] = value[a[i]] + value[b[i]]; break;
        case OP.SUB: value[i] = value[a[i]] - value[b[i]]; break;
        case OP.MUL: value[i] = value[a[i]] * value[b[i]]; break;
        case OP.DIV: value[i] = value[a[i]] / value[b[i]]; break;
        case OP.ADDC: value[i] = value[a[i]] + c[i]; break;
        case OP.MULC: value[i] = value[a[i]] * c[i]; break;
        case OP.NEG: value[i] = -value[a[i]]; break;
        case OP.SQRT: value[i] = Math.sqrt(value[a[i]]); break;
        case OP.SIN: value[i] = Math.sin(value[a[i]]); break;
        case OP.COS: value[i] = Math.cos(value[a[i]]); break;
      }
    }
  }

  backward() {
    const { op, a, b, c, value, grad } = this;
    for (let i = this.length - 1; i >= 0; i--) {
      const g = grad[i];
      if (g === 0) continue;
      switch (op[i]) {
        case OP.LEAF: case OP.CONST: break;
        case OP.ADD: grad[a[i]] += g; grad[b[i]] += g; break;
        case OP.SUB: grad[a[i]] += g; grad[b[i]] -= g; break;
        case OP.MUL: grad[a[i]] += g * value[b[i]]; grad[b[i]] += g * value[a[i]]; break;
        case OP.DIV:
          grad[a[i]] += g / value[b[i]];
          grad[b[i]] -= g * value[a[i]] / value[b[i]] ** 2;
          break;
        case OP.ADDC: grad[a[i]] += g; break;
        case OP.MULC: grad[a[i]] += g * c[i]; break;
        case OP.NEG: grad[a[i]] -= g; break;
        case OP.SQRT: grad[a[i]] += g * 0.5 / value[i]; break;
        case OP.SIN: grad[a[i]] += g * Math.cos(value[a[i]]); break;
        case OP.COS: grad[a[i]] -= g * Math.sin(value[a[i]]); break;
      }
    }
  }

  readGrads(out) {
    for (const { group, nodes, indices } of this.runs) {
      const values = out[group];
      for (let i = 0; i < nodes.length; i++) values[indices[i]] += this.grad[nodes[i]];
    }
  }
}
