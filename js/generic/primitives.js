function loop(tape, ax, ay, cx, cy) {
  const curves = new Array(ax.length * 6);
  for (let i = 0; i < ax.length; i++) {
    const next = (i + 1) % ax.length;
    curves[6 * i] = ax[i];
    curves[6 * i + 1] = ay[i];
    curves[6 * i + 2] = cx[i];
    curves[6 * i + 3] = cy[i];
    curves[6 * i + 4] = ax[next];
    curves[6 * i + 5] = ay[next];
  }
  return curves;
}

export function circle(tape, x, y, radius, segments = 8) {
  const ax = [], ay = [], cx = [], cy = [];
  const sec = 1 / Math.cos(Math.PI / segments);
  for (let i = 0; i < segments; i++) {
    const angle = i * 2 * Math.PI / segments;
    const middle = angle + Math.PI / segments;
    ax.push(tape.add(x, tape.mulc(radius, Math.cos(angle))));
    ay.push(tape.add(y, tape.mulc(radius, Math.sin(angle))));
    cx.push(tape.add(x, tape.mulc(radius, sec * Math.cos(middle))));
    cy.push(tape.add(y, tape.mulc(radius, sec * Math.sin(middle))));
  }
  return loop(tape, ax, ay, cx, cy);
}

// Sharp-cornered stroke: the capsule's straight long edges without the cap
// overhang — ends sit exactly at the segment endpoints. width must be
// positive: a negative value flips the edge offsets and inverts the outline.
export function rectStroke(tape, x0, y0, x1, y1, width) {
  const dx = tape.sub(x1, x0), dy = tape.sub(y1, y0);
  const length = tape.sqrt(tape.addc(tape.add(tape.mul(dx, dx), tape.mul(dy, dy)), 1e-12));
  const scale = tape.div(width, length);
  const nx = tape.mul(tape.neg(dy), scale), ny = tape.mul(dx, scale);
  const mid = (a, b) => tape.mulc(tape.add(a, b), 0.5);
  const ax = [tape.add(x0, nx), tape.add(x1, nx), tape.sub(x1, nx), tape.sub(x0, nx)];
  const ay = [tape.add(y0, ny), tape.add(y1, ny), tape.sub(y1, ny), tape.sub(y0, ny)];
  const cx = ax.map((_, i) => mid(ax[i], ax[(i + 1) % 4]));
  const cy = ay.map((_, i) => mid(ay[i], ay[(i + 1) % 4]));
  return loop(tape, ax, ay, cx, cy);
}

// width must be positive: a negative value flips the cap and edge offsets and
// turns the outline inside out.
export function roundCapsule(tape, x0, y0, x1, y1, width) {
  const dx = tape.sub(x1, x0), dy = tape.sub(y1, y0);
  const length = tape.sqrt(tape.addc(tape.add(tape.mul(dx, dx), tape.mul(dy, dy)), 1e-12));
  const scale = tape.div(width, length);
  const nx = tape.mul(tape.neg(dy), scale), ny = tape.mul(dx, scale);
  const ux = tape.mul(dx, scale), uy = tape.mul(dy, scale);
  const mid = (a, b) => tape.mulc(tape.add(a, b), 0.5);
  const ax = [
    tape.add(x0, nx), tape.add(x1, nx), tape.add(x1, ux),
    tape.sub(x1, nx), tape.sub(x0, nx), tape.sub(x0, ux),
  ];
  const ay = [
    tape.add(y0, ny), tape.add(y1, ny), tape.add(y1, uy),
    tape.sub(y1, ny), tape.sub(y0, ny), tape.sub(y0, uy),
  ];
  const cx = [
    tape.add(mid(x0, x1), nx),
    tape.add(tape.add(x1, nx), ux),
    tape.sub(tape.add(x1, ux), nx),
    tape.sub(mid(x0, x1), nx),
    tape.sub(tape.sub(x0, nx), ux),
    tape.sub(tape.add(x0, nx), ux),
  ];
  const cy = [
    tape.add(mid(y0, y1), ny),
    tape.add(tape.add(y1, ny), uy),
    tape.sub(tape.add(y1, uy), ny),
    tape.sub(mid(y0, y1), ny),
    tape.sub(tape.sub(y0, ny), uy),
    tape.sub(tape.add(y0, ny), uy),
  ];
  return loop(tape, ax, ay, cx, cy);
}
