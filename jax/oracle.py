"""Small JAX oracle for the Windfoil shader's analytic filter kernels."""

import jax
import jax.numpy as jnp


_TINY = 1e-12
_FOLD_EPS = 1e-5
_COMPOSITE_EPS = 1e-5

# Per-axis (offset, weight) taps in units of s; must match js/filter-kernels.js.
_GL3_NODE = 0.3872983346207417  # sqrt(3/5) / 2
_CUBIC_OUTER = 0.021433470507544582  # 125/5832
_CUBIC_MID = 0.102880658436214  # 25/243
_CUBIC_INNER = 0.22890946502057613  # 445/1944
_CUBIC_CENTER = 0.2935528120713306  # 214/729
_KERNEL_TAPS = {
    "box": ((0.0, 1.0),),
    "tent": ((-_GL3_NODE, 5.0 / 18.0), (0.0, 8.0 / 18.0), (_GL3_NODE, 5.0 / 18.0)),
    "cubic": (
        (-3.0 * _GL3_NODE, _CUBIC_OUTER),
        (-2.0 * _GL3_NODE, _CUBIC_MID),
        (-_GL3_NODE, _CUBIC_INNER),
        (0.0, _CUBIC_CENTER),
        (_GL3_NODE, _CUBIC_INNER),
        (2.0 * _GL3_NODE, _CUBIC_MID),
        (3.0 * _GL3_NODE, _CUBIC_OUTER),
    ),
}


def _kernel_taps(kernel):
    taps = _KERNEL_TAPS.get(kernel)
    if taps is None:
        raise ValueError(f"unsupported filter kernel: {kernel!r}")
    return taps


def loop_to_curves(anchors, controls):
    """Build closed quadratics: anchor[i], control[i], anchor[i+1]."""
    anchors = jnp.asarray(anchors)
    controls = jnp.asarray(controls)
    return jnp.stack((anchors, controls, jnp.roll(anchors, -1, axis=-2)), axis=-2)


def _nonzero(x):
    return jnp.where(jnp.abs(x) < _TINY, jnp.where(x < 0.0, -_TINY, _TINY), x)


def _degenerate(piece):
    q1, q2, q3 = piece[..., 0, :], piece[..., 1, :], piece[..., 2, :]
    span = jnp.abs(q2 - q1).sum(-1) + jnp.abs(q3 - q2).sum(-1)
    return span == 0.0


def mono_root(a2, a1, a0, e1, value, rising):
    """Solve a monotone quadratic on [0, 1], saturating at its endpoints."""
    c = a0 - value
    disc = jnp.maximum(a1 * a1 - 4.0 * a2 * c, 0.0)
    guard = 1e-14 * (a1 * a1 + jnp.abs(4.0 * a2 * c)) + 1e-30
    sq = jnp.sqrt(disc + guard)
    q = -0.5 * (a1 + jnp.where(a1 >= 0.0, sq, -sq))
    r1 = q / _nonzero(a2)
    r2 = c / _nonzero(q)
    t = jnp.clip(jnp.where((a1 < 0.0) == rising, r1, r2), 0.0, 1.0)
    sat0 = jnp.where(rising, a0 >= value, a0 <= value)
    sat1 = jnp.where(rising, e1 <= value, e1 >= value)
    return jnp.where(sat0, 0.0, jnp.where(sat1, 1.0, t))


def _integrate_inside(a2, a1, x0, ta, tb, hx):
    mid = 0.5 * (ta + tb)
    half = 0.5 * (tb - ta)
    x_mid = (a2[..., 0] * mid + a1[..., 0]) * mid + x0 + hx
    dx_mid = 2.0 * a2[..., 0] * mid + a1[..., 0]
    dy_mid = 2.0 * a2[..., 1] * mid + a1[..., 1]
    return (2.0 * half * x_mid * dy_mid
            + (2.0 / 3.0) * half**3 * (a2[..., 0] * dy_mid + 2.0 * a2[..., 1] * dx_mid))


def integrate_piece(piece, wlo, whi, hx):
    """Integrate one xy-monotone quadratic over a pixel box."""
    q1, q2, q3 = piece[..., 0, :], piece[..., 1, :], piece[..., 2, :]
    a2 = q1 - 2.0 * q2 + q3
    a1 = 2.0 * (q2 - q1)

    y_rising = q3[..., 1] >= q1[..., 1]
    t_lo = mono_root(a2[..., 1], a1[..., 1], q1[..., 1], q3[..., 1],
                     jnp.where(y_rising, wlo, whi), y_rising)
    t_hi = mono_root(a2[..., 1], a1[..., 1], q1[..., 1], q3[..., 1],
                     jnp.where(y_rising, whi, wlo), y_rising)
    x_rising = q3[..., 0] >= q1[..., 0]
    t_left = jnp.clip(
        mono_root(a2[..., 0], a1[..., 0], q1[..., 0], q3[..., 0], -hx, x_rising),
        t_lo, t_hi,
    )
    t_right = jnp.clip(
        mono_root(a2[..., 0], a1[..., 0], q1[..., 0], q3[..., 0], hx, x_rising),
        t_lo, t_hi,
    )
    t1 = jnp.where(x_rising, t_left, t_right)
    t2 = jnp.maximum(jnp.where(x_rising, t_right, t_left), t1)
    inside = _integrate_inside(a2, a1, q1[..., 0], t1, t2, hx)
    ra = jnp.where(x_rising, t2, t_lo)
    rb = jnp.where(x_rising, t_hi, t1)
    mid = 0.5 * (ra + rb)
    right = jnp.maximum(rb - ra, 0.0) * (2.0 * a2[..., 1] * mid + a1[..., 1]) * (2.0 * hx)
    area = jnp.where(t_hi > t_lo, inside + right, 0.0)

    hull_min = jnp.minimum(q1[..., 0], jnp.minimum(q2[..., 0], q3[..., 0]))
    hull_max = jnp.maximum(q1[..., 0], jnp.maximum(q2[..., 0], q3[..., 0]))
    telescoped = (2.0 * hx) * (
        jnp.clip(q3[..., 1], wlo, whi) - jnp.clip(q1[..., 1], wlo, whi)
    )
    area = jnp.where(hull_min >= hx, telescoped, area)
    area = jnp.where(hull_max < -hx, 0.0, area)
    return jnp.where(_degenerate(piece), 0.0, area)


def split_monotone(curves):
    """Split every quadratic into three fixed-shape xy-monotone candidates."""
    curves = jnp.asarray(curves)
    b0, b1, b2 = curves[..., 0, :], curves[..., 1, :], curves[..., 2, :]
    a1 = 2.0 * (b1 - b0)
    denominator = 2.0 * (b0 - 2.0 * b1 + b2)
    eps = 1e-12 * (jnp.abs(a1) + jnp.abs(denominator)) + 1e-30
    denominator = jnp.where(
        jnp.abs(denominator) < eps,
        jnp.where(denominator < 0.0, -eps, eps),
        denominator,
    )
    extrema = jnp.clip(-a1 / denominator, 0.0, 1.0)
    t1 = jax.lax.stop_gradient(jnp.minimum(extrema[..., 0], extrema[..., 1]))
    t2 = jax.lax.stop_gradient(jnp.maximum(extrema[..., 0], extrema[..., 1]))

    def blossom(s, t):
        w0 = (1.0 - s) * (1.0 - t)
        w1 = s * (1.0 - t) + t * (1.0 - s)
        w2 = s * t
        return w0[..., None] * b0 + w1[..., None] * b1 + w2[..., None] * b2

    zero, one = jnp.zeros_like(t1), jnp.ones_like(t1)
    m1, m2 = blossom(t1, t1), blossom(t2, t2)
    p0 = jnp.stack((b0, blossom(zero, t1), m1), axis=-2)
    p1 = jnp.stack((m1, blossom(t1, t2), m2), axis=-2)
    p2 = jnp.stack((m2, blossom(t2, one), b2), axis=-2)
    return jnp.stack((p0, p1, p2), axis=-3)


def _as_float(curves):
    curves = jnp.asarray(curves)
    return curves if jnp.issubdtype(curves.dtype, jnp.inexact) else curves.astype(jnp.float32)


def _box_area(pieces, centers, hx, hy):
    relative = pieces - centers[..., None, None, :]
    q1, q2, q3 = relative[..., 0, :], relative[..., 1, :], relative[..., 2, :]
    outside_y = (jnp.maximum(q1[..., 1], q3[..., 1]) < -hy) | (
        jnp.minimum(q1[..., 1], q3[..., 1]) > hy
    )
    outside_left = jnp.maximum(q1[..., 0], jnp.maximum(q2[..., 0], q3[..., 0])) < -hx
    areas = integrate_piece(relative, -hy, hy, hx)
    return jnp.where(outside_y | outside_left, 0.0, areas).sum(-1)


def _winding(pieces, centers, size, kernel="box"):
    taps = _kernel_taps(kernel)
    sx, sy = size[0], size[1]
    hx, hy = 0.5 * sx, 0.5 * sy
    total = 0.0
    for node_y, weight_y in taps:
        for node_x, weight_x in taps:
            offset = jnp.stack((node_x * sx, node_y * sy), axis=-1)
            total = total + (weight_x * weight_y) * _box_area(
                pieces, centers + offset, hx, hy
            )
    product = sx * sy
    return total / jnp.where(product == 0.0, 1.0, product)


def _fold_nonzero(winding):
    magnitude = jnp.abs(winding)
    coverage = jnp.clip(magnitude, 0.0, 1.0)
    # Match WGSL sign(0) and the backward pass's zero-winding early exit.
    live = (magnitude != 0.0) & (magnitude < 1.0 - _FOLD_EPS)
    return jnp.where(live, coverage, jax.lax.stop_gradient(coverage))


def _fold_evenodd(winding):
    magnitude = jnp.abs(winding)
    phase = magnitude - 2.0 * jnp.floor(0.5 * magnitude)
    coverage = jnp.minimum(phase, 2.0 - phase)
    live = (phase != 0.0) & (jnp.abs(phase - 1.0) >= _FOLD_EPS)
    return jnp.where(live, coverage, jax.lax.stop_gradient(coverage))


def _fill_rule_code(fill_rule):
    if fill_rule == "nonzero":
        return 0
    if fill_rule == "evenodd":
        return 1
    raise ValueError(f"unsupported fill rule: {fill_rule!r}")


def _fold_coverage(winding, fill_rule_code):
    return jnp.where(fill_rule_code == 1, _fold_evenodd(winding), _fold_nonzero(winding))


def _coverage_image(curves, height, width, s, scale, origin, fill_rule_code,
                    kernel="box"):
    curves = _as_float(curves)
    pieces = split_monotone(curves).reshape(-1, 3, 2)
    size = jnp.broadcast_to(jnp.asarray(s, dtype=curves.dtype), (2,))
    xs = origin[0] + (jnp.arange(width, dtype=curves.dtype) + 0.5) * scale
    ys = origin[1] + (jnp.arange(height, dtype=curves.dtype) + 0.5) * scale
    xx, yy = jnp.meshgrid(xs, ys)
    centers = jnp.stack((xx, yy), axis=-1)
    return _fold_coverage(_winding(pieces, centers, size, kernel), fill_rule_code)


def coverage_image(curves, height, width, s, *, scale=1.0, origin=(0.0, 0.0),
                   fill_rule="nonzero", kernel="box"):
    """Render Windfoil's filter-averaged-winding fill approximation on a grid."""
    _kernel_taps(kernel)
    return _coverage_image(
        curves, height, width, s, scale, origin, _fill_rule_code(fill_rule), kernel
    )


_BLEND_MODES = ("src-over", "add", "multiply", "screen")


def tonemap_reinhard(image, exposure=1.0):
    """Reinhard exposure tonemap, T(x) = k*x / (1 + k*x).

    The shader applies this at the loss boundary only: scenes composite in
    linear light (add mode may exceed 1) and comparison happens in display
    space, with ``exposure`` as the learnable scalar.
    """
    scaled = exposure * jnp.asarray(image)
    return scaled / (1.0 + scaled)


def tonemap_reinhard_white(image, exposure=1.0, white=1.0):
    """White-normalized Reinhard, T(x) = R(k*x) / R(k*W) with R(u) = u/(1+u).

    T(W) = 1 exactly (display white is reachable) and T -> x/W as k -> 0.
    Like plain Reinhard it has a pole at x = -1/k, so its domain is
    nonnegative linear light; ``exposure`` and ``white`` are both learnable.
    """
    scaled = exposure * jnp.asarray(image)
    kw = exposure * white
    return (scaled / (1.0 + scaled)) * ((1.0 + kw) / kw)


def tonemap_smooth(image, exposure=1.0, white=1.0):
    """Signed-safe sigmoid tonemap, T(x) = s(k*x) / s(k*W), s(u) = u/sqrt(1+u^2).

    Odd and C-infinity on all of R (no pole), T(W) = 1, and T -> x/W as
    k -> 0 -- the operator for signed (raw identity-transfer) scenes.
    """
    scaled = exposure * jnp.asarray(image)
    kw = exposure * white
    return (scaled / jnp.sqrt(1.0 + scaled * scaled)) * (
        jnp.sqrt(1.0 + kw * kw) / kw
    )


def composite_scene(coverages, colors, alphas, background=(1.0, 1.0, 1.0),
                    blend="src-over"):
    """Composite bottom-to-top layers using the shader's top-down evaluation.

    ``add``, ``multiply``, and ``screen`` are order-independent, matching the
    shader's unsorted evaluation: ``bg + sum(color * a)``,
    ``bg * prod(1 - a * (1 - color))``, and
    ``1 - (1 - bg) * prod(1 - a * color)`` with layer alpha ``a = coverage *
    opacity``, exactly as in src-over.
    """
    if blend not in _BLEND_MODES:
        raise ValueError(f"unsupported blend mode: {blend!r}")
    coverages = jnp.asarray(coverages)
    colors = jnp.asarray(colors, dtype=coverages.dtype)
    alphas = jnp.asarray(alphas, dtype=coverages.dtype)
    background = jnp.asarray(background, dtype=coverages.dtype)

    if blend != "src-over":
        spatial = (1,) * (coverages.ndim - 1)
        color = colors.reshape(colors.shape[0], *spatial, colors.shape[-1])
        opacity = alphas.reshape(alphas.shape[0], *spatial, 1)
        a = coverages[..., None] * opacity
        if blend == "add":
            return background + (color * a).sum(0)
        if blend == "multiply":
            return background * jnp.prod(1.0 - a * (1.0 - color), axis=0)
        return 1.0 - (1.0 - background) * jnp.prod(1.0 - a * color, axis=0)

    image = jnp.zeros(coverages.shape[1:] + (colors.shape[-1],), dtype=coverages.dtype)
    transmittance = jnp.ones(coverages.shape[1:], dtype=coverages.dtype)

    def add_layer(carry, layer):
        image, transmittance = carry
        coverage, color, opacity = layer
        active = transmittance >= _COMPOSITE_EPS
        alpha = jnp.where(active, coverage * opacity, 0.0)
        image = image + color * (alpha * transmittance)[..., None]
        return (image, transmittance * (1.0 - alpha)), None

    (image, transmittance), _ = jax.lax.scan(
        add_layer, (image, transmittance), (coverages[::-1], colors[::-1], alphas[::-1])
    )
    return image + background * transmittance[..., None]


def render_scene(curves, colors, alphas, height, width, s, *,
                 background=(1.0, 1.0, 1.0), scale=1.0, origin=(0.0, 0.0),
                 fill_rules=None, blend="src-over", kernel="box"):
    """Render stacked closed-curve shapes; index 0 is the bottom layer."""
    _kernel_taps(kernel)
    curves = _as_float(curves)
    if fill_rules is None:
        rule_codes = jnp.zeros((curves.shape[0],), dtype=jnp.uint32)
    else:
        if len(fill_rules) != curves.shape[0]:
            raise ValueError("fill_rules must contain one rule per shape")
        rule_codes = jnp.asarray([_fill_rule_code(rule) for rule in fill_rules], dtype=jnp.uint32)
    coverages = jax.vmap(
        lambda shape, rule: _coverage_image(
            shape, height, width, s, scale, origin, rule, kernel
        )
    )(curves, rule_codes)
    return composite_scene(coverages, colors, alphas, background, blend)


def render_scene_ragged(curves, colors, alphas, height, width, s, *,
                        background=(1.0, 1.0, 1.0), scale=1.0, origin=(0.0, 0.0),
                        fill_rules=None, blend="src-over", shape_s=None,
                        kernel="box"):
    """Render shapes with differing curve counts; ``curves`` is a Python sequence.

    ``shape_s`` optionally overrides the global filter per shape (one scalar
    or ``(sx, sy)`` pair per shape, indexable like an array) and is
    differentiable -- the reference for the shader's per-shape blur training.
    """
    _kernel_taps(kernel)
    count = len(curves)
    if fill_rules is None:
        fill_rules = ("nonzero",) * count
    if len(fill_rules) != count:
        raise ValueError("fill_rules must contain one rule per shape")
    coverages = jnp.stack([
        _coverage_image(
            _as_float(shape), height, width,
            s if shape_s is None else shape_s[index],
            scale, origin, _fill_rule_code(rule), kernel,
        )
        for index, (shape, rule) in enumerate(zip(curves, fill_rules))
    ])
    return composite_scene(coverages, colors, alphas, background, blend)


def scene_image(anchors, controls, colors, alphas, height, width, s, *,
                background=(1.0, 1.0, 1.0), scale=1.0, origin=(0.0, 0.0),
                fill_rules=None, blend="src-over", kernel="box"):
    """VJP-friendly scene entry point over loop anchors and controls."""
    curves = jax.vmap(loop_to_curves)(_as_float(anchors), _as_float(controls))
    return render_scene(
        curves, colors, alphas, height, width, s,
        background=background, scale=scale, origin=origin, fill_rules=fill_rules,
        blend=blend, kernel=kernel,
    )
