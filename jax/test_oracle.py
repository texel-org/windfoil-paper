"""Small CPU checks for the JAX Windfoil oracle."""

import pathlib
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from oracle import (  # noqa: E402
    _fold_evenodd,
    _fold_nonzero,
    composite_scene,
    coverage_image,
    integrate_piece,
    loop_to_curves,
    render_scene,
    render_scene_ragged,
    scene_image,
    tonemap_reinhard,
    tonemap_reinhard_white,
    tonemap_smooth,
)


def circle_loop(center, radius, count):
    angle = jnp.arange(count) * (2.0 * jnp.pi / count)
    middle = angle + jnp.pi / count
    center = jnp.asarray(center)
    anchors = center + radius * jnp.stack((jnp.cos(angle), jnp.sin(angle)), axis=-1)
    controls = center + (radius / jnp.cos(jnp.pi / count)) * jnp.stack(
        (jnp.cos(middle), jnp.sin(middle)), axis=-1
    )
    return anchors, controls


def rectangle(x0, y0, x1, y1, dtype=jnp.float64):
    anchors = jnp.asarray(((x0, y0), (x1, y0), (x1, y1), (x0, y1)), dtype=dtype)
    controls = 0.5 * (anchors + jnp.roll(anchors, -1, axis=0))
    return loop_to_curves(anchors, controls)


def pentagram(center=(16.5, 16.5), radius=12.0, dtype=jnp.float64):
    angles = -0.5 * jnp.pi + jnp.arange(5) * (2.0 * jnp.pi / 5.0)
    outer = jnp.asarray(center, dtype=dtype) + radius * jnp.stack(
        (jnp.cos(angles), jnp.sin(angles)), axis=-1
    )
    anchors = outer[jnp.asarray((0, 2, 4, 1, 3))]
    controls = 0.5 * (anchors + jnp.roll(anchors, -1, axis=0))
    return anchors, controls


def test_rectangle():
    height, width, size = 24, 28, 3.25
    x0, y0, x1, y1 = 5.3, 4.4, 21.7, 18.2
    actual = np.asarray(coverage_image(rectangle(x0, y0, x1, y1), height, width, size))
    xs, ys = np.arange(width) + 0.5, np.arange(height) + 0.5
    ox = np.clip(np.minimum(x1, xs + size / 2) - np.maximum(x0, xs - size / 2), 0, None)
    oy = np.clip(np.minimum(y1, ys + size / 2) - np.maximum(y0, ys - size / 2), 0, None)
    expected = (oy[:, None] * ox[None, :]) / size**2
    error = np.max(np.abs(actual - expected))
    assert error < 2e-10, error


def test_fill_rules():
    anchors, controls = pentagram()
    curves = loop_to_curves(anchors, controls)
    default = np.asarray(coverage_image(curves, 33, 33, 1.0))
    nonzero = np.asarray(coverage_image(curves, 33, 33, 1.0, fill_rule="nonzero"))
    evenodd = np.asarray(coverage_image(curves, 33, 33, 1.0, fill_rule="evenodd"))
    assert np.max(np.abs(default - nonzero)) == 0.0
    assert nonzero[16, 16] > 1.0 - 1e-12
    assert evenodd[16, 16] < 1e-12

    colors = jnp.asarray(((0.1, 0.2, 0.3),))
    alphas = jnp.asarray((1.0,))
    scene = np.asarray(render_scene(
        curves[None], colors, alphas, 33, 33, 1.0, fill_rules=("evenodd",)
    ))
    assert np.max(np.abs(scene[16, 16] - 1.0)) < 1e-12

    rectangle_curves = rectangle(4.0, 5.0, 12.0, 13.0)
    ragged = np.asarray(render_scene_ragged(
        (curves, rectangle_curves),
        jnp.asarray(((0.1, 0.2, 0.3), (0.4, 0.5, 0.6))),
        jnp.asarray((1.0, 0.7)),
        33,
        33,
        1.0,
        fill_rules=("evenodd", "nonzero"),
    ))
    assert ragged.shape == (33, 33, 3)
    ragged_image, ragged_pullback = jax.vjp(
        lambda value: render_scene_ragged(
            value,
            jnp.asarray(((0.1, 0.2, 0.3), (0.4, 0.5, 0.6))),
            jnp.asarray((1.0, 0.7)),
            33,
            33,
            1.0,
            fill_rules=("evenodd", "nonzero"),
        ),
        (curves, rectangle_curves),
    )
    ragged_grads = ragged_pullback(jnp.ones_like(ragged_image))[0]
    assert len(ragged_grads) == 2
    assert ragged_grads[0].shape != ragged_grads[1].shape
    assert all(np.isfinite(np.asarray(grad)).all() for grad in ragged_grads)

    try:
        coverage_image(curves, 1, 1, 1.0, fill_rule="winding")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid fill rule was accepted")


def test_fill_rule_cusp_gradients_match_shader():
    assert float(jax.grad(_fold_nonzero)(jnp.asarray(0.0))) == 0.0
    assert float(jax.grad(_fold_nonzero)(jnp.asarray(1.0))) == 0.0
    assert float(jax.grad(_fold_evenodd)(jnp.asarray(0.0))) == 0.0
    assert float(jax.grad(_fold_evenodd)(jnp.asarray(1.0))) == 0.0
    assert float(jax.grad(_fold_evenodd)(jnp.asarray(2.0))) == 0.0


def test_evenodd_geometry_gradient():
    anchors, controls = pentagram(radius=10.7)
    yy, xx = jnp.meshgrid(jnp.arange(25), jnp.arange(27), indexing="ij")
    weights = 0.7 + 0.2 * jnp.sin(0.31 * xx + 0.17 * yy)

    def loss(a):
        curves = loop_to_curves(a, controls)
        coverage = coverage_image(
            curves, 25, 27, 2.7, origin=(3.0, 4.0), fill_rule="evenodd"
        )
        return jnp.mean(coverage * weights)

    index = (2, 0)
    autodiff = float(jax.grad(loss)(anchors)[index])
    step = 1e-5
    plus = anchors.at[index].add(step)
    minus = anchors.at[index].add(-step)
    reference = float((loss(plus) - loss(minus)) / (2.0 * step))
    relative = abs(autodiff - reference) / max(abs(reference), 1e-10)
    assert relative < 3e-4, (autodiff, reference, relative)


def test_geometry_gradient():
    rng = np.random.default_rng(4)
    anchors, controls = circle_loop((14.0, 14.0), 7.0, 6)
    anchors = jnp.asarray(np.asarray(anchors) + rng.normal(0, 0.6, (6, 2)))
    controls = jnp.asarray(np.asarray(controls) + rng.normal(0, 0.8, (6, 2)))
    target = scene_image(
        (anchors + jnp.asarray((0.4, -0.3)))[None], controls[None],
        jnp.asarray(((0.2, 0.6, 0.9),)), jnp.asarray((0.8,)), 14, 14, 2.7,
    )

    def loss(a, c):
        image = scene_image(
            a[None], c[None], jnp.asarray(((0.2, 0.6, 0.9),)),
            jnp.asarray((0.8,)), 14, 14, 2.7,
        )
        return jnp.mean((image - target) ** 2)

    grad_a, grad_c = jax.grad(loss, argnums=(0, 1))(anchors, controls)

    def finite_difference(which, index, step=1e-5):
        values = [anchors, controls]
        plus = values.copy()
        minus = values.copy()
        plus[which] = plus[which].at[index].add(step)
        minus[which] = minus[which].at[index].add(-step)
        return float((loss(*plus) - loss(*minus)) / (2.0 * step))

    for which, index, autodiff in (
        (0, (2, 0), float(grad_a[2, 0])),
        (1, (4, 1), float(grad_c[4, 1])),
    ):
        reference = finite_difference(which, index)
        relative = abs(autodiff - reference) / max(abs(reference), 1e-10)
        assert relative < 2e-4, (autodiff, reference, relative)


def test_degenerate_finiteness():
    cases = (
        (((8, 8), (8, 8), (8, 8)),),
        (((2, 8), (8, 8), (14, 8)),),
        (((8, 2), (8, 8), (8, 14)),),
        (((8, 8), (8 + 1e-7, 8), (8, 8 + 1e-7)),),
    )
    for case in cases:
        curves = jnp.asarray(case, dtype=jnp.float32)
        for size in (1e-3, 1.0, 100.0):
            value, grad = jax.value_and_grad(
                lambda q: coverage_image(q, 12, 12, size).sum()
            )(curves)
            assert bool(jnp.isfinite(value))
            assert bool(jnp.isfinite(grad).all())


def test_short_piece_at_large_coordinate_is_not_degenerate():
    # This is representative of the source SVG's 0.001-unit edges around
    # x=330. A relative-to-coordinate gate used to erase the edge and leave a
    # faint, bbox-wide winding residue on otherwise transparent scanlines.
    piece = jnp.asarray(
        ((330.0, 0.0), (330.0, 0.0005), (330.0, 0.001)),
        dtype=jnp.float32,
    )
    area = float(integrate_piece(piece, -1.0, 1.0, 0.1))
    assert np.isclose(area, 0.0002, rtol=2e-5), area


def test_scale_invariance():
    anchors, controls = circle_loop((12.0, 12.0), 6.5, 7)
    curves = loop_to_curves(anchors, controls)
    base = np.asarray(coverage_image(curves, 12, 12, 3.3, scale=2.0))
    for factor in (1e-8, 1e-3, 1e6):
        scaled = np.asarray(coverage_image(
            curves * factor, 12, 12, 3.3 * factor, scale=2.0 * factor
        ))
        assert np.max(np.abs(base - scaled)) < 2e-9


def test_compositing():
    coverage = jnp.ones((2, 1, 1))
    colors = jnp.asarray(((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    actual = np.asarray(composite_scene(coverage, colors, jnp.asarray((0.5, 0.5))))[0, 0]
    expected = np.asarray((0.5, 0.25, 0.75))
    assert np.max(np.abs(actual - expected)) < 1e-12

    coverage = jnp.ones((3, 1, 1))
    colors = jnp.asarray(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
    actual = np.asarray(composite_scene(coverage, colors, jnp.asarray((1.0, 0.999, 0.999))))[0, 0]
    expected = np.asarray((0.000001, 0.001, 0.999001))
    assert np.max(np.abs(actual - expected)) < 1e-12


def test_blend_modes():
    coverage = jnp.ones((2, 1, 1))
    colors = jnp.asarray(((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    alphas = jnp.asarray((0.5, 0.5))
    background = (0.1, 0.2, 0.3)

    added = np.asarray(composite_scene(
        coverage, colors, alphas, background, blend="add"
    ))[0, 0]
    assert np.max(np.abs(added - np.asarray((0.6, 0.2, 0.8)))) < 1e-12

    # multiply: bg * prod(1 - a * (1 - color)); factors (0.5-red, 0.5-blue).
    multiplied = np.asarray(composite_scene(
        coverage, colors, alphas, background, blend="multiply"
    ))[0, 0]
    expected = np.asarray(background) * np.asarray((1.0, 0.5, 0.5)) * np.asarray((0.5, 0.5, 1.0))
    assert np.max(np.abs(multiplied - expected)) < 1e-12

    # screen: 1 - (1 - bg) * prod(1 - a * color) -- multiply in complement space.
    screened = np.asarray(composite_scene(
        coverage, colors, alphas, background, blend="screen"
    ))[0, 0]
    expected = 1.0 - (1.0 - np.asarray(background)) \
        * np.asarray((0.5, 1.0, 1.0)) * np.asarray((1.0, 1.0, 0.5))
    assert np.max(np.abs(screened - expected)) < 1e-12

    # Order independence: reversing the layer stack changes src-over but not
    # add or multiply -- the shader relies on this to skip its painter sort.
    coverage = jnp.asarray(np.linspace(0.2, 1.0, 8).reshape(2, 2, 2))
    colors = jnp.asarray(((0.9, 0.1, 0.4), (0.2, 0.8, 0.6)))
    alphas = jnp.asarray((0.7, 0.6))
    for blend in ("add", "multiply", "screen"):
        forward_order = composite_scene(coverage, colors, alphas, background, blend=blend)
        reverse_order = composite_scene(
            coverage[::-1], colors[::-1], alphas[::-1], background, blend=blend
        )
        assert np.max(np.abs(np.asarray(forward_order - reverse_order))) < 1e-12, blend
    over = composite_scene(coverage, colors, alphas, background)
    over_reversed = composite_scene(coverage[::-1], colors[::-1], alphas[::-1], background)
    assert np.max(np.abs(np.asarray(over - over_reversed))) > 1e-3

    try:
        composite_scene(coverage, colors, alphas, background, blend="overlay")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid blend mode was accepted")


def test_tonemap():
    assert abs(float(tonemap_reinhard(1.0, 1.0)) - 0.5) < 1e-12
    assert abs(float(tonemap_reinhard(3.0, 1.0)) - 0.75) < 1e-12
    # d/dk [k*x / (1 + k*x)] = x / (1 + k*x)^2
    grad_k = float(jax.grad(lambda k: tonemap_reinhard(2.0, k))(jnp.float64(0.5)))
    assert abs(grad_k - 2.0 / 4.0) < 1e-12
    # HDR values stay inside [0, 1): additive accumulation cannot blow out.
    values = tonemap_reinhard(jnp.asarray([0.0, 1.0, 10.0, 1000.0]), 2.0)
    assert float(values[0]) == 0.0
    assert bool((values[1:] > 0.0).all()) and bool((values < 1.0).all())
    # The white-normalized operators hit display 1 exactly at x = W ...
    for operator in (tonemap_reinhard_white, tonemap_smooth):
        for k, w in ((0.7, 1.9), (2.3, 0.6)):
            assert abs(float(operator(w, k, w)) - 1.0) < 1e-12
        # ... and tend to the pure linear rescale x / W as k -> 0.
        near = float(operator(0.4, 1e-4, 1.6))
        assert abs(near - 0.4 / 1.6) < 1e-4
    # smooth is odd (signed linear light maps symmetrically) and monotone
    # across zero; the Reinhard pair never sees negative light by contract.
    xs = jnp.asarray([-10.0, -1.0, -0.25, 0.0, 0.25, 1.0, 10.0])
    values = tonemap_smooth(xs, 1.3, 2.0)
    assert bool(jnp.allclose(values + tonemap_smooth(-xs, 1.3, 2.0), 0.0, atol=1e-12))
    assert bool((jnp.diff(values) > 0.0).all())
    grad_x = float(jax.grad(lambda x: tonemap_smooth(x, 1.3, 2.0))(jnp.float64(-0.7)))
    assert grad_x > 0.0 and jnp.isfinite(grad_x)


if __name__ == "__main__":
    test_rectangle()
    test_fill_rules()
    test_fill_rule_cusp_gradients_match_shader()
    test_evenodd_geometry_gradient()
    test_geometry_gradient()
    test_degenerate_finiteness()
    test_short_piece_at_large_coordinate_is_not_degenerate()
    test_scale_invariance()
    test_compositing()
    test_blend_modes()
    test_tonemap()
    print("oracle tests passed")
