import json
import pathlib
import sys

import jax
import jax.numpy as jnp


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "jax"))

from oracle import (  # noqa: E402
    render_scene_ragged,
    tonemap_reinhard,
    tonemap_reinhard_white,
    tonemap_smooth,
)


source = json.loads(pathlib.Path(sys.argv[1]).read_text())
curves = tuple(
    jnp.asarray(shape, dtype=jnp.float32).reshape(-1, 3, 2)
    for shape in source["curves"]
)
colors = jnp.asarray(source["colors"], dtype=jnp.float32)
alphas = jnp.asarray(source["alphas"], dtype=jnp.float32)
cotangent = jnp.asarray(source["cotangent"], dtype=jnp.float32).reshape(
    source["height"], source["width"], 3
)
settings = source["settings"]


shape_s = source.get("shapeS")
filters = None if shape_s is None else jnp.asarray(shape_s, dtype=jnp.float32)


def render(curves_value, colors_value, alphas_value, filters_value=None):
    return render_scene_ragged(
        curves_value,
        colors_value,
        alphas_value,
        source["height"],
        source["width"],
        settings["s"],
        background=settings["bg"],
        scale=settings["scale"],
        origin=settings["origin"],
        fill_rules=source.get("fillRules"),
        blend=source.get("blend", "src-over"),
        shape_s=filters_value,
    )


s_grads = None
k_grad = None
w_grad = None
tonemap = source.get("tonemap")
if tonemap == "reinhard":
    exposure = jnp.float32(source.get("exposure", 1.0))

    def render_display(curves_value, colors_value, alphas_value, k):
        return tonemap_reinhard(render(curves_value, colors_value, alphas_value), k)

    image, pullback = jax.vjp(render_display, curves, colors, alphas, exposure)
    curve_grads, color_grads, alpha_grads, k_grad = pullback(cotangent)
elif tonemap in ("reinhard-white", "smooth"):
    exposure = jnp.float32(source.get("exposure", 1.0))
    white = jnp.float32(source.get("white", 1.0))
    operator = tonemap_reinhard_white if tonemap == "reinhard-white" else tonemap_smooth

    def render_display(curves_value, colors_value, alphas_value, k, w):
        return operator(render(curves_value, colors_value, alphas_value), k, w)

    image, pullback = jax.vjp(render_display, curves, colors, alphas, exposure, white)
    curve_grads, color_grads, alpha_grads, k_grad, w_grad = pullback(cotangent)
elif filters is None:
    image, pullback = jax.vjp(render, curves, colors, alphas)
    curve_grads, color_grads, alpha_grads = pullback(cotangent)
else:
    image, pullback = jax.vjp(render, curves, colors, alphas, filters)
    curve_grads, color_grads, alpha_grads, s_grads = pullback(cotangent)
result = {
    "image": image.reshape(-1).tolist(),
    "curveGrads": jnp.concatenate([grad.reshape(-1) for grad in curve_grads]).tolist(),
    "colorGrads": color_grads.reshape(-1).tolist(),
    "alphaGrads": alpha_grads.reshape(-1).tolist(),
    **({} if s_grads is None else {"sGrads": s_grads.reshape(-1).tolist()}),
    **({} if k_grad is None else {"kGrad": float(k_grad)}),
    **({} if w_grad is None else {"wGrad": float(w_grad)}),
}
pathlib.Path(sys.argv[2]).write_text(json.dumps(result))
