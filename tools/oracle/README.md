# WebGPU/JAX parity

Run the deterministic forward and VJP checks for both `nonzero` and `evenodd`
with Node/Dawn:

```sh
node tools/oracle/check.js
```

The micro-edge case keeps a real 0.001-unit segment at a large coordinate so a
relative degeneracy cutoff cannot silently reopen a contour. The evenodd case
mixes both rules and unequal per-shape curve counts so the exchange also
exercises the JAX oracle's ragged-scene path.

JAX must be installed in `.venv`, or set `PYTHON` to another interpreter. Pass
an ignored output directory to retain the exchanged JSON files:

```sh
node tools/oracle/check.js output/oracle
```
