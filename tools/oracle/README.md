# WebGPU/JAX parity

Requires the root JavaScript install, a WebGPU GPU, and JAX in `.venv`
(or another interpreter selected by `PYTHON`).

```sh
npm run test:oracle
node tools/oracle/check.js output/oracle  # retain exchanged JSON files
```

Checks forward renders and VJPs for `nonzero` and `evenodd` fill rules.
