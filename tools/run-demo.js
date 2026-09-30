#!/usr/bin/env node

import { spawnSync } from 'node:child_process';

import { WINDFOIL_ENVIRONMENTS, windfoilCommand } from './windfoil-runtime.js';

const [demo, ...args] = process.argv.slice(2);
const entries = {
  l2: ['demos/cli.js', 'l2'],
  clip: ['demos/cli.js', 'clip'],
  lines: ['demos/cli.js', 'lines'],
  plot: ['demos/cli.js', 'plot'],
  roundtrip: ['demos/roundtrip/cli.js'],
  render: ['demos/render/cli.js'],
};
if (!entries[demo]) {
  console.error('usage: tools/run-demo.js <l2|clip|lines|plot|roundtrip|render> [options]');
  process.exit(2);
}

const runtime = process.env.WF_RUNTIME ?? 'node';
const cli = [...entries[demo], ...args];
if (!WINDFOIL_ENVIRONMENTS.includes(runtime)) {
  console.error(`invalid WF_RUNTIME=${JSON.stringify(runtime)} (expected node, deno, or deno-dawn)`);
  process.exit(2);
}
const { command, args: commandArgs, env } = windfoilCommand(runtime, cli);

const result = spawnSync(command, commandArgs, {
  cwd: process.cwd(),
  env,
  stdio: 'inherit',
});

if (result.error) {
  console.error(`${command}: ${result.error.message}`);
  process.exit(1);
}
if (result.signal) process.kill(process.pid, result.signal);
process.exit(result.status ?? 1);
