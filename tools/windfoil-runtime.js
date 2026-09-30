const HOSTS = {
  node: { backend: 'dawn', prefix: [] },
  deno: { backend: 'wgpu', prefix: ['run', '--unstable-webgpu', '-A', '--node-modules-dir=auto'] },
  'deno-dawn': { backend: 'dawn', prefix: ['run', '-A', '--node-modules-dir=auto'] },
};

export const WINDFOIL_ENVIRONMENTS = Object.freeze(Object.keys(HOSTS));

export function windfoilBackend(environment) {
  const host = HOSTS[environment];
  if (!host) throw new Error(`unsupported Windfoil environment: ${environment}`);
  return host.backend;
}

export function windfoilCommand(environment, args) {
  const host = HOSTS[environment];
  if (!host) throw new Error(`unsupported Windfoil environment: ${environment}`);
  const command = environment === 'node'
    ? process.env.NODE_BIN ?? process.execPath
    : process.env.DENO_BIN ?? 'deno';
  return {
    command,
    args: [...host.prefix, ...args],
    env: { ...process.env, WF_WEBGPU_BACKEND: host.backend },
  };
}
