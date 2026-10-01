#!/usr/bin/env bash
# One-shot pod setup for the three-engine benchmark (Windfoil, DiffVG, Bézier).
# Idempotent: safe to re-run. Assumes an NVIDIA CUDA pod with the repo at $ROOT.
#
# Create the pod with NVIDIA_DRIVER_CAPABILITIES including `graphics` (`all`
# works). Without it the driver exposes no working Vulkan ICD and Windfoil
# silently falls back to the llvmpipe CPU rasterizer -- correct pictures,
# worthless timings. The "vulkan" step below detects that and repairs it.
set -euo pipefail

ROOT=${ROOT:-/root/wf}
NODE_VERSION=${NODE_VERSION:-v22.23.1}
export PATH=/usr/local/cuda/bin:/root/node/bin:$PATH
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}

step() { printf '\n=== %s ===\n' "$1"; }

step "system packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
# libegl1/libglvnd0/libglx0 are the GLVND dispatch libraries. Dawn reports the
# unhelpful "no drivers" when libEGL.so.1 is missing, which stock CUDA images
# routinely lack.
apt-get install -y -qq build-essential cmake ninja-build git curl kmod \
  mesa-vulkan-drivers vulkan-tools libgl1 libegl1 libglvnd0 libglx0 libxext6

step "vulkan"
# Stock cloud CUDA containers often ship the *compute* half of the driver only:
# libGLX_nvidia.so.0 is present and exports the ICD entry points, but every one
# resolves to NULL. Probe for that before trusting anything Windfoil times.
nvidia_icd_works() {
  python3 - <<'PY' 2>/dev/null
import ctypes, sys
try:
    lib = ctypes.CDLL("libGLX_nvidia.so.0")
    fn = lib.vk_icdGetInstanceProcAddr
    fn.restype = ctypes.c_void_p
    fn.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    sys.exit(0 if fn(None, b"vkCreateInstance") else 1)
except Exception:
    sys.exit(1)
PY
}

if ! nvidia_icd_works; then
  # Install the userspace half of the *exact* host driver version. The kernel
  # module is already loaded and is left alone, so CUDA keeps working. The
  # installer's "Unable to delete existing file (Device or resource busy)"
  # errors and its systemctl failure are expected: those are the files the
  # container runtime bind-mounts read-only.
  driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
  echo "NVIDIA Vulkan ICD is non-functional; installing userspace driver ${driver}"
  curl -fsSL -o /tmp/nvidia.run \
    "https://us.download.nvidia.com/XFree86/Linux-x86_64/${driver}/NVIDIA-Linux-x86_64-${driver}.run"
  sh /tmp/nvidia.run --no-kernel-module --silent --no-questions --ui=none \
    --no-nouveau-check --no-rebuild-initramfs --install-libglvnd || true
  rm -f /tmp/nvidia.run
  ldconfig
  nvidia_icd_works || {
    echo "ERROR: NVIDIA Vulkan ICD still unusable -- Windfoil would run on llvmpipe." >&2
    exit 1
  }
fi

# Pin the ICD for every later command: llvmpipe stays installed and enumerable,
# and Dawn will pick it given the chance. Written as a loop rather than `ls a b
# | head -1` because ls exits non-zero when either path is absent, which under
# `set -euo pipefail` kills the script with no diagnostic at all.
WF_VULKAN_ICD=
for candidate in /etc/vulkan/icd.d/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json; do
  if [ -f "$candidate" ]; then
    WF_VULKAN_ICD=$candidate
    break
  fi
done
if [ -z "$WF_VULKAN_ICD" ]; then
  echo "ERROR: no nvidia_icd.json found; Dawn would fall back to llvmpipe." >&2
  exit 1
fi
export VK_DRIVER_FILES="$WF_VULKAN_ICD"
export WF_WEBGPU_BACKEND=dawn
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/tmp/xdg}
mkdir -p "$XDG_RUNTIME_DIR" && chmod 700 "$XDG_RUNTIME_DIR"
vulkaninfo --summary 2>/dev/null | grep -m1 deviceName || true
for line in \
  "export VK_DRIVER_FILES=$WF_VULKAN_ICD" \
  "export WF_WEBGPU_BACKEND=dawn" \
  "export XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR"; do
  grep -qxF "$line" /root/.bashrc 2>/dev/null || echo "$line" >> /root/.bashrc
done

step "node ${NODE_VERSION}"
if [ ! -x /root/node/bin/node ]; then
  curl -fsSL -o /tmp/node.tar.xz \
    "https://nodejs.org/dist/${NODE_VERSION}/node-${NODE_VERSION}-linux-x64.tar.xz"
  mkdir -p /root/node
  tar -xJf /tmp/node.tar.xz -C /root/node --strip-components=1
  rm -f /tmp/node.tar.xz
fi
node --version
grep -q '/root/node/bin' /root/.bashrc 2>/dev/null || \
  echo 'export PATH=/root/node/bin:/usr/local/cuda/bin:$PATH' >> /root/.bashrc

step "javascript deps"
cd "$ROOT"
npm install --no-audit --no-fund

step "bezier submodules (non-recursive: upstream's gsplat gitlink has no URL)"
# Both are cloned at their pinned commits; bench/bezier/run.py re-checks the SHA
# and refuses a dirty or moved checkout. Done by URL rather than `git submodule`
# so this also works from an archive of the repo, where there is no parent .git.
clone_pinned() {
  local path=$1 url=$2 commit=$3
  if [ -e "$path/.git" ] && [ "$(git -C "$path" rev-parse HEAD)" = "$commit" ]; then
    echo "ok   $path @ ${commit:0:8}"
    return
  fi
  rm -rf "$path"
  git clone -q "$url" "$path"
  git -C "$path" checkout -q "$commit"
  echo "done $path @ ${commit:0:8}"
}
clone_pinned bench/bezier/upstream \
  https://github.com/xiliu8006/Bezier_splatting.git 9612a228bc0662e26e06840f2ed2b187bc366f8c
clone_pinned bench/bezier/gsplat \
  https://github.com/XingtongGe/gsplat.git bcca3ecae966a052e3bf8dd1ff9910cf7b8f851d

step "fixtures"
node tools/fetch-fixtures.js all

step "diffvg (built for this GPU's compute capability)"
if [ ! -d "$ROOT/.venv-diffvg" ]; then
  python3 -m venv --system-site-packages "$ROOT/.venv-diffvg"
fi
"$ROOT/.venv-diffvg/bin/python" -m pip install -q --upgrade pip
"$ROOT/.venv-diffvg/bin/python" -m pip install -q -r bench/diffvg/requirements.txt
if ! "$ROOT/.venv-diffvg/bin/python" -c 'import pydiffvg' 2>/dev/null; then
  rm -rf /root/diffvg
  git clone -q --recursive https://github.com/BachiLi/diffvg.git /root/diffvg
  (
    cd /root/diffvg
    git checkout -q 85802a71fbcc72d79cb75716eb4da4392fd09532
    git submodule update --init --recursive -q
    (cd pybind11 && git fetch -q --depth 1 origin tag v2.13.6 && git checkout -q --detach v2.13.6)
    # diffvg's legacy find_package(CUDA) ignores CMAKE_CUDA_ARCHITECTURES and emits
    # no -gencode, so nvcc builds only its default targets. Without the pod GPU's
    # arch the module imports and renders forward, then dies in the first backward
    # pass with `radix_sort: cudaErrorInvalidDevice`.
    ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
    echo "building diffvg for sm_${ARCH}"
    sed -i "s/find_package(CUDA 10 REQUIRED)/find_package(CUDA 10 REQUIRED)\n    list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_${ARCH},code=sm_${ARCH})/" CMakeLists.txt
    DIFFVG_CUDA=1 "$ROOT/.venv-diffvg/bin/python" setup.py install
  )
fi
"$ROOT/.venv-diffvg/bin/python" -c 'import pydiffvg, torch; assert torch.cuda.is_available()'

step "bezier splatting / gsplat"
if [ ! -d "$ROOT/.venv-bezier" ]; then
  python3 -m venv --system-site-packages "$ROOT/.venv-bezier"
fi
"$ROOT/.venv-bezier/bin/python" -m pip install -q --upgrade pip
"$ROOT/.venv-bezier/bin/python" -m pip install -q -r bench/bezier/requirements.txt
if ! "$ROOT/.venv-bezier/bin/python" -c 'import gsplat' 2>/dev/null; then
  ARCH_DOT=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
  echo "building gsplat for compute ${ARCH_DOT}"
  TORCH_CUDA_ARCH_LIST="$ARCH_DOT" "$ROOT/.venv-bezier/bin/python" \
    -m pip install --no-build-isolation -e bench/bezier/gsplat
fi
"$ROOT/.venv-bezier/bin/python" -c 'import gsplat, torch; assert gsplat.__file__ and torch.cuda.is_available()'

step "clip loss server"
# The default sweep's `clip` stage talks to demos/clip/loss_server.py, which
# `npm run clip:server` runs out of .venv. open_clip and websockets only --
# requirements-clip.txt pins torch 2.11, but the pod's CUDA build is what every
# other engine here uses, and --system-site-packages keeps it.
if [ ! -d "$ROOT/.venv" ]; then
  python3 -m venv --system-site-packages "$ROOT/.venv"
fi
"$ROOT/.venv/bin/python" -m pip install -q --upgrade pip
"$ROOT/.venv/bin/python" -m pip install -q open_clip_torch==3.3.0 websockets==16.1
"$ROOT/.venv/bin/python" -c 'import open_clip, websockets, torch; assert torch.cuda.is_available()'

step "report tools"
# Install the plotting dependency in the Bézier venv.
"$ROOT/.venv-bezier/bin/python" -m pip install -q -r bench/requirements.txt

step "done"
# node, npm and the Vulkan ICD pin were appended to ~/.bashrc, which the shell
# that ran this script has not read: without this, `npm` is not found, and a
# Windfoil run that did start could fall back to llvmpipe.
echo "First, in this shell (new shells pick it up by themselves):"
echo "  source ~/.bashrc"
echo
echo "then verify every engine is really on the GPU:"
echo "  $ROOT/.venv-diffvg/bin/python bench/gpu-verify.py --engine=diffvg"
echo "  $ROOT/.venv-bezier/bin/python bench/gpu-verify.py --engine=bezier"
echo
echo "then (a sweep is hours long -- run it inside tmux or under nohup; an"
echo "interrupted one resumes; --skip=clip unless \`npm run clip:server\` is up):"
echo "  npm run bench -- --skip=clip"
echo "  $ROOT/.venv-bezier/bin/python bench/report.py output/<run>"
