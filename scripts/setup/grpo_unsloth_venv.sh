#!/usr/bin/env bash
set -euo pipefail

# Setup script for GRPO + Unsloth QLoRA — A100 (CUDA 12.4) target.
#
# Usage:
#   bash scripts/setup/grpo_unsloth_venv.sh .venv_grpo_unsloth
#
# Optional env vars:
#   PYTHON_BIN=python3.12
#   CUDA_TAG=cu124           # cu121 / cu124 — must match driver on target machine
#   TORCH_VERSION=2.5.1      # PyTorch version to install
#
# Why torch is installed BEFORE unsloth:
#   uv does not honour the index-url embedded in unsloth's extras (e.g.
#   unsloth[cu124-torch250]). Letting unsloth pull torch resolves to the CPU
#   build from PyPI. Installing torch from the official pytorch CUDA wheel
#   index first guarantees the correct CUDA-enabled build, then unsloth is
#   installed without extras so it adopts the already-present torch.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${1:-.venv_grpo_unsloth}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
CUDA_TAG="${CUDA_TAG:-cu124}"
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"

cd "${ROOT_DIR}"

echo "=== Creating venv: ${VENV_DIR} ==="
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip wheel uv
python -m pip install --upgrade "setuptools==80.10.2"

# --- Step 1: torch with CUDA (must be before unsloth) ---
echo "=== Installing torch ${TORCH_VERSION}+${CUDA_TAG} ==="
uv pip install \
  --index-url "https://download.pytorch.org/whl/${CUDA_TAG}" \
  --extra-index-url https://pypi.org/simple \
  "torch==${TORCH_VERSION}+${CUDA_TAG}"

# Verify torch sees CUDA before continuing
python - <<'PYCHECK'
import torch, sys
if not torch.cuda.is_available():
    print(f"ERROR: torch {torch.__version__} has no CUDA. Check CUDA driver and CUDA_TAG.", file=sys.stderr)
    sys.exit(1)
print(f"torch {torch.__version__} — CUDA OK: {torch.cuda.get_device_name(0)}")
PYCHECK

# --- Step 2: unsloth (no extras — adopts torch already installed above) ---
echo "=== Installing Unsloth ==="
uv pip install unsloth

# --- Step 3: remaining GRPO packages ---
echo "=== Installing GRPO packages ==="
uv pip install -r requirements/grpo_unsloth_qlora.txt

python -m pip check

echo ""
echo "=== Unsloth QLoRA environment ready at ${VENV_DIR} ==="
echo "Activate: source ${VENV_DIR}/bin/activate"
echo ""
echo "Smoke test:"
echo "  python -c \""
echo "    import torch, unsloth, trl, transformers, bitsandbytes as bnb;"
echo "    print(torch.__version__, unsloth.__version__, trl.__version__);"
echo "    print('CUDA:', torch.cuda.get_device_name(0));"
echo "    print('bf16:', torch.cuda.is_bf16_supported())"
echo "  \""
