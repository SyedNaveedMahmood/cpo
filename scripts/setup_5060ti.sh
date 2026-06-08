#!/usr/bin/env bash
# Setup script for RTX 5060 Ti 16GB (Blackwell, sm100 / CUDA 12.8+)
set -euo pipefail

echo "=== CPO/HCH Paper 1 — RTX 5060 Ti setup ==="

python -m pip install --upgrade pip setuptools wheel

# RTX 5060 Ti (Blackwell, sm_100) requires PyTorch built against CUDA 12.8+.
# The cu128 index contains Blackwell-compatible wheels.
python -m pip install --upgrade torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# bitsandbytes 0.45+ has first-class support for sm_100 (Blackwell).
# Earlier versions fall back to CUDA emulation which is slower and may
# produce slightly different quantization results.
python -m pip install --upgrade "bitsandbytes>=0.45.0"

# Install the CPO package and remaining dependencies.
python -m pip install -e .

# Verify the setup.
python - <<'PYCHECK'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print("gpu:", props.name)
    print(f"  VRAM: {props.total_memory / 1024**3:.1f} GiB")
    print(f"  CUDA capability: sm_{props.major}{props.minor}")
    if props.major < 10:
        print("  NOTE: sm < 10 detected.  If this is not Blackwell, cu128 "
              "wheels still work but bfloat16 4-bit quant may be slower.")
import bitsandbytes as bnb
print("bitsandbytes:", getattr(bnb, "__version__", "unknown"))
PYCHECK
