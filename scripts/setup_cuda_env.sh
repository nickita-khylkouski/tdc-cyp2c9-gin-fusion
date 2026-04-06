#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=${1:-.venv}

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip wheel setuptools

python -m pip install \
  torch==2.2.1+cu121 \
  torchvision==0.17.1+cu121 \
  torchaudio==2.2.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121

python -m pip install \
  dgl==2.4.0+cu121 \
  --find-links https://data.dgl.ai/wheels/torch-2.2/cu121/repo.html

python -m pip install -r requirements.txt

python - <<'PY'
import torch
import dgl
from dgllife.model import load_pretrained
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
print("dgl", dgl.__version__)
model = load_pretrained("gin_supervised_masking")
print("gin_loaded", model is not None)
PY
