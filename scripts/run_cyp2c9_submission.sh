#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

python scripts/tdc_admet_pretrained_gin_fusion.py \
  --benchmark CYP2C9_Veith \
  --prepared-dir .chem/prepared_gin_gpu \
  --feature-set maplight_exact \
  --fingerprint-bits 1024 \
  --model catboost \
  --embedding-kind gin_supervised_masking \
  --evaluate-split test \
  --device cuda \
  --pooling mean \
  --seeds 1 2 3 4 5 \
  --run-name cyp_gin_fusion_test_all5_cuda
