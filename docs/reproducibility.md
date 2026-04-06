# Reproducibility

## Scope

This repository packages the frozen `TDC.CYP2C9_Veith` submission candidate and the core scripts needed to reproduce the same run family.

## 1. Get the benchmark data

Follow the instructions in [`../data/README.md`](../data/README.md) to obtain the official TDC benchmark split and place the prepared data under:

```text
.chem/prepared_gin_gpu/
```

## 2. Build the environment

```bash
bash scripts/setup_cuda_env.sh .venv
source .venv/bin/activate
```

## 3. Run the frozen submission configuration

```bash
bash scripts/run_cyp2c9_submission.sh
```

## 4. Compare against the frozen packet

Expected frozen summary:

- benchmark: `TDC.CYP2C9_Veith`
- metric: `AUPRC`
- mean: `0.8580798922058548`
- std: `0.001236962051413275`

Reference packet:

- [`../results/cyp2c9_veith/frozen_submission_summary.json`](../results/cyp2c9_veith/frozen_submission_summary.json)

## Notes

- This release is intentionally narrow: it covers the frozen submission candidate, not the later private ablation queue.
- The package assumes access to CUDA-capable hardware for the published reproduction path.
