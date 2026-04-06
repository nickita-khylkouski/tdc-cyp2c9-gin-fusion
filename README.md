# TDC CYP2C9 GIN Fusion

Frozen public release for a `TDC.CYP2C9_Veith` benchmark submission based on descriptor-graph fusion.

## Abstract

This repository packages a frozen five-run benchmark result for `TDC.CYP2C9_Veith`, a scaffold-split molecular classification task evaluated with `AUPRC`. The model combines an `exact-MapLight 1024` chemistry feature block with pretrained `gin_supervised_masking` graph embeddings from DGL-LifeSci and a `CatBoost` fusion head. The goal of this release is reproducibility and benchmark transparency: it exposes the runner, the preserved configuration, the submission packet, and a concise compliance audit for the released result.

## Model card

| Component | Choice |
| --- | --- |
| Chemistry features | `exact-MapLight 1024` |
| Graph encoder | `gin_supervised_masking` |
| Fusion head | `CatBoost` |
| Benchmark | `TDC.CYP2C9_Veith` |
| Metric | `AUPRC` |
| Split | `Scaffold` |
| Runs | `5` |

## Frozen result

Frozen submission candidate on `TDC.CYP2C9_Veith`:

| Field | Value |
| --- | --- |
| Benchmark | `TDC.CYP2C9_Veith` |
| Metric | `AUPRC` |
| Split | `Scaffold` |
| Runs | `5` |
| Score | `0.8580798922058548 ± 0.001236962051413275` |

Per-seed test scores:

- `0.8580033686993394`
- `0.8586647011725154`
- `0.8578495349256301`
- `0.8560421348549657`
- `0.8598397213768233`

At the time this run was frozen, that score was effectively tied with the public `#1` row on the TDC `CYP2C9_Veith` leaderboard.

## Method summary

The pipeline fuses two complementary signal sources:

1. hand-crafted medicinal chemistry features from the `exact-MapLight 1024` block
2. pretrained graph representations from `gin_supervised_masking`

Those features are concatenated and scored with a `CatBoost` classifier.

The descriptor block and graph embedding block are concatenated and scored with a `CatBoost` classifier. This repository packages the frozen run configuration, the core reproduction scripts, and the benchmark submission materials for that exact configuration.

## Why this repo exists

This repository is the clean public release for one frozen benchmark entry:

- the reproducible runner
- the frozen result packet
- the submission notes
- the compliance audit

It is intentionally smaller than the private research workspace that produced it.

## Repository layout

- [`scripts/tdc_admet_baseline.py`](scripts/tdc_admet_baseline.py): feature generation, benchmark loading, model utilities
- [`scripts/tdc_admet_pretrained_gin_fusion.py`](scripts/tdc_admet_pretrained_gin_fusion.py): GIN embedding fusion runner
- [`scripts/tdc_admet_graph_fusion.py`](scripts/tdc_admet_graph_fusion.py): graph-fusion helper utilities
- [`results/cyp2c9_veith/frozen_submission_summary.json`](results/cyp2c9_veith/frozen_submission_summary.json): frozen 5-seed submission packet
- [`results/cyp2c9_veith/run_config.json`](results/cyp2c9_veith/run_config.json): preserved run configuration
- [`results/cyp2c9_veith/seed_1_metrics.json`](results/cyp2c9_veith/seed_1_metrics.json): sample seed metrics artifact
- [`docs/submission.md`](docs/submission.md): paste-ready TDC submission notes
- [`docs/audit.md`](docs/audit.md): compliance and similarity audit

## Reproduction sketch

1. Prepare official TDC benchmark splits for `CYP2C9_Veith`.
2. Build `exact-MapLight 1024` features for train/valid/test.
3. Load pretrained `gin_supervised_masking` via DGL-LifeSci.
4. Extract graph embeddings for train/valid/test.
5. Concatenate descriptor and embedding blocks.
6. Fit `CatBoost` over 5 scaffold-split seeds.
7. Evaluate `AUPRC` on the official TDC test split and aggregate mean/std.

The core runner is:

```bash
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
```

## Dependencies

Core dependencies:

- `torch==2.2.1+cu121`
- `dgl==2.4.0+cu121`
- `dgllife`
- `rdkit`
- `catboost`
- `numpy`
- `pandas`
- `scikit-learn`

See [`requirements.txt`](requirements.txt).

## Submission status

This package is structured for TDC leaderboard submission, but the repository itself does not submit anything automatically. The benchmark submission flow is summarized in [`docs/submission.md`](docs/submission.md).

## Notes on scope

- This is a benchmark release, not a claim of a new model architecture.
- The method is in the same broad family as prior descriptor-plus-graph leaderboard entries, but this repository contains a distinct implementation and a frozen result packet for one exact run family.
- Later exploratory post-test sweeps are intentionally excluded from this public release.

## Benchmark links

- GitHub repository: <https://github.com/nickita-khylkouski/tdc-cyp2c9-gin-fusion>
- TDC leaderboard guide: <https://tdcommons.ai/benchmark/overview/>
- TDC `CYP2C9_Veith` leaderboard: <https://tdcommons.ai/benchmark/admet_group/10cyp2c9i/>

## License

This repository is released under the MIT License. Third-party dependencies keep their own licenses.
