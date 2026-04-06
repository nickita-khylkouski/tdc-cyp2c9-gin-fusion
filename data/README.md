# Data

## Benchmark

This repository targets:

- `TDC.CYP2C9_Veith`

Official benchmark pages:

- overview: <https://tdcommons.ai/benchmark/overview/>
- leaderboard: <https://tdcommons.ai/benchmark/admet_group/10cyp2c9i/>

## Data source

Obtain the benchmark data through the official TDC Python package and benchmark interfaces. This repository does not redistribute the benchmark dataset itself.

## Expected prepared layout

The reproduction scripts expect prepared per-seed splits under:

```text
.chem/prepared_gin_gpu/cyp2c9_veith/
  1/train.csv
  1/val.csv
  test.csv
  2/train.csv
  2/val.csv
  ...
  5/train.csv
  5/val.csv
```

## Required columns

The training pipeline expects CSV files with:

- a SMILES column such as `Drug`, `SMILES`, or `smiles`
- a target column such as `Y`

## Why data is not bundled

The benchmark data belongs to the TDC benchmark distribution. The public release includes code, configuration, and result packets, but not redistributed benchmark tables.
