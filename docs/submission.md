# TDC Submission Packet

## Benchmark

- `TDC.CYP2C9_Veith`
- metric: `AUPRC`
- split: `Scaffold`

## Frozen result

- mean: `0.8580798922058548`
- std: `0.001236962051413275`
- runs: `5`

## Model name

`MapLight-GIN-CatBoost Fusion`

## Short description

Hybrid CYP2C9 inhibitor classifier using `exact-MapLight 1024` chemical features, pretrained `gin_supervised_masking` graph embeddings from DGL-LifeSci, and a `CatBoost` fusion head. The final score is reported over 5 scaffold-split runs.

## Hardware

- NVIDIA H100 GPUs

## Code link

- <https://github.com/nickita-khylkouski/tdc-cyp2c9-gin-fusion>

## Benchmark links

- TDC overview: <https://tdcommons.ai/benchmark/overview/>
- CYP2C9_Veith leaderboard: <https://tdcommons.ai/benchmark/admet_group/10cyp2c9i/>
- submission form: <https://forms.gle/HYupGaV7WDuutbr9A>
