# TDC Form Answers

This file is designed to be pasted into the public TDC submission form linked in [`submission.md`](submission.md).

Because the Google Form requires sign-in, field labels may vary slightly from what is visible publicly. The content below is the exact text to paste into the corresponding sections.

## Benchmark / dataset

`TDC.CYP2C9_Veith`

## Metric

`AUPRC`

## Result

`0.8580798922058548 ± 0.001236962051413275`

## Number of runs

`5`

## Model name

`MapLight-GIN-CatBoost Fusion`

## Short method description

Hybrid CYP2C9 inhibitor classifier using exact-MapLight 1024 chemical features, pretrained gin_supervised_masking graph embeddings from DGL-LifeSci, and a CatBoost fusion head. The reported result is the mean and standard deviation over 5 scaffold-split runs on TDC.CYP2C9_Veith.

## Hardware

`NVIDIA H100 GPUs`

## Code / repository link

`https://github.com/nickita-khylkouski/tdc-cyp2c9-gin-fusion`

## Additional notes

This repository is the frozen public release for the submitted benchmark run. It excludes later post-test exploratory sweeps and packages the exact configuration, result summary, and audit notes for the submission candidate.
