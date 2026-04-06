# Compliance Audit

## Rule fit

This repository packages the frozen `test_all5` candidate for `TDC.CYP2C9_Veith`.

Why it is a defensible benchmark submission:

- the saved result is a 5-run `test` result, not a `valid`-only winner
- the run was promoted from `valid` before its final test value was known
- the method uses public libraries and public pretrained graph models rather than a benchmark-specific endpoint predictor

## Similarity to existing leaderboard methods

The method is in the same broad family as the public `MapLight + GNN` row:

- chemistry feature block
- graph representation block
- fused predictor

But it is not the same implementation:

- graph backbone: `gin_supervised_masking` from DGL-LifeSci
- fusion head: `CatBoost`
- repo-local feature fusion and experiment scripts

This should be described as a strong hybrid benchmark implementation, not as a novel architecture paper claim.

## Things not included

- old public `ADMET-AI` direct-prediction path
- post-test-selected micro-variants
- any result that was only chosen on `test`
