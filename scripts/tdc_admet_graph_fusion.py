#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.tdc_admet_baseline import (
    build_model,
    ensure_dir,
    evaluate_predictions,
    find_smiles_column,
    find_target_column,
    infer_task_type,
    json_default,
    load_benchmark_frames,
    load_prepared_split_frames,
    predict_scores,
    smiles_to_matrix,
    utc_now,
    write_json,
)


def run_cmd(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
    proc_env = os.environ.copy()
    proc_env.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    if env:
        proc_env.update(env)
    subprocess.run(cmd, cwd=cwd, env=proc_env, check=True)


def find_fingerprint_output(output_stem: Path) -> Path:
    exact = output_stem if output_stem.suffix == ".npz" else output_stem.with_suffix(".npz")
    if exact.exists():
        return exact
    stem = exact.stem
    matches = sorted(output_stem.parent.glob(f"{stem}_*.npz"))
    if not matches:
        raise FileNotFoundError(f"Could not find chemprop fingerprint output for stem {output_stem}")
    return matches[0]


def load_fingerprint_matrix(npz_path: Path) -> np.ndarray:
    payload = np.load(npz_path)
    if "H" in payload.files:
        return np.asarray(payload["H"], dtype=np.float32)
    if not payload.files:
        raise ValueError(f"No arrays found in {npz_path}")
    return np.asarray(payload[payload.files[0]], dtype=np.float32)


def ensure_chemprop_fingerprints(
    *,
    split_csv: Path,
    output_stem: Path,
    model_dir: Path,
    ffn_block_index: int,
    device: str,
    num_workers: int,
    molecule_featurizers: list[str],
) -> np.ndarray:
    existing = None
    try:
        existing = find_fingerprint_output(output_stem)
    except FileNotFoundError:
        existing = None
    if existing is not None:
        return load_fingerprint_matrix(existing)

    cmd = [
        "chemprop",
        "fingerprint",
        "-i",
        str(split_csv),
        "-o",
        str(output_stem.with_suffix(".npz")),
        "--model-paths",
        str(model_dir),
        "--ffn-block-index",
        str(ffn_block_index),
        "--smiles-columns",
        "smiles",
        "--num-workers",
        str(num_workers),
    ]
    if molecule_featurizers:
        cmd.extend(["--molecule-featurizers", *molecule_featurizers])
    env: dict[str, str] | None = None
    if device != "cpu":
        env = {"CUDA_VISIBLE_DEVICES": device}
        cmd.extend(["--accelerator", "gpu", "--devices", "1"])
    else:
        cmd.extend(["--accelerator", "cpu"])

    run_cmd(cmd, ROOT, env=env)
    return load_fingerprint_matrix(find_fingerprint_output(output_stem))


def infer_metric_name(benchmark_name: str) -> str:
    benchmark = benchmark_name.strip().lower()
    if benchmark in {"cyp2c9_veith", "cyp2d6_veith", "cyp3a4_veith", "cyp2c9_substrate_carbonmangels", "cyp2d6_substrate_carbonmangels"}:
        return "auprc"
    return "auroc"


def find_target_column_flexible(frame: pd.DataFrame) -> str:
    if "target" in frame.columns:
        return "target"
    return find_target_column(frame)


def fit_seed(
    *,
    benchmark_name: str,
    seed: int,
    data_dir: Path,
    prepared_dir: Path | None,
    chemprop_run_root: Path,
    feature_set: str,
    fingerprint_bits: int,
    model_name: str,
    cpu_only_model: bool,
    ffn_block_index: int,
    device: str,
    num_workers: int,
    molecule_featurizers: list[str],
    output_dir: Path,
    evaluate_split: str,
) -> dict[str, Any]:
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from types import SimpleNamespace

    chemprop_seed_root = chemprop_run_root / f"seed-{seed}"
    direct_train = chemprop_seed_root / "train.csv"
    direct_valid = chemprop_seed_root / "valid.csv"
    direct_test = chemprop_seed_root / "test.csv"
    nested_train = chemprop_seed_root / "splits" / "train.csv"
    nested_valid = chemprop_seed_root / "splits" / "valid.csv"
    nested_test = chemprop_seed_root / "splits" / "test.csv"

    if direct_train.exists() and direct_valid.exists() and direct_test.exists():
        frame_set = SimpleNamespace(
            benchmark=benchmark_name,
            task_type=infer_task_type(benchmark_name),
            train=pd.read_csv(direct_train),
            valid=pd.read_csv(direct_valid),
            test=pd.read_csv(direct_test),
        )
    elif nested_train.exists() and nested_valid.exists() and nested_test.exists():
        frame_set = SimpleNamespace(
            benchmark=benchmark_name,
            task_type=infer_task_type(benchmark_name),
            train=pd.read_csv(nested_train),
            valid=pd.read_csv(nested_valid),
            test=pd.read_csv(nested_test),
        )
    elif prepared_dir is not None:
        frame_set = load_prepared_split_frames(prepared_dir=prepared_dir, benchmark_name=benchmark_name, seed=seed)
    else:
        frame_set = load_benchmark_frames(data_dir=data_dir, benchmark_name=benchmark_name, seed=seed)

    smiles_col = find_smiles_column(frame_set.train)
    target_col = find_target_column_flexible(frame_set.train)

    train_base, feature_names, train_stats = smiles_to_matrix(
        frame_set.train[smiles_col].astype(str).tolist(),
        feature_set=feature_set,
        fingerprint_bits=fingerprint_bits,
    )
    valid_base, _, valid_stats = smiles_to_matrix(
        frame_set.valid[smiles_col].astype(str).tolist(),
        feature_set=feature_set,
        fingerprint_bits=fingerprint_bits,
    )
    test_base, _, test_stats = smiles_to_matrix(
        frame_set.test[smiles_col].astype(str).tolist(),
        feature_set=feature_set,
        fingerprint_bits=fingerprint_bits,
    )

    seed_dir = output_dir / f"seed_{seed}"
    ensure_dir(seed_dir)
    split_dir = ensure_dir(seed_dir / "splits")
    train_csv = split_dir / "train.csv"
    valid_csv = split_dir / "valid.csv"
    test_csv = split_dir / "test.csv"
    frame_set.train[[smiles_col, target_col]].rename(columns={smiles_col: "smiles", target_col: "target"}).to_csv(train_csv, index=False)
    frame_set.valid[[smiles_col, target_col]].rename(columns={smiles_col: "smiles", target_col: "target"}).to_csv(valid_csv, index=False)
    frame_set.test[[smiles_col, target_col]].rename(columns={smiles_col: "smiles", target_col: "target"}).to_csv(test_csv, index=False)

    chemprop_seed_dir = chemprop_seed_root / "chemprop"
    if not chemprop_seed_dir.exists():
        raise FileNotFoundError(f"Missing chemprop seed dir: {chemprop_seed_dir}")

    train_fp = ensure_chemprop_fingerprints(
        split_csv=train_csv,
        output_stem=seed_dir / f"train_fps_block{ffn_block_index}",
        model_dir=chemprop_seed_dir,
        ffn_block_index=ffn_block_index,
        device=device,
        num_workers=num_workers,
        molecule_featurizers=molecule_featurizers,
    )
    valid_fp = ensure_chemprop_fingerprints(
        split_csv=valid_csv,
        output_stem=seed_dir / f"valid_fps_block{ffn_block_index}",
        model_dir=chemprop_seed_dir,
        ffn_block_index=ffn_block_index,
        device=device,
        num_workers=num_workers,
        molecule_featurizers=molecule_featurizers,
    )
    test_fp = ensure_chemprop_fingerprints(
        split_csv=test_csv,
        output_stem=seed_dir / f"test_fps_block{ffn_block_index}",
        model_dir=chemprop_seed_dir,
        ffn_block_index=ffn_block_index,
        device=device,
        num_workers=num_workers,
        molecule_featurizers=molecule_featurizers,
    )

    train_matrix = np.concatenate([train_base, train_fp], axis=1)
    valid_matrix = np.concatenate([valid_base, valid_fp], axis=1)
    test_matrix = np.concatenate([test_base, test_fp], axis=1)

    y_train = frame_set.train[target_col].to_numpy()
    y_valid = frame_set.valid[target_col].to_numpy()
    y_test = frame_set.test[target_col].to_numpy()

    model = build_model(model_name=model_name, task_type=frame_set.task_type, seed=seed, use_gpu=not cpu_only_model)
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("model", model),
        ]
    )
    pipeline.fit(train_matrix, y_train)
    pred_valid = predict_scores(pipeline, valid_matrix, task_type=frame_set.task_type)
    pred_test = predict_scores(pipeline, test_matrix, task_type=frame_set.task_type)

    eval_target = y_valid if evaluate_split == "valid" else y_test
    eval_scores = pred_valid if evaluate_split == "valid" else pred_test
    metric_name = infer_metric_name(benchmark_name)
    metric_value = evaluate_predictions(metric_name=metric_name, true=eval_target, pred=eval_scores)

    pd.DataFrame(
        {"smiles": frame_set.valid[smiles_col].astype(str), "target": y_valid, "prediction": pred_valid}
    ).to_csv(seed_dir / "predictions_valid.csv", index=False)
    pd.DataFrame(
        {"smiles": frame_set.test[smiles_col].astype(str), "target": y_test, "prediction": pred_test}
    ).to_csv(seed_dir / "predictions_test.csv", index=False)

    payload = {
        "benchmark": benchmark_name,
        "seed": seed,
        "model": model_name,
        "feature_set": feature_set,
        "fingerprint_bits": fingerprint_bits,
        "ffn_block_index": ffn_block_index,
        "chemprop_run_root": str(chemprop_run_root),
        "chemprop_fp_dim": int(train_fp.shape[1]),
        "evaluate_split": evaluate_split,
        "metric_name": metric_name,
        "metric_value": float(metric_value),
        "counts": {"train": int(len(y_train)), "valid": int(len(y_valid)), "test": int(len(y_test))},
        "feature_stats": {
            "base_feature_count": len(feature_names),
            "graph_feature_count": int(train_fp.shape[1]),
            "train": train_stats,
            "valid": valid_stats,
            "test": test_stats,
        },
        "created_at": utc_now(),
    }
    write_json(seed_dir / "metrics.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Fuse Chemprop fingerprints with exact-MapLight features and fit a safe tabular head.")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--chemprop-run-root", required=True)
    parser.add_argument("--feature-set", default="maplight_exact", choices=["maplight_exact", "maplight_cyp_mech", "morgan_rdkit2d", "classical", "all"])
    parser.add_argument("--fingerprint-bits", default=1024, type=int)
    parser.add_argument("--ffn-block-index", default=0, type=int)
    parser.add_argument("--model", default="catboost", choices=["catboost", "extratrees", "rf", "logreg"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--evaluate-split", default="valid", choices=["valid", "test"])
    parser.add_argument("--data-dir", default=".chem/tdc_admet/data")
    parser.add_argument("--prepared-dir", default=None)
    parser.add_argument("--output-root", default=".chem/runs/tdc_admet_graph_fusion")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--cpu-only-model", action="store_true")
    parser.add_argument("--device", default="0")
    parser.add_argument("--num-workers", default=8, type=int)
    parser.add_argument("--molecule-featurizers", default="morgan_binary,rdkit_2d")
    args = parser.parse_args()

    run_name = args.run_name or (
        f"{args.benchmark.lower()}__graphfusion__{args.feature_set}__block{args.ffn_block_index}__"
        f"{args.evaluate_split}__{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    output_dir = ensure_dir(Path(args.output_root) / run_name)
    chemprop_run_root = Path(args.chemprop_run_root)
    prepared_dir = Path(args.prepared_dir) if args.prepared_dir else None
    molecule_featurizers = [item.strip() for item in args.molecule_featurizers.split(",") if item.strip()]

    write_json(
        output_dir / "config.json",
        {
            "benchmark": args.benchmark,
            "chemprop_run_root": str(chemprop_run_root),
            "feature_set": args.feature_set,
            "fingerprint_bits": args.fingerprint_bits,
            "ffn_block_index": args.ffn_block_index,
            "model": args.model,
            "seeds": args.seeds,
            "evaluate_split": args.evaluate_split,
            "cpu_only_model": args.cpu_only_model,
            "device": args.device,
            "num_workers": args.num_workers,
            "molecule_featurizers": molecule_featurizers,
            "created_at": utc_now(),
        },
    )

    seed_payloads = []
    for seed in args.seeds:
        seed_payloads.append(
            fit_seed(
                benchmark_name=args.benchmark,
                seed=seed,
                data_dir=Path(args.data_dir),
                prepared_dir=prepared_dir,
                chemprop_run_root=chemprop_run_root,
                feature_set=args.feature_set,
                fingerprint_bits=args.fingerprint_bits,
                model_name=args.model,
                cpu_only_model=args.cpu_only_model,
                ffn_block_index=args.ffn_block_index,
                device=args.device,
                num_workers=args.num_workers,
                molecule_featurizers=molecule_featurizers,
                output_dir=output_dir,
                evaluate_split=args.evaluate_split,
            )
        )

    scores = np.asarray([payload["metric_value"] for payload in seed_payloads], dtype=np.float64)
    summary = {
        "run_name": run_name,
        "benchmark": args.benchmark,
        "model": args.model,
        "feature_set": args.feature_set,
        "fingerprint_bits": args.fingerprint_bits,
        "ffn_block_index": args.ffn_block_index,
        "evaluate_split": args.evaluate_split,
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores, ddof=0)),
        "seeds": args.seeds,
        "chemprop_run_root": str(chemprop_run_root),
        "created_at": utc_now(),
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
