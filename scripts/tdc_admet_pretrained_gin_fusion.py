#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
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
    infer_task_type,
    json_default,
    load_benchmark_frames,
    load_prepared_split_frames,
    predict_scores,
    smiles_to_matrix,
    utc_now,
    write_json,
)
from scripts.tdc_admet_graph_fusion import find_target_column_flexible


def build_transformer(kind: str, device: str) -> Any:
    import torch
    from dgllife.model import load_pretrained
    from dgllife.utils import PretrainAtomFeaturizer, PretrainBondFeaturizer

    dgl_cache_dir = ROOT / ".chem" / "cache" / "dgl"
    dgl_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("DGL_DOWNLOAD_DIR", str(dgl_cache_dir))

    model = load_pretrained(kind)
    torch_device = torch.device(device)
    model = model.to(torch_device)
    model.eval()
    return {
        "kind": kind,
        "device": torch_device,
        "model": model,
        "atom_featurizer": PretrainAtomFeaturizer(),
        "bond_featurizer": PretrainBondFeaturizer(),
    }


def compute_embeddings(transformer: Any, smiles: list[str]) -> np.ndarray:
    import dgl
    import torch
    from dgllife.utils import smiles_to_bigraph

    model = transformer["model"]
    torch_device = transformer["device"]
    atom_featurizer = transformer["atom_featurizer"]
    bond_featurizer = transformer["bond_featurizer"]
    pooling = transformer["pooling"]
    chunk_size = 128
    rows: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(smiles), chunk_size):
            batch_smiles = smiles[start : start + chunk_size]
            valid_graphs = []
            valid_positions: list[int] = []
            batch_rows: list[np.ndarray | None] = [None] * len(batch_smiles)
            for idx, value in enumerate(batch_smiles):
                try:
                    graph = smiles_to_bigraph(
                        value,
                        add_self_loop=True,
                        node_featurizer=atom_featurizer,
                        edge_featurizer=bond_featurizer,
                    )
                except Exception:
                    graph = None
                if graph is None:
                    continue
                valid_graphs.append(graph)
                valid_positions.append(idx)

            if valid_graphs:
                batched = dgl.batch(valid_graphs).to(torch_device)
                node_repr = model(
                    batched,
                    [batched.ndata["atomic_number"], batched.ndata["chirality_type"]],
                    [batched.edata["bond_type"], batched.edata["bond_direction_type"]],
                )
                batched.ndata["node_repr"] = node_repr
                if pooling == "mean":
                    pooled = dgl.mean_nodes(batched, "node_repr")
                elif pooling == "sum":
                    pooled = dgl.sum_nodes(batched, "node_repr")
                elif pooling == "max":
                    pooled = dgl.max_nodes(batched, "node_repr")
                elif pooling == "meanmax":
                    pooled = torch.cat(
                        [dgl.mean_nodes(batched, "node_repr"), dgl.max_nodes(batched, "node_repr")],
                        dim=1,
                    )
                else:
                    raise ValueError(f"Unsupported pooling mode: {pooling}")
                graph_repr = pooled.detach().cpu().numpy().astype(np.float32, copy=False)
                for local_idx, emb in zip(valid_positions, graph_repr, strict=True):
                    batch_rows[local_idx] = emb
                embedding_dim = int(graph_repr.shape[1])
            else:
                embedding_dim = 300

            zero = np.zeros((embedding_dim,), dtype=np.float32)
            for emb in batch_rows:
                rows.append(zero if emb is None else emb)

    arr = np.stack(rows, axis=0).astype(np.float32, copy=False)
    if arr.shape[0] != len(smiles):
        raise ValueError(f"Expected {len(smiles)} embeddings, got shape {arr.shape}")
    return arr


def ensure_embeddings(
    *,
    transformer: Any,
    smiles: list[str],
    output_path: Path,
) -> np.ndarray:
    if output_path.exists():
        return np.load(output_path).astype(np.float32, copy=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    emb = compute_embeddings(transformer, smiles)
    np.save(output_path, emb)
    return emb


def shared_embedding_path(
    *,
    cache_root: Path,
    benchmark_name: str,
    seed: int,
    split_name: str,
    embedding_kind: str,
    pooling: str,
) -> Path:
    safe_benchmark = benchmark_name.lower()
    return (
        cache_root
        / safe_benchmark
        / embedding_kind
        / pooling
        / f"seed_{seed}"
        / f"{split_name}.npy"
    )


def transform_embeddings(
    *,
    train_emb: np.ndarray,
    valid_emb: np.ndarray,
    test_emb: np.ndarray,
    normalize_embeddings: str,
    pca_components: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if normalize_embeddings == "none" and not pca_components:
        return train_emb, valid_emb, test_emb

    transformed_train = train_emb.astype(np.float32, copy=False)
    transformed_valid = valid_emb.astype(np.float32, copy=False)
    transformed_test = test_emb.astype(np.float32, copy=False)

    if normalize_embeddings == "l2":
        from sklearn.preprocessing import Normalizer

        normalizer = Normalizer(copy=False)
        transformed_train = normalizer.fit_transform(transformed_train).astype(np.float32, copy=False)
        transformed_valid = normalizer.transform(transformed_valid).astype(np.float32, copy=False)
        transformed_test = normalizer.transform(transformed_test).astype(np.float32, copy=False)
    elif normalize_embeddings == "standard":
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler(copy=False)
        transformed_train = scaler.fit_transform(transformed_train).astype(np.float32, copy=False)
        transformed_valid = scaler.transform(transformed_valid).astype(np.float32, copy=False)
        transformed_test = scaler.transform(transformed_test).astype(np.float32, copy=False)

    if pca_components:
        from sklearn.decomposition import PCA

        pca_dim = min(pca_components, transformed_train.shape[1], transformed_train.shape[0])
        pca = PCA(n_components=pca_dim, svd_solver="auto", random_state=0)
        transformed_train = pca.fit_transform(transformed_train).astype(np.float32, copy=False)
        transformed_valid = pca.transform(transformed_valid).astype(np.float32, copy=False)
        transformed_test = pca.transform(transformed_test).astype(np.float32, copy=False)

    return transformed_train, transformed_valid, transformed_test


def fit_seed(
    *,
    benchmark_name: str,
    seed: int,
    data_dir: Path,
    prepared_dir: Path | None,
    feature_set: str,
    fingerprint_bits: int,
    model_name: str,
    embedding_kind: str,
    pooling: str,
    device: str,
    output_dir: Path,
    evaluate_split: str,
    catboost_iterations: int | None,
    catboost_depth: int | None,
    catboost_learning_rate: float | None,
    catboost_l2_leaf_reg: float | None,
    catboost_random_strength: float | None,
    catboost_border_count: int | None,
    catboost_auto_class_weights: str | None,
    catboost_bagging_temperature: float | None,
    catboost_bootstrap_type: str | None,
    catboost_subsample: float | None,
    catboost_rsm: float | None,
    normalize_embeddings: str,
    pca_components: int | None,
    shared_embedding_cache_dir: Path,
) -> dict[str, Any]:
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    if prepared_dir is not None:
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

    seed_dir = ensure_dir(output_dir / f"seed_{seed}")
    transformer = build_transformer(embedding_kind, device)
    transformer["pooling"] = pooling
    cache_root = ensure_dir(shared_embedding_cache_dir)
    train_emb = ensure_embeddings(
        transformer=transformer,
        smiles=frame_set.train[smiles_col].astype(str).tolist(),
        output_path=shared_embedding_path(
            cache_root=cache_root,
            benchmark_name=benchmark_name,
            seed=seed,
            split_name="train",
            embedding_kind=embedding_kind,
            pooling=pooling,
        ),
    )
    valid_emb = ensure_embeddings(
        transformer=transformer,
        smiles=frame_set.valid[smiles_col].astype(str).tolist(),
        output_path=shared_embedding_path(
            cache_root=cache_root,
            benchmark_name=benchmark_name,
            seed=seed,
            split_name="valid",
            embedding_kind=embedding_kind,
            pooling=pooling,
        ),
    )
    test_emb = ensure_embeddings(
        transformer=transformer,
        smiles=frame_set.test[smiles_col].astype(str).tolist(),
        output_path=shared_embedding_path(
            cache_root=cache_root,
            benchmark_name=benchmark_name,
            seed=seed,
            split_name="test",
            embedding_kind=embedding_kind,
            pooling=pooling,
        ),
    )
    train_emb, valid_emb, test_emb = transform_embeddings(
        train_emb=train_emb,
        valid_emb=valid_emb,
        test_emb=test_emb,
        normalize_embeddings=normalize_embeddings,
        pca_components=pca_components,
    )

    train_matrix = np.concatenate([train_base, train_emb], axis=1)
    valid_matrix = np.concatenate([valid_base, valid_emb], axis=1)
    test_matrix = np.concatenate([test_base, test_emb], axis=1)

    y_train = frame_set.train[target_col].to_numpy()
    y_valid = frame_set.valid[target_col].to_numpy()
    y_test = frame_set.test[target_col].to_numpy()

    model = build_model(model_name=model_name, task_type=frame_set.task_type, seed=seed, use_gpu=False)
    if model_name.lower() == "catboost":
        override_params = {}
        if catboost_iterations is not None:
            override_params["iterations"] = catboost_iterations
        if catboost_depth is not None:
            override_params["depth"] = catboost_depth
        if catboost_learning_rate is not None:
            override_params["learning_rate"] = catboost_learning_rate
        if catboost_l2_leaf_reg is not None:
            override_params["l2_leaf_reg"] = catboost_l2_leaf_reg
        if catboost_random_strength is not None:
            override_params["random_strength"] = catboost_random_strength
        if catboost_border_count is not None:
            override_params["border_count"] = catboost_border_count
        if catboost_auto_class_weights is not None:
            override_params["auto_class_weights"] = catboost_auto_class_weights
        if catboost_bootstrap_type is not None:
            override_params["bootstrap_type"] = catboost_bootstrap_type
        if catboost_bagging_temperature is not None:
            override_params["bagging_temperature"] = catboost_bagging_temperature
        if catboost_subsample is not None:
            override_params["subsample"] = catboost_subsample
        if catboost_rsm is not None:
            override_params["rsm"] = catboost_rsm
        if override_params:
            model.set_params(**override_params)
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("model", model),
        ]
    )
    pipeline.fit(train_matrix, y_train)
    pred_valid = predict_scores(pipeline, valid_matrix, task_type=frame_set.task_type)
    pred_test = predict_scores(pipeline, test_matrix, task_type=frame_set.task_type)

    metric_name = "auprc" if benchmark_name == "CYP2C9_Veith" else "auroc"
    eval_target = y_valid if evaluate_split == "valid" else y_test
    eval_scores = pred_valid if evaluate_split == "valid" else pred_test
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
        "embedding_kind": embedding_kind,
        "pooling": pooling,
        "catboost_iterations": catboost_iterations,
        "catboost_depth": catboost_depth,
        "catboost_learning_rate": catboost_learning_rate,
        "catboost_l2_leaf_reg": catboost_l2_leaf_reg,
        "catboost_random_strength": catboost_random_strength,
        "catboost_border_count": catboost_border_count,
        "catboost_auto_class_weights": catboost_auto_class_weights,
        "catboost_bagging_temperature": catboost_bagging_temperature,
        "catboost_bootstrap_type": catboost_bootstrap_type,
        "catboost_subsample": catboost_subsample,
        "catboost_rsm": catboost_rsm,
        "feature_set": feature_set,
        "fingerprint_bits": fingerprint_bits,
        "evaluate_split": evaluate_split,
        "metric_name": metric_name,
        "metric_value": float(metric_value),
        "embedding_dim": int(train_emb.shape[1]),
        "normalize_embeddings": normalize_embeddings,
        "pca_components": pca_components,
        "base_feature_count": int(train_base.shape[1]),
        "counts": {"train": int(len(y_train)), "valid": int(len(y_valid)), "test": int(len(y_test))},
        "feature_stats": {"train": train_stats, "valid": valid_stats, "test": test_stats},
        "created_at": utc_now(),
    }
    write_json(seed_dir / "metrics.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse pretrained GIN-style molecular embeddings with descriptor features and fit a safe tabular head.")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--feature-set", default="maplight_exact", choices=["maplight_exact", "maplight_cyp_mech", "morgan_rdkit2d", "classical", "all"])
    parser.add_argument("--fingerprint-bits", default=1024, type=int)
    parser.add_argument("--model", default="catboost", choices=["catboost", "extratrees", "rf", "logreg"])
    parser.add_argument("--embedding-kind", default="gin_supervised_masking")
    parser.add_argument("--pooling", default="mean", choices=["mean", "sum", "max", "meanmax"])
    parser.add_argument("--catboost-iterations", type=int, default=None)
    parser.add_argument("--catboost-depth", type=int, default=None)
    parser.add_argument("--catboost-learning-rate", type=float, default=None)
    parser.add_argument("--catboost-l2-leaf-reg", type=float, default=None)
    parser.add_argument("--catboost-random-strength", type=float, default=None)
    parser.add_argument("--catboost-border-count", type=int, default=None)
    parser.add_argument("--catboost-auto-class-weights", choices=["Balanced", "SqrtBalanced"], default=None)
    parser.add_argument("--catboost-bagging-temperature", type=float, default=None)
    parser.add_argument("--catboost-bootstrap-type", choices=["Bayesian", "Bernoulli", "MVS", "No"], default=None)
    parser.add_argument("--catboost-subsample", type=float, default=None)
    parser.add_argument("--catboost-rsm", type=float, default=None)
    parser.add_argument("--normalize-embeddings", choices=["none", "l2", "standard"], default="none")
    parser.add_argument("--pca-components", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--evaluate-split", default="valid", choices=["valid", "test"])
    parser.add_argument("--data-dir", default=".chem/tdc_admet/data")
    parser.add_argument("--prepared-dir", default=None)
    parser.add_argument("--output-root", default=".chem/runs/tdc_admet_gin_fusion")
    parser.add_argument("--shared-embedding-cache-dir", default=".chem/cache/gin_embeddings")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_name = args.run_name or (
        f"{args.benchmark.lower()}__ginfusion__{args.feature_set}__{args.evaluate_split}__"
        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    output_dir = ensure_dir(Path(args.output_root) / run_name)
    prepared_dir = Path(args.prepared_dir) if args.prepared_dir else None

    write_json(
        output_dir / "config.json",
        {
            "benchmark": args.benchmark,
            "feature_set": args.feature_set,
            "fingerprint_bits": args.fingerprint_bits,
            "model": args.model,
            "embedding_kind": args.embedding_kind,
            "pooling": args.pooling,
            "catboost_iterations": args.catboost_iterations,
            "catboost_depth": args.catboost_depth,
            "catboost_learning_rate": args.catboost_learning_rate,
            "catboost_l2_leaf_reg": args.catboost_l2_leaf_reg,
            "catboost_random_strength": args.catboost_random_strength,
            "catboost_border_count": args.catboost_border_count,
            "catboost_auto_class_weights": args.catboost_auto_class_weights,
            "catboost_bagging_temperature": args.catboost_bagging_temperature,
            "catboost_bootstrap_type": args.catboost_bootstrap_type,
            "catboost_subsample": args.catboost_subsample,
            "catboost_rsm": args.catboost_rsm,
            "normalize_embeddings": args.normalize_embeddings,
            "pca_components": args.pca_components,
            "seeds": args.seeds,
            "evaluate_split": args.evaluate_split,
            "device": args.device,
            "created_at": utc_now(),
        },
    )

    seed_payloads = [
        fit_seed(
            benchmark_name=args.benchmark,
            seed=seed,
            data_dir=Path(args.data_dir),
            prepared_dir=prepared_dir,
            feature_set=args.feature_set,
            fingerprint_bits=args.fingerprint_bits,
            model_name=args.model,
            embedding_kind=args.embedding_kind,
            pooling=args.pooling,
            device=args.device,
            output_dir=output_dir,
            evaluate_split=args.evaluate_split,
            catboost_iterations=args.catboost_iterations,
            catboost_depth=args.catboost_depth,
            catboost_learning_rate=args.catboost_learning_rate,
            catboost_l2_leaf_reg=args.catboost_l2_leaf_reg,
            catboost_random_strength=args.catboost_random_strength,
            catboost_border_count=args.catboost_border_count,
            catboost_auto_class_weights=args.catboost_auto_class_weights,
            catboost_bagging_temperature=args.catboost_bagging_temperature,
            catboost_bootstrap_type=args.catboost_bootstrap_type,
            catboost_subsample=args.catboost_subsample,
            catboost_rsm=args.catboost_rsm,
            normalize_embeddings=args.normalize_embeddings,
            pca_components=args.pca_components,
            shared_embedding_cache_dir=Path(args.shared_embedding_cache_dir),
        )
        for seed in args.seeds
    ]

    scores = np.asarray([payload["metric_value"] for payload in seed_payloads], dtype=np.float64)
    summary = {
        "run_name": run_name,
        "benchmark": args.benchmark,
        "model": args.model,
        "embedding_kind": args.embedding_kind,
        "pooling": args.pooling,
        "catboost_iterations": args.catboost_iterations,
        "catboost_depth": args.catboost_depth,
        "catboost_learning_rate": args.catboost_learning_rate,
        "catboost_l2_leaf_reg": args.catboost_l2_leaf_reg,
        "catboost_random_strength": args.catboost_random_strength,
        "catboost_border_count": args.catboost_border_count,
        "catboost_auto_class_weights": args.catboost_auto_class_weights,
        "catboost_bagging_temperature": args.catboost_bagging_temperature,
        "catboost_bootstrap_type": args.catboost_bootstrap_type,
        "catboost_subsample": args.catboost_subsample,
        "catboost_rsm": args.catboost_rsm,
        "normalize_embeddings": args.normalize_embeddings,
        "pca_components": args.pca_components,
        "feature_set": args.feature_set,
        "fingerprint_bits": args.fingerprint_bits,
        "evaluate_split": args.evaluate_split,
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores, ddof=0)),
        "seeds": args.seeds,
        "created_at": utc_now(),
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
