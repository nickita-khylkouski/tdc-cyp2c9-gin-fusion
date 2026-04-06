#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CLASSIFICATION_BENCHMARKS = {
    "hia_hou",
    "pgp_broccatelli",
    "bioavailability_ma",
    "bbb_martins",
    "cyp2c9_veith",
    "cyp2d6_veith",
    "cyp3a4_veith",
    "cyp2c9_substrate_carbonmangels",
    "cyp2d6_substrate_carbonmangels",
    "cyp3a4_substrate_carbonmangels",
    "herg",
    "ames",
    "dili",
}

REGRESSION_BENCHMARKS = {
    "caco2_wang",
    "lipophilicity_astrazeneca",
    "solubility_aqsoldb",
    "ppbr_az",
    "vdss_lombardo",
    "half_life_obach",
    "clearance_hepatocyte_az",
    "clearance_microsome_az",
    "ld50_zhu",
}

MAPLIGHT_DESCRIPTOR_NAMES = [
    "BalabanJ", "BertzCT", "Chi0", "Chi0n", "Chi0v", "Chi1",
    "Chi1n", "Chi1v", "Chi2n", "Chi2v", "Chi3n", "Chi3v", "Chi4n", "Chi4v",
    "EState_VSA1", "EState_VSA10", "EState_VSA11", "EState_VSA2", "EState_VSA3",
    "EState_VSA4", "EState_VSA5", "EState_VSA6", "EState_VSA7", "EState_VSA8",
    "EState_VSA9", "ExactMolWt", "FpDensityMorgan1", "FpDensityMorgan2",
    "FpDensityMorgan3", "FractionCSP3", "HallKierAlpha", "HeavyAtomCount",
    "HeavyAtomMolWt", "Ipc", "Kappa1", "Kappa2", "Kappa3", "LabuteASA",
    "MaxAbsEStateIndex", "MaxAbsPartialCharge", "MaxEStateIndex", "MaxPartialCharge",
    "MinAbsEStateIndex", "MinAbsPartialCharge", "MinEStateIndex", "MinPartialCharge",
    "MolLogP", "MolMR", "MolWt", "NHOHCount", "NOCount", "NumAliphaticCarbocycles",
    "NumAliphaticHeterocycles", "NumAliphaticRings", "NumAromaticCarbocycles",
    "NumAromaticHeterocycles", "NumAromaticRings", "NumHAcceptors", "NumHDonors",
    "NumHeteroatoms", "NumRadicalElectrons", "NumRotatableBonds",
    "NumSaturatedCarbocycles", "NumSaturatedHeterocycles", "NumSaturatedRings",
    "NumValenceElectrons", "PEOE_VSA1", "PEOE_VSA10", "PEOE_VSA11", "PEOE_VSA12",
    "PEOE_VSA13", "PEOE_VSA14", "PEOE_VSA2", "PEOE_VSA3", "PEOE_VSA4", "PEOE_VSA5",
    "PEOE_VSA6", "PEOE_VSA7", "PEOE_VSA8", "PEOE_VSA9", "RingCount", "SMR_VSA1",
    "SMR_VSA10", "SMR_VSA2", "SMR_VSA3", "SMR_VSA4", "SMR_VSA5", "SMR_VSA6", "SMR_VSA7",
    "SMR_VSA8", "SMR_VSA9", "SlogP_VSA1", "SlogP_VSA10", "SlogP_VSA11", "SlogP_VSA12",
    "SlogP_VSA2", "SlogP_VSA3", "SlogP_VSA4", "SlogP_VSA5", "SlogP_VSA6", "SlogP_VSA7",
    "SlogP_VSA8", "SlogP_VSA9", "TPSA", "VSA_EState1", "VSA_EState10", "VSA_EState2",
    "VSA_EState3", "VSA_EState4", "VSA_EState5", "VSA_EState6", "VSA_EState7",
    "VSA_EState8", "VSA_EState9", "fr_Al_COO", "fr_Al_OH", "fr_Al_OH_noTert", "fr_ArN",
    "fr_Ar_COO", "fr_Ar_N", "fr_Ar_NH", "fr_Ar_OH", "fr_COO", "fr_COO2", "fr_C_O",
    "fr_C_O_noCOO", "fr_C_S", "fr_HOCCN", "fr_Imine", "fr_NH0", "fr_NH1", "fr_NH2",
    "fr_N_O", "fr_Ndealkylation1", "fr_Ndealkylation2", "fr_Nhpyrrole", "fr_SH",
    "fr_aldehyde", "fr_alkyl_carbamate", "fr_alkyl_halide", "fr_allylic_oxid",
    "fr_amide", "fr_amidine", "fr_aniline", "fr_aryl_methyl", "fr_azide", "fr_azo",
    "fr_barbitur", "fr_benzene", "fr_benzodiazepine", "fr_bicyclic", "fr_diazo",
    "fr_dihydropyridine", "fr_epoxide", "fr_ester", "fr_ether", "fr_furan", "fr_guanido",
    "fr_halogen", "fr_hdrzine", "fr_hdrzone", "fr_imidazole", "fr_imide", "fr_isocyan",
    "fr_isothiocyan", "fr_ketone", "fr_ketone_Topliss", "fr_lactam", "fr_lactone",
    "fr_methoxy", "fr_morpholine", "fr_nitrile", "fr_nitro", "fr_nitro_arom",
    "fr_nitro_arom_nonortho", "fr_nitroso", "fr_oxazole", "fr_oxime",
    "fr_para_hydroxylation", "fr_phenol", "fr_phenol_noOrthoHbond", "fr_phos_acid",
    "fr_phos_ester", "fr_piperdine", "fr_piperzine", "fr_priamide", "fr_prisulfonamd",
    "fr_pyridine", "fr_quatN", "fr_sulfide", "fr_sulfonamd", "fr_sulfone",
    "fr_term_acetylene", "fr_tetrazole", "fr_thiazole", "fr_thiocyan", "fr_thiophene",
    "fr_unbrch_alkane", "fr_urea", "qed",
]

CYP_MECH_FEATURE_NAMES = [
    "num_bridgehead_atoms",
    "num_spiro_atoms",
    "num_amide_bonds",
    "num_hetero_aromatic_rings",
    "num_ring_hetero_atoms",
    "max_ring_size",
    "fused_ring_atom_fraction",
    "aromatic_atom_fraction",
    "halogen_atom_fraction",
    "benzylic_carbon_count",
    "tertiary_amine_count",
    "aryl_ether_count",
]

BENCHMARK_METRICS = {
    "caco2_wang": "mae",
    "hia_hou": "auroc",
    "pgp_broccatelli": "auroc",
    "bioavailability_ma": "auroc",
    "lipophilicity_astrazeneca": "mae",
    "solubility_aqsoldb": "mae",
    "bbb_martins": "auroc",
    "ppbr_az": "mae",
    "vdss_lombardo": "spearman",
    "cyp2c9_veith": "auprc",
    "cyp2d6_veith": "auprc",
    "cyp3a4_veith": "auprc",
    "cyp2c9_substrate_carbonmangels": "auprc",
    "cyp2d6_substrate_carbonmangels": "auprc",
    "cyp3a4_substrate_carbonmangels": "auroc",
    "half_life_obach": "spearman",
    "clearance_hepatocyte_az": "spearman",
    "clearance_microsome_az": "spearman",
    "ld50_zhu": "mae",
    "herg": "auroc",
    "ames": "auroc",
    "dili": "auroc",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=json_default) + "\n")


@dataclass
class BenchmarkFrameSet:
    benchmark: str
    task_type: str
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


def infer_task_type(benchmark_name: str) -> str:
    key = benchmark_name.lower()
    if key in CLASSIFICATION_BENCHMARKS:
        return "classification"
    if key in REGRESSION_BENCHMARKS:
        return "regression"
    raise ValueError(f"Unknown benchmark type for {benchmark_name}")


def infer_metric_name(benchmark_name: str) -> str:
    key = benchmark_name.lower()
    if key not in BENCHMARK_METRICS:
        raise ValueError(f"Unknown benchmark metric for {benchmark_name}")
    return BENCHMARK_METRICS[key]


def find_smiles_column(frame: pd.DataFrame) -> str:
    for candidate in ("Drug", "SMILES", "smiles", "X", "Mol", "molecule"):
        if candidate in frame.columns:
            return candidate
    raise KeyError(f"Failed to find a SMILES column in columns={list(frame.columns)}")


def find_target_column(frame: pd.DataFrame) -> str:
    for candidate in ("Y", "y", "Label", "Target"):
        if candidate in frame.columns:
            return candidate
    raise KeyError(f"Failed to find a target column in columns={list(frame.columns)}")


def canonicalize_smiles_string(smiles: str) -> str:
    from rdkit import Chem

    if not isinstance(smiles, str) or not smiles.strip():
        return ""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, canonical=True)


def apply_duplicate_policy(
    frame: pd.DataFrame,
    smiles_col: str,
    target_col: str,
    task_type: str,
    duplicate_policy: str,
) -> tuple[pd.DataFrame, dict[str, float] | None]:
    if duplicate_policy == "none":
        return frame, None

    working = frame.copy()
    working["_canonical_smiles"] = working[smiles_col].astype(str).map(canonicalize_smiles_string)
    working["_row_order"] = np.arange(len(working))

    total_rows = float(len(working))
    grouped = working.groupby("_canonical_smiles", dropna=False, sort=False)
    kept_rows: list[pd.Series] = []
    conflict_groups = 0
    duplicate_groups = 0
    dropped_rows = 0

    for canonical, group in grouped:
        if canonical == "":
            kept_rows.extend(group.sort_values("_row_order").to_dict("records"))
            continue
        if len(group) > 1:
            duplicate_groups += 1
        unique_targets = sorted({float(value) for value in group[target_col].tolist()})
        if task_type == "classification":
            if duplicate_policy == "drop_conflicts":
                if len(unique_targets) > 1:
                    conflict_groups += 1
                    dropped_rows += int(len(group))
                    continue
                kept_rows.append(group.sort_values("_row_order").iloc[0].to_dict())
            elif duplicate_policy == "majority_vote":
                positives = int(np.sum(group[target_col].to_numpy() > 0.5))
                negatives = int(len(group) - positives)
                if positives == negatives:
                    conflict_groups += 1
                    dropped_rows += int(len(group))
                    continue
                row = group.sort_values("_row_order").iloc[0].copy()
                row[target_col] = 1 if positives > negatives else 0
                kept_rows.append(row.to_dict())
            else:
                raise ValueError(f"Unsupported duplicate_policy={duplicate_policy}")
        else:
            if duplicate_policy == "drop_conflicts":
                row = group.sort_values("_row_order").iloc[0].copy()
                row[target_col] = float(np.mean(group[target_col].to_numpy(dtype=float)))
                kept_rows.append(row.to_dict())
            elif duplicate_policy == "majority_vote":
                row = group.sort_values("_row_order").iloc[0].copy()
                row[target_col] = float(np.mean(group[target_col].to_numpy(dtype=float)))
                kept_rows.append(row.to_dict())
            else:
                raise ValueError(f"Unsupported duplicate_policy={duplicate_policy}")

    collapsed = pd.DataFrame(kept_rows)
    if "_row_order" in collapsed.columns:
        collapsed = collapsed.sort_values("_row_order").drop(columns=["_row_order"], errors="ignore")
    collapsed = collapsed.drop(columns=["_canonical_smiles"], errors="ignore").reset_index(drop=True)

    stats = {
        "original_train_rows": total_rows,
        "kept_train_rows": float(len(collapsed)),
        "duplicate_groups": float(duplicate_groups),
        "conflict_groups": float(conflict_groups),
        "dropped_rows": float(dropped_rows),
    }
    return collapsed, stats


def load_benchmark_frames(data_dir: Path, benchmark_name: str, seed: int) -> BenchmarkFrameSet:
    from tdc.benchmark_group import admet_group

    group = admet_group(path=str(data_dir))
    benchmark_payload = group.get(benchmark_name)
    normalized_name = benchmark_payload.get("name", benchmark_name)
    train_frame, valid_frame = group.get_train_valid_split(seed=seed, benchmark=benchmark_name)
    test_frame = benchmark_payload["test"]
    task_type = infer_task_type(str(normalized_name))
    return BenchmarkFrameSet(
        benchmark=str(normalized_name),
        task_type=task_type,
        train=train_frame.copy(),
        valid=valid_frame.copy(),
        test=test_frame.copy(),
    )


def load_prepared_split_frames(prepared_dir: Path, benchmark_name: str, seed: int) -> BenchmarkFrameSet:
    benchmark_dir = prepared_dir / benchmark_name.lower()
    if not benchmark_dir.exists():
        raise FileNotFoundError(f"Prepared benchmark directory not found: {benchmark_dir}")
    train_frame = pd.read_csv(benchmark_dir / str(seed) / "train.csv")
    valid_frame = pd.read_csv(benchmark_dir / str(seed) / "val.csv")
    test_frame = pd.read_csv(benchmark_dir / "test.csv")
    return BenchmarkFrameSet(
        benchmark=benchmark_name,
        task_type=infer_task_type(benchmark_name),
        train=train_frame,
        valid=valid_frame,
        test=test_frame,
    )


def canonicalize_mol(smiles: str):
    from rdkit import Chem

    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return mol


def murcko_scaffold(smiles: str) -> str:
    from rdkit.Chem.Scaffolds import MurckoScaffold

    mol = canonicalize_mol(smiles)
    if mol is None:
        return ""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol) or ""
    except Exception:
        return ""


def compute_support_features(
    train_smiles: list[str],
    train_targets: np.ndarray,
    query_smiles: list[str],
    fingerprint_bits: int = 2048,
) -> tuple[np.ndarray, list[str]]:
    from rdkit import DataStructs
    from rdkit.Chem import AllChem

    feature_names = [
        "support_max_tanimoto",
        "support_mean_top5_tanimoto",
        "support_mean_top10_tanimoto",
        "support_max_pos_tanimoto",
        "support_max_neg_tanimoto",
        "support_mean_top5_pos_tanimoto",
        "support_mean_top5_neg_tanimoto",
        "support_top10_pos_fraction",
        "support_scaffold_seen",
        "support_log_train_scaffold_count",
    ]

    train_fps = []
    train_scaffolds = []
    train_labels = []
    scaffold_counts: dict[str, int] = {}

    for smiles, target in zip(train_smiles, train_targets):
        mol = canonicalize_mol(smiles)
        if mol is None:
            continue
        train_fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=fingerprint_bits))
        scaffold = murcko_scaffold(smiles)
        train_scaffolds.append(scaffold)
        train_labels.append(int(float(target) >= 0.5))
        scaffold_counts[scaffold] = scaffold_counts.get(scaffold, 0) + 1

    positive_indices = [idx for idx, label in enumerate(train_labels) if label == 1]
    negative_indices = [idx for idx, label in enumerate(train_labels) if label == 0]

    def top_mean(values: list[float], k: int) -> float:
        if not values:
            return 0.0
        ranked = sorted(values, reverse=True)[:k]
        return float(sum(ranked) / len(ranked))

    rows: list[list[float]] = []
    for smiles in query_smiles:
        mol = canonicalize_mol(smiles)
        if mol is None or not train_fps:
            rows.append([0.0] * len(feature_names))
            continue

        query_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=fingerprint_bits)
        similarities = list(DataStructs.BulkTanimotoSimilarity(query_fp, train_fps))
        pos_scores = [similarities[idx] for idx in positive_indices]
        neg_scores = [similarities[idx] for idx in negative_indices]
        top10_indices = sorted(range(len(similarities)), key=lambda idx: similarities[idx], reverse=True)[:10]
        top10_pos_fraction = (
            sum(train_labels[idx] for idx in top10_indices) / len(top10_indices) if top10_indices else 0.0
        )

        scaffold = murcko_scaffold(smiles)
        scaffold_count = scaffold_counts.get(scaffold, 0)
        rows.append(
            [
                float(max(similarities) if similarities else 0.0),
                top_mean(similarities, 5),
                top_mean(similarities, 10),
                float(max(pos_scores) if pos_scores else 0.0),
                float(max(neg_scores) if neg_scores else 0.0),
                top_mean(pos_scores, 5),
                top_mean(neg_scores, 5),
                float(top10_pos_fraction),
                float(1.0 if scaffold_count > 0 else 0.0),
                float(math.log1p(scaffold_count)),
            ]
        )

    return np.asarray(rows, dtype=np.float32), feature_names


def compute_neighbor_agreement_weights(
    train_smiles: list[str],
    train_targets: np.ndarray,
    fingerprint_bits: int = 2048,
    top_k: int = 10,
) -> tuple[np.ndarray, dict[str, float]]:
    from rdkit import DataStructs
    from rdkit.Chem import AllChem

    train_fps = []
    train_labels: list[int] = []
    valid_indices: list[int] = []

    for idx, (smiles, target) in enumerate(zip(train_smiles, train_targets)):
        mol = canonicalize_mol(smiles)
        if mol is None:
            continue
        train_fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=fingerprint_bits))
        train_labels.append(int(float(target) >= 0.5))
        valid_indices.append(idx)

    weights = np.full(len(train_smiles), 0.5, dtype=np.float32)
    agreement_scores: list[float] = []

    for local_idx, source_idx in enumerate(valid_indices):
        sims = list(DataStructs.BulkTanimotoSimilarity(train_fps[local_idx], train_fps))
        neighbor_pairs = [(idx, sim) for idx, sim in enumerate(sims) if idx != local_idx]
        neighbor_pairs.sort(key=lambda item: item[1], reverse=True)
        top_neighbors = neighbor_pairs[:top_k]
        if not top_neighbors:
            weights[source_idx] = 0.5
            agreement_scores.append(0.5)
            continue

        sim_sum = sum(sim for _, sim in top_neighbors)
        if sim_sum <= 0:
            weighted_agreement = 0.5
        else:
            weighted_agreement = sum(
                sim * float(train_labels[idx] == train_labels[local_idx]) for idx, sim in top_neighbors
            ) / sim_sum
        mean_similarity = sum(sim for _, sim in top_neighbors) / len(top_neighbors)
        score = 0.7 * weighted_agreement + 0.3 * mean_similarity
        agreement_scores.append(score)
        weights[source_idx] = float(0.2 + 0.8 * score)

    stats = {
        "mean_weight": float(np.mean(weights)),
        "std_weight": float(np.std(weights)),
        "min_weight": float(np.min(weights)),
        "max_weight": float(np.max(weights)),
        "mean_agreement_score": float(np.mean(agreement_scores)) if agreement_scores else 0.0,
    }
    return weights, stats


def smiles_to_matrix(smiles_list: list[str], feature_set: str, fingerprint_bits: int) -> tuple[np.ndarray, list[str], dict[str, int]]:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, Descriptors
    from rdkit.Chem.rdReducedGraphs import GetErGFingerprint
    from rdkit.Chem.rdMolDescriptors import GetHashedMorganFingerprint
    from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

    try:
        from rdkit.Avalon import pyAvalonTools  # type: ignore
    except Exception:
        pyAvalonTools = None

    is_maplight = feature_set in {"maplight_exact", "maplight_cyp_mech"}
    use_cyp_mech = feature_set == "maplight_cyp_mech"
    use_morgan = feature_set in {"morgan", "all", "classical", "morgan_rdkit2d"}
    use_avalon = feature_set in {"avalon", "all", "classical"} and pyAvalonTools is not None
    use_erg = feature_set in {"erg", "all", "classical"}
    use_rdkit2d = feature_set in {"rdkit2d", "all", "classical", "morgan_rdkit2d"}
    use_maplight_morgan = is_maplight
    use_maplight_avalon = is_maplight and pyAvalonTools is not None
    use_maplight_erg = is_maplight
    use_maplight_rdkit = is_maplight

    rdkit_descriptor_names = [name for name, _ in Descriptors._descList]
    maplight_calculator = MolecularDescriptorCalculator(MAPLIGHT_DESCRIPTOR_NAMES) if use_maplight_rdkit else None
    rows: list[np.ndarray] = []
    invalid_idx: list[int] = []
    feature_names: list[str] = []
    block_sizes: dict[str, int] = {}

    if use_morgan:
        block_sizes["morgan"] = fingerprint_bits
        feature_names.extend([f"morgan_{idx}" for idx in range(fingerprint_bits)])
    if use_avalon:
        block_sizes["avalon"] = fingerprint_bits
        feature_names.extend([f"avalon_{idx}" for idx in range(fingerprint_bits)])
    if use_erg:
        probe = GetErGFingerprint(Chem.MolFromSmiles("CC"))
        erg_size = int(np.asarray(probe, dtype=np.float32).shape[0])
        block_sizes["erg"] = erg_size
        feature_names.extend([f"erg_{idx}" for idx in range(erg_size)])
    if use_rdkit2d:
        block_sizes["rdkit2d"] = len(rdkit_descriptor_names)
        feature_names.extend([f"rdkit2d_{name}" for name in rdkit_descriptor_names])
    if use_maplight_morgan:
        block_sizes["maplight_morgan_count"] = fingerprint_bits
        feature_names.extend([f"maplight_morgan_count_{idx}" for idx in range(fingerprint_bits)])
    if use_maplight_avalon:
        block_sizes["maplight_avalon_count"] = fingerprint_bits
        feature_names.extend([f"maplight_avalon_count_{idx}" for idx in range(fingerprint_bits)])
    if use_maplight_erg:
        probe = GetErGFingerprint(Chem.MolFromSmiles("CC"))
        erg_size = int(np.asarray(probe, dtype=np.float32).shape[0])
        block_sizes["maplight_erg"] = erg_size
        feature_names.extend([f"maplight_erg_{idx}" for idx in range(erg_size)])
    if use_maplight_rdkit:
        block_sizes["maplight_rdkit"] = len(MAPLIGHT_DESCRIPTOR_NAMES)
        feature_names.extend([f"maplight_rdkit_{name}" for name in MAPLIGHT_DESCRIPTOR_NAMES])
    if use_cyp_mech:
        block_sizes["cyp_mech"] = len(CYP_MECH_FEATURE_NAMES)
        feature_names.extend([f"cyp_mech_{name}" for name in CYP_MECH_FEATURE_NAMES])

    benzylic_pattern = Chem.MolFromSmarts("[c][CH2,CH1,CH0;!$(C=[O,N,S])]")
    tertiary_amine_pattern = Chem.MolFromSmarts("[NX3;H0]([#6])([#6])[#6]")
    aryl_ether_pattern = Chem.MolFromSmarts("[a]-O-[#6]")

    def cyp_mech_features(mol: Chem.Mol) -> np.ndarray:
        from rdkit.Chem import rdMolDescriptors

        ring_info = mol.GetRingInfo()
        atom_rings = ring_info.AtomRings()
        heavy_atoms = max(1, mol.GetNumHeavyAtoms())
        aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
        halogens = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() in {9, 17, 35, 53})
        ring_hetero_atoms = {
            atom_idx
            for ring in atom_rings
            for atom_idx in ring
            if mol.GetAtomWithIdx(atom_idx).GetAtomicNum() not in {1, 6}
        }
        hetero_aromatic_rings = 0
        fused_ring_atoms = 0
        ring_membership = [ring_info.NumAtomRings(idx) for idx in range(mol.GetNumAtoms())]
        for ring in atom_rings:
            ring_atoms = [mol.GetAtomWithIdx(idx) for idx in ring]
            if any(atom.GetAtomicNum() not in {1, 6} for atom in ring_atoms) and any(atom.GetIsAromatic() for atom in ring_atoms):
                hetero_aromatic_rings += 1
        fused_ring_atoms = sum(1 for count in ring_membership if count > 1)
        max_ring_size = max((len(ring) for ring in atom_rings), default=0)
        benzylic_count = len(mol.GetSubstructMatches(benzylic_pattern)) if benzylic_pattern is not None else 0
        tertiary_amine_count = len(mol.GetSubstructMatches(tertiary_amine_pattern)) if tertiary_amine_pattern is not None else 0
        aryl_ether_count = len(mol.GetSubstructMatches(aryl_ether_pattern)) if aryl_ether_pattern is not None else 0

        values = [
            float(rdMolDescriptors.CalcNumBridgeheadAtoms(mol)),
            float(rdMolDescriptors.CalcNumSpiroAtoms(mol)),
            float(rdMolDescriptors.CalcNumAmideBonds(mol)),
            float(hetero_aromatic_rings),
            float(len(ring_hetero_atoms)),
            float(max_ring_size),
            float(fused_ring_atoms / heavy_atoms),
            float(aromatic_atoms / heavy_atoms),
            float(halogens / heavy_atoms),
            float(benzylic_count),
            float(tertiary_amine_count),
            float(aryl_ether_count),
        ]
        return np.asarray(values, dtype=np.float32)

    for idx, smiles in enumerate(smiles_list):
        mol = canonicalize_mol(smiles)
        if mol is None:
            invalid_idx.append(idx)
            vector_parts: list[np.ndarray] = []
            for size in block_sizes.values():
                vector_parts.append(np.zeros(size, dtype=np.float32))
            rows.append(np.concatenate(vector_parts, dtype=np.float32))
            continue

        vector_parts = []
        if use_morgan:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=fingerprint_bits)
            arr = np.zeros((fingerprint_bits,), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)
            vector_parts.append(arr)
        if use_avalon:
            fp = pyAvalonTools.GetAvalonFP(mol, nBits=fingerprint_bits)  # type: ignore[union-attr]
            arr = np.zeros((fingerprint_bits,), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)
            vector_parts.append(arr)
        if use_erg:
            arr = np.asarray(GetErGFingerprint(mol), dtype=np.float32)
            vector_parts.append(arr)
        if use_rdkit2d:
            desc_values = []
            for _, func in Descriptors._descList:
                try:
                    value = float(func(mol))
                    if math.isfinite(value):
                        desc_values.append(value)
                    else:
                        desc_values.append(0.0)
                except Exception:
                    desc_values.append(0.0)
            vector_parts.append(np.asarray(desc_values, dtype=np.float32))
        if use_maplight_morgan:
            arr = np.zeros((fingerprint_bits,), dtype=np.float32)
            fp = GetHashedMorganFingerprint(mol, radius=2, nBits=fingerprint_bits)
            DataStructs.ConvertToNumpyArray(fp, arr)
            vector_parts.append(arr)
        if use_maplight_avalon:
            arr = np.zeros((fingerprint_bits,), dtype=np.float32)
            fp = pyAvalonTools.GetAvalonCountFP(mol, nBits=fingerprint_bits)  # type: ignore[union-attr]
            DataStructs.ConvertToNumpyArray(fp, arr)
            vector_parts.append(arr)
        if use_maplight_erg:
            arr = np.asarray(GetErGFingerprint(mol), dtype=np.float32)
            vector_parts.append(arr)
        if use_maplight_rdkit:
            assert maplight_calculator is not None
            desc_values = []
            for value in maplight_calculator.CalcDescriptors(mol):
                try:
                    numeric = float(value)
                except Exception:
                    numeric = 0.0
                if not math.isfinite(numeric):
                    numeric = 0.0
                desc_values.append(numeric)
            vector_parts.append(np.asarray(desc_values, dtype=np.float32))
        if use_cyp_mech:
            vector_parts.append(cyp_mech_features(mol))

        rows.append(np.concatenate(vector_parts, dtype=np.float32))

    matrix = np.vstack(rows).astype(np.float32, copy=False)
    return matrix, feature_names, {"invalid_smiles": len(invalid_idx), **block_sizes}


def build_model(model_name: str, task_type: str, seed: int, use_gpu: bool):
    model_key = model_name.lower()
    if task_type == "classification":
        if model_key == "logreg":
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression(
                max_iter=4000,
                solver="saga",
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            )
        if model_key == "rf":
            from sklearn.ensemble import RandomForestClassifier

            return RandomForestClassifier(
                n_estimators=500,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=-1,
            )
        if model_key == "extratrees":
            from sklearn.ensemble import ExtraTreesClassifier

            return ExtraTreesClassifier(
                n_estimators=700,
                max_depth=None,
                min_samples_leaf=1,
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=-1,
            )
        if model_key == "lgbm":
            from lightgbm import LGBMClassifier

            return LGBMClassifier(
                n_estimators=1600,
                learning_rate=0.03,
                num_leaves=255,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight="balanced",
                random_state=seed,
                verbosity=-1,
                device_type="gpu" if use_gpu else "cpu",
            )
        if model_key == "xgboost":
            from xgboost import XGBClassifier

            tree_method = "hist"
            device = "cuda" if use_gpu else "cpu"
            return XGBClassifier(
                n_estimators=900,
                max_depth=8,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                min_child_weight=1.0,
                random_state=seed,
                eval_metric="logloss",
                tree_method=tree_method,
                device=device,
                n_jobs=8,
            )
        if model_key == "catboost":
            from catboost import CatBoostClassifier

            return CatBoostClassifier(
                iterations=1200,
                depth=8,
                learning_rate=0.03,
                loss_function="Logloss",
                eval_metric="AUC",
                random_strength=2.0,
                random_seed=seed,
                verbose=False,
                task_type="GPU" if use_gpu else "CPU",
            )
        if model_key == "mlp":
            return None
    else:
        if model_key == "logreg":
            from sklearn.linear_model import ElasticNet

            return ElasticNet(alpha=0.01, l1_ratio=0.2, random_state=seed, max_iter=5000)
        if model_key == "rf":
            from sklearn.ensemble import RandomForestRegressor

            return RandomForestRegressor(
                n_estimators=500,
                max_depth=None,
                min_samples_leaf=2,
                random_state=seed,
                n_jobs=-1,
            )
        if model_key == "extratrees":
            from sklearn.ensemble import ExtraTreesRegressor

            return ExtraTreesRegressor(
                n_estimators=700,
                max_depth=None,
                min_samples_leaf=1,
                random_state=seed,
                n_jobs=-1,
            )
        if model_key == "lgbm":
            from lightgbm import LGBMRegressor

            return LGBMRegressor(
                n_estimators=1600,
                learning_rate=0.03,
                num_leaves=255,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=seed,
                verbosity=-1,
                device_type="gpu" if use_gpu else "cpu",
            )
        if model_key == "xgboost":
            from xgboost import XGBRegressor

            tree_method = "hist"
            device = "cuda" if use_gpu else "cpu"
            return XGBRegressor(
                n_estimators=900,
                max_depth=8,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=seed,
                eval_metric="mae",
                tree_method=tree_method,
                device=device,
                n_jobs=8,
            )
        if model_key == "catboost":
            from catboost import CatBoostRegressor

            return CatBoostRegressor(
                iterations=1200,
                depth=8,
                learning_rate=0.03,
                loss_function="MAE",
                eval_metric="MAE",
                random_strength=2.0,
                random_seed=seed,
                verbose=False,
                task_type="GPU" if use_gpu else "CPU",
            )
        if model_key == "mlp":
            return None

    raise ValueError(f"Unsupported model={model_name} for task_type={task_type}")


def predict_scores(model: Any, matrix: np.ndarray, task_type: str) -> np.ndarray:
    if task_type == "classification":
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(matrix)
            if probs.ndim == 2 and probs.shape[1] >= 2:
                return probs[:, 1].astype(np.float32)
            return probs.astype(np.float32).reshape(-1)
        if hasattr(model, "decision_function"):
            decision = model.decision_function(matrix)
            return np.asarray(decision, dtype=np.float32).reshape(-1)
    preds = model.predict(matrix)
    return np.asarray(preds, dtype=np.float32).reshape(-1)


def fit_and_score_seed(
    benchmark_name: str,
    data_dir: Path,
    output_dir: Path,
    seed: int,
    feature_set: str,
    model_name: str,
    fingerprint_bits: int,
    evaluate_split: str,
    use_gpu: bool,
    prepared_dir: Path | None,
    include_support_features: bool,
    sample_weight_mode: str,
    drop_bottom_fraction: float,
    duplicate_policy: str,
    positive_class_weight: float,
    mlp_hidden_scale: float,
    mlp_dropout_1: float,
    mlp_dropout_2: float,
    mlp_epochs: int,
    mlp_label_smoothing: float,
    mlp_focal_gamma: float,
) -> dict[str, Any]:
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if prepared_dir is not None:
        frame_set = load_prepared_split_frames(prepared_dir=prepared_dir, benchmark_name=benchmark_name, seed=seed)
    else:
        frame_set = load_benchmark_frames(data_dir=data_dir, benchmark_name=benchmark_name, seed=seed)
    smiles_col = find_smiles_column(frame_set.train)
    target_col = find_target_column(frame_set.train)
    duplicate_stats: dict[str, float] | None = None
    frame_set.train, duplicate_stats = apply_duplicate_policy(
        frame=frame_set.train,
        smiles_col=smiles_col,
        target_col=target_col,
        task_type=frame_set.task_type,
        duplicate_policy=duplicate_policy,
    )

    train_matrix, feature_names, train_feature_stats = smiles_to_matrix(
        frame_set.train[smiles_col].astype(str).tolist(),
        feature_set=feature_set,
        fingerprint_bits=fingerprint_bits,
    )
    valid_matrix, _, valid_feature_stats = smiles_to_matrix(
        frame_set.valid[smiles_col].astype(str).tolist(),
        feature_set=feature_set,
        fingerprint_bits=fingerprint_bits,
    )
    test_matrix, _, test_feature_stats = smiles_to_matrix(
        frame_set.test[smiles_col].astype(str).tolist(),
        feature_set=feature_set,
        fingerprint_bits=fingerprint_bits,
    )

    y_train = frame_set.train[target_col].to_numpy()
    y_valid = frame_set.valid[target_col].to_numpy()
    y_test = frame_set.test[target_col].to_numpy()

    if include_support_features:
        train_smiles = frame_set.train[smiles_col].astype(str).tolist()
        support_train, support_feature_names = compute_support_features(
            train_smiles=train_smiles,
            train_targets=y_train,
            query_smiles=train_smiles,
        )
        support_valid, _ = compute_support_features(
            train_smiles=train_smiles,
            train_targets=y_train,
            query_smiles=frame_set.valid[smiles_col].astype(str).tolist(),
        )
        support_test, _ = compute_support_features(
            train_smiles=train_smiles,
            train_targets=y_train,
            query_smiles=frame_set.test[smiles_col].astype(str).tolist(),
        )
        train_matrix = np.concatenate([train_matrix, support_train], axis=1)
        valid_matrix = np.concatenate([valid_matrix, support_valid], axis=1)
        test_matrix = np.concatenate([test_matrix, support_test], axis=1)
        feature_names = feature_names + support_feature_names

    sample_weight_stats: dict[str, float] | None = None
    train_sample_weights: np.ndarray | None = None
    if frame_set.task_type == "classification" and sample_weight_mode == "neighbor_agreement":
        train_sample_weights, sample_weight_stats = compute_neighbor_agreement_weights(
            train_smiles=frame_set.train[smiles_col].astype(str).tolist(),
            train_targets=y_train,
            fingerprint_bits=fingerprint_bits,
        )
        if drop_bottom_fraction > 0:
            threshold = float(np.quantile(train_sample_weights, drop_bottom_fraction))
            keep_mask = train_sample_weights >= threshold
            train_matrix = train_matrix[keep_mask]
            y_train = y_train[keep_mask]
            train_sample_weights = train_sample_weights[keep_mask]
            if sample_weight_stats is not None:
                sample_weight_stats["drop_bottom_fraction"] = float(drop_bottom_fraction)
                sample_weight_stats["drop_threshold"] = threshold
                sample_weight_stats["kept_train_count"] = float(np.sum(keep_mask))

    if frame_set.task_type == "classification" and positive_class_weight != 1.0:
        if train_sample_weights is None:
            train_sample_weights = np.ones_like(y_train, dtype=np.float32)
        train_sample_weights = train_sample_weights.astype(np.float32, copy=False)
        train_sample_weights[y_train == 1] *= float(positive_class_weight)
        train_sample_weights /= max(float(np.mean(train_sample_weights)), 1e-8)
        if sample_weight_stats is None:
            sample_weight_stats = {}
        sample_weight_stats["positive_class_weight"] = float(positive_class_weight)
        sample_weight_stats["positive_fraction"] = float(np.mean(y_train == 1))

    model_key = model_name.lower()
    model = build_model(model_name=model_name, task_type=frame_set.task_type, seed=seed, use_gpu=use_gpu)
    if model_key == "mlp":
        prediction_scores = train_torch_mlp(
            train_matrix=train_matrix,
            train_targets=y_train,
            eval_matrix=valid_matrix if evaluate_split == "valid" else test_matrix,
            valid_matrix=valid_matrix,
            valid_targets=y_valid,
            task_type=frame_set.task_type,
            seed=seed,
            use_gpu=use_gpu,
            train_sample_weights=train_sample_weights,
            hidden_scale=mlp_hidden_scale,
            dropout_1=mlp_dropout_1,
            dropout_2=mlp_dropout_2,
            epochs=mlp_epochs,
            label_smoothing=mlp_label_smoothing,
            focal_gamma=mlp_focal_gamma,
        )
        pipeline = None
    elif model_name.lower() in {"logreg", "rf"}:
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                ("scaler", StandardScaler(with_mean=False)),
                ("model", model),
            ]
        )
    else:
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                ("model", model),
            ]
        )

    if pipeline is not None:
        fit_kwargs = {}
        if train_sample_weights is not None:
            fit_kwargs["model__sample_weight"] = train_sample_weights
        pipeline.fit(train_matrix, y_train, **fit_kwargs)

    if pipeline is not None:
        prediction_scores_valid = predict_scores(pipeline, valid_matrix, task_type=frame_set.task_type)
        prediction_scores_test = predict_scores(pipeline, test_matrix, task_type=frame_set.task_type)
    else:
        prediction_scores_valid = prediction_scores if evaluate_split == "valid" else train_torch_mlp(
            train_matrix=train_matrix,
            train_targets=y_train,
            eval_matrix=valid_matrix,
            valid_matrix=valid_matrix,
            valid_targets=y_valid,
            task_type=frame_set.task_type,
            seed=seed,
            use_gpu=use_gpu,
            train_sample_weights=train_sample_weights,
            hidden_scale=mlp_hidden_scale,
            dropout_1=mlp_dropout_1,
            dropout_2=mlp_dropout_2,
            epochs=mlp_epochs,
            label_smoothing=mlp_label_smoothing,
            focal_gamma=mlp_focal_gamma,
        )
        prediction_scores_test = prediction_scores if evaluate_split == "test" else train_torch_mlp(
            train_matrix=train_matrix,
            train_targets=y_train,
            eval_matrix=test_matrix,
            valid_matrix=valid_matrix,
            valid_targets=y_valid,
            task_type=frame_set.task_type,
            seed=seed,
            use_gpu=use_gpu,
            train_sample_weights=train_sample_weights,
            hidden_scale=mlp_hidden_scale,
            dropout_1=mlp_dropout_1,
            dropout_2=mlp_dropout_2,
            epochs=mlp_epochs,
            label_smoothing=mlp_label_smoothing,
            focal_gamma=mlp_focal_gamma,
        )

    prediction_scores = prediction_scores_valid if evaluate_split == "valid" else prediction_scores_test
    predictions_dict = {
        frame_set.benchmark: {
            "valid": prediction_scores_valid,
            "test": prediction_scores_test,
        }
    }
    split_target = y_valid if evaluate_split == "valid" else y_test
    metric_name = infer_metric_name(frame_set.benchmark)
    metric_value = evaluate_predictions(metric_name=metric_name, true=split_target, pred=prediction_scores)
    official_metrics = {frame_set.benchmark: {metric_name: round(float(metric_value), 3)}}

    seed_dir = ensure_dir(output_dir / f"seed_{seed}")
    prediction_frame_valid = pd.DataFrame(
        {
            "smiles": frame_set.valid[smiles_col].astype(str),
            "target": y_valid,
            "prediction": prediction_scores_valid,
        }
    )
    prediction_frame_test = pd.DataFrame(
        {
            "smiles": frame_set.test[smiles_col].astype(str),
            "target": y_test,
            "prediction": prediction_scores_test,
        }
    )
    prediction_frame_valid.to_csv(seed_dir / "predictions_valid.csv", index=False)
    prediction_frame_test.to_csv(seed_dir / "predictions_test.csv", index=False)

    payload = {
        "benchmark": frame_set.benchmark,
        "task_type": frame_set.task_type,
        "model": model_name,
        "feature_set": feature_set,
        "include_support_features": include_support_features,
        "sample_weight_mode": sample_weight_mode,
        "drop_bottom_fraction": drop_bottom_fraction,
        "duplicate_policy": duplicate_policy,
        "positive_class_weight": positive_class_weight,
        "mlp_hidden_scale": mlp_hidden_scale,
        "mlp_dropout_1": mlp_dropout_1,
        "mlp_dropout_2": mlp_dropout_2,
        "mlp_epochs": mlp_epochs,
        "mlp_label_smoothing": mlp_label_smoothing,
        "mlp_focal_gamma": mlp_focal_gamma,
        "seed": seed,
        "evaluate_split": evaluate_split,
        "official_metrics": official_metrics[frame_set.benchmark],
        "feature_names": feature_names,
        "feature_stats": {
            "train": train_feature_stats,
            "valid": valid_feature_stats,
            "test": test_feature_stats,
            "feature_count": len(feature_names),
        },
        "sample_weight_stats": sample_weight_stats,
        "duplicate_stats": duplicate_stats,
        "counts": {
            "train": int(frame_set.train.shape[0]),
            "valid": int(frame_set.valid.shape[0]),
            "test": int(frame_set.test.shape[0]),
        },
        "created_at": utc_now(),
    }
    write_json(seed_dir / "metrics.json", payload)
    append_jsonl(output_dir / "seed_metrics.jsonl", payload)
    return {
        "predictions": predictions_dict,
        "individual_result": official_metrics,
        "payload": payload,
    }


def train_torch_mlp(
    train_matrix: np.ndarray,
    train_targets: np.ndarray,
    eval_matrix: np.ndarray,
    valid_matrix: np.ndarray,
    valid_targets: np.ndarray,
    task_type: str,
    seed: int,
    use_gpu: bool,
    train_sample_weights: np.ndarray | None,
    hidden_scale: float,
    dropout_1: float,
    dropout_2: float,
    epochs: int,
    label_smoothing: float,
    focal_gamma: float,
) -> np.ndarray:
    import torch
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    np.random.seed(seed)

    imputer = SimpleImputer(strategy="constant", fill_value=0.0)
    scaler = StandardScaler()
    train_np = scaler.fit_transform(imputer.fit_transform(train_matrix)).astype(np.float32)
    valid_np = scaler.transform(imputer.transform(valid_matrix)).astype(np.float32)
    eval_np = scaler.transform(imputer.transform(eval_matrix)).astype(np.float32)

    train_y = np.asarray(train_targets, dtype=np.float32).reshape(-1, 1)
    valid_y = np.asarray(valid_targets, dtype=np.float32).reshape(-1, 1)
    if label_smoothing > 0 and task_type == "classification":
        train_y = train_y * (1.0 - label_smoothing) + 0.5 * label_smoothing

    sample_weight_array: np.ndarray | None = None
    if train_sample_weights is not None:
        sample_weight_array = np.asarray(train_sample_weights, dtype=np.float32).reshape(-1, 1)
        sample_weight_array = sample_weight_array / max(float(np.mean(sample_weight_array)), 1e-8)

    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    input_dim = train_np.shape[1]
    hidden_1 = min(4096, max(256, int(input_dim * hidden_scale)))
    hidden_2 = min(1024, max(128, int(hidden_1 * 0.25)))

    model = nn.Sequential(
        nn.Linear(input_dim, hidden_1),
        nn.ReLU(),
        nn.LayerNorm(hidden_1),
        nn.Dropout(dropout_1),
        nn.Linear(hidden_1, hidden_2),
        nn.ReLU(),
        nn.LayerNorm(hidden_2),
        nn.Dropout(dropout_2),
        nn.Linear(hidden_2, 1),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    if sample_weight_array is not None:
        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(train_np), torch.from_numpy(train_y), torch.from_numpy(sample_weight_array)),
            batch_size=min(512, max(64, len(train_np) // 4)),
            shuffle=True,
            drop_last=False,
        )
    else:
        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(train_np), torch.from_numpy(train_y)),
            batch_size=min(512, max(64, len(train_np) // 4)),
            shuffle=True,
            drop_last=False,
        )
    valid_x = torch.from_numpy(valid_np).to(device)
    valid_y_tensor = torch.from_numpy(valid_y).to(device)

    def compute_train_loss(logits: Any, target: Any, batch_weight: Any) -> Any:
        if task_type == "classification":
            base = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
            if focal_gamma > 0:
                probs = torch.sigmoid(logits)
                pt = target * probs + (1.0 - target) * (1.0 - probs)
                base = ((1.0 - pt) ** focal_gamma) * base
        else:
            base = (logits - target) ** 2
        if batch_weight is not None:
            base = base * batch_weight
        return base.mean()

    def compute_valid_loss(logits: Any, target: Any) -> Any:
        if task_type == "classification":
            return nn.functional.binary_cross_entropy_with_logits(logits, target)
        return nn.functional.mse_loss(logits, target)

    best_state = None
    best_loss = float("inf")
    patience = 10
    stale = 0
    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            if len(batch) == 3:
                batch_x, batch_y, batch_weight = batch
                batch_weight = batch_weight.to(device)
            else:
                batch_x, batch_y = batch
                batch_weight = None
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = compute_train_loss(logits, batch_y, batch_weight)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            valid_logits = model(valid_x)
            valid_loss = compute_valid_loss(valid_logits, valid_y_tensor).item()
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        eval_logits = model(torch.from_numpy(eval_np).to(device)).detach().cpu().numpy().reshape(-1)
    if task_type == "classification":
        return (1.0 / (1.0 + np.exp(-eval_logits))).astype(np.float32)
    return eval_logits.astype(np.float32)


def evaluate_predictions(metric_name: str, true: np.ndarray, pred: np.ndarray) -> float:
    from scipy.stats import spearmanr
    from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score

    true_arr = np.asarray(true)
    pred_arr = np.asarray(pred)
    metric = metric_name.lower()
    if metric == "auroc":
        return float(roc_auc_score(true_arr, pred_arr))
    if metric == "auprc":
        return float(average_precision_score(true_arr, pred_arr))
    if metric == "mae":
        return float(mean_absolute_error(true_arr, pred_arr))
    if metric == "spearman":
        score = spearmanr(true_arr, pred_arr).statistic
        if score is None or not math.isfinite(score):
            return 0.0
        return float(score)
    raise ValueError(f"Unsupported metric {metric_name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run reproducible TDC ADMET baseline experiments.")
    parser.add_argument("--benchmark", required=True, help="TDC benchmark name, e.g. DILI or CYP2C9_Veith")
    parser.add_argument("--model", default="mlp", choices=["catboost", "xgboost", "lgbm", "logreg", "rf", "extratrees", "mlp"])
    parser.add_argument(
        "--feature-set",
        default="classical",
        choices=["classical", "all", "morgan", "avalon", "erg", "rdkit2d", "maplight_exact", "maplight_cyp_mech", "morgan_rdkit2d"],
    )
    parser.add_argument("--evaluate-split", default="valid", choices=["valid", "test"])
    parser.add_argument("--fingerprint-bits", default=2048, type=int)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--data-dir", default=".chem/tdc_admet/data")
    parser.add_argument("--output-root", default=".chem/tdc_admet/runs")
    parser.add_argument("--prepared-dir", default=None, help="Optional root of pre-prepared benchmark CSV splits.")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--notes", default="")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument(
        "--include-support-features",
        action="store_true",
        help="Append split-aware nearest-neighbor and scaffold-support features built from the training split only.",
    )
    parser.add_argument(
        "--sample-weight-mode",
        default="none",
        choices=["none", "neighbor_agreement"],
        help="Optional training-time sample weighting strategy for classification tasks.",
    )
    parser.add_argument(
        "--drop-bottom-fraction",
        default=0.0,
        type=float,
        help="Optionally drop the lowest-weight training fraction after computing sample weights.",
    )
    parser.add_argument(
        "--duplicate-policy",
        default="none",
        choices=["none", "drop_conflicts", "majority_vote"],
        help="Optional train-only duplicate handling based on canonical SMILES.",
    )
    parser.add_argument(
        "--positive-class-weight",
        default=1.0,
        type=float,
        help="Optional multiplier applied to positive-class training examples for classification tasks.",
    )
    parser.add_argument("--mlp-hidden-scale", default=0.5, type=float)
    parser.add_argument("--mlp-dropout-1", default=0.2, type=float)
    parser.add_argument("--mlp-dropout-2", default=0.1, type=float)
    parser.add_argument("--mlp-epochs", default=80, type=int)
    parser.add_argument("--mlp-label-smoothing", default=0.0, type=float)
    parser.add_argument("--mlp-focal-gamma", default=0.0, type=float)
    args = parser.parse_args()

    run_name = args.run_name or (
        f"{args.benchmark.lower()}__{args.model}__{args.feature_set}__{args.evaluate_split}__{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    output_dir = ensure_dir(Path(args.output_root) / run_name)
    ensure_dir(Path(args.data_dir))

    config_payload = {
        "benchmark": args.benchmark,
        "model": args.model,
        "feature_set": args.feature_set,
        "include_support_features": args.include_support_features,
        "sample_weight_mode": args.sample_weight_mode,
        "drop_bottom_fraction": args.drop_bottom_fraction,
        "duplicate_policy": args.duplicate_policy,
        "positive_class_weight": args.positive_class_weight,
        "mlp_hidden_scale": args.mlp_hidden_scale,
        "mlp_dropout_1": args.mlp_dropout_1,
        "mlp_dropout_2": args.mlp_dropout_2,
        "mlp_epochs": args.mlp_epochs,
        "mlp_label_smoothing": args.mlp_label_smoothing,
        "mlp_focal_gamma": args.mlp_focal_gamma,
        "evaluate_split": args.evaluate_split,
        "fingerprint_bits": args.fingerprint_bits,
        "seeds": args.seeds,
        "data_dir": str(Path(args.data_dir).resolve()),
        "output_dir": str(output_dir.resolve()),
        "prepared_dir": str(Path(args.prepared_dir).resolve()) if args.prepared_dir else None,
        "notes": args.notes,
        "cpu_only": args.cpu_only,
        "created_at": utc_now(),
    }
    write_json(output_dir / "config.json", config_payload)

    run_results = []
    for seed in args.seeds:
        run_results.append(
            fit_and_score_seed(
                benchmark_name=args.benchmark,
                data_dir=Path(args.data_dir),
                output_dir=output_dir,
                seed=seed,
                feature_set=args.feature_set,
                model_name=args.model,
                fingerprint_bits=args.fingerprint_bits,
                evaluate_split=args.evaluate_split,
                use_gpu=not args.cpu_only,
                prepared_dir=Path(args.prepared_dir) if args.prepared_dir else None,
                include_support_features=args.include_support_features,
                sample_weight_mode=args.sample_weight_mode,
                drop_bottom_fraction=args.drop_bottom_fraction,
                duplicate_policy=args.duplicate_policy,
                positive_class_weight=args.positive_class_weight,
                mlp_hidden_scale=args.mlp_hidden_scale,
                mlp_dropout_1=args.mlp_dropout_1,
                mlp_dropout_2=args.mlp_dropout_2,
                mlp_epochs=args.mlp_epochs,
                mlp_label_smoothing=args.mlp_label_smoothing,
                mlp_focal_gamma=args.mlp_focal_gamma,
            )
        )

    benchmark_key = next(iter(run_results[0]["individual_result"].keys()))
    metric_key = next(iter(run_results[0]["individual_result"][benchmark_key].keys()))
    scores = [float(item["individual_result"][benchmark_key][metric_key]) for item in run_results]
    combined = {
        benchmark_key: {
            metric_key: [round(float(np.mean(scores)), 3), round(float(np.std(scores, ddof=0)), 3)]
        }
    }
    metric_payload = {
        "run_name": run_name,
        "benchmark": benchmark_key,
        "model": args.model,
        "feature_set": args.feature_set,
        "include_support_features": args.include_support_features,
        "sample_weight_mode": args.sample_weight_mode,
        "drop_bottom_fraction": args.drop_bottom_fraction,
        "duplicate_policy": args.duplicate_policy,
        "positive_class_weight": args.positive_class_weight,
        "mlp_hidden_scale": args.mlp_hidden_scale,
        "mlp_dropout_1": args.mlp_dropout_1,
        "mlp_dropout_2": args.mlp_dropout_2,
        "mlp_epochs": args.mlp_epochs,
        "mlp_label_smoothing": args.mlp_label_smoothing,
        "mlp_focal_gamma": args.mlp_focal_gamma,
        "evaluate_split": args.evaluate_split,
        "aggregate_metrics": combined[benchmark_key],
        "seeds": args.seeds,
        "created_at": utc_now(),
    }
    write_json(output_dir / "summary.json", metric_payload)
    append_jsonl(Path(args.output_root) / "summary.jsonl", metric_payload)
    print(json.dumps(metric_payload, indent=2, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
