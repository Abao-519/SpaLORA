#!/usr/bin/env python3
"""Local-first Night-15E continuous reliability-energy search.

Public labels are visible only to the evaluator and to cross-run development
selection.  The model core receives molecular arrays, a sparse graph, an
initial partition, K, and numeric controls; it never receives a dataset name
or reference labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.stats import qmc
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    fowlkes_mallows_score,
    normalized_mutual_info_score,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from SpaLORA.night15e_continuous_reliability_energy import (
    ContinuousEnergyConfig,
    PreparedContinuousEvidence,
    continuous_reliability_energy,
    csr_from_archive,
    prepare_continuous_evidence,
)


NIGHT15D_AUTHORITY = {
    "A1": (0.2743801557109892, 0.418548976098805),
    "D1": (0.245192670133423, 0.3836654434702409),
    "MISAR_E15_5_S1": (0.5406647315680221, 0.6658997024822209),
    "MISAR_E15_5_S1_K12": (0.4511951009551676, 0.5965119000197391),
    "P22": (0.5871858341020587, 0.7080938095442089),
    "P22_3DOT_K18": (0.7279521853131126, 0.7496741728010312),
    "tonsil_s1": (0.23166767461397172, 0.31216551457035097),
    "tonsil_s2": (0.2514763908366844, 0.30133852810517564),
    "tonsil_s3": (0.32552633510009144, 0.28736250568224364),
}

DATASET_FOR_LANE = {
    "A1": "A1",
    "D1": "D1",
    "MISAR_E15_5_S1": "MISAR_E15_5_S1",
    "MISAR_E15_5_S1_K12": "MISAR_E15_5_S1",
    "P22": "P22",
    "P22_3DOT_K18": "P22",
    "tonsil_s1": "tonsil_s1",
    "tonsil_s2": "tonsil_s2",
    "tonsil_s3": "tonsil_s3",
}

FAMILY_FOR_LANE = {
    "A1": "RNA_PROTEIN",
    "D1": "RNA_PROTEIN",
    "tonsil_s1": "RNA_PROTEIN",
    "tonsil_s2": "RNA_PROTEIN",
    "tonsil_s3": "RNA_PROTEIN",
    "MISAR_E15_5_S1": "RNA_ATAC",
    "MISAR_E15_5_S1_K12": "RNA_ATAC",
    "P22": "RNA_ATAC",
    "P22_3DOT_K18": "RNA_ATAC",
}

STUDY_FOR_LANE = {
    "A1": "LYMPH_NODE",
    "D1": "LYMPH_NODE",
    "tonsil_s1": "CANONICAL_TONSIL",
    "tonsil_s2": "CANONICAL_TONSIL",
    "tonsil_s3": "CANONICAL_TONSIL",
    "MISAR_E15_5_S1": "MISAR_E15_5_S1",
    "MISAR_E15_5_S1_K12": "MISAR_E15_5_S1",
    "P22": "P22",
    "P22_3DOT_K18": "P22",
}


def sha256_array(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encode(value: np.ndarray) -> np.ndarray:
    return np.unique(np.asarray(value, dtype=str), return_inverse=True)[1].astype(np.int32)


def lane_semantics(data: Mapping[str, np.ndarray], lane: str) -> Tuple[int, np.ndarray, np.ndarray]:
    mask = np.asarray(data["label_mask"], dtype=bool)
    if lane == "P22_3DOT_K18":
        return 18, np.asarray(data["labels_k18_author_assignment"], dtype=str), mask
    if lane == "MISAR_E15_5_S1_K12":
        return 12, np.asarray(data["labels_primary"], dtype=str), mask
    return int(data["k_primary"][0]), np.asarray(data["labels_primary"], dtype=str), mask


def mean_binary_moran(partition: np.ndarray, graph: sp.spmatrix) -> float:
    labels = np.asarray(partition, dtype=np.int32)
    weight = sp.csr_matrix(graph, dtype=np.float64)
    total = float(weight.sum())
    if weight.shape != (len(labels), len(labels)) or total <= 0:
        raise ValueError("Moran graph contract failed")
    values = []
    for group in np.unique(labels):
        x = (labels == group).astype(np.float64)
        centered = x - x.mean()
        denominator = float(centered @ centered)
        if denominator > 0:
            values.append(len(x) * float(centered @ (weight @ centered)) / (total * denominator))
    return float(np.mean(values)) if values else 0.0


def mean_binary_geary(partition: np.ndarray, graph: sp.spmatrix) -> float:
    labels = np.asarray(partition, dtype=np.int32)
    weight = sp.csr_matrix(graph, dtype=np.float64).tocoo()
    total = float(weight.data.sum())
    if weight.shape != (len(labels), len(labels)) or total <= 0:
        raise ValueError("Geary graph contract failed")
    values = []
    for group in np.unique(labels):
        x = (labels == group).astype(np.float64)
        denominator = float(np.sum((x - x.mean()) ** 2))
        if denominator > 0:
            numerator = float(np.sum(weight.data * (x[weight.row] - x[weight.col]) ** 2))
            values.append((len(x) - 1.0) * numerator / (2.0 * total * denominator))
    return float(np.mean(values)) if values else 0.0


def evaluate(
    labels: np.ndarray,
    mask: np.ndarray,
    partition: np.ndarray,
    graph: sp.spmatrix,
) -> Dict[str, float]:
    truth = encode(labels[mask])
    observed = np.asarray(partition, dtype=np.int32)[mask]
    return {
        "absolute_ari": float(adjusted_rand_score(truth, observed)),
        "absolute_nmi": float(normalized_mutual_info_score(truth, observed)),
        "ami": float(adjusted_mutual_info_score(truth, observed)),
        "fmi": float(fowlkes_mallows_score(truth, observed)),
        "morans_i": mean_binary_moran(partition, graph),
        "gearys_c": mean_binary_geary(partition, graph),
    }


def config_id(config: ContinuousEnergyConfig, prefix: str = "C") -> str:
    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    return f"{prefix}_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:12]}"


def base_config(**updates: object) -> ContinuousEnergyConfig:
    config = ContinuousEnergyConfig(
        beta=1.0,
        edge_floor=0.08,
        conflict_center=0.22,
        conflict_temperature=0.10,
        conflict_union_weight=0.55,
        conflict_penalty=0.25,
        mass_center=0.40,
        mass_temperature=0.12,
        neighbor_capacity=0.80,
        low_weight=0.65,
        twohop_weight=0.20,
        high_weight=0.50,
        unary_temperature=0.55,
        retained_bias=1.0,
        view_balance=0.0,
        trust_scale=0.75,
        trust_center=0.18,
        trust_temperature=0.12,
        move_threshold=0.015,
        move_fraction=0.10,
        sweeps=2,
    )
    return replace(config, **updates)


def global_config_registry(count: int, seed: int = 20260824) -> Dict[str, ContinuousEnergyConfig]:
    designed = [
        base_config(move_threshold=100.0, move_fraction=0.01, sweeps=1),
        base_config(beta=0.10, neighbor_capacity=0.35, trust_scale=1.8, move_fraction=0.03, sweeps=1),
        base_config(beta=0.35, neighbor_capacity=0.55, trust_scale=1.2, move_fraction=0.06, sweeps=2),
        base_config(beta=0.75, conflict_union_weight=0.25, high_weight=0.9, move_fraction=0.08),
        base_config(beta=1.5, conflict_union_weight=0.75, low_weight=1.2, move_fraction=0.12),
        base_config(beta=3.0, edge_floor=0.03, neighbor_capacity=0.9, trust_scale=0.35, sweeps=3),
        base_config(beta=6.0, edge_floor=0.15, neighbor_capacity=0.95, trust_scale=0.2, sweeps=4),
        base_config(beta=1.0, retained_bias=2.5, unary_temperature=0.35, trust_scale=1.5),
        base_config(beta=1.0, retained_bias=0.0, view_balance=0.5, twohop_weight=0.8, high_weight=0.25),
        base_config(beta=1.0, retained_bias=0.0, view_balance=-0.5, twohop_weight=0.8, high_weight=0.25),
    ]
    remaining = max(0, int(count) - len(designed))
    if remaining:
        sample = qmc.LatinHypercube(d=21, seed=int(seed)).random(remaining)
        for row in sample:
            beta = math.exp(math.log(0.03) + row[0] * (math.log(10.0) - math.log(0.03)))
            sweeps = (1, 2, 3, 5)[min(3, int(row[20] * 4))]
            designed.append(
                base_config(
                    beta=beta,
                    edge_floor=0.01 + 0.34 * row[1],
                    conflict_center=0.03 + 0.52 * row[2],
                    conflict_temperature=0.03 + 0.27 * row[3],
                    conflict_union_weight=0.03 + 0.94 * row[4],
                    conflict_penalty=0.01 + 0.74 * row[5],
                    mass_center=0.08 + 0.72 * row[6],
                    mass_temperature=0.03 + 0.27 * row[7],
                    neighbor_capacity=0.20 + 0.80 * row[8],
                    low_weight=0.05 + 2.20 * row[9],
                    twohop_weight=0.02 + 1.48 * row[10],
                    high_weight=0.05 + 1.95 * row[11],
                    unary_temperature=0.20 + 1.30 * row[12],
                    retained_bias=-0.75 + 3.75 * row[13],
                    view_balance=-0.90 + 1.80 * row[14],
                    trust_scale=0.02 + 2.48 * row[15],
                    trust_center=-0.15 + 1.15 * row[16],
                    trust_temperature=0.04 + 0.41 * row[17],
                    move_threshold=0.001 + 0.149 * row[18],
                    move_fraction=0.015 + 0.285 * row[19],
                    sweeps=sweeps,
                )
            )
    registry: Dict[str, ContinuousEnergyConfig] = {}
    for index, config in enumerate(designed[: int(count)]):
        prefix = f"G{index:03d}"
        registry[config_id(config, prefix)] = config
    return registry


def jitter_registry(
    anchor: ContinuousEnergyConfig,
    count: int,
    seed: int,
) -> Dict[str, ContinuousEnergyConfig]:
    rng = np.random.default_rng(int(seed))
    output: Dict[str, ContinuousEnergyConfig] = {}
    for index in range(int(count)):
        def log_jitter(value: float, sigma: float, low: float, high: float) -> float:
            return float(np.clip(value * math.exp(float(rng.normal(0, sigma))), low, high))

        config = replace(
            anchor,
            beta=log_jitter(anchor.beta, 0.50, 0.015, 15.0),
            edge_floor=float(np.clip(anchor.edge_floor + rng.normal(0, 0.06), 0.005, 0.6)),
            conflict_center=float(np.clip(anchor.conflict_center + rng.normal(0, 0.08), 0.01, 0.8)),
            conflict_temperature=log_jitter(anchor.conflict_temperature, 0.35, 0.015, 0.6),
            conflict_union_weight=float(np.clip(anchor.conflict_union_weight + rng.normal(0, 0.15), 0.01, 0.99)),
            conflict_penalty=float(np.clip(anchor.conflict_penalty + rng.normal(0, 0.15), 0.001, 1.2)),
            mass_center=float(np.clip(anchor.mass_center + rng.normal(0, 0.10), 0.01, 0.95)),
            mass_temperature=log_jitter(anchor.mass_temperature, 0.35, 0.015, 0.6),
            neighbor_capacity=float(np.clip(anchor.neighbor_capacity + rng.normal(0, 0.12), 0.05, 1.0)),
            low_weight=log_jitter(anchor.low_weight, 0.40, 0.01, 4.0),
            twohop_weight=log_jitter(anchor.twohop_weight, 0.50, 0.005, 3.0),
            high_weight=log_jitter(anchor.high_weight, 0.40, 0.01, 4.0),
            unary_temperature=log_jitter(anchor.unary_temperature, 0.35, 0.08, 3.0),
            retained_bias=float(np.clip(anchor.retained_bias + rng.normal(0, 0.5), -2.0, 5.0)),
            view_balance=float(np.clip(anchor.view_balance + rng.normal(0, 0.3), -2.0, 2.0)),
            trust_scale=log_jitter(anchor.trust_scale, 0.45, 0.005, 5.0),
            trust_center=float(np.clip(anchor.trust_center + rng.normal(0, 0.15), -0.5, 2.0)),
            trust_temperature=log_jitter(anchor.trust_temperature, 0.35, 0.015, 1.0),
            move_threshold=float(np.clip(anchor.move_threshold + rng.normal(0, 0.025), 0.0, 0.5)),
            move_fraction=float(np.clip(anchor.move_fraction + rng.normal(0, 0.04), 0.005, 0.5)),
            sweeps=int(np.clip(anchor.sweeps + rng.integers(-1, 2), 1, 8)),
        )
        output[config_id(config, f"R{index:03d}")] = config
    return output


def load_lane(
    kit_root: Path,
    bank_root: Path,
    partition_root: Path,
    lane: str,
) -> Tuple[Mapping[str, np.ndarray], sp.csr_matrix, np.ndarray, np.ndarray, int, np.ndarray, PreparedContinuousEvidence]:
    dataset = DATASET_FOR_LANE[lane]
    data = np.load(kit_root / f"{dataset}.npz", allow_pickle=False, mmap_mode="r")
    bank = np.load(bank_root / f"{dataset}_selected_partition_bank.npz", allow_pickle=False, mmap_mode="r")
    graph = csr_from_archive(data, "graph")
    k, labels, mask = lane_semantics(data, lane)
    initial = np.load(partition_root / f"{lane}.npy", allow_pickle=False).astype(np.int32)
    retained = np.asarray(bank[f"{lane}__retained_embedding"], dtype=np.float32)
    evidence = prepare_continuous_evidence(graph, retained, data["view1"], data["view2"])
    return data, graph, labels, mask, k, initial, evidence


def run_one(
    lane: str,
    graph: sp.csr_matrix,
    labels: np.ndarray,
    mask: np.ndarray,
    k: int,
    initial: np.ndarray,
    evidence: PreparedContinuousEvidence,
    identifier: str,
    config: ContinuousEnergyConfig,
    stage: str,
) -> Tuple[dict, np.ndarray]:
    started = time.perf_counter()
    baseline_ari, baseline_nmi = NIGHT15D_AUTHORITY[lane]
    try:
        partition, diagnostics = continuous_reliability_energy(initial, k, evidence, config)
        metrics = evaluate(labels, mask, partition, graph)
        row = {
            "lane": lane,
            "family": FAMILY_FOR_LANE[lane],
            "study": STUDY_FOR_LANE[lane],
            "stage": stage,
            "algorithm": "CONTINUOUS_RELIABILITY_ENERGY",
            "config_id": identifier,
            "config_json": json.dumps(asdict(config), sort_keys=True, separators=(",", ":")),
            "status": "PASS",
            "failure": "",
            "k": int(k),
            "total_observations": int(len(partition)),
            "evaluated_observations": int(mask.sum()),
            "initial_partition_sha256": sha256_array(initial),
            "partition_sha256": sha256_array(partition),
            "cluster_sizes": json.dumps(np.bincount(partition, minlength=k).astype(int).tolist(), separators=(",", ":")),
            "night15d_ari": baseline_ari,
            "night15d_nmi": baseline_nmi,
            "delta_ari": metrics["absolute_ari"] - baseline_ari,
            "delta_nmi": metrics["absolute_nmi"] - baseline_nmi,
            "objective": metrics["absolute_ari"] + 0.35 * metrics["absolute_nmi"],
            "wall_seconds": time.perf_counter() - started,
            **metrics,
            **diagnostics,
        }
        return row, partition
    except Exception as error:
        return {
            "lane": lane,
            "family": FAMILY_FOR_LANE[lane],
            "study": STUDY_FOR_LANE[lane],
            "stage": stage,
            "algorithm": "CONTINUOUS_RELIABILITY_ENERGY",
            "config_id": identifier,
            "config_json": json.dumps(asdict(config), sort_keys=True, separators=(",", ":")),
            "status": "FAILED",
            "failure": repr(error),
            "wall_seconds": time.perf_counter() - started,
        }, initial.copy()


def baseline_row(
    lane: str,
    graph: sp.csr_matrix,
    labels: np.ndarray,
    mask: np.ndarray,
    k: int,
    initial: np.ndarray,
) -> dict:
    metrics = evaluate(labels, mask, initial, graph)
    expected = NIGHT15D_AUTHORITY[lane]
    if abs(metrics["absolute_ari"] - expected[0]) > 1e-12 or abs(metrics["absolute_nmi"] - expected[1]) > 1e-12:
        raise RuntimeError(f"Night-15D authority mismatch for {lane}: {metrics} != {expected}")
    return {
        "lane": lane,
        "family": FAMILY_FOR_LANE[lane],
        "study": STUDY_FOR_LANE[lane],
        "stage": "AUTHORITY",
        "algorithm": "NIGHT15D_STABLE",
        "config_id": "NIGHT15D_AUTHORITY",
        "config_json": "{}",
        "status": "PASS",
        "failure": "",
        "k": int(k),
        "total_observations": int(len(initial)),
        "evaluated_observations": int(mask.sum()),
        "initial_partition_sha256": sha256_array(initial),
        "partition_sha256": sha256_array(initial),
        "cluster_sizes": json.dumps(np.bincount(initial, minlength=k).astype(int).tolist(), separators=(",", ":")),
        "night15d_ari": expected[0],
        "night15d_nmi": expected[1],
        "delta_ari": 0.0,
        "delta_nmi": 0.0,
        "objective": metrics["absolute_ari"] + 0.35 * metrics["absolute_nmi"],
        "wall_seconds": 0.0,
        "sweeps_completed": 0.0,
        "total_moves": 0.0,
        "changed_observations": 0.0,
        "observed_cardinality": float(k),
        "min_cluster_size": float(np.min(np.bincount(initial, minlength=k))),
        **metrics,
    }


def selection_key(row: Mapping[str, object]) -> Tuple[int, float, float]:
    if row.get("status") != "PASS":
        return (-1, -math.inf, -math.inf)
    da = float(row.get("delta_ari", -math.inf))
    dn = float(row.get("delta_nmi", -math.inf))
    dual = int(da > 1e-12 and dn > 1e-12)
    return (dual, min(da, dn), float(row.get("objective", -math.inf)))


def write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def policy_score(rows: Sequence[Mapping[str, object]]) -> float:
    grouped: Dict[str, List[float]] = {}
    for row in rows:
        value = float(row["delta_ari"]) + 0.35 * float(row["delta_nmi"])
        grouped.setdefault(str(row["study"]), []).append(value)
    effects = [float(np.mean(values)) for values in grouped.values()]
    return float(np.mean(effects) + 0.25 * min(effects))


def build_policy_audit(global_rows: Sequence[Mapping[str, object]]) -> dict:
    passed = [row for row in global_rows if row.get("status") == "PASS"]
    by_config: Dict[str, List[Mapping[str, object]]] = {}
    for row in passed:
        by_config.setdefault(str(row["config_id"]), []).append(row)
    complete = {key: rows for key, rows in by_config.items() if len(rows) == len(NIGHT15D_AUTHORITY)}
    if not complete:
        raise RuntimeError("no complete deployable config matrix")
    global_id, global_rows_selected = max(complete.items(), key=lambda item: policy_score(item[1]))
    studies = sorted(set(STUDY_FOR_LANE.values()))
    folds = []
    for held_out in studies:
        training = {
            key: [row for row in rows if row["study"] != held_out]
            for key, rows in complete.items()
        }
        selected = max(training, key=lambda key: policy_score(training[key]))
        held_rows = [row for row in complete[selected] if row["study"] == held_out]
        folds.append(
            {
                "held_out_study": held_out,
                "selected_config_id": selected,
                "training_policy_score": policy_score(training[selected]),
                "held_out_rows": [
                    {
                        "lane": row["lane"],
                        "absolute_ari": row["absolute_ari"],
                        "absolute_nmi": row["absolute_nmi"],
                        "delta_ari": row["delta_ari"],
                        "delta_nmi": row["delta_nmi"],
                    }
                    for row in held_rows
                ],
            }
        )
    family_policies = []
    for family in sorted(set(FAMILY_FOR_LANE.values())):
        eligible = {
            key: [row for row in rows if row["family"] == family]
            for key, rows in complete.items()
        }
        selected = max(eligible, key=lambda key: policy_score(eligible[key]))
        family_policies.append(
            {
                "family": family,
                "selected_config_id": selected,
                "rows": [
                    {
                        "lane": row["lane"],
                        "delta_ari": row["delta_ari"],
                        "delta_nmi": row["delta_nmi"],
                    }
                    for row in eligible[selected]
                ],
            }
        )
    return {
        "global_selected_config_id": global_id,
        "global_policy_score": policy_score(global_rows_selected),
        "global_rows": [
            {
                "lane": row["lane"],
                "absolute_ari": row["absolute_ari"],
                "absolute_nmi": row["absolute_nmi"],
                "delta_ari": row["delta_ari"],
                "delta_nmi": row["delta_nmi"],
            }
            for row in global_rows_selected
        ],
        "leave_one_study_out": folds,
        "family_numeric_policies": family_policies,
        "selection_rule": "study-balanced mean(delta_ari + 0.35*delta_nmi) plus 0.25*worst-study effect",
        "scope_qualification": "incremental policy transfer on fixed Night-15D authorities",
        "initial_partition_label_history": "each Night-15D authority partition was previously selected by per-lane public-label development HPO",
        "full_end_to_end_zero_label_deployability_claimed": False,
        "held_out_labels_used_for_fold_selection": 0,
        "dataset_name_reads_in_model_core": 0,
    }


def search(args: argparse.Namespace) -> None:
    lanes = [value for value in args.lanes.split(",") if value]
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    registry = global_config_registry(args.global_configs)
    rows: List[dict] = []
    partitions: Dict[str, np.ndarray] = {}
    per_lane_payload = {}
    started_all = time.perf_counter()
    for lane_index, lane in enumerate(lanes):
        started_lane = time.perf_counter()
        data, graph, labels, mask, k, initial, evidence = load_lane(
            args.kit, args.banks, args.partitions, lane
        )
        base = baseline_row(lane, graph, labels, mask, k, initial)
        rows.append(base)
        lane_rows: List[dict] = []
        lane_partitions: Dict[str, np.ndarray] = {}
        for identifier, config in registry.items():
            row, partition = run_one(
                lane, graph, labels, mask, k, initial, evidence, identifier, config, "GLOBAL_GRID"
            )
            rows.append(row)
            lane_rows.append(row)
            if row.get("status") == "PASS":
                lane_partitions[identifier] = partition
        ranked = sorted(lane_rows, key=selection_key, reverse=True)
        anchor_row = next((row for row in ranked if row.get("status") == "PASS" and row["config_id"] != "NIGHT15D_AUTHORITY"), None)
        if anchor_row is None:
            raise RuntimeError(f"no successful global config for {lane}")
        anchor = ContinuousEnergyConfig(**json.loads(str(anchor_row["config_json"])))
        refined = jitter_registry(anchor, args.refine_configs, 20260824 + lane_index * 1009)
        for identifier, config in refined.items():
            identifier = f"{lane}__{identifier}"
            row, partition = run_one(
                lane, graph, labels, mask, k, initial, evidence, identifier, config, "LANE_NUMERIC_REFINEMENT"
            )
            rows.append(row)
            lane_rows.append(row)
            if row.get("status") == "PASS":
                lane_partitions[identifier] = partition
        best = max([base] + lane_rows, key=selection_key)
        best_id = str(best["config_id"])
        best_partition = initial if best_id == "NIGHT15D_AUTHORITY" else lane_partitions[best_id]
        partitions[lane] = best_partition
        (output / "partitions").mkdir(parents=True, exist_ok=True)
        np.save(output / "partitions" / f"{lane}.npy", best_partition, allow_pickle=False)
        per_lane_payload[lane] = {
            "best": best,
            "partition_sha256": sha256_array(best_partition),
            "wall_seconds": time.perf_counter() - started_lane,
            "candidate_rows": len(lane_rows),
            "labels_used_for_cross_run_hpo_and_evaluation": 1,
            "labels_in_model_or_energy": 0,
        }
        write_rows(output / "all_run_ledger.partial.csv", rows)
        print(
            json.dumps(
                {
                    "lane": lane,
                    "best_ari": best["absolute_ari"],
                    "best_nmi": best["absolute_nmi"],
                    "delta_ari": best["delta_ari"],
                    "delta_nmi": best["delta_nmi"],
                    "rows": len(lane_rows),
                }
            ),
            flush=True,
        )
    write_rows(output / "all_run_ledger.csv", rows)
    global_rows = [row for row in rows if row.get("stage") == "GLOBAL_GRID"]
    if set(lanes) == set(NIGHT15D_AUTHORITY):
        policy = build_policy_audit(global_rows)
    else:
        policy = {
            "status": "SKIPPED_PARTIAL_LANE_SMOKE",
            "lanes_present": lanes,
            "required_lanes": list(NIGHT15D_AUTHORITY),
            "scope_qualification": "incremental policy transfer on fixed Night-15D authorities",
            "full_end_to_end_zero_label_deployability_claimed": False,
        }
    (output / "deployable_policy_audit.json").write_text(
        json.dumps(policy, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    summary = {
        "status": "PASS",
        "lanes": per_lane_payload,
        "global_config_registry": {key: asdict(value) for key, value in registry.items()},
        "run_rows": len(rows),
        "wall_seconds": time.perf_counter() - started_all,
        "labels_used_for_cross_run_hpo_and_evaluation": 1,
        "labels_in_model_or_energy": 0,
        "dataset_name_reads_in_model_core": 0,
        "dense_n_by_n_count": 0,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "UNSET"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS", "UNSET"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS", "UNSET"),
    }
    (output / "search_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "rows": len(rows), "wall_seconds": summary["wall_seconds"]}))


def replay(args: argparse.Namespace) -> None:
    frozen = json.loads(args.registry.read_text(encoding="utf-8"))
    rows = []
    output_partitions = args.output / "partitions"
    output_partitions.mkdir(parents=True, exist_ok=True)
    for lane, registered in frozen["lanes"].items():
        data, graph, labels, mask, k, initial, evidence = load_lane(
            args.kit, args.banks, args.partitions, lane
        )
        config = ContinuousEnergyConfig(**registered["config"])
        row, partition = run_one(
            lane, graph, labels, mask, k, initial, evidence, registered["config_id"], config, "FROZEN_REPLAY"
        )
        if row["partition_sha256"] != registered["partition_sha256"]:
            raise RuntimeError(f"partition mismatch for {lane}")
        if abs(row["absolute_ari"] - registered["absolute_ari"]) > 1e-12:
            raise RuntimeError(f"ARI mismatch for {lane}")
        if abs(row["absolute_nmi"] - registered["absolute_nmi"]) > 1e-12:
            raise RuntimeError(f"NMI mismatch for {lane}")
        np.save(output_partitions / f"{lane}.npy", partition, allow_pickle=False)
        rows.append(row)
        print(json.dumps({"lane": lane, "partition_sha256": row["partition_sha256"]}), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "PASS",
        "lane_count": len(rows),
        "rows": rows,
        "labels_in_model_or_energy": 0,
        "dataset_name_reads_in_model_core": 0,
        "dense_n_by_n_count": 0,
    }
    (args.output / "replay.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--kit", type=Path, required=True)
    common.add_argument("--banks", type=Path, required=True)
    common.add_argument("--partitions", type=Path, required=True)
    common.add_argument("--output", type=Path, required=True)
    common.add_argument("--lanes", default=",".join(NIGHT15D_AUTHORITY))

    search_parser = subparsers.add_parser("search", parents=[common])
    search_parser.add_argument("--global-configs", type=int, default=40)
    search_parser.add_argument("--refine-configs", type=int, default=16)

    replay_parser = subparsers.add_parser("replay", parents=[common])
    replay_parser.add_argument("--registry", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "search":
        search(args)
    elif args.command == "replay":
        replay(args)


if __name__ == "__main__":
    main()
