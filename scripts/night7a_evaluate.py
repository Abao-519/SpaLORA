#!/usr/bin/env python3
"""Single post-lock evaluator for all twelve Night-7A development candidates."""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from SpaLORA.night1_evaluation import _mean_cluster_moran  # noqa: E402
from SpaLORA.night3b_metrics import mean_one_vs_rest_geary, symmetric_knn_adjacency  # noqa: E402
from SpaLORA.night7a_consensus import (  # noqa: E402
    CANDIDATE_ORDER, DATASETS, G00, K_BY_DATASET,
    atomic_json, array_sha, canonical_json_sha, parse_registry,
    sha256_file,
)
from SpaLORA.night7a_firewall import Night7AFirewall  # noqa: E402

OUT = REPO / "outputs/night7a_handoff"
RAW = Path("/root/autodl-fs/night7a_consensus_20260818")
REG = REPO / "protocols/night7a/SpaLORA_Night7A_Consensus_Registry_2026-08-18.json"
HISTORICAL_METRICS = {
    "night6c": {
        "path": REPO / "protocols/night7a/historical/night6c_per_seed_metrics.csv",
        "sha256": "284b015c005b3b877a6b06ca565b0823285b7d57f39941eb225118f45ce8bd15",
        "datasets": {"a1", "tonsil"},
    },
    "night6d": {
        "path": REPO / "protocols/night7a/historical/night6d_per_seed_metrics.csv",
        "sha256": "97beb1adca50f753ccb35aa95a35d3a54a5f6758d3599844eaca2ac6b7cb1a1e",
        "datasets": {"d1", "p22"},
    },
}
LABEL_SOURCES = {
    "a1": {"path": "/root/autodl-fs/Human lymph node/A1/A1_groundtruth.csv",
           "kind": "csv", "id": "Barcode", "label": "manual-anno",
           "id_rule": "strip_s1_prefix"},
    "tonsil": {"path": "/root/autodl-fs/datasets/human_tonsil_official/section1/s1_adata_rna.h5ad",
               "kind": "h5ad_low_level", "label": "final_annot"},
    "d1": {"path": "/root/autodl-fs/Human lymph node/D1/D1_groundtruth.csv",
           "kind": "csv", "id": "Barcode", "label": "manual-anno"},
    "p22": {"path": "/root/autodl-fs/P22 mouse brain coronal section/MouseBrain_groundtruth.csv",
            "kind": "csv", "id": "Barcode", "label": "manual-anno"},
}
COMPLEXITY = {candidate: rank for rank, candidate in enumerate((
    "C01_G00_H05", "C00_G04_H05_CONFIRMED",
    "C02_DUAL_ARITHMETIC_MEAN", "C03_DUAL_ELEMENTWISE_MAX",
    "C04_DUAL_ELEMENTWISE_MIN", "C05_DUAL_HARMONIC_INTERSECTION",
    "C06_DUAL_ROW_STOCHASTIC_MEAN", "C08_SIX_VIEW_SUPPORT_MEDIAN",
    "C07_DUAL_LOCAL_RELIABILITY", "C10_DUAL_MEAN_SPATIAL05",
    "C11_DUAL_MEAN_SPATIAL10", "C09_DUAL_SPARSE_SNF10",
))}


def decode(values) -> np.ndarray:
    result = []
    for value in np.asarray(values).reshape(-1):
        if isinstance(value, bytes):
            result.append(value.decode("utf-8"))
        else:
            result.append(str(value))
    return np.asarray(result, dtype=str)


def read_h5ad_obs_low_level(path: Path, column: str) -> tuple[np.ndarray, np.ndarray]:
    """Read one post-lock obs column without calling anndata.read_h5ad."""
    with h5py.File(path, "r") as handle:
        obs = handle["obs"]
        index_key = obs.attrs.get("_index", "_index")
        if isinstance(index_key, bytes):
            index_key = index_key.decode()
        ids = decode(obs[index_key][()])
        node = obs[column]
        if isinstance(node, h5py.Dataset):
            values = decode(node[()])
        else:
            codes = np.asarray(node["codes"][()], dtype=np.int64)
            categories = decode(node["categories"][()])
            if np.any(codes < 0):
                raise RuntimeError(f"missing categorical labels in {path}/{column}")
            values = categories[codes]
    return ids, values


def load_base(dataset: str, seed: int = 0) -> tuple[list[str], np.ndarray]:
    root = RAW / "base" / dataset / f"seed_{seed}"
    manifest = json.loads((root / "base_manifest.json").read_text())
    ids = list(map(str, manifest["ids"]))
    coordinate_path = Path(manifest["source_pair"][0]["coordinates_path"])
    if sha256_file(coordinate_path) != manifest["source_pair"][0]["coordinates_sha256"]:
        raise RuntimeError(f"coordinate SHA changed before evaluation: {dataset}")
    return ids, np.load(coordinate_path, allow_pickle=False)


def load_labels_once(preflight: dict) -> tuple[dict, dict]:
    labels, audit = {}, {}
    snapshot_root = RAW / "evaluation_label_snapshots"
    snapshot_root.mkdir(parents=True, exist_ok=True)
    firewall = Night7AFirewall()
    firewall.lock_transforms_and_preflight(360, preflight["status"] == "LOCKED_PRE_LABEL")
    for dataset in DATASETS:
        firewall.read_development_label("evaluator")
        ids, _ = load_base(dataset)
        source = LABEL_SOURCES[dataset]
        path = Path(source["path"])
        observed_file_sha = sha256_file(path)
        expected_file_sha = preflight["development_label_byte_hashes"][dataset]
        if observed_file_sha != expected_file_sha:
            raise RuntimeError(f"development label byte hash changed: {dataset}")
        if source["kind"] == "csv":
            table = pd.read_csv(path)
            raw_id_values = table[source["id"]]
            label_values = table[source["label"]]
            if raw_id_values.isna().any() or label_values.isna().any():
                raise RuntimeError(f"missing ID or label in source: {dataset}")
            raw_ids = raw_id_values.astype(str)
            if source.get("id_rule") == "strip_s1_prefix":
                raw_ids = raw_ids.str.replace(r"^s1-", "", regex=True)
                model_ids = pd.Index(ids).str.replace(r"^s1-", "", regex=True)
            else:
                model_ids = pd.Index(ids)
            values = label_values.astype(str)
            if not raw_ids.is_unique:
                raise RuntimeError(f"invalid label source: {dataset}")
            mapping = pd.Series(values.to_numpy(), index=raw_ids.to_numpy())
            valid = model_ids.isin(mapping.index)
            positions = np.flatnonzero(valid)
            aligned = mapping.reindex(model_ids[valid]).to_numpy(dtype=str)
        else:
            raw_ids, values = read_h5ad_obs_low_level(path, source["label"])
            if not pd.Index(raw_ids).is_unique:
                raise RuntimeError(f"duplicate H5AD observation IDs: {dataset}")
            mapping = pd.Series(values, index=raw_ids)
            model_ids = pd.Index(ids)
            valid = model_ids.isin(mapping.index)
            positions = np.flatnonzero(valid)
            aligned = mapping.reindex(model_ids[valid]).to_numpy(dtype=str)
        if len(positions) != len(ids) or len(np.unique(aligned)) != K_BY_DATASET[dataset]:
            raise RuntimeError(
                f"label replay mismatch {dataset}: aligned={len(positions)}/{len(ids)}, "
                f"K={len(np.unique(aligned))}/{K_BY_DATASET[dataset]}"
            )
        snapshot = snapshot_root / f"{dataset}_labels_locked.npz"
        snapshot_tmp = snapshot.with_name(snapshot.name + ".tmp.npz")
        np.savez_compressed(snapshot_tmp, observation_id=np.asarray(ids, dtype=str),
                            label=np.asarray(aligned, dtype=str))
        os.replace(snapshot_tmp, snapshot)
        labels[dataset] = (positions, aligned)
        audit[dataset] = {
            "authorized_role": "single_evaluator_process",
            "source_path": str(path), "source_file_sha256": observed_file_sha,
            "source_kind": source["kind"], "anndata_read_h5ad_calls": 0,
            "rows": len(ids), "aligned_rows": len(positions),
            "known_k": K_BY_DATASET[dataset], "unique_labels": len(np.unique(aligned)),
            "ordered_label_vector_sha256": array_sha(np.asarray(aligned, dtype=str)),
            "snapshot_path": str(snapshot), "snapshot_sha256": sha256_file(snapshot),
            "used_for_training_or_transform": False,
        }
    firewall.close_evaluator()
    if firewall.development_label_reads != 4 or firewall.fresh_label_reads != 0:
        raise RuntimeError("label firewall access count mismatch")
    return labels, audit


def verify_all_prelabel_evaluator_inputs(transform: dict, preflight: dict) -> pd.DataFrame:
    """Byte-only total verification before the first development label is decoded."""
    for dataset, source in LABEL_SOURCES.items():
        path = Path(source["path"])
        if sha256_file(path) != preflight["development_label_byte_hashes"][dataset]:
            raise RuntimeError(f"development label byte hash changed before window: {dataset}")
        # Coordinate and base-manifest checks never deserialize an obs/label field.
        load_base(dataset)
    for authority, spec in HISTORICAL_METRICS.items():
        observed = sha256_file(spec["path"])
        if observed != spec["sha256"]:
            raise RuntimeError(f"historical metric authority SHA mismatch: {authority}: {observed}")
    for item in transform["transforms"]:
        audit_path = Path(item["candidate_audit_path"])
        if sha256_file(audit_path) != item["candidate_audit_sha256"]:
            raise RuntimeError(f"candidate audit changed before label window: {audit_path}")
        for artifact in item.get("artifacts", {}).values():
            path = Path(artifact["path"])
            if (not path.is_file() or path.stat().st_size != artifact["size_bytes"] or
                    sha256_file(path) != artifact["sha256"]):
                raise RuntimeError(f"candidate artifact changed before label window: {path}")
        weights = item.get("local_reliability_weights")
        if weights:
            path = Path(weights["path"])
            if (not path.is_file() or path.stat().st_size != weights["size_bytes"] or
                    sha256_file(path) != weights["sha256"]):
                raise RuntimeError(f"reliability weights changed before label window: {path}")
    source_predictions = pd.read_csv(OUT / "source_prediction_index.csv")
    if len(source_predictions) != 120:
        raise RuntimeError("historical prediction index is not 120/120")
    for row in source_predictions.itertuples(index=False):
        path = Path(row.path)
        if (not path.is_file() or path.stat().st_size != int(row.size_bytes) or
                sha256_file(path) != row.sha256):
            raise RuntimeError(f"historical prediction changed before label window: {path}")
    return source_predictions


def metrics(true: np.ndarray, pred_labeled: np.ndarray,
            pred_all: np.ndarray, coords: np.ndarray) -> dict:
    graph = symmetric_knn_adjacency(coords, 18)
    rows, cols = graph.nonzero()
    neighbor = float(np.mean(pred_all[rows] == pred_all[cols]))
    moran = float(_mean_cluster_moran(pred_all, graph))
    geary, _ = mean_one_vs_rest_geary(pred_all, graph)
    ari = float(adjusted_rand_score(true, pred_labeled))
    nmi = float(normalized_mutual_info_score(true, pred_labeled))
    return {"ari": ari, "nmi": nmi, "q": (ari + nmi) / 2,
            "neighbor_agreement": neighbor, "moran_i": moran,
            "geary_c": float(geary), "boundary_disagreement": 1 - neighbor}


def exact_sign_flip(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    observed = float(values.mean())
    count = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        count += float(np.mean(values * np.asarray(signs))) >= observed - 1e-15
    return count / (2 ** len(values))


def bootstrap(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.RandomState(20260818)
    indices = rng.randint(0, len(values), size=(100000, len(values)))
    means = values[indices].mean(axis=1)
    return {"replicates": 100000, "seed": 20260818,
            "ci_lower": float(np.quantile(means, .025)),
            "ci_upper": float(np.quantile(means, .975))}


def holm(raw: dict[str, float]) -> dict[str, float]:
    ordered = sorted(raw, key=lambda key: (raw[key], key))
    result, running = {}, 0.0
    m = len(ordered)
    for rank, key in enumerate(ordered):
        running = max(running, min(1.0, raw[key] * (m - rank)))
        result[key] = running
    return result


def spatial_fail(row: pd.Series) -> bool:
    neighbor = float(row.get("mean_delta_neighbor_agreement",
                             row.get("mean_delta_neighbor")))
    moran = float(row.get("mean_delta_moran_i", row.get("mean_delta_moran")))
    geary = float(row.get("mean_delta_geary_c", row.get("mean_delta_geary")))
    return bool((neighbor < -.03 and moran < -.03) or
                (geary > .03 and (neighbor < -.03 or moran < -.03)))


def directional_improvement(metric: str, raw_delta: np.ndarray) -> np.ndarray:
    values = np.asarray(raw_delta, dtype=np.float64)
    return -values if metric in {"geary_c", "boundary_disagreement"} else values


def generalization_components(group: pd.DataFrame) -> dict:
    complete = len(group) == 4 and bool(group.complete.all())
    if not complete:
        return {
            "complete": False, "q_all": False, "nmi_all": False,
            "ari_positive": 0, "ari_floor": False, "wins_gate": False,
            "median_gate": False, "spatial_gate": False,
            "generalization": False,
        }
    q_all = bool((group.mean_delta_q >= .005).all())
    nmi_all = bool((group.mean_delta_nmi > 0).all())
    ari_positive = int((group.mean_delta_ari > 0).sum())
    ari_floor = bool((group.mean_delta_ari >= -.005).all())
    wins = {row.dataset: int(row.wins_delta_q) for row in group.itertuples()}
    wins_gate = (wins["a1"] >= 4 and wins["tonsil"] >= 4 and
                 wins["d1"] >= 7 and wins["p22"] >= 7)
    median_gate = bool((group.median_delta_q > 0).all())
    spatial = group.copy()
    spatial["spatial_fail"] = spatial.apply(spatial_fail, axis=1)
    spatial_gate = not bool(spatial.spatial_fail.any())
    return {
        "complete": True, "q_all": q_all, "nmi_all": nmi_all,
        "ari_positive": ari_positive, "ari_floor": ari_floor,
        "wins_gate": wins_gate, "median_gate": median_gate,
        "spatial_gate": spatial_gate,
        "generalization": bool(
            q_all and nmi_all and ari_positive >= 3 and ari_floor and
            wins_gate and median_gate and spatial_gate
        ),
    }


def dual_complexity_options(macro: float, worst: float, c00_macro: float,
                            c00_worst: float, per_dataset_vs_c00: list[float]) -> tuple[bool, bool]:
    option_a = macro >= c00_macro + .0075 and min(per_dataset_vs_c00) >= -.002
    option_b = (worst >= c00_worst + .005 and macro >= c00_macro and
                min(per_dataset_vs_c00) >= -.002)
    return bool(option_a), bool(option_b)


def historical_metric_replay(frame: pd.DataFrame) -> dict:
    """Post-lock replay of the three authoritative H00/H05 metric rows per unit."""
    mapping = {
        "REFERENCE_G00_H00": (G00, "H00_FUSED_PCA20_MCLUST_EEE"),
        "C00_G04_H05_CONFIRMED": ("G04_SP10_F10_EUC_UNION", "H05_EQUAL3_AFFINITY_SPECTRAL"),
        "C01_G00_H05": (G00, "H05_EQUAL3_AFFINITY_SPECTRAL"),
    }
    metric_names = ("ari", "nmi", "q", "neighbor_agreement", "moran_i",
                    "geary_c", "boundary_disagreement")
    rows, maximum = [], 0.0
    expected_total = 0
    for authority, spec in HISTORICAL_METRICS.items():
        path = spec["path"]
        observed_sha = sha256_file(path)
        if observed_sha != spec["sha256"]:
            raise RuntimeError(
                f"historical metric authority SHA mismatch: {authority}: {observed_sha}"
            )
        history = pd.read_csv(path)
        for dataset in DATASETS:
            if dataset not in spec["datasets"]:
                continue
            seeds = range(5) if dataset in {"a1", "tonsil"} else range(10)
            for seed in seeds:
                for candidate_id, (graph_id, head_id) in mapping.items():
                    expected_total += 1
                    expected = history[
                        (history.dataset == dataset) &
                        (history.seed.astype(int) == seed) &
                        (history.graph_id == graph_id) &
                        (history.head_id == head_id)
                    ]
                    observed = frame[
                        (frame.dataset == dataset) &
                        (frame.seed.astype(int) == seed) &
                        (frame.candidate_id == candidate_id)
                    ]
                    if len(expected) != 1 or len(observed) != 1:
                        raise RuntimeError(
                            f"historical replay cardinality mismatch: "
                            f"{dataset}/{seed}/{candidate_id}: {len(expected)}/{len(observed)}"
                        )
                    errors = {
                        metric: abs(float(observed.iloc[0][metric]) -
                                    float(expected.iloc[0][metric]))
                        for metric in metric_names
                    }
                    maximum = max(maximum, *errors.values())
                    rows.append({
                        "authority": authority, "dataset": dataset, "seed": seed,
                        "candidate_id": candidate_id, "graph_id": graph_id,
                        "head_id": head_id, "maximum_absolute_error": max(errors.values()),
                        "metric_absolute_errors": errors,
                    })
    if expected_total != 90 or len(rows) != 90 or maximum > 1e-12:
        raise RuntimeError(
            f"historical metric replay failed: rows={len(rows)}/90; max_error={maximum}"
        )
    return {
        "status": "PASS", "rows": "90/90", "tolerance": 1e-12,
        "maximum_absolute_error": maximum,
        "historical_final_weights_claimed_loadable": False,
        "label_snapshot_reuse": "not available in Night-6C/Night-6D authoritative raw delivery; byte-locked sources and 90-row numerical replay used",
        "authorities": {
            key: {"path": str(spec["path"]), "sha256": spec["sha256"]}
            for key, spec in HISTORICAL_METRICS.items()
        },
        "cells": rows,
    }


def main() -> None:
    transform_path = OUT / "locked_consensus_transform_manifest.json"
    transform = json.loads(transform_path.read_text())
    if (transform["status"] != "LOCKED" or not transform["locked_before_label_access"] or
            transform["formal_transform_attempts"] != 360 or len(transform["transforms"]) != 360):
        raise RuntimeError("360-cell transform manifest is not total locked")
    expected_keys = [(dataset, seed, candidate)
                     for dataset in DATASETS
                     for seed in (range(5) if dataset in {"a1", "tonsil"} else range(10))
                     for candidate in CANDIDATE_ORDER]
    observed_keys = [(row["dataset"], int(row["seed"]), row["candidate_id"])
                     for row in transform["transforms"]]
    if observed_keys != expected_keys or len(set(observed_keys)) != 360:
        raise RuntimeError("formal transform key order changed")
    preflight_path = OUT / "benchmark_and_data_preflight_lock.json"
    preflight = json.loads(preflight_path.read_text())
    if preflight["status"] != "LOCKED_PRE_LABEL" or preflight["fresh_external_label_reads"] != 0:
        raise RuntimeError("benchmark/data preflight is not pre-label locked")
    for path, expected in preflight["locked_outputs"].items():
        if sha256_file(REPO / path) != expected:
            raise RuntimeError(f"preflight output changed before evaluator: {path}")

    source_predictions = verify_all_prelabel_evaluator_inputs(transform, preflight)

    # This is the single authorized opening of all four already-used development labels.
    labels, label_audit = load_labels_once(preflight)
    base = {dataset: load_base(dataset) for dataset in DATASETS}
    reference_paths = {
        (row.dataset, int(row.seed)): Path(row.path)
        for row in source_predictions.itertuples(index=False)
        if row.graph_id == G00 and row.head_id == "H00_FUSED_PCA20_MCLUST_EEE"
    }
    rows = []
    # Fixed fresh reference metrics.
    for dataset in DATASETS:
        ids, coords = base[dataset]
        pos, true = labels[dataset]
        seeds = range(5) if dataset in {"a1", "tonsil"} else range(10)
        for seed in seeds:
            table = pd.read_csv(reference_paths[(dataset, seed)])
            if table.observation_id.astype(str).tolist() != ids:
                raise RuntimeError(f"reference observation order changed {dataset}/{seed}")
            pred = table.cluster.to_numpy(dtype=np.int64)
            rows.append({"dataset": dataset, "seed": seed,
                         "candidate_id": "REFERENCE_G00_H00", "role": "reference",
                         **metrics(true, pred[pos], pred, coords),
                         "cluster_file_sha256": sha256_file(reference_paths[(dataset, seed)])})
    # All successful formal candidates. Numerical failures remain missing and fail gates.
    for item in transform["transforms"]:
        if item["status"] != "success":
            continue
        dataset, seed = item["dataset"], int(item["seed"])
        ids, coords = base[dataset]
        pos, true = labels[dataset]
        path = Path(item["artifacts"]["clusters.csv"]["path"])
        if sha256_file(path) != item["artifacts"]["clusters.csv"]["sha256"]:
            raise RuntimeError(f"candidate cluster changed before evaluation: {path}")
        table = pd.read_csv(path)
        if table.observation_id.astype(str).tolist() != ids:
            raise RuntimeError(f"candidate observation order changed: {path}")
        pred = table.cluster.to_numpy(dtype=np.int64)
        rows.append({"dataset": dataset, "seed": seed,
                     "candidate_id": item["candidate_id"], "role": "candidate",
                     **metrics(true, pred[pos], pred, coords),
                     "cluster_file_sha256": sha256_file(path)})
    frame = pd.DataFrame(rows).sort_values(["dataset", "seed", "role", "candidate_id"])
    frame.to_csv(OUT / "per_seed_metrics.csv", index=False)
    replay = historical_metric_replay(frame)
    atomic_json(OUT / "historical_metric_replay.json", replay)
    reference = frame[frame.role == "reference"].set_index(["dataset", "seed"])
    deltas = []
    for row in frame[frame.role == "candidate"].itertuples(index=False):
        ref = reference.loc[(row.dataset, int(row.seed))]
        deltas.append({
            "dataset": row.dataset, "seed": int(row.seed),
            "candidate_id": row.candidate_id,
            **{f"delta_{key}": float(getattr(row, key) - ref[key]) for key in
               ("ari", "nmi", "q", "neighbor_agreement", "moran_i", "geary_c",
                "boundary_disagreement")},
            **{key: float(getattr(row, key)) for key in ("ari", "nmi", "q",
                "neighbor_agreement", "moran_i", "geary_c", "boundary_disagreement")},
        })
    delta = pd.DataFrame(deltas)
    delta.to_csv(OUT / "paired_delta_vs_g00h00.csv", index=False)
    c00 = delta[delta.candidate_id == "C00_G04_H05_CONFIRMED"].set_index(["dataset", "seed"])
    versus_c00 = []
    for row in delta.itertuples(index=False):
        comp = c00.loc[(row.dataset, int(row.seed))]
        versus_c00.append({"dataset": row.dataset, "seed": int(row.seed),
                           "candidate_id": row.candidate_id,
                           "delta_ari_vs_c00": row.ari - comp.ari,
                           "delta_nmi_vs_c00": row.nmi - comp.nmi,
                           "delta_q_vs_c00": row.q - comp.q})
    pd.DataFrame(versus_c00).to_csv(OUT / "paired_delta_vs_c00.csv", index=False)

    metric_names = ("ari", "nmi", "q", "neighbor_agreement", "moran_i",
                    "geary_c", "boundary_disagreement")
    lower_is_better = {"geary_c", "boundary_disagreement"}
    summaries, statistics, raw_p = [], {}, {}
    for candidate in CANDIDATE_ORDER:
        statistics[candidate] = {}
        for dataset in DATASETS:
            group = delta[(delta.candidate_id == candidate) & (delta.dataset == dataset)]
            expected_n = 5 if dataset in {"a1", "tonsil"} else 10
            if len(group) != expected_n:
                summaries.append({"candidate_id": candidate, "dataset": dataset,
                                  "successful_cells": len(group), "expected_cells": expected_n,
                                  "complete": False})
                raw_p[f"{candidate}:{dataset}"] = 1.0
                statistics[candidate][dataset] = {
                    "status": "NOT_TESTED_INCOMPLETE_CELLS",
                    "successful_cells": len(group), "expected_cells": expected_n,
                    "q_exact_sign_flip_raw_p": 1.0,
                }
                continue
            row = {"candidate_id": candidate, "dataset": dataset,
                   "successful_cells": len(group), "expected_cells": expected_n,
                   "complete": True}
            statistics[candidate][dataset] = {"metrics": {}}
            for metric in metric_names:
                row[f"mean_{metric}"] = float(group[metric].mean())
                values = group[f"delta_{metric}"].to_numpy(dtype=float)
                row[f"mean_delta_{metric}"] = float(values.mean())
                row[f"median_delta_{metric}"] = float(np.median(values))
                row[f"sd_delta_{metric}"] = float(np.std(values, ddof=1))
                directional = directional_improvement(metric, values)
                row[f"wins_delta_{metric}"] = int(np.sum(directional > 0))
                statistics[candidate][dataset]["metrics"][metric] = {
                    "improvement_direction": "negative_delta" if metric in lower_is_better else "positive_delta",
                    "exact_sign_flip_raw_p": exact_sign_flip(directional),
                    "paired_bootstrap_raw_delta": bootstrap(values),
                }
            p = exact_sign_flip(group.delta_q.to_numpy())
            raw_p[f"{candidate}:{dataset}"] = p
            statistics[candidate][dataset]["q_exact_sign_flip_raw_p"] = p
            statistics[candidate][dataset]["delta_q_bootstrap"] = bootstrap(
                group.delta_q.to_numpy()
            )
            summaries.append(row)
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "four_dataset_summary.csv", index=False)
    macro_rows = []
    for candidate in CANDIDATE_ORDER:
        group = summary[(summary.candidate_id == candidate) & (summary.complete)]
        row = {"candidate_id": candidate, "complete_datasets": len(group),
               "dataset_balanced": len(group) == 4}
        if len(group) == 4:
            for metric in metric_names:
                row[f"macro_mean_{metric}"] = float(group[f"mean_{metric}"].mean())
                row[f"macro_mean_delta_{metric}"] = float(group[f"mean_delta_{metric}"].mean())
            row["worst_dataset_mean_delta_q"] = float(group.mean_delta_q.min())
            row["total_directional_q_wins"] = int(group.wins_delta_q.sum())
        macro_rows.append(row)
    pd.DataFrame(macro_rows).to_csv(
        OUT / "dataset_balanced_macro_summary.csv", index=False
    )
    adjusted = holm(raw_p)
    for key, value in adjusted.items():
        candidate, dataset = key.split(":", 1)
        statistics[candidate][dataset]["q_exploratory_holm_p_48"] = value
    atomic_json(OUT / "exploratory_statistics.json", {
        "family": "12 candidates x 4 development datasets; exploratory",
        "holm_family_size": 48, "bootstrap_replicates": 100000,
        "tested_or_conservatively_filled_hypotheses": len(raw_p),
        "incomplete_candidate_dataset_p_policy": "raw p=1.0; candidate remains ineligible",
        "bootstrap_seed": 20260818, "results": statistics,
    })

    gates = []
    c00_summary = summary[summary.candidate_id == "C00_G04_H05_CONFIRMED"].set_index("dataset")
    c00_macro = float(c00_summary.mean_delta_q.mean())
    c00_worst = float(c00_summary.mean_delta_q.min())
    dual = set(CANDIDATE_ORDER[2:])
    for candidate in CANDIDATE_ORDER:
        group = summary[summary.candidate_id == candidate]
        components = generalization_components(group)
        complete = components["complete"]
        if complete:
            q_all = components["q_all"]; nmi_all = components["nmi_all"]
            ari_positive = components["ari_positive"]; ari_floor = components["ari_floor"]
            wins_gate = components["wins_gate"]; median_gate = components["median_gate"]
            spatial_gate = components["spatial_gate"]
            generalization = components["generalization"]
            macro = float(group.mean_delta_q.mean())
            worst = float(group.mean_delta_q.min())
            versus = [float(group.set_index("dataset").loc[d, "mean_q"] - c00_summary.loc[d, "mean_q"]) for d in DATASETS]
            option_a, option_b = dual_complexity_options(
                macro, worst, c00_macro, c00_worst, versus
            )
            complexity_gate = (option_a or option_b) if candidate in dual else True
            total_wins = int(group.wins_delta_q.sum())
        else:
            q_all = nmi_all = ari_floor = wins_gate = median_gate = spatial_gate = False
            ari_positive = 0; generalization = False; macro = worst = float("nan")
            option_a = option_b = False; complexity_gate = candidate not in dual; total_wins = 0
        gates.append({
            "candidate_id": candidate, "complete": complete,
            "mean_delta_q_all_datasets_ge_0_005": q_all,
            "mean_delta_nmi_positive_all": nmi_all,
            "positive_mean_delta_ari_dataset_count": ari_positive,
            "mean_delta_ari_floor_pass": ari_floor,
            "q_wins_gate_pass": wins_gate, "median_delta_q_all_positive": median_gate,
            "spatial_protection_all_pass": spatial_gate,
            "generalization_gate_pass": generalization,
            "dual_graph": candidate in dual,
            "complexity_option_a": option_a, "complexity_option_b": option_b,
            "complexity_gate_pass": complexity_gate,
            "eligible": generalization and complexity_gate,
            "worst_dataset_mean_delta_q": worst,
            "dataset_balanced_macro_mean_delta_q": macro,
            "total_paired_q_wins": total_wins,
            "future_complexity_rank": COMPLEXITY[candidate],
        })
    gate = pd.DataFrame(gates)
    gate.to_csv(OUT / "candidate_gate_table.csv", index=False)
    eligible = gate[gate.eligible].copy()
    if not eligible.empty:
        eligible = eligible.sort_values(
            ["worst_dataset_mean_delta_q", "dataset_balanced_macro_mean_delta_q",
             "total_paired_q_wins", "future_complexity_rank", "candidate_id"],
            ascending=[False, False, False, True, True],
        )
    selected = "C00_G04_H05_CONFIRMED"
    if not eligible.empty:
        # The taskbook has priority over the registry where they differ: C01 is
        # a mechanism-only single-graph comparator and cannot replace the
        # Night-6D-confirmed C00.  Only a dual-graph candidate that clears both
        # locked gates and outranks C00 may become the new structure.
        selectable = [
            row for row in eligible.itertuples(index=False)
            if row.candidate_id in dual or row.candidate_id == "C00_G04_H05_CONFIRMED"
        ]
        if selectable:
            selected = str(selectable[0].candidate_id)
    terminal = ("KEEP_CONFIRMED_G04_H05_FOR_EXTERNAL_VALIDATION"
                if selected == "C00_G04_H05_CONFIRMED" else
                "LOCK_NEW_CONSENSUS_FOR_FRESH_EXTERNAL_VALIDATION")
    atomic_json(OUT / "spatial_protection.json", {
        "rule": "(neighbor<-0.03 and Moran<-0.03) or (Geary>0.03 and (neighbor<-0.03 or Moran<-0.03))",
        "lower_is_better": ["geary_c", "boundary_disagreement"],
        "rows": [{"candidate_id": row.candidate_id, "dataset": row.dataset,
                  "mean_delta_neighbor": row.mean_delta_neighbor_agreement,
                  "mean_delta_moran": row.mean_delta_moran_i,
                  "mean_delta_geary": row.mean_delta_geary_c,
                  "mean_delta_boundary": row.mean_delta_boundary_disagreement,
                  "spatial_gate_failed": spatial_fail(pd.Series({
                      "mean_delta_neighbor": row.mean_delta_neighbor_agreement,
                      "mean_delta_moran": row.mean_delta_moran_i,
                      "mean_delta_geary": row.mean_delta_geary_c,
                  }))} for row in summary[summary.complete].itertuples(index=False)]
    })
    atomic_json(OUT / "label_window_audit.json", {
        "status": "PASS", "single_authorized_evaluator_process": True,
        "transform_manifest_sha256": sha256_file(transform_path),
        "preflight_lock_sha256": sha256_file(preflight_path),
        "transforms_terminal_before_access": "360/360",
        "datasets_opened_together": list(DATASETS), "datasets": label_audit,
        "fresh_external_label_reads": 0, "anndata_read_h5ad_calls": 0,
        "labels_used_for_training_or_transform": False,
        "return_to_affinity_or_clustering_after_access": False,
    })
    atomic_json(OUT / "night7a_decision.json", {
        "schema_version": 1, "terminal_status": terminal,
        "selected_structure": selected,
        "ranked_eligible_candidates": eligible.candidate_id.astype(str).tolist(),
        "dual_graph_candidates_passing_both_gates": eligible[
            eligible.candidate_id.isin(dual)
        ].candidate_id.astype(str).tolist(),
        "c01_role": "mechanism-only single-graph comparator; taskbook forbids it from replacing confirmed C00",
        "selected_structure_confirmatory_status": (
            "retains Night-6D confirmation" if selected == "C00_G04_H05_CONFIRMED"
            else "development-selected; requires genuinely fresh external confirmation"
        ),
        "reference": "same-dataset same-seed G00/H00",
        "all_four_datasets_are_development_in_night7a": True,
        "generalization_and_complexity_gates_locked_before_labels": True,
        "parameter_tuning": False, "seed_search": False,
        "scientific_training": 0, "checkpoint_forward": 0, "gpu_use": 0,
    })
    print(json.dumps({"terminal_status": terminal, "selected_structure": selected,
                      "metric_rows": len(frame), "candidate_metric_rows": len(delta)},
                     sort_keys=True))


if __name__ == "__main__":
    main()
