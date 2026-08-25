#!/usr/bin/env python3
"""Independent post-lock evaluator for Night-18D human placenta."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import resource
import time

import anndata as ad
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import (
    adjusted_mutual_info_score, adjusted_rand_score, fowlkes_mallows_score,
    homogeneity_score, normalized_mutual_info_score, v_measure_score,
)


def array_sha(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value); digest = hashlib.sha256(); digest.update(value.dtype.str.encode()); digest.update(np.asarray(value.shape, dtype=np.int64).tobytes()); digest.update(value.tobytes()); return digest.hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def graph_from_npz(carrier, index: int) -> sp.csr_matrix:
    return sp.csr_matrix((carrier[f"graph{index}__data"], carrier[f"graph{index}__indices"], carrier[f"graph{index}__indptr"]), shape=tuple(carrier[f"graph{index}__shape"]))


def categorical_spatial(partition: np.ndarray, graph: sp.csr_matrix) -> tuple[float, float, float]:
    labels = np.asarray(partition, dtype=np.int32); n = len(labels)
    graph = sp.csr_matrix(graph, dtype=np.float64).maximum(sp.csr_matrix(graph, dtype=np.float64).T).tocsr(); graph.setdiag(0); graph.eliminate_zeros()
    rows, cols = graph.nonzero(); weights = graph.data; weight_sum = float(weights.sum())
    moran, geary = [], []
    for group in np.unique(labels):
        x = (labels == group).astype(np.float64); centered = x - x.mean(); denominator = float(centered @ centered)
        if denominator <= 0: continue
        moran.append(float(n / weight_sum * np.sum(weights * centered[rows] * centered[cols]) / denominator))
        geary.append(float((n - 1) / (2 * weight_sum) * np.sum(weights * (x[rows] - x[cols]) ** 2) / denominator))
    upper = sp.triu(graph, k=1, format="coo")
    agreement = float(np.average(labels[upper.row] == labels[upper.col], weights=upper.data))
    return float(np.mean(moran)), float(np.mean(geary)), agreement


def metrics(truth: np.ndarray, partition: np.ndarray, graph: sp.csr_matrix) -> dict[str, object]:
    _, encoded = np.unique(partition, return_inverse=True); encoded = encoded.astype(np.int32)
    sizes = np.bincount(encoded); moran, geary, agreement = categorical_spatial(encoded, graph)
    return {
        "ari": adjusted_rand_score(truth, encoded), "nmi": normalized_mutual_info_score(truth, encoded),
        "ami": adjusted_mutual_info_score(truth, encoded), "fmi": fowlkes_mallows_score(truth, encoded),
        "homogeneity": homogeneity_score(truth, encoded), "v_measure": v_measure_score(truth, encoded),
        "moran_indicator_macro": moran, "geary_indicator_macro": geary, "neighbor_agreement": agreement,
        "observed_k": int(len(sizes)), "min_cluster_size": int(sizes.min()), "cluster_sizes": json.dumps([int(value) for value in sizes]),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    started = time.perf_counter(); artifact_path, manifest_path = Path(args.artifact), Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] != "FORMAL_PRODUCER_LOCKED_BEFORE_LABEL_EVALUATION" or manifest["labels_read"] != 0:
        raise RuntimeError("producer was not formally locked before evaluation")
    if manifest["artifact_sha256"] != file_sha(artifact_path): raise RuntimeError("artifact hash mismatch")
    with np.load(artifact_path, allow_pickle=False) as artifact:
        ids = artifact["ids"].astype("U"); candidate_ids = artifact["candidate_ids"].astype("U"); partitions = artifact["partitions"].astype(np.int32)
        selection_names = artifact["selection_names"].astype("U"); selection_indices = artifact["selection_indices"].astype(np.int32)
    records = manifest["records"]
    if len(records) != len(candidate_ids) or any(records[index]["candidate_id"] != candidate_ids[index] for index in range(len(records))):
        raise RuntimeError("record/candidate order mismatch")
    if any(records[index]["partition_sha256"] != array_sha(partitions[index]) for index in range(len(records))):
        raise RuntimeError("candidate partition authority mismatch")
    with np.load(args.carrier, allow_pickle=False) as carrier:
        if not np.array_equal(ids, carrier["ids"].astype("U")): raise RuntimeError("carrier/artifact ordered IDs mismatch")
        graph = graph_from_npz(carrier, 0)
    reference = ad.read_h5ad(args.rna)
    reference_ids = np.asarray(reference.obs_names.astype(str), dtype="U")
    if not np.array_equal(ids, reference_ids): raise RuntimeError("reference/artifact ordered IDs mismatch")
    labels = np.asarray(reference.obs["cell_type"].astype(str), dtype="U")
    if np.any(labels == "") or len(np.unique(labels)) != args.k: raise RuntimeError("reference labels invalid")
    label_names, truth = np.unique(labels, return_inverse=True); truth = truth.astype(np.int32)
    candidate_rows = []
    for index, partition in enumerate(partitions):
        row = {
            "candidate_index": index, "candidate_id": str(candidate_ids[index]), "profile_id": records[index]["profile_id"],
            "arm": records[index]["arm"], "start_index": records[index]["start_index"], "start_name": records[index]["start_name"],
            "partition_sha256": records[index]["partition_sha256"], "changed_observations": records[index]["changed_observations"],
            "structure_feasible": records[index]["feasibility"]["feasible"],
        }
        row.update(metrics(truth, partition, graph)); candidate_rows.append(row)
    selected_rows = []
    for name, index in zip(selection_names, selection_indices):
        row = {"selection_name": str(name), **candidate_rows[int(index)]}; selected_rows.append(row)
    primary = "RNA_CHROMATIN_FAMILY_GEOMETRIC_CENTER"
    selected_by_arm = {row["arm"]: row for row in selected_rows if row["profile_id"] == primary and row["selection_name"].endswith("STRUCTURE_FEASIBLE_MEDOID")}
    if set(selected_by_arm) != {"NO_OP_START", "L2_LOWPASS_MATCHED", "FULL_FROZEN_ENERGY", "REGISTERED_SCALE_ONLY", "NO_SELF_RETURN_STAY", "PAIRWISE_ZERO_KEEP_STAY", "PURE_DYNAMIC_UNARY", "SINGLE_SITE_SAME_ENERGY"}:
        raise RuntimeError("primary selected-arm table incomplete")
    paired_rows = []
    primary_rows = [row for row in candidate_rows if row["profile_id"] == primary]
    for start_index in sorted({int(row["start_index"]) for row in primary_rows}):
        by_arm = {row["arm"]: row for row in primary_rows if int(row["start_index"]) == start_index}
        full, noop = by_arm["FULL_FROZEN_ENERGY"], by_arm["NO_OP_START"]
        paired_rows.append({
            "start_index": start_index, "start_name": full["start_name"], "full_ari": full["ari"], "full_nmi": full["nmi"],
            "noop_ari": noop["ari"], "noop_nmi": noop["nmi"], "delta_ari": full["ari"] - noop["ari"], "delta_nmi": full["nmi"] - noop["nmi"],
            "full_changed_observations": full["changed_observations"], "full_structure_feasible": full["structure_feasible"],
        })
    full = selected_by_arm["FULL_FROZEN_ENERGY"]
    controls = [row for arm, row in selected_by_arm.items() if arm != "FULL_FROZEN_ENERGY"]
    strongest_ari = max(controls, key=lambda row: row["ari"]); strongest_nmi = max(controls, key=lambda row: row["nmi"])
    delta_ari = full["ari"] - strongest_ari["ari"]; delta_nmi = full["nmi"] - strongest_nmi["nmi"]
    paired_dual = sum(row["delta_ari"] > 0 and row["delta_nmi"] > 0 for row in paired_rows)
    full_unique = all(full["partition_sha256"] != row["partition_sha256"] for row in controls)
    external_signal = bool(delta_ari > 0 and delta_nmi > 0 and max(delta_ari, delta_nmi) >= .01 and paired_dual >= 7 and full_unique)
    if external_signal: classification = "EXTERNAL_FROZEN_METHOD_SIGNAL"
    elif (delta_ari > 0 and delta_nmi > 0) or paired_dual >= 7: classification = "EXTERNAL_LOCAL_SIGNAL"
    else: classification = "SCIENTIFIC_NEGATIVE"
    full_candidates = [row for row in primary_rows if row["arm"] == "FULL_FROZEN_ENERGY"]
    oracle = max(full_candidates, key=lambda row: (row["ari"], row["nmi"]))
    per_arm_oracle = {
        arm: max([row for row in primary_rows if row["arm"] == arm], key=lambda row: (row["ari"], row["nmi"]))
        for arm in sorted(selected_by_arm)
    }
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "all_candidate_metrics.csv", candidate_rows); write_csv(output / "selected_main_table.csv", selected_rows); write_csv(output / "paired_start_ablation.csv", paired_rows)
    summary = {
        "status": "EVALUATED_AFTER_FORMAL_PARTITION_LOCK", "classification": classification,
        "headline_selection": next(row for row in selected_rows if row["selection_name"] == "FULL_FROZEN_ENERGY__STRUCTURE_FEASIBLE_MEDOID"),
        "strongest_control_by_ari": strongest_ari, "strongest_control_by_nmi": strongest_nmi,
        "delta_vs_strongest_ari": delta_ari, "delta_vs_strongest_nmi": delta_nmi,
        "paired_dual_gain_starts": paired_dual, "paired_start_count": len(paired_rows), "headline_partition_unique_from_all_controls": full_unique,
        "label_assisted_locked_candidate_oracle": oracle,
        "label_assisted_locked_candidate_oracle_by_arm": per_arm_oracle,
        "reference": {"k": args.k, "n_eval": len(labels), "label_names": label_names.tolist(), "label_counts": {name: int(np.sum(labels == name)) for name in label_names}, "ordered_labels_sha256": array_sha(labels)},
        "label_flow": {"producer_labels_read": 0, "evaluator_label_column": "cell_type", "labels_opened_after_artifact_sha256": file_sha(artifact_path)},
        "wall_seconds": time.perf_counter() - started, "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }
    (output / "evaluation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--artifact", required=True); parser.add_argument("--manifest", required=True); parser.add_argument("--carrier", required=True)
    parser.add_argument("--rna", required=True); parser.add_argument("--k", type=int, default=10); parser.add_argument("--output-dir", required=True); run(parser.parse_args())


if __name__ == "__main__": main()
