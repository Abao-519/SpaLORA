#!/usr/bin/env python3
"""Independent public-annotation evaluator for locked Night-17A artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import resource
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    average_precision_score,
    fowlkes_mallows_score,
    homogeneity_score,
    normalized_mutual_info_score,
    roc_auc_score,
    v_measure_score,
)

from SpaLORA.night17a_ceup import encode_partition, sha256_array


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_reference(kind: str, path: Path, ids: np.ndarray, label_key: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    if kind == "npz":
        with np.load(path, allow_pickle=False) as ref:
            reference_ids = np.asarray(ref["ids"])
            labels = np.asarray(ref[label_key])
            mask = np.asarray(ref["label_mask"], dtype=bool) if "label_mask" in ref.files else np.ones(labels.size, dtype=bool)
        if not np.array_equal(reference_ids, ids):
            raise ValueError("reference and producer ordered IDs are not byte-exact")
    elif kind == "h5ad":
        import anndata as ad

        ref = ad.read_h5ad(path)
        if label_key not in ref.obs:
            raise ValueError(f"missing reference column {label_key}")
        index = {str(value): i for i, value in enumerate(ref.obs_names.tolist())}
        if any(str(value) not in index for value in ids.tolist()):
            raise ValueError("producer IDs are not a subset of reference IDs")
        order = np.asarray([index[str(value)] for value in ids.tolist()], dtype=np.int64)
        series = ref.obs[label_key].iloc[order]
        mask = np.asarray(series.notna(), dtype=bool)
        labels = np.asarray(series.astype(str))
    else:
        raise ValueError(kind)
    if labels.shape[0] != ids.size or mask.shape[0] != ids.size:
        raise ValueError("reference observation mismatch")
    clean = encode_partition(labels[mask])
    metadata = {
        "reference_path": str(path.resolve()),
        "reference_sha256": file_sha256(path),
        "label_key": label_key,
        "evaluated_observations": int(mask.sum()),
        "label_k": int(np.unique(clean).size),
        "ordered_label_sha256": sha256_array(np.asarray(labels[mask]).astype("U")),
    }
    return labels, mask, metadata


def graph_from_edges(n: int, edge_i: np.ndarray, edge_j: np.ndarray, edge_w: np.ndarray) -> sp.csr_matrix:
    return sp.csr_matrix(
        (np.concatenate([edge_w, edge_w]), (np.concatenate([edge_i, edge_j]), np.concatenate([edge_j, edge_i]))),
        shape=(n, n),
    )


def categorical_spatial_metrics(labels: np.ndarray, graph: sp.csr_matrix) -> Dict[str, float]:
    labels = encode_partition(labels)
    coo = sp.triu(graph, k=1).tocoo()
    total_weight = float(coo.data.sum())
    same = labels[coo.row] == labels[coo.col]
    neighbor_agreement = float(coo.data[same].sum() / total_weight) if total_weight > 0 else float("nan")
    morans: List[float] = []
    gearys: List[float] = []
    sizes: List[int] = []
    n = labels.size
    for cluster in range(int(labels.max()) + 1):
        indicator = (labels == cluster).astype(np.float64)
        sizes.append(int(indicator.sum()))
        centered = indicator - indicator.mean()
        denom = float(np.sum(centered * centered))
        if denom <= 0 or total_weight <= 0:
            continue
        cross = float(np.sum(coo.data * centered[coo.row] * centered[coo.col]))
        sqdiff = float(np.sum(coo.data * (indicator[coo.row] - indicator[coo.col]) ** 2))
        morans.append((n / total_weight) * cross / denom)
        gearys.append(((n - 1) / (2.0 * total_weight)) * sqdiff / denom)
    weights = np.asarray(sizes, dtype=np.float64)
    weights = weights / weights.sum()
    return {
        "neighbor_agreement": neighbor_agreement,
        "moran_macro_ovr": float(np.mean(morans)),
        "geary_macro_ovr": float(np.mean(gearys)),
        "moran_weighted_ovr": float(np.sum(np.asarray(morans) * weights[: len(morans)])),
        "geary_weighted_ovr": float(np.sum(np.asarray(gearys) * weights[: len(gearys)])),
    }


def partition_components(partition: np.ndarray, graph: sp.csr_matrix) -> Tuple[int, List[int]]:
    counts = []
    for cluster in range(int(partition.max()) + 1):
        nodes = np.flatnonzero(partition == cluster)
        count = connected_components(graph[nodes][:, nodes], directed=False, return_labels=False) if nodes.size else 0
        counts.append(int(count))
    return int(sum(counts)), counts


def metric_row(true_labels: np.ndarray, prediction: np.ndarray, mask: np.ndarray, graph: sp.csr_matrix) -> Dict[str, object]:
    truth = encode_partition(true_labels[mask])
    pred_eval = encode_partition(prediction[mask])
    spatial = categorical_spatial_metrics(prediction, graph)
    components_total, components_by_cluster = partition_components(encode_partition(prediction), graph)
    full_sizes = np.bincount(encode_partition(prediction)).astype(int).tolist()
    eval_sizes = np.bincount(pred_eval).astype(int).tolist()
    return {
        "ari": adjusted_rand_score(truth, pred_eval),
        "nmi": normalized_mutual_info_score(truth, pred_eval),
        "ami": adjusted_mutual_info_score(truth, pred_eval),
        "fmi": fowlkes_mallows_score(truth, pred_eval),
        "homogeneity": homogeneity_score(truth, pred_eval),
        "v_measure": v_measure_score(truth, pred_eval),
        **spatial,
        "min_cluster_size_full": min(full_sizes),
        "cluster_sizes_full": json.dumps(full_sizes, separators=(",", ":")),
        "min_cluster_size_eval": min(eval_sizes),
        "cluster_sizes_eval": json.dumps(eval_sizes, separators=(",", ":")),
        "components_total": components_total,
        "components_by_cluster": json.dumps(components_by_cluster, separators=(",", ":")),
    }


def safe_binary_scores(truth: np.ndarray, score: np.ndarray) -> Tuple[float, float]:
    if np.unique(truth).size < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(truth, score)), float(average_precision_score(truth, score))


def graph_local_edge_blocks(
    ids: np.ndarray,
    graph: sp.csr_matrix,
    edge_i: np.ndarray,
    edge_j: np.ndarray,
    n_blocks: int = 64,
) -> np.ndarray:
    """Build deterministic graph-local node blocks, then assign edges to them."""
    graph = sp.csr_matrix(graph)
    n = graph.shape[0]
    target = max(1, int(math.ceil(n / float(n_blocks))))
    hashes = np.asarray(
        [hashlib.sha256(f"night17a-local-block|{value}".encode("utf-8")).digest() for value in ids.tolist()],
        dtype="S32",
    )
    seed_order = np.argsort(hashes, kind="mergesort")
    node_block = np.full(n, -1, dtype=np.int32)
    block = 0
    for seed in seed_order.tolist():
        if node_block[seed] >= 0:
            continue
        queue = [seed]
        node_block[seed] = block
        filled = 1
        cursor = 0
        while cursor < len(queue) and filled < target:
            node = queue[cursor]
            cursor += 1
            neighbors = graph.indices[graph.indptr[node] : graph.indptr[node + 1]]
            neighbors = neighbors[np.argsort(hashes[neighbors], kind="mergesort")]
            for neighbor in neighbors.tolist():
                if node_block[neighbor] < 0:
                    node_block[neighbor] = block
                    queue.append(neighbor)
                    filled += 1
                    if filled >= target:
                        break
        block += 1
    if np.any(node_block < 0):
        raise RuntimeError("graph-local block assignment incomplete")
    return np.minimum(node_block[edge_i], node_block[edge_j]).astype(np.int32)


def bootstrap_ap_delta(truth: np.ndarray, learned: np.ndarray, control: np.ndarray, blocks: np.ndarray, seed: int = 20260825, replicates: int = 199) -> Tuple[float, float, float]:
    rng = np.random.RandomState(seed)
    unique = np.unique(blocks)
    deltas = []
    for _ in range(replicates):
        sampled = rng.choice(unique, size=unique.size, replace=True)
        indices = np.concatenate([np.flatnonzero(blocks == block) for block in sampled])
        if np.unique(truth[indices]).size < 2:
            continue
        deltas.append(average_precision_score(truth[indices], learned[indices]) - average_precision_score(truth[indices], control[indices]))
    if not deltas:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(deltas)), float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--producer-json", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--reference-kind", choices=("npz", "h5ad"), required=True)
    parser.add_argument("--label-key", required=True)
    parser.add_argument("--lane", required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--metrics-output", required=True)
    parser.add_argument("--edge-output", required=True)
    args = parser.parse_args()
    started = time.time()
    artifact_path = Path(args.artifact)
    producer = json.loads(Path(args.producer_json).read_text(encoding="utf-8"))
    if producer["artifact_sha256"] != file_sha256(artifact_path):
        raise ValueError("producer artifact hash mismatch")
    with np.load(artifact_path, allow_pickle=False) as artifact:
        data = {key: np.asarray(artifact[key]) for key in artifact.files}
    labels, mask, reference_meta = load_reference(args.reference_kind, Path(args.reference), data["ids"], args.label_key)
    if reference_meta["label_k"] != args.k:
        raise ValueError(f"reference K {reference_meta['label_k']} != registered K {args.k}")
    graph = graph_from_edges(data["ids"].size, data["edge_i"], data["edge_j"], data["edge_w"])
    diagnostics = {record["arm"]: record for record in producer["arm_diagnostics"]}
    metric_rows = []
    for arm, partition in zip(data["arm_ids"].tolist(), data["arm_partitions"]):
        row = {"lane": args.lane, "arm": str(arm), "n": int(data["ids"].size), "eval_n": int(mask.sum()), "k": args.k}
        row.update(metric_row(labels, partition, mask, graph))
        row.update({
            "changed_spots": diagnostics[str(arm)]["changed_spots"],
            "initial_energy": diagnostics[str(arm)]["initial_energy"],
            "final_energy": diagnostics[str(arm)]["final_energy"],
            "energy_delta": diagnostics[str(arm)]["final_energy"] - diagnostics[str(arm)]["initial_energy"],
            "positive_mass": diagnostics[str(arm)]["positive_mass"],
            "negative_mass": diagnostics[str(arm)]["negative_mass"],
            "positive_coverage": diagnostics[str(arm)]["positive_coverage"],
            "negative_coverage": diagnostics[str(arm)]["negative_coverage"],
            "partition_sha256": producer["partition_sha256"][str(arm)],
            "producer_wall_seconds": producer["wall_seconds"],
            "producer_peak_rss_mb": producer["peak_rss_mb"],
            "producer_peak_gpu_mb": producer["peak_gpu_mb"],
        })
        metric_rows.append(row)

    edge_valid = mask[data["edge_i"]] & mask[data["edge_j"]]
    same = (labels[data["edge_i"]] == labels[data["edge_j"]])[edge_valid].astype(np.int8)
    boundary = 1 - same
    blocks = graph_local_edge_blocks(data["ids"], graph, data["edge_i"], data["edge_j"])[edge_valid]
    sources = {
        "LEARNED_FULL": (data["learned_q_positive"], data["learned_q_negative"]),
        "PERMUTED": (data["permuted_q_positive"], data["permuted_q_negative"]),
        "STATIC": (data["static_q_positive"], data["static_q_negative"]),
        "SINGLE_VIEW1": (data["single1_q_positive"], data["single1_q_negative"]),
        "SINGLE_VIEW2": (data["single2_q_positive"], data["single2_q_negative"]),
        "UNCERTAINTY_OFF": (data["no_uncertainty_q_positive"], data["no_uncertainty_q_negative"]),
    }
    edge_rows = []
    learned_support = data["learned_q_positive"][edge_valid]
    learned_boundary = data["learned_q_negative"][edge_valid]
    for name, (support_score, boundary_score) in sources.items():
        support = support_score[edge_valid]
        bound = boundary_score[edge_valid]
        support_auc, support_ap = safe_binary_scores(same, support)
        boundary_auc, boundary_ap = safe_binary_scores(boundary, bound)
        support_delta = bootstrap_ap_delta(same, learned_support, support, blocks) if name != "LEARNED_FULL" else (0.0, 0.0, 0.0)
        boundary_delta = bootstrap_ap_delta(boundary, learned_boundary, bound, blocks, seed=20260826) if name != "LEARNED_FULL" else (0.0, 0.0, 0.0)
        edge_rows.append({
            "lane": args.lane,
            "source": name,
            "eligible_edges": int(edge_valid.sum()),
            "same_domain_fraction": float(same.mean()),
            "support_auroc": support_auc,
            "support_auprc": support_ap,
            "boundary_auroc": boundary_auc,
            "boundary_auprc": boundary_ap,
            "learned_minus_source_support_auprc_boot_mean": support_delta[0],
            "learned_minus_source_support_auprc_ci_low": support_delta[1],
            "learned_minus_source_support_auprc_ci_high": support_delta[2],
            "learned_minus_source_boundary_auprc_boot_mean": boundary_delta[0],
            "learned_minus_source_boundary_auprc_ci_low": boundary_delta[1],
            "learned_minus_source_boundary_auprc_ci_high": boundary_delta[2],
        })

    metrics_output = Path(args.metrics_output)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    with metrics_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    edge_output = Path(args.edge_output)
    with edge_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(edge_rows[0]))
        writer.writeheader()
        writer.writerows(edge_rows)
    audit = {
        "schema": "night17a-ceup-p0-evaluator-v1",
        "lane": args.lane,
        "producer_artifact_sha256": file_sha256(artifact_path),
        "producer_locked_before_label_load": True,
        "reference": reference_meta,
        "metrics_sha256": file_sha256(metrics_output),
        "edge_identifiability_sha256": file_sha256(edge_output),
        "bootstrap_replicates": 199,
        "bootstrap_semantics": "deterministic graph-local node blocks; cross-block edges assigned to lower block index",
        "wall_seconds": time.time() - started,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }
    metrics_output.with_suffix(".evaluator.json").write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"lane": args.lane, "status": "PASS", "metrics": str(metrics_output)}, sort_keys=True))


if __name__ == "__main__":
    main()
