#!/usr/bin/env python3
"""Independent post-lock evaluator for Night-18A partitions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, fowlkes_mallows_score, normalized_mutual_info_score


def sha_array(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode()); digest.update(np.asarray(value.shape, dtype=np.int64).tobytes()); digest.update(value.tobytes())
    return digest.hexdigest()


def csr(archive: np.lib.npyio.NpzFile, prefix: str) -> sp.csr_matrix:
    return sp.csr_matrix((archive[f"{prefix}__data"], archive[f"{prefix}__indices"], archive[f"{prefix}__indptr"]),
                         shape=tuple(int(x) for x in archive[f"{prefix}__shape"]))


def spatial_metrics(partition: np.ndarray, graph: sp.csr_matrix) -> dict[str, float]:
    graph = sp.csr_matrix(graph, dtype=np.float64).maximum(sp.csr_matrix(graph).T).tocsr()
    graph.setdiag(0); graph.eliminate_zeros()
    rows = np.repeat(np.arange(graph.shape[0]), np.diff(graph.indptr)); cols = graph.indices
    total_weight = max(float(graph.data.sum()), 1e-12)
    neighbour = float(np.sum(graph.data * (partition[rows] == partition[cols])) / total_weight)
    n = len(partition); degree = np.asarray(graph.sum(1)).ravel(); weight_sum = max(float(graph.sum()), 1e-12)
    morans, gearys = [], []
    for cluster in np.unique(partition):
        x = (partition == cluster).astype(np.float64); centered = x - x.mean(); denom = max(float(np.sum(centered ** 2)), 1e-12)
        morans.append(float(n / weight_sum * np.sum(graph.data * centered[rows] * centered[cols]) / denom))
        gearys.append(float((n - 1) / (2 * weight_sum) * np.sum(graph.data * (x[rows] - x[cols]) ** 2) / denom))
    return {"neighbor_agreement": neighbour, "moran_macro_indicator": float(np.mean(morans)),
            "geary_macro_indicator": float(np.mean(gearys))}


def run(args: argparse.Namespace) -> None:
    with np.load(args.artifact, allow_pickle=False) as artifact:
        ids = artifact["ids"].astype("U"); partitions = artifact["partitions"].astype(np.int32)
        candidate_ids = artifact["candidate_ids"].astype("U"); names = artifact["selection_names"].astype("U")
        selection_indices = artifact["selection_indices"].astype(np.int32)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    with np.load(args.authority, allow_pickle=False) as authority:
        authority_ids = authority["ids"].astype("U"); labels = authority["labels_primary"].astype("U")
        mask = authority["label_mask"].astype(bool); graph = csr(authority, "operator4")
    if not np.array_equal(ids, authority_ids):
        raise ValueError("producer/evaluator ordered-ID mismatch")
    rows = []
    for name, index in zip(names, selection_indices):
        partition = partitions[int(index)]; predicted = partition[mask]; truth = labels[mask]
        sizes_full = np.bincount(partition, minlength=args.k); sizes_eval = np.bincount(partition[mask], minlength=args.k)
        metrics = spatial_metrics(partition, graph)
        record = {"lane": manifest["lane"], "config_id": manifest["config"]["config_id"], "training_seed": manifest["training_seed"],
                  "endpoint": str(name), "candidate_id": str(candidate_ids[int(index)]),
                  "partition_sha256": sha_array(partition), "n_total": len(partition), "n_eval": int(mask.sum()), "k": args.k,
                  "ari": adjusted_rand_score(truth, predicted), "nmi": normalized_mutual_info_score(truth, predicted),
                  "ami": adjusted_mutual_info_score(truth, predicted), "fmi": fowlkes_mallows_score(truth, predicted),
                  "min_cluster_size_full": int(sizes_full.min()), "cluster_sizes_full": json.dumps(sizes_full.tolist()),
                  "min_cluster_size_eval": int(sizes_eval.min()), "cluster_sizes_eval": json.dumps(sizes_eval.tolist()),
                  "wall_seconds": manifest["wall_seconds"], "gpu_peak_mib": manifest["gpu_peak_mib"],
                  "peak_rss_mib": manifest["peak_rss_mib"], **metrics}
        expected = manifest["selections"][str(name)]["partition_sha256"]
        if expected != record["partition_sha256"]:
            raise RuntimeError("locked partition hash mismatch")
        rows.append(record)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    output.with_suffix(".json").write_text(json.dumps({"status": "PASS", "label_reads": 1, "rows": len(rows),
                                                       "authority_labels_sha256": sha_array(labels),
                                                       "authority_mask_sha256": sha_array(mask)}, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--artifact", required=True); parser.add_argument("--manifest", required=True)
    parser.add_argument("--authority", required=True); parser.add_argument("--k", type=int, required=True); parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__": main()
