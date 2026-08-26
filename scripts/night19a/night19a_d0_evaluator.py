#!/usr/bin/env python3
"""Independent post-lock annotation evaluator for Night-19A D0 artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    fowlkes_mallows_score,
    normalized_mutual_info_score,
)

from SpaLORA.night17b_sfrd import csr_from_carrier, encode_partition, same_head_partition, sha256_array


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_reference(kind: str, path: Path, ids: np.ndarray, label_key: str, mask_key: str):
    if kind == "npz":
        with np.load(path, allow_pickle=False) as archive:
            ref_ids = np.asarray(archive["ids"])
            labels = np.asarray(archive[label_key])
            mask = np.asarray(archive[mask_key], dtype=bool) if mask_key in archive.files else np.ones(ids.size, bool)
        if not np.array_equal(ref_ids, ids):
            raise ValueError("NPZ reference ordered IDs differ")
    elif kind == "h5ad":
        import anndata as ad

        frame = ad.read_h5ad(path)
        if label_key not in frame.obs:
            raise ValueError("H5AD reference label key missing")
        lookup = {str(value): index for index, value in enumerate(frame.obs_names.tolist())}
        if any(str(value) not in lookup for value in ids.tolist()):
            raise ValueError("producer ID absent from H5AD reference")
        order = [lookup[str(value)] for value in ids.tolist()]
        series = frame.obs[label_key].iloc[order]
        mask = np.asarray(series.notna(), dtype=bool)
        labels = np.asarray(series.astype(str))
    else:
        raise ValueError(kind)
    return labels, mask


def categorical_spatial(partition: np.ndarray, graph: sp.csr_matrix):
    labels = encode_partition(partition)
    upper = sp.triu(graph, k=1).tocoo()
    total = float(upper.data.sum())
    same = labels[upper.row] == labels[upper.col]
    agreement = float(upper.data[same].sum() / total) if total > 0 else float("nan")
    moran = []
    geary = []
    n = labels.size
    for cluster in range(int(labels.max()) + 1):
        x = (labels == cluster).astype(np.float64)
        centered = x - x.mean()
        denominator = float(np.sum(centered * centered))
        if denominator <= 0 or total <= 0:
            continue
        moran.append((n / total) * float(np.sum(upper.data * centered[upper.row] * centered[upper.col])) / denominator)
        geary.append(
            ((n - 1) / (2.0 * total))
            * float(np.sum(upper.data * (x[upper.row] - x[upper.col]) ** 2))
            / denominator
        )
    return agreement, float(np.mean(moran)), float(np.mean(geary))


def aligned_changed_count(reference: np.ndarray, candidate: np.ndarray) -> int:
    reference = encode_partition(reference)
    candidate = encode_partition(candidate)
    k = max(int(reference.max()), int(candidate.max())) + 1
    contingency = np.zeros((k, k), dtype=np.int64)
    np.add.at(contingency, (candidate, reference), 1)
    row, col = linear_sum_assignment(-contingency)
    mapping = np.arange(k, dtype=np.int32)
    mapping[row] = col
    return int(np.sum(mapping[candidate] != reference))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--producer-json", required=True)
    parser.add_argument("--carrier", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--reference-kind", choices=("npz", "h5ad"), required=True)
    parser.add_argument("--label-key", required=True)
    parser.add_argument("--mask-key", default="label_mask")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    started = time.time()
    artifact_path = Path(args.artifact)
    producer_path = Path(args.producer_json)
    producer = json.loads(producer_path.read_text(encoding="utf-8"))
    if file_sha256(artifact_path) != producer["artifact_sha256"]:
        raise ValueError("locked artifact SHA mismatch before annotation load")
    with np.load(artifact_path, allow_pickle=False) as archive:
        artifact = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(args.carrier, allow_pickle=False) as archive:
        carrier = {key: np.asarray(archive[key]) for key in archive.files}
    if not np.array_equal(artifact["ids"], carrier["ids"]):
        raise ValueError("artifact/carrier IDs differ")
    locked_partition_sha = sha256_array(artifact["partition"])
    locked_representation_sha = sha256_array(artifact["representation"])
    if locked_partition_sha != producer["partition_sha256"] or locked_representation_sha != producer["representation_sha256"]:
        raise ValueError("locked artifact semantic SHA mismatch")
    # This is the first annotation read in the process.
    reference_path = Path(args.reference)
    labels, mask = load_reference(
        args.reference_kind, reference_path, artifact["ids"], args.label_key, args.mask_key
    )
    truth = encode_partition(labels[mask])
    if np.unique(truth).size != int(producer["k"]):
        raise ValueError("reference K differs from registered K")
    graph = csr_from_carrier(carrier, "graph0")
    anchor_partition = same_head_partition(artifact["anchor_representation"], int(producer["k"]), seed=0)
    outputs = (
        ("REGISTERED_RELATION_SMOOTHED_ANCHOR_KMEANS", anchor_partition),
        ("D0_STANDARD_SUM_FINAL_KMEANS", artifact["partition"]),
    )
    rows = []
    for profile, raw_partition in outputs:
        partition = encode_partition(raw_partition)
        prediction = encode_partition(partition[mask])
        if np.unique(partition).size != int(producer["k"]):
            raise ValueError("evaluation partition violates exact K")
        agreement, moran, geary = categorical_spatial(partition, graph)
        sizes = np.bincount(partition, minlength=int(producer["k"])).astype(int)
        rows.append(
            {
                "lane": producer["lane"],
                "training_seed": int(producer["training_seed"]),
                "profile": profile,
                "ari": adjusted_rand_score(truth, prediction),
                "nmi": normalized_mutual_info_score(truth, prediction),
                "ami": adjusted_mutual_info_score(truth, prediction),
                "fmi": fowlkes_mallows_score(truth, prediction),
                "neighbor_agreement": agreement,
                "moran_macro_ovr": moran,
                "geary_macro_ovr": geary,
                "min_cluster_size": int(sizes.min()),
                "cluster_sizes": json.dumps(sizes.tolist(), separators=(",", ":")),
                "partition_sha256": sha256_array(partition),
                "changed_spots_vs_anchor": aligned_changed_count(anchor_partition, partition),
                "evaluator_label_reads": 1,
            }
        )
    output = Path(args.output)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    audit = {
        "schema": "night19a-gradient-d0-evaluator-v1",
        "lane": producer["lane"],
        "training_seed": int(producer["training_seed"]),
        "locked_artifact_verified_before_annotation_load": True,
        "locked_partition_sha256": locked_partition_sha,
        "locked_representation_sha256": locked_representation_sha,
        "reference_path": str(reference_path.resolve()),
        "reference_sha256": file_sha256(reference_path),
        "ordered_label_sha256": sha256_array(np.asarray(labels[mask]).astype("U")),
        "n_evaluated": int(mask.sum()),
        "reference_k": int(np.unique(truth).size),
        "metrics_sha256": file_sha256(output),
        "producer_label_reads": 0,
        "evaluator_label_reads": 1,
        "wall_seconds": float(time.time() - started),
        "peak_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
    }
    output.with_suffix(".evaluator.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
