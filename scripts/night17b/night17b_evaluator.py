#!/usr/bin/env python3
"""Independent public-annotation evaluator for locked Night-17B partitions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    fowlkes_mallows_score,
    normalized_mutual_info_score,
)

from SpaLORA.night17b_sfrd import (
    csr_from_carrier,
    encode_partition,
    sha256_array,
    training_seed_from_run_ids,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_reference(kind: str, path: Path, ids: np.ndarray, key: str):
    if kind == "npz":
        with np.load(path, allow_pickle=False) as archive:
            ref_ids = np.asarray(archive["ids"])
            labels = np.asarray(archive[key])
            mask = np.asarray(archive["label_mask"], dtype=bool) if "label_mask" in archive.files else np.ones(ids.size, dtype=bool)
        if not np.array_equal(ref_ids, ids):
            raise ValueError("reference ordered IDs differ from producer")
    elif kind == "h5ad":
        import anndata as ad

        frame = ad.read_h5ad(path)
        if key not in frame.obs:
            raise ValueError(f"missing reference column {key}")
        lookup = {str(value): index for index, value in enumerate(frame.obs_names.tolist())}
        if any(str(value) not in lookup for value in ids.tolist()):
            raise ValueError("producer ID absent from reference")
        order = [lookup[str(value)] for value in ids.tolist()]
        series = frame.obs[key].iloc[order]
        mask = np.asarray(series.notna(), dtype=bool)
        labels = np.asarray(series.astype(str))
    else:
        raise ValueError(kind)
    return labels, mask


def categorical_spatial(partition: np.ndarray, graph):
    labels = encode_partition(partition)
    upper = __import__("scipy.sparse", fromlist=["triu"]).triu(graph, k=1).tocoo()
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
        geary.append(((n - 1) / (2.0 * total)) * float(np.sum(upper.data * (x[upper.row] - x[upper.col]) ** 2)) / denominator)
    return agreement, float(np.mean(moran)), float(np.mean(geary))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--producer-json", required=True)
    parser.add_argument("--carrier", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--reference-kind", choices=("npz", "h5ad"), required=True)
    parser.add_argument("--label-key", required=True)
    parser.add_argument("--lane", required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    started = time.time()
    artifact_path = Path(args.artifact)
    producer = json.loads(Path(args.producer_json).read_text(encoding="utf-8"))
    if producer["artifact_sha256"] != file_sha256(artifact_path):
        raise ValueError("locked artifact hash mismatch")
    with np.load(artifact_path, allow_pickle=False) as archive:
        artifact = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(args.carrier, allow_pickle=False) as archive:
        carrier = {key: np.asarray(archive[key]) for key in archive.files}
    if not np.array_equal(artifact["ids"], carrier["ids"]):
        raise ValueError("artifact/carrier IDs differ")
    labels, mask = load_reference(args.reference_kind, Path(args.reference), artifact["ids"], args.label_key)
    truth = encode_partition(labels[mask])
    if np.unique(truth).size != args.k:
        raise ValueError("registered reference K mismatch")
    graph = csr_from_carrier(carrier, "graph0")
    rows = []
    for index, partition in enumerate(artifact["partitions"]):
        prediction = encode_partition(partition)
        pred_eval = encode_partition(prediction[mask])
        if np.unique(prediction).size != args.k:
            raise ValueError("producer partition violates exact K")
        agreement, moran, geary = categorical_spatial(prediction, graph)
        sizes = np.bincount(prediction, minlength=args.k).astype(int)
        rows.append(
            {
                "lane": args.lane,
                "run_id": str(artifact["run_ids"][index]),
                "arm": str(artifact["arm_ids"][index]),
                "config_id": str(artifact["config_ids"][index]),
                "ari": adjusted_rand_score(truth, pred_eval),
                "nmi": normalized_mutual_info_score(truth, pred_eval),
                "ami": adjusted_mutual_info_score(truth, pred_eval),
                "fmi": fowlkes_mallows_score(truth, pred_eval),
                "neighbor_agreement": agreement,
                "moran_macro_ovr": moran,
                "geary_macro_ovr": geary,
                "min_cluster_size": int(sizes.min()),
                "cluster_sizes": json.dumps(sizes.tolist(), separators=(",", ":")),
                "partition_sha256": sha256_array(prediction),
                "representation_sha256": str(artifact["representation_sha256"][index]),
                "training_seed": training_seed_from_run_ids(
                    str(artifact["run_ids"][index]), artifact["run_ids"].tolist()
                ),
                "evaluator_label_reads": 1,
            }
        )
    output = Path(args.output)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    audit = {
        "schema": "night17b-sfrd-evaluator-v1",
        "lane": args.lane,
        "producer_locked_before_label_load": True,
        "evaluated_observations": int(mask.sum()),
        "reference_k": int(np.unique(truth).size),
        "ordered_label_sha256": sha256_array(np.asarray(labels[mask]).astype("U")),
        "reference_path": str(Path(args.reference).resolve()),
        "reference_sha256": file_sha256(Path(args.reference)),
        "metrics_sha256": file_sha256(output),
        "wall_seconds": time.time() - started,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }
    output.with_suffix(".evaluator.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
