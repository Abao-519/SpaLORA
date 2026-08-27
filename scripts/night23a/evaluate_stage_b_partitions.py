"""Independent benchmark evaluator for already locked and replayed Stage-B partitions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    fowlkes_mallows_score,
    homogeneity_score,
    normalized_mutual_info_score,
    v_measure_score,
)

from SpaLORA.night23a_xbed import array_sha, file_sha, load_csr


def encode(value: np.ndarray) -> np.ndarray:
    return np.unique(np.asarray(value).astype(str), return_inverse=True)[1].astype(np.int32)


def ordered_reference_hash(ids: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> str:
    payload = b"\0".join(f"{i}\t{y}".encode() for i, y, keep in zip(ids, labels, mask) if keep)
    return hashlib.sha256(payload).hexdigest()


def categorical_spatial_metrics(partition: np.ndarray, graph: sp.csr_matrix) -> tuple[float, float, float]:
    partition = encode(partition)
    graph = sp.csr_matrix(graph, dtype=np.float64).maximum(sp.csr_matrix(graph).T).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    total = float(graph.sum())
    if total <= 0:
        raise RuntimeError("empty spatial graph")
    n = len(partition)
    upper = sp.triu(graph, k=1, format="coo")
    moran, geary = [], []
    for group in range(int(partition.max()) + 1):
        indicator = (partition == group).astype(np.float64)
        centered = indicator - indicator.mean()
        denominator = float(np.sum(centered**2))
        if denominator <= 0:
            continue
        moran.append(float(n / total * (centered @ (graph @ centered)) / denominator))
        numerator = float(np.sum(upper.data * (indicator[upper.row] - indicator[upper.col]) ** 2))
        geary.append(float((n - 1) / total * numerator / denominator))
    graph_rows = np.repeat(np.arange(n), np.diff(graph.indptr))
    agreement = float(np.sum(graph.data * (partition[graph_rows] == partition[graph.indices])) / total)
    return float(np.mean(moran)), float(np.mean(geary)), agreement


def load_reference(args) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    if args.reference_kind == "npz":
        with np.load(args.reference, allow_pickle=False) as archive:
            ids = np.asarray(archive[args.reference_id_key]).astype("U")
            labels = np.asarray(archive[args.label_key]).astype("U")
            mask = np.asarray(archive[args.mask_key], dtype=bool)
        provenance = f"npz:{args.label_key}:{args.mask_key}"
    elif args.reference_kind == "h5ad":
        value = ad.read_h5ad(args.reference)
        ids = np.asarray(value.obs_names.astype(str)).astype("U")
        series = value.obs[args.label_key]
        mask = ~np.asarray(series.isna(), dtype=bool)
        labels = np.asarray(series.astype(str)).astype("U")
        provenance = f"h5ad-obs:{args.label_key}"
    else:
        with Path(args.reference).open(newline="", encoding="utf-8") as handle:
            records = list(csv.DictReader(handle, delimiter="\t"))
        ids = np.asarray([row[args.reference_id_key] for row in records]).astype("U")
        labels = np.asarray([row[args.label_key] for row in records]).astype("U")
        mask = np.ones(len(ids), dtype=bool)
        provenance = f"tsv:{args.label_key}"
    if not (len(ids) == len(labels) == len(mask)) or len(np.unique(ids)) != len(ids):
        raise RuntimeError("reference authority invalid")
    return ids, labels, mask, provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--carrier", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--reference-kind", choices=("npz", "h5ad", "tsv"), required=True)
    parser.add_argument("--reference-id-key", default="ids")
    parser.add_argument("--label-key", required=True)
    parser.add_argument("--mask-key", default="label_mask")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    replay = json.loads(Path(args.replay).read_text(encoding="utf-8"))
    if replay["status"] != "PASS" or replay["labels_read"] != 0 or file_sha(args.bank) != manifest["bank_sha256"]:
        raise RuntimeError("locked/replay authority failure")
    with np.load(args.bank, allow_pickle=False) as archive:
        ids = np.asarray(archive["ids"]).astype("U")
        candidate_ids = np.asarray(archive["candidate_ids"]).astype("U")
        partitions = np.asarray(archive["partitions"], dtype=np.int32)
    with np.load(args.carrier, allow_pickle=False) as archive:
        carrier_ids = np.asarray(archive["ids"]).astype("U")
        spatial = load_csr(archive, "graph0")
    if not np.array_equal(ids, carrier_ids) or array_sha(partitions) != manifest["partitions_sha256"]:
        raise RuntimeError("bank/carrier authority mismatch")
    reference_ids, labels, mask, provenance = load_reference(args)
    lookup = {identifier: index for index, identifier in enumerate(reference_ids)}
    if not set(ids).issubset(lookup):
        raise RuntimeError("carrier IDs absent from reference")
    order = np.asarray([lookup[value] for value in ids], dtype=np.int64)
    labels, mask = labels[order], mask[order]
    truth = labels[mask]
    if len(np.unique(truth)) != args.k:
        raise RuntimeError("reference K mismatch")
    output_rows = []
    for candidate_id, partition in zip(candidate_ids, partitions):
        encoded = encode(partition)
        sizes_full = np.bincount(encoded, minlength=args.k)
        sizes_eval = np.bincount(encoded[mask], minlength=args.k)
        if len(np.unique(encoded)) != args.k or np.any(sizes_full <= 0):
            raise RuntimeError("locked partition exact-K/no-empty failure")
        prediction = partition[mask]
        moran, geary, agreement = categorical_spatial_metrics(partition, spatial)
        output_rows.append(
            {
                "lane": args.lane,
                "candidate_id": candidate_id,
                "partition_sha256": array_sha(partition),
                "ari": adjusted_rand_score(truth, prediction),
                "nmi": normalized_mutual_info_score(truth, prediction),
                "ami": adjusted_mutual_info_score(truth, prediction),
                "fmi": fowlkes_mallows_score(truth, prediction),
                "homogeneity": homogeneity_score(truth, prediction),
                "v_measure": v_measure_score(truth, prediction),
                "morans_i_macro": moran,
                "gearys_c_macro": geary,
                "neighbor_agreement": agreement,
                "n_total": len(ids),
                "n_eval": int(mask.sum()),
                "k": args.k,
                "cluster_sizes_full": json.dumps(sizes_full.astype(int).tolist()),
                "cluster_sizes_eval": json.dumps(sizes_eval.astype(int).tolist()),
                "min_cluster_size_full": int(sizes_full.min()),
                "min_cluster_size_eval": int(sizes_eval.min()),
            }
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    audit = {
        "schema": "night23a-stage-b-independent-evaluation-v1",
        "lane": args.lane,
        "rows": len(output_rows),
        "partition_locked_and_replayed_before_reference_read": True,
        "producer_labels_read": manifest["labels_read"],
        "evaluator_reference_reads": 1,
        "reference_kind": args.reference_kind,
        "reference_provenance": provenance,
        "reference_file_sha256": file_sha(args.reference),
        "ordered_reference_sha256": ordered_reference_hash(ids, labels, mask),
        "reference_category_counts": {str(v): int(np.sum(truth == v)) for v in sorted(np.unique(truth))},
        "output_sha256": file_sha(output),
    }
    output.with_suffix(".json").write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"lane": args.lane, "rows": len(output_rows), "output_sha256": audit["output_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
