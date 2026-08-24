#!/usr/bin/env python3
"""Independent evaluator for already-locked Night-16F partitions."""

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

from scripts.night16f.night16f_producer import load_csr


def encode(value: np.ndarray) -> np.ndarray:
    return np.unique(np.asarray(value).astype(str), return_inverse=True)[1].astype(np.int32)


def ordered_reference_hash(ids: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> str:
    payload = b"\0".join(
        f"{identifier}\t{label}".encode("utf-8")
        for identifier, label, keep in zip(ids, labels, mask)
        if keep
    )
    return hashlib.sha256(payload).hexdigest()


def categorical_spatial_metrics(
    partition: np.ndarray, graph: sp.csr_matrix
) -> tuple[float, float, float]:
    partition = encode(partition)
    graph = sp.csr_matrix(graph, dtype=np.float64).maximum(sp.csr_matrix(graph).T).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    total_weight = float(graph.sum())
    if total_weight <= 0:
        raise ValueError("spatial graph has no positive weight")
    n = len(partition)
    upper = sp.triu(graph, k=1, format="coo")
    moran: list[float] = []
    geary: list[float] = []
    for group in range(int(partition.max()) + 1):
        indicator = (partition == group).astype(np.float64)
        centered = indicator - float(np.mean(indicator))
        denominator = float(np.sum(centered**2))
        if denominator <= 0:
            continue
        moran.append(float(n / total_weight * (centered @ (graph @ centered)) / denominator))
        numerator = float(
            np.sum(upper.data * (indicator[upper.row] - indicator[upper.col]) ** 2)
        )
        geary.append(float((n - 1) / total_weight * numerator / denominator))
    rows = np.repeat(np.arange(n), np.diff(graph.indptr))
    agreement = float(
        np.sum(graph.data * (partition[rows] == partition[graph.indices])) / total_weight
    )
    return float(np.mean(moran)), float(np.mean(geary)), agreement


def load_reference(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    if args.reference_kind == "npz":
        with np.load(args.reference, allow_pickle=False) as archive:
            ids = np.asarray(archive[args.reference_id_key]).astype("U")
            labels = np.asarray(archive[args.label_key]).astype("U")
            mask = np.asarray(archive[args.mask_key], dtype=bool)
        provenance = f"npz:{args.label_key}:{args.mask_key}"
    elif args.reference_kind == "h5ad":
        value = ad.read_h5ad(args.reference)
        ids = np.asarray(value.obs_names.astype(str))
        series = value.obs[args.label_key]
        missing = np.asarray(series.isna(), dtype=bool)
        labels = np.asarray(series.astype(str)).astype("U")
        mask = ~missing
        provenance = f"h5ad-obs:{args.label_key}"
    else:
        with Path(args.reference).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        ids = np.asarray([row[args.reference_id_key] for row in rows]).astype("U")
        labels = np.asarray([row[args.label_key] for row in rows]).astype("U")
        mask = np.ones(len(ids), dtype=bool)
        provenance = f"tsv:{args.label_key}"
    if not (len(ids) == len(labels) == len(mask)) or len(np.unique(ids)) != len(ids):
        raise ValueError("reference ID/label/mask contract invalid")
    return ids, labels, mask, provenance


def run(args: argparse.Namespace) -> None:
    producer = json.loads(Path(args.producer_json).read_text())
    with np.load(args.partition_bank, allow_pickle=False) as bank:
        bank_ids = np.asarray(bank["ids"]).astype("U")
        partitions = np.asarray(bank["partitions"], dtype=np.int32)
    with np.load(args.carrier, allow_pickle=False) as carrier:
        carrier_ids = np.asarray(carrier["ids"]).astype("U")
        graph = load_csr(carrier, "graph1")
    if not np.array_equal(bank_ids, carrier_ids):
        raise ValueError("partition-bank/carrier ID mismatch")
    reference_ids, reference_labels, reference_mask, provenance = load_reference(args)
    lookup = {identifier: index for index, identifier in enumerate(reference_ids)}
    if not set(carrier_ids).issubset(lookup):
        raise ValueError("carrier IDs absent from reference")
    order = np.asarray([lookup[x] for x in carrier_ids], dtype=np.int64)
    labels = reference_labels[order]
    mask = reference_mask[order]
    if not np.any(mask):
        raise ValueError("empty evaluation mask")
    true_labels = labels[mask]
    truth_k = len(np.unique(true_labels))
    if truth_k != int(args.k):
        raise ValueError(f"reference K mismatch: observed {truth_k}, expected {args.k}")
    label_counts = {
        str(label): int(np.sum(true_labels == label)) for label in sorted(np.unique(true_labels))
    }
    results: list[dict[str, object]] = []
    for source in producer["rows"]:
        row = {
            key: source.get(key, "")
            for key in (
                "candidate_id", "start_id", "start_index", "start_role", "arm",
                "status", "failure", "partition_sha256", "initial_partition_sha256",
                "changed_from_initial", "wall_seconds",
            )
        }
        if source["status"] == "PASS":
            partition = partitions[int(source["partition_index"])]
            encoded_full = encode(partition)
            observed_k = len(np.unique(encoded_full))
            full_sizes = np.bincount(encoded_full, minlength=int(args.k))
            eval_sizes = np.bincount(encoded_full[mask], minlength=int(args.k))
            if observed_k != int(args.k) or np.any(full_sizes <= 0):
                raise RuntimeError("locked partition violates exact-K/no-empty contract")
            prediction = partition[mask]
            moran, geary, agreement = categorical_spatial_metrics(partition, graph)
            row.update(
                absolute_ari=float(adjusted_rand_score(true_labels, prediction)),
                absolute_nmi=float(normalized_mutual_info_score(true_labels, prediction)),
                ami=float(adjusted_mutual_info_score(true_labels, prediction)),
                fmi=float(fowlkes_mallows_score(true_labels, prediction)),
                homogeneity=float(homogeneity_score(true_labels, prediction)),
                v_measure=float(v_measure_score(true_labels, prediction)),
                morans_i_macro=moran,
                gearys_c_macro=geary,
                neighbor_agreement=agreement,
                n_total=int(len(partition)),
                n_evaluated=int(np.sum(mask)),
                k=int(args.k),
                cluster_sizes_full=json.dumps([int(x) for x in full_sizes]),
                cluster_sizes_eval=json.dumps([int(x) for x in eval_sizes]),
                min_cluster_size_full=int(full_sizes.min()),
                min_cluster_size_eval=int(eval_sizes.min()),
                evaluator_label_reads=1,
            )
        results.append(row)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in results for key in row})
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)
    audit = {
        "schema": "night16f-independent-evaluator-v1",
        "lane": args.lane,
        "data_id": args.data_id,
        "reference_kind": args.reference_kind,
        "reference_provenance": provenance,
        "reference_path": str(Path(args.reference).resolve()),
        "n_reference_rows": int(len(reference_ids)),
        "n_total": int(len(carrier_ids)),
        "n_evaluated": int(np.sum(mask)),
        "reference_k": int(truth_k),
        "reference_category_counts": label_counts,
        "ordered_reference_sha256": ordered_reference_hash(carrier_ids, labels, mask),
        "producer_label_reads": int(producer.get("producer_label_reads", -1)),
        "evaluator_label_reads": 1,
        "labels_loaded_after_locked_partition_artifact": True,
        "rows": len(results),
    }
    output.with_suffix(".evaluator.json").write_text(json.dumps(audit, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--carrier", required=True)
    parser.add_argument("--partition-bank", required=True)
    parser.add_argument("--producer-json", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--reference-kind", choices=("npz", "h5ad", "tsv"), required=True)
    parser.add_argument("--reference-id-key", default="ids")
    parser.add_argument("--label-key", required=True)
    parser.add_argument("--mask-key", default="label_mask")
    parser.add_argument("--data-id", required=True)
    parser.add_argument("--lane", required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
