#!/usr/bin/env python3
"""Independent post-lock evaluator for Night-17E candidate partitions."""

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

from scripts.night17e.night17e_producer import load_csr, sha256_file


def encode(value: np.ndarray) -> np.ndarray:
    return np.unique(np.asarray(value).astype(str), return_inverse=True)[1].astype(np.int32)


def categorical_spatial_metrics(partition: np.ndarray, graph: sp.csr_matrix) -> tuple[float, float, float]:
    partition = encode(partition)
    graph = sp.csr_matrix(graph, dtype=np.float64).maximum(sp.csr_matrix(graph).T).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    total_weight = float(graph.sum())
    if total_weight <= 0:
        raise ValueError("spatial graph has no positive weight")
    n = len(partition)
    upper = sp.triu(graph, k=1, format="coo")
    moran, geary = [], []
    for group in range(int(partition.max()) + 1):
        indicator = (partition == group).astype(np.float64)
        centered = indicator - float(np.mean(indicator))
        denominator = float(np.sum(centered**2))
        if denominator <= 0:
            continue
        moran.append(float(n / total_weight * (centered @ (graph @ centered)) / denominator))
        numerator = float(np.sum(upper.data * (indicator[upper.row] - indicator[upper.col]) ** 2))
        geary.append(float((n - 1) / total_weight * numerator / denominator))
    rows = np.repeat(np.arange(n), np.diff(graph.indptr))
    agreement = float(np.sum(graph.data * (partition[rows] == partition[graph.indices])) / total_weight)
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
        ids = np.asarray(value.obs_names.astype(str)).astype("U")
        series = value.obs[args.label_key]
        missing = np.asarray(series.isna(), dtype=bool)
        labels = np.asarray(series.astype(str)).astype("U")
        mask = ~missing
        provenance = f"h5ad-obs:{args.label_key}"
    else:
        raise ValueError("unsupported reference kind")
    if not (len(ids) == len(labels) == len(mask)) or len(np.unique(ids)) != len(ids):
        raise ValueError("reference ID/label/mask contract invalid")
    return ids, labels, mask, provenance


def run(args: argparse.Namespace) -> None:
    producer_path = Path(args.partition_bank).with_suffix(".producer.json")
    producer = json.loads(producer_path.read_text(encoding="utf-8"))
    with np.load(args.partition_bank, allow_pickle=False) as bank:
        bank_ids = np.asarray(bank["ids"]).astype("U")
        partitions = np.asarray(bank["partitions"], dtype=np.int32)
    with np.load(args.carrier, allow_pickle=False) as carrier:
        carrier_ids = np.asarray(carrier["ids"]).astype("U")
        graph = load_csr(carrier, "graph1")
    if not np.array_equal(bank_ids, carrier_ids):
        raise ValueError("partition-bank/carrier ordered ID mismatch")
    reference_ids, reference_labels, reference_mask, provenance = load_reference(args)
    lookup = {identifier: index for index, identifier in enumerate(reference_ids)}
    if not set(carrier_ids).issubset(lookup):
        raise ValueError("carrier IDs absent from reference")
    order = np.asarray([lookup[x] for x in carrier_ids], dtype=np.int64)
    labels = reference_labels[order]
    mask = reference_mask[order]
    truth = labels[mask]
    if len(np.unique(truth)) != int(args.k):
        raise ValueError("reference K mismatch")
    results = []
    for source in producer["rows"]:
        row = {
            "lane": args.lane,
            "candidate_id": source["run_id"],
            "start_id": source.get("start_id", ""),
            "start_role": source.get("start_role", ""),
            "arm": source.get("arm", ""),
            "config_id": source.get("config_id", ""),
            "partition_sha256": source.get("partition_sha256", ""),
            "status": source["status"],
            "failure": source.get("failure", ""),
            "changed_from_initial": source.get("changed_from_initial", ""),
            "wall_seconds": source.get("wall_seconds", ""),
        }
        if source["status"] == "PASS":
            partition = partitions[int(source["partition_index"])]
            full = encode(partition)
            sizes_full = np.bincount(full, minlength=int(args.k))
            sizes_eval = np.bincount(full[mask], minlength=int(args.k))
            if np.unique(full).size != int(args.k) or np.any(sizes_full <= 0):
                raise RuntimeError("locked partition violates exact K/no empty")
            prediction = partition[mask]
            moran, geary, agreement = categorical_spatial_metrics(partition, graph)
            row.update(
                absolute_ari=float(adjusted_rand_score(truth, prediction)),
                absolute_nmi=float(normalized_mutual_info_score(truth, prediction)),
                ami=float(adjusted_mutual_info_score(truth, prediction)),
                fmi=float(fowlkes_mallows_score(truth, prediction)),
                homogeneity=float(homogeneity_score(truth, prediction)),
                v_measure=float(v_measure_score(truth, prediction)),
                morans_i_macro=moran,
                gearys_c_macro=geary,
                neighbor_agreement=agreement,
                n_total=len(partition),
                n_evaluated=int(np.sum(mask)),
                k=int(args.k),
                cluster_sizes_full=json.dumps(sizes_full.astype(int).tolist()),
                cluster_sizes_eval=json.dumps(sizes_eval.astype(int).tolist()),
                min_cluster_size_full=int(np.min(sizes_full)),
                min_cluster_size_eval=int(np.min(sizes_eval)),
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
    reference_payload = b"\0".join(
        f"{identifier}\t{label}".encode() for identifier, label, keep in zip(carrier_ids, labels, mask) if keep
    )
    output.with_suffix(".evaluator.json").write_text(
        json.dumps(
            {
                "schema": "night17e-independent-evaluator-v1",
                "lane": args.lane,
                "reference_provenance": provenance,
                "reference_k": int(args.k),
                "n_total": len(carrier_ids),
                "n_evaluated": int(np.sum(mask)),
                "ordered_reference_sha256": hashlib.sha256(reference_payload).hexdigest(),
                "partition_artifact_sha256": sha256_file(Path(args.partition_bank)),
                "producer_manifest_sha256": sha256_file(producer_path),
                "labels_loaded_only_after_partition_lock": True,
                "producer_label_reads": 0,
                "evaluator_label_reads": 1,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--carrier", required=True)
    parser.add_argument("--partition-bank", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--reference-kind", choices=("npz", "h5ad"), required=True)
    parser.add_argument("--reference-id-key", default="ids")
    parser.add_argument("--label-key", required=True)
    parser.add_argument("--mask-key", default="label_mask")
    parser.add_argument("--lane", required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

