#!/usr/bin/env python3
"""Independent public-benchmark evaluator for locked Night-16E partitions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

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

from SpaLORA.night15e_continuous_reliability_energy import csr_from_archive


def encode_partition(value: np.ndarray) -> np.ndarray:
    return np.unique(np.asarray(value), return_inverse=True)[1].astype(np.int32)


def cluster_sizes_full_eval(
    partition: np.ndarray, mask: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    partition = encode_partition(partition)
    mask = np.asarray(mask, dtype=bool)
    if len(partition) != len(mask):
        raise ValueError("evaluation mask length mismatch")
    full = np.bincount(partition, minlength=int(k))
    evaluated = np.bincount(partition[mask], minlength=int(k))
    return full, evaluated


def categorical_spatial_metrics(
    partition: np.ndarray, graph: sp.csr_matrix
) -> tuple[float, float, float]:
    partition = encode_partition(partition)
    graph = sp.csr_matrix(graph, dtype=np.float64)
    graph = graph.maximum(graph.T).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    total_weight = float(graph.sum())
    n = len(partition)
    moran = []
    geary = []
    weights = []
    for group in range(int(partition.max()) + 1):
        indicator = (partition == group).astype(np.float64)
        centered = indicator - float(np.mean(indicator))
        denominator = float(np.sum(centered**2))
        if denominator <= 0:
            continue
        moran.append(float(n / total_weight * (centered @ (graph @ centered)) / denominator))
        coo = sp.triu(graph, k=1, format="coo")
        numerator = float(np.sum(coo.data * (indicator[coo.row] - indicator[coo.col]) ** 2))
        geary.append(float((n - 1) / total_weight * numerator / denominator))
        weights.append(float(np.mean(indicator)))
    neighbor_agreement = float(
        np.sum(graph.data * (partition[np.repeat(np.arange(n), np.diff(graph.indptr))] == partition[graph.indices]))
        / total_weight
    )
    return float(np.mean(moran)), float(np.mean(geary)), neighbor_agreement


def evaluate(args: argparse.Namespace) -> None:
    producer = json.loads(Path(args.producer_json).read_text())
    with np.load(args.partition_bank, allow_pickle=False) as bank:
        partitions = np.asarray(bank["partitions"], dtype=np.int32)
    with np.load(Path(args.kit_root) / f"{args.data_id}.npz", allow_pickle=False) as archive:
        if args.label_key not in archive.files:
            raise KeyError(f"label key not found: {args.label_key}")
        if args.mask_key not in archive.files:
            raise KeyError(f"mask key not found: {args.mask_key}")
        labels = np.asarray(archive[args.label_key])
        mask = np.asarray(archive[args.mask_key], dtype=bool)
        graph = csr_from_archive(archive, "graph")
    truth = labels[mask]
    rows = []
    for source in producer["rows"]:
        row = {key: source.get(key, "") for key in (
            "candidate_id", "profile_id", "base_id", "variant", "config_sha256",
            "status", "failure", "partition_sha256", "changed_from_strong_start",
            "wall_seconds",
        )}
        row["evaluator_label_key"] = args.label_key
        row["evaluator_mask_key"] = args.mask_key
        if source["status"] == "PASS":
            partition = partitions[int(source["partition_index"])]
            prediction = partition[mask]
            full_sizes, eval_sizes = cluster_sizes_full_eval(partition, mask, args.k)
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
                n_total=int(len(partition)),
                n_evaluated=int(np.sum(mask)),
                k=int(args.k),
                cluster_sizes_full=json.dumps([int(x) for x in full_sizes]),
                cluster_sizes_eval=json.dumps([int(x) for x in eval_sizes]),
                min_cluster_size_full=int(full_sizes.min()),
                min_cluster_size_eval=int(eval_sizes.min()),
                evaluator_label_reads=1,
            )
        rows.append(row)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kit-root", required=True)
    parser.add_argument("--data-id", required=True)
    parser.add_argument("--k", required=True, type=int)
    parser.add_argument("--label-key", default="labels_primary")
    parser.add_argument("--mask-key", default="label_mask")
    parser.add_argument("--partition-bank", required=True)
    parser.add_argument("--producer-json", required=True)
    parser.add_argument("--output", required=True)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
