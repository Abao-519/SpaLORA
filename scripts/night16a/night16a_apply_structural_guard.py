#!/usr/bin/env python3
"""Apply one fixed, label-free microcluster guard to a candidate arena."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from SpaLORA.night15e_continuous_reliability_energy import csr_from_archive, reduce_full
from scripts.night15g.night15g_optional_view_search import evaluate
from scripts.night16a.night16a_balanced_repair_probe import repair
from scripts.night16a.night16a_calibrated_arena import LANE_DATASET, partition_descriptors


def sha(value):
    value = np.ascontiguousarray(value); digest = hashlib.sha256(); digest.update(value.dtype.str.encode()); digest.update(np.asarray(value.shape, np.int64).tobytes()); digest.update(value.tobytes()); return digest.hexdigest()


def semantics(data, lane):
    if lane == "P22_3DOT_K18":
        return 18, np.asarray(data["labels_k18_author_assignment"]), np.ones(len(data["ids"]), bool)
    if lane == "MISAR_E15_5_S1_K12":
        return 12, np.asarray(data["labels_primary"]), np.asarray(data["label_mask"], bool)
    return int(data["k_primary"][0]), np.asarray(data["labels_primary"]), np.asarray(data["label_mask"], bool)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arena", type=Path, required=True)
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--banks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--relative-threshold", type=float, default=0.05)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.arena / "calibrated_arena_ledger.csv")
    archive = np.load(args.arena / "calibrated_arena_partitions.npz", allow_pickle=False)
    rows, payload, audit = [], {}, {}
    for lane, group in frame.groupby("lane", sort=False):
        dataset = LANE_DATASET[lane]
        data = np.load(args.kit / f"{dataset}.npz", allow_pickle=False, mmap_mode="r")
        bank = np.load(args.banks / f"{dataset}_selected_partition_bank.npz", allow_pickle=False)
        k, labels, mask = semantics(data, lane)
        graph = csr_from_archive(data, "graph")
        feature = reduce_full(bank[f"{lane}__retained_embedding"], min(48, bank[f"{lane}__retained_embedding"].shape[1]))
        blocks = {
            "retained": reduce_full(bank[f"{lane}__retained_embedding"], min(32, bank[f"{lane}__retained_embedding"].shape[1])),
            "view1": reduce_full(data["view1"], min(24, data["view1"].shape[1])),
            "view2": reduce_full(data["view2"], min(24, data["view2"].shape[1])),
        }
        changed = 0
        for _, source in group.iterrows():
            initial = np.asarray(archive[str(source.partition_key)], dtype=np.int32)
            try:
                partition, threshold, removed, sizes = repair(
                    initial, feature, k, args.relative_threshold, 0.5, "mean"
                )
                status, failure = "PASS", ""
            except Exception as exc:
                partition = initial.copy(); sizes = np.bincount(partition, minlength=k)
                threshold, removed, status, failure = 0, 0, "FAILED", f"{type(exc).__name__}: {exc}"
            changed += int(not np.array_equal(partition, initial))
            metrics = evaluate(labels, mask, partition, graph)
            digest = sha(partition)
            key = f"{lane}__{source.config_id}__guard__{digest[:12]}"
            payload[key] = partition.astype(np.int32)
            row = source.to_dict()
            row.update({
                "candidate_key": f"{source.candidate_key}::STRUCTURAL_GUARD_005",
                "partition_key": key, "partition_sha256": digest,
                "status": status, "failure": failure,
                "absolute_ari": metrics["absolute_ari"], "absolute_nmi": metrics["absolute_nmi"],
                "ami": metrics["ami"], "fmi": metrics["fmi"],
                "min_cluster_size": int(sizes.min()), "cluster_sizes": json.dumps(sizes.tolist()),
                "structural_guard_threshold": threshold, "structural_guard_removed_clusters": removed,
                "structural_guard_changed": int(not np.array_equal(partition, initial)),
                **partition_descriptors(partition, k, graph, blocks),
            })
            rows.append(row)
        audit[lane] = {"rows": len(group), "changed_partitions": changed}
    columns = sorted({key for row in rows for key in row})
    with (args.output / "calibrated_arena_ledger.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(args.output / "calibrated_arena_partitions.npz", **payload)
    (args.output / "structural_guard_manifest.json").write_text(json.dumps({
        "relative_threshold": args.relative_threshold,
        "split_quantile": 0.5,
        "feature": "retained",
        "dispersion_mode": "mean",
        "label_reads_before_partition": 0,
        "evaluation_after_partition": 1,
        "lanes": audit,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
