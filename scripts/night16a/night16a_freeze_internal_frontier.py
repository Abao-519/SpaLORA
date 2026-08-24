#!/usr/bin/env python3
"""Freeze the label-assisted Night-16A development score frontier.

This artifact is an oracle/development ledger and is never relabelled as the
self-calibrated output.  It preserves the exact selected rows and partitions
needed to audit genuine score-frontier advances.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from SpaLORA.night15e_continuous_reliability_energy import csr_from_archive
from scripts.night15g.night15g_optional_view_search import evaluate


LANE_DATASET = {
    "A1": "A1", "D1": "D1", "tonsil_s1": "tonsil_s1", "tonsil_s2": "tonsil_s2",
    "tonsil_s3": "tonsil_s3", "P22": "P22", "P22_3DOT_K18": "P22",
    "MISAR_E15_5_S1": "MISAR_E15_5_S1", "MISAR_E15_5_S1_K12": "MISAR_E15_5_S1",
}

CURRENT = {
    "A1": (0.2760026589984753, 0.42173983941218224),
    "D1": (0.28887600833487515, 0.41376160498731795),
    "tonsil_s1": (0.23622005623915696, 0.3168843080420388),
    "tonsil_s2": (0.2575602926237894, 0.31187103018718415),
    "tonsil_s3": (0.349880567011789, 0.3092665192529602),
    "P22": (0.5939121542899773, 0.7142428513607495),
    "P22_3DOT_K18": (0.7396852743147485, 0.7542182135683022),
    "MISAR_E15_5_S1": (0.5414237853091904, 0.6667977615565593),
    "MISAR_E15_5_S1_K12": (0.45317629302756757, 0.5986289047196983),
}


def sha(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(); digest.update(value.dtype.str.encode()); digest.update(np.asarray(value.shape, np.int64).tobytes()); digest.update(value.tobytes())
    return digest.hexdigest()


def semantics(data, lane):
    if lane == "P22_3DOT_K18":
        return 18, np.asarray(data["labels_k18_author_assignment"]), np.ones(len(data["ids"]), bool)
    if lane == "MISAR_E15_5_S1_K12":
        return 12, np.asarray(data["labels_primary"]), np.asarray(data["label_mask"], bool)
    return int(data["k_primary"][0]), np.asarray(data["labels_primary"]), np.asarray(data["label_mask"], bool)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kit", type=Path, required=True)
    p.add_argument("--night15g-work", type=Path, required=True)
    p.add_argument("--night15f-partitions", type=Path, required=True)
    p.add_argument("--arena", type=Path, required=True)
    p.add_argument("--d1-repair", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True); (args.output / "partitions").mkdir(exist_ok=True)
    arena = pd.read_csv(args.arena / "calibrated_arena_ledger.csv")
    arena_archive = np.load(args.arena / "calibrated_arena_partitions.npz", allow_pickle=False)
    d1 = pd.read_csv(args.d1_repair / "balanced_repair_ledger.csv")
    d1_archive = np.load(args.d1_repair / "partitions.npz", allow_pickle=False)
    selected = {}
    for lane in LANE_DATASET:
        old_ari, old_nmi = CURRENT[lane]
        if lane == "D1":
            eligible = d1[(d1.status == "PASS") & (d1.min_cluster_size.astype(int) >= 34)].copy()
            eligible = eligible[(eligible.absolute_ari > old_ari) & (eligible.absolute_nmi > old_nmi)]
            eligible["objective"] = eligible.absolute_ari + 0.35 * eligible.absolute_nmi
            row_index = int(eligible.objective.idxmax())
            row = d1.loc[row_index]
            partition = np.asarray(d1_archive[f"p{row_index:04d}"], dtype=np.int32)
            source = {
                "source": "label_assisted_generic_structural_repair",
                "source_row_index": row_index,
                "feature": row.feature,
                "relative_threshold": float(row.relative_threshold),
                "split_quantile": float(row.split_quantile),
                "dispersion_mode": row.dispersion_mode,
            }
        else:
            eligible = arena[(arena.lane == lane) & (arena.status == "PASS")].copy()
            eligible = eligible[(eligible.absolute_ari > old_ari + 1e-12) & (eligible.absolute_nmi > old_nmi + 1e-12)]
            if eligible.empty:
                if lane in {"A1", "tonsil_s3"}:
                    path = (
                        args.night15g_work / "profile_replay_rev3_a1" / "partitions" / "A1__balanced.npy"
                        if lane == "A1"
                        else args.night15g_work / "profile_replay_rev3_tonsil" / "partitions" / "tonsil_s3__balanced.npy"
                    )
                else:
                    path = args.night15f_partitions / f"{lane}.npy"
                partition = np.load(path, allow_pickle=False)
                source = {"source": "previous_current_best_no_new_dual_frontier", "source_path": str(path)}
            else:
                eligible["objective"] = (eligible.absolute_ari - old_ari) + 0.35 * (eligible.absolute_nmi - old_nmi)
                row = eligible.loc[eligible.objective.idxmax()]
                key = f"{lane}__{row.config_id}__{str(row.partition_sha256)[:12]}"
                partition = np.asarray(arena_archive[key], dtype=np.int32)
                source = {
                    "source": "label_assisted_self_calibrated_energy_grid",
                    "config_id": row.config_id,
                    "constants_index": int(row.constants_index),
                    "start_name": row.start_name,
                    "source_partition_key": key,
                }
        dataset = LANE_DATASET[lane]
        data = np.load(args.kit / f"{dataset}.npz", allow_pickle=False, mmap_mode="r")
        k, labels, mask = semantics(data, lane)
        metric = evaluate(labels, mask, partition, csr_from_archive(data, "graph"))
        sizes = np.bincount(partition, minlength=k)
        path = args.output / "partitions" / f"{lane}.npy"; np.save(path, partition, allow_pickle=False)
        selected[lane] = {
            **source,
            "old_ari": old_ari, "old_nmi": old_nmi,
            "absolute_ari": metric["absolute_ari"], "absolute_nmi": metric["absolute_nmi"],
            "ami": metric["ami"], "fmi": metric["fmi"], "morans_i": metric["morans_i"], "gearys_c": metric["gearys_c"],
            "delta_ari": metric["absolute_ari"] - old_ari, "delta_nmi": metric["absolute_nmi"] - old_nmi,
            "cluster_sizes": sizes.tolist(), "min_cluster_size": int(sizes.min()),
            "partition_sha256": sha(partition), "labels_used_for_cross_run_selection": True,
            "eligible_for_automatic_main_result": False,
        }
    (args.output / "internal_frontier_registry.json").write_text(json.dumps({
        "status": "NIGHT16A_LABEL_ASSISTED_INTERNAL_SCORE_FRONTIER_FROZEN",
        "interpretation": "development oracle only; never substituted for self-calibrated main output",
        "lanes": selected,
    }, indent=2) + "\n", encoding="utf-8")
    with (args.output / "internal_frontier_table.csv").open("w", newline="", encoding="utf-8") as handle:
        flat = [{"lane": lane, **{k: v for k, v in row.items() if not isinstance(v, (list, dict))}, "cluster_sizes": json.dumps(row["cluster_sizes"])} for lane, row in selected.items()]
        writer = csv.DictWriter(handle, fieldnames=sorted({k for row in flat for k in row})); writer.writeheader(); writer.writerows(flat)


if __name__ == "__main__":
    main()
