#!/usr/bin/env python3
"""Cross-study selection of a compact family/global calibration template.

Every lane starts from the same deterministic retained-view KMeans generator.
One calibration-constant row is chosen on other studies, while current-data
observable statistics still generate the actual energy parameters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.night16a.night16a_crossfit_calibrator import FAMILY, STUDY, array_sha256, file_sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--descriptors", type=Path, required=True)
    parser.add_argument("--partitions", type=Path, required=True)
    parser.add_argument("--training-metrics-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", choices=("GLOBAL", "FAMILY"), default="FAMILY")
    parser.add_argument("--minimum-relative-cluster", type=float, default=0.02)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True); (args.output / "partitions").mkdir(exist_ok=True)
    descriptors = pd.read_csv(args.descriptors)
    descriptors["study"] = descriptors["lane"].map(STUDY)
    descriptors["family"] = descriptors["lane"].map(FAMILY)
    archive = np.load(args.partitions, allow_pickle=False)
    registry = {
        "status": "NIGHT16A_CROSSFIT_CONSTANT_POLICY_FROZEN",
        "scope": args.scope,
        "start_generator": "KMeans(retained, K, random_state=0, n_init=30)",
        "minimum_relative_cluster": args.minimum_relative_cluster,
        "held_out_reference_columns_opened": 0,
        "descriptor_sha256": file_sha256(args.descriptors),
        "partition_archive_sha256": file_sha256(args.partitions),
        "lanes": {}, "checkpoints": {},
    }
    for held_out in sorted(descriptors.study.unique()):
        metrics_path = args.training_metrics_root / f"training_without_{held_out}.csv"
        metrics = pd.read_csv(metrics_path)
        if held_out in set(metrics.study):
            raise RuntimeError(f"{held_out}: held-out metric leakage")
        training = descriptors.merge(metrics.drop(columns=["study"]), on=["candidate_key", "lane"], validate="one_to_one")
        held_family = descriptors.loc[descriptors.study == held_out, "family"].iloc[0]
        if args.scope == "FAMILY":
            training = training[training.family == held_family].copy()
        if training.empty:
            raise RuntimeError(f"{held_out}: empty training block")
        training["raw_target"] = training.absolute_ari.astype(float) + 0.35 * training.absolute_nmi.astype(float)
        normalized = []
        for _, group in training.groupby("lane", sort=False):
            low, high = group.raw_target.min(), group.raw_target.max()
            score = (group.raw_target - low) / max(float(high - low), 1e-8)
            normalized.append(pd.Series(score.to_numpy(), index=group.index))
        training["normalized_target"] = pd.concat(normalized).sort_index()
        ranking = (
            training.groupby("config_id", as_index=False)
            .agg(training_score=("normalized_target", "mean"), training_lane_count=("lane", "nunique"))
            .sort_values(["training_score", "config_id"], ascending=[False, True], kind="mergesort")
        )
        held_rows = descriptors[descriptors.study == held_out].copy()
        eligible_configs = []
        for config_id in ranking.config_id:
            block = held_rows[held_rows.config_id == config_id]
            if len(block) == held_rows.lane.nunique() and np.all(
                block.selector_min_cluster_relative_to_equal.astype(float) >= args.minimum_relative_cluster
            ):
                eligible_configs.append(config_id)
        if not eligible_configs:
            raise RuntimeError(f"{held_out}: no structurally eligible transferred constant row")
        selected_config = eligible_configs[0]
        selected_rank = int(ranking.index[ranking.config_id == selected_config][0])
        training_score = float(ranking.loc[ranking.config_id == selected_config, "training_score"].iloc[0])
        checkpoint = held_rows[held_rows.config_id == selected_config].iloc[0]
        constants = {
            column.removeprefix("calibration_constant__"): (
                checkpoint[column].item() if isinstance(checkpoint[column], np.generic) else checkpoint[column]
            )
            for column in descriptors
            if column.startswith("calibration_constant__")
        }
        registry["checkpoints"][held_out] = {
            "selected_config_id": selected_config,
            "training_score": training_score,
            "training_metrics_file": metrics_path.name,
            "training_metrics_sha256": file_sha256(metrics_path),
            "training_studies": sorted(set(training.study)),
            "training_lanes": sorted(set(training.lane)),
            "calibration_constants": constants,
            "structural_rank_skip_count": selected_rank,
        }
        for _, row in held_rows[held_rows.config_id == selected_config].iterrows():
            partition = np.asarray(archive[str(row.partition_key)], dtype=np.int32)
            np.save(args.output / "partitions" / f"{row.lane}.npy", partition, allow_pickle=False)
            registry["lanes"][row.lane] = {
                "held_out_study": held_out, "family": row.family,
                "candidate_key": row.candidate_key, "partition_key": row.partition_key,
                "partition_sha256": array_sha256(partition), "config_id": selected_config,
                "start_name": row.start_name,
                "cluster_sizes": np.bincount(partition, minlength=int(row.k)).tolist(),
                "parameter_source": "other-study constant selection plus current observable-statistic formula",
            }
    (args.output / "frozen_crossfit_registry.json").write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
