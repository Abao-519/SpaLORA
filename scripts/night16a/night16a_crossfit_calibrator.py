#!/usr/bin/env python3
"""Freeze cross-study self-calibrated partitions without held-out references.

The candidate descriptor table contains no public metrics.  For each held-out
study this runner opens a physically separate training-metric file that has no
row from that study, fits the same ridge meta-calibrator, and selects a
structurally eligible candidate using numeric descriptors only.  The frozen
partitions are evaluated by a separate process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


STUDY = {
    "A1": "LYMPH_NODE",
    "D1": "LYMPH_NODE",
    "tonsil_s1": "TONSIL",
    "tonsil_s2": "TONSIL",
    "tonsil_s3": "TONSIL",
    "P22": "P22",
    "P22_3DOT_K18": "P22",
    "MISAR_E15_5_S1": "MISAR",
    "MISAR_E15_5_S1_K12": "MISAR",
}

FAMILY = {
    "A1": "RNA_PROTEIN", "D1": "RNA_PROTEIN", "tonsil_s1": "RNA_PROTEIN",
    "tonsil_s2": "RNA_PROTEIN", "tonsil_s3": "RNA_PROTEIN",
    "P22": "RNA_ATAC", "P22_3DOT_K18": "RNA_ATAC",
    "MISAR_E15_5_S1": "RNA_ATAC", "MISAR_E15_5_S1_K12": "RNA_ATAC",
}

FORBIDDEN_DESCRIPTOR_EXACT = {
    "absolute_ari", "absolute_nmi", "ami", "fmi", "morans_i", "gearys_c", "score_target"
}


def array_sha256(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def robust_by_lane(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    output = np.zeros((len(frame), len(columns)), dtype=np.float64)
    for _, group in frame.groupby("lane", sort=False):
        values = group[columns].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
        median = np.median(values, axis=0)
        mad = np.median(np.abs(values - median), axis=0)
        scale = np.maximum(1.4826 * mad, np.std(values, axis=0))
        output[group.index] = np.clip((values - median) / np.maximum(scale, 1e-8), -6.0, 6.0)
    return output


def normalized_training_target(frame: pd.DataFrame) -> np.ndarray:
    raw = frame["absolute_ari"].astype(float).to_numpy() + 0.35 * frame["absolute_nmi"].astype(float).to_numpy()
    target = np.zeros(len(frame), dtype=np.float64)
    for _, group in frame.groupby("lane", sort=False):
        values = raw[group.index]
        target[group.index] = (values - values.min()) / max(float(values.max() - values.min()), 1e-8)
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--descriptors", type=Path, required=True)
    parser.add_argument("--partitions", type=Path, required=True)
    parser.add_argument("--training-metrics-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", choices=("GLOBAL", "FAMILY"), default="FAMILY")
    parser.add_argument("--ridge-alpha", type=float, default=0.1)
    parser.add_argument("--minimum-relative-cluster", type=float, default=0.02)
    parser.add_argument("--feature-profile", choices=("compact", "full"), default="compact")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "partitions").mkdir(exist_ok=True)

    descriptors = pd.read_csv(args.descriptors)
    forbidden = [
        column
        for column in descriptors
        if column.lower() in FORBIDDEN_DESCRIPTOR_EXACT
        or any(token in column.lower() for token in ("label", "truth", "reference_assignment"))
    ]
    if forbidden:
        raise RuntimeError(f"descriptor firewall violation: {forbidden}")
    descriptors["study"] = descriptors["lane"].map(STUDY)
    descriptors["family"] = descriptors["lane"].map(FAMILY)
    if descriptors[["study", "family"]].isna().any().any():
        raise RuntimeError("unregistered lane in descriptor file")
    descriptors = descriptors[
        descriptors["selector_min_cluster_relative_to_equal"].astype(float) >= args.minimum_relative_cluster
    ].copy().reset_index(drop=True)
    if args.feature_profile == "compact":
        numeric = [column for column in descriptors.columns if column.startswith("selector_")]
    else:
        numeric = [
            column for column in descriptors.columns
            if column.startswith(("selector_", "calibration_stat__", "calibration_constant__"))
        ]
    if not numeric:
        raise RuntimeError("no numeric self-calibration descriptors")
    x_numeric = robust_by_lane(descriptors, numeric)
    start = descriptors["start_name"].astype(str)
    start_classes = sorted({name.rsplit("::s", 1)[0] for name in start})
    x_start = np.column_stack([start.str.startswith(name + "::s").astype(float) for name in start_classes])
    x = np.concatenate((x_numeric, x_start), axis=1)
    archive = np.load(args.partitions, allow_pickle=False)
    registry = {
        "status": "NIGHT16A_CROSSFIT_SELF_CALIBRATED_PARTITIONS_FROZEN",
        "scope": args.scope,
        "ridge_alpha": args.ridge_alpha,
        "feature_profile": args.feature_profile,
        "minimum_relative_cluster": args.minimum_relative_cluster,
        "descriptor_file_sha256": file_sha256(args.descriptors),
        "partition_archive_sha256": file_sha256(args.partitions),
        "descriptor_features": numeric,
        "start_classes": start_classes,
        "held_out_reference_columns_opened": 0,
        "lanes": {},
        "checkpoints": {},
    }
    for held_out in sorted(descriptors["study"].unique()):
        metric_path = args.training_metrics_root / f"training_without_{held_out}.csv"
        metrics = pd.read_csv(metric_path)
        if held_out in set(metrics["study"].astype(str)):
            raise RuntimeError(f"{held_out}: training metrics contain held-out study")
        metric_values = metrics.drop(columns=["study"], errors="ignore")
        train_frame = descriptors.merge(metric_values, on=["candidate_key", "lane"], how="inner", validate="one_to_one")
        if args.scope == "FAMILY":
            held_family = descriptors.loc[descriptors["study"] == held_out, "family"].iloc[0]
            train_frame = train_frame[train_frame["family"] == held_family].copy()
        if train_frame.empty:
            raise RuntimeError(f"{held_out}: empty calibration training set")
        training_indices = descriptors.index[descriptors["candidate_key"].isin(train_frame["candidate_key"])].to_numpy()
        train_frame = train_frame.set_index("candidate_key").loc[descriptors.loc[training_indices, "candidate_key"]].reset_index()
        target = normalized_training_target(train_frame)
        model = Ridge(alpha=args.ridge_alpha)
        model.fit(x[training_indices], target)
        registry["checkpoints"][held_out] = {
            "training_metrics_file": metric_path.name,
            "training_metrics_sha256": file_sha256(metric_path),
            "training_studies": sorted(set(train_frame["study"])),
            "training_lanes": sorted(set(train_frame["lane"])),
            "coef": model.coef_.tolist(),
            "intercept": float(model.intercept_),
        }
        test_indices = descriptors.index[descriptors["study"] == held_out].to_numpy()
        predicted = model.predict(x[test_indices])
        for lane in descriptors.loc[test_indices, "lane"].unique():
            lane_mask = descriptors.loc[test_indices, "lane"].to_numpy() == lane
            lane_indices = test_indices[lane_mask]
            local_prediction = predicted[lane_mask]
            best = int(lane_indices[int(np.argmax(local_prediction))])
            row = descriptors.loc[best]
            partition = np.asarray(archive[str(row["partition_key"])], dtype=np.int32)
            destination = args.output / "partitions" / f"{lane}.npy"
            np.save(destination, partition, allow_pickle=False)
            registry["lanes"][lane] = {
                "held_out_study": held_out,
                "family": FAMILY[lane],
                "candidate_key": row["candidate_key"],
                "config_id": row["config_id"],
                "start_name": row["start_name"],
                "partition_key": row["partition_key"],
                "partition_sha256": array_sha256(partition),
                "predicted_quality": float(np.max(local_prediction)),
                "cluster_sizes": np.bincount(partition, minlength=int(row["k"])).tolist(),
                "parameter_source": "cross-study ridge over observable candidate and calibration descriptors",
            }
    (args.output / "frozen_crossfit_registry.json").write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
