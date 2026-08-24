#!/usr/bin/env python3
"""Prepare physically separated descriptor and cross-study metric artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


STUDY = {
    "A1": "LYMPH_NODE", "D1": "LYMPH_NODE",
    "tonsil_s1": "TONSIL", "tonsil_s2": "TONSIL", "tonsil_s3": "TONSIL",
    "P22": "P22", "P22_3DOT_K18": "P22",
    "MISAR_E15_5_S1": "MISAR", "MISAR_E15_5_S1_K12": "MISAR",
}

METRIC_COLUMNS = {
    "absolute_ari", "absolute_nmi", "ami", "fmi", "morans_i", "gearys_c",
    "delta_ari", "delta_nmi", "score_target",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.development_ledger)
    frame = frame[frame["status"] == "PASS"].copy().reset_index(drop=True)
    frame["study"] = frame["lane"].map(STUDY)
    if frame["study"].isna().any():
        raise RuntimeError("unregistered lane")
    metrics = frame[["candidate_key", "lane", "study", "absolute_ari", "absolute_nmi", "ami", "fmi"]].copy()
    descriptor_columns = [column for column in frame if column not in METRIC_COLUMNS and column != "study"]
    descriptors = frame[descriptor_columns].copy()
    # Partition-agreement and edge-similarity descriptors may legitimately
    # contain the character sequence ``ari``.  Reject semantic reference
    # tokens and exact public-metric fields rather than substring matching.
    forbidden = [
        column
        for column in descriptors
        if column.lower() in METRIC_COLUMNS
        or any(token in column.lower() for token in ("label", "truth", "reference_assignment"))
    ]
    if forbidden:
        raise RuntimeError(f"redaction failure: {forbidden}")
    descriptors.to_csv(args.output / "candidate_descriptors_no_public_metrics.csv", index=False)
    files = {}
    for held_out in sorted(frame["study"].unique()):
        subset = metrics[metrics["study"] != held_out].copy()
        path = args.output / f"training_without_{held_out}.csv"
        subset.to_csv(path, index=False)
        files[held_out] = {"rows": len(subset), "studies": sorted(set(subset["study"])), "lanes": sorted(set(subset["lane"]))}
    (args.output / "split_manifest.json").write_text(
        json.dumps({
            "development_rows": len(frame),
            "descriptor_rows": len(descriptors),
            "descriptor_public_metric_columns": 0,
            "training_files": files,
        }, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
