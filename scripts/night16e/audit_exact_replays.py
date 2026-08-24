#!/usr/bin/env python3
"""Verify exact frozen-family, score-frontier, and new-unit replays."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


LANES = ("A1", "D1", "tonsil_s1", "tonsil_s2", "tonsil_s3", "P22", "MISAR_E15_5_S1")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare_group(first_root: Path, second_root: Path) -> dict[str, object]:
    rows = []
    for lane in LANES:
        first_manifest = json.loads((first_root / lane / "partitions.producer.json").read_text())
        second_manifest = json.loads((second_root / lane / "partitions.producer.json").read_text())
        first_partitions = [
            (row["candidate_id"], row["partition_sha256"], row.get("cluster_sizes_full"))
            for row in first_manifest["rows"]
        ]
        second_partitions = [
            (row["candidate_id"], row["partition_sha256"], row.get("cluster_sizes_full"))
            for row in second_manifest["rows"]
        ]
        first_eval = pd.read_csv(first_root / lane / "evaluation.csv")
        second_eval = pd.read_csv(second_root / lane / "evaluation.csv")
        columns = [
            "candidate_id",
            "partition_sha256",
            "absolute_ari",
            "absolute_nmi",
            "ami",
            "fmi",
            "cluster_sizes_full",
            "cluster_sizes_eval",
        ]
        exact = first_partitions == second_partitions and first_eval[columns].equals(second_eval[columns])
        if not exact:
            raise RuntimeError(f"fresh-process mismatch: {lane}")
        rows.append(
            {
                "lane": lane,
                "candidate_count": len(first_partitions),
                "partition_and_metric_exact": True,
                "producer_label_reads": first_manifest["producer_label_reads"],
                "dense_n_by_n_count": first_manifest["dense_n_by_n_count"],
            }
        )
    return {"lane_count": len(rows), "exact_lane_count": len(rows), "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--work", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    work = Path(args.work)
    human_first = work / "human_run1"
    human_second = work / "human_run2"
    first_manifest = json.loads((human_first / "partitions.producer.json").read_text())
    second_manifest = json.loads((human_second / "partitions.producer.json").read_text())
    human_first_rows = [
        (row["candidate_id"], row["partition_sha256"], row["cluster_sizes_full"])
        for row in first_manifest["rows"]
    ]
    human_second_rows = [
        (row["candidate_id"], row["partition_sha256"], row["cluster_sizes_full"])
        for row in second_manifest["rows"]
    ]
    human_eval_first = pd.read_csv(human_first / "evaluation.csv")
    human_eval_second = pd.read_csv(human_second / "evaluation.csv")
    metric_columns = ["candidate_id", "partition_sha256", "absolute_ari", "absolute_nmi", "ami", "fmi"]
    human_exact = human_first_rows == human_second_rows and human_eval_first[metric_columns].equals(
        human_eval_second[metric_columns]
    )
    if not human_exact:
        raise RuntimeError("human-hippocampus fresh-process mismatch")
    repo = Path(args.repo)
    source_paths = [
        repo / "SpaLORA/night16e_tsre.py",
        repo / "scripts/night16e/night16e_producer.py",
        repo / "scripts/night16e/night16e_evaluator.py",
        repo / "scripts/night16e/human_hippocampus_producer.py",
        repo / "scripts/night16e/human_hippocampus_evaluator.py",
    ]
    result = {
        "schema": "night16e-final-exact-replay-audit-v1",
        "family_frozen": compare_group(work / "family_run1", work / "family_run2"),
        "score_frontier": compare_group(work / "frontier_run1", work / "frontier_run2"),
        "human_hippocampus": {
            "candidate_count": len(human_first_rows),
            "partition_and_metric_exact": True,
            "producer_label_reads": first_manifest["producer_label_reads"],
            "dense_n_by_n_count": first_manifest["dense_n_by_n_count"],
        },
        "source_sha256": {str(path.relative_to(repo)): sha256(path) for path in source_paths},
    }
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
