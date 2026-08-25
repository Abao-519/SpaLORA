#!/usr/bin/env python
"""Fit a small transparent rank grid using training studies only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SpaLORA.night17d_learned_evidence import fit_weight_config, weight_grid  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path):
    return list(csv.DictReader(path.open(encoding="utf-8")))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--training", action="append", required=True, help="LANE=FEATURES=EVALUATION")
    parser.add_argument("--training-authority", action="append", required=True, help="LANE=SINGLE_ROW_CSV")
    parser.add_argument("--held-out-lane", default="NONE")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    lane_records, evaluation, source_hashes = {}, {}, {}
    for specification in args.training:
        lane, feature_name, evaluation_name = specification.split("=", 2)
        feature_path, evaluation_path = Path(feature_name), Path(evaluation_name)
        if lane == args.held_out_lane:
            raise ValueError("held-out lane was supplied to training fit")
        feature_rows = read_rows(feature_path)
        eval_rows = read_rows(evaluation_path)
        lane_records[lane] = feature_rows
        evaluation[lane] = {row["candidate_id"]: row for row in eval_rows}
        if set(evaluation[lane]) != {row["candidate_id"] for row in feature_rows}:
            raise ValueError(f"candidate/evaluation mismatch for {lane}")
        source_hashes[lane] = {"features": sha(feature_path), "evaluation": sha(evaluation_path)}
    authority = {}
    for specification in args.training_authority:
        lane, filename = specification.split("=", 1)
        path = Path(filename)
        rows = read_rows(path)
        if len(rows) != 1 or rows[0]["lane"] != lane:
            raise ValueError(f"authority slice for {lane} is not exactly one matching row")
        if lane == args.held_out_lane or lane not in lane_records:
            raise ValueError(f"authority slice for non-training lane {lane}")
        authority[lane] = {
            "absolute_ari": float(rows[0]["absolute_ari"]),
            "absolute_nmi": float(rows[0]["absolute_nmi"]),
        }
        source_hashes[lane]["authority"] = sha(path)
    if set(authority) != set(lane_records):
        raise ValueError("training authority slices do not exactly match training lanes")
    winner, summaries, detail = fit_weight_config(
        lane_records, evaluation, authority, weight_grid(freeze["grid"])
    )
    with (args.output_dir / "grid_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(summaries)
    with (args.output_dir / "grid_training_detail.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(detail)
    result = {
        "schema": "night17d-fitted-selector-v1",
        "training_lanes": sorted(lane_records),
        "held_out_lane_not_loaded": args.held_out_lane,
        "fitted_weights": winner.__dict__,
        "config_id": winner.config_id,
        "grid_size": len(summaries),
        "training_label_derived_evaluation_files_read": len(evaluation),
        "held_out_evaluation_files_read": 0,
        "held_out_authority_files_read": 0,
        "source_hashes": source_hashes,
        "freeze_sha256": sha(args.freeze),
    }
    (args.output_dir / "fitted_config.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
