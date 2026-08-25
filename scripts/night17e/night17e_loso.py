#!/usr/bin/env python3
"""Strict two-process LOSO calibration and held-out application for Night-17E.

The ``fit`` command is only allowed to open training-lane evaluation CSV files.
The ``apply`` command opens the held-out producer manifest, but no labels or
evaluation metrics.  The ``evaluate`` command is the first process allowed to
open the held-out evaluation table.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable


TOL = 1e-12
CONFIG_COMPLEXITY = {
    "L01_MIX025": 0.25,
    "L02_MIX050": 0.50,
    "L03_MIX075": 0.75,
    "L04_MIX050_POWER2": 0.60,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(paths: Iterable[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def training_score(rows: list[dict[str, str]], config_id: str) -> tuple:
    details = []
    for lane in sorted({row["lane"] for row in rows}):
        lane_rows = [row for row in rows if row["lane"] == lane]
        baseline = [
            row
            for row in lane_rows
            if row["start_id"] == "METHOD_NIGHT16H_START" and row["arm"] == "INPUT_START"
        ]
        learned = [
            row
            for row in lane_rows
            if row["start_id"] == "METHOD_NIGHT16H_START"
            and row["arm"] == "LEARNED_RELATION"
            and row["config_id"] == config_id
        ]
        if len(baseline) != 1 or len(learned) != 1:
            raise ValueError(f"missing or ambiguous method-start row for {lane}/{config_id}")
        d_ari = float(learned[0]["absolute_ari"]) - float(baseline[0]["absolute_ari"])
        d_nmi = float(learned[0]["absolute_nmi"]) - float(baseline[0]["absolute_nmi"])
        details.append((lane, d_ari, d_nmi))
    dual_count = sum(d_ari > TOL and d_nmi > TOL for _, d_ari, d_nmi in details)
    worst_dari = min(d_ari for _, d_ari, _ in details)
    mean_dari = sum(d_ari for _, d_ari, _ in details) / len(details)
    mean_dnmi = sum(d_nmi for _, _, d_nmi in details) / len(details)
    return (
        dual_count,
        worst_dari,
        mean_dari,
        mean_dnmi,
        -CONFIG_COMPLEXITY[config_id],
        config_id,
        details,
    )


def fit(args: argparse.Namespace) -> None:
    paths = [Path(path) for path in args.training_evaluation]
    rows = read_rows(paths)
    lanes = sorted({row["lane"] for row in rows})
    if args.heldout_lane in lanes:
        raise ValueError("held-out lane was physically present in fit inputs")
    if set(lanes) != set(args.expected_training_lane):
        raise ValueError(f"training-lane contract mismatch: {lanes}")
    configs = sorted(CONFIG_COMPLEXITY)
    scored = {config: training_score(rows, config) for config in configs}
    selected = sorted(
        configs,
        key=lambda config: (
            -scored[config][0],
            -scored[config][1],
            -scored[config][2],
            -scored[config][3],
            CONFIG_COMPLEXITY[config],
            config,
        ),
    )[0]
    payload = {
        "schema": "night17e-strict-loso-fit-v1",
        "heldout_lane": args.heldout_lane,
        "training_lanes": lanes,
        "training_evaluation_files": [str(path) for path in paths],
        "training_evaluation_sha256": {str(path): sha256_file(path) for path in paths},
        "heldout_evaluation_files_read": 0,
        "selected_config_id": selected,
        "mechanical_rule": [
            "maximum count of training lanes with delta ARI>0 and delta NMI>0",
            "maximum worst training-lane delta ARI",
            "maximum mean delta ARI",
            "maximum mean delta NMI",
            "lower declared complexity",
            "lexically smaller config id",
        ],
        "config_scores": {
            config: {
                "dual_positive_count": scored[config][0],
                "worst_delta_ari": scored[config][1],
                "mean_delta_ari": scored[config][2],
                "mean_delta_nmi": scored[config][3],
                "complexity": CONFIG_COMPLEXITY[config],
                "per_lane": [
                    {"lane": lane, "delta_ari": d_ari, "delta_nmi": d_nmi}
                    for lane, d_ari, d_nmi in scored[config][-1]
                ],
            }
            for config in configs
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def apply(args: argparse.Namespace) -> None:
    fit_payload = json.loads(Path(args.fit_json).read_text(encoding="utf-8"))
    if fit_payload["heldout_lane"] != args.heldout_lane:
        raise ValueError("fit/held-out mismatch")
    producer_path = Path(args.producer_json)
    producer = json.loads(producer_path.read_text(encoding="utf-8"))
    config_id = fit_payload["selected_config_id"]
    rows = [
        row
        for row in producer["rows"]
        if row.get("start_id") == "METHOD_NIGHT16H_START"
        and row.get("config_id") == config_id
        and row.get("arm") == "LEARNED_RELATION"
        and row.get("status") == "PASS"
    ]
    if len(rows) != 1:
        raise ValueError("held-out producer row is missing or ambiguous")
    row = rows[0]
    payload = {
        "schema": "night17e-strict-loso-heldout-selection-v1",
        "lane": args.heldout_lane,
        "fit_json_sha256": sha256_file(Path(args.fit_json)),
        "producer_json_sha256": sha256_file(producer_path),
        "heldout_evaluation_files_read": 0,
        "selected_config_id": config_id,
        "candidate_id": row["run_id"],
        "partition_sha256": row["partition_sha256"],
        "start_id": row["start_id"],
        "arm": row["arm"],
    }
    Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def evaluate(args: argparse.Namespace) -> None:
    selection = json.loads(Path(args.selection_json).read_text(encoding="utf-8"))
    rows = read_rows([Path(args.heldout_evaluation)])
    matches = [row for row in rows if row["candidate_id"] == selection["candidate_id"]]
    if len(matches) != 1:
        raise ValueError("held-out evaluation row is missing or ambiguous")
    row = matches[0]
    if row["partition_sha256"] != selection["partition_sha256"]:
        raise ValueError("held-out selection/evaluation partition SHA mismatch")
    payload = dict(selection)
    payload.update(
        schema="night17e-strict-loso-heldout-evaluation-v1",
        heldout_evaluation_file=str(args.heldout_evaluation),
        heldout_evaluation_sha256=sha256_file(Path(args.heldout_evaluation)),
        heldout_evaluation_files_read=1,
        metrics={
            key: row[key]
            for key in (
                "absolute_ari",
                "absolute_nmi",
                "ami",
                "fmi",
                "morans_i_macro",
                "gearys_c_macro",
                "neighbor_agreement",
                "cluster_sizes_full",
                "min_cluster_size_full",
                "changed_from_initial",
            )
        },
    )
    Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    fit_parser = sub.add_parser("fit")
    fit_parser.add_argument("--training-evaluation", action="append", required=True)
    fit_parser.add_argument("--expected-training-lane", action="append", required=True)
    fit_parser.add_argument("--heldout-lane", required=True)
    fit_parser.add_argument("--output", required=True)
    fit_parser.set_defaults(function=fit)
    apply_parser = sub.add_parser("apply")
    apply_parser.add_argument("--fit-json", required=True)
    apply_parser.add_argument("--producer-json", required=True)
    apply_parser.add_argument("--heldout-lane", required=True)
    apply_parser.add_argument("--output", required=True)
    apply_parser.set_defaults(function=apply)
    eval_parser = sub.add_parser("evaluate")
    eval_parser.add_argument("--selection-json", required=True)
    eval_parser.add_argument("--heldout-evaluation", required=True)
    eval_parser.add_argument("--output", required=True)
    eval_parser.set_defaults(function=evaluate)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
