#!/usr/bin/env python3
"""Replay frozen Night-16A cross-study calibrator selections.

This process opens only the redacted descriptor table and the frozen JSON
checkpoint.  It reconstructs the exact robust descriptor transform, restores
the recorded ridge coefficients, and verifies that every held-out lane selects
the registered candidate.  Public reference arrays and metric tables are not
opened.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from scripts.night16a.night16a_crossfit_calibrator import FAMILY, STUDY, robust_by_lane


FORBIDDEN = {
    "absolute_ari", "absolute_nmi", "ami", "fmi", "morans_i", "gearys_c", "score_target"
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--descriptors", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()

    registry = json.loads((args.frozen / "frozen_crossfit_registry.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(args.descriptors)
    forbidden = [
        column for column in frame
        if column.lower() in FORBIDDEN
        or any(token in column.lower() for token in ("label", "truth", "reference_assignment"))
    ]
    if forbidden:
        raise RuntimeError(f"descriptor firewall violation: {forbidden}")

    frame["study"] = frame["lane"].map(STUDY)
    frame["family"] = frame["lane"].map(FAMILY)
    frame = frame[
        frame["selector_min_cluster_relative_to_equal"].astype(float)
        >= float(registry["minimum_relative_cluster"])
    ].copy().reset_index(drop=True)
    if registry["feature_profile"] == "compact":
        numeric = [column for column in frame if column.startswith("selector_")]
    else:
        numeric = [
            column for column in frame
            if column.startswith(("selector_", "calibration_stat__", "calibration_constant__"))
        ]
    if numeric != registry["descriptor_features"]:
        raise RuntimeError("descriptor feature order differs from frozen checkpoint")
    x_numeric = robust_by_lane(frame, numeric)
    start = frame["start_name"].astype(str)
    start_classes = registry["start_classes"]
    x_start = np.column_stack([start.str.startswith(name + "::s").astype(float) for name in start_classes])
    x = np.concatenate((x_numeric, x_start), axis=1)

    rows = []
    for held_out, checkpoint in registry["checkpoints"].items():
        coef = np.asarray(checkpoint["coef"], dtype=np.float64)
        intercept = float(checkpoint["intercept"])
        test_indices = frame.index[frame["study"] == held_out].to_numpy()
        prediction = x[test_indices] @ coef + intercept
        held_lanes = frame.loc[test_indices, "lane"].unique()
        for lane in held_lanes:
            mask = frame.loc[test_indices, "lane"].to_numpy() == lane
            lane_indices = test_indices[mask]
            local = prediction[mask]
            selected = frame.loc[int(lane_indices[int(np.argmax(local))]), "candidate_key"]
            expected = registry["lanes"][lane]["candidate_key"]
            rows.append({
                "lane": lane,
                "held_out_study": held_out,
                "selected_candidate": selected,
                "expected_candidate": expected,
                "selection_exact": bool(selected == expected),
            })

    result = {
        "status": "NIGHT16A_CALIBRATOR_CHECKPOINT_REPLAY_PASS"
        if all(row["selection_exact"] for row in rows)
        else "NIGHT16A_CALIBRATOR_CHECKPOINT_REPLAY_FAILED",
        "lanes": len(rows),
        "selection_exact_lanes": int(sum(row["selection_exact"] for row in rows)),
        "held_out_reference_arrays_opened": 0,
        "public_metric_columns_opened": 0,
        "checkpoint_coefficients_reloaded": len(registry["checkpoints"]),
        "wall_seconds": time.perf_counter() - started,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))
    if result["selection_exact_lanes"] != len(rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
