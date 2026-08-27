"""Mechanically apply the frozen Night-23A Stage-A identifiability gate."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


MODEL = "logistic_probability"
CONTROLS = ["spatial_affinity_score", "arise_like_intersection_score", "shuffled_teacher_probability"]


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True)
    parser.add_argument("--evaluation-dir", required=True)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--teacher-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    contract = json.loads(Path(args.contract).read_text(encoding="utf-8"))
    gate = contract["stage_a_gate"]
    main_lanes = gate["main_lanes"]
    evaluation_dir, prediction_dir, teacher_dir = map(Path, [args.evaluation_dir, args.prediction_dir, args.teacher_dir])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    pooled_target, pooled_scores = [], {MODEL: [], **{name: [] for name in CONTROLS}}
    per_lane = {}
    for lane in contract["lanes"]:
        current = read_csv(evaluation_dir / f"{lane}.csv")
        rows.extend(current)
        per_lane[lane] = {row["model"]: float(row["auroc"]) for row in current}
        if lane in main_lanes:
            with np.load(prediction_dir / f"{lane}.npz", allow_pickle=False) as prediction:
                for name in pooled_scores:
                    pooled_scores[name].append(np.asarray(prediction[name], dtype=np.float64))
            with np.load(teacher_dir / f"{lane}.npz", allow_pickle=False) as teacher:
                pooled_target.append(np.asarray(teacher["relation"], dtype=np.uint8))
    pooled_target_value = np.concatenate(pooled_target)
    pooled = {}
    for name, values in pooled_scores.items():
        score = np.concatenate(values)
        pooled[name] = {
            "auroc": float(roc_auc_score(pooled_target_value, score)),
            "auprc": float(average_precision_score(pooled_target_value, score)),
        }
    lane_threshold_pass = {
        lane: per_lane[lane][MODEL] > float(gate["minimum_main_lanes_auroc_strictly_above"])
        for lane in main_lanes
    }
    control_wins = {
        control: sum(per_lane[lane][MODEL] > per_lane[lane][control] for lane in main_lanes)
        for control in CONTROLS
    }
    stage_b_authorized = (
        sum(lane_threshold_pass.values()) >= int(gate["minimum_main_lane_count"])
        and pooled[MODEL]["auroc"] > float(gate["minimum_pooled_main_auroc_strictly_above"])
        and all(value >= int(gate["minimum_main_lanes_outperform_each_control"]) for value in control_wins.values())
        and all(pooled[MODEL]["auroc"] > pooled[control]["auroc"] for control in CONTROLS)
    )
    with (output_dir / "edge_transfer_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    decision = {
        "schema": "night23a-stage-a-gate-v1",
        "primary_model": MODEL,
        "main_lanes": main_lanes,
        "lane_auroc": {lane: per_lane[lane][MODEL] for lane in main_lanes},
        "lane_threshold_pass": lane_threshold_pass,
        "lane_threshold_pass_count": int(sum(lane_threshold_pass.values())),
        "pooled_main": pooled,
        "control_lane_win_counts": control_wins,
        "stage_b_authorized": bool(stage_b_authorized),
        "classification_if_stopped": "NO_CROSS_STUDY_EDGE_IDENTIFIABILITY",
        "mlp_is_diagnostic_and_cannot_rescue_primary_gate": True,
        "benchmark_reference_labels_read": 0,
        "heldout_teacher_used_only_after_prediction_lock": True,
    }
    (output_dir / "stage_a_decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
