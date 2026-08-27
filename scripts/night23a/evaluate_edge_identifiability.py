"""Open held-out teacher relations only after edge predictions/checkpoint/replay are locked."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from SpaLORA.night23a_xbed import array_sha, file_sha


def ece(y: np.ndarray, score: np.ndarray, bins: int = 10) -> float:
    total = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        mask = (score >= edges[index]) & (score < edges[index + 1] if index + 1 < bins else score <= 1.0)
        if mask.any():
            total += mask.mean() * abs(float(score[mask].mean()) - float(y[mask].mean()))
    return float(total)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prediction_path, teacher_path = Path(args.prediction), Path(args.teacher)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    replay = json.loads(Path(args.replay).read_text(encoding="utf-8"))
    if replay["status"] != "PASS" or replay["heldout_teacher_files_read"] != 0:
        raise RuntimeError("fresh replay authority failure")
    if file_sha(prediction_path) != manifest["prediction_artifact_sha256"]:
        raise RuntimeError("locked prediction SHA mismatch")
    teacher_manifest = json.loads(teacher_path.with_suffix(".json").read_text(encoding="utf-8"))
    with np.load(prediction_path, allow_pickle=False) as prediction:
        scores = {key: np.asarray(prediction[key], dtype=np.float64) for key in prediction.files if key.endswith("score") or key.endswith("probability")}
        rows, cols = np.asarray(prediction["rows"]), np.asarray(prediction["cols"])
    # First held-out teacher read in this evaluator process.
    with np.load(teacher_path, allow_pickle=False) as teacher:
        target = np.asarray(teacher["relation"], dtype=np.uint8)
    if file_sha(teacher_path) != teacher_manifest["artifact_sha256"] or array_sha(target) != teacher_manifest["relation_sha256"]:
        raise RuntimeError("held-out teacher authority mismatch")
    if len(target) != len(rows) or array_sha(rows) != manifest["rows_sha256"] or array_sha(cols) != manifest["cols_sha256"]:
        raise RuntimeError("held-out target/edge mismatch")
    rows_out = []
    for model, score in sorted(scores.items()):
        if len(score) != len(target) or not np.all(np.isfinite(score)):
            raise RuntimeError(f"invalid score: {model}")
        rows_out.append(
            {
                "heldout_lane": manifest["heldout_lane"],
                "role": manifest["heldout_role"],
                "model": model,
                "auroc": roc_auc_score(target, score),
                "auprc": average_precision_score(target, score),
                "brier": brier_score_loss(target, score),
                "ece10": ece(target, score),
                "edge_count": len(target),
                "positive_fraction": float(target.mean()),
                "prediction_sha256": array_sha(score),
                "teacher_relation_sha256": array_sha(target),
            }
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_out[0]))
        writer.writeheader()
        writer.writerows(rows_out)
    audit = {
        "schema": "night23a-edge-identifiability-evaluation-v1",
        "heldout_lane": manifest["heldout_lane"],
        "prediction_locked_before_teacher_read": True,
        "fresh_process_replay_passed_before_teacher_read": True,
        "teacher_file_sha256": file_sha(teacher_path),
        "teacher_relation_sha256": array_sha(target),
        "benchmark_reference_labels_read": 0,
        "heldout_teacher_read_count": 1,
        "output_sha256": file_sha(output),
    }
    output.with_suffix(".json").write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
