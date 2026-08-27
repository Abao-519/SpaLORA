"""Open held-out teacher only after Night-23C partitions and replay are locked."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from SpaLORA.night23a_xbed import array_sha, file_sha


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prediction", required=True); p.add_argument("--manifest", required=True)
    p.add_argument("--replay", required=True); p.add_argument("--teacher", required=True); p.add_argument("--output", required=True)
    args = p.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8")); replay = json.loads(Path(args.replay).read_text(encoding="utf-8"))
    if replay["status"] != "PASS" or replay["labels_read"] != 0 or file_sha(args.prediction) != manifest["prediction_sha256"]:
        raise RuntimeError("relation evaluation lock/replay failure")
    teacher_manifest = json.loads(Path(args.teacher).with_suffix(".json").read_text(encoding="utf-8"))
    if file_sha(args.teacher) != teacher_manifest["artifact_sha256"]: raise RuntimeError("heldout teacher authority mismatch")
    with np.load(args.teacher, allow_pickle=False) as z: target = np.asarray(z["relation"], dtype=np.uint8)
    if array_sha(target) != teacher_manifest["relation_sha256"]: raise RuntimeError("heldout relation array mismatch")
    with np.load(args.prediction, allow_pickle=False) as z:
        values = {key: np.asarray(z[key]) for key in z.files}
    rows = []
    for name in ("logistic_probability", "mlp_probability", "calibrated_probability", "shuffled_calibrated_probability"):
        score = np.asarray(values[name], dtype=np.float64)
        rows.append({"model": name, "auroc": float(roc_auc_score(target, score)), "auprc": float(average_precision_score(target, score)),
                     "brier": float(brier_score_loss(target, score))})
    within, boundary, unknown = (np.asarray(values[key], dtype=bool) for key in ("within", "boundary", "unknown"))
    if np.any(within & boundary) or not np.all(within | boundary | unknown): raise RuntimeError("tri-state relation artifact invalid")
    tri = {"within_count": int(within.sum()), "boundary_count": int(boundary.sum()), "unknown_count": int(unknown.sum()),
           "within_fraction": float(within.mean()), "boundary_fraction": float(boundary.mean()), "unknown_fraction": float(unknown.mean()),
           "within_purity": float(np.mean(target[within] == 1)) if np.any(within) else None,
           "boundary_purity": float(np.mean(target[boundary] == 0)) if np.any(boundary) else None,
           "covered_accuracy": float(np.mean(np.concatenate([target[within] == 1, target[boundary] == 0]))) if np.any(within | boundary) else None}
    output = {"schema": "night23c-heldout-relation-diagnostic-v1", "lane": manifest["lane"], "partition_and_replay_locked_before_teacher_read": True,
              "benchmark_reference_labels_read": 0, "teacher_file_sha256": file_sha(args.teacher), "relation_sha256": array_sha(target),
              "metrics": rows, "tri_state": tri}
    Path(args.output).write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__": main()
