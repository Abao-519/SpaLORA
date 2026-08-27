"""Fresh-process exact replay for a locked LOSO edge checkpoint/prediction artifact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from SpaLORA.night23a_xbed import array_sha, file_sha, predict_logistic, predict_mlp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    feature_path, prediction_path, checkpoint_path = map(Path, [args.features, args.prediction, args.checkpoint])
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if file_sha(prediction_path) != manifest["prediction_artifact_sha256"]:
        raise RuntimeError("prediction artifact SHA mismatch")
    if file_sha(checkpoint_path) != manifest["checkpoint_sha256"]:
        raise RuntimeError("checkpoint SHA mismatch")
    if file_sha(feature_path) != manifest["heldout_feature_sha256"]:
        raise RuntimeError("heldout feature SHA mismatch")
    with np.load(feature_path, allow_pickle=False) as archive:
        features = np.asarray(archive["features"], dtype=np.float32)
        rows = np.asarray(archive["rows"], dtype=np.int32)
        cols = np.asarray(archive["cols"], dtype=np.int32)
    with np.load(prediction_path, allow_pickle=False) as archive:
        stored = {key: np.asarray(archive[key]) for key in archive.files}
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    observed = {
        "logistic_probability": predict_logistic(checkpoint["logistic"], features),
        "mlp_probability": predict_mlp(checkpoint["mlp"], features, device="cpu"),
        "shuffled_teacher_probability": predict_logistic(checkpoint["shuffled_logistic"], features),
    }
    for name, value in observed.items():
        if not np.array_equal(value, stored[name]):
            raise RuntimeError(f"fresh prediction replay mismatch: {name}")
        if array_sha(value) != manifest["prediction_array_sha256"][name]:
            raise RuntimeError(f"fresh prediction SHA mismatch: {name}")
    if not np.array_equal(rows, stored["rows"]) or not np.array_equal(cols, stored["cols"]):
        raise RuntimeError("fresh edge replay mismatch")
    output = {
        "schema": "night23a-edge-fresh-replay-v1",
        "status": "PASS",
        "heldout_lane": manifest["heldout_lane"],
        "prediction_artifact_sha256": file_sha(prediction_path),
        "checkpoint_sha256": file_sha(checkpoint_path),
        "replayed_prediction_arrays": sorted(observed),
        "heldout_teacher_files_read": 0,
        "benchmark_reference_labels_read": 0,
    }
    Path(args.output).write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
