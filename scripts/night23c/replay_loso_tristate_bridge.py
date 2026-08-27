"""Fresh-process exact checkpoint, prediction and partition replay for Night-23C."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from SpaLORA.night23a_xbed import array_sha, file_sha
from SpaLORA.night23c_tristate_bridge import ARMS, BridgeConfig, build_arm_weights, predict_outer_bridge, produce_partitions
from SpaLORA.night23a_xbed import EdgeModelConfig


def config_from_checkpoint(value: dict) -> BridgeConfig:
    model = value["model"]
    return BridgeConfig(model=EdgeModelConfig(**model), relation_scale=value["relation_scale"], calibration_method=value["calibration_method"],
                        calibration_min_purity=value["calibration_min_purity"],
                        calibration_min_state_fraction=value["calibration_min_state_fraction"],
                        calibration_quantile_grid_step=value["calibration_quantile_grid_step"],
                        uncalibrated_lower=value["uncalibrated_lower"], uncalibrated_upper=value["uncalibrated_upper"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--feature", required=True); p.add_argument("--carrier", required=True)
    p.add_argument("--checkpoint", required=True); p.add_argument("--prediction", required=True)
    p.add_argument("--bank", required=True); p.add_argument("--manifest", required=True); p.add_argument("--output", required=True)
    args = p.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if file_sha(args.checkpoint) != manifest["checkpoint_sha256"] or file_sha(args.prediction) != manifest["prediction_sha256"] or file_sha(args.bank) != manifest["bank_sha256"]:
        raise RuntimeError("replay file authority mismatch")
    with np.load(args.feature, allow_pickle=False) as z:
        ids = np.asarray(z["ids"]).astype("U"); rows = np.asarray(z["rows"], dtype=np.int32); cols = np.asarray(z["cols"], dtype=np.int32); features = np.asarray(z["features"], dtype=np.float32)
    with np.load(args.carrier, allow_pickle=False) as z:
        carrier_ids = np.asarray(z["ids"]).astype("U"); retained = np.asarray(z["retained"], dtype=np.float64)
    if not np.array_equal(ids, carrier_ids): raise RuntimeError("replay carrier ID mismatch")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = config_from_checkpoint(checkpoint["bridge_config"])
    prediction = predict_outer_bridge(checkpoint["bridge"], features)
    shuffled = predict_outer_bridge(checkpoint["shuffled_bridge"], features)
    with np.load(args.prediction, allow_pickle=False) as z:
        if not np.array_equal(ids, np.asarray(z["ids"]).astype("U")) or not np.array_equal(rows, z["rows"]) or not np.array_equal(cols, z["cols"]):
            raise RuntimeError("prediction authority mismatch")
        for key, value in prediction.items():
            if not np.array_equal(np.asarray(value), np.asarray(z[key])): raise RuntimeError(f"prediction replay mismatch: {key}")
        for key, value in shuffled.items():
            if not np.array_equal(np.asarray(value), np.asarray(z[f"shuffled_{key}"])): raise RuntimeError(f"shuffled replay mismatch: {key}")
    weights = build_arm_weights(features, prediction, shuffled, config)
    partitions, _ = produce_partitions(retained, rows, cols, int(manifest["k"]), weights)
    with np.load(args.bank, allow_pickle=False) as z:
        stored_ids = np.asarray(z["ids"]).astype("U"); candidate_ids = np.asarray(z["candidate_ids"]).astype("U"); stored = np.asarray(z["partitions"], dtype=np.int32)
    if not np.array_equal(ids, stored_ids) or candidate_ids.tolist() != list(ARMS) or not np.array_equal(partitions, stored):
        raise RuntimeError("fresh-process partition replay mismatch")
    output = {"schema": "night23c-fresh-process-replay-v1", "status": "PASS", "lane": manifest["lane"],
              "candidate_count": len(ARMS), "checkpoint_prediction_exact": True, "partition_bank_exact": True,
              "partitions_sha256": array_sha(partitions), "bank_sha256": file_sha(args.bank), "labels_read": 0, "teacher_files_read": 0}
    Path(args.output).write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__": main()
