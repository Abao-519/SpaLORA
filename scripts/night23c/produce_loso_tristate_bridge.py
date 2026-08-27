"""Train one outer LOSO bridge and lock held-out partitions without reading held-out teacher/reference."""
from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch

from SpaLORA.night23a_xbed import EdgeModelConfig, array_sha, file_sha, load_csr, validate_teacher
from SpaLORA.night23c_tristate_bridge import (
    ARMS,
    BridgeConfig,
    bridge_config_dict,
    build_arm_weights,
    fit_outer_bridge,
    predict_outer_bridge,
    produce_partitions,
)


def load_feature(path: Path) -> tuple[dict, dict]:
    manifest = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if file_sha(path) != manifest["artifact_sha256"] or manifest["teacher_files_read"] != 0:
        raise RuntimeError("feature authority failure")
    with np.load(path, allow_pickle=False) as z:
        value = {key: np.asarray(z[key]) for key in ("ids", "rows", "cols", "features", "feature_names")}
    if array_sha(value["features"]) != manifest["features_sha256"]:
        raise RuntimeError("feature array authority failure")
    return value, manifest


def load_source_teacher(path: Path, feature_manifest: dict) -> tuple[np.ndarray, dict]:
    manifest = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if file_sha(path) != manifest["artifact_sha256"] or manifest["feature_artifact_sha256"] != feature_manifest["artifact_sha256"]:
        raise RuntimeError("source teacher authority failure")
    with np.load(path, allow_pickle=False) as z:
        relation = np.asarray(z["relation"], dtype=np.uint8)
    if array_sha(relation) != manifest["relation_sha256"]:
        raise RuntimeError("source relation SHA mismatch")
    return relation, manifest


def make_config(contract: dict) -> BridgeConfig:
    model = contract["model"]
    return BridgeConfig(
        model=EdgeModelConfig(
            sample_per_class_per_study=int(model["sample_per_class_per_study"]),
            logistic_c=float(model["logistic_c"]), mlp_hidden=int(model["mlp_hidden"]),
            mlp_steps=int(model["mlp_steps"]), mlp_batch_per_study=int(model["mlp_batch_per_study"]),
            mlp_learning_rate=float(model["mlp_learning_rate"]), weight_decay=float(model["weight_decay"]),
        ),
        relation_scale=float(contract["consumer"]["relation_scale"]),
        calibration_method=str(contract["calibration"]["method"]),
        calibration_min_purity=float(contract["calibration"]["minimum_state_purity"]),
        calibration_min_state_fraction=float(contract["calibration"]["minimum_state_fraction"]),
        calibration_quantile_grid_step=float(contract["calibration"]["quantile_grid_step"]),
        uncalibrated_lower=float(contract["controls"]["uncalibrated_lower"]),
        uncalibrated_upper=float(contract["controls"]["uncalibrated_upper"]),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--contract", required=True)
    p.add_argument("--feature-dir", required=True)
    p.add_argument("--teacher-dir", required=True)
    p.add_argument("--heldout", required=True)
    p.add_argument("--carrier", required=True)
    p.add_argument("--core-source", required=True)
    p.add_argument("--output-prefix", required=True)
    args = p.parse_args()
    started = time.perf_counter()
    contract_path = Path(args.contract)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    primary = list(contract["primary_lanes"])
    if args.heldout not in primary:
        raise RuntimeError("heldout must be a primary lane")
    sources = [lane for lane in primary if lane != args.heldout]
    if len(sources) != 2:
        raise RuntimeError("outer LOSO source count mismatch")
    feature_dir, teacher_dir = Path(args.feature_dir), Path(args.teacher_dir)
    heldout_path = feature_dir / f"{args.heldout}.npz"
    heldout, heldout_manifest = load_feature(heldout_path)
    source, source_authority = [], {}
    for lane in sources:
        feature_path, teacher_path = feature_dir / f"{lane}.npz", teacher_dir / f"{lane}.npz"
        feature, feature_manifest = load_feature(feature_path)
        target, teacher_manifest = load_source_teacher(teacher_path, feature_manifest)
        if len(target) != len(feature["features"]):
            raise RuntimeError("source feature/relation length mismatch")
        source.append({"lane": lane, "features": feature["features"], "target": target})
        source_authority[lane] = {"feature_sha256": file_sha(feature_path), "teacher_sha256": file_sha(teacher_path),
                                  "relation_sha256": teacher_manifest["relation_sha256"]}
    config = make_config(contract)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bridge = fit_outer_bridge(source, config, device=device, shuffled=False)
    shuffled_bridge = fit_outer_bridge(source, config, device=device, shuffled=True)
    prediction = predict_outer_bridge(bridge, heldout["features"])
    shuffled_prediction = predict_outer_bridge(shuffled_bridge, heldout["features"])
    prefix = Path(args.output_prefix); prefix.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path, prediction_path = prefix.with_suffix(".pt"), prefix.with_name(prefix.name + "_prediction.npz")
    bank_path, manifest_path = prefix.with_name(prefix.name + "_partitions.npz"), prefix.with_suffix(".json")
    checkpoint = {
        "schema": "night23c-loso-tristate-checkpoint-v1", "heldout": args.heldout, "source_lanes": sources,
        "bridge_config": bridge_config_dict(config), "bridge": bridge, "shuffled_bridge": shuffled_bridge,
        "contract_sha256": file_sha(contract_path), "core_source_sha256": file_sha(args.core_source),
    }
    torch.save(checkpoint, checkpoint_path)
    reloaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    reloaded_prediction = predict_outer_bridge(reloaded["bridge"], heldout["features"])
    reloaded_shuffled = predict_outer_bridge(reloaded["shuffled_bridge"], heldout["features"])
    for name in prediction:
        if not np.array_equal(np.asarray(prediction[name]), np.asarray(reloaded_prediction[name])):
            raise RuntimeError(f"checkpoint reload prediction mismatch: {name}")
    for name in shuffled_prediction:
        if not np.array_equal(np.asarray(shuffled_prediction[name]), np.asarray(reloaded_shuffled[name])):
            raise RuntimeError(f"shuffled checkpoint reload mismatch: {name}")
    np.savez_compressed(prediction_path, ids=heldout["ids"], rows=heldout["rows"], cols=heldout["cols"],
                        **prediction, **{f"shuffled_{key}": value for key, value in shuffled_prediction.items()})
    with np.load(args.carrier, allow_pickle=False) as z:
        carrier_ids = np.asarray(z["ids"]).astype("U")
        retained = np.asarray(z["retained"], dtype=np.float64)
        spatial = load_csr(z, "graph0")
    if not np.array_equal(np.asarray(heldout["ids"]).astype("U"), carrier_ids):
        raise RuntimeError("heldout feature/carrier ID mismatch")
    weights = build_arm_weights(heldout["features"], prediction, shuffled_prediction, config)
    partitions, weight_ledger = produce_partitions(retained, heldout["rows"], heldout["cols"], int(contract["lanes"][args.heldout]["k"]), weights)
    ledger = []
    for index, (arm, partition) in enumerate(zip(ARMS, partitions)):
        structure = validate_teacher(partition, int(contract["lanes"][args.heldout]["k"]), spatial)
        ledger.append({"candidate_id": arm, "partition_index": index, "partition_sha256": array_sha(partition), **structure, **weight_ledger[index]})
    np.savez_compressed(bank_path, ids=heldout["ids"], candidate_ids=np.asarray(ARMS, dtype="U"), partitions=partitions,
                        rows=heldout["rows"], cols=heldout["cols"])
    manifest = {
        "schema": "night23c-loso-tristate-partition-bank-v1", "lane": args.heldout,
        "role": "PRIMARY", "k": int(contract["lanes"][args.heldout]["k"]), "n": len(heldout["ids"]),
        "feature_shape": list(heldout["features"].shape), "carrier_shape": list(retained.shape), "union_edge_count": len(heldout["rows"]),
        "source_lanes": sources, "source_authority": source_authority, "heldout_feature_sha256": file_sha(heldout_path),
        "heldout_teacher_path_opened": False, "heldout_teacher_arrays_read": 0, "benchmark_reference_labels_read": 0,
        "labels_read": 0,
        "checkpoint_sha256": file_sha(checkpoint_path), "prediction_sha256": file_sha(prediction_path),
        "bank_sha256": file_sha(bank_path), "partitions_sha256": array_sha(partitions), "ids_sha256": array_sha(np.asarray(heldout["ids"]).astype("U")),
        "rows_sha256": array_sha(heldout["rows"]), "cols_sha256": array_sha(heldout["cols"]), "carrier_sha256": file_sha(args.carrier),
        "contract_sha256": file_sha(contract_path), "core_source_sha256": file_sha(args.core_source), "candidate_ids": list(ARMS), "rows": ledger,
        "calibration": {"primary": bridge["thresholds"], "shuffled": shuffled_bridge["thresholds"]},
        "inner_crossfit": {"primary": bridge["inner_crossfit"], "shuffled": shuffled_bridge["inner_crossfit"]},
        "same_process_checkpoint_reload": "PASS", "train_device": device,
        "mlp_parameter_l1_change": bridge["mlp"]["parameter_l1_change"], "wall_seconds": time.perf_counter() - started,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"lane": args.heldout, "sources": sources, "candidate_count": len(ARMS), "thresholds": bridge["thresholds"],
                      "bank_sha256": manifest["bank_sha256"], "wall_seconds": manifest["wall_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
