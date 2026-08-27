"""Train shared source-study edge models and predict one held-out study without opening its teacher."""
from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch

from SpaLORA.night23a_xbed import (
    EdgeModelConfig,
    array_sha,
    file_sha,
    fit_logistic,
    fit_mlp,
    predict_logistic,
    predict_mlp,
    source_lanes,
)


def load_feature(path: Path) -> tuple[dict, dict]:
    manifest = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if file_sha(path) != manifest["artifact_sha256"] or manifest["teacher_files_read"] != 0:
        raise RuntimeError("feature artifact authority failure")
    with np.load(path, allow_pickle=False) as archive:
        value = {key: np.asarray(archive[key]) for key in archive.files}
    if array_sha(value["features"]) != manifest["features_sha256"]:
        raise RuntimeError("feature array SHA mismatch")
    return value, manifest


def load_teacher(path: Path, feature_manifest: dict) -> tuple[np.ndarray, dict]:
    manifest = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if file_sha(path) != manifest["artifact_sha256"]:
        raise RuntimeError("teacher artifact authority failure")
    if manifest["feature_artifact_sha256"] != feature_manifest["artifact_sha256"]:
        raise RuntimeError("teacher/feature artifact mismatch")
    with np.load(path, allow_pickle=False) as archive:
        relation = np.asarray(archive["relation"], dtype=np.uint8)
    if array_sha(relation) != manifest["relation_sha256"]:
        raise RuntimeError("teacher relation SHA mismatch")
    return relation, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--teacher-dir", required=True)
    parser.add_argument("--heldout", required=True)
    parser.add_argument("--output", required=True, help="Output prefix without suffix")
    args = parser.parse_args()
    started = time.perf_counter()
    contract_path = Path(args.contract)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    all_lanes = list(contract["lanes"])
    sources = source_lanes(all_lanes, args.heldout)
    feature_dir, teacher_dir = Path(args.feature_dir), Path(args.teacher_dir)
    heldout_feature_path = feature_dir / f"{args.heldout}.npz"
    heldout, heldout_manifest = load_feature(heldout_feature_path)
    source = []
    source_manifest = {}
    for lane in sources:
        feature_path = feature_dir / f"{lane}.npz"
        teacher_path = teacher_dir / f"{lane}.npz"
        feature, feature_manifest = load_feature(feature_path)
        relation, relation_manifest = load_teacher(teacher_path, feature_manifest)
        if len(relation) != len(feature["features"]):
            raise RuntimeError("source feature/teacher length mismatch")
        source.append({"lane": lane, "features": feature["features"], "target": relation})
        source_manifest[lane] = {
            "feature_sha256": file_sha(feature_path),
            "teacher_sha256": file_sha(teacher_path),
            "relation_sha256": relation_manifest["relation_sha256"],
            "same_fraction": relation_manifest["same_edge_fraction"],
        }
    model_config = EdgeModelConfig(
        sample_per_class_per_study=contract["model"]["sample_per_class_per_study"],
        logistic_c=contract["model"]["logistic_c"],
        mlp_hidden=contract["model"]["mlp_hidden"],
        mlp_steps=contract["model"]["mlp_steps"],
        mlp_batch_per_study=contract["model"]["mlp_batch_per_study"],
        mlp_learning_rate=contract["model"]["mlp_learning_rate"],
        weight_decay=contract["model"]["weight_decay"],
    )
    logistic = fit_logistic(source, model_config, shuffled=False)
    shuffled = fit_logistic(source, model_config, shuffled=True)
    train_device = "cuda" if torch.cuda.is_available() else "cpu"
    mlp = fit_mlp(source, model_config, device=train_device)
    features = heldout["features"]
    probabilities = {
        "logistic_probability": predict_logistic(logistic, features),
        "mlp_probability": predict_mlp(mlp, features, device="cpu"),
        "shuffled_teacher_probability": predict_logistic(shuffled, features),
        "spatial_affinity_score": np.asarray(heldout["spatial_score"], dtype=np.float64),
        "arise_like_intersection_score": np.asarray(heldout["intersection_score"], dtype=np.float64),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output.with_suffix(".pt")
    prediction_path = output.with_suffix(".npz")
    checkpoint = {
        "schema": "night23a-loso-edge-checkpoint-v1",
        "heldout": args.heldout,
        "source_lanes": sources,
        "feature_names": heldout["feature_names"].tolist(),
        "contract_sha256": file_sha(contract_path),
        "model_config": model_config.__dict__,
        "logistic": logistic,
        "shuffled_logistic": shuffled,
        "mlp": mlp,
    }
    torch.save(checkpoint, checkpoint_path)
    reloaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    reload_logistic = predict_logistic(reloaded["logistic"], features)
    reload_shuffled = predict_logistic(reloaded["shuffled_logistic"], features)
    reload_mlp = predict_mlp(reloaded["mlp"], features, device="cpu")
    if not (
        np.array_equal(probabilities["logistic_probability"], reload_logistic)
        and np.array_equal(probabilities["shuffled_teacher_probability"], reload_shuffled)
        and np.array_equal(probabilities["mlp_probability"], reload_mlp)
    ):
        raise RuntimeError("same-process strict checkpoint reload failed")
    np.savez_compressed(
        prediction_path,
        ids=heldout["ids"],
        rows=heldout["rows"],
        cols=heldout["cols"],
        **probabilities,
    )
    manifest = {
        "schema": "night23a-loso-edge-prediction-v1",
        "heldout_lane": args.heldout,
        "heldout_role": contract["lanes"][args.heldout]["role"],
        "source_lanes": sources,
        "source_authority": source_manifest,
        "heldout_feature_path": str(heldout_feature_path),
        "heldout_feature_sha256": file_sha(heldout_feature_path),
        "heldout_teacher_path_opened": False,
        "heldout_teacher_arrays_read": 0,
        "benchmark_reference_labels_read": 0,
        "prediction_count": len(features),
        "prediction_array_sha256": {name: array_sha(value) for name, value in probabilities.items()},
        "rows_sha256": array_sha(heldout["rows"]),
        "cols_sha256": array_sha(heldout["cols"]),
        "checkpoint_sha256": file_sha(checkpoint_path),
        "prediction_artifact_sha256": file_sha(prediction_path),
        "same_process_strict_reload": "PASS",
        "mlp_parameter_l1_change": mlp["parameter_l1_change"],
        "mlp_loss_ledger": mlp["loss_ledger"],
        "train_device": train_device,
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
