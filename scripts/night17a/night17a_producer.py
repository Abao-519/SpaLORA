#!/usr/bin/env python3
"""Label-free real-lane producer for Night-17A CEUP-P0."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import time
from pathlib import Path
from typing import Dict, Mapping

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from SpaLORA.night17a_ceup import (
    ARM_IDS,
    MaskedLinearPredictor,
    canonical_undirected_graph,
    crossfit_directed_utilities,
    deterministic_node_folds,
    fused_molecular_representation,
    kmeans_start,
    load_csr,
    load_predictor,
    make_arm_evidence,
    normalize_pairwise_weights,
    normalize_utility_tensor,
    prototype_unary,
    sha256_array,
    signed_energy,
    signed_icm_partition,
    static_relation_evidence,
    standardize,
    symmetric_signed_evidence,
    train_masked_predictor,
)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_config(path: Path) -> Dict[str, object]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema",
        "seed",
        "graph_prefix",
        "n_masks",
        "n_node_folds",
        "training_steps",
        "learning_rate",
        "weight_decay",
        "pairwise_strength",
        "repulsion_strength",
        "trust_strength",
        "max_sweeps",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"config missing {missing}")
    if config["schema"] != "night17a-ceup-p0-frozen-v1":
        raise ValueError("unexpected Night-17A config schema")
    return config


def load_numeric_carrier(path: Path, graph_prefix: str) -> Dict[str, object]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"ids", "view1", "view2", f"{graph_prefix}__data", f"{graph_prefix}__indices", f"{graph_prefix}__indptr", f"{graph_prefix}__shape"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"carrier missing {missing}")
        ids = np.asarray(archive["ids"])
        view1 = np.asarray(archive["view1"], dtype=np.float32)
        view2 = np.asarray(archive["view2"], dtype=np.float32)
        graph = load_csr(archive, graph_prefix)
        accessed = sorted(required)
    if len(set(ids.tolist())) != ids.size:
        raise ValueError("carrier IDs are not unique")
    if view1.shape[0] != ids.size or view2.shape[0] != ids.size or graph.shape != (ids.size, ids.size):
        raise ValueError("carrier observation mismatch")
    if not np.isfinite(view1).all() or not np.isfinite(view2).all():
        raise ValueError("non-finite carrier view")
    return {"ids": ids, "view1": view1, "view2": view2, "graph": graph, "accessed": accessed}


def fit_models(view1, view2, graph, node_folds, config, device):
    common = dict(
        n_masks=int(config["n_masks"]),
        steps=int(config["training_steps"]),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        device=device,
    )
    seed = int(config["seed"])
    specs = {
        "cross_v1_from_v2": (view1, view2, seed + 101),
        "cross_v2_from_v1": (view2, view1, seed + 211),
        "single_v1_from_v1": (view1, view1, seed + 307),
        "single_v2_from_v2": (view2, view2, seed + 401),
    }
    fits = {}
    for name, (target, source, model_seed) in specs.items():
        fits[name] = []
        for fold in range(int(config["n_node_folds"])):
            train_rows = np.flatnonzero(node_folds != fold)
            fits[name].append(
                train_masked_predictor(target, source, graph, seed=model_seed, train_rows=train_rows, **common)
            )
    return fits


def checkpoint_payload(fits, view1_dim, view2_dim, config):
    return {
        "schema": "night17a-ceup-p0-checkpoint-v1",
        "config": config,
        "dimensions": {"view1": int(view1_dim), "view2": int(view2_dim)},
        "models": {
            name: [
                {
                    "state_dict": dict(fit.state_dict),
                    "masks": [np.asarray(x, dtype=np.int64) for x in fit.masks],
                    "initial_loss": fit.initial_loss,
                    "final_loss": fit.final_loss,
                    "parameter_update_norm": fit.parameter_update_norm,
                }
                for fit in fold_fits
            ]
            for name, fold_fits in fits.items()
        },
    }


def strict_reload_checkpoint(path: Path, view1_dim: int, view2_dim: int, device: str):
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch 2.0 compatibility
        payload = torch.load(path, map_location=device)
    if payload.get("schema") != "night17a-ceup-p0-checkpoint-v1":
        raise ValueError("checkpoint schema mismatch")
    if payload["dimensions"] != {"view1": int(view1_dim), "view2": int(view2_dim)}:
        raise ValueError("checkpoint dimension mismatch")
    dims = {
        "cross_v1_from_v2": (view1_dim, view2_dim),
        "cross_v2_from_v1": (view2_dim, view1_dim),
        "single_v1_from_v1": (view1_dim, view1_dim),
        "single_v2_from_v2": (view2_dim, view2_dim),
    }
    models = {}
    metadata = {}
    for name, (target_dim, source_dim) in dims.items():
        models[name] = []
        metadata[name] = []
        for record in payload["models"][name]:
            models[name].append(load_predictor(target_dim, source_dim, record["state_dict"], device=device))
            metadata[name].append({
                "masks": [np.asarray(x, dtype=np.int64) for x in record["masks"]],
                "initial_loss": float(record["initial_loss"]),
                "final_loss": float(record["final_loss"]),
                "parameter_update_norm": float(record["parameter_update_norm"]),
            })
    return payload, models, metadata


def run_pipeline(carrier_path: Path, config_path: Path, checkpoint_path: Path, k: int, lane: str, mode: str, device: str):
    started = time.time()
    config = read_config(config_path)
    numeric = load_numeric_carrier(carrier_path, str(config["graph_prefix"]))
    ids = numeric["ids"]
    view1, _, _ = standardize(numeric["view1"])
    view2, _, _ = standardize(numeric["view2"])
    edge_i, edge_j, edge_w, graph = canonical_undirected_graph(numeric["graph"])
    node_folds = deterministic_node_folds(ids, int(config["n_node_folds"]))

    if mode == "train":
        fits = fit_models(view1, view2, graph, node_folds, config, device)
        payload = checkpoint_payload(fits, view1.shape[1], view2.shape[1], config)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint_path.with_suffix(".tmp.pt")
        torch.save(payload, temporary)
        temporary.replace(checkpoint_path)
    payload, models, model_metadata = strict_reload_checkpoint(checkpoint_path, view1.shape[1], view2.shape[1], device)
    if payload["config"] != config:
        raise ValueError("checkpoint/config mismatch")
    if any(meta["parameter_update_norm"] <= 0.0 for folds in model_metadata.values() for meta in folds):
        raise RuntimeError("optimizer did not update every registered predictor")

    raw_v1 = crossfit_directed_utilities(
        models["cross_v1_from_v2"], [x["masks"] for x in model_metadata["cross_v1_from_v2"]], node_folds, view1, view2, graph,
        edge_i, edge_j, edge_w, device=device,
    )
    raw_v2 = crossfit_directed_utilities(
        models["cross_v2_from_v1"], [x["masks"] for x in model_metadata["cross_v2_from_v1"]], node_folds, view2, view1, graph,
        edge_i, edge_j, edge_w, device=device,
    )
    raw_four = np.concatenate([raw_v1, raw_v2], axis=1)
    utility_mean4, utility_uncertainty4, q_direction4, utility_scales4 = normalize_utility_tensor(raw_four)
    learned = symmetric_signed_evidence(q_direction4)
    q_no_uncertainty = np.tanh(utility_mean4)
    learned_no_uncertainty = symmetric_signed_evidence(q_no_uncertainty)

    raw_single1 = crossfit_directed_utilities(
        models["single_v1_from_v1"], [x["masks"] for x in model_metadata["single_v1_from_v1"]], node_folds, view1, view1, graph,
        edge_i, edge_j, edge_w, device=device,
    )
    raw_single2 = crossfit_directed_utilities(
        models["single_v2_from_v2"], [x["masks"] for x in model_metadata["single_v2_from_v2"]], node_folds, view2, view2, graph,
        edge_i, edge_j, edge_w, device=device,
    )
    single1 = symmetric_signed_evidence(normalize_utility_tensor(raw_single1)[2])
    single2 = symmetric_signed_evidence(normalize_utility_tensor(raw_single2)[2])
    static = static_relation_evidence(view1, view2, edge_i, edge_j)

    fused = fused_molecular_representation(view1, view2)
    start = kmeans_start(fused, k, seed=int(config["seed"]))
    unary = prototype_unary(fused, start, k)
    unary = unary + float(config["trust_strength"]) * (np.arange(k)[None, :] != start[:, None])
    edge_w_normalized = normalize_pairwise_weights(edge_w, edge_i, edge_j, ids.size)
    arm_partitions = []
    arm_diagnostics = []
    for arm in ARM_IDS:
        evidence = make_arm_evidence(
            arm, learned, learned_no_uncertainty, static, single1, single2, ids, edge_i, edge_j, edge_w,
        )
        attraction = float(config["pairwise_strength"]) * edge_w_normalized * evidence["q_positive"]
        repulsion = float(config["repulsion_strength"]) * edge_w_normalized * evidence["q_negative"]
        initial_energy = signed_energy(start, unary, edge_i, edge_j, attraction, repulsion)
        if arm == "NO_OP":
            partition = start.copy()
            diagnostic = {
                "energy_trace": [initial_energy, initial_energy],
                "accepted_per_sweep": [0],
                "changed_spots": 0,
                "cluster_sizes": np.bincount(start, minlength=k).astype(int).tolist(),
            }
        else:
            partition, diagnostic = signed_icm_partition(
                start, unary, edge_i, edge_j, attraction, repulsion,
                max_sweeps=int(config["max_sweeps"]),
            )
        diagnostic.update(
            {
                "arm": arm,
                "initial_energy": initial_energy,
                "final_energy": float(diagnostic["energy_trace"][-1]),
                "positive_mass": float(np.sum(edge_w * evidence["q_positive"])),
                "negative_mass": float(np.sum(edge_w * evidence["q_negative"])),
                "positive_coverage": float(np.mean(evidence["q_positive"] > 0)),
                "negative_coverage": float(np.mean(evidence["q_negative"] > 0)),
            }
        )
        arm_partitions.append(partition.astype(np.int32))
        arm_diagnostics.append(diagnostic)

    permuted = make_arm_evidence(
        "EDGE_PERMUTATION", learned, learned_no_uncertainty, static, single1, single2,
        ids, edge_i, edge_j, edge_w,
    )

    result = {
        "ids": ids,
        "edge_i": edge_i,
        "edge_j": edge_j,
        "edge_w": edge_w,
        "start_partition": start.astype(np.int32),
        "arm_ids": np.asarray(ARM_IDS),
        "arm_partitions": np.stack(arm_partitions),
        "utility_raw_four": raw_four.astype(np.float32),
        "utility_mean_four": utility_mean4.astype(np.float32),
        "utility_uncertainty_four": utility_uncertainty4.astype(np.float32),
        "utility_q_four": q_direction4.astype(np.float32),
        "utility_scales_four": utility_scales4.astype(np.float64),
        "node_folds": node_folds.astype(np.int8),
        "learned_q_positive": learned["q_positive"].astype(np.float32),
        "learned_q_negative": learned["q_negative"].astype(np.float32),
        "learned_discordance": learned["discordance"].astype(np.float32),
        "learned_relation": learned["relation"].astype(np.int8),
        "no_uncertainty_q_positive": learned_no_uncertainty["q_positive"].astype(np.float32),
        "no_uncertainty_q_negative": learned_no_uncertainty["q_negative"].astype(np.float32),
        "static_q_positive": static["q_positive"].astype(np.float32),
        "static_q_negative": static["q_negative"].astype(np.float32),
        "single1_q_positive": single1["q_positive"].astype(np.float32),
        "single1_q_negative": single1["q_negative"].astype(np.float32),
        "single2_q_positive": single2["q_positive"].astype(np.float32),
        "single2_q_negative": single2["q_negative"].astype(np.float32),
        "permuted_q_positive": permuted["q_positive"].astype(np.float32),
        "permuted_q_negative": permuted["q_negative"].astype(np.float32),
    }
    manifest = {
        "schema": "night17a-ceup-p0-producer-v1",
        "lane": lane,
        "k": int(k),
        "seed": int(config["seed"]),
        "mode": mode,
        "device": device,
        "carrier_path": str(carrier_path.resolve()),
        "carrier_sha256": file_sha256(carrier_path),
        "config_path": str(config_path.resolve()),
        "config_sha256": file_sha256(config_path),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "carrier_keys_accessed": numeric["accessed"],
        "labels_accessed": 0,
        "annotation_keys_accessed": 0,
        "n_observations": int(ids.size),
        "view1_shape": list(view1.shape),
        "view2_shape": list(view2.shape),
        "graph_shape": list(graph.shape),
        "graph_nnz": int(graph.nnz),
        "undirected_edges": int(edge_i.size),
        "full_degree_fixed_for_loo": True,
        "renormalization_after_edge_deletion": False,
        "four_directional_utilities_preserved": True,
        "checkpoint_strict_reload": "PASS",
        "optimizer_updates": {name: [meta["parameter_update_norm"] for meta in folds] for name, folds in model_metadata.items()},
        "training_losses": {name: [{"initial": meta["initial_loss"], "final": meta["final_loss"]} for meta in folds] for name, folds in model_metadata.items()},
        "node_crossfit_folds": int(config["n_node_folds"]),
        "node_crossfit_semantics": "receiver target loss excluded from its fold model; graph source features remain transductive",
        "explicit_start_trust_strength": float(config["trust_strength"]),
        "arm_diagnostics": arm_diagnostics,
        "ids_sha256": sha256_array(ids),
        "edge_sha256": sha256_array(np.stack([edge_i, edge_j], axis=1)),
        "utility_sha256": sha256_array(result["utility_raw_four"]),
        "partition_sha256": {arm: sha256_array(part) for arm, part in zip(ARM_IDS, arm_partitions)},
        "wall_seconds": time.time() - started,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "peak_gpu_mb": float(torch.cuda.max_memory_allocated() / (1024 ** 2)) if device.startswith("cuda") else 0.0,
    }
    return result, manifest


def write_result(output: Path, result: Mapping[str, np.ndarray], manifest: Mapping[str, object]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **result)
    temporary.replace(output)
    with np.load(output, allow_pickle=False) as replay:
        if not np.array_equal(replay["arm_partitions"], result["arm_partitions"]):
            raise RuntimeError("producer artifact reload partition mismatch")
        if not np.array_equal(replay["utility_raw_four"], result["utility_raw_four"]):
            raise RuntimeError("producer artifact reload utility mismatch")
    manifest = dict(manifest)
    manifest["artifact_path"] = str(output.resolve())
    manifest["artifact_sha256"] = file_sha256(output)
    manifest["artifact_reload"] = "PASS"
    output.with_suffix(".producer.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--carrier", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lane", required=True)
    parser.add_argument("--k", required=True, type=int)
    parser.add_argument("--mode", choices=("train", "replay"), default="train")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    result, manifest = run_pipeline(
        Path(args.carrier), Path(args.config), Path(args.checkpoint), args.k, args.lane, args.mode, args.device
    )
    write_result(Path(args.output), result, manifest)
    print(json.dumps({"lane": args.lane, "artifact": str(Path(args.output).resolve()), "status": "PASS"}, sort_keys=True))


if __name__ == "__main__":
    main()
