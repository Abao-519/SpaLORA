#!/usr/bin/env python3
"""Hash-locked Night-2C main, tutorial, and technical-replicate runner."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch


os.environ.setdefault("R_HOME", "/opt/R/4.0.3/lib/R")
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from SpaLORA.night1_evaluation import evaluate, load_evaluation_labels
from SpaLORA.night2c_loss_audit import (
    ParityLockedTrainer, VARIANTS, compute_locked_asr_weights, critical_hashes,
    environment_payload, sha256_file,
)
from SpaLORA.preprocess import fix_seed
from SpaLORA.SpaLORA_pyG import Train_SpaLORA
from scripts.night1_benchmark import cluster_exact, prepare_legacy


PARENT = "c283449b188f510e98c2826cbb856f296367aa03"
DATASETS = ("a1", "placenta", "p22")
REQUIRED_RUN_FILES = ("metrics.json", "clusters.csv", "attention.npz", "embedding.npz",
                      "loss_components.csv", "observation_ids.csv")


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()


def hash_tensor(value: torch.Tensor) -> str:
    value = value.detach().cpu()
    if value.is_sparse:
        value = value.coalesce()
        content = value.indices().contiguous().numpy().tobytes() + value.values().contiguous().numpy().tobytes()
    else:
        content = value.contiguous().numpy().tobytes()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode()); digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(content)
    return digest.hexdigest()


def locked_input_hash(trainer: ParityLockedTrainer) -> str:
    digest = hashlib.sha256()
    for value in (trainer.features1, trainer.features2) + trainer.adjacencies:
        digest.update(hash_tensor(value).encode())
    digest.update("\n".join(trainer.gene_names).encode()); digest.update("\n".join(trainer.obs_names).encode())
    return digest.hexdigest()


def load_context() -> tuple:
    config_path = REPO / "configs/night2c_numerical_equivalence_factorial.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    gate = json.loads((REPO / "results/night2c/gate_status.json").read_text(encoding="utf-8"))
    if gate.get("p0c_pass") is not True or gate.get("factorial_authorized") is not True:
        raise RuntimeError("P0C has not authorized any Night-2C training")
    if config["parent_commit"] != PARENT or config["seeds"] != [0, 1, 2, 3, 4] or tuple(config["variants"]) != VARIANTS:
        raise AssertionError("immutable Night-2C configuration drift")
    environment = environment_payload()
    hashes = critical_hashes(REPO, config)
    if hashes != gate.get("critical_hashes"):
        raise RuntimeError("critical source/config/input hash drift after P0C")
    if environment["fingerprint"] != gate.get("environment_fingerprint"):
        raise RuntimeError("environment fingerprint drift after P0C")
    return config_path, config, gate, environment, hashes


def run_dir_for(kind: str, dataset: str, variant: str, seed: int, repeat: int = 0) -> Path:
    if kind == "main":
        return REPO / "results/night2c/raw" / dataset / variant / ("seed_%d" % seed)
    if kind == "technical":
        return REPO / "results/night2c/technical" / dataset / variant / ("seed_%d" % seed) / ("repeat_%d" % repeat)
    raise ValueError(kind)


def existing_is_valid(path: Path, metadata: dict) -> bool:
    if not all((path / name).is_file() for name in REQUIRED_RUN_FILES):
        return False
    if (path / "failure.json").exists():
        return False
    try:
        payload = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    except Exception:
        return False
    return all(payload.get(key) == value for key, value in metadata.items())


def validate_attention(output: dict, n_obs: int) -> dict:
    result = {}
    for name in ("alpha", "alpha_omics1", "alpha_omics2"):
        value = np.asarray(output[name])
        if value.shape != (n_obs, 2):
            raise AssertionError("%s shape %r" % (name, value.shape))
        result[name] = float(np.max(np.abs(value.sum(axis=1) - 1.0)))
    return result


def fsync_path(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def save_prediction_artifacts(run_dir: Path, output: dict, obs_names: np.ndarray,
                              predicted: np.ndarray = None) -> None:
    embedding_path = run_dir / "embedding.npz"
    attention_path = run_dir / "attention.npz"
    ids_path = run_dir / "observation_ids.csv"
    np.savez_compressed(embedding_path,
                        SpaLORA=np.asarray(output["SpaLORA"], dtype=np.float32),
                        emb_latent_omics1=np.asarray(output["emb_latent_omics1"], dtype=np.float32),
                        emb_latent_omics2=np.asarray(output["emb_latent_omics2"], dtype=np.float32))
    np.savez_compressed(attention_path,
                        alpha=np.asarray(output["alpha"], dtype=np.float32),
                        alpha_omics1=np.asarray(output["alpha_omics1"], dtype=np.float32),
                        alpha_omics2=np.asarray(output["alpha_omics2"], dtype=np.float32))
    pd.DataFrame({"observation_id": obs_names.astype(str)}).to_csv(ids_path, index=False)
    for path in (embedding_path, attention_path, ids_path):
        fsync_path(path)
    if predicted is not None:
        cluster_path = run_dir / "clusters.csv"
        pd.DataFrame({"observation_id": obs_names.astype(str), "cluster": predicted}).to_csv(cluster_path, index=False)
        fsync_path(cluster_path)


def run_one(config_path: Path, config: dict, environment: dict, hashes: dict,
            dataset: str, variant: str, seed: int, kind: str = "main", repeat: int = 0) -> Path:
    if dataset not in DATASETS or variant not in VARIANTS or seed not in config["seeds"]:
        raise ValueError("unregistered dataset/variant/seed")
    if kind == "technical" and not (dataset == "placenta" and seed == 0 and repeat in (1, 2)
                                     and variant in ("locked_unweighted", "locked_legacy_loss_replay")):
        raise ValueError("unregistered technical replicate")
    cfg = config["datasets"][dataset]
    run_dir = run_dir_for(kind, dataset, variant, seed, repeat)
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"dataset": dataset, "variant": variant, "seed": seed, "run_kind": kind,
                "technical_repeat": repeat, "parent_commit": PARENT,
                "config_sha256": sha256_file(config_path),
                "environment_fingerprint": environment["fingerprint"], "critical_hashes": hashes}
    if existing_is_valid(run_dir, metadata):
        print("SKIP_VALID", kind, dataset, variant, seed, repeat, flush=True)
        return run_dir / "metrics.json"
    # A partial or mismatched run is never silently resumed.
    existing = [path.name for path in run_dir.iterdir()]
    if existing:
        raise RuntimeError("partial or mismatched run requires diagnosis, not overwrite: %s %r" % (run_dir, existing))
    failure_path = run_dir / "failure.json"
    started = time.perf_counter()
    try:
        # Exactly one seed reset at Night-1's pre-preparation position.
        fix_seed(seed)
        torch.use_deterministic_algorithms(False)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        preprocessing_start = time.perf_counter()
        data, obs_names, coordinates = prepare_legacy(dataset, cfg)
        asr_names = asr_weights = None
        if variant == "locked_asr_hvg_legacy_scale_diagnostic":
            asr_names, asr_weights = compute_locked_asr_weights(dataset, cfg, config, data)
        trainer = ParityLockedTrainer(data, cfg, variant, seed, torch.device("cuda:0"), asr_weights)
        if asr_names is not None and not np.array_equal(asr_names, trainer.gene_names):
            raise AssertionError("V4 ASR/model gene order differs")
        input_fingerprint = locked_input_hash(trainer)
        preprocessing_seconds = time.perf_counter() - preprocessing_start
        training_start = time.perf_counter()
        output, loss_rows = trainer.train(); torch.cuda.synchronize()
        training_seconds = time.perf_counter() - training_start
        embedding = np.asarray(output["SpaLORA"], dtype=np.float32)
        attention_deviation = validate_attention(output, len(obs_names))
        # Required durable order: embeddings/attention/IDs/loss, then clusters, then labels.
        save_prediction_artifacts(run_dir, output, obs_names.astype(str).to_numpy())
        loss_path = run_dir / "loss_components.csv"
        with loss_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(loss_rows[0])); writer.writeheader(); writer.writerows(loss_rows)
        fsync_path(loss_path)
        clustering_start = time.perf_counter()
        predicted = cluster_exact(embedding, cfg["n_clusters"], config["clustering_seed"])
        clustering_seconds = time.perf_counter() - clustering_start
        save_prediction_artifacts(run_dir, output, obs_names.astype(str).to_numpy(), predicted)
        # This is deliberately the first semantic label access.
        evaluation_positions, true_labels = load_evaluation_labels(dataset, cfg, obs_names)
        evaluation_start = time.perf_counter()
        metrics = evaluate(true_labels, predicted[evaluation_positions], predicted, embedding,
                           coordinates, cfg["spatial_neighbors"])
        evaluation_seconds = time.perf_counter() - evaluation_start
        payload = dict(metadata)
        payload.update({"schema_version": 1, "code_commit_at_run": git_value("rev-parse", "HEAD"),
                        "data_sha256": {"rna": sha256_file(Path(cfg["rna"])),
                                        "modality2": sha256_file(Path(cfg["modality2"]))},
                        "locked_input_sha256": input_fingerprint,
                        "n_observations_trained": int(len(obs_names)),
                        "n_observations_evaluated": int(evaluation_positions.size),
                        "n_clusters_requested": int(cfg["n_clusters"]),
                        "n_clusters_observed": int(np.unique(predicted).size), "metrics": metrics,
                        "attention_max_row_sum_deviation": attention_deviation,
                        "final_attention_means": {
                            "cross_omics_rna": float(np.asarray(output["alpha"])[:, 0].mean()),
                            "rna_spatial": float(np.asarray(output["alpha_omics1"])[:, 0].mean()),
                            "modality2_spatial": float(np.asarray(output["alpha_omics2"])[:, 0].mean())},
                        "m_bad": float(trainer.legacy.weight_vector_omics1.mean().detach().cpu()),
                        "timings": {"preprocessing_seconds": preprocessing_seconds,
                                    "training_seconds": training_seconds,
                                    "clustering_seconds": clustering_seconds,
                                    "evaluation_seconds": evaluation_seconds,
                                    "total_seconds": time.perf_counter() - started},
                        "memory": {"gpu_peak_allocated_mib": float(torch.cuda.max_memory_allocated() / 1024 ** 2),
                                   "process_peak_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)},
                        "label_access_after_unsupervised_serialization": True})
        metrics_path = run_dir / "metrics.json"
        metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"); fsync_path(metrics_path)
        print("DONE", kind, dataset, variant, seed, repeat, "ARI=%.6f" % metrics["ari"], flush=True)
        return metrics_path
    except Exception as exc:
        failure_path.write_text(json.dumps({"dataset": dataset, "variant": variant, "seed": seed,
                                            "run_kind": kind, "repeat": repeat, "error": repr(exc),
                                            "traceback": traceback.format_exc()}, indent=2), encoding="utf-8")
        raise
    finally:
        gc.collect(); torch.cuda.empty_cache()


def tutorial_existing_valid(path: Path, metadata: dict) -> bool:
    required = ("metrics.json", "clusters.csv", "attention.npz", "embedding.npz", "observation_ids.csv")
    if not all((path / name).is_file() for name in required) or (path / "failure.json").exists():
        return False
    payload = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    return all(payload.get(k) == v for k, v in metadata.items())


def run_tutorial(config_path, config, environment, hashes, dataset) -> Path:
    cfg = config["datasets"][dataset]
    run_dir = REPO / "results/night2c/tutorial2022" / dataset
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"dataset": dataset, "variant": "public_tutorial_exact", "model_seed": 2022,
                "clustering_seed": 2020, "parent_commit": PARENT,
                "config_sha256": sha256_file(config_path),
                "environment_fingerprint": environment["fingerprint"], "critical_hashes": hashes}
    if tutorial_existing_valid(run_dir, metadata):
        print("SKIP_VALID_TUTORIAL", dataset, flush=True); return run_dir / "metrics.json"
    existing = [path.name for path in run_dir.iterdir()]
    if existing:
        raise RuntimeError("partial or mismatched tutorial run: %s %r" % (run_dir, existing))
    failure = run_dir / "failure.json"; started = time.perf_counter()
    try:
        fix_seed(config["tutorial_model_seed"]); torch.use_deterministic_algorithms(False)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        data, obs_names, coordinates = prepare_legacy(dataset, cfg)
        trainer = Train_SpaLORA(data, datatype=cfg["legacy_datatype"], device=torch.device("cuda:0"),
                                random_seed=2022)
        output = trainer.train(); torch.cuda.synchronize()
        embedding = np.asarray(output["SpaLORA"], dtype=np.float32)
        save_prediction_artifacts(run_dir, output, obs_names.astype(str).to_numpy())
        predicted = cluster_exact(embedding, cfg["n_clusters"], config["clustering_seed"])
        save_prediction_artifacts(run_dir, output, obs_names.astype(str).to_numpy(), predicted)
        evaluation_positions, true_labels = load_evaluation_labels(dataset, cfg, obs_names)
        metrics = evaluate(true_labels, predicted[evaluation_positions], predicted, embedding,
                           coordinates, cfg["spatial_neighbors"])
        payload = dict(metadata)
        payload.update({"schema_version": 1, "code_commit_at_run": git_value("rev-parse", "HEAD"),
                        "metrics": metrics, "n_observations_trained": int(len(obs_names)),
                        "n_observations_evaluated": int(evaluation_positions.size),
                        "n_clusters_observed": int(np.unique(predicted).size),
                        "timings": {"total_seconds": time.perf_counter() - started},
                        "memory": {"gpu_peak_allocated_mib": float(torch.cuda.max_memory_allocated() / 1024 ** 2)},
                        "label_access_after_unsupervised_serialization": True})
        path = run_dir / "metrics.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"); fsync_path(path)
        print("DONE_TUTORIAL", dataset, "ARI=%.6f" % metrics["ari"], flush=True); return path
    except Exception as exc:
        failure.write_text(json.dumps({"dataset": dataset, "error": repr(exc),
                                       "traceback": traceback.format_exc()}, indent=2), encoding="utf-8")
        raise


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("main", "tutorial", "technical"), required=True)
    parser.add_argument("--dataset", choices=DATASETS + ("all",), default="all")
    parser.add_argument("--variant", choices=VARIANTS + ("all",), default="all")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--repeat", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path, config, _, environment, hashes = load_context()
    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    if args.mode == "tutorial":
        if args.variant != "all" or args.seed is not None or args.repeat != 0:
            raise ValueError("tutorial takes only --dataset")
        for dataset in datasets:
            run_tutorial(config_path, config, environment, hashes, dataset)
        return
    if args.mode == "technical":
        if args.dataset != "placenta" or args.variant == "all" or args.seed != 0:
            raise ValueError("technical requires placenta, one registered variant, seed 0")
        run_one(config_path, config, environment, hashes, "placenta", args.variant, 0,
                kind="technical", repeat=args.repeat)
        return
    if args.repeat != 0:
        raise ValueError("main repeat must be zero")
    seeds = config["seeds"] if args.seed is None else [args.seed]
    if any(seed not in config["seeds"] for seed in seeds):
        raise ValueError("seeds are fixed to [0,1,2,3,4]")
    requested = list(config["variants"]) if args.variant == "all" else [args.variant]
    # Preregistered order: dataset -> seed -> V0..V4 rotated left by seed index.
    for dataset in datasets:
        for seed in seeds:
            rotated = list(config["variants"])[seed:] + list(config["variants"])[:seed]
            for variant in rotated:
                if variant in requested:
                    run_one(config_path, config, environment, hashes, dataset, variant, seed)


if __name__ == "__main__":
    main()
