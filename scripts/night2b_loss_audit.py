#!/usr/bin/env python3
"""Run the authorized Night-2B parity-locked factorial and tutorials."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
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
from SpaLORA.night2b_loss_audit import ParityLockedTrainer, VARIANTS, compute_locked_asr_weights
from SpaLORA.preprocess import fix_seed
from SpaLORA.SpaLORA_pyG import Train_SpaLORA
from scripts.night1_benchmark import cluster_exact, prepare_legacy


SOURCE_HASH_CACHE = {}


def sha256(path: Path) -> str:
    if str(path) in SOURCE_HASH_CACHE:
        return SOURCE_HASH_CACHE[str(path)]
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    SOURCE_HASH_CACHE[str(path)] = value
    return value


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()


def environment_payload() -> dict:
    try:
        r_version = subprocess.check_output(["/opt/R/4.0.3/bin/R", "--version"], text=True).splitlines()[0]
        mclust = subprocess.check_output(
            ["/opt/R/4.0.3/bin/Rscript", "-e", "cat(as.character(packageVersion('mclust')))"], text=True
        ).strip()
    except Exception as exc:
        r_version, mclust = "unavailable: %r" % exc, "unavailable"
    payload = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "r": r_version,
        "mclust": mclust,
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
    }
    payload["fingerprint"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


def hash_tensor(value: torch.Tensor) -> str:
    value = value.detach().cpu()
    if value.is_sparse:
        value = value.coalesce()
        content = value.indices().contiguous().numpy().tobytes() + value.values().contiguous().numpy().tobytes()
    else:
        content = value.contiguous().numpy().tobytes()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(content)
    return digest.hexdigest()


def locked_input_hash(trainer: ParityLockedTrainer) -> str:
    digest = hashlib.sha256()
    for value in (trainer.features1, trainer.features2) + trainer.adjacencies:
        digest.update(hash_tensor(value).encode())
    digest.update("\n".join(trainer.gene_names).encode())
    digest.update("\n".join(trainer.obs_names).encode())
    return digest.hexdigest()


def validate_attention(output: dict, n_obs: int) -> dict:
    result = {}
    for name in ("alpha", "alpha_omics1", "alpha_omics2"):
        value = np.asarray(output[name])
        if value.shape != (n_obs, 2):
            raise AssertionError("%s shape %r" % (name, value.shape))
        result[name] = float(np.max(np.abs(value.sum(axis=1) - 1.0)))
    return result


def config_and_environment() -> tuple:
    config_path = REPO / "configs" / "night2b_parity_locked_loss_audit.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    night1 = json.loads((REPO / config["night1_config"]).read_text(encoding="utf-8"))
    environment = environment_payload()
    reports = REPO / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "night2b_environment.json").write_text(json.dumps(environment, indent=2, sort_keys=True), encoding="utf-8")
    return config_path, config, night1, environment


def gate_authorized() -> None:
    gate = json.loads((REPO / "results" / "night2b" / "gate_status.json").read_text(encoding="utf-8"))
    if gate.get("p0b_pass") is not True or gate.get("factorial_authorized") is not True:
        raise RuntimeError("P0B has not authorized Night-2B training")


def existing_is_valid(metrics_path: Path, config_hash: str, environment: dict, dataset: str, variant: str, seed: int) -> bool:
    required = ("clusters.csv", "attention.npz", "embedding.npz", "loss_components.csv")
    if not metrics_path.is_file() or not all((metrics_path.parent / name).is_file() for name in required):
        return False
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        payload.get("dataset") == dataset
        and payload.get("variant") == variant
        and payload.get("seed") == seed
        and payload.get("config_sha256") == config_hash
        and payload.get("parent_commit") == "16f0cc43673617c73527110962b7ca115c59b4c6"
        and payload.get("environment_fingerprint") == environment["fingerprint"]
    )


def save_prediction_artifacts(run_dir: Path, output: dict, obs_names: np.ndarray, predicted: np.ndarray = None) -> None:
    np.savez_compressed(
        run_dir / "embedding.npz",
        SpaLORA=np.asarray(output["SpaLORA"], dtype=np.float32),
        emb_latent_omics1=np.asarray(output["emb_latent_omics1"], dtype=np.float32),
        emb_latent_omics2=np.asarray(output["emb_latent_omics2"], dtype=np.float32),
    )
    np.savez_compressed(
        run_dir / "attention.npz",
        alpha=np.asarray(output["alpha"], dtype=np.float32),
        alpha_omics1=np.asarray(output["alpha_omics1"], dtype=np.float32),
        alpha_omics2=np.asarray(output["alpha_omics2"], dtype=np.float32),
    )
    if predicted is not None:
        pd.DataFrame({"observation_id": obs_names.astype(str), "cluster": predicted}).to_csv(
            run_dir / "clusters.csv", index=False
        )


def run_one(
    config_path: Path,
    config: dict,
    night1: dict,
    environment: dict,
    dataset: str,
    variant: str,
    seed: int,
) -> Path:
    cfg = night1["datasets"][dataset]
    run_dir = REPO / "results" / "night2b" / "raw" / dataset / variant / ("seed_%d" % seed)
    metrics_path = run_dir / "metrics.json"
    config_hash = sha256(config_path)
    if existing_is_valid(metrics_path, config_hash, environment, dataset, variant, seed):
        print("SKIP_VALID", dataset, variant, seed, flush=True)
        return metrics_path
    run_dir.mkdir(parents=True, exist_ok=True)
    failure_path = run_dir / "failure.json"
    if failure_path.exists():
        failure_path.unlink()
    started = time.perf_counter()
    try:
        # Exactly one seed reset at the same pre-preparation position as Night-1.
        fix_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        preprocessing_start = time.perf_counter()
        data, obs_names, coordinates = prepare_legacy(dataset, cfg)
        asr_weights = None
        asr_names = None
        if variant == "locked_asr_hvg_legacy_scale_diagnostic":
            asr_names, asr_weights = compute_locked_asr_weights(dataset, cfg, night1, data)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        trainer = ParityLockedTrainer(data, cfg, variant, seed, device, asr_weights)
        if asr_names is not None and not np.array_equal(asr_names, trainer.gene_names):
            raise AssertionError("V4 ASR/model selected-gene order differs")
        input_fingerprint = locked_input_hash(trainer)
        preprocessing_seconds = time.perf_counter() - preprocessing_start

        training_start = time.perf_counter()
        output, loss_rows = trainer.train()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        training_seconds = time.perf_counter() - training_start
        embedding = np.asarray(output["SpaLORA"], dtype=np.float32)
        attention_deviation = validate_attention(output, len(obs_names))
        # Embeddings/attention are durably serialized before clustering and label access.
        save_prediction_artifacts(run_dir, output, obs_names.astype(str).to_numpy())
        with (run_dir / "loss_components.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(loss_rows[0]))
            writer.writeheader()
            writer.writerows(loss_rows)

        clustering_start = time.perf_counter()
        predicted = cluster_exact(embedding, cfg["n_clusters"], config["clustering_seed"])
        clustering_seconds = time.perf_counter() - clustering_start
        # Unsupervised assignments are serialized before the first ground-truth access.
        save_prediction_artifacts(run_dir, output, obs_names.astype(str).to_numpy(), predicted)

        evaluation_positions, true_labels = load_evaluation_labels(dataset, cfg, obs_names)
        evaluation_start = time.perf_counter()
        metrics = evaluate(
            true_labels,
            predicted[evaluation_positions],
            predicted,
            embedding,
            coordinates,
            cfg["spatial_neighbors"],
        )
        evaluation_seconds = time.perf_counter() - evaluation_start
        payload = {
            "schema_version": 1,
            "dataset": dataset,
            "variant": variant,
            "seed": seed,
            "parent_commit": config["parent_commit"],
            "code_commit_at_run": git_value("rev-parse", "HEAD"),
            "config_sha256": config_hash,
            "environment_fingerprint": environment["fingerprint"],
            "environment": environment,
            "data_sha256": {"rna": sha256(Path(cfg["rna"])), "modality2": sha256(Path(cfg["modality2"]))},
            "locked_input_sha256": input_fingerprint,
            "n_observations_trained": int(len(obs_names)),
            "n_observations_evaluated": int(evaluation_positions.size),
            "n_clusters_requested": int(cfg["n_clusters"]),
            "n_clusters_observed": int(np.unique(predicted).size),
            "metrics": metrics,
            "attention_max_row_sum_deviation": attention_deviation,
            "final_attention_means": {
                "cross_omics_rna": float(np.asarray(output["alpha"])[:, 0].mean()),
                "rna_spatial": float(np.asarray(output["alpha_omics1"])[:, 0].mean()),
                "modality2_spatial": float(np.asarray(output["alpha_omics2"])[:, 0].mean()),
            },
            "m_bad": float(trainer.legacy.weight_vector_omics1.mean().detach().cpu()),
            "timings": {
                "preprocessing_seconds": preprocessing_seconds,
                "training_seconds": training_seconds,
                "clustering_seconds": clustering_seconds,
                "evaluation_seconds": evaluation_seconds,
                "total_seconds": time.perf_counter() - started,
            },
            "memory": {
                "gpu_peak_allocated_mib": float(torch.cuda.max_memory_allocated() / 1024 ** 2) if torch.cuda.is_available() else 0.0,
                "process_peak_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
            },
        }
        metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print("DONE", dataset, variant, seed, "ARI=%.6f" % metrics["ari"], "seconds=%.1f" % payload["timings"]["total_seconds"], flush=True)
        return metrics_path
    except Exception as exc:
        failure_path.write_text(
            json.dumps({"dataset": dataset, "variant": variant, "seed": seed, "error": repr(exc), "traceback": traceback.format_exc()}, indent=2),
            encoding="utf-8",
        )
        raise
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def tutorial_existing_valid(path: Path, config_hash: str, environment: dict, dataset: str) -> bool:
    required = ("metrics.json", "clusters.csv", "attention.npz", "embedding.npz")
    if not all((path / name).is_file() for name in required):
        return False
    payload = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    return payload.get("dataset") == dataset and payload.get("model_seed") == 2022 and payload.get("config_sha256") == config_hash and payload.get("environment_fingerprint") == environment["fingerprint"]


def run_tutorial(config_path: Path, config: dict, night1: dict, environment: dict, dataset: str) -> Path:
    cfg = night1["datasets"][dataset]
    run_dir = REPO / "results" / "night2b" / "tutorial2022" / dataset
    run_dir.mkdir(parents=True, exist_ok=True)
    config_hash = sha256(config_path)
    if tutorial_existing_valid(run_dir, config_hash, environment, dataset):
        print("SKIP_VALID_TUTORIAL", dataset, flush=True)
        return run_dir / "metrics.json"
    failure = run_dir / "failure.json"
    if failure.exists():
        failure.unlink()
    started = time.perf_counter()
    try:
        fix_seed(config["tutorial_model_seed"])
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        data, obs_names, coordinates = prepare_legacy(dataset, cfg)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        trainer = Train_SpaLORA(data, datatype=cfg["legacy_datatype"], device=device, random_seed=2022)
        output = trainer.train()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        embedding = np.asarray(output["SpaLORA"], dtype=np.float32)
        save_prediction_artifacts(run_dir, output, obs_names.astype(str).to_numpy())
        predicted = cluster_exact(embedding, cfg["n_clusters"], config["clustering_seed"])
        save_prediction_artifacts(run_dir, output, obs_names.astype(str).to_numpy(), predicted)
        evaluation_positions, true_labels = load_evaluation_labels(dataset, cfg, obs_names)
        metrics = evaluate(true_labels, predicted[evaluation_positions], predicted, embedding, coordinates, cfg["spatial_neighbors"])
        payload = {
            "schema_version": 1,
            "dataset": dataset,
            "variant": "public_tutorial_exact",
            "model_seed": 2022,
            "clustering_seed": 2020,
            "parent_commit": config["parent_commit"],
            "code_commit_at_run": git_value("rev-parse", "HEAD"),
            "config_sha256": config_hash,
            "environment_fingerprint": environment["fingerprint"],
            "environment": environment,
            "metrics": metrics,
            "n_observations_trained": int(len(obs_names)),
            "n_observations_evaluated": int(evaluation_positions.size),
            "n_clusters_observed": int(np.unique(predicted).size),
            "timings": {"total_seconds": time.perf_counter() - started},
            "memory": {"gpu_peak_allocated_mib": float(torch.cuda.max_memory_allocated() / 1024 ** 2) if torch.cuda.is_available() else 0.0},
        }
        (run_dir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print("DONE_TUTORIAL", dataset, "ARI=%.6f" % metrics["ari"], flush=True)
        return run_dir / "metrics.json"
    except Exception as exc:
        failure.write_text(json.dumps({"dataset": dataset, "error": repr(exc), "traceback": traceback.format_exc()}, indent=2), encoding="utf-8")
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["a1", "placenta", "p22", "all"], default="all")
    parser.add_argument("--variant", choices=list(VARIANTS) + ["all"], default="all")
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--tutorial2022", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gate_authorized()
    config_path, config, night1, environment = config_and_environment()
    datasets = list(night1["datasets"]) if args.dataset == "all" else [args.dataset]
    if args.tutorial2022:
        for dataset in datasets:
            run_tutorial(config_path, config, night1, environment, dataset)
        return
    variants = list(config["variants"]) if args.variant == "all" else [args.variant]
    seeds = config["seeds"] if not args.seeds else args.seeds
    if any(seed not in config["seeds"] for seed in seeds):
        raise ValueError("Seeds must remain %r" % config["seeds"])
    for dataset in datasets:
        for variant in variants:
            for seed in seeds:
                run_one(config_path, config, night1, environment, dataset, variant, seed)


if __name__ == "__main__":
    main()
