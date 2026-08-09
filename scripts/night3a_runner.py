#!/usr/bin/env python3
"""Hash-locked, label-free runner for the 60 preregistered Night-3A runs."""

from __future__ import annotations

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

import anndata as ad
import numpy as np
import pandas as pd
import torch


os.environ.setdefault("R_HOME", "/opt/R/4.0.3/lib/R")
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CONFIG_PATH = REPO / "configs/night3a_ige_feasibility.json"
REQUIRED_RUN_FILES = (
    "embedding.npz", "attention.npz", "observation_ids.csv", "clusters.csv",
    "loss_trajectory.csv", "checkpoint_index.csv", "run_config.json",
    "model_final.pt", "run_manifest.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(str(temporary), str(path))


def fsync_path(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def verify_lock(config: dict, lock: dict) -> None:
    checks = {"config": sha256_file(CONFIG_PATH) == lock["config_sha256"]}
    checks.update({
        "source:" + name: (REPO / name).is_file() and sha256_file(REPO / name) == expected
        for name, expected in lock["source_sha256"].items()
    })
    checks.update({
        "data:" + name: Path(name).is_file() and sha256_file(Path(name)) == expected
        for name, expected in lock["data_sha256"].items()
    })
    order = REPO / config["run_order"]["manifest"]
    checks["run_order"] = order.is_file() and sha256_file(order) == lock["run_order_sha256"]
    if not all(checks.values()):
        raise RuntimeError("Night-3A lock drift: %r" % [name for name, passed in checks.items() if not passed])


def preprocessing_config(config: dict) -> dict:
    pre = config["preprocessing"]
    return {
        "min_cells": pre["min_cells"],
        "alpha": pre["alpha_compatibility_only"],
        "rescue_non_hvg": pre["rescue_non_hvg_compatibility_only"],
        "moran_shrinkage_tau": pre["moran_shrinkage_tau_compatibility_only"],
        "feature_graph": {"k": pre["feature_graph_k"], "metric": pre["feature_graph_metric"]},
    }


def cluster_exact(embedding: np.ndarray, n_clusters: int, random_seed: int) -> np.ndarray:
    from SpaLORA.utils import clustering

    if random_seed != 2020:
        raise AssertionError("Clustering seed must remain 2020")
    clustered = ad.AnnData(np.zeros((embedding.shape[0], 1), dtype=np.float32))
    clustered.obsm["SpaLORA"] = np.asarray(embedding, dtype=np.float32)
    clustering(clustered, key="SpaLORA", add_key="SpaLORA", n_clusters=n_clusters,
               use_pca=True, n_comps=20)
    return clustered.obs["SpaLORA"].astype(int).to_numpy()


def validate_attention(output: dict, n_obs: int) -> dict:
    result = {}
    for key in ("alpha", "alpha_omics1", "alpha_omics2"):
        value = np.asarray(output[key])
        if value.shape != (n_obs, 2) or not np.isfinite(value).all():
            raise AssertionError("%s invalid shape/nonfinite" % key)
        result[key] = float(np.max(np.abs(value.sum(axis=1) - 1.0)))
        if result[key] > 1e-6:
            raise AssertionError("%s attention rows do not sum to one" % key)
    return result


def run_directory(output: Path, dataset: str, variant: str, seed: int) -> Path:
    return output / "runs" / dataset / variant / ("seed_%d" % seed)


def existing_valid(path: Path, identity: dict) -> bool:
    if not all((path / name).is_file() for name in REQUIRED_RUN_FILES) or (path / "failure.json").exists():
        return False
    try:
        manifest = json.loads((path / "run_manifest.json").read_text(encoding="utf-8"))
    except Exception:
        return False
    if any(manifest.get(key) != value for key, value in identity.items()):
        return False
    return all(
        (path / name).is_file() and sha256_file(path / name) == expected
        for name, expected in manifest.get("artifact_sha256", {}).items()
    )


def write_csv(path: Path, rows: list) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    fsync_path(path)


def run_one(config: dict, lock: dict, output: Path, prepared_cache: dict, prep_seconds: dict,
            dataset: str, variant: str, seed: int, ordinal: int) -> Path:
    from SpaLORA.night3a_ige import Night3ATrainer, input_sha256, registered_variant_contract

    cfg = config["datasets"][dataset]
    run_dir = run_directory(output, dataset, variant, seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    identity = {
        "schema_version": 1,
        "dataset": dataset,
        "variant": variant,
        "seed": seed,
        "run_order_ordinal": ordinal,
        "config_lock_sha256": sha256_file(output / "config_lock.json"),
        "config_sha256": lock["config_sha256"],
    }
    if existing_valid(run_dir, identity):
        print("SKIP_VALID %02d/60 %s %s seed=%d" % (ordinal, dataset, variant, seed), flush=True)
        return run_dir / "run_manifest.json"
    existing = [path.name for path in run_dir.iterdir()]
    if existing:
        raise RuntimeError("Partial/mismatched run cannot be overwritten: %s %r" % (run_dir, existing))
    failure_path = run_dir / "failure.json"
    started = time.perf_counter()
    try:
        prepared = prepared_cache[dataset]
        device = torch.device("cuda:0")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        trainer = Night3ATrainer(prepared.data, cfg, variant, seed, device, config["ige_epsilon"])
        locked_input = input_sha256(
            prepared.data, prepared.obs_names.astype(str), prepared.data["selected_gene_names"]
        )
        training_started = time.perf_counter()
        result = trainer.train()
        torch.cuda.synchronize()
        training_seconds = time.perf_counter() - training_started
        attention_deviation = validate_attention(result.output, len(prepared.obs_names))

        embedding_path = run_dir / "embedding.npz"
        attention_path = run_dir / "attention.npz"
        ids_path = run_dir / "observation_ids.csv"
        loss_path = run_dir / "loss_trajectory.csv"
        checkpoint_path = run_dir / "checkpoint_index.csv"
        config_path = run_dir / "run_config.json"
        model_path = run_dir / "model_final.pt"
        cluster_path = run_dir / "clusters.csv"
        np.savez_compressed(
            embedding_path,
            SpaLORA=np.asarray(result.output["SpaLORA"], dtype=np.float32),
            emb_latent_omics1=np.asarray(result.output["emb_latent_omics1"], dtype=np.float32),
            emb_latent_omics2=np.asarray(result.output["emb_latent_omics2"], dtype=np.float32),
        )
        np.savez_compressed(
            attention_path,
            alpha=np.asarray(result.output["alpha"], dtype=np.float32),
            alpha_omics1=np.asarray(result.output["alpha_omics1"], dtype=np.float32),
            alpha_omics2=np.asarray(result.output["alpha_omics2"], dtype=np.float32),
        )
        pd.DataFrame({"observation_id": prepared.obs_names.astype(str)}).to_csv(ids_path, index=False)
        write_csv(loss_path, result.logs)
        write_csv(checkpoint_path, [
            {
                "step": row["step"],
                "fraction_of_training": row["fraction_of_training"],
                "state_sha256": row["checkpoint_state_sha256"],
                "state_file_saved": row["step"] == int(cfg["epochs"]),
            }
            for row in result.logs
        ])
        torch.save(
            {
                "state_dict": {name: value.detach().cpu() for name, value in result.model.state_dict().items()},
                "dataset": dataset,
                "variant": variant,
                "seed": seed,
                "final_state_sha256": result.final_state_sha256,
            },
            model_path,
        )
        atomic_json(
            config_path,
            {
                "schema_version": 1,
                "dataset": dataset,
                "variant": variant,
                "seed": seed,
                "epochs": cfg["epochs"],
                "embedding_dim": cfg["embedding_dim"],
                "optimizer": {"name": "Adam", "learning_rate": 1e-4, "weight_decay": 0.0},
                "raw_loss_reduction": "torch.nn.functional.mse_loss mean over every element",
                "variant_contract": registered_variant_contract(config)[variant],
                "frozen_coefficients": result.coefficients,
                "initial_raw_losses": result.initial_losses,
                "ige_probe": None if result.probe is None else {
                    "rms_gradients": result.probe["gradients"],
                    "weights": result.probe["weights"],
                    "repeat_within_gpu_envelope": result.probe["repeat_within_gpu_envelope"],
                    "state_unchanged": result.probe["state_unchanged"],
                    "rng_unchanged": result.probe["rng_unchanged"],
                    "reload_forward_within_envelope": result.probe["reload_forward_within_envelope"],
                },
                "ground_truth_access": False,
                "config_lock_sha256": identity["config_lock_sha256"],
            },
        )
        for path in (embedding_path, attention_path, ids_path, model_path, config_path):
            fsync_path(path)

        clustering_started = time.perf_counter()
        predicted = cluster_exact(
            np.asarray(result.output["SpaLORA"], dtype=np.float32), cfg["n_clusters"],
            config["clustering"]["random_seed"],
        )
        clustering_seconds = time.perf_counter() - clustering_started
        if len(predicted) != len(prepared.obs_names) or len(np.unique(predicted)) != cfg["n_clusters"]:
            raise AssertionError("Clustering count/length mismatch")
        pd.DataFrame({"observation_id": prepared.obs_names.astype(str), "cluster": predicted}).to_csv(
            cluster_path, index=False
        )
        fsync_path(cluster_path)

        artifacts = {
            name: sha256_file(run_dir / name)
            for name in REQUIRED_RUN_FILES
            if name != "run_manifest.json"
        }
        manifest = dict(identity)
        manifest.update(
            {
                "n_observations": int(len(prepared.obs_names)),
                "n_selected_genes": int(len(prepared.data["selected_gene_names"])),
                "n_clusters": int(len(np.unique(predicted))),
                "locked_input_sha256": locked_input,
                "initial_state_sha256": result.initial_state_sha256,
                "final_state_sha256": result.final_state_sha256,
                "artifact_sha256": artifacts,
                "attention_max_row_sum_deviation": attention_deviation,
                "timings": {
                    "preprocessing_cache_seconds": prep_seconds[dataset],
                    "training_seconds": training_seconds,
                    "clustering_seconds": clustering_seconds,
                    "run_seconds_excluding_shared_preprocessing": time.perf_counter() - started,
                },
                "resources": {
                    "gpu_peak_allocated_mib": float(torch.cuda.max_memory_allocated() / 1024 ** 2),
                    "gpu_peak_reserved_mib": float(torch.cuda.max_memory_reserved() / 1024 ** 2),
                    "process_peak_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
                },
                "ground_truth_access": False,
                "semantic_label_values_read": False,
            }
        )
        atomic_json(run_dir / "run_manifest.json", manifest)
        fsync_path(run_dir / "run_manifest.json")
        print(
            "DONE %02d/60 %s %s seed=%d train=%.1fs total=%.1fs" %
            (ordinal, dataset, variant, seed, training_seconds, time.perf_counter() - started),
            flush=True,
        )
        return run_dir / "run_manifest.json"
    except Exception as exc:
        atomic_json(
            failure_path,
            {
                "dataset": dataset,
                "variant": variant,
                "seed": seed,
                "run_order_ordinal": ordinal,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    opened_paths = []

    def audit_hook(event, args):
        if event == "open" and args:
            try:
                opened_paths.append(str(Path(args[0]).resolve()))
            except Exception:
                pass

    sys.addaudithook(audit_hook)
    from SpaLORA.night1_pipeline import prepare_corrected

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    output = REPO / config["paths"]["output_root"]
    lock = json.loads((output / "config_lock.json").read_text(encoding="utf-8"))
    verify_lock(config, lock)
    gate = json.loads((output / "night3a_gate_status.json").read_text(encoding="utf-8"))
    if gate.get("p0a_pass") is not True or gate.get("p0b_pass") is not True or gate.get("main_60_authorized") is not True:
        raise RuntimeError("P0A/P0B did not authorize the 60-run factorial")
    if gate.get("config_lock_sha256") != sha256_file(output / "config_lock.json"):
        raise RuntimeError("Gate/config lock mismatch")
    order_path = REPO / config["run_order"]["manifest"]
    order_payload = json.loads(order_path.read_text(encoding="utf-8"))
    runs = order_payload["runs"]
    if len(runs) != 60 or len({(x["dataset"], x["variant"], x["seed"]) for x in runs}) != 60:
        raise AssertionError("Locked run manifest is not the 60-cell factorial")

    # Cache each immutable label-free preprocessing result exactly once.
    prepared_cache, prep_seconds = {}, {}
    pre_cfg = preprocessing_config(config)
    for dataset, cfg in config["datasets"].items():
        started = time.perf_counter()
        prepared_cache[dataset] = prepare_corrected(dataset, cfg, pre_cfg, "corrected_unweighted")
        prep_seconds[dataset] = time.perf_counter() - started
        print("PREPARED %s n=%d seconds=%.1f" %
              (dataset, len(prepared_cache[dataset].obs_names), prep_seconds[dataset]), flush=True)

    completed = []
    try:
        for ordinal, run in enumerate(runs, 1):
            path = run_one(
                config, lock, output, prepared_cache, prep_seconds,
                run["dataset"], run["variant"], int(run["seed"]), ordinal,
            )
            completed.append(path)
    except Exception:
        failures = sorted(str(path.relative_to(output)) for path in output.rglob("failure.json"))
        atomic_json(output / "failure_index.json", {"schema_version": 1, "failures": failures})
        raise

    forbidden_paths = {
        str(Path(cfg["ground_truth"]).resolve())
        for cfg in config["datasets"].values()
        if cfg["ground_truth"].startswith("/")
    }
    forbidden_opened = sorted(forbidden_paths.intersection(set(opened_paths)))
    forbidden_imported = sorted(
        name for name in config["label_firewall"]["training_must_not_import"] if name in sys.modules
    )
    label_access = {
        "schema_version": 1,
        "stage": "training_60",
        "ground_truth_files_opened": forbidden_opened,
        "forbidden_modules_imported": forbidden_imported,
        "semantic_label_values_read": False,
        "opened_path_count": len(set(opened_paths)),
        "passed": not forbidden_opened and not forbidden_imported,
    }
    atomic_json(output / "training_label_firewall.json", label_access)
    if not label_access["passed"]:
        raise RuntimeError("Label firewall violation: %r" % label_access)
    if len(completed) != 60:
        raise AssertionError("Expected 60 completed runs")

    locked_rows = []
    for ordinal, run in enumerate(runs, 1):
        path = run_directory(output, run["dataset"], run["variant"], int(run["seed"])) / "run_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        locked_rows.append(
            {
                "ordinal": ordinal,
                "dataset": run["dataset"],
                "variant": run["variant"],
                "seed": int(run["seed"]),
                "run_manifest": str(path.relative_to(REPO)),
                "run_manifest_sha256": sha256_file(path),
                "artifact_sha256": manifest["artifact_sha256"],
            }
        )
    locked_manifest_path = output / "locked_60_run_manifest.json"
    atomic_json(
        locked_manifest_path,
        {
            "schema_version": 1,
            "locked_before_any_semantic_label_access": True,
            "run_count": 60,
            "failures": 0,
            "preregistered_order_sha256": sha256_file(order_path),
            "config_lock_sha256": sha256_file(output / "config_lock.json"),
            "runs": locked_rows,
        },
    )
    atomic_json(
        output / "training_complete.json",
        {
            "schema_version": 1,
            "training_complete": True,
            "run_count": 60,
            "failure_count": 0,
            "locked_60_run_manifest_sha256": sha256_file(locked_manifest_path),
            "label_firewall_sha256": sha256_file(output / "training_label_firewall.json"),
            "semantic_label_values_read": False,
        },
    )
    atomic_json(output / "failure_index.json", {"schema_version": 1, "failures": []})
    print("TRAINING_LOCKED 60/60 manifest=%s" % sha256_file(locked_manifest_path), flush=True)


if __name__ == "__main__":
    main()
