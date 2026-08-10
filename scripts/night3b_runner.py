#!/usr/bin/env python3
"""Locked 120-run Night-3B trainer using only published Night-3AF caches."""

from __future__ import annotations

import csv
import gc
import json
import os
import resource
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

from SpaLORA.night3af_cache import load_cache
from SpaLORA.night3a_ige import LOSS_KEYS, input_sha256
from SpaLORA.night3b_ablation import (
    VARIANTS, Night3BTrainer, active_loss_mask, initial_weighted_gradient_shares,
    variant_contract,
)
from SpaLORA.night3b_protocol import (
    ScientificWindow, assert_training_payload_label_free, atomic_json,
    cache_directory, ground_truth_csv_paths, load_cache_index, sha256_file,
    training_cfg, verify_night3b_lock,
)


CONFIG_PATH = REPO / "configs/night3b_ablation_interpretability.json"
REQUIRED_RUN_FILES = (
    "embedding.npz", "attention.npz", "clusters.csv", "observation_ids.csv",
    "loss_trajectory.csv", "gradient_influence_trajectory.csv",
    "coefficient_probe.json", "checkpoint_index.csv", "run_manifest.json",
)


def fsync_path(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def write_csv(path: Path, rows: list) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
        handle.flush(); os.fsync(handle.fileno())


def run_dir(output: Path, dataset: str, variant: str, seed: int) -> Path:
    return output / "runs" / dataset / variant / ("seed_%d" % seed)


def existing_valid(directory: Path, identity: dict) -> bool:
    if not all((directory / name).is_file() for name in REQUIRED_RUN_FILES) or (directory / "failure.json").exists():
        return False
    try:
        manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
    except Exception:
        return False
    if any(manifest.get(key) != value for key, value in identity.items()):
        return False
    return all(
        (directory / name).is_file() and sha256_file(directory / name) == expected
        for name, expected in manifest.get("artifact_sha256", {}).items()
    )


def coefficient_payload(result, variant: str) -> dict:
    probe = result.probe or {}
    shares = initial_weighted_gradient_shares(probe, result.coefficients)
    return {
        "schema_version": 1,
        "variant": variant,
        "active_loss_mask": active_loss_mask(variant),
        "raw_initial_losses": result.initial_losses,
        "raw_rms_gradients": probe.get("gradients", {}),
        "frozen_coefficients": result.coefficients,
        "initial_weighted_gradient_shares": shares,
        "active_coefficient_sum": float(sum(
            result.coefficients[name] for name in LOSS_KEYS if active_loss_mask(variant)[name]
        )),
        "probe_state_unchanged": bool(probe.get("state_unchanged", False)),
        "probe_grad_fields_unchanged": bool(probe.get("grad_fields_unchanged", False)),
        "probe_rng_unchanged": bool(probe.get("rng_unchanged", False)),
        "initial_state_sha256": result.initial_state_sha256,
        "dynamic_coefficient_update": False,
        "semantic_label_access": False,
    }


def run_one(config: dict, lock: dict, output: Path, prepared_cache: dict,
            dataset: str, variant: str, seed: int, ordinal: int, cache_row: dict) -> Path:
    from scripts.night3a_runner import cluster_exact, validate_attention

    cfg = training_cfg(config["datasets"][dataset])
    prepared = prepared_cache[dataset]
    assert_training_payload_label_free(prepared.data, cfg, ground_truth_csv_paths(config))
    directory = run_dir(output, dataset, variant, seed)
    directory.mkdir(parents=True, exist_ok=True)
    identity = {
        "schema_version": 1, "dataset": dataset, "variant": variant, "seed": int(seed),
        "run_order_ordinal": int(ordinal),
        "config_lock_sha256": sha256_file(output / "config_lock.json"),
        "config_sha256": lock["config_sha256"],
    }
    if existing_valid(directory, identity):
        print("SKIP_VALID %03d/120 %s %s seed=%d" % (ordinal, dataset, variant, seed), flush=True)
        return directory / "run_manifest.json"
    existing = [path.name for path in directory.iterdir()]
    if existing:
        raise RuntimeError("Partial or mismatched run cannot be overwritten: %s %r" % (directory, existing))

    failure_path = directory / "failure.json"
    started = time.perf_counter()
    try:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        trainer = Night3BTrainer(prepared.data, cfg, variant, seed, torch.device("cuda:0"), config["ige_epsilon"])
        locked_input = input_sha256(prepared.data, prepared.obs_names, prepared.data["selected_gene_names"])
        if locked_input != cache_row["canonical_model_input_sha256"]:
            raise RuntimeError("Run did not consume the published cache")
        training_started = time.perf_counter()
        result = trainer.train(); torch.cuda.synchronize()
        training_seconds = time.perf_counter() - training_started
        attention_deviation = validate_attention(result.output, len(prepared.obs_names))

        embedding_path = directory / "embedding.npz"
        attention_path = directory / "attention.npz"
        ids_path = directory / "observation_ids.csv"
        cluster_path = directory / "clusters.csv"
        loss_path = directory / "loss_trajectory.csv"
        gradient_path = directory / "gradient_influence_trajectory.csv"
        coefficient_path = directory / "coefficient_probe.json"
        checkpoint_path = directory / "checkpoint_index.csv"
        np.savez_compressed(
            embedding_path, SpaLORA=np.asarray(result.output["SpaLORA"], np.float32),
            emb_latent_omics1=np.asarray(result.output["emb_latent_omics1"], np.float32),
            emb_latent_omics2=np.asarray(result.output["emb_latent_omics2"], np.float32),
        )
        np.savez_compressed(
            attention_path, alpha=np.asarray(result.output["alpha"], np.float32),
            alpha_omics1=np.asarray(result.output["alpha_omics1"], np.float32),
            alpha_omics2=np.asarray(result.output["alpha_omics2"], np.float32),
        )
        pd.DataFrame({"observation_id": prepared.obs_names.astype(str)}).to_csv(ids_path, index=False)
        write_csv(loss_path, result.logs)
        write_csv(gradient_path, result.gradient_logs)
        atomic_json(coefficient_path, coefficient_payload(result, variant))
        write_csv(checkpoint_path, [{
            "step": row["step"], "fraction_of_training": row["fraction_of_training"],
            "state_sha256": row["checkpoint_state_sha256"], "state_file_saved": False,
        } for row in result.logs])

        clustering_started = time.perf_counter()
        predicted = cluster_exact(np.asarray(result.output["SpaLORA"], np.float32),
                                  cfg["n_clusters"], config["clustering"]["random_seed"])
        clustering_seconds = time.perf_counter() - clustering_started
        if len(predicted) != len(prepared.obs_names) or len(np.unique(predicted)) != cfg["n_clusters"]:
            raise AssertionError("Clustering count or observation length mismatch")
        pd.DataFrame({"observation_id": prepared.obs_names.astype(str), "cluster": predicted}).to_csv(
            cluster_path, index=False
        )
        for path in (embedding_path, attention_path, ids_path, cluster_path, loss_path,
                     gradient_path, coefficient_path, checkpoint_path):
            fsync_path(path)

        artifacts = {
            name: sha256_file(directory / name)
            for name in REQUIRED_RUN_FILES if name != "run_manifest.json"
        }
        manifest = dict(identity)
        manifest.update({
            "source_commit_at_start": lock["parent_commit"],
            "model_source_sha256": lock["source_sha256"]["SpaLORA/model_corrected.py"],
            "night3b_source_lock_sha256": sha256_file(output / "config_lock.json"),
            "n_observations": int(len(prepared.obs_names)),
            "n_selected_genes": int(len(prepared.data["selected_gene_names"])),
            "n_clusters": int(len(np.unique(predicted))),
            "locked_input_sha256": locked_input,
            "deterministic_cache_manifest_sha256": cache_row["manifest_sha256"],
            "deterministic_cache_content_sha256": cache_row["canonical_cache_content_sha256"],
            "cache_directory": str(cache_directory(config, cache_row)),
            "initial_state_sha256": result.initial_state_sha256,
            "final_state_sha256": result.final_state_sha256,
            "active_loss_mask": active_loss_mask(variant),
            "attention_mode": variant_contract(variant)["attention_mode"],
            "frozen_coefficients": result.coefficients,
            "epochs": int(cfg["epochs"]),
            "optimizer": {"name": "Adam", "learning_rate": 1e-4, "weight_decay": 0.0},
            "artifact_sha256": artifacts,
            "attention_max_row_sum_deviation": attention_deviation,
            "timings": {
                "shared_cache_load_seconds": 0.0,
                "training_seconds": training_seconds,
                "clustering_seconds": clustering_seconds,
                "run_seconds_excluding_shared_cache_load": time.perf_counter() - started,
            },
            "resources": {
                "gpu_peak_allocated_mib": float(torch.cuda.max_memory_allocated() / 1024 ** 2),
                "gpu_peak_reserved_mib": float(torch.cuda.max_memory_reserved() / 1024 ** 2),
                "process_peak_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
            },
            "semantic_label_access": False,
            "diagnostics_state_neutral": all(
                row["diagnostic_parameter_state_unchanged"]
                and row["diagnostic_grad_fields_unchanged"]
                and row["diagnostic_rng_state_unchanged"]
                and row["diagnostic_optimizer_state_unchanged"]
                for row in result.gradient_logs
            ),
        })
        atomic_json(directory / "run_manifest.json", manifest)
        fsync_path(directory / "run_manifest.json")
        print("DONE %03d/120 %s %s seed=%d train=%.1fs total=%.1fs" % (
            ordinal, dataset, variant, seed, training_seconds, time.perf_counter() - started
        ), flush=True)
        return directory / "run_manifest.json"
    except Exception as exc:
        atomic_json(failure_path, {
            "dataset": dataset, "variant": variant, "seed": seed,
            "run_order_ordinal": ordinal, "error": repr(exc),
            "traceback": traceback.format_exc(),
        })
        raise
    finally:
        gc.collect(); torch.cuda.empty_cache()


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    output = REPO / config["paths"]["output_root"]
    lock = json.loads((output / "config_lock.json").read_text(encoding="utf-8"))
    verify_night3b_lock(REPO, CONFIG_PATH, config, lock, output, "training_120")
    gate = json.loads((output / "night3b_gate_status.json").read_text(encoding="utf-8"))
    if not (gate.get("p0_arch_pass") and gate.get("main_120_authorized")):
        raise RuntimeError("P0-ARCH did not authorize 120 runs")
    order_path = REPO / config["run_order"]["manifest"]
    order = json.loads(order_path.read_text(encoding="utf-8"))["runs"]
    if len(order) != 120 or len({(r["dataset"], r["variant"], r["seed"]) for r in order}) != 120:
        raise AssertionError("Locked order is not the exact 120-cell factorial")
    if [row["ordinal"] for row in order] != list(range(1, 121)):
        raise AssertionError("Locked ordinals are invalid")

    cache_index = load_cache_index(config)
    window = ScientificWindow(config, output, "training_120").install()
    completed = []
    try:
        prepared_cache = {}
        forbidden = ground_truth_csv_paths(config)
        for dataset in ("a1", "placenta", "p22"):
            row = cache_index["datasets"][dataset]
            prepared = load_cache(cache_directory(config, row), row["manifest_sha256"])
            assert_training_payload_label_free(prepared.data, training_cfg(config["datasets"][dataset]), forbidden)
            observed = input_sha256(prepared.data, prepared.obs_names, prepared.data["selected_gene_names"])
            if observed != row["canonical_model_input_sha256"]:
                raise RuntimeError("Published deterministic cache input hash mismatch")
            prepared_cache[dataset] = prepared
            print("CACHE_LOADED %s n=%d hash=%s" % (dataset, len(prepared.obs_names), observed), flush=True)

        for cell in order:
            completed.append(run_one(
                config, lock, output, prepared_cache,
                cell["dataset"], cell["variant"], int(cell["seed"]),
                int(cell["ordinal"]), cache_index["datasets"][cell["dataset"]],
            ))
        if len(completed) != 120:
            raise AssertionError("Expected 120 completed runs")

        locked_rows = []
        dataset_hashes = {dataset: set() for dataset in config["datasets"]}
        for cell in order:
            path = run_dir(output, cell["dataset"], cell["variant"], int(cell["seed"])) / "run_manifest.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            dataset_hashes[cell["dataset"]].add(manifest["locked_input_sha256"])
            locked_rows.append({
                "ordinal": int(cell["ordinal"]), "dataset": cell["dataset"],
                "variant": cell["variant"], "seed": int(cell["seed"]),
                "run_manifest": str(path.relative_to(REPO)),
                "run_manifest_sha256": sha256_file(path),
                "artifact_sha256": manifest["artifact_sha256"],
                "locked_input_sha256": manifest["locked_input_sha256"],
            })
        if any(len(values) != 1 for values in dataset_hashes.values()):
            raise RuntimeError("Runs did not use one immutable cache hash per dataset")
        locked_path = output / "locked_120_run_manifest.json"
        atomic_json(locked_path, {
            "schema_version": 1, "locked_before_any_semantic_label_access": True,
            "run_count": 120, "failures": 0,
            "preregistered_order_sha256": sha256_file(order_path),
            "config_lock_sha256": sha256_file(output / "config_lock.json"),
            "dataset_cache_hashes": {key: list(values)[0] for key, values in dataset_hashes.items()},
            "runs": locked_rows,
        })
        firewall = window.close(passed=True)
        if not firewall["passed"]:
            raise RuntimeError("Training label firewall failed")
        atomic_json(output / "training_complete.json", {
            "schema_version": 1, "training_complete": True,
            "run_count": 120, "failure_count": 0,
            "locked_120_run_manifest_sha256": sha256_file(locked_path),
            "scientific_window_firewall_sha256": sha256_file(output / "scientific_window_label_firewall.json"),
            "semantic_label_values_read": False,
        })
        atomic_json(output / "failure_index.json", {"schema_version": 1, "failures": []})
        print("TRAINING_LOCKED 120/120 manifest=%s" % sha256_file(locked_path), flush=True)
    except Exception:
        try:
            window.close(passed=False)
        except Exception:
            pass
        failures = sorted(str(path.relative_to(output)) for path in output.rglob("failure.json"))
        atomic_json(output / "failure_index.json", {"schema_version": 1, "failures": failures})
        raise


if __name__ == "__main__":
    main()
