#!/usr/bin/env python3
"""Night-3A-R 60-run label-free runner with weighted-gradient diagnostics."""

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

from SpaLORA.night3ar_protocol import (
    ScientificWindow, assert_training_payload_label_free, atomic_json,
    ground_truth_csv_paths, sha256_file, training_cfg, verify_lock,
)


CONFIG_PATH = REPO / "configs/night3ar_ige_feasibility.json"
REQUIRED_RUN_FILES = (
    "embedding.npz", "attention.npz", "observation_ids.csv", "clusters.csv",
    "loss_trajectory.csv", "gradient_influence_trajectory.csv", "checkpoint_index.csv",
    "run_config.json", "model_final.pt", "run_manifest.json",
)


def preprocessing_config(config: dict) -> dict:
    pre = config["preprocessing"]
    return {
        "min_cells": pre["min_cells"], "alpha": pre["alpha_compatibility_only"],
        "rescue_non_hvg": pre["rescue_non_hvg_compatibility_only"],
        "moran_shrinkage_tau": pre["moran_shrinkage_tau_compatibility_only"],
        "feature_graph": {"k": pre["feature_graph_k"], "metric": pre["feature_graph_metric"]},
    }


def fsync_path(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def write_csv(path: Path, rows: list) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        handle.flush(); os.fsync(handle.fileno())


def run_dir(output: Path, dataset: str, variant: str, seed: int) -> Path:
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


def run_one(config: dict, lock: dict, output: Path, prepared_cache: dict, prep_seconds: dict,
            dataset: str, variant: str, seed: int, ordinal: int) -> Path:
    from SpaLORA.night3a_ige import input_sha256, registered_variant_contract
    from SpaLORA.night3ar_ige import Night3ARTrainer
    from scripts.night3a_runner import cluster_exact, validate_attention

    full_cfg = config["datasets"][dataset]
    cfg = training_cfg(full_cfg)
    prepared = prepared_cache[dataset]
    assert_training_payload_label_free(prepared.data, cfg, ground_truth_csv_paths(config))
    directory = run_dir(output, dataset, variant, seed)
    directory.mkdir(parents=True, exist_ok=True)
    identity = {
        "schema_version": 1, "dataset": dataset, "variant": variant, "seed": seed,
        "run_order_ordinal": ordinal,
        "config_lock_sha256": sha256_file(output / "config_lock.json"),
        "config_sha256": lock["config_sha256"],
    }
    if existing_valid(directory, identity):
        print("SKIP_VALID %02d/60 %s %s seed=%d" % (ordinal, dataset, variant, seed), flush=True)
        return directory / "run_manifest.json"
    existing = [path.name for path in directory.iterdir()]
    if existing:
        raise RuntimeError("Partial/mismatched Night-3A-R run cannot be overwritten: %s %r" % (directory, existing))
    failure_path = directory / "failure.json"
    started = time.perf_counter()
    try:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        trainer = Night3ARTrainer(prepared.data, cfg, variant, seed, torch.device("cuda:0"), config["ige_epsilon"])
        locked_input = input_sha256(prepared.data, prepared.obs_names.astype(str), prepared.data["selected_gene_names"])
        training_started = time.perf_counter()
        result = trainer.train(); torch.cuda.synchronize()
        training_seconds = time.perf_counter() - training_started
        attention_deviation = validate_attention(result.output, len(prepared.obs_names))

        embedding_path = directory / "embedding.npz"
        attention_path = directory / "attention.npz"
        ids_path = directory / "observation_ids.csv"
        loss_path = directory / "loss_trajectory.csv"
        gradient_path = directory / "gradient_influence_trajectory.csv"
        checkpoint_path = directory / "checkpoint_index.csv"
        config_path = directory / "run_config.json"
        model_path = directory / "model_final.pt"
        cluster_path = directory / "clusters.csv"
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
        write_csv(checkpoint_path, [{
            "step": row["step"], "fraction_of_training": row["fraction_of_training"],
            "state_sha256": row["checkpoint_state_sha256"],
            "state_file_saved": row["step"] == int(cfg["epochs"]),
        } for row in result.logs])
        torch.save({
            "state_dict": {name: value.detach().cpu() for name, value in result.model.state_dict().items()},
            "dataset": dataset, "variant": variant, "seed": seed,
            "final_state_sha256": result.final_state_sha256,
        }, model_path)
        atomic_json(config_path, {
            "schema_version": 1, "dataset": dataset, "variant": variant, "seed": seed,
            "epochs": cfg["epochs"], "embedding_dim": cfg["embedding_dim"],
            "optimizer": {"name": "Adam", "learning_rate": 1e-4, "weight_decay": 0.0},
            "raw_loss_reduction": "torch.nn.functional.mse_loss mean over every element",
            "variant_contract": registered_variant_contract(config)[variant],
            "frozen_coefficients": result.coefficients, "initial_raw_losses": result.initial_losses,
            "gradient_influence_definition": "abs(frozen_coefficient_k) * RMS_gradient(raw_loss_k)",
            "scalar_contribution_is_descriptive_only": True,
            "ground_truth_path_in_trainer": False, "semantic_label_access": False,
            "config_lock_sha256": identity["config_lock_sha256"],
        })
        for path in (embedding_path, attention_path, ids_path, model_path, config_path): fsync_path(path)

        clustering_started = time.perf_counter()
        predicted = cluster_exact(np.asarray(result.output["SpaLORA"], np.float32),
                                  cfg["n_clusters"], config["clustering"]["random_seed"])
        clustering_seconds = time.perf_counter() - clustering_started
        if len(predicted) != len(prepared.obs_names) or len(np.unique(predicted)) != cfg["n_clusters"]:
            raise AssertionError("Clustering count/length mismatch")
        pd.DataFrame({"observation_id": prepared.obs_names.astype(str), "cluster": predicted}).to_csv(
            cluster_path, index=False
        ); fsync_path(cluster_path)

        artifacts = {name: sha256_file(directory / name) for name in REQUIRED_RUN_FILES if name != "run_manifest.json"}
        manifest = dict(identity)
        manifest.update({
            "n_observations": int(len(prepared.obs_names)),
            "n_selected_genes": int(len(prepared.data["selected_gene_names"])),
            "n_clusters": int(len(np.unique(predicted))), "locked_input_sha256": locked_input,
            "initial_state_sha256": result.initial_state_sha256,
            "final_state_sha256": result.final_state_sha256,
            "artifact_sha256": artifacts, "attention_max_row_sum_deviation": attention_deviation,
            "timings": {
                "preprocessing_cache_seconds": prep_seconds[dataset],
                "training_seconds": training_seconds, "clustering_seconds": clustering_seconds,
                "run_seconds_excluding_shared_preprocessing": time.perf_counter() - started,
            },
            "resources": {
                "gpu_peak_allocated_mib": float(torch.cuda.max_memory_allocated() / 1024 ** 2),
                "gpu_peak_reserved_mib": float(torch.cuda.max_memory_reserved() / 1024 ** 2),
                "process_peak_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
            },
            "semantic_label_access": False,
            "diagnostics_state_neutral": all(
                row["diagnostic_parameter_state_unchanged"] and row["diagnostic_grad_fields_unchanged"]
                and row["diagnostic_rng_state_unchanged"] and row["diagnostic_optimizer_state_unchanged"]
                for row in result.gradient_logs
            ),
        })
        atomic_json(directory / "run_manifest.json", manifest); fsync_path(directory / "run_manifest.json")
        print("DONE %02d/60 %s %s seed=%d train=%.1fs total=%.1fs" %
              (ordinal, dataset, variant, seed, training_seconds, time.perf_counter() - started), flush=True)
        return directory / "run_manifest.json"
    except Exception as exc:
        atomic_json(failure_path, {
            "dataset": dataset, "variant": variant, "seed": seed, "run_order_ordinal": ordinal,
            "error": repr(exc), "traceback": traceback.format_exc(),
        })
        raise
    finally:
        gc.collect(); torch.cuda.empty_cache()


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    output = REPO / config["paths"]["output_root"]
    lock = json.loads((output / "config_lock.json").read_text(encoding="utf-8"))
    verify_lock(REPO, CONFIG_PATH, config, lock, output, "training_60")
    gate = json.loads((output / "night3ar_gate_status.json").read_text(encoding="utf-8"))
    if not (gate.get("p0ar_pass") and gate.get("p0br_pass") and gate.get("main_60_authorized")):
        raise RuntimeError("P0A-R/P0B-R did not authorize main training")
    order_path = REPO / config["run_order"]["manifest"]
    order = json.loads(order_path.read_text(encoding="utf-8"))["runs"]
    if len(order) != 60 or len({(x["dataset"], x["variant"], x["seed"]) for x in order}) != 60:
        raise AssertionError("Locked order is not the exact 60-cell factorial")

    window = ScientificWindow(config, output, "training_60").install()
    completed = []
    try:
        from SpaLORA.night1_pipeline import prepare_corrected

        prepared_cache, prep_seconds = {}, {}
        pre_cfg = preprocessing_config(config)
        for dataset, cfg in config["datasets"].items():
            started = time.perf_counter()
            prepared_cache[dataset] = prepare_corrected(dataset, cfg, pre_cfg, "corrected_unweighted")
            assert_training_payload_label_free(
                prepared_cache[dataset].data, training_cfg(cfg), ground_truth_csv_paths(config)
            )
            prep_seconds[dataset] = time.perf_counter() - started
            print("PREPARED %s n=%d seconds=%.1f" %
                  (dataset, len(prepared_cache[dataset].obs_names), prep_seconds[dataset]), flush=True)
        for ordinal, cell in enumerate(order, 1):
            completed.append(run_one(
                config, lock, output, prepared_cache, prep_seconds,
                cell["dataset"], cell["variant"], int(cell["seed"]), ordinal,
            ))
        if len(completed) != 60:
            raise AssertionError("Expected 60 completed runs")
        locked_rows = []
        for ordinal, cell in enumerate(order, 1):
            path = run_dir(output, cell["dataset"], cell["variant"], int(cell["seed"])) / "run_manifest.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            locked_rows.append({
                "ordinal": ordinal, "dataset": cell["dataset"], "variant": cell["variant"],
                "seed": int(cell["seed"]), "run_manifest": str(path.relative_to(REPO)),
                "run_manifest_sha256": sha256_file(path), "artifact_sha256": manifest["artifact_sha256"],
            })
        locked_path = output / "locked_60_run_manifest.json"
        atomic_json(locked_path, {
            "schema_version": 1, "locked_before_any_semantic_label_access": True,
            "run_count": 60, "failures": 0,
            "preregistered_order_sha256": sha256_file(order_path),
            "config_lock_sha256": sha256_file(output / "config_lock.json"), "runs": locked_rows,
        })
        firewall = window.close(passed=True)
        if not firewall["passed"]:
            raise RuntimeError("Training scientific-window firewall failed")
        atomic_json(output / "training_complete.json", {
            "schema_version": 1, "training_complete": True, "run_count": 60, "failure_count": 0,
            "locked_60_run_manifest_sha256": sha256_file(locked_path),
            "scientific_window_firewall_sha256": sha256_file(output / "scientific_window_label_firewall.json"),
            "semantic_label_values_read": False,
        })
        atomic_json(output / "failure_index.json", {"schema_version": 1, "failures": []})
        print("TRAINING_LOCKED 60/60 manifest=%s" % sha256_file(locked_path), flush=True)
    except Exception:
        try: window.close(passed=False)
        except Exception: pass
        failures = sorted(str(path.relative_to(output)) for path in output.rglob("failure.json"))
        atomic_json(output / "failure_index.json", {"schema_version": 1, "failures": failures})
        raise


if __name__ == "__main__":
    main()
