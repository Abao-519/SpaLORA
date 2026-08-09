#!/usr/bin/env python3
"""Run the preregistered 3-dataset x 5-seed Night-3A IGE hard gate."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CONFIG_PATH = REPO / "configs/night3a_ige_feasibility.json"


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


def verify_lock(config: dict, lock: dict) -> list:
    failures = []
    if sha256_file(CONFIG_PATH) != lock.get("config_sha256"):
        failures.append("config SHA drift after P0A")
    for name, expected in lock.get("source_sha256", {}).items():
        path = REPO / name
        if not path.is_file() or sha256_file(path) != expected:
            failures.append("source SHA drift: %s" % name)
    for name, expected in lock.get("data_sha256", {}).items():
        path = Path(name)
        if not path.is_file() or sha256_file(path) != expected:
            failures.append("data SHA drift: %s" % name)
    order = REPO / config["run_order"]["manifest"]
    manifest = REPO / config["paths"]["output_root"] / "data_manifest.csv"
    if not order.is_file() or sha256_file(order) != lock.get("run_order_sha256"):
        failures.append("run order SHA drift")
    if not manifest.is_file() or sha256_file(manifest) != lock.get("data_manifest_sha256"):
        failures.append("data manifest SHA drift")
    return failures


def preprocessing_config(config: dict) -> dict:
    pre = config["preprocessing"]
    return {
        "min_cells": pre["min_cells"],
        "alpha": pre["alpha_compatibility_only"],
        "rescue_non_hvg": pre["rescue_non_hvg_compatibility_only"],
        "moran_shrinkage_tau": pre["moran_shrinkage_tau_compatibility_only"],
        "feature_graph": {"k": pre["feature_graph_k"], "metric": pre["feature_graph_metric"]},
    }


def write_csv(path: Path, rows: list) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


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
    from SpaLORA.night3a_ige import (
        LOSS_KEYS, Night3ATrainer, input_sha256, legacy_m_bad, model_state_sha256,
        run_initial_probe,
    )

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    output = REPO / config["paths"]["output_root"]
    lock_path = output / "config_lock.json"
    if not lock_path.is_file():
        raise RuntimeError("P0A config lock is missing")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    failures = verify_lock(config, lock)
    if failures:
        raise RuntimeError("P0A lock failed before P0B: %r" % failures)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gradient_rows = []
    parameter_rows = []
    weight_rows = []
    cells = []
    pre_cfg = preprocessing_config(config)
    for dataset, cfg in config["datasets"].items():
        prepared = prepare_corrected(dataset, cfg, pre_cfg, "corrected_unweighted")
        prepared_input_hash = input_sha256(
            prepared.data, prepared.obs_names.astype(str), prepared.data["selected_gene_names"]
        )
        for seed in config["seeds"]:
            trainer = Night3ATrainer(prepared.data, cfg, "IGE", seed, device, config["ige_epsilon"])
            model = trainer.new_model()
            ige_initial_hash = model_state_sha256(model)
            probe = run_initial_probe(model, trainer.forward, config["ige_epsilon"])
            c0 = Night3ATrainer(prepared.data, cfg, "C0", seed, device, config["ige_epsilon"])
            c0_model = c0.new_model()
            c0_initial_hash = model_state_sha256(c0_model)
            c0_input_hash = input_sha256(
                prepared.data, prepared.obs_names.astype(str), prepared.data["selected_gene_names"]
            )
            m_bad = float(legacy_m_bad(trainer.features1).detach().cpu())
            weights = np.asarray([probe["weights"][name] for name in LOSS_KEYS], dtype=np.float64)
            gradients = np.asarray([probe["gradients"][name] for name in LOSS_KEYS], dtype=np.float64)
            checks = {
                "finite_positive_gradients": bool(np.all(np.isfinite(gradients)) and np.all(gradients > 0)),
                "finite_positive_weights": bool(np.all(np.isfinite(weights)) and np.all(weights > 0)),
                "weight_range": bool(
                    np.all(weights >= config["ige_weight_min"]) and np.all(weights <= config["ige_weight_max"])
                ),
                "weight_sum": bool(
                    abs(float(weights.sum()) - config["ige_weight_sum"]) <= config["ige_weight_sum_tolerance"]
                ),
                "probe_repeat_stable": bool(
                    probe["repeat_exact"] if device.type == "cpu" else probe["repeat_within_gpu_envelope"]
                ),
                "probe_state_unchanged": bool(probe["state_unchanged"]),
                "probe_grad_fields_unchanged": bool(probe["grad_fields_unchanged"]),
                "probe_rng_unchanged": bool(probe["rng_unchanged"]),
                "reload_forward_within_envelope": bool(probe["reload_forward_within_envelope"]),
                "initial_parameters_identical": ige_initial_hash == c0_initial_hash,
                "inputs_graphs_order_identical": prepared_input_hash == c0_input_hash,
                "locked_m_bad_matches": abs(m_bad - float(cfg["locked_m_bad_expected"])) <= 1e-6,
            }
            passed = all(checks.values())
            if not passed:
                failures.append("%s seed %d: %r" % (dataset, seed, [k for k, v in checks.items() if not v]))
            for name in LOSS_KEYS:
                gradient_rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "loss": name,
                        "raw_initial_loss": probe["raw_losses"][name],
                        "rms_gradient": probe["gradients"][name],
                        "repeat_rms_gradient": probe["repeat_gradients"][name],
                        "repeat_max_abs_difference_all_losses": probe["repeat_gradient_max_abs_difference"],
                    }
                )
                weight_rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "loss": name,
                        "ige_weight": probe["weights"][name],
                        "repeat_ige_weight": probe["repeat_weights"][name],
                        "weight_sum": float(weights.sum()),
                    }
                )
            for row in probe["gradient_parameter_rows"]:
                parameter_rows.append({"dataset": dataset, "seed": seed, "repeat": 1, **row})
            for row in probe["repeat_gradient_parameter_rows"]:
                parameter_rows.append({"dataset": dataset, "seed": seed, "repeat": 2, **row})
            cells.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "device": str(device),
                    "passed": passed,
                    "checks": checks,
                    "input_sha256": prepared_input_hash,
                    "ige_initial_state_sha256": ige_initial_hash,
                    "c0_initial_state_sha256": c0_initial_hash,
                    "m_bad": m_bad,
                    "gradient_values": probe["gradients"],
                    "ige_weights": probe["weights"],
                    "repeat_exact": probe["repeat_exact"],
                    "repeat_within_gpu_envelope": probe["repeat_within_gpu_envelope"],
                    "reload_forward_max_abs_difference": probe["reload_forward_max_abs_difference"],
                    "reload_forward_envelope": probe["reload_forward_envelope"],
                    "rng_before_sha256": probe["rng_before_sha256"],
                    "rng_after_sha256": probe["rng_after_sha256"],
                }
            )
            del trainer, model, c0, c0_model, probe
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    forbidden_paths = {
        str(Path(cfg["ground_truth"]).resolve())
        for cfg in config["datasets"].values()
        if cfg["ground_truth"].startswith("/")
    }
    forbidden_opened = sorted(forbidden_paths.intersection(set(opened_paths)))
    forbidden_imported = sorted(
        name for name in config["label_firewall"]["training_must_not_import"] if name in sys.modules
    )
    if forbidden_opened:
        failures.append("ground-truth file opened during P0B: %r" % forbidden_opened)
    if forbidden_imported:
        failures.append("forbidden evaluation module imported during P0B: %r" % forbidden_imported)

    write_csv(output / "ige_initial_gradients.csv", gradient_rows)
    write_csv(output / "ige_gradient_parameters.csv", parameter_rows)
    write_csv(output / "ige_weights.csv", weight_rows)
    p0b = {
        "schema_version": 1,
        "stage": "P0B",
        "passed": len(failures) == 0,
        "cells_completed": len(cells),
        "cells_required": 15,
        "failures": failures,
        "cells": cells,
        "label_firewall": {
            "ground_truth_files_opened": forbidden_opened,
            "forbidden_modules_imported": forbidden_imported,
            "semantic_label_values_read": False,
            "opened_path_count": len(set(opened_paths)),
        },
        "config_lock_sha256": sha256_file(lock_path),
    }
    atomic_json(output / "night3a_p0b.json", p0b)
    gate = {
        "schema_version": 1,
        "p0a_pass": True,
        "p0b_pass": p0b["passed"],
        "main_60_authorized": p0b["passed"],
        "config_lock_sha256": sha256_file(lock_path),
        "night3a_p0b_sha256": sha256_file(output / "night3a_p0b.json"),
    }
    atomic_json(output / "night3a_gate_status.json", gate)
    if failures:
        atomic_json(output / "night3a_p0b_failure.json", p0b)
        atomic_json(output / "failure_index.json", {"schema_version": 1, "failures": [p0b]})
        print("P0B_FAIL", json.dumps(failures, indent=2))
        raise SystemExit(2)
    print("P0B_PASS cells=15 gradients=60 weights=60")


if __name__ == "__main__":
    main()
