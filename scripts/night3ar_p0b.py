#!/usr/bin/env python3
"""Night-3A-R P0B-R probes with corrected scientific-window semantics."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from SpaLORA.night3ar_protocol import (
    ScientificWindow, assert_training_payload_label_free, atomic_json,
    ground_truth_csv_paths, sha256_file, training_cfg, verify_lock,
)


CONFIG_PATH = REPO / "configs/night3ar_ige_feasibility.json"


def preprocessing_config(config: dict) -> dict:
    pre = config["preprocessing"]
    return {
        "min_cells": pre["min_cells"], "alpha": pre["alpha_compatibility_only"],
        "rescue_non_hvg": pre["rescue_non_hvg_compatibility_only"],
        "moran_shrinkage_tau": pre["moran_shrinkage_tau_compatibility_only"],
        "feature_graph": {"k": pre["feature_graph_k"], "metric": pre["feature_graph_metric"]},
    }


def write_csv(path: Path, rows: list) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def within_envelope(old: float, new: float, config: dict) -> bool:
    tolerance = (
        config["p0b_equivalence_margin"] * config["p0b_float32_epsilon_multiplier"]
        * float(torch.finfo(torch.float32).eps) * max(1.0, abs(float(old)), abs(float(new)))
    )
    return abs(float(old) - float(new)) <= tolerance


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    output = REPO / config["paths"]["output_root"]
    lock_path = output / "config_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    integrity_reads = verify_lock(REPO, CONFIG_PATH, config, lock, output, "p0br")
    old_p0b = json.loads((REPO / config["paths"]["old_output_root"] / "night3a_p0b.json").read_text(encoding="utf-8"))
    old_cells = {(row["dataset"], int(row["seed"])): row for row in old_p0b["cells"]}
    old_gradients = {}
    old_weights = {}
    old_raw_losses = {}
    for row in old_p0b["cells"]:
        old_gradients[(row["dataset"], int(row["seed"]))] = row["gradient_values"]
        old_weights[(row["dataset"], int(row["seed"]))] = row["ige_weights"]
    with (REPO / config["paths"]["old_output_root"] / "ige_initial_gradients.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        for row in csv.DictReader(handle):
            old_raw_losses.setdefault((row["dataset"], int(row["seed"])), {})[row["loss"]] = float(row["raw_initial_loss"])

    window = ScientificWindow(config, output, "p0br").install()
    failures = []
    cells, gradient_rows, weight_rows, parameter_rows = [], [], [], []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    try:
        from SpaLORA.night1_pipeline import prepare_corrected
        from SpaLORA.night3a_ige import (
            LOSS_KEYS, Night3ATrainer, input_sha256, legacy_m_bad,
            model_state_sha256, run_initial_probe,
        )

        pre_cfg = preprocessing_config(config)
        forbidden = ground_truth_csv_paths(config)
        for dataset, full_cfg in config["datasets"].items():
            prepared = prepare_corrected(dataset, full_cfg, pre_cfg, "corrected_unweighted")
            cfg = training_cfg(full_cfg)
            assert_training_payload_label_free(prepared.data, cfg, forbidden)
            prepared_input_hash = input_sha256(
                prepared.data, prepared.obs_names.astype(str), prepared.data["selected_gene_names"]
            )
            for seed in config["seeds"]:
                trainer = Night3ATrainer(prepared.data, cfg, "IGE", seed, device, config["ige_epsilon"])
                model = trainer.new_model()
                initial_hash = model_state_sha256(model)
                probe = run_initial_probe(model, trainer.forward, config["ige_epsilon"])
                c0 = Night3ATrainer(prepared.data, cfg, "C0", seed, device, config["ige_epsilon"])
                c0_model = c0.new_model()
                c0_hash = model_state_sha256(c0_model)
                m_bad = float(legacy_m_bad(trainer.features1).detach().cpu())
                weights = np.asarray([probe["weights"][name] for name in LOSS_KEYS], dtype=np.float64)
                gradients = np.asarray([probe["gradients"][name] for name in LOSS_KEYS], dtype=np.float64)
                q = weights * gradients
                shares = q / q.sum()
                old = old_cells[(dataset, seed)]
                old_raw = old_raw_losses[(dataset, seed)]
                comparison_fields = {
                    "input_sha_exact": prepared_input_hash == old["input_sha256"],
                    "initial_state_sha_exact": initial_hash == old["ige_initial_state_sha256"],
                    "c0_initial_state_sha_exact": c0_hash == old["c0_initial_state_sha256"],
                    "m_bad_within_envelope": within_envelope(old["m_bad"], m_bad, config),
                    "raw_initial_losses_within_envelope": all(
                        within_envelope(old_raw[name], probe["raw_losses"][name], config) for name in LOSS_KEYS
                    ),
                    "rms_gradients_within_envelope": all(
                        within_envelope(old_gradients[(dataset, seed)][name], probe["gradients"][name], config)
                        for name in LOSS_KEYS
                    ),
                    "ige_weights_within_envelope": all(
                        within_envelope(old_weights[(dataset, seed)][name], probe["weights"][name], config)
                        for name in LOSS_KEYS
                    ),
                    "repeatability_equal": bool(old["repeat_exact"] == probe["repeat_exact"]),
                    "reload_envelope_pass": bool(probe["reload_forward_within_envelope"]),
                    "state_rng_preserved": bool(
                        probe["state_unchanged"] and probe["grad_fields_unchanged"] and probe["rng_unchanged"]
                    ),
                }
                checks = {
                    "finite_positive_gradients": bool(np.all(np.isfinite(gradients)) and np.all(gradients > 0)),
                    "finite_positive_weights": bool(np.all(np.isfinite(weights)) and np.all(weights > 0)),
                    "weight_range": bool(np.all(weights >= config["ige_weight_min"]) and np.all(weights <= config["ige_weight_max"])),
                    "weight_sum": bool(abs(float(weights.sum()) - config["ige_weight_sum"]) <= config["ige_weight_sum_tolerance"]),
                    "initial_weighted_influence_equal": bool(
                        np.max(np.abs(q - q.mean())) <= config["p0b_equivalence_margin"]
                        * config["p0b_float32_epsilon_multiplier"] * torch.finfo(torch.float32).eps
                        * max(1.0, float(np.max(q)))
                    ),
                    "initial_gradient_shares_quarter": bool(np.max(np.abs(shares - 0.25)) <= 1e-5),
                    "initial_parameters_identical": initial_hash == c0_hash,
                    "locked_m_bad_matches": abs(m_bad - float(cfg["locked_m_bad_expected"])) <= 1e-6,
                }
                passed = all(checks.values()) and all(comparison_fields.values())
                if not passed:
                    failures.append("%s seed %d failed %r" % (
                        dataset, seed,
                        [name for name, value in {**checks, **comparison_fields}.items() if not value],
                    ))
                for index, name in enumerate(LOSS_KEYS):
                    gradient_rows.append({
                        "dataset": dataset, "seed": seed, "loss": name,
                        "raw_initial_loss": probe["raw_losses"][name],
                        "rms_gradient": probe["gradients"][name],
                        "old_rms_gradient": old_gradients[(dataset, seed)][name],
                        "within_old_envelope": within_envelope(old_gradients[(dataset, seed)][name], probe["gradients"][name], config),
                    })
                    weight_rows.append({
                        "dataset": dataset, "seed": seed, "loss": name,
                        "ige_weight": probe["weights"][name],
                        "old_ige_weight": old_weights[(dataset, seed)][name],
                        "weighted_gradient_influence": float(q[index]),
                        "weighted_gradient_share": float(shares[index]),
                        "within_old_envelope": within_envelope(old_weights[(dataset, seed)][name], probe["weights"][name], config),
                    })
                for row in probe["gradient_parameter_rows"]:
                    parameter_rows.append({"dataset": dataset, "seed": seed, **row})
                cells.append({
                    "dataset": dataset, "seed": seed, "passed": passed,
                    "checks": checks, "old_probe_comparison": comparison_fields,
                    "input_sha256": prepared_input_hash,
                    "ige_initial_state_sha256": initial_hash,
                    "c0_initial_state_sha256": c0_hash,
                    "m_bad": m_bad, "raw_losses": probe["raw_losses"],
                    "gradient_values": probe["gradients"], "ige_weights": probe["weights"],
                    "initial_weighted_gradient_influence": {name: float(q[i]) for i, name in enumerate(LOSS_KEYS)},
                    "initial_weighted_gradient_share": {name: float(shares[i]) for i, name in enumerate(LOSS_KEYS)},
                    "repeat_exact": probe["repeat_exact"],
                    "repeat_within_gpu_envelope": probe["repeat_within_gpu_envelope"],
                    "reload_forward_within_envelope": probe["reload_forward_within_envelope"],
                })
                del trainer, model, c0, c0_model, probe
                if torch.cuda.is_available(): torch.cuda.empty_cache()
        window_payload = window.close(passed=not failures)
    except Exception as exc:
        failures.append("scientific window exception: %r" % exc)
        window_payload = window.close(passed=False)

    if not window_payload["passed"]:
        failures.append("P0B-R scientific-window firewall failed")
    write_csv(output / "ige_initial_gradients.csv", gradient_rows)
    write_csv(output / "ige_gradient_parameters.csv", parameter_rows)
    write_csv(output / "ige_weights.csv", weight_rows)
    p0br = {
        "schema_version": 1, "stage": "P0B-R", "passed": not failures,
        "cells_completed": len(cells), "cells_required": 15,
        "old_probe_cells_within_envelope": sum(all(row["old_probe_comparison"].values()) for row in cells),
        "failures": failures, "cells": cells,
        "integrity_read_count": len(integrity_reads),
        "scientific_window_firewall": window_payload,
        "semantic_label_values_read": False,
        "config_lock_sha256": sha256_file(lock_path),
    }
    atomic_json(output / "night3ar_p0b.json", p0br)
    gate = {
        "schema_version": 1, "p0ar_pass": True, "p0br_pass": p0br["passed"],
        "main_60_authorized": p0br["passed"],
        "previous_night3a_status": config["previous_night3a_status"],
        "config_lock_sha256": sha256_file(lock_path),
        "night3ar_p0b_sha256": sha256_file(output / "night3ar_p0b.json"),
    }
    atomic_json(output / "night3ar_gate_status.json", gate)
    if failures:
        atomic_json(output / "night3ar_p0b_failure.json", p0br)
        atomic_json(output / "failure_index.json", {"schema_version": 1, "failures": [p0br]})
        print("P0B_R_FAIL", json.dumps(failures, indent=2)); raise SystemExit(2)
    print("P0B_R_PASS cells=15 old_envelope=15 scientific_window=PASS")


if __name__ == "__main__":
    main()
