#!/usr/bin/env python3
"""Night-3A-F P0B-F: 15 IGE probes on the published deterministic cache."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))

from SpaLORA.night3af_cache import load_cache
from SpaLORA.night3af_protocol import (
    ScientificWindow, assert_training_payload_label_free, atomic_json,
    ground_truth_csv_paths, load_cache_index, sha256_file, training_cfg,
    verify_night3af_lock,
)


CONFIG_PATH = REPO / "configs/night3af_deterministic_pca.json"


def write_csv(path: Path, rows: list) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    output = REPO / config["paths"]["output_root"]
    p0d = json.loads((output / "night3af_p0d.json").read_text(encoding="utf-8"))
    if not p0d.get("passed"): raise RuntimeError("P0D did not authorize P0B-F")
    lock_path = output / "config_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    reads = verify_night3af_lock(REPO, CONFIG_PATH, config, lock, output, "p0bf")
    cache_index = load_cache_index(output)
    window = ScientificWindow(config, output, "p0bf").install()
    failures, cells, gradient_rows, weight_rows, parameter_rows = [], [], [], [], []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    try:
        from SpaLORA.night3a_ige import (
            LOSS_KEYS, Night3ATrainer, input_sha256, legacy_m_bad,
            model_state_sha256, run_initial_probe,
        )
        forbidden = ground_truth_csv_paths(config)
        for dataset, full_cfg in config["datasets"].items():
            cache_row = cache_index["datasets"][dataset]
            prepared = load_cache(REPO / cache_row["directory"], cache_row["manifest_sha256"])
            cfg = training_cfg(full_cfg)
            assert_training_payload_label_free(prepared.data, cfg, forbidden)
            cache_hash = input_sha256(prepared.data, prepared.obs_names, prepared.data["selected_gene_names"])
            if cache_hash != cache_row["canonical_model_input_sha256"]:
                raise RuntimeError("Published cache hash mismatch")
            for seed in config["seeds"]:
                trainer = Night3ATrainer(prepared.data, cfg, "IGE", seed, device, config["ige_epsilon"])
                model = trainer.new_model(); initial_hash = model_state_sha256(model)
                probe = run_initial_probe(model, trainer.forward, config["ige_epsilon"])
                c0 = Night3ATrainer(prepared.data, cfg, "C0", seed, device, config["ige_epsilon"])
                c0_hash = model_state_sha256(c0.new_model())
                gradients = np.asarray([probe["gradients"][name] for name in LOSS_KEYS], np.float64)
                weights = np.asarray([probe["weights"][name] for name in LOSS_KEYS], np.float64)
                q = weights * gradients; shares = q / q.sum()
                m_bad = float(legacy_m_bad(trainer.features1).detach().cpu())
                q_tol = (config["p0b_equivalence_margin"] * config["p0b_float32_epsilon_multiplier"]
                         * float(torch.finfo(torch.float32).eps) * max(1.0, float(np.max(q))))
                checks = {
                    "finite_positive_gradients": bool(np.all(np.isfinite(gradients)) and np.all(gradients > 0)),
                    "finite_positive_weights": bool(np.all(np.isfinite(weights)) and np.all(weights > 0)),
                    "weight_range": bool(np.all(weights >= config["ige_weight_min"]) and np.all(weights <= config["ige_weight_max"])),
                    "weight_sum": abs(float(weights.sum()) - config["ige_weight_sum"]) <= config["ige_weight_sum_tolerance"],
                    "initial_weighted_influence_equal": bool(np.max(np.abs(q - q.mean())) <= q_tol),
                    "initial_gradient_shares_quarter": bool(np.max(np.abs(shares - .25)) <= 1e-5),
                    "initial_parameters_identical": initial_hash == c0_hash,
                    "state_rng_preserved": bool(probe["state_unchanged"] and probe["grad_fields_unchanged"] and probe["rng_unchanged"]),
                    "repeat_within_gpu_envelope": bool(probe["repeat_within_gpu_envelope"]),
                    "reload_within_gpu_envelope": bool(probe["reload_forward_within_envelope"]),
                    "cache_hash_exact": cache_hash == cache_row["canonical_model_input_sha256"],
                    "locked_m_bad_matches": abs(m_bad - float(cfg["locked_m_bad_expected"])) <= 1e-6,
                }
                passed = all(checks.values())
                if not passed:
                    failures.append("%s seed %d: %r" % (dataset, seed, [k for k, v in checks.items() if not v]))
                for i, name in enumerate(LOSS_KEYS):
                    gradient_rows.append({"dataset": dataset, "seed": seed, "loss": name,
                                          "raw_initial_loss": probe["raw_losses"][name],
                                          "rms_gradient": probe["gradients"][name]})
                    weight_rows.append({"dataset": dataset, "seed": seed, "loss": name,
                                        "ige_weight": probe["weights"][name],
                                        "weighted_gradient_influence": float(q[i]),
                                        "weighted_gradient_share": float(shares[i])})
                for row in probe["gradient_parameter_rows"]:
                    parameter_rows.append({"dataset": dataset, "seed": seed, **row})
                cells.append({"dataset": dataset, "seed": seed, "passed": passed, "checks": checks,
                              "cache_model_input_sha256": cache_hash,
                              "ige_initial_state_sha256": initial_hash, "c0_initial_state_sha256": c0_hash,
                              "m_bad": m_bad, "raw_losses": probe["raw_losses"],
                              "gradient_values": probe["gradients"], "ige_weights": probe["weights"],
                              "initial_weighted_gradient_share": {name: float(shares[i]) for i, name in enumerate(LOSS_KEYS)}})
                del trainer, model, c0, probe
                if torch.cuda.is_available(): torch.cuda.empty_cache()
        firewall = window.close(passed=not failures)
    except Exception as exc:
        failures.append("scientific-window exception: %r" % exc)
        firewall = window.close(passed=False)
    if not firewall["passed"]: failures.append("P0B-F label firewall failed")
    write_csv(output / "ige_initial_gradients.csv", gradient_rows)
    write_csv(output / "ige_gradient_parameters.csv", parameter_rows)
    write_csv(output / "ige_weights.csv", weight_rows)
    p0bf = {
        "schema_version": 1, "stage": "P0B-F", "passed": not failures,
        "cells_completed": len(cells), "cells_required": 15,
        "cells_passed": sum(row["passed"] for row in cells), "failures": failures,
        "cells": cells, "integrity_read_count": len(reads),
        "scientific_window_firewall": firewall, "semantic_label_values_read": False,
        "config_lock_sha256": sha256_file(lock_path),
        "old_probe_equivalence_not_required": True,
    }
    atomic_json(output / "night3af_p0b.json", p0bf)
    atomic_json(output / "night3af_gate_status.json", {
        "schema_version": 1, "p0d_pass": True, "p0bf_pass": p0bf["passed"],
        "main_60_authorized": p0bf["passed"],
        "previous_night3ar_status": config["previous_night3ar_status"],
        "night3af_p0b_sha256": sha256_file(output / "night3af_p0b.json"),
    })
    atomic_json(output / "failure_index.json", {"schema_version": 1, "failures": [] if not failures else [p0bf]})
    if failures:
        atomic_json(output / "night3af_p0b_failure.json", p0bf)
        print("P0B_F_FAIL", json.dumps(failures, indent=2)); raise SystemExit(2)
    print("P0B_F_PASS cells=15 cache_hashes=3 scientific_window=PASS", flush=True)


if __name__ == "__main__":
    main()
