#!/usr/bin/env python3
"""Label-free Night-2C P0C numerical-equivalence authorization gate."""

from __future__ import annotations

import csv
import gc
import json
import math
import os
import subprocess
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


os.environ.setdefault("R_HOME", "/opt/R/4.0.3/lib/R")
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from SpaLORA.model import Encoder_overall
from SpaLORA.night2c_loss_audit import (
    ParityLockedTrainer,
    atomic_json,
    critical_hashes,
    environment_payload,
)
from SpaLORA.preprocess import fix_seed
from SpaLORA.SpaLORA_pyG import Train_SpaLORA
from scripts.night1_benchmark import prepare_legacy
from scripts.night2b_parity_locked_audit import consumed_input_audit, frozen_components


DATASETS = ("a1", "placenta", "p22")
PAIRS = (("L1", "L2", "within_legacy"), ("G1", "G2", "within_generalized"),
         ("L1", "G1", "cross"), ("L1", "G2", "cross"),
         ("L2", "G1", "cross"), ("L2", "G2", "cross"))
LOSS_FIELDS = ("rna", "mod2", "corr1", "corr2", "total")


def exact_tensor(a: torch.Tensor, b: torch.Tensor) -> dict:
    same = tuple(a.shape) == tuple(b.shape) and a.dtype == b.dtype and torch.equal(a, b)
    difference = 0.0
    if tuple(a.shape) == tuple(b.shape) and a.numel() and a.dtype == b.dtype:
        difference = float(torch.max(torch.abs(a.detach().cpu() - b.detach().cpu())))
    return {"pass": bool(same), "shape": list(a.shape), "dtype": str(a.dtype),
            "maximum_absolute_difference": difference}


def exact_named(first: dict, second: dict) -> dict:
    names = list(first) == list(second)
    tensors = {name: exact_tensor(first[name], second[name]) for name in first if name in second}
    return {"pass": names and len(first) == len(second) and all(x["pass"] for x in tensors.values()),
            "names_exact": names, "tensors": tensors}


def shared_forward_audit(reference, generalized, initial_state) -> dict:
    model = generalized.new_model()
    model.load_state_dict(initial_state)
    result = generalized.forward(model)
    public = frozen_components(reference, result)
    dispatched = generalized.components(result)
    public_all = {name: public[name] for name in LOSS_FIELDS}
    public_all["gene_weights"] = reference.weight_vector_omics1
    public_all["m_bad"] = reference.weight_vector_omics1.mean()
    generalized_all = {name: dispatched[name] for name in LOSS_FIELDS}
    generalized_all["gene_weights"] = dispatched["gene_weights"]
    generalized_all["m_bad"] = dispatched["m_bad"]
    comparison = exact_named(public_all, generalized_all)
    comparison["public_values"] = {name: float(public[name].detach().cpu()) for name in LOSS_FIELDS}
    comparison["generalized_values"] = {name: float(dispatched[name].detach().cpu()) for name in LOSS_FIELDS}
    comparison["operation_descriptions"] = {
        "public_rna": "mean((features_omics1 - emb_recon_omics1)**2 * public_weight_vector.unsqueeze(0))",
        "generalized_v3_rna": "mean(squared_error * legacy_bug_weight_vector(features_omics1).unsqueeze(0))",
        "total": "f0*rna + f1*modality2_mse + f2*corr1_mse + f3*corr2_mse",
        "shared_graph": "both formulae consumed the identical result dict from one model forward",
    }
    del model, result, public, dispatched
    return comparison


def optimizer_named(optimizer, model, include_zeros=False) -> dict:
    result = {}
    for name, parameter in model.named_parameters():
        state = optimizer.state.get(parameter, {})
        if state:
            for key in ("step", "exp_avg", "exp_avg_sq"):
                value = state[key]
                if not torch.is_tensor(value):
                    value = torch.tensor(value, device=parameter.device)
                result[name + ":" + key] = value.detach().cpu().clone()
        elif include_zeros:
            result[name + ":step"] = torch.zeros((), dtype=torch.float32)
            result[name + ":exp_avg"] = torch.zeros_like(parameter, device="cpu")
            result[name + ":exp_avg_sq"] = torch.zeros_like(parameter, device="cpu")
    return result


def independent_cpu_step(data, cfg, seed, gpu_initial_state) -> dict:
    device = torch.device("cpu")
    reference = Train_SpaLORA(data, datatype=cfg["legacy_datatype"], device=device,
                              random_seed=seed, learning_rate=1e-4, weight_decay=0.0,
                              epochs=cfg["epochs"], dim_output=cfg["embedding_dim"])
    generalized = ParityLockedTrainer(data, cfg, "locked_legacy_loss_replay", seed, device)
    initial = {name: value.detach().cpu().clone() for name, value in gpu_initial_state.items()}
    ref_model = Encoder_overall(reference.dim_input1, reference.dim_output1,
                                reference.dim_input2, reference.dim_output2).cpu()
    gen_model = generalized.new_model()
    ref_model.load_state_dict(initial)
    gen_model.load_state_dict(initial)
    ref_opt = torch.optim.Adam(ref_model.parameters(), lr=1e-4, weight_decay=0.0)
    gen_opt = torch.optim.Adam(gen_model.parameters(), lr=1e-4, weight_decay=0.0)
    ref_output = ref_model(reference.features_omics1, reference.features_omics2,
                           reference.adj_spatial_omics1, reference.adj_feature_omics1,
                           reference.adj_spatial_omics2, reference.adj_feature_omics2)
    gen_output = generalized.forward(gen_model)
    ref_loss = frozen_components(reference, ref_output)
    gen_loss = generalized.components(gen_output)
    ref_opt.zero_grad(); ref_loss["total"].backward()
    gen_opt.zero_grad(); gen_loss["total"].backward()
    output_check = exact_named(ref_output, gen_output)
    loss_check = exact_named({x: ref_loss[x] for x in LOSS_FIELDS}, {x: gen_loss[x] for x in LOSS_FIELDS})
    gradient_check = exact_named(
        {n: p.grad for n, p in ref_model.named_parameters()},
        {n: p.grad for n, p in gen_model.named_parameters()},
    )
    ref_opt.step(); gen_opt.step()
    parameter_check = exact_named(dict(ref_model.named_parameters()), dict(gen_model.named_parameters()))
    adam_check = exact_named(optimizer_named(ref_opt, ref_model), optimizer_named(gen_opt, gen_model))
    checks = (output_check, loss_check, gradient_check, parameter_check, adam_check)
    return {"pass": all(x["pass"] for x in checks), "outputs": output_check,
            "loss_components": loss_check, "gradients": gradient_check,
            "updated_parameters": parameter_check, "adam_state": adam_check}


def tensor_distances(a: torch.Tensor, b: torch.Tensor) -> tuple:
    a = a.detach().cpu().to(torch.float64).reshape(-1)
    b = b.detach().cpu().to(torch.float64).reshape(-1)
    if a.numel() != b.numel():
        return float("inf"), float("inf"), float("inf")
    delta = a - b
    raw = float(torch.max(torch.abs(delta))) if delta.numel() else 0.0
    rel = float(torch.linalg.vector_norm(delta) /
                max(float(torch.linalg.vector_norm(a)), float(torch.linalg.vector_norm(b)), 1e-12))
    scaled = raw / max(float(torch.max(torch.abs(a))) if a.numel() else 0.0,
                       float(torch.max(torch.abs(b))) if b.numel() else 0.0, 1.0)
    return raw, rel, scaled


def capture(model, optimizer, trainer, implementation: str) -> dict:
    optimizer.zero_grad()
    result = trainer.forward(model)
    components = frozen_components(trainer.legacy, result) if implementation == "legacy" else trainer.components(result)
    components["total"].backward()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return {
        "outputs": {name: value.detach().cpu().clone() for name, value in result.items()},
        "losses": {name: components[name].detach().cpu().clone() for name in LOSS_FIELDS},
        "gradients": {name: value.grad.detach().cpu().clone() for name, value in model.named_parameters()},
        "parameters": {name: value.detach().cpu().clone() for name, value in model.named_parameters()},
        "adam": optimizer_named(optimizer, model, include_zeros=True),
    }


def advance(model, optimizer, trainer, implementation: str) -> None:
    optimizer.zero_grad()
    result = trainer.forward(model)
    components = frozen_components(trainer.legacy, result) if implementation == "legacy" else trainer.components(result)
    components["total"].backward()
    optimizer.step()


def compare_captures(dataset, checkpoint, block, captures, rows) -> None:
    for left, right, pair_kind in PAIRS:
        for family in ("outputs", "losses", "gradients", "parameters", "adam"):
            if list(captures[left][family]) != list(captures[right][family]):
                raise AssertionError("named tensor drift for %s/%s" % (family, pair_kind))
            for name in captures[left][family]:
                raw, rel, scaled = tensor_distances(captures[left][family][name], captures[right][family][name])
                rows.append({"dataset": dataset, "checkpoint": checkpoint, "block": block,
                             "pair_kind": pair_kind, "left": left, "right": right,
                             "family": family, "tensor": name, "raw_max_abs": raw,
                             "relative_l2": rel, "scaled_max": scaled})


def summarize_envelope(rows, eta, margin) -> list:
    # First retain the worst named tensor in each family/pair/block/metric, as preregistered.
    worst = defaultdict(list)
    for row in rows:
        for metric in ("relative_l2", "scaled_max"):
            key = (row["dataset"], row["checkpoint"], row["family"], metric,
                   row["pair_kind"], row["left"], row["right"], row["block"])
            worst[key].append((float(row[metric]), row["tensor"], float(row["raw_max_abs"])))
    pooled = defaultdict(lambda: {"within": [], "cross": []})
    named = {}
    for key, values in worst.items():
        value, tensor, raw = max(values, key=lambda item: item[0])
        dataset, checkpoint, family, metric, kind, left, right, block = key
        bucket = "cross" if kind == "cross" else "within"
        pooled[(dataset, checkpoint, family, metric)][bucket].append(value)
        named[(dataset, checkpoint, family, metric, kind, left, right, block)] = (tensor, value, raw)
    summary = []
    for key in sorted(pooled):
        within = np.asarray(pooled[key]["within"], dtype=float)
        cross = np.asarray(pooled[key]["cross"], dtype=float)
        finite = bool(within.size and cross.size and np.all(np.isfinite(within)) and np.all(np.isfinite(cross)))
        within_q95 = float(np.quantile(within, 0.95)) if within.size else float("nan")
        within_median = float(np.median(within)) if within.size else float("nan")
        cross_q95 = float(np.quantile(cross, 0.95)) if cross.size else float("nan")
        cross_median = float(np.median(cross)) if cross.size else float("nan")
        e95, emed = max(within_q95, eta), max(within_median, eta)
        passed = finite and cross_q95 <= margin * e95 and cross_median <= margin * emed
        summary.append({"dataset": key[0], "checkpoint": key[1], "family": key[2], "metric": key[3],
                        "within_count": int(within.size), "cross_count": int(cross.size),
                        "within_q95": within_q95, "within_median": within_median,
                        "cross_q95": cross_q95, "cross_median": cross_median,
                        "eta": eta, "E95": e95, "Emed": emed, "margin": margin,
                        "q95_ratio": cross_q95 / e95, "median_ratio": cross_median / emed,
                        "finite": finite, "pass": passed})
    return summary


def gpu_envelope(dataset, data, cfg, seed, orders, checkpoints) -> tuple:
    device = torch.device("cuda:0")
    reference = Train_SpaLORA(data, datatype=cfg["legacy_datatype"], device=device,
                              random_seed=seed, learning_rate=1e-4, weight_decay=0.0,
                              epochs=cfg["epochs"], dim_output=cfg["embedding_dim"])
    generalized = ParityLockedTrainer(data, cfg, "locked_legacy_loss_replay", seed, device)
    initial_model = generalized.new_model()
    initial_state = {name: value.detach().clone() for name, value in initial_model.state_dict().items()}
    shared = shared_forward_audit(reference, generalized, initial_state)
    rows = []
    for block, order in enumerate(orders, 1):
        models = {}; optimizers = {}
        for trajectory in ("L1", "L2", "G1", "G2"):
            model = generalized.new_model(); model.load_state_dict(initial_state)
            models[trajectory] = model
            optimizers[trajectory] = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)
        current = 0
        for checkpoint in checkpoints:
            while current < checkpoint:
                for trajectory in order:
                    advance(models[trajectory], optimizers[trajectory], generalized,
                            "legacy" if trajectory.startswith("L") else "generalized")
                current += 1
            captures = {}
            for trajectory in order:
                captures[trajectory] = capture(models[trajectory], optimizers[trajectory], generalized,
                                                "legacy" if trajectory.startswith("L") else "generalized")
            compare_captures(dataset, checkpoint, block, captures, rows)
            del captures
        del models, optimizers
        gc.collect(); torch.cuda.empty_cache()
        print("P0C_BLOCK", dataset, block, "DONE", flush=True)
    return initial_state, shared, rows


def deterministic_probe(dataset) -> dict:
    command = [sys.executable, str(REPO / "scripts/night2c_deterministic_probe.py"), dataset]
    completed = subprocess.run(command, cwd=str(REPO), text=True, capture_output=True)
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except Exception:
        payload = {"success": False, "parse_error": True, "stdout": completed.stdout, "stderr": completed.stderr}
    payload["exit_status"] = completed.returncode
    return payload


def write_csv(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise AssertionError("refusing to write empty P0C evidence")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def audit_dataset(dataset, cfg, config) -> tuple:
    print("P0C_DATASET", dataset, "START", flush=True)
    seed = int(config["p0c"]["seed"])
    fix_seed(seed)
    torch.use_deterministic_algorithms(False)
    data, _, _ = prepare_legacy(dataset, cfg)
    device = torch.device("cuda:0")
    reference = Train_SpaLORA(data, datatype=cfg["legacy_datatype"], device=device,
                              random_seed=seed, learning_rate=1e-4, weight_decay=0.0,
                              epochs=cfg["epochs"], dim_output=cfg["embedding_dim"])
    generalized = ParityLockedTrainer(data, cfg, "locked_legacy_loss_replay", seed, device)
    consumed = consumed_input_audit(reference, generalized, data, cfg)
    canonical = generalized.new_model()
    canonical_state = {name: value.detach().clone() for name, value in canonical.state_dict().items()}
    reference_model = Encoder_overall(reference.dim_input1, reference.dim_output1,
                                      reference.dim_input2, reference.dim_output2).to(device)
    generalized_model = generalized.new_model()
    reference_model.load_state_dict(canonical_state); generalized_model.load_state_dict(canonical_state)
    consumed["initial_model_state"] = exact_named(reference_model.state_dict(), generalized_model.state_dict())
    expected = config["bad_weight_expected_mean"][str(generalized.features1.shape[1])]
    consumed["m_bad_expected"] = expected
    consumed["m_bad_tolerance"] = config["bad_weight_mean_tolerance"]
    consumed["m_bad_pass"] = abs(consumed["m_bad"] - expected) <= config["bad_weight_mean_tolerance"]
    consumed["pass"] = consumed["pass"] and consumed["m_bad_pass"] and consumed["initial_model_state"]["pass"]
    del reference, generalized, canonical, reference_model, generalized_model, canonical_state
    gc.collect(); torch.cuda.empty_cache()

    initial, shared, pairwise = gpu_envelope(dataset, data, cfg, seed,
                                              config["p0c"]["execution_orders"],
                                              config["p0c"]["checkpoints_after_updates"])
    cpu = independent_cpu_step(data, cfg, seed, initial)
    probe = deterministic_probe(dataset)
    del data, initial
    gc.collect(); torch.cuda.empty_cache()
    return {"consumed_state": consumed, "shared_forward_v3": shared,
            "cpu_independent_one_step": cpu, "deterministic_algorithm_probe": probe}, pairwise


def main() -> int:
    config_path = REPO / "configs/night2c_numerical_equivalence_factorial.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["parent_commit"] != "c283449b188f510e98c2826cbb856f296367aa03":
        raise AssertionError("Night-2C parent commit drift")
    if not torch.cuda.is_available():
        raise RuntimeError("P0C requires the target CUDA GPU")
    environment = environment_payload()
    hashes = critical_hashes(REPO, config)
    report = {"schema_version": 1, "gate": "P0C", "ground_truth_accessed": False,
              "label_access_flag": False, "environment": environment,
              "critical_hashes": hashes, "datasets": {}}
    all_pairwise = []
    for dataset in DATASETS:
        try:
            item, rows = audit_dataset(dataset, config["datasets"][dataset], config)
            report["datasets"][dataset] = item
            all_pairwise.extend(rows)
        except Exception as exc:
            report["datasets"][dataset] = {"pass": False, "error": repr(exc),
                                            "traceback": traceback.format_exc()}
        finally:
            gc.collect(); torch.cuda.empty_cache()
    eta = float(config["p0c"]["float32_epsilon_multiplier"]) * float(np.finfo(np.float32).eps)
    margin = float(config["p0c"]["equivalence_margin"])
    summary = summarize_envelope(all_pairwise, eta, margin) if all_pairwise else []
    for dataset in DATASETS:
        item = report["datasets"][dataset]
        cells = [row for row in summary if row["dataset"] == dataset]
        exact_pass = all(item.get(key, {}).get("pass") is True for key in
                         ("consumed_state", "shared_forward_v3", "cpu_independent_one_step"))
        item["gpu_envelope"] = {"pass": bool(cells) and all(row["pass"] for row in cells),
                                "cell_count": len(cells),
                                "failed_cells": [row for row in cells if not row["pass"]]}
        item["pass"] = exact_pass and item["gpu_envelope"]["pass"]
    report["p0c_pass"] = all(report["datasets"].get(name, {}).get("pass") is True for name in DATASETS)
    report["factorial_authorized"] = report["p0c_pass"]
    report["envelope"] = {"eta": eta, "margin": margin, "summary_cell_count": len(summary),
                          "failed_cell_count": sum(not row["pass"] for row in summary)}
    results = REPO / "results/night2c"; reports = REPO / "reports"
    results.mkdir(parents=True, exist_ok=True); reports.mkdir(exist_ok=True)
    if all_pairwise:
        write_csv(results / "p0c_pairwise_distances.csv", all_pairwise)
    if summary:
        write_csv(results / "p0c_summary.csv", summary)
    atomic_json(reports / "night2c_p0c.json", report)
    atomic_json(reports / "night2c_environment.json", environment)
    reason = "all exact audits and every normalized GPU noise-envelope cell passed" if report["p0c_pass"] else "one or more preregistered P0C exact audits or GPU envelope cells failed"
    gate = {"status": "authorized" if report["p0c_pass"] else "stopped_by_p0c_hard_gate",
            "p0c_pass": report["p0c_pass"], "factorial_authorized": report["p0c_pass"],
            "ground_truth_accessed_during_p0c": False, "ground_truth_used_for_setting_selection": False,
            "seed_search_performed": False, "asr_modified": False,
            "critical_hashes": hashes, "environment_fingerprint": environment["fingerprint"],
            "main_runs": {"planned_if_authorized": 75, "completed": 0},
            "tutorial_runs": {"planned_if_authorized": 3, "completed": 0},
            "technical_runs": {"planned_if_authorized": 4, "completed": 0},
            "reason": reason}
    atomic_json(results / "gate_status.json", gate)
    print(json.dumps({"p0c_pass": report["p0c_pass"], "failed_cells": report["envelope"]["failed_cell_count"],
                      "reason": reason}, sort_keys=True), flush=True)
    return 0 if report["p0c_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
