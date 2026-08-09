#!/usr/bin/env python3
"""P0B: prove loss-only replay on exact frozen legacy-consumed inputs."""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
import platform
import sys
import traceback
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import scipy
import sklearn
import torch
import torch.nn.functional as F


os.environ.setdefault("R_HOME", "/opt/R/4.0.3/lib/R")
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from SpaLORA.model import Encoder_overall
from SpaLORA.night2b_loss_audit import ParityLockedTrainer, legacy_bug_weight_vector
from SpaLORA.preprocess import fix_seed
from SpaLORA.SpaLORA_pyG import Train_SpaLORA
from scripts.night1_benchmark import prepare_legacy


ABS_TOL = 1e-7
REL_TOL = 1e-7


def sha_tensor(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def tensor_report(first: torch.Tensor, second: torch.Tensor, exact_required: bool = False) -> dict:
    shape_equal = tuple(first.shape) == tuple(second.shape)
    dtype_equal = first.dtype == second.dtype
    if not shape_equal or not dtype_equal:
        return {
            "pass": False,
            "shape_equal": shape_equal,
            "dtype_equal": dtype_equal,
            "first_shape": list(first.shape),
            "second_shape": list(second.shape),
            "first_dtype": str(first.dtype),
            "second_dtype": str(second.dtype),
        }
    a = first.detach()
    b = second.detach()
    exact = bool(torch.equal(a, b))
    floating = a.dtype.is_floating_point or a.dtype.is_complex
    if a.numel() and floating:
        difference = torch.abs(a - b)
        max_abs = float(difference.max().cpu())
        denom = torch.maximum(torch.abs(a), torch.abs(b)).clamp_min(torch.finfo(a.dtype).tiny)
        max_rel = float((difference / denom).max().cpu())
        close = bool(torch.allclose(a, b, atol=ABS_TOL, rtol=REL_TOL))
    elif a.numel():
        max_abs = float(torch.abs(a - b).max().cpu())
        max_rel = 0.0 if exact else float("inf")
        close = exact
    else:
        max_abs = max_rel = 0.0
        close = True
    return {
        "pass": exact if exact_required else close,
        "exact_equal": exact,
        "allclose": close,
        "exact_required": exact_required,
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "max_absolute_difference": max_abs,
        "max_relative_difference": max_rel,
        "first_sha256": sha_tensor(a),
        "second_sha256": sha_tensor(b),
    }


def sparse_report(first: torch.Tensor, second: torch.Tensor) -> dict:
    if not first.is_sparse or not second.is_sparse:
        return {"pass": False, "error": "both adjacency tensors must be sparse COO"}
    a, b = first.coalesce(), second.coalesce()
    indices = tensor_report(a.indices(), b.indices(), exact_required=True)
    values = tensor_report(a.values(), b.values(), exact_required=True)
    return {
        "pass": tuple(a.shape) == tuple(b.shape) and a._nnz() == b._nnz() and indices["pass"] and values["pass"],
        "shape_equal": tuple(a.shape) == tuple(b.shape),
        "shape": list(a.shape),
        "first_nnz": int(a._nnz()),
        "second_nnz": int(b._nnz()),
        "indices": indices,
        "values": values,
    }


def state_report(first: torch.nn.Module, second: torch.nn.Module) -> dict:
    a, b = first.state_dict(), second.state_dict()
    names_equal = list(a) == list(b)
    tensors = {name: tensor_report(a[name], b[name], exact_required=True) for name in a if name in b}
    return {
        "pass": names_equal and len(a) == len(b) and all(item["pass"] for item in tensors.values()),
        "parameter_names_exact": names_equal,
        "first_count": len(a),
        "second_count": len(b),
        "tensors": tensors,
    }


def forward_report(first: Dict[str, torch.Tensor], second: Dict[str, torch.Tensor]) -> dict:
    names_equal = list(first) == list(second)
    outputs = {name: tensor_report(first[name], second[name]) for name in first if name in second}
    return {
        "pass": names_equal and len(first) == len(second) and all(item["pass"] for item in outputs.values()),
        "names_exact": names_equal,
        "outputs": outputs,
    }


def frozen_components(trainer: Train_SpaLORA, result: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    diff = trainer.features_omics1 - result["emb_recon_omics1"]
    rna = torch.mean((diff ** 2) * trainer.weight_vector_omics1.unsqueeze(0))
    mod2 = F.mse_loss(trainer.features_omics2, result["emb_recon_omics2"])
    corr1 = F.mse_loss(result["emb_latent_omics1"], result["emb_latent_omics1_across_recon"])
    corr2 = F.mse_loss(result["emb_latent_omics2"], result["emb_latent_omics2_across_recon"])
    factors = trainer.weight_factors
    total = factors[0] * rna + factors[1] * mod2 + factors[2] * corr1 + factors[3] * corr2
    return {"rna": rna, "mod2": mod2, "corr1": corr1, "corr2": corr2, "total": total}


def compare_losses(first: Dict[str, torch.Tensor], second: Dict[str, torch.Tensor]) -> dict:
    fields = ("rna", "mod2", "corr1", "corr2", "total")
    values = {field: tensor_report(first[field], second[field]) for field in fields}
    return {"pass": all(item["pass"] for item in values.values()), "components": values}


def gradient_report(first: torch.nn.Module, second: torch.nn.Module) -> dict:
    first_named, second_named = dict(first.named_parameters()), dict(second.named_parameters())
    names_equal = list(first_named) == list(second_named)
    gradients = {}
    missing = []
    for name in first_named:
        if name not in second_named or first_named[name].grad is None or second_named[name].grad is None:
            missing.append(name)
            continue
        gradients[name] = tensor_report(first_named[name].grad, second_named[name].grad)
    return {
        "pass": names_equal and not missing and all(item["pass"] for item in gradients.values()),
        "parameter_names_exact": names_equal,
        "missing_gradients": missing,
        "gradients": gradients,
    }


def optimizer_report(
    first: torch.optim.Optimizer,
    second: torch.optim.Optimizer,
    first_model: torch.nn.Module,
    second_model: torch.nn.Module,
) -> dict:
    first_params = list(first_model.parameters())
    second_params = list(second_model.parameters())
    items = {}
    missing = []
    for index, (a_param, b_param) in enumerate(zip(first_params, second_params)):
        a_state, b_state = first.state.get(a_param, {}), second.state.get(b_param, {})
        if set(a_state) != set(b_state):
            missing.append(index)
            continue
        state_items = {}
        for name in sorted(a_state):
            a_value, b_value = a_state[name], b_state[name]
            if torch.is_tensor(a_value) and torch.is_tensor(b_value):
                state_items[name] = tensor_report(a_value, b_value)
            else:
                state_items[name] = {"pass": a_value == b_value, "first": a_value, "second": b_value}
        items[str(index)] = {
            "pass": all(value["pass"] for value in state_items.values()),
            "state": state_items,
        }
    return {
        "pass": len(first_params) == len(second_params) and not missing and all(x["pass"] for x in items.values()),
        "first_parameter_count": len(first_params),
        "second_parameter_count": len(second_params),
        "state_key_mismatches": missing,
        "parameters": items,
    }


def model_parameter_report(first: torch.nn.Module, second: torch.nn.Module) -> dict:
    a, b = dict(first.named_parameters()), dict(second.named_parameters())
    names_equal = list(a) == list(b)
    values = {name: tensor_report(a[name], b[name]) for name in a if name in b}
    return {
        "pass": names_equal and len(a) == len(b) and all(item["pass"] for item in values.values()),
        "names_exact": names_equal,
        "parameters": values,
    }


def forward_on_reference(model: Encoder_overall, trainer: Train_SpaLORA) -> Dict[str, torch.Tensor]:
    return model(
        trainer.features_omics1,
        trainer.features_omics2,
        trainer.adj_spatial_omics1,
        trainer.adj_feature_omics1,
        trainer.adj_spatial_omics2,
        trainer.adj_feature_omics2,
    )


def single_step_audit(
    reference: Train_SpaLORA,
    generalized: ParityLockedTrainer,
    initial_state: Dict[str, torch.Tensor],
) -> dict:
    ref_model = Encoder_overall(
        reference.dim_input1, reference.dim_output1, reference.dim_input2, reference.dim_output2
    ).to(reference.device)
    gen_model = generalized.new_model()
    ref_model.load_state_dict(initial_state)
    gen_model.load_state_dict(initial_state)
    ref_optimizer = torch.optim.Adam(ref_model.parameters(), lr=0.0001, weight_decay=0.0)
    gen_optimizer = torch.optim.Adam(gen_model.parameters(), lr=0.0001, weight_decay=0.0)
    ref_forward = forward_on_reference(ref_model, reference)
    gen_forward = generalized.forward(gen_model)
    forward_comparison = forward_report(ref_forward, gen_forward)
    ref_losses = frozen_components(reference, ref_forward)
    gen_losses = generalized.components(gen_forward)
    loss_comparison = compare_losses(ref_losses, gen_losses)
    ref_optimizer.zero_grad()
    ref_losses["total"].backward()
    gen_optimizer.zero_grad()
    gen_losses["total"].backward()
    gradients = gradient_report(ref_model, gen_model)
    ref_optimizer.step()
    gen_optimizer.step()
    parameters = model_parameter_report(ref_model, gen_model)
    optimizer = optimizer_report(ref_optimizer, gen_optimizer, ref_model, gen_model)
    return {
        "pass": all(x["pass"] for x in (forward_comparison, loss_comparison, gradients, parameters, optimizer)),
        "forward": forward_comparison,
        "losses": loss_comparison,
        "gradients": gradients,
        "updated_parameters": parameters,
        "optimizer_state": optimizer,
    }


def five_step_audit(
    reference: Train_SpaLORA,
    generalized: ParityLockedTrainer,
    initial_state: Dict[str, torch.Tensor],
) -> dict:
    ref_model = Encoder_overall(
        reference.dim_input1, reference.dim_output1, reference.dim_input2, reference.dim_output2
    ).to(reference.device)
    gen_model = generalized.new_model()
    ref_model.load_state_dict(initial_state)
    gen_model.load_state_dict(initial_state)
    ref_optimizer = torch.optim.Adam(ref_model.parameters(), lr=0.0001, weight_decay=0.0)
    gen_optimizer = torch.optim.Adam(gen_model.parameters(), lr=0.0001, weight_decay=0.0)
    steps = []
    passed = True
    for step in range(5):
        ref_forward = forward_on_reference(ref_model, reference)
        gen_forward = generalized.forward(gen_model)
        ref_losses = frozen_components(reference, ref_forward)
        gen_losses = generalized.components(gen_forward)
        losses = compare_losses(ref_losses, gen_losses)
        ref_optimizer.zero_grad()
        ref_losses["total"].backward()
        gen_optimizer.zero_grad()
        gen_losses["total"].backward()
        gradients = gradient_report(ref_model, gen_model)
        ref_optimizer.step()
        gen_optimizer.step()
        parameters = model_parameter_report(ref_model, gen_model)
        optimizer = optimizer_report(ref_optimizer, gen_optimizer, ref_model, gen_model)
        parameter_max = max(
            item.get("max_absolute_difference", float("inf")) for item in parameters["parameters"].values()
        )
        row = {
            "step": step + 1,
            "reference_losses": {name: float(ref_losses[name].detach().cpu()) for name in ("rna", "mod2", "corr1", "corr2", "total")},
            "generalized_losses": {name: float(gen_losses[name].detach().cpu()) for name in ("rna", "mod2", "corr1", "corr2", "total")},
            "loss_comparison": losses,
            "gradient_pass": gradients["pass"],
            "parameter_pass": parameters["pass"],
            "optimizer_state_pass": optimizer["pass"],
            "maximum_parameter_difference": parameter_max,
        }
        row["pass"] = losses["pass"] and gradients["pass"] and parameters["pass"] and optimizer["pass"]
        passed = passed and row["pass"]
        steps.append(row)
    return {"pass": passed, "steps": steps}


def consumed_input_audit(reference: Train_SpaLORA, generalized: ParityLockedTrainer, data: dict, cfg: dict) -> dict:
    dense = {
        "features_omics1": tensor_report(reference.features_omics1, generalized.features1, exact_required=True),
        "features_omics2": tensor_report(reference.features_omics2, generalized.features2, exact_required=True),
        "legacy_bug_weight_vector": tensor_report(
            reference.weight_vector_omics1,
            legacy_bug_weight_vector(generalized.features1),
            exact_required=True,
        ),
    }
    adjacency_names = (
        "adj_spatial_omics1", "adj_feature_omics1", "adj_spatial_omics2", "adj_feature_omics2"
    )
    sparse = {
        name: sparse_report(getattr(reference, name), generalized.adjacencies[index])
        for index, name in enumerate(adjacency_names)
    }
    reference_genes = data["adata_omics1"].var_names[
        data["adata_omics1"].var["highly_variable"].to_numpy(dtype=bool)
    ].astype(str).to_numpy()
    settings = {
        "loss_factors_exact": list(reference.weight_factors) == list(generalized.factors) == list(cfg["loss_factors"]),
        "epochs_exact": int(reference.epochs) == generalized.epochs == int(cfg["epochs"]),
        "embedding_dimension_exact": int(reference.dim_output) == generalized.embedding_dim == int(cfg["embedding_dim"]),
        "learning_rate": 0.0001,
        "weight_decay": 0.0,
        "optimizer": "torch.optim.Adam",
    }
    genes_exact = np.array_equal(reference_genes, generalized.gene_names)
    obs_exact = np.array_equal(data["adata_omics1"].obs_names.astype(str).to_numpy(), generalized.obs_names)
    m_bad = float(reference.weight_vector_omics1.mean().detach().cpu())
    return {
        "pass": all(item["pass"] for item in dense.values()) and all(item["pass"] for item in sparse.values())
        and genes_exact and obs_exact and all(settings[key] for key in ("loss_factors_exact", "epochs_exact", "embedding_dimension_exact")),
        "dense_tensors": dense,
        "sparse_adjacencies": sparse,
        "ordered_gene_names_exact": bool(genes_exact),
        "ordered_gene_names_count": int(reference_genes.size),
        "observation_ids_exact": bool(obs_exact),
        "observation_count": int(generalized.obs_names.size),
        "settings": settings,
        "m_bad": m_bad,
        "m_bad_expected": 2.290322960 if reference_genes.size == 2000 else 2.289938091,
    }


def audit_dataset(dataset: str, cfg: dict, seed: int, device: torch.device) -> dict:
    print("P0B_START", dataset, flush=True)
    fix_seed(seed)
    data, _, _ = prepare_legacy(dataset, cfg)
    reference = Train_SpaLORA(
        data,
        datatype=cfg["legacy_datatype"],
        device=device,
        random_seed=seed,
        learning_rate=0.0001,
        weight_decay=0.0,
        epochs=cfg["epochs"],
        dim_output=cfg["embedding_dim"],
    )
    generalized = ParityLockedTrainer(data, cfg, "locked_legacy_loss_replay", seed, device)
    consumed = consumed_input_audit(reference, generalized, data, cfg)

    initial_reference = Encoder_overall(
        reference.dim_input1, reference.dim_output1, reference.dim_input2, reference.dim_output2
    ).to(device)
    initial_state = {name: value.detach().clone() for name, value in initial_reference.state_dict().items()}
    initial_generalized = generalized.new_model()
    initial_generalized.load_state_dict(initial_state)
    initial = state_report(initial_reference, initial_generalized)
    one_step = single_step_audit(reference, generalized, initial_state)
    trajectory = five_step_audit(reference, generalized, initial_state)
    expected_mean = 2.290322960 if generalized.features1.shape[1] == 2000 else 2.289938091
    mean_error = abs(float(reference.weight_vector_omics1.mean().detach().cpu()) - expected_mean)
    result = {
        "pass": consumed["pass"] and initial["pass"] and one_step["pass"] and trajectory["pass"] and mean_error <= 1e-6,
        "dataset": dataset,
        "seed": seed,
        "consumed_inputs": consumed,
        "initial_model_state": initial,
        "v3_one_adam_step": one_step,
        "v3_five_step_trajectory": trajectory,
        "bad_weight_mean_error": mean_error,
    }
    print("P0B_DONE", dataset, "PASS" if result["pass"] else "FAIL", flush=True)
    return result


def main() -> int:
    night1 = json.loads((REPO / "configs" / "night1.json").read_text(encoding="utf-8"))
    config = json.loads((REPO / "configs" / "night2b_parity_locked_loss_audit.json").read_text(encoding="utf-8"))
    if config["parent_commit"] != "16f0cc43673617c73527110962b7ca115c59b4c6":
        raise AssertionError("Night-2B parent drift")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    report = {
        "schema_version": 1,
        "gate": "P0B",
        "ground_truth_accessed": False,
        "declared_absolute_tolerance": ABS_TOL,
        "declared_relative_tolerance": REL_TOL,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "datasets": {},
    }
    for dataset in ("a1", "placenta", "p22"):
        try:
            report["datasets"][dataset] = audit_dataset(dataset, night1["datasets"][dataset], config["p0b_seed"], device)
        except Exception as exc:
            report["datasets"][dataset] = {
                "dataset": dataset,
                "pass": False,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    report["p0b_pass"] = all(item.get("pass") is True for item in report["datasets"].values())
    report["factorial_authorized"] = report["p0b_pass"]
    reports = REPO / "reports"
    results = REPO / "results" / "night2b"
    reports.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    (reports / "night2b_parity_locked.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    rows = []
    for dataset, item in report["datasets"].items():
        consumed = item.get("consumed_inputs", {})
        one = item.get("v3_one_adam_step", {})
        trajectory = item.get("v3_five_step_trajectory", {})
        rows.append({
            "dataset": dataset,
            "p0b_pass": item.get("pass", False),
            "consumed_inputs_pass": consumed.get("pass", False),
            "initial_state_pass": item.get("initial_model_state", {}).get("pass", False),
            "forward_loss_gradient_adam_pass": one.get("pass", False),
            "five_step_trajectory_pass": trajectory.get("pass", False),
            "m_bad": consumed.get("m_bad", ""),
            "error": item.get("error", ""),
        })
    with (results / "p0b_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    gate = {
        "status": "authorized" if report["p0b_pass"] else "stopped_by_p0b_hard_gate",
        "p0b_pass": report["p0b_pass"],
        "factorial_authorized": report["factorial_authorized"],
        "main_runs": {"planned_if_authorized": 75, "completed": 0},
        "tutorial_runs": {"planned_if_authorized": 3, "completed": 0},
        "ground_truth_accessed_during_p0b": False,
    }
    (results / "gate_status.json").write_text(json.dumps(gate, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(gate, sort_keys=True), flush=True)
    return 0 if report["p0b_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
