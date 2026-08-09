"""Night-3A-R weighted-gradient influence diagnostics with unchanged training math."""

from __future__ import annotations

import copy
import hashlib
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .night3a_ige import (
    LOSS_KEYS,
    Night3ATrainer,
    TrainingResult,
    _clone_state,
    _rng_snapshot,
    calibrated_total,
    coefficients_for_variant,
    global_gradient_norm,
    legacy_m_bad,
    model_state_sha256,
    raw_losses,
    required_record_steps,
    rms_gradient_probe,
    rng_snapshot_sha256,
    run_initial_probe,
    state_dict_sha256,
    tensor_sha256,
)


def optimizer_state_sha256(optimizer: torch.optim.Optimizer) -> str:
    state = optimizer.state_dict()
    digest = hashlib.sha256(repr(state["param_groups"]).encode("utf-8"))
    for parameter_id in sorted(state["state"]):
        digest.update(str(parameter_id).encode("ascii"))
        for key, value in sorted(state["state"][parameter_id].items()):
            digest.update(str(key).encode("utf-8"))
            if isinstance(value, torch.Tensor):
                digest.update(tensor_sha256(value).encode("ascii"))
            else:
                digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


def weighted_gradient_influence(model: torch.nn.Module, forward,
                                coefficients: Mapping[str, float], eps: float,
                                optimizer: Optional[torch.optim.Optimizer] = None) -> dict:
    """Measure |coefficient| * raw-loss RMS gradient without changing training state."""
    state_before = model_state_sha256(model)
    grad_before = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }
    rng_before = _rng_snapshot()
    optimizer_before = None if optimizer is None else optimizer_state_sha256(optimizer)
    result = forward(model)
    losses = raw_losses(result, forward.features1, forward.features2)
    gradients, parameter_rows = rms_gradient_probe(model, result, losses, eps)
    influence = {name: abs(float(coefficients[name])) * gradients[name] for name in LOSS_KEYS}
    total = float(sum(influence.values()))
    if not np.isfinite(total) or total <= 0:
        raise FloatingPointError("Weighted-gradient influence is non-finite or non-positive")
    shares = {name: influence[name] / total for name in LOSS_KEYS}
    grad_unchanged = True
    for name, parameter in model.named_parameters():
        previous = grad_before[name]
        if previous is None:
            grad_unchanged &= parameter.grad is None
        else:
            grad_unchanged &= parameter.grad is not None and torch.equal(previous, parameter.grad)
    payload = {
        "raw_rms_gradients": gradients,
        "weighted_gradient_influence": influence,
        "weighted_gradient_share": shares,
        "parameter_rows": parameter_rows,
        "parameter_state_unchanged": model_state_sha256(model) == state_before,
        "grad_fields_unchanged": bool(grad_unchanged),
        "rng_state_unchanged": rng_snapshot_sha256(_rng_snapshot()) == rng_snapshot_sha256(rng_before),
        "optimizer_state_unchanged": (
            True if optimizer is None else optimizer_state_sha256(optimizer) == optimizer_before
        ),
    }
    if not all(payload[name] for name in (
        "parameter_state_unchanged", "grad_fields_unchanged",
        "rng_state_unchanged", "optimizer_state_unchanged",
    )):
        raise AssertionError("Gradient-influence diagnostic changed training state")
    return payload


@dataclass
class Night3ARTrainingResult:
    output: Dict[str, np.ndarray]
    logs: List[dict]
    gradient_logs: List[dict]
    model: torch.nn.Module
    probe: Optional[dict]
    coefficients: Dict[str, float]
    initial_losses: Dict[str, float]
    initial_state_sha256: str
    final_state_sha256: str


class Night3ARTrainer(Night3ATrainer):
    """Night3ATrainer plus state-neutral gradient influence at locked checkpoints."""

    def _gradient_record(self, model, optimizer, step, coefficients) -> dict:
        diagnostic = weighted_gradient_influence(
            model, self.forward, coefficients, self.eps, optimizer=optimizer
        )
        row = {
            "step": int(step),
            "fraction_of_training": float(step / int(self.cfg["epochs"])),
            "diagnostic_parameter_state_unchanged": diagnostic["parameter_state_unchanged"],
            "diagnostic_grad_fields_unchanged": diagnostic["grad_fields_unchanged"],
            "diagnostic_rng_state_unchanged": diagnostic["rng_state_unchanged"],
            "diagnostic_optimizer_state_unchanged": diagnostic["optimizer_state_unchanged"],
        }
        for name in LOSS_KEYS:
            short = name.replace("L_", "").replace("_raw", "")
            row[short + "_raw_rms_gradient"] = diagnostic["raw_rms_gradients"][name]
            row[short + "_weighted_gradient_influence"] = diagnostic["weighted_gradient_influence"][name]
            row[short + "_weighted_gradient_share"] = diagnostic["weighted_gradient_share"][name]
        return row

    def train(self) -> Night3ARTrainingResult:
        model = self.new_model()
        initial_state = _clone_state(model)
        initial_state_hash = state_dict_sha256(initial_state)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)
        initial_result = self.forward(model)
        initial_raw_tensors = raw_losses(initial_result, self.features1, self.features2)
        initial_losses = {name: float(initial_raw_tensors[name].detach().cpu()) for name in LOSS_KEYS}
        probe: Optional[dict] = None
        gradients = {name: 1.0 for name in LOSS_KEYS}
        if self.variant == "IGE":
            probe = run_initial_probe(model, self.forward, self.eps)
            gradients = probe["gradients"]
            model.load_state_dict(initial_state)
            if model_state_sha256(model) != initial_state_hash:
                raise AssertionError("IGE formal training did not reload the exact initial state")
        m_bad = float(legacy_m_bad(self.features1).detach().cpu())
        coefficients = coefficients_for_variant(
            self.variant, initial_losses, gradients, self.cfg["loss_factors"], m_bad, self.eps
        )
        if self.variant == "IGE":
            values = np.asarray(list(coefficients.values()), dtype=np.float64)
            if np.any(values < 1e-3) or np.any(values > 1e3) or abs(float(values.sum()) - 4.0) > 1e-6:
                raise RuntimeError("IGE coefficients violate the preregistered P0B range/sum gate")

        started = time.perf_counter()
        record_steps = set(required_record_steps(int(self.cfg["epochs"])))
        logs: List[dict] = []
        gradient_logs: List[dict] = []
        initial_result = self.forward(model)
        initial_raw_tensors = raw_losses(initial_result, self.features1, self.features2)
        initial_total = calibrated_total(initial_raw_tensors, coefficients)
        initial_total_gradients = torch.autograd.grad(
            initial_total, [p for p in model.parameters() if p.requires_grad], allow_unused=True
        )
        initial_grad_norm = math.sqrt(sum(
            float(torch.sum(gradient.detach().double() ** 2).cpu())
            for gradient in initial_total_gradients if gradient is not None
        ))
        logs.append(self._record(model, 0, coefficients, initial_grad_norm, started))
        gradient_logs.append(self._gradient_record(model, optimizer, 0, coefficients))

        for step in range(1, int(self.cfg["epochs"]) + 1):
            result = self.forward(model)
            losses = raw_losses(result, self.features1, self.features2)
            total = calibrated_total(losses, coefficients)
            optimizer.zero_grad()
            total.backward()
            grad_norm = global_gradient_norm(model)
            optimizer.step()
            if step in record_steps:
                logs.append(self._record(model, step, coefficients, grad_norm, started))
                gradient_logs.append(self._gradient_record(model, optimizer, step, coefficients))
        expected = sorted(record_steps)
        if [row["step"] for row in logs] != expected or [row["step"] for row in gradient_logs] != expected:
            raise AssertionError("Required Night-3A-R checkpoints are incomplete")
        if not all(row["embedding_finite"] for row in logs):
            raise FloatingPointError("Embedding NaN/Inf detected")

        model.eval()
        with torch.no_grad():
            result = self.forward(model)
        output = {
            "emb_latent_omics1": F.normalize(result["emb_latent_omics1"], p=2, eps=1e-12, dim=1).cpu().numpy(),
            "emb_latent_omics2": F.normalize(result["emb_latent_omics2"], p=2, eps=1e-12, dim=1).cpu().numpy(),
            "SpaLORA": F.normalize(result["emb_latent_combined"], p=2, eps=1e-12, dim=1).cpu().numpy(),
            "alpha_omics1": result["alpha_omics1"].cpu().numpy(),
            "alpha_omics2": result["alpha_omics2"].cpu().numpy(),
            "alpha": result["alpha"].cpu().numpy(),
        }
        return Night3ARTrainingResult(
            output=output,
            logs=logs,
            gradient_logs=gradient_logs,
            model=model,
            probe=probe,
            coefficients=coefficients,
            initial_losses=initial_losses,
            initial_state_sha256=initial_state_hash,
            final_state_sha256=model_state_sha256(model),
        )

