"""Preregistered Night-3B architecture and loss ablations.

The FULL_IGE path delegates to the published Night-3A-R trainer.  Registered
attention ablations retain the identical parameter set and initialization and
change only the forward combination rule.  Registered loss-drop variants set
one coefficient exactly to zero and recompute the other three label-free IGE
coefficients on the same variant/seed initialization.
"""

from __future__ import annotations

import math
import time
from typing import Dict, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from .night3a_ige import (
    LOSS_KEYS,
    _clone_state,
    calibrated_total,
    global_gradient_norm,
    model_state_sha256,
    raw_losses,
    required_record_steps,
    run_initial_probe,
    state_dict_sha256,
)
from .night3ar_ige import Night3ARTrainer, Night3ARTrainingResult


VARIANTS = (
    "FULL_IGE",
    "DROP_RNA_RECON",
    "DROP_MOD2_RECON",
    "DROP_CORR1",
    "DROP_CORR2",
    "UNIFORM_WITHIN",
    "UNIFORM_CROSS",
    "UNIFORM_ALL",
)

DROP_BY_VARIANT = {
    "DROP_RNA_RECON": "L_rna_recon_raw",
    "DROP_MOD2_RECON": "L_mod2_recon_raw",
    "DROP_CORR1": "L_corr1_raw",
    "DROP_CORR2": "L_corr2_raw",
}

ATTENTION_MODE_BY_VARIANT = {
    "FULL_IGE": "learned",
    "DROP_RNA_RECON": "learned",
    "DROP_MOD2_RECON": "learned",
    "DROP_CORR1": "learned",
    "DROP_CORR2": "learned",
    "UNIFORM_WITHIN": "uniform_within",
    "UNIFORM_CROSS": "uniform_cross",
    "UNIFORM_ALL": "uniform_all",
}


def active_loss_mask(variant: str) -> Dict[str, bool]:
    if variant not in VARIANTS:
        raise ValueError("Unregistered Night-3B variant: %s" % variant)
    dropped = DROP_BY_VARIANT.get(variant)
    return {name: name != dropped for name in LOSS_KEYS}


def active_ige_coefficients(
    gradients: Mapping[str, float], mask: Mapping[str, bool], eps: float
) -> Dict[str, float]:
    """Apply the locked active-set geometric IGE formula with total sum four."""
    active = [name for name in LOSS_KEYS if bool(mask[name])]
    values = np.asarray([float(gradients[name]) for name in active], dtype=np.float64)
    if not active or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("Active gradients must be finite and strictly positive")
    geo = float(np.exp(np.mean(np.log(values + float(eps)))))
    ratios = geo / (values + float(eps))
    normalized = 4.0 * ratios / ratios.sum()
    result = {name: 0.0 for name in LOSS_KEYS}
    for name, value in zip(active, normalized):
        result[name] = float(value)
    active_values = np.asarray([result[name] for name in active], dtype=np.float64)
    if not np.all(np.isfinite(active_values)) or np.any(active_values <= 0):
        raise AssertionError("Active coefficients must be finite and positive")
    if abs(float(active_values.sum()) - 4.0) > 1e-6:
        raise AssertionError("Active coefficient sum is not four")
    for name in LOSS_KEYS:
        if not mask[name] and result[name] != 0.0:
            raise AssertionError("Dropped coefficient is not exact zero")
    return result


def variant_contract(variant: str) -> dict:
    mask = active_loss_mask(variant)
    return {
        "variant": variant,
        "active_loss_mask": mask,
        "dropped_loss": DROP_BY_VARIANT.get(variant),
        "attention_mode": ATTENTION_MODE_BY_VARIANT[variant],
        "coefficient_formula": "active-set initial RMS-gradient equalization; active sum=4",
        "dynamic_coefficient_update": False,
        "parameter_objects_removed": False,
    }


class Night3BTrainer(Night3ARTrainer):
    """Night-3A-R training math with only registered Night-3B switches."""

    def __init__(self, data, cfg, variant: str, seed: int, device, eps: float = 1e-12):
        if variant not in VARIANTS:
            raise ValueError(variant)
        # The parent IGE path supplies the exact published FULL_IGE behavior.
        super().__init__(data, cfg, "IGE", seed, device, eps)
        self.night3b_variant = variant
        self.attention_mode = ATTENTION_MODE_BY_VARIANT[variant]

    def new_model(self):
        model = super().new_model()
        model.set_attention_mode(self.attention_mode)
        return model

    def train(self) -> Night3ARTrainingResult:
        # This delegation is intentionally exact for FULL_IGE and differs for
        # attention ablations only through new_model().attention_mode.
        if self.night3b_variant not in DROP_BY_VARIANT:
            result = super().train()
            if result.probe is not None:
                result.probe["registered_coefficients"] = dict(result.coefficients)
                result.probe["active_loss_mask"] = active_loss_mask(self.night3b_variant)
                result.probe["night3b_variant"] = self.night3b_variant
            return result
        return self._train_loss_drop()

    def _train_loss_drop(self) -> Night3ARTrainingResult:
        model = self.new_model()
        initial_state = _clone_state(model)
        initial_state_hash = state_dict_sha256(initial_state)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)
        initial_result = self.forward(model)
        initial_raw_tensors = raw_losses(initial_result, self.features1, self.features2)
        initial_losses = {name: float(initial_raw_tensors[name].detach().cpu()) for name in LOSS_KEYS}

        probe = run_initial_probe(model, self.forward, self.eps)
        model.load_state_dict(initial_state)
        if model_state_sha256(model) != initial_state_hash:
            raise AssertionError("Loss-drop training did not reload the exact initial state")
        mask = active_loss_mask(self.night3b_variant)
        coefficients = active_ige_coefficients(probe["gradients"], mask, self.eps)
        probe["registered_coefficients"] = dict(coefficients)
        probe["active_loss_mask"] = dict(mask)
        probe["night3b_variant"] = self.night3b_variant

        started = time.perf_counter()
        record_steps = set(required_record_steps(int(self.cfg["epochs"])))
        logs = []
        gradient_logs = []
        initial_result = self.forward(model)
        initial_raw_tensors = raw_losses(initial_result, self.features1, self.features2)
        initial_total = calibrated_total(initial_raw_tensors, coefficients)
        initial_gradients = torch.autograd.grad(
            initial_total, [p for p in model.parameters() if p.requires_grad], allow_unused=True
        )
        initial_grad_norm = math.sqrt(sum(
            float(torch.sum(gradient.detach().double() ** 2).cpu())
            for gradient in initial_gradients if gradient is not None
        ))
        logs.append(self._record(model, 0, coefficients, initial_grad_norm, started))
        gradient_logs.append(self._gradient_record(model, optimizer, 0, coefficients))

        for step in range(1, int(self.cfg["epochs"]) + 1):
            result = self.forward(model)
            losses = raw_losses(result, self.features1, self.features2)
            total = calibrated_total(losses, coefficients)
            optimizer.zero_grad()
            total.backward()
            gradient_norm = global_gradient_norm(model)
            optimizer.step()
            if step in record_steps:
                logs.append(self._record(model, step, coefficients, gradient_norm, started))
                gradient_logs.append(self._gradient_record(model, optimizer, step, coefficients))

        expected = sorted(record_steps)
        if [row["step"] for row in logs] != expected or [row["step"] for row in gradient_logs] != expected:
            raise AssertionError("Required Night-3B checkpoints are incomplete")
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


def initial_weighted_gradient_shares(probe: Mapping[str, object], coefficients: Mapping[str, float]) -> dict:
    gradients = probe["gradients"]
    influence = {name: abs(float(coefficients[name])) * float(gradients[name]) for name in LOSS_KEYS}
    total = float(sum(influence.values()))
    return {name: influence[name] / total for name in LOSS_KEYS}


__all__ = [
    "VARIANTS", "DROP_BY_VARIANT", "ATTENTION_MODE_BY_VARIANT", "Night3BTrainer",
    "active_loss_mask", "active_ige_coefficients", "initial_weighted_gradient_shares",
    "variant_contract",
]
