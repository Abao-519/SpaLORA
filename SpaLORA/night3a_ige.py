"""Label-free Night-3A loss calibration on the corrected sparse pipeline.

This module deliberately contains no evaluation or ground-truth loader.  The
four registered variants differ only in the frozen scalar coefficients applied
to the same four raw losses.
"""

from __future__ import annotations

import copy
import hashlib
import io
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .model_corrected import EncoderOverallCorrected
from .preprocess import fix_seed


VARIANTS = ("C0", "C1", "IGE", "ILN")
LOSS_KEYS = (
    "L_rna_recon_raw",
    "L_mod2_recon_raw",
    "L_corr1_raw",
    "L_corr2_raw",
)
ATTENTION_KEYS = (
    "cross_omics_rna_attention",
    "cross_omics_modality2_attention",
    "rna_spatial_attention",
    "rna_feature_attention",
    "modality2_spatial_attention",
    "modality2_feature_attention",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    value = value.detach().cpu()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    if value.is_sparse:
        value = value.coalesce()
        digest.update(value.indices().contiguous().numpy().tobytes())
        digest.update(value.values().contiguous().numpy().tobytes())
    else:
        digest.update(value.contiguous().numpy().tobytes())
    return digest.hexdigest()


def state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        digest.update(tensor_sha256(state[name]).encode("ascii"))
    return digest.hexdigest()


def model_state_sha256(model: torch.nn.Module) -> str:
    return state_dict_sha256(model.state_dict())


def input_sha256(data: Mapping[str, object], obs_names: Sequence[str], gene_names: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for key in (
        "features_omics1",
        "features_omics2",
        "adj_spatial_omics1",
        "adj_feature_omics1",
        "adj_spatial_omics2",
        "adj_feature_omics2",
    ):
        value = data[key]
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        digest.update(key.encode("utf-8"))
        digest.update(tensor_sha256(tensor).encode("ascii"))
    digest.update("\n".join(map(str, obs_names)).encode("utf-8"))
    digest.update("\n".join(map(str, gene_names)).encode("utf-8"))
    return digest.hexdigest()


def raw_losses(result: Mapping[str, torch.Tensor], features1: torch.Tensor,
               features2: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Expose the four original mean-reduced losses before any calibration."""
    return {
        "L_rna_recon_raw": F.mse_loss(features1, result["emb_recon_omics1"]),
        "L_mod2_recon_raw": F.mse_loss(features2, result["emb_recon_omics2"]),
        "L_corr1_raw": F.mse_loss(
            result["emb_latent_omics1"], result["emb_latent_omics1_across_recon"]
        ),
        "L_corr2_raw": F.mse_loss(
            result["emb_latent_omics2"], result["emb_latent_omics2_across_recon"]
        ),
    }


def legacy_m_bad(features1: torch.Tensor) -> torch.Tensor:
    """Locked scalar induced by the public argsort bug; its shape is not used."""
    if features1.ndim != 2:
        raise ValueError("features1 must be locations-by-genes")
    n_genes = features1.shape[1]
    average = features1.mean(dim=0)
    bad_percentile = torch.argsort(average, descending=False).to(features1.dtype) / n_genes
    weights = 1.0 + 5.0 * torch.sigmoid(-10.0 * (bad_percentile - 0.25))
    return weights.mean()


def ige_weights_from_gradients(gradients: Mapping[str, float], eps: float) -> Dict[str, float]:
    values = np.asarray([float(gradients[name]) for name in LOSS_KEYS], dtype=np.float64)
    if values.shape != (4,) or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("IGE gradients must be finite and strictly positive")
    geo = float(np.exp(np.mean(np.log(values + float(eps)))))
    raw = geo / (values + float(eps))
    normalized = 4.0 * raw / raw.sum()
    return {name: float(value) for name, value in zip(LOSS_KEYS, normalized)}


def _rng_snapshot() -> Dict[str, object]:
    result: Dict[str, object] = {
        "cpu": torch.get_rng_state().clone(),
        "numpy": copy.deepcopy(np.random.get_state()),
    }
    if torch.cuda.is_available():
        result["cuda"] = [state.clone() for state in torch.cuda.get_rng_state_all()]
    return result


def rng_snapshot_sha256(snapshot: Mapping[str, object]) -> str:
    buffer = io.BytesIO()
    torch.save(dict(snapshot), buffer)
    return sha256_bytes(buffer.getvalue())


def _gradients_equal(first: Mapping[str, float], second: Mapping[str, float]) -> bool:
    return all(float(first[name]) == float(second[name]) for name in LOSS_KEYS)


def rms_gradient_probe(model: torch.nn.Module, result: Mapping[str, torch.Tensor],
                       losses: Mapping[str, torch.Tensor], eps: float) -> Tuple[Dict[str, float], List[dict]]:
    """Compute independent RMS gradients with allow_unused=True and no .grad writes."""
    named_parameters = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    parameters = [value for _, value in named_parameters]
    summaries: Dict[str, float] = {}
    rows: List[dict] = []
    del result  # the retained graph is owned by losses
    for loss_name in LOSS_KEYS:
        gradients = torch.autograd.grad(
            losses[loss_name], parameters, retain_graph=True, create_graph=False, allow_unused=True
        )
        sq_sum = torch.zeros((), dtype=torch.float64, device=losses[loss_name].device)
        n_elem = 0
        for (parameter_name, parameter), gradient in zip(named_parameters, gradients):
            has_gradient = gradient is not None
            if has_gradient:
                detached = gradient.detach()
                l2 = float(torch.linalg.vector_norm(detached).cpu())
                rms = float(torch.sqrt(torch.mean(detached.double() ** 2)).cpu())
                sq_sum = sq_sum + torch.sum(detached.double() ** 2)
                n_elem += int(parameter.numel())
            else:
                l2 = 0.0
                rms = 0.0
            rows.append(
                {
                    "loss": loss_name,
                    "parameter": parameter_name,
                    "has_gradient": bool(has_gradient),
                    "numel": int(parameter.numel()),
                    "l2_norm": l2,
                    "rms": rms,
                }
            )
        value = torch.sqrt(sq_sum / (float(n_elem) + float(eps)))
        summaries[loss_name] = float(value.detach().cpu())
        for row in rows:
            if row["loss"] == loss_name:
                row["loss_sq_sum"] = float(sq_sum.detach().cpu())
                row["loss_non_none_numel"] = int(n_elem)
                row["loss_rms_gradient"] = summaries[loss_name]
    return summaries, rows


def _clone_state(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _max_tensor_difference(first: Mapping[str, torch.Tensor], second: Mapping[str, torch.Tensor]) -> float:
    differences = []
    for key in first:
        differences.append(float(torch.max(torch.abs(first[key].detach() - second[key].detach())).cpu()))
    return max(differences) if differences else 0.0


def run_initial_probe(model: torch.nn.Module, forward, eps: float) -> dict:
    """Run two non-mutating probes and a reload-forward numerical-envelope check."""
    initial_state = _clone_state(model)
    initial_state_hash = state_dict_sha256(initial_state)
    parameter_grad_before = {
        name: None if value.grad is None else value.grad.detach().clone()
        for name, value in model.named_parameters()
    }
    rng_before = _rng_snapshot()

    result1 = forward(model)
    losses1 = raw_losses(result1, forward.features1, forward.features2)
    gradients1, rows1 = rms_gradient_probe(model, result1, losses1, eps)
    weights1 = ige_weights_from_gradients(gradients1, eps)

    result2 = forward(model)
    losses2 = raw_losses(result2, forward.features1, forward.features2)
    gradients2, rows2 = rms_gradient_probe(model, result2, losses2, eps)
    weights2 = ige_weights_from_gradients(gradients2, eps)
    rng_after = _rng_snapshot()

    state_unchanged = state_dict_sha256(model.state_dict()) == initial_state_hash
    grad_fields_unchanged = True
    for name, parameter in model.named_parameters():
        previous = parameter_grad_before[name]
        if previous is None:
            grad_fields_unchanged &= parameter.grad is None
        else:
            grad_fields_unchanged &= parameter.grad is not None and torch.equal(previous, parameter.grad)
    rng_unchanged = rng_snapshot_sha256(rng_before) == rng_snapshot_sha256(rng_after)

    model.load_state_dict(initial_state)
    reloaded = forward(model)
    reload_raw = raw_losses(reloaded, forward.features1, forward.features2)
    first_raw = {name: losses1[name].detach() for name in LOSS_KEYS}
    reload_values = {name: reload_raw[name].detach() for name in LOSS_KEYS}
    max_reload_difference = _max_tensor_difference(first_raw, reload_values)
    max_scale = max([1.0] + [abs(float(value.cpu())) for value in first_raw.values()])
    envelope = 2.0 * 32.0 * float(torch.finfo(torch.float32).eps) * max_scale
    repeat_raw_values = {name: float(losses2[name].detach().cpu()) for name in LOSS_KEYS}
    repeat_raw_max = max(abs(float(losses1[name].detach().cpu()) - repeat_raw_values[name]) for name in LOSS_KEYS)
    repeat_gradient_max = max(abs(gradients1[name] - gradients2[name]) for name in LOSS_KEYS)
    repeat_weight_max = max(abs(weights1[name] - weights2[name]) for name in LOSS_KEYS)
    gradient_scale = max([1.0] + list(map(abs, gradients1.values())))
    weight_scale = max([1.0] + list(map(abs, weights1.values())))
    repeat_stable = (
        repeat_raw_max <= envelope
        and repeat_gradient_max <= 2.0 * 32.0 * float(torch.finfo(torch.float32).eps) * gradient_scale
        and repeat_weight_max <= 2.0 * 32.0 * float(torch.finfo(torch.float32).eps) * weight_scale
    )

    return {
        "initial_state": initial_state,
        "initial_state_sha256": initial_state_hash,
        "raw_losses": {name: float(losses1[name].detach().cpu()) for name in LOSS_KEYS},
        "repeat_raw_losses": repeat_raw_values,
        "gradients": gradients1,
        "repeat_gradients": gradients2,
        "weights": weights1,
        "repeat_weights": weights2,
        "gradient_parameter_rows": rows1,
        "repeat_gradient_parameter_rows": rows2,
        "repeat_exact": (
            all(float(losses1[name].detach().cpu()) == float(losses2[name].detach().cpu()) for name in LOSS_KEYS)
            and _gradients_equal(gradients1, gradients2)
            and all(weights1[name] == weights2[name] for name in LOSS_KEYS)
        ),
        "repeat_raw_max_abs_difference": repeat_raw_max,
        "repeat_gradient_max_abs_difference": repeat_gradient_max,
        "repeat_weight_max_abs_difference": repeat_weight_max,
        "repeat_within_gpu_envelope": bool(repeat_stable),
        "state_unchanged": bool(state_unchanged),
        "grad_fields_unchanged": bool(grad_fields_unchanged),
        "rng_unchanged": bool(rng_unchanged),
        "rng_before_sha256": rng_snapshot_sha256(rng_before),
        "rng_after_sha256": rng_snapshot_sha256(rng_after),
        "reload_forward_max_abs_difference": max_reload_difference,
        "reload_forward_envelope": envelope,
        "reload_forward_within_envelope": bool(max_reload_difference <= envelope),
    }


def coefficients_for_variant(variant: str, raw_initial: Mapping[str, float], gradients: Mapping[str, float],
                             legacy_factors: Sequence[float], m_bad: float,
                             eps: float) -> Dict[str, float]:
    if variant not in VARIANTS:
        raise ValueError("Unregistered variant: %s" % variant)
    if len(legacy_factors) != 4:
        raise ValueError("Exactly four legacy factors are required")
    if variant == "C0":
        values = list(map(float, legacy_factors))
    elif variant == "C1":
        values = list(map(float, legacy_factors))
        values[0] *= float(m_bad)
    elif variant == "IGE":
        weights = ige_weights_from_gradients(gradients, eps)
        values = [weights[name] for name in LOSS_KEYS]
    else:
        values = [1.0 / (float(raw_initial[name]) + float(eps)) for name in LOSS_KEYS]
    if not np.all(np.isfinite(values)) or np.any(np.asarray(values) <= 0):
        raise AssertionError("Frozen loss coefficients must be finite and positive")
    return {name: float(value) for name, value in zip(LOSS_KEYS, values)}


def calibrated_total(losses: Mapping[str, torch.Tensor], coefficients: Mapping[str, float]) -> torch.Tensor:
    return sum(losses[name] * float(coefficients[name]) for name in LOSS_KEYS)


def required_record_steps(epochs: int) -> List[int]:
    if epochs < 1:
        raise ValueError("epochs must be positive")
    return sorted(
        {
            0, 1, min(5, epochs), min(10, epochs), min(20, epochs), min(40, epochs),
            min(80, epochs), min(int(round(0.20 * epochs)), epochs),
            min(int(round(0.50 * epochs)), epochs), epochs,
        }
    )


def attention_means(result: Mapping[str, torch.Tensor]) -> Dict[str, float]:
    return {
        "cross_omics_rna_attention": float(result["alpha"][:, 0].mean().detach().cpu()),
        "cross_omics_modality2_attention": float(result["alpha"][:, 1].mean().detach().cpu()),
        "rna_spatial_attention": float(result["alpha_omics1"][:, 0].mean().detach().cpu()),
        "rna_feature_attention": float(result["alpha_omics1"][:, 1].mean().detach().cpu()),
        "modality2_spatial_attention": float(result["alpha_omics2"][:, 0].mean().detach().cpu()),
        "modality2_feature_attention": float(result["alpha_omics2"][:, 1].mean().detach().cpu()),
    }


def global_gradient_norm(model: torch.nn.Module) -> float:
    squared = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            squared += float(torch.sum(parameter.grad.detach().double() ** 2).cpu())
    return math.sqrt(squared)


@dataclass
class TrainingResult:
    output: Dict[str, np.ndarray]
    logs: List[dict]
    model: torch.nn.Module
    probe: Optional[dict]
    coefficients: Dict[str, float]
    initial_losses: Dict[str, float]
    initial_state_sha256: str
    final_state_sha256: str


class _Forward:
    def __init__(self, features1: torch.Tensor, features2: torch.Tensor,
                 adjacencies: Sequence[torch.Tensor]):
        self.features1 = features1
        self.features2 = features2
        self.adjacencies = tuple(adjacencies)

    def __call__(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        return model(self.features1, self.features2, *self.adjacencies)


class Night3ATrainer:
    """One registered run using immutable corrected-sparse inputs."""

    def __init__(self, data: Mapping[str, object], cfg: Mapping[str, object], variant: str,
                 seed: int, device: torch.device, eps: float = 1e-12):
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant = variant
        self.seed = int(seed)
        self.cfg = dict(cfg)
        self.device = device
        self.eps = float(eps)
        self.features1 = torch.as_tensor(data["features_omics1"], dtype=torch.float32, device=device)
        self.features2 = torch.as_tensor(data["features_omics2"], dtype=torch.float32, device=device)
        self.adjacencies = tuple(
            data[name].to(device) for name in (
                "adj_spatial_omics1", "adj_feature_omics1",
                "adj_spatial_omics2", "adj_feature_omics2",
            )
        )
        if any(not adjacency.is_sparse for adjacency in self.adjacencies):
            raise AssertionError("Corrected adjacency must remain sparse end to end")
        self.forward = _Forward(self.features1, self.features2, self.adjacencies)

    def new_model(self) -> EncoderOverallCorrected:
        fix_seed(self.seed)
        return EncoderOverallCorrected(
            self.features1.shape[1], int(self.cfg["embedding_dim"]),
            self.features2.shape[1], int(self.cfg["embedding_dim"]),
        ).to(self.device)

    def _record(self, model: torch.nn.Module, step: int, coefficients: Mapping[str, float],
                gradient_norm: float, started: float) -> dict:
        model.eval()
        with torch.no_grad():
            result = self.forward(model)
            losses = raw_losses(result, self.features1, self.features2)
            contributions = {name: float((losses[name] * float(coefficients[name])).cpu()) for name in LOSS_KEYS}
            total = float(sum(contributions.values()))
            if not np.isfinite(total) or total <= 0:
                raise FloatingPointError("Non-finite or non-positive calibrated total loss")
            combined = result["emb_latent_combined"]
            row = {
                "step": int(step),
                "fraction_of_training": float(step / int(self.cfg["epochs"])),
                "wall_seconds": float(__import__("time").perf_counter() - started),
                "gradient_norm": float(gradient_norm),
                "embedding_l2_mean": float(torch.linalg.vector_norm(combined, dim=1).mean().cpu()),
                "embedding_finite": bool(torch.isfinite(combined).all().cpu()),
                "total_loss": total,
                "checkpoint_state_sha256": model_state_sha256(model),
            }
            for name in LOSS_KEYS:
                short = name.replace("L_", "").replace("_raw", "")
                row[name] = float(losses[name].cpu())
                row[short + "_coefficient"] = float(coefficients[name])
                row[short + "_contribution"] = contributions[name]
                row[short + "_contribution_fraction"] = contributions[name] / total
            row.update(attention_means(result))
        model.train()
        return row

    def train(self) -> TrainingResult:
        import time

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
        # Initial gradient of the exact calibrated objective, without populating .grad.
        initial_result = self.forward(model)
        initial_raw_tensors = raw_losses(initial_result, self.features1, self.features2)
        initial_total = calibrated_total(initial_raw_tensors, coefficients)
        gradients_initial = torch.autograd.grad(
            initial_total, [p for p in model.parameters() if p.requires_grad], allow_unused=True
        )
        initial_grad_norm = math.sqrt(sum(
            float(torch.sum(g.detach().double() ** 2).cpu()) for g in gradients_initial if g is not None
        ))
        logs.append(self._record(model, 0, coefficients, initial_grad_norm, started))

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
        if [row["step"] for row in logs] != sorted(record_steps):
            raise AssertionError("Required Night-3A checkpoints are incomplete")
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
        return TrainingResult(
            output=output,
            logs=logs,
            model=model,
            probe=probe,
            coefficients=coefficients,
            initial_losses=initial_losses,
            initial_state_sha256=initial_state_hash,
            final_state_sha256=model_state_sha256(model),
        )


def registered_variant_contract(config: Mapping[str, object]) -> Dict[str, dict]:
    """Return the only permitted per-variant differences for audit/tests."""
    return {
        "C0": {"calibration": "legacy_dataset_gamma", "rna_global_scale": 1.0, "gene_shape": False},
        "C1": {"calibration": "legacy_dataset_gamma", "rna_global_scale": "locked_m_bad", "gene_shape": False},
        "IGE": {"calibration": "initial_gradient_equalization", "rna_global_scale": None, "gene_shape": False},
        "ILN": {"calibration": "initial_loss_normalization", "rna_global_scale": None, "gene_shape": False},
    }
