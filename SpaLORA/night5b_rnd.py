"""Locked Night-5B clean-room candidates; this module never reads semantic labels."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

from .night3a_ige import (LOSS_KEYS, _clone_state, calibrated_total, global_gradient_norm,
                          model_state_sha256, raw_losses, required_record_steps,
                          run_initial_probe, state_dict_sha256)
from .night3b_ablation import active_ige_coefficients
from .night5a_rnd import (Night5AModel, Night5ATrainer, Night5ATrainingResult,
                          canonical_sha256, sha256_file)
from .preprocess import fix_seed


REFERENCE_MAP = {
    "B00_C00_FULL_IGE": "C00_FULL_IGE",
    "B01_C04_SHRINK25": "C04_SHRINK25",
    "B02_C09_RNA_ANCHOR10": "C09_RNA_ANCHOR10",
    "B03_C10_MNN_TRIPLET01": "C10_MNN_TRIPLET01",
}
SECONDLOOK_MAP = {
    "B04_SECONDLOOK_RELIABILITY25": "C06_RELIABILITY25",
    "B05_SECONDLOOK_RELIABILITY50": "C07_RELIABILITY50",
    "B06_SECONDLOOK_RNA_ANCHOR05": "C08_RNA_ANCHOR05",
    "B07_SECONDLOOK_HYBRID_IGE50": "C14_HYBRID_IGE50",
    "B08_SECONDLOOK_DGI01": "C16_DGI01",
}
DIFFUSION_MAP = {
    "B17_C09_DIFFUSE10": "C09_RNA_ANCHOR10",
    "B18_C09_DIFFUSE25": "C09_RNA_ANCHOR10",
    "B19_C10_DIFFUSE10": "C10_MNN_TRIPLET01",
    "B20_C10_DIFFUSE25": "C10_MNN_TRIPLET01",
}
WITHHELD_TOKENS = ("p22", "d1", "gse198353", "night4b")


def load_registry(path: Path) -> dict:
    registry = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = [row.get("id") for row in registry.get("candidates", [])]
    expected = ["B%02d" % value for value in range(25)]
    if len(ids) != 25 or [value.split("_", 1)[0] for value in ids] != expected or len(set(ids)) != 25:
        raise RuntimeError("Registry must contain exactly B00-B24 once and in order")
    if registry.get("parent_commit") != "f9aeed223d38a897e190ac55ca741af1246071d2":
        raise RuntimeError("Night-5B registry parent drift")
    if any(x in (" ".join(ids)).lower() for x in WITHHELD_TOKENS):
        raise RuntimeError("Withheld dataset token in candidate registry")
    return registry


def registry_contracts(registry: Mapping[str, object]) -> Dict[str, dict]:
    result = {}
    for row in registry["candidates"]:
        contract = dict(row)
        contract["config_sha256"] = canonical_sha256(row)
        result[row["id"]] = contract
    if len(result) != 25 or len({row["config_sha256"] for row in result.values()}) != 25:
        raise RuntimeError("Night-5B candidate configuration SHAs are not unique")
    return result


def assert_development_dataset(dataset: str) -> None:
    value = str(dataset).lower()
    if value not in ("a1", "placenta") or any(token in value for token in WITHHELD_TOKENS):
        raise RuntimeError("Night-5B refuses withheld/non-development dataset: %s" % dataset)


def latent_reliability_weights(first: np.ndarray, second: np.ndarray, k: int = 20,
                               epsilon: float = 1e-12) -> np.ndarray:
    """PCA-free within/cross local-predictability weights."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.ndim != 2 or second.ndim != 2 or first.shape[0] != second.shape[0] or len(first) < 2:
        raise ValueError("Latent modalities must be aligned 2D matrices")
    k = min(max(1, int(k)), len(first) - 1)

    def standardize(x):
        x = x - x.mean(0, keepdims=True)
        return x / np.maximum(x.std(0, keepdims=True), epsilon)

    first, second = standardize(first), standardize(second)
    n1 = NearestNeighbors(n_neighbors=k + 1).fit(first).kneighbors(return_distance=False)[:, 1:]
    n2 = NearestNeighbors(n_neighbors=k + 1).fit(second).kneighbors(return_distance=False)[:, 1:]

    def score(target, own, cross):
        own_error = np.mean((target - target[own].mean(1)) ** 2, axis=1)
        cross_error = np.mean((target - target[cross].mean(1)) ** 2, axis=1)
        denom = own_error + cross_error
        result = np.divide(cross_error, denom + epsilon)
        result[denom <= epsilon] = 0.5
        return np.clip(result, 0.001, 0.999)

    scores = np.column_stack((score(first, n1, n2), score(second, n2, n1)))
    result = scores / np.maximum(scores.sum(1, keepdims=True), epsilon)
    if not np.isfinite(result).all() or np.max(np.abs(result.sum(1) - 1.0)) > 1e-12:
        raise AssertionError("Invalid latent reliability weights")
    return result.astype(np.float32)


def reliability_audit(weights: np.ndarray) -> dict:
    w = np.asarray(weights, np.float64)
    entropy = -np.sum(np.clip(w, 1e-12, 1.0) * np.log(np.clip(w, 1e-12, 1.0)), axis=1)
    return {
        "shape": list(w.shape), "mean": w.mean(0).tolist(), "minimum": w.min(0).tolist(),
        "maximum": w.max(0).tolist(), "entropy_mean": float(entropy.mean()),
        "extreme_fraction": float(np.mean((w.min(1) < 0.1) | (w.max(1) > 0.9))),
        "row_sum_max_deviation": float(np.max(np.abs(w.sum(1) - 1.0))),
    }


def undirected_edges(adjacency: torch.Tensor) -> torch.Tensor:
    if not adjacency.is_sparse:
        raise AssertionError("Spatial graph densified")
    idx = adjacency.coalesce().indices()
    keep = idx[0] < idx[1]
    edges = idx[:, keep].t().contiguous()
    if not len(edges):
        return torch.empty((0, 2), dtype=torch.long, device=idx.device)
    return torch.unique(edges, dim=0)


def laplacian_loss(embedding: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("Edges must have shape [m,2]")
    if not len(edges):
        return embedding.sum() * 0.0
    z = F.normalize(embedding, p=2, dim=1, eps=1e-12)
    return torch.sum((z[edges[:, 0]] - z[edges[:, 1]]) ** 2, dim=1).mean()


def single_step_diffusion(embedding: np.ndarray, adjacency: torch.Tensor, alpha: float) -> np.ndarray:
    if float(alpha) not in (0.0, 0.1, 0.25):
        raise ValueError("Unregistered diffusion alpha")
    z = torch.as_tensor(np.asarray(embedding, np.float32))
    if float(alpha) == 0.0:
        return np.asarray(embedding, np.float32).copy()
    if not adjacency.is_sparse:
        raise AssertionError("Diffusion graph must remain sparse")
    out = (1.0 - float(alpha)) * z + float(alpha) * torch.sparse.mm(adjacency.cpu(), z)
    return F.normalize(out, p=2, dim=1, eps=1e-12).numpy().astype(np.float32)


class Night5BModel(Night5AModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("latent_reliability_active_flag", torch.zeros((), dtype=torch.uint8), persistent=True)

    def activate_latent_reliability(self, weights: np.ndarray) -> None:
        tensor = torch.as_tensor(weights, dtype=self.reliability_weights.dtype,
                                 device=self.reliability_weights.device)
        if tensor.shape != (self.reliability_weights.shape[0], 2):
            raise ValueError("Latent reliability shape mismatch")
        self.reliability_weights.copy_(tensor)
        self.latent_reliability_active_flag.fill_(1)

    def _combine(self, first, second, layer, location):
        if location == "cross" and bool(self.latent_reliability_active_flag.item()):
            stacked = torch.stack((first, second), dim=1)
            alpha = self.reliability_weights.to(dtype=first.dtype, device=first.device)
            return torch.sum(stacked * alpha.unsqueeze(-1), dim=1), alpha
        return super()._combine(first, second, layer, location)


class Night5BTrainer(Night5ATrainer):
    def __init__(self, data: Mapping[str, object], cfg: Mapping[str, object], candidate: Mapping[str, object],
                 seed: int, device: torch.device, artifacts: Mapping[str, object], eps: float = 1e-12):
        self.night5b_candidate = dict(candidate)
        proxy = dict(candidate)
        proxy["id"] = "C09_RNA_ANCHOR10" if candidate.get("rna_anchor_eta") == 1.0 else (
            "C08_RNA_ANCHOR05" if candidate.get("rna_anchor_eta") == 0.5 else "C04_SHRINK25")
        super().__init__(data, cfg, proxy, seed, device, artifacts, eps)
        self.candidate = self.night5b_candidate
        self.candidate_id = str(candidate["id"])
        self.edges = undirected_edges(self.adjacencies[0])
        self.laplacian_coefficient = None
        self.laplacian_probe = None
        if "laplacian_target_gradient_fraction" in candidate:
            self._initialize_laplacian_coefficient()

    def new_model(self):
        fix_seed(self.seed)
        candidate = self.night5b_candidate
        attention_policy = str(candidate.get("attention", "shrink_to_uniform"))
        learned_fraction = 0.25 if attention_policy == "shrink_to_uniform" else None
        return Night5BModel(
            self.features1.shape[1], int(self.cfg["embedding_dim"]), self.features2.shape[1],
            int(self.cfg["embedding_dim"]), attention_policy=attention_policy,
            learned_fraction=learned_fraction,
            reliability_weights=np.full((len(self.features1), 2), 0.5, np.float32),
            residual_encoder=False, dgi_head="dgi_weight" in candidate,
        ).to(self.device)

    @staticmethod
    def _rms(grads: Sequence[Optional[torch.Tensor]]) -> float:
        values = [g.detach().reshape(-1).double() for g in grads if g is not None]
        if not values:
            return 0.0
        packed = torch.cat(values)
        return float(torch.sqrt(torch.mean(packed ** 2)).cpu())

    def _initialize_laplacian_coefficient(self) -> None:
        model = self.new_model()
        result = self.forward(model)
        losses = raw_losses(result, self.features1, self.features2)
        params = [p for p in model.parameters() if p.requires_grad]
        active_rms = []
        for name in LOSS_KEYS:
            if name == "L_corr2_raw":
                continue
            active_rms.append(self._rms(torch.autograd.grad(losses[name], params, retain_graph=True, allow_unused=True)))
        spatial = laplacian_loss(result["emb_latent_combined"], self.edges)
        spatial_rms = self._rms(torch.autograd.grad(spatial, params, allow_unused=True))
        active_mean = float(np.mean(active_rms))
        target = float(self.night5b_candidate["laplacian_target_gradient_fraction"])
        self.laplacian_coefficient = target * active_mean / (spatial_rms + self.eps)
        self.laplacian_probe = {"target_fraction": target, "active_raw_loss_gradient_rms_mean": active_mean,
                                "spatial_raw_gradient_rms": spatial_rms,
                                "frozen_coefficient": float(self.laplacian_coefficient),
                                "weighted_spatial_gradient_rms": float(self.laplacian_coefficient * spatial_rms)}
        if not np.isfinite(self.laplacian_coefficient) or self.laplacian_coefficient <= 0:
            raise FloatingPointError("Invalid frozen Laplacian coefficient")

    def _auxiliary_loss(self, model, result):
        candidate = self.night5b_candidate
        z = result["emb_latent_combined"]
        pieces, audit = [], []
        if "triplet_weight" in candidate:
            rows = torch.as_tensor(self.artifacts["triplets"], dtype=torch.long, device=self.device)
            raw = F.triplet_margin_loss(z[rows[:, 0]], z[rows[:, 1]], z[rows[:, 2]],
                                        margin=float(candidate["triplet_margin"]), reduction="mean")
            pieces.append(raw * float(candidate["triplet_weight"])); audit.append(("mnn_triplet", float(raw.detach().cpu()), float(candidate["triplet_weight"])))
        if "dgi_weight" in candidate:
            permutation = torch.as_tensor(self.artifacts["dgi_permutation"], dtype=torch.long, device=self.device)
            corrupted = model(self.features1[permutation], self.features2[permutation], *self.adjacencies)
            summary = torch.sigmoid(z.mean(0))
            positive = torch.sum(torch.mm(z, model.dgi_bilinear) * summary, dim=1)
            negative = torch.sum(torch.mm(corrupted["emb_latent_combined"], model.dgi_bilinear) * summary, dim=1)
            raw = 0.5 * (F.binary_cross_entropy_with_logits(positive, torch.ones_like(positive)) +
                         F.binary_cross_entropy_with_logits(negative, torch.zeros_like(negative)))
            pieces.append(raw * float(candidate["dgi_weight"])); audit.append(("dgi", float(raw.detach().cpu()), float(candidate["dgi_weight"])))
        if self.laplacian_coefficient is not None:
            raw = laplacian_loss(z, self.edges)
            pieces.append(raw * float(self.laplacian_coefficient)); audit.append(("laplacian", float(raw.detach().cpu()), float(self.laplacian_coefficient)))
        total = sum(pieces, z.sum() * 0.0)
        return total, {"name": "+".join(x[0] for x in audit) or "none", "raw": float(sum(x[1] for x in audit)),
                       "weight": float(sum(x[2] for x in audit))}

    def train(self):
        if self.candidate_id not in ("B14_LATENT_RELIABILITY25", "B15_LATENT_RELIABILITY50"):
            result = super().train()
            if self.laplacian_probe is not None:
                result.auxiliary["laplacian_gradient_probe"] = self.laplacian_probe
            return result
        return self._train_latent_reliability()

    def _train_latent_reliability(self):
        model = self.new_model(); initial_state = _clone_state(model); initial_hash = state_dict_sha256(initial_state)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)
        initial = self.forward(model); initial_raw = raw_losses(initial, self.features1, self.features2)
        initial_losses = {k: float(v.detach().cpu()) for k, v in initial_raw.items()}
        probe = run_initial_probe(model, self.forward, self.eps); model.load_state_dict(initial_state)
        coefficients = active_ige_coefficients(probe["gradients"], {k: k != "L_corr2_raw" for k in LOSS_KEYS}, self.eps)
        started = time.perf_counter(); record_steps = set(required_record_steps(int(self.cfg["epochs"]))); logs=[]; audit=None
        result = self.forward(model)
        base = calibrated_total(raw_losses(result, self.features1, self.features2), coefficients)
        grads = torch.autograd.grad(base, [p for p in model.parameters() if p.requires_grad], allow_unused=True)
        logs.append(self._record(model, 0, coefficients, math.sqrt(sum(float(torch.sum(g.detach().double() ** 2).cpu()) for g in grads if g is not None)), started))
        for step in range(1, int(self.cfg["epochs"]) + 1):
            result = self.forward(model); total = calibrated_total(raw_losses(result, self.features1, self.features2), coefficients)
            optimizer.zero_grad(); total.backward(); grad_norm = global_gradient_norm(model); optimizer.step()
            if step == 100:
                model.eval()
                with torch.no_grad(): warm = self.forward(model)
                raw_weights = latent_reliability_weights(warm["emb_latent_omics1"].cpu().numpy(), warm["emb_latent_omics2"].cpu().numpy(), 20)
                rho = float(self.night5b_candidate["reliability_fraction_rho"])
                weights = (1.0 - rho) * 0.5 + rho * raw_weights
                model.activate_latent_reliability(weights.astype(np.float32)); model.train()
                swapped = latent_reliability_weights(warm["emb_latent_omics2"].cpu().numpy(), warm["emb_latent_omics1"].cpu().numpy(), 20)
                audit = reliability_audit(weights); audit.update({"rho": rho, "warmup_epoch": 100,
                    "raw_modality_swap_max_error": float(np.max(np.abs(raw_weights - swapped[:, ::-1]))),
                    "weights_sha256": hashlib.sha256(np.ascontiguousarray(weights.astype(np.float32)).tobytes()).hexdigest(),
                    "frozen_after_warmup": True})
            if step in record_steps: logs.append(self._record(model, step, coefficients, grad_norm, started))
        if audit is None or not bool(model.latent_reliability_active_flag.item()):
            raise AssertionError("Latent reliability was not frozen at warmup")
        model.eval()
        with torch.no_grad(): result = self.forward(model)
        output = {"emb_latent_omics1": F.normalize(result["emb_latent_omics1"], p=2, dim=1).cpu().numpy(),
                  "emb_latent_omics2": F.normalize(result["emb_latent_omics2"], p=2, dim=1).cpu().numpy(),
                  "SpaLORA": F.normalize(result["emb_latent_combined"], p=2, dim=1).cpu().numpy(),
                  "alpha_omics1": result["alpha_omics1"].cpu().numpy(), "alpha_omics2": result["alpha_omics2"].cpu().numpy(), "alpha": result["alpha"].cpu().numpy()}
        return Night5ATrainingResult(output, logs, model, probe, coefficients, initial_losses, initial_hash,
            model_state_sha256(model), {"latent_reliability": audit, "active_loss_names": [k for k,v in coefficients.items() if v],
            "active_loss_coefficient_sum": float(sum(coefficients.values())), "corr2_objective_contribution_exact_zero": coefficients["L_corr2_raw"] == 0.0,
            "parameter_count": int(sum(p.numel() for p in model.parameters()))})


def source_candidate_contract(night5a_contracts: Mapping[str, dict], source_id: str) -> dict:
    if source_id not in night5a_contracts:
        raise RuntimeError("Missing locked Night-5A source candidate: %s" % source_id)
    return dict(night5a_contracts[source_id])


__all__ = ["DIFFUSION_MAP", "REFERENCE_MAP", "SECONDLOOK_MAP", "Night5BModel", "Night5BTrainer",
           "assert_development_dataset", "laplacian_loss", "latent_reliability_weights", "load_registry",
           "registry_contracts", "reliability_audit", "single_step_diffusion", "source_candidate_contract",
           "undirected_edges"]
