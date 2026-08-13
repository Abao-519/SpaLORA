"""Fail-closed runtime semantic contracts for the Night-5C correction."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import torch.nn as nn

from SpaLORA.night5b_rnd import Night5BModel, Night5BTrainer
from SpaLORA.preprocess import fix_seed


CORRECTIVE_IDS = (
    "B21_C09_LAPLACIAN005", "B22_C09_LAPLACIAN010",
    "B23_C10_LAPLACIAN005", "B24_C10_LAPLACIAN010",
)


def canonical_sha256(payload: Mapping[str, object]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ParameterlessAttention(nn.Module):
    """Uniform two-modality attention with no parameters or buffers."""
    def forward(self, stacked: torch.Tensor) -> torch.Tensor:
        if stacked.ndim != 3 or stacked.shape[1] != 2:
            raise ValueError("Expected [n,2,d] modality stack")
        return torch.full((stacked.shape[0], 2), 0.5, dtype=stacked.dtype, device=stacked.device)


class Night5CUniformModel(Night5BModel):
    """Night-5B architecture with genuinely parameterless uniform fusion."""
    def __init__(self, *args, **kwargs):
        kwargs["attention_policy"] = "uniform_all"
        kwargs["learned_fraction"] = None
        super().__init__(*args, **kwargs)
        self.attention1 = ParameterlessAttention()
        self.attention2 = ParameterlessAttention()
        self.cross_attention = ParameterlessAttention()
        self.attention_policy = "uniform_all"
        self.learned_fraction = None


class Night5CTrainer(Night5BTrainer):
    """Corrected trainer restricted to the four preregistered Laplacian candidates."""
    def new_model(self):
        candidate = self.night5b_candidate
        if candidate.get("id") not in CORRECTIVE_IDS:
            raise RuntimeError("Night-5C trainer refuses non-corrective candidate")
        if candidate.get("attention") != "uniform_all":
            raise RuntimeError("Night-5C corrective candidate must declare uniform_all")
        fix_seed(self.seed)
        return Night5CUniformModel(
            self.features1.shape[1], int(self.cfg["embedding_dim"]), self.features2.shape[1],
            int(self.cfg["embedding_dim"]), reliability_weights=np.full((len(self.features1), 2), 0.5, np.float32),
            residual_encoder=False, dgi_head=False,
        ).to(self.device)


def load_corrective_registry(path: Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = [row.get("id") for row in payload.get("candidates", [])]
    if ids != list(CORRECTIVE_IDS):
        raise RuntimeError("Corrective registry must contain exactly B21-B24 in order")
    return payload


def declared_contract(candidate: Mapping[str, object]) -> dict:
    return {
        "candidate_id": candidate["id"],
        "actual_attention_policy": "uniform_all",
        "actual_learned_fraction": None,
        "enabled_mechanisms": sorted(
            ["ige_base", "laplacian"] +
            (["rna_anchor"] if "rna_anchor_eta" in candidate else []) +
            (["mnn_triplet"] if "triplet_weight" in candidate else [])
        ),
        "learnable_attention_parameter_count": 0,
        "uniform_weight": [0.5, 0.5],
    }


def resolved_contract(trainer: Night5CTrainer, model: nn.Module) -> dict:
    attention_names = [name for name, _ in model.named_parameters() if "attention" in name.lower()]
    mechanisms = ["ige_base", "laplacian"]
    candidate = trainer.night5b_candidate
    if "rna_anchor_eta" in candidate:
        mechanisms.append("rna_anchor")
    if "triplet_weight" in candidate:
        mechanisms.append("mnn_triplet")
    return {
        "candidate_id": candidate["id"],
        "actual_attention_policy": getattr(model, "attention_policy", None),
        "actual_learned_fraction": getattr(model, "learned_fraction", None),
        "enabled_mechanisms": sorted(mechanisms),
        "learnable_attention_parameter_count": int(sum(p.numel() for n, p in model.named_parameters() if "attention" in n.lower())),
        "learnable_attention_parameter_names": attention_names,
        "uniform_weight": [0.5, 0.5],
        "laplacian_frozen_coefficient": float(trainer.laplacian_coefficient),
    }


def contract_sha(contract: Mapping[str, object]) -> str:
    return canonical_sha256({k: v for k, v in contract.items() if k != "laplacian_frozen_coefficient"})


def assert_contract_match(declared: Mapping[str, object], resolved: Mapping[str, object]) -> None:
    keys = ("candidate_id", "actual_attention_policy", "actual_learned_fraction",
            "enabled_mechanisms", "learnable_attention_parameter_count", "uniform_weight")
    mismatches = {key: {"declared": declared.get(key), "resolved": resolved.get(key)}
                  for key in keys if declared.get(key) != resolved.get(key)}
    if mismatches:
        raise RuntimeError("Runtime semantic contract mismatch: %s" % json.dumps(mismatches, sort_keys=True))
    if resolved.get("learnable_attention_parameter_names"):
        raise RuntimeError("Uniform model retains learnable attention parameter names")


def assert_uniform_forward(trainer: Night5CTrainer, model: nn.Module) -> dict:
    with torch.no_grad():
        output = trainer.forward(model)
    deviations = {}
    for key in ("alpha_omics1", "alpha_omics2", "alpha"):
        values = output[key]
        expected = torch.full_like(values, 0.5)
        deviation = float(torch.max(torch.abs(values - expected)).cpu())
        deviations[key] = deviation
        if not torch.equal(values, expected):
            raise RuntimeError("Uniform attention forward mismatch for %s" % key)
    return deviations


def generic_resolved_contract(candidate: Mapping[str, object]) -> dict:
    """Explicit semantic inventory for all locked B00-B24 candidates."""
    cid = str(candidate["id"])
    prefix = cid.split("_", 1)[0]
    if prefix in ("B00",):
        policy, fraction = "learned", 1.0
    elif prefix in ("B01", "B09", "B10", "B11", "B12", "B13", "B16"):
        policy, fraction = "shrink_to_uniform", 0.25
    elif prefix in ("B04", "B05"):
        policy, fraction = "frozen_local_reliability", None
    elif prefix in ("B14", "B15"):
        policy, fraction = "shrink_to_uniform_with_latent_cross_after_warmup", 0.25
    else:
        policy, fraction = "uniform_all", None
    mechanisms = ["ige_base"]
    family = str(candidate.get("family", ""))
    if "anchor" in family or "rna_anchor_eta" in candidate:
        mechanisms.append("rna_anchor")
    if "mnn" in family or "triplet_weight" in candidate:
        mechanisms.append("mnn_triplet")
    if "dgi" in family or "dgi_weight" in candidate:
        mechanisms.append("dgi")
    if "reliability" in family:
        mechanisms.append("local_reliability")
    if "diffusion" in family:
        mechanisms.append("posthoc_spatial_diffusion")
    if "laplacian" in family or "laplacian_target_gradient_fraction" in candidate:
        mechanisms.append("laplacian")
    return {"candidate_id": cid, "actual_attention_policy": policy,
            "actual_learned_fraction": fraction, "enabled_mechanisms": sorted(set(mechanisms))}


__all__ = ["CORRECTIVE_IDS", "Night5CTrainer", "Night5CUniformModel", "ParameterlessAttention",
           "assert_contract_match", "assert_uniform_forward", "canonical_sha256", "contract_sha",
           "declared_contract", "generic_resolved_contract", "load_corrective_registry", "resolved_contract"]
