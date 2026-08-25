"""Zero-start, uncertainty-gated relation refinement for Night-17C.

The trainable producer starts byte-exactly from a deterministic relation-smoothed
carrier. Candidate disagreement creates a node trust gate: unreliable nodes keep
the smooth carrier (self-return), while reliable nodes may receive a bounded
learned residual. Public annotations are absent from this module.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch

from SpaLORA.night17b_sfrd import (
    RelationPosterior,
    row_normalize,
    sha256_array,
    standardize,
)


def stratified_permute_relation(
    posterior: RelationPosterior,
    is_spatial: np.ndarray,
    seed: int = 20260826,
) -> RelationPosterior:
    """Permute relation locations within edge-type strata, preserving each distribution."""

    strata = np.asarray(is_spatial, dtype=bool)
    if strata.size != posterior.probability_same.size:
        raise ValueError("edge stratum length differs from relation posterior")
    order = np.arange(strata.size, dtype=np.int64)
    for offset, value in enumerate((False, True)):
        indices = np.flatnonzero(strata == value)
        rng = np.random.RandomState(int(seed) + offset)
        order[indices] = indices[rng.permutation(indices.size)]
    return RelationPosterior(
        probability_same=posterior.probability_same[order].copy(),
        uncertainty=posterior.uncertainty[order].copy(),
        positive_weight=posterior.positive_weight[order].copy(),
        negative_weight=posterior.negative_weight[order].copy(),
        candidate_weights=posterior.candidate_weights.copy(),
        selected_candidate_count=posterior.selected_candidate_count,
    )


def node_trust_gate(
    n: int,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    is_spatial: np.ndarray,
    posterior: RelationPosterior,
) -> tuple[np.ndarray, Mapping[str, float]]:
    """Aggregate pair confidence and robustly map it to a [0,1] node gate."""

    confidence = (1.0 - posterior.uncertainty.astype(np.float64)) * np.abs(
        2.0 * posterior.probability_same.astype(np.float64) - 1.0
    )
    pair_scale = np.where(np.asarray(is_spatial, dtype=bool), 1.0, 0.5)
    weighted = confidence * pair_scale
    mass = np.zeros(n, dtype=np.float64)
    degree = np.zeros(n, dtype=np.float64)
    np.add.at(mass, pair_i, weighted)
    np.add.at(mass, pair_j, weighted)
    np.add.at(degree, pair_i, pair_scale)
    np.add.at(degree, pair_j, pair_scale)
    raw = mass / np.maximum(degree, 1e-12)
    positive = raw[degree > 0]
    if positive.size == 0:
        raise ValueError("node trust gate has no supported node")
    lower = float(np.quantile(positive, 0.25))
    upper = float(np.quantile(positive, 0.75))
    if upper - lower < 1e-6:
        # A nearly unanimous relation bank is informative rather than untrusted.
        # Falling back to the already bounded raw confidence avoids a degenerate
        # all-zero gate without adding a lane-specific threshold.
        gate = np.clip(raw, 0.0, 1.0)
        calibration_mode = "RAW_CONFIDENCE_DEGENERATE_IQR"
    else:
        gate = np.clip((raw - lower) / (upper - lower), 0.0, 1.0)
        calibration_mode = "ROBUST_IQR"
    gate[degree <= 0] = 0.0
    diagnostics = {
        "raw_min": float(raw.min()),
        "raw_median": float(np.median(raw)),
        "raw_max": float(raw.max()),
        "calibration_q25": lower,
        "calibration_q75": upper,
        "gate_zero_fraction": float(np.mean(gate == 0.0)),
        "gate_one_fraction": float(np.mean(gate == 1.0)),
        "gate_mean": float(gate.mean()),
        "calibration_mode": calibration_mode,
    }
    return gate.astype(np.float32), diagnostics


class ZeroStartRelationEncoder(torch.nn.Module):
    """Shared residual core whose final layer is exactly zero at construction."""

    def __init__(
        self,
        view1_dim: int,
        view2_dim: int,
        retained_dim: int,
        hidden_dim: int,
        residual_scale: float,
    ) -> None:
        super().__init__()
        self.view1_adapter = torch.nn.Linear(view1_dim, hidden_dim)
        self.view2_adapter = torch.nn.Linear(view2_dim, hidden_dim)
        self.retained_adapter = torch.nn.Linear(retained_dim, hidden_dim)
        self.smooth_adapter = torch.nn.Linear(retained_dim, hidden_dim)
        self.hidden = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 4, hidden_dim),
            torch.nn.GELU(),
        )
        self.residual_out = torch.nn.Linear(hidden_dim, retained_dim)
        torch.nn.init.zeros_(self.residual_out.weight)
        torch.nn.init.zeros_(self.residual_out.bias)
        self.residual_scale = float(residual_scale)

    def forward(
        self,
        view1: torch.Tensor,
        view2: torch.Tensor,
        retained: torch.Tensor,
        smooth: torch.Tensor,
        trust_gate: torch.Tensor,
        view_mask: tuple[float, float] = (1.0, 1.0),
    ) -> torch.Tensor:
        h1 = torch.tanh(self.view1_adapter(view1)) * float(view_mask[0])
        h2 = torch.tanh(self.view2_adapter(view2)) * float(view_mask[1])
        hr = torch.tanh(self.retained_adapter(retained))
        hs = torch.tanh(self.smooth_adapter(smooth))
        residual = torch.tanh(self.residual_out(self.hidden(torch.cat([h1, h2, hr, hs], dim=1))))
        return smooth + self.residual_scale * trust_gate[:, None] * residual


def balanced_soft_relation_loss(
    predicted_same: torch.Tensor,
    target_same: torch.Tensor,
    relation_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Equal-weight positive/negative soft BCE, independent of pair prevalence."""

    positive_mass = torch.sum(relation_weight * target_same)
    negative_mass = torch.sum(relation_weight * (1.0 - target_same))
    if float(positive_mass.detach().cpu()) <= 0 or float(negative_mass.detach().cpu()) <= 0:
        raise ValueError("balanced relation loss requires positive and negative soft mass")
    positive_loss = -torch.sum(relation_weight * target_same * torch.log(predicted_same)) / positive_mass
    negative_loss = -torch.sum(
        relation_weight * (1.0 - target_same) * torch.log(1.0 - predicted_same)
    ) / negative_mass
    return 0.5 * (positive_loss + negative_loss), positive_loss, negative_loss, positive_mass, negative_mass


def _state_sha(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode("utf-8"))
        digest.update(state[key].detach().cpu().numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ZeroStartTrainResult:
    representation: np.ndarray
    state_dict: Mapping[str, torch.Tensor]
    diagnostics: Mapping[str, object]


def train_zero_start(
    view1: np.ndarray,
    view2: np.ndarray,
    retained: np.ndarray,
    smooth: np.ndarray,
    trust_gate: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    is_spatial: np.ndarray,
    posterior: RelationPosterior,
    config: Mapping[str, object],
    seed: int,
    view_mask: tuple[float, float] = (1.0, 1.0),
    device: str = "cpu",
) -> ZeroStartTrainResult:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.manual_seed_all(int(seed))
    v1 = torch.as_tensor(standardize(view1), device=device)
    v2 = torch.as_tensor(standardize(view2), device=device)
    base = torch.as_tensor(row_normalize(standardize(retained)), device=device)
    smooth_tensor = torch.as_tensor(np.asarray(smooth, dtype=np.float32), device=device)
    gate = torch.as_tensor(np.asarray(trust_gate, dtype=np.float32), device=device)
    ii = torch.as_tensor(pair_i, dtype=torch.long, device=device)
    jj = torch.as_tensor(pair_j, dtype=torch.long, device=device)
    target = torch.as_tensor(posterior.probability_same, device=device)
    confidence = torch.as_tensor(1.0 - posterior.uncertainty, device=device)
    edge_scale = torch.as_tensor(np.where(is_spatial, 1.0, 0.5).astype(np.float32), device=device)
    relation_weight = confidence * edge_scale
    model = ZeroStartRelationEncoder(
        view1.shape[1],
        view2.shape[1],
        retained.shape[1],
        int(config["hidden_dim"]),
        float(config["residual_scale"]),
    ).to(device)
    model.eval()
    with torch.no_grad():
        step0 = model(v1, v2, base, smooth_tensor, gate, view_mask=view_mask).cpu().numpy().astype(np.float32)
    if not np.array_equal(step0, np.asarray(smooth, dtype=np.float32)):
        raise RuntimeError("zero-start representation is not byte-exact to smooth carrier")
    initial_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    initial_sha = _state_sha(initial_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    max_gradient = 0.0
    last = {}
    model.train()
    for step in range(int(config["steps"])):
        optimizer.zero_grad(set_to_none=True)
        representation = model(v1, v2, base, smooth_tensor, gate, view_mask=view_mask)
        normalized = torch.nn.functional.normalize(representation, dim=1)
        cosine = torch.sum(normalized[ii] * normalized[jj], dim=1)
        predicted_same = torch.clamp((cosine + 1.0) / 2.0, 1e-5, 1.0 - 1e-5)
        relation_loss, positive_relation_loss, negative_relation_loss, positive_mass, negative_mass = (
            balanced_soft_relation_loss(predicted_same, target, relation_weight)
        )
        low_trust = 1.0 + float(config["self_return_weight"]) * (1.0 - gate)
        anchor_loss = torch.mean(low_trust[:, None] * (representation - smooth_tensor) ** 2)
        masked_view = (1.0, 0.0) if step % 2 == 0 else (0.0, 1.0)
        masked = model(v1, v2, base, smooth_tensor, gate, view_mask=masked_view)
        consistency_loss = torch.mean(gate[:, None] * (masked - representation.detach()) ** 2)
        std = torch.sqrt(representation.var(dim=0, unbiased=False) + 1e-5)
        variance_loss = torch.mean(torch.relu(0.08 - std) ** 2)
        loss = (
            float(config["relation_weight"]) * relation_loss
            + float(config["anchor_weight"]) * anchor_loss
            + float(config["consistency_weight"]) * consistency_loss
            + float(config["variance_weight"]) * variance_loss
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite zero-start loss")
        loss.backward()
        gradient = float(
            np.sqrt(
                sum(
                    float(torch.sum(parameter.grad.detach() ** 2).cpu())
                    for parameter in model.parameters()
                    if parameter.grad is not None
                )
            )
        )
        max_gradient = max(max_gradient, gradient)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        last = {
            "loss": float(loss.detach().cpu()),
            "relation_loss": float(relation_loss.detach().cpu()),
            "positive_relation_loss": float(positive_relation_loss.detach().cpu()),
            "negative_relation_loss": float(negative_relation_loss.detach().cpu()),
            "effective_positive_relation_mass": float(positive_mass.detach().cpu()),
            "effective_negative_relation_mass": float(negative_mass.detach().cpu()),
            "anchor_loss": float(anchor_loss.detach().cpu()),
            "consistency_loss": float(consistency_loss.detach().cpu()),
            "variance_loss": float(variance_loss.detach().cpu()),
        }
    model.eval()
    with torch.no_grad():
        final = model(v1, v2, base, smooth_tensor, gate, view_mask=view_mask).cpu().numpy().astype(np.float32)
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    parameter_change = float(
        np.sqrt(sum(float(torch.sum((state[key] - initial_state[key]) ** 2)) for key in state))
    )
    diagnostics = {
        **last,
        "actual_optimizer_steps": int(config["steps"]),
        "max_gradient_norm": max_gradient,
        "parameter_l2_change": parameter_change,
        "initial_parameter_sha256": initial_sha,
        "final_parameter_sha256": _state_sha(state),
        "step0_representation_sha256": sha256_array(step0),
        "smooth_representation_sha256": sha256_array(np.asarray(smooth, dtype=np.float32)),
        "step0_exact_smooth": True,
        "representation_sha256": sha256_array(final),
        "max_absolute_residual": float(np.max(np.abs(final - smooth))),
        "zero_gate_max_absolute_residual": float(
            np.max(np.abs(final[trust_gate == 0] - smooth[trust_gate == 0])) if np.any(trust_gate == 0) else 0.0
        ),
    }
    return ZeroStartTrainResult(final, state, diagnostics)


def reload_zero_start(
    state_dict: Mapping[str, torch.Tensor],
    view1: np.ndarray,
    view2: np.ndarray,
    retained: np.ndarray,
    smooth: np.ndarray,
    trust_gate: np.ndarray,
    config: Mapping[str, object],
    view_mask: tuple[float, float] = (1.0, 1.0),
    device: str = "cpu",
) -> np.ndarray:
    model = ZeroStartRelationEncoder(
        view1.shape[1], view2.shape[1], retained.shape[1], int(config["hidden_dim"]), float(config["residual_scale"])
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    with torch.no_grad():
        result = model(
            torch.as_tensor(standardize(view1), device=device),
            torch.as_tensor(standardize(view2), device=device),
            torch.as_tensor(row_normalize(standardize(retained)), device=device),
            torch.as_tensor(np.asarray(smooth, dtype=np.float32), device=device),
            torch.as_tensor(np.asarray(trust_gate, dtype=np.float32), device=device),
            view_mask=view_mask,
        )
    return result.cpu().numpy().astype(np.float32)
