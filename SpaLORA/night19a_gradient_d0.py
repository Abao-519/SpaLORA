"""Night-19A D0 primitives for auditable gradient-conflict diagnosis.

The module reuses the frozen Night-17C zero-start residual network and loss
semantics.  It never imports annotations.  Gradients are flattened in one
fixed ``model.named_parameters()`` coordinate system; unused coordinates are
explicit zeros, so pairwise inner products cannot silently compare different
parameter subsets.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import torch
from scipy.stats import rankdata

from SpaLORA.night17b_sfrd import RelationPosterior, row_normalize, sha256_array, standardize
from SpaLORA.night17c_zero_start import ZeroStartRelationEncoder, balanced_soft_relation_loss


LOSS_NAMES = ("relation", "anchor", "consistency", "variance")
CHECKPOINT_STEPS = (0, 1, 5, 20, 40)
GRADIENT_ZERO_EPS = 1e-12


def state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        digest.update(state[name].detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def ordered_trainable_parameters(
    model: torch.nn.Module,
) -> Tuple[Tuple[str, torch.nn.Parameter], ...]:
    ordered = tuple((name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad)
    if not ordered:
        raise ValueError("model has no trainable parameters")
    if len({name for name, _ in ordered}) != len(ordered):
        raise ValueError("duplicate parameter name")
    return ordered


def flatten_loss_gradient(
    loss: torch.Tensor,
    ordered: Sequence[Tuple[str, torch.nn.Parameter]],
    retain_graph: bool = True,
) -> Tuple[torch.Tensor, Tuple[str, ...]]:
    """Flatten one loss gradient in a fixed parameter coordinate system.

    ``None``/unused gradients are replaced by same-shaped zeros.  The returned
    tuple of active names is diagnostic only; it never changes coordinates.
    """

    parameters = tuple(parameter for _, parameter in ordered)
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
        create_graph=False,
    )
    pieces = []
    active = []
    for (name, parameter), gradient in zip(ordered, gradients):
        if gradient is None:
            pieces.append(torch.zeros_like(parameter).reshape(-1))
        else:
            pieces.append(gradient.detach().reshape(-1))
            if bool(torch.any(gradient.detach() != 0)):
                active.append(name)
    return torch.cat(pieces), tuple(active)


def safe_gradient_pair(vector_a: torch.Tensor, vector_b: torch.Tensor) -> Mapping[str, object]:
    norm_a = float(torch.linalg.vector_norm(vector_a).detach().cpu())
    norm_b = float(torch.linalg.vector_norm(vector_b).detach().cpu())
    dot = float(torch.dot(vector_a, vector_b).detach().cpu())
    if norm_a <= GRADIENT_ZERO_EPS or norm_b <= GRADIENT_ZERO_EPS:
        cosine = None
    else:
        cosine = float(dot / (norm_a * norm_b))
    return {
        "dot": dot,
        "cosine": cosine,
        "negative_inner_product": bool(dot < 0.0 and cosine is not None),
        "norm_a": norm_a,
        "norm_b": norm_b,
    }


def evidence_support_strata(
    posterior: RelationPosterior,
    is_spatial: np.ndarray,
) -> Tuple[np.ndarray, Mapping[str, object]]:
    """Mechanical low/mid/high strata from registered relation support.

    Ties receive identical midranks.  Spatial edges retain the same 1.0 versus
    bounded feature-neighbour 0.5 scale used by Night-17C.
    """

    probability = np.asarray(posterior.probability_same, dtype=np.float64)
    uncertainty = np.asarray(posterior.uncertainty, dtype=np.float64)
    if probability.shape != uncertainty.shape or probability.shape != np.asarray(is_spatial).shape:
        raise ValueError("relation evidence arrays have incompatible shapes")
    support = (1.0 - uncertainty) * np.abs(2.0 * probability - 1.0)
    support *= np.where(np.asarray(is_spatial, dtype=bool), 1.0, 0.5)
    if not np.all(np.isfinite(support)):
        raise ValueError("nonfinite evidence support")
    if support.size < 3:
        raise ValueError("at least three relation edges are required")
    percentile = (rankdata(support, method="average") - 1.0) / max(1.0, float(support.size - 1))
    strata = np.where(percentile < (1.0 / 3.0), 0, np.where(percentile < (2.0 / 3.0), 1, 2)).astype(np.int8)
    diagnostics = {
        "support_min": float(support.min()),
        "support_median": float(np.median(support)),
        "support_max": float(support.max()),
        "support_unique_count": int(np.unique(support).size),
        "stratum_counts": [int(np.sum(strata == index)) for index in range(3)],
        "stratum_support_means": [
            float(np.mean(support[strata == index])) if np.any(strata == index) else None for index in range(3)
        ],
        "spatial_fraction": float(np.mean(np.asarray(is_spatial, dtype=bool))),
    }
    return strata, diagnostics


@dataclass(frozen=True)
class PreparedTensors:
    view1: torch.Tensor
    view2: torch.Tensor
    retained: torch.Tensor
    anchor: torch.Tensor
    gate: torch.Tensor
    pair_i: torch.Tensor
    pair_j: torch.Tensor
    target: torch.Tensor
    relation_weight: torch.Tensor
    strata: torch.Tensor


def prepare_tensors(
    view1: np.ndarray,
    view2: np.ndarray,
    retained: np.ndarray,
    anchor: np.ndarray,
    trust_gate: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    is_spatial: np.ndarray,
    posterior: RelationPosterior,
    strata: np.ndarray,
    device: str,
) -> PreparedTensors:
    return PreparedTensors(
        view1=torch.as_tensor(standardize(view1), device=device),
        view2=torch.as_tensor(standardize(view2), device=device),
        retained=torch.as_tensor(row_normalize(standardize(retained)), device=device),
        anchor=torch.as_tensor(np.asarray(anchor, dtype=np.float32), device=device),
        gate=torch.as_tensor(np.asarray(trust_gate, dtype=np.float32), device=device),
        pair_i=torch.as_tensor(np.asarray(pair_i), dtype=torch.long, device=device),
        pair_j=torch.as_tensor(np.asarray(pair_j), dtype=torch.long, device=device),
        target=torch.as_tensor(np.asarray(posterior.probability_same, dtype=np.float32), device=device),
        relation_weight=torch.as_tensor(
            (1.0 - np.asarray(posterior.uncertainty, dtype=np.float32))
            * np.where(np.asarray(is_spatial, dtype=bool), 1.0, 0.5).astype(np.float32),
            device=device,
        ),
        strata=torch.as_tensor(np.asarray(strata, dtype=np.int64), dtype=torch.long, device=device),
    )


def loss_components(
    model: ZeroStartRelationEncoder,
    tensors: PreparedTensors,
    config: Mapping[str, object],
    completed_steps: int,
) -> Tuple[torch.Tensor, Mapping[str, torch.Tensor], Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]]:
    """Return representation, raw losses, weighted losses and relation strata.

    The diagnostic consistency mask is the mask that the next Night-17C update
    would use.  Step 0 therefore has the original first-update mask.
    """

    representation = model(
        tensors.view1,
        tensors.view2,
        tensors.retained,
        tensors.anchor,
        tensors.gate,
        view_mask=(1.0, 1.0),
    )
    normalized = torch.nn.functional.normalize(representation, dim=1)
    cosine = torch.sum(normalized[tensors.pair_i] * normalized[tensors.pair_j], dim=1)
    predicted_same = torch.clamp((cosine + 1.0) / 2.0, 1e-5, 1.0 - 1e-5)
    relation, _, _, positive_mass, negative_mass = balanced_soft_relation_loss(
        predicted_same, tensors.target, tensors.relation_weight
    )
    low_trust = 1.0 + float(config["self_return_weight"]) * (1.0 - tensors.gate)
    anchor = torch.mean(low_trust[:, None] * (representation - tensors.anchor) ** 2)
    view_mask = (1.0, 0.0) if int(completed_steps) % 2 == 0 else (0.0, 1.0)
    masked = model(
        tensors.view1,
        tensors.view2,
        tensors.retained,
        tensors.anchor,
        tensors.gate,
        view_mask=view_mask,
    )
    consistency = torch.mean(tensors.gate[:, None] * (masked - representation.detach()) ** 2)
    std = torch.sqrt(representation.var(dim=0, unbiased=False) + 1e-5)
    variance = torch.mean(torch.relu(0.08 - std) ** 2)
    raw = {
        "relation": relation,
        "anchor": anchor,
        "consistency": consistency,
        "variance": variance,
    }
    weights = {
        "relation": float(config["relation_weight"]),
        "anchor": float(config["anchor_weight"]),
        "consistency": float(config["consistency_weight"]),
        "variance": float(config["variance_weight"]),
    }
    weighted = {name: raw[name] * weights[name] for name in LOSS_NAMES}
    relation_strata: Dict[str, torch.Tensor] = {}
    for index, name in enumerate(("LOW", "MID", "HIGH")):
        mask = (tensors.strata == index).to(predicted_same.dtype)
        positive = -torch.sum(
            mask * tensors.relation_weight * tensors.target * torch.log(predicted_same)
        ) / positive_mass
        negative = -torch.sum(
            mask * tensors.relation_weight * (1.0 - tensors.target) * torch.log(1.0 - predicted_same)
        ) / negative_mass
        relation_strata["relation_" + name.lower()] = float(config["relation_weight"]) * 0.5 * (positive + negative)
    return representation, raw, weighted, relation_strata


def gradient_diagnostic(
    model: ZeroStartRelationEncoder,
    tensors: PreparedTensors,
    config: Mapping[str, object],
    completed_steps: int,
) -> Mapping[str, object]:
    representation, raw, weighted, relation_strata = loss_components(model, tensors, config, completed_steps)
    ordered = ordered_trainable_parameters(model)
    vectors: Dict[str, torch.Tensor] = {}
    active: Dict[str, Tuple[str, ...]] = {}
    for name in list(LOSS_NAMES) + list(relation_strata):
        loss = weighted[name] if name in weighted else relation_strata[name]
        vector, names = flatten_loss_gradient(loss, ordered, retain_graph=True)
        vectors[name] = vector
        active[name] = names
    pairs = {}
    for left_index, left in enumerate(LOSS_NAMES):
        for right in LOSS_NAMES[left_index + 1 :]:
            pairs[left + "__" + right] = dict(safe_gradient_pair(vectors[left], vectors[right]))
    stratum_vs_anchor = {
        name: dict(safe_gradient_pair(vectors[name], vectors["anchor"])) for name in relation_strata
    }
    gradient_norms = {name: float(torch.linalg.vector_norm(vector).detach().cpu()) for name, vector in vectors.items()}
    total = sum(weighted.values())
    return {
        "completed_steps": int(completed_steps),
        "zero_start_boundary": bool(int(completed_steps) == 0),
        "diagnostic_view_mask": [1.0, 0.0] if int(completed_steps) % 2 == 0 else [0.0, 1.0],
        "loss_values": {name: float(raw[name].detach().cpu()) for name in LOSS_NAMES},
        "weighted_loss_values": {name: float(weighted[name].detach().cpu()) for name in LOSS_NAMES},
        "weighted_total_loss": float(total.detach().cpu()),
        "gradient_norms": gradient_norms,
        "active_parameter_names": {name: list(values) for name, values in active.items()},
        "gradient_coordinate_count": int(next(iter(vectors.values())).numel()),
        "pair_metrics": pairs,
        "relation_stratum_vs_anchor": stratum_vs_anchor,
        "relation_stratum_loss_sum_error": float(
            abs(sum(float(value.detach().cpu()) for value in relation_strata.values()) - float(weighted["relation"].detach().cpu()))
        ),
        "representation_sha256": sha256_array(representation.detach().cpu().numpy().astype(np.float32)),
        "max_absolute_residual": float(torch.max(torch.abs(representation - tensors.anchor)).detach().cpu()),
    }


def build_model(
    view1_dim: int,
    view2_dim: int,
    retained_dim: int,
    config: Mapping[str, object],
    seed: int,
    device: str,
) -> ZeroStartRelationEncoder:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(int(seed))
    return ZeroStartRelationEncoder(
        int(view1_dim),
        int(view2_dim),
        int(retained_dim),
        int(config["hidden_dim"]),
        float(config["residual_scale"]),
    ).to(device)


def zero_start_representation(model: ZeroStartRelationEncoder, tensors: PreparedTensors) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        value = model(
            tensors.view1,
            tensors.view2,
            tensors.retained,
            tensors.anchor,
            tensors.gate,
            view_mask=(1.0, 1.0),
        )
    return value.detach().cpu().numpy().astype(np.float32)


def train_standard_sum_d0(
    model: ZeroStartRelationEncoder,
    tensors: PreparedTensors,
    config: Mapping[str, object],
) -> Mapping[str, object]:
    """Run the frozen Night-17C weighted-sum trajectory and collect D0 records."""

    expected_steps = int(config["steps"])
    if expected_steps != CHECKPOINT_STEPS[-1]:
        raise ValueError("D0 checkpoint contract requires exactly 40 updates")
    initial = zero_start_representation(model, tensors)
    anchor = tensors.anchor.detach().cpu().numpy().astype(np.float32)
    if not np.array_equal(initial, anchor):
        raise RuntimeError("step-0 representation is not byte-exact to registered anchor")
    initial_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    trajectory = [gradient_diagnostic(model, tensors, config, 0)]
    model.train()
    for update_index in range(expected_steps):
        optimizer.zero_grad(set_to_none=True)
        _, _, weighted, _ = loss_components(model, tensors, config, update_index)
        total = sum(weighted.values())
        if not torch.isfinite(total):
            raise FloatingPointError("nonfinite D0 objective")
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        completed = update_index + 1
        if completed in CHECKPOINT_STEPS:
            trajectory.append(gradient_diagnostic(model, tensors, config, completed))
    final = zero_start_representation(model, tensors)
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    parameter_change = float(
        np.sqrt(sum(float(torch.sum((state[name] - initial_state[name]) ** 2)) for name in state))
    )
    if parameter_change <= 0:
        raise RuntimeError("D0 optimizer did not change parameters")
    return {
        "initial_representation": initial,
        "final_representation": final,
        "state_dict": state,
        "trajectory": trajectory,
        "initial_state_sha256": state_sha256(initial_state),
        "final_state_sha256": state_sha256(state),
        "parameter_l2_change": parameter_change,
        "final_representation_sha256": sha256_array(final),
    }

