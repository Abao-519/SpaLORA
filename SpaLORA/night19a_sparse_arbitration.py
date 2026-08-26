"""Night-19A Stage-A sparse-evidence gradient arbitration.

The model and losses are the frozen Night-17C zero-start residual scaffold.  The
new operation acts only in gradient space: relation gradients are split by
registered sparse evidence strata and only their components conflicting with
the strong-anchor protection direction are continuously projected.  No labels
or dataset identities enter this module.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import torch

from SpaLORA.night17b_sfrd import sha256_array
from SpaLORA.night19a_gradient_d0 import (
    PreparedTensors,
    build_model,
    flatten_loss_gradient,
    loss_components,
    ordered_trainable_parameters,
    safe_gradient_pair,
    state_sha256,
    zero_start_representation,
)


TRAINED_ARMS = (
    "STANDARD_WEIGHTED_SUM",
    "VANILLA_PCGRAD",
    "GLOBAL_MIN_NORM",
    "EVIDENCE_CONDITIONED_ARBITRATION_FULL",
    "EVIDENCE_PERMUTED_MASS_MATCHED",
    "TOPOLOGY_DISABLED",
    "ARBITRATION_DISABLED_SAME_LOSSES",
)
ALL_ARMS = ("STRONG_START_NO_TRAIN",) + TRAINED_ARMS
EVIDENCE_PROJECTION_STRENGTHS = np.asarray([5.0 / 6.0, 0.5, 1.0 / 6.0], dtype=np.float64)
MASS_MATCH_ATOL = 1e-8
MASS_MATCH_RTOL = 1e-8


def deterministic_stratified_permutation(
    ids: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    is_spatial: np.ndarray,
    strata: np.ndarray,
) -> np.ndarray:
    """Permute stratum locations separately within spatial/feature edge types.

    Ordering is SHA-256 of ordered endpoint IDs and edge type, never Python's
    process-randomized hash.  The operation is a bijection within each type.
    """

    import hashlib

    ids = np.asarray(ids).astype(str)
    pair_i = np.asarray(pair_i, dtype=np.int64)
    pair_j = np.asarray(pair_j, dtype=np.int64)
    is_spatial = np.asarray(is_spatial, dtype=bool)
    strata = np.asarray(strata, dtype=np.int64)
    if not (pair_i.shape == pair_j.shape == is_spatial.shape == strata.shape):
        raise ValueError("edge arrays differ in shape")
    output = strata.copy()
    for flag in (False, True):
        indices = np.flatnonzero(is_spatial == flag)
        if indices.size <= 1:
            continue
        tokens = []
        for index in indices:
            left, right = sorted((ids[pair_i[index]], ids[pair_j[index]]))
            token = f"{int(flag)}|{left}|{right}".encode("utf-8")
            tokens.append(hashlib.sha256(token).digest())
        order = indices[np.argsort(np.asarray(tokens, dtype="S32"), kind="mergesort")]
        shift = max(1, int(order.size // 3))
        output[order] = strata[np.roll(order, shift)]
        if not np.array_equal(np.sort(output[indices]), np.sort(strata[indices])):
            raise RuntimeError("stratified permutation is not a bijection")
    if np.array_equal(output, strata):
        raise RuntimeError("evidence permutation unexpectedly identity")
    return output


def _mass_scale(values: np.ndarray, weights: np.ndarray, target: float) -> Tuple[np.ndarray, float]:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0 or not (0 <= target <= total + MASS_MATCH_ATOL):
        raise ValueError("invalid projection mass target")
    if target <= MASS_MATCH_ATOL:
        return np.zeros_like(values), 0.0
    low, high = 0.0, 1.0
    while float(np.sum(weights * np.clip(high * values, 0.0, 1.0))) < target and high < 1e12:
        high *= 2.0
    for _ in range(100):
        middle = 0.5 * (low + high)
        mass = float(np.sum(weights * np.clip(middle * values, 0.0, 1.0)))
        if mass < target:
            low = middle
        else:
            high = middle
    adjusted = np.clip(high * values, 0.0, 1.0)
    error = float(np.sum(weights * adjusted) - target)
    if abs(error) > MASS_MATCH_ATOL + MASS_MATCH_RTOL * abs(target):
        raise RuntimeError("projection mass matching failed")
    return adjusted, error


def projection_groups_and_strengths(
    ids: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    is_spatial: np.ndarray,
    relation_weight: np.ndarray,
    strata: np.ndarray,
    permuted: bool,
) -> Tuple[np.ndarray, np.ndarray, Mapping[str, object]]:
    """Build six edge-type x evidence-stratum projection groups.

    For the negative control, stratum positions are permuted within edge type.
    Projection-strength mass is then matched exactly *within each edge type* to
    the unpermuted evidence assignment.
    """

    is_spatial = np.asarray(is_spatial, dtype=bool)
    relation_weight = np.asarray(relation_weight, dtype=np.float64)
    strata = np.asarray(strata, dtype=np.int64)
    assigned = deterministic_stratified_permutation(ids, pair_i, pair_j, is_spatial, strata) if permuted else strata.copy()
    original_strength = EVIDENCE_PROJECTION_STRENGTHS[strata]
    assigned_strength = EVIDENCE_PROJECTION_STRENGTHS[assigned]
    errors = {}
    for flag, label in ((False, "feature"), (True, "spatial")):
        mask = is_spatial == flag
        target = float(np.sum(relation_weight[mask] * original_strength[mask]))
        if permuted:
            adjusted, error = _mass_scale(assigned_strength[mask], relation_weight[mask], target)
            assigned_strength[mask] = adjusted
        else:
            error = float(np.sum(relation_weight[mask] * assigned_strength[mask]) - target)
        errors[label] = error
    groups = (is_spatial.astype(np.int64) * 3 + assigned).astype(np.int64)
    group_strengths = np.zeros(6, dtype=np.float64)
    for group in range(6):
        values = assigned_strength[groups == group]
        if values.size:
            if float(values.max() - values.min()) > 1e-12:
                raise RuntimeError("one projection group has nonconstant strength")
            group_strengths[group] = float(values[0])
    diagnostics = {
        "permuted": bool(permuted),
        "permutation_nonidentity": bool(not np.array_equal(assigned, strata)),
        "group_counts": [int(np.sum(groups == index)) for index in range(6)],
        "group_strengths": group_strengths.tolist(),
        "projection_mass_errors_by_edge_type": errors,
        "maximum_absolute_mass_error": float(max(abs(value) for value in errors.values())),
    }
    return groups, group_strengths, diagnostics


def _relation_group_losses(
    representation: torch.Tensor,
    tensors: PreparedTensors,
    config: Mapping[str, object],
    groups: torch.Tensor,
    group_count: int,
) -> Mapping[int, torch.Tensor]:
    normalized = torch.nn.functional.normalize(representation, dim=1)
    cosine = torch.sum(normalized[tensors.pair_i] * normalized[tensors.pair_j], dim=1)
    predicted = torch.clamp((cosine + 1.0) / 2.0, 1e-5, 1.0 - 1e-5)
    positive_mass = torch.sum(tensors.relation_weight * tensors.target).clamp_min(1e-8)
    negative_mass = torch.sum(tensors.relation_weight * (1.0 - tensors.target)).clamp_min(1e-8)
    losses: Dict[int, torch.Tensor] = {}
    for group in range(group_count):
        mask = (groups == group).to(predicted.dtype)
        positive = -torch.sum(mask * tensors.relation_weight * tensors.target * torch.log(predicted)) / positive_mass
        negative = -torch.sum(mask * tensors.relation_weight * (1.0 - tensors.target) * torch.log(1.0 - predicted)) / negative_mass
        losses[group] = float(config["relation_weight"]) * 0.5 * (positive + negative)
    return losses


def _project_simplex(vector: torch.Tensor) -> torch.Tensor:
    sorted_values, _ = torch.sort(vector, descending=True)
    cumulative = torch.cumsum(sorted_values, dim=0) - 1.0
    indices = torch.arange(1, vector.numel() + 1, device=vector.device, dtype=vector.dtype)
    valid = sorted_values - cumulative / indices > 0
    rho = int(torch.nonzero(valid, as_tuple=False)[-1])
    theta = cumulative[rho] / float(rho + 1)
    return torch.clamp(vector - theta, min=0.0)


def global_min_norm_gradient(gradients: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    matrix = torch.stack(tuple(gradients), dim=0)
    norms = torch.linalg.vector_norm(matrix, dim=1)
    active = norms > 1e-12
    weights = torch.zeros((matrix.shape[0],), device=matrix.device, dtype=matrix.dtype)
    if not bool(torch.any(active)):
        return torch.zeros_like(matrix[0]), weights
    active_matrix = matrix[active]
    gram = active_matrix @ active_matrix.T
    active_weights = torch.full(
        (active_matrix.shape[0],), 1.0 / active_matrix.shape[0], device=matrix.device, dtype=matrix.dtype
    )
    lipschitz = float(2.0 * torch.linalg.eigvalsh(gram).max().detach().cpu()) + 1e-12
    for _ in range(128):
        active_weights = _project_simplex(active_weights - (2.0 * (gram @ active_weights)) / lipschitz)
    weights[active] = active_weights
    return torch.sum(weights[:, None] * matrix, dim=0), weights


def vanilla_pcgrad_gradient(gradients: Sequence[torch.Tensor]) -> torch.Tensor:
    originals = tuple(gradient.detach() for gradient in gradients)
    projected = []
    for index, gradient in enumerate(originals):
        value = gradient.clone()
        for offset in range(1, len(originals)):
            other = originals[(index + offset) % len(originals)]
            denominator = torch.dot(other, other)
            dot = torch.dot(value, other)
            if float(dot.detach().cpu()) < 0.0 and float(denominator.detach().cpu()) > 1e-24:
                value = value - dot / denominator * other
        projected.append(value)
    return torch.stack(projected, dim=0).sum(dim=0)


def _assign_flat_gradient(ordered, gradient: torch.Tensor) -> None:
    offset = 0
    for _, parameter in ordered:
        count = parameter.numel()
        value = gradient[offset : offset + count].reshape_as(parameter)
        parameter.grad = value.clone()
        offset += count
    if offset != gradient.numel():
        raise ValueError("flat gradient coordinate count mismatch")


def train_stage_a_arm(
    arm: str,
    tensors: PreparedTensors,
    topology_disabled_tensors: PreparedTensors,
    config: Mapping[str, object],
    seed: int,
    groups: np.ndarray,
    group_strengths: np.ndarray,
    permuted_groups: np.ndarray,
    permuted_group_strengths: np.ndarray,
    device: str,
) -> Mapping[str, object]:
    if arm not in TRAINED_ARMS:
        raise ValueError(arm)
    model = build_model(tensors.view1.shape[1], tensors.view2.shape[1], tensors.retained.shape[1], config, seed, device)
    initial = zero_start_representation(model, tensors)
    anchor = tensors.anchor.detach().cpu().numpy().astype(np.float32)
    if not np.array_equal(initial, anchor):
        raise RuntimeError("Stage-A step0 is not exact registered anchor")
    initial_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    arbitration_events = 0
    warmup_updates = 0
    relation_norm_match_errors = []
    total_updates = int(config["steps"])
    min_norm_weights = []
    pairwise_diagnostics = []
    for step in range(total_updates):
        optimizer.zero_grad(set_to_none=True)
        active_tensors = topology_disabled_tensors if arm == "TOPOLOGY_DISABLED" else tensors
        representation, _, weighted, _ = loss_components(model, active_tensors, config, step)
        ordered = ordered_trainable_parameters(model)
        protection_loss = weighted["anchor"] + weighted["consistency"] + weighted["variance"]
        protection_gradient, _ = flatten_loss_gradient(protection_loss, ordered, retain_graph=True)
        task_gradients = []
        for name in ("relation", "anchor", "consistency", "variance"):
            gradient, _ = flatten_loss_gradient(weighted[name], ordered, retain_graph=True)
            task_gradients.append(gradient)
        if arm in ("STANDARD_WEIGHTED_SUM", "ARBITRATION_DISABLED_SAME_LOSSES"):
            if arm == "STANDARD_WEIGHTED_SUM":
                update = torch.stack(task_gradients, dim=0).sum(dim=0)
            else:
                group_tensor = torch.as_tensor(groups, dtype=torch.long, device=device)
                relation_groups = _relation_group_losses(representation, active_tensors, config, group_tensor, 6)
                group_gradients = [flatten_loss_gradient(loss, ordered, retain_graph=True)[0] for loss in relation_groups.values()]
                update = protection_gradient + torch.stack(group_gradients, dim=0).sum(dim=0)
        elif arm == "VANILLA_PCGRAD":
            update = vanilla_pcgrad_gradient(task_gradients)
        elif arm == "GLOBAL_MIN_NORM":
            update, weights = global_min_norm_gradient(task_gradients)
            min_norm_weights.append(weights.detach().cpu().numpy().tolist())
        else:
            if arm == "EVIDENCE_PERMUTED_MASS_MATCHED":
                selected_groups = permuted_groups
                selected_strengths = permuted_group_strengths
            else:
                selected_groups = groups
                selected_strengths = group_strengths
            group_tensor = torch.as_tensor(selected_groups, dtype=torch.long, device=device)
            relation_groups = _relation_group_losses(representation, active_tensors, config, group_tensor, 6)
            adjusted = []
            original_group_gradients = []
            for group, loss in relation_groups.items():
                gradient, _ = flatten_loss_gradient(loss, ordered, retain_graph=True)
                original_group_gradients.append(gradient)
                pair = safe_gradient_pair(gradient, protection_gradient)
                dot = torch.dot(gradient, protection_gradient)
                denominator = torch.dot(protection_gradient, protection_gradient)
                strength = float(selected_strengths[group])
                if step >= 2 and pair["cosine"] is not None and float(dot.detach().cpu()) < 0 and float(denominator.detach().cpu()) > 1e-24:
                    gradient = gradient - strength * dot / denominator * protection_gradient
                    if strength > 0:
                        arbitration_events += 1
                adjusted.append(gradient)
                if step in (0, 4, 19, 39):
                    pairwise_diagnostics.append({
                        "completed_steps_before_update": step,
                        "group": int(group),
                        "projection_strength": strength,
                        "pre_projection_cosine": pair["cosine"],
                        "pre_projection_norm_ratio": (
                            min(float(pair["norm_a"]), float(pair["norm_b"]))
                            / max(float(pair["norm_a"]), float(pair["norm_b"]), 1e-30)
                        ),
                    })
            original_relation = torch.stack(original_group_gradients, dim=0).sum(dim=0)
            adjusted_relation = torch.stack(adjusted, dim=0).sum(dim=0)
            if step < 2:
                adjusted_relation = original_relation
                warmup_updates += 1
            else:
                original_norm = torch.linalg.vector_norm(original_relation)
                adjusted_norm = torch.linalg.vector_norm(adjusted_relation)
                if float(original_norm.detach().cpu()) > 1e-24 and float(adjusted_norm.detach().cpu()) > 1e-24:
                    adjusted_relation = adjusted_relation * (original_norm / adjusted_norm)
                relation_norm_match_errors.append(
                    float(abs(torch.linalg.vector_norm(adjusted_relation) - original_norm).detach().cpu())
                )
            update = protection_gradient + adjusted_relation
        if not torch.all(torch.isfinite(update)):
            raise FloatingPointError("nonfinite manually arbitrated gradient")
        _assign_flat_gradient(ordered, update)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    final = zero_start_representation(model, tensors)
    final_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    parameter_change = float(np.sqrt(sum(float(torch.sum((final_state[name] - initial_state[name]) ** 2)) for name in final_state)))
    if parameter_change <= 0:
        raise RuntimeError("Stage-A optimizer did not change parameters")
    return {
        "state_dict": final_state,
        "final_representation": final,
        "representation_sha256": sha256_array(final),
        "initial_state_sha256": state_sha256(initial_state),
        "final_state_sha256": state_sha256(final_state),
        "parameter_l2_change": parameter_change,
        "arbitration_projection_events": int(arbitration_events),
        "zero_start_warmup_updates_without_surgery": int(warmup_updates),
        "maximum_relation_gradient_norm_match_error": max(relation_norm_match_errors) if relation_norm_match_errors else 0.0,
        "min_norm_weights_mean": np.mean(min_norm_weights, axis=0).tolist() if min_norm_weights else None,
        "projection_diagnostics": pairwise_diagnostics,
    }


def make_topology_disabled_tensors(tensors: PreparedTensors, is_spatial: np.ndarray) -> PreparedTensors:
    keep = torch.as_tensor(~np.asarray(is_spatial, dtype=bool), device=tensors.relation_weight.device)
    modified = tensors.relation_weight * keep.to(tensors.relation_weight.dtype)
    if float(modified.sum().detach().cpu()) <= 0:
        raise ValueError("topology-disabled relation bank has no feature edges")
    return replace(tensors, relation_weight=modified)
