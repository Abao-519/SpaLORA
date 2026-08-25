"""Structure-feasible relation distillation (SFRD) P0 primitives.

The module consumes only numeric modality carriers, registered sparse graphs,
and locked candidate partitions plus their pre-evaluation evidence.  Public
annotations are deliberately absent from this producer-side implementation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import scipy.sparse as sp
import torch
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


def sha256_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def encode_partition(values: np.ndarray) -> np.ndarray:
    _, encoded = np.unique(np.asarray(values), return_inverse=True)
    return encoded.astype(np.int32)


def training_seed_from_run_ids(run_id: str, all_run_ids: Sequence[str]) -> int:
    """Recover the producer seed without hard-coding evaluator metadata."""
    candidates = [str(run_id)] + [str(value) for value in all_run_ids]
    for value in candidates:
        if "__S" not in value:
            continue
        suffix = value.rsplit("__S", 1)[1]
        if suffix.isdigit():
            return int(suffix)
    raise ValueError("no training seed encoded in locked run IDs")


def exceeds_all_matched_controls(
    full_ari: float,
    full_nmi: float,
    control_metrics: Sequence[tuple[float, float]],
    tolerance: float = 1e-12,
) -> tuple[bool, float, float]:
    """Strict scientific gate against the coordinate-wise strongest control."""
    if not control_metrics:
        raise ValueError("at least one matched control is required")
    strongest_ari = max(float(value[0]) for value in control_metrics)
    strongest_nmi = max(float(value[1]) for value in control_metrics)
    passed = float(full_ari) > strongest_ari + tolerance and float(full_nmi) > strongest_nmi + tolerance
    return passed, strongest_ari, strongest_nmi


def standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    return ((values - mean) / np.maximum(std, 1e-6)).astype(np.float32)


def row_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    return (values / np.maximum(norm, 1e-8)).astype(np.float32)


def csr_from_carrier(carrier: Mapping[str, np.ndarray], prefix: str = "graph0") -> sp.csr_matrix:
    return sp.csr_matrix(
        (
            np.asarray(carrier[f"{prefix}__data"]),
            np.asarray(carrier[f"{prefix}__indices"], dtype=np.int32),
            np.asarray(carrier[f"{prefix}__indptr"], dtype=np.int32),
        ),
        shape=tuple(int(value) for value in carrier[f"{prefix}__shape"]),
    )


def canonical_pair_bank(
    graph: sp.csr_matrix,
    retained: np.ndarray,
    feature_neighbors: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Union registered spatial edges with bounded retained-feature kNN pairs."""

    graph = sp.csr_matrix(graph).maximum(sp.csr_matrix(graph).T).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    upper = sp.triu(graph, k=1).tocoo()
    spatial_pairs = np.column_stack([upper.row, upper.col]).astype(np.int64)
    n = retained.shape[0]
    k = min(max(1, int(feature_neighbors)), max(1, n - 1))
    neighbors = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=1)
    index = neighbors.fit(row_normalize(standardize(retained))).kneighbors(return_distance=False)
    source = np.repeat(np.arange(n, dtype=np.int64), k)
    target = index[:, 1 : k + 1].reshape(-1).astype(np.int64)
    lo = np.minimum(source, target)
    hi = np.maximum(source, target)
    feature_pairs = np.column_stack([lo, hi])
    feature_pairs = feature_pairs[feature_pairs[:, 0] != feature_pairs[:, 1]]
    all_pairs = np.vstack([spatial_pairs, feature_pairs])
    order = np.lexsort((all_pairs[:, 1], all_pairs[:, 0]))
    all_pairs = all_pairs[order]
    unique = np.ones(len(all_pairs), dtype=bool)
    unique[1:] = np.any(all_pairs[1:] != all_pairs[:-1], axis=1)
    pairs = all_pairs[unique]
    spatial_codes = set((int(i) << 32) | int(j) for i, j in spatial_pairs.tolist())
    is_spatial = np.asarray(
        [((int(i) << 32) | int(j)) in spatial_codes for i, j in pairs], dtype=bool
    )
    return pairs[:, 0].astype(np.int64), pairs[:, 1].astype(np.int64), is_spatial


def _percentile(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 1:
        return np.ones(1, dtype=np.float64)
    return (rankdata(values, method="average") - 1.0) / (values.size - 1.0)


def candidate_mask(records: Sequence[Mapping[str, object]], bank_mode: str) -> np.ndarray:
    feasible = np.asarray(
        [str(row["feasible_SMALLEST_SCALE_INTERNAL_EDGE"]).lower() == "true" for row in records],
        dtype=bool,
    )
    if bank_mode == "FULL_BANK":
        return feasible
    if bank_mode != "UNBIASED_BANK":
        raise ValueError(bank_mode)
    unbiased = np.asarray(
        [
            str(row["candidate_id"]).startswith("KMEANS_RETAINED_")
            or str(row["candidate_id"]).startswith("PATH_UNIFORM__KMEANS_RETAINED_")
            for row in records
        ],
        dtype=bool,
    )
    return feasible & unbiased


@dataclass(frozen=True)
class RelationPosterior:
    probability_same: np.ndarray
    uncertainty: np.ndarray
    positive_weight: np.ndarray
    negative_weight: np.ndarray
    candidate_weights: np.ndarray
    selected_candidate_count: int


def relation_posterior(
    partitions: np.ndarray,
    records: Sequence[Mapping[str, object]],
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    bank_mode: str = "FULL_BANK",
    weighted: bool = True,
) -> RelationPosterior:
    mask = candidate_mask(records, bank_mode)
    indices = np.flatnonzero(mask)
    if indices.size < 2:
        raise ValueError(f"{bank_mode} has fewer than two feasible candidates")
    if weighted:
        axes = np.column_stack(
            [
                [float(records[index]["molecular_joint"]) for index in indices],
                [float(records[index]["topology_joint"]) for index in indices],
                [float(records[index]["persistence"]) for index in indices],
            ]
        )
        evidence = np.mean(np.column_stack([_percentile(axes[:, col]) for col in range(3)]), axis=1)
        logits = (evidence - evidence.max()) / 0.35
        weights = np.exp(logits)
        weights /= weights.sum()
    else:
        weights = np.full(indices.size, 1.0 / indices.size, dtype=np.float64)
    same = partitions[indices][:, pair_i] == partitions[indices][:, pair_j]
    probability = np.asarray(weights @ same.astype(np.float64), dtype=np.float64)
    eps = 1e-12
    entropy = -(probability * np.log(probability + eps) + (1.0 - probability) * np.log(1.0 - probability + eps))
    uncertainty = np.clip(entropy / np.log(2.0), 0.0, 1.0)
    confidence = np.abs(2.0 * probability - 1.0) * (1.0 - uncertainty)
    positive = confidence * (probability > 0.5)
    negative = confidence * (probability < 0.5)
    all_weights = np.zeros(len(records), dtype=np.float64)
    all_weights[indices] = weights
    return RelationPosterior(
        probability_same=probability.astype(np.float32),
        uncertainty=uncertainty.astype(np.float32),
        positive_weight=positive.astype(np.float32),
        negative_weight=negative.astype(np.float32),
        candidate_weights=all_weights,
        selected_candidate_count=int(indices.size),
    )


def permute_relation(posterior: RelationPosterior, seed: int = 20260826) -> RelationPosterior:
    rng = np.random.RandomState(seed)
    order = rng.permutation(posterior.probability_same.size)
    probability = posterior.probability_same[order]
    uncertainty = posterior.uncertainty[order]
    positive = posterior.positive_weight[order]
    negative = posterior.negative_weight[order]
    return RelationPosterior(
        probability_same=probability,
        uncertainty=uncertainty,
        positive_weight=positive,
        negative_weight=negative,
        candidate_weights=posterior.candidate_weights.copy(),
        selected_candidate_count=posterior.selected_candidate_count,
    )


class SFRDEncoder(torch.nn.Module):
    """Small residual encoder with modality adapters and an identity anchor."""

    def __init__(self, view1_dim: int, view2_dim: int, retained_dim: int, hidden_dim: int, residual_scale: float):
        super().__init__()
        self.view1_adapter = torch.nn.Linear(view1_dim, hidden_dim)
        self.view2_adapter = torch.nn.Linear(view2_dim, hidden_dim)
        self.retained_adapter = torch.nn.Linear(retained_dim, hidden_dim)
        self.fuse = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 3, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, retained_dim),
        )
        self.residual_scale = float(residual_scale)

    def forward(
        self,
        view1: torch.Tensor,
        view2: torch.Tensor,
        retained: torch.Tensor,
        view_mask: tuple[float, float] = (1.0, 1.0),
    ) -> torch.Tensor:
        h1 = torch.tanh(self.view1_adapter(view1)) * float(view_mask[0])
        h2 = torch.tanh(self.view2_adapter(view2)) * float(view_mask[1])
        hr = torch.tanh(self.retained_adapter(retained))
        residual = torch.tanh(self.fuse(torch.cat([h1, h2, hr], dim=1)))
        return torch.nn.functional.normalize(retained + self.residual_scale * residual, dim=1)


@dataclass(frozen=True)
class TrainResult:
    representation: np.ndarray
    state_dict: Mapping[str, torch.Tensor]
    diagnostics: Mapping[str, object]


def _state_sha(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode("utf-8"))
        digest.update(state[key].detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def train_residual(
    view1: np.ndarray,
    view2: np.ndarray,
    retained: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    is_spatial: np.ndarray,
    posterior: RelationPosterior,
    config: Mapping[str, object],
    seed: int,
    view_mask: tuple[float, float] = (1.0, 1.0),
    device: str = "cpu",
) -> TrainResult:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.manual_seed_all(int(seed))
    view1_np = standardize(view1)
    view2_np = standardize(view2)
    retained_np = row_normalize(standardize(retained))
    v1 = torch.as_tensor(view1_np, device=device)
    v2 = torch.as_tensor(view2_np, device=device)
    base = torch.as_tensor(retained_np, device=device)
    ii = torch.as_tensor(pair_i, dtype=torch.long, device=device)
    jj = torch.as_tensor(pair_j, dtype=torch.long, device=device)
    edge_scale = torch.as_tensor(np.where(is_spatial, 1.0, 0.5).astype(np.float32), device=device)
    pos = torch.as_tensor(posterior.positive_weight, device=device) * edge_scale
    neg = torch.as_tensor(posterior.negative_weight, device=device) * edge_scale
    model = SFRDEncoder(
        view1.shape[1], view2.shape[1], retained.shape[1], int(config["hidden_dim"]), float(config["residual_scale"])
    ).to(device)
    initial_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    initial_sha = _state_sha(initial_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    max_gradient = 0.0
    last = {}
    model.train()
    for step in range(int(config["steps"])):
        optimizer.zero_grad(set_to_none=True)
        embedding = model(v1, v2, base, view_mask=view_mask)
        squared = torch.sum((embedding[ii] - embedding[jj]) ** 2, dim=1)
        distance = torch.sqrt(squared + 1e-8)
        positive_loss = torch.sum(pos * squared) / torch.clamp(pos.sum(), min=1e-8)
        negative_loss = torch.sum(neg * torch.relu(float(config["margin"]) - distance) ** 2) / torch.clamp(neg.sum(), min=1e-8)
        identity_loss = torch.mean((embedding - base) ** 2)
        std = torch.sqrt(embedding.var(dim=0, unbiased=False) + 1e-5)
        variance_loss = torch.mean(torch.relu(0.08 - std) ** 2)
        masked_view = (1.0, 0.0) if step % 2 == 0 else (0.0, 1.0)
        masked = model(v1, v2, base, view_mask=masked_view)
        consistency_loss = torch.mean((masked - embedding.detach()) ** 2)
        loss = (
            float(config["relation_weight"]) * (positive_loss + negative_loss)
            + float(config["anchor_weight"]) * identity_loss
            + float(config["consistency_weight"]) * consistency_loss
            + float(config["variance_weight"]) * variance_loss
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite SFRD loss")
        loss.backward()
        gradient = float(
            np.sqrt(sum(float(torch.sum(parameter.grad.detach() ** 2).cpu()) for parameter in model.parameters() if parameter.grad is not None))
        )
        max_gradient = max(max_gradient, gradient)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        last = {
            "loss": float(loss.detach().cpu()),
            "positive_loss": float(positive_loss.detach().cpu()),
            "negative_loss": float(negative_loss.detach().cpu()),
            "identity_loss": float(identity_loss.detach().cpu()),
            "consistency_loss": float(consistency_loss.detach().cpu()),
            "variance_loss": float(variance_loss.detach().cpu()),
        }
    model.eval()
    with torch.no_grad():
        result = model(v1, v2, base, view_mask=view_mask).cpu().numpy().astype(np.float32)
    final_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    final_sha = _state_sha(final_state)
    parameter_delta = float(
        np.sqrt(sum(float(torch.sum((final_state[key] - initial_state[key]) ** 2)) for key in final_state))
    )
    diagnostics = {
        **last,
        "actual_optimizer_steps": int(config["steps"]),
        "max_gradient_norm": max_gradient,
        "parameter_l2_change": parameter_delta,
        "initial_parameter_sha256": initial_sha,
        "final_parameter_sha256": final_sha,
        "representation_sha256": sha256_array(result),
    }
    return TrainResult(result, final_state, diagnostics)


def deterministic_relation_smooth(
    retained: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    posterior: RelationPosterior,
    alpha: float = 0.2,
) -> np.ndarray:
    base = row_normalize(standardize(retained))
    n = base.shape[0]
    weight = posterior.positive_weight.astype(np.float64)
    graph = sp.csr_matrix(
        (np.concatenate([weight, weight]), (np.concatenate([pair_i, pair_j]), np.concatenate([pair_j, pair_i]))),
        shape=(n, n),
    )
    degree = np.asarray(graph.sum(axis=1)).reshape(-1)
    aggregate = graph @ base
    aggregate = aggregate / np.maximum(degree[:, None], 1e-8)
    unchanged = degree <= 0
    aggregate[unchanged] = base[unchanged]
    return row_normalize((1.0 - alpha) * base + alpha * aggregate)


def same_head_partition(representation: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    model = KMeans(n_clusters=int(k), random_state=int(seed), n_init=20, algorithm="lloyd")
    return encode_partition(model.fit_predict(np.asarray(representation, dtype=np.float64)))


def reload_representation(
    state_dict: Mapping[str, torch.Tensor],
    view1: np.ndarray,
    view2: np.ndarray,
    retained: np.ndarray,
    config: Mapping[str, object],
    view_mask: tuple[float, float] = (1.0, 1.0),
    device: str = "cpu",
) -> np.ndarray:
    model = SFRDEncoder(
        view1.shape[1], view2.shape[1], retained.shape[1], int(config["hidden_dim"]), float(config["residual_scale"])
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    with torch.no_grad():
        result = model(
            torch.as_tensor(standardize(view1), device=device),
            torch.as_tensor(standardize(view2), device=device),
            torch.as_tensor(row_normalize(standardize(retained)), device=device),
            view_mask=view_mask,
        )
    return result.cpu().numpy().astype(np.float32)
