"""Night-17G trainable cross-modal signed-boundary objective (CSBO).

This clean-room P0 keeps a mature residual multimodal autoencoder/DEC scaffold
separate from the tested object: continuous attraction, consensus-boundary
repulsion, and cross-modal conflict abstention on registered sparse edges.
Annotations are not accepted by this module.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.cluster import KMeans
from scipy.stats import rankdata


def sha256_array(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def standardize(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    return ((value - value.mean(0, keepdims=True)) / np.maximum(value.std(0, keepdims=True), 1e-5)).astype(np.float32)


def row_normalize(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    return (value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-6)).astype(np.float32)


def encode_partition(value: np.ndarray) -> np.ndarray:
    _, encoded = np.unique(np.asarray(value), return_inverse=True)
    return encoded.astype(np.int32)


def _ordinal_rank(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    # Midranks prevent equal molecular similarities from acquiring artificial
    # evidence merely because sparse edges arrived in a different order.
    return (rankdata(value, method="average") - 1.0) / max(value.size - 1, 1)


def _stable_edge_hash(left: str, right: str, salt: str) -> int:
    a, b = sorted((str(left), str(right)))
    return int.from_bytes(hashlib.sha256(f"{salt}\0{a}\0{b}".encode()).digest()[:8], "little")


@dataclass(frozen=True)
class EdgeStates:
    rows: np.ndarray
    cols: np.ndarray
    base_weight: np.ndarray
    attraction: np.ndarray
    boundary: np.ndarray
    conflict: np.ndarray
    state_sha256: str
    diagnostics: Mapping[str, float]


def build_edge_states(
    view1: np.ndarray,
    view2: np.ndarray,
    graph: sp.csr_matrix,
    ids: np.ndarray,
) -> EdgeStates:
    """Build three continuous states from dual-view local rank evidence."""

    graph = sp.csr_matrix(graph, dtype=np.float64).maximum(sp.csr_matrix(graph, dtype=np.float64).T)
    graph.setdiag(0)
    graph.eliminate_zeros()
    upper = sp.triu(graph, k=1, format="coo")
    if upper.nnz == 0 or np.any(upper.row == upper.col):
        raise ValueError("registered graph has no valid non-self sparse edge")
    rows, cols = upper.row.astype(np.int64), upper.col.astype(np.int64)
    order = np.lexsort((np.asarray(ids)[cols].astype("U"), np.asarray(ids)[rows].astype("U")))
    rows, cols = rows[order], cols[order]
    base = np.asarray(upper.data[order], dtype=np.float64)
    if not np.all(np.isfinite(base)) or np.any(base <= 0):
        raise ValueError("base edge weights must be positive and finite")
    x1, x2 = row_normalize(standardize(view1)), row_normalize(standardize(view2))
    sim1 = np.sum(x1[rows] * x1[cols], axis=1)
    sim2 = np.sum(x2[rows] * x2[cols], axis=1)
    rank1, rank2 = _ordinal_rank(sim1), _ordinal_rank(sim2)
    # Product decomposition gives an exact, threshold-free three-state simplex:
    # both views similar -> attraction; both dissimilar -> boundary; discordant -> conflict.
    attraction = (rank1 * rank2).astype(np.float32)
    boundary = ((1.0 - rank1) * (1.0 - rank2)).astype(np.float32)
    conflict = (rank1 * (1.0 - rank2) + (1.0 - rank1) * rank2).astype(np.float32)
    closure = attraction.astype(np.float64) + boundary.astype(np.float64) + conflict.astype(np.float64)
    if not np.allclose(closure, 1.0, rtol=0.0, atol=2e-7):
        raise RuntimeError("three-state edge simplex failed numerical closure")
    if min(float(attraction.sum()), float(boundary.sum()), float(conflict.sum())) <= 0:
        raise RuntimeError("a three-state edge mass is degenerate")
    payload = np.column_stack([rows, cols, attraction, boundary, conflict])
    return EdgeStates(
        rows=rows,
        cols=cols,
        base_weight=base.astype(np.float32),
        attraction=attraction,
        boundary=boundary,
        conflict=conflict,
        state_sha256=sha256_array(payload),
        diagnostics={
            "edge_count": int(len(rows)),
            "attraction_mean": float(attraction.mean()),
            "boundary_mean": float(boundary.mean()),
            "conflict_mean": float(conflict.mean()),
            "simplex_max_absolute_error": float(np.max(np.abs(closure - 1.0))),
            "view_rank_spearman": float(np.corrcoef(rank1, rank2)[0, 1]),
        },
    )


def permute_edge_states(states: EdgeStates, ids: np.ndarray, seed: int = 20260826) -> EdgeStates:
    """ID-stable permutation inside base-weight quartiles, preserving state mass."""

    quantiles = np.quantile(states.base_weight, [0.25, 0.5, 0.75])
    strata = np.digitize(states.base_weight, quantiles, right=True)
    permutation = np.arange(len(states.rows), dtype=np.int64)
    for group in range(4):
        index = np.flatnonzero(strata == group)
        if index.size < 2:
            continue
        destination = index[np.argsort([
            _stable_edge_hash(ids[states.rows[i]], ids[states.cols[i]], f"dst-{seed}-{group}") for i in index
        ], kind="mergesort")]
        source = index[np.argsort([
            _stable_edge_hash(ids[states.rows[i]], ids[states.cols[i]], f"src-{seed}-{group}") for i in index
        ], kind="mergesort")]
        permutation[destination] = source
    if np.array_equal(permutation, np.arange(len(permutation))):
        raise RuntimeError("edge-state permutation is identity")
    channels = []
    mass_errors = {}
    for name in ("attraction", "boundary", "conflict"):
        original = np.asarray(getattr(states, name), dtype=np.float64)
        changed = original[permutation].copy()
        target_mass = float(np.sum(states.base_weight.astype(np.float64) * original))
        current_mass = float(np.sum(states.base_weight.astype(np.float64) * changed))
        if not np.isfinite(target_mass) or not np.isfinite(current_mass) or min(target_mass, current_mass) <= 0:
            raise RuntimeError(f"invalid base-weighted {name} mass")
        changed *= target_mass / current_mass
        final_mass = float(np.sum(states.base_weight.astype(np.float64) * changed))
        absolute_error = abs(final_mass - target_mass)
        relative_error = absolute_error / max(abs(target_mass), 1e-12)
        if absolute_error > 1e-8 and relative_error > 1e-10:
            raise RuntimeError(f"permuted {name} base-weighted mass mismatch")
        channels.append(changed.astype(np.float32))
        mass_errors[f"{name}_base_weighted_mass_absolute_error"] = absolute_error
        mass_errors[f"{name}_base_weighted_mass_relative_error"] = relative_error
    payload = np.column_stack([states.rows, states.cols, *channels])
    return EdgeStates(
        rows=states.rows.copy(), cols=states.cols.copy(), base_weight=states.base_weight.copy(),
        attraction=channels[0], boundary=channels[1], conflict=channels[2], state_sha256=sha256_array(payload),
        diagnostics={**states.diagnostics, "permutation_changed_fraction": float(np.mean(permutation != np.arange(len(permutation)))),
                     "permuted_control_probability_simplex": False, **mass_errors},
    )


@dataclass(frozen=True)
class CSBOConfig:
    config_id: str
    hidden_dim: int
    residual_scale: float
    learning_rate: float
    steps: int
    reconstruction_weight: float
    anchor_weight: float
    prototype_weight: float
    edge_weight: float
    repulsion_margin: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class SignedBoundaryCore(torch.nn.Module):
    def __init__(self, d1: int, d2: int, d: int, hidden: int, k: int, scale: float, prototypes: np.ndarray):
        super().__init__()
        self.a1 = torch.nn.Linear(d1, hidden)
        self.a2 = torch.nn.Linear(d2, hidden)
        self.ar = torch.nn.Linear(d, hidden)
        self.fuse = torch.nn.Sequential(torch.nn.Linear(hidden * 4, hidden), torch.nn.GELU())
        self.residual = torch.nn.Linear(hidden, d)
        torch.nn.init.zeros_(self.residual.weight)
        torch.nn.init.zeros_(self.residual.bias)
        self.decoder1 = torch.nn.Linear(d, d1)
        self.decoder2 = torch.nn.Linear(d, d2)
        self.prototypes = torch.nn.Parameter(torch.as_tensor(prototypes, dtype=torch.float32))
        self.scale = float(scale)
        self.k = int(k)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, base: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h1, h2, hr = torch.tanh(self.a1(x1)), torch.tanh(self.a2(x2)), torch.tanh(self.ar(base))
        interaction = h1 * h2
        delta = torch.tanh(self.residual(self.fuse(torch.cat([h1, h2, hr, interaction], 1))))
        z = base + self.scale * delta
        return z, self.decoder1(z), self.decoder2(z)


def endpoint_partition(representation: np.ndarray, initial: np.ndarray, k: int) -> np.ndarray:
    initial = encode_partition(initial)
    if np.unique(initial).size != k:
        raise ValueError("initial partition K mismatch")
    centroids = np.stack([representation[initial == group].mean(0) for group in range(k)])
    result = KMeans(n_clusters=k, init=centroids, n_init=1, random_state=0, max_iter=300).fit_predict(representation)
    if np.unique(result).size != k:
        raise RuntimeError("endpoint violated exact K")
    return result.astype(np.int32)


def _edge_weights(states: EdgeStates, arm: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if arm == "BACKBONE_NO_CSBO":
        return np.zeros_like(states.attraction), np.zeros_like(states.boundary), np.zeros_like(states.conflict)
    if arm == "FULL_CSBO":
        return states.attraction, states.boundary, states.conflict
    if arm == "UNSIGNED_ONLY":
        return np.ones_like(states.attraction), np.zeros_like(states.boundary), states.conflict
    if arm == "BOUNDARY_TO_ABSTAIN":
        return states.attraction, np.zeros_like(states.boundary), states.conflict
    if arm == "CONFLICT_AS_POSITIVE":
        return np.clip(states.attraction + states.conflict, 0, 1), states.boundary, np.zeros_like(states.conflict)
    raise ValueError(f"unknown arm {arm}")


def _state_sha(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode())
        digest.update(state[key].detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def train_core(
    view1: np.ndarray, view2: np.ndarray, retained: np.ndarray, initial: np.ndarray,
    states: EdgeStates, config: CSBOConfig, arm: str, seed: int, device: str,
) -> tuple[np.ndarray, np.ndarray, Mapping[str, torch.Tensor], dict[str, object]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    x1n, x2n = standardize(view1), standardize(view2)
    base_np = row_normalize(standardize(retained))
    initial = encode_partition(initial)
    prototypes = np.stack([base_np[initial == group].mean(0) for group in range(int(initial.max()) + 1)])
    model = SignedBoundaryCore(x1n.shape[1], x2n.shape[1], base_np.shape[1], config.hidden_dim,
                               prototypes.shape[0], config.residual_scale, prototypes).to(device)
    x1 = torch.as_tensor(x1n, device=device)
    x2 = torch.as_tensor(x2n, device=device)
    base = torch.as_tensor(base_np, device=device)
    rows = torch.as_tensor(states.rows, dtype=torch.long, device=device)
    cols = torch.as_tensor(states.cols, dtype=torch.long, device=device)
    base_w = torch.as_tensor(states.base_weight / np.maximum(np.mean(states.base_weight), 1e-8), device=device)
    # The producer constructs the ID-stable permuted state once and passes it here.
    effective_arm = "FULL_CSBO" if arm == "PERMUTED_EDGE_STATES" else arm
    attr_np, bound_np, conflict_np = _edge_weights(states, effective_arm)
    attr = torch.as_tensor(attr_np, device=device) * base_w
    bound = torch.as_tensor(bound_np, device=device) * base_w
    conflict = torch.as_tensor(conflict_np, device=device) * base_w
    initial_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    last: dict[str, float] = {}
    max_gradient = 0.0
    for _ in range(config.steps):
        optimizer.zero_grad(set_to_none=True)
        z, r1, r2 = model(x1, x2, base)
        reconstruction = 0.5 * (torch.mean((r1 - x1) ** 2) + torch.mean((r2 - x2) ** 2))
        anchor = torch.mean((z - base) ** 2)
        distance = torch.sum((z[:, None, :] - model.prototypes[None, :, :]) ** 2, dim=2)
        q = 1.0 / (1.0 + distance)
        q = q / torch.sum(q, dim=1, keepdim=True)
        p = q.detach() ** 2 / torch.clamp(torch.sum(q.detach(), dim=0, keepdim=True), min=1e-8)
        p = p / torch.sum(p, dim=1, keepdim=True)
        prototype = torch.mean(torch.sum(p * (torch.log(torch.clamp(p, min=1e-8)) - torch.log(torch.clamp(q, min=1e-8))), dim=1))
        zn = torch.nn.functional.normalize(z, dim=1)
        cosine = torch.sum(zn[rows] * zn[cols], dim=1)
        attraction = torch.sum(attr * (1.0 - cosine)) / torch.clamp(torch.sum(attr), min=1e-8)
        repulsion = torch.sum(bound * torch.relu(cosine - config.repulsion_margin) ** 2) / torch.clamp(torch.sum(bound), min=1e-8)
        edge_loss = attraction + repulsion if effective_arm != "BACKBONE_NO_CSBO" else torch.zeros((), device=device)
        loss = (config.reconstruction_weight * reconstruction + config.anchor_weight * anchor
                + config.prototype_weight * prototype + config.edge_weight * edge_loss)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite training loss")
        loss.backward()
        gradient = float(np.sqrt(sum(float(torch.sum(p.grad.detach() ** 2).cpu()) for p in model.parameters() if p.grad is not None)))
        max_gradient = max(max_gradient, gradient)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        last = {"loss": float(loss.detach().cpu()), "reconstruction_loss": float(reconstruction.detach().cpu()),
                "anchor_loss": float(anchor.detach().cpu()), "prototype_loss": float(prototype.detach().cpu()),
                "attraction_loss": float(attraction.detach().cpu()), "repulsion_loss": float(repulsion.detach().cpu())}
    model.eval()
    with torch.no_grad():
        representation = model(x1, x2, base)[0].cpu().numpy().astype(np.float32)
    partition = endpoint_partition(representation, initial, prototypes.shape[0])
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    parameter_change = float(np.sqrt(sum(float(torch.sum((state[k] - initial_state[k]) ** 2)) for k in state)))
    if parameter_change <= 0 or max_gradient <= 0:
        raise RuntimeError("trainable core did not update")
    diagnostics = {**last, "actual_optimizer_steps": config.steps, "max_gradient_norm": max_gradient,
                   "parameter_change_l2": parameter_change, "initial_state_sha256": _state_sha(initial_state),
                   "final_state_sha256": _state_sha(state), "representation_sha256": sha256_array(representation),
                   "partition_sha256": sha256_array(partition), "attraction_mass": float(np.sum(attr_np * states.base_weight)),
                   "boundary_mass": float(np.sum(bound_np * states.base_weight)), "conflict_mass": float(np.sum(conflict_np * states.base_weight))}
    return representation, partition, state, diagnostics


def reload_core(
    state: Mapping[str, torch.Tensor], view1: np.ndarray, view2: np.ndarray, retained: np.ndarray,
    initial: np.ndarray, config: CSBOConfig, device: str,
) -> np.ndarray:
    x1, x2, base = standardize(view1), standardize(view2), row_normalize(standardize(retained))
    initial = encode_partition(initial)
    prototypes = np.stack([base[initial == group].mean(0) for group in range(int(initial.max()) + 1)])
    model = SignedBoundaryCore(x1.shape[1], x2.shape[1], base.shape[1], config.hidden_dim,
                               prototypes.shape[0], config.residual_scale, prototypes).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(x1, device=device), torch.as_tensor(x2, device=device),
                     torch.as_tensor(base, device=device))[0].cpu().numpy().astype(np.float32)
