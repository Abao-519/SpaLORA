"""Night-17A counterfactual edge-utility partitioning (CEUP) P0 core.

The producer is label-free.  It learns small masked cross-modal linear
predictors, estimates exact fixed-denominator leave-one-edge-out utilities,
and uses a signed sparse Potts energy with fixed K.  Public annotations are
only consumed by the separate evaluator.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - evaluator-only local environments
    torch = None
    nn = None


SEED = 20260825


def sha256_array(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
    h.update(arr.tobytes())
    return h.hexdigest()


def encode_partition(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    mapping: Dict[object, int] = {}
    out = np.empty(labels.shape[0], dtype=np.int32)
    for i, value in enumerate(labels.tolist()):
        if value not in mapping:
            mapping[value] = len(mapping)
        out[i] = mapping[value]
    return out


def standardize(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x64 = np.asarray(x, dtype=np.float64)
    mean = x64.mean(axis=0)
    std = x64.std(axis=0)
    std[std < 1e-8] = 1.0
    return ((x64 - mean) / std).astype(np.float32), mean, std


def load_csr(archive: np.lib.npyio.NpzFile, prefix: str) -> sp.csr_matrix:
    shape = tuple(int(x) for x in archive[f"{prefix}__shape"].tolist())
    return sp.csr_matrix(
        (
            np.asarray(archive[f"{prefix}__data"]),
            np.asarray(archive[f"{prefix}__indices"], dtype=np.int32),
            np.asarray(archive[f"{prefix}__indptr"], dtype=np.int32),
        ),
        shape=shape,
    )


def canonical_undirected_graph(graph: sp.csr_matrix) -> Tuple[np.ndarray, np.ndarray, np.ndarray, sp.csr_matrix]:
    graph = sp.csr_matrix(graph, dtype=np.float64)
    graph.setdiag(0.0)
    graph.eliminate_zeros()
    sym = ((graph + graph.T) * 0.5).tocsr()
    sym.data = np.maximum(sym.data, 0.0)
    sym.eliminate_zeros()
    upper = sp.triu(sym, k=1).tocoo()
    order = np.lexsort((upper.col, upper.row))
    edge_i = np.asarray(upper.row[order], dtype=np.int32)
    edge_j = np.asarray(upper.col[order], dtype=np.int32)
    edge_w = np.asarray(upper.data[order], dtype=np.float64)
    if edge_i.size == 0 or np.any(edge_w <= 0):
        raise ValueError("registered graph has no positive undirected edges")
    return edge_i, edge_j, edge_w, sym


def deterministic_feature_masks(n_features: int, n_masks: int, seed: int) -> List[np.ndarray]:
    if n_features < n_masks:
        raise ValueError("feature dimension is smaller than mask count")
    rng = np.random.RandomState(seed)
    order = rng.permutation(n_features)
    return [np.sort(block).astype(np.int64) for block in np.array_split(order, n_masks)]


class MaskedLinearPredictor(nn.Module):
    def __init__(self, target_dim: int, source_dim: int):
        super().__init__()
        self.target_dim = int(target_dim)
        self.source_dim = int(source_dim)
        self.linear = nn.Linear(target_dim + source_dim, target_dim)

    def forward(self, target_masked: "torch.Tensor", neighbor_source: "torch.Tensor") -> "torch.Tensor":
        return self.linear(torch.cat([target_masked, neighbor_source], dim=1))


@dataclass
class PredictorFit:
    state_dict: Mapping[str, "torch.Tensor"]
    masks: List[np.ndarray]
    loss_trace: List[float]
    initial_loss: float
    final_loss: float
    parameter_update_norm: float


def row_normalized_message_fixed_degree(graph: sp.csr_matrix, source: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Full message with a frozen full-degree denominator.

    Removing an edge later subtracts only its contribution.  Other neighbor
    weights are never re-normalized.
    """
    graph = sp.csr_matrix(graph, dtype=np.float64)
    degree = np.asarray(graph.sum(axis=1)).ravel()
    safe = np.where(degree > 0, degree, 1.0)
    message = graph.dot(np.asarray(source, dtype=np.float64)) / safe[:, None]
    return np.asarray(message, dtype=np.float32), safe.astype(np.float64)


def train_masked_predictor(
    target: np.ndarray,
    source: np.ndarray,
    graph: sp.csr_matrix,
    *,
    seed: int,
    n_masks: int = 4,
    steps: int = 64,
    lr: float = 0.02,
    weight_decay: float = 1e-3,
    device: str = "cpu",
    train_rows: np.ndarray | None = None,
) -> PredictorFit:
    if torch is None:
        raise RuntimeError("PyTorch is required for CEUP predictor training")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    target = np.asarray(target, dtype=np.float32)
    source = np.asarray(source, dtype=np.float32)
    message, _ = row_normalized_message_fixed_degree(graph, source)
    masks = deterministic_feature_masks(target.shape[1], n_masks, seed + 17)
    model = MaskedLinearPredictor(target.shape[1], source.shape[1]).to(device)
    initial_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    target_t = torch.from_numpy(target).to(device)
    message_t = torch.from_numpy(message).to(device)
    if train_rows is None:
        train_rows_np = np.arange(target.shape[0], dtype=np.int64)
    else:
        train_rows_np = np.asarray(train_rows, dtype=np.int64)
    if train_rows_np.size == 0:
        raise ValueError("empty node-crossfit training fold")
    train_rows_t = torch.from_numpy(train_rows_np).to(device)

    loss_trace: List[float] = []
    for step in range(int(steps)):
        held = masks[step % len(masks)]
        observed = target_t.clone()
        observed[:, held] = 0.0
        pred = model(observed, message_t)
        loss = ((pred[train_rows_t][:, held] - target_t[train_rows_t][:, held]) ** 2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_trace.append(float(loss.detach().cpu()))
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    update_norm = math.sqrt(
        sum(float(torch.sum((state[key] - initial_state[key]) ** 2)) for key in state)
    )
    return PredictorFit(
        state_dict=state,
        masks=masks,
        loss_trace=loss_trace,
        initial_loss=loss_trace[0],
        final_loss=loss_trace[-1],
        parameter_update_norm=update_norm,
    )


def deterministic_node_folds(ids: np.ndarray, n_folds: int) -> np.ndarray:
    if n_folds < 2:
        raise ValueError("node cross-fitting requires at least two folds")
    folds = np.empty(len(ids), dtype=np.int32)
    for i, value in enumerate(np.asarray(ids).tolist()):
        digest = hashlib.sha256(f"night17a-node-fold|{value}".encode("utf-8")).digest()
        folds[i] = int.from_bytes(digest[:8], "little") % int(n_folds)
    if np.unique(folds).size != n_folds:
        raise ValueError("deterministic node folds are incomplete")
    return folds


def crossfit_directed_utilities(
    models: Sequence[MaskedLinearPredictor],
    masks_by_fold: Sequence[Sequence[np.ndarray]],
    node_folds: np.ndarray,
    target: np.ndarray,
    source: np.ndarray,
    graph: sp.csr_matrix,
    edge_i: np.ndarray,
    edge_j: np.ndarray,
    edge_w: np.ndarray,
    *,
    device: str = "cpu",
) -> np.ndarray:
    if len(models) != len(masks_by_fold):
        raise ValueError("cross-fit model/mask count mismatch")
    output = None
    filled = np.zeros((edge_i.size, 2), dtype=bool)
    for fold, (model, masks) in enumerate(zip(models, masks_by_fold)):
        current = directed_leave_one_edge_out_utilities(
            model, masks, target, source, graph, edge_i, edge_j, edge_w, device=device
        )
        if output is None:
            output = np.empty_like(current)
        receive_i = node_folds[edge_i] == fold
        receive_j = node_folds[edge_j] == fold
        output[receive_i, 0, :] = current[receive_i, 0, :]
        output[receive_j, 1, :] = current[receive_j, 1, :]
        filled[receive_i, 0] = True
        filled[receive_j, 1] = True
    if output is None or not filled.all():
        raise RuntimeError("node-crossfit utility assembly left missing directed edges")
    return output


def load_predictor(target_dim: int, source_dim: int, state: Mapping[str, "torch.Tensor"], device: str = "cpu") -> MaskedLinearPredictor:
    if torch is None:
        raise RuntimeError("PyTorch is required for CEUP predictor reload")
    model = MaskedLinearPredictor(target_dim, source_dim).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def directed_leave_one_edge_out_utilities(
    model: MaskedLinearPredictor,
    masks: Sequence[np.ndarray],
    target: np.ndarray,
    source: np.ndarray,
    graph: sp.csr_matrix,
    edge_i: np.ndarray,
    edge_j: np.ndarray,
    edge_w: np.ndarray,
    *,
    device: str = "cpu",
) -> np.ndarray:
    """Return exact fixed-degree utilities [edge, orientation(i<-j,j<-i), mask]."""
    target = np.asarray(target, dtype=np.float32)
    source = np.asarray(source, dtype=np.float32)
    message, degree = row_normalized_message_fixed_degree(graph, source)
    target_t = torch.from_numpy(target).to(device)
    message_t = torch.from_numpy(message).to(device)
    source_t = torch.from_numpy(source).to(device)
    utilities = np.zeros((edge_i.size, 2, len(masks)), dtype=np.float64)
    weight = model.linear.weight.detach().to(device)[:, target.shape[1] :]
    with torch.no_grad():
        for mask_index, held_np in enumerate(masks):
            held = torch.as_tensor(held_np, dtype=torch.long, device=device)
            observed = target_t.clone()
            observed[:, held] = 0.0
            prediction = model(observed, message_t)
            residual = target_t[:, held] - prediction[:, held]
            held_weight = weight[held, :]
            for orientation, (receiver_np, sender_np) in enumerate(((edge_i, edge_j), (edge_j, edge_i))):
                receiver = torch.from_numpy(np.asarray(receiver_np, dtype=np.int64)).to(device)
                sender = torch.from_numpy(np.asarray(sender_np, dtype=np.int64)).to(device)
                alpha = torch.from_numpy((edge_w / degree[receiver_np]).astype(np.float32)).to(device)
                contribution = (source_t[sender] @ held_weight.T) * alpha[:, None]
                r = residual[receiver]
                delta = (2.0 * (r * contribution).sum(dim=1) + (contribution * contribution).sum(dim=1)) / float(len(held_np))
                utilities[:, orientation, mask_index] = delta.cpu().numpy().astype(np.float64)
    return utilities


def scalar_edge_utility_recompute(
    model: MaskedLinearPredictor,
    held: np.ndarray,
    target: np.ndarray,
    source: np.ndarray,
    graph: sp.csr_matrix,
    receiver: int,
    sender: int,
    weight: float,
) -> float:
    """Reference scalar LOO recomputation with the full-degree denominator fixed."""
    message, degree = row_normalized_message_fixed_degree(graph, source)
    observed = np.asarray(target, dtype=np.float32).copy()
    observed[:, held] = 0.0
    with torch.no_grad():
        full = model(
            torch.from_numpy(observed), torch.from_numpy(message)
        ).numpy()[receiver, held]
        cross_weight = model.linear.weight[:, target.shape[1] :].detach().numpy()[held]
    contribution = (float(weight) / float(degree[receiver])) * (np.asarray(source[sender]) @ cross_weight.T)
    without = full - contribution
    truth = np.asarray(target[receiver, held], dtype=np.float64)
    return float(np.mean((truth - without) ** 2) - np.mean((truth - full) ** 2))


def normalize_utility_tensor(raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Normalize per direction/mask, retaining directions rather than averaging them."""
    raw = np.asarray(raw, dtype=np.float64)
    normalized = np.empty_like(raw)
    scales = np.empty(raw.shape[1:], dtype=np.float64)
    for d in range(raw.shape[1]):
        for m in range(raw.shape[2]):
            values = raw[:, d, m]
            scale = float(np.median(np.abs(values)))
            if not np.isfinite(scale) or scale < 1e-12:
                scale = float(np.mean(np.abs(values)) + 1e-12)
            normalized[:, d, m] = values / scale
            scales[d, m] = scale
    mean = normalized.mean(axis=2)
    uncertainty = normalized.std(axis=2, ddof=1) / math.sqrt(normalized.shape[2])
    q_direction = np.tanh(mean / (uncertainty + 0.25))
    return mean, uncertainty, q_direction, scales


def symmetric_signed_evidence(q_direction: np.ndarray) -> Dict[str, np.ndarray]:
    """Common-positive attracts, common-negative repels, discordance rejects."""
    q_direction = np.asarray(q_direction, dtype=np.float64)
    positive = np.maximum(q_direction, 0.0)
    negative = np.maximum(-q_direction, 0.0)
    q_positive = positive.min(axis=1)
    q_negative = negative.min(axis=1)
    has_positive = positive.max(axis=1) > 0.0
    has_negative = negative.max(axis=1) > 0.0
    discordance = np.minimum(positive.max(axis=1), negative.max(axis=1))
    relation = np.zeros(q_direction.shape[0], dtype=np.int8)
    relation[q_positive > 0] = 1
    relation[q_negative > 0] = -1
    relation[has_positive & has_negative] = 0
    # Discordant edges are exact reject even if a numerical minimum leaked.
    conflict = has_positive & has_negative
    q_positive[conflict] = 0.0
    q_negative[conflict] = 0.0
    return {
        "q_positive": q_positive,
        "q_negative": q_negative,
        "q_signed": q_positive - q_negative,
        "discordance": discordance,
        "relation": relation,
    }


def deterministic_permutation(ids: np.ndarray, edge_i: np.ndarray, edge_j: np.ndarray, values: np.ndarray) -> np.ndarray:
    keys = []
    for i, j in zip(edge_i.tolist(), edge_j.tolist()):
        a, b = sorted((str(ids[i]), str(ids[j])))
        keys.append(hashlib.sha256(f"night17a|{a}|{b}".encode("utf-8")).digest())
    order = np.argsort(np.asarray(keys, dtype="S32"), kind="mergesort")
    shifted = np.roll(order, 1)
    out = np.empty_like(values)
    out[order] = np.asarray(values)[shifted]
    return out


def static_relation_evidence(view1: np.ndarray, view2: np.ndarray, edge_i: np.ndarray, edge_j: np.ndarray) -> Dict[str, np.ndarray]:
    d1 = np.sqrt(np.sum((view1[edge_i] - view1[edge_j]) ** 2, axis=1))
    d2 = np.sqrt(np.sum((view2[edge_i] - view2[edge_j]) ** 2, axis=1))
    s1 = np.exp(-d1 / (np.median(d1[d1 > 0]) + 1e-12))
    s2 = np.exp(-d2 / (np.median(d2[d2 > 0]) + 1e-12))
    qdir = np.stack([2 * s1 - 1, 2 * s1 - 1, 2 * s2 - 1, 2 * s2 - 1], axis=1)
    return symmetric_signed_evidence(qdir)


def fused_molecular_representation(view1: np.ndarray, view2: np.ndarray) -> np.ndarray:
    v1, _, _ = standardize(view1)
    v2, _, _ = standardize(view2)
    return np.concatenate([v1 / math.sqrt(v1.shape[1]), v2 / math.sqrt(v2.shape[1])], axis=1).astype(np.float32)


def kmeans_start(fused: np.ndarray, k: int, seed: int = SEED) -> np.ndarray:
    labels = KMeans(n_clusters=int(k), random_state=int(seed), n_init=20, max_iter=300).fit_predict(fused)
    return encode_partition(labels)


def prototype_unary(fused: np.ndarray, start: np.ndarray, k: int) -> np.ndarray:
    centers = np.vstack([fused[start == c].mean(axis=0) for c in range(k)])
    unary = np.sum((fused[:, None, :] - centers[None, :, :]) ** 2, axis=2).astype(np.float64)
    assigned = unary[np.arange(unary.shape[0]), start]
    scale = float(np.median(assigned[assigned > 1e-12])) if np.any(assigned > 1e-12) else 1.0
    return unary / max(scale, 1e-12)


def normalize_pairwise_weights(edge_w: np.ndarray, edge_i: np.ndarray, edge_j: np.ndarray, n: int) -> np.ndarray:
    degree = np.bincount(np.concatenate([edge_i, edge_j]), weights=np.concatenate([edge_w, edge_w]), minlength=n)
    scale = float(np.mean(degree[degree > 0]))
    return np.asarray(edge_w, dtype=np.float64) / max(scale, 1e-12)


def signed_energy(
    labels: np.ndarray,
    unary: np.ndarray,
    edge_i: np.ndarray,
    edge_j: np.ndarray,
    attraction: np.ndarray,
    repulsion: np.ndarray,
) -> float:
    labels = np.asarray(labels, dtype=np.int32)
    different = labels[edge_i] != labels[edge_j]
    same = ~different
    return float(unary[np.arange(labels.size), labels].sum() + attraction[different].sum() + repulsion[same].sum())


def signed_icm_partition(
    start: np.ndarray,
    unary: np.ndarray,
    edge_i: np.ndarray,
    edge_j: np.ndarray,
    attraction: np.ndarray,
    repulsion: np.ndarray,
    *,
    max_sweeps: int = 5,
    tolerance: float = 1e-10,
) -> Tuple[np.ndarray, Dict[str, object]]:
    labels = np.asarray(start, dtype=np.int32).copy()
    n, k = unary.shape
    if np.unique(labels).size != k:
        raise ValueError("start does not have exact K")
    adjacency: List[List[Tuple[int, int]]] = [[] for _ in range(n)]
    for e, (i, j) in enumerate(zip(edge_i.tolist(), edge_j.tolist())):
        adjacency[i].append((j, e))
        adjacency[j].append((i, e))
    counts = np.bincount(labels, minlength=k).astype(np.int64)
    energy_trace = [signed_energy(labels, unary, edge_i, edge_j, attraction, repulsion)]
    accepted_per_sweep: List[int] = []
    for _ in range(int(max_sweeps)):
        accepted = 0
        for node in range(n):
            old = int(labels[node])
            current = float(unary[node, old])
            for neighbor, edge in adjacency[node]:
                current += repulsion[edge] if labels[neighbor] == old else attraction[edge]
            best_label = old
            best_cost = current
            if counts[old] <= 1:
                continue
            for candidate in range(k):
                if candidate == old:
                    continue
                cost = float(unary[node, candidate])
                for neighbor, edge in adjacency[node]:
                    cost += repulsion[edge] if labels[neighbor] == candidate else attraction[edge]
                if cost < best_cost - tolerance:
                    best_cost = cost
                    best_label = candidate
            if best_label != old:
                labels[node] = best_label
                counts[old] -= 1
                counts[best_label] += 1
                accepted += 1
        new_energy = signed_energy(labels, unary, edge_i, edge_j, attraction, repulsion)
        if new_energy > energy_trace[-1] + 1e-7:
            raise RuntimeError("accepted ICM moves increased registered signed energy")
        energy_trace.append(new_energy)
        accepted_per_sweep.append(accepted)
        if accepted == 0:
            break
    if np.unique(labels).size != k or np.any(np.bincount(labels, minlength=k) == 0):
        raise RuntimeError("signed decoder violated exact-K/no-empty contract")
    return labels, {
        "energy_trace": energy_trace,
        "accepted_per_sweep": accepted_per_sweep,
        "changed_spots": int(np.count_nonzero(labels != start)),
        "cluster_sizes": np.bincount(labels, minlength=k).astype(int).tolist(),
    }


def make_arm_evidence(
    arm: str,
    learned: Mapping[str, np.ndarray],
    learned_no_uncertainty: Mapping[str, np.ndarray],
    static: Mapping[str, np.ndarray],
    single1: Mapping[str, np.ndarray],
    single2: Mapping[str, np.ndarray],
    ids: np.ndarray,
    edge_i: np.ndarray,
    edge_j: np.ndarray,
    edge_w: np.ndarray,
) -> Dict[str, np.ndarray]:
    zeros = np.zeros(edge_i.size, dtype=np.float64)
    if arm == "NO_OP":
        return {"q_positive": zeros.copy(), "q_negative": zeros.copy()}
    if arm == "LEARNED_SIGNED_FULL":
        return {"q_positive": learned["q_positive"].copy(), "q_negative": learned["q_negative"].copy()}
    if arm == "LEARNED_POSITIVE_ONLY":
        return {"q_positive": learned["q_positive"].copy(), "q_negative": zeros.copy()}
    if arm == "UNCERTAINTY_OFF":
        return {"q_positive": learned_no_uncertainty["q_positive"].copy(), "q_negative": learned_no_uncertainty["q_negative"].copy()}
    if arm == "STATIC_CMBF_TSRE":
        return {"q_positive": static["q_positive"].copy(), "q_negative": static["q_negative"].copy()}
    if arm == "SINGLE_VIEW1_UTILITY":
        return {"q_positive": single1["q_positive"].copy(), "q_negative": single1["q_negative"].copy()}
    if arm == "SINGLE_VIEW2_UTILITY":
        return {"q_positive": single2["q_positive"].copy(), "q_negative": single2["q_negative"].copy()}
    if arm == "UNIFORM_POSITIVE_MASS":
        mass = float(np.sum(edge_w * learned["q_positive"]) / np.sum(edge_w))
        return {"q_positive": np.full(edge_i.size, mass, dtype=np.float64), "q_negative": zeros.copy()}
    if arm == "EDGE_PERMUTATION":
        return {
            "q_positive": deterministic_permutation(ids, edge_i, edge_j, learned["q_positive"]),
            "q_negative": deterministic_permutation(ids, edge_i, edge_j, learned["q_negative"]),
        }
    raise KeyError(arm)


ARM_IDS = (
    "NO_OP",
    "UNIFORM_POSITIVE_MASS",
    "STATIC_CMBF_TSRE",
    "LEARNED_POSITIVE_ONLY",
    "LEARNED_SIGNED_FULL",
    "EDGE_PERMUTATION",
    "SINGLE_VIEW1_UTILITY",
    "SINGLE_VIEW2_UTILITY",
    "UNCERTAINTY_OFF",
)
