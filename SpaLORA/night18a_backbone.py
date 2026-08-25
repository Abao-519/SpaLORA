"""Night-18A clean-room sparse multi-view backbone and portable decoder helpers.

This module accepts numeric matrices, registered sparse graphs, and known K.
It has no annotation or dataset-name input.  The representation scaffold uses
two modality adapters, sparse spatial/feature propagation, masked
reconstruction, a small DEC-style separation term, and a bounded residual
around an auditable retained embedding.  The downstream candidate generator
and structural selector operate on the same locked embedding.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Mapping, Sequence

import numpy as np
import scipy.sparse as sp
import torch
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors


def sha256_array(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def encode_partition(value: np.ndarray) -> np.ndarray:
    _, encoded = np.unique(np.asarray(value), return_inverse=True)
    return encoded.astype(np.int32)


def standardize(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError("numeric view must be a finite matrix")
    center = np.median(value, axis=0, keepdims=True)
    mad = np.median(np.abs(value - center), axis=0, keepdims=True) * 1.4826
    std = np.std(value, axis=0, keepdims=True)
    scale = np.where(mad > 1e-5, mad, np.where(std > 1e-5, std, 1.0))
    return np.clip((value - center) / scale, -12.0, 12.0).astype(np.float32)


def row_normalize(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    return (value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-6)).astype(np.float32)


def prepare_graph(graph: sp.csr_matrix, add_self: bool = False) -> sp.csr_matrix:
    graph = sp.csr_matrix(graph, dtype=np.float64).maximum(sp.csr_matrix(graph, dtype=np.float64).T).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    if add_self:
        graph = graph + sp.eye(graph.shape[0], dtype=np.float64, format="csr")
    degree = np.asarray(graph.sum(1)).ravel()
    inv = np.zeros_like(degree)
    inv[degree > 0] = 1.0 / degree[degree > 0]
    graph = sp.diags(inv) @ graph
    graph.sort_indices()
    return graph.astype(np.float32)


def feature_graph(view: np.ndarray, k: int = 8) -> sp.csr_matrix:
    """Construct a bounded sparse cosine kNN graph without dense N by N work."""

    x = row_normalize(standardize(view))
    neighbours = NearestNeighbors(n_neighbors=min(k + 1, len(x)), metric="cosine", n_jobs=1)
    neighbours.fit(x)
    distance, index = neighbours.kneighbors(x, return_distance=True)
    rows = np.repeat(np.arange(len(x), dtype=np.int64), index.shape[1] - 1)
    cols = index[:, 1:].reshape(-1)
    weight = np.clip(1.0 - distance[:, 1:].reshape(-1), 1e-4, 1.0)
    graph = sp.csr_matrix((weight, (rows, cols)), shape=(len(x), len(x)))
    return prepare_graph(graph, add_self=True)


def scipy_to_torch(graph: sp.csr_matrix, device: str) -> torch.Tensor:
    graph = sp.coo_matrix(graph)
    indices = torch.as_tensor(np.vstack([graph.row, graph.col]), dtype=torch.long, device=device)
    values = torch.as_tensor(graph.data, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(indices, values, graph.shape, device=device).coalesce()


@dataclass(frozen=True)
class BackboneConfig:
    config_id: str
    hidden_dim: int = 96
    residual_scale: float = 0.05
    learning_rate: float = 1e-3
    steps: int = 80
    mask_rate: float = 0.18
    reconstruction_weight: float = 1.0
    cross_view_weight: float = 0.20
    spatial_smooth_weight: float = 0.04
    prototype_weight: float = 0.05
    anchor_weight: float = 0.10
    use_graph: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class SparseModalityEncoder(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input = torch.nn.Linear(input_dim, hidden_dim)
        self.spatial = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.feature = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm = torch.nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, spatial: torch.Tensor, feature: torch.Tensor, use_graph: bool) -> torch.Tensor:
        base = torch.nn.functional.gelu(self.input(x))
        if use_graph:
            message = self.spatial(torch.sparse.mm(spatial, base)) + self.feature(torch.sparse.mm(feature, base))
            base = base + 0.5 * message
        return torch.nn.functional.gelu(self.norm(base))


class SparseMultiViewBackbone(torch.nn.Module):
    def __init__(self, d1: int, d2: int, latent_dim: int, hidden_dim: int, k: int, residual_scale: float, prototypes: np.ndarray):
        super().__init__()
        self.encoder1 = SparseModalityEncoder(d1, hidden_dim)
        self.encoder2 = SparseModalityEncoder(d2, hidden_dim)
        self.fuse = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 4, hidden_dim),
            torch.nn.GELU(),
            torch.nn.LayerNorm(hidden_dim),
        )
        self.residual = torch.nn.Linear(hidden_dim, latent_dim)
        torch.nn.init.zeros_(self.residual.weight)
        torch.nn.init.zeros_(self.residual.bias)
        self.decoder1 = torch.nn.Sequential(torch.nn.Linear(latent_dim, hidden_dim), torch.nn.GELU(), torch.nn.Linear(hidden_dim, d1))
        self.decoder2 = torch.nn.Sequential(torch.nn.Linear(latent_dim, hidden_dim), torch.nn.GELU(), torch.nn.Linear(hidden_dim, d2))
        self.prototypes = torch.nn.Parameter(torch.as_tensor(prototypes, dtype=torch.float32))
        self.residual_scale = float(residual_scale)
        self.k = int(k)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, base: torch.Tensor, spatial: torch.Tensor,
                feature1: torch.Tensor, feature2: torch.Tensor, use_graph: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h1 = self.encoder1(x1, spatial, feature1, use_graph)
        h2 = self.encoder2(x2, spatial, feature2, use_graph)
        fused = self.fuse(torch.cat([h1, h2, h1 * h2, torch.abs(h1 - h2)], dim=1))
        delta = torch.tanh(self.residual(fused))
        z = base + self.residual_scale * delta
        return z, self.decoder1(z), self.decoder2(z)


def _state_sha(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _student_assignment(z: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    distance = torch.sum((z[:, None, :] - prototypes[None, :, :]) ** 2, dim=2)
    q = 1.0 / (1.0 + distance)
    return q / torch.clamp(q.sum(dim=1, keepdim=True), min=1e-8)


def train_backbone(view1: np.ndarray, view2: np.ndarray, retained: np.ndarray, spatial_graph: sp.csr_matrix,
                   k: int, config: BackboneConfig, seed: int, device: str) -> tuple[np.ndarray, Mapping[str, torch.Tensor], dict[str, object]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    x1n, x2n = standardize(view1), standardize(view2)
    base_np = row_normalize(standardize(retained))
    spatial = prepare_graph(spatial_graph, add_self=True)
    f1, f2 = feature_graph(x1n), feature_graph(x2n)
    base_partition = KMeans(n_clusters=k, n_init=20, random_state=0).fit_predict(base_np)
    prototypes = np.stack([base_np[base_partition == group].mean(0) for group in range(k)])
    model = SparseMultiViewBackbone(x1n.shape[1], x2n.shape[1], base_np.shape[1], config.hidden_dim,
                                    k, config.residual_scale, prototypes).to(device)
    initial_state_sha = _state_sha(model)
    x1 = torch.as_tensor(x1n, device=device)
    x2 = torch.as_tensor(x2n, device=device)
    base = torch.as_tensor(base_np, device=device)
    tsp, tf1, tf2 = scipy_to_torch(spatial, device), scipy_to_torch(f1, device), scipy_to_torch(f2, device)
    upper = sp.triu(prepare_graph(spatial_graph), k=1, format="coo")
    edge_rows = torch.as_tensor(upper.row, dtype=torch.long, device=device)
    edge_cols = torch.as_tensor(upper.col, dtype=torch.long, device=device)
    edge_weight = torch.as_tensor(upper.data, dtype=torch.float32, device=device)
    edge_weight = edge_weight / torch.clamp(edge_weight.sum(), min=1e-8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 7919)
    losses: list[dict[str, float]] = []
    model.train()
    for step in range(config.steps):
        mask1 = torch.rand(len(x1), generator=generator, device=device) < config.mask_rate
        mask2 = torch.rand(len(x2), generator=generator, device=device) < config.mask_rate
        if not mask1.any() or not mask2.any():
            raise RuntimeError("masked reconstruction produced an empty mask")
        input1, input2 = x1.clone(), x2.clone()
        input1[mask1] = 0.0
        input2[mask2] = 0.0
        z, recon1, recon2 = model(input1, input2, base, tsp, tf1, tf2, config.use_graph)
        reconstruction = torch.nn.functional.mse_loss(recon1[mask1], x1[mask1]) + torch.nn.functional.mse_loss(recon2[mask2], x2[mask2])
        z_drop1, _, _ = model(torch.zeros_like(x1), x2, base, tsp, tf1, tf2, config.use_graph)
        z_drop2, _, _ = model(x1, torch.zeros_like(x2), base, tsp, tf1, tf2, config.use_graph)
        cross_view = 0.5 * (torch.nn.functional.mse_loss(z, z_drop1) + torch.nn.functional.mse_loss(z, z_drop2))
        if len(edge_rows):
            smooth = torch.sum(edge_weight[:, None] * (z[edge_rows] - z[edge_cols]) ** 2)
        else:
            smooth = torch.zeros((), device=device)
        q = _student_assignment(z, model.prototypes)
        target = (q.detach() ** 2) / torch.clamp(q.detach().sum(dim=0, keepdim=True), min=1e-8)
        target = target / torch.clamp(target.sum(dim=1, keepdim=True), min=1e-8)
        prototype = torch.nn.functional.kl_div(torch.log(torch.clamp(q, min=1e-8)), target, reduction="batchmean")
        anchor = torch.nn.functional.mse_loss(z, base)
        loss = (config.reconstruction_weight * reconstruction + config.cross_view_weight * cross_view
                + config.spatial_smooth_weight * smooth + config.prototype_weight * prototype
                + config.anchor_weight * anchor)
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite backbone loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(torch.sqrt(sum(torch.sum(parameter.grad ** 2) for parameter in model.parameters() if parameter.grad is not None)).detach().cpu())
        optimizer.step()
        if step in (0, config.steps - 1):
            losses.append({"step": step, "loss": float(loss.detach().cpu()), "reconstruction": float(reconstruction.detach().cpu()),
                           "cross_view": float(cross_view.detach().cpu()), "smooth": float(smooth.detach().cpu()),
                           "prototype": float(prototype.detach().cpu()), "anchor": float(anchor.detach().cpu()),
                           "gradient_norm": gradient_norm})
    model.eval()
    with torch.no_grad():
        representation, _, _ = model(x1, x2, base, tsp, tf1, tf2, config.use_graph)
    representation_np = representation.detach().cpu().numpy().astype(np.float32)
    final_state_sha = _state_sha(model)
    if final_state_sha == initial_state_sha:
        raise RuntimeError("optimizer did not change model parameters")
    if not np.isfinite(representation_np).all():
        raise RuntimeError("nonfinite learned representation")
    diagnostics = {
        "optimizer_steps": config.steps,
        "initial_state_sha256": initial_state_sha,
        "final_state_sha256": final_state_sha,
        "parameter_changed": True,
        "representation_sha256": sha256_array(representation_np),
        "representation_delta_frobenius": float(np.linalg.norm(representation_np - base_np)),
        "view1_shape": list(view1.shape), "view2_shape": list(view2.shape),
        "retained_shape": list(retained.shape), "spatial_graph_nnz": int(spatial_graph.nnz),
        "feature_graph1_nnz": int(f1.nnz), "feature_graph2_nnz": int(f2.nnz),
        "loss_snapshots": losses,
    }
    return representation_np, {key: value.detach().cpu() for key, value in model.state_dict().items()}, diagnostics


def reload_representation(view1: np.ndarray, view2: np.ndarray, retained: np.ndarray, spatial_graph: sp.csr_matrix,
                          k: int, config: BackboneConfig, state: Mapping[str, torch.Tensor], device: str) -> np.ndarray:
    x1n, x2n = standardize(view1), standardize(view2)
    base_np = row_normalize(standardize(retained))
    spatial = prepare_graph(spatial_graph, add_self=True)
    f1, f2 = feature_graph(x1n), feature_graph(x2n)
    base_partition = KMeans(n_clusters=k, n_init=20, random_state=0).fit_predict(base_np)
    prototypes = np.stack([base_np[base_partition == group].mean(0) for group in range(k)])
    model = SparseMultiViewBackbone(x1n.shape[1], x2n.shape[1], base_np.shape[1], config.hidden_dim,
                                    k, config.residual_scale, prototypes).to(device)
    result = model.load_state_dict(dict(state), strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("strict checkpoint load failed")
    model.eval()
    with torch.no_grad():
        representation, _, _ = model(torch.as_tensor(x1n, device=device), torch.as_tensor(x2n, device=device),
                                     torch.as_tensor(base_np, device=device), scipy_to_torch(spatial, device),
                                     scipy_to_torch(f1, device), scipy_to_torch(f2, device), config.use_graph)
    return representation.detach().cpu().numpy().astype(np.float32)


def graph_diffuse(representation: np.ndarray, graph: sp.csr_matrix, strength: float) -> np.ndarray:
    graph = prepare_graph(graph, add_self=False)
    return ((1.0 - strength) * representation + strength * (graph @ representation)).astype(np.float32)


def generate_candidate_bank(representation: np.ndarray, graph: sp.csr_matrix, k: int) -> tuple[np.ndarray, list[dict[str, object]]]:
    candidates: list[np.ndarray] = []
    records: list[dict[str, object]] = []

    def add(candidate_id: str, partition: np.ndarray, source: str, complexity: int) -> None:
        partition = encode_partition(partition)
        if len(np.unique(partition)) != k:
            raise RuntimeError("candidate violates exact K")
        candidates.append(partition)
        records.append({"candidate_id": candidate_id, "source": source, "complexity": complexity,
                        "candidate_sha256": sha256_array(partition)})

    add("COMMON_KMEANS_N20_S0", KMeans(k, n_init=20, random_state=0).fit_predict(representation), "RAW", 0)
    for seed in range(6):
        add(f"RAW_KMEANS_N1_S{seed}", KMeans(k, n_init=1, random_state=seed).fit_predict(representation), "RAW", 1)
    for seed in (0, 1):
        add(f"RAW_GMM_DIAG_S{seed}", GaussianMixture(k, covariance_type="diag", n_init=1, random_state=seed,
                                                     reg_covar=1e-5, max_iter=200).fit_predict(representation), "RAW_GMM", 2)
    for level, strength in enumerate((0.10, 0.20, 0.35), start=1):
        changed = graph_diffuse(representation, graph, strength)
        for seed in (0, 1):
            add(f"GRAPH_L{level}_KMEANS_S{seed}", KMeans(k, n_init=1, random_state=seed).fit_predict(changed),
                f"GRAPH_{strength:.2f}", 2 + level)
    partitions = np.stack(candidates).astype(np.int32)
    if len({row["candidate_id"] for row in records}) != len(records):
        raise RuntimeError("duplicate candidate ID")
    return partitions, records


def molecular_separation(view: np.ndarray, partition: np.ndarray) -> float:
    view = np.asarray(view, dtype=np.float64)
    partition = encode_partition(partition)
    center = view.mean(0)
    total = float(np.sum((view - center) ** 2))
    within = 0.0
    for group in np.unique(partition):
        block = view[partition == group]
        within += float(np.sum((block - block.mean(0)) ** 2))
    return float(max(total - within, 0.0) / max(total, 1e-12))


def topology_evidence(graph: sp.csr_matrix, partition: np.ndarray) -> float:
    graph = sp.csr_matrix(graph, dtype=np.float64)
    rows = np.repeat(np.arange(graph.shape[0]), np.diff(graph.indptr))
    partition = encode_partition(partition)
    agreement = float(np.sum(graph.data * (partition[rows] == partition[graph.indices])) / max(graph.data.sum(), 1e-12))
    probability = np.bincount(partition).astype(np.float64) / len(partition)
    chance = float(np.sum(probability ** 2))
    return float((agreement - chance) / max(1.0 - chance, 1e-12))


def feasibility(partition: np.ndarray, graph: sp.csr_matrix, k: int) -> dict[str, object]:
    partition = encode_partition(partition)
    sizes = np.bincount(partition, minlength=k)
    if len(sizes) != k or np.any(sizes == 0):
        return {"exact_k": False, "feasible": False, "min_cluster_size": 0, "min_internal_edges": 0}
    upper = sp.triu(sp.csr_matrix(graph).maximum(sp.csr_matrix(graph).T), k=1, format="coo")
    same = partition[upper.row] == partition[upper.col]
    counts = np.bincount(partition[upper.row[same]], minlength=k)
    feasible = bool(sizes.min() >= 2 and counts.min() >= 1)
    return {"exact_k": True, "feasible": feasible, "min_cluster_size": int(sizes.min()),
            "cluster_sizes": [int(x) for x in sizes], "min_internal_edges": int(counts.min())}


def _partition_similarity(partitions: np.ndarray) -> np.ndarray:
    result = np.eye(len(partitions), dtype=np.float64)
    for left in range(len(partitions)):
        for right in range(left):
            value = adjusted_rand_score(partitions[left], partitions[right])
            result[left, right] = result[right, left] = value
    return result


def decode_embedding(representation: np.ndarray, raw_view1: np.ndarray, raw_view2: np.ndarray,
                     graphs: Sequence[sp.csr_matrix], k: int) -> tuple[np.ndarray, list[dict[str, object]], dict[str, int]]:
    partitions, records = generate_candidate_bank(representation, graphs[0], k)
    standardized = [standardize(representation), standardize(raw_view1), standardize(raw_view2)]
    for index, partition in enumerate(partitions):
        explained = [molecular_separation(view, partition) for view in standardized]
        records[index]["molecular_joint"] = float(np.exp(np.mean(np.log(np.maximum(explained, 1e-12)))))
        records[index]["topology_joint"] = float(np.mean([topology_evidence(graph, partition) for graph in graphs]))
        records[index].update(feasibility(partition, graphs[0], k))
    feasible_indices = [index for index, row in enumerate(records) if bool(row["feasible"])]
    if not feasible_indices:
        raise RuntimeError("no structurally feasible candidate")
    similarity = _partition_similarity(partitions)
    medoid = min(feasible_indices, key=lambda index: (-float((similarity[index, feasible_indices].sum() - 1.0) /
                                                             max(len(feasible_indices) - 1, 1)), str(records[index]["candidate_id"])))
    molecular_values = np.asarray([records[index]["molecular_joint"] for index in feasible_indices])
    topology_values = np.asarray([records[index]["topology_joint"] for index in feasible_indices])
    molecular_rank = (rankdata(molecular_values, method="average") - 1.0) / max(len(feasible_indices) - 1, 1)
    topology_rank = (rankdata(topology_values, method="average") - 1.0) / max(len(feasible_indices) - 1, 1)
    molecular_local = int(np.argmax(molecular_rank))
    topology_local = int(np.argmax(topology_rank))
    structured_local = molecular_local if topology_rank[molecular_local] >= 0.5 else topology_local
    selections = {"COMMON_KMEANS": 0, "FEASIBLE_MEDOID": int(medoid),
                  "NIGHT16H_FIXED_STRUCTURED": int(feasible_indices[structured_local]),
                  "COMMON_GMM": next(index for index, row in enumerate(records) if row["candidate_id"] == "RAW_GMM_DIAG_S0")}
    return partitions, records, selections
