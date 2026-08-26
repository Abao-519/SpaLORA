"""Clean-room Night-21A anchored multiscale compositional fusion (AMCF).

The model consumes only numeric modality views, an auditable retained carrier,
and registered sparse graphs.  It constructs per-modality self/mean/high-pass
channels at several graph scales, composes them with node-wise weights, fuses
the two modalities continuously, and applies a bounded zero-initialized
residual to the retained carrier.  No annotation or dataset identifier enters
the model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Mapping, Sequence

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.cluster import KMeans


def sha256_array(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def standardize(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError("view must be a finite two-dimensional array")
    center = np.median(value, axis=0, keepdims=True)
    mad = 1.4826 * np.median(np.abs(value - center), axis=0, keepdims=True)
    std = np.std(value, axis=0, keepdims=True)
    scale = np.where(mad > 1e-5, mad, np.where(std > 1e-5, std, 1.0))
    return np.clip((value - center) / scale, -12.0, 12.0).astype(np.float32)


def row_normalize(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    return (value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-6)).astype(np.float32)


def prepare_graph(graph: sp.csr_matrix) -> sp.csr_matrix:
    graph = sp.csr_matrix(graph, dtype=np.float64)
    graph = graph.maximum(graph.T).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    if graph.shape[0] != graph.shape[1] or np.any(graph.data < 0) or not np.isfinite(graph.data).all():
        raise ValueError("registered graph must be finite, square, sparse, and nonnegative")
    degree = np.asarray(graph.sum(axis=1)).ravel()
    inverse = np.zeros_like(degree)
    inverse[degree > 0] = 1.0 / degree[degree > 0]
    result = (sp.diags(inverse) @ graph).tocsr()
    result.sort_indices()
    return result.astype(np.float32)


def multiscale_texture_bank(view: np.ndarray, graphs: Sequence[sp.csr_matrix], include_gradient: bool) -> tuple[np.ndarray, list[str]]:
    """Return self plus sparse low/high graph channels.

    The high-pass channel is X-PX.  It depends on the registered graph but not
    on a coordinate axis, so rigid coordinate rotation leaves it unchanged
    when graph registration is unchanged.
    """

    x = standardize(view)
    channels = [x]
    names = ["SELF"]
    for index, raw in enumerate(graphs):
        operator = prepare_graph(raw)
        mean = np.asarray(operator @ x, dtype=np.float32)
        channels.append(mean)
        names.append(f"MEAN_S{index}")
        if include_gradient:
            channels.append((x - mean).astype(np.float32))
            names.append(f"GRADIENT_S{index}")
    return np.stack(channels, axis=0), names


@dataclass(frozen=True)
class AMCFConfig:
    config_id: str
    arm: str
    hidden_dim: int = 48
    steps: int = 60
    learning_rate: float = 1e-3
    residual_bound: float = 0.12
    reconstruction_weight: float = 1.0
    cross_modal_weight: float = 0.05
    anchor_weight: float = 0.35
    prototype_weight: float = 0.02
    hierarchical: bool = True
    include_gradient: bool = True
    use_anchor: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


ARM_CONFIG = {
    "MEAN_ONLY_ANCHORED": dict(hierarchical=False, include_gradient=False, use_anchor=True),
    "HIERARCHICAL_WITHOUT_ANCHOR": dict(hierarchical=True, include_gradient=True, use_anchor=False),
    "ANCHOR_WITHOUT_GRADIENT": dict(hierarchical=True, include_gradient=False, use_anchor=True),
    "ANCHOR_WITHOUT_HIERARCHICAL_FUSION": dict(hierarchical=False, include_gradient=True, use_anchor=True),
    "FULL_COMPOSITION": dict(hierarchical=True, include_gradient=True, use_anchor=True),
}


def make_config(arm: str, profile: str = "FAMILY_BASE_V1", steps: int = 60) -> AMCFConfig:
    if arm not in ARM_CONFIG:
        raise KeyError(arm)
    return AMCFConfig(config_id=f"{profile}__{arm}", arm=arm, steps=steps, **ARM_CONFIG[arm])


class AMCFModel(torch.nn.Module):
    def __init__(self, d1: int, d2: int, latent_dim: int, channels: int, config: AMCFConfig,
                 initial_prototypes: np.ndarray):
        super().__init__()
        h = config.hidden_dim
        self.adapter1 = torch.nn.Linear(d1, h)
        self.adapter2 = torch.nn.Linear(d2, h)
        self.channel_score1 = torch.nn.Linear(h, 1)
        self.channel_score2 = torch.nn.Linear(h, 1)
        self.cross_score = torch.nn.Sequential(torch.nn.Linear(h * 3, h), torch.nn.GELU(), torch.nn.Linear(h, 2))
        self.fused_projector = torch.nn.Sequential(torch.nn.Linear(h * 3, h), torch.nn.GELU(), torch.nn.LayerNorm(h))
        self.residual = torch.nn.Linear(h, latent_dim)
        self.node_gate = torch.nn.Sequential(torch.nn.Linear(h * 3, h), torch.nn.GELU(), torch.nn.Linear(h, 1))
        self.decoder1 = torch.nn.Linear(latent_dim, d1)
        self.decoder2 = torch.nn.Linear(latent_dim, d2)
        self.prototypes = torch.nn.Parameter(torch.as_tensor(initial_prototypes, dtype=torch.float32))
        self.channels = int(channels)
        self.config = config
        torch.nn.init.zeros_(self.residual.weight)
        torch.nn.init.zeros_(self.residual.bias)

    def _compose(self, bank: torch.Tensor, adapter: torch.nn.Linear, scorer: torch.nn.Linear) -> tuple[torch.Tensor, torch.Tensor]:
        # bank: channels x observations x features
        encoded = torch.nn.functional.gelu(adapter(bank))
        if self.config.hierarchical:
            weights = torch.softmax(scorer(encoded).squeeze(-1), dim=0)
        else:
            weights = torch.full(encoded.shape[:2], 1.0 / encoded.shape[0], dtype=encoded.dtype, device=encoded.device)
        return torch.sum(weights[:, :, None] * encoded, dim=0), weights

    def forward(self, bank1: torch.Tensor, bank2: torch.Tensor, base: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        h1, weights1 = self._compose(bank1, self.adapter1, self.channel_score1)
        h2, weights2 = self._compose(bank2, self.adapter2, self.channel_score2)
        cross_input = torch.cat([h1, h2, torch.abs(h1 - h2)], dim=1)
        if self.config.hierarchical:
            modality_weights = torch.softmax(self.cross_score(cross_input), dim=1)
        else:
            modality_weights = torch.full((len(h1), 2), 0.5, dtype=h1.dtype, device=h1.device)
        mixed = modality_weights[:, :1] * h1 + modality_weights[:, 1:] * h2
        fused = self.fused_projector(torch.cat([mixed, h1 * h2, torch.abs(h1 - h2)], dim=1))
        if self.config.use_anchor:
            delta = torch.tanh(self.residual(fused))
            gate = torch.sigmoid(self.node_gate(cross_input))
            z = base + self.config.residual_bound * gate * delta
        else:
            # Matched non-anchored atomic arm: same fusion, unconstrained output.
            z = self.residual(fused)
            gate = torch.ones((len(h1), 1), dtype=h1.dtype, device=h1.device)
        return z, {"channel_weights1": weights1, "channel_weights2": weights2,
                   "modality_weights": modality_weights, "gate": gate,
                   "h1": h1, "h2": h2}


def _state_sha(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _assign(z: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    distance = torch.sum((z[:, None, :] - prototypes[None, :, :]) ** 2, dim=2)
    q = 1.0 / (1.0 + distance)
    return q / torch.clamp(q.sum(1, keepdim=True), min=1e-8)


def _model_and_inputs(view1: np.ndarray, view2: np.ndarray, retained: np.ndarray,
                      graphs: Sequence[sp.csr_matrix], k: int, config: AMCFConfig,
                      seed: int, device: str) -> tuple[AMCFModel, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, list[str]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    bank1_np, names = multiscale_texture_bank(view1, graphs, config.include_gradient)
    bank2_np, names2 = multiscale_texture_bank(view2, graphs, config.include_gradient)
    if names != names2:
        raise RuntimeError("modality texture schemas differ")
    base_np = row_normalize(standardize(retained))
    initial = KMeans(k, n_init=20, random_state=0).fit_predict(base_np)
    prototypes = np.stack([base_np[initial == group].mean(0) for group in range(k)]).astype(np.float32)
    model = AMCFModel(view1.shape[1], view2.shape[1], retained.shape[1], len(names), config, prototypes).to(device)
    return (model, torch.as_tensor(bank1_np, device=device), torch.as_tensor(bank2_np, device=device),
            torch.as_tensor(base_np, device=device), base_np, names)


def train_amcf(view1: np.ndarray, view2: np.ndarray, retained: np.ndarray, graphs: Sequence[sp.csr_matrix],
               k: int, config: AMCFConfig, seed: int, device: str) -> tuple[np.ndarray, Mapping[str, torch.Tensor], dict[str, object]]:
    model, bank1, bank2, base, base_np, channel_names = _model_and_inputs(view1, view2, retained, graphs, k, config, seed, device)
    model.eval()
    with torch.no_grad():
        initial_z, initial_aux = model(bank1, bank2, base)
    identity_error = float(torch.max(torch.abs(initial_z - base)).cpu()) if config.use_anchor else None
    if config.use_anchor and identity_error != 0.0:
        raise RuntimeError("bounded residual is not exact identity at initialization")
    initial_state_sha = _state_sha(model)
    target1 = torch.as_tensor(standardize(view1), device=device)
    target2 = torch.as_tensor(standardize(view2), device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    snapshots: list[dict[str, float]] = []
    model.train()
    for step in range(config.steps):
        z, aux = model(bank1, bank2, base)
        reconstruction = torch.nn.functional.mse_loss(model.decoder1(z), target1) + torch.nn.functional.mse_loss(model.decoder2(z), target2)
        cross_modal = 1.0 - torch.nn.functional.cosine_similarity(aux["h1"], aux["h2"], dim=1).mean()
        anchor = torch.nn.functional.mse_loss(z, base) if config.use_anchor else torch.zeros((), device=device)
        q = _assign(z, model.prototypes)
        target = (q.detach() ** 2) / torch.clamp(q.detach().sum(0, keepdim=True), min=1e-8)
        target = target / torch.clamp(target.sum(1, keepdim=True), min=1e-8)
        prototype = torch.nn.functional.kl_div(torch.log(torch.clamp(q, min=1e-8)), target, reduction="batchmean")
        loss = (config.reconstruction_weight * reconstruction + config.cross_modal_weight * cross_modal
                + config.anchor_weight * anchor + config.prototype_weight * prototype)
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite AMCF loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(torch.sqrt(sum(torch.sum(parameter.grad ** 2) for parameter in model.parameters() if parameter.grad is not None)).detach().cpu())
        optimizer.step()
        if step in (0, config.steps - 1):
            snapshots.append({"step": step, "loss": float(loss.detach().cpu()), "reconstruction": float(reconstruction.detach().cpu()),
                              "cross_modal": float(cross_modal.detach().cpu()), "anchor": float(anchor.detach().cpu()),
                              "prototype": float(prototype.detach().cpu()), "gradient_norm": gradient_norm})
    model.eval()
    with torch.no_grad():
        representation, aux = model(bank1, bank2, base)
    representation_np = representation.detach().cpu().numpy().astype(np.float32)
    final_state_sha = _state_sha(model)
    if final_state_sha == initial_state_sha or not np.isfinite(representation_np).all():
        raise RuntimeError("training did not produce a finite parameter update")
    delta = representation_np - base_np
    diagnostics = {
        "identity_initial_max_abs": identity_error, "parameter_changed": True,
        "initial_state_sha256": initial_state_sha, "final_state_sha256": final_state_sha,
        "representation_sha256": sha256_array(representation_np), "optimizer_steps": config.steps,
        "channel_names": channel_names, "channel_count_per_modality": len(channel_names),
        "residual_frobenius": float(np.linalg.norm(delta)), "residual_max_row_norm": float(np.linalg.norm(delta, axis=1).max()),
        "residual_bound": config.residual_bound, "loss_snapshots": snapshots,
        "gate_min": float(aux["gate"].min().cpu()), "gate_max": float(aux["gate"].max().cpu()), "gate_mean": float(aux["gate"].mean().cpu()),
        "modality1_weight_mean": float(aux["modality_weights"][:, 0].mean().cpu()),
        "modality2_weight_mean": float(aux["modality_weights"][:, 1].mean().cpu()),
        "channel1_weight_means": [float(x) for x in aux["channel_weights1"].mean(1).cpu()],
        "channel2_weight_means": [float(x) for x in aux["channel_weights2"].mean(1).cpu()],
        "view1_shape": list(view1.shape), "view2_shape": list(view2.shape), "retained_shape": list(retained.shape),
        "graph_shapes": [list(graph.shape) for graph in graphs], "graph_nnz": [int(graph.nnz) for graph in graphs],
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
    }
    return representation_np, {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}, diagnostics


def reload_amcf(view1: np.ndarray, view2: np.ndarray, retained: np.ndarray, graphs: Sequence[sp.csr_matrix],
                k: int, config: AMCFConfig, seed: int, state: Mapping[str, torch.Tensor], device: str = "cpu") -> np.ndarray:
    model, bank1, bank2, base, _, _ = _model_and_inputs(view1, view2, retained, graphs, k, config, seed, device)
    result = model.load_state_dict(dict(state), strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("strict checkpoint reload failed")
    model.eval()
    with torch.no_grad():
        representation, _ = model(bank1, bank2, base)
    return representation.detach().cpu().numpy().astype(np.float32)


def common_kmeans_endpoint(representation: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    representation = row_normalize(standardize(representation))
    partition = KMeans(k, n_init=20, random_state=seed).fit_predict(representation)
    _, encoded = np.unique(partition, return_inverse=True)
    encoded = encoded.astype(np.int32)
    if len(np.unique(encoded)) != k:
        raise RuntimeError("common endpoint violated exact K")
    return encoded

