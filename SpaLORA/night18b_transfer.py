"""Night-18B clean-room graph measurement-response calibration.

The module estimates each modality's effective spatial response from robust
graph roughness and maps both views to a shared tissue bandwidth using a
bounded forward (smoothing) or regularized inverse (unsharp) operator.  The
calibration is analytic and label-free; a matched Night-18A residual backbone
then consumes the calibrated views.  No dataset identifier is accepted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Mapping, Sequence

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.cluster import KMeans

from SpaLORA.night18a_backbone import (
    SparseMultiViewBackbone,
    _state_sha,
    _student_assignment,
    decode_embedding,
    feature_graph,
    prepare_graph,
    row_normalize,
    scipy_to_torch,
    sha256_array,
    standardize,
)


@dataclass(frozen=True)
class TransferBackboneConfig:
    config_id: str = "T01_COMMON_NO_GRAPH"
    hidden_dim: int = 96
    residual_scale: float = 0.05
    learning_rate: float = 1e-3
    steps: int = 80
    mask_rate: float = 0.18
    reconstruction_weight: float = 1.0
    cross_view_weight: float = 0.20
    prototype_weight: float = 0.05
    anchor_weight: float = 0.10
    distortion_penalty: float = 0.025
    use_graph: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


ARMS = (
    "MATCHED_BACKBONE",
    "FULL_RESPONSE_CALIBRATION",
    "LOWPASS_ONLY_CALIBRATION",
    "SHARPEN_ONLY_CALIBRATION",
    "SHARED_SCALE_CONTROL",
    "SWAPPED_RESPONSE_CONTROL",
)

LOWPASS_GRID = ((0.10, 0.00), (0.20, 0.05), (0.35, 0.10), (0.50, 0.15))
SHARPEN_GRID = ((0.10, 0.15), (0.10, 0.20), (0.15, 0.35), (0.20, 0.50))


def _graph_authority_sha(graph: sp.csr_matrix) -> str:
    graph = sp.csr_matrix(graph)
    digest = hashlib.sha256()
    for value in (graph.indptr, graph.indices, graph.data, np.asarray(graph.shape, dtype=np.int64)):
        value = np.ascontiguousarray(value)
        digest.update(value.dtype.str.encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def robust_graph_roughness(value: np.ndarray, graph: sp.csr_matrix) -> float:
    """Median feature-wise residual energy under a registered graph operator."""

    value = standardize(value).astype(np.float64)
    operator = prepare_graph(graph, add_self=False).astype(np.float64)
    residual = value - operator @ value
    numerator = np.mean(residual * residual, axis=0)
    denominator = np.mean(value * value, axis=0)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator > 1e-10)
    if not np.any(valid):
        raise ValueError("no informative features for graph-response calibration")
    result = float(np.median(numerator[valid] / denominator[valid]))
    if not np.isfinite(result) or result <= 0:
        raise ValueError("invalid graph roughness")
    return result


def _normalized_laplacian(graph: sp.csr_matrix) -> sp.csr_matrix:
    graph = sp.csr_matrix(graph, dtype=np.float64)
    graph = graph.maximum(graph.T).tocsr(); graph.setdiag(0); graph.eliminate_zeros()
    degree = np.asarray(graph.sum(1)).ravel()
    inv = np.zeros_like(degree); inv[degree > 0] = 1.0 / np.sqrt(degree[degree > 0])
    adjacency = sp.diags(inv) @ graph @ sp.diags(inv)
    return (sp.eye(graph.shape[0], dtype=np.float64, format="csr") - adjacency).tocsr()


def _apply_response(value: np.ndarray, graph: sp.csr_matrix, beta: float, gamma: float,
                    iterations: int = 16) -> np.ndarray:
    """Apply (I + beta L)^-1 (I + gamma L) by fixed Richardson iterations.

    For beta >= 0 and a normalized Laplacian with spectrum in [0, 2], the
    fixed step 1/(1+beta) has contraction beta/(1+beta).  The registered grid
    keeps beta <= .5, so sixteen steps are a stable sparse approximation.
    """

    value = standardize(value).astype(np.float64)
    if beta < 0 or gamma < 0:
        raise ValueError("response coefficients must be nonnegative")
    if beta == 0 and gamma == 0:
        return value.astype(np.float32)
    laplacian = _normalized_laplacian(graph)
    right = value + gamma * (laplacian @ value)
    output = value.copy()
    step = 1.0 / (1.0 + beta)
    for _ in range(iterations):
        residual = right - (output + beta * (laplacian @ output))
        output = output + step * residual
    if not np.isfinite(output).all():
        raise RuntimeError("nonfinite response-calibrated view")
    return output.astype(np.float32)


def _candidate_plan(value: np.ndarray, graph: sp.csr_matrix, target: float,
                    allowed: str, distortion_penalty: float) -> tuple[dict[str, float | str], np.ndarray]:
    original = standardize(value)
    initial = robust_graph_roughness(original, graph)
    candidates: list[tuple[float, float]] = [(0.0, 0.0)]
    if allowed in ("full", "lowpass") and initial > target * (1.0 + 1e-7):
        candidates.extend(LOWPASS_GRID)
    if allowed in ("full", "sharpen") and initial < target * (1.0 - 1e-7):
        candidates.extend(SHARPEN_GRID)
    scale = max(float(np.mean(original.astype(np.float64) ** 2)), 1e-12)
    scored = []
    for beta, gamma in candidates:
        transformed = _apply_response(original, graph, beta, gamma)
        roughness = robust_graph_roughness(transformed, graph)
        distortion = float(np.sqrt(np.mean((transformed - original) ** 2) / scale))
        score = float(abs(np.log(max(roughness, 1e-12) / max(target, 1e-12))) + distortion_penalty * distortion)
        scored.append((score, 0 if beta == gamma else 1, beta + gamma, beta, gamma, roughness, distortion, transformed))
    best = min(scored, key=lambda row: (row[0], row[1], row[2], row[3], row[4]))
    plan = {"beta": float(best[3]), "gamma": float(best[4]), "initial_roughness": initial,
            "corrected_roughness": float(best[5]), "distortion": float(best[6]), "objective": float(best[0]),
            "response_type": "identity" if best[3] == best[4] else ("lowpass" if best[3] > best[4] else "sharpen")}
    return plan, best[7]


def _shared_response(value1: np.ndarray, value2: np.ndarray, graph: sp.csr_matrix,
                     target: float, distortion_penalty: float) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    originals = [standardize(value1), standardize(value2)]
    scale = [max(float(np.mean(value.astype(np.float64) ** 2)), 1e-12) for value in originals]
    scored = []
    for beta, gamma in ((0.0, 0.0), *LOWPASS_GRID, *SHARPEN_GRID):
        transformed = [_apply_response(value, graph, beta, gamma) for value in originals]
        roughness = [robust_graph_roughness(value, graph) for value in transformed]
        distortion = [float(np.sqrt(np.mean((transformed[i] - originals[i]) ** 2) / scale[i])) for i in range(2)]
        score = float(sum(abs(np.log(max(value, 1e-12) / max(target, 1e-12))) for value in roughness)
                      + distortion_penalty * sum(distortion))
        scored.append((score, beta, gamma, roughness, distortion, transformed))
    best = min(scored, key=lambda row: (row[0], row[1] + row[2], row[1], row[2]))
    plan = {"kind": "shared_rational_response", "beta": float(best[1]), "gamma": float(best[2]),
            "corrected_roughness": best[3], "distortion": best[4], "objective": float(best[0])}
    return plan, best[5][0], best[5][1]


def calibrate_views(view1: np.ndarray, view2: np.ndarray, graph: sp.csr_matrix, arm: str,
                    distortion_penalty: float = 0.025) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm}")
    initial = [robust_graph_roughness(view1, graph), robust_graph_roughness(view2, graph)]
    target = float(np.sqrt(initial[0] * initial[1]))
    if arm == "MATCHED_BACKBONE":
        output1, output2 = standardize(view1), standardize(view2)
        plans: object = [{"response_type": "identity", "beta": 0.0, "gamma": 0.0},
                         {"response_type": "identity", "beta": 0.0, "gamma": 0.0}]
    elif arm == "SHARED_SCALE_CONTROL":
        plans, output1, output2 = _shared_response(view1, view2, graph, target, distortion_penalty)
    else:
        allowed = {"FULL_RESPONSE_CALIBRATION": "full", "LOWPASS_ONLY_CALIBRATION": "lowpass",
                   "SHARPEN_ONLY_CALIBRATION": "sharpen", "SWAPPED_RESPONSE_CONTROL": "full"}[arm]
        plan1, candidate1 = _candidate_plan(view1, graph, target, allowed, distortion_penalty)
        plan2, candidate2 = _candidate_plan(view2, graph, target, allowed, distortion_penalty)
        if arm == "SWAPPED_RESPONSE_CONTROL":
            output1 = _apply_response(view1, graph, float(plan2["beta"]), float(plan2["gamma"]))
            output2 = _apply_response(view2, graph, float(plan1["beta"]), float(plan1["gamma"]))
            plans = [{"beta": plan2["beta"], "gamma": plan2["gamma"]},
                     {"beta": plan1["beta"], "gamma": plan1["gamma"]}]
        else:
            output1, output2, plans = candidate1, candidate2, [plan1, plan2]
    corrected = [robust_graph_roughness(output1, graph), robust_graph_roughness(output2, graph)]
    diagnostics = {"arm": arm, "initial_roughness": initial, "target_roughness": target,
                   "corrected_roughness": corrected, "plans": plans,
                   "log_gap_before": float(abs(np.log(initial[0] / initial[1]))),
                   "log_gap_after": float(abs(np.log(corrected[0] / corrected[1]))),
                   "graph_sha256": _graph_authority_sha(graph)}
    return output1.astype(np.float32), output2.astype(np.float32), diagnostics


def train_transfer_backbone(view1: np.ndarray, view2: np.ndarray, retained: np.ndarray,
                            spatial_graph: sp.csr_matrix, k: int, config: TransferBackboneConfig,
                            arm: str, seed: int, device: str) -> tuple[np.ndarray, Mapping[str, torch.Tensor], dict[str, object]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    encoded1, encoded2, response = calibrate_views(view1, view2, spatial_graph, arm, config.distortion_penalty)
    target1, target2 = standardize(view1), standardize(view2)
    base_np = row_normalize(standardize(retained))
    spatial = prepare_graph(spatial_graph, add_self=True)
    f1, f2 = feature_graph(encoded1), feature_graph(encoded2)
    base_partition = KMeans(n_clusters=k, n_init=20, random_state=0).fit_predict(base_np)
    prototypes = np.stack([base_np[base_partition == group].mean(0) for group in range(k)])
    model = SparseMultiViewBackbone(encoded1.shape[1], encoded2.shape[1], base_np.shape[1], config.hidden_dim,
                                    k, config.residual_scale, prototypes).to(device)
    initial_state_sha = _state_sha(model)
    x1, x2 = torch.as_tensor(encoded1, device=device), torch.as_tensor(encoded2, device=device)
    y1, y2 = torch.as_tensor(target1, device=device), torch.as_tensor(target2, device=device)
    base = torch.as_tensor(base_np, device=device)
    tsp, tf1, tf2 = scipy_to_torch(spatial, device), scipy_to_torch(f1, device), scipy_to_torch(f2, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    generator = torch.Generator(device=device); generator.manual_seed(seed + 7919)
    snapshots: list[dict[str, float]] = []
    model.train()
    for step in range(config.steps):
        mask1 = torch.rand(len(x1), generator=generator, device=device) < config.mask_rate
        mask2 = torch.rand(len(x2), generator=generator, device=device) < config.mask_rate
        if not mask1.any() or not mask2.any():
            raise RuntimeError("empty masked-reconstruction mask")
        input1, input2 = x1.clone(), x2.clone(); input1[mask1] = 0; input2[mask2] = 0
        z, recon1, recon2 = model(input1, input2, base, tsp, tf1, tf2, config.use_graph)
        reconstruction = torch.nn.functional.mse_loss(recon1[mask1], y1[mask1]) + torch.nn.functional.mse_loss(recon2[mask2], y2[mask2])
        z_drop1, _, _ = model(torch.zeros_like(x1), x2, base, tsp, tf1, tf2, config.use_graph)
        z_drop2, _, _ = model(x1, torch.zeros_like(x2), base, tsp, tf1, tf2, config.use_graph)
        cross_view = 0.5 * (torch.nn.functional.mse_loss(z, z_drop1) + torch.nn.functional.mse_loss(z, z_drop2))
        q = _student_assignment(z, model.prototypes)
        target = (q.detach() ** 2) / torch.clamp(q.detach().sum(0, keepdim=True), min=1e-8)
        target = target / torch.clamp(target.sum(1, keepdim=True), min=1e-8)
        prototype = torch.nn.functional.kl_div(torch.log(torch.clamp(q, min=1e-8)), target, reduction="batchmean")
        anchor = torch.nn.functional.mse_loss(z, base)
        loss = (config.reconstruction_weight * reconstruction + config.cross_view_weight * cross_view
                + config.prototype_weight * prototype + config.anchor_weight * anchor)
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite backbone loss")
        optimizer.zero_grad(set_to_none=True); loss.backward()
        gradient_norm = float(torch.sqrt(sum(torch.sum(p.grad ** 2) for p in model.parameters() if p.grad is not None)).detach().cpu())
        optimizer.step()
        if step in (0, config.steps - 1):
            snapshots.append({"step": step, "loss": float(loss.detach().cpu()), "gradient_norm": gradient_norm,
                              "reconstruction": float(reconstruction.detach().cpu()), "cross_view": float(cross_view.detach().cpu()),
                              "prototype": float(prototype.detach().cpu()), "anchor": float(anchor.detach().cpu())})
    model.eval()
    with torch.no_grad():
        representation, _, _ = model(x1, x2, base, tsp, tf1, tf2, config.use_graph)
    representation_np = representation.detach().cpu().numpy().astype(np.float32)
    final_state_sha = _state_sha(model)
    if final_state_sha == initial_state_sha:
        raise RuntimeError("optimizer did not update parameters")
    diagnostics = {"parameter_changed": True, "optimizer_steps": config.steps,
                   "initial_state_sha256": initial_state_sha, "final_state_sha256": final_state_sha,
                   "representation_sha256": sha256_array(representation_np),
                   "representation_delta_frobenius": float(np.linalg.norm(representation_np - base_np)),
                   "response": response, "loss_snapshots": snapshots,
                   "view1_shape": list(view1.shape), "view2_shape": list(view2.shape),
                   "retained_shape": list(retained.shape), "spatial_graph_nnz": int(spatial_graph.nnz),
                   "feature_graph1_nnz": int(f1.nnz), "feature_graph2_nnz": int(f2.nnz)}
    return representation_np, {key: value.detach().cpu() for key, value in model.state_dict().items()}, diagnostics


def reload_transfer_representation(view1: np.ndarray, view2: np.ndarray, retained: np.ndarray,
                                   spatial_graph: sp.csr_matrix, k: int, config: TransferBackboneConfig,
                                   arm: str, state: Mapping[str, torch.Tensor], device: str) -> np.ndarray:
    encoded1, encoded2, _ = calibrate_views(view1, view2, spatial_graph, arm, config.distortion_penalty)
    base_np = row_normalize(standardize(retained)); spatial = prepare_graph(spatial_graph, add_self=True)
    f1, f2 = feature_graph(encoded1), feature_graph(encoded2)
    base_partition = KMeans(n_clusters=k, n_init=20, random_state=0).fit_predict(base_np)
    prototypes = np.stack([base_np[base_partition == group].mean(0) for group in range(k)])
    model = SparseMultiViewBackbone(encoded1.shape[1], encoded2.shape[1], base_np.shape[1], config.hidden_dim,
                                    k, config.residual_scale, prototypes).to(device)
    loaded = model.load_state_dict(dict(state), strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError("strict checkpoint load failed")
    model.eval()
    with torch.no_grad():
        output, _, _ = model(torch.as_tensor(encoded1, device=device), torch.as_tensor(encoded2, device=device),
                             torch.as_tensor(base_np, device=device), scipy_to_torch(spatial, device),
                             scipy_to_torch(f1, device), scipy_to_torch(f2, device), config.use_graph)
    return output.detach().cpu().numpy().astype(np.float32)


__all__ = ["ARMS", "TransferBackboneConfig", "calibrate_views", "decode_embedding",
           "reload_transfer_representation", "robust_graph_roughness", "train_transfer_backbone"]
