#!/usr/bin/env python3
"""Night-14B score sprint runner.

Labels are loaded only by the evaluator inherited from Night-13B.  The model,
loss, optimizer, graph filter, checkpoint, and reload functions receive no
label-bearing object.  Dataset identifiers are used by this orchestration
layer solely to locate registered public inputs and per-dataset numerical HPO;
they never enter ``UnifiedEdgeStateModel.forward``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/night13b"))

import night13b_run as n13b  # noqa: E402
from SpaLORA.night6c_pipeline import run_head  # noqa: E402
from SpaLORA.night14a_tcf import seed_everything  # noqa: E402
from SpaLORA.night14b_atac import (  # noqa: E402
    UnifiedEdgeStateModel,
    anchored_majority_refine,
    array_sha256,
    canonical_sha256,
    diffuse,
    edge_state_loss,
    empirical_edge_states,
    initialize_cuda_device,
    multiscale_filter,
    spatial_operator,
    state_sha256,
)


ROOT = Path("/root/autodl-fs/night14b_atac_score_acceleration_20260823")
N14A = Path("/root/autodl-fs/night14a_topology_conflict_sprint_20260823")
N14A_RUN = N14A / "formal/development_cycle3/C15_BAL_XREC_600_WEAK_ALIGN"
N02_RUN = Path("/root/autodl-fs/night9b_racf_20260820/r1/r1-u009/attempt_001")
DATASETS = ("P22", "MISAR_E15_5_S1")
LABEL_POLICY = "PUBLIC_BENCHMARK_HPO_ONLY_NOT_IN_UNSUPERVISED_TRAINING"


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False,
                  allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_row(payload: Mapping[str, object], partition: np.ndarray) -> dict:
    return n13b.partition_metrics(
        payload["labels"], payload["label_mask"], np.asarray(partition),
        payload["metric_graph"],
    )


def views_from_archive(dataset: str, source: str) -> Dict[str, np.ndarray]:
    if source == "N02":
        if dataset != "P22":
            raise ValueError("N02 is a registered P22 historical reference only")
        value = np.load(N02_RUN / "views.npz", allow_pickle=False)
        return {key: np.asarray(value[key], dtype=np.float32) for key in (
            "emb_latent_omics1", "emb_latent_omics2", "SpaLORA_fused"
        )}
    if source == "N14A_C15":
        value = np.load(N14A_RUN / dataset / "seed_0/roundtrip_expected.npz",
                        allow_pickle=False)
        return {
            "emb_latent_omics1": np.asarray(value["z1"], dtype=np.float32),
            "emb_latent_omics2": np.asarray(value["z2"], dtype=np.float32),
            "SpaLORA_fused": np.asarray(value["fused"], dtype=np.float32),
        }
    raise ValueError("unknown representation source")


def filter_grid() -> List[dict]:
    rows: List[dict] = [{
        "filter_id": "I00_IDENTITY", "kind": "IDENTITY", "graph_k": 6,
        "beta": 0.0, "steps": 0,
    }]
    for graph_k in (4, 6, 8, 12, 18, 24):
        for beta in (0.2, 0.4, 0.6, 0.8):
            rows.append({
                "filter_id": "L_K%02d_B%02d" % (graph_k, round(beta * 10)),
                "kind": "FIXED_LOW", "graph_k": graph_k, "beta": beta,
                "steps": 1,
            })
    for graph_k in (6, 12, 18):
        for beta in (0.4, 0.6, 0.8):
            rows.append({
                "filter_id": "L2_K%02d_B%02d" % (graph_k, round(beta * 10)),
                "kind": "FIXED_LOW", "graph_k": graph_k, "beta": beta,
                "steps": 2,
            })
    multiscale = [
        ("M00", (4, 12), (0.2, 0.4, 0.4)),
        ("M01", (6, 18), (0.2, 0.5, 0.3)),
        ("M02", (8, 24), (0.1, 0.6, 0.3)),
        ("M03", (4, 18), (0.4, 0.4, 0.2)),
    ]
    for identifier, ks, weights in multiscale:
        rows.append({
            "filter_id": identifier, "kind": "MULTISCALE", "graph_k": list(ks),
            "weights": list(weights), "beta": None, "steps": 1,
        })
    # Rank-calibrated TSPR configs: the formula is the same in both datasets.
    for graph_k in (6, 12, 18):
        for interior in (0.40, 0.55, 0.70):
            for conflict in (0.20, 0.35):
                rows.append({
                    "filter_id": "T_K%02d_I%02d_C%02d" % (
                        graph_k, round(interior * 100), round(conflict * 100)
                    ),
                    "kind": "TSPR", "graph_k": graph_k,
                    "interior_quantile": interior, "boundary_quantile": 0.25,
                    "conflict_quantile": conflict, "floor": 0.0,
                    "beta": 0.8, "steps": 1,
                })
    identifiers = [row["filter_id"] for row in rows]
    if len(identifiers) != len(set(identifiers)) or not 40 <= len(rows) <= 100:
        raise RuntimeError("cheap filter grid contract failed")
    return rows


def apply_filter_config(
    views: Mapping[str, np.ndarray], payload: Mapping[str, object], config: Mapping[str, object],
) -> Tuple[Dict[str, np.ndarray], dict]:
    kind = str(config["kind"])
    if kind == "IDENTITY":
        return {key: np.asarray(value, dtype=np.float32).copy()
                for key, value in views.items()}, {"dense_n_by_n_count": 0}
    if kind == "MULTISCALE":
        operators = [spatial_operator(payload["coordinates"], int(k))
                     for k in config["graph_k"]]
        result = {key: multiscale_filter(value, operators, config["weights"])
                  for key, value in views.items()}
        return result, {"dense_n_by_n_count": 0}
    operator = spatial_operator(payload["coordinates"], int(config["graph_k"]))
    diagnostics: dict = {"dense_n_by_n_count": 0}
    if kind == "TSPR":
        operator, diagnostics = empirical_edge_states(
            views["emb_latent_omics1"], views["emb_latent_omics2"], operator,
            float(config["interior_quantile"]),
            float(config["boundary_quantile"]),
            float(config["conflict_quantile"]), float(config["floor"]),
        )
        if operator.nnz == 0:
            return {key: np.asarray(value, dtype=np.float32).copy()
                    for key, value in views.items()}, {
                        **diagnostics, "exact_identity_fallback": True,
                    }
    result = {
        key: diffuse(value, operator, float(config["beta"]), int(config["steps"]))
        for key, value in views.items()
    }
    return result, {**diagnostics, "exact_identity_fallback": False}


def evaluate_fast(
    dataset: str, source: str, filter_config: Mapping[str, object],
    payload: Mapping[str, object], views: Mapping[str, np.ndarray], k: int,
) -> List[dict]:
    filtered, diagnostics = apply_filter_config(views, payload, filter_config)
    fused = filtered["SpaLORA_fused"]
    rows: List[dict] = []
    for seed in (0,):
        # The cheap screen only ranks 40--100 frozen filters.  The selected
        # configurations are rerun with the registered H05 spectral endpoint;
        # using three restarts here avoids spending the sprint on a disposable
        # pre-screen while remaining deterministic.
        partition = KMeans(int(k), random_state=seed, n_init=3).fit_predict(fused)
        rows.append({
            "stage": "CHEAP", "dataset": dataset, "source": source,
            "filter_id": filter_config["filter_id"], "head": "KMEANS_FUSED_NINIT3_SCREEN",
            "cluster_k": int(k), "endpoint_seed": seed,
            "embedding_sha256": array_sha256(fused),
            "partition_sha256": array_sha256(partition.astype(np.int64)),
            "label_policy": LABEL_POLICY, "status": "PASS", **diagnostics,
            **metric_row(payload, partition),
        })
    return rows


def spectral_shortlist(
    cheap: pd.DataFrame, dataset: str, source: str, k: int,
    payload: Mapping[str, object], views: Mapping[str, np.ndarray],
    configs: Mapping[str, Mapping[str, object]],
) -> List[dict]:
    subset = cheap[(cheap.dataset == dataset) & (cheap.source == source)
                   & (cheap.cluster_k == int(k))].sort_values(
                       ["absolute_ari", "absolute_nmi"], ascending=False
                   )
    forced = {"I00_IDENTITY"}
    if dataset == "P22" and source == "N14A_C15":
        forced.add("L_K12_B08")
    if dataset == "MISAR_E15_5_S1":
        forced.update({"L_K12_B06", "L2_K06_B06"})
    selected = list(dict.fromkeys(
        list(subset.filter_id.head(2).astype(str)) + sorted(forced)
    ))
    rows: List[dict] = []
    for filter_id in selected:
        config = configs[filter_id]
        filtered, diagnostics = apply_filter_config(views, payload, config)
        started = time.perf_counter()
        partition, head_audit = run_head(
            {"id": "H05_EQUAL3_AFFINITY_SPECTRAL"}, filtered, int(k),
            payload["coordinates"], payload["ids"],
        )
        base_row = {
            "stage": "SHORTLIST", "dataset": dataset, "source": source,
            "filter_id": filter_id, "head": "H05_EQUAL3_AFFINITY_SPECTRAL",
            "cluster_k": int(k), "endpoint_seed": 2020,
            "embedding_sha256": array_sha256(filtered["SpaLORA_fused"]),
            "partition_sha256": array_sha256(np.asarray(partition, dtype=np.int64)),
            "endpoint_wall_seconds": float(time.perf_counter() - started),
            "label_policy": LABEL_POLICY, "status": "PASS", **diagnostics,
            **{("head_" + key): value for key, value in head_audit.items()
               if isinstance(value, (str, int, float, bool))},
            **metric_row(payload, partition),
        }
        rows.append(base_row)
        # One common spatial Potts grid is evaluated after the molecular head.
        for head_k, anchor, iterations in ((18, 0.0, 2), (24, 0.2, 10),
                                            (30, 0.3, 10), (12, 0.2, 10)):
            operator = spatial_operator(payload["coordinates"], head_k)
            refined = anchored_majority_refine(
                partition, operator, int(k), anchor, iterations
            )
            rows.append({
                **base_row,
                "head": "H05_PLUS_ANCHORED_MAJORITY",
                "head_graph_k": head_k, "head_anchor": anchor,
                "head_iterations": iterations,
                "partition_sha256": array_sha256(refined.astype(np.int64)),
                **metric_row(payload, refined),
            })
    return rows


def stage0_protocol(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    target = pd.DataFrame([
        {
            "lane": "P22_PROJECT_K9", "dataset_version": "project P22 paired h5ad",
            "annotation": "MouseBrain_groundtruth.csv coarse manual-anno",
            "mask": "9196/9196", "k": 9, "endpoint": "N02 native H05",
            "reported_or_internal_ari": 0.5063, "reported_or_internal_nmi": 0.6562,
            "source": "Night-14A audited N02 native context", "directly_comparable": True,
        },
        {
            "lane": "P22_PAPER_K18", "dataset_version": "3d-OT Dataset7 Mouse Brain ATAC",
            "annotation": "3d-OT h5ad truth; artifact not present at Stage 0",
            "mask": "pending exact artifact", "k": 18, "endpoint": "mclust per official tutorial",
            "reported_or_internal_ari": 0.390, "reported_or_internal_nmi": np.nan,
            "source": "3d-OT paper/docs; provisional Worker-2 target",
            "directly_comparable": False,
        },
        {
            "lane": "MISAR_PROJECT_K7", "dataset_version": "MISAR E15.5 S1 1949 spots",
            "annotation": "deposited carrier Y, seven unique classes",
            "mask": "1949/1949", "k": 7, "endpoint": "Night-14A support-only",
            "reported_or_internal_ari": 0.3137, "reported_or_internal_nmi": 0.4924,
            "source": "Night-14A audited support-only ablation", "directly_comparable": True,
        },
        {
            "lane": "MISAR_PAPER_K12", "dataset_version": "MISAR E15.5 S1 1949 spots",
            "annotation": "deposited carrier Y; head K is 12",
            "mask": "1949/1949", "k": 12, "endpoint": "SEPAR clustering K12",
            "reported_or_internal_ari": 0.644, "reported_or_internal_nmi": np.nan,
            "source": "SEPAR paper and official Tutorial 4", "directly_comparable": "PROTOCOL_CONTEXT_ONLY",
        },
    ])
    target.to_csv(output / "reported_high_water_target_board.csv", index=False)
    protocol = target[["lane", "dataset_version", "annotation", "mask", "k", "endpoint",
                       "directly_comparable"]].copy()
    protocol["labels_allowed_for_hpo"] = True
    protocol["labels_in_unsupervised_loss"] = False
    protocol.to_csv(output / "protocol_registry.csv", index=False)


def cheap_search(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    grid = filter_grid()
    atomic_json(output / "cheap_filter_grid.json", {
        "config_count": len(grid), "grid": grid,
        "formula_changes_have_distinct_filter_ids": True,
    })
    configs = {row["filter_id"]: row for row in grid}
    cheap_rows: List[dict] = []
    spectral_rows: List[dict] = []
    for dataset in DATASETS:
        payload = n13b.base_payload(dataset)
        sources = ("N14A_C15", "N02") if dataset == "P22" else ("N14A_C15",)
        ks = (9,) if dataset == "P22" else (7, 12)
        for source in sources:
            views = views_from_archive(dataset, source)
            for config in grid:
                for k in ks:
                    cheap_rows.extend(evaluate_fast(
                        dataset, source, config, payload, views, k
                    ))
            cheap = pd.DataFrame(cheap_rows)
            cheap.to_csv(output / "cheap_filter_search.partial.csv", index=False)
            for k in ks:
                spectral_rows.extend(spectral_shortlist(
                    cheap, dataset, source, k, payload, views, configs
                ))
            pd.DataFrame(spectral_rows).to_csv(
                output / "shortlist_spectral_search.partial.csv", index=False
            )
    cheap = pd.DataFrame(cheap_rows)
    spectral = pd.DataFrame(spectral_rows)
    cheap.to_csv(output / "cheap_filter_search.csv", index=False)
    spectral.to_csv(output / "shortlist_spectral_search.csv", index=False)
    (output / "cheap_filter_search.partial.csv").unlink(missing_ok=True)
    (output / "shortlist_spectral_search.partial.csv").unlink(missing_ok=True)
    combined = pd.concat([cheap, spectral], ignore_index=True, sort=False)
    combined.to_csv(output / "stage1_all_run_ledger.csv", index=False)
    best = combined.sort_values(["dataset", "cluster_k", "absolute_ari", "absolute_nmi"],
                                ascending=[True, True, False, False]).groupby(
                                    ["dataset", "cluster_k"], as_index=False
                                ).head(10)
    best.to_csv(output / "stage1_best_run_board.csv", index=False)
    atomic_json(output / "stage1_manifest.json", {
        "filter_config_count": len(grid),
        "cheap_row_count": len(cheap), "shortlist_row_count": len(spectral),
        "all_failures_preserved": True, "labels_in_model_or_filter": False,
        "labels_used_by_evaluator_and_cross_run_hpo": True,
        "dataset_name_in_core_forward": False,
    })


def torch_graph(operator: sp.spmatrix, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    coo = operator.tocoo()
    edge_index = torch.as_tensor(np.vstack((coo.row, coo.col)), dtype=torch.long,
                                 device=device)
    edge_weight = torch.as_tensor(coo.data, dtype=torch.float32, device=device)
    return edge_index, edge_weight


def validate_training_config(config: Mapping[str, object]) -> None:
    required = {
        "candidate_id", "edge_mode", "base_config", "initial_low_strength",
        "initial_high_strength", "support_scale", "support_center",
        "conflict_scale", "conflict_center", "boundary_scale",
        "boundary_center", "edge_loss_weights", "optimizer", "learning_rate",
        "weight_decay", "steps", "gradient_clip", "freeze_base",
        "post_graph_k", "post_beta", "post_steps", "head_graph_k",
        "head_anchor", "head_iterations", "cluster_ks",
    }
    if set(config) != required:
        raise ValueError("training config schema mismatch")
    if str(config["edge_mode"]) not in UnifiedEdgeStateModel.MODES:
        raise ValueError("training edge mode mismatch")
    if set(config["edge_loss_weights"]) != {
        "edge_reconstruction", "trusted_smoothness", "rejected_edge_retention"
    }:
        raise ValueError("edge loss schema mismatch")
    if not config["cluster_ks"] or any(int(k) <= 1 for k in config["cluster_ks"]):
        raise ValueError("cluster_ks must contain positive registered endpoints")


def load_training_input(dataset: str) -> Tuple[dict, np.ndarray, np.ndarray, sp.csr_matrix]:
    payload = n13b.base_payload(dataset)
    x1 = StandardScaler().fit_transform(
        np.asarray(payload["view1"], dtype=np.float64)
    ).astype(np.float32)
    x2 = StandardScaler().fit_transform(
        np.asarray(payload["view2"], dtype=np.float64)
    ).astype(np.float32)
    graph = spatial_operator(payload["coordinates"], 8)
    return payload, x1, x2, graph


def train_one(
    dataset: str, config: Mapping[str, object], seed: int, output: Path,
    warmstart_base: Optional[Path], fresh_reload: bool,
) -> dict:
    validate_training_config(config)
    seed_everything(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    initialize_cuda_device(device)
    payload, x1_np, x2_np, graph = load_training_input(dataset)
    edge_index, edge_weight = torch_graph(graph, device)
    x1 = torch.as_tensor(x1_np, device=device)
    x2 = torch.as_tensor(x2_np, device=device)
    model = UnifiedEdgeStateModel(x1.shape[1], x2.shape[1], config).to(device)
    warmstart_sha = None
    if warmstart_base is not None:
        checkpoint = torch.load(warmstart_base, map_location="cpu")
        model.base.load_state_dict(checkpoint["state_dict"], strict=True)
        warmstart_sha = file_sha256(warmstart_base)
    if bool(config["freeze_base"]):
        for parameter in model.base.parameters():
            parameter.requires_grad_(False)
    initial_sha = state_sha256(model.state_dict())
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    started = time.perf_counter()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    trace: List[dict] = []
    first_gradient = None
    for step in range(int(config["steps"])):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        result = model(x1, x2, edge_index, edge_weight)
        loss, audit = edge_state_loss(
            model, result, x1, x2, edge_index, config["edge_loss_weights"]
        )
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite Night-14B loss")
        loss.backward()
        norm = torch.sqrt(sum(
            (parameter.grad.detach().square().sum() for parameter in trainable
             if parameter.grad is not None),
            torch.zeros((), device=device),
        ))
        if not torch.isfinite(norm) or float(norm) <= 0:
            raise RuntimeError("zero or non-finite Night-14B gradient")
        if first_gradient is None:
            first_gradient = float(norm.detach().cpu())
        torch.nn.utils.clip_grad_norm_(trainable, float(config["gradient_clip"]))
        optimizer.step()
        if step == 0 or step == int(config["steps"]) - 1 or (
            (step + 1) % max(1, int(config["steps"]) // 10) == 0
        ):
            trace.append({"step": step + 1, **audit})
    end_event.record()
    torch.cuda.synchronize(device)
    gpu_seconds = float(start_event.elapsed_time(end_event) / 1000.0)
    model.eval()
    with torch.no_grad():
        result = model(x1, x2, edge_index, edge_weight)
    views = {
        "emb_latent_omics1": result["z1"].detach().cpu().numpy().astype(np.float32),
        "emb_latent_omics2": result["z2"].detach().cpu().numpy().astype(np.float32),
        "SpaLORA_fused": result["edge_fused"].detach().cpu().numpy().astype(np.float32),
    }
    post = spatial_operator(payload["coordinates"], int(config["post_graph_k"]))
    filtered = {
        key: diffuse(value, post, float(config["post_beta"]), int(config["post_steps"]))
        for key, value in views.items()
    }
    endpoint_results: List[dict] = []
    partitions: Dict[int, np.ndarray] = {}
    head_operator = spatial_operator(
        payload["coordinates"], int(config["head_graph_k"])
    )
    for cluster_k in map(int, config["cluster_ks"]):
        partition, head_audit = run_head(
            {"id": "H05_EQUAL3_AFFINITY_SPECTRAL"}, filtered, cluster_k,
            payload["coordinates"], payload["ids"],
        )
        refined = anchored_majority_refine(
            partition, head_operator, cluster_k, float(config["head_anchor"]),
            int(config["head_iterations"]),
        )
        partitions[cluster_k] = np.asarray(refined, dtype=np.int64)
        endpoint_results.append({
            "cluster_k": cluster_k,
            "partition_sha256": array_sha256(partitions[cluster_k]),
            "head_audit": head_audit,
            **metric_row(payload, partitions[cluster_k]),
        })
    output.mkdir(parents=True, exist_ok=False)
    checkpoint_path = output / "checkpoint.pt"
    torch.save({
        "state_dict": {key: value.detach().cpu()
                       for key, value in model.state_dict().items()},
        "config": dict(config), "config_sha256": canonical_sha256(config),
        "input1": int(x1.shape[1]), "input2": int(x2.shape[1]),
        "seed": int(seed), "dataset": dataset,
    }, checkpoint_path)
    roundtrip_values = {
        "x1": x1_np, "x2": x2_np,
        "edge_index": edge_index.detach().cpu().numpy(),
        "edge_weight": edge_weight.detach().cpu().numpy(),
        "expected_z1": views["emb_latent_omics1"],
        "expected_z2": views["emb_latent_omics2"],
        "expected_fused": views["SpaLORA_fused"],
    }
    roundtrip_values.update({
        "expected_partition_k%d" % cluster_k: partition
        for cluster_k, partition in partitions.items()
    })
    np.savez_compressed(output / "roundtrip_input.npz", **roundtrip_values)
    pd.DataFrame(trace).to_csv(output / "loss_trace.csv", index=False)
    cluster_rows = []
    for cluster_k, partition in partitions.items():
        cluster_rows.extend({
            "cluster_k": cluster_k, "observation_id": str(observation_id),
            "cluster": int(cluster),
        } for observation_id, cluster in zip(payload["ids"], partition))
    pd.DataFrame(cluster_rows).to_csv(output / "clusters.csv", index=False)
    final_sha = state_sha256(model.state_dict())
    metrics = dict(endpoint_results[0])
    audit = {
        "dataset": dataset, "candidate_id": config["candidate_id"],
        "model_seed": int(seed), "status": "PASS",
        "config_sha256": canonical_sha256(config),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "warmstart_base_checkpoint": None if warmstart_base is None else str(warmstart_base),
        "warmstart_base_sha256": warmstart_sha,
        "initial_state_sha256": initial_sha, "final_state_sha256": final_sha,
        "parameters_changed": initial_sha != final_sha,
        "optimizer_steps": int(config["steps"]),
        "trainable_parameter_count": int(sum(p.numel() for p in trainable)),
        "first_gradient_norm": first_gradient,
        "raw_shapes": payload["raw_shapes"],
        "processed_shapes": [list(x1_np.shape), list(x2_np.shape)],
        "latent_shapes": {key: list(value.shape) for key, value in views.items()},
        "ordered_id_sha256": n13b.ordered_id_sha256(payload["ids"]),
        "embedding_sha256": {key: array_sha256(value) for key, value in filtered.items()},
        "partition_sha256": {
            str(cluster_k): array_sha256(partition)
            for cluster_k, partition in partitions.items()
        },
        "endpoint_results": endpoint_results,
        "labels_in_loss_gradient_or_checkpoint_selection": False,
        "labels_used_for_cross_run_hpo_and_evaluation": True,
        "dataset_name_in_model_forward": False,
        "wall_seconds": float(time.perf_counter() - started),
        "gpu_seconds": gpu_seconds,
        "peak_gpu_mib": float(torch.cuda.max_memory_allocated() / 1048576.0),
        "peak_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
        **{key: value for key, value in metrics.items()
           if key not in {"partition_sha256", "head_audit"}},
    }
    atomic_json(output / "training_audit.json", audit)
    if fresh_reload:
        subprocess.run([
            sys.executable, str(Path(__file__).resolve()), "reload",
            "--run-dir", str(output),
        ], cwd=str(REPO), check=True)
        reload_audit = read_json(output / "fresh_process_reload.json")
        audit["fresh_process_reload"] = reload_audit
        audit["status"] = "PASS" if reload_audit["status"] == "PASS" else "FAIL"
        atomic_json(output / "training_audit.json", audit)
    return audit


def reload_run(run_dir: Path) -> None:
    checkpoint = torch.load(run_dir / "checkpoint.pt", map_location="cpu")
    config = checkpoint["config"]
    if canonical_sha256(config) != checkpoint["config_sha256"]:
        raise RuntimeError("checkpoint config SHA mismatch")
    values = np.load(run_dir / "roundtrip_input.npz", allow_pickle=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = UnifiedEdgeStateModel(
        int(checkpoint["input1"]), int(checkpoint["input2"]), config
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    x1 = torch.as_tensor(values["x1"], dtype=torch.float32, device=device)
    x2 = torch.as_tensor(values["x2"], dtype=torch.float32, device=device)
    edge_index = torch.as_tensor(values["edge_index"], dtype=torch.long, device=device)
    edge_weight = torch.as_tensor(values["edge_weight"], dtype=torch.float32, device=device)
    with torch.no_grad():
        result = model(x1, x2, edge_index, edge_weight)
    actual = {
        "z1": result["z1"].detach().cpu().numpy().astype(np.float32),
        "z2": result["z2"].detach().cpu().numpy().astype(np.float32),
        "fused": result["edge_fused"].detach().cpu().numpy().astype(np.float32),
    }
    maximum = {
        key: float(np.max(np.abs(actual[key] - values["expected_" + key])))
        for key in actual
    }
    payload = n13b.base_payload(str(checkpoint["dataset"]))
    filtered = {
        "emb_latent_omics1": actual["z1"],
        "emb_latent_omics2": actual["z2"],
        "SpaLORA_fused": actual["fused"],
    }
    post = spatial_operator(payload["coordinates"], int(config["post_graph_k"]))
    filtered = {
        key: diffuse(value, post, float(config["post_beta"]), int(config["post_steps"]))
        for key, value in filtered.items()
    }
    partition_exact = {}
    for cluster_k in map(int, config["cluster_ks"]):
        partition, _ = run_head(
            {"id": "H05_EQUAL3_AFFINITY_SPECTRAL"}, filtered, cluster_k,
            payload["coordinates"], payload["ids"],
        )
        refined = anchored_majority_refine(
            partition,
            spatial_operator(payload["coordinates"], int(config["head_graph_k"])),
            cluster_k, float(config["head_anchor"]),
            int(config["head_iterations"]),
        )
        partition_exact[str(cluster_k)] = bool(np.array_equal(
            np.asarray(refined, dtype=np.int64),
            values["expected_partition_k%d" % cluster_k],
        ))
    audit = {
        "fresh_process": True, "strict_state_load": True,
        "state_sha256": state_sha256(model.state_dict()),
        "embedding_max_abs": maximum,
        "embedding_numerical_roundtrip": all(value <= 1e-5 for value in maximum.values()),
        "partition_exact_roundtrip": partition_exact,
        "endpoint_roundtrip": all(partition_exact.values()),
    }
    audit["status"] = "PASS" if (
        audit["embedding_numerical_roundtrip"] and audit["endpoint_roundtrip"]
    ) else "FAIL"
    atomic_json(run_dir / "fresh_process_reload.json", audit)


def train_batch(args: argparse.Namespace) -> None:
    config = read_json(Path(args.config))
    datasets = [item for item in args.datasets.split(",") if item]
    seeds = [int(item) for item in args.seeds.split(",") if item]
    output = Path(args.output)
    rows: List[dict] = []
    for dataset in datasets:
        for seed in seeds:
            warmstart = None
            if args.warmstart_night14a:
                warmstart = N14A_RUN / dataset / ("seed_%d" % seed) / "checkpoint.pt"
                if not warmstart.exists():
                    # Initial screens use the registered seed-0 warm start.
                    warmstart = N14A_RUN / dataset / "seed_0/checkpoint.pt"
            run_dir = output / str(config["candidate_id"]) / dataset / ("seed_%d" % seed)
            try:
                rows.append(train_one(
                    dataset, config, seed, run_dir, warmstart, args.fresh_reload
                ))
            except Exception as error:
                run_dir.mkdir(parents=True, exist_ok=True)
                failure = {
                    "dataset": dataset, "candidate_id": config["candidate_id"],
                    "model_seed": seed, "status": "FAIL",
                    "error_type": type(error).__name__, "error": str(error),
                }
                atomic_json(run_dir / "failure.json", failure)
                rows.append(failure)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / (str(config["candidate_id"]) + "_manifest.json"), {
        "config": config, "config_sha256": canonical_sha256(config),
        "datasets": datasets, "seeds": seeds, "runs": rows,
        "failures_preserved": True,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    protocol = sub.add_parser("protocol")
    protocol.add_argument("--output", default=str(ROOT / "stage0"))
    cheap = sub.add_parser("cheap-search")
    cheap.add_argument("--output", default=str(ROOT / "stage1"))
    train = sub.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--datasets", required=True)
    train.add_argument("--seeds", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--warmstart-night14a", action="store_true")
    train.add_argument("--fresh-reload", action="store_true")
    reload_parser = sub.add_parser("reload")
    reload_parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    if args.mode == "protocol":
        stage0_protocol(Path(args.output))
    elif args.mode == "cheap-search":
        cheap_search(Path(args.output))
    elif args.mode == "reload":
        reload_run(Path(args.run_dir))
    else:
        train_batch(args)


if __name__ == "__main__":
    main()
