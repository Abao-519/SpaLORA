#!/usr/bin/env python3
"""Night-14A train/evaluate runner with matched common endpoints.

Labels are loaded only by the evaluator inherited from Night-13B.  Model
forward, loss, gradients, optimizer updates, checkpoints, and TCF filtering
receive no label-bearing object.
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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/night13b"))

import night13b_run as n13b  # noqa: E402
from SpaLORA.night13c_core import consensus_medoid  # noqa: E402
from SpaLORA.night14a_tcf import (  # noqa: E402
    UnifiedGraphAutoencoder, apply_tcf, canonical_sha256, loss_components,
    seed_everything, state_sha256, unsupervised_loss,
)


ROOT = Path("/root/autodl-fs/night14a_topology_conflict_sprint_20260823")
DEVELOPMENT = ("A1", "tonsil_s1", "P22", "MISAR_E15_5_S1")
CONFIRMATION = ("D1", "tonsil_s2", "tonsil_s3")
ALL_DATASETS = DEVELOPMENT + CONFIRMATION
DEFAULT_ENDPOINT_SEEDS = tuple(range(20))
REFERENCE_SELECTION = {
    "A1": "C00_G04_MODEL_SEED0",
    "tonsil_s1": "SIMPLE_CONCAT",
    "P22": "N02_HIER_MODEL_SEED0",
    "MISAR_E15_5_S1": "SIMPLE_CONCAT",
    "D1": "SIMPLE_CONCAT",
    "tonsil_s2": "RNA_ONLY",
    "tonsil_s3": "SIMPLE_CONCAT",
}
NATIVE_CONTEXT = [
    {"dataset": "A1", "method": "C00_H05_NATIVE", "ari": 0.2692,
     "nmi": 0.4087, "semantic_lane": "NATIVE_FULL_PIPELINE_CONTEXT"},
    {"dataset": "D1", "method": "C00_H05_NATIVE", "ari": 0.2412,
     "nmi": 0.3777, "semantic_lane": "NATIVE_FULL_PIPELINE_CONTEXT"},
    {"dataset": "P22", "method": "F00_NATIVE", "ari": 0.4677,
     "nmi": 0.6334, "semantic_lane": "NATIVE_FULL_PIPELINE_CONTEXT"},
    {"dataset": "P22", "method": "N02_NATIVE", "ari": 0.5063,
     "nmi": 0.6562, "semantic_lane": "NATIVE_FULL_PIPELINE_CONTEXT"},
]


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
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


def array_sha256(value: np.ndarray) -> str:
    value = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_config(config: Mapping[str, object]) -> None:
    required = {
        "candidate_id", "backbone", "hidden_dim", "latent_dim", "depth",
        "dropout", "graph_k", "steps", "learning_rate", "weight_decay",
        "gradient_clip", "loss_weights",
    }
    if set(config) != required:
        raise ValueError("training config schema mismatch")
    if str(config["backbone"]) not in UnifiedGraphAutoencoder.BACKBONES:
        raise ValueError("unregistered backbone")
    if min(int(config[x]) for x in ("hidden_dim", "latent_dim", "depth",
                                    "graph_k", "steps")) <= 0:
        raise ValueError("nonpositive training configuration")
    expected_weights = {
        "private_recon", "cross_recon", "alignment", "graph_smooth",
        "topology_agreement", "variance", "covariance",
    }
    if set(config["loss_weights"]) != expected_weights:
        raise ValueError("loss weight schema mismatch")


def validate_filter(config: Mapping[str, object]) -> None:
    required = {
        "filter_id", "variant", "support_scale", "support_center",
        "conflict_scale", "max_low", "max_high", "node_scale",
        "node_center", "high_center", "global_scale", "global_center",
        "roughness_scale", "roughness_center",
    }
    optional = {
        "global_gate_floor", "trust_evidence_fraction",
        "integrity_scale", "integrity_center",
    }
    keys = set(config)
    if not required.issubset(keys) or (keys - required - optional):
        raise ValueError("filter config schema mismatch")
    if str(config["variant"]) not in {
        "IDENTITY", "FIXED_LOW", "SUPPORT_LOW", "TCF_LOW_HIGH"
    }:
        raise ValueError("unregistered filter variant")


def scipy_graph_to_torch(graph: sp.spmatrix,
                         device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    coo = graph.tocoo()
    index = torch.as_tensor(
        np.vstack([coo.row, coo.col]), dtype=torch.long, device=device
    )
    weight = torch.as_tensor(coo.data, dtype=torch.float32, device=device)
    return index, weight


def make_graph(payload: Mapping[str, object], k: int) -> sp.csr_matrix:
    return n13b.row_stochastic(
        n13b.sparse_spatial_graph(payload["coordinates"], payload["ids"], k=int(k))
    )


def prepare_inputs(payload: Mapping[str, object]) -> Tuple[np.ndarray, np.ndarray]:
    x1 = StandardScaler().fit_transform(
        np.asarray(payload["view1"], dtype=np.float64)
    ).astype(np.float32)
    x2 = StandardScaler().fit_transform(
        np.asarray(payload["view2"], dtype=np.float64)
    ).astype(np.float32)
    if not np.all(np.isfinite(x1)) or not np.all(np.isfinite(x2)):
        raise RuntimeError("non-finite registered input")
    return x1, x2


def reorder(ids: Sequence[str], target: Sequence[str], embedding: np.ndarray) -> np.ndarray:
    ids = list(map(str, ids))
    target = list(map(str, target))
    if len(set(ids)) != len(ids) or set(ids) != set(target):
        raise RuntimeError("strong-reference observation identity mismatch")
    lookup = {value: index for index, value in enumerate(ids)}
    return np.asarray(embedding, dtype=np.float32)[[lookup[x] for x in target]]


def strong_reference_embedding(dataset: str, method: str,
                               payload: Mapping[str, object]) -> np.ndarray:
    if method == "SIMPLE_CONCAT":
        return np.asarray(payload["embedding"], dtype=np.float32)
    if method == "RNA_ONLY":
        value = StandardScaler().fit_transform(
            np.asarray(payload["view1"], dtype=np.float64)
        )
        components = max(1, min(64, value.shape[0] - 1, value.shape[1]))
        if components < value.shape[1]:
            value = PCA(n_components=components, random_state=0).fit_transform(value)
        return np.asarray(value, dtype=np.float32)
    if method == "C00_G04_MODEL_SEED0":
        runs = {
            "A1": Path("/root/autodl-fs/night6c_raw_runs_20260817/r1/"
                       "G04_SP10_F10_EUC_UNION/a1/seed_0/attempt_001"),
            "D1": Path("/root/autodl-fs/night6d_raw_runs_20260817/"
                       "G04_SP10_F10_EUC_UNION/d1/seed_0/attempt_001"),
            "tonsil_s1": Path("/root/autodl-fs/night6c_raw_runs_20260817/r1/"
                              "G04_SP10_F10_EUC_UNION/tonsil/seed_0/attempt_001"),
        }
        run = runs[dataset]
        archive = np.load(run / "views.npz", allow_pickle=False)
        ids = pd.read_csv(run / "observation_ids.csv").iloc[:, 0].astype(str)
        return reorder(ids, payload["ids"], archive["SpaLORA_fused"])
    if method == "F00_R02_MODEL_SEED0":
        path = Path("/root/autodl-fs/night7b_score_rnd_20260818/adapter_stage/"
                    "R1/formal/R02/u020/attempt_001/worker/embedding.npy")
        ids = Path("/root/autodl-fs/night7b_score_rnd_20260818/source/u020/"
                   "observation_ids.txt").read_text(encoding="utf-8").splitlines()
        return reorder(ids, payload["ids"], np.load(path, allow_pickle=False))
    if method == "N02_HIER_MODEL_SEED0":
        run = Path("/root/autodl-fs/night9b_racf_20260820/r1/r1-u009/attempt_001")
        resolved = read_json(run / "resolved_config.json")
        ids = pd.read_csv(resolved["observation_ids_path"]).iloc[:, 0].astype(str)
        embedding = np.load(run / "views.npz", allow_pickle=False)["SpaLORA_fused"]
        return reorder(ids, payload["ids"], embedding)
    raise ValueError("unknown reference method")


def endpoint_rows(dataset: str, method: str, model_seed: Optional[int],
                  embedding: np.ndarray, payload: Mapping[str, object],
                  endpoint_seeds: Sequence[int], n_init: int,
                  phase: str, candidate_id: Optional[str] = None,
                  filter_id: Optional[str] = None) -> Tuple[List[dict], dict, np.ndarray]:
    rows: List[dict] = []
    partitions: List[np.ndarray] = []
    embedding = np.asarray(embedding, dtype=np.float32)
    embedding_sha = array_sha256(embedding)
    for endpoint_seed in endpoint_seeds:
        started = time.perf_counter()
        partition = KMeans(
            n_clusters=int(payload["k"]), random_state=int(endpoint_seed),
            n_init=int(n_init),
        ).fit_predict(embedding).astype(np.int64)
        partitions.append(partition)
        metrics = n13b.partition_metrics(
            payload["labels"], payload["label_mask"], partition,
            payload["metric_graph"],
        )
        rows.append({
            "dataset": dataset, "phase": phase, "method": method,
            "candidate_id": candidate_id, "filter_id": filter_id,
            "model_seed": model_seed, "endpoint_seed": int(endpoint_seed),
            "endpoint_n_init": int(n_init), "k": int(payload["k"]),
            "total_observations": int(len(payload["ids"])),
            "evaluated_observations": int(np.sum(payload["label_mask"])),
            "ordered_id_sha256": n13b.ordered_id_sha256(payload["ids"]),
            "embedding_sha256": embedding_sha,
            "partition_sha256": array_sha256(partition),
            "endpoint_wall_seconds": float(time.perf_counter() - started),
            **metrics,
        })
    consensus, medoid_seed, agreement = consensus_medoid(partitions)
    consensus_metrics = n13b.partition_metrics(
        payload["labels"], payload["label_mask"], consensus,
        payload["metric_graph"],
    )
    summary = {
        "dataset": dataset, "phase": phase, "method": method,
        "candidate_id": candidate_id, "filter_id": filter_id,
        "model_seed": model_seed, "endpoint_seed_count": len(endpoint_seeds),
        "endpoint_n_init": int(n_init), "embedding_sha256": embedding_sha,
        "ari_mean": float(np.mean([x["absolute_ari"] for x in rows])),
        "ari_sd": (float(np.std([x["absolute_ari"] for x in rows], ddof=1))
                   if len(rows) > 1 else 0.0),
        "ari_min": float(np.min([x["absolute_ari"] for x in rows])),
        "nmi_mean": float(np.mean([x["absolute_nmi"] for x in rows])),
        "nmi_sd": (float(np.std([x["absolute_nmi"] for x in rows], ddof=1))
                   if len(rows) > 1 else 0.0),
        "nmi_min": float(np.min([x["absolute_nmi"] for x in rows])),
        "ami_mean": float(np.mean([x["ami"] for x in rows])),
        "fmi_mean": float(np.mean([x["fmi"] for x in rows])),
        "morans_i_mean": float(np.mean([x["morans_i"] for x in rows])),
        "gearys_c_mean": float(np.mean([x["gearys_c"] for x in rows])),
        "consensus_method": "PAIRWISE_ARI_MEDOID",
        "consensus_medoid_seed": int(medoid_seed),
        "consensus_mean_pairwise_ari": float(agreement),
        "consensus_partition_sha256": array_sha256(consensus),
        "consensus_ari": consensus_metrics["absolute_ari"],
        "consensus_nmi": consensus_metrics["absolute_nmi"],
    }
    return rows, summary, consensus


def retag_reused_endpoint(rows: Sequence[Mapping[str, object]],
                          summary: Mapping[str, object],
                          filter_id: str,
                          source_filter_id: str) -> Tuple[List[dict], dict]:
    """Reuse one endpoint result for a byte-identical embedding.

    Re-running multithreaded KMeans on the same bytes can differ at numerical
    tie boundaries.  Reuse makes the identity contract semantic rather than a
    claim about an implementation's thread scheduling.
    """
    reused_rows: List[dict] = []
    for old in rows:
        value = dict(old)
        value["filter_id"] = filter_id
        value["endpoint_reused"] = True
        value["endpoint_reuse_of_filter_id"] = source_filter_id
        value["source_endpoint_wall_seconds"] = value["endpoint_wall_seconds"]
        value["endpoint_wall_seconds"] = 0.0
        reused_rows.append(value)
    reused_summary = dict(summary)
    reused_summary["filter_id"] = filter_id
    reused_summary["endpoint_reused"] = True
    reused_summary["endpoint_reuse_of_filter_id"] = source_filter_id
    return reused_rows, reused_summary


def roundtrip_status(rows: Sequence[Mapping[str, object]]) -> dict:
    """Separate numerical checkpoint replay from non-primary ablation ties."""
    required = {"W00_IDENTITY", "W02_TCF_FINAL"}
    required_rows = [row for row in rows if row.get("object") in required]
    return {
        "all_embedding_numerical_roundtrip": all(
            bool(row["numerical_roundtrip"]) for row in rows
        ),
        "all_filter_partition_exact": all(
            bool(row.get("partition_exact", True)) for row in rows
        ),
        "required_partition_objects": sorted(required),
        "required_filter_partition_exact": (
            {str(row.get("object")) for row in required_rows} == required
            and all(bool(row.get("partition_exact", False)) for row in required_rows)
        ),
    }


def cached_partition(embedding: np.ndarray, k: int, seed: int, n_init: int,
                     cache: Dict[str, Tuple[np.ndarray, str]],
                     object_name: str) -> Tuple[np.ndarray, bool, Optional[str]]:
    """Cluster each byte-identical embedding only once within an audit."""
    embedding_sha = array_sha256(np.asarray(embedding, dtype=np.float32))
    if embedding_sha in cache:
        partition, source = cache[embedding_sha]
        return partition.copy(), True, source
    partition = KMeans(
        int(k), random_state=int(seed), n_init=int(n_init)
    ).fit_predict(embedding).astype(np.int64)
    cache[embedding_sha] = (partition.copy(), str(object_name))
    return partition, False, None


def run_references(datasets: Sequence[str], output: Path,
                   endpoint_seeds: Sequence[int], n_init: int) -> None:
    all_rows: List[dict] = []
    summaries: List[dict] = []
    artifacts: List[dict] = []
    for dataset in datasets:
        payload = n13b.base_payload(dataset)
        methods = ["SIMPLE_CONCAT", "RNA_ONLY"]
        if dataset in {"A1", "D1", "tonsil_s1"}:
            methods.append("C00_G04_MODEL_SEED0")
        if dataset == "P22":
            methods.extend(["F00_R02_MODEL_SEED0", "N02_HIER_MODEL_SEED0"])
        for method in methods:
            embedding = strong_reference_embedding(dataset, method, payload)
            rows, summary, _ = endpoint_rows(
                dataset, method, None, embedding, payload, endpoint_seeds,
                n_init, "REFERENCE",
            )
            all_rows.extend(rows)
            summaries.append(summary)
            artifacts.append({
                "dataset": dataset, "method": method,
                "embedding_shape": list(embedding.shape),
                "embedding_sha256": array_sha256(embedding),
                "ordered_id_sha256": n13b.ordered_id_sha256(payload["ids"]),
                "status": "PASS",
            })
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(output / "reference_endpoint_rows.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "reference_endpoint_summary.csv", index=False)
    pd.DataFrame(artifacts).to_csv(output / "reference_artifact_audit.csv", index=False)
    pd.DataFrame(NATIVE_CONTEXT).to_csv(output / "native_full_pipeline_context.csv", index=False)


def train_one(dataset: str, config: Mapping[str, object], model_seed: int,
              steps_override: Optional[int] = None) -> Tuple[dict, dict, dict]:
    validate_config(config)
    seed_everything(model_seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    payload = n13b.base_payload(dataset)
    x1_np, x2_np = prepare_inputs(payload)
    graph = make_graph(payload, int(config["graph_k"]))
    edge_index, edge_weight = scipy_graph_to_torch(graph, device)
    x1 = torch.as_tensor(x1_np, device=device)
    x2 = torch.as_tensor(x2_np, device=device)
    model = UnifiedGraphAutoencoder(x1.shape[1], x2.shape[1], config).to(device)
    initial_sha = state_sha256(model.state_dict())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    steps = int(config["steps"] if steps_override is None else steps_override)
    if steps <= 0:
        raise ValueError("optimizer steps must be positive")
    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    first_gradient_norm = None
    final_loss = None
    loss_trace: List[dict] = []
    for step in range(steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(x1, x2, edge_index, edge_weight)
        components = loss_components(output, x1, x2, edge_index)
        loss, component_audit = unsupervised_loss(
            model, components, config["loss_weights"]
        )
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite unsupervised loss")
        loss.backward()
        gradients = [p.grad.detach().square().sum() for p in model.parameters()
                     if p.grad is not None]
        norm = torch.sqrt(sum(gradients, torch.zeros((), device=device)))
        if not torch.isfinite(norm) or float(norm) <= 0:
            raise RuntimeError("non-finite or zero gradient")
        if first_gradient_norm is None:
            first_gradient_norm = float(norm.detach().cpu())
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip"]))
        optimizer.step()
        final_loss = component_audit
        if step in {0, steps - 1} or (step + 1) % max(1, steps // 10) == 0:
            loss_trace.append({"step": step + 1, **component_audit})
    gpu_seconds = 0.0
    if torch.cuda.is_available():
        end_event.record()
        torch.cuda.synchronize(device)
        gpu_seconds = float(start_event.elapsed_time(end_event) / 1000.0)
    model.eval()
    with torch.no_grad():
        output = model(x1, x2, edge_index, edge_weight)
    embeddings = {
        key: output[key].detach().cpu().numpy().astype(np.float32)
        for key in ("z1", "z2", "fused")
    }
    final_sha = state_sha256(model.state_dict())
    audit = {
        "dataset": dataset, "candidate_id": config["candidate_id"],
        "backbone": config["backbone"], "model_seed": int(model_seed),
        "optimizer_steps": steps,
        "trainable_parameter_count": int(sum(p.numel() for p in model.parameters()
                                             if p.requires_grad)),
        "initial_state_sha256": initial_sha, "final_state_sha256": final_sha,
        "parameters_changed": initial_sha != final_sha,
        "first_gradient_norm": first_gradient_norm, "final_loss": final_loss,
        "loss_trace": loss_trace,
        "raw_shapes": payload["raw_shapes"],
        "processed_shapes": [list(x1_np.shape), list(x2_np.shape)],
        "latent_shapes": {key: list(value.shape) for key, value in embeddings.items()},
        "total_observations": int(len(payload["ids"])),
        "evaluated_observations": int(np.sum(payload["label_mask"])),
        "ordered_id_sha256": n13b.ordered_id_sha256(payload["ids"]),
        "sparse_graph_nnz": int(graph.nnz), "dense_n_by_n_count": 0,
        "labels_in_loss_gradient_or_checkpoint_selection": False,
        "dataset_name_routing": False,
        "wall_seconds": float(time.perf_counter() - started),
        "gpu_seconds": gpu_seconds,
        "peak_gpu_mib": float(torch.cuda.max_memory_allocated() / 1048576.0)
            if torch.cuda.is_available() else 0.0,
        "peak_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
        "embedding_sha256": {key: array_sha256(value)
                              for key, value in embeddings.items()},
    }
    run_objects = {
        "model": model, "x1": x1_np, "x2": x2_np,
        "edge_index": edge_index.detach().cpu().numpy(),
        "edge_weight": edge_weight.detach().cpu().numpy(),
        "embeddings": embeddings,
    }
    return payload, audit, run_objects


def filter_embeddings(run_objects: Mapping[str, object],
                      filters: Sequence[Mapping[str, object]],
                      device: torch.device) -> Tuple[Dict[str, np.ndarray], Dict[str, dict]]:
    z1 = torch.as_tensor(run_objects["embeddings"]["z1"], device=device)
    z2 = torch.as_tensor(run_objects["embeddings"]["z2"], device=device)
    fused = torch.as_tensor(run_objects["embeddings"]["fused"], device=device)
    edge_index = torch.as_tensor(run_objects["edge_index"], dtype=torch.long, device=device)
    values: Dict[str, np.ndarray] = {}
    audits: Dict[str, dict] = {}
    for config in filters:
        validate_filter(config)
        output, audit = apply_tcf(z1, z2, fused, edge_index, config)
        value = output.detach().cpu().numpy().astype(np.float32)
        values[str(config["filter_id"])] = value
        audits[str(config["filter_id"])] = {
            **audit, "filter_config_sha256": canonical_sha256(config),
            "embedding_sha256": array_sha256(value),
        }
    return values, audits


def save_and_evaluate(run_dir: Path, payload: Mapping[str, object],
                      config: Mapping[str, object], filters: Sequence[Mapping[str, object]],
                      audit: dict, run_objects: Mapping[str, object], phase: str,
                      endpoint_seeds: Sequence[int], n_init: int,
                      fresh_reload: bool) -> Tuple[List[dict], List[dict]]:
    run_dir.mkdir(parents=True, exist_ok=True)
    model = run_objects["model"]
    checkpoint = run_dir / "checkpoint.pt"
    torch.save({
        "state_dict": {key: value.detach().cpu()
                       for key, value in model.state_dict().items()},
        "config": dict(config), "config_sha256": canonical_sha256(config),
        "input1": int(run_objects["x1"].shape[1]),
        "input2": int(run_objects["x2"].shape[1]),
    }, checkpoint)
    np.savez_compressed(
        run_dir / "roundtrip_input.npz", x1=run_objects["x1"],
        x2=run_objects["x2"], edge_index=run_objects["edge_index"],
        edge_weight=run_objects["edge_weight"],
    )
    device = next(model.parameters()).device
    filtered, filter_audits = filter_embeddings(run_objects, filters, device)
    expected = {"z1": run_objects["embeddings"]["z1"],
                "z2": run_objects["embeddings"]["z2"],
                "fused": run_objects["embeddings"]["fused"], **filtered}
    np.savez_compressed(run_dir / "roundtrip_expected.npz", **expected)
    rows: List[dict] = []
    summaries: List[dict] = []
    expected_partitions: Dict[str, np.ndarray] = {}
    endpoint_cache: Dict[str, Tuple[List[dict], dict, str]] = {}
    first_partition_cache: Dict[str, np.ndarray] = {}
    for filter_id, embedding in filtered.items():
        embedding_sha = array_sha256(embedding)
        if embedding_sha in endpoint_cache:
            old_endpoint, old_summary, source_filter_id = endpoint_cache[embedding_sha]
            endpoint, summary = retag_reused_endpoint(
                old_endpoint, old_summary, filter_id, source_filter_id
            )
        else:
            endpoint, summary, _ = endpoint_rows(
                str(audit["dataset"]), "NIGHT14A_UNIFIED_TCF", int(audit["model_seed"]),
                embedding, payload, endpoint_seeds, n_init, phase,
                str(config["candidate_id"]), filter_id,
            )
            summary["endpoint_reused"] = False
            summary["endpoint_reuse_of_filter_id"] = None
            for value in endpoint:
                value["endpoint_reused"] = False
                value["endpoint_reuse_of_filter_id"] = None
                value["source_endpoint_wall_seconds"] = value["endpoint_wall_seconds"]
            endpoint_cache[embedding_sha] = (
                [dict(value) for value in endpoint], dict(summary), filter_id
            )
        rows.extend(endpoint)
        summaries.append({**summary, **filter_audits[filter_id]})
        if embedding_sha not in first_partition_cache:
            first_partition_cache[embedding_sha] = KMeans(
                int(payload["k"]), random_state=int(endpoint_seeds[0]), n_init=int(n_init)
            ).fit_predict(embedding).astype(np.int64)
        expected_partitions[filter_id] = first_partition_cache[embedding_sha].copy()
    np.savez_compressed(run_dir / "expected_partitions.npz", **expected_partitions)
    audit.update({
        "checkpoint_sha256": file_sha256(checkpoint),
        "config_sha256": canonical_sha256(config),
        "filter_audits": filter_audits,
        "offline_filter_count": len(filters),
        "single_frozen_checkpoint_for_filter_grid": True,
        "unique_endpoint_embedding_count": len(endpoint_cache),
        "byte_identical_endpoint_reuse_count": len(filters) - len(endpoint_cache),
    })
    atomic_json(run_dir / "training_audit.json", audit)
    pd.DataFrame(rows).to_csv(run_dir / "endpoint_rows.csv", index=False)
    pd.DataFrame(summaries).to_csv(run_dir / "endpoint_summary.csv", index=False)
    atomic_json(run_dir / "resolved_filters.json", list(filters))
    if fresh_reload:
        subprocess.run([
            sys.executable, str(Path(__file__).resolve()), "reload",
            "--run-dir", str(run_dir), "--n-init", str(n_init),
            "--endpoint-first-seed", str(endpoint_seeds[0]),
            "--k", str(payload["k"]),
        ], cwd=str(REPO), check=True)
        reload_audit = read_json(run_dir / "fresh_process_reload.json")
        audit["fresh_process_reload"] = reload_audit
        audit["status"] = reload_audit["status"]
    else:
        audit["status"] = "PASS"
    atomic_json(run_dir / "training_audit.json", audit)
    return rows, summaries


def reload_run(run_dir: Path, n_init: int, endpoint_first_seed: int, k: int) -> None:
    checkpoint = torch.load(run_dir / "checkpoint.pt", map_location="cpu")
    if canonical_sha256(checkpoint["config"]) != checkpoint["config_sha256"]:
        raise RuntimeError("checkpoint config hash mismatch")
    config = checkpoint["config"]
    values = np.load(run_dir / "roundtrip_input.npz", allow_pickle=False)
    expected = np.load(run_dir / "roundtrip_expected.npz", allow_pickle=False)
    filters = read_json(run_dir / "resolved_filters.json")
    model = UnifiedGraphAutoencoder(
        int(checkpoint["input1"]), int(checkpoint["input2"]), config
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    x1 = torch.as_tensor(values["x1"], dtype=torch.float32)
    x2 = torch.as_tensor(values["x2"], dtype=torch.float32)
    edge_index = torch.as_tensor(values["edge_index"], dtype=torch.long)
    edge_weight = torch.as_tensor(values["edge_weight"], dtype=torch.float32)
    with torch.no_grad():
        output = model(x1, x2, edge_index, edge_weight)
    run_objects = {
        "embeddings": {key: output[key].numpy().astype(np.float32)
                       for key in ("z1", "z2", "fused")},
        "edge_index": values["edge_index"],
    }
    filtered, _ = filter_embeddings(run_objects, filters, torch.device("cpu"))
    actual = {**run_objects["embeddings"], **filtered}
    partitions = np.load(run_dir / "expected_partitions.npz", allow_pickle=False)
    rows = []
    reload_partition_cache: Dict[str, Tuple[np.ndarray, str]] = {}
    for key in expected.files:
        maximum = float(np.max(np.abs(actual[key] - expected[key])))
        row = {"object": key, "max_abs_difference": maximum,
               "numerical_roundtrip": bool(maximum <= 1e-5),
               "byte_exact": bool(np.array_equal(actual[key], expected[key]))}
        if key in filtered:
            observed, reused, source = cached_partition(
                actual[key], int(k), int(endpoint_first_seed), int(n_init),
                reload_partition_cache, key,
            )
            row["partition_exact"] = bool(np.array_equal(observed, partitions[key]))
            row["partition_reused"] = reused
            row["partition_reuse_of_object"] = source
        rows.append(row)
    status_fields = roundtrip_status(rows)
    audit = {
        "fresh_process": True, "strict_state_load": True,
        "final_state_sha256": state_sha256(model.state_dict()),
        "rows": rows,
        **status_fields,
        "nonprimary_ablation_partition_difference_is_status_blocking": False,
    }
    audit["status"] = "PASS" if (
        audit["all_embedding_numerical_roundtrip"]
        and audit["required_filter_partition_exact"]
    ) else "FAIL"
    atomic_json(run_dir / "fresh_process_reload.json", audit)


def run_batch(args: argparse.Namespace) -> None:
    config = read_json(Path(args.config))
    validate_config(config)
    filters = read_json(Path(args.filters))
    for value in filters:
        validate_filter(value)
    datasets = tuple(x for x in args.datasets.split(",") if x)
    seeds = tuple(int(x) for x in args.seeds.split(",") if x)
    endpoint_seeds = tuple(range(int(args.endpoint_seed_count)))
    output = Path(args.output)
    all_rows: List[dict] = []
    all_summaries: List[dict] = []
    manifests: List[dict] = []
    for dataset in datasets:
        for seed in seeds:
            payload, audit, objects = train_one(
                dataset, config, seed, args.steps_override
            )
            run_dir = output / str(config["candidate_id"]) / dataset / ("seed_%d" % seed)
            rows, summaries = save_and_evaluate(
                run_dir, payload, config, filters, audit, objects, args.phase,
                endpoint_seeds, args.n_init, args.fresh_reload,
            )
            all_rows.extend(rows)
            all_summaries.extend(summaries)
            manifests.append(read_json(run_dir / "training_audit.json"))
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(
        output / (str(config["candidate_id"]) + "_endpoint_rows.csv"), index=False
    )
    pd.DataFrame(all_summaries).to_csv(
        output / (str(config["candidate_id"]) + "_endpoint_summary.csv"), index=False
    )
    atomic_json(output / (str(config["candidate_id"]) + "_manifest.json"), {
        "candidate": config, "config_sha256": canonical_sha256(config),
        "filters": filters, "filters_sha256": canonical_sha256(filters),
        "datasets": list(datasets), "model_seeds": list(seeds),
        "endpoint_seeds": list(endpoint_seeds), "endpoint_n_init": args.n_init,
        "run_count": len(manifests),
        "all_status_pass": all(x.get("status") == "PASS" for x in manifests),
        "labels_in_loss_gradient_or_checkpoint_selection": False,
        "dataset_name_routing": False, "dense_n_by_n_count": 0,
        "runs": manifests,
    })


def preflight(output: Path) -> None:
    rows = []
    for dataset in ALL_DATASETS:
        started = time.perf_counter()
        payload = n13b.base_payload(dataset)
        x1, x2 = prepare_inputs(payload)
        graph = make_graph(payload, 8)
        rows.append({
            "dataset": dataset, "registered_adapter": n13b.DATASETS[dataset]["adapter"],
            "registered_phase_night13b": n13b.DATASETS[dataset]["phase"],
            "night14a_phase": "development" if dataset in DEVELOPMENT else "confirmation",
            "raw_shapes": payload["raw_shapes"],
            "processed_shapes": [list(x1.shape), list(x2.shape)],
            "simple_embedding_shape": list(payload["embedding"].shape),
            "total_observations": int(len(payload["ids"])),
            "evaluated_observations": int(np.sum(payload["label_mask"])),
            "ordered_id_sha256": n13b.ordered_id_sha256(payload["ids"]),
            "k": int(payload["k"]), "sparse_graph_nnz": int(graph.nnz),
            "dense_n_by_n_count": 0, "finite": bool(np.isfinite(x1).all()
                                                       and np.isfinite(x2).all()),
            "load_seconds": float(time.perf_counter() - started),
        })
    atomic_json(output, {"passed": all(x["finite"] for x in rows), "rows": rows})


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("preflight")
    p.add_argument("--output", default=str(ROOT / "p0/real_input_preflight.json"))
    reference = sub.add_parser("references")
    reference.add_argument("--datasets", default=",".join(ALL_DATASETS))
    reference.add_argument("--output", default=str(ROOT / "references"))
    reference.add_argument("--endpoint-seed-count", type=int, default=20)
    reference.add_argument("--n-init", type=int, default=20)
    run = sub.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--filters", required=True)
    run.add_argument("--datasets", required=True)
    run.add_argument("--seeds", required=True)
    run.add_argument("--phase", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--steps-override", type=int)
    run.add_argument("--endpoint-seed-count", type=int, default=20)
    run.add_argument("--n-init", type=int, default=20)
    run.add_argument("--fresh-reload", action="store_true")
    reload_parser = sub.add_parser("reload")
    reload_parser.add_argument("--run-dir", required=True)
    reload_parser.add_argument("--n-init", type=int, required=True)
    reload_parser.add_argument("--endpoint-first-seed", type=int, required=True)
    reload_parser.add_argument("--k", type=int, required=True)
    args = parser.parse_args()
    if args.mode == "preflight":
        preflight(Path(args.output))
    elif args.mode == "references":
        run_references(tuple(x for x in args.datasets.split(",") if x),
                       Path(args.output), tuple(range(args.endpoint_seed_count)),
                       args.n_init)
    elif args.mode == "reload":
        reload_run(Path(args.run_dir), args.n_init, args.endpoint_first_seed, args.k)
    else:
        run_batch(args)


if __name__ == "__main__":
    main()
