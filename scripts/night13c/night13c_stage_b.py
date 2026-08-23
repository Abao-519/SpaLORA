#!/usr/bin/env python3
"""Train, reload and evaluate Night-13C unified-core candidates."""
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
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/night13b"))

import night13b_run as n13b  # noqa: E402
from SpaLORA.night13b_unified import model_state_sha256  # noqa: E402
from SpaLORA.night13c_core import (  # noqa: E402
    TrainableUnifiedCore, canonical_sha256, consensus_medoid, seed_everything,
)

ROOT = Path("/root/autodl-fs/night13c_endpoint_robustness_trainable_core_20260823")
OUT = ROOT / "stage_b"

REFERENCES = {
    "A1": ("C00_G04_COMMON_BRIDGE", 0.2304603078741277, 0.3760111948360975),
    "tonsil_s1": ("SIMPLE_CONCAT", 0.1418066616082619, 0.2842230497175844),
    "P22": ("N02_HIER_ONLY_COMMON_BRIDGE", 0.420270597573084, 0.5771380492633564),
    "MISAR_E15_5_S1": ("SIMPLE_CONCAT", 0.2021466839622214, 0.3561877505161702),
    "D1": ("SIMPLE_CONCAT", 0.2427545621080531, 0.3630270844289118),
    "tonsil_s2": ("RNA_ONLY", 0.1817289187343056, 0.26201392999773837),
    "tonsil_s3": ("SIMPLE_CONCAT", 0.1969333595315539, 0.2497791360710971),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False,
                  allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(tmp), str(path))


def read_config(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {"candidate_id", "mechanism", "latent", "steps", "learning_rate",
                "weight_decay", "graph_residual", "final_residual", "anchor_weight"}
    if set(value) != required:
        raise ValueError("candidate config schema mismatch")
    if value["mechanism"] not in TrainableUnifiedCore.MECHANISMS:
        raise ValueError("candidate mechanism mismatch")
    if int(value["steps"]) <= 0:
        raise ValueError("optimizer steps must be positive")
    return value


def pad_anchor(value: np.ndarray, latent: int) -> np.ndarray:
    source = np.asarray(value, dtype=np.float32)
    if source.shape[1] > latent:
        return source[:, :latent].copy()
    if source.shape[1] == latent:
        return source.copy()
    result = np.zeros((source.shape[0], latent), dtype=np.float32)
    result[:, :source.shape[1]] = source
    return result


def prepare_dataset(name: str, latent: int) -> Tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    payload = n13b.base_payload(name)
    x1 = StandardScaler().fit_transform(np.asarray(payload["view1"], dtype=np.float64)).astype(np.float32)
    x2 = StandardScaler().fit_transform(np.asarray(payload["view2"], dtype=np.float64)).astype(np.float32)
    anchor = pad_anchor(np.asarray(payload["embedding"], dtype=np.float32), latent)
    anchor_target = StandardScaler().fit_transform(anchor.astype(np.float64)).astype(np.float32)
    rows, cols = payload["operators"][0].nonzero()
    keep = rows != cols
    edges = np.vstack([rows[keep], cols[keep]]).astype(np.int64)
    if edges.shape[1] == 0:
        raise RuntimeError("sparse graph has no non-self edges")
    return payload, x1, x2, anchor, anchor_target, edges


def align_to_anchor(learned: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    left = np.asarray(learned, dtype=np.float64)
    right = np.asarray(anchor, dtype=np.float64)
    left_center = left - left.mean(axis=0, keepdims=True)
    right_center = right - right.mean(axis=0, keepdims=True)
    u, _, vt = np.linalg.svd(left_center.T @ right_center, full_matrices=False)
    aligned = left_center @ (u @ vt)
    source_std = aligned.std(axis=0, ddof=0)
    target_std = right_center.std(axis=0, ddof=0)
    scale = np.divide(target_std, source_std, out=np.ones_like(target_std),
                      where=source_std > 1e-8)
    aligned = aligned * scale + right.mean(axis=0, keepdims=True)
    return aligned.astype(np.float32)


def final_embedding(model: TrainableUnifiedCore, x1: torch.Tensor, x2: torch.Tensor,
                    edges: torch.Tensor, anchor: np.ndarray, residual: float) -> Tuple[np.ndarray, dict]:
    model.eval()
    with torch.no_grad():
        output = model(x1, x2, edges)
    learned = output["fused"].detach().cpu().numpy().astype(np.float32)
    graph_anchor = np.asarray(anchor, dtype=np.float32)
    if model.mechanism == "EDGE_RELIABILITY":
        anchor_tensor = torch.as_tensor(anchor, dtype=torch.float32, device=x1.device)
        propagated = model._propagate(anchor_tensor, edges, output["edge_reliability"])
        beta = model.residual * output["node_reliability"][:, None]
        graph_anchor = ((1.0 - beta) * anchor_tensor + beta * propagated).detach().cpu().numpy().astype(np.float32)
    aligned = align_to_anchor(learned, graph_anchor)
    fused = ((1.0 - float(residual)) * graph_anchor + float(residual) * aligned).astype(np.float32)
    diagnostics = {
        "learned_embedding_sha256": n13b.array_sha256(learned),
        "aligned_embedding_sha256": n13b.array_sha256(aligned),
        "graph_anchor_embedding_sha256": n13b.array_sha256(graph_anchor),
        "final_embedding_sha256": n13b.array_sha256(fused),
        "edge_reliability_mean": None if output["edge_reliability"].numel() == 0 else
            float(output["edge_reliability"].mean().cpu()),
        "edge_reliability_sd": None if output["edge_reliability"].numel() == 0 else
            float(output["edge_reliability"].std().cpu()),
    }
    return fused, diagnostics


def train(name: str, config: Mapping[str, object], seed: int,
          steps_override: int | None = None) -> Tuple[dict, dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seed_everything(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    payload, x1_np, x2_np, anchor, anchor_target_np, edges_np = prepare_dataset(
        name, int(config["latent"]))
    x1 = torch.as_tensor(x1_np, device=device)
    x2 = torch.as_tensor(x2_np, device=device)
    anchor_target = torch.as_tensor(anchor_target_np, device=device)
    edges = torch.as_tensor(edges_np, dtype=torch.long, device=device)
    model = TrainableUnifiedCore(x1.shape[1], x2.shape[1], int(config["latent"]),
                                 str(config["mechanism"]),
                                 residual=float(config["graph_residual"])).to(device)
    initial_state = model_state_sha256(model.state_dict())
    parameter_count = int(sum(value.numel() for value in model.parameters() if value.requires_grad))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]),
                                  weight_decay=float(config["weight_decay"]))
    steps = int(config["steps"] if steps_override is None else steps_override)
    if steps <= 0:
        raise ValueError("nonpositive optimizer steps")
    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        begin_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        begin_event.record()
    final_components: Dict[str, float] = {}
    first_gradient_norm = None
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        dropout_mask = (False, False)
        if config["mechanism"] == "MODALITY_DROPOUT_CROSS_RECON":
            drop_first = bool(torch.rand((), device=device) < 0.5)
            dropout_mask = (drop_first, not drop_first)
        output = model(x1, x2, edges, dropout_mask=dropout_mask)
        loss, components = model.loss(output, x1, x2)
        anchor_loss = torch.nn.functional.mse_loss(output["fused"], anchor_target)
        loss = loss + float(config["anchor_weight"]) * anchor_loss
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite training loss")
        loss.backward()
        norm = torch.sqrt(sum((p.grad.detach().square().sum() for p in model.parameters()
                               if p.grad is not None), torch.zeros((), device=device)))
        if not torch.isfinite(norm) or norm <= 0:
            raise RuntimeError("non-finite or zero gradient")
        if first_gradient_norm is None:
            first_gradient_norm = float(norm.detach().cpu())
        optimizer.step()
        final_components = {key: float(value.detach().cpu()) for key, value in components.items()}
        final_components.update({"anchor": float(anchor_loss.detach().cpu()),
                                 "total": float(loss.detach().cpu())})
    gpu_seconds = 0.0
    if torch.cuda.is_available():
        end_event.record()
        torch.cuda.synchronize(device)
        gpu_seconds = float(begin_event.elapsed_time(end_event) / 1000.0)
    fused, diagnostics = final_embedding(model, x1, x2, edges, anchor,
                                         float(config["final_residual"]))
    audit = {
        "dataset": name, "candidate_id": config["candidate_id"],
        "mechanism": config["mechanism"], "training_seed": int(seed),
        "optimizer_steps": steps, "trainable_parameter_count": parameter_count,
        "initial_state_sha256": initial_state,
        "final_state_sha256": model_state_sha256(model.state_dict()),
        "parameters_changed": initial_state != model_state_sha256(model.state_dict()),
        "first_gradient_norm": first_gradient_norm, "final_loss_components": final_components,
        "raw_shapes": payload["raw_shapes"],
        "processed_shapes": [list(x1_np.shape), list(x2_np.shape)],
        "anchor_shape": list(anchor.shape), "sparse_edge_count": int(edges_np.shape[1]),
        "total_observations": int(len(payload["ids"])),
        "evaluated_observations": int(np.sum(payload["label_mask"])),
        "ordered_id_sha256": n13b.ordered_id_sha256(payload["ids"]),
        "labels_in_loss_gradient_or_checkpoint_selection": False,
        "dataset_name_routing": False, "dense_n_by_n_count": 0,
        "wall_seconds": float(time.perf_counter() - started),
        "gpu_seconds": gpu_seconds,
        "peak_gpu_mib": float(torch.cuda.max_memory_allocated() / 1048576.0)
            if torch.cuda.is_available() else 0.0,
        "peak_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
        **diagnostics,
    }
    return payload, audit, fused, x1_np, x2_np, anchor, edges_np, model


def endpoint_rows(payload: Mapping[str, object], audit: Mapping[str, object],
                  fused: np.ndarray, count: int) -> Tuple[List[dict], dict, np.ndarray]:
    rows: List[dict] = []
    partitions: List[np.ndarray] = []
    reference_method, reference_ari, reference_nmi = REFERENCES[str(audit["dataset"])]
    for endpoint_seed in range(count):
        started = time.perf_counter()
        model = KMeans(int(payload["k"]), random_state=endpoint_seed, n_init=20)
        partition = model.fit_predict(fused).astype(np.int64)
        partitions.append(partition)
        metrics = n13b.partition_metrics(payload["labels"], payload["label_mask"],
                                          partition, payload["metric_graph"])
        rows.append({
            "dataset": audit["dataset"], "candidate_id": audit["candidate_id"],
            "mechanism": audit["mechanism"], "training_seed": audit["training_seed"],
            "endpoint_seed": endpoint_seed, "k": int(payload["k"]),
            "total_observations": audit["total_observations"],
            "evaluated_observations": audit["evaluated_observations"],
            "ordered_id_sha256": audit["ordered_id_sha256"],
            "embedding_sha256": audit["final_embedding_sha256"],
            "partition_sha256": n13b.array_sha256(partition),
            "reference_method": reference_method, "reference_ari": reference_ari,
            "reference_nmi": reference_nmi,
            "delta_ari": metrics["absolute_ari"] - reference_ari,
            "delta_nmi": metrics["absolute_nmi"] - reference_nmi,
            "win_both": bool(metrics["absolute_ari"] > reference_ari and
                             metrics["absolute_nmi"] > reference_nmi),
            "inertia": float(model.inertia_),
            "endpoint_wall_seconds": float(time.perf_counter() - started),
            **metrics,
        })
    consensus, medoid_seed, agreement = consensus_medoid(partitions)
    metrics = n13b.partition_metrics(payload["labels"], payload["label_mask"],
                                      consensus, payload["metric_graph"])
    summary = {
        "dataset": audit["dataset"], "candidate_id": audit["candidate_id"],
        "mechanism": audit["mechanism"], "training_seed": audit["training_seed"],
        "endpoint_seed_count": count,
        "ari_mean": float(np.mean([x["absolute_ari"] for x in rows])),
        "ari_sd": float(np.std([x["absolute_ari"] for x in rows], ddof=1)) if count > 1 else 0.0,
        "nmi_mean": float(np.mean([x["absolute_nmi"] for x in rows])),
        "nmi_sd": float(np.std([x["absolute_nmi"] for x in rows], ddof=1)) if count > 1 else 0.0,
        "delta_ari_mean": float(np.mean([x["delta_ari"] for x in rows])),
        "delta_nmi_mean": float(np.mean([x["delta_nmi"] for x in rows])),
        "win_both_rate": float(np.mean([x["win_both"] for x in rows])),
        "consensus_method": "PAIRWISE_ARI_MEDOID", "consensus_medoid_seed": medoid_seed,
        "consensus_mean_pairwise_ari": agreement,
        "consensus_partition_sha256": n13b.array_sha256(consensus),
        "consensus_ari": metrics["absolute_ari"], "consensus_nmi": metrics["absolute_nmi"],
        "consensus_ami": metrics["ami"], "consensus_fmi": metrics["fmi"],
        "consensus_morans_i": metrics["morans_i"], "consensus_gearys_c": metrics["gearys_c"],
        "consensus_delta_ari": metrics["absolute_ari"] - reference_ari,
        "consensus_delta_nmi": metrics["absolute_nmi"] - reference_nmi,
        "optimizer_steps": audit["optimizer_steps"],
        "trainable_parameter_count": audit["trainable_parameter_count"],
        "wall_seconds": audit["wall_seconds"], "gpu_seconds": audit["gpu_seconds"],
        "peak_gpu_mib": audit["peak_gpu_mib"], "peak_rss_mib": audit["peak_rss_mib"],
    }
    return rows, summary, consensus


def save_run(run_dir: Path, config: Mapping[str, object], payload: Mapping[str, object],
             audit: dict, fused: np.ndarray, x1: np.ndarray, x2: np.ndarray,
             anchor: np.ndarray, edges: np.ndarray, model: TrainableUnifiedCore,
             endpoint_count: int, do_reload: bool) -> Tuple[List[dict], dict]:
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "checkpoint.pt"
    torch.save({"state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "config": dict(config), "config_sha256": canonical_sha256(config),
                "input1": x1.shape[1], "input2": x2.shape[1]}, checkpoint)
    audit["checkpoint_sha256"] = file_sha256(checkpoint)
    np.savez_compressed(run_dir / "roundtrip_input.npz", x1=x1, x2=x2, anchor=anchor,
                        edges=edges, expected=fused)
    rows, summary, consensus = endpoint_rows(payload, audit, fused, endpoint_count)
    np.save(run_dir / "consensus_partition.npy", consensus, allow_pickle=False)
    seed0 = np.asarray([row for row in rows if row["endpoint_seed"] == 0][0]["partition_sha256"])
    expected_partition = KMeans(int(payload["k"]), random_state=0,
                                n_init=20).fit_predict(fused).astype(np.int64)
    np.save(run_dir / "expected_partition.npy", expected_partition, allow_pickle=False)
    atomic_json(run_dir / "training_audit.json", audit)
    pd.DataFrame(rows).to_csv(run_dir / "endpoint_metrics.csv", index=False)
    atomic_json(run_dir / "endpoint_summary.json", summary)
    if do_reload:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "reload",
                        "--run-dir", str(run_dir)], cwd=str(REPO), check=True)
        reload_audit = json.loads((run_dir / "fresh_process_reload.json").read_text(encoding="utf-8"))
        audit["fresh_process_reload"] = reload_audit
        audit["status"] = "PASS" if reload_audit["status"] == "PASS" else "FAIL"
        atomic_json(run_dir / "training_audit.json", audit)
    else:
        audit["status"] = "PASS"
    return rows, summary


def reload_run(run_dir: Path) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(run_dir / "checkpoint.pt", map_location="cpu")
    if canonical_sha256(checkpoint["config"]) != checkpoint["config_sha256"]:
        raise RuntimeError("checkpoint config hash mismatch")
    config = checkpoint["config"]
    values = np.load(run_dir / "roundtrip_input.npz", allow_pickle=False)
    model = TrainableUnifiedCore(int(checkpoint["input1"]), int(checkpoint["input2"]),
                                 int(config["latent"]), str(config["mechanism"]),
                                 residual=float(config["graph_residual"])).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    x1 = torch.as_tensor(values["x1"], dtype=torch.float32, device=device)
    x2 = torch.as_tensor(values["x2"], dtype=torch.float32, device=device)
    edges = torch.as_tensor(values["edges"], dtype=torch.long, device=device)
    fused, diagnostics = final_embedding(model, x1, x2, edges, values["anchor"],
                                         float(config["final_residual"]))
    difference = float(np.max(np.abs(fused - values["expected"])))
    expected_partition = np.load(run_dir / "expected_partition.npy", allow_pickle=False)
    partition = KMeans(len(np.unique(expected_partition)), random_state=0,
                       n_init=20).fit_predict(fused).astype(np.int64)
    audit = {
        "fresh_process": True, "strict_state_load": True,
        "embedding_max_abs_difference": difference,
        "embedding_numerical_roundtrip": bool(difference <= 1e-5),
        "partition_exact": bool(np.array_equal(partition, expected_partition)),
        "final_state_sha256": model_state_sha256(model.state_dict()),
        **diagnostics,
    }
    audit["status"] = "PASS" if (audit["embedding_numerical_roundtrip"] and
                                      audit["partition_exact"]) else "FAIL"
    atomic_json(run_dir / "fresh_process_reload.json", audit)


def run_batch(args: argparse.Namespace) -> None:
    config = read_config(Path(args.config))
    config_sha = canonical_sha256(config)
    datasets = [x for x in args.datasets.split(",") if x]
    seeds = [int(x) for x in args.seeds.split(",") if x]
    all_rows: List[dict] = []
    summaries: List[dict] = []
    audits: List[dict] = []
    output = Path(args.output)
    for dataset in datasets:
        for seed in seeds:
            payload, audit, fused, x1, x2, anchor, edges, model = train(
                dataset, config, seed, steps_override=args.steps_override)
            audit["config_sha256"] = config_sha
            run_dir = output / str(config["candidate_id"]) / dataset / f"seed_{seed}"
            rows, summary = save_run(run_dir, config, payload, audit, fused, x1, x2,
                                     anchor, edges, model, args.endpoint_seed_count,
                                     args.fresh_reload)
            all_rows.extend(rows)
            summaries.append(summary)
            audits.append(audit)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(output / f"{config['candidate_id']}_endpoint_metrics.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / f"{config['candidate_id']}_summary.csv", index=False)
    atomic_json(output / f"{config['candidate_id']}_run_manifest.json", {
        "candidate": config, "config_sha256": config_sha, "datasets": datasets,
        "seeds": seeds, "endpoint_seed_count": args.endpoint_seed_count,
        "run_count": len(audits), "all_status_pass": all(x.get("status") == "PASS" for x in audits),
        "optimizer_steps_positive": all(int(x["optimizer_steps"]) > 0 for x in audits),
        "parameters_changed": all(bool(x["parameters_changed"]) for x in audits),
        "unique_final_embedding_sha256": len(set(x["final_embedding_sha256"] for x in audits)),
        "audits": audits,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    run = sub.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--datasets", required=True)
    run.add_argument("--seeds", required=True)
    run.add_argument("--endpoint-seed-count", type=int, default=10)
    run.add_argument("--steps-override", type=int)
    run.add_argument("--output", required=True)
    run.add_argument("--fresh-reload", action="store_true")
    reload_parser = sub.add_parser("reload")
    reload_parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    if args.mode == "reload":
        reload_run(Path(args.run_dir))
    else:
        run_batch(args)


if __name__ == "__main__":
    main()
