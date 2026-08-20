#!/usr/bin/env python3
"""Train, transform and fresh-process reload one opaque Night-9B MF-RACF cell."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import resource
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from SpaLORA.night6c_pipeline import array_sha as n6_array_sha, run_head  # noqa: E402
from SpaLORA.night6d_pipeline import HEADS  # noqa: E402
from SpaLORA.night7a_consensus import canonical_partition  # noqa: E402
from SpaLORA.night9b_racf import (  # noqa: E402
    FORBIDDEN_KEYS, RACFModel, VIEW_KEYS, array_sha, canonical_json_sha,
    fixed_permutation, output_views, racf_loss, rna_common_graph, row_l2_np,
    spatial_operator, sparse_sha, tensor_state_sha, torch_sparse,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_npz(path: Path, **values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **values)
    os.replace(tmp, path)


def atomic_torch(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(value, tmp); os.replace(tmp, path)


def artifact(path: Path) -> dict:
    return {"path": str(path), "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def configure(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark = False


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text())
    forbidden = FORBIDDEN_KEYS & set(cfg)
    if forbidden:
        raise RuntimeError(f"identity/label keys rejected: {sorted(forbidden)}")
    required = {"schema_version", "unit_id", "stage", "candidate", "candidate_config_sha256",
                "seed", "K", "views_path", "reference_path", "coordinates_path",
                "observation_ids_path", "output_dir", "spatial_k", "epochs",
                "optimizer", "learning_rate", "weight_decay", "smoke"}
    if set(cfg) != required:
        raise RuntimeError(f"config key mismatch: missing={sorted(required-set(cfg))} extra={sorted(set(cfg)-required)}")
    if canonical_json_sha(cfg["candidate"]) != cfg["candidate_config_sha256"]:
        raise RuntimeError("candidate config SHA mismatch")
    return cfg


def load_inputs(cfg: dict) -> dict:
    ids_path = Path(cfg["observation_ids_path"])
    if ids_path.suffix.lower() == ".csv":
        with ids_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or set(rows[0]) != {"observation_id"}:
            raise RuntimeError("observation-id CSV schema mismatch")
        ids = [str(row["observation_id"]) for row in rows]
    else:
        ids = [x.strip() for x in ids_path.read_text().splitlines() if x.strip()]
    source = np.load(cfg["views_path"], allow_pickle=False)
    x_rna = row_l2_np(source["emb_latent_omics1"])
    x_aux = row_l2_np(source["emb_latent_omics2"])
    ref_path = Path(cfg["reference_path"])
    if ref_path.suffix == ".npy":
        reference = row_l2_np(np.load(ref_path, allow_pickle=False))
    else:
        z = np.load(ref_path, allow_pickle=False)
        reference = row_l2_np(z["SpaLORA_fused"])
    coords = np.asarray(np.load(cfg["coordinates_path"], allow_pickle=False), dtype=np.float64)
    if not (len(ids) == len(x_rna) == len(x_aux) == len(reference) == len(coords)):
        raise RuntimeError("opaque input row count mismatch")
    if x_rna.shape != x_aux.shape or reference.ndim != 2:
        raise RuntimeError("inherited family source/reference shape mismatch")
    return {"ids": ids, "x_rna": x_rna, "x_aux": x_aux,
            "reference": reference, "coords": coords}


def make_graphs(cfg: dict, inputs: dict) -> tuple[sp.csr_matrix, sp.csr_matrix | None, dict]:
    spatial = spatial_operator(inputs["coords"], inputs["ids"], int(cfg["spatial_k"]))
    common = None; common_audit = {"active": False, "union_fallback": False}
    k = cfg["candidate"]["common_graph_k"]
    if k is not None:
        common, common_audit = rna_common_graph(inputs["x_rna"], inputs["coords"], inputs["ids"], int(k))
        common_audit["active"] = True
    return spatial, common, {"spatial_operator_sha256": sparse_sha(spatial),
                             "common_graph": common_audit}


def optimizer_for(cfg: dict, model: torch.nn.Module):
    if cfg["optimizer"] == "Adam":
        return torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]),
                                weight_decay=float(cfg["weight_decay"]))
    if cfg["optimizer"] == "AdamW":
        return torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]),
                                 weight_decay=float(cfg["weight_decay"]))
    raise RuntimeError("unregistered inherited optimizer")


def evaluate_model(model: RACFModel, tensors: dict, permutation: torch.Tensor) -> tuple[dict, dict]:
    model.eval()
    with torch.no_grad():
        out = model(tensors["x_rna"], tensors["x_aux"], tensors["spatial"],
                    tensors["common"], tensors["reference"])
        _, losses = racf_loss(model, out, tensors["x_rna"], tensors["x_aux"],
                              tensors["reference"], permutation)
    return dict(output_views(out)), losses


def save_clusters(path: Path, ids: list[str], labels: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["observation_id", "cluster"])
        writer.writerows(zip(ids, map(int, labels)))
    os.replace(tmp, path)


def run_train(config_path: Path) -> None:
    cfg = load_config(config_path); outdir = Path(cfg["output_dir"])
    if outdir.exists() and any(outdir.iterdir()):
        raise RuntimeError("refusing to overwrite an existing attempt")
    outdir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter(); configure(int(cfg["seed"]))
    if not torch.cuda.is_available():
        raise RuntimeError("formal Night-9B training requires CUDA")
    device = torch.device("cuda"); torch.cuda.reset_peak_memory_stats(device)
    inputs = load_inputs(cfg); spatial, common, graph_audit = make_graphs(cfg, inputs)
    model = RACFModel(inputs["x_rna"].shape[1], cfg["candidate"],
                      latent_dim=inputs["reference"].shape[1]).to(device)
    optimizer = optimizer_for(cfg, model)
    tensors = {
        "x_rna": torch.as_tensor(inputs["x_rna"], device=device),
        "x_aux": torch.as_tensor(inputs["x_aux"], device=device),
        "reference": torch.as_tensor(inputs["reference"], device=device),
        "spatial": torch_sparse(spatial, device),
        "common": None if common is None else torch_sparse(common, device),
    }
    permutation_np = fixed_permutation(len(inputs["ids"]), int(cfg["seed"]))
    permutation = torch.as_tensor(permutation_np, device=device)
    trace = []; gradient = None
    train_started = time.perf_counter()
    for epoch in range(int(cfg["epochs"])):
        model.train(); optimizer.zero_grad(set_to_none=True)
        output = model(tensors["x_rna"], tensors["x_aux"], tensors["spatial"],
                       tensors["common"], tensors["reference"])
        loss, row = racf_loss(model, output, tensors["x_rna"], tensors["x_aux"],
                              tensors["reference"], permutation)
        if loss.device.type != "cuda" or not torch.isfinite(loss):
            raise RuntimeError("non-finite or non-CUDA loss")
        loss.backward()
        gradient = {name: {"present": p.grad is not None,
                           "finite": bool(p.grad is not None and torch.isfinite(p.grad).all()),
                           "norm": None if p.grad is None else float(p.grad.norm().detach().cpu())}
                    for name, p in model.named_parameters()}
        required_prefix = ["rna_feature", "rna_spatial", "aux_common", "decoder_rna", "decoder_aux"]
        if cfg["candidate"]["hierarchical_fusion"]:
            required_prefix += ["stage1_logits", "stage2_logits"]
        if cfg["candidate"]["dgi"]:
            required_prefix += ["dgi_discriminator"]
        bad = [name for name, status in gradient.items()
               if any(name.startswith(prefix) for prefix in required_prefix)
               and not (status["present"] and status["finite"])]
        if bad:
            raise RuntimeError(f"enabled branch gradient contract failed: {bad}")
        optimizer.step(); row["epoch"] = epoch + 1; trace.append(row)
    training_seconds = time.perf_counter() - train_started
    views, final_losses = evaluate_model(model, tensors, permutation)
    reliability = views["alpha_omics2"]
    if cfg["candidate"]["reliability_gate"]:
        if not (np.allclose(reliability.sum(1), 1.0, atol=1e-7) and
                reliability.min() >= .1 - 1e-7 and reliability.max() <= .9 + 1e-7):
            raise RuntimeError("runtime reliability contract failed")

    views_path = outdir / "views.npz"; atomic_npz(views_path, **views)
    fixed_path = outdir / "fixed_indices.npz"; atomic_npz(fixed_path, dgi_permutation=permutation_np)
    checkpoint_path = outdir / "model_final.pt"
    atomic_torch(checkpoint_path, {"schema_version": "night9b-racf-checkpoint-v1",
                                   "model_state": model.state_dict(),
                                   "optimizer_state": optimizer.state_dict(), "config": cfg})
    config_copy = outdir / "resolved_config.json"; atomic_json(config_copy, cfg)
    trace_path = outdir / "loss_curve.csv"
    fields = ["epoch", "reconstruction", "cooperation", "reference", "dgi", "total"]
    with trace_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(trace)
    transform_started = time.perf_counter()
    labels, head_audit = run_head(HEADS["H05_EQUAL3_AFFINITY_SPECTRAL"], views,
                                  int(cfg["K"]), inputs["coords"], inputs["ids"], outdir / "head")
    transform_seconds = time.perf_counter() - transform_started
    clusters_path = outdir / "clusters.csv"; save_clusters(clusters_path, inputs["ids"], labels)
    state_sha = tensor_state_sha(model.state_dict())
    manifest_path = outdir / "training_manifest.json"
    files = [checkpoint_path, views_path, fixed_path, config_copy, trace_path,
             clusters_path, outdir / "head/affinity.npz"]
    manifest = {
        "schema_version": "night9b-racf-training-manifest-v1", "status": "SUCCESS_PRE_LABEL",
        "unit_id": cfg["unit_id"], "stage": cfg["stage"], "candidate_id": cfg["candidate"]["id"],
        "candidate_config_sha256": cfg["candidate_config_sha256"], "seed": int(cfg["seed"]),
        "K": int(cfg["K"]), "label_access": False, "scientific_retry": 0, "fallback": False,
        "cuda_used": True, "cuda_device": torch.cuda.get_device_name(device),
        "epochs": int(cfg["epochs"]), "optimizer": cfg["optimizer"],
        "learning_rate": cfg["learning_rate"], "weight_decay": cfg["weight_decay"],
        "model_tensor_state_sha256": state_sha,
        "view_canonical_sha256": {key: array_sha(value) for key, value in views.items()},
        "partition_canonical_sha256": n6_array_sha(canonical_partition(labels)),
        "head": head_audit, "graph_semantics": graph_audit,
        "dgi_permutation_sha256": array_sha(permutation_np),
        "reliability_quantiles": np.quantile(reliability, [0, .01, .5, .99, 1], axis=0).tolist(),
        "gradient_audit_final": gradient, "final_losses": final_losses,
        "runtime": {"training_seconds": training_seconds,
                    "transform_seconds": transform_seconds,
                    "serialization_and_setup_seconds": time.perf_counter() - started - training_seconds - transform_seconds,
                    "total_seconds": time.perf_counter() - started},
        "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024 ** 2,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "inputs": {key: artifact(Path(cfg[key])) for key in
                   ("views_path", "reference_path", "coordinates_path", "observation_ids_path")},
        "artifacts": {path.name: artifact(path) for path in files},
    }
    atomic_json(manifest_path, manifest)
    print(json.dumps({"status": manifest["status"], "unit_id": cfg["unit_id"],
                      "runtime_seconds": manifest["runtime"]["total_seconds"]}, sort_keys=True))


def run_reload(config_path: Path, output_dir: Path) -> None:
    cfg = load_config(config_path); configure(int(cfg["seed"]))
    if not torch.cuda.is_available(): raise RuntimeError("reload requires CUDA")
    device = torch.device("cuda"); inputs = load_inputs(cfg)
    spatial, common, _ = make_graphs(cfg, inputs)
    tensors = {"x_rna": torch.as_tensor(inputs["x_rna"], device=device),
               "x_aux": torch.as_tensor(inputs["x_aux"], device=device),
               "reference": torch.as_tensor(inputs["reference"], device=device),
               "spatial": torch_sparse(spatial, device),
               "common": None if common is None else torch_sparse(common, device)}
    checkpoint = torch.load(output_dir / "model_final.pt", map_location=device, weights_only=False)
    model = RACFModel(inputs["x_rna"].shape[1], cfg["candidate"],
                      latent_dim=inputs["reference"].shape[1]).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    manifest = json.loads((output_dir / "training_manifest.json").read_text())
    if tensor_state_sha(model.state_dict()) != manifest["model_tensor_state_sha256"]:
        raise RuntimeError("reload state SHA mismatch")
    perm = torch.as_tensor(fixed_permutation(len(inputs["ids"]), int(cfg["seed"])), device=device)
    computed, _ = evaluate_model(model, tensors, perm)
    stored = np.load(output_dir / "views.npz", allow_pickle=False)
    view_audit = {}
    for key in VIEW_KEYS:
        delta = float(np.max(np.abs(computed[key] - stored[key])))
        view_audit[key] = {"exact": bool(np.array_equal(computed[key], stored[key])),
                           "max_abs": delta, "sha256": array_sha(computed[key])}
        if not view_audit[key]["exact"]:
            raise RuntimeError(f"fresh-process view parity failed: {key}")
    labels, head_audit = run_head(HEADS["H05_EQUAL3_AFFINITY_SPECTRAL"], computed,
                                  int(cfg["K"]), inputs["coords"], inputs["ids"])
    with (output_dir / "clusters.csv").open(newline="", encoding="utf-8") as handle:
        stored_labels = np.asarray([int(r["cluster"]) for r in csv.DictReader(handle)])
    exact = bool(np.array_equal(canonical_partition(labels), canonical_partition(stored_labels)))
    if not exact: raise RuntimeError("fresh-process H05 partition parity failed")
    atomic_json(output_dir / "reload_audit.json", {
        "schema_version": 1, "status": "PASS", "fresh_process": True,
        "label_access": False, "state_sha256": tensor_state_sha(model.state_dict()),
        "views": view_audit, "h05_partition_exact": exact,
        "h05_partition_sha256": n6_array_sha(canonical_partition(labels)), "head": head_audit,
    })
    print(json.dumps({"status": "RELOAD_PASS", "unit_id": cfg["unit_id"]}, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("train"); p.add_argument("--config", type=Path, required=True)
    p = sub.add_parser("reload"); p.add_argument("--config", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.mode == "train": run_train(args.config)
    else: run_reload(args.config, args.output)


if __name__ == "__main__": main()
