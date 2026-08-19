#!/usr/bin/env python3
"""Train or reload one opaque Night-8A MF-SPC cell."""
from __future__ import annotations

import argparse
import csv
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
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from SpaLORA.night8a_mfspc import (  # noqa: E402
    EMAScaler, MFSPCModel, RNA_EPIGENOME, array_sha,
    centered_cross_covariance_loss, dgi_loss, file_sha, fixed_triplets,
    parameter_grad_audit, prototype_loss, resolve_modules, rna_anchor_support,
    row_l2, row_l2_np, select_family, tensor_state_sha, vicreg_loss,
)


FORBIDDEN_KEYS = {
    "dataset", "dataset_name", "tissue", "platform", "file_name", "labels",
    "label", "ground_truth", "ari", "nmi", "q", "metrics", "evaluator_path",
}


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_save(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    writer(tmp)
    os.replace(tmp, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    def write(target: Path) -> None:
        with target.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
    atomic_save(path, write)


def configure_seed(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def verify_config(config: dict) -> tuple[str, tuple[str, ...]]:
    if FORBIDDEN_KEYS & set(config):
        raise RuntimeError(f"forbidden identity/label keys: {sorted(FORBIDDEN_KEYS & set(config))}")
    required = {"schema_version", "stage", "config_id", "unit_id", "seed", "K",
                "assay_metadata", "registered_modules", "worker_input", "family_reference",
                "output_dir", "epochs", "learning_rate", "weight_decay"}
    if set(config) - (required | {"rna_anchor_spatial_support", "smoke"}):
        raise RuntimeError(f"unknown config keys: {sorted(set(config) - required - {'rna_anchor_spatial_support', 'smoke'})}")
    if not required <= set(config):
        raise RuntimeError(f"missing config keys: {sorted(required - set(config))}")
    family = select_family(config["assay_metadata"])
    active = resolve_modules(config["registered_modules"], family)
    return family, active


def load_inputs(config: dict, family: str, active: tuple[str, ...]) -> dict:
    worker = Path(config["worker_input"])
    payload = json.loads(worker.read_text())
    if FORBIDDEN_KEYS & set(payload):
        raise RuntimeError("worker input contains forbidden key")
    allowed = {"K", "g00_views", "g04_views", "observation_count", "observation_ids",
               "ordered_observation_sha256", "pseudo_affinity", "pseudo_partition", "s00", "s04", "unit_id"}
    if set(payload) - allowed:
        raise RuntimeError(f"unexpected worker payload keys: {sorted(set(payload) - allowed)}")
    if str(payload["unit_id"]) != str(config["unit_id"]) or int(payload["K"]) != int(config["K"]):
        raise RuntimeError("opaque unit or K mismatch")
    ids = [x.strip() for x in Path(payload["observation_ids"]).read_text().splitlines() if x.strip()]
    z = np.load(payload["g04_views"], allow_pickle=False)
    x1 = row_l2_np(z["emb_latent_omics1"])
    x2 = row_l2_np(z["emb_latent_omics2"])
    if len(ids) != len(x1) or x1.shape != x2.shape:
        raise RuntimeError("view/observation mismatch")
    reference_path = Path(config["family_reference"])
    if reference_path.suffix == ".npy":
        reference = row_l2_np(np.load(reference_path, allow_pickle=False))
    else:
        ref = np.load(reference_path, allow_pickle=False)
        reference = row_l2_np(ref["SpaLORA_fused"])
    if len(reference) != len(x1):
        raise RuntimeError("family reference shape mismatch")
    anchor_audit = {"active": False}
    if "RNA_ANCHOR" in active:
        if family != RNA_EPIGENOME:
            raise RuntimeError("RNA_ANCHOR active outside epigenome family")
        spatial_path = Path(config["rna_anchor_spatial_support"])
        spatial = sp.load_npz(spatial_path)
        anchor = rna_anchor_support(spatial, x1, ids, k=10)
        x1 = row_l2_np(.5 * x1 + .5 * anchor.dot(x1))
        x2 = row_l2_np(.5 * x2 + .5 * anchor.dot(x2))
        anchor_audit = {"active": True, "support_path": str(spatial_path),
                        "support_file_sha256": file_sha(spatial_path), "nnz": int(anchor.nnz),
                        "self_loops": int(np.sum(anchor.diagonal() != 0)),
                        "dense_n_by_n": False}
    return {"payload": payload, "ids": ids, "x1": x1, "x2": x2,
            "reference": reference, "anchor_audit": anchor_audit}


def build_fixed(active: tuple[str, ...], inputs: dict, seed: int) -> dict:
    n = len(inputs["ids"]); result = {}
    rng = np.random.default_rng(int(seed) + 8675309)
    result["dgi_permutation"] = rng.permutation(n).astype(np.int64)
    if "SMART_TRIPLET" in active:
        positive, negative, audit = fixed_triplets(inputs["x1"], inputs["x2"],
                                                   inputs["ids"], seed, k=3,
                                                   farthest_fraction=.60)
        result.update({"triplet_positive": positive, "triplet_negative": negative,
                       "triplet_audit": audit})
    return result


def forward_loss(model: MFSPCModel, x1: torch.Tensor, x2: torch.Tensor,
                 reference: torch.Tensor, active: tuple[str, ...], fixed: dict,
                 scaler: EMAScaler, epoch: int) -> tuple[torch.Tensor, dict, dict, torch.Tensor | None]:
    output = model(x1, x2, reference)
    base = .5 * (F.mse_loss(output["recon1"], x1) + F.mse_loss(output["recon2"], x2))
    base = base + .10 * F.mse_loss(output["candidate"], reference)
    raw: dict[str, torch.Tensor] = {}
    details: dict[str, torch.Tensor] = {}
    occupancy = None
    if "SP" in active:
        raw["SP"] = .5 * (centered_cross_covariance_loss(output["shared1"], output["private1"])
                          + centered_cross_covariance_loss(output["shared2"], output["private2"]))
    if "RR10" in active or "RR30" in active:
        raw["RR"] , rr = vicreg_loss(output["shared1"], output["shared2"]); details.update(rr)
    if "PROTO" in active:
        raw["PROTO"], proto, occupancy = prototype_loss(model, output); details.update(proto)
    if "DGI" in active:
        perm = torch.as_tensor(fixed["dgi_permutation"], device=x1.device)
        raw["DGI"] = dgi_loss(output["candidate"], perm)
    if "SMART_TRIPLET" in active:
        pos = torch.as_tensor(fixed["triplet_positive"], device=x1.device)
        neg = torch.as_tensor(fixed["triplet_negative"], device=x1.device)
        raw["SMART_TRIPLET"] = F.triplet_margin_loss(output["shared1"], output["shared2"][pos],
                                                       output["shared2"][neg], margin=.50)
    weights = {"SP": .05, "RR": .30 if "RR30" in active else .10,
               "PROTO": .10, "DGI": .10, "SMART_TRIPLET": .10}
    total = base
    log = {"base_raw": float(base.detach().cpu())}
    for name in sorted(raw):
        factor = scaler.factor(name, base, raw[name], epoch)
        weighted = weights[name] * factor * raw[name]
        total = total + weighted
        log[f"{name}_raw"] = float(raw[name].detach().cpu())
        log[f"{name}_scale"] = factor
        log[f"{name}_weighted"] = float(weighted.detach().cpu())
    for key, value in details.items():
        log[key] = float(value.detach().cpu())
    log["total"] = float(total.detach().cpu())
    return total, output, log, occupancy


def artifact_row(path: Path) -> dict:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": file_sha(path)}


def run_train(config_path: Path) -> None:
    config = json.loads(config_path.read_text())
    family, active = verify_config(config)
    out = Path(config["output_dir"])
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("refusing to overwrite an existing attempt")
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter(); configure_seed(int(config["seed"]))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    inputs = load_inputs(config, family, active)
    fixed = build_fixed(active, inputs, int(config["seed"]))
    dim = int(inputs["x1"].shape[1]); fused_dim = int(inputs["reference"].shape[1])
    model = MFSPCModel(dim, active, int(config["K"]), fused_dim=fused_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]),
                                 weight_decay=float(config["weight_decay"]))
    x1 = torch.as_tensor(inputs["x1"], device=device)
    x2 = torch.as_tensor(inputs["x2"], device=device)
    reference = torch.as_tensor(inputs["reference"], device=device)
    if any(x.device.type != "cuda" for x in (x1, x2, reference)) or next(model.parameters()).device.type != "cuda":
        raise RuntimeError("CUDA placement contract failed")
    scaler = EMAScaler(beta=.99, warmup_epochs=10)
    rows = []; grad_audit = None
    epochs = int(config["epochs"])
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss, output, row, occupancy = forward_loss(model, x1, x2, reference, active, fixed, scaler, epoch)
        if loss.device.type != "cuda" or not torch.isfinite(loss):
            raise RuntimeError("non-finite or non-CUDA loss")
        loss.backward()
        grad_audit = parameter_grad_audit(model)
        expected = [v for k, v in grad_audit.items() if v["requires_grad"] and not k.startswith("prototypes")]
        if not expected or not all(v["grad_present"] and v["grad_finite"] for v in expected):
            raise RuntimeError("missing or non-finite parameter gradient")
        optimizer.step()
        if occupancy is not None:
            model.update_teacher(occupancy, .99)
        row.update({"epoch": epoch + 1,
                    "gradient_norm": float(torch.sqrt(sum((p.grad.detach().square().sum()
                                                            for p in model.parameters() if p.grad is not None))).cpu())})
        rows.append(row)
    model.eval()
    with torch.no_grad():
        output = model(x1, x2, reference)
    arrays = {
        "emb_latent_omics1": output["shared1"].detach().cpu().numpy(),
        "emb_latent_omics2": output["shared2"].detach().cpu().numpy(),
        "private_omics1": output["private1"].detach().cpu().numpy(),
        "private_omics2": output["private2"].detach().cpu().numpy(),
        "SpaLORA_fused": output["fused"].detach().cpu().numpy(),
    }
    embedding_path = out / "embeddings.npz"
    atomic_npz(embedding_path, **arrays)
    fixed_path = out / "fixed_indices.npz"
    fixed_arrays = {k: v for k, v in fixed.items() if isinstance(v, np.ndarray)}
    atomic_npz(fixed_path, **fixed_arrays)
    trace_path = out / "loss_trace.csv"
    fields = sorted(set().union(*(r.keys() for r in rows)), key=lambda x: (x != "epoch", x))
    with trace_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    checkpoint = {
        "schema_version": "night8a-mfspc-checkpoint-v1",
        "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
        "config": config, "family": family, "active_modules": active,
        "scaler_state": {"base_ema": scaler.base_ema, "aux_ema": scaler.aux_ema,
                         "beta": scaler.beta, "warmup_epochs": scaler.warmup_epochs},
        "torch_rng_state": torch.get_rng_state(), "cuda_rng_state": torch.cuda.get_rng_state_all(),
    }
    checkpoint_path = out / "model_final.pt"
    atomic_save(checkpoint_path, lambda p: torch.save(checkpoint, p))
    config_copy = out / "resolved_config.json"; atomic_json(config_copy, config)
    metrics_placeholder = out / "metrics_placeholder.json"
    atomic_json(metrics_placeholder, {"status": "SEALED_PRE_LABEL", "label_access": False,
                                      "ari": None, "nmi": None, "q": None})
    manifest_path = out / "training_manifest.json"
    artifacts = {p.name: artifact_row(p) for p in
                 (checkpoint_path, embedding_path, fixed_path, trace_path, config_copy, metrics_placeholder)}
    manifest = {
        "schema_version": "night8a-training-manifest-v1", "status": "SUCCESS_PRE_LABEL",
        "stage": config["stage"], "config_id": config["config_id"], "unit_id": config["unit_id"],
        "seed": int(config["seed"]), "K": int(config["K"]), "family": family,
        "registered_modules": list(config["registered_modules"]), "active_modules": list(active),
        "resolved_runtime_sha256": __import__("SpaLORA.night8a_mfspc", fromlist=["canonical_json_sha"]).canonical_json_sha(
            {"family": family, "modules": active, "K": int(config["K"]),
             "input_dim": dim, "fused_dim": fused_dim}),
        "assay_metadata": config["assay_metadata"], "label_access": False,
        "dataset_identity_received_by_trainer": False, "dense_n_by_n_created": False,
        "cuda_required": True, "cuda_used": True, "cuda_device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "epochs": epochs, "family_reference": artifact_row(Path(config["family_reference"])),
        "worker_input": artifact_row(Path(config["worker_input"])),
        "anchor_audit": inputs["anchor_audit"], "fixed_index_audit": fixed.get("triplet_audit"),
        "model_tensor_state_sha256": tensor_state_sha(model.state_dict()),
        "embedding_canonical_sha256": {k: array_sha(v) for k, v in arrays.items()},
        "gradient_audit_final": grad_audit, "artifacts": artifacts,
        "runtime_seconds": time.perf_counter() - start,
        "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024 ** 2),
        "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    atomic_json(manifest_path, manifest)
    print(json.dumps({"status": manifest["status"], "unit_id": config["unit_id"],
                      "config_id": config["config_id"], "runtime_seconds": manifest["runtime_seconds"]},
                     sort_keys=True), flush=True)


def run_reload(config_path: Path, output: Path) -> None:
    config = json.loads(config_path.read_text()); family, active = verify_config(config)
    manifest = json.loads((output / "training_manifest.json").read_text())
    checkpoint_path = output / "model_final.pt"
    if file_sha(checkpoint_path) != manifest["artifacts"]["model_final.pt"]["sha256"]:
        raise RuntimeError("checkpoint file SHA mismatch")
    configure_seed(int(config["seed"])); device = torch.device("cuda")
    inputs = load_inputs(config, family, active)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = MFSPCModel(inputs["x1"].shape[1], active, int(config["K"]),
                       fused_dim=inputs["reference"].shape[1]).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True); model.eval()
    if tensor_state_sha(model.state_dict()) != manifest["model_tensor_state_sha256"]:
        raise RuntimeError("canonical model state SHA mismatch")
    with torch.no_grad():
        result = model(torch.as_tensor(inputs["x1"], device=device),
                       torch.as_tensor(inputs["x2"], device=device),
                       torch.as_tensor(inputs["reference"], device=device))
    stored = np.load(output / "embeddings.npz", allow_pickle=False)
    computed = {"emb_latent_omics1": result["shared1"].cpu().numpy(),
                "emb_latent_omics2": result["shared2"].cpu().numpy(),
                "private_omics1": result["private1"].cpu().numpy(),
                "private_omics2": result["private2"].cpu().numpy(),
                "SpaLORA_fused": result["fused"].cpu().numpy()}
    audits = {}
    for key, value in computed.items():
        delta = float(np.max(np.abs(value - stored[key]))) if value.size else 0.0
        audits[key] = {"max_abs_delta": delta, "allclose_1e-7": bool(np.allclose(value, stored[key], atol=1e-7, rtol=1e-7)),
                       "computed_canonical_sha256": array_sha(value),
                       "stored_canonical_sha256": array_sha(stored[key])}
        if not audits[key]["allclose_1e-7"]:
            raise RuntimeError(f"reload embedding mismatch: {key}")
    atomic_json(output / "reload_audit.json", {
        "status": "PASS", "fresh_process": True, "label_access": False,
        "checkpoint_file_sha256": file_sha(checkpoint_path), "state_sha256": tensor_state_sha(model.state_dict()),
        "embeddings": audits,
    })
    print(json.dumps({"status": "RELOAD_PASS", "unit_id": config["unit_id"],
                      "config_id": config["config_id"]}, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="mode", required=True)
    train = sub.add_parser("train"); train.add_argument("--config", type=Path, required=True)
    reload_p = sub.add_parser("reload"); reload_p.add_argument("--config", type=Path, required=True)
    reload_p.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.mode == "train": run_train(args.config)
    else: run_reload(args.config, args.output)


if __name__ == "__main__":
    main()
