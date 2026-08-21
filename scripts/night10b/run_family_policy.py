#!/usr/bin/env python3
"""Single public CLI for the frozen Night-10B family policy."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import scipy.sparse as sp
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from SpaLORA.family_policy import resolve_family_policy  # noqa: E402
from SpaLORA.family_runtime import (  # noqa: E402
    atomic_json, atomic_npy, authority_audit, file_sha256, load_q00_rows,
    ordered_observation_sha256,
    replay_q00_row, resource_snapshot, verify_replay_row,
)
from SpaLORA.night3a_ige import model_state_sha256  # noqa: E402
from SpaLORA.night3af_cache import load_cache, sha256_file  # noqa: E402
from SpaLORA.night5a_rnd import Night5ATrainer  # noqa: E402
from SpaLORA.night6c_pipeline import (  # noqa: E402
    BASE_C04, atomic_torch_save, forward_model, load_graph_data, load_views,
    run_head, save_views, sparse_sha,
)
from SpaLORA.night6d_pipeline import HEADS as NIGHT6D_HEADS  # noqa: E402
from SpaLORA.night7a_consensus import (  # noqa: E402
    build_base_affinities, candidate_affinity, canonical_partition,
)
from SpaLORA.night7b_adaptive import run_partition  # noqa: E402
from SpaLORA.night8b_pipeline import adapter_endpoint  # noqa: E402
from SpaLORA.night10a_qcrd import canonical_array_sha256  # noqa: E402


PYTHON = Path("/root/miniconda3/envs/SpaLORA/bin/python")
ADAPTER_TRAINER = REPO / "scripts/night7b_train.py"
NIGHT7B_REGISTRY = REPO / (
    "protocols/night7b/"
    "SpaLORA_Night7B_Adaptive_Relational_Fusion_Registry_2026-08-18.json"
)


def _write_ids(path: Path, values: list[str]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("\n".join(values) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _write_clusters(path: Path, ids: list[str], labels: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["observation_id", "cluster"])
        writer.writeheader()
        for name, label in zip(ids, labels):
            writer.writerow({"observation_id": name, "cluster": int(label)})
    os.replace(tmp, path)


def _read_clusters(path: Path, ids: list[str]) -> np.ndarray:
    names: list[str] = []
    labels: list[int] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            names.append(str(row["observation_id"]))
            labels.append(int(row["cluster"]))
    if names != ids:
        raise RuntimeError("smoke endpoint observation order drift")
    return np.asarray(labels, dtype=np.int64)


def _train_backbone(spec: Mapping[str, Any], seed: int, output: Path) -> dict[str, Any]:
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    base_dir = Path(spec["base_cache_dir"])
    graph_dir = Path(spec["graph_cache_dir"])
    base_manifest_sha = sha256_file(base_dir / "manifest.json")
    prepared = load_cache(base_dir, base_manifest_sha)
    data, graph_manifest = load_graph_data(prepared, graph_dir)
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for fresh smoke")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model_config = dict(spec["model_config"])
    trainer = Night5ATrainer(data, model_config, BASE_C04, int(seed), device, {}, 1e-12)
    result = trainer.train()
    views = forward_model(result.model, data, device)
    view_contracts = save_views(
        output / "views.npz", views, prepared.obs_names.astype(str)
    )
    ids = prepared.obs_names.astype(str).tolist()
    _write_ids(output / "observation_ids.txt", ids)
    state_sha = model_state_sha256(result.model)
    if state_sha != result.final_state_sha256:
        raise RuntimeError("fresh backbone final state SHA mismatch")
    checkpoint = {
        "backbone_id": spec["backbone_id"],
        "base_manifest_sha256": base_manifest_sha,
        "canonical_tensor_state_sha256": state_sha,
        "graph_manifest_sha256": sha256_file(graph_dir / "manifest.json"),
        "model_config": model_config,
        "model_state_dict": {
            key: value.detach().cpu() for key, value in result.model.state_dict().items()
        },
        "seed": int(seed),
    }
    atomic_torch_save(output / "model_final.pt", checkpoint)
    manifest = {
        "backbone_id": spec["backbone_id"],
        "base_cache_dir": str(base_dir),
        "base_manifest_sha256": base_manifest_sha,
        "checkpoint_file_sha256": file_sha256(output / "model_final.pt"),
        "checkpoint_path": str(output / "model_final.pt"),
        "final_tensor_state_sha256": state_sha,
        "fresh_full_training": True,
        "graph_cache_dir": str(graph_dir),
        "graph_manifest_sha256": sha256_file(graph_dir / "manifest.json"),
        "label_reads": 0,
        "model_config": model_config,
        "observation_count": len(ids),
        "ordered_observation_sha256": ordered_observation_sha256(ids),
        "seed": int(seed),
        "status": "PASS",
        "view_contracts": view_contracts,
        "views_file_sha256": file_sha256(output / "views.npz"),
        **resource_snapshot(started),
    }
    atomic_json(output / "training_manifest.json", manifest)
    return manifest


def _reload_backbone(spec: Mapping[str, Any], seed: int, output: Path) -> tuple[dict[str, Any], dict[str, np.ndarray], list[str], np.ndarray]:
    checkpoint = torch.load(output / "model_final.pt", map_location="cuda", weights_only=False)
    base_dir = Path(spec["base_cache_dir"])
    graph_dir = Path(spec["graph_cache_dir"])
    prepared = load_cache(base_dir, checkpoint["base_manifest_sha256"])
    data, graph_manifest = load_graph_data(prepared, graph_dir)
    if sha256_file(graph_dir / "manifest.json") != checkpoint["graph_manifest_sha256"]:
        raise RuntimeError("fresh backbone graph manifest drift")
    device = torch.device("cuda:0")
    trainer = Night5ATrainer(
        data, checkpoint["model_config"], BASE_C04, int(seed), device, {}, 1e-12
    )
    model = trainer.new_model()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    state_sha = model_state_sha256(model)
    if state_sha != checkpoint["canonical_tensor_state_sha256"]:
        raise RuntimeError("fresh backbone strict state reload mismatch")
    observed = forward_model(model, data, device)
    expected = load_views(output / "views.npz")
    maximum = 0.0
    for key in expected:
        if not np.allclose(observed[key], expected[key], atol=1e-6, rtol=1e-5):
            raise RuntimeError(f"fresh backbone forward reload mismatch: {key}")
        maximum = max(maximum, float(np.max(np.abs(observed[key] - expected[key]))))
    audit = {
        "atol": 1e-6,
        "backbone_id": spec["backbone_id"],
        "checkpoint_file_sha256": file_sha256(output / "model_final.pt"),
        "forward_allclose": True,
        "fresh_process": True,
        "label_reads": 0,
        "max_absolute_error": maximum,
        "rtol": 1e-5,
        "state_load_strict": True,
        "status": "PASS",
    }
    atomic_json(output / "reload_audit.json", audit)
    return audit, observed, prepared.obs_names.astype(str).tolist(), np.asarray(prepared.coordinates)


def _run_adapter(command: list[str], log_path: Path, timeout: int = 1800) -> None:
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
    }
    with log_path.open("wb") as handle:
        completed = subprocess.run(
            command, stdout=handle, stderr=subprocess.STDOUT,
            timeout=timeout, env=env,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"R02 subprocess failed; inspect {log_path}")


def _adapter_inputs(
    output: Path, config: Mapping[str, Any], views: Mapping[str, Path],
    ids: list[str], s00: sp.csr_matrix, s04: sp.csr_matrix,
) -> tuple[Path, Path]:
    unit = output / "adapter_input"
    unit.mkdir(parents=True, exist_ok=False)
    ids_path = unit / "observation_ids.txt"
    _write_ids(ids_path, ids)
    s00_path, s04_path = unit / "s00.npz", unit / "s04.npz"
    sp.save_npz(s00_path, s00, compressed=True)
    sp.save_npz(s04_path, s04, compressed=True)
    contract = {
        "K": int(config["K"]),
        "g00_views": str(views["G00_SP18_F20_CORR_UNION"]),
        "g04_views": str(views["G04_SP10_F10_EUC_UNION"]),
        "observation_count": len(ids),
        "observation_ids": str(ids_path),
        "ordered_observation_sha256": ordered_observation_sha256(ids),
        "pseudo_affinity": "",
        "pseudo_partition": "",
        "s00": str(s00_path),
        "s04": str(s04_path),
        "unit_id": "opaque_fresh_smoke_rna_atac_seed0",
    }
    atomic_json(unit / "worker_input.json", contract)
    recipe = output / "r02_config.json"
    atomic_json(recipe, {
        "epochs": 160, "fusion": "equal", "losses": ["RECON", "MNN"],
        "recipe_id": "R02", "seed": int(config["seed"]),
    })
    return unit, recipe


def run_smoke(config_path: Path, output: Path) -> dict[str, Any]:
    started = time.perf_counter()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    policy = resolve_family_policy(config["primary_assay"], config["auxiliary_assay"])
    if policy.recipe.recipe_id != config["recipe_id"]:
        raise RuntimeError("smoke recipe does not match explicit assay pair")
    if output.exists():
        raise RuntimeError("fresh smoke output already exists")
    output.mkdir(parents=True)
    backbone_rows = []
    view_paths: dict[str, Path] = {}
    ids: list[str] | None = None
    coords: np.ndarray | None = None
    for spec in config["backbones"]:
        target = output / "backbones" / spec["backbone_id"]
        backbone_rows.append(_train_backbone(spec, int(config["seed"]), target))
        view_paths[spec["backbone_id"]] = target / "views.npz"
        local_ids = target.joinpath("observation_ids.txt").read_text(encoding="utf-8").splitlines()
        prepared = load_cache(Path(spec["base_cache_dir"]), spec.get("base_manifest_sha256"))
        if ids is None:
            ids, coords = local_ids, np.asarray(prepared.coordinates)
        elif ids != local_ids or not np.array_equal(coords, prepared.coordinates):
            raise RuntimeError("fresh dual-backbone observation order drift")
    assert ids is not None and coords is not None
    endpoint_dir = output / "endpoint"
    endpoint_dir.mkdir()
    if policy.recipe.family == "RNA_PROTEIN":
        with np.load(view_paths[policy.recipe.backbones[0]], allow_pickle=False) as payload:
            views = {key: np.asarray(payload[key]) for key in (
                "emb_latent_omics1", "emb_latent_omics2", "SpaLORA_fused"
            )}
        labels, aux = run_head(
            NIGHT6D_HEADS["H05_EQUAL3_AFFINITY_SPECTRAL"], views,
            int(config["K"]), coords, ids, endpoint_dir,
        )
        canonical = canonical_partition(labels)
        _write_clusters(endpoint_dir / "clusters.csv", ids, canonical)
        endpoint = {
            "canonical_partition_sha256": canonical_array_sha256(canonical),
            "cluster_count": len(np.unique(canonical)),
            "endpoint": policy.recipe.endpoint,
            "label_reads": 0,
            "sparse_affinity": True,
            "status": "PASS",
            "transform_audit": aux,
        }
    else:
        with np.load(view_paths["G00_SP18_F20_CORR_UNION"], allow_pickle=False) as p:
            v00 = {key: np.asarray(p[key]) for key in (
                "emb_latent_omics1", "emb_latent_omics2", "SpaLORA_fused"
            )}
        with np.load(view_paths["G04_SP10_F10_EUC_UNION"], allow_pickle=False) as p:
            v04 = {key: np.asarray(p[key]) for key in (
                "emb_latent_omics1", "emb_latent_omics2", "SpaLORA_fused"
            )}
        base = build_base_affinities(v00, v04, ids, coords)
        c06, c06_aux = candidate_affinity(
            "C06_DUAL_ROW_STOCHASTIC_MEAN", base, ids
        )
        c06_path = output / "c06_affinity.npz"
        sp.save_npz(c06_path, c06, compressed=True)
        unit, recipe = _adapter_inputs(
            output, config, view_paths, ids, base["S_G00"], base["S_G04"]
        )
        worker = output / "adapter"
        _run_adapter([
            str(PYTHON), str(ADAPTER_TRAINER), "train", "--unit-dir", str(unit),
            "--config", str(recipe), "--output", str(worker),
        ], output / "adapter_train.log")
        _run_adapter([
            str(PYTHON), str(ADAPTER_TRAINER), "reload", "--unit-dir", str(unit),
            "--config", str(recipe), "--output", str(worker),
        ], output / "adapter_reload.log")
        embedding = np.load(worker / "embedding.npy", allow_pickle=False)
        affinity = adapter_endpoint(embedding, c06, ids)
        registry = json.loads(NIGHT7B_REGISTRY.read_text(encoding="utf-8"))
        labels, aux = run_partition(
            "H01", affinity, int(config["K"]),
            registry["partition_heads"]["LEIDEN_EXACT_K"]["resolution_grid"],
        )
        canonical = canonical_partition(labels)
        _write_clusters(endpoint_dir / "clusters.csv", ids, canonical)
        endpoint = {
            "adapter_checkpoint_sha256": file_sha256(worker / "model_final.pt"),
            "adapter_embedding_sha256": canonical_array_sha256(embedding),
            "adapter_reload_audit_sha256": file_sha256(worker / "reload_forward_audit.json"),
            "c06_aux": c06_aux,
            "c06_canonical_sha256": sparse_sha(c06),
            "canonical_partition_sha256": canonical_array_sha256(canonical),
            "cluster_count": len(np.unique(canonical)),
            "endpoint": policy.recipe.endpoint,
            "label_reads": 0,
            "sparse_affinity": True,
            "status": "PASS",
            "transform_audit": aux,
        }
    atomic_json(endpoint_dir / "endpoint_manifest.json", endpoint)
    manifest = {
        "assay_pair": [config["primary_assay"], config["auxiliary_assay"]],
        "backbones": backbone_rows,
        "config_file_sha256": file_sha256(config_path),
        "endpoint": endpoint,
        "fresh_full_training": True,
        "label_reads": 0,
        "recipe_id": policy.recipe.recipe_id,
        "resolved_policy": policy.as_dict(),
        "schema": "spalora.night10b.fresh_smoke_manifest.v1",
        "seed": int(config["seed"]),
        "smoke_id": config["smoke_id"],
        "status": "PASS",
        **resource_snapshot(started),
    }
    atomic_json(output / "smoke_manifest.json", manifest)
    return manifest


def reload_smoke(config_path: Path, output: Path) -> dict[str, Any]:
    started = time.perf_counter()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    policy = resolve_family_policy(config["primary_assay"], config["auxiliary_assay"])
    observed_views: dict[str, dict[str, np.ndarray]] = {}
    ids: list[str] | None = None
    coords: np.ndarray | None = None
    backbones = []
    for spec in config["backbones"]:
        target = output / "backbones" / spec["backbone_id"]
        audit, views, local_ids, local_coords = _reload_backbone(
            spec, int(config["seed"]), target
        )
        observed_views[spec["backbone_id"]] = views
        backbones.append(audit)
        if ids is None:
            ids, coords = local_ids, local_coords
        elif ids != local_ids or not np.array_equal(coords, local_coords):
            raise RuntimeError("fresh reload dual-backbone order drift")
    assert ids is not None and coords is not None
    saved = _read_clusters(output / "endpoint/clusters.csv", ids)
    if policy.recipe.family == "RNA_PROTEIN":
        labels, _ = run_head(
            NIGHT6D_HEADS["H05_EQUAL3_AFFINITY_SPECTRAL"],
            observed_views["G04_SP10_F10_EUC_UNION"], int(config["K"]),
            coords, ids, None,
        )
    else:
        base = build_base_affinities(
            observed_views["G00_SP18_F20_CORR_UNION"],
            observed_views["G04_SP10_F10_EUC_UNION"], ids, coords,
        )
        c06, _ = candidate_affinity("C06_DUAL_ROW_STOCHASTIC_MEAN", base, ids)
        if sparse_sha(c06) != sparse_sha(sp.load_npz(output / "c06_affinity.npz")):
            raise RuntimeError("fresh C06 rebuild mismatch")
        unit, recipe, worker = output / "adapter_input", output / "r02_config.json", output / "adapter"
        _run_adapter([
            str(PYTHON), str(ADAPTER_TRAINER), "reload", "--unit-dir", str(unit),
            "--config", str(recipe), "--output", str(worker),
        ], output / "adapter_reload_fresh_process.log")
        embedding = np.load(worker / "embedding.npy", allow_pickle=False)
        affinity = adapter_endpoint(embedding, c06, ids)
        registry = json.loads(NIGHT7B_REGISTRY.read_text(encoding="utf-8"))
        labels, _ = run_partition(
            "H01", affinity, int(config["K"]),
            registry["partition_heads"]["LEIDEN_EXACT_K"]["resolution_grid"],
        )
    canonical = canonical_partition(labels)
    exact = np.array_equal(canonical, saved)
    if not exact:
        raise RuntimeError("fresh smoke endpoint partition round-trip mismatch")
    audit = {
        "backbones": backbones,
        "canonical_partition_exact": True,
        "canonical_partition_sha256": canonical_array_sha256(canonical),
        "fresh_process": True,
        "label_reads": 0,
        "schema": "spalora.night10b.checkpoint_roundtrip_audit.v1",
        "smoke_id": config["smoke_id"],
        "status": "PASS",
        **resource_snapshot(started),
    }
    atomic_json(output / "checkpoint_roundtrip_audit.json", audit)
    return audit


def _q00_lookup(repo: Path, unit_id: str) -> dict[str, Any]:
    rows = load_q00_rows(repo / (
        "outputs/night10a_rev2_handoff/r2_lock/"
        "r2_total_lock_independent_audit.json"
    ))
    matches = [row for row in rows if row["unit_id"] == unit_id]
    if len(matches) != 1:
        raise RuntimeError("formal replay unit is not uniquely registered")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    resolve = sub.add_parser("resolve")
    resolve.add_argument("--primary-assay", required=True)
    resolve.add_argument("--auxiliary-assay", required=True)
    resolve.add_argument("--output", type=Path, required=True)
    authority = sub.add_parser("authority")
    authority.add_argument("--output", type=Path, required=True)
    replay = sub.add_parser("replay")
    replay.add_argument("--unit-id", required=True)
    replay.add_argument("--output", type=Path, required=True)
    verify = sub.add_parser("verify-row")
    verify.add_argument("--unit-id", required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--config", type=Path, required=True)
    smoke.add_argument("--output", type=Path, required=True)
    smoke_reload = sub.add_parser("smoke-reload")
    smoke_reload.add_argument("--config", type=Path, required=True)
    smoke_reload.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "resolve":
        value = resolve_family_policy(args.primary_assay, args.auxiliary_assay).as_dict()
        atomic_json(args.output, value)
    elif args.command == "authority":
        atomic_json(args.output, authority_audit(REPO))
    elif args.command == "replay":
        q00 = _q00_lookup(REPO, args.unit_id)
        atomic_json(args.output, replay_q00_row(q00, REPO))
    elif args.command == "verify-row":
        q00 = _q00_lookup(REPO, args.unit_id)
        atomic_json(args.output, verify_replay_row(args.manifest, q00, REPO))
    elif args.command == "smoke":
        run_smoke(args.config, args.output)
    else:
        reload_smoke(args.config, args.output)


if __name__ == "__main__":
    main()
