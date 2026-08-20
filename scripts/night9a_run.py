#!/usr/bin/env python3
"""Pre-label P0, reference construction, chain runner, and stage locks for Night-9A.

This file is intentionally label-free.  The only P22 label reader is
``scripts/night9a_evaluate.py`` and MISAR Y is never referenced here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp
import sklearn
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from SpaLORA.night3a_ige import model_state_sha256
from SpaLORA.night3af_cache import load_cache
from SpaLORA.night5a_rnd import Night5ATrainer
from SpaLORA.night6c_pipeline import array_sha, forward_model, load_graph_data, sparse_sha
from SpaLORA.night6d_pipeline import DATASET_CFG as N6D_CFG
from SpaLORA.night7a_consensus import atomic_sparse
from SpaLORA.night8b_pipeline import (
    BASE_C04, G00, G04, MISAR_CFG, adapter_endpoint, base_affinities,
    load_graph as load_misar_graph, observation_sha,
)
from SpaLORA.night9a_efficient import (
    HEAD_ID, VIEW_KEYS, affinity_fidelity, canonical_json_sha,
    eigen_kmeans100, exact_partition_equal, load_projection_operators,
    projection_views, sha256_file,
)


RAW = Path("/root/autodl-fs/night9a_efficient_topology_transfer_20260820")
OUT = REPO / "outputs/night9a"
REGISTRY = REPO / "protocols/night9a/SpaLORA_Night9A_Efficient_Topology_Transfer_Registry_2026-08-20.json"
TASKBOOK = REPO / "protocols/night9a/SpaLORA_Night9A_Efficient_Topology_Transfer_Taskbook_2026-08-20.md"
N7B_ROOT = Path("/root/autodl-fs/night7b_score_rnd_20260818")
N6D_RAW = Path("/root/autodl-fs/night6d_raw_runs_20260817")
N6D_CACHE = Path("/root/autodl-fs/night6d_cache_20260817")
N8B_RAW = Path("/root/autodl-fs/night8b_raw_runs_20260820")
N8B_HEAD = Path("/root/autodl-fs/night8b_head_recovery_20260820")
N8B_EVAL = Path("/root/autodl-fs/night8b_cardinality_safe_eval_20260820")
P22_BASE = Path("/root/autodl-fs/night3af_p0d_builds_20260810/process_a/p22")
BASE_COMMIT = "991ba9dcbd7108c2b3b4c9b4b8e233c5f394da5a"
BASE_TAG = "night8b-cardinality-safe-eval-final-20260820"


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def atomic_npz(path: Path, values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **values)
    os.replace(tmp, path)


def atomic_torch(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def save_clusters(path: Path, ids, labels) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame({"observation_id": ids, "cluster": labels}).to_csv(tmp, index=False)
    os.replace(tmp, path)


def load_registry() -> dict:
    value = json.loads(REGISTRY.read_text())
    if value["base"]["commit"] != BASE_COMMIT or value["base"]["tag"] != BASE_TAG:
        raise RuntimeError("registry base authority mismatch")
    expected = [f"E{i:02d}_" for i in range(9)]
    observed = [row["id"][:4] for row in value["candidates"]]
    if observed != expected:
        raise RuntimeError("candidate registry order drift")
    return value


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def file_row(path: Path) -> dict:
    return {"path": str(path), "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def load_npz_views(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as handle:
        missing = set(VIEW_KEYS) - set(handle.files)
        if missing:
            raise RuntimeError(f"view archive missing keys: {sorted(missing)}")
        result = {key: np.asarray(handle[key], dtype=np.float32) for key in VIEW_KEYS}
    if len({value.shape[0] for value in result.values()}) != 1:
        raise RuntimeError("view observation count mismatch")
    if any(value.ndim != 2 or not np.isfinite(value).all() for value in result.values()):
        raise RuntimeError("non-finite or malformed views")
    return result


def read_ids(path: Path) -> np.ndarray:
    ids = np.asarray([value.strip() for value in path.read_text().splitlines()
                      if value.strip()], dtype=str)
    if len(ids) != len(set(ids)):
        raise RuntimeError("observation IDs are not unique")
    return ids


def p22_adapter_cell(seed: int) -> Path:
    hits = sorted((N7B_ROOT / "adapter_stage").glob(
        f"*/formal/R02/u{20 + seed:03d}/attempt_001"))
    if len(hits) != 1:
        raise RuntimeError(f"P22 full F00 adapter cell is not unique for seed {seed}")
    return hits[0]


def dataset_paths(dataset: str, seed: int) -> dict:
    if not 0 <= int(seed) <= 9:
        raise ValueError(seed)
    if dataset == "p22":
        source = N7B_ROOT / "source" / f"u{20 + seed:03d}"
        adapter = p22_adapter_cell(seed)
        teacher_transform = adapter / "transforms/E1_ADAPTER_C06_MEAN/H01"
        return {
            "dataset": dataset, "seed": seed, "K": 9,
            "ids": source / "observation_ids.txt", "coordinates": source / "coordinates.npy",
            "source_views": source / "g04_views.npz",
            "source_checkpoint": N6D_RAW / G04 / "p22" / f"seed_{seed}/attempt_001/model_final.pt",
            "source_manifest": N6D_RAW / G04 / "p22" / f"seed_{seed}/attempt_001/run_manifest.json",
            "teacher_checkpoint": N6D_RAW / G00 / "p22" / f"seed_{seed}/attempt_001/model_final.pt",
            "teacher_manifest": N6D_RAW / G00 / "p22" / f"seed_{seed}/attempt_001/run_manifest.json",
            "teacher_affinity": teacher_transform / "affinity.npz",
            "teacher_embedding": adapter / "worker/embedding.npy",
            "teacher_adapter_manifest": adapter / "worker/training_manifest.json",
            "u00_affinity": source / "s04.npz",
            "target_graph_dir": N6D_CACHE / "graphs/p22" / G00,
            "source_graph_dir": N6D_CACHE / "graphs/p22" / G04,
        }
    if dataset == "misar":
        base = N8B_RAW / "formal/base" / f"seed_{seed}/attempt_001"
        adapter = N8B_RAW / "formal/adapter/formal" / f"seed_{seed}/attempt_001"
        return {
            "dataset": dataset, "seed": seed, "K": 12,
            "ids": N8B_RAW / "formal/adapter/inputs" / f"seed_{seed}/observation_ids.txt",
            "coordinates": N8B_RAW / "cache/base/coordinates.npy",
            "source_views": base / G04 / "views.npz",
            "source_checkpoint": base / G04 / "model_final.pt",
            "source_manifest": base / G04 / "training_manifest.json",
            "teacher_checkpoint": base / G00 / "model_final.pt",
            "teacher_manifest": base / G00 / "training_manifest.json",
            # Reuse the exact affinity carriers named by the locked Night-8B
            # uniform-head transform manifests.  The recovery partition
            # directory intentionally contains clusters/manifests only.
            "teacher_affinity": N8B_RAW / f"formal/transforms/F00/seed_{seed}/affinity.npz",
            "teacher_partition": N8B_HEAD / f"partitions/HR_F00/seed_{seed}/clusters.csv",
            "teacher_partition_manifest": N8B_HEAD / f"partitions/HR_F00/seed_{seed}/transform_manifest.json",
            "teacher_embedding": adapter / "worker/embedding.npy",
            "teacher_adapter_manifest": adapter / "worker/training_manifest.json",
            "u00_affinity": N8B_RAW / f"formal/adapter/inputs/seed_{seed}/s04.npz",
            "u00_partition": N8B_HEAD / f"partitions/HR_U00/seed_{seed}/clusters.csv",
            "u00_partition_manifest": N8B_HEAD / f"partitions/HR_U00/seed_{seed}/transform_manifest.json",
            "target_graph_dir": N8B_RAW / "cache/graphs" / G00,
            "source_graph_dir": N8B_RAW / "cache/graphs" / G04,
        }
    raise KeyError(dataset)


def load_checkpoint_state(path: Path) -> tuple[dict, dict]:
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict) or not state:
        raise RuntimeError(f"checkpoint lacks model_state_dict: {path}")
    return state, checkpoint


def make_trainer(dataset: str, graph_id: str, seed: int, epochs: int):
    if graph_id not in (G00, G04):
        raise KeyError(graph_id)
    if dataset == "p22":
        prepared = load_cache(P22_BASE)
        data, manifest = load_graph_data(prepared, N6D_CACHE / "graphs/p22" / graph_id)
        cfg = dict(N6D_CFG["p22"]); cfg["epochs"] = int(epochs)
    elif dataset == "misar":
        prepared, data, manifest = load_misar_graph(graph_id)
        cfg = dict(MISAR_CFG); cfg["epochs"] = int(epochs)
    else:
        raise KeyError(dataset)
    trainer = Night5ATrainer(data, cfg, BASE_C04, int(seed), torch.device("cuda:0"), {}, 1e-12)
    return prepared, data, manifest, trainer, cfg


class WarmTransferredTrainer(Night5ATrainer):
    def __init__(self, *args, transferred_state, **kwargs):
        self._transferred_state = transferred_state
        super().__init__(*args, **kwargs)

    def new_model(self):
        model = super().new_model()
        model.load_state_dict(self._transferred_state, strict=True)
        return model


def strict_transfer_model(dataset: str, seed: int, epochs: int):
    paths = dataset_paths(dataset, seed)
    source_state, source_checkpoint = load_checkpoint_state(paths["source_checkpoint"])
    prepared, data, graph_manifest, template, cfg = make_trainer(dataset, G00, seed, epochs)
    trainer = WarmTransferredTrainer(data, cfg, BASE_C04, int(seed), torch.device("cuda:0"),
                                     {}, 1e-12, transferred_state=source_state)
    model = trainer.new_model()
    source_state_sha = source_checkpoint.get("final_tensor_state_sha256") or source_checkpoint.get(
        "canonical_tensor_state_sha256")
    if source_state_sha and model_state_sha256(model) != source_state_sha:
        raise RuntimeError("warm-start initial state does not equal source G04 final state")
    return prepared, data, graph_manifest, trainer, model, source_checkpoint, cfg


def reference_root(dataset: str, seed: int) -> Path:
    return RAW / "p0/references" / dataset / f"seed_{seed}"


def load_clusters(path: Path, expected_ids: np.ndarray) -> np.ndarray:
    table = pd.read_csv(path)
    ids = table["observation_id"].astype(str).to_numpy()
    if not np.array_equal(ids, expected_ids):
        raise RuntimeError("partition observation order mismatch")
    return table["cluster"].to_numpy(dtype=np.int64)


def build_references() -> list[dict]:
    rows = []
    for dataset in ("p22", "misar"):
        for seed in range(10):
            paths = dataset_paths(dataset, seed)
            ids = read_ids(paths["ids"])
            target = reference_root(dataset, seed)
            if target.exists():
                raise RuntimeError(f"refusing to overwrite reference: {target}")
            target.mkdir(parents=True)
            method_rows = []
            for method, affinity_path in (("U00", paths["u00_affinity"]),
                                          ("FULL_F00", paths["teacher_affinity"])):
                affinity = sp.load_npz(affinity_path)
                labels, head = eigen_kmeans100(affinity, paths["K"])
                mdir = target / method; mdir.mkdir()
                save_clusters(mdir / "clusters.csv", ids, labels)
                atomic_json(mdir / "reference_manifest.json", {
                    "status": "success", "dataset": dataset, "seed": seed,
                    "method": method, "K": paths["K"], "label_access": False,
                    "source_affinity": file_row(affinity_path),
                    "clusters": file_row(mdir / "clusters.csv"), "head": head,
                })
                if dataset == "misar":
                    locked_path = paths["u00_partition"] if method == "U00" else paths["teacher_partition"]
                    locked_manifest_path = (paths["u00_partition_manifest"] if method == "U00"
                                            else paths["teacher_partition_manifest"])
                    locked_manifest = json.loads(locked_manifest_path.read_text())
                    if Path(locked_manifest["input_path"]) != affinity_path:
                        raise RuntimeError(f"MISAR locked affinity path drift: {method} seed {seed}")
                    if locked_manifest["input_file_sha256"] != sha256_file(affinity_path):
                        raise RuntimeError(f"MISAR locked affinity SHA drift: {method} seed {seed}")
                    locked = load_clusters(locked_path, ids)
                    if not exact_partition_equal(labels, locked):
                        raise RuntimeError(f"MISAR locked uniform head parity failed: {method} seed {seed}")
                method_rows.append(file_row(mdir / "reference_manifest.json"))
            rows.append({"dataset": dataset, "seed": seed, "methods": method_rows})
    atomic_json(OUT / "p0_reference_manifest.json", {
        "status": "PASS", "row_count": 20, "label_access": False,
        "misar_Y_access": 0, "rows": rows,
    })
    return rows


def used_authority_files() -> list[Path]:
    files = []
    for dataset in ("p22", "misar"):
        for seed in range(10):
            row = dataset_paths(dataset, seed)
            for key in ("ids", "coordinates", "source_views", "source_checkpoint",
                        "source_manifest", "teacher_checkpoint", "teacher_manifest",
                        "teacher_affinity", "teacher_embedding", "teacher_adapter_manifest",
                        "u00_affinity"):
                files.append(Path(row[key]))
            for graph in (row["target_graph_dir"], row["source_graph_dir"]):
                files.extend(sorted(graph.glob("*")))
            if dataset == "misar":
                files.extend([row["teacher_partition"], row["u00_partition"],
                              row["teacher_partition_manifest"], row["u00_partition_manifest"]])
    unique = []
    seen = set()
    for path in files:
        if path not in seen:
            seen.add(path); unique.append(path)
    return unique


def authority_snapshot(path: Path) -> dict:
    rows = []
    for item in used_authority_files():
        if not item.is_file():
            raise RuntimeError(f"missing historical authority artifact: {item}")
        rows.append(file_row(item))
    payload = {"status": "PASS", "row_count": len(rows), "rows": rows,
               "misar_Y_access": 0, "lineage_raw_Y_read_count": 2}
    atomic_json(path, payload)
    return payload


def p0() -> None:
    registry = load_registry()
    if not TASKBOOK.is_file():
        raise RuntimeError("taskbook is missing from repository protocol directory")
    if git("rev-parse", f"{BASE_TAG}^{{commit}}") != BASE_COMMIT:
        raise RuntimeError("base tag peel mismatch")
    if git("merge-base", "--is-ancestor", BASE_COMMIT, "HEAD") not in ("",):
        raise RuntimeError("base is not an ancestor")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU card mode is required")
    gpu = torch.cuda.get_device_name(0)
    if "4080" not in gpu:
        raise RuntimeError(f"unexpected GPU: {gpu}")
    final_guard = json.loads((N8B_EVAL / "evaluation/final_Y_read_guard.json").read_text())
    if final_guard.get("after") != 2 or not final_guard.get("third_read_forbidden"):
        raise RuntimeError("MISAR final Y-read guard mismatch")
    before = authority_snapshot(RAW / "manifests/teacher_authority_before.json")
    compat = []
    for dataset in ("p22", "misar"):
        prepared, data, graph_manifest, trainer, model, checkpoint, cfg = strict_transfer_model(
            dataset, 0, 0)
        state = checkpoint["model_state_dict"]
        actual = model.state_dict()
        contracts = []
        for name in sorted(state):
            contracts.append({"name": name, "shape": list(state[name].shape),
                              "dtype": str(state[name].dtype),
                              "target_shape": list(actual[name].shape),
                              "target_dtype": str(actual[name].dtype)})
        model.eval()
        with torch.no_grad():
            views = forward_model(model, data, torch.device("cuda:0"))
        compat.append({
            "dataset": dataset, "seed": 0, "strict_load": True,
            "state_key_count": len(contracts), "state_contracts": contracts,
            "initial_state_equals_source_final": True,
            "view_shapes": {k: list(v.shape) for k, v in views.items()},
            "views_finite": all(np.isfinite(v).all() for v in views.values()),
            "graph_cache_sha256": graph_manifest["canonical_graph_cache_sha256"],
        })
    # Parameter-free candidates are checked by two independent calls.
    projection = []
    paths = dataset_paths("p22", 0)
    views = load_npz_views(paths["source_views"])
    p00 = load_projection_operators(paths["target_graph_dir"])
    p04 = load_projection_operators(paths["source_graph_dir"])
    for row in registry["candidates"][5:]:
        first = projection_views(row["id"], views, p00, p04)
        second = projection_views(row["id"], views, p00, p04)
        hashes1 = {k: array_sha(v) for k, v in first.items()}
        hashes2 = {k: array_sha(v) for k, v in second.items()}
        if hashes1 != hashes2:
            raise RuntimeError(f"projection determinism failed: {row['id']}")
        projection.append({"candidate_id": row["id"], "first": hashes1,
                           "second": hashes2, "exact": True})
    # Same head exactness is tested on one real P22 affinity.
    affinity = sp.load_npz(paths["u00_affinity"])
    labels1, head1 = eigen_kmeans100(affinity, paths["K"])
    labels2, head2 = eigen_kmeans100(affinity, paths["K"])
    if not np.array_equal(labels1, labels2):
        raise RuntimeError("head determinism failed")
    build_references()
    payload = {
        "status": "P0_AUTHORITY_SEMANTICS_PROFILE_PASS",
        "base_commit": BASE_COMMIT, "base_tag": BASE_TAG,
        "registry_sha256": sha256_file(REGISTRY), "taskbook_sha256": sha256_file(TASKBOOK),
        "candidate_count": len(registry["candidates"]), "candidate_ids": [x["id"] for x in registry["candidates"]],
        "compatibility": compat, "projection_determinism": projection,
        "head_determinism": {"exact": True, "first": head1, "second": head2},
        "teacher_authority_before_sha256": sha256_file(RAW / "manifests/teacher_authority_before.json"),
        "teacher_authority_rows": before["row_count"],
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "cuda_runtime": torch.version.cuda, "gpu": gpu,
                        "numpy": np.__version__, "scipy": scipy.__version__,
                        "sklearn": sklearn.__version__},
        "label_firewall": {"misar_Y_access": 0, "lineage_raw_Y_read_count": 2,
                           "third_read_forbidden": True, "p22_label_access": 0},
    }
    atomic_json(OUT / "p0_authority_semantics_profile.json", payload)
    print(json.dumps({"event": "p0_complete", "status": payload["status"]}))


def candidate_by_id(candidate_id: str) -> dict:
    values = [row for row in load_registry()["candidates"] if row["id"] == candidate_id]
    if len(values) != 1:
        raise KeyError(candidate_id)
    return values[0]


def gpu_peak() -> float:
    return float(torch.cuda.max_memory_allocated(0) / 1048576.0) if torch.cuda.is_available() else 0.0


def construct_candidate_views(dataset: str, seed: int, candidate: dict,
                              target: Path) -> tuple[dict, dict]:
    paths = dataset_paths(dataset, seed)
    source_views = load_npz_views(paths["source_views"])
    started = time.perf_counter()
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(0)
    if candidate["family"] == "precomputed_topology_projection":
        target_ops = load_projection_operators(paths["target_graph_dir"])
        source_ops = load_projection_operators(paths["source_graph_dir"])
        views = projection_views(candidate["id"], source_views, target_ops, source_ops)
        audit = {
            "route": "parameter_free_precomputed_topology_projection",
            "target_epochs": 0, "strict_load": False,
            "learned_parameter_count": 0, "model_state_saved": False,
            "projection_operator_sha256": {
                "target": {k: sparse_sha(v) for k, v in target_ops.items()},
                "source": {k: sparse_sha(v) for k, v in source_ops.items()},
            },
        }
    else:
        epochs = int(candidate["target_epochs"])
        prepared, data, graph_manifest, trainer, model, checkpoint, cfg = strict_transfer_model(
            dataset, seed, epochs)
        source_final = checkpoint.get("final_tensor_state_sha256") or checkpoint.get(
            "canonical_tensor_state_sha256")
        initial = model_state_sha256(model)
        if source_final and initial != source_final:
            raise RuntimeError("transferred initial state mismatch")
        if epochs == 0:
            model.eval()
            with torch.no_grad():
                views = forward_model(model, data, torch.device("cuda:0"))
            final = initial; logs = []
        else:
            result = trainer.train()
            model = result.model
            views = forward_model(model, data, torch.device("cuda:0"))
            final = model_state_sha256(model); logs = result.logs
            pd.DataFrame(logs).to_csv(target / "transfer_loss_curve.csv", index=False)
        checkpoint_out = target / "topology_expert_model_final.pt"
        atomic_torch(checkpoint_out, {
            "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "candidate_id": candidate["id"], "seed": seed,
            "target_epochs": epochs, "initial_state_sha256": initial,
            "final_state_sha256": final, "source_checkpoint_sha256": sha256_file(paths["source_checkpoint"]),
        })
        audit = {
            "route": "strict_checkpoint_transfer_then_fixed_finetune",
            "target_epochs": epochs, "strict_load": True,
            "initial_state_equals_source_final": True,
            "initial_state_sha256": initial, "final_state_sha256": final,
            "checkpoint": file_row(checkpoint_out), "fixed_final_epoch": True,
            "early_stopping": False, "scheduler": None, "optimizer": "Adam",
            "learning_rate": 1e-4, "weight_decay": 0.0,
            "recorded_steps": [int(x["step"]) for x in logs],
        }
    runtime = float(time.perf_counter() - started)
    if any(not np.isfinite(value).all() for value in views.values()):
        raise RuntimeError("candidate topology expert produced non-finite views")
    audit.update({"runtime_seconds": runtime, "peak_gpu_mib": gpu_peak(),
                  "view_array_sha256": {k: array_sha(v) for k, v in views.items()}})
    return views, audit


def adapter_run(dataset: str, seed: int, target: Path, candidate_views: dict,
                source_views: dict, ids: np.ndarray, coords: np.ndarray) -> tuple[sp.csr_matrix, dict]:
    values, c06, _ = base_affinities(candidate_views, source_views, ids.tolist(), coords)
    inputs = target / "adapter_inputs"; inputs.mkdir()
    atomic_npz(inputs / "candidate_views.npz", candidate_views)
    atomic_npz(inputs / "source_views.npz", source_views)
    for name, value in (("s00.npz", values["S_G00"]), ("s04.npz", values["S_G04"]),
                        ("c06_affinity.npz", c06)):
        atomic_sparse(inputs / name, value)
    (inputs / "observation_ids.txt").write_text("\n".join(ids.tolist()) + "\n")
    worker = {
        "unit_id": f"night9a_{seed:02d}", "K": 9 if dataset == "p22" else 12,
        "observation_count": len(ids), "ordered_observation_sha256": observation_sha(ids),
        "g00_views": str(inputs / "candidate_views.npz"),
        "g04_views": str(inputs / "source_views.npz"),
        "observation_ids": str(inputs / "observation_ids.txt"),
        "s00": str(inputs / "s00.npz"), "s04": str(inputs / "s04.npz"),
        "pseudo_partition": "", "pseudo_affinity": "",
    }
    atomic_json(inputs / "worker_input.json", worker)
    config = {"recipe_id": "R02", "losses": ["RECON", "MNN"], "fusion": "equal",
              "seed": seed, "epochs": 160}
    atomic_json(target / "adapter_config.json", config)
    output = target / "adapter_worker"
    command = [sys.executable, str(REPO / "scripts/night7b_train.py"), "train",
               "--unit-dir", str(inputs), "--config", str(target / "adapter_config.json"),
               "--output", str(output)]
    started = time.perf_counter()
    subprocess.run(command, cwd=REPO, check=True)
    subprocess.run([sys.executable, str(REPO / "scripts/night7b_train.py"), "reload",
                    "--unit-dir", str(inputs), "--config", str(target / "adapter_config.json"),
                    "--output", str(output)], cwd=REPO, check=True)
    elapsed = float(time.perf_counter() - started)
    manifest = json.loads((output / "training_manifest.json").read_text())
    reload_audit = json.loads((output / "reload_forward_audit.json").read_text())
    if reload_audit.get("status") != "PASS":
        raise RuntimeError("R02 adapter reload audit failed")
    embedding = np.load(output / "embedding.npy", allow_pickle=False)
    affinity = adapter_endpoint(embedding, c06, ids.tolist())
    return affinity, {
        "recipe_id": "R02", "losses": ["RECON", "MNN"], "fusion": "equal",
        "epochs": 160, "optimizer": "AdamW", "learning_rate": 1e-3,
        "weight_decay": 1e-5, "checkpoint_policy": "fixed_final_epoch",
        "runtime_seconds": elapsed, "worker_manifest": file_row(output / "training_manifest.json"),
        "reload_audit": file_row(output / "reload_forward_audit.json"),
        "embedding": file_row(output / "embedding.npy"),
        "peak_gpu_mib": float(manifest.get("peak_gpu_mib", gpu_peak())),
        "label_access": False,
    }


def one_chain(stage: str, dataset: str, seed: int, candidate_id: str,
              root: Path, smoke: bool) -> None:
    candidate = candidate_by_id(candidate_id)
    target = root / stage / candidate_id / dataset / f"seed_{seed}/attempt_001"
    if target.exists():
        raise RuntimeError(f"refusing to overwrite chain: {target}")
    target.mkdir(parents=True)
    started = time.perf_counter()
    paths = dataset_paths(dataset, seed)
    ids = read_ids(paths["ids"])
    coords = np.load(paths["coordinates"], allow_pickle=False)
    source_views = load_npz_views(paths["source_views"])
    if any(value.shape[0] != len(ids) for value in source_views.values()) or len(coords) != len(ids):
        raise RuntimeError("chain input observation order/size mismatch")
    candidate_views, transfer = construct_candidate_views(dataset, seed, candidate, target)
    atomic_npz(target / "candidate_virtual_g00_views.npz", candidate_views)
    affinity, adapter = adapter_run(dataset, seed, target, candidate_views, source_views, ids, coords)
    atomic_sparse(target / "affinity.npz", affinity)
    labels, head = eigen_kmeans100(affinity, paths["K"])
    save_clusters(target / "clusters.csv", ids, labels)
    teacher_affinity = sp.load_npz(paths["teacher_affinity"])
    teacher_labels = load_clusters(reference_root(dataset, seed) / "FULL_F00/clusters.csv", ids)
    fidelity = affinity_fidelity(affinity, teacher_affinity, labels, teacher_labels)
    source_manifest = json.loads(paths["source_manifest"].read_text())
    teacher_manifest = json.loads(paths["teacher_manifest"].read_text())
    source_seconds = float(source_manifest["runtime_seconds"])
    teacher_seconds = float(teacher_manifest["runtime_seconds"])
    u00_ref = json.loads((reference_root(dataset, seed) / "U00/reference_manifest.json").read_text())
    # Night-6D and Night-8B used different field names for the same measured
    # CUDA allocated-memory envelope.  Both are historical authority fields;
    # fail closed instead of silently treating a missing measurement as zero.
    source_peak_value = source_manifest.get(
        "peak_gpu_mib", source_manifest.get("peak_gpu_allocated_mib"))
    if source_peak_value is None:
        raise RuntimeError("source manifest lacks a measured CUDA peak")
    source_peak = float(source_peak_value)
    candidate_peak = max(source_peak, float(transfer["peak_gpu_mib"]), float(adapter["peak_gpu_mib"]))
    candidate_e2e = source_seconds + float(transfer["runtime_seconds"]) + float(adapter["runtime_seconds"]) + float(head["runtime_seconds"])
    u00_e2e = source_seconds + float(u00_ref["head"]["runtime_seconds"])
    manifest = {
        "schema_version": 1, "status": "smoke_invalid_for_science" if smoke else "success",
        "stage": stage, "candidate": candidate, "candidate_id": candidate_id,
        "dataset": dataset, "seed": seed, "K": paths["K"],
        "scientific_chain": not smoke, "scientific_retry": 0, "fallback": 0,
        "label_access": {"p22": 0, "misar_Y": 0}, "cuda_training_used": True,
        "source_backbone": {"graph_id": G04, "checkpoint": file_row(paths["source_checkpoint"]),
                            "manifest": file_row(paths["source_manifest"]),
                            "historical_runtime_seconds": source_seconds,
                            "historical_peak_gpu_mib": source_peak},
        "full_teacher": {"graph_id": G00, "checkpoint": file_row(paths["teacher_checkpoint"]),
                         "manifest": file_row(paths["teacher_manifest"]),
                         "historical_runtime_seconds": teacher_seconds},
        "topology_expert": transfer, "adapter": adapter, "head": head,
        "fidelity_vs_full_f00": fidelity,
        "artifacts": {
            "views": file_row(target / "candidate_virtual_g00_views.npz"),
            "affinity": file_row(target / "affinity.npz"),
            "clusters": file_row(target / "clusters.csv"),
        },
        "resource": {
            "actual_incremental_wall_seconds": float(time.perf_counter() - started),
            "candidate_end_to_end_seconds": candidate_e2e,
            "u00_end_to_end_seconds": u00_e2e,
            "runtime_ratio_vs_u00": candidate_e2e / u00_e2e,
            "candidate_peak_gpu_mib": candidate_peak,
            "u00_peak_gpu_mib": source_peak,
            "peak_gpu_ratio_vs_u00": candidate_peak / max(source_peak, 1e-12),
            "checkpoint_reuse_does_not_make_backbone_cost_zero": True,
        },
        "ordered_observation_sha256": observation_sha(ids),
        "code_commit": git("rev-parse", "HEAD"),
    }
    atomic_json(target / "chain_manifest.json", manifest)
    print(json.dumps({"event": "chain_complete", "stage": stage,
                      "candidate": candidate_id, "dataset": dataset, "seed": seed,
                      "status": manifest["status"],
                      "runtime_ratio": manifest["resource"]["runtime_ratio_vs_u00"]}))


def stage_plan(stage: str) -> list[tuple[str, str, int]]:
    registry = load_registry()
    if stage == "R1":
        candidates = [row["id"] for row in registry["candidates"]]
        seeds = [0, 1, 2]
    elif stage == "R2":
        lock = json.loads((OUT / "R1_shortlist.json").read_text())
        candidates = lock["shortlist_candidate_ids"]; seeds = [3, 4]
    elif stage == "R3":
        lock = json.loads((OUT / "night9a_final_candidate_lock.json").read_text())
        candidates = [lock["final_candidate_id"]]; seeds = [5, 6, 7, 8, 9]
    else:
        raise KeyError(stage)
    return [(candidate, dataset, seed) for candidate in candidates
            for dataset in ("p22", "misar") for seed in seeds]


def existing_attempt_count() -> int:
    return sum(1 for _ in (RAW / "formal").glob("R*/E*/**/attempt_001"))


def run_stage(stage: str) -> None:
    if json.loads((OUT / "p0_authority_semantics_profile.json").read_text())["status"] != "P0_AUTHORITY_SEMANTICS_PROFILE_PASS":
        raise RuntimeError("P0 gate not passed")
    if git("status", "--porcelain", "--", "SpaLORA", "scripts", "tests"):
        raise RuntimeError("scientific code must be committed before formal chains")
    plan = stage_plan(stage)
    expected = {"R1": 54, "R2": 12, "R3": 10}[stage]
    if len(plan) > expected:
        raise RuntimeError("stage formal chain budget exceeded")
    rows = []
    for ordinal, (candidate, dataset, seed) in enumerate(plan, 1):
        target = RAW / "formal" / stage / candidate / dataset / f"seed_{seed}/attempt_001"
        print(json.dumps({"event": "chain_start", "stage": stage, "ordinal": ordinal,
                          "planned": len(plan), "candidate": candidate,
                          "dataset": dataset, "seed": seed}), flush=True)
        command = [sys.executable, str(Path(__file__).resolve()), "chain",
                   "--stage", stage, "--candidate", candidate, "--dataset", dataset,
                   "--seed", str(seed), "--root", str(RAW / "formal")]
        started = time.perf_counter()
        proc = subprocess.Popen(command, cwd=REPO, start_new_session=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        output = ""
        try:
            output, _ = proc.communicate(timeout=1200)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                more, _ = proc.communicate(timeout=60); output += more
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL); more, _ = proc.communicate(); output += more
            proc.returncode = 124
        log = RAW / "logs" / f"{stage}_{ordinal:03d}_{candidate}_{dataset}_seed{seed}.log"
        log.parent.mkdir(parents=True, exist_ok=True); log.write_text(output)
        manifest_path = target / "chain_manifest.json"
        if proc.returncode == 0 and manifest_path.is_file():
            row = json.loads(manifest_path.read_text())
            row["chain_manifest_sha256"] = sha256_file(manifest_path)
        else:
            partial = []
            if target.exists():
                partial = [file_row(path) for path in sorted(target.rglob("*")) if path.is_file()]
            failure = {
                "status": "RESOURCE_CENSORED_WALLTIME_20M" if proc.returncode == 124 else "FAILED_NO_RETRY",
                "stage": stage, "candidate_id": candidate, "dataset": dataset, "seed": seed,
                "returncode": proc.returncode, "scientific_retry": 0, "fallback": 0,
                "elapsed_seconds": float(time.perf_counter() - started),
                "log": file_row(log), "partial_artifacts": partial,
                "label_access": {"p22": 0, "misar_Y": 0},
            }
            failure_path = RAW / "formal" / stage / "failures" / f"{candidate}_{dataset}_seed{seed}.json"
            atomic_json(failure_path, failure); row = {**failure, "failure_manifest": file_row(failure_path)}
        rows.append(row)
        if existing_attempt_count() > 80:
            raise RuntimeError("Night-9A total attempt budget exceeded")
    lock = {
        "status": "LOCKED", "stage": stage, "locked_before_p22_label_access": True,
        "planned_chain_count": len(plan), "formal_chain_count": len(rows),
        "success_count": sum(row["status"] == "success" for row in rows),
        "failure_count": sum(row["status"] != "success" for row in rows),
        "scientific_retry": 0, "fallback": 0, "rows": rows,
        "label_access": {"p22": 0, "misar_Y": 0},
        "registry_sha256": sha256_file(REGISTRY), "code_commit": git("rev-parse", "HEAD"),
    }
    atomic_json(OUT / f"locked_{stage}_manifest.json", lock)
    print(json.dumps({"event": "stage_lock", "stage": stage,
                      "success": lock["success_count"], "failures": lock["failure_count"]}))


def smoke() -> None:
    rows = []
    for candidate in [row["id"] for row in load_registry()["candidates"]]:
        started = time.perf_counter()
        try:
            one_chain("SMOKE", "p22", 0, candidate, RAW / "smoke", True)
            manifest = RAW / "smoke/SMOKE" / candidate / "p22/seed_0/attempt_001/chain_manifest.json"
            rows.append({"candidate_id": candidate, "status": "PASS",
                         "elapsed_seconds": time.perf_counter() - started,
                         "manifest": file_row(manifest)})
        except Exception as exc:
            rows.append({"candidate_id": candidate, "status": "FAIL",
                         "exception_type": type(exc).__name__, "message": str(exc),
                         "traceback": traceback.format_exc(),
                         "elapsed_seconds": time.perf_counter() - started})
            break
    status = "PASS" if len(rows) == 9 and all(x["status"] == "PASS" for x in rows) else "FAIL"
    atomic_json(OUT / "p0_real_seed_smoke_profile.json", {
        "status": status, "scientific_chain_count": 0, "label_access": False,
        "dataset": "p22", "seed": 0, "rows": rows,
    })
    if status != "PASS":
        raise RuntimeError("real-seed P0 smoke failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("p0")
    sub.add_parser("references")
    sub.add_parser("smoke")
    chain = sub.add_parser("chain")
    chain.add_argument("--stage", required=True); chain.add_argument("--candidate", required=True)
    chain.add_argument("--dataset", choices=("p22", "misar"), required=True)
    chain.add_argument("--seed", type=int, required=True); chain.add_argument("--root", type=Path, required=True)
    chain.add_argument("--smoke", action="store_true")
    stage = sub.add_parser("stage"); stage.add_argument("stage", choices=("R1", "R2", "R3"))
    args = parser.parse_args()
    if args.mode == "p0": p0()
    elif args.mode == "references": build_references()
    elif args.mode == "smoke": smoke()
    elif args.mode == "chain": one_chain(args.stage, args.dataset, args.seed,
                                           args.candidate, args.root, args.smoke)
    else: run_stage(args.stage)


if __name__ == "__main__":
    main()
