from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import pathlib
import subprocess
import sys
import time
from dataclasses import asdict, fields
from typing import Optional

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from SpaLORA.night10a_qcrd import (
    EPS, MASKED_CANDIDATES, FrozenQuality, QCRDAdapter,
    canonical_array_sha256, canonical_sparse_sha256, canonical_state_sha256,
    corrected_views, deterministic_mask, fourier_coordinates, frozen_quality,
    frozen_reference_harmonizer, qcrd_forward, row_normalize,
)
from SpaLORA.night6c_pipeline import self_tuning_affinity, spectral
from SpaLORA.night7b_adaptive import row_sparse_strict, sym_zero


REPO = pathlib.Path("/root/autodl-fs/SpaLORA-night10a-rev1")
RAW = pathlib.Path("/root/autodl-fs/night10a_rev2_qcrd_20260821")
REV1_RAW = pathlib.Path("/root/autodl-fs/night10a_rev1_qcrd_20260821")
SOURCE = pathlib.Path("/root/autodl-fs/night7b_score_rnd_20260818")
PYTHON = "/root/miniconda3/envs/SpaLORA/bin/python"
CONTRACT = REPO / "protocols/night10a_rev2/night10a_qcrd_rev2_dimension_and_reuse_contract.json"
PREFLIGHT = RAW / "p0_rev2/real_runtime_preflight.json"
REV1_ARTIFACT_MANIFEST = REV1_RAW / "official_compact/handoff/raw_artifact_manifest.csv"
CANDIDATES = [
    "Q01_GLOBAL_QUALITY_BLEND", "Q02_SPOT_QUALITY_BLEND", "Q03_MASKED_RESIDUAL",
    "Q04_SPOT_GATED_MASKED_RESIDUAL", "Q05_BOUNDARY_GATED_RESIDUAL",
    "Q06_COORDINATE_PRIOR_RESIDUAL", "Q07_CONFIDENCE_MNN_RESIDUAL",
]
DATA = {
    "a1": {"base": 0, "K": 10, "family": "RNA+protein", "r1": range(3), "r2": range(5)},
    "tonsil": {"base": 5, "K": 4, "family": "RNA+protein", "r1": range(3), "r2": range(5)},
    "d1": {"base": 10, "K": 10, "family": "RNA+protein", "r1": range(3), "r2": range(10)},
    "p22": {"base": 20, "K": 9, "family": "RNA+ATAC", "r1": range(3), "r2": range(10)},
}
LOSS_WEIGHTS = {"align": 1.0, "mask": 1.0, "anchor": .25, "correction": .05, "boundary": .25, "mnn": .10}
ACCEPTED_TRAINING_STATUSES = {"CHECKPOINT_ROUNDTRIP_PASS", "REUSED_CHECKPOINT_ROUNDTRIP_PASS"}
ACCEPTED_TRANSFORM_STATUSES = {"PASS", "REUSED_PASS"}


def sha(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def atomic_json(path: pathlib.Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_npz(path: pathlib.Path, **values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp = pathlib.Path(str(path) + ".tmp.npz")
    np.savez_compressed(tmp, **values); os.replace(tmp, path)


def canonical_json_sha256(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def rev1_artifact_index() -> dict[str, dict]:
    rows = {}
    with REV1_ARTIFACT_MANIFEST.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[row["relative_path"]] = row
    return rows


def verify_rev1_artifact(path: pathlib.Path, index: Optional[dict[str, dict]] = None) -> dict:
    if index is None: index = rev1_artifact_index()
    relative = str(path.relative_to(REV1_RAW)).replace("\\", "/")
    expected = index.get(relative)
    if expected is None: raise AssertionError(f"REV1 artifact is absent from preserved manifest: {relative}")
    actual_size = path.stat().st_size; actual_sha = sha(path)
    if actual_size != int(expected["size"]) or actual_sha != expected["sha256"]:
        raise AssertionError(f"REV1 artifact drift: {relative}")
    return {"relative_path": relative, "size": actual_size, "sha256": actual_sha}


def unit(dataset: str, seed: int) -> str:
    return f"u{DATA[dataset]['base'] + seed:03d}"


def source_spec(unit_id: str) -> dict:
    return json.loads((SOURCE / f"source/{unit_id}/worker_input.json").read_text())


def ids(unit_id: str) -> list[str]:
    return (SOURCE / f"source/{unit_id}/observation_ids.txt").read_text().splitlines()


def views(unit_id: str):
    x = np.load(SOURCE / f"source/{unit_id}/g04_views.npz")
    return tuple(np.asarray(x[k], dtype=np.float32) for k in ("emb_latent_omics1", "emb_latent_omics2", "SpaLORA_fused"))


def graph_path(dataset: str) -> pathlib.Path:
    cache = "night6c_cache_20260817" if dataset in {"a1", "tonsil"} else "night6d_cache_20260817"
    return pathlib.Path("/root/autodl-fs") / cache / f"graphs/{dataset}/G04_SP10_F10_EUC_UNION/adj_spatial_omics1_support.npz"


def coords_path(dataset: str, unit_id: Optional[str] = None) -> pathlib.Path:
    # Night-7B source packs lock the exact ordered coordinates for every unit.
    if unit_id is None: unit_id = unit(dataset, 0)
    return SOURCE / f"source/{unit_id}/coordinates.npy"


def r02_root(unit_id: str) -> pathlib.Path:
    for stage in ("R1", "R2"):
        path = SOURCE / f"adapter_stage/{stage}/formal/R02/{unit_id}/attempt_001"
        if (path/"worker/embedding.npy").exists() and (path/"transforms/E1_ADAPTER_C06_MEAN/H01/clusters.csv").exists():
            return path
    raise FileNotFoundError(f"no complete authoritative R02 endpoint for {unit_id}")


def read_cluster(path: pathlib.Path) -> tuple[list[str], np.ndarray]:
    names, labels = [], []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle): names.append(row["observation_id"]); labels.append(int(row["cluster"]))
    return names, np.asarray(labels, dtype=np.int64)


def reference(unit_id: str, family: str):
    if family == "RNA+protein":
        return views(unit_id)[2], np.load(SOURCE / f"source/{unit_id}/c00_partition.npy"), None
    root = r02_root(unit_id)
    zf = np.load(root / "worker/embedding.npy").astype(np.float32)
    names, part = read_cluster(root / "transforms/E1_ADAPTER_C06_MEAN/H01/clusters.csv")
    if names != ids(unit_id): raise AssertionError("R02 order mismatch")
    return zf, part, SOURCE / f"source/{unit_id}/c06_affinity.npz"


def quality_dir(unit_id: str) -> pathlib.Path:
    return RAW / f"raw/inputs/{unit_id}"


def atomic_npy(path: pathlib.Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = pathlib.Path(str(path) + ".tmp.npy"); np.save(tmp, value); os.replace(tmp, path)


def _assert_real_array(name: str, value: np.ndarray, n: int, columns: Optional[int] = None) -> None:
    if not isinstance(value, np.ndarray) or value.ndim != 2:
        raise AssertionError(f"{name} must be a two-dimensional numpy array")
    if value.shape[0] != n or value.shape[1] <= 0:
        raise AssertionError(f"{name} shape mismatch: {value.shape}")
    if columns is not None and value.shape[1] != columns:
        raise AssertionError(f"{name} feature dimension mismatch: {value.shape[1]} != {columns}")
    if not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all():
        raise AssertionError(f"{name} dtype/finite gate failed")


def _quality_arrays(quality: FrozenQuality) -> dict[str, np.ndarray]:
    arrays = {}
    for field in fields(FrozenQuality):
        value = getattr(quality, field.name)
        if isinstance(value, np.ndarray): arrays[field.name] = value
    arrays["zero_degree_count"] = np.asarray(quality.zero_degree_count, dtype=np.int64)
    return arrays


def _prepare_quality(dataset: str, seed: int, unit_id: str, z1: np.ndarray,
                     z2: np.ndarray, zf_aligned: np.ndarray, pf: np.ndarray,
                     graph: sp.spmatrix, names: list[str], out: pathlib.Path) -> dict:
    old = REV1_RAW / f"raw/inputs/{unit_id}"
    if (old / "input_manifest.json").is_file():
        index = rev1_artifact_index()
        evidence = [verify_rev1_artifact(old / name, index) for name in
                    ("input_manifest.json", "p1.npy", "p2.npy", "pf.npy", "quality.npz")]
        old_manifest = json.loads((old / "input_manifest.json").read_text(encoding="utf-8"))
        if old_manifest["unit_id"] != unit_id or int(old_manifest["seed"]) != seed:
            raise AssertionError("REV1 frozen-quality identity mismatch")
        for name, key in (("p1.npy", "p1_sha256"), ("p2.npy", "p2_sha256"),
                          ("pf.npy", "pf_sha256"), ("quality.npz", "quality_sha256")):
            if sha(old / name) != old_manifest[key]:
                raise AssertionError(f"REV1 frozen-quality declared SHA mismatch: {name}")
        if not np.array_equal(np.load(old / "pf.npy"), pf):
            raise AssertionError("REV1 reference partition drift")
        return {
            "quality_source_mode": "REV1_VERIFIED_REUSE", "quality_file": str(old / "quality.npz"),
            "quality_sha256": sha(old / "quality.npz"), "p1_file": str(old / "p1.npy"),
            "p1_sha256": sha(old / "p1.npy"), "p2_file": str(old / "p2.npy"),
            "p2_sha256": sha(old / "p2.npy"), "pf_file": str(old / "pf.npy"),
            "pf_sha256": sha(old / "pf.npy"), "rev1_preserved_evidence": evidence,
            "zero_degree_count": int(old_manifest["zero_degree_count"]),
            "quality_diagnostics": old_manifest.get("quality_diagnostics", {}),
        }
    a1 = self_tuning_affinity(row_normalize(z1), 10, names)
    p1 = spectral(a1, DATA[dataset]["K"])
    a2 = self_tuning_affinity(row_normalize(z2), 10, names)
    p2 = spectral(a2, DATA[dataset]["K"])
    quality = frozen_quality(z1, z2, zf_aligned, p1, p2, graph, DATA[dataset]["K"], 10)
    atomic_npy(out / "p1.npy", p1); atomic_npy(out / "p2.npy", p2); atomic_npy(out / "pf.npy", pf)
    atomic_npz(out / "quality.npz", **_quality_arrays(quality))
    return {
        "quality_source_mode": "REV2_COMPUTED_LABEL_FREE", "quality_file": str(out / "quality.npz"),
        "quality_sha256": sha(out / "quality.npz"), "p1_file": str(out / "p1.npy"),
        "p1_sha256": sha(out / "p1.npy"), "p2_file": str(out / "p2.npy"),
        "p2_sha256": sha(out / "p2.npy"), "pf_file": str(out / "pf.npy"),
        "pf_sha256": sha(out / "pf.npy"), "rev1_preserved_evidence": [],
        "zero_degree_count": int(quality.zero_degree_count),
        "quality_diagnostics": quality.diagnostics,
    }


def validate_input_manifest(manifest: dict) -> None:
    required_files = [
        (manifest["views_file"], manifest["views_file_sha256"]),
        (manifest["zf_raw_file"], manifest["zf_raw_file_sha256"]),
        (manifest["spatial_graph_file"], manifest["spatial_graph_file_sha256"]),
        (manifest["coordinates_file"], manifest["coordinates_file_sha256"]),
        (manifest["quality_file"], manifest["quality_sha256"]),
        (manifest["p1_file"], manifest["p1_sha256"]),
        (manifest["p2_file"], manifest["p2_sha256"]),
        (manifest["pf_file"], manifest["pf_sha256"]),
    ]
    for name, expected in required_files:
        path = pathlib.Path(name)
        if not path.is_file() or sha(path) != expected:
            raise AssertionError(f"registered input file drift: {path}")
    if manifest.get("c06_file"):
        path = pathlib.Path(manifest["c06_file"])
        if not path.is_file() or sha(path) != manifest["c06_sha256"]:
            raise AssertionError("registered C06 file drift")
    if manifest["harmonizer_mode"] == "rectangular_expansion":
        for key in ("projection", "zf_aligned"):
            path = pathlib.Path(manifest[f"{key}_file"])
            if not path.is_file() or sha(path) != manifest[f"{key}_file_sha256"]:
                raise AssertionError(f"frozen harmonizer file drift: {path}")
    if manifest["contract_sha256"] != sha(CONTRACT) or manifest["label_reads"] != 0:
        raise AssertionError("input authority or label firewall drift")
    names = ids(manifest["unit_id"])
    ordered = hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()
    if ordered != manifest["ordered_observation_sha256"]:
        raise AssertionError("ordered observation identity drift")
    z1, z2, _ = views(manifest["unit_id"])
    zf_raw, pf, _ = reference(manifest["unit_id"], manifest["family"])
    if (canonical_array_sha256(z1) != manifest["z1_canonical_sha256"] or
            canonical_array_sha256(z2) != manifest["z2_canonical_sha256"] or
            canonical_array_sha256(zf_raw) != manifest["zf_raw_canonical_sha256"] or
            canonical_array_sha256(np.asarray(pf)) != manifest["reference_partition_canonical_sha256"]):
        raise AssertionError("registered view/reference canonical SHA drift")
    graph = sp.load_npz(manifest["spatial_graph_file"]); coords = np.load(manifest["coordinates_file"])
    if (graph.shape != tuple(manifest["graph_shape"]) or
            canonical_sparse_sha256(graph) != manifest["spatial_graph_canonical_sha256"] or
            canonical_array_sha256(coords) != manifest["coordinates_canonical_sha256"]):
        raise AssertionError("registered graph/coordinate canonical SHA drift")
    if not (np.isfinite(z1).all() and np.isfinite(z2).all() and np.isfinite(zf_raw).all() and
            np.isfinite(graph.data).all() and np.isfinite(coords).all()):
        raise AssertionError("registered real input finite gate failed")
    quality = load_quality(pathlib.Path(manifest["quality_file"]))
    for field in fields(FrozenQuality):
        value = getattr(quality, field.name)
        if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise AssertionError(f"registered frozen quality non-finite: {field.name}")
    for name in ("local_residual_1", "local_residual_2", "entropy_1", "entropy_2",
                 "support_1", "support_2", "spot_logits", "spot_weights",
                 "disagreement", "fused_residual", "boundary_risk", "base_gate"):
        if len(getattr(quality, name)) != manifest["N"]:
            raise AssertionError(f"registered frozen quality cardinality drift: {name}")


def prepare_one(dataset: str, seed: int) -> dict:
    if dataset not in DATA or seed not in DATA[dataset]["r2"]:
        raise ValueError("unit is outside the preregistered REV2 matrix")
    unit_id = unit(dataset, seed); out = quality_dir(unit_id); manifest_file = out / "input_manifest.json"
    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text(encoding="utf-8")); validate_input_manifest(manifest)
        return manifest
    out.mkdir(parents=True, exist_ok=True)
    z1, z2, _ = views(unit_id); zf_raw, pf, c06 = reference(unit_id, DATA[dataset]["family"])
    names = ids(unit_id); n = len(names); graph_file = graph_path(dataset); coord_file = coords_path(dataset, unit_id)
    graph = sp.load_npz(graph_file); coords = np.load(coord_file)
    _assert_real_array("z1", z1, n); _assert_real_array("z2", z2, n, z1.shape[1])
    _assert_real_array("zf_raw", zf_raw, n); _assert_real_array("coordinates", coords, n)
    if coords.shape[1] != 2 or len(pf) != n or not np.isfinite(pf).all():
        raise AssertionError("coordinate/reference partition schema mismatch")
    if graph.shape != (n, n) or not np.isfinite(graph.data).all():
        raise AssertionError("spatial graph schema mismatch")
    ordered_sha = hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()
    first_h = frozen_reference_harmonizer(
        z1, z2, zf_raw, ordered_sha, ordered_sha, ordered_sha)
    second_h = frozen_reference_harmonizer(
        z1, z2, zf_raw, ordered_sha, ordered_sha, ordered_sha)
    if first_h.mode != second_h.mode:
        raise AssertionError("harmonizer mode is not deterministic")
    projection_file = aligned_file = None
    projection_file_sha = aligned_file_sha = None
    if first_h.mode == "identity":
        if first_h.zf_aligned is not zf_raw or canonical_array_sha256(first_h.zf_aligned) != canonical_array_sha256(zf_raw):
            raise AssertionError("identity harmonizer is not byte-exact")
    else:
        if (canonical_array_sha256(first_h.projection) != canonical_array_sha256(second_h.projection) or
                canonical_array_sha256(first_h.zf_aligned) != canonical_array_sha256(second_h.zf_aligned)):
            raise AssertionError("rectangular harmonizer SHA is not deterministic")
        projection_file = out / "reference_projection.npy"; aligned_file = out / "zf_aligned.npy"
        atomic_npy(projection_file, first_h.projection); atomic_npy(aligned_file, first_h.zf_aligned)
        projection_file_sha = sha(projection_file); aligned_file_sha = sha(aligned_file)
    d_view = int(z1.shape[1]); d_ref = int(zf_raw.shape[1]); d_coords = int(fourier_coordinates(coords).shape[1])
    if d_coords != 16:
        raise AssertionError("registered Q06 coordinate width must be 16")
    quality = _prepare_quality(dataset, seed, unit_id, z1, z2, first_h.zf_aligned,
                               np.asarray(pf), graph, names, out)
    view_file = SOURCE / f"source/{unit_id}/g04_views.npz"
    zf_file = r02_root(unit_id) / "worker/embedding.npy" if dataset == "p22" else view_file
    manifest = {
        "schema": "spalora.night10a.rev2.real_input.v1", "unit_id": unit_id,
        "dataset": dataset, "seed": seed, "K": DATA[dataset]["K"],
        "family": DATA[dataset]["family"], "N": n, "d_z1": d_view,
        "d_z2": int(z2.shape[1]), "d_zf_raw": d_ref,
        "d_zf_aligned": int(first_h.zf_aligned.shape[1]), "d_coords": d_coords,
        "z1_dtype": str(z1.dtype), "z2_dtype": str(z2.dtype),
        "zf_raw_dtype": str(zf_raw.dtype), "zf_aligned_dtype": str(first_h.zf_aligned.dtype),
        "all_finite": True, "label_reads": 0,
        "ordered_observation_sha256": ordered_sha,
        "views_file": str(view_file), "views_file_sha256": sha(view_file),
        "z1_canonical_sha256": canonical_array_sha256(z1),
        "z2_canonical_sha256": canonical_array_sha256(z2),
        "zf_raw_file": str(zf_file), "zf_raw_file_sha256": sha(zf_file),
        "zf_raw_canonical_sha256": canonical_array_sha256(zf_raw),
        "zf_aligned_canonical_sha256": canonical_array_sha256(first_h.zf_aligned),
        "spatial_graph_file": str(graph_file), "spatial_graph_file_sha256": sha(graph_file),
        "spatial_graph_canonical_sha256": canonical_sparse_sha256(graph),
        "graph_shape": list(graph.shape), "graph_nnz": int(graph.nnz),
        "coordinates_file": str(coord_file), "coordinates_file_sha256": sha(coord_file),
        "coordinates_canonical_sha256": canonical_array_sha256(coords),
        "reference_partition_canonical_sha256": canonical_array_sha256(np.asarray(pf)),
        "c06_file": str(c06) if c06 else None, "c06_sha256": sha(c06) if c06 else None,
        "harmonizer_mode": first_h.mode, "harmonizer_audit": first_h.audit,
        "harmonizer_sha256": first_h.audit.get("projection_canonical_sha256") or
                              first_h.audit["zf_aligned_canonical_sha256"],
        "projection_file": str(projection_file) if projection_file else None,
        "projection_file_sha256": projection_file_sha,
        "zf_aligned_file": str(aligned_file) if aligned_file else None,
        "zf_aligned_file_sha256": aligned_file_sha,
        "adapter_input_width": 3 * d_view,
        "q06_adapter_input_width": 3 * d_view + d_coords,
        "contract_sha256": sha(CONTRACT), "implementation_commit": git_head(),
        **quality,
    }
    manifest["registered_input_sha256"] = canonical_json_sha256({
        key: manifest[key] for key in (
            "unit_id", "seed", "K", "N", "d_z1", "d_z2", "d_zf_raw",
            "d_zf_aligned", "d_coords", "ordered_observation_sha256",
            "views_file_sha256", "z1_canonical_sha256", "z2_canonical_sha256",
            "zf_raw_file_sha256", "zf_raw_canonical_sha256",
            "zf_aligned_canonical_sha256", "spatial_graph_file_sha256",
            "spatial_graph_canonical_sha256", "coordinates_file_sha256",
            "quality_sha256", "p1_sha256", "p2_sha256", "pf_sha256",
            "harmonizer_mode", "harmonizer_sha256", "contract_sha256")})
    atomic_json(manifest_file, manifest); validate_input_manifest(manifest); return manifest


def load_zf_aligned(manifest: dict, z1: np.ndarray, z2: np.ndarray,
                    zf_raw: np.ndarray) -> np.ndarray:
    ordered = manifest["ordered_observation_sha256"]
    recomputed = frozen_reference_harmonizer(z1, z2, zf_raw, ordered, ordered, ordered)
    if recomputed.mode != manifest["harmonizer_mode"]:
        raise AssertionError("frozen harmonizer mode drift")
    if recomputed.mode == "identity":
        if recomputed.zf_aligned is not zf_raw:
            raise AssertionError("identity harmonizer copied its input")
        aligned = recomputed.zf_aligned
    else:
        projection = np.load(manifest["projection_file"])
        aligned = np.load(manifest["zf_aligned_file"])
        if canonical_array_sha256(projection) != recomputed.audit["projection_canonical_sha256"]:
            raise AssertionError("frozen projection canonical SHA drift")
        if canonical_array_sha256(aligned) != recomputed.audit["zf_aligned_canonical_sha256"]:
            raise AssertionError("frozen aligned reference canonical SHA drift")
    if canonical_array_sha256(aligned) != manifest["zf_aligned_canonical_sha256"]:
        raise AssertionError("registered aligned reference drift")
    return aligned


def load_quality(path: pathlib.Path) -> FrozenQuality:
    x = np.load(path); values = {}
    for field in fields(FrozenQuality):
        if field.name == "diagnostics": values[field.name] = {}
        elif field.name == "zero_degree_count": values[field.name] = int(x[field.name])
        else: values[field.name] = x[field.name]
    return FrozenQuality(**values)


def load_runtime_unit(cfg: dict) -> dict:
    manifest_path = pathlib.Path(cfg["input_manifest"])
    if sha(manifest_path) != cfg["input_manifest_sha256"]:
        raise AssertionError("input manifest drift")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")); validate_input_manifest(manifest)
    matches = [row for row in json.loads(PREFLIGHT.read_text())["rows"]
               if row["dataset"] == cfg["dataset"] and int(row["seed"]) == int(cfg["seed"])
               and row["candidate"] == cfg["candidate"]]
    if (len(matches) != 1 or matches[0]["preflight_row_sha256"] != cfg["preflight_row_sha256"] or
            matches[0]["input_manifest_sha256"] != cfg["input_manifest_sha256"] or
            matches[0]["registered_input_sha256"] != cfg["registered_input_sha256"] or
            matches[0]["harmonizer_sha256"] != cfg["harmonizer_sha256"]):
        raise AssertionError("formal config is not bound to the matching locked preflight row")
    if (cfg["implementation_sha256"] != sha(REPO / "SpaLORA/night10a_qcrd.py") or
            cfg["runner_sha256"] != sha(REPO / "scripts/night10a/night10a_rev2_run.py")):
        raise AssertionError("formal implementation source drift")
    z1, z2, _ = views(cfg["unit_id"]); zf_raw, _, _ = reference(cfg["unit_id"], cfg["family"])
    aligned = load_zf_aligned(manifest, z1, z2, zf_raw)
    quality = load_quality(pathlib.Path(manifest["quality_file"]))
    partition = np.load(manifest["pf_file"]); graph = sp.load_npz(manifest["spatial_graph_file"])
    if not (z1.shape == z2.shape == aligned.shape == (manifest["N"], manifest["d_zf_aligned"])):
        raise AssertionError("formal runtime view schema drift")
    if graph.shape != (manifest["N"], manifest["N"]) or len(partition) != manifest["N"]:
        raise AssertionError("formal runtime graph/partition schema drift")
    return {"manifest": manifest, "z1": z1, "z2": z2, "zf_aligned": aligned,
            "quality": quality, "partition": partition, "graph": graph}


def config_path(stage: str, dataset: str, seed: int, candidate: str) -> pathlib.Path:
    return REPO / f"protocols/night10a_rev2/formal_configs/{stage}/{dataset}/seed_{seed}/{candidate}.json"


def preflight_row_map() -> dict[tuple[str, int, str], dict]:
    if not PREFLIGHT.is_file():
        raise RuntimeError("formal config generation requires the locked 210-row preflight")
    value = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    rows = value.get("rows", [])
    if value.get("status") != "PASS" or len(rows) != 210 or not all(row.get("status") == "PASS" for row in rows):
        raise RuntimeError("formal config generation requires 210/210 passing real rows")
    result = {(row["dataset"], int(row["seed"]), row["candidate"]): row for row in rows}
    if len(result) != 210: raise AssertionError("preflight row keys are not unique")
    return result


def prepare_configs(stage: str, selected=None):
    rows = []; preflight = preflight_row_map()
    candidates = CANDIDATES if selected is None else selected
    for dataset, spec in DATA.items():
        seeds = spec[stage]
        for seed in seeds:
            if stage == "r2" and seed in spec["r1"]:
                continue  # Exact R1 cells are reused; only missing seed cells are trained.
            unit_id = unit(dataset, seed)
            input_manifest = prepare_one(dataset, seed)
            input_sha = sha(quality_dir(unit_id)/"input_manifest.json")
            for candidate in candidates:
                smoke = preflight[(dataset, int(seed), candidate)]
                if (smoke["input_manifest_sha256"] != input_sha or
                        smoke["registered_input_sha256"] != input_manifest["registered_input_sha256"] or
                        smoke["harmonizer_sha256"] != input_manifest["harmonizer_sha256"] or
                        smoke["contract_sha256"] != sha(CONTRACT) or
                        smoke["code_commit"] != input_manifest["implementation_commit"]):
                    raise AssertionError("formal/preflight binding mismatch")
                path = config_path(stage, dataset, seed, candidate)
                cfg = {
                    "schema": "spalora.night10a.rev2.formal_config.v1", "stage": stage,
                    "dataset": dataset, "family": spec["family"], "seed": seed, "unit_id": unit_id,
                    "K": spec["K"], "candidate": candidate, "input_manifest": str(quality_dir(unit_id)/"input_manifest.json"),
                    "input_manifest_sha256": input_sha, "epochs": 120, "optimizer": "AdamW",
                    "learning_rate": .001, "weight_decay": .0001, "gradient_clip_norm": 5.0,
                    "hidden_width": 64, "residual_rank": 16, "dropout": .1,
                    "amp": False, "early_stopping": False, "best_epoch": False,
                    "scientific_retry": 0, "fallback": 0, "transform_timeout_seconds": 1800,
                    "label_access": 0, "preflight_row_sha256": smoke["preflight_row_sha256"],
                    "preflight_code_commit": smoke["code_commit"],
                    "registered_input_sha256": input_manifest["registered_input_sha256"],
                    "harmonizer_mode": input_manifest["harmonizer_mode"],
                    "harmonizer_sha256": input_manifest["harmonizer_sha256"],
                    "implementation_sha256": sha(REPO/"SpaLORA/night10a_qcrd.py"),
                    "runner_sha256": sha(REPO/"scripts/night10a/night10a_rev2_run.py"),
                    "semantic_contract_sha256": sha(CONTRACT),
                    "h05_endpoint_source_sha256": sha(REPO/"SpaLORA/night6c_pipeline.py"),
                    "p22_endpoint_source_sha256": sha(REPO/"SpaLORA/night7b_adaptive.py"),
                }
                atomic_json(path, cfg); rows.append({"path": str(path.relative_to(REPO)), "sha256": sha(path), **cfg})
    atomic_json(REPO/f"protocols/night10a_rev2/formal_configs/{stage}_index.json", {"stage":stage,"count":len(rows),"rows":rows})
    return rows


def torch_loss(out, first, second, zf_aligned, quality, candidate, mask, boundary_rows, boundary_cols):
    gate = out["gate"].view(-1)
    align = torch.sum(gate*(1-F.cosine_similarity(out["student_corrected"],out["teacher"].detach(),dim=1,eps=EPS)))/(torch.sum(gate)+EPS)
    if candidate in MASKED_CANDIDATES:
        mf=mask.to(first.dtype); num=torch.sum(mf*(out["pred_student"]-out["clean_student"].detach())**2,dim=1)
        den=torch.sum(mf*out["clean_student"].detach()**2,dim=1)+EPS; masked=torch.mean(num/den)
    else: masked=align.new_zeros(())
    anchor=.5*torch.mean((1-F.cosine_similarity(out["z1c"],first,dim=1,eps=EPS))+(1-F.cosine_similarity(out["z2c"],second,dim=1,eps=EPS)))
    correction=torch.mean((out["correction"].norm(dim=1)/.25)**2)
    if len(boundary_rows):
        new=torch.sum(out["zc"][boundary_rows]*out["zc"][boundary_cols],dim=1)
        old=torch.sum(zf_aligned[boundary_rows]*zf_aligned[boundary_cols],dim=1); boundary=torch.mean(F.relu(new-old)**2)
    else: boundary=align.new_zeros(())
    if candidate=="Q07_CONFIDENCE_MNN_RESIDUAL" and len(quality.mnn_rows):
        r=torch.as_tensor(quality.mnn_rows,dtype=torch.long,device=first.device); c=torch.as_tensor(quality.mnn_cols,dtype=torch.long,device=first.device)
        h1=torch.as_tensor(quality.entropy_1[quality.mnn_rows],dtype=first.dtype,device=first.device); h2=torch.as_tensor(quality.entropy_2[quality.mnn_cols],dtype=first.dtype,device=first.device)
        b1=torch.as_tensor(quality.boundary_risk[quality.mnn_rows],dtype=first.dtype,device=first.device); b2=torch.as_tensor(quality.boundary_risk[quality.mnn_cols],dtype=first.dtype,device=first.device)
        w=torch.sqrt((1-h1).clamp_min(0)*(1-h2).clamp_min(0))*(1-b1)*(1-b2)
        mnn=torch.sum(w*(1-F.cosine_similarity(out["z1c"][r],out["z2c"][c],dim=1,eps=EPS)))/(torch.sum(w)+EPS)
    else: mnn=align.new_zeros(())
    total=align+masked+.25*anchor+.05*correction+.25*boundary+(.1*mnn if candidate=="Q07_CONFIDENCE_MNN_RESIDUAL" else 0)
    return {"align":align,"mask":masked,"anchor":anchor,"correction":correction,"boundary":boundary,"mnn":mnn,"total":total}


def cell_dir(cfg) -> pathlib.Path:
    return RAW / f"{cfg['stage']}/{cfg['dataset']}/seed_{cfg['seed']}/{cfg['candidate']}"


def train(config_file: pathlib.Path):
    cfg=json.loads(config_file.read_text()); outdir=cell_dir(cfg); final=outdir/"training_manifest.json"
    if final.exists():
        status=json.loads(final.read_text()).get("status")
        if status in ACCEPTED_TRAINING_STATUSES: return
        raise AssertionError(f"existing training manifest has invalid status={status}")
    outdir.mkdir(parents=True,exist_ok=True); start=time.time()
    runtime=load_runtime_unit(cfg); unit_id=cfg["unit_id"]
    z1=runtime["z1"]; z2=runtime["z2"]; zf_aligned=runtime["zf_aligned"]
    pf=runtime["partition"]; q=runtime["quality"]; graph=runtime["graph"]
    upper=sp.triu(graph.maximum(graph.T),k=1).tocoo(); keep=pf[upper.row]!=pf[upper.col]
    device=torch.device("cuda"); torch.manual_seed(cfg["seed"]); torch.cuda.manual_seed_all(cfg["seed"]); np.random.seed(cfg["seed"])
    first=torch.as_tensor(row_normalize(z1),dtype=torch.float32,device=device); second=torch.as_tensor(row_normalize(z2),dtype=torch.float32,device=device); fused=torch.as_tensor(zf_aligned,dtype=torch.float32,device=device)
    coord=None
    if cfg["candidate"]=="Q06_COORDINATE_PRIOR_RESIDUAL": coord=torch.as_tensor(fourier_coordinates(np.load(runtime["manifest"]["coordinates_file"])),dtype=torch.float32,device=device)
    model=QCRDAdapter(first.shape[1],0 if coord is None else coord.shape[1],64,16,.1).to(device)
    expected_width=3*first.shape[1]+(0 if coord is None else coord.shape[1])
    if model.input.in_features!=expected_width or expected_width!=(runtime["manifest"]["q06_adapter_input_width"] if coord is not None else runtime["manifest"]["adapter_input_width"]):
        raise AssertionError("formal adapter input width mismatch")
    initial=canonical_state_sha256(model.state_dict()); optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    br=torch.as_tensor(upper.row[keep],dtype=torch.long,device=device); bc=torch.as_tensor(upper.col[keep],dtype=torch.long,device=device)
    input_artifact=cfg["input_manifest_sha256"]; curves=[]; peak=0
    for epoch in range(120):
        model.train(); optimizer.zero_grad(set_to_none=True); mask=None
        if cfg["candidate"] in MASKED_CANDIDATES: mask=torch.as_tensor(deterministic_mask(cfg["candidate"],input_artifact,cfg["seed"],epoch,len(first),first.shape[1]),device=device)
        fo=qcrd_forward(model,first,second,fused,q,cfg["candidate"],coord,mask); losses=torch_loss(fo,first,second,fused,q,cfg["candidate"],mask,br,bc)
        if not torch.isfinite(losses["total"]): raise FloatingPointError("nonfinite formal loss")
        losses["total"].backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); optimizer.step()
        peak=max(peak,torch.cuda.max_memory_allocated())
        curves.append({"epoch":epoch,**{k:float(v.detach().cpu()) for k,v in losses.items()}})
    model.eval()
    with torch.no_grad(): a,b,c,cor=corrected_views(model,first,second,fused,q,cfg["candidate"],coord)
    atomic_npz(outdir/"corrected_views.npz",z1c=a.cpu().numpy(),z2c=b.cpu().numpy(),zc=c.cpu().numpy(),correction=cor.cpu().numpy())
    ckpt={"state_dict":model.state_dict(),"config":cfg,"initial_state_sha256":initial}; tmp=outdir/"model_final.pt.tmp"; torch.save(ckpt,tmp); os.replace(tmp,outdir/"model_final.pt")
    with (outdir/"loss_curve.csv.tmp").open("w",newline="") as h:
        w=csv.DictWriter(h,fieldnames=list(curves[0])); w.writeheader(); w.writerows(curves)
    os.replace(outdir/"loss_curve.csv.tmp",outdir/"loss_curve.csv")
    expected=np.load(outdir/"corrected_views.npz"); expected_sha={k:canonical_array_sha256(expected[k]) for k in expected.files}
    manifest={"status":"TRAINED_AWAITING_RELOAD","config":cfg,"config_file_sha256":sha(config_file),"model_file_sha256":sha(outdir/"model_final.pt"),"canonical_state_sha256":canonical_state_sha256(model.state_dict()),"initial_state_sha256":initial,"view_shas":expected_sha,"loss_curve_sha256":sha(outdir/"loss_curve.csv"),"epochs":120,"device":torch.cuda.get_device_name(0),"peak_gpu_bytes":peak,"runtime_seconds":time.time()-start,"label_reads":0,"scientific_retry":0,"fallback":0}
    atomic_json(final,manifest)
    cmd=[PYTHON,str(REPO/"scripts/night10a/night10a_rev2_run.py"),"reload",str(config_file)]
    proc=subprocess.run(cmd,cwd=REPO,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=600)
    (outdir/"reload.log").write_text(proc.stdout)
    if proc.returncode: raise RuntimeError("fresh reload failed")


def reload(config_file: pathlib.Path):
    cfg=json.loads(config_file.read_text()); outdir=cell_dir(cfg); manifest=json.loads((outdir/"training_manifest.json").read_text())
    runtime=load_runtime_unit(cfg); unit_id=cfg["unit_id"]; z1=runtime["z1"]; z2=runtime["z2"]; q=runtime["quality"]
    first=torch.as_tensor(row_normalize(z1),dtype=torch.float32,device="cuda"); second=torch.as_tensor(row_normalize(z2),dtype=torch.float32,device="cuda"); fused=torch.as_tensor(runtime["zf_aligned"],dtype=torch.float32,device="cuda")
    coord=None
    if cfg["candidate"]=="Q06_COORDINATE_PRIOR_RESIDUAL": coord=torch.as_tensor(fourier_coordinates(np.load(runtime["manifest"]["coordinates_file"])),dtype=torch.float32,device="cuda")
    ckpt=torch.load(outdir/"model_final.pt",map_location="cuda"); model=QCRDAdapter(first.shape[1],0 if coord is None else coord.shape[1],64,16,.1).cuda().eval(); model.load_state_dict(ckpt["state_dict"])
    with torch.no_grad(): a,b,c,cor=corrected_views(model,first,second,fused,q,cfg["candidate"],coord)
    got={"z1c":a.cpu().numpy(),"z2c":b.cpu().numpy(),"zc":c.cpu().numpy(),"correction":cor.cpu().numpy()}; expected=np.load(outdir/"corrected_views.npz")
    checks={k:{"max_abs":float(np.max(np.abs(got[k]-expected[k]))),"allclose":bool(np.allclose(got[k],expected[k],rtol=1e-6,atol=1e-7)),"sha_exact":canonical_array_sha256(got[k])==manifest["view_shas"][k]} for k in got}
    if not all(x["allclose"] for x in checks.values()): raise AssertionError("reload view mismatch")
    atomic_json(outdir/"reload_audit.json",{"status":"PASS","fresh_process":True,"checks":checks,"label_reads":0})
    manifest["status"]="CHECKPOINT_ROUNDTRIP_PASS"; manifest["reload_audit_sha256"]=sha(outdir/"reload_audit.json"); atomic_json(outdir/"training_manifest.json",manifest)


def transform(config_file: pathlib.Path):
    cfg=json.loads(config_file.read_text()); outdir=cell_dir(cfg); manifest=json.loads((outdir/"training_manifest.json").read_text())
    if manifest["status"] not in ACCEPTED_TRAINING_STATUSES: raise AssertionError("training not locked")
    if (outdir/"transform_manifest.json").exists():
        status=json.loads((outdir/"transform_manifest.json").read_text()).get("status")
        if status in ACCEPTED_TRANSFORM_STATUSES: return
        raise AssertionError(f"existing transform manifest has invalid status={status}")
    x=np.load(outdir/"corrected_views.npz"); names=ids(cfg["unit_id"]); start=time.time()
    if cfg["family"]=="RNA+protein":
        aff=[self_tuning_affinity(row_normalize(x[k]),10,names) for k in ("z1c","z2c","zc")]; affinity=sym_zero(sum(aff)/3); endpoint="H05_EQUAL3_AFFINITY_SPECTRAL"
    else:
        az=self_tuning_affinity(row_normalize(x["zc"]),10,names); c06=sp.load_npz(SOURCE/f"source/{cfg['unit_id']}/c06_affinity.npz"); affinity=sym_zero(.5*row_sparse_strict(az)+.5*row_sparse_strict(c06)); endpoint="E1_ADAPTER_C06_MEAN/H01"
    clusters=spectral(affinity,cfg["K"]); sp.save_npz(outdir/"affinity.npz",affinity)
    tmp=outdir/"clusters.csv.tmp"
    with tmp.open("w",newline="") as h:
        w=csv.writer(h); w.writerow(["observation_id","cluster"]); w.writerows(zip(names,clusters.tolist()))
    os.replace(tmp,outdir/"clusters.csv")
    atomic_json(outdir/"transform_manifest.json",{"status":"PASS","endpoint":endpoint,"clusters_sha256":sha(outdir/"clusters.csv"),"affinity_sha256":sha(outdir/"affinity.npz"),"affinity_canonical_sha256":canonical_sparse_sha256(affinity),"partition_sha256":canonical_array_sha256(clusters),"runtime_seconds":time.time()-start,"label_reads":0})


def run_stage(stage: str):
    index=json.loads((REPO/f"protocols/night10a_rev2/formal_configs/{stage}_index.json").read_text())
    failures=[]; started=time.time(); run_start=(RAW/"p0_rev2/p0_rev2_authority.json").stat().st_mtime
    for ordinal,row in enumerate(index["rows"],1):
        cfg=REPO/row["path"]
        try: train(cfg)
        except Exception as exc:
            failures.append({"ordinal":ordinal,"config":str(cfg),"phase":"training","error":repr(exc)})
            atomic_json(cell_dir(row)/"failure.json",failures[-1])
        if time.time()-run_start>12*3600:
            failures.append({"ordinal":ordinal,"config":str(cfg),"phase":"training","error":"WALLCLOCK_BUDGET_REACHED"}); break
    # Fixed endpoint: three independent subprocesses x three BLAS threads <= 12 CPUs.
    tasks=[]
    for ordinal,row in enumerate(index["rows"],1):
        cfg=REPO/row["path"]; out=cell_dir(row)
        if not (out/"training_manifest.json").exists() or (out/"failure.json").exists(): continue
        tasks.append((ordinal,row,cfg,out))

    def run_transform(task):
        ordinal,row,cfg,out=task
        if time.time()-run_start>12*3600:
            error={"ordinal":ordinal,"config":str(cfg),"phase":"transform","error":"WALLCLOCK_BUDGET_REACHED"}; atomic_json(out/"failure.json",error); return error
        try:
            proc=subprocess.run([PYTHON,str(REPO/"scripts/night10a/night10a_rev2_run.py"),"transform",str(cfg)],cwd=REPO,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=1800,env={**os.environ,"OMP_NUM_THREADS":"3","MKL_NUM_THREADS":"3","OPENBLAS_NUM_THREADS":"3"})
            (out/"transform.log").write_text(proc.stdout)
            if proc.returncode: raise RuntimeError("transform subprocess nonzero")
            return None
        except subprocess.TimeoutExpired: error={"ordinal":ordinal,"config":str(cfg),"phase":"transform","error":"TIMEOUT_1800S"}
        except Exception as exc: error={"ordinal":ordinal,"config":str(cfg),"phase":"transform","error":repr(exc)}
        atomic_json(out/"failure.json",error); return error

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for error in pool.map(run_transform,tasks):
            if error is not None: failures.append(error)
    manifests=list(RAW.glob(f"{stage}/**/transform_manifest.json")); atomic_json(RAW/f"{stage}/{stage}_lock_manifest.json",{"stage":stage,"registered":len(index["rows"]),"locked":len(manifests),"failures":failures,"label_reads":0,"elapsed_seconds":time.time()-started})


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("mode",choices=["prepare-r1","prepare-r2","run-r1","run-r2","train","reload","transform"]); parser.add_argument("arg",nargs="?"); args=parser.parse_args()
    if args.mode=="prepare-r1": prepare_configs("r1")
    elif args.mode=="prepare-r2": prepare_configs("r2",json.loads(args.arg))
    elif args.mode=="run-r1": run_stage("r1")
    elif args.mode=="run-r2": run_stage("r2")
    elif args.mode=="train": train(pathlib.Path(args.arg))
    elif args.mode=="reload": reload(pathlib.Path(args.arg))
    elif args.mode=="transform": transform(pathlib.Path(args.arg))


if __name__=="__main__": main()
