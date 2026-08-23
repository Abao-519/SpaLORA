#!/usr/bin/env python3
"""Build the compact Night-15B Windows compute kit from registered artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import anndata as ad
import numpy as np
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/night13b"))

import night13b_run as n13b  # noqa: E402

from SpaLORA.night15b_sapr import build_stability_anchors  # noqa: E402


N13C = Path("/root/autodl-fs/night13c_endpoint_robustness_trainable_core_20260823")
N14B = Path("/root/autodl-fs/night14b_atac_score_acceleration_20260823")
N15A = Path("/root/autodl-fs/night15a_multimodal_contribution_and_score_stability_20260823")
DEFAULT_ROOT = Path("/root/autodl-fs/night15b_stability_anchored_prototype_score_sprint_20260824/local_compute_kit")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def add_embedding(target: Dict[str, np.ndarray], key: str, value: np.ndarray, n: int) -> None:
    value = np.asarray(value, dtype=np.float32)
    if value.ndim != 2 or value.shape[0] != n or not np.isfinite(value).all():
        raise ValueError(f"invalid embedding {key}: {value.shape}")
    target[f"emb__{key}"] = value


def load_views(path: Path) -> Mapping[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=False)


def reorder(ids: Sequence[str], target: Sequence[str], value: np.ndarray) -> np.ndarray:
    ids = list(map(str, ids))
    target = list(map(str, target))
    if ids == target:
        return np.asarray(value, dtype=np.float32)
    if len(ids) != len(set(ids)) or set(ids) != set(target):
        raise RuntimeError("registered embedding observation identity mismatch")
    lookup = {item: index for index, item in enumerate(ids)}
    return np.asarray(value, dtype=np.float32)[[lookup[item] for item in target]]


def add_registered_embeddings(name: str, payload: Mapping[str, object], arrays: Dict[str, np.ndarray]) -> list[dict]:
    n = len(payload["ids"])
    audit: list[dict] = []
    add_embedding(arrays, "simple", payload["embedding"], n)
    audit.append({"key": "simple", "source": "Night-13B base_payload", "status": "PASS"})

    c00_paths = {
        "A1": Path("/root/autodl-fs/night6c_raw_runs_20260817/r1/G04_SP10_F10_EUC_UNION/a1/seed_0/attempt_001"),
        "D1": Path("/root/autodl-fs/night6d_raw_runs_20260817/G04_SP10_F10_EUC_UNION/d1/seed_0/attempt_001"),
        "tonsil_s1": Path("/root/autodl-fs/night6c_raw_runs_20260817/r1/G04_SP10_F10_EUC_UNION/tonsil/seed_0/attempt_001"),
    }
    if name in c00_paths:
        run = c00_paths[name]
        archive = load_views(run / "views.npz")
        ids = np.loadtxt(run / "observation_ids.csv", dtype=str, delimiter=",", skiprows=1, usecols=0)
        z = reorder(ids.tolist(), list(map(str, payload["ids"])), archive["SpaLORA_fused"])
        add_embedding(arrays, "C00_G04", z, n)
        audit.append({"key": "C00_G04", "source": str(run / "views.npz"), "sha256": sha256(run / "views.npz"), "status": "PASS"})

    if name == "P22":
        f00 = Path("/root/autodl-fs/night7b_score_rnd_20260818/adapter_stage/R1/formal/R02/u020/attempt_001/worker/embedding.npy")
        fids = Path("/root/autodl-fs/night7b_score_rnd_20260818/source/u020/observation_ids.txt").read_text(encoding="utf-8").splitlines()
        add_embedding(arrays, "F00_R02", reorder(fids, payload["ids"], np.load(f00, allow_pickle=False)), n)
        audit.append({"key": "F00_R02", "source": str(f00), "sha256": sha256(f00), "status": "PASS"})
        n02run = Path("/root/autodl-fs/night9b_racf_20260820/r1/r1-u009/attempt_001")
        config = json.loads((n02run / "resolved_config.json").read_text(encoding="utf-8"))
        import pandas as pd
        nids = pd.read_csv(config["observation_ids_path"]).iloc[:, 0].astype(str).tolist()
        n02archive = load_views(n02run / "views.npz")
        add_embedding(arrays, "N02_HIER", reorder(nids, payload["ids"], n02archive["SpaLORA_fused"]), n)
        audit.append({"key": "N02_HIER", "source": str(n02run / "views.npz"), "sha256": sha256(n02run / "views.npz"), "status": "PASS"})
        locked = N15A / "locked_reference_bridge_v1/P22/seed_0/views.npz"
        archive = load_views(locked)
        if list(map(str, archive["ids"])) != list(map(str, payload["ids"])):
            raise RuntimeError("P22 locked bridge ordered ID mismatch")
        for key in ("z1", "z2", "base_fused", "mcdf"):
            add_embedding(arrays, f"LOCKED_{key}", archive[key], n)
        audit.append({"key": "LOCKED_BRIDGE", "source": str(locked), "sha256": sha256(locked), "status": "PASS"})
    elif name == "MISAR_E15_5_S1":
        candidates = {
            "MORAN_R20": N15A / "development_moran_v1/R20_MCDF_GATED/MISAR_E15_5_S1/seed_0/views.npz",
            "MORAN_R40": N15A / "development_r40_v1/MORAN/R40_MCDF_MASKED_MODALITY/MISAR_E15_5_S1/seed_0/views.npz",
            "SEPAR_R40": N15A / "development_r40_v1/SEPAR_EXACT/R40_MCDF_MASKED_MODALITY/MISAR_E15_5_S1/seed_0/views.npz",
        }
        for prefix, path in candidates.items():
            archive = load_views(path)
            if list(map(str, archive["ids"])) != list(map(str, payload["ids"])):
                raise RuntimeError(f"{prefix} ordered ID mismatch")
            for key in ("z1", "z2", "base_fused", "mcdf"):
                add_embedding(arrays, f"{prefix}_{key}", archive[key], n)
            audit.append({"key": prefix, "source": str(path), "sha256": sha256(path), "status": "PASS"})
    elif name in {"A1", "tonsil_s1"}:
        for candidate in ("R20_MCDF_GATED", "R40_MCDF_MASKED_MODALITY"):
            path = N15A / f"side_development_v1/{candidate}/{name}/seed_0/views.npz"
            archive = load_views(path)
            if list(map(str, archive["ids"])) != list(map(str, payload["ids"])):
                raise RuntimeError(f"{name}/{candidate} ordered ID mismatch")
            for key in ("z1", "z2", "base_fused", "mcdf"):
                add_embedding(arrays, f"{candidate}_{key}", archive[key], n)
            audit.append({"key": candidate, "source": str(path), "sha256": sha256(path), "status": "PASS"})
    return audit


def teacher_bank(name: str, n: int) -> tuple[np.ndarray, list[str]]:
    items: list[np.ndarray] = []
    names: list[str] = []
    stage_a = N13C / "stage_a"
    for variant in ("IDENTITY", "P4_ONLY", "P18_ONLY", "B10_MIXTURE", "FIXED_BETA_0.04", "FIXED_BETA_0.16", "FIXED_BETA_0.32"):
        path = stage_a / f"consensus_{name}_{variant}.npy"
        if path.is_file():
            value = np.load(path, allow_pickle=False).astype(np.int32)
            if len(value) != n:
                raise RuntimeError(f"teacher length mismatch {path}")
            items.append(value)
            names.append(f"N13C_{variant}")
    formal = N14B / "formal_frozen/formal_partitions.npz"
    if name in {"P22", "MISAR_E15_5_S1"} and formal.is_file():
        archive = np.load(formal, allow_pickle=False)
        prefixes = ("F30_P22_K9", "F31_P22_K9") if name == "P22" else ("F32_MISAR_K7",)
        for key in sorted(k for k in archive.files if k.startswith(prefixes)):
            value = archive[key].astype(np.int32)
            if len(value) == n:
                items.append(value)
                names.append(f"N14B_{key}")
    if len(items) < 3:
        raise RuntimeError(f"insufficient teacher partitions for {name}")
    return np.stack(items), names


def csr_arrays(prefix: str, matrix: sp.spmatrix, arrays: Dict[str, np.ndarray]) -> None:
    matrix = sp.csr_matrix(matrix, dtype=np.float32)
    arrays[f"{prefix}__data"] = matrix.data.astype(np.float32)
    arrays[f"{prefix}__indices"] = matrix.indices.astype(np.int32)
    arrays[f"{prefix}__indptr"] = matrix.indptr.astype(np.int32)
    arrays[f"{prefix}__shape"] = np.asarray(matrix.shape, dtype=np.int64)


def build_dataset(name: str, root: Path) -> dict:
    started = time.perf_counter()
    payload = n13b.base_payload(name)
    n = len(payload["ids"])
    arrays: Dict[str, np.ndarray] = {
        "ids": np.asarray(payload["ids"], dtype=str),
        "coordinates": np.asarray(payload["coordinates"], dtype=np.float32),
        "labels_primary": np.asarray(payload["labels"], dtype=str),
        "label_mask": np.asarray(payload["label_mask"], dtype=bool),
        "view1": np.asarray(payload["view1"], dtype=np.float32),
        "view2": np.asarray(payload["view2"], dtype=np.float32),
        "k_primary": np.asarray([int(n13b.DATASETS[name]["k"])], dtype=np.int32),
    }
    audit = add_registered_embeddings(name, payload, arrays)
    partitions, partition_names = teacher_bank(name, n)
    arrays["teacher_partitions"] = partitions
    arrays["teacher_names"] = np.asarray(partition_names, dtype=str)
    if name == "MISAR_E15_5_S1":
        formal = np.load(N14B / "formal_frozen/formal_partitions.npz", allow_pickle=False)
        keys12 = sorted(key for key in formal.files if key.startswith("F33_MISAR_K12"))
        values12 = [formal[key].astype(np.int32) for key in keys12]
        if len(values12) < 3 or any(len(value) != n for value in values12):
            raise RuntimeError("MISAR K12 teacher partition bank is incomplete")
        arrays["teacher_partitions_k12"] = np.stack(values12)
        arrays["teacher_names_k12"] = np.asarray([f"N14B_{key}" for key in keys12], dtype=str)
    csr_arrays("graph", payload["metric_graph"], arrays)
    csr_arrays("operator4", payload["operators"][0], arrays)
    csr_arrays("operator18", payload["operators"][1], arrays)

    lanes = [{"lane": name, "k": int(n13b.DATASETS[name]["k"]), "reference": "primary"}]
    if name == "MISAR_E15_5_S1":
        lanes.append({"lane": "MISAR_E15_5_S1_K12", "k": 12, "reference": "same public K7 Y; prediction cardinality K12"})
    if name == "P22":
        carrier_path = N15A / "protocol_inputs/3dot_zenodo_15089427/3d-OT.h5ad"
        carrier = ad.read_h5ad(carrier_path, backed="r")
        if list(map(str, carrier.obs_names)) != list(map(str, payload["ids"])):
            raise RuntimeError("3d-OT K18 carrier ordered ID mismatch")
        labels18 = carrier.obs["3d-OT"].astype(str).to_numpy()
        if len(np.unique(labels18)) != 18:
            raise RuntimeError("3d-OT author assignment is not K18")
        arrays["labels_k18_author_assignment"] = labels18.astype(str)
        for key in ("3d-OT", "X_pca", "emb_pca", "feat"):
            if key in carrier.obsm:
                value = np.asarray(carrier.obsm[key], dtype=np.float32)
                if value.ndim == 2 and value.shape[0] == n and np.isfinite(value).all():
                    add_embedding(arrays, f"3DOT_CONTEXT_{key}", value, n)
        lanes.append({"lane": "P22_3DOT_K18", "k": 18, "reference": "author 3d-OT assignment; context, not independent truth"})

    path = root / f"{name}.npz"
    np.savez_compressed(path, **arrays)
    meta = {
        "dataset": name,
        "family": "RNA+protein" if n13b.DATASETS[name]["adapter"] == "protein" else "RNA+ATAC",
        "total_observations": n,
        "evaluated_observations": int(np.sum(payload["label_mask"])),
        "ordered_id_sha256": n13b.ordered_id_sha256(payload["ids"]),
        "view_shapes": [list(arrays["view1"].shape), list(arrays["view2"].shape)],
        "embedding_keys": sorted(key[5:] for key in arrays if key.startswith("emb__")),
        "teacher_partition_count": int(partitions.shape[0]),
        "teacher_names": partition_names,
        "lanes": lanes,
        "artifact": path.name,
        "artifact_size": path.stat().st_size,
        "artifact_sha256": sha256(path),
        "source_audit": audit,
        "wall_seconds": time.perf_counter() - started,
    }
    atomic_json(root / f"{name}.json", meta)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    datasets = list(n13b.DATASETS)
    results = [build_dataset(name, args.output) for name in datasets]
    gse = N15A / "protocol_inputs/GSE213264/dedup_audit.json"
    gse_audit = json.loads(gse.read_text(encoding="utf-8")) if gse.is_file() else {"status": "MISSING"}
    manifest = {
        "schema": "night15b_local_compute_kit_v1",
        "datasets": results,
        "gse213264": {
            "status": "UNSUPPORTED_LABEL_PROTOCOL_NOT_CLOSED",
            "reason": "RNA/protein matrices were provenance-audited but no canonical spatial label/mask/coordinate protocol was closed",
            "dedup_audit": gse_audit,
        },
        "raw_fragments_included": 0,
        "dense_n_by_n_included": 0,
        "labels_usage": "evaluator and cross-run HPO only; absent from SAPR APIs and loss",
    }
    atomic_json(args.output / "kit_manifest.json", manifest)
    indexed = []
    for path in sorted(item for item in args.output.iterdir() if item.is_file()):
        indexed.append({"path": path.name, "size": path.stat().st_size, "sha256": sha256(path)})
    atomic_json(args.output / "kit_index.json", {"count": len(indexed), "files": indexed})
    print(json.dumps({"status": "PASS", "count": len(indexed), "bytes": sum(x["size"] for x in indexed)}))


if __name__ == "__main__":
    main()
