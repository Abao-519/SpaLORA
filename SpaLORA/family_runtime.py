"""Label-free runtime helpers for Night-10B exact replay and fresh smokes."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import resource
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import scipy.sparse as sp

from .family_policy import ResolvedPolicy, resolve_family_policy
from .family_recipes import canonical_json_sha256
from .night10a_qcrd import canonical_array_sha256
from .night3af_cache import sha256_file
from .night6c_pipeline import run_head
from .night6d_pipeline import HEADS as NIGHT6D_HEADS
from .night7b_adaptive import run_partition
from .night8b_pipeline import adapter_endpoint


DEFAULT_SOURCE_ROOT = Path("/root/autodl-fs/night7b_score_rnd_20260818")
DEFAULT_INPUT_ROOT = Path(
    "/root/autodl-fs/night10a_rev2_qcrd_20260821/raw/inputs"
)
DEFAULT_Q00_AUDIT = Path(
    "outputs/night10a_rev2_handoff/r2_lock/"
    "r2_total_lock_independent_audit.json"
)
NIGHT7B_REGISTRY = Path(
    "protocols/night7b/"
    "SpaLORA_Night7B_Adaptive_Relational_Fusion_Registry_2026-08-18.json"
)

EXPECTED_UNIT_ORDER = tuple(f"u{i:03d}" for i in range(30))
EXPECTED_DATA_STEWARD_ROWS = (
    tuple(("a1", seed) for seed in range(5))
    + tuple(("tonsil", seed) for seed in range(5))
    + tuple(("d1", seed) for seed in range(10))
    + tuple(("p22", seed) for seed in range(10))
)


def file_sha256(path: str | Path) -> str:
    return sha256_file(Path(path))


def atomic_json(path: str | Path, value: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with tmp.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(tmp, target)


def atomic_npy(path: str | Path, value: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp.npy")
    np.save(tmp, value, allow_pickle=False)
    os.replace(tmp, target)


def git_head(repo: str | Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(repo), text=True,
    ).strip()


def implementation_sha256(repo: str | Path) -> tuple[str, list[dict[str, Any]]]:
    root = Path(repo)
    relative = (
        "SpaLORA/family_recipes.py",
        "SpaLORA/family_policy.py",
        "SpaLORA/family_runtime.py",
        "scripts/night10b/run_family_policy.py",
    )
    rows = [
        {
            "path": name,
            "sha256": file_sha256(root / name),
            "size": (root / name).stat().st_size,
        }
        for name in relative
    ]
    return canonical_json_sha256(rows), rows


def ensure_sparse_square(value: sp.spmatrix, n: int, name: str) -> sp.csr_matrix:
    if not sp.issparse(value) or value.shape != (n, n):
        raise RuntimeError(f"{name} must be sparse and have shape ({n},{n})")
    out = value.tocsr()
    if not np.isfinite(out.data).all():
        raise RuntimeError(f"{name} contains non-finite values")
    return out


def reject_dense_n_by_n(value: object, n: int, name: str) -> None:
    if isinstance(value, np.ndarray) and value.ndim == 2 and value.shape == (n, n):
        raise RuntimeError(f"dense N-by-N guard rejected {name}")


def load_q00_rows(path: str | Path = DEFAULT_Q00_AUDIT) -> list[dict[str, Any]]:
    audit = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = list(audit.get("q00_references", []))
    actual_units = tuple(str(row.get("unit_id")) for row in rows)
    actual_steward = tuple(
        (str(row.get("dataset")), int(row.get("seed"))) for row in rows
    )
    if (
        len(rows) != 30
        or audit.get("q00_reference_aliases_locked") != 30
        or actual_units != EXPECTED_UNIT_ORDER
        or actual_steward != EXPECTED_DATA_STEWARD_ROWS
    ):
        raise RuntimeError("Q00 authority is not the exact locked 30-row set")
    if any(int(row.get("label_reads_during_training", -1)) != 0 for row in rows):
        raise RuntimeError("Q00 authority reports a training label read")
    return rows


def assay_pair_for_steward(data_steward_id: str) -> tuple[str, str]:
    """Data-steward boundary; its output, not the identifier, reaches resolver."""
    mapping = {
        "a1": ("RNA", "PROTEIN"),
        "tonsil": ("RNA", "PROTEIN"),
        "d1": ("RNA", "PROTEIN"),
        "p22": ("RNA", "ATAC"),
    }
    if data_steward_id not in mapping:
        raise RuntimeError("unregistered content-addressed data-steward ID")
    return mapping[data_steward_id]


def _ordered_ids(source_dir: Path) -> list[str]:
    values = source_dir.joinpath("observation_ids.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    if not values or len(set(values)) != len(values):
        raise RuntimeError("ordered observations are empty or duplicated")
    return values


def ordered_observation_sha256(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(map(str, ids)).encode("utf-8")).hexdigest()


def _read_partition_csv(path: Path, ids: Sequence[str]) -> np.ndarray:
    names: list[str] = []
    labels: list[int] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            names.append(str(row["observation_id"]))
            labels.append(int(row["cluster"]))
    if names != list(ids):
        raise RuntimeError("partition observation order drift")
    return np.asarray(labels, dtype=np.int64)


def _r02_root(source_root: Path, unit_id: str) -> Path:
    candidates = []
    for stage in ("R1", "R2"):
        path = (
            source_root / "adapter_stage" / stage / "formal" / "R02"
            / unit_id / "attempt_001"
        )
        if (
            path.joinpath("worker/embedding.npy").is_file()
            and path.joinpath(
                "transforms/E1_ADAPTER_C06_MEAN/H01/clusters.csv"
            ).is_file()
        ):
            candidates.append(path)
    if len(candidates) != 1:
        raise RuntimeError(f"R02 authority is not unique for {unit_id}")
    return candidates[0]


def _artifact_row(path: Path) -> dict[str, Any]:
    return {"path": str(path), "size": path.stat().st_size, "sha256": file_sha256(path)}


def _load_reference(
    source_root: Path, unit_id: str, policy: ResolvedPolicy,
) -> dict[str, Any]:
    source = source_root / "source" / unit_id
    worker = json.loads(source.joinpath("worker_input.json").read_text(encoding="utf-8"))
    ids = _ordered_ids(source)
    if str(worker.get("unit_id")) != unit_id or int(worker.get("observation_count")) != len(ids):
        raise RuntimeError("source worker identity/cardinality drift")
    if worker.get("ordered_observation_sha256") != ordered_observation_sha256(ids):
        raise RuntimeError("source worker ordered-observation drift")
    k = int(worker["K"])
    if policy.recipe.family == "RNA_PROTEIN":
        views_path = source / "g04_views.npz"
        with np.load(views_path, allow_pickle=False) as payload:
            views = {
                name: np.asarray(payload[name])
                for name in (
                    "emb_latent_omics1", "emb_latent_omics2", "SpaLORA_fused"
                )
            }
        embedding = views["SpaLORA_fused"]
        partition_path = source / "c00_partition.npy"
        partition = np.load(partition_path, allow_pickle=False).astype(np.int64, copy=False)
        coords_path = source / "coordinates.npy"
        coords = np.load(coords_path, allow_pickle=False)
        artifacts = [
            _artifact_row(source / "worker_input.json"), _artifact_row(views_path),
            _artifact_row(partition_path), _artifact_row(coords_path),
            _artifact_row(source / "observation_ids.txt"),
        ]
        return {
            "artifacts": artifacts, "coordinates": coords, "embedding": embedding,
            "ids": ids, "k": k, "partition": partition, "source_dir": source,
            "views": views,
        }
    if policy.recipe.family == "RNA_EPIGENOME":
        root = _r02_root(source_root, unit_id)
        embedding_path = root / "worker/embedding.npy"
        partition_path = root / "transforms/E1_ADAPTER_C06_MEAN/H01/clusters.csv"
        c06_path = source / "c06_affinity.npz"
        embedding = np.load(embedding_path, allow_pickle=False).astype(np.float32, copy=False)
        partition = _read_partition_csv(partition_path, ids)
        c06 = ensure_sparse_square(sp.load_npz(c06_path), len(ids), "C06")
        artifacts = [
            _artifact_row(source / "worker_input.json"), _artifact_row(embedding_path),
            _artifact_row(partition_path), _artifact_row(c06_path),
            _artifact_row(source / "observation_ids.txt"),
        ]
        return {
            "artifacts": artifacts, "c06": c06, "embedding": embedding,
            "ids": ids, "k": k, "partition": partition, "source_dir": source,
        }
    raise RuntimeError("resolved family has no frozen runtime")


def _validate_night10a_input(
    input_root: Path, q00: Mapping[str, Any], reference: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    path = input_root / str(q00["unit_id"]) / "input_manifest.json"
    if not path.is_file() or file_sha256(path) != q00["input_manifest_sha256"]:
        raise RuntimeError("Night-10A input-manifest file SHA drift")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    checks = {
        "registered_input_sha256": q00["registered_input_sha256"],
        "ordered_observation_sha256": q00["ordered_observation_sha256"],
    }
    for key, expected in checks.items():
        if manifest.get(key) != expected:
            raise RuntimeError(f"Night-10A manifest binding drift: {key}")
    if canonical_array_sha256(np.asarray(reference["embedding"])) != q00["reference_embedding_sha256"]:
        raise RuntimeError("source reference embedding canonical SHA drift")
    # Night-10A froze this field with canonical_array_sha256 on the stored
    # integer partition itself.  Relabelling clusters here would change that
    # authority (for example, the historical C00 files use labels 1..K).
    if canonical_array_sha256(np.asarray(reference["partition"])) != q00["reference_partition_sha256"]:
        raise RuntimeError("source reference partition array SHA drift")
    return path, manifest


def _endpoint_partition(reference: Mapping[str, Any], policy: ResolvedPolicy) -> np.ndarray:
    n = len(reference["ids"])
    reject_dense_n_by_n(reference["embedding"], n, "reference_embedding")
    if policy.recipe.family == "RNA_PROTEIN":
        labels, _ = run_head(
            NIGHT6D_HEADS["H05_EQUAL3_AFFINITY_SPECTRAL"], reference["views"],
            reference["k"], reference["coordinates"], reference["ids"], None,
        )
        return np.asarray(labels, dtype=np.int64)
    affinity = adapter_endpoint(
        reference["embedding"], reference["c06"], reference["ids"]
    )
    ensure_sparse_square(affinity, n, "E1_ADAPTER_C06_MEAN")
    registry = json.loads(NIGHT7B_REGISTRY.read_text(encoding="utf-8"))
    labels, _ = run_partition(
        "H01", affinity, reference["k"],
        registry["partition_heads"]["LEIDEN_EXACT_K"]["resolution_grid"],
    )
    return np.asarray(labels, dtype=np.int64)


def replay_q00_row(
    q00: Mapping[str, Any], repo: str | Path, source_root: str | Path = DEFAULT_SOURCE_ROOT,
    input_root: str | Path = DEFAULT_INPUT_ROOT,
) -> dict[str, Any]:
    started = time.perf_counter()
    pair = assay_pair_for_steward(str(q00["dataset"]))
    policy = resolve_family_policy(*pair)
    source = _load_reference(Path(source_root), str(q00["unit_id"]), policy)
    input_path, _ = _validate_night10a_input(Path(input_root), q00, source)
    replay_partition = _endpoint_partition(source, policy)
    replay_partition_sha = canonical_array_sha256(replay_partition)
    if replay_partition_sha != q00["reference_partition_sha256"]:
        raise RuntimeError("endpoint replay canonical partition SHA mismatch")
    impl_sha, _ = implementation_sha256(repo)
    resolved = policy.as_dict()
    binding = {
        "assay_pair": list(pair),
        "dataset_data_steward_id": str(q00["dataset"]),
        "implementation_sha256": impl_sha,
        "input_manifest_sha256": file_sha256(input_path),
        "label_objects_deserialized": 0,
        "ordered_observation_sha256": q00["ordered_observation_sha256"],
        "registered_input_sha256": q00["registered_input_sha256"],
        "replay_embedding_mode": "AUTHORITATIVE_NO_OP_ALIAS_REHASHED",
        "replay_embedding_sha256": q00["reference_embedding_sha256"],
        "replay_partition_sha256": replay_partition_sha,
        "resolved_family": policy.recipe.family,
        "resolved_policy_sha256": resolved["resolved_policy_sha256"],
        "resolved_recipe_id": policy.recipe.recipe_id,
        "resolved_recipe_config_sha256": policy.recipe.config_sha256,
        "seed": int(q00["seed"]),
        "source_artifacts": source["artifacts"],
        "source_reference_embedding_sha256": q00["reference_embedding_sha256"],
        "source_reference_partition_sha256": q00["reference_partition_sha256"],
        "unit_id": str(q00["unit_id"]),
    }
    row_sha = canonical_json_sha256(binding)
    return {
        **binding,
        "fresh_process": True,
        "label_reads": 0,
        "replay_manifest_sha256": row_sha,
        "runtime_seconds": time.perf_counter() - started,
        "status": "PASS",
    }


def verify_replay_row(
    manifest_path: str | Path, q00: Mapping[str, Any], repo: str | Path,
    source_root: str | Path = DEFAULT_SOURCE_ROOT, input_root: str | Path = DEFAULT_INPUT_ROOT,
) -> dict[str, Any]:
    saved = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    replayed = replay_q00_row(q00, repo, source_root, input_root)
    deterministic = {
        key: replayed[key]
        for key in replayed
        if key not in {"runtime_seconds", "fresh_process", "status", "label_reads"}
    }
    saved_deterministic = {key: saved.get(key) for key in deterministic}
    exact = deterministic == saved_deterministic
    if not exact:
        raise RuntimeError("fresh-process row replay binding mismatch")
    return {
        "fresh_process": True,
        "label_objects_deserialized": 0,
        "manifest_file_sha256": file_sha256(manifest_path),
        "row_binding_exact": True,
        "status": "PASS",
        "unit_id": q00["unit_id"],
    }


def authority_audit(
    repo: str | Path, q00_path: str | Path = DEFAULT_Q00_AUDIT,
    source_root: str | Path = DEFAULT_SOURCE_ROOT, input_root: str | Path = DEFAULT_INPUT_ROOT,
) -> dict[str, Any]:
    root = Path(repo)
    rows = load_q00_rows(root / q00_path if not Path(q00_path).is_absolute() else q00_path)
    impl_sha, source_rows = implementation_sha256(root)
    audited = []
    for q00 in rows:
        pair = assay_pair_for_steward(str(q00["dataset"]))
        policy = resolve_family_policy(*pair)
        reference = _load_reference(Path(source_root), q00["unit_id"], policy)
        input_path, _ = _validate_night10a_input(Path(input_root), q00, reference)
        audited.append({
            "assay_pair": list(pair),
            "data_steward_id": q00["dataset"],
            "input_manifest": _artifact_row(input_path),
            "reference_artifacts": reference["artifacts"],
            "reference_embedding_sha256": q00["reference_embedding_sha256"],
            "reference_partition_sha256": q00["reference_partition_sha256"],
            "resolved_family": policy.recipe.family,
            "resolved_recipe_id": policy.recipe.recipe_id,
            "seed": q00["seed"],
            "unit_id": q00["unit_id"],
        })
    return {
        "authority_rows": audited,
        "conflicting_frozen_implementations": 0,
        "implementation_sha256": impl_sha,
        "label_paths_reachable_from_worker": False,
        "label_reads": 0,
        "q00_rows": len(audited),
        "schema": "spalora.night10b.authority_audit.v1",
        "source_files": source_rows,
        "status": "PASS",
        "unique_recipe_sources": {
            "C00_G04_H05_CONFIRMED": [
                "SpaLORA/night6c_pipeline.py", "SpaLORA/night6d_pipeline.py"
            ],
            "F00_R02_FULL": [
                "SpaLORA/night7b_adaptive.py", "scripts/night7b_train.py",
                "scripts/night7b_adapter_stage.py",
            ],
        },
    }


def resource_snapshot(started: float) -> dict[str, Any]:
    try:
        import torch
        gpu = float(torch.cuda.max_memory_allocated() / 1048576.0) if torch.cuda.is_available() else 0.0
    except Exception:
        gpu = 0.0
    return {
        "peak_gpu_mib": gpu,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "runtime_seconds": time.perf_counter() - started,
    }


__all__ = [
    "DEFAULT_INPUT_ROOT", "DEFAULT_Q00_AUDIT", "DEFAULT_SOURCE_ROOT",
    "assay_pair_for_steward", "atomic_json", "atomic_npy", "authority_audit",
    "ensure_sparse_square", "file_sha256", "implementation_sha256",
    "load_q00_rows", "reject_dense_n_by_n", "replay_q00_row",
    "resource_snapshot", "verify_replay_row",
]
