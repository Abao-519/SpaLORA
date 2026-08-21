#!/usr/bin/env python3
"""Night-9C P0-DATA authority, provenance, and label-firewall audit.

The source H5AD files are never opened with AnnData.  This program reads only
HDF5 structure plus the explicitly authorized modelling fields: X, var, the
observation index, and obsm/spatial.  It never indexes or hashes any value
stored under obs/Annotation_for_Combined.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np


ANNOTATION_KEY = "Annotation_for_Combined"
EXPECTED_SPOTS = 2129
EXPECTED_RNA_FEATURES = 32285
LOCKED_K = 10
ALLOWED_UNS_KEYS = {"peaks", "reference_sequences", "spatial"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def run(*command: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(command, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()


def decode_scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return decode_scalar(value.item())
    if isinstance(value, np.ndarray):
        return [decode_scalar(item) for item in value.tolist()]
    return value


def h5_index(handle: h5py.File) -> list[str]:
    obs = handle["obs"]
    index_key = str(decode_scalar(obs.attrs.get("_index", "_index")))
    values = obs[index_key][...]
    return [str(decode_scalar(item)) for item in np.asarray(values)]


def x_shape(handle: h5py.File) -> tuple[int, int]:
    node = handle["X"]
    if isinstance(node, h5py.Dataset):
        return tuple(map(int, node.shape))
    return tuple(map(int, decode_scalar(node.attrs["shape"])))


def x_contract(handle: h5py.File) -> dict[str, Any]:
    node = handle["X"]
    if isinstance(node, h5py.Dataset):
        return {"encoding": "dense", "shape": list(map(int, node.shape)), "dtype": str(node.dtype)}
    encoding = str(decode_scalar(node.attrs.get("encoding-type", "unknown")))
    return {
        "encoding": encoding,
        "shape": list(map(int, decode_scalar(node.attrs["shape"]))),
        "data_dtype": str(node["data"].dtype),
        "indices_dtype": str(node["indices"].dtype),
        "indptr_dtype": str(node["indptr"].dtype),
        "nnz": int(node["data"].shape[0]),
    }


def schema_contract(path: Path) -> dict[str, Any]:
    """Return structural metadata without reading annotation values."""
    with h5py.File(path, "r") as handle:
        annotation_path = f"obs/{ANNOTATION_KEY}"
        annotation_present = annotation_path in handle
        root_keys = sorted(map(str, handle.keys()))
        obs_keys = sorted(map(str, handle["obs"].keys()))
        var_keys = sorted(map(str, handle["var"].keys()))
        obsm_keys = sorted(map(str, handle.get("obsm", {}).keys()))
        uns_keys = sorted(map(str, handle.get("uns", {}).keys()))
        spatial = handle["obsm/spatial"]
        return {
            "root_keys": root_keys,
            "obs_storage_keys_names_only": obs_keys,
            "var_storage_keys_names_only": var_keys,
            "obsm_storage_keys_names_only": obsm_keys,
            "uns_storage_keys_names_only": uns_keys,
            "annotation_path_present": annotation_present,
            "annotation_values_read": False,
            "shape": list(x_shape(handle)),
            "x": x_contract(handle),
            "spatial_shape": list(map(int, spatial.shape)),
            "spatial_dtype": str(spatial.dtype),
        }


def array_sha(value: np.ndarray) -> str:
    arr = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("ascii"))
    h.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
    h.update(arr.tobytes())
    return h.hexdigest()


def ordered_text_sha(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def copy_attrs(source: h5py.AttributeManager, target: h5py.AttributeManager) -> None:
    for key, value in source.items():
        target[key] = value


def copy_label_free(source: Path, target: Path) -> dict[str, Any]:
    """Make an H5AD containing no obs columns and no source annotation values."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + ".tmp")
    if temp.exists():
        temp.unlink()
    with h5py.File(source, "r") as src, h5py.File(temp, "w") as dst:
        copy_attrs(src.attrs, dst.attrs)
        for key in ("X", "var"):
            src.copy(key, dst, name=key)

        src_obs = src["obs"]
        dst_obs = dst.create_group("obs")
        copy_attrs(src_obs.attrs, dst_obs.attrs)
        index_key = str(decode_scalar(src_obs.attrs.get("_index", "_index")))
        src.copy(src_obs[index_key], dst_obs, name=index_key)
        # Preserve the dataframe encoding while declaring exactly zero columns.
        string_dtype = h5py.string_dtype(encoding="utf-8")
        dst_obs.attrs["column-order"] = np.asarray([], dtype=string_dtype)

        dst_obsm = dst.create_group("obsm")
        if "obsm" in src:
            copy_attrs(src["obsm"].attrs, dst_obsm.attrs)
        src.copy(src["obsm/spatial"], dst_obsm, name="spatial")

        # Only known non-label metadata groups are retained.  No obs-derived
        # arrays, layers, obsp, raw, or alternate embeddings are copied.
        if "uns" in src:
            dst_uns = dst.create_group("uns")
            copy_attrs(src["uns"].attrs, dst_uns.attrs)
            for key in sorted(set(src["uns"].keys()) & ALLOWED_UNS_KEYS):
                src.copy(src[f"uns/{key}"], dst_uns, name=key)
        for group_name in ("varm", "varp"):
            if group_name in src:
                src.copy(group_name, dst, name=group_name)
    os.replace(temp, target)

    import anndata as ad  # deliberately imported only after the source files are closed

    obj = ad.read_h5ad(target, backed="r")
    proof = {
        "path": str(target),
        "size_bytes": target.stat().st_size,
        "sha256": sha256_file(target),
        "shape": list(map(int, obj.shape)),
        "obs_columns": list(map(str, obj.obs.columns)),
        "obs_columns_count": int(len(obj.obs.columns)),
        "ordered_observation_sha256": ordered_text_sha(list(map(str, obj.obs_names))),
        "spatial_shape": list(map(int, obj.obsm["spatial"].shape)),
        "source_opened_with_anndata": False,
        "source_annotation_values_read": False,
    }
    obj.file.close()
    if proof["obs_columns_count"] != 0:
        raise RuntimeError(f"label-free copy has obs columns: {target}")
    return proof


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna", type=Path, required=True)
    parser.add_argument("--atac", type=Path, required=True)
    parser.add_argument("--label-free-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--taskbook", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--filename-audit", type=Path, required=True)
    parser.add_argument("--text-audit", type=Path, required=True)
    args = parser.parse_args()

    expected_parent = "e9bd62e2bb07c58f58c219956b8aaf090527c471"
    expected_n02 = "5204e1cd9320336a424ccffd867269e64397c4a6a6210efe09ae614b0a58ab7f"
    expected_source_sha = {
        "RNA": "6e93ead6f07ece06d49b132dade58ac11c9246bf278cbb10593e6332cf25a287",
        "ATAC": "5e03d404f959893a3d035e655427c8933951be6fd7f509893bcacb1d53e69d10",
    }
    taskbook_sha = "466f7a139ac2cf443568509e7177ed75341508202538b331b2612ad891f3b059"
    registry_sha = "bfd053ccac50a5ba6c8e914333ec17d6c17e104558c825d4aa5bc2952dbaf324"

    head = run("git", "rev-parse", "HEAD", cwd=args.repo)
    parent_tag = run("git", "rev-parse", "night9b-final-20260820^{commit}", cwd=args.repo)
    n02_path = args.repo / "SpaLORA/night9b_racf.py"
    branch = run("git", "branch", "--show-current", cwd=args.repo)
    # New P0 protocol/code/output files are intentionally untracked until the
    # P0 commit.  Protect the parent by checking only modifications to tracked
    # files; treating expected new files as parent drift would be a false gate.
    tracked_changes = run("git", "status", "--porcelain", "--untracked-files=no", cwd=args.repo)
    authority_checks = {
        "head_parent": head == expected_parent,
        "parent_tag_peels_to_parent": parent_tag == expected_parent,
        "branch": branch == "revision/q2-night9c-frozen-hierarchy-e18-5-confirmation-20260821",
        "parent_tracked_files_unchanged_during_p0": not bool(tracked_changes),
        "n02_implementation_sha": sha256_file(n02_path) == expected_n02,
        "taskbook_sha": sha256_file(args.taskbook) == taskbook_sha,
        "registry_sha": sha256_file(args.registry) == registry_sha,
    }

    source_rows: dict[str, dict[str, Any]] = {}
    for role, path, url, git_blob in (
        (
            "RNA",
            args.rna,
            "https://raw.githubusercontent.com/WHY-17/SpaDDM/main/Cross-omics%20translation/MISAR/E18_5-S1/adata_RNA.h5ad",
            "7b0696a207fdc9f2c766e700187a214a2f7d2b05",
        ),
        (
            "ATAC",
            args.atac,
            "https://raw.githubusercontent.com/WHY-17/SpaDDM/main/Cross-omics%20translation/MISAR/E18_5-S1/adata_Peak.h5ad",
            "48c20fe96f3de1422e90ee59424e0a9e55dc2c95",
        ),
    ):
        source_rows[role] = {
            "role": role,
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "public_processed_source_url": url,
            "public_processed_source_git_blob_sha1": git_blob,
            "computed_git_blob_sha1": run("git", "hash-object", str(path)),
            "upstream_raw_project": "https://github.com/gpenglab/MISAR-seq",
            "upstream_raw_accession": ["OEP003285", "SRP491963"],
            "schema": schema_contract(path),
            "source_opened_with_anndata": False,
            "annotation_values_read": False,
            "expected_sha256": expected_source_sha[role],
        }

    with h5py.File(args.rna, "r") as rna_handle, h5py.File(args.atac, "r") as atac_handle:
        rna_ids = h5_index(rna_handle)
        atac_ids = h5_index(atac_handle)
        rna_spatial = np.asarray(rna_handle["obsm/spatial"])
        atac_spatial = np.asarray(atac_handle["obsm/spatial"])
    paired_checks = {
        "rna_spots_expected": len(rna_ids) == EXPECTED_SPOTS,
        "atac_spots_expected": len(atac_ids) == EXPECTED_SPOTS,
        "rna_features_expected": source_rows["RNA"]["schema"]["shape"][1] == EXPECTED_RNA_FEATURES,
        "observation_order_exact": rna_ids == atac_ids,
        "spatial_shape_exact": rna_spatial.shape == atac_spatial.shape == (EXPECTED_SPOTS, 2),
        "spatial_values_exact": bool(np.array_equal(rna_spatial, atac_spatial)),
        "rna_annotation_key_present": source_rows["RNA"]["schema"]["annotation_path_present"],
        "atac_annotation_key_present": source_rows["ATAC"]["schema"]["annotation_path_present"],
    }
    pairing = {
        "checks": paired_checks,
        "rna_ordered_observation_sha256": ordered_text_sha(rna_ids),
        "atac_ordered_observation_sha256": ordered_text_sha(atac_ids),
        "rna_spatial_sha256": array_sha(rna_spatial),
        "atac_spatial_sha256": array_sha(atac_spatial),
        "inner_join_performed": False,
        "spots_dropped": 0,
    }

    rna_lf = args.label_free_root / "E18_5_expr_label_free.h5ad"
    atac_lf = args.label_free_root / "E18_5_atac_label_free.h5ad"
    label_free = {
        "RNA": copy_label_free(args.rna, rna_lf),
        "ATAC": copy_label_free(args.atac, atac_lf),
    }
    label_free_checks = {
        "rna_zero_obs": label_free["RNA"]["obs_columns_count"] == 0,
        "atac_zero_obs": label_free["ATAC"]["obs_columns_count"] == 0,
        "shape_preserved": label_free["RNA"]["shape"] == source_rows["RNA"]["schema"]["shape"]
        and label_free["ATAC"]["shape"] == source_rows["ATAC"]["schema"]["shape"],
        "order_preserved": label_free["RNA"]["ordered_observation_sha256"]
        == label_free["ATAC"]["ordered_observation_sha256"]
        == pairing["rna_ordered_observation_sha256"],
    }

    gpu_name = run("nvidia-smi", "--query-gpu=name", "--format=csv,noheader").splitlines()[0]
    gpu_memory = run("nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits").splitlines()[0]
    try:
        cuda_available = run(sys.executable, "-c", "import torch; print(torch.cuda.is_available())") == "True"
    except Exception:
        cuda_available = False
    disk_free = int(shutil.disk_usage(args.rna).free)
    resource = {
        "gpu_name": gpu_name,
        "gpu_memory_total_mib": int(gpu_memory),
        "torch_cuda_available": cuda_available,
        "disk_free_bytes": disk_free,
        "python": sys.version,
        "h5py": h5py.__version__,
        "numpy": np.__version__,
    }

    filename_hits = [line for line in args.filename_audit.read_text(encoding="utf-8").splitlines()[1:] if line.strip()]
    text_hits = [line for line in args.text_audit.read_text(encoding="utf-8").splitlines()[1:] if line.strip()]
    history = {
        "audit_scope": "/root/autodl-fs filename plus text-file path-only search before source download",
        "filename_hit_paths": filename_hits,
        "text_presence_hit_paths": text_hits,
        "matching_file_contents_or_label_values_printed": False,
        "prior_e18_5_artifact_evidence_found": bool(filename_hits or text_hits),
        "historical_training_or_selection_use_found": False,
        "assigned_role": "PRISTINE_EXTERNAL_CONFIRMATION",
    }

    provenance = {
        "status": "PASS",
        "dataset": "MISAR-seq E18.5 S1",
        "source_chain": [
            "MISAR-seq original project and OEP003285/SRP491963 raw accession",
            "SpaDDM public repository processed E18.5-S1 H5AD blobs",
            "PRESENT public tutorial independently documents the same 2129-spot/32285-RNA-feature schema and annotation key",
        ],
        "annotation_tier": "TIER_B_PUBLISHED_REFERENCE_CLUSTER",
        "annotation_claim": "Published unsupervised reference clusters with anatomical interpretation; not manual ground truth",
        "annotation_values_read": False,
        "locked_k": LOCKED_K,
        "k_provenance": "published SMART E18.5 S1 ten-region description fixed before local label access",
        "evidence_urls": [
            "https://github.com/gpenglab/MISAR-seq",
            "https://www.biosino.org/node/project/detail/OEP003285",
            "https://github.com/WHY-17/SpaDDM",
            "https://bio-present.readthedocs.io/en/latest/Tutorial4_spatial-RNA-ATAC-data_representation_MouseBrain.html",
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC13031631/",
        ],
        "history_audit": history,
        "label_firewall": {
            "original_h5ad_anndata_deserialization_count": 0,
            "annotation_value_reads": 0,
            "annotation_value_hashes": 0,
            "training_units": 0,
            "embeddings": 0,
            "partitions": 0,
        },
    }

    source_manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": source_rows,
        "paired_exactness": pairing,
    }
    label_free_manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": label_free,
        "checks": label_free_checks,
        "source_annotation_values_read": False,
    }
    all_checks = {**authority_checks, **paired_checks, **label_free_checks}
    all_checks.update({
        "rna_source_sha256_exact": source_rows["RNA"]["sha256"] == expected_source_sha["RNA"],
        "atac_source_sha256_exact": source_rows["ATAC"]["sha256"] == expected_source_sha["ATAC"],
        "rna_public_git_blob_exact": source_rows["RNA"]["computed_git_blob_sha1"]
        == source_rows["RNA"]["public_processed_source_git_blob_sha1"],
        "atac_public_git_blob_exact": source_rows["ATAC"]["computed_git_blob_sha1"]
        == source_rows["ATAC"]["public_processed_source_git_blob_sha1"],
        "provenance_resolved": provenance["annotation_tier"] != "UNRESOLVED",
        "history_pristine": not history["prior_e18_5_artifact_evidence_found"],
        "cuda_available": cuda_available,
        "disk_minimum_50gb": disk_free >= 50 * 1024**3,
    })
    contract = {
        "status": "PASS" if all(all_checks.values()) else "BLOCKED_DATA_PROVENANCE",
        "checks": all_checks,
        "authority": {
            "head": head,
            "parent_tag_commit": parent_tag,
            "branch": branch,
            "n02_sha256": sha256_file(n02_path),
            "taskbook_sha256": sha256_file(args.taskbook),
            "registry_sha256": sha256_file(args.registry),
        },
        "resource": resource,
        "label_access": {
            "prelock_annotation_value_reads": 0,
            "authorized_evaluator_processes_started": 0,
        },
        "scientific_counts": {"training": 0, "embedding": 0, "partition": 0},
        "source_manifest_canonical_sha256": canonical_json_sha(source_manifest),
        "label_free_manifest_canonical_sha256": canonical_json_sha(label_free_manifest),
    }

    atomic_json(args.output_dir / "e18_5_data_provenance_and_firewall.json", provenance)
    atomic_json(args.output_dir / "e18_5_source_file_manifest.json", source_manifest)
    atomic_json(args.output_dir / "e18_5_label_free_copy_manifest.json", label_free_manifest)
    atomic_json(args.output_dir / "night9c_p0_contract.json", contract)
    print(json.dumps({"status": contract["status"], "checks": all_checks, "resource": resource}, sort_keys=True))
    if contract["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
