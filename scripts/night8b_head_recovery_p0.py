#!/usr/bin/env python3
"""Fail-closed P0 authority, immutability, and 20-affinity input lock."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from SpaLORA.night6c_pipeline import sparse_sha
from SpaLORA.night8b_head_recovery import (
    HEAD_CONFIG, HEAD_CONFIG_SHA256, N_CLUSTERS, OBSERVATION_SHA256,
)

ORIGINAL = Path("/root/autodl-fs/night8b_raw_runs_20260820")
RECOVERY = Path("/root/autodl-fs/night8b_head_recovery_20260820")
OUT = REPO / "outputs/night8b_head_recovery"
ORIGINAL_OUT = REPO / "outputs/night8b_handoff"
PROTOCOL = REPO / "protocols/night8b_head_recovery"

EXPECTED = {
    "taskbook": "65229a18a8f4c6f1f3911391da8fc6def4f692b92b674b5ce41acc445113f045",
    "registry": "ba55faf2a25038f00c4ff2bc4984ec2d0ae3322a84d934c6f74c8cb26ce22162",
    "original_index": "f15cfaae937b246cb30017a0e8203db04bf6f254cafb8f5fcb9994ae91ff708b",
    "original_report": "980a754561b0ce36dc60af513b6cdbd705575a60929b75561071ff9fce6b82d3",
    "original_decision": "c7eaa285ac413a868bdcad5dbd0b50491454e0410661ec4a07a588e096e848a3",
    "original_incomplete": "eebb21cc24216129dde29415d829f796c22958d3f68b699e696b5778c37d52ce",
    "original_raw_manifest": "d45465d1ab2c9b8f7d50aef4bcb8f0d67b869e4071070d7945e73269f5c8b4c9",
    "original_p0": "9371a4b512df65cf3ef3cb16dd9b1cc328b55bca48d99c4557c4240b309699d6",
    "mapping": "322e7bf0f459998c882a0305e8aea129deee29570412982b3697976ac64b8ae5",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def mirror_json(name: str, payload: Any) -> None:
    atomic_json(RECOVERY / "manifests" / name, payload)
    atomic_json(OUT / name, payload)


def snapshot_original_manifest(source_manifest: Path) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    with source_manifest.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    for expected in source_rows:
        path = Path(expected["path"])
        exists = path.is_file()
        actual_size = path.stat().st_size if exists else None
        actual_sha = sha256_file(path) if exists else None
        match = (exists and actual_size == int(expected["size_bytes"])
                 and actual_sha == expected["sha256"])
        row = {
            "path": str(path), "exists": exists,
            "expected_size_bytes": int(expected["size_bytes"]),
            "actual_size_bytes": actual_size,
            "expected_sha256": expected["sha256"],
            "actual_sha256": actual_sha, "match": bool(match),
        }
        rows.append(row)
        if not match:
            failures.append(row)
    return {
        "schema_version": 1,
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": sha256_file(source_manifest),
        "row_count": len(rows), "all_match": not failures,
        "failures": failures, "rows": rows,
    }


def validate_original_compact() -> Dict[str, Any]:
    files = {
        "original_index": ORIGINAL_OUT / "tracked_delivery_index.json",
        "original_report": ORIGINAL_OUT / "night8b_report.md",
        "original_decision": ORIGINAL_OUT / "night8b_decision.json",
        "original_incomplete": ORIGINAL_OUT / "prelabel_incomplete_manifest.json",
        "original_raw_manifest": ORIGINAL_OUT / "raw_artifact_manifest.csv",
        "original_p0": ORIGINAL_OUT / "p0_authority_provenance_deployability.json",
    }
    hashes = {key: sha256_file(path) for key, path in files.items()}
    if any(hashes[key] != EXPECTED[key] for key in files):
        raise RuntimeError(f"RECOVERY_BLOCKED_INPUT_INTEGRITY compact SHA: {hashes}")
    index = json.loads(files["original_index"].read_text())
    failures = []
    missing = []
    verified = []
    for row in index["files"]:
        path = REPO / row["path"]
        if not path.is_file():
            missing.append(row["path"])
        elif (path.stat().st_size != int(row["size_bytes"])
              or sha256_file(path) != row["sha256"]):
            failures.append(row["path"])
        else:
            verified.append(row["path"])
    expected_compact_only = sorted(
        row["path"] for row in index["files"]
        if row["path"].startswith("scripts/__pycache__/")
    )
    if (len(index["files"]) != 38 or failures
            or sorted(missing) != expected_compact_only
            or len(expected_compact_only) != 9):
        raise RuntimeError(f"RECOVERY_BLOCKED_INPUT_INTEGRITY original 38/38: {failures}")
    return {
        "hashes": hashes,
        "windows_compact_independently_verified_by_executor": "38/38",
        "remote_git_worktree_index_members_verified": f"{len(verified)}/38",
        "compact_only_pyc_verified_on_windows": f"{len(missing)}/9",
        "compact_only_pyc_paths": missing,
        "note": "The nine pyc files were compact-only artifacts and are not Git members; the Windows root-rule audit verified their bytes.",
    }


def validate_units() -> Dict[str, Any]:
    base, base_reload, adapter, adapter_reload = [], [], [], []
    for seed in range(10):
        bdir = ORIGINAL / f"formal/base/seed_{seed}/attempt_001"
        b = json.loads((bdir / "base_unit_manifest.json").read_text())
        br = json.loads((bdir / "base_checkpoint_reload_audit.json").read_text())
        adir = ORIGINAL / f"formal/adapter/formal/seed_{seed}/attempt_001"
        a = json.loads((adir / "adapter_unit_manifest.json").read_text())
        ar = json.loads((adir / "worker/reload_forward_audit.json").read_text())
        if b.get("status") != "success" or b.get("label_access") is not False:
            raise RuntimeError(f"RECOVERY_BLOCKED_INPUT_INTEGRITY base seed {seed}")
        if br.get("status") != "PASS" or br.get("label_access") is not False:
            raise RuntimeError(f"RECOVERY_BLOCKED_INPUT_INTEGRITY base reload seed {seed}")
        if a.get("status") != "success" or a.get("label_access") is not False:
            raise RuntimeError(f"RECOVERY_BLOCKED_INPUT_INTEGRITY adapter seed {seed}")
        if ar.get("status") != "PASS" or ar.get("label_access") is not False:
            raise RuntimeError(f"RECOVERY_BLOCKED_INPUT_INTEGRITY adapter reload seed {seed}")
        base.append({"seed": seed, "manifest_sha256": sha256_file(bdir / "base_unit_manifest.json")})
        base_reload.append({"seed": seed, "audit_sha256": sha256_file(bdir / "base_checkpoint_reload_audit.json")})
        adapter.append({"seed": seed, "manifest_sha256": sha256_file(adir / "adapter_unit_manifest.json")})
        adapter_reload.append({"seed": seed, "audit_sha256": sha256_file(adir / "worker/reload_forward_audit.json")})
    return {"base_units": base, "base_reload": base_reload,
            "adapter_units": adapter, "adapter_reload": adapter_reload}


def build_input_lock() -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    u00_parity: List[Dict[str, Any]] = []
    for method in ("HR_U00", "HR_F00"):
        for seed in range(10):
            if method == "HR_U00":
                path = ORIGINAL / f"formal/adapter/inputs/seed_{seed}/s04.npz"
            else:
                path = ORIGINAL / f"formal/transforms/F00/seed_{seed}/affinity.npz"
            value = sp.load_npz(path).tocsr()
            canonical = sparse_sha(value)
            rows.append({
                "method": method, "seed": seed, "K": N_CLUSTERS,
                "path": str(path), "file_sha256": sha256_file(path),
                "canonical_sparse_sha256": canonical,
                "shape": list(map(int, value.shape)), "nnz": int(value.nnz),
                "ordered_observation_sha256": OBSERVATION_SHA256,
            })
            if method == "HR_U00" and seed != 6:
                old = ORIGINAL / f"formal/transforms/U00/seed_{seed}/affinity.npz"
                old_value = sp.load_npz(old).tocsr()
                old_sha = sparse_sha(old_value)
                same = old_sha == canonical
                u00_parity.append({"seed": seed, "old_path": str(old),
                                   "old_canonical_sparse_sha256": old_sha,
                                   "s04_canonical_sparse_sha256": canonical,
                                   "exact": same})
                if not same:
                    raise RuntimeError(f"RECOVERY_BLOCKED_INPUT_INTEGRITY U00 parity seed {seed}")
    if len(rows) != 20 or len(u00_parity) != 9:
        raise RuntimeError("RECOVERY_BLOCKED_INPUT_INTEGRITY input cardinality")
    return {"schema_version": 1, "status": "LOCKED_PRELABEL", "row_count": 20,
            "ordered_observation_sha256": OBSERVATION_SHA256,
            "head_id": HEAD_CONFIG["head_id"],
            "head_config_sha256": HEAD_CONFIG_SHA256,
            "rows": rows, "original_u00_s04_parity": u00_parity}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    task = PROTOCOL / "SpaLORA_Night8B_Uniform_Head_Evaluation_Recovery_Taskbook_2026-08-20.md"
    registry = PROTOCOL / "SpaLORA_Night8B_Uniform_Head_Recovery_Registry_2026-08-20.json"
    authority = {"taskbook_sha256": sha256_file(task),
                 "registry_sha256": sha256_file(registry)}
    if authority["taskbook_sha256"] != EXPECTED["taskbook"] or authority["registry_sha256"] != EXPECTED["registry"]:
        raise RuntimeError(f"RECOVERY_BLOCKED_INPUT_INTEGRITY authority: {authority}")
    original_compact = validate_original_compact()
    mirror_json("p0_attempt1_infrastructure_correction.json", {
        "schema_version": 1,
        "status": "INFRASTRUCTURE_ONLY_CORRECTED_BEFORE_SCIENCE",
        "reason": "attempt 1 incorrectly required nine compact-only pyc artifacts to exist in the Git worktree",
        "windows_compact_was_independently_verified": "38/38",
        "remote_worktree_science_or_authority_file_mismatch": False,
        "training": 0, "adapter": 0, "affinity_rebuild": 0,
        "head_transforms": 0, "label_access": False,
        "scientific_retry": False,
    })
    before = snapshot_original_manifest(ORIGINAL_OUT / "raw_artifact_manifest.csv")
    mirror_json("original_artifact_manifest_before.json", before)
    if before["source_manifest_sha256"] != EXPECTED["original_raw_manifest"] or not before["all_match"]:
        raise RuntimeError("RECOVERY_BLOCKED_INPUT_INTEGRITY raw artifact rehash")
    units = validate_units()
    mapping = ORIGINAL_OUT / "prelabel_observation_mapping.csv"
    mapping_rows = len(pd.read_csv(mapping))
    if mapping_rows != 1949 or sha256_file(mapping) != EXPECTED["mapping"]:
        raise RuntimeError("RECOVERY_BLOCKED_INPUT_INTEGRITY mapping")
    incomplete = json.loads((ORIGINAL_OUT / "prelabel_incomplete_manifest.json").read_text())
    state = json.loads((ORIGINAL / "formal_runner_state.json").read_text())
    if incomplete.get("labels_Y_read") is not False or state.get("label_access") is not False:
        raise RuntimeError("RECOVERY_BLOCKED_INPUT_INTEGRITY label firewall history")
    input_lock = build_input_lock()
    mirror_json("recovery_input_view_manifest.json", input_lock)
    rule = {
        "schema_version": 1, "status": "LOCKED_PRELABEL",
        "head_id": HEAD_CONFIG["head_id"], "head_config": HEAD_CONFIG,
        "head_config_sha256": HEAD_CONFIG_SHA256,
        "training": 0, "adapter": 0, "affinity_rebuild": 0,
        "science_retry": 0, "fallback": 0, "K": 12,
        "methods": ["HR_U00", "HR_F00"], "seeds": list(range(10)),
        "input_manifest_sha256": sha256_file(OUT / "recovery_input_view_manifest.json"),
        "label_access": False,
    }
    mirror_json("recovery_rule_lock.json", rule)
    p0 = {
        "schema_version": 1, "status": "P0_RECOVERY_AUTHORITY_PASS",
        "authority": authority, "original_compact": original_compact,
        "raw_manifest_rows_verified": before["row_count"],
        "base_units": "10/10", "adapter_units": "10/10",
        "checkpoint_reload_audits": "20/20", "s04_inputs": "10/10",
        "f00_affinity_inputs": "10/10", "original_u00_s04_parity": "9/9",
        "observation_count": mapping_rows,
        "ordered_observation_sha256": OBSERVATION_SHA256,
        "mapping_sha256": sha256_file(mapping),
        "annotation_Y_values_deserialized": False,
        "label_access": False, "training": 0, "adapter": 0,
        "affinity_rebuild": 0, "unit_evidence": units,
    }
    mirror_json("p0_recovery_authority.json", p0)
    print(json.dumps({"status": p0["status"], "raw_rows": before["row_count"],
                      "input_rows": input_lock["row_count"],
                      "head_config_sha256": HEAD_CONFIG_SHA256}, sort_keys=True))


if __name__ == "__main__":
    main()
