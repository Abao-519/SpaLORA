#!/usr/bin/env python3
"""Build the allowlisted Night-12A compact and incremental Git bundle."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from SpaLORA.night12a_schema_p0 import atomic_json, file_sha256

REPO = Path(__file__).resolve().parents[2]
ROOT = Path("/root/autodl-fs/night12a_delivery_20260822/official_compact")
PARENT = "5eb3d881dfdf5e2cd1f553ccf0d51241e4fc044b"
TAG = "night12a-final-20260822"


def copy(relative: str):
    source = REPO / relative
    target = ROOT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def main():
    if ROOT.exists():
        raise FileExistsError(ROOT)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO,
                                   text=True).strip()
    peel = subprocess.check_output(["git", "rev-parse", f"{TAG}^{{commit}}"],
                                   cwd=REPO, text=True).strip()
    if head != peel:
        raise ValueError("final tag does not peel to HEAD")
    files = [
        "SpaLORA/night12a_schema_p0.py",
        "scripts/night12a/night12a_schema_audit.py",
        "scripts/night12a/night12a_smoke.py",
        "scripts/night12a/night12a_freeze.py",
        "scripts/night12a/night12a_finalize.py",
        "scripts/night12a/night12a_delivery.py",
        "tests/test_night12a_schema_p0.py",
        "configs/night12a/night12a_manifest_contract.json",
        "configs/night12a/night12a_schema_and_real_path_p0_contract.json",
    ]
    files += [str(path.relative_to(REPO)) for path in
              sorted((REPO / "outputs/night12a_handoff").iterdir()) if path.is_file()]
    for relative in files:
        copy(relative)
    bundle = ROOT / "git/night12a_incremental.bundle"
    bundle.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "bundle", "create", str(bundle), TAG, f"^{PARENT}"],
                   cwd=REPO, check=True)
    subprocess.run(["git", "bundle", "verify", str(bundle)], cwd=REPO, check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    heads = subprocess.check_output(["git", "bundle", "list-heads", str(bundle)],
                                    cwd=REPO, text=True).splitlines()
    if not any(line.split()[0] == head and line.endswith(f"refs/tags/{TAG}") for line in heads):
        raise ValueError("bundle does not list the unique final tag")
    manifest = {
        "schema": "spalora.night12a.delivery_manifest.v1",
        "final_commit": head, "final_tag": TAG, "parent_commit": PARENT,
        "bundle_path": "git/night12a_incremental.bundle",
        "bundle_sha256": file_sha256(bundle),
        "bundle_list_heads": heads,
        "raw_or_large_artifacts_included": False,
    }
    atomic_json(ROOT / "delivery_manifest.json", manifest)
    records = []
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and path.name != "compact_delivery_index.json":
            records.append({"path": path.relative_to(ROOT).as_posix(),
                            "size": path.stat().st_size,
                            "sha256": file_sha256(path)})
    index = {
        "schema": "spalora.night12a.compact_index.v1",
        "root_relative": True,
        "indexed_file_count": len(records),
        "files": records,
        "validation": {"missing": 0, "size_mismatch": 0,
                       "sha_mismatch": 0, "extras": 0},
    }
    atomic_json(ROOT / "compact_delivery_index.json", index)
    print(json.dumps({"indexed": len(records),
                      "index_sha256": file_sha256(ROOT / "compact_delivery_index.json"),
                      "bundle_sha256": manifest["bundle_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
