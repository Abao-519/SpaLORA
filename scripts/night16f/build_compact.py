#!/usr/bin/env python3
"""Build the root-relative Night-16F compact delivery.

The compact intentionally excludes raw matrices, numeric carriers, partitions,
embeddings, and checkpoints.  Every copied file other than the root index is
indexed by root-relative path, byte size, and SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


REPO_FILES = (
    "SpaLORA/night16f_support_attribution.py",
    "configs/night16f/night16f_formula_and_ablation_contract.json",
    "tasks/night16f/Night16F_Formula_and_Ablation_Contract.md",
    "tests/test_night16f_support_attribution.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--bundle-verify", type=Path, required=True)
    parser.add_argument("--push-log", type=Path, required=True)
    parser.add_argument("--final-commit", required=True)
    parser.add_argument("--final-tag", required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    output = args.output.resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    for relative in REPO_FILES:
        copy_file(repo / relative, output / relative)

    for source in sorted((repo / "scripts/night16f").glob("*")):
        if source.is_file() and source.suffix in {".py", ".sh"}:
            relative = source.relative_to(repo)
            copy_file(source, output / relative)

    for source in sorted((repo / "outputs/night16f_handoff").rglob("*")):
        if source.is_file():
            relative = source.relative_to(repo)
            copy_file(source, output / relative)

    copy_file(args.bundle.resolve(), output / "bundle/night16f_incremental.bundle")
    copy_file(args.bundle_verify.resolve(), output / "audit/bundle_verify.txt")
    copy_file(args.push_log.resolve(), output / "audit/git_push.log")

    manifest = {
        "schema": "night16f-final-delivery-manifest-v1",
        "final_commit": args.final_commit,
        "final_tag": args.final_tag,
        "bundle_sha256": sha256(output / "bundle/night16f_incremental.bundle"),
        "classification": "LOCAL_SIGNAL",
        "status": "NIGHT16F_HUMAN_RELATION_SPECIFIC_SUPPORT_LOCAL_SIGNAL_NO_MELANOMA_TRANSFER",
        "compact_exclusions": [
            "raw data",
            "numeric carriers",
            "partitions",
            "embeddings",
            "checkpoints",
        ],
        "shutdown_dispatched": False,
        "instance_instruction": "KEEP_ON_FOR_NIGHTTIME_CHAINED_TASKS",
    }
    (output / "final_delivery_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    files = sorted(
        path for path in output.rglob("*")
        if path.is_file() and path.name != "compact_delivery_index.json"
    )
    rows = [
        {
            "path": path.relative_to(output).as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in files
    ]
    index = {
        "schema": "night16f-root-relative-compact-index-v1",
        "indexed_count": len(rows),
        "files": rows,
    }
    (output / "compact_delivery_index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "indexed_count": len(rows),
        "index_sha256": sha256(output / "compact_delivery_index.json"),
        "bundle_sha256": manifest["bundle_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
