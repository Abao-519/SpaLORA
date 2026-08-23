#!/usr/bin/env python3
"""Build the Night-15A compact with a root-relative size/SHA-256 index."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


FORBIDDEN_SUFFIXES = {
    ".npy", ".npz", ".pt", ".pth", ".ckpt", ".h5", ".h5ad", ".mtx",
    ".png", ".jpg", ".jpeg", ".tif", ".tiff",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def copy_tree(source: Path, destination: Path) -> None:
    for path in sorted(source.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def run(repo: Path, bundle: Path, git_audit: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    selected = [
        "SpaLORA/night15a_mcdf.py",
        "configs/night15a",
        "contracts/night15a",
        "outputs/night15a_handoff",
        "scripts/night15a",
        "tests/night15a",
    ]
    for relative in selected:
        source = repo / relative
        destination = output / relative
        if source.is_dir():
            copy_tree(source, destination)
        elif source.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        else:
            raise FileNotFoundError(source)
    shutil.copy2(bundle, output / "night15a_incremental.bundle")
    shutil.copy2(git_audit, output / "git_and_delivery_audit.json")

    files = []
    forbidden = []
    large_artifact_tokens = ("checkpoint", "embedding", "affinity", "raw_data")
    suspicious = []
    for path in sorted(candidate for candidate in output.rglob("*") if candidate.is_file()):
        relative = path.relative_to(output).as_posix()
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            forbidden.append(relative)
        if any(token in relative.lower() for token in large_artifact_tokens):
            # JSON/CSV audits may mention these concepts; binary payloads are already suffix-blocked.
            if path.suffix.lower() not in {".json", ".csv", ".md", ".txt", ".py"}:
                suspicious.append(relative)
        files.append({"path": relative, "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    if forbidden or suspicious:
        raise RuntimeError(f"forbidden compact payloads: suffix={forbidden}, suspicious={suspicious}")
    index = {
        "schema_version": "night15a-root-relative-size-sha256-v1",
        "index_file_excluded_from_entries": True,
        "indexed_count": len(files),
        "total_size_bytes": sum(row["size_bytes"] for row in files),
        "forbidden_extension_count": 0,
        "raw_checkpoint_embedding_affinity_count": 0,
        "expected_missing": 0,
        "expected_size_mismatch": 0,
        "expected_sha_mismatch": 0,
        "expected_extras": 0,
        "files": files,
    }
    atomic_json(output / "compact_delivery_index.json", index)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--git-audit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(Path(args.repo), Path(args.bundle), Path(args.git_audit), Path(args.output))


if __name__ == "__main__":
    main()
