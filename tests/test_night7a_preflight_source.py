from __future__ import annotations

import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "night7a_preflight.py"
SPEC = importlib.util.spec_from_file_location("night7a_preflight", SCRIPT)
assert SPEC and SPEC.loader
PREFLIGHT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREFLIGHT)


def test_partial_git_directory_is_not_a_verified_source(tmp_path, monkeypatch):
    monkeypatch.setattr(PREFLIGHT, "SOURCES", tmp_path)
    target = tmp_path / "Method"
    (target / ".git").mkdir(parents=True)

    actual, failures = PREFLIGHT.clone("Method", "https://github.com/example/repository")

    assert actual == target
    assert failures[0]["operation"] == "preexisting_incomplete_source"


def test_snapshot_requires_commit_branch_and_source_file(tmp_path):
    target = tmp_path / "source"
    target.mkdir()
    metadata = {
        "resolved_commit": "a" * 40,
        "default_branch": "main",
    }
    (target / ".night7a_source_snapshot.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    assert PREFLIGHT._snapshot_metadata(target) is None

    (target / "README.md").write_text("source", encoding="utf-8")
    assert PREFLIGHT._snapshot_metadata(target) == metadata


def test_safe_extract_rejects_path_traversal(tmp_path):
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        member = tarfile.TarInfo("../escape.txt")
        payload = b"not allowed"
        member.size = len(payload)
        handle.addfile(member, io.BytesIO(payload))

    with pytest.raises(RuntimeError, match="unsafe tar path"):
        PREFLIGHT._safe_extract(archive, tmp_path / "extract")

