import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/night7a_package.py"
SPEC = importlib.util.spec_from_file_location("night7a_package", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_root_aware_delivery_index_and_byte_drift(monkeypatch, tmp_path):
    repo = tmp_path / "repo"; out = repo / "outputs/night7a_handoff"
    repo.mkdir(); out.mkdir(parents=True)
    repo_file = repo / "code.py"; output_file = out / "result.json"
    repo_file.write_text("code\n", encoding="utf-8")
    output_file.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(MODULE, "REPO", repo)
    monkeypatch.setattr(MODULE, "OUT", out)
    index = {"files": [
        {"path": "code.py", "root": "repo", "size_bytes": repo_file.stat().st_size,
         "sha256": MODULE.sha256_file(repo_file)},
        {"path": "result.json", "size_bytes": output_file.stat().st_size,
         "sha256": MODULE.sha256_file(output_file)},
    ]}
    MODULE.verify_internal(index)
    output_file.write_text("drift\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="internal index mismatch"):
        MODULE.verify_internal(index)


def test_final_tag_and_remote_ref_guard():
    commit = "a" * 40
    MODULE.validate_final_refs(commit, commit, f"{commit}\trefs/heads/revision", "tagline", f"{commit}\tpeeled")
    with pytest.raises(RuntimeError, match="does not peel"):
        MODULE.validate_final_refs(commit, "b" * 40, "", "", "")
    with pytest.raises(RuntimeError, match="remote branch"):
        MODULE.validate_final_refs(commit, commit, "", "", "")
