#!/usr/bin/env python3
"""Build the small Night-7B handoff without copying raw arrays or checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RAW = Path("/root/autodl-fs/night7b_score_rnd_20260818")
OUT = REPO / "outputs/night7b_handoff"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def run(*args) -> str:
    return subprocess.check_output(args, cwd=REPO, text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--bundle", type=Path, required=True); args = parser.parse_args()
    target = RAW / "official_compact"
    if target.exists():
        raise RuntimeError("compact target already exists; preserve it and use an audited new attempt")
    target.mkdir(parents=True)
    for source in sorted(OUT.rglob("*")):
        if source.is_file():
            copy_file(source, target / "handoff" / source.relative_to(OUT))
    for source in sorted((REPO / "protocols/night7b").rglob("*")):
        if source.is_file():
            copy_file(source, target / "protocols/night7b" / source.relative_to(REPO / "protocols/night7b"))
    for relative in (
        "SpaLORA/night7b_adaptive.py", "scripts/night7b_p0.py", "scripts/night7b_train.py",
        "scripts/night7b_heads.py", "scripts/night7b_evaluate.py", "scripts/night7b_adapter_stage.py",
        "scripts/night7b_adapter_evaluate.py", "scripts/night7b_finalize.py", "scripts/night7b_package.py",
        "tests/test_night7b_adaptive.py", "tests/test_night7b_firewall.py", "tests/test_night7b_stage_contracts.py",
    ):
        copy_file(REPO / relative, target / relative)
    copy_file(args.bundle, target / "git" / args.bundle.name)
    final_git = {
        "branch":run("git","branch","--show-current"), "head":run("git","rev-parse","HEAD"),
        "final_tag":"night7b-final-20260818",
        "tag_peels_to":run("git","rev-list","-n","1","night7b-final-20260818"),
        "remote_branch":run("git","ls-remote","origin","refs/heads/revision/q2-night7b-adaptive-relational-score-rnd-20260818").split()[0],
        "remote_tag":run("git","ls-remote","origin","refs/tags/night7b-final-20260818").split()[0],
        "force_push_used":False, "force_with_lease_used":False,
    }
    (target / "handoff/git_audit_final.json").write_text(json.dumps(final_git, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    records = []
    for path in sorted(target.rglob("*")):
        if path.is_file() and path.name != "compact_delivery_index.json":
            records.append({"path":path.relative_to(target).as_posix(), "size_bytes":path.stat().st_size,
                            "sha256":sha(path)})
    root = hashlib.sha256("\n".join(x["sha256"] for x in records).encode()).hexdigest()
    index = {"schema_version":1, "files":records,
             "root_rule":"sha256(newline_join(file_sha256_in_lexical_path_order))", "root_sha256":root}
    (target / "compact_delivery_index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    total = sum(path.stat().st_size for path in target.rglob("*") if path.is_file())
    if total >= 25 * 1024 * 1024:
        raise RuntimeError("compact exceeds 25 MiB: %d" % total)
    print(json.dumps({"target":str(target), "files":len(records)+1, "bytes":total,
                      "index_sha256":sha(target / "compact_delivery_index.json"), "root_sha256":root}, sort_keys=True))


if __name__ == "__main__":
    main()
