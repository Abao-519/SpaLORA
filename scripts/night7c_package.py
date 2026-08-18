#!/usr/bin/env python3
"""Build the compact Night-7C planner handoff without raw scientific files."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs/night7c_handoff"
STAGE = Path("/root/autodl-fs/night7c_compact_stage_6444db8/official_compact")
ARCHIVE = Path("/root/autodl-fs/night7c_planner_handoff_20260818.tar.gz")
PARENT = "32d6ed947b313423805ee0f80c9dada06bb6a28d"
BRANCH = "revision/q2-night7c-conflict-gated-mnn-rnd-20260818"


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    if STAGE.parent.exists() or ARCHIVE.exists():
        raise RuntimeError("compact stage or archive already exists")
    STAGE.mkdir(parents=True)
    tracked = json.loads((OUT / "delivery_index.json").read_text())
    paths = [row["path"] for row in tracked["files"]]
    paths.append("outputs/night7c_handoff/delivery_index.json")
    for relative in paths:
        source = REPO / relative
        target = STAGE / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    bundle = STAGE / "git/night7b_to_night7c_20260818.bundle"
    bundle.parent.mkdir(parents=True)
    subprocess.run(["git", "bundle", "create", str(bundle), BRANCH, "^" + PARENT],
                   cwd=REPO, check=True)
    subprocess.run(["git", "bundle", "verify", str(bundle)], cwd=REPO, check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    rows = []
    for path in sorted((x for x in STAGE.rglob("*") if x.is_file()),
                       key=lambda x: x.relative_to(STAGE).as_posix()):
        rows.append({"path": path.relative_to(STAGE).as_posix(),
                     "size_bytes": path.stat().st_size, "sha256": sha(path)})
    root = hashlib.sha256("\n".join(row["sha256"] for row in rows).encode()).hexdigest()
    index = {
        "schema_version": 1, "status": "IMPLEMENTATION_SEMANTICS_INVALID",
        "file_count": len(rows), "files": rows,
        "root_rule": "sha256(newline_join(file_sha256_in_lexical_path_order))",
        "root_sha256": root,
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "branch": BRANCH, "final_tag_created_at_packaging": False,
        "raw_files_included": False,
    }
    (STAGE / "compact_delivery_index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    total = sum(path.stat().st_size for path in STAGE.rglob("*") if path.is_file())
    if total >= 25 * 1024 * 1024:
        raise RuntimeError("compact handoff exceeds 25 MiB")
    with tarfile.open(ARCHIVE, "w:gz") as handle:
        handle.add(STAGE, arcname="official_compact")
    print(json.dumps({"files": len(rows), "root_sha256": root,
                      "stage_bytes": total, "archive": str(ARCHIVE),
                      "archive_sha256": sha(ARCHIVE)}, sort_keys=True))


if __name__ == "__main__":
    main()
