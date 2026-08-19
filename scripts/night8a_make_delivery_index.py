#!/usr/bin/env python3
"""Create the non-self-referential tracked Night-8A delivery index."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs/night8a_handoff"


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    target = OUT / "delivery_index.json"
    files = []
    for path in sorted(x for x in OUT.rglob("*") if x.is_file() and x != target):
        files.append({"path": path.relative_to(REPO).as_posix(), "sha256": sha(path),
                      "size_bytes": path.stat().st_size})
    root = hashlib.sha256("\n".join(x["sha256"] for x in sorted(files, key=lambda x: x["path"])).encode()).hexdigest()
    decision = json.loads((OUT / "night8a_decision.json").read_text(encoding="utf-8"))
    payload = {
        "schema_version": 1, "status": decision["status"], "file_count": len(files), "files": files,
        "root_rule": "sha256(newline_join(file_sha256_in_lexical_path_order))", "root_sha256": root,
        "self_referential_fields": False, "raw_files_included": False,
    }
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"file_count": len(files), "root_sha256": root, "index_sha256": sha(target)}, sort_keys=True))


if __name__ == "__main__":
    main()
