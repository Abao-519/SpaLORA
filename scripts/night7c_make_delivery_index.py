#!/usr/bin/env python3
"""Create the tracked non-self-referential Night-7C delivery index."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs/night7c_handoff"


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    explicit = [
        REPO / "SpaLORA/night7c_conflict.py",
        REPO / "SpaLORA/night7c_firewall.py",
        REPO / "scripts/night7c_p0_authority.py",
        REPO / "scripts/night7c_p1.py",
        REPO / "scripts/night7c_fail_closed_finalize.py",
        REPO / "scripts/night7c_make_delivery_index.py",
        REPO / "tests/test_night7c_conflict.py",
        REPO / "tests/test_night7c_firewall.py",
    ]
    files = explicit
    files += sorted((REPO / "protocols/night7c").glob("*"))
    files += sorted(path for path in OUT.rglob("*")
                    if path.is_file() and path.name != "delivery_index.json")
    rows = []
    for path in sorted(set(files), key=lambda x: x.relative_to(REPO).as_posix()):
        relative = path.relative_to(REPO).as_posix()
        rows.append({"path": relative, "size_bytes": path.stat().st_size,
                     "sha256": sha(path)})
    root = hashlib.sha256("\n".join(row["sha256"] for row in rows).encode()).hexdigest()
    index = {
        "schema_version": 1, "status": "IMPLEMENTATION_SEMANTICS_INVALID",
        "root_rule": "sha256(newline_join(file_sha256_in_lexical_path_order))",
        "root_sha256": root, "file_count": len(rows), "files": rows,
        "formal_training": 0, "formal_transforms": 0, "label_access": False,
    }
    (OUT / "delivery_index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"files": len(rows), "root_sha256": root}, sort_keys=True))


if __name__ == "__main__":
    main()
