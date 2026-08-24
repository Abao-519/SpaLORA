#!/usr/bin/env python3
"""Build the pre-commit Night-16H handoff content manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/night16h_handoff"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    roots = [
        ROOT / "SpaLORA/night16h_feasible_selector.py",
        ROOT / "configs/night16h/night16h_feasible_selector_contract.json",
        ROOT / "tests/test_night16h_feasible_selector.py",
    ]
    roots.extend(sorted((ROOT / "scripts/night16h").glob("*.py")))
    roots.extend(sorted(path for path in OUTPUT.glob("*") if path.name != "handoff_manifest.json"))
    rows = [
        {
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "size": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in roots
    ]
    result = {
        "schema": "night16h-handoff-content-manifest-v1",
        "status": "NIGHT16H_UNIVERSAL_FEASIBILITY_CROSS_EVIDENCE_SELECTOR_SIGNAL",
        "classification": "LOCAL_SIGNAL",
        "parent_commit": "be35ca5ff6dcaa189cfa4f59f5c9bd97dcf02f31",
        "parent_tag": "night16g-final-20260824",
        "branch": "revision/q2-night16h-feasible-multiview-selector-20260824",
        "intended_final_tag": "night16h-final-20260824",
        "formal_lane_count": 4,
        "candidate_count": 356,
        "fresh_process_replay": "8/8",
        "targeted_tests": "11/11",
        "files": rows,
        "shutdown_dispatched": False,
        "machine_state_instruction": "KEEP_ON_FOR_NIGHTTIME_CHAINED_TASKS",
    }
    (OUTPUT / "handoff_manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "files": len(rows)}))


if __name__ == "__main__":
    main()
