#!/usr/bin/env python3
"""Dependency-free runner for the targeted Night-16B test module."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import time
import traceback


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("night16b_targeted_tests", args.test_file)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    rows = []
    for name in sorted(item for item in dir(module) if item.startswith("test_")):
        started = time.perf_counter()
        try:
            getattr(module, name)()
            status, failure = "PASS", ""
        except Exception:
            status, failure = "FAILED", traceback.format_exc()
        rows.append(
            {
                "test": name,
                "status": status,
                "failure": failure,
                "wall_seconds": time.perf_counter() - started,
            }
        )
    payload = {
        "status": "PASS" if all(row["status"] == "PASS" for row in rows) else "FAILED",
        "passed": sum(row["status"] == "PASS" for row in rows),
        "failed": sum(row["status"] != "PASS" for row in rows),
        "tests": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
