#!/usr/bin/env python3
"""Minimal dependency-free runner for the targeted Night-16A tests."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import time
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("night16a_targeted_tests", args.test_file)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    rows = []
    started = time.perf_counter()
    for name in sorted(item for item in dir(module) if item.startswith("test_")):
        case_started = time.perf_counter()
        try:
            getattr(module, name)()
            status, detail = "PASS", ""
        except Exception:
            status, detail = "FAILED", traceback.format_exc()
        rows.append({"test": name, "status": status, "detail": detail, "wall_seconds": time.perf_counter() - case_started})
    payload = {
        "status": "PASS" if all(row["status"] == "PASS" for row in rows) else "FAILED",
        "passed": sum(row["status"] == "PASS" for row in rows),
        "failed": sum(row["status"] != "PASS" for row in rows),
        "wall_seconds": time.perf_counter() - started,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("status", "passed", "failed", "wall_seconds")}, indent=2))
    raise SystemExit(0 if payload["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
