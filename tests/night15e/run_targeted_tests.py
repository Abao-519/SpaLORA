#!/usr/bin/env python3
"""Dependency-free targeted test runner for Night-15E."""

from __future__ import annotations

import importlib
import json
import sys
import time
import traceback


def main() -> None:
    module = importlib.import_module("tests.night15e.test_continuous_reliability_energy")
    tests = sorted(
        (name, value)
        for name, value in vars(module).items()
        if name.startswith("test_") and callable(value)
    )
    rows = []
    for name, test in tests:
        started = time.perf_counter()
        try:
            test()
            rows.append({"test": name, "status": "PASS", "wall_seconds": time.perf_counter() - started})
        except Exception as error:
            rows.append(
                {
                    "test": name,
                    "status": "FAIL",
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                    "wall_seconds": time.perf_counter() - started,
                }
            )
    result = {
        "status": "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL",
        "passed": sum(row["status"] == "PASS" for row in rows),
        "total": len(rows),
        "rows": rows,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
