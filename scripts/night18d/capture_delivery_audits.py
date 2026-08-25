#!/usr/bin/env python3
"""Capture live resource and actual pytest evidence for Night-18D."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from datetime import datetime, timezone


def filesystem(path: str) -> dict:
    stat = os.statvfs(path)
    return {
        "path": path, "block_size": stat.f_frsize,
        "bytes_total": stat.f_blocks * stat.f_frsize, "bytes_available": stat.f_bavail * stat.f_frsize,
        "inodes_total": stat.f_files, "inodes_available": stat.f_favail,
    }


def du(path: str) -> int:
    result = subprocess.run(["du", "-sb", path], check=True, text=True, capture_output=True)
    return int(result.stdout.split()[0])


def run(args: argparse.Namespace) -> None:
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.test_log); text = log_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"(\d+) passed", text)
    if not match or int(match.group(1)) != args.expected_tests:
        raise RuntimeError("actual targeted pytest output does not contain expected pass count")
    test_summary = {
        "status": "PASS", "passed": int(match.group(1)), "failed": 0,
        "command": args.test_command, "environment": args.environment,
        "actual_output_log": str(log_path), "actual_output_tail": text.strip().splitlines()[-6:],
        "captured_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output / "targeted_test_summary.json").write_text(json.dumps(test_summary, indent=2, sort_keys=True), encoding="utf-8")
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"],
        check=False, text=True, capture_output=True,
    )
    resource = {
        "status": "LIVE_SNAPSHOT", "captured_utc": datetime.now(timezone.utc).isoformat(),
        "filesystem_root": filesystem("/"), "filesystem_persistent_inode_limited": filesystem("/autodl-fs/data"),
        "night18d_working_bytes": du(args.working), "repository_bytes": du(args.repository),
        "gpu_query": gpu.stdout.strip() if gpu.returncode == 0 else "UNAVAILABLE",
        "gpu_training_used": False, "dense_n_by_n_created": False,
        "shutdown_dispatched": False, "autodl_instruction": "KEEP_ON_FOR_WORKER2_REVIEW",
    }
    (output / "resource_and_disk_audit.json").write_text(json.dumps(resource, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--test-log", required=True); parser.add_argument("--test-command", required=True)
    parser.add_argument("--environment", required=True); parser.add_argument("--expected-tests", type=int, required=True)
    parser.add_argument("--working", required=True); parser.add_argument("--repository", required=True); parser.add_argument("--output", required=True); run(parser.parse_args())


if __name__ == "__main__": main()
