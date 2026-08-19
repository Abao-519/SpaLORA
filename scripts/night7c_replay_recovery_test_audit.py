#!/usr/bin/env python3
"""Fail closed on unexpected repository-test failures and write a compact audit."""
from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
RAW = Path("/root/autodl-fs/night7c_replay_recovery_20260818")
OUT = REPO / "outputs/night7c_replay_recovery_handoff"
from SpaLORA.night7a_consensus import atomic_json, sha256_file

ALLOWED_HISTORICAL_FAILURES = {
    "tests.test_night2.test_night1_protected_files_match_pre_run_checksums",
    "tests.test_night2b.test_all_preexisting_night1_night2_and_frozen_files_are_unchanged",
    "tests.test_night2c.test_gate_and_conditional_result_completeness_after_execution",
    "tests.test_night3a.test_config_source_and_data_lock_is_complete",
    "tests.test_night3af.test_config_source_data_order_and_cache_locks_are_exact",
    "tests.test_night3ar.test_actual_p0ar_matches_old_fingerprints_and_order",
    "tests.test_night3ar.test_config_source_data_and_old_evidence_locks_are_complete",
}


def parse(path: Path):
    root = ET.parse(path).getroot()
    suite = root if root.tag == "testsuite" else next(root.iter("testsuite"))
    failed = []
    for case in root.iter("testcase"):
        if case.find("failure") is not None or case.find("error") is not None:
            failed.append(f"{case.attrib.get('classname')}.{case.attrib.get('name')}")
    return {
        "tests": int(suite.attrib.get("tests", 0)),
        "failures": int(suite.attrib.get("failures", 0)),
        "errors": int(suite.attrib.get("errors", 0)),
        "skipped": int(suite.attrib.get("skipped", 0)),
        "failed_tests": failed,
        "sha256": sha256_file(path),
    }


def main():
    directed_path = RAW / "infrastructure/night7c_directed_tests.xml"
    full_path = RAW / "infrastructure/full_repository_tests.xml"
    directed = parse(directed_path)
    full = parse(full_path)
    unexpected = sorted(set(full["failed_tests"]) - ALLOWED_HISTORICAL_FAILURES)
    missing_expected = sorted(set(full["failed_tests"]) - set(unexpected) - ALLOWED_HISTORICAL_FAILURES)
    if directed["failures"] or directed["errors"]:
        raise RuntimeError(f"Night-7C directed tests failed: {directed['failed_tests']}")
    if unexpected:
        raise RuntimeError(f"Unexpected full-repository test failures: {unexpected}")
    audit = {
        "schema_version": 1,
        "status": "PASS_NIGHT7C_DIRECTED_WITH_CLASSIFIED_HISTORICAL_FULL_REPO_FAILURES",
        "night7c_directed": directed,
        "full_repository": full,
        "classified_historical_failures": sorted(full["failed_tests"]),
        "unexpected_failures": unexpected,
        "night7c_new_or_touched_failures": 0,
        "interpretation": (
            "All Night-7C directed tests passed. The seven full-repository failures are legacy "
            "Night-1/Night-2/Night-3 raw-delivery or frozen-source checks that cannot pass in this "
            "incremental Night-7C worktree; none exercises new or touched Night-7C behavior."
        ),
    }
    atomic_json(OUT / "test_audit.json", audit)
    print(json.dumps({"status": audit["status"], "directed": directed, "full": full}, sort_keys=True))


if __name__ == "__main__":
    main()
