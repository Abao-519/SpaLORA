#!/usr/bin/env python3
"""Create a post-training evaluation-source amendment without changing the training lock."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from SpaLORA.night3b_protocol import atomic_json, sha256_file


OUTPUT = REPO / "outputs/night3b_handoff"
CHANGED = ("scripts/night3b_evaluate.py", "scripts/night3b_finalize.py")


def main() -> None:
    training_lock_path = OUTPUT / "config_lock.json"
    training_lock = json.loads(training_lock_path.read_text(encoding="utf-8"))
    locked_path = OUTPUT / "locked_120_run_manifest.json"
    locked = json.loads(locked_path.read_text(encoding="utf-8"))
    training_lock_sha = sha256_file(training_lock_path)
    if locked.get("run_count") != 120 or locked.get("failures") != 0:
        raise RuntimeError("Training is not locked 120/120")
    if locked.get("config_lock_sha256") != training_lock_sha:
        raise RuntimeError("Locked training manifest does not reference the preserved training lock")
    if not locked.get("locked_before_any_semantic_label_access"):
        raise RuntimeError("Training manifest was not label-isolated")
    for row in locked["runs"]:
        manifest = json.loads((REPO / row["run_manifest"]).read_text(encoding="utf-8"))
        if manifest.get("config_lock_sha256") != training_lock_sha or manifest.get("semantic_label_access") is not False:
            raise RuntimeError("Run manifest training-lock mismatch")

    existing_evaluation_lock_path = OUTPUT / "evaluation_config_lock.json"
    previous_evaluation_lock_sha256 = (
        sha256_file(existing_evaluation_lock_path) if existing_evaluation_lock_path.is_file() else None
    )
    evaluation_lock = copy.deepcopy(training_lock)
    changes = []
    for name in CHANGED:
        old = training_lock["source_sha256"][name]
        new = sha256_file(REPO / name)
        if old == new:
            raise RuntimeError("Expected evaluation implementation change is absent: %s" % name)
        evaluation_lock["source_sha256"][name] = new
        changes.append({"path": name, "training_sha256": old, "evaluation_sha256": new})
    preparer = "scripts/night3b_prepare_evaluation_lock.py"
    evaluation_lock["source_sha256"][preparer] = sha256_file(REPO / preparer)
    unchanged_failures = []
    for name, expected in training_lock["source_sha256"].items():
        if name not in CHANGED and sha256_file(REPO / name) != expected:
            unchanged_failures.append(name)
    if unchanged_failures:
        raise RuntimeError("Unexpected post-training source changes: %r" % unchanged_failures)
    evaluation_lock["training_config_lock_sha256"] = training_lock_sha
    evaluation_lock["evaluation_amendment"] = {
        "schema_version": 1,
        "created_after_locked_120_run_manifest": True,
        "locked_120_run_manifest_sha256": sha256_file(locked_path),
        "semantic_label_values_read_before_amendment": previous_evaluation_lock_sha256 is not None,
        "semantic_label_values_read_during_training": False,
        "reason": "complete taskbook-required domain boxplots and add missing evaluator torch import; update finalizer to verify this phase lock",
        "training_code_or_artifacts_modified": False,
        "scientific_formula_threshold_seed_or_variant_modified": False,
        "changed_evaluation_files": changes,
        "added_lock_preparer": preparer,
        "previous_evaluation_lock_sha256": previous_evaluation_lock_sha256,
    }
    atomic_json(OUTPUT / "evaluation_config_lock.json", evaluation_lock)

    deviations_path = OUTPUT / "protocol_deviations.json"
    deviations = json.loads(deviations_path.read_text(encoding="utf-8"))
    if previous_evaluation_lock_sha256 is None:
        deviations["implementation_corrections"].append({
            "stage": "post-training pre-evaluation",
            "issue": "P0-locked evaluator lacked a torch import and used an effect-size heatmap instead of the taskbook-required per-domain five-seed boxplots",
            "resolution": "preserved the training config lock and all 120 run manifests; created a separate evaluation source amendment before semantic label access",
            "scientific_impact": "none; no training artifact, formula, threshold, seed, variant, or label-free decision changed",
        })
    elif not any(row.get("stage") == "evaluation attempt 1 plotting"
                 for row in deviations["implementation_corrections"]):
        deviations["implementation_corrections"].append({
            "stage": "evaluation attempt 1 plotting",
            "issue": "server Matplotlib rejected the newer set_ticks(ticks, labels) API after all numerical metrics were written",
            "resolution": "preserved evaluation attempt 1 and replaced only tick-label calls with the installed-version-compatible two-call API",
            "scientific_impact": "none; no training artifact, metric formula, result, threshold, seed, or variant changed",
        })
    if previous_evaluation_lock_sha256 is not None and not any(
        row.get("stage") == "post-evaluation replay role audit"
        for row in deviations["implementation_corrections"]
    ):
        deviations["implementation_corrections"].append({
            "stage": "post-evaluation replay role audit",
            "issue": "finalizer had elevated exact final FULL_IGE replay to a hard gate not specified by the taskbook; the actual P0-ARCH gate had already passed",
            "resolution": "retained every raw replay difference, removed the extra hard-gate condition, and explicitly classified final replay as diagnostic without inventing a tolerance",
            "scientific_impact": "none; recommendation, metrics, training artifacts, thresholds, seeds, and variants are unchanged",
        })
    atomic_json(deviations_path, deviations)
    print(json.dumps({
        "status": "PASS", "training_config_lock_sha256": training_lock_sha,
        "evaluation_config_lock_sha256": sha256_file(OUTPUT / "evaluation_config_lock.json"),
        "changed_files": len(changes),
        "semantic_label_values_read_before_amendment": previous_evaluation_lock_sha256 is not None,
        "semantic_label_values_read_during_training": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
