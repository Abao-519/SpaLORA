#!/usr/bin/env python3
"""Create the fail-closed Night-7C handoff after the preregistered P1 stop."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from SpaLORA.night7a_consensus import atomic_json, sha256_file  # noqa: E402

OUT = REPO / "outputs/night7c_handoff"
RAW = Path("/root/autodl-fs/night7c_conflict_rnd_20260818")
STATUS = "IMPLEMENTATION_SEMANTICS_INVALID"
ERROR = 0.000027619302272713364
TOLERANCE = 0.0000001


def write_json(name: str, value: dict) -> None:
    atomic_json(OUT / name, value)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def main() -> None:
    p0 = json.loads((OUT / "p0_authority_audit.json").read_text())
    if p0["status"] != "PASS":
        raise RuntimeError("P0 authority was not PASS")
    invalid_source = Path(__file__).resolve().parents[1] / "outputs/night7c_handoff/invalid_attempts"
    if not invalid_source.exists():
        invalid_source.mkdir(parents=True)

    p1 = {
        "schema_version": 1, "status": "FAIL", "terminal_status": STATUS,
        "label_access": False, "formal_training": 0, "formal_transforms": 0,
        "fresh_reload_pass_before_stop": 5, "planned_units": 30,
        "first_failed_unit": "u005",
        "m_initial_recomputed": 0.09747149795293808,
        "m_initial_repeat_after_rng_restore": 0.09747149795293808,
        "m_initial_historical": 0.09749911725521079,
        "m_initial_absolute_error": ERROR,
        "m_initial_tolerance": TOLERANCE,
        "fixed_input_and_mnn_sha_verified": True,
        "checkpoint_and_final_endpoint_parity_verified_before_m_initial_gate": True,
        "hard_stop": "P1 m_initial parity did not satisfy <=1e-7",
    }
    write_json("p1_semantic_contract.json", p1)
    write_json("routing_transform_manifest.json", {
        "schema_version": 1, "status": "NOT_STARTED_DUE_P1_HARD_STOP",
        "planned": 240, "attempted": 0, "successful": 0, "label_access": False,
        "locked_before_evaluation": False, "candidates": ["T%02d" % i for i in range(2, 10)],
    })
    write_json("weighted_mnn_training_manifest.json", {
        "schema_version": 1, "status": "NOT_STARTED_DUE_P1_HARD_STOP",
        "planned": 48, "attempted": 0, "successful": 0, "scientific_retry": 0,
        "label_access": False, "candidates": ["W%02d" % i for i in range(6)],
    })
    write_json("weighted_mnn_transform_manifest.json", {
        "schema_version": 1, "status": "NOT_STARTED_DUE_P1_HARD_STOP",
        "planned": 48, "attempted": 0, "successful": 0, "label_access": False,
        "locked_before_evaluation": False,
    })
    write_json("label_window_audit.json", {
        "schema_version": 1, "status": "NEVER_OPENED", "window_count": 0,
        "label_access": False, "original_h5ad_reads": 0, "low_level_obs_value_reads": 0,
        "reason": "P1 semantic hard stop before P2/T/W",
    })
    write_json("gate_audit.json", {
        "schema_version": 1, "status": "NOT_EVALUATED", "terminal_status": STATUS,
        "routing_gate": "NOT_APPLICABLE", "weighted_mnn_promotion": "NOT_APPLICABLE",
        "reason": "No formal outputs and no label window after P1 hard stop.",
    })
    write_json("budget_audit.json", {
        "schema_version": 1, "terminal_status": STATUS,
        "formal_training": {"used": 0, "limit": 48},
        "formal_routing_transforms": {"used": 0, "limit": 240},
        "formal_weighted_transforms": {"used": 0, "limit": 48},
        "scientific_retry": {"used": 0, "limit": 0},
        "implementation_corrections": {"used": 2, "limit": 8},
        "total_training_attempts": {"used": 0, "limit": 56},
    })
    write_json("runtime_report.json", {
        "schema_version": 1, "status": "P2_NOT_STARTED_DUE_P1_HARD_STOP",
        "current_gpu": p0["environment"]["gpu"],
        "historical_training_gpu": "NVIDIA GeForce RTX 4080",
        "historical_training_cells_positive_gpu_memory": "124/124",
        "current_cuda_smoke": p0["environment"]["cuda_smoke"],
        "parallel_transform_parity_speed_test": "NOT_RUN",
        "gpu_spectral_diagnostic": "NOT_RUN",
        "scientific_interpretation": "No runtime acceleration conclusion was generated.",
    })
    shutil.copyfile(RAW / "full_test.log", OUT / "full_test_log.txt")
    write_json("test_audit.json", {
        "schema_version": 1, "night7c_directed": "9 passed",
        "full_repository": "277 passed, 7 failed, 84 warnings",
        "night7c_touched_function_failures": 0,
        "legacy_failures": 7,
        "legacy_failure_classification": [
            "old result files absent from isolated worktree",
            "old protected manifests compare later scientific source revisions",
            "old Night-3 handoff paths not materialized in this isolated worktree"
        ],
        "full_test_log_sha256": sha256_file(OUT / "full_test_log.txt"),
    })
    failure_rows = [
        {"stage": "P0-AUTHORITY", "unit_id": "", "classification": "implementation_correction",
         "reason": "observation SHA semantic mismatch in auditor", "formal": 0, "label_access": False},
        {"stage": "P1-SEMANTICS", "unit_id": "u005", "classification": "terminal_semantic_failure",
         "reason": "m_initial absolute error %.17g exceeds %.17g" % (ERROR, TOLERANCE),
         "formal": 0, "label_access": False},
    ]
    with (OUT / "failure_audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(failure_rows[0]))
        writer.writeheader(); writer.writerows(failure_rows)

    raw_entries = []
    for path in sorted((RAW / "p1_features").glob("u*/**/*")):
        if path.is_file():
            raw_entries.append({"path": str(path), "size_bytes": path.stat().st_size,
                                "sha256": sha256_file(path)})
    write_json("raw_artifact_index.json", {
        "schema_version": 1, "remote_root": str(RAW), "files": raw_entries,
        "note": "Partial preformal P1 probes are invalid evidence and remain remote only."
    })

    summary = (
        "Night-7C 在 P1 语义一致性门按预注册规则停止。前 5 个 R02 单元 exact；"
        "u005 的初始 MNN 复算与历史首条 loss 相差 2.76193e-05，超过 1e-7 容差。"
        "因此没有启动 P2、240 个路由 transform、48 次 weighted-MNN 训练、48 个 transform，"
        "也没有打开标签。终态为 IMPLEMENTATION_SEMANTICS_INVALID；不能产生方法优劣结论。\n"
    )
    (OUT / "plain_language_summary.txt").write_text(summary)
    report = f"""# SpaLORA Night-7C report

## Outcome

Terminal status: `{STATUS}`.

Night-7C stopped at the preregistered P1 semantic gate. Five R02 units completed
fresh-process checkpoint, endpoint-affinity, partition, and initial-MNN parity.
The next unit, `u005`, produced an initial MNN value of
`0.09747149795293808` twice, including after restoring the checkpoint's exact
pre-forward RNG state. The locked Night-7B loss curve contains
`0.09749911725521079`; absolute error `{ERROR:.17g}` exceeds the registered
`1e-7` tolerance.

## Integrity interpretation

P0 passed all authority inputs, Night-7B compact 67/67 and root SHA, 30 locked
input units with 60 six-view archives, 124/124 historical CUDA training cells,
120 R02/R08 specialist transforms, checkpoint hashes, and the deny-by-default
label firewall. Historical files report `NVIDIA GeForce RTX 4080`; the current
instance reports `NVIDIA GeForce RTX 4080 SUPER`. This hardware difference is a
plausible explanation for the replay deviation, but it is an inference and is
not used to waive the exact tolerance.

## Work not authorized after the hard stop

- P2 serial/parallel runtime experiment: not run.
- Stage T routing: 0/240 transforms.
- Stage W weighted-MNN pilot: 0/48 training and 0/48 transforms.
- Label window and evaluation: never opened.
- Router or weighted-MNN scientific conclusion: none.

## Budgets and failures

Scientific retry was 0. No seed search, fallback, threshold change, H02 repeat,
or label access occurred. Two preformal implementation corrections are retained:
the observation-id SHA audit semantic correction and the saved RNG tensor-device
correction. Partial P1 files are marked invalid and remain under `{RAW}`.

## Tests

Night-7C directed tests: 9 passed. Full repository: 277 passed and 7 legacy
failures caused by old raw/result locks absent or intentionally not materialized
in the isolated worktree; no Night-7C touched-function test failed.

## Scientific limitation

This terminal state is an implementation/replay validity result, not evidence
for or against conflict routing, weighted MNN, P22 performance, or unified
cross-dataset improvement.
"""
    (OUT / "night7c_report.md").write_text(report)
    write_json("git_audit_pre_final.json", {
        "schema_version": 1, "branch": git("branch", "--show-current"),
        "parent_commit": "32d6ed947b313423805ee0f80c9dada06bb6a28d",
        "parent_tag_peel": git("rev-parse", "night7b-final-20260818^{}"),
        "force_push": False, "final_tag_created": False,
    })
    print(json.dumps({"status": STATUS, "output": str(OUT),
                      "formal_training": 0, "formal_transforms": 0,
                      "label_access": False}, sort_keys=True))


if __name__ == "__main__":
    main()
