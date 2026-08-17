"""Fail-closed path, phase, and payload firewall for Night-6D."""
from __future__ import annotations

import json
import os
from pathlib import Path


ROLES = {"data_steward", "trainer_transformer", "evaluator"}
ORIGINAL_D1 = "/root/autodl-fs/human lymph node/d1/"
ORIGINAL_P22 = "/root/autodl-fs/p22 mouse brain coronal section/"
GROUND_TRUTH_NAMES = {"d1_groundtruth.csv", "mousebrain_groundtruth.csv"}
TRAINER_ALLOWED_ROOTS = (
    "/root/autodl-fs/night6d_data_20260817/",
    "/root/autodl-fs/night6d_cache_20260817/",
    "/root/autodl-fs/night6d_raw_runs_20260817/",
    "/root/autodl-fs/night3af_p0d_builds_20260810/process_a/p22/",
    "/root/autodl-fs/spalora-night6d/",
)
ALWAYS_PROTECTED = (
    "gse198353", "night4b", "a1_groundtruth", "final_annot",
    "night6a_raw", "night6a_metric", "per_seed_metrics",
)


class FirewallViolation(RuntimeError):
    pass


def _norm(path) -> str:
    return os.path.abspath(os.fspath(path)).replace("\\", "/").lower()


def guard_path(path, *, role: str, operation: str, phase_locked: bool = False,
               audit_log=None) -> Path:
    if role not in ROLES:
        raise FirewallViolation(f"unknown role: {role}")
    value = _norm(path)
    basename = Path(value).name
    allowed = False
    reason = "role/path denied"
    if any(token in value for token in ALWAYS_PROTECTED):
        allowed, reason = False, "protected dataset/result path"
    elif role == "trainer_transformer":
        allowed = any(value.startswith(root) for root in TRAINER_ALLOWED_ROOTS)
        reason = "label-free/cache/raw path" if allowed else "trainer path outside allowlist"
        if value.startswith(ORIGINAL_D1) or value.startswith(ORIGINAL_P22) or basename in GROUND_TRUTH_NAMES:
            allowed, reason = False, "original data or ground truth forbidden to trainer/transformer"
    elif role == "data_steward":
        if basename in GROUND_TRUTH_NAMES:
            allowed = operation == "byte_hash_only"
            reason = "ground truth byte hash only" if allowed else "ground truth parse forbidden pre-lock"
        elif value.startswith(ORIGINAL_D1):
            allowed = operation in {"byte_hash_only", "low_level_hdf5_copy"}
            reason = "authorized low-level D1 copy" if allowed else "D1 source operation forbidden"
        elif value.startswith(ORIGINAL_P22):
            allowed = operation == "byte_hash_only"
            reason = "P22 source byte hash only" if allowed else "P22 originals forbidden"
        else:
            allowed = any(value.startswith(root) for root in TRAINER_ALLOWED_ROOTS)
            reason = "authorized cache/output stewardship" if allowed else "data-steward path outside allowlist"
    elif role == "evaluator":
        allowed = bool(phase_locked and basename in GROUND_TRUTH_NAMES and operation == "parse_ground_truth")
        reason = "single post-lock evaluation window" if allowed else "evaluator access outside locked label window"
    record = {
        "path": value,
        "role": role,
        "operation": operation,
        "phase_locked": bool(phase_locked),
        "allowed": bool(allowed),
        "reason": reason,
    }
    if audit_log:
        target = Path(audit_log)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    if not allowed:
        raise FirewallViolation(f"{reason}: {path}")
    return Path(path)


def reject_transform_payload(**kwargs) -> None:
    forbidden = {
        "labels", "label_vector", "ground_truth", "ari", "nmi", "q",
        "evaluator_path", "single_seed_metric", "intermediate_epoch_metric",
        "checkpoint_metric", "cluster_annotation_alignment",
    }
    present = sorted(forbidden.intersection(kwargs))
    if present:
        raise FirewallViolation(f"label/result payload rejected: {present}")

