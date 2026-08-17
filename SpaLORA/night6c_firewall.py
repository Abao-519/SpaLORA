"""Fail-closed role, path, and payload firewall for Night-6C."""
from __future__ import annotations
import json
import os
from pathlib import Path


ORIGINAL_TONSIL_ROOT = "/root/autodl-fs/datasets/human_tonsil_official/section1"
PROTECTED_TOKENS = (
    "/p22/", "p22_mouse", "/d1/", "human lymph node d1", "human_lymph_node_d1",
    "gse198353", "night4b", "night5d", "night6a_raw", "night6a_metric",
)
ROLES = {"data_steward", "trainer_transformer", "evaluator"}


class FirewallViolation(RuntimeError):
    pass


def _norm(path) -> str:
    return os.path.abspath(os.fspath(path)).replace("\\", "/").lower()


def guard_path(path, *, role: str, operation: str, phase_locked: bool = False,
               audit_log=None) -> Path:
    if role not in ROLES:
        raise FirewallViolation(f"unknown role: {role}")
    normalized = _norm(path)
    allowed = not any(token in normalized for token in PROTECTED_TOKENS)
    reason = "role/path allowed" if allowed else "protected dataset/result path"
    original = normalized.startswith(ORIGINAL_TONSIL_ROOT.lower())
    if original:
        if role == "evaluator" and phase_locked:
            allowed, reason = True, "post-lock evaluator access"
        else:
            allowed, reason = False, "original tonsil forbidden outside post-lock evaluator"
    record = {"path": normalized, "role": role, "operation": operation,
              "phase_locked": bool(phase_locked), "allowed": allowed,
              "reason": reason}
    if audit_log:
        target = Path(audit_log); target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
    if not allowed:
        raise FirewallViolation(f"{reason}: {path}")
    return Path(path)


def reject_transform_payload(**kwargs) -> None:
    forbidden = {"labels", "label_vector", "ground_truth", "ari", "nmi",
                 "evaluator_path", "single_seed_metric", "intermediate_epoch_metric"}
    present = sorted(forbidden.intersection(kwargs))
    if present:
        raise FirewallViolation(f"label/result payload rejected: {present}")
