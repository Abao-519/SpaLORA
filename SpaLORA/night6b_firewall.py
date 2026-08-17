"""Fail-closed role and path firewall for Night-6B."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

ORIGINAL_TONSIL_ROOT = "/root/autodl-fs/datasets/human_tonsil_official/section1"
PROTECTED_SUBSTRINGS = (
    "p22",
    "human_lymph_node_d1",
    "gse198353",
    "night4b",
    "night5d",
    "night6a_raw_runs",
)
AUTHORIZED_ROLES = {"data_steward", "trainer_transformer", "evaluator"}


class FirewallViolation(RuntimeError):
    pass


def _norm(path: os.PathLike | str) -> str:
    return os.path.abspath(os.fspath(path)).replace("\\", "/").lower()


def guard_path(
    path: os.PathLike | str,
    *,
    role: str,
    operation: str,
    phase_locked: bool = False,
    audit_log: Optional[os.PathLike | str] = None,
) -> Path:
    if role not in AUTHORIZED_ROLES:
        raise FirewallViolation(f"unknown Night-6B role: {role}")
    normalized = _norm(path)
    allowed = True
    reason = "role/path allowed"
    for token in PROTECTED_SUBSTRINGS:
        if token in normalized:
            allowed = False
            reason = f"protected path token: {token}"
            break
    if normalized.startswith(ORIGINAL_TONSIL_ROOT.lower()):
        if role == "data_steward":
            allowed = True
            reason = "authorized ontology/source-copy access"
        elif role == "evaluator" and phase_locked:
            allowed = True
            reason = "authorized evaluator access after total lock"
        else:
            allowed = False
            reason = "original tonsil is forbidden to trainer/transformer and pre-lock evaluator"
    record = {
        "path": normalized,
        "role": role,
        "operation": operation,
        "phase_locked": bool(phase_locked),
        "allowed": allowed,
        "reason": reason,
    }
    if audit_log is not None:
        log = Path(audit_log)
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    if not allowed:
        raise FirewallViolation(f"{reason}: {path}")
    return Path(path)


def reject_label_payload(**kwargs) -> None:
    forbidden = {"labels", "label_vector", "ari", "nmi", "evaluator_path", "ground_truth"}
    present = sorted(forbidden.intersection(kwargs))
    if present:
        raise FirewallViolation(f"label/evaluator payload rejected: {present}")

