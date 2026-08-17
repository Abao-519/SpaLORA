"""Fail-closed Night-7A role/phase guard for label and transform operations."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Night7AFirewall:
    phase: str = "prelock"
    evaluator_open_count: int = 0
    development_label_reads: int = 0
    fresh_label_reads: int = 0

    def byte_hash(self, *, fresh: bool = False) -> None:
        # Byte hashing without decoding is permitted in every pre-dispatch phase.
        if self.phase == "closed":
            raise RuntimeError("no remote work is permitted after evaluator closure")

    def lock_transforms_and_preflight(self, transforms: int, preflight_locked: bool) -> None:
        if self.phase != "prelock" or transforms != 360 or not preflight_locked:
            raise RuntimeError("label window requires 360 terminal transforms and preflight lock")
        self.phase = "evaluator"
        self.evaluator_open_count += 1

    def read_development_label(self, role: str) -> None:
        if self.phase != "evaluator" or role != "evaluator" or self.evaluator_open_count != 1:
            raise PermissionError("development labels are evaluator-only after total lock")
        self.development_label_reads += 1

    def read_fresh_label(self, role: str) -> None:
        self.fresh_label_reads += 1
        raise PermissionError("fresh external per-spot labels are forbidden in Night-7A")

    def transform(self, role: str) -> None:
        if self.phase != "prelock" or role != "transformer":
            raise PermissionError("transform operations are prelock transformer-only")

    def close_evaluator(self) -> None:
        if self.phase != "evaluator":
            raise RuntimeError("evaluator is not open")
        self.phase = "closed"

