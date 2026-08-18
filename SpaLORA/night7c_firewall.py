"""Fail-closed path and role firewall for Night-7C."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


PRELOCK = "PRELOCK"
LOCKED = "LOCKED"
EVALUATOR = "EVALUATOR"


class LabelFirewallError(RuntimeError):
    pass


class LabelFirewall:
    def __init__(self, allowed_files: Iterable[Path], denied_roots: Iterable[Path]):
        self.allowed = {str(Path(x).resolve()) for x in allowed_files}
        self.denied_roots = tuple(str(Path(x).resolve()) for x in denied_roots)
        self.state = PRELOCK
        self.window_count = 0

    def _resolved(self, path: Path) -> str:
        return str(Path(path).resolve())

    def authorize_locked_outputs(self) -> None:
        if self.state != PRELOCK:
            raise LabelFirewallError("total lock can only be declared once")
        self.state = LOCKED

    def open_evaluator_window(self) -> None:
        if self.state != LOCKED or self.window_count != 0:
            raise LabelFirewallError("single evaluator window contract violated")
        self.window_count = 1
        self.state = EVALUATOR

    def require_modeling_read(self, path: Path) -> None:
        target = self._resolved(path)
        if self.state != PRELOCK:
            raise LabelFirewallError("modeling reads are forbidden after total lock")
        if target not in self.allowed:
            raise LabelFirewallError("modeling read is not on the SHA whitelist")
        if target.lower().endswith(".h5ad"):
            raise LabelFirewallError("h5ad is forbidden before total lock")
        if any(target == root or target.startswith(root + os.sep) for root in self.denied_roots):
            raise LabelFirewallError("denied authority path")

    def require_label_read(self, path: Path) -> None:
        if self.state != EVALUATOR or self.window_count != 1:
            raise LabelFirewallError("label read outside the single evaluator window")
        target = self._resolved(path)
        if not any(target == root or target.startswith(root + os.sep)
                   for root in self.denied_roots):
            raise LabelFirewallError("evaluator attempted an unregistered label path")

    def require_transform_allowed(self) -> None:
        if self.state != PRELOCK:
            raise LabelFirewallError("training or transform after total lock")


def reject_low_level_obs_read(path: Path, object_name: str) -> None:
    if str(object_name).strip("/").split("/")[0] == "obs":
        raise LabelFirewallError("low-level obs value read is forbidden")
    if str(Path(path)).lower().endswith(".h5ad"):
        raise LabelFirewallError("original h5ad read is forbidden")
