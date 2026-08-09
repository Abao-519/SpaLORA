"""Night-3A-R protocol boundaries for integrity and scientific label access."""

from __future__ import annotations

import builtins
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))
    directory_fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def canonical_path(value: object) -> str:
    return str(Path(str(value)).resolve())


def ground_truth_csv_paths(config: Mapping[str, object]) -> List[str]:
    return sorted(
        canonical_path(cfg["ground_truth"])
        for cfg in config["datasets"].values()
        if str(cfg["ground_truth"]).startswith("/")
    )


def integrity_read(path: Path, expected: Optional[str], purpose: str) -> dict:
    actual = sha256_file(path)
    return {
        "path": canonical_path(path),
        "expected_sha256": expected,
        "actual_sha256": actual,
        "size_bytes": int(path.stat().st_size),
        "purpose": purpose,
        "access_class": "integrity_byte_read",
        "content_returned": False,
        "semantic_parse": False,
        "match": expected is None or actual == expected,
    }


def record_integrity_reads(output: Path, stage: str, reads: Sequence[dict]) -> None:
    path = output / "integrity_read_manifest.json"
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = {
            "schema_version": 1,
            "integrity_reads_are_allowed_before_scientific_window": True,
            "semantic_content_returned": False,
            "stages": {},
        }
    payload["stages"][stage] = list(reads)
    payload["all_reads_match"] = all(
        row["match"] for stage_rows in payload["stages"].values() for row in stage_rows
    )
    payload["total_read_records"] = sum(len(rows) for rows in payload["stages"].values())
    atomic_json(path, payload)


def verify_lock(repo: Path, config_path: Path, config: dict, lock: dict, output: Path,
                stage: str) -> List[dict]:
    """Verify immutable bytes before the scientific window; never return content."""
    reads: List[dict] = []
    reads.append(integrity_read(config_path, lock["config_sha256"], "locked configuration"))
    for name, expected in sorted(lock["source_sha256"].items()):
        reads.append(integrity_read(repo / name, expected, "locked source"))
    for name, expected in sorted(lock["data_sha256"].items()):
        purpose = "ground-truth integrity only" if canonical_path(name) in ground_truth_csv_paths(config) else "training input integrity"
        reads.append(integrity_read(Path(name), expected, purpose))
    order = repo / config["run_order"]["manifest"]
    reads.append(integrity_read(order, lock["run_order_sha256"], "preregistered run order"))
    manifest = output / "data_manifest.csv"
    reads.append(integrity_read(manifest, lock["data_manifest_sha256"], "P0A-R data manifest"))
    if not all(row["match"] for row in reads):
        record_integrity_reads(output, stage, reads)
        raise RuntimeError("Night-3A-R byte-integrity lock mismatch: %r" % [row for row in reads if not row["match"]])
    record_integrity_reads(output, stage, reads)
    return reads


def training_cfg(cfg: Mapping[str, object]) -> dict:
    """Return the only dataset fields allowed to enter the trainer."""
    keys = (
        "n_clusters", "hvg", "spatial_neighbors", "legacy_datatype",
        "embedding_dim", "epochs", "loss_factors", "locked_m_bad_expected",
    )
    result = {key: cfg[key] for key in keys}
    assert not any("ground" in key.lower() or "label" in key.lower() for key in result)
    return result


def assert_training_payload_label_free(data: Mapping[str, object], cfg: Mapping[str, object],
                                       forbidden_paths: Iterable[str]) -> None:
    forbidden = set(map(str, forbidden_paths))
    for key in data:
        lowered = str(key).lower()
        if "ground_truth" in lowered or "label" in lowered or "cell_type" in lowered:
            raise AssertionError("Label-like key entered training payload: %s" % key)
    for key, value in cfg.items():
        lowered = str(key).lower()
        if "ground" in lowered or "label" in lowered or "cell_type" in lowered:
            raise AssertionError("Label-like config key entered trainer: %s" % key)
        if isinstance(value, (str, Path)) and canonical_path(value) in forbidden:
            raise AssertionError("Ground-truth path entered trainer")


class ScientificWindow:
    """Audit and enforce semantic label isolation after byte verification."""

    def __init__(self, config: dict, output: Path, stage: str):
        self.config = config
        self.output = output
        self.stage = stage
        self.forbidden_paths = set(ground_truth_csv_paths(config))
        self.forbidden_modules = tuple(config["label_firewall"]["forbidden_modules_before_manifest_lock"])
        self.opened_ground_truth_paths: List[str] = []
        self.forbidden_imports: List[str] = []
        self.parser_guard_triggers = 0
        self.opened_path_count = 0
        self._original_read_csv = pd.read_csv
        self._installed = False

    def _audit_hook(self, event, args):
        if event == "open" and args:
            try:
                resolved = canonical_path(args[0])
            except Exception:
                return
            self.opened_path_count += 1
            if resolved in self.forbidden_paths:
                self.opened_ground_truth_paths.append(resolved)
                raise RuntimeError("Ground-truth CSV opened inside scientific window: %s" % resolved)
        if event == "import" and args:
            name = str(args[0])
            if any(name == forbidden or name.startswith(forbidden + ".") for forbidden in self.forbidden_modules):
                self.forbidden_imports.append(name)
                raise RuntimeError("Evaluator import inside scientific window: %s" % name)

    def _guarded_read_csv(self, filepath_or_buffer, *args, **kwargs):
        if isinstance(filepath_or_buffer, (str, os.PathLike)):
            try:
                resolved = canonical_path(filepath_or_buffer)
            except Exception:
                resolved = ""
            if resolved in self.forbidden_paths:
                self.parser_guard_triggers += 1
                raise RuntimeError("Ground-truth CSV parser blocked inside scientific window: %s" % resolved)
        return self._original_read_csv(filepath_or_buffer, *args, **kwargs)

    def install(self) -> "ScientificWindow":
        if self._installed:
            raise RuntimeError("Scientific window already installed")
        already = [name for name in self.forbidden_modules if name in sys.modules]
        if already:
            raise RuntimeError("Forbidden evaluator module imported before scientific window: %r" % already)
        sys.addaudithook(self._audit_hook)
        pd.read_csv = self._guarded_read_csv
        self._installed = True
        return self

    def payload(self, passed: bool = True) -> dict:
        return {
            "stage": self.stage,
            "integrity_verification_completed_before_window": True,
            "ground_truth_csv_opened_inside_window": sorted(set(self.opened_ground_truth_paths)),
            "forbidden_evaluator_imports": sorted(set(self.forbidden_imports)),
            "ground_truth_parser_guard_trigger_count": int(self.parser_guard_triggers),
            "semantic_label_values_read": False,
            "opened_path_count": int(self.opened_path_count),
            "passed": bool(
                passed and not self.opened_ground_truth_paths
                and not self.forbidden_imports and self.parser_guard_triggers == 0
            ),
        }

    def close(self, passed: bool = True) -> dict:
        if self._installed:
            pd.read_csv = self._original_read_csv
        stage_payload = self.payload(passed)
        path = self.output / "scientific_window_label_firewall.json"
        if path.is_file():
            aggregate = json.loads(path.read_text(encoding="utf-8"))
        else:
            aggregate = {
                "schema_version": 1,
                "integrity_reads_classified_outside_scientific_window": True,
                "stages": {},
            }
        aggregate["stages"][self.stage] = stage_payload
        aggregate["semantic_label_values_read"] = False
        aggregate["passed"] = all(row["passed"] for row in aggregate["stages"].values())
        atomic_json(path, aggregate)
        return stage_payload

