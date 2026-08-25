#!/usr/bin/env python3
"""Mechanically aggregate P22 and MISAR Night-15F configs before placenta evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


INTEGER_FIELDS = {"expansion_cycles", "sweeps"}


def canonical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def aggregate(left, right, field: str = ""):
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right): raise ValueError(f"config fields differ at {field}")
        return {key: aggregate(left[key], right[key], key) for key in sorted(left)}
    if field in INTEGER_FIELDS:
        return int(math.floor((float(left) + float(right)) / 2.0 + 0.5))
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        a, b = float(left), float(right)
        if a > 0 and b > 0:
            return float(math.exp(0.5 * (math.log(a) + math.log(b))))
        return float(0.5 * (a + b))
    if left != right: raise ValueError(f"non-numeric config mismatch at {field}")
    return left


def assert_numeric_tree(value, field: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            assert_numeric_tree(child, f"{field}.{key}")
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"family config leaf must be numeric and non-bool: {field}={type(value).__name__}")
    if not math.isfinite(float(value)):
        raise ValueError(f"family config leaf is non-finite: {field}")


def run(args: argparse.Namespace) -> None:
    source = Path(args.registry)
    registry = json.loads(source.read_text(encoding="utf-8"))
    p22 = registry["lanes"]["P22"]["config"]
    misar = registry["lanes"]["MISAR_E15_5_S1"]["config"]
    assert_numeric_tree(p22, "P22")
    assert_numeric_tree(misar, "MISAR_E15_5_S1")
    primary = aggregate(p22, misar)
    assert_numeric_tree(primary, "primary")
    output = {
        "status": "FROZEN_BEFORE_PLACENTA_EVALUATION",
        "schema": "night18d-rna-chromatin-family-default-v1",
        "source_registry": str(source),
        "source_registry_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_configs": {"P22": p22, "MISAR_E15_5_S1": misar},
        "source_config_sha256": {"P22": canonical_sha(p22), "MISAR_E15_5_S1": canonical_sha(misar)},
        "source_config_types": "RECURSIVE_FINITE_NUMERIC_NON_BOOL_ASSERTED",
        "aggregation_rule": "recursive component-wise: geometric mean iff both values strictly positive; otherwise arithmetic midpoint; expansion_cycles/sweeps use half-up rounded midpoint",
        "primary_profile_id": "RNA_CHROMATIN_FAMILY_GEOMETRIC_CENTER",
        "primary_config": primary,
        "primary_config_sha256": canonical_sha(primary),
        "sensitivity_profiles": {"P22_FROZEN": p22, "MISAR_FROZEN": misar},
        "placenta_metrics_read": 0,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--registry", required=True); parser.add_argument("--output", required=True); run(parser.parse_args())


if __name__ == "__main__": main()
