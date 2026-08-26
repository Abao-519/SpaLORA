#!/usr/bin/env python3
"""Freeze Night-18E base-energy and CCSR configuration registries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def canonical_sha(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run(args: argparse.Namespace) -> None:
    parent = json.loads(Path(args.night15f_registry).read_text())
    family = json.loads(Path(args.night18d_family).read_text())
    base = {
        "schema": "night18e-base-energy-registry-v1",
        "source_semantics": {
            "P22_K9": "Night-15F per-lane public-benchmark frozen energy",
            "MISAR_K7": "Night-15F per-lane public-benchmark frozen energy",
            "PLACENTA_K10": "Night-18D RNA-chromatin family geometric-center energy",
            "HUMAN_HIPPOCAMPUS_K7": "Night-18D RNA-chromatin family geometric-center energy",
        },
        "lanes": {
            "P22_K9": parent["lanes"]["P22"]["config"],
            "MISAR_K7": parent["lanes"]["MISAR_E15_5_S1"]["config"],
            "PLACENTA_K10": family["primary_config"],
            "HUMAN_HIPPOCAMPUS_K7": family["primary_config"],
        },
        "labels_read": 0,
    }
    base["lane_config_sha256"] = {
        lane: canonical_sha(value) for lane, value in base["lanes"].items()
    }
    configs = []
    for quantile in (0.50, 0.65, 0.80):
        for support in (2, 3):
            for keep in (False, True):
                value = {
                    "margin_rank_quantile": quantile,
                    "minimum_view_support": support,
                    "keep_untrusted_original_self_return": keep,
                    "epsilon_relative": 1.0e-6,
                }
                value["config_id"] = (
                    f"Q{int(quantile*100):02d}_S{support}_KEEP{int(keep)}_E1E6"
                )
                value["config_sha256"] = canonical_sha(value)
                configs.append(value)
    certificate = {
        "schema": "night18e-certificate-config-registry-v1",
        "status": "FROZEN_BEFORE_PARTITION_PRODUCTION",
        "p0_config_id": "Q65_S2_KEEP1_E1E6",
        "selection_rule": (
            "maximize independent dual-gain discovery-study count; then worst-study "
            "delta ARI, mean delta ARI, mean delta NMI, lower protected fraction"
        ),
        "configs": configs,
        "labels_read": 0,
    }
    certificate["registry_sha256"] = canonical_sha(certificate)
    for output, value in ((args.output_base, base), (args.output_certificate, certificate)):
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--night15f-registry", required=True)
    parser.add_argument("--night18d-family", required=True)
    parser.add_argument("--output-base", required=True)
    parser.add_argument("--output-certificate", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
