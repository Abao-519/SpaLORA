#!/usr/bin/env python3
"""Matched component and solver ablations on the frozen Night-15F profile."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from SpaLORA.night15e_continuous_reliability_energy import ContinuousEnergyConfig
from SpaLORA.night15f_multiscale_expansion import (
    ExpansionEnergyConfig,
    continuous_multiscale_expansion,
    continuous_multiscale_single_site,
)
from scripts.night15e.night15e_continuous_search import evaluate, sha256_array
from scripts.night15f.night15f_solver_search import load_expansion_lane, write_rows


def run_variant(name, solver, config, initial, k, evidence, labels, mask, graph, full_metrics):
    started = time.perf_counter()
    partition, diagnostics = solver(initial, k, evidence, config)
    metrics = evaluate(labels, mask, partition, graph)
    return {
        "variant": name,
        "solver": diagnostics.get("solver", "ALPHA_EXPANSION"),
        "absolute_ari": metrics["absolute_ari"],
        "absolute_nmi": metrics["absolute_nmi"],
        "ami": metrics["ami"],
        "fmi": metrics["fmi"],
        "morans_i": metrics["morans_i"],
        "gearys_c": metrics["gearys_c"],
        "delta_ari_vs_full": metrics["absolute_ari"] - full_metrics["absolute_ari"],
        "delta_nmi_vs_full": metrics["absolute_nmi"] - full_metrics["absolute_nmi"],
        "partition_sha256": sha256_array(partition),
        "cluster_sizes": json.dumps(np.bincount(partition, minlength=k).astype(int).tolist(), separators=(",", ":")),
        "changed_observations": diagnostics["changed_observations"],
        "cycle_energy_ledger_json": diagnostics["cycle_energy_ledger_json"],
        "config_json": json.dumps(asdict(config), sort_keys=True, separators=(",", ":")),
        "wall_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--banks", type=Path, required=True)
    parser.add_argument("--partitions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frozen = json.loads(args.registry.read_text(encoding="utf-8"))
    rows = []
    for lane, registered in frozen["lanes"].items():
        _, graph, labels, mask, k, initial, evidence = load_expansion_lane(
            args.kit, args.banks, args.partitions, lane
        )
        raw = registered["config"]
        if raw is None:
            raise RuntimeError(f"matched ablation requires a non-noop frozen config: {lane}")
        full = ExpansionEnergyConfig(**{**raw, "local": ContinuousEnergyConfig(**raw["local"])})
        expected = {"absolute_ari": registered["absolute_ari"], "absolute_nmi": registered["absolute_nmi"]}
        variants = [
            ("FULL", continuous_multiscale_expansion, full),
            ("SINGLE_SITE_SAME_ENERGY", continuous_multiscale_single_site, full),
            ("REGISTERED_SCALE_ONLY", continuous_multiscale_expansion,
             replace(full, scale_fine=1e-8, scale_registered=1.0, scale_broad=1e-8)),
            ("NO_SELF_RETURN_STAY", continuous_multiscale_expansion,
             replace(full, self_return_strength=0.0)),
            ("NO_SIZE_PRIOR", continuous_multiscale_expansion,
             replace(full, size_prior=0.0)),
            ("PAIRWISE_ZERO_KEEP_STAY", continuous_multiscale_expansion,
             replace(full, pairwise_beta=0.0)),
            ("PURE_DYNAMIC_UNARY", continuous_multiscale_expansion,
             replace(full, pairwise_beta=0.0, self_return_strength=0.0,
                     local=replace(full.local, trust_scale=0.0))),
        ]
        for name, solver, config in variants:
            row = run_variant(name, solver, config, initial, k, evidence, labels, mask, graph, expected)
            row["lane"] = lane
            row["k"] = k
            if name == "FULL":
                if abs(row["absolute_ari"] - expected["absolute_ari"]) > 1e-12:
                    raise RuntimeError(f"full ARI mismatch: {lane}")
                if abs(row["absolute_nmi"] - expected["absolute_nmi"]) > 1e-12:
                    raise RuntimeError(f"full NMI mismatch: {lane}")
            rows.append(row)
        print(json.dumps({"lane": lane, "variants": len(variants)}), flush=True)
    write_rows(args.output, rows)
    print(json.dumps({"status": "PASS", "rows": len(rows)}))


if __name__ == "__main__":
    main()
