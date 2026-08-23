#!/usr/bin/env python3
"""Conservative near-authority refinement for lanes without a dual gain."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from SpaLORA.night15e_continuous_reliability_energy import ContinuousEnergyConfig
from SpaLORA.night15f_multiscale_expansion import ExpansionEnergyConfig
from scripts.night15f.night15f_solver_search import (
    authority_row,
    config_id,
    load_expansion_lane,
    run_one,
    selection_key,
    write_rows,
)


def registry(local: ContinuousEnergyConfig, count: int, seed: int):
    rng = np.random.default_rng(seed)
    output = {}
    for index in range(count):
        scale = rng.dirichlet([0.45, 0.8, 0.45])
        candidate_local = replace(
            local,
            trust_scale=float(np.clip(local.trust_scale * np.exp(rng.uniform(np.log(0.5), np.log(60.0))), 0.001, 120.0)),
            trust_center=float(np.clip(local.trust_center + rng.normal(0.0, 0.08), -0.8, 2.5)),
            trust_temperature=float(np.clip(local.trust_temperature * np.exp(rng.normal(0.0, 0.18)), 0.01, 1.5)),
            retained_bias=float(np.clip(local.retained_bias + rng.normal(0.0, 0.18), -3.0, 6.0)),
            view_balance=float(np.clip(local.view_balance + rng.normal(0.0, 0.12), -3.0, 3.0)),
        )
        config = ExpansionEnergyConfig(
            local=candidate_local,
            scale_fine=float(scale[0]),
            scale_registered=float(scale[1]),
            scale_broad=float(scale[2]),
            pairwise_beta=float(np.exp(rng.uniform(np.log(0.4), np.log(160.0)))),
            self_return_strength=float(np.exp(rng.uniform(np.log(1.5), np.log(400.0)))),
            size_prior=float(rng.uniform(0.0, 0.35)),
            expansion_cycles=int(rng.choice([1, 1, 1, 2])),
            capacity_scale=1000000.0,
        )
        output[config_id(config, f"C{index:04d}")] = config
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--banks", type=Path, required=True)
    parser.add_argument("--partitions", type=Path, required=True)
    parser.add_argument("--night15e-registry", type=Path, required=True)
    parser.add_argument("--lanes", required=True)
    parser.add_argument("--count", type=int, default=600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frozen = json.loads(args.night15e_registry.read_text(encoding="utf-8"))
    rows = []
    for lane_index, lane in enumerate(args.lanes.split(",")):
        _, graph, labels, mask, k, initial, evidence = load_expansion_lane(
            args.kit, args.banks, args.partitions, lane
        )
        base = authority_row(lane, graph, labels, mask, k, initial)
        rows.append(base)
        lane_rows = [base]
        partitions = {"NIGHT15E_AUTHORITY": initial}
        local = ContinuousEnergyConfig(**frozen["lanes"][lane]["config"])
        for identifier, config in registry(local, args.count, 20260824 + 7919 * lane_index).items():
            row, partition = run_one(
                lane, graph, labels, mask, k, initial, evidence, identifier, config, "CONSERVATIVE_REFINE"
            )
            rows.append(row)
            lane_rows.append(row)
            if row["status"] == "PASS":
                partitions[identifier] = partition
        best = max(lane_rows, key=selection_key)
        (args.output / "partitions").mkdir(exist_ok=True)
        np.save(args.output / "partitions" / f"{lane}.npy", partitions[str(best["config_id"])], allow_pickle=False)
        write_rows(args.output / "all_run_ledger.partial.csv", rows)
        print(json.dumps({"lane": lane, "ari": best["absolute_ari"], "nmi": best["absolute_nmi"],
                          "delta_ari": best["delta_ari"], "delta_nmi": best["delta_nmi"],
                          "changed": best.get("changed_observations", 0)}), flush=True)
    write_rows(args.output / "all_run_ledger.csv", rows)


if __name__ == "__main__":
    main()
