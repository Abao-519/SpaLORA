#!/usr/bin/env python3
"""Label-closed Stage-A producer for direct feasible-posterior graph cuts."""

from __future__ import annotations

import argparse
import csv
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np

from SpaLORA.night15e_continuous_reliability_energy import ContinuousEnergyConfig
from SpaLORA.night15f_multiscale_expansion import ExpansionEnergyConfig, prepare_expansion_evidence
from SpaLORA.night16e_tsre import partition_sha256
from SpaLORA.night17e_lrcc import LRCCConfig, relation_conditioned_pairwise
from SpaLORA.night17f_direct_relation_cut import (
    ANALYTIC_ARM,
    DISABLED_ARM,
    PERMUTED_ARM,
    PRIMARY_ARM,
    UNIFORM_ARM,
    candidate_mask,
    direct_relation_cut,
    prepare_direct_relation_evidence,
)
from scripts.night17e.night17e_producer import load_carrier, sha256_array, sha256_file


ARM_MAP = {
    PRIMARY_ARM: "LEARNED_RELATION",
    ANALYTIC_ARM: "ZERO_RELATION",
    PERMUTED_ARM: "PERMUTED_RELATION",
    UNIFORM_ARM: "UNIFORM_MASS_MATCHED",
    DISABLED_ARM: "RELATION_DISABLED",
}


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def load_config(value: dict[str, object]) -> LRCCConfig:
    base_value = dict(value["base"])
    local = ContinuousEnergyConfig(**dict(base_value.pop("local")))
    base = ExpansionEnergyConfig(local=local, **base_value)
    extra = {field.name: value[field.name] for field in fields(LRCCConfig) if field.name != "base"}
    return LRCCConfig(base=base, **extra)


def load_candidate_bank(path: Path, ids: np.ndarray, evidence_csv: Path, lane: str):
    with np.load(path, allow_pickle=False) as archive:
        bank_ids = np.asarray(archive["ids"]).astype("U")
        candidate_ids = np.asarray(archive["candidate_ids"]).astype("U")
        partitions = np.asarray(archive["partitions"], dtype=np.int32)
    if not np.array_equal(bank_ids, ids):
        raise ValueError("candidate-bank/carrier ordered ID mismatch")
    if partitions.shape != (len(candidate_ids), len(ids)) or len(np.unique(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate bank shape/ID contract invalid")
    if len(candidate_ids) != 89:
        raise ValueError("Night-16H authority must contain exactly 89 candidates")
    with evidence_csv.open(newline="", encoding="utf-8-sig") as handle:
        raw = [row for row in csv.DictReader(handle) if row["lane"] == lane]
    by_id = {row["candidate_id"]: row for row in raw}
    if len(by_id) != len(raw) or set(by_id) != set(candidate_ids.tolist()):
        raise ValueError("sanitized candidate evidence/candidate-bank ID set mismatch")
    records = [by_id[candidate_id] for candidate_id in candidate_ids]
    if [row["candidate_id"] for row in records] != candidate_ids.tolist():
        raise RuntimeError("sanitized records were not aligned in candidate-bank order")
    forbidden = set(records[0]) - {
        "lane", "candidate_id", "molecular_joint", "topology_joint", "persistence",
        "feasible_SMALLEST_SCALE_INTERNAL_EDGE",
    }
    if forbidden:
        raise ValueError(f"producer evidence contains forbidden columns: {sorted(forbidden)}")
    selected = candidate_mask(records, "UNBIASED_BANK")
    if int(np.sum(selected)) < 2:
        raise ValueError("Night-16H UNBIASED_BANK has fewer than two feasible candidates")
    return candidate_ids, partitions, records


def run(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    registry_path = Path(args.registry)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry["lane"] != args.lane or registry["bank_mode"] != "UNBIASED_BANK":
        raise ValueError("registry lane/bank-mode mismatch")
    carrier_path = Path(args.carrier)
    bank_path = Path(args.candidate_bank)
    evidence_csv = Path(args.candidate_evidence)
    carrier = load_carrier(carrier_path)
    candidate_ids, candidate_partitions, records = load_candidate_bank(
        bank_path, carrier["ids"], evidence_csv, args.lane
    )
    match = np.flatnonzero(candidate_ids == registry["start"]["source_id"])
    if len(match) != 1:
        raise ValueError("locked Night-16H method start missing")
    initial = candidate_partitions[int(match[0])].astype(np.int32)
    if np.unique(initial).size != int(args.k):
        raise ValueError("locked method start K mismatch")
    initial_sha = partition_sha256(initial)
    evidence = prepare_expansion_evidence(
        carrier["graphs"], carrier["retained"], carrier["view1"], carrier["view2"]
    )

    partitions: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    relation_manifest: dict[str, object] = {}

    def append(partition: np.ndarray, row: dict[str, object]) -> None:
        partition = np.asarray(partition, dtype=np.int32)
        sizes = np.bincount(partition, minlength=int(args.k))
        if np.unique(partition).size != int(args.k) or np.any(sizes <= 0):
            raise RuntimeError("partition violates exact K/no empty")
        row.update(
            partition_index=len(partitions),
            partition_sha256=partition_sha256(partition),
            cluster_sizes=sizes.astype(int).tolist(),
            min_cluster_size=int(np.min(sizes)),
            status="PASS",
            failure="",
        )
        partitions.append(partition)
        rows.append(row)

    append(
        initial,
        {
            "run_id": "METHOD_NIGHT16H_START__INPUT_START",
            "start_id": registry["start"]["start_id"],
            "start_role": registry["start"]["role"],
            "arm": "INPUT_START",
            "config_id": "INPUT",
            "initial_partition_sha256": initial_sha,
            "changed_from_initial": 0,
            "wall_seconds": 0.0,
        },
    )
    for candidate in registry["candidates"]:
        config = load_config(candidate["config"])
        relation, posterior_diagnostics = prepare_direct_relation_evidence(
            evidence,
            candidate_partitions,
            records,
            config,
            bank_mode=registry["bank_mode"],
        )
        preflight = {}
        capacities = {}
        for arm in registry["arms"]:
            _, _, capacity, _, diagnostic = relation_conditioned_pairwise(
                evidence, relation, config, ARM_MAP[arm], carrier["ids"]
            )
            capacities[arm] = capacity
            preflight[arm] = {
                "scale_mass_absolute_errors": [
                    float(item["mass_match_absolute_error"])
                    for item in diagnostic["scale_diagnostics"]
                ],
                "scale_mass_relative_errors": [
                    float(item["mass_match_relative_error"])
                    for item in diagnostic["scale_diagnostics"]
                ],
                "mean_pair_weight": float(diagnostic["mean_pair_weight"]),
            }
        primary_delta = float(np.max(np.abs(capacities[PRIMARY_ARM] - capacities[DISABLED_ARM])))
        if not np.isfinite(primary_delta) or primary_delta <= 0:
            raise RuntimeError("direct posterior did not alter pairwise capacity")
        relation_manifest[candidate["config_id"]] = {
            "posterior_diagnostics": posterior_diagnostics,
            "capacity_preflight": preflight,
            "direct_vs_disabled_max_abs_capacity_delta": primary_delta,
        }
        for arm in registry["arms"]:
            item_started = time.perf_counter()
            row = {
                "run_id": f"METHOD_NIGHT16H_START__{candidate['config_id']}__{arm}",
                "start_id": registry["start"]["start_id"],
                "start_role": registry["start"]["role"],
                "arm": arm,
                "config_id": candidate["config_id"],
                "config_sha256": hashlib.sha256(canonical_json(candidate["config"]).encode()).hexdigest(),
                "initial_partition_sha256": initial_sha,
                "status": "FAILED",
                "failure": "",
            }
            try:
                partition, diagnostics = direct_relation_cut(
                    initial, args.k, carrier["ids"], evidence, relation, config, arm
                )
                row.update(
                    changed_from_initial=int(np.sum(partition != initial)),
                    diagnostics=diagnostics,
                    wall_seconds=float(time.perf_counter() - item_started),
                )
                append(partition, row)
            except Exception as exc:
                row["failure"] = f"{type(exc).__name__}: {exc}"
                row["wall_seconds"] = float(time.perf_counter() - item_started)
                rows.append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        ids=np.asarray(carrier["ids"]),
        run_ids=np.asarray([row["run_id"] for row in rows if row["status"] == "PASS"]).astype("U"),
        partitions=np.stack(partitions).astype(np.int32),
    )
    temporary.replace(output)
    with np.load(output, allow_pickle=False) as replay:
        if not np.array_equal(replay["ids"].astype("U"), carrier["ids"]):
            raise RuntimeError("saved ordered IDs failed reload")
        if not np.array_equal(replay["partitions"], np.stack(partitions)):
            raise RuntimeError("saved partitions failed reload")
    manifest = {
        "schema": "night17f-stage-a-producer-v1",
        "scientific_role": registry["scientific_role"],
        "lane": args.lane,
        "k": int(args.k),
        "n": int(len(carrier["ids"])),
        "registry_sha256": sha256_file(registry_path),
        "carrier_sha256": sha256_file(carrier_path),
        "candidate_bank_sha256": sha256_file(bank_path),
        "sanitized_candidate_evidence_sha256": sha256_file(evidence_csv),
        "sanitized_candidate_evidence_columns": list(records[0]),
        "ordered_id_sha256": sha256_array(np.asarray(carrier["ids"])),
        "candidate_id_sha256": sha256_array(candidate_ids),
        "candidate_partition_bank_sha256": sha256_array(candidate_partitions),
        "candidate_record_order_sha256": hashlib.sha256(
            canonical_json(records).encode("utf-8")
        ).hexdigest(),
        "candidate_count": len(candidate_ids),
        "unbiased_candidate_count": int(np.sum(candidate_mask(records, "UNBIASED_BANK"))),
        "relation_manifest": relation_manifest,
        "rows": rows,
        "partition_artifact_sha256": sha256_file(output),
        "artifact_reload": "PASS",
        "producer_label_reads": 0,
        "dense_n_by_n_count": 0,
        "trainable_parameter_count": 0,
        "thread_limit": 1,
        "wall_seconds": float(time.perf_counter() - started),
        "peak_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
    }
    output.with_suffix(".producer.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"lane": args.lane, "pass": len(partitions), "rows": len(rows)}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--carrier", required=True)
    parser.add_argument("--candidate-bank", required=True)
    parser.add_argument("--candidate-evidence", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--lane", required=True)
    parser.add_argument("--k", required=True, type=int)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
