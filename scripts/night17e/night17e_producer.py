#!/usr/bin/env python3
"""Label-closed Night-17E LRCC partition producer."""

from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np
import scipy.sparse as sp
import torch

from SpaLORA.night15e_continuous_reliability_energy import ContinuousEnergyConfig
from SpaLORA.night15f_multiscale_expansion import ExpansionEnergyConfig, prepare_expansion_evidence
from SpaLORA.night16e_tsre import partition_sha256
from SpaLORA.night17c_zero_start import reload_zero_start
from SpaLORA.night17e_lrcc import (
    LRCCConfig,
    lrcc_expansion,
    prepare_relation_evidence,
    relation_conditioned_pairwise,
)


Z01 = {
    "config_id": "Z01_CONSERVATIVE",
    "hidden_dim": 32,
    "residual_scale": 0.05,
    "learning_rate": 0.0007,
    "steps": 40,
    "relation_weight": 1.0,
    "anchor_weight": 4.0,
    "self_return_weight": 4.0,
    "consistency_weight": 0.5,
    "variance_weight": 0.1,
}


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_array(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_csr(archive: np.lib.npyio.NpzFile, prefix: str) -> sp.csr_matrix:
    return sp.csr_matrix(
        (
            np.asarray(archive[f"{prefix}__data"]),
            np.asarray(archive[f"{prefix}__indices"], dtype=np.int32),
            np.asarray(archive[f"{prefix}__indptr"], dtype=np.int32),
        ),
        shape=tuple(np.asarray(archive[f"{prefix}__shape"], dtype=np.int64)),
    )


def load_carrier(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as archive:
        output = {
            "ids": np.asarray(archive["ids"]).astype("U"),
            "view1": np.asarray(archive["view1"], dtype=np.float32),
            "view2": np.asarray(archive["view2"], dtype=np.float32),
            "retained": np.asarray(archive["retained"], dtype=np.float32),
            "start_ids": np.asarray(archive["start_ids"]).astype("U"),
            "start_partitions": np.asarray(archive["start_partitions"], dtype=np.int32),
            "graphs": tuple(load_csr(archive, f"graph{i}") for i in range(3)),
        }
    n = len(output["ids"])
    if not all(len(output[key]) == n for key in ("view1", "view2", "retained")):
        raise ValueError("carrier observation mismatch")
    return output


def load_config(value: dict[str, object]) -> LRCCConfig:
    base_value = dict(value["base"])
    local = ContinuousEnergyConfig(**dict(base_value.pop("local")))
    base = ExpansionEnergyConfig(local=local, **base_value)
    extra = {field.name: value[field.name] for field in fields(LRCCConfig) if field.name != "base"}
    return LRCCConfig(base=base, **extra)


def _seed_root(seed: int) -> Path:
    return Path("/root/night17c_p0_working/formal") if seed == 0 else Path(
        f"/root/night17c_p0_working/confirmation/seed{seed}"
    )


def load_frozen_representations(
    lane: str, carrier: dict[str, object]
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict[str, object]]]:
    learned: list[np.ndarray] = []
    zero: list[np.ndarray] = []
    authority: list[dict[str, object]] = []
    for seed in (0, 1, 2):
        root = _seed_root(seed) / lane
        artifact_path = root / "producer.npz"
        checkpoint_path = root / "checkpoint.pt"
        producer_manifest_path = root / "producer.producer.json"
        with np.load(artifact_path, allow_pickle=False) as artifact:
            if not np.array_equal(np.asarray(artifact["ids"]).astype("U"), carrier["ids"]):
                raise ValueError("Night-17C artifact/carrier ordered ID mismatch")
            zero_representation = np.asarray(artifact["unbiased_smooth"], dtype=np.float32)
            gate = np.asarray(artifact["primary_node_gate"], dtype=np.float32)
            run_ids = np.asarray(artifact["run_ids"]).astype("U")
            arm_ids = np.asarray(artifact["arm_ids"]).astype("U")
            config_ids = np.asarray(artifact["config_ids"]).astype("U")
            representation_hashes = np.asarray(artifact["representation_sha256"]).astype("U")
            match = np.flatnonzero(
                (arm_ids == "UNBIASED_FULL") & (config_ids == "Z01_CONSERVATIVE")
            )
            zero_match = np.flatnonzero(run_ids == "BASELINE__ZERO_RESIDUAL")
            if len(match) != 1 or len(zero_match) != 1:
                raise ValueError("Night-17C Z01/zero representation authority missing")
            expected_learned_sha = str(representation_hashes[int(match[0])])
            expected_zero_sha = str(representation_hashes[int(zero_match[0])])
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        producer_manifest = json.loads(producer_manifest_path.read_text(encoding="utf-8"))
        replay_device = str(producer_manifest.get("device", "cpu"))
        if replay_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("Night-17C authority requires CUDA strict replay but CUDA is unavailable")
        configs = [dict(value) for value in checkpoint["configs"]]
        matches = [value for value in configs if value.get("config_id") == "Z01_CONSERVATIVE"]
        if matches != [Z01]:
            raise ValueError("Night-17C checkpoint Z01 config mismatch")
        state = checkpoint["unbiased_full_state_dict"]["Z01_CONSERVATIVE"]
        representation = reload_zero_start(
            state,
            carrier["view1"],
            carrier["view2"],
            carrier["retained"],
            zero_representation,
            gate,
            Z01,
            device=replay_device,
        )
        learned_sha = sha256_array(representation)
        zero_sha = sha256_array(zero_representation)
        if learned_sha != expected_learned_sha or zero_sha != expected_zero_sha:
            raise RuntimeError("Night-17C strict checkpoint representation replay mismatch")
        learned.append(representation)
        zero.append(zero_representation)
        authority.append(
            {
                "seed": seed,
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "artifact_sha256": sha256_file(artifact_path),
                "original_training_device": replay_device,
                "learned_representation_sha256": learned_sha,
                "zero_representation_sha256": zero_sha,
            }
        )
    return learned, zero, authority


def load_starts(
    registry: dict[str, object], carrier: dict[str, object], candidate_bank_path: Path
) -> list[dict[str, object]]:
    with np.load(candidate_bank_path, allow_pickle=False) as bank:
        if not np.array_equal(np.asarray(bank["ids"]).astype("U"), carrier["ids"]):
            raise ValueError("candidate-bank/carrier ordered ID mismatch")
        candidate_ids = np.asarray(bank["candidate_ids"]).astype("U")
        candidate_partitions = np.asarray(bank["partitions"], dtype=np.int32)
    carrier_start_ids = np.asarray(carrier["start_ids"]).astype("U")
    carrier_partitions = np.asarray(carrier["start_partitions"], dtype=np.int32)
    output = []
    for value in registry["starts"]:
        if value["source"] == "candidate_bank":
            match = np.flatnonzero(candidate_ids == value["source_id"])
            if len(match) != 1:
                raise ValueError(f"candidate-bank start missing: {value['source_id']}")
            partition = candidate_partitions[int(match[0])]
        elif value["source"] == "carrier_start":
            match = np.flatnonzero(carrier_start_ids == value["source_id"])
            if len(match) != 1:
                raise ValueError(f"carrier start missing: {value['source_id']}")
            partition = carrier_partitions[int(match[0])]
        else:
            raise ValueError("unknown start source")
        output.append(
            {
                **value,
                "partition": np.asarray(partition, dtype=np.int32),
                "partition_sha256": partition_sha256(partition),
            }
        )
    return output


def run(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    registry = json.loads(Path(args.registry).read_text(encoding="utf-8"))
    carrier = load_carrier(Path(args.carrier))
    starts = load_starts(registry, carrier, Path(args.candidate_bank))
    for start in starts:
        if np.unique(start["partition"]).size != int(args.k):
            raise ValueError("start partition K mismatch")
    learned, zero, checkpoint_authority = load_frozen_representations(args.lane, carrier)
    evidence = prepare_expansion_evidence(
        carrier["graphs"], carrier["retained"], carrier["view1"], carrier["view2"]
    )
    relation_cache = {}
    config_cache = {}
    capacity_preflight = {}
    for candidate in registry["candidates"]:
        config = load_config(candidate["config"])
        config_cache[candidate["config_id"]] = config
        relation_cache[candidate["config_id"]] = prepare_relation_evidence(
            evidence, learned, zero, config
        )
        _, _, learned_capacity, _, learned_diag = relation_conditioned_pairwise(
            evidence, relation_cache[candidate["config_id"]], config, "LEARNED_RELATION", carrier["ids"]
        )
        _, _, disabled_capacity, _, _ = relation_conditioned_pairwise(
            evidence, relation_cache[candidate["config_id"]], config, "RELATION_DISABLED", carrier["ids"]
        )
        capacity_delta = float(np.max(np.abs(learned_capacity - disabled_capacity)))
        if capacity_delta <= 0:
            raise RuntimeError("nontrivial learned support did not alter pairwise capacity")
        matched_mass_errors = {}
        for matched_arm in ("ZERO_RELATION", "PERMUTED_RELATION", "UNIFORM_MASS_MATCHED"):
            _, _, _, _, matched_diag = relation_conditioned_pairwise(
                evidence,
                relation_cache[candidate["config_id"]],
                config,
                matched_arm,
                carrier["ids"],
            )
            matched_mass_errors[matched_arm] = {
                "absolute": [
                    float(value["mass_match_absolute_error"])
                    for value in matched_diag["scale_diagnostics"]
                ],
                "relative": [
                    float(value["mass_match_relative_error"])
                    for value in matched_diag["scale_diagnostics"]
                ],
            }
        capacity_preflight[candidate["config_id"]] = {
            "learned_vs_disabled_max_abs_capacity_delta": capacity_delta,
            "matched_control_mass_errors": matched_mass_errors,
            "max_matched_absolute_mass_error": max(
                max(value["absolute"]) for value in matched_mass_errors.values()
            ),
            "max_matched_relative_mass_error": max(
                max(value["relative"]) for value in matched_mass_errors.values()
            ),
        }
    partitions: list[np.ndarray] = []
    rows: list[dict[str, object]] = []

    def append(partition: np.ndarray, row: dict[str, object]) -> None:
        partition = np.asarray(partition, dtype=np.int32)
        sizes = np.bincount(partition, minlength=int(args.k))
        if np.unique(partition).size != int(args.k) or np.any(sizes <= 0):
            raise RuntimeError("locked partition violates exact K/no empty")
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

    for start in starts:
        append(
            start["partition"].copy(),
            {
                "run_id": f"{start['start_id']}__INPUT_START",
                "start_id": start["start_id"],
                "start_role": start["role"],
                "source_id": start["source_id"],
                "arm": "INPUT_START",
                "config_id": "INPUT",
                "config_sha256": hashlib.sha256(b"null").hexdigest(),
                "initial_partition_sha256": start["partition_sha256"],
                "changed_from_initial": 0,
                "wall_seconds": 0.0,
            },
        )
        for candidate in registry["candidates"]:
            config = config_cache[candidate["config_id"]]
            relation = relation_cache[candidate["config_id"]]
            for arm in registry["arms"]:
                candidate_started = time.perf_counter()
                row = {
                    "run_id": f"{start['start_id']}__{candidate['config_id']}__{arm}",
                    "start_id": start["start_id"],
                    "start_role": start["role"],
                    "source_id": start["source_id"],
                    "arm": arm,
                    "config_id": candidate["config_id"],
                    "config_sha256": hashlib.sha256(
                        canonical_json(candidate["config"]).encode()
                    ).hexdigest(),
                    "initial_partition_sha256": start["partition_sha256"],
                    "status": "FAILED",
                    "failure": "",
                }
                try:
                    partition, diagnostics = lrcc_expansion(
                        start["partition"],
                        args.k,
                        carrier["ids"],
                        evidence,
                        relation,
                        config,
                        arm,
                    )
                    row.update(
                        changed_from_initial=int(np.sum(partition != start["partition"])),
                        diagnostics=diagnostics,
                        wall_seconds=float(time.perf_counter() - candidate_started),
                    )
                    append(partition, row)
                except Exception as exc:
                    row["failure"] = f"{type(exc).__name__}: {exc}"
                    row["wall_seconds"] = float(time.perf_counter() - candidate_started)
                    rows.append(row)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        ids=np.asarray(carrier["ids"]),
        run_ids=np.asarray([row.get("run_id", "") for row in rows if row["status"] == "PASS"]).astype("U"),
        partitions=np.stack(partitions).astype(np.int32),
    )
    temporary.replace(output)
    with np.load(output, allow_pickle=False) as replay:
        if not np.array_equal(np.asarray(replay["partitions"]), np.stack(partitions)):
            raise RuntimeError("saved partition artifact reload mismatch")
    manifest = {
        "schema": "night17e-lrcc-producer-v1",
        "lane": args.lane,
        "k": int(args.k),
        "n": int(len(carrier["ids"])),
        "carrier": str(Path(args.carrier).resolve()),
        "carrier_sha256": sha256_file(Path(args.carrier)),
        "candidate_bank": str(Path(args.candidate_bank).resolve()),
        "candidate_bank_sha256": sha256_file(Path(args.candidate_bank)),
        "ordered_id_sha256": sha256_array(np.asarray(carrier["ids"])),
        "checkpoint_authority": checkpoint_authority,
        "capacity_preflight": capacity_preflight,
        "learned_representation_bank_sha256": sha256_array(np.stack(learned)),
        "zero_representation_bank_sha256": sha256_array(np.stack(zero)),
        "rows": rows,
        "partition_artifact_sha256": sha256_file(output),
        "artifact_reload": "PASS",
        "producer_label_reads": 0,
        "dense_n_by_n_count": 0,
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
    parser.add_argument("--registry", required=True)
    parser.add_argument("--lane", required=True)
    parser.add_argument("--k", required=True, type=int)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
