#!/usr/bin/env python3
"""Fresh-process reconstruction of frozen strict Night-16A partitions.

No public reference array is opened.  The process reconstructs generic starts,
observable statistics, calibrated parameters, unified energy and optional-view
fallback, then verifies the locked partition hashes.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from SpaLORA.night15e_continuous_reliability_energy import csr_from_archive
from SpaLORA.night15f_multiscale_expansion import prepare_expansion_evidence
from SpaLORA.night15g_optional_morphology_energy import prepare_optional_morphology_evidence
from SpaLORA.night16a_self_calibrating_energy import (
    CalibrationConstants,
    calibrate_from_statistics,
    observable_statistics,
    run_calibrated_energy,
)
from scripts.night16a.night16a_calibrated_arena import LANE_DATASET, optional_view


def sha(value):
    value = np.ascontiguousarray(value); h = hashlib.sha256(); h.update(value.dtype.str.encode()); h.update(np.asarray(value.shape, np.int64).tobytes()); h.update(value.tobytes()); return h.hexdigest()


def find_start(archive, lane, name):
    index = 0
    while f"{lane}__p{index:03d}" in archive:
        if str(archive[f"{lane}__n{index:03d}"][0]) == name:
            return np.asarray(archive[f"{lane}__p{index:03d}"], dtype=np.int32)
        index += 1
    raise RuntimeError(f"{lane}: missing start {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--banks", type=Path, required=True)
    parser.add_argument("--morphology", type=Path, required=True)
    parser.add_argument("--start-probe", type=Path, required=True)
    parser.add_argument("--descriptors", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    registry = json.loads((args.frozen / "frozen_crossfit_registry.json").read_text(encoding="utf-8"))
    descriptor = pd.read_csv(args.descriptors)
    if any(column in descriptor for column in ("absolute_ari", "absolute_nmi", "ami", "fmi")):
        raise RuntimeError("public metrics are reachable in replay descriptor file")
    starts = np.load(args.start_probe / "start_bank_partitions.npz", allow_pickle=False)
    rows = []
    for lane, locked in registry["lanes"].items():
        row = descriptor[descriptor.candidate_key == locked["candidate_key"]]
        if len(row) != 1:
            raise RuntimeError(f"{lane}: candidate descriptor is not unique")
        row = row.iloc[0]
        dataset = LANE_DATASET[lane]
        data = np.load(args.kit / f"{dataset}.npz", allow_pickle=False, mmap_mode="r")
        bank = np.load(args.banks / f"{dataset}_selected_partition_bank.npz", allow_pickle=False, mmap_mode="r")
        graph = csr_from_archive(data, "graph")
        graphs = (csr_from_archive(data, "operator4"), graph, csr_from_archive(data, "operator18"))
        retained = np.asarray(bank[f"{lane}__retained_embedding"], dtype=np.float32)
        initial = find_start(starts, lane, str(row.start_name))
        calibration_starts = [
            find_start(starts, lane, name)
            for name in sorted(set(descriptor.loc[descriptor.lane == lane, "start_name"].astype(str)))
        ]
        k = int(row.k)
        morph, presence = optional_view(args.morphology / f"{dataset}_morphology_views.npz")
        evidence = prepare_expansion_evidence(graphs, retained, data["view1"], data["view2"])
        statistics = observable_statistics(
            evidence, retained, data["view1"], data["view2"], calibration_starts, k, morph
        )
        constants = CalibrationConstants(**{
            field.name: row[f"calibration_constant__{field.name}"]
            for field in fields(CalibrationConstants)
        })
        calibrated = calibrate_from_statistics(statistics, constants)
        optional_cached = None
        if morph is not None:
            optional_cached = prepare_optional_morphology_evidence(
                graphs, retained, (data["view1"], data["view2"]), (morph,), (presence,)
            )
        partition, diagnostics = run_calibrated_energy(
            initial, k, evidence, calibrated, graphs, retained, data["view1"], data["view2"],
            morph, presence, optional_cached,
        )
        observed = sha(partition)
        if observed != locked["partition_sha256"]:
            raise RuntimeError(f"{lane}: replay hash {observed} != {locked['partition_sha256']}")
        expected = np.load(args.frozen / "partitions" / f"{lane}.npy", allow_pickle=False)
        if not np.array_equal(partition, expected):
            raise RuntimeError(f"{lane}: replay bytes differ from frozen partition")
        rows.append({
            "lane": lane, "partition_sha256": observed, "byte_exact": True,
            "total_observations": len(partition), "k": k,
            "retained_shape": list(retained.shape), "view1_shape": list(data["view1"].shape),
            "view2_shape": list(data["view2"].shape),
            "optional_shape": None if morph is None else list(morph.shape),
            "graph_shape_nnz": [int(graph.shape[0]), int(graph.shape[1]), int(graph.nnz)],
            "calibrated_parameters": {
                "pairwise_beta": calibrated.config.pairwise_beta,
                "self_return_strength": calibrated.config.self_return_strength,
                "size_prior": calibrated.config.size_prior,
                "scale_weights": [calibrated.config.scale_fine, calibrated.config.scale_registered, calibrated.config.scale_broad],
                "optional_weight": calibrated.optional_config.morphology_unary_weight,
            },
            "diagnostics": {key: value for key, value in diagnostics.items() if isinstance(value, (int, float, str))},
        })
    payload = {
        "status": "NIGHT16A_STRICT_FRESH_PROCESS_REPLAY_PASS",
        "lanes": len(rows), "byte_exact_lanes": sum(row["byte_exact"] for row in rows),
        "held_out_reference_arrays_opened": 0, "public_metric_columns_in_descriptors": 0,
        "dense_n_by_n_count": 0, "wall_seconds": time.perf_counter() - started,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in payload if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
