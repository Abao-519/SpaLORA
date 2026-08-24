#!/usr/bin/env python3
"""Matched post-freeze mechanism ablations for the strict automatic path."""

from __future__ import annotations

import argparse
import csv
from dataclasses import fields, replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

from SpaLORA.night15e_continuous_reliability_energy import csr_from_archive
from SpaLORA.night15f_multiscale_expansion import prepare_expansion_evidence
from SpaLORA.night15g_optional_morphology_energy import prepare_optional_morphology_evidence
from SpaLORA.night16a_self_calibrating_energy import (
    CalibratedEnergy, CalibrationConstants, calibrate_from_statistics,
    observable_statistics, run_calibrated_energy,
)
from scripts.night15g.night15g_optional_view_search import evaluate
from scripts.night16a.night16a_calibrated_arena import LANE_DATASET, lane_semantics, optional_view
from scripts.night16a.night16a_final_replay import find_start, sha


def main():
    p = argparse.ArgumentParser()
    for name in ("kit", "banks", "morphology", "start_probe", "descriptors", "frozen", "output"):
        p.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = p.parse_args()
    registry = json.loads((args.frozen / "frozen_crossfit_registry.json").read_text(encoding="utf-8"))
    descriptor = pd.read_csv(args.descriptors)
    starts_archive = np.load(args.start_probe / "start_bank_partitions.npz", allow_pickle=False)
    rows = []
    for lane, locked in registry["lanes"].items():
        source = descriptor[descriptor.candidate_key == locked["candidate_key"]].iloc[0]
        dataset = LANE_DATASET[lane]
        data = np.load(args.kit / f"{dataset}.npz", allow_pickle=False, mmap_mode="r")
        bank = np.load(args.banks / f"{dataset}_selected_partition_bank.npz", allow_pickle=False)
        k, labels, mask = lane_semantics(data, lane)
        graph = csr_from_archive(data, "graph")
        graphs = (csr_from_archive(data, "operator4"), graph, csr_from_archive(data, "operator18"))
        retained = np.asarray(bank[f"{lane}__retained_embedding"], dtype=np.float32)
        initial = find_start(starts_archive, lane, source.start_name)
        calibration_starts = [
            find_start(starts_archive, lane, name)
            for name in sorted(set(descriptor.loc[descriptor.lane == lane, "start_name"].astype(str)))
        ]
        morph, presence = optional_view(args.morphology / f"{dataset}_morphology_views.npz")
        evidence = prepare_expansion_evidence(graphs, retained, data["view1"], data["view2"])
        constants = CalibrationConstants(**{
            field.name: source[f"calibration_constant__{field.name}"]
            for field in fields(CalibrationConstants)
        })
        statistics = observable_statistics(evidence, retained, data["view1"], data["view2"], calibration_starts, k, morph)
        calibrated = calibrate_from_statistics(statistics, constants)
        optional_cached = None if morph is None else prepare_optional_morphology_evidence(
            graphs, retained, (data["view1"], data["view2"]), (morph,), (presence,)
        )
        equal_local = replace(calibrated.config.local, unary_temperature=1_000_000.0, retained_bias=0.0, view_balance=0.0)
        variants = {
            "FULL": calibrated,
            "NO_SELF_RETURN": replace(calibrated, config=replace(calibrated.config, self_return_strength=0.0), optional_config=replace(calibrated.optional_config, base=replace(calibrated.config, self_return_strength=0.0))),
            "REGISTERED_SCALE_ONLY": replace(calibrated, config=replace(calibrated.config, scale_fine=0.0, scale_registered=1.0, scale_broad=0.0), optional_config=replace(calibrated.optional_config, base=replace(calibrated.config, scale_fine=0.0, scale_registered=1.0, scale_broad=0.0))),
            "FIXED_EQUAL_MODALITY_WEIGHTS": replace(calibrated, config=replace(calibrated.config, local=equal_local), optional_config=replace(calibrated.optional_config, base=replace(calibrated.config, local=equal_local))),
        }
        for ablation, calibration in variants.items():
            partition, _ = run_calibrated_energy(
                initial, k, evidence, calibration, graphs, retained, data["view1"], data["view2"], morph, presence, optional_cached
            )
            metric = evaluate(labels, mask, partition, graph); sizes = np.bincount(partition, minlength=k)
            rows.append({"lane": lane, "ablation": ablation, **metric, "min_cluster_size": int(sizes.min()), "partition_sha256": sha(partition)})
        if morph is not None:
            for ablation, current in (("MORPHOLOGY_MISSING", None), ("MORPHOLOGY_PERMUTED", morph[np.random.default_rng(20260824).permutation(len(morph))])):
                current_presence = None if current is None else presence
                current_statistics = observable_statistics(evidence, retained, data["view1"], data["view2"], calibration_starts, k, current)
                current_calibration = calibrate_from_statistics(current_statistics, constants)
                current_optional = None if current is None else prepare_optional_morphology_evidence(
                    graphs, retained, (data["view1"], data["view2"]), (current,), (current_presence,)
                )
                partition, _ = run_calibrated_energy(
                    initial, k, evidence, current_calibration, graphs, retained, data["view1"], data["view2"], current, current_presence, current_optional
                )
                metric = evaluate(labels, mask, partition, graph); sizes = np.bincount(partition, minlength=k)
                rows.append({"lane": lane, "ablation": ablation, **metric, "min_cluster_size": int(sizes.min()), "partition_sha256": sha(partition)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__": main()
