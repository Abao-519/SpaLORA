#!/usr/bin/env python3
"""Build the preregistered label-free placenta relation bank from Night-19B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from SpaLORA.night17b_sfrd import sha256_array
from SpaLORA.night19c_zero_start_transfer import (
    bank_authority_sha,
    file_sha256,
    select_placenta_bank,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-artifact", required=True)
    parser.add_argument("--source-producer", required=True)
    parser.add_argument("--output-bank", required=True)
    parser.add_argument("--output-manifest", required=True)
    args = parser.parse_args()
    artifact_path = Path(args.source_artifact)
    producer_path = Path(args.source_producer)
    producer = json.loads(producer_path.read_text(encoding="utf-8"))
    if file_sha256(artifact_path) != producer["artifact_sha256"]:
        raise ValueError("Night-19B source artifact SHA mismatch")
    if producer.get("producer_label_reads") != 0 or producer.get("carrier_annotation_arrays_accessed") != 0:
        raise ValueError("Night-19B producer label-flow authority is not clean")
    with np.load(artifact_path, allow_pickle=False) as archive:
        ids = np.asarray(archive["ids"])
        candidate_ids = archive["candidate_ids"].astype(str)
        partitions = np.asarray(archive["partitions"], dtype=np.int32)
    if candidate_ids.tolist() != producer["candidate_ids"]:
        raise ValueError("Night-19B candidate ID order differs from manifest")
    source_partition_sha = [sha256_array(row) for row in partitions]
    if source_partition_sha != producer["partition_sha256"]:
        raise ValueError("Night-19B partition SHA authority mismatch")
    selected = select_placenta_bank(candidate_ids)
    selected_ids = candidate_ids[selected]
    selected_partitions = partitions[selected]
    weights = np.full(16, 1.0 / 16.0, dtype=np.float64)
    output = Path(args.output_bank)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        ids=ids,
        candidate_ids=selected_ids.astype("U"),
        partitions=selected_partitions,
        candidate_weights=weights,
    )
    selected_sha = [sha256_array(row) for row in selected_partitions]
    manifest = {
        "schema": "night19c-placenta-unbiased-bank-v1",
        "bank_id": "PLACENTA_UNBIASED_BANK_V1",
        "source_artifact": str(artifact_path.resolve()),
        "source_artifact_sha256": file_sha256(artifact_path),
        "source_producer": str(producer_path.resolve()),
        "source_producer_sha256": file_sha256(producer_path),
        "source_candidates_locked_before_evaluation": True,
        "source_producer_label_reads": 0,
        "candidate_count": 16,
        "candidate_ids": selected_ids.tolist(),
        "partition_sha256": selected_sha,
        "candidate_weights": weights.tolist(),
        "candidate_weighting": "EXPLICIT_EQUAL_WEIGHT_1_OVER_16",
        "bank_authority_sha256": bank_authority_sha(selected_ids, selected_partitions),
        "ordered_ids_sha256": sha256_array(ids.astype("U")),
        "output_bank_sha256": file_sha256(output),
        "included_semantics": "2_PROFILES_X_8_NON_ABLATION_NON_PERMUTED_ARMS",
        "excluded_semantics": ["CSAD_CONFLICT_DISABLED", "CSAD_MODALITY_EDGE_PERMUTED"],
        "labels_read": 0,
    }
    Path(args.output_manifest).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
