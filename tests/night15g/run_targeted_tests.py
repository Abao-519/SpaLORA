#!/usr/bin/env python3
"""Risk-focused targeted tests for the frozen Night-15G delivery."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.night15g.night15g_optional_view_search import array_sha256  # noqa: E402


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--replay1", type=Path, required=True)
    parser.add_argument("--replay2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    project = args.project_root.resolve()
    started = time.perf_counter()
    results = []

    def test(name, fn):
        before = time.perf_counter()
        fn()
        results.append({"test": name, "status": "PASS", "wall_seconds": time.perf_counter() - before})

    core_path = workspace / "SpaLORA" / "night15g_optional_morphology_energy.py"
    replay_path = workspace / "scripts" / "night15g" / "night15g_frozen_artifact_replay.py"
    core_text = core_path.read_text(encoding="utf-8")
    tree = ast.parse(core_text)

    test(
        "core_has_no_dataset_or_lane_routing",
        lambda: check(
            not any(token in core_text for token in ("A1", "D1", "P22", "MISAR", "tonsil_s")),
            "core contains a dataset identifier",
        ),
    )
    test(
        "core_has_no_reference_label_input",
        lambda: check(
            not any(
                isinstance(node, ast.Name) and node.id in {"labels", "ground_truth", "annotation"}
                for node in ast.walk(tree)
            ),
            "core reads a reference-label symbol",
        ),
    )
    test(
        "core_has_no_dense_n_by_n_conversion",
        lambda: check("toarray(" not in core_text and "todense(" not in core_text, "dense sparse conversion found"),
    )

    replay1 = json.loads(args.replay1.read_text(encoding="utf-8"))
    replay2 = json.loads(args.replay2.read_text(encoding="utf-8"))
    test(
        "fresh_process_replay_counts",
        lambda: check(
            replay1["scientific_partition_profiles"] == replay2["scientific_partition_profiles"] == 12
            and replay1["missing_view_fallback_lanes"] == replay2["missing_view_fallback_lanes"] == 6,
            "replay count mismatch",
        ),
    )
    test(
        "fresh_process_partition_and_metric_exactness",
        lambda: check(
            replay1["rows"] == replay2["rows"],
            "fresh-process results differ",
        ),
    )
    test(
        "missing_view_fallback_is_exact",
        lambda: check(
            all(
                row["evidence_path"] == "presence_mask_zero_returns_night15f_authority"
                for row in replay1["rows"]
                if row["profile"] == "missing_view_exact_fallback"
            ),
            "missing-view fallback did not report exact return",
        ),
    )
    optional_ablation = workspace / "working" / "profile_replay_rev3_tonsil" / "matched_ablation.csv"
    test(
        "matched_optional_view_ablation_is_preserved",
        lambda: check(
            "MISSING_VIEW" in optional_ablation.read_text(encoding="utf-8")
            and "PERMUTED_MORPHOLOGY" in optional_ablation.read_text(encoding="utf-8"),
            "required optional-view ablations are absent",
        ),
    )

    morphology = workspace / "working" / "morphology"
    extraction = json.loads((morphology / "morphology_extraction_audit.json").read_text(encoding="utf-8"))
    test(
        "morphology_artifact_sha_and_ordered_id_contract",
        lambda: check(
            all(
                sha256_file(morphology / f"{item['unit_id']}_morphology_views.npz") == item["output_sha256"]
                and item["missing_after_transform"] == 0
                and item["all_patches_in_bounds"]
                for item in extraction["units"]
            ),
            "morphology artifact or registration mismatch",
        ),
    )

    d1_final = np.load(
        workspace / "working" / "seeded_refinement_d1_stage2_rev1" / "partitions" / "D1__balanced.npy",
        allow_pickle=False,
    ).astype(np.int32)
    test(
        "d1_singleton_is_preserved_not_backfilled",
        lambda: check(np.min(np.bincount(d1_final, minlength=10)) == 1, "D1 singleton was hidden or backfilled"),
    )
    test(
        "corrected_block_weight_semantics_registered",
        lambda: check(
            "post_weight_column_standardization" in (
                ledger_text := (
                    workspace / "working" / "formal_search_rev3_deterministic" / "D1_all_run_ledger.csv"
                ).read_text(encoding="utf-8")
            )
            and "post_weight_column_standardization\"\":false" in ledger_text,
            "corrected block-weight semantics absent",
        ),
    )
    test(
        "source_frozen_before_replays",
        lambda: check(
            max(core_path.stat().st_mtime, replay_path.stat().st_mtime)
            <= min(args.replay1.stat().st_mtime, args.replay2.stat().st_mtime),
            "core or replay source was modified after replay",
        ),
    )

    payload = {
        "status": "NIGHT15G_TARGETED_TESTS_PASS",
        "passed": len(results),
        "failed": 0,
        "wall_seconds": time.perf_counter() - started,
        "core_sha256": sha256_file(core_path),
        "replay_source_sha256": sha256_file(replay_path),
        "tests": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "passed": payload["passed"]}, indent=2))


if __name__ == "__main__":
    main()
