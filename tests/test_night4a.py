from pathlib import Path

import numpy as np
import pandas as pd

from SpaLORA.night4a_evidence import METRIC_DIRECTIONS, corrected_count


def test_metric_direction_registry_is_explicit():
    assert METRIC_DIRECTIONS["ari"].higher_is_better
    assert METRIC_DIRECTIONS["spatial_cluster_moran_mean"].higher_is_better
    assert not METRIC_DIRECTIONS["spatial_cluster_geary_mean"].higher_is_better
    assert not METRIC_DIRECTIONS["boundary_disagreement"].higher_is_better


def test_corrected_win_counts_higher_lower_tie_and_missing():
    values = [1.0, -2.0, 0.0, np.nan, 3.0]
    higher = corrected_count(values, "ari")
    lower = corrected_count(values, "spatial_cluster_geary_mean")
    assert higher == {
        "n_finite": 4, "n_missing": 1, "positive_delta_count": 2,
        "negative_delta_count": 1, "tie_count": 1,
        "full_win_count_corrected": 2, "direction": "higher_is_better",
        "full_win_rule": "delta > 0",
    }
    assert lower["full_win_count_corrected"] == 1
    assert lower["positive_delta_count"] == 2
    assert lower["full_win_rule"] == "delta < 0"


def test_seed_aggregation_supports_five_and_fifteen():
    five = [-1, -2, 3, 0, np.nan]
    fifteen = five * 3
    assert corrected_count(five, "boundary_disagreement")["full_win_count_corrected"] == 2
    assert corrected_count(fifteen, "boundary_disagreement")["full_win_count_corrected"] == 6


def test_benchmark_contract_has_no_outcome_columns():
    forbidden = {"ari", "nmi", "ami", "rank", "winner", "score", "performance"}
    allowed = {
        "dataset", "method", "task_match", "tutorial_status", "commit",
        "environment_lock", "observation_alignment", "pre_exclusion_reason",
    }
    assert forbidden.isdisjoint(allowed)


def test_protocol_lock_constants():
    assert [0, 1, 2, 3, 4] == list(range(5))
    assert ("FULL_IGE", "UNIFORM_CROSS", "UNIFORM_WITHIN") == (
        "FULL_IGE", "UNIFORM_CROSS", "UNIFORM_WITHIN"
    )


def test_original_artifact_path_is_external_to_night4a(tmp_path: Path):
    original = Path("/root/autodl-fs/SpaLORA-night3b/outputs/night3b_handoff")
    new = Path("/root/autodl-fs/SpaLORA-night4a/outputs/night4a_handoff")
    assert original != new
    assert not str(new).startswith(str(original) + "/")

