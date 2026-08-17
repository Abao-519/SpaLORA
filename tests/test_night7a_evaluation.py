import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/night7a_evaluate.py"
SPEC = importlib.util.spec_from_file_location("night7a_evaluate", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_exact_sign_flip_enumerates_full_space():
    assert MODULE.exact_sign_flip(np.array([1.0, 2.0, 3.0])) == 1 / 8


def test_bootstrap_is_fixed_100000_and_deterministic():
    values = np.array([-.1, .2, .3, .4, .5])
    first = MODULE.bootstrap(values)
    second = MODULE.bootstrap(values)
    assert first == second
    assert first["replicates"] == 100000
    assert first["seed"] == 20260818


def test_holm_step_down_monotonicity():
    observed = MODULE.holm({"a": .01, "b": .02, "c": .9})
    assert observed["a"] == .03
    assert observed["b"] == .04
    assert observed["c"] == .9


def test_spatial_gate_and_lower_is_better_direction():
    assert MODULE.spatial_fail({"mean_delta_neighbor": -.04,
                                "mean_delta_moran": -.04,
                                "mean_delta_geary": .00})
    assert MODULE.spatial_fail({"mean_delta_neighbor": -.04,
                                "mean_delta_moran": .00,
                                "mean_delta_geary": .04})
    assert not MODULE.spatial_fail({"mean_delta_neighbor": .01,
                                    "mean_delta_moran": .02,
                                    "mean_delta_geary": -.03})
    raw = np.array([-.2, .1, -.05])
    assert np.array_equal(MODULE.directional_improvement("geary_c", raw), -raw)
    assert np.array_equal(MODULE.directional_improvement("q", raw), raw)


def test_generalization_and_complexity_thresholds_are_exact():
    group = pd.DataFrame({
        "dataset": ["a1", "tonsil", "d1", "p22"],
        "complete": [True] * 4,
        "mean_delta_q": [.005, .006, .007, .008],
        "mean_delta_nmi": [.001] * 4,
        "mean_delta_ari": [.001, .001, .001, -.005],
        "wins_delta_q": [4, 4, 7, 7],
        "median_delta_q": [.001] * 4,
        "mean_delta_neighbor_agreement": [0.0] * 4,
        "mean_delta_moran_i": [0.0] * 4,
        "mean_delta_geary_c": [0.0] * 4,
    })
    observed = MODULE.generalization_components(group)
    assert observed["generalization"] is True
    group.loc[group.dataset == "p22", "mean_delta_ari"] = -.0050000001
    assert MODULE.generalization_components(group)["generalization"] is False

    option_a, option_b = MODULE.dual_complexity_options(
        macro=.0175, worst=.012, c00_macro=.010, c00_worst=.007,
        per_dataset_vs_c00=[-.002, 0, .01, .02],
    )
    assert option_a is True and option_b is True
    option_a, option_b = MODULE.dual_complexity_options(
        macro=.0175, worst=.012, c00_macro=.010, c00_worst=.007,
        per_dataset_vs_c00=[-.0020000001, 0, .01, .02],
    )
    assert option_a is False and option_b is False
