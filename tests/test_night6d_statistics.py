from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.night6d_evaluate import bootstrap, exact_sign_flip, holm, spatial_summary


def test_exact_sign_flip_complete_enumeration_and_extremes():
    positive = exact_sign_flip(np.ones(10))
    zero = exact_sign_flip(np.zeros(10))
    assert positive["enumerations"] == 1024
    assert positive["tail_count"] == 1
    assert positive["raw_p"] == 1 / 1024
    assert zero["tail_count"] == 1024
    assert zero["raw_p"] == 1.0


def test_holm_step_down_is_monotone_and_key_stable():
    adjusted = holm({"p22": .04, "d1": .01})
    assert adjusted == {"d1": .02, "p22": .04}


def test_bootstrap_is_deterministic_and_uses_paired_rows():
    frame = pd.DataFrame({
        "delta_ari": np.arange(10) / 100,
        "delta_nmi": np.arange(10) / 200,
        "delta_q": np.arange(10) * .0075,
    })
    assert bootstrap(frame, n=1000) == bootstrap(frame, n=1000)


def test_spatial_gate_direction_and_boundary_report_only():
    protected = pd.DataFrame({
        "delta_neighbor": [-.02] * 10, "delta_moran": [-.04] * 10,
        "delta_geary": [.04] * 10, "delta_boundary": [.02] * 10,
    })
    failed = protected.copy()
    failed["delta_neighbor"] = -.04
    assert spatial_summary(protected)["spatial_gate_failed"] is False
    assert spatial_summary(failed)["spatial_gate_failed"] is True
    assert spatial_summary(failed)["boundary_disagreement_role"] == "report_only"
