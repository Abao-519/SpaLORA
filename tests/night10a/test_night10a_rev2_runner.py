from __future__ import annotations

import inspect

import numpy as np
import pytest

from scripts.night10a import night10a_rev2_p0 as p0
from scripts.night10a import night10a_rev2_run as runner


def test_real_matrix_and_formal_cardinality_are_preregistered():
    assert sum(len(list(spec["r2"])) for spec in runner.DATA.values()) == 30
    assert len(runner.CANDIDATES) == 7
    assert sum(len(list(spec["r2"])) * len(runner.CANDIDATES)
               for spec in runner.DATA.values()) == 210
    assert sum(len(list(spec["r1"])) * len(runner.CANDIDATES)
               for spec in runner.DATA.values()) == 84


def test_prepare_one_has_full_real_schema_and_frozen_harmonizer_gates():
    source = inspect.getsource(runner.prepare_one)
    for token in ("d_z1", "d_z2", "d_zf_raw", "d_zf_aligned", "d_coords",
                  "graph_shape", "ordered_observation_sha256", "adapter_input_width",
                  "q06_adapter_input_width", "all_finite", "harmonizer_sha256"):
        assert token in source
    assert "frozen_reference_harmonizer" in source
    assert "dataset == \"p22\"" not in inspect.getsource(runner.load_zf_aligned)


def test_real_schema_dtype_gate_accepts_numeric_coordinates_only():
    runner._assert_real_array("coordinates", np.arange(12).reshape(6, 2), 6, floating=False)
    with pytest.raises(AssertionError):
        runner._assert_real_array("view", np.arange(12).reshape(6, 2), 6)


def test_formal_config_requires_exact_preflight_binding_and_zero_labels():
    source = inspect.getsource(runner.prepare_configs)
    assert "preflight_row_sha256" in source
    assert "registered_input_sha256" in source
    assert "harmonizer_sha256" in source
    assert '"label_access": 0' in source


def test_p0_row_executes_full_runtime_actions_once():
    source = inspect.getsource(p0.one_runtime_row)
    for token in ("qcrd_forward", "torch_loss", ".backward()", "optimizer.step()",
                  "torch.save", "load_state_dict", "corrected_views"):
        assert token in source
    assert '"optimizer_steps": 1' in source
    assert p0.LIMIT_SECONDS == 90 * 60


def test_no_label_loader_or_scientific_retry_in_p0_and_runner():
    combined = inspect.getsource(p0) + inspect.getsource(runner)
    assert "load_labels(" not in combined
    assert "from scripts.night7b_evaluate import" not in combined
    assert '"scientific_retry": 0' in combined
    assert '"fallback": 0' in combined
    assert "ThreadPoolExecutor(max_workers=4)" in combined
    assert "timeout=1800" in combined
    assert '"-m","scripts.night10a.night10a_rev2_run"' in combined
