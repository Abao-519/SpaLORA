from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from SpaLORA.family_runtime import (
    EXPECTED_DATA_STEWARD_ROWS, EXPECTED_UNIT_ORDER, load_q00_rows,
    sparse_roundtrip_audit,
)


REPO = Path(__file__).resolve().parents[2]


def test_q00_authority_is_exact_30_row_set() -> None:
    rows = load_q00_rows(
        REPO / "outputs/night10a_rev2_handoff/r2_lock/"
        "r2_total_lock_independent_audit.json"
    )
    assert len(rows) == 30
    assert tuple(row["unit_id"] for row in rows) == EXPECTED_UNIT_ORDER
    assert tuple((row["dataset"], row["seed"]) for row in rows) == EXPECTED_DATA_STEWARD_ROWS
    assert len({row["reference_embedding_sha256"] for row in rows}) == 30
    assert len({row["reference_partition_sha256"] for row in rows}) == 30


def test_training_worker_has_no_label_module_import() -> None:
    source = (REPO / "scripts/night10b/run_family_policy.py").read_text(encoding="utf-8").lower()
    assert "night10a_rev2_evaluate" not in source
    assert "annotation_reader" not in source
    assert "misar_y" not in source
    assert "adjusted_rand" not in source


def test_sparse_roundtrip_accepts_float_jitter_but_rejects_support_drift() -> None:
    expected = sp.csr_matrix(np.asarray([[0.0, 0.25], [0.75, 0.0]]))
    observed = expected.copy()
    observed.data += np.asarray([2e-9, -3e-9])
    audit = sparse_roundtrip_audit(observed, expected)
    assert audit["sparse_support_exact"] is True
    assert audit["max_absolute_error"] <= 3.1e-9
    drift = sp.csr_matrix(np.asarray([[0.1, 0.25], [0.75, 0.0]]))
    with pytest.raises(RuntimeError, match="support mismatch"):
        sparse_roundtrip_audit(drift, expected)
