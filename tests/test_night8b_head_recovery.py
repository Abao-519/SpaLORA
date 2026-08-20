from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from SpaLORA.night8b_head_recovery import (
    HEAD_CONFIG, HEAD_CONFIG_SHA256, canonical_affinity,
    canonical_json_sha, recovery_eigen_kmeans100, run_twice,
)


REPO = Path(__file__).resolve().parents[1]


def ring_affinity(n: int = 120) -> sp.csr_matrix:
    row = np.arange(n)
    col = (row + 1) % n
    directed = sp.coo_matrix((np.ones(n), (row, col)), shape=(n, n))
    return (directed + directed.T).tocsr()


def test_head_signature_is_identity_blind_and_single_argument():
    signature = inspect.signature(recovery_eigen_kmeans100)
    assert list(signature.parameters) == ["affinity"]


def test_head_config_sha_is_canonical_and_locked():
    assert HEAD_CONFIG_SHA256 == canonical_json_sha(HEAD_CONFIG)
    assert HEAD_CONFIG["eigensolver"]["k"] == 12
    assert HEAD_CONFIG["kmeans"] == {
        "n_clusters": 12, "n_init": 100,
        "random_state": 2020, "algorithm": "lloyd",
    }


def test_affinity_contract_symmetrizes_and_zeros_diagonal():
    value = sp.csr_matrix(np.asarray([[7.0, 2.0], [0.0, 3.0]]))
    actual = canonical_affinity(value).toarray()
    np.testing.assert_array_equal(actual, np.asarray([[0.0, 1.0], [1.0, 0.0]]))


def test_zero_degree_fails_closed():
    value = sp.block_diag((ring_affinity(120), sp.csr_matrix((1, 1))), format="csr")
    with pytest.raises(RuntimeError, match="RECOVERY_BLOCKED_HEAD_NUMERICS"):
        recovery_eigen_kmeans100(value)


def test_same_process_repeat_is_exact_and_K12():
    labels, audit = run_twice(ring_affinity())
    assert len(np.unique(labels)) == 12
    assert audit["deterministic_exact"] is True
    assert audit["first_partition_sha256"] == audit["second_partition_sha256"]


def test_prelabel_contract_if_present():
    out = REPO / "outputs/night8b_head_recovery"
    path = out / "recovery_input_view_manifest.json"
    if not path.is_file():
        pytest.skip("P0 input lock not generated yet")
    payload = json.loads(path.read_text())
    assert payload["status"] == "LOCKED_PRELABEL"
    assert payload["row_count"] == 20
    assert len(payload["rows"]) == 20
    assert {row["method"] for row in payload["rows"]} == {"HR_U00", "HR_F00"}
    assert all(row["K"] == 12 for row in payload["rows"])
    assert len({row["head_config_sha256"] for row in [payload]}) == 1


def test_total_lock_contract_if_present():
    path = REPO / "outputs/night8b_head_recovery/locked_recovery_partition_manifest.json"
    if not path.is_file():
        pytest.skip("partition total lock not generated yet")
    payload = json.loads(path.read_text())
    assert payload["status"] == "TOTAL_LOCKED_BEFORE_LABEL_ACCESS"
    assert payload["row_count"] == 20
    assert payload["exact_K"] == "20/20"
    assert payload["deterministic_exact"] == "20/20"
    assert payload["same_head_config"] is True
    assert payload["training"] == payload["adapter"] == payload["affinity_rebuild"] == 0
