import sys
from pathlib import Path


RUNNER_DIR = Path(__file__).resolve().parents[2] / "scripts" / "night14a"
sys.path.insert(0, str(RUNNER_DIR))

import numpy as np

from night14a_run import (  # noqa: E402
    cached_partition, retag_reused_endpoint, roundtrip_status,
)


def test_byte_identical_endpoint_reuse_preserves_partition_and_metrics():
    rows = [{
        "filter_id": "identity", "endpoint_seed": 3,
        "partition_sha256": "abc", "absolute_ari": 0.4,
        "endpoint_wall_seconds": 2.5,
    }]
    summary = {"filter_id": "identity", "ari_mean": 0.4}
    reused_rows, reused_summary = retag_reused_endpoint(
        rows, summary, "tcf_reject", "identity"
    )
    assert reused_rows[0]["partition_sha256"] == "abc"
    assert reused_rows[0]["absolute_ari"] == 0.4
    assert reused_rows[0]["endpoint_wall_seconds"] == 0.0
    assert reused_rows[0]["source_endpoint_wall_seconds"] == 2.5
    assert reused_rows[0]["endpoint_reuse_of_filter_id"] == "identity"
    assert reused_summary["endpoint_reused"] is True


def test_roundtrip_status_does_not_promote_nonprimary_ablation_tie_to_checkpoint_failure():
    rows = [
        {"object": "z1", "numerical_roundtrip": True},
        {"object": "W00_IDENTITY", "numerical_roundtrip": True,
         "partition_exact": True},
        {"object": "W01_FIXED_LOW60", "numerical_roundtrip": True,
         "partition_exact": False},
        {"object": "W02_TCF_FINAL", "numerical_roundtrip": True,
         "partition_exact": True},
    ]
    audit = roundtrip_status(rows)
    assert audit["all_embedding_numerical_roundtrip"] is True
    assert audit["all_filter_partition_exact"] is False
    assert audit["required_filter_partition_exact"] is True


def test_fresh_reload_reuses_partition_for_byte_identical_embeddings():
    embedding = np.asarray([[0.0], [0.0], [1.0], [1.0]], dtype=np.float32)
    cache = {}
    first, reused_first, source_first = cached_partition(
        embedding, 2, 0, 10, cache, "W00_IDENTITY"
    )
    second, reused_second, source_second = cached_partition(
        embedding.copy(), 2, 0, 10, cache, "W02_TCF_FINAL"
    )
    assert reused_first is False and source_first is None
    assert reused_second is True and source_second == "W00_IDENTITY"
    assert np.array_equal(first, second)
