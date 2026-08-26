import numpy as np

from SpaLORA.night19c_zero_start_transfer import (
    PLACENTA_INCLUDED_ARMS,
    PLACENTA_PROFILES,
    Z01_CONSERVATIVE,
    bank_authority_sha,
    equal_weight_relation_posterior,
    select_placenta_bank,
)
from scripts.night17c.night17c_producer import CONFIGS


def candidate_ids():
    result = []
    for profile in PLACENTA_PROFILES:
        for arm in (*PLACENTA_INCLUDED_ARMS, "CSAD_CONFLICT_DISABLED", "CSAD_MODALITY_EDGE_PERMUTED"):
            result.append(f"{profile}__{arm}__E0")
    return result


def test_explicit_selection_preserves_original_semantic_ids():
    ids = candidate_ids()
    indices = select_placenta_bank(ids)
    selected = [ids[index] for index in indices]
    assert len(selected) == 16
    assert all("KMEANS_RETAINED" not in value for value in selected)
    assert not any("CONFLICT_DISABLED" in value or "PERMUTED" in value for value in selected)


def test_equal_weight_posterior_matches_manual_probability():
    partitions = np.asarray([[0, 0, 1, index % 2] for index in range(16)], dtype=np.int32)
    pair_i = np.asarray([0, 0, 1], dtype=np.int64)
    pair_j = np.asarray([1, 2, 3], dtype=np.int64)
    posterior = equal_weight_relation_posterior(partitions, pair_i, pair_j)
    manual = np.mean(partitions[:, pair_i] == partitions[:, pair_j], axis=0)
    assert np.allclose(posterior.probability_same, manual)
    assert np.array_equal(posterior.candidate_weights, np.full(16, 1.0 / 16.0))
    assert posterior.selected_candidate_count == 16


def test_equal_weight_posterior_is_candidate_order_invariant():
    rng = np.random.RandomState(7)
    partitions = rng.randint(0, 3, size=(16, 12), dtype=np.int32)
    pair_i = np.asarray([0, 2, 4, 6], dtype=np.int64)
    pair_j = np.asarray([1, 3, 5, 7], dtype=np.int64)
    a = equal_weight_relation_posterior(partitions, pair_i, pair_j)
    b = equal_weight_relation_posterior(partitions[::-1], pair_i, pair_j)
    assert np.array_equal(a.probability_same, b.probability_same)
    assert np.array_equal(a.uncertainty, b.uncertainty)


def test_z01_configuration_is_byte_semantic_match_to_night17c():
    authority = next(config for config in CONFIGS if config["config_id"] == "Z01_CONSERVATIVE")
    assert Z01_CONSERVATIVE == authority


def test_bank_authority_hash_binds_ids_partitions_and_weights():
    ids = [f"candidate-{index}" for index in range(16)]
    partitions = np.tile(np.arange(8, dtype=np.int32), (16, 1))
    baseline = bank_authority_sha(ids, partitions)
    changed = partitions.copy(); changed[0, 0] = 9
    assert bank_authority_sha(ids, changed) != baseline
    changed_ids = ids.copy(); changed_ids[0] = "different"
    assert bank_authority_sha(changed_ids, partitions) != baseline
