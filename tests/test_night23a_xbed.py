import numpy as np
import scipy.sparse as sp
import torch

from SpaLORA.night23a_xbed import (
    EdgeModelConfig,
    FEATURE_NAMES,
    STAGE_B_ARMS,
    array_sha,
    build_union_edge_features,
    confidence_abstention_weights,
    deterministic_balanced_indices,
    fit_logistic,
    fit_mlp,
    predict_logistic,
    predict_mlp,
    source_lanes,
    spectral_exact_k_partition,
    stage_b_arm_weights,
    teacher_relations,
)


def ring_graph(n):
    rows = np.arange(n)
    cols = (rows + 1) % n
    graph = sp.coo_matrix((np.ones(n), (rows, cols)), shape=(n, n))
    return (graph + graph.T).tocsr()


def test_teacher_relation_is_cluster_label_permutation_invariant():
    partition = np.asarray([0, 0, 1, 2, 2, 1])
    rows = np.asarray([0, 0, 1, 2, 3])
    cols = np.asarray([1, 2, 5, 4, 5])
    relabeled = np.asarray([7, 7, 3, 9, 9, 3])
    assert len(np.unique(partition)) == len(np.unique(relabeled)) == 3
    assert np.array_equal(teacher_relations(partition, rows, cols), teacher_relations(relabeled, rows, cols))


def test_study_balanced_sampler_balances_relation_classes():
    y = np.asarray([0] * 15 + [1] * 5, dtype=np.uint8)
    index = deterministic_balanced_indices(y, 10, "toy")
    assert np.sum(y[index] == 0) == 5
    assert np.sum(y[index] == 1) == 5


def test_heldout_source_exclusion_is_fail_closed():
    assert source_lanes(["a", "b", "c", "d"], "c") == ["a", "b", "d"]


def test_sparse_union_and_feature_schema_are_finite():
    rng = np.random.default_rng(3)
    n = 24
    result = build_union_edge_features(rng.normal(size=(n, 5)), rng.normal(size=(n, 7)), rng.normal(size=(n, 6)), ring_graph(n), k=4)
    assert len(result["rows"]) < n * n // 2
    assert result["features"].shape == (len(result["rows"]), len(FEATURE_NAMES))
    assert np.all(np.isfinite(result["features"]))
    assert np.all(result["rows"] < result["cols"])


def test_node_permutation_equivariance_for_edge_features():
    rng = np.random.default_rng(7)
    n = 22
    views = [rng.normal(size=(n, d)) for d in (5, 6, 7)]
    graph = ring_graph(n)
    base = build_union_edge_features(*views, graph, k=4)
    perm = rng.permutation(n)
    moved = build_union_edge_features(*(value[perm] for value in views), graph[perm][:, perm], k=4)
    base_key = base["rows"].astype(np.int64) * n + base["cols"]
    original_rows, original_cols = perm[moved["rows"]], perm[moved["cols"]]
    moved_key = np.minimum(original_rows, original_cols) * n + np.maximum(original_rows, original_cols)
    order_base, order_moved = np.argsort(base_key), np.argsort(moved_key)
    assert np.array_equal(base_key[order_base], moved_key[order_moved])
    assert np.allclose(base["features"][order_base], moved["features"][order_moved], atol=1e-6)


def test_logistic_and_mlp_update_and_checkpoint_reload(tmp_path):
    rng = np.random.default_rng(9)
    source = []
    for lane in ("a", "b", "c"):
        x = rng.normal(size=(80, 6)).astype(np.float32)
        y = (x[:, 0] + 0.3 * x[:, 1] > 0).astype(np.uint8)
        source.append({"lane": lane, "features": x, "target": y})
    config = EdgeModelConfig(sample_per_class_per_study=30, mlp_hidden=8, mlp_steps=8, mlp_batch_per_study=24)
    logistic = fit_logistic(source, config)
    mlp = fit_mlp(source, config, device="cpu")
    assert np.linalg.norm(logistic["coef"]) > 0
    assert mlp["parameter_l1_change"] > 0
    path = tmp_path / "checkpoint.pt"
    torch.save({"logistic": logistic, "mlp": mlp}, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    target = rng.normal(size=(23, 6)).astype(np.float32)
    assert np.array_equal(predict_logistic(logistic, target), predict_logistic(loaded["logistic"], target))
    assert np.array_equal(predict_mlp(mlp, target), predict_mlp(loaded["mlp"], target))


def test_spectral_partitioner_is_exact_k_and_sparse():
    n = 30
    graph = ring_graph(n)
    upper = sp.triu(graph, k=1).tocoo()
    partition = spectral_exact_k_partition(n, upper.row, upper.col, upper.data, k=3, seed=2)
    assert len(np.unique(partition)) == 3
    assert len(partition) == n


def test_feature_contract_has_no_identity_or_dataset_fields():
    forbidden = {"id", "lane", "dataset", "tissue", "cluster"}
    for name in FEATURE_NAMES:
        assert not any(token in name.lower() for token in forbidden)


def test_stage_b_capacity_formula_and_arm_semantics():
    m = 17
    features = np.zeros((m, len(FEATURE_NAMES)), dtype=np.float32)
    features[:, FEATURE_NAMES.index("registered_spatial_edge")] = np.arange(m) % 2
    features[:, FEATURE_NAMES.index("retained_mutual")] = np.arange(m) % 3 == 0
    features[:, FEATURE_NAMES.index("view1_mutual")] = np.arange(m) % 4 == 0
    features[:, FEATURE_NAMES.index("view2_mutual")] = np.arange(m) % 5 == 0
    probability = np.linspace(0.02, 0.98, m)
    prediction = {
        "logistic_probability": probability,
        "mlp_probability": probability[::-1].copy(),
        "shuffled_teacher_probability": np.roll(probability, 3),
        "arise_like_intersection_score": (
            features[:, FEATURE_NAMES.index("registered_spatial_edge")]
            * features[:, FEATURE_NAMES.index("retained_mutual")]
        ),
    }
    weights = stage_b_arm_weights(features, prediction)
    assert tuple(weights) == STAGE_B_ARMS
    assert np.allclose(weights["FULL_XBED"], confidence_abstention_weights(probability))
    assert np.array_equal(weights["RAW_EQUAL_UNION"], np.ones(m))
    assert np.array_equal(
        weights["SPATIAL_ONLY"], features[:, FEATURE_NAMES.index("registered_spatial_edge")]
    )
    assert all(np.all(np.isfinite(value)) and np.all(value >= 0) for value in weights.values())
    assert not np.array_equal(weights["FULL_XBED"], weights["RAW_EQUAL_UNION"])


def test_stage_b_partition_replay_is_exact():
    n = 34
    graph = ring_graph(n)
    upper = sp.triu(graph, k=1).tocoo()
    p = np.linspace(0.1, 0.9, len(upper.data))
    weights = confidence_abstention_weights(p)
    first = spectral_exact_k_partition(n, upper.row, upper.col, weights, k=4, seed=23)
    second = spectral_exact_k_partition(n, upper.row, upper.col, weights, k=4, seed=23)
    assert len(np.unique(first)) == 4
    assert np.array_equal(first, second)
    assert array_sha(first) == array_sha(second)
