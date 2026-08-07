import inspect
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from SpaLORA.model_corrected import AttentionCorrected, EncoderOverallCorrected
from SpaLORA.night1_pipeline import (
    calculate_asr_scores,
    log_normalize_counts,
    moran_i_sparse,
    normalize_graph_sparse,
    percentile_rank,
    select_gene_mask,
    symmetric_knn_graph,
    weighted_gene_mse,
)
from SpaLORA.preprocess import fix_seed


def test_ground_truth_is_loaded_only_after_training_and_clustering():
    import scripts.night1_benchmark as benchmark

    source = inspect.getsource(benchmark.run_one)
    label_position = source.index("load_evaluation_labels")
    assert source.index("trainer.train()") < label_position
    assert source.index("cluster_exact") < label_position


def test_raw_counts_are_not_mutated_by_log_normalization():
    counts = sp.csr_matrix(np.array([[0, 2, 1], [5, 0, 3]], dtype=np.float32))
    before = counts.copy()
    xlog = log_normalize_counts(counts)
    assert (counts != before).nnz == 0
    assert sp.issparse(xlog)
    assert not np.allclose(xlog.toarray(), counts.toarray())


def test_inverse_abundance_rank_rewards_lower_mean_expression():
    means = np.array([0.1, 1.0, 10.0])
    abundance = 1.0 - percentile_rank(means)
    assert np.allclose(abundance, [1.0, 0.5, 0.0])


def test_sparse_moran_matches_direct_dense_formula():
    x = sp.csr_matrix(np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 0.0], [3.0, 0.0]]))
    w = sp.csr_matrix(
        np.array(
            [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]],
            dtype=float,
        )
    )
    observed = moran_i_sparse(x, w, chunk_size=1)
    expected = []
    for values in x.toarray().T:
        z = values - values.mean()
        expected.append((len(values) / w.sum()) * (z @ w.toarray() @ z) / (z @ z))
    assert np.allclose(observed, expected, atol=1e-12)


def test_reliability_shrinkage_is_n_over_n_plus_20():
    counts = sp.csr_matrix(np.array([[1, 0], [1, 0], [0, 1]], dtype=float))
    xlog = log_normalize_counts(counts)
    w = sp.csr_matrix(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float))
    table = calculate_asr_scores(counts, xlog, w, tau=20.0)
    assert np.allclose(table["reliability_r"], [2 / 22, 1 / 21])
    assert np.allclose(
        table["q_asr"],
        table["abundance_score_a"] * table["spatial_rank_s"] * table["reliability_r"],
    )


def test_weighted_loss_uses_normalized_per_gene_mse():
    diff = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    uniform = torch.ones(2)
    weighted = torch.tensor([1.0, 2.0])
    assert torch.allclose(weighted_gene_mse(diff, uniform), torch.mean(diff ** 2))
    per_gene = torch.mean(diff ** 2, dim=0)
    assert torch.allclose(weighted_gene_mse(diff, weighted), torch.sum(per_gene * weighted) / 3.0)


def test_rescue_adds_exact_top_non_hvg_genes_with_deterministic_ties():
    hvg = np.array([True, False, False, True, False])
    q = np.array([0.0, 0.9, 0.9, 0.0, 0.2])
    selected, rescued = select_gene_mask(hvg, q, "asr_rescue", 2)
    assert selected.tolist() == [True, True, True, True, False]
    assert rescued.tolist() == [False, True, True, False, False]


def test_adjacency_pipeline_stays_sparse_and_has_bounded_nnz():
    rng = np.random.RandomState(0)
    coordinates = rng.normal(size=(2000, 2))
    graph = symmetric_knn_graph(coordinates, 8)
    normalized = normalize_graph_sparse(graph)
    assert sp.isspmatrix_csr(graph)
    assert graph.nnz <= 2 * 2000 * 8
    assert normalized.is_sparse
    assert normalized._nnz() <= graph.nnz + 2000


def test_feature_graph_metric_is_configurable():
    features = np.array([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [0.0, 1.0]])
    correlation = symmetric_knn_graph(features, 1, metric="correlation")
    euclidean = symmetric_knn_graph(features, 1, metric="euclidean")
    assert (correlation != euclidean).nnz > 0


def test_attention_softmax_is_explicitly_per_observation():
    fix_seed(7)
    attention = AttentionCorrected(3, 3)
    first = torch.randn(5, 3)
    second = torch.randn(5, 3)
    _, alpha = attention(first, second)
    assert alpha.shape == (5, 2)
    assert torch.allclose(alpha.sum(dim=1), torch.ones(5), atol=1e-7)
    assert "dim=1" in inspect.getsource(AttentionCorrected.forward)


def _tiny_model_output(seed):
    fix_seed(seed)
    identity = normalize_graph_sparse(sp.csr_matrix((4, 4)))
    model = EncoderOverallCorrected(3, 2, 2, 2)
    result = model(torch.arange(12, dtype=torch.float32).reshape(4, 3), torch.ones(4, 2), identity, identity, identity, identity)
    return result["emb_latent_combined"].detach().numpy()


def test_deterministic_smoke_repeats_to_tolerance():
    assert np.allclose(_tiny_model_output(123), _tiny_model_output(123), atol=1e-7)


def test_preregistered_config_is_fixed_to_required_search_space():
    config = json.loads((Path(__file__).parents[1] / "configs" / "night1.json").read_text(encoding="utf-8"))
    assert config["seeds"] == [0, 1, 2, 3, 4]
    assert config["alpha"] == 1.0
    assert config["rescue_non_hvg"] == 1000
    assert config["datasets"]["a1"]["n_clusters"] == 10
    assert config["datasets"]["placenta"]["n_clusters"] == 10
    assert config["datasets"]["p22"]["n_clusters"] == 9
