import ast
import inspect
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from SpaLORA.night13a_runner import (
    UnifiedZeroStepAutoencoder,
    evaluate_embedding,
    protein_adapter,
    run_zero_step,
    simple_embedding,
)


def test_model_signature_is_identity_blind():
    model = UnifiedZeroStepAutoencoder([3, 4], 2)
    names = set(model.forward.__code__.co_varnames)
    assert not names & {"dataset", "tissue", "family", "path", "label"}


def test_runner_has_no_dataset_name_routing():
    source = (inspect.getsource(run_zero_step) + inspect.getsource(simple_embedding)).casefold()
    assert "dataset" not in source and "tissue" not in source
    assert "timepoint" not in source and "family" not in source


def test_protein_adapter_and_sparse_metrics_are_finite():
    rng = np.random.RandomState(0)
    raw = sp.csr_matrix(rng.poisson(2, size=(40, 7)))
    view = protein_adapter(raw, components=4)
    assert view.shape == (40, 4)
    coords = np.column_stack([np.arange(40), np.arange(40) % 5])
    result = evaluate_embedding(view, np.repeat(np.arange(4), 10), coords,
                                [f"s{i}" for i in range(40)], 4)
    assert result["dense_n_by_n_count"] == 0
    assert np.isfinite(result["absolute_ari"])


def test_no_dense_n_by_n_constructor_in_project_runner():
    source = Path("SpaLORA/night13a_runner.py").read_text(encoding="utf-8")
    forbidden = ["pairwise_distances(", "np.zeros((n, n", ".toarray()  # adjacency"]
    assert not any(token in source for token in forbidden)


def test_registered_observation_filter_is_explicit_in_loader_signature():
    from SpaLORA.night13a_runner import load_h5ad_pair
    assert "observation_ids" in inspect.signature(load_h5ad_pair).parameters


def test_label_prefix_is_only_an_explicit_io_alignment_option():
    source = Path("scripts/night13a/night13a_unified_runner.py").read_text(encoding="utf-8")
    assert '--label-id-prefix' in source
    assert 'dataset' not in source.casefold()
    assert '.notna()' in source
