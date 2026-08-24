import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "night16c" / "night16c_new_unit_p0.py"
SPEC = importlib.util.spec_from_file_location("night16c_new_unit_p0", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_guarded_central_start_rejects_singleton_even_if_central():
    # Starts 0 and 1 share a singleton and are mutually central.  Start 2 is
    # less central but satisfies the generic minimum-cluster contract.
    singleton = np.array([0] + [1] * 9 + [2] * 10, dtype=np.int32)
    near_singleton = np.array([0] + [1] * 10 + [2] * 9, dtype=np.int32)
    guarded = np.array([0] * 5 + [1] * 5 + [2] * 10, dtype=np.int32)
    selected, _, minimum_sizes, threshold, valid = MODULE._select_guarded_central_start(
        np.stack([singleton, near_singleton, guarded]), 3
    )
    assert threshold == 5
    assert minimum_sizes == [1, 1, 5]
    assert valid.tolist() == [False, False, True]
    assert selected == 2


def test_sparse_graph_has_no_dense_pairwise_materialization():
    coords = np.stack([np.arange(20), np.zeros(20)], axis=1).astype(float)
    graph = MODULE._spatial_graph(coords, neighbours=3)
    assert graph.shape == (20, 20)
    assert graph.nnz < 20 * 20
    assert np.all(graph.diagonal() == 0)
