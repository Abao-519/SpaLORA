import ast
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from SpaLORA.night13b_unified import (
    ContentAdaptiveGraphResidual, gradient_probe_loss, mean_binary_geary,
    row_stochastic, scipy_to_torch,
)


CONFIG = {
    "threshold": .66, "slope": 40.0, "max_residual": .7,
    "fine_k": 4, "broad_k": 18,
}


def ring(n):
    rows = np.arange(n)
    cols = (rows + 1) % n
    graph = sp.coo_matrix((np.ones(n * 2),
                           (np.r_[rows, cols], np.r_[cols, rows])), shape=(n, n))
    return row_stochastic(graph)


def test_identity_gate_and_finite_gradient():
    torch.manual_seed(3)
    x = torch.randn(24, 8)
    operator = scipy_to_torch(ring(24), torch.device("cpu"))
    model = ContentAdaptiveGraphResidual(CONFIG)
    output = model(x, operator, operator)
    loss = gradient_probe_loss(output, x)
    loss.backward()
    assert torch.isfinite(output["fused"]).all()
    assert torch.isfinite(model.threshold_offset.grad)
    assert float(model.threshold_offset.grad.abs()) > 0


def test_forbidden_identity_or_metric_config_fails_closed():
    for key in ("dataset", "family", "label", "ari"):
        bad = dict(CONFIG)
        bad[key] = "forbidden"
        try:
            ContentAdaptiveGraphResidual(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(key)


def test_sparse_only_and_geary_finite():
    graph = ring(20)
    assert sp.isspmatrix_csr(graph)
    assert graph.nnz < graph.shape[0] * graph.shape[1]
    value = mean_binary_geary(np.arange(20) % 3, graph)
    assert np.isfinite(value)


def test_model_source_has_no_dataset_routing_or_label_metric_import():
    path = Path(__file__).resolve().parents[2] / "SpaLORA/night13b_unified.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    identity_names = {"dataset", "dataset_name", "family", "tissue", "path"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.IfExp)):
            test_names = {item.id for item in ast.walk(node.test)
                          if isinstance(item, ast.Name)}
            assert not (identity_names & test_names)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
        elif isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
    assert "sklearn.metrics" not in imported
