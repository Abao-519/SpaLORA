import ast
import inspect
import textwrap
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from SpaLORA.night15b_sapr import (
    SAPRCore,
    align_partition,
    build_stability_anchors,
    module_semantics,
    sapr_loss,
    scipy_csr_to_torch,
)


def toy_graph(n=12):
    row = np.arange(n)
    col = (row + 1) % n
    return sp.csr_matrix((np.ones(n * 2), (np.r_[row, col], np.r_[col, row])), shape=(n, n))


def test_alignment_and_stability_are_deterministic():
    reference = np.repeat(np.arange(3), 4)
    candidate = np.asarray([2 if x == 0 else 0 if x == 1 else 1 for x in reference])
    assert np.array_equal(align_partition(reference, candidate, 3), reference)
    partitions = [reference, candidate, reference.copy(), candidate.copy()]
    first = build_stability_anchors(partitions, toy_graph(), 3)
    second = build_stability_anchors(partitions, toy_graph(), 3)
    assert np.array_equal(first.consensus, reference)
    assert np.array_equal(first.consensus, second.consensus)
    assert np.all(first.confidence == 1.0)
    assert np.isfinite(first.boundary).all()


def test_majority_consensus_preserves_registered_k_without_labels():
    partitions = [
        np.asarray([0, 0, 0, 0, 1, 1, 2, 2]),
        np.asarray([0, 0, 0, 0, 1, 1, 2, 2]),
        np.asarray([0, 0, 0, 0, 1, 2, 1, 2]),
        np.asarray([0, 0, 0, 0, 2, 1, 1, 2]),
        np.asarray([0, 0, 0, 0, 1, 2, 2, 1]),
    ]
    observed = build_stability_anchors(partitions, toy_graph(8), 3)
    assert set(np.unique(observed.consensus)) == {0, 1, 2}
    assert observed.cardinality_repair_count >= 0


def test_core_has_no_dataset_or_label_routing_api():
    semantics = module_semantics()
    assert semantics["ground_truth_argument_count"] == 0
    assert semantics["dataset_name_argument_count"] == 0
    assert semantics["dense_n_by_n_count"] == 0
    tree = ast.parse(textwrap.dedent(inspect.getsource(SAPRCore)))
    names = {node.id.lower() for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "dataset" not in names
    assert "labels" not in names
    assert "ground_truth" not in names


def test_real_formula_forward_loss_backward_and_strict_reload():
    torch.manual_seed(17)
    n, k = 12, 3
    graph = toy_graph(n)
    partitions = [
        np.repeat(np.arange(k), 4),
        np.tile(np.arange(k), 4),
        np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 2]),
    ]
    anchors = build_stability_anchors(partitions, graph, k)
    device = torch.device("cpu")
    sparse = scipy_csr_to_torch(graph, device)
    base = torch.randn(n, 8)
    view1 = torch.randn(n, 6)
    view2 = torch.randn(n, 5)
    consensus = torch.as_tensor(anchors.consensus, dtype=torch.long)
    confidence = torch.as_tensor(anchors.confidence)
    boundary = torch.as_tensor(anchors.boundary)
    interior = torch.as_tensor(anchors.interior_weight)
    model = SAPRCore(8, 6, 5, 8, k, hidden_dim=16, dropout=0.0)
    model.initialize_prototypes(base, consensus, interior)
    output = model(base, view1, view2, sparse, confidence, boundary)
    losses = sapr_loss(
        output, consensus, confidence, boundary, sparse,
        {"anchor": 1.0, "trust": 0.5, "cross_view": 0.2, "boundary": 0.2, "balance": 0.05},
    )
    assert output["embedding"].shape == (n, 8)
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    assert any(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    clone = SAPRCore(8, 6, 5, 8, k, hidden_dim=16, dropout=0.0)
    clone.load_state_dict(model.state_dict(), strict=True)
    observed = clone(base, view1, view2, sparse, confidence, boundary)["embedding"]
    assert torch.equal(output["embedding"], observed)


def test_minimal_contribution_control_preserves_frozen_retained_embedding():
    runner = Path(__file__).parents[1] / "scripts" / "night15b" / "night15b_train_sapr.py"
    source = runner.read_text(encoding="utf-8")
    assert '"retained": retained' in source
    assert '"RETAINED_TEACHER": payload["retained"]' in source
    assert '"RETAINED_TEACHER": payload["base"]' not in source
    assert 'n_init=1, max_iter=120' in source
