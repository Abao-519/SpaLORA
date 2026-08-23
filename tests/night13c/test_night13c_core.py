import ast
import inspect

import numpy as np
import scipy.sparse as sp
import torch

from SpaLORA.night13b_unified import ContentAdaptiveGraphResidual, row_stochastic, scipy_to_torch
from SpaLORA.night13c_core import (
    TrainableUnifiedCore, aligned_partition_change, consensus_medoid,
    deterministic_residual_variants, seed_everything,
)


def ring_graph(n: int, width: int = 2) -> sp.csr_matrix:
    rows, cols = [], []
    for i in range(n):
        for offset in range(1, width + 1):
            rows.extend([i, i])
            cols.extend([(i - offset) % n, (i + offset) % n])
    return row_stochastic(sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)))


def test_b10_float32_semantics_match_locked_implementation():
    rng = np.random.default_rng(4)
    z = rng.normal(size=(31, 7)).astype(np.float32)
    fine, broad = ring_graph(31, 2), ring_graph(31, 5)
    observed = deterministic_residual_variants(z, fine, broad)
    config = {"threshold": .66, "slope": 40., "max_residual": .70,
              "fine_k": 4, "broad_k": 18}
    model = ContentAdaptiveGraphResidual(config)
    expected = model(torch.as_tensor(z), scipy_to_torch(fine, torch.device("cpu")),
                     scipy_to_torch(broad, torch.device("cpu")))["fused"].detach().numpy()
    assert float(np.max(np.abs(observed.b10_mixture - expected))) <= 1e-6


def test_consensus_and_partition_change_are_label_permutation_safe():
    a = np.array([0, 0, 1, 1, 2, 2])
    b = np.array([2, 2, 0, 0, 1, 1])
    c = np.array([0, 1, 0, 1, 2, 2])
    consensus, index, score = consensus_medoid([a, b, c])
    assert index in (0, 1)
    assert aligned_partition_change(a, b) == 0.0
    assert np.array_equal(consensus, [a, b][index])
    assert np.isfinite(score)


def test_model_api_has_no_dataset_or_label_identity():
    signature = inspect.signature(TrainableUnifiedCore.forward)
    assert set(signature.parameters) == {"self", "x1", "x2", "edges", "dropout_mask"}
    tree = ast.parse(inspect.getsource(TrainableUnifiedCore))
    literals = {node.value.lower() for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    assert literals.isdisjoint({"a1", "p22", "misar", "tonsil", "label",
                                "labels", "ari", "nmi"})


def test_all_three_mechanisms_have_positive_trainable_updates_and_seed_effect():
    edges = torch.tensor([[0, 1, 2, 3, 1, 2, 3, 0],
                          [1, 2, 3, 0, 0, 1, 2, 3]], dtype=torch.long)
    x1 = torch.randn(4, 3)
    x2 = torch.randn(4, 5)
    hashes = set()
    for mechanism in sorted(TrainableUnifiedCore.MECHANISMS):
        seed_everything(11)
        model = TrainableUnifiedCore(3, 5, 8, mechanism, .2)
        initial = torch.cat([p.detach().flatten() for p in model.parameters()]).clone()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        output = model(x1, x2, edges,
                       dropout_mask=(True, False) if mechanism.startswith("MODALITY") else (False, False))
        loss, _ = model.loss(output, x1, x2)
        loss.backward()
        optimizer.step()
        final = torch.cat([p.detach().flatten() for p in model.parameters()])
        assert not torch.equal(initial, final)
        hashes.add(final.numpy().tobytes())
    assert len(hashes) == 3
    seed_everything(11)
    left = TrainableUnifiedCore(3, 5, 8, "EDGE_RELIABILITY")
    seed_everything(12)
    right = TrainableUnifiedCore(3, 5, 8, "EDGE_RELIABILITY")
    assert any(not torch.equal(a, b) for a, b in zip(left.parameters(), right.parameters()))
