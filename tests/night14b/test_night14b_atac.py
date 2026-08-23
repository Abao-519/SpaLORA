import ast
import inspect

import numpy as np
import scipy.sparse as sp
import torch

import SpaLORA.night14b_atac as n14b


def base_config():
    return {
        "candidate_id": "TEST",
        "edge_mode": "TSPR",
        "base_config": {
            "candidate_id": "TEST_BASE",
            "backbone": "CR_BALANCED_XREC",
            "hidden_dim": 16,
            "latent_dim": 8,
            "depth": 1,
            "dropout": 0.0,
            "graph_k": 2,
            "steps": 2,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "gradient_clip": 5.0,
            "loss_weights": {
                "private_recon": 1.0,
                "cross_recon": 0.5,
                "alignment": 0.05,
                "graph_smooth": 0.01,
                "topology_agreement": 0.05,
                "variance": 0.01,
                "covariance": 0.001,
            },
        },
        "initial_low_strength": 0.6,
        "initial_high_strength": 0.0,
        "support_scale": 8.0,
        "support_center": 0.6,
        "conflict_scale": 8.0,
        "conflict_center": 0.2,
        "boundary_scale": 8.0,
        "boundary_center": 0.4,
    }


def toy_graph(n=12):
    row = np.repeat(np.arange(n), 2)
    col = np.column_stack(((np.arange(n) - 1) % n,
                           (np.arange(n) + 1) % n)).reshape(-1)
    graph = sp.coo_matrix((np.ones(len(row)), (row, col)), shape=(n, n))
    graph = n14b.row_stochastic(graph)
    coo = graph.tocoo()
    edge_index = torch.as_tensor(np.vstack((coo.row, coo.col)), dtype=torch.long)
    edge_weight = torch.as_tensor(coo.data, dtype=torch.float32)
    return graph, edge_index, edge_weight


def test_identity_diffusion_is_byte_exact():
    value = np.arange(48, dtype=np.float32).reshape(12, 4)
    graph, _, _ = toy_graph()
    observed = n14b.diffuse(value, graph, beta=0.0, steps=3)
    assert np.array_equal(observed, value)


def test_rank_calibrated_states_are_sparse_and_deterministic():
    rng = np.random.default_rng(7)
    z1 = rng.normal(size=(12, 5)).astype(np.float32)
    z2 = rng.normal(size=(12, 5)).astype(np.float32)
    graph, _, _ = toy_graph()
    first, audit1 = n14b.empirical_edge_states(z1, z2, graph, .55, .25, .35)
    second, audit2 = n14b.empirical_edge_states(z1, z2, graph, .55, .25, .35)
    assert sp.issparse(first)
    assert first.shape == (12, 12)
    assert (first != second).nnz == 0
    assert audit1 == audit2
    assert audit1["dense_n_by_n_count"] == 0


def test_majority_head_preserves_registered_cardinality():
    graph, _, _ = toy_graph()
    initial = np.tile(np.arange(3), 4)
    observed = n14b.anchored_majority_refine(initial, graph, 3, anchor=2.0,
                                              iterations=5)
    assert len(np.unique(observed)) == 3
    assert observed.shape == initial.shape


def test_unified_model_has_finite_nonzero_gradient():
    torch.manual_seed(3)
    _, edge_index, edge_weight = toy_graph()
    x1 = torch.randn(12, 6)
    x2 = torch.randn(12, 7)
    config = base_config()
    model = n14b.UnifiedEdgeStateModel(6, 7, config)
    output = model(x1, x2, edge_index, edge_weight)
    loss, audit = n14b.edge_state_loss(
        model, output, x1, x2, edge_index,
        {"edge_reconstruction": .5, "trusted_smoothness": .02,
         "rejected_edge_retention": .02},
    )
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters()
                 if parameter.grad is not None]
    assert torch.isfinite(loss)
    assert gradients
    assert all(torch.isfinite(value).all() for value in gradients)
    assert sum(float(value.abs().sum()) for value in gradients) > 0
    assert np.isfinite(audit["total_with_edge_state"])


def test_model_source_has_no_dataset_identity_routing():
    source = inspect.getsource(n14b.UnifiedEdgeStateModel)
    tree = ast.parse(source)
    names = {node.id.lower() for node in ast.walk(tree) if isinstance(node, ast.Name)}
    strings = {
        node.value.lower() for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    forbidden = {"dataset", "tissue", "p22", "misar", "path", "label", "ari", "nmi"}
    assert not (names & forbidden)
    assert not any(any(item in value for item in forbidden) for value in strings)


def test_cuda_allocator_initialization_order_when_gpu_available():
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda:0")
    n14b.initialize_cuda_device(device)
    assert torch.cuda.is_initialized()
