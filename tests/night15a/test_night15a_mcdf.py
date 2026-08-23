import ast
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from SpaLORA.night15a_mcdf import (
    MCDFUnifiedCore,
    cluster_known_k,
    matched_control_matrix,
    mcdf_unsupervised_loss,
    sparse_weighted_sum,
)


def test_score_source_controls_have_matched_shapes_and_no_label_argument():
    rng = np.random.RandomState(7)
    views = {
        "emb_latent_omics1": rng.normal(size=(24, 6)).astype(np.float32),
        "emb_latent_omics2": rng.normal(size=(24, 5)).astype(np.float32),
        "SpaLORA_fused": rng.normal(size=(24, 7)).astype(np.float32),
    }
    coords = rng.normal(size=(24, 2))
    for control in (
        "COORDINATE_ONLY",
        "RNA_ONLY",
        "ATAC_ONLY",
        "RNA_PLUS_COORDINATES",
        "ATAC_PLUS_COORDINATES",
        "FUSED_FULL",
        "FUSED_WITHOUT_COORDINATES",
        "FUSED_WITHOUT_GRAPH_FILTER",
        "FUSED_WITHOUT_SPATIAL_REFINEMENT",
    ):
        value = matched_control_matrix(
            control, views, views, coords, "FUSED", 4, "LINEAR", 1.2
        )
        assert value.shape[0] == 24
        assert np.isfinite(value).all()
        partition = cluster_known_k(value, 3, "KMEANS", 0, 4)
        assert partition.shape == (24,)


def test_sparse_expert_mixture_stays_sparse_and_convex():
    a = sp.eye(5, format="csr")
    b = sp.diags([1.0, 1.0, 1.0, 1.0], offsets=1, shape=(5, 5), format="csr")
    result = sparse_weighted_sum({"identity": a, "neighbor": b}, {"identity": 0.4, "neighbor": 0.6})
    assert sp.isspmatrix_csr(result)
    assert result.nnz <= a.nnz + b.nnz


def test_core_module_has_no_dataset_or_label_routing():
    source = Path(__file__).resolve().parents[2] / "SpaLORA" / "night15a_mcdf.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    names = {node.id.lower() for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "dataset" not in names
    assert "labels" not in names


def test_mcdf_forward_loss_backward_for_all_mechanism_modes():
    base = {
        "backbone": "CR_BALANCED_XREC",
        "hidden_dim": 16,
        "latent_dim": 8,
        "depth": 1,
        "dropout": 0.0,
        "loss_weights": {
            "private_recon": 1.0,
            "cross_recon": 0.5,
            "alignment": 0.1,
            "graph_smooth": 0.1,
            "topology_agreement": 0.1,
            "variance": 0.1,
            "covariance": 0.01,
        },
    }
    loss_weights = {
        "mcdf_reconstruction": 0.5,
        "cross_view_consistency": 0.1,
        "variance": 0.1,
        "covariance": 0.01,
        "maximum_geometry_gate": 0.4,
        "minimum_molecular_gate": 0.6,
        "minimum_gate_entropy": 0.5,
        "coordinate_dominance": 0.2,
        "molecular_contribution": 0.2,
        "gate_entropy": 0.1,
    }
    row = torch.arange(20, dtype=torch.long)
    col = torch.roll(row, shifts=1)
    edge_index = torch.stack((torch.cat((row, col)), torch.cat((col, row))))
    edge_weight = torch.ones(edge_index.shape[1])
    x1 = torch.randn(20, 6)
    x2 = torch.randn(20, 5)
    for mode in ("IDENTITY", "FIXED_EQUAL", "GATED", "GATED_NO_GEOMETRY"):
        config = {
            "mode": mode,
            "base_config": base,
            "gate_hidden": 12,
            "initial_residual_strength": 0.6,
            "support_power": 2.0,
            "joint_power": 2.0,
        }
        model = MCDFUnifiedCore(6, 5, config)
        output = model(x1, x2, edge_index, edge_weight)
        assert output["mcdf"].shape == (20, 8)
        assert output["expert_gate"].shape == (20, 4)
        assert torch.allclose(output["expert_gate"].sum(1), torch.ones(20))
        loss, audit = mcdf_unsupervised_loss(
            model, output, x1, x2, edge_index, loss_weights
        )
        assert torch.isfinite(loss)
        loss.backward()
        assert audit["total_with_mcdf"] > 0


def test_masked_modality_candidate_is_frozen_and_bounded():
    path = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "night15a"
        / "candidate_grid.json"
    )
    grid = json.loads(path.read_text(encoding="utf-8"))
    selected = [
        row for row in grid["candidates"]
        if row["candidate_id"] == "R40_MCDF_MASKED_MODALITY"
    ]
    assert len(selected) == 1
    probability = selected[0]["modality_dropout_probability"]
    assert 0.0 < probability < 0.5
