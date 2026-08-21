from __future__ import annotations

import inspect

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from SpaLORA.night10a_qcrd import (
    QCRDAdapter, canonical_array_sha256, fourier_coordinates,
    frozen_quality, frozen_reference_harmonizer, qcrd_forward,
    qcrd_loss_components, row_normalize,
)


def random_views(n=192, d_view=128, d_ref=64, seed=7):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(n, d_view)).astype(np.float32),
            rng.normal(size=(n, d_view)).astype(np.float32),
            rng.normal(size=(n, d_ref)).astype(np.float32))


def test_identity_returns_authoritative_array_object_and_exact_sha():
    z1, z2, _ = random_views(d_ref=128); raw = z1.copy(); order = "a" * 64
    result = frozen_reference_harmonizer(z1, z2, raw, order, order, order)
    assert result.mode == "identity" and result.projection is None
    assert result.zf_aligned is raw
    assert canonical_array_sha256(result.zf_aligned) == canonical_array_sha256(raw)
    assert result.audit["identity_same_object"] and result.audit["identity_sha_exact"]


def test_rectangular_procrustes_shape_rank_orthogonality_geometry_and_sha():
    z1, z2, raw = random_views(); order = "b" * 64
    first = frozen_reference_harmonizer(z1, z2, raw, order, order, order)
    second = frozen_reference_harmonizer(z1, z2, raw, order, order, order)
    assert first.mode == "rectangular_expansion"
    assert first.projection.shape == (64, 128) and first.zf_aligned.shape == (192, 128)
    assert first.projection.dtype == np.float64 and first.zf_aligned.dtype == np.float32
    assert first.audit["rank"] == 64
    assert first.audit["orthogonality_max_abs"] <= 1e-10
    assert first.audit["geometry_subset_max_abs"] <= 1e-10
    assert first.audit["geometry_full_max_abs_certified_bound"] <= 1e-8
    assert canonical_array_sha256(first.projection) == canonical_array_sha256(second.projection)
    assert canonical_array_sha256(first.zf_aligned) == canonical_array_sha256(second.zf_aligned)


@pytest.mark.parametrize("mutation", ["contraction", "rank", "nan", "order", "cardinality"])
def test_harmonizer_fail_closed_negative_contract(mutation):
    z1, z2, raw = random_views(); order = "c" * 64
    if mutation == "contraction": raw = np.zeros((len(z1), 256), dtype=np.float32)
    elif mutation == "rank": raw = np.ones_like(raw)
    elif mutation == "nan": raw[0, 0] = np.nan
    elif mutation == "cardinality": raw = raw[:-1]
    orders = (order, order, "d" * 64) if mutation == "order" else (order, order, order)
    with pytest.raises((ValueError, TypeError)):
        frozen_reference_harmonizer(z1, z2, raw, *orders)


def test_no_dataset_routing_and_all_consumers_use_zf_aligned():
    import SpaLORA.night10a_qcrd as module
    assert "dataset" not in inspect.signature(module.frozen_reference_harmonizer).parameters
    forward_source = inspect.getsource(module.qcrd_forward)
    loss_source = inspect.getsource(module.qcrd_loss_components)
    assert "model(model_student, teacher, zf_aligned" in forward_source
    assert "first_c + second_c + zf_aligned" in forward_source
    assert "old_cos = torch.sum(zf_aligned[rows] * zf_aligned[cols]" in loss_source


def test_mixed_dimension_q06_full_forward_loss_backward_and_checkpoint_state():
    n = 192; z1, z2, raw = random_views(n=n); order = "e" * 64
    harmonized = frozen_reference_harmonizer(z1, z2, raw, order, order, order)
    rows = np.arange(n - 1)
    graph = sp.coo_matrix((np.ones(2 * (n - 1)),
                           (np.r_[rows, rows + 1], np.r_[rows + 1, rows])),
                          shape=(n, n)).tocsr()
    p1 = np.arange(n) % 5; p2 = (np.arange(n) + 1) % 5; pf = np.arange(n) % 7
    quality = frozen_quality(z1, z2, harmonized.zf_aligned, p1, p2, graph, 7, mnn_k=3)
    coords = fourier_coordinates(np.c_[np.arange(n), np.arange(n) % 11])
    first = torch.tensor(row_normalize(z1)); second = torch.tensor(row_normalize(z2))
    aligned = torch.tensor(harmonized.zf_aligned); coord = torch.tensor(coords)
    model = QCRDAdapter(128, coord_features=16)
    assert model.input.in_features == 400
    output = qcrd_forward(model, first, second, aligned, quality,
                          "Q06_COORDINATE_PRIOR_RESIDUAL", coord,
                          torch.zeros((n, 128), dtype=torch.bool))
    losses = qcrd_loss_components(output, first, second, aligned, quality,
                                  "Q06_COORDINATE_PRIOR_RESIDUAL", graph, pf,
                                  torch.zeros((n, 128), dtype=torch.bool))
    losses["total"].backward()
    assert output["zc"].shape == (n, 128) and torch.isfinite(output["zc"]).all()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())

