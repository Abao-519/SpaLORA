from __future__ import annotations

import inspect

import numpy as np
import scipy.sparse as sp
import torch

from SpaLORA.night10a_qcrd import QCRDAdapter, corrected_views, fourier_coordinates, frozen_quality


def fixture(n=48, d=8):
    rng = np.random.RandomState(7)
    labels = np.repeat(np.arange(4), n // 4)
    centers = rng.normal(size=(4, d))
    a = centers[labels] + 0.05 * rng.normal(size=(n, d))
    b = centers[labels] + 0.20 * rng.normal(size=(n, d))
    fused = (a + b) / 2
    rows = np.repeat(np.arange(n), 2)
    cols = np.column_stack([(np.arange(n) - 1) % n, (np.arange(n) + 1) % n]).ravel()
    graph = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    coords = np.column_stack([np.arange(n), np.zeros(n)])
    return a.astype("float32"), b.astype("float32"), fused.astype("float32"), labels, graph, coords


def test_quality_is_frozen_and_teacher_swaps_with_evidence():
    a, b, fused, labels, graph, _ = fixture()
    q = frozen_quality(a, b, fused, labels, graph, mnn_k=5)
    swapped = frozen_quality(b, a, fused, labels, graph, mnn_k=5)
    assert q.spot_weights.flags.writeable
    assert np.allclose(q.global_weights, swapped.global_weights[::-1], atol=1e-6)
    assert np.allclose(q.spot_weights, swapped.spot_weights[:, ::-1], atol=1e-6)
    model = QCRDAdapter(a.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    assert all(not isinstance(x, torch.Tensor) or not x.requires_grad for x in (q.global_weights, q.spot_weights))
    assert optimizer.param_groups[0]["params"]


def test_teacher_stop_gradient_and_neutral_quality():
    src = inspect.getsource(corrected_views)
    assert "teacher.detach()" in src
    a, _, fused, labels, graph, _ = fixture()
    q = frozen_quality(a, a.copy(), fused, labels, graph, mnn_k=5)
    assert abs(float(q.global_weights[0]) - 0.5) < 1e-6
    model = QCRDAdapter(a.shape[1])
    with torch.no_grad():
        out = corrected_views(model, torch.tensor(a), torch.tensor(a), torch.tensor(fused), torch.tensor(a), q, "Q02_SPOT_QUALITY_BLEND")
    assert torch.isfinite(out[2]).all()
    assert float(out[3].norm(dim=1).max()) <= 0.25 + 1e-6


def test_boundary_suppresses_q05_correction():
    a, b, fused, labels, graph, _ = fixture()
    q = frozen_quality(a, b, fused, labels, graph, mnn_k=5)
    model = QCRDAdapter(a.shape[1])
    model.eval()
    args = [model, torch.tensor(a), torch.tensor(b), torch.tensor(fused), torch.tensor(fused), q]
    with torch.no_grad():
        q04 = corrected_views(*args, "Q04_SPOT_GATED_MASKED_RESIDUAL")[3].norm(dim=1).numpy()
        q05 = corrected_views(*args, "Q05_BOUNDARY_GATED_RESIDUAL")[3].norm(dim=1).numpy()
    high = q.boundary_risk >= np.quantile(q.boundary_risk, 0.75)
    assert np.all(q05 <= q04 + 1e-7)
    assert q05[high].mean() < q04[high].mean()


def test_same_module_for_families_and_no_dataset_routing():
    assert "dataset" not in inspect.signature(QCRDAdapter).parameters
    assert "dataset" not in inspect.signature(corrected_views).parameters
    assert QCRDAdapter(64).__class__ is QCRDAdapter(128).__class__


def test_sparse_mnn_and_no_dense_pairwise_call():
    import SpaLORA.night10a_qcrd as module
    src = inspect.getsource(module)
    assert "pairwise_distances(" not in src
    assert "cdist(" not in src
    a, b, fused, labels, graph, _ = fixture()
    q = frozen_quality(a, b, fused, labels, graph, mnn_k=5)
    assert q.mnn_rows.ndim == q.mnn_cols.ndim == 1
    assert len(q.mnn_rows) <= 5 * len(a)


def test_coordinate_features_and_checkpoint_round_trip():
    a, b, fused, labels, graph, coords = fixture()
    q = frozen_quality(a, b, fused, labels, graph, mnn_k=5)
    cf = fourier_coordinates(coords, 4)
    torch.manual_seed(3)
    first = QCRDAdapter(a.shape[1], coord_features=cf.shape[1])
    first.eval()
    args = (torch.tensor(a), torch.tensor(b), torch.tensor(fused), torch.tensor(fused), q, "Q06_COORDINATE_PRIOR_RESIDUAL", torch.tensor(cf))
    with torch.no_grad(): expected = corrected_views(first, *args)[2]
    second = QCRDAdapter(a.shape[1], coord_features=cf.shape[1])
    second.load_state_dict(first.state_dict()); second.eval()
    with torch.no_grad(): observed = corrected_views(second, *args)[2]
    assert torch.equal(expected, observed)
