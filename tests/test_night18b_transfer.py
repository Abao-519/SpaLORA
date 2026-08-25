import inspect

import numpy as np
import scipy.sparse as sp

from SpaLORA.night18a_backbone import decode_embedding, prepare_graph, sha256_array
from SpaLORA.night18b_transfer import (
    ARMS,
    TransferBackboneConfig,
    _apply_response,
    calibrate_views,
    reload_transfer_representation,
    robust_graph_roughness,
    train_transfer_backbone,
)


def toy():
    rng = np.random.default_rng(17); n = 48
    rows = np.arange(n)
    cols = np.column_stack(((rows - 1) % n, (rows + 1) % n)).reshape(-1)
    graph = sp.csr_matrix((np.ones(2 * n), (np.repeat(rows, 2), cols)), shape=(n, n))
    latent = np.sin(np.linspace(0, 4 * np.pi, n))[:, None]
    smooth = np.hstack([latent + .03 * rng.normal(size=(n, 1)) for _ in range(6)]).astype(np.float32)
    rough = (smooth + .9 * rng.normal(size=smooth.shape)).astype(np.float32)
    retained = rng.normal(size=(n, 12)).astype(np.float32)
    return rough, smooth, retained, graph


def test_rational_identity_constant_and_finite():
    rough, _, _, graph = toy()
    identity = _apply_response(rough, graph, 0.0, 0.0)
    assert np.array_equal(identity, _apply_response(rough, graph, 0.0, 0.0))
    for beta, gamma in ((.5, .15), (.1, .5)):
        output = _apply_response(rough, graph, beta, gamma)
        assert np.isfinite(output).all() and output.shape == rough.shape


def test_full_calibration_reduces_roughness_gap_and_is_data_derived():
    rough, smooth, _, graph = toy()
    left, right, diagnostics = calibrate_views(rough, smooth, graph, "FULL_RESPONSE_CALIBRATION")
    assert diagnostics["log_gap_after"] < diagnostics["log_gap_before"]
    assert robust_graph_roughness(left, graph) == diagnostics["corrected_roughness"][0]
    assert robust_graph_roughness(right, graph) == diagnostics["corrected_roughness"][1]
    assert any(float(plan["beta"]) != float(plan["gamma"]) for plan in diagnostics["plans"])


def test_controls_are_deterministic_and_registered():
    rough, smooth, _, graph = toy()
    hashes = {}
    for arm in ARMS:
        left, right, diagnostics = calibrate_views(rough, smooth, graph, arm)
        hashes[arm] = (sha256_array(left), sha256_array(right))
        replay = calibrate_views(rough, smooth, graph, arm)
        assert hashes[arm] == (sha256_array(replay[0]), sha256_array(replay[1]))
        assert diagnostics["graph_sha256"] == replay[2]["graph_sha256"]
    assert hashes["MATCHED_BACKBONE"] != hashes["FULL_RESPONSE_CALIBRATION"]
    assert hashes["FULL_RESPONSE_CALIBRATION"] != hashes["SWAPPED_RESPONSE_CONTROL"]


def test_real_train_parameter_change_strict_reload_and_exact_k():
    rough, smooth, retained, graph = toy()
    config = TransferBackboneConfig(config_id="TEST", hidden_dim=16, residual_scale=.04, steps=3)
    rep, state, diagnostics = train_transfer_backbone(
        rough, smooth, retained, graph, 3, config, "FULL_RESPONSE_CALIBRATION", 0, "cpu")
    replay = reload_transfer_representation(
        rough, smooth, retained, graph, 3, config, "FULL_RESPONSE_CALIBRATION", state, "cpu")
    assert diagnostics["parameter_changed"] and np.array_equal(rep, replay)
    graphs = [prepare_graph(graph), prepare_graph(graph), prepare_graph(graph)]
    partitions, records, selections = decode_embedding(rep, rough, smooth, graphs, 3)
    assert partitions.shape == (15, len(rough)) and all(len(np.unique(row)) == 3 for row in partitions)
    assert len(records) == 15 and len(selections) == 4


def test_seed_and_graph_order_are_effective_and_deterministic():
    rough, smooth, retained, graph = toy(); config = TransferBackboneConfig(config_id="TEST", hidden_dim=12, steps=2)
    left, _, _ = train_transfer_backbone(rough, smooth, retained, graph, 3, config, "MATCHED_BACKBONE", 0, "cpu")
    right, _, _ = train_transfer_backbone(rough, smooth, retained, graph, 3, config, "MATCHED_BACKBONE", 1, "cpu")
    assert sha256_array(left) != sha256_array(right)
    coo = graph.tocoo(); permuted = sp.coo_matrix((coo.data[::-1], (coo.row[::-1], coo.col[::-1])), shape=graph.shape).tocsr()
    a = calibrate_views(rough, smooth, graph, "FULL_RESPONSE_CALIBRATION")
    b = calibrate_views(rough, smooth, permuted, "FULL_RESPONSE_CALIBRATION")
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_core_has_no_dataset_name_branch():
    source = inspect.getsource(calibrate_views).lower() + inspect.getsource(train_transfer_backbone).lower()
    for token in ("a1", "p22", "misar", "tonsil", "hippocampus"):
        assert token not in source
