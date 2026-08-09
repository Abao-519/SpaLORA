"""Hard P0D/P0B-F contracts for SpaLORA Night-3A-F."""

import copy
import inspect
import json
import random
import shutil
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from SpaLORA.night1_pipeline import PreparedData, normalize_graph_sparse
from SpaLORA.night3a_ige import LOSS_KEYS, input_sha256
from SpaLORA.night3af_cache import load_cache, save_cache, sha256_file, verify_cache
from SpaLORA.night3ar_ige import weighted_gradient_influence
from SpaLORA.night3a_ige import Night3ATrainer
from SpaLORA.preprocess import pca, pca_deterministic


REPO = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO / "configs/night3af_deterministic_pca.json"
OUTPUT = REPO / "outputs/night3af_handoff"


def matrix():
    return np.arange(800, dtype=np.float32).reshape(40, 20) / 37.0


def numpy_equal(a, b):
    return a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2:] == b[2:]


def synthetic_prepared():
    rng = np.random.RandomState(4)
    n, f1, f2 = 7, 5, 4
    graph = normalize_graph_sparse(sp.eye(n, format="csr", dtype=np.float32))
    data = {
        "features_omics1": rng.normal(size=(n, f1)).astype(np.float32),
        "features_omics2": rng.normal(size=(n, f2)).astype(np.float32),
        "weight_vector_omics1": np.ones(f1, dtype=np.float32),
        "selected_gene_names": np.asarray(["g%d" % i for i in range(f1)]),
        "rna_pca_scores": rng.normal(size=(n, 3)).astype(np.float32),
        "rna_pca_explained_variance": np.arange(3, dtype=np.float64),
        "rna_pca_explained_variance_ratio": np.asarray([.5, .3, .2]),
        "rna_pca_metadata": {"svd_solver_requested": "randomized", "svd_solver_resolved": "randomized",
                             "random_state": 0, "numpy_global_rng_unchanged": True},
        "adj_spatial_omics1": graph, "adj_spatial_omics2": graph,
        "adj_feature_omics1": graph, "adj_feature_omics2": graph,
    }
    return PreparedData(data, pd.Index(["x%d" % i for i in range(n)]),
                        np.arange(n * 2).reshape(n, 2), pd.DataFrame())


def synthetic_training_data():
    return {key: value for key, value in synthetic_prepared().data.items()
            if key.startswith("features_") or key.startswith("adj_")}


def test_deterministic_pca_byte_exact_across_external_numpy_seeds():
    obj = ad.AnnData(matrix())
    np.random.seed(11); first = pca_deterministic(obj, n_comps=7, random_state=0, svd_solver="randomized")
    np.random.seed(987654); np.random.random(101); second = pca_deterministic(
        obj, n_comps=7, random_state=0, svd_solver="randomized"
    )
    assert np.array_equal(first, second) and first.tobytes() == second.tobytes()


def test_deterministic_pca_preserves_python_numpy_and_torch_rng():
    random.seed(9); np.random.seed(9); torch.manual_seed(9)
    py_before = copy.deepcopy(random.getstate()); np_before = copy.deepcopy(np.random.get_state())
    torch_before = torch.get_rng_state().clone()
    scores, metadata = pca_deterministic(ad.AnnData(matrix()), n_comps=5, return_metadata=True)
    assert scores.shape == (40, 5) and metadata["svd_solver_resolved"] == "randomized"
    assert random.getstate() == py_before and numpy_equal(np.random.get_state(), np_before)
    assert torch.equal(torch.get_rng_state(), torch_before)


def test_legacy_pca_signature_and_constructor_remain_unchanged():
    assert list(inspect.signature(pca).parameters) == ["adata", "use_reps", "n_comps"]
    source = inspect.getsource(pca)
    assert "PCA(n_components=n_comps)" in source
    assert "random_state" not in source and "svd_solver" not in source


def test_corrected_pipeline_explicitly_uses_locked_pca_parameters():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["preprocessing"]["pca_svd_solver"] == "randomized"
    assert config["preprocessing"]["pca_random_state"] == 0
    source = (REPO / "SpaLORA/night1_pipeline.py").read_text(encoding="utf-8")
    assert "pca_deterministic(" in source and 'config["pca_random_state"]' in source


def test_cache_round_trip_preserves_canonical_input_and_sparse_graphs(tmp_path):
    prepared = synthetic_prepared(); directory = tmp_path / "cache"
    manifest = save_cache(directory, "fake", prepared, {"pca_svd_solver": "randomized", "pca_random_state": 0})
    loaded = load_cache(directory, sha256_file(directory / "manifest.json"))
    observed = input_sha256(loaded.data, loaded.obs_names, loaded.data["selected_gene_names"])
    assert observed == manifest["canonical_model_input_sha256"]
    assert all(loaded.data[name].is_sparse and loaded.data[name].is_coalesced() for name in (
        "adj_spatial_omics1", "adj_spatial_omics2", "adj_feature_omics1", "adj_feature_omics2"
    ))


def test_cache_damage_partial_and_extra_files_are_rejected(tmp_path):
    original = tmp_path / "original"
    save_cache(original, "fake", synthetic_prepared(), {"pca_svd_solver": "randomized", "pca_random_state": 0})
    damaged = tmp_path / "damaged"; shutil.copytree(original, damaged)
    with (damaged / "features_omics1.npy").open("ab") as handle: handle.write(b"damage")
    with pytest.raises(RuntimeError): verify_cache(damaged)
    partial = tmp_path / "partial"; shutil.copytree(original, partial); (partial / "coordinates.npy").unlink()
    with pytest.raises(RuntimeError): verify_cache(partial)
    extra = tmp_path / "extra"; shutil.copytree(original, extra); (extra / "junk").write_text("x")
    with pytest.raises(RuntimeError): verify_cache(extra)


def test_p0d_two_independent_processes_are_three_of_three_byte_exact():
    a = json.loads((OUTPUT / "p0d_process_a.json").read_text(encoding="utf-8"))
    b = json.loads((OUTPUT / "p0d_process_b.json").read_text(encoding="utf-8"))
    comparison = json.loads((OUTPUT / "p0d_cross_process_comparison.json").read_text(encoding="utf-8"))
    assert a["external_rng_seed"] != b["external_rng_seed"]
    assert comparison["passed"] and comparison["byte_exact_dataset_count"] == 3
    assert all(all(row.values()) for row in comparison["datasets"].values())


def test_published_cache_is_verified_and_identical_for_all_variants_and_seeds():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    index = json.loads((OUTPUT / "preprocessing_cache_manifest.json").read_text(encoding="utf-8"))
    for dataset, row in index["datasets"].items():
        cache = load_cache(REPO / row["directory"], row["manifest_sha256"])
        hashes = {input_sha256(cache.data, cache.obs_names, cache.data["selected_gene_names"])
                  for _variant in config["variants"] for _seed in config["seeds"]}
        assert hashes == {row["canonical_model_input_sha256"]}


def test_published_graph_files_are_indices_values_not_dense_adjacencies():
    index = json.loads((OUTPUT / "preprocessing_cache_manifest.json").read_text(encoding="utf-8"))
    for row in index["datasets"].values():
        names = {path.name for path in (REPO / row["directory"]).iterdir()}
        assert all(name + "_indices.npy" in names and name + "_values.npy" in names for name in (
            "adj_spatial_omics1", "adj_spatial_omics2", "adj_feature_omics1", "adj_feature_omics2"
        ))
        assert not any("dense_adj" in name for name in names)


def test_runner_reads_only_cache_and_never_recomputes_preprocessing():
    runner = (REPO / "scripts/night3af_runner.py").read_text(encoding="utf-8")
    assert "load_cache(" in runner and "prepare_corrected" not in runner
    assert "pca(" not in runner and "kneighbors_graph" not in runner
    assert "SpaLORA.night1_evaluation" not in runner and "load_evaluation_labels" not in runner


def test_p0bf_uses_new_cache_as_target_not_old_probe_equivalence():
    source = (REPO / "scripts/night3af_p0b.py").read_text(encoding="utf-8")
    assert "load_cache(" in source and "old_probe_comparison" not in source
    assert "old_envelope" not in source and "initial_gradient_shares_quarter" in source


def test_evaluator_import_occurs_only_after_locked_manifest_preconditions():
    source = (REPO / "scripts/night3af_evaluate.py").read_text(encoding="utf-8")
    assert source.index("verify_preconditions(config, output)") < source.index(
        "from SpaLORA.night1_evaluation import evaluate, load_evaluation_labels"
    )


def test_weighted_gradient_diagnostic_is_state_and_rng_neutral_cpu():
    cfg = {"embedding_dim": 3, "epochs": 1, "loss_factors": [1.9, 2.5, 1.5, 10.0]}
    trainer = Night3ATrainer(synthetic_training_data(), cfg, "IGE", 2, torch.device("cpu"))
    model = trainer.new_model(); optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    result = weighted_gradient_influence(model, trainer.forward, {name: 1.0 for name in LOSS_KEYS}, 1e-12, optimizer)
    assert all(result[name] for name in (
        "parameter_state_unchanged", "grad_fields_unchanged", "rng_state_unchanged", "optimizer_state_unchanged"
    ))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_weighted_gradient_diagnostic_is_state_and_rng_neutral_gpu():
    cfg = {"embedding_dim": 3, "epochs": 1, "loss_factors": [1.9, 2.5, 1.5, 10.0]}
    trainer = Night3ATrainer(synthetic_training_data(), cfg, "IGE", 2, torch.device("cuda:0"))
    model = trainer.new_model(); optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    result = weighted_gradient_influence(model, trainer.forward, {name: 1.0 for name in LOSS_KEYS}, 1e-12, optimizer)
    assert all(result[name] for name in (
        "parameter_state_unchanged", "grad_fields_unchanged", "rng_state_unchanged", "optimizer_state_unchanged"
    ))


def test_label_firewall_outputs_show_no_semantic_access():
    firewall = json.loads((OUTPUT / "scientific_window_label_firewall.json").read_text(encoding="utf-8"))
    assert firewall["semantic_label_values_read"] is False and firewall["passed"] is True
    assert all(row["passed"] for row in firewall["stages"].values())


def test_config_source_data_order_and_cache_locks_are_exact():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    lock = json.loads((OUTPUT / "config_lock.json").read_text(encoding="utf-8"))
    assert sha256_file(CONFIG_PATH) == lock["config_sha256"]
    assert set(config["source_lock_files"]) == set(lock["source_sha256"])
    assert all(sha256_file(REPO / name) == expected for name, expected in lock["source_sha256"].items())
    assert sha256_file(OUTPUT / "preprocessing_cache_manifest.json") == lock["preprocessing_cache_manifest_sha256"]
    new_order = json.loads((REPO / config["run_order"]["manifest"]).read_text(encoding="utf-8"))
    old_order = json.loads((REPO / config["run_order"]["old_manifest"]).read_text(encoding="utf-8"))
    assert new_order == old_order and len(new_order["runs"]) == 60


def test_all_old_protection_manifests_exist_and_new_output_is_isolated():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    for name in ("night3ar", "night3a", "night2c"):
        assert Path(config["paths"]["protected_%s_manifest" % name]).is_file()
        assert str(OUTPUT) != config["paths"]["protected_%s_root" % name]


def test_no_asr_seed_solver_or_scale_search_space_expansion():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["variants"] == ["C0", "C1", "IGE", "ILN"]
    assert config["seeds"] == [0, 1, 2, 3, 4]
    assert config["preprocessing"]["pca_random_state"] == 0
    assert config["preprocessing"]["pca_svd_solver"] == "randomized"
