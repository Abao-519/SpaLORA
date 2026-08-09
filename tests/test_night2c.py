import csv
import hashlib
import inspect
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

from SpaLORA.model import Encoder_overall
from SpaLORA.night2c_loss_audit import compute_loss_components, legacy_bug_weight_vector
from scripts.night2c_p0c_gate import summarize_envelope, tensor_distances
from scripts.night2c_summarize import contrast_values, sign_flip_p


REPO = Path(__file__).resolve().parents[1]
CONFIG = json.loads((REPO / "configs/night2c_numerical_equivalence_factorial.json").read_text(encoding="utf-8"))
PROTECTED = Path("/root/autodl-fs/night2b_preexisting_20260809/protected_before.sha256")
PROTECTED_ROOT = Path("/root/autodl-fs/SpaLORA-night1")


def fixture_inputs():
    torch.manual_seed(7)
    features1 = torch.randn(4, 3)
    features2 = torch.randn(4, 5)
    indices = torch.arange(4).repeat(2, 1)
    adjacency = torch.sparse_coo_tensor(indices, torch.ones(4), (4, 4)).coalesce()
    return features1, features2, (adjacency,) * 4


def test_exact_same_forward_v3_loss_identity():
    features1, features2, adjacency = fixture_inputs()
    model = Encoder_overall(3, 2, 5, 2)
    result = model(features1, features2, *adjacency)
    dispatched = compute_loss_components(result, features1, features2, [1.9, 2.5, 1.5, 10.0],
                                         "locked_legacy_loss_replay")
    public_rna = torch.mean(((features1 - result["emb_recon_omics1"]) ** 2) *
                            legacy_bug_weight_vector(features1).unsqueeze(0))
    public_total = 1.9 * public_rna + 2.5 * torch.nn.functional.mse_loss(features2, result["emb_recon_omics2"])
    public_total += 1.5 * torch.nn.functional.mse_loss(result["emb_latent_omics1"], result["emb_latent_omics1_across_recon"])
    public_total += 10.0 * torch.nn.functional.mse_loss(result["emb_latent_omics2"], result["emb_latent_omics2_across_recon"])
    assert torch.equal(public_rna, dispatched["rna"])
    assert torch.equal(public_total, dispatched["total"])


def test_exact_cpu_independent_one_step_small_fixture():
    features1, features2, adjacency = fixture_inputs()
    torch.manual_seed(11)
    first = Encoder_overall(3, 2, 5, 2)
    state = {name: value.detach().clone() for name, value in first.state_dict().items()}
    second = Encoder_overall(3, 2, 5, 2); first.load_state_dict(state); second.load_state_dict(state)
    opts = [torch.optim.Adam(first.parameters(), lr=1e-4, weight_decay=0.0),
            torch.optim.Adam(second.parameters(), lr=1e-4, weight_decay=0.0)]
    outputs = [model(features1, features2, *adjacency) for model in (first, second)]
    losses = [compute_loss_components(output, features1, features2, [1.9, 2.5, 1.5, 10.0],
                                      "locked_legacy_loss_replay")["total"] for output in outputs]
    assert all(torch.equal(outputs[0][name], outputs[1][name]) for name in outputs[0])
    assert torch.equal(losses[0], losses[1])
    for opt, loss in zip(opts, losses):
        opt.zero_grad(); loss.backward(); opt.step()
    assert all(torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters()))
    for p1, p2 in zip(first.parameters(), second.parameters()):
        assert set(opts[0].state[p1]) == set(opts[1].state[p2])
        assert all(torch.equal(opts[0].state[p1][key], opts[1].state[p2][key])
                   for key in opts[0].state[p1])


def test_normalized_distance_formulas_and_float32_floor():
    raw, rel, scaled = tensor_distances(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 3.0]))
    assert raw == 1.0
    assert rel == pytest.approx(1.0 / np.sqrt(10.0))
    assert scaled == pytest.approx(1.0 / 3.0)
    eta = 32 * np.finfo(np.float32).eps
    assert eta == pytest.approx(3.814697265625e-6)


def synthetic_rows(cross_value):
    rows = []
    for block in range(1, 9):
        for left, right, kind, value in (("L1", "L2", "within_legacy", 0.0),
                                         ("G1", "G2", "within_generalized", 0.0),
                                         ("L1", "G1", "cross", cross_value),
                                         ("L1", "G2", "cross", cross_value),
                                         ("L2", "G1", "cross", cross_value),
                                         ("L2", "G2", "cross", cross_value)):
            rows.append({"dataset": "a1", "checkpoint": 0, "block": block,
                         "pair_kind": kind, "left": left, "right": right,
                         "family": "outputs", "tensor": "x", "raw_max_abs": value,
                         "relative_l2": value, "scaled_max": value})
    return rows


def test_envelope_synthetic_pass_and_deliberate_failure():
    eta = 32 * np.finfo(np.float32).eps
    assert all(row["pass"] for row in summarize_envelope(synthetic_rows(eta), eta, 2.0))
    assert all(not row["pass"] for row in summarize_envelope(synthetic_rows(3 * eta), eta, 2.0))


def test_eight_orders_are_exact_and_position_balanced():
    expected = [["L1", "L2", "G1", "G2"], ["L2", "G1", "G2", "L1"],
                ["G1", "G2", "L1", "L2"], ["G2", "L1", "L2", "G1"],
                ["G2", "G1", "L2", "L1"], ["L1", "G2", "G1", "L2"],
                ["L2", "L1", "G2", "G1"], ["G1", "L2", "L1", "G2"]]
    assert CONFIG["p0c"]["execution_orders"] == expected
    for position in range(4):
        assert Counter(order[position] for order in expected) == Counter({"L1": 2, "L2": 2, "G1": 2, "G2": 2})


def test_hash_authorization_guard_and_hard_stop_are_source_level_mandatory():
    import scripts.night2c_factorial as runner
    source = inspect.getsource(runner.load_context)
    assert "critical_hashes" in source and "environment fingerprint drift" in source
    assert "P0C has not authorized" in source


def test_p0c_code_cannot_load_ground_truth():
    import scripts.night2c_p0c_gate as gate
    source = inspect.getsource(gate)
    assert "load_evaluation_labels" not in source
    assert "ground_truth_accessed\": False" in source


def test_serialization_precedes_first_label_access():
    import scripts.night2c_factorial as runner
    source = inspect.getsource(runner.run_one)
    first_save = source.index("save_prediction_artifacts")
    cluster = source.index("cluster_exact")
    cluster_save = source.index("save_prediction_artifacts", first_save + 1)
    labels = source.index("load_evaluation_labels")
    assert first_save < cluster < cluster_save < labels


def test_exact_seed_variant_optimizer_and_dataset_restrictions():
    assert CONFIG["seeds"] == [0, 1, 2, 3, 4]
    assert CONFIG["clustering_seed"] == 2020 and CONFIG["tutorial_model_seed"] == 2022
    assert CONFIG["learning_rate"] == 1e-4 and CONFIG["weight_decay"] == 0.0
    assert len(CONFIG["variants"]) == 5 and len(CONFIG["datasets"]) == 3
    assert CONFIG["parent_commit"] == "c283449b188f510e98c2826cbb856f296367aa03"


def test_night2c_result_paths_are_isolated():
    import scripts.night2c_factorial as runner
    source = inspect.getsource(runner)
    assert "results/night2c" in source
    assert "results/night2b/raw" not in source and "results/night2/raw" not in source


def test_factorial_contrast_formulas():
    result = contrast_values({0: 1, 1: 3, 2: 4, 3: 10, 4: 12})
    assert result == {"uniform_scale_effect": 2, "legacy_shape_effect": 3,
                      "scale_at_legacy_shape": 6, "legacy_shape_at_full_scale": 7,
                      "factorial_scale_main": 4, "factorial_shape_main": 5,
                      "interaction": 4, "asr_vs_uniform_same_scale": 9,
                      "asr_vs_legacy_same_scale": 2}
    assert sign_flip_p([1, 1, 1, 1, 1]) == 2 / 32


def test_resume_rejects_partial_or_mismatched_run(tmp_path):
    import scripts.night2c_factorial as runner
    (tmp_path / "metrics.json").write_text("{}", encoding="utf-8")
    assert runner.existing_is_valid(tmp_path, {"dataset": "a1"}) is False


def test_original_protected_manifest_still_matches_309_non_gitignore_files():
    if not PROTECTED.exists() or not PROTECTED_ROOT.exists():
        pytest.skip("target protected manifest/root unavailable")
    matched = 0
    mismatches = []
    for line in PROTECTED.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(None, 1); relative = relative.strip()
        if relative == ".gitignore":
            continue  # Night-2B's already documented sole protected-file change.
        path = PROTECTED_ROOT / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            mismatches.append(relative)
        else:
            matched += 1
    assert matched == 309 and mismatches == []


def test_gate_and_conditional_result_completeness_after_execution():
    gate_path = REPO / "results/night2c/gate_status.json"
    if not gate_path.exists():
        pytest.skip("P0C has not executed")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    report = json.loads((REPO / "reports/night2c_p0c.json").read_text(encoding="utf-8"))
    assert gate["p0c_pass"] is report["p0c_pass"]
    assert report["ground_truth_accessed"] is False
    main = list((REPO / "results/night2c/raw").glob("*/*/seed_*/metrics.json"))
    tutorials = list((REPO / "results/night2c/tutorial2022").glob("*/metrics.json"))
    technical = list((REPO / "results/night2c/technical").glob("*/*/seed_*/repeat_*/metrics.json"))
    if gate["factorial_authorized"]:
        if (REPO / "reports/night2c_completion.json").exists():
            assert (len(main), len(tutorials), len(technical)) == (75, 3, 4)
    else:
        assert (len(main), len(tutorials), len(technical)) == (0, 0, 0)


def test_required_compact_outputs_after_finalization():
    if not (REPO / "reports/night2c_completion.json").exists():
        pytest.skip("finalizer has not run")
    for name in ("p0c_summary.csv", "p0c_pairwise_distances.csv"):
        assert (REPO / "results/night2c" / name).is_file()
    assert all((REPO / "results/night2c" / name).is_file() for name in (
        "per_seed_metrics.csv", "summary.csv", "paired_deltas.csv", "factorial_effects.csv",
        "loss_components.csv", "attention_summary.csv", "per_domain_f1.csv",
        "v3_replay_audit.csv", "paper_repro_audit.csv", "placenta_seed0_technical_replicates.csv"))
