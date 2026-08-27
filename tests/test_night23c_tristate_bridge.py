import numpy as np

from SpaLORA.night23a_xbed import EdgeModelConfig, FEATURE_NAMES
from SpaLORA.night23c_tristate_bridge import (
    ARMS,
    BridgeConfig,
    apply_platt,
    build_arm_weights,
    fit_platt_study_class_balanced,
    select_source_only_thresholds,
    studywise_crossfit_mlp,
)


def test_inner_crossfit_excludes_validation_study_and_updates_model():
    rng = np.random.default_rng(3)
    source = []
    for lane, shift in (("A", 0.0), ("B", 0.2)):
        x = rng.normal(size=(80, 6)).astype(np.float32)
        y = ((x[:, 0] + x[:, 1] + shift) > 0).astype(np.uint8)
        source.append({"lane": lane, "features": x, "target": y})
    cfg = EdgeModelConfig(sample_per_class_per_study=20, mlp_hidden=4, mlp_steps=4, mlp_batch_per_study=16, mlp_learning_rate=.01)
    out = studywise_crossfit_mlp(source, cfg, shuffled=False, device="cpu")
    assert {x["lane"] for x in out} == {"A", "B"}
    assert all(x["lane"] not in x["training_lanes"] and len(x["training_lanes"]) == 1 for x in out)
    assert all(np.isfinite(x["score"]).all() for x in out)


def test_source_calibration_and_thresholds_are_source_only_and_nonempty():
    crossfit = [
        {"lane": "A", "score": np.r_[np.linspace(.02, .30, 50), np.linspace(.70, .98, 50)], "target": np.r_[np.zeros(50), np.ones(50)].astype(np.uint8)},
        {"lane": "B", "score": np.r_[np.linspace(.03, .35, 50), np.linspace(.65, .97, 50)], "target": np.r_[np.zeros(50), np.ones(50)].astype(np.uint8)},
    ]
    calibrator = fit_platt_study_class_balanced(crossfit)
    cfg = BridgeConfig(model=EdgeModelConfig(), calibration_min_purity=0.0, calibration_min_state_fraction=.01, calibration_quantile_grid_step=.05)
    thresholds = select_source_only_thresholds(crossfit, calibrator, cfg)
    assert 0 <= thresholds["lower"] < thresholds["upper"] <= 1
    assert thresholds["worst_selective_utility"] > 0
    assert thresholds["worst_purity"] >= .8
    assert np.all(np.diff(apply_platt(calibrator, np.linspace(.01, .99, 100))) > 0)


def test_full_unknown_erases_while_retained_support_control_is_distinct():
    m = 8
    features = np.zeros((m, len(FEATURE_NAMES)), dtype=np.float32)
    retained_index = FEATURE_NAMES.index("retained_mutual")
    features[:, retained_index] = np.asarray([0, 1, 1, 0, 1, 0, 1, 0])
    calibrated = np.asarray([.95, .05, .50, .55, .45, .90, .10, .51])
    within, boundary = calibrated >= .8, calibrated <= .2
    unknown = ~(within | boundary)
    primary = {"calibrated_probability": calibrated, "mlp_probability": calibrated, "logistic_probability": calibrated,
               "within": within, "boundary": boundary, "unknown": unknown}
    shuffled = {**primary}
    cfg = BridgeConfig(model=EdgeModelConfig(), relation_scale=2.0)
    weights = build_arm_weights(features, primary, shuffled, cfg)
    assert tuple(weights) == ARMS
    pos, neg, scale = weights["FULL_NESTED_CALIBRATED_TRISTATE_SIGNED"]
    assert scale == 2.0 and np.all((pos + neg)[unknown] == 0)
    support_pos = weights["UNKNOWN_RETAINED_SUPPORT_FULL"][0]
    assert np.any(support_pos[unknown] > 0)
    assert not np.array_equal(pos, support_pos)


def test_positive_negative_states_are_disjoint_for_every_arm():
    rng = np.random.default_rng(7); m = 20
    features = rng.random((m, len(FEATURE_NAMES))).astype(np.float32)
    score = np.linspace(.01, .99, m)
    within, boundary = score >= .75, score <= .25; unknown = ~(within | boundary)
    pred = {"calibrated_probability": score, "mlp_probability": score, "logistic_probability": score,
            "within": within, "boundary": boundary, "unknown": unknown}
    weights = build_arm_weights(features, pred, pred, BridgeConfig(model=EdgeModelConfig(), relation_scale=2.0))
    for positive, negative, scale in weights.values():
        assert np.all(np.isfinite(positive + negative)) and np.all(positive >= 0) and np.all(negative >= 0)
        assert not np.any((positive > 0) & (negative > 0))
        assert scale in (0.0, 2.0)
