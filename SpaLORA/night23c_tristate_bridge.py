"""Nested source-calibrated tri-state relation bridge for Night-23C."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression

from SpaLORA.night23a_xbed import (
    EdgeModelConfig,
    FEATURE_NAMES,
    fit_logistic,
    fit_mlp,
    predict_logistic,
    predict_mlp,
    stable_seed,
)
from SpaLORA.night23b_signed_bridge import carrier_preserving_signed_partition, tri_state_from_thresholds


ARMS = (
    "CARRIER_ONLY",
    "RETAINED_ONLY_UNSIGNED",
    "LOGISTIC_DIRECT_NONNEGATIVE",
    "MLP_DIRECT_NONNEGATIVE",
    "CALIBRATED_POSITIVE_ONLY",
    "SIGNED_UNCALIBRATED",
    "UNKNOWN_RETAINED_SUPPORT_FULL",
    "FULL_NESTED_CALIBRATED_TRISTATE_SIGNED",
    "SHUFFLED_TEACHER_NEGATIVE_CONTROL",
)


@dataclass(frozen=True)
class BridgeConfig:
    model: EdgeModelConfig
    relation_scale: float = 2.0
    calibration_method: str = "STUDYWISE_MIDRANK"
    calibration_min_purity: float = 0.80
    calibration_min_state_fraction: float = 0.01
    calibration_quantile_grid_step: float = 0.02
    uncalibrated_lower: float = 0.25
    uncalibrated_upper: float = 0.75


def _shuffled_target(item: dict) -> np.ndarray:
    y = np.asarray(item["target"], dtype=np.uint8)
    rng = np.random.default_rng(stable_seed(item["lane"] + "__night23c_shuffled"))
    return y[rng.permutation(len(y))]


def _model_sources(source: list[dict], shuffled: bool) -> list[dict]:
    return [
        {"lane": item["lane"], "features": item["features"], "target": _shuffled_target(item) if shuffled else item["target"]}
        for item in source
    ]


def studywise_crossfit_mlp(source: list[dict], config: EdgeModelConfig, shuffled: bool, device: str) -> list[dict]:
    """Predict each source study using a model trained on the other source studies."""
    modeled = _model_sources(source, shuffled)
    output = []
    for validation in modeled:
        training = [item for item in modeled if item["lane"] != validation["lane"]]
        if not training or any(item["lane"] == validation["lane"] for item in training):
            raise RuntimeError("inner study-wise split leakage")
        state = fit_mlp(training, config, device=device)
        score = predict_mlp(state, validation["features"], device="cpu")
        output.append(
            {
                "lane": validation["lane"],
                "score": score,
                "target": np.asarray(validation["target"], dtype=np.uint8),
                "training_lanes": [item["lane"] for item in training],
            }
        )
    return output


def fit_platt_study_class_balanced(crossfit: list[dict]) -> dict:
    xs, ys, ws = [], [], []
    for item in crossfit:
        score = np.clip(np.asarray(item["score"], dtype=np.float64), 1e-6, 1 - 1e-6)
        target = np.asarray(item["target"], dtype=np.uint8)
        weight = np.zeros(len(target), dtype=np.float64)
        for label in (0, 1):
            index = target == label
            if not np.any(index):
                raise RuntimeError("source cross-fit relation lacks one class")
            weight[index] = 0.5 / float(index.sum())
        xs.append(logit(score).reshape(-1, 1)); ys.append(target); ws.append(weight)
    x, y, weight = np.vstack(xs), np.concatenate(ys), np.concatenate(ws)
    model = LogisticRegression(C=1000.0, solver="lbfgs", max_iter=500, random_state=2303)
    model.fit(x, y, sample_weight=weight)
    coef = float(model.coef_[0, 0])
    if not np.isfinite(coef) or coef <= 0:
        raise RuntimeError("source-only calibration is not monotonic increasing")
    return {"method": "PLATT", "coef": coef, "intercept": float(model.intercept_[0]), "study_count": len(crossfit)}


def apply_platt(state: dict, score: np.ndarray) -> np.ndarray:
    score = np.clip(np.asarray(score, dtype=np.float64), 1e-6, 1 - 1e-6)
    output = expit(float(state["coef"]) * logit(score) + float(state["intercept"]))
    if not np.all(np.isfinite(output)) or np.any((output < 0) | (output > 1)):
        raise RuntimeError("invalid calibrated probabilities")
    return output


def apply_calibrator(state: dict, score: np.ndarray) -> np.ndarray:
    if state["method"] == "STUDYWISE_MIDRANK":
        score = np.asarray(score, dtype=np.float64)
        output = (rankdata(score, method="average") - 0.5) / len(score)
        return output.astype(np.float64)
    if state["method"] == "PLATT":
        return apply_platt(state, score)
    raise RuntimeError("unknown calibration method")


def select_source_only_thresholds(crossfit: list[dict], calibrator: dict, config: BridgeConfig) -> dict:
    calibrated = [
        {"lane": item["lane"], "score": apply_calibrator(calibrator, item["score"]), "target": item["target"]}
        for item in crossfit
    ]
    pooled = np.concatenate([item["score"] for item in calibrated])
    step = float(config.calibration_quantile_grid_step)
    quantiles = np.arange(step, 1.0, step)
    grid = np.unique(np.concatenate([[0.0], np.quantile(pooled, quantiles), [1.0]])).astype(np.float64)
    candidates = []
    for lower in grid[:-1]:
        for upper in grid[1:]:
            if not lower < upper:
                continue
            per_study = []
            valid = True
            for item in calibrated:
                score, target = item["score"], np.asarray(item["target"], dtype=np.uint8)
                boundary, within = score <= lower, score >= upper
                boundary_fraction, within_fraction = float(boundary.mean()), float(within.mean())
                if boundary_fraction < config.calibration_min_state_fraction or within_fraction < config.calibration_min_state_fraction:
                    valid = False; break
                boundary_purity = float(np.mean(target[boundary] == 0))
                within_purity = float(np.mean(target[within] == 1))
                if min(boundary_purity, within_purity) < config.calibration_min_purity:
                    valid = False; break
                coverage = boundary_fraction + within_fraction
                y0, y1 = target == 0, target == 1
                correct = 0.5 * (float(np.mean(boundary[y0])) + float(np.mean(within[y1])))
                wrong = 0.5 * (float(np.mean(within[y0])) + float(np.mean(boundary[y1])))
                utility = correct - wrong
                per_study.append(
                    {"lane": item["lane"], "boundary_purity": boundary_purity, "within_purity": within_purity,
                     "boundary_fraction": boundary_fraction, "within_fraction": within_fraction, "coverage": coverage,
                     "selective_utility": utility}
                )
            if valid:
                coverages = [item["coverage"] for item in per_study]
                purities = [min(item["boundary_purity"], item["within_purity"]) for item in per_study]
                utilities = [item["selective_utility"] for item in per_study]
                candidates.append((min(utilities), float(np.mean(utilities)), min(purities), min(coverages),
                                   float(np.mean(coverages)), upper - lower, lower, upper, per_study))
    if not candidates:
        raise RuntimeError("no source-only tri-state threshold pair satisfies frozen purity/coverage contract")
    # Worst-study selective utility first; unknown has zero utility, correct confident states are rewarded,
    # and confident errors are penalized. Remaining keys prefer mean utility, purity/coverage and simplicity.
    candidates.sort(key=lambda x: (-x[0], -x[1], -x[2], -x[3], -x[4], -x[5], x[6], x[7]))
    best = candidates[0]
    return {"lower": float(best[6]), "upper": float(best[7]), "worst_selective_utility": float(best[0]),
            "mean_selective_utility": float(best[1]), "worst_purity": float(best[2]),
            "worst_coverage": float(best[3]), "mean_coverage": float(best[4]), "per_study": best[8],
            "candidate_count": len(candidates)}


def fit_outer_bridge(source: list[dict], config: BridgeConfig, device: str, shuffled: bool = False) -> dict:
    modeled = _model_sources(source, shuffled)
    crossfit = studywise_crossfit_mlp(source, config.model, shuffled=shuffled, device=device)
    if config.calibration_method == "STUDYWISE_MIDRANK":
        calibrator = {"method": "STUDYWISE_MIDRANK", "study_count": len(crossfit)}
    elif config.calibration_method == "PLATT":
        calibrator = {"method": "PLATT", **fit_platt_study_class_balanced(crossfit)}
    else:
        raise RuntimeError("unsupported calibration method")
    thresholds = select_source_only_thresholds(crossfit, calibrator, config)
    mlp = fit_mlp(modeled, config.model, device=device)
    logistic = fit_logistic(modeled, config.model, shuffled=False)
    return {
        "schema": "night23c-outer-bridge-state-v1", "source_lanes": [item["lane"] for item in modeled],
        "shuffled_teacher": bool(shuffled), "mlp": mlp, "logistic": logistic,
        "calibrator": calibrator, "thresholds": thresholds,
        "inner_crossfit": [{"lane": item["lane"], "training_lanes": item["training_lanes"],
                             "score_min": float(np.min(item["score"])), "score_max": float(np.max(item["score"]))} for item in crossfit],
    }


def predict_outer_bridge(state: dict, features: np.ndarray) -> dict:
    mlp = predict_mlp(state["mlp"], features, device="cpu")
    logistic = predict_logistic(state["logistic"], features)
    calibrated = apply_calibrator(state["calibrator"], mlp)
    positive, negative, unknown = tri_state_from_thresholds(
        calibrated, float(state["thresholds"]["lower"]), float(state["thresholds"]["upper"])
    )
    return {"mlp_probability": mlp, "logistic_probability": logistic, "calibrated_probability": calibrated,
            "within": positive, "boundary": negative, "unknown": unknown}


def build_arm_weights(features: np.ndarray, prediction: dict, shuffled_prediction: dict, config: BridgeConfig) -> dict:
    features = np.asarray(features, dtype=np.float64)
    retained = features[:, FEATURE_NAMES.index("retained_mutual")]
    m = len(features)
    zero = np.zeros(m, dtype=np.float64)
    calibrated = np.asarray(prediction["calibrated_probability"], dtype=np.float64)
    within = np.asarray(prediction["within"], dtype=bool)
    boundary = np.asarray(prediction["boundary"], dtype=bool)
    unknown = np.asarray(prediction["unknown"], dtype=bool)
    raw_mlp = np.asarray(prediction["mlp_probability"], dtype=np.float64)
    uncal_within, uncal_boundary, _ = tri_state_from_thresholds(raw_mlp, config.uncalibrated_lower, config.uncalibrated_upper)
    shuffled_cal = np.asarray(shuffled_prediction["calibrated_probability"], dtype=np.float64)
    shuffled_within = np.asarray(shuffled_prediction["within"], dtype=bool)
    shuffled_boundary = np.asarray(shuffled_prediction["boundary"], dtype=bool)
    full_positive = calibrated * within
    full_negative = (1.0 - calibrated) * boundary
    weights = {
        "CARRIER_ONLY": (zero, zero, 0.0),
        "RETAINED_ONLY_UNSIGNED": (retained, zero, config.relation_scale),
        "LOGISTIC_DIRECT_NONNEGATIVE": (np.asarray(prediction["logistic_probability"], dtype=np.float64), zero, config.relation_scale),
        "MLP_DIRECT_NONNEGATIVE": (raw_mlp, zero, config.relation_scale),
        "CALIBRATED_POSITIVE_ONLY": (full_positive, zero, config.relation_scale),
        "SIGNED_UNCALIBRATED": (raw_mlp * uncal_within, (1.0 - raw_mlp) * uncal_boundary, config.relation_scale),
        "UNKNOWN_RETAINED_SUPPORT_FULL": (full_positive + retained * unknown, full_negative, config.relation_scale),
        "FULL_NESTED_CALIBRATED_TRISTATE_SIGNED": (full_positive, full_negative, config.relation_scale),
        "SHUFFLED_TEACHER_NEGATIVE_CONTROL": (shuffled_cal * shuffled_within, (1.0 - shuffled_cal) * shuffled_boundary, config.relation_scale),
    }
    if tuple(weights) != ARMS:
        raise RuntimeError("arm ordering mismatch")
    for arm, (positive, negative, scale) in weights.items():
        if positive.shape != (m,) or negative.shape != (m,) or np.any(positive < 0) or np.any(negative < 0):
            raise RuntimeError(f"invalid arm weights: {arm}")
        if np.any((positive > 0) & (negative > 0)):
            raise RuntimeError(f"attraction/repulsion collision: {arm}")
        if arm == "FULL_NESTED_CALIBRATED_TRISTATE_SIGNED" and np.any((positive + negative)[unknown] != 0):
            raise RuntimeError("unknown edges restored by FULL")
        if scale != 0 and not np.any(positive + negative > 0):
            raise RuntimeError(f"empty relation arm: {arm}")
    return weights


def produce_partitions(carrier: np.ndarray, rows: np.ndarray, cols: np.ndarray, k: int, weights: dict) -> tuple[np.ndarray, list[dict]]:
    partitions, ledger = [], []
    for arm in ARMS:
        positive, negative, scale = weights[arm]
        partition, representation = carrier_preserving_signed_partition(
            carrier, rows, cols, positive, negative, k=k, relation_scale=float(scale)
        )
        partitions.append(partition)
        ledger.append({"candidate_id": arm, "positive_mass": float(positive.sum()), "negative_mass": float(negative.sum()),
                       "positive_nonzero": int(np.count_nonzero(positive)), "negative_nonzero": int(np.count_nonzero(negative)),
                       "relation_scale": float(scale), "representation_shape": list(representation.shape)})
    return np.stack(partitions).astype(np.int32), ledger


def bridge_config_dict(config: BridgeConfig) -> dict:
    value = asdict(config)
    return value
