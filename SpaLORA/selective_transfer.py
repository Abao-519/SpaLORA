"""Night-11A label-free selective cross-modal transfer pilot.

The implementation is intentionally identity blind: callers provide only an
explicit unit id, ordered observation ids, a direction, and numeric arrays.
No dataset, tissue, annotation, or scientific metric is accepted.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.special import expit


ARMS = (
    "B0_SELF_ONLY",
    "B1_ALWAYS_TRANSFER",
    "B2_UNCERTAINTY_ONLY",
    "B3_SELECTIVE_NULL",
)


def _bucket(parts: Iterable[object], modulo: int) -> int:
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % modulo


def spot_folds(unit_id: str, observation_ids: Sequence[str]) -> np.ndarray:
    return np.asarray(
        [_bucket((unit_id, x, "spot-fold"), 2) for x in observation_ids],
        dtype=np.int8,
    )


def feature_folds(unit_id: str, direction: str, dimension: int) -> np.ndarray:
    return np.asarray(
        [_bucket((unit_id, direction, j, "feature-fold"), 5) for j in range(dimension)],
        dtype=np.int8,
    )


@dataclass(frozen=True)
class RidgeFit:
    mean: np.ndarray
    scale: np.ndarray
    kept: np.ndarray
    target_mean: np.ndarray
    coefficient: np.ndarray
    zero_variance_count: int


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> RidgeFit:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    kept = np.isfinite(scale) & (scale > 0.0)
    xs = (x[:, kept] - mean[kept]) / scale[kept]
    target_mean = y.mean(axis=0)
    yc = y - target_mean
    gram = xs.T @ xs
    gram.flat[:: gram.shape[0] + 1] += float(alpha)
    coefficient = np.linalg.solve(gram, xs.T @ yc)
    return RidgeFit(mean, scale, kept, target_mean, coefficient,
                    int((~kept).sum()))


def predict_ridge(fit: RidgeFit, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    xs = (x[:, fit.kept] - fit.mean[fit.kept]) / fit.scale[fit.kept]
    return xs @ fit.coefficient + fit.target_mean


def apply_gate(self_prediction: np.ndarray, transfer_prediction: np.ndarray,
               gate: np.ndarray) -> np.ndarray:
    """Combine paths and preserve a byte-exact null boundary."""
    self_prediction = np.asarray(self_prediction)
    transfer_prediction = np.asarray(transfer_prediction)
    gate = np.asarray(gate)
    if np.count_nonzero(gate) == 0:
        return self_prediction.copy()
    if np.all(gate == 1):
        return transfer_prediction.copy()
    return self_prediction + gate[:, None] * (transfer_prediction - self_prediction)


def _mad4(values: np.ndarray) -> np.ndarray:
    med = np.median(values, axis=1)
    return np.median(np.abs(values - med[:, None]), axis=1)


def _damage_features(unit_id: str, direction: str, eval_fold: int,
                     name: str, indices: np.ndarray, fraction: float = 0.30) -> np.ndarray:
    ranked = sorted(((_bucket((unit_id, direction, eval_fold, name, int(j), "damage"),
                               2 ** 63 - 1), int(j)) for j in indices))
    count = int(math.ceil(fraction * len(indices)))
    return np.asarray([j for _, j in ranked[:count]], dtype=np.int64)


def _damage_input(x: np.ndarray, train: np.ndarray, patch: np.ndarray,
                  features: np.ndarray) -> np.ndarray:
    out = np.asarray(x, dtype=np.float64).copy()
    if len(features):
        med = np.median(out[np.ix_(train, features)], axis=0)
        out[np.ix_(patch, features)] = med[None, :]
    return out


def _inputs_for_fold(unit_id: str, direction: str, condition: str,
                     eval_feature_fold: int, train_rows: np.ndarray,
                     patch: np.ndarray, receiving_g00: np.ndarray,
                     receiving_g04: np.ndarray, auxiliary_g00: np.ndarray,
                     auxiliary_g04: np.ndarray, permutation: Optional[np.ndarray],
                     feature_fold: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                        np.ndarray, np.ndarray, np.ndarray]:
    evidence = ((eval_feature_fold + 1) % 5, (eval_feature_fold + 2) % 5)
    observable = np.flatnonzero(~np.isin(feature_fold,
                                        (eval_feature_fold,) + evidence))
    recv00 = np.asarray(receiving_g00, dtype=np.float64)
    recv04 = np.asarray(receiving_g04, dtype=np.float64)
    aux00 = np.asarray(auxiliary_g00, dtype=np.float64)
    aux04 = np.asarray(auxiliary_g04, dtype=np.float64)
    if condition == "LOCAL_TARGET_DAMAGE":
        f00 = _damage_features(unit_id, direction, eval_feature_fold, "receiving-g00",
                               np.arange(recv00.shape[1]))
        f04 = _damage_features(unit_id, direction, eval_feature_fold, "observable-g04",
                               observable)
        recv00 = _damage_input(recv00, train_rows, patch, f00)
        recv04 = _damage_input(recv04, train_rows, patch, f04)
    elif condition == "LOCAL_AUXILIARY_CONFLICT":
        if permutation is None or len(permutation) != len(patch):
            raise ValueError("registered patch permutation is required")
        aux00 = aux00.copy(); aux04 = aux04.copy()
        aux00[patch] = aux00[permutation]
        aux04[patch] = aux04[permutation]
    elif condition != "CLEAN_HOLDOUT":
        raise ValueError("unregistered condition")
    self_x = np.concatenate((recv00, recv04[:, observable]), axis=1)
    return self_x, np.concatenate((self_x, aux00), axis=1), np.concatenate((self_x, aux04), axis=1), observable, np.asarray(evidence), recv04


def run_direction(*, unit_id: str, direction: str,
                  observation_ids: Sequence[str], condition: str,
                  receiving_g00: np.ndarray, receiving_g04: np.ndarray,
                  auxiliary_g00: np.ndarray, auxiliary_g04: np.ndarray,
                  patch: np.ndarray, permutation: Optional[np.ndarray],
                  alpha: float = 1.0) -> Dict[str, object]:
    """Run the fixed two-by-five cross-fit pilot for one transfer direction."""
    n, d = receiving_g04.shape
    if any(np.asarray(a).shape[0] != n for a in
           (receiving_g00, auxiliary_g00, auxiliary_g04)):
        raise ValueError("observation contract mismatch")
    if len(observation_ids) != n:
        raise ValueError("ordered observation id mismatch")
    sf = spot_folds(unit_id, observation_ids)
    ff = feature_folds(unit_id, direction, d)
    arm_predictions = {arm: np.empty((n, d), dtype=np.float64) for arm in ARMS}
    gate_all = np.empty((n, d), dtype=np.float64)
    utility_all = np.empty((n, d), dtype=np.float64)
    uncertainty_all = np.empty((n, d), dtype=np.float64)
    zero_variance = 0
    for heldout_fold in (0, 1):
        test = np.flatnonzero(sf == heldout_fold)
        train = np.flatnonzero(sf != heldout_fold)
        for e in range(5):
            eval_idx = np.flatnonzero(ff == e)
            self_x, transfer00_x, transfer04_x, _, evidence_folds, target_input = _inputs_for_fold(
                unit_id, direction, condition, e, train, np.asarray(patch, dtype=np.int64),
                receiving_g00, receiving_g04, auxiliary_g00, auxiliary_g04,
                permutation, ff)
            evidence_indices = [np.flatnonzero(ff == int(ef)) for ef in evidence_folds]
            target_idx = np.concatenate(evidence_indices + [eval_idx])
            y = np.asarray(receiving_g04, dtype=np.float64)[:, target_idx]
            fs = fit_ridge(self_x[train], y[train], alpha)
            f0 = fit_ridge(transfer00_x[train], y[train], alpha)
            f4 = fit_ridge(transfer04_x[train], y[train], alpha)
            zero_variance += fs.zero_variance_count + f0.zero_variance_count + f4.zero_variance_count
            ps_test = predict_ridge(fs, self_x[test])
            p0_test = predict_ridge(f0, transfer00_x[test])
            p4_test = predict_ridge(f4, transfer04_x[test])
            ps_train = predict_ridge(fs, self_x[train])
            p0_train = predict_ridge(f0, transfer00_x[train])
            p4_train = predict_ridge(f4, transfer04_x[train])
            widths = [len(x) for x in evidence_indices]
            starts = np.cumsum([0] + widths)
            def evidence_stats(rows, ps, p0, p4):
                clean = np.asarray(receiving_g04, dtype=np.float64)[rows]
                vals = []
                for j, idx in enumerate(evidence_indices):
                    sl = slice(starts[j], starts[j + 1])
                    ls = np.mean((ps[:, sl] - clean[:, idx]) ** 2, axis=1)
                    vals.append(ls - np.mean((p0[:, sl] - clean[:, idx]) ** 2, axis=1))
                    vals.append(ls - np.mean((p4[:, sl] - clean[:, idx]) ** 2, axis=1))
                v = np.column_stack(vals)
                return v.mean(axis=1), _mad4(v)
            u_test, s_test = evidence_stats(test, ps_test, p0_test, p4_test)
            u_train, s_train = evidence_stats(train, ps_train, p0_train, p4_train)
            temperature = max(float(np.median(np.abs(u_train))), 1e-6)
            s_scale = max(float(np.median(s_train)), 1e-6)
            g2 = np.clip(np.exp(-s_test / s_scale), 0.0, 1.0)
            g3 = expit((u_test - s_test) / temperature)
            eval_sl = slice(starts[-1], starts[-1] + len(eval_idx))
            ps_eval = ps_test[:, eval_sl]
            pt_eval = 0.5 * (p0_test[:, eval_sl] + p4_test[:, eval_sl])
            arm_predictions["B0_SELF_ONLY"][np.ix_(test, eval_idx)] = ps_eval
            arm_predictions["B1_ALWAYS_TRANSFER"][np.ix_(test, eval_idx)] = pt_eval
            arm_predictions["B2_UNCERTAINTY_ONLY"][np.ix_(test, eval_idx)] = apply_gate(ps_eval, pt_eval, g2)
            arm_predictions["B3_SELECTIVE_NULL"][np.ix_(test, eval_idx)] = apply_gate(ps_eval, pt_eval, g3)
            gate_all[np.ix_(test, eval_idx)] = g3[:, None]
            utility_all[np.ix_(test, eval_idx)] = u_test[:, None]
            uncertainty_all[np.ix_(test, eval_idx)] = s_test[:, None]
    clean = np.asarray(receiving_g04, dtype=np.float64)
    mse = {arm: np.mean((pred - clean) ** 2, axis=1) for arm, pred in arm_predictions.items()}
    oracle = mse["B0_SELF_ONLY"] - mse["B1_ALWAYS_TRANSFER"]
    gate_spot = gate_all.mean(axis=1)
    exact_null = apply_gate(arm_predictions["B0_SELF_ONLY"],
                            arm_predictions["B1_ALWAYS_TRANSFER"],
                            np.zeros(n, dtype=np.float64))
    if exact_null.tobytes() != arm_predictions["B0_SELF_ONLY"].tobytes():
        raise AssertionError("exact-null boundary failed")
    return {
        "arm_predictions": arm_predictions,
        "mse": mse,
        "oracle_utility": oracle,
        "gate": gate_spot,
        "gate_matrix": gate_all,
        "pilot_utility": utility_all.mean(axis=1),
        "uncertainty": uncertainty_all.mean(axis=1),
        "spot_folds": sf,
        "feature_folds": ff,
        "zero_variance_removed_total": int(zero_variance),
        "exact_null_pass": True,
    }
