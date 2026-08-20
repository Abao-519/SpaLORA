import importlib.util
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


ROOT = Path(__file__).resolve().parents[1]


def load(name):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_contingency_evaluator_matches_sklearn():
    independent = load("night8a_eval_recovery_independent.py")
    y = np.array(["a", "a", "b", "b", "c", "c", "c"])
    pred = np.array([2, 2, 0, 1, 1, 1, 0])
    ari, nmi = independent.contingency_metrics(y, pred)
    assert abs(ari - adjusted_rand_score(y, pred)) <= 1e-12
    assert abs(nmi - normalized_mutual_info_score(y, pred)) <= 1e-12


def test_independent_spatial_matches_primary():
    primary = load("night8a_eval_recovery_primary.py")
    independent = load("night8a_eval_recovery_independent.py")
    pred = np.array([0, 0, 1, 1, 2, 2])
    rows = np.array([0, 1, 1, 2, 2, 3, 3, 4, 4, 5])
    cols = np.array([1, 0, 2, 1, 3, 2, 4, 3, 5, 4])
    a = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(6, 6))
    left = primary.spatial(pred, a)
    right = independent.spatial(pred, a)
    keys = ["neighbor_agreement", "moran_i", "geary_c", "boundary_disagreement"]
    assert max(abs(left[k] - right[i]) for i, k in enumerate(keys)) <= 1e-12


def test_window_has_no_training_or_transform_entrypoint():
    text = (ROOT / "scripts/night8a_eval_recovery_window.py").read_text()
    assert "night8a_train.py" not in text
    assert "night8a_transform.py" not in text
    assert "night8a_external" not in text
    assert "CUDA_VISIBLE_DEVICES=''" in text
