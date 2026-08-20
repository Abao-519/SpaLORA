"""Static and synthetic gates that do not access the sealed MISAR labels."""
from __future__ import annotations
import ast,runpy
from pathlib import Path
import numpy as np
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score

ROOT=Path(__file__).resolve().parents[1]

def test_prelock_code_only_uses_y_metadata_not_values():
    source=(ROOT/'scripts/night8b_p0.py').read_text(encoding='utf-8')
    assert 'h["Y"].shape' in source and 'h["Y"].dtype' in source
    assert 'h["Y"][:]' not in source and "h['Y'][:]" not in source

def test_training_path_has_no_annotation_carrier_dependency():
    for rel in ('SpaLORA/night8b_pipeline.py','scripts/night8b_train.py','scripts/night8b_formal.py','scripts/night8b_lock.py'):
        source=(ROOT/rel).read_text(encoding='utf-8'); tree=ast.parse(source)
        assert 'MISAR_seq_mouse_E15_brain_ATAC_data.h5' not in source
        assert not any(isinstance(n,ast.Constant) and n.value=='Y' for n in ast.walk(tree))

def test_no_retry_and_timeout_are_literal():
    source=(ROOT/'scripts/night8b_formal.py').read_text(encoding='utf-8')
    assert 'no automatic resume/retry' in source
    assert '3600, state' in source and '1200, state' in source
    assert '"scientific_retry": 0' in source and '"fallback": 0' in source

def test_independent_contingency_primary_matches_sklearn():
    ns=runpy.run_path(str(ROOT/'scripts/night8b_independent_metrics.py'))
    truth=np.asarray(['a','a','a','b','b','c','c','c','c','d'])
    pred=np.asarray([0,0,1,1,1,2,2,3,3,3])
    assert abs(ns['ari'](truth,pred)-adjusted_rand_score(truth,pred))<=1e-12
    assert abs(ns['nmi'](truth,pred)-normalized_mutual_info_score(truth,pred))<=1e-12

def test_label_window_is_one_way_and_after_push():
    source=(ROOT/'scripts/night8b_evaluate.py').read_text(encoding='utf-8')
    assert "prelabel_push_audit.json" in source
    assert "h['Y'][:]" in source
    assert "night8b_train.py" not in source
    assert "return_to_training_transform_or_clustering':False" in source

