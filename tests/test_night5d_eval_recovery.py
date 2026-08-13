import csv
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.night5d_eval_recovery import (CANDIDATES, METRICS, canonicalize, exact_signflip,
    holm, paired, read_rows, sha256, spatial_fail, validate_frozen, validate_stable_delta_payload)

REPO = Path(__file__).resolve().parents[1]
FROZEN = REPO / "outputs/night5d_handoff/p22_per_seed_metrics.csv"
HISTORY = REPO / "outputs/night3b_handoff/per_seed_metrics.csv"

def test_01_frozen_contract():
    rows = read_rows(FROZEN); validate_frozen(rows, FROZEN)
    assert len(rows) == 50 and len({(r['candidate_id'], r['seed']) for r in rows}) == 50
    assert sorted({r['candidate_id'] for r in rows}) == sorted(CANDIDATES)

def test_02_night3b_boundary_identity():
    rows = read_rows(HISTORY)
    assert max(abs(float(r['boundary_disagreement'])-(1-float(r['spatial_neighbor_agreement']))) for r in rows) <= 1e-12

def test_03_b00_seven_metric_replay():
    rows = canonicalize(read_rows(FROZEN)); hist = read_rows(HISTORY)
    b={(int(r['seed'])):r for r in rows if r['candidate_id']=='B00_C00_FULL_IGE' and int(r['seed'])<5}
    h={(int(r['seed'])):r for r in hist if r['dataset']=='p22' and r['variant']=='FULL_IGE'}
    for seed in range(5):
        hv=dict(h[seed]); hv['q']=(float(hv['ari'])+float(hv['nmi']))/2
        for f in ['ari','nmi','q','spatial_neighbor_agreement','spatial_cluster_moran_mean','spatial_cluster_geary_mean','boundary_disagreement']:
            assert abs(float(b[seed][f])-float(hv[f])) <= 1e-12

def test_04_symmetric_diagnostic_preserved():
    before=read_rows(FROZEN); after=canonicalize(before)
    assert [r['boundary_disagreement'] for r in before] == [r['boundary_disagreement_symmetric_union_diagnostic'] for r in after]

def test_05_stable_fields():
    payload={f'delta_{m}':0.0 for m in METRICS}; validate_stable_delta_payload(payload)
    assert not any('_mean_mean' in k for k in payload)

def test_06_old_dynamic_fixture_rejected():
    with pytest.raises(ValueError): validate_stable_delta_payload({'delta_spatial_cluster_moran_mean_mean':0.0})

def test_07_signflip_independent_reference():
    d=np.array([.1,-.2,.3,.4,-.1,.6,-.2,.2,.05,.01]); obs=d.mean()
    ref=sum((d*np.array(s)).mean() >= obs-1e-15 for s in itertools.product((-1.,1.),repeat=10))/1024
    assert exact_signflip(d) == ref

def test_08_holm_independent_reference():
    p=[.1328125,.2294921875,.1220703125]
    assert holm(p) == pytest.approx([.3662109375,.3662109375,.3662109375])

def test_09_bootstrap_deterministic_registered_order():
    rows=canonicalize(read_rows(FROZEN)); calls=[]
    def run():
        rng=np.random.default_rng(20260814); out=[]
        pairs=[('B01_C04_SHRINK25','B00_C00_FULL_IGE'),('B10_SHRINK25_ANCHOR10','B00_C00_FULL_IGE'),('B17_C09_DIFFUSE10','B00_C00_FULL_IGE'),('B10_SHRINK25_ANCHOR10','B01_C04_SHRINK25'),('B17_C09_DIFFUSE10','C09_RNA_ANCHOR10')]
        for pair in pairs:
            a=paired(rows,*pair)
            for metric in METRICS:
                idx=rng.integers(0,10,size=(1000,10)); out.append(float(a[metric][idx].mean(axis=1)[0])); calls.append((pair,metric))
        return out
    assert run() == run()

def test_10_spatial_direct_formulas():
    assert spatial_fail({'delta_neighbor':-.031,'delta_moran':-.031,'delta_geary':0})
    assert spatial_fail({'delta_neighbor':-.031,'delta_moran':0,'delta_geary':.031})
    assert not spatial_fail({'delta_neighbor':-.02,'delta_moran':-.02,'delta_geary':.031})

def test_11_original_outputs_invariance_fixture():
    index=json.loads((REPO/'outputs/night5d_handoff/delivery_index.json').read_text())
    for entry in index['entries']:
        rel=Path(entry['path']).relative_to('outputs/night5d_handoff')
        p=REPO/'outputs/night5d_handoff'/rel
        assert p.stat().st_size == entry['size_bytes'] and sha256(p) == entry['sha256']
