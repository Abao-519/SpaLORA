import json
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse


CONFIG = Path("configs/night11a/night11a_frozen_config.json")


@pytest.mark.parametrize("unit_id,shape", [("u000", (3484, 64)), ("u020", (9196, 128))])
def test_real_family_shapes_and_observation_contract(unit_id, shape):
    config = json.loads(CONFIG.read_text()); u = config["units"][unit_id]
    root = Path(u["source_root"])
    if not root.exists(): pytest.skip("remote real authority unavailable")
    ids = (root / "observation_ids.txt").read_text().splitlines()
    for name in ["g00_views.npz", "g04_views.npz"]:
        z = np.load(root / name)
        assert z["emb_latent_omics1"].shape == shape
        assert z["emb_latent_omics2"].shape == shape
        assert z["SpaLORA_fused"].shape == shape
        assert all(z[k].dtype == np.float32 for k in
                   ["emb_latent_omics1", "emb_latent_omics2", "SpaLORA_fused"])
    graph = sparse.load_npz(u["spatial_graph"])
    assert len(ids) == shape[0] and graph.shape == (shape[0], shape[0])


def test_frozen_config_has_no_dataset_router_or_scientific_search():
    text = CONFIG.read_text().lower()
    assert "dataset_name" not in text and "threshold_search" not in text
    config = json.loads(text)
    assert config["ridge_alpha"] == 1.0
    assert config["replicate_seeds"] == [17, 29, 43]
    assert len(config["conditions"]) == 3 and len(config["arms"]) == 4
