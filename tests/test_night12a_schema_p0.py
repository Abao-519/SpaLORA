import ast
import csv
import gzip
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from SpaLORA.night12a_schema_p0 import (
    UnifiedZeroStepAutoencoder,
    canonical_pixel_id,
    gene_score_intervals,
    inspect_csv_matrix,
    map_adt_targets,
    reconstruction_loss,
    sparse_spatial_graph,
)


ROOT = Path(__file__).resolve().parents[1]


def test_download_accession_whitelist_is_closed():
    rows = list(csv.DictReader((ROOT / "outputs/night12a_handoff/accession_file_manifest.csv").open()))
    assert len(rows) == 23
    assert all(row["authorization"] == "ALLOW" for row in rows)
    allowed = (
        "https://ftp.ncbi.nlm.nih.gov/geo/",
        "https://data.nemoarchive.org/",
        "https://ftp.ncbi.nlm.nih.gov/gene/",
        "https://ftp.ensembl.org/pub/release-79/",
    )
    assert all(row["url"].startswith(allowed) for row in rows)
    text = "\n".join(row["filename"] for row in rows).casefold()
    for forbidden in ["fastq", "gse308623_raw.tar", "gse263333", "gse213264"]:
        assert forbidden not in text


def test_pixel_rule_and_orientation_do_not_rewrite_features(tmp_path):
    matrix = tmp_path / "matrix.csv.gz"
    with gzip.open(matrix, "wt", newline="") as handle:
        handle.write(',"A-1","B-1"\n')
        handle.write('Gene-1,1,0\n')
        handle.write('Gene,0,2\n')
    audit = inspect_csv_matrix(matrix, ["A", "B"])
    assert audit["orientation"] == "features_by_rows_spots_by_columns"
    assert audit["ordered_observation_ids"] == ["A", "B"]
    assert audit["feature_ids"] == ["Gene-1", "Gene"]
    assert canonical_pixel_id("A-1") == "A"
    assert canonical_pixel_id("A-2") == "A-2"


def test_atac_interval_is_strand_aware_and_mm10_prefixed():
    registry = {"unique": {
        "A": {"gene_id": "g1", "chrom": "chr1", "start_1based": 10001,
              "end_1based": 12000, "strand": "+"},
        "B": {"gene_id": "g2", "chrom": "chr2", "start_1based": 20001,
              "end_1based": 22000, "strand": "-"},
    }}
    rows = gene_score_intervals(registry, ["B", "A"], upstream_bp=5000)
    assert [row["feature"] for row in rows] == ["A", "B"]
    assert rows[0]["start_0based"] == 5000 and rows[0]["end_0based_exclusive"] == 12000
    assert rows[1]["start_0based"] == 20000 and rows[1]["end_0based_exclusive"] == 27000
    assert all(row["chrom"].startswith("chr") for row in rows)


def test_adt_mapping_has_four_fail_closed_states():
    gene_info = {"exact_casefold": {
        "good": [{"gene_id": "1", "symbol": "Good"}],
        "many": [{"gene_id": "2", "symbol": "Many"},
                 {"gene_id": "3", "symbol": "Many2"}],
    }}
    rows = map_adt_targets(["Good", "Many", "Isotype_X", "guess-me"], gene_info)
    assert [row["status"] for row in rows] == ["unique", "ambiguous", "control", "unmapped"]
    assert rows[-1]["canonical_identifier"] == ""


def test_single_model_signature_and_no_identity_routing():
    source = (ROOT / "SpaLORA/night12a_schema_p0.py").read_text()
    tree = ast.parse(source)
    model = next(node for node in tree.body
                 if isinstance(node, ast.ClassDef) and node.name == "UnifiedZeroStepAutoencoder")
    forward = next(node for node in model.body
                   if isinstance(node, ast.FunctionDef) and node.name == "forward")
    assert [arg.arg for arg in forward.args.args] == ["self", "view1", "view2"]
    lowered = ast.get_source_segment(source, model).casefold()
    for forbidden in ["dataset", "tissue", "timepoint", "family", "path", "label"]:
        assert forbidden not in lowered


def test_sparse_graph_and_zero_step_forward_are_finite():
    coords = np.asarray([[0, 0], [1, 0], [0, 1], [1, 1], [2, 1], [1, 2], [2, 2]], dtype=float)
    graph = sparse_spatial_graph(coords, list("ABCDEFG"), k=2)
    assert sp.isspmatrix_csr(graph)
    assert graph.nnz < graph.shape[0] ** 2
    torch.manual_seed(20260822)
    model = UnifiedZeroStepAutoencoder([3, 4], 64)
    left, right = torch.ones((7, 3)), torch.ones((7, 4))
    result = model(left, right)
    assert result["fused"].shape == (7, 64)
    assert torch.isfinite(reconstruction_loss(result, left, right))
    assert all(parameter.grad is None for parameter in model.parameters())


def test_label_and_scientific_metric_symbols_are_unreachable():
    paths = [ROOT / "SpaLORA/night12a_schema_p0.py",
             ROOT / "scripts/night12a/night12a_smoke.py"]
    symbols = set()
    for path in paths:
        tree = ast.parse(path.read_text())
        symbols.update(node.id.casefold() for node in ast.walk(tree)
                       if isinstance(node, ast.Name))
        symbols.update(node.attr.casefold() for node in ast.walk(tree)
                       if isinstance(node, ast.Attribute))
    for forbidden in ["adjusted_rand_score", "normalized_mutual_info_score",
                      "adjusted_mutual_info_score", "fowlkes_mallows_score"]:
        assert forbidden not in symbols
