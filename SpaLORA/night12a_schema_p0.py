"""Night-12A clean-room schema and zero-step real-path primitives.

No function in this module accepts a dataset, tissue, timepoint, family, label,
or evaluation metric.  Upstream no-license code was inspected for semantics only;
this implementation is independent and restricted to registered feature matrices,
identifiers, coordinates, fragments and official identifier annotations.
"""
from __future__ import annotations

import bisect
import csv
import gzip
import hashlib
import itertools
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.neighbors import NearestNeighbors

from .night6c_pipeline import array_sha, run_head, sparse_sha
from .night6d_pipeline import HEADS


PIXEL_SUFFIX = re.compile(r"-1$")
FORBIDDEN_MODEL_FIELDS = {
    "dataset", "tissue", "timepoint", "family", "path", "label", "ground_truth",
    "ari", "nmi", "ami", "fmi", "q",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode("utf-8")).hexdigest()


def canonical_json_sha256(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False,
                  allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def atomic_torch(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    torch.save(value, temp)
    with temp.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temp, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temp, **arrays)
    os.replace(temp, path)


def canonical_pixel_id(value: str) -> str:
    """Frozen pure-format transform: remove exactly one terminal '-1'."""
    return PIXEL_SUFFIX.sub("", str(value).strip())


def _open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8", newline="") if path.suffix == ".gz" else path.open("r", encoding="utf-8", newline="")


def read_coordinates(path: Path) -> Dict[str, object]:
    ordered, rows, duplicate = [], {}, []
    with _open_text(Path(path)) as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) != 6:
                raise ValueError("coordinate row must have exactly six fields")
            raw = str(row[0])
            key = canonical_pixel_id(raw)
            if key in rows:
                duplicate.append(key)
                continue
            try:
                record = {
                    "raw_id": raw,
                    "in_tissue": int(row[1]),
                    "array_row": int(row[2]),
                    "array_col": int(row[3]),
                    "pixel_col": float(row[4]),
                    "pixel_row": float(row[5]),
                }
            except ValueError as exc:
                raise ValueError("coordinate numeric schema invalid") from exc
            ordered.append(key)
            rows[key] = record
    if duplicate:
        raise ValueError("coordinate identifiers collide after frozen transform")
    return {
        "ordered_ids": ordered,
        "records": rows,
        "ordered_id_sha256": text_sha256(ordered),
        "raw_shape": [len(ordered), 6],
        "dtype": "mixed(str,int,int,int,float,float)",
    }


def _numeric_row(values: Sequence[str]) -> Tuple[int, float, bool]:
    nnz = 0
    total = 0.0
    integer_like = True
    for value in values:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("non-finite matrix value")
        if number != 0.0:
            nnz += 1
        total += number
        integer_like = integer_like and abs(number - round(number)) <= 1e-9
    return nnz, total, integer_like


def inspect_csv_matrix(path: Path, coordinate_ids: Sequence[str]) -> Dict[str, object]:
    """Stream a CSV/TSV matrix and determine orientation from registered IDs."""
    delimiter = "\t" if ".tsv" in Path(path).name else ","
    coordinate_set = set(map(str, coordinate_ids))
    with _open_text(Path(path)) as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        header = next(reader)
        if len(header) < 2:
            raise ValueError("matrix header has fewer than two columns")
        first_row = next(reader)
        if len(first_row) == len(header):
            header_index_field_present = True
            header_raw = [x.strip('"') for x in header[1:]]
        elif len(first_row) == len(header) + 1:
            header_index_field_present = False
            header_raw = [x.strip('"') for x in header]
        else:
            raise ValueError("matrix header/data width contract invalid")
        header_ids = [canonical_pixel_id(x) for x in header_raw]
        header_hits = sum(x in coordinate_set for x in header_ids)
        row_count = 0
        width = len(header_raw)
        row_raw: List[str] = []
        row_ids: List[str] = []
        nnz = 0
        integer_like = True
        for row in itertools.chain([first_row], reader):
            if len(row) != width + 1:
                raise ValueError("matrix row width drift")
            row_count += 1
            row_raw.append(row[0].strip('"'))
            row_ids.append(canonical_pixel_id(row_raw[-1]))
            row_nnz, _, row_int = _numeric_row(row[1:])
            nnz += row_nnz
            integer_like = integer_like and row_int
    row_hits = sum(x in coordinate_set for x in row_ids)
    if header_hits and row_hits:
        raise ValueError("matrix orientation ambiguous against coordinate IDs")
    if header_hits == len(header_ids):
        orientation = "features_by_rows_spots_by_columns"
        observations, features = header_ids, row_raw
        obs_by_feature_shape = [len(header_ids), row_count]
    elif row_hits == len(row_ids):
        orientation = "spots_by_rows_features_by_columns"
        observations, features = row_ids, header_raw
        obs_by_feature_shape = [row_count, width]
    else:
        raise ValueError(
            f"matrix IDs do not close against coordinates: header={header_hits}/{len(header_ids)} "
            f"rows={row_hits}/{len(row_ids)}"
        )
    if len(set(observations)) != len(observations):
        raise ValueError("observation IDs collide after frozen transform")
    if len(set(features)) != len(features):
        raise ValueError("feature identifiers are not unique")
    entries = int(obs_by_feature_shape[0]) * int(obs_by_feature_shape[1])
    return {
        "source_path": str(Path(path).resolve()),
        "source_shape": [row_count, width],
        "observation_by_feature_shape": obs_by_feature_shape,
        "header_index_field_present": header_index_field_present,
        "orientation": orientation,
        "dtype": "integer_count" if integer_like else "float64",
        "sparse_on_disk": False,
        "nonzero": int(nnz),
        "density": float(nnz / entries) if entries else 0.0,
        "ordered_observation_ids": observations,
        "feature_ids": features,
        "ordered_observation_sha256": text_sha256(observations),
        "ordered_feature_sha256": text_sha256(features),
    }


def load_selected_counts(path: Path, audit: Mapping[str, object],
                         selected_features: Sequence[str]) -> Dict[str, object]:
    """Read all values for library size while retaining only registered features."""
    selected = list(selected_features)
    feature_position = {name: i for i, name in enumerate(selected)}
    if len(feature_position) != len(selected):
        raise ValueError("selected features are not unique")
    obs = list(audit["ordered_observation_ids"])
    output = np.zeros((len(obs), len(selected)), dtype=np.float32)
    library = np.zeros(len(obs), dtype=np.float64)
    delimiter = "\t" if ".tsv" in Path(path).name else ","
    with _open_text(Path(path)) as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        header = next(reader)
        if audit["orientation"] == "features_by_rows_spots_by_columns":
            header_values = (header[1:] if audit["header_index_field_present"]
                             else header)
            if [canonical_pixel_id(x.strip('"')) for x in header_values] != obs:
                raise ValueError("matrix observation order changed after preflight")
            for row in reader:
                values = np.asarray(row[1:], dtype=np.float64)
                if values.shape != (len(obs),):
                    raise ValueError("feature-row width drift")
                library += values
                name = row[0].strip('"')
                if name in feature_position:
                    output[:, feature_position[name]] = values.astype(np.float32)
        elif audit["orientation"] == "spots_by_rows_features_by_columns":
            header_values = (header[1:] if audit["header_index_field_present"] else header)
            all_features = [x.strip('"') for x in header_values]
            indices = [all_features.index(name) for name in selected]
            for row_index, row in enumerate(reader):
                if canonical_pixel_id(row[0].strip('"')) != obs[row_index]:
                    raise ValueError("matrix observation order changed after preflight")
                values = np.asarray(row[1:], dtype=np.float64)
                library[row_index] = values.sum(dtype=np.float64)
                output[row_index] = values[indices].astype(np.float32)
        else:
            raise ValueError("unsupported frozen matrix orientation")
    if np.any(library <= 0) or np.any(~np.isfinite(library)):
        raise ValueError("raw count matrix contains zero or non-finite library")
    return {"counts": output, "library_size": library, "observations": obs}


def normalize_counts(counts: np.ndarray, library_size: np.ndarray) -> np.ndarray:
    value = np.asarray(counts, dtype=np.float64)
    normalized = np.log1p(value * (10000.0 / np.asarray(library_size))[:, None])
    if not np.all(np.isfinite(normalized)):
        raise ValueError("non-finite normalized counts")
    return normalized.astype(np.float32)


def seurat_clr_counts(counts: np.ndarray) -> np.ndarray:
    value = np.asarray(counts, dtype=np.float64)
    if value.ndim != 2 or np.any(value < 0) or np.any(~np.isfinite(value)):
        raise ValueError("CLR input must be a finite non-negative matrix")
    denominator = np.exp(np.log1p(value).sum(axis=1) / float(value.shape[1]))
    normalized = np.log1p(value / denominator[:, None])
    if not np.all(np.isfinite(normalized)):
        raise ValueError("non-finite CLR output")
    return normalized


def scale_features(counts: np.ndarray) -> np.ndarray:
    value = np.asarray(counts, dtype=np.float64)
    mean = value.mean(axis=0, dtype=np.float64)
    std = value.std(axis=0, ddof=1, dtype=np.float64)
    std[std == 0] = 1.0
    scaled = (value - mean) / std
    if not np.all(np.isfinite(scaled)):
        raise ValueError("non-finite scaled feature matrix")
    return scaled.astype(np.float32)


def parse_ensembl79_genes(path: Path) -> Dict[str, object]:
    """Parse unique gene symbols from Ensembl release 79 GRCm38 GTF."""
    pattern = re.compile(r'(\w+) "([^"]+)"')
    candidates: Dict[str, List[dict]] = defaultdict(list)
    with _open_text(Path(path)) as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2] != "gene":
                continue
            attrs = dict(pattern.findall(fields[8]))
            symbol = attrs.get("gene_name")
            gene_id = attrs.get("gene_id")
            if not symbol or not gene_id:
                continue
            chrom = fields[0]
            if not chrom.startswith("chr"):
                chrom = "chrM" if chrom == "MT" else "chr" + chrom
            candidates[symbol].append({
                "gene_id": gene_id,
                "symbol": symbol,
                "chrom": chrom,
                "start_1based": int(fields[3]),
                "end_1based": int(fields[4]),
                "strand": fields[6],
                "gene_biotype": attrs.get("gene_biotype"),
            })
    unique = {symbol: rows[0] for symbol, rows in candidates.items() if len(rows) == 1}
    ambiguous = {symbol: rows for symbol, rows in candidates.items() if len(rows) != 1}
    return {"unique": unique, "ambiguous": ambiguous}


def parse_ncbi_gene_info(path: Path) -> Dict[str, object]:
    exact: Dict[str, List[dict]] = defaultdict(list)
    with _open_text(Path(path)) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            record = {"gene_id": row["GeneID"], "symbol": row["Symbol"]}
            tokens = {row["Symbol"]}
            if row.get("Synonyms") and row["Synonyms"] != "-":
                tokens.update(row["Synonyms"].split("|"))
            for token in tokens:
                exact[token.casefold()].append(record)
    return {"exact_casefold": exact}


def map_adt_targets(targets: Sequence[str], gene_info: Mapping[str, object]) -> List[dict]:
    rows = []
    lookup = gene_info["exact_casefold"]
    for raw in targets:
        raw = str(raw)
        if raw.casefold().startswith("isotype"):
            rows.append({"raw_target": raw, "canonical_identifier": "",
                         "gene_id": "", "status": "control",
                         "authority": "deposited isotype-control name"})
            continue
        matches = {(x["gene_id"], x["symbol"]) for x in lookup.get(raw.casefold(), [])}
        if len(matches) == 1:
            gene_id, symbol = next(iter(matches))
            rows.append({"raw_target": raw, "canonical_identifier": symbol,
                         "gene_id": gene_id, "status": "unique",
                         "authority": "NCBI Gene exact Symbol/Synonym token"})
        elif len(matches) > 1:
            rows.append({"raw_target": raw, "canonical_identifier": "",
                         "gene_id": "", "status": "ambiguous",
                         "authority": "NCBI Gene exact token maps to multiple GeneIDs"})
        else:
            rows.append({"raw_target": raw, "canonical_identifier": "",
                         "gene_id": "", "status": "unmapped",
                         "authority": "no exact NCBI Gene Symbol/Synonym token"})
    return rows


def gene_score_intervals(
    registry: Mapping[str, object],
    symbols: Sequence[str],
    upstream_bp: int = 5000,
) -> List[dict]:
    """Build strand-aware, zero-based half-open gene-body plus upstream intervals."""
    if upstream_bp < 0:
        raise ValueError("upstream_bp must be non-negative")
    unique = registry["unique"]
    rows: List[dict] = []
    for symbol in sorted(set(map(str, symbols))):
        if symbol not in unique:
            raise ValueError(f"symbol lacks unique Ensembl79 authority: {symbol}")
        gene = unique[symbol]
        start = int(gene["start_1based"]) - 1
        end = int(gene["end_1based"])
        strand = str(gene["strand"])
        if strand == "+":
            start = max(0, start - upstream_bp)
        elif strand == "-":
            end += upstream_bp
        else:
            raise ValueError(f"invalid gene strand for {symbol}")
        rows.append({
            "feature": symbol,
            "gene_id": gene["gene_id"],
            "chrom": gene["chrom"],
            "start_0based": start,
            "end_0based_exclusive": end,
            "strand": strand,
            "upstream_bp": upstream_bp,
        })
    return rows


def scan_fragments(
    path: Path,
    observation_ids: Sequence[str],
    intervals: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    """Stream a coordinate-sorted fragments file into sparse selected gene scores."""
    obs = list(map(str, observation_ids))
    obs_index = {value: index for index, value in enumerate(obs)}
    if len(obs_index) != len(obs):
        raise ValueError("duplicate registered observation identifiers")
    features = [str(row["feature"]) for row in intervals]
    feature_index = {value: index for index, value in enumerate(features)}
    if len(feature_index) != len(features):
        raise ValueError("duplicate registered gene-score feature")
    by_chrom: Dict[str, List[dict]] = defaultdict(list)
    for row in intervals:
        by_chrom[str(row["chrom"])].append(dict(row))
    for chrom in by_chrom:
        by_chrom[chrom].sort(
            key=lambda row: (
                int(row["start_0based"]),
                int(row["end_0based_exclusive"]),
                str(row["feature"]),
            )
        )
    starts = {
        chrom: [int(row["start_0based"]) for row in rows]
        for chrom, rows in by_chrom.items()
    }
    sweep_state: Dict[str, dict] = {}
    values: Dict[Tuple[int, int], float] = defaultdict(float)
    all_barcodes = set()
    registered_barcodes = set()
    line_count = 0
    multiplicity_sum = 0
    registered_depth = np.zeros(len(obs), dtype=np.float64)
    with _open_text(Path(path)) as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 4:
                raise ValueError("fragments row has fewer than four fields")
            chrom, start_raw, end_raw, barcode_raw = fields[:4]
            start, end = int(start_raw), int(end_raw)
            if start < 0 or end <= start:
                raise ValueError("invalid fragment coordinate")
            multiplicity = int(fields[4]) if len(fields) >= 5 else 1
            if multiplicity <= 0:
                raise ValueError("invalid fragment multiplicity")
            barcode = canonical_pixel_id(barcode_raw)
            all_barcodes.add(barcode)
            line_count += 1
            multiplicity_sum += multiplicity
            row_index = obs_index.get(barcode)
            if row_index is None:
                continue
            registered_barcodes.add(barcode)
            registered_depth[row_index] += float(multiplicity)
            chrom_rows = by_chrom.get(chrom)
            if not chrom_rows:
                continue
            midpoint = (start + end) // 2
            state = sweep_state.setdefault(
                chrom, {"cursor": 0, "active": [], "last_midpoint": -1})
            if midpoint >= state["last_midpoint"]:
                cursor = int(state["cursor"])
                while cursor < len(chrom_rows) and starts[chrom][cursor] <= midpoint:
                    state["active"].append(chrom_rows[cursor])
                    cursor += 1
                state["cursor"] = cursor
                state["active"] = [
                    row for row in state["active"]
                    if midpoint < int(row["end_0based_exclusive"])
                ]
            else:
                upper = bisect.bisect_right(starts[chrom], midpoint)
                state["active"] = [row for row in chrom_rows[:upper]
                                   if midpoint < int(row["end_0based_exclusive"])]
            state["last_midpoint"] = midpoint
            for interval in state["active"]:
                    column = feature_index[str(interval["feature"])]
                    values[(row_index, column)] += float(multiplicity)
    if values:
        keys = list(values)
        matrix = sp.coo_matrix(
            (
                np.asarray([values[key] for key in keys], dtype=np.float32),
                (
                    np.asarray([key[0] for key in keys], dtype=np.int64),
                    np.asarray([key[1] for key in keys], dtype=np.int64),
                ),
            ),
            shape=(len(obs), len(features)),
        ).tocsr()
    else:
        matrix = sp.csr_matrix((len(obs), len(features)), dtype=np.float32)
    return {
        "counts": matrix,
        "fragment_rows": line_count,
        "fragment_multiplicity_sum": multiplicity_sum,
        "all_barcode_count": len(all_barcodes),
        "all_barcode_sha256": text_sha256(sorted(all_barcodes)),
        "registered_barcode_count": len(registered_barcodes),
        "registered_barcode_sha256": text_sha256(sorted(registered_barcodes)),
        "registered_coordinate_count": len(obs),
        "registered_coordinate_sha256": text_sha256(obs),
        "registered_missing_count": len(set(obs) - registered_barcodes),
        "registered_fragment_depth": registered_depth,
    }


def normalize_gene_scores(counts: sp.spmatrix,
                          fragment_library_size: np.ndarray) -> np.ndarray:
    matrix = counts.tocsr().astype(np.float64)
    library = np.asarray(fragment_library_size, dtype=np.float64)
    if library.shape != (matrix.shape[0],):
        raise ValueError("fragment library shape mismatch")
    if np.any(library <= 0) or np.any(~np.isfinite(library)):
        raise ValueError("registered fragment library has zero or non-finite depth")
    normalized = matrix.multiply((10000.0 / library)[:, None])
    normalized.data = np.log1p(normalized.data)
    value = normalized.toarray().astype(np.float32)
    if not np.all(np.isfinite(value)):
        raise ValueError("non-finite normalized gene scores")
    return value


def sparse_spatial_graph(coordinates: np.ndarray, ids: Sequence[str],
                         k: int = 6) -> sp.csr_matrix:
    coords = np.asarray(coordinates, dtype=np.float64)
    if len(coords) <= k:
        raise ValueError("observation count must exceed spatial k")
    model = NearestNeighbors(n_neighbors=k + 1, metric="euclidean",
                             algorithm="kd_tree").fit(coords)
    distances, indices = model.kneighbors(coords)
    rows, cols = [], []
    names = np.asarray(ids, dtype=str)
    for index in range(len(coords)):
        candidates = [(float(d), str(names[j]), int(j))
                      for d, j in zip(distances[index], indices[index])
                      if int(j) != index]
        candidates.sort(key=lambda item: (item[0], item[1]))
        chosen = [item[2] for item in candidates[:k]]
        rows.extend([index] * len(chosen))
        cols.extend(chosen)
    directed = sp.coo_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)),
                             shape=(len(coords), len(coords))).tocsr()
    graph = directed.maximum(directed.T)
    graph.setdiag(0)
    graph.eliminate_zeros()
    if graph.nnz >= graph.shape[0] * graph.shape[0]:
        raise ValueError("dense N by N graph forbidden")
    return graph


class UnifiedZeroStepAutoencoder(torch.nn.Module):
    """One class/signature/formula for both registered two-modality families."""

    def __init__(self, input_dims: Sequence[int], latent_dim: int = 64):
        super().__init__()
        if len(input_dims) != 2 or any(int(x) <= 0 for x in input_dims):
            raise ValueError("exactly two positive input dimensions required")
        self.input_dims = tuple(int(x) for x in input_dims)
        self.latent_dim = int(latent_dim)
        self.encoders = torch.nn.ModuleList(
            [torch.nn.Linear(dim, self.latent_dim) for dim in self.input_dims]
        )
        self.decoders = torch.nn.ModuleList(
            [torch.nn.Linear(self.latent_dim, dim) for dim in self.input_dims]
        )

    def forward(self, view1: torch.Tensor, view2: torch.Tensor) -> Dict[str, torch.Tensor]:
        values = (view1, view2)
        if tuple(int(x.shape[1]) for x in values) != self.input_dims:
            raise ValueError("input width differs from frozen projection dimensions")
        private = [torch.tanh(layer(value)) for layer, value in zip(self.encoders, values)]
        reconstructed = [layer(value) for layer, value in zip(self.decoders, private)]
        fused = (private[0] + private[1]) * 0.5
        return {
            "private1": private[0], "private2": private[1], "fused": fused,
            "reconstruction1": reconstructed[0], "reconstruction2": reconstructed[1],
        }


def reconstruction_loss(result: Mapping[str, torch.Tensor],
                        view1: torch.Tensor, view2: torch.Tensor) -> torch.Tensor:
    loss = torch.nn.functional.mse_loss(result["reconstruction1"], view1)
    loss = loss + torch.nn.functional.mse_loss(result["reconstruction2"], view2)
    if not torch.isfinite(loss):
        raise ValueError("non-finite reconstruction loss")
    return loss


def h05_endpoint(private1: np.ndarray, private2: np.ndarray, fused: np.ndarray,
                 coordinates: np.ndarray, ids: Sequence[str], k: int,
                 artifact_dir: Path | None = None) -> Tuple[np.ndarray, dict]:
    views = {
        "emb_latent_omics1": np.asarray(private1),
        "emb_latent_omics2": np.asarray(private2),
        "SpaLORA_fused": np.asarray(fused),
    }
    labels, aux = run_head(
        HEADS["H05_EQUAL3_AFFINITY_SPECTRAL"], views, int(k),
        np.asarray(coordinates), list(ids), artifact_dir,
    )
    return canonical_partition(labels), aux


def canonical_partition(labels: Sequence[object]) -> np.ndarray:
    mapping: Dict[object, int] = {}
    result = np.empty(len(labels), dtype=np.int64)
    for index, label in enumerate(labels):
        if label not in mapping:
            mapping[label] = len(mapping)
        result[index] = mapping[label]
    return result


def fixed_linked_features(left: Iterable[str], right: Iterable[str],
                          cap: int = 256) -> List[str]:
    linked = sorted(set(map(str, left)) & set(map(str, right)))
    return linked[:int(cap)]


def assert_model_identity_blind(model: UnifiedZeroStepAutoencoder) -> None:
    names = {name.casefold() for name, _ in model.named_modules()}
    if names & FORBIDDEN_MODEL_FIELDS:
        raise ValueError("model contains forbidden identity-bearing module name")

