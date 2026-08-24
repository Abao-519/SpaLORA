#!/usr/bin/env python3
"""Extract deterministic morphology views without reading reference labels.

The manifest supplies paths and coordinate transforms.  Dataset identifiers are
used only by this I/O layer; the downstream optional-view energy receives only
numeric view arrays and a presence mask.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import time
from typing import Iterable

import anndata as ad
import numpy as np
from PIL import Image


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_id_sha256(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_gzip_image(path: Path) -> tuple[np.ndarray, str]:
    with gzip.open(path, "rb") as handle:
        decoded = handle.read()
    image = np.asarray(Image.open(io.BytesIO(decoded)).convert("RGB"), dtype=np.uint8)
    return image, hashlib.sha256(decoded).hexdigest()


def load_geo_unit(record: dict) -> tuple[list[str], np.ndarray, np.ndarray, dict]:
    adata = ad.read_h5ad(record["h5ad_path"], backed="r")
    ordered_ids = list(map(str, adata.obs_names))
    prefix = str(record["strip_exact_prefix"])
    if not all(value.startswith(prefix) for value in ordered_ids):
        raise ValueError("registered exact prefix is not present on every observation")
    deposited_ids = [value[len(prefix) :] for value in ordered_ids]
    with gzip.open(record["positions_gz_path"], "rt", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or set(rows[0]) != {
        "barcode",
        "in_tissue",
        "array_row",
        "array_col",
        "pxl_row_in_fullres",
        "pxl_col_in_fullres",
    }:
        raise ValueError("unexpected GEO tissue-position schema")
    by_id = {row["barcode"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("duplicate deposited barcodes")
    missing = sorted(set(deposited_ids) - set(by_id))
    if missing:
        raise ValueError(f"{len(missing)} h5ad observations are absent from positions")
    aligned = [by_id[value] for value in deposited_ids]
    if any(int(row["in_tissue"]) != 1 for row in aligned):
        raise ValueError("registered h5ad includes a non-tissue GEO position")
    scale = float(record["hires_scalef"])
    xy = np.asarray(
        [
            [float(row["pxl_col_in_fullres"]) * scale, float(row["pxl_row_in_fullres"]) * scale]
            for row in aligned
        ],
        dtype=np.float32,
    )
    image, decoded_sha = load_gzip_image(Path(record["image_gz_path"]))
    audit = {
        "coordinate_source": "GEO_tissue_positions_fullres_pixel_columns",
        "id_transform": f"strip_exact_prefix:{prefix}",
        "missing_after_transform": 0,
        "extra_deposited_positions": len(set(by_id) - set(deposited_ids)),
        "hires_scalef": scale,
        "hires_scale_authority": record["hires_scale_authority"],
        "compressed_image_sha256": sha256_file(Path(record["image_gz_path"])),
        "decoded_image_sha256": decoded_sha,
    }
    adata.file.close()
    return ordered_ids, xy, image, audit


def load_embedded_unit(record: dict) -> tuple[list[str], np.ndarray, np.ndarray, dict]:
    adata = ad.read_h5ad(record["h5ad_path"], backed="r")
    ordered_ids = list(map(str, adata.obs_names))
    library_id = str(record["library_id"])
    spatial = adata.uns["spatial"][library_id]
    image = np.asarray(spatial["images"]["hires"])
    if image.dtype.kind == "f":
        image = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        image = np.asarray(image, dtype=np.uint8)
    scale = float(spatial["scalefactors"]["tissue_hires_scalef"])
    coordinates = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    xy = coordinates * scale
    decoded_sha = hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
    audit = {
        "coordinate_source": "h5ad_obsm_spatial_xy",
        "id_transform": "byte_exact_identity",
        "missing_after_transform": 0,
        "extra_deposited_positions": 0,
        "hires_scalef": scale,
        "hires_scale_authority": "h5ad_uns_spatial_scalefactors_tissue_hires_scalef",
        "compressed_image_sha256": None,
        "decoded_image_sha256": decoded_sha,
    }
    adata.file.close()
    return ordered_ids, xy, image, audit


def crop_rgb(image: np.ndarray, x: float, y: float, radius: int) -> np.ndarray:
    # A deposited Visium centre can legitimately lie on the edge of a cropped
    # hires image.  Keep every crop inside the deposited image by shifting the
    # window inward; never synthesize reflection/constant pixels.
    cx = int(np.clip(np.rint(x), radius, image.shape[1] - radius))
    cy = int(np.clip(np.rint(y), radius, image.shape[0] - radius))
    x0, x1 = cx - radius, cx + radius
    y0, y1 = cy - radius, cy + radius
    if x0 < 0 or y0 < 0 or x1 > image.shape[1] or y1 > image.shape[0]:
        raise AssertionError("boundary-clamped patch still crosses the image")
    patch = image[y0:y1, x0:x1]
    if patch.shape != (2 * radius, 2 * radius, 3):
        raise AssertionError("unexpected patch shape")
    return patch


def handcrafted_one(patch: np.ndarray) -> np.ndarray:
    value = patch.astype(np.float32) / 255.0
    flat = value.reshape(-1, 3)
    rgb = np.concatenate(
        [
            flat.mean(axis=0),
            flat.std(axis=0),
            np.quantile(flat, [0.1, 0.5, 0.9], axis=0).reshape(-1),
        ]
    )
    pil_hsv = np.asarray(Image.fromarray(patch).convert("HSV"), dtype=np.float32) / 255.0
    hsv_flat = pil_hsv.reshape(-1, 3)
    hsv = np.concatenate([hsv_flat.mean(axis=0), hsv_flat.std(axis=0)])
    gray = np.asarray(Image.fromarray(patch).convert("L"), dtype=np.float32) / 255.0
    gy, gx = np.gradient(gray)
    magnitude = np.sqrt(gx * gx + gy * gy)
    laplacian = (
        -4.0 * gray
        + np.roll(gray, 1, axis=0)
        + np.roll(gray, -1, axis=0)
        + np.roll(gray, 1, axis=1)
        + np.roll(gray, -1, axis=1)
    )
    hist = np.histogram(gray, bins=8, range=(0.0, 1.0), density=False)[0].astype(np.float32)
    hist /= max(float(hist.sum()), 1.0)
    entropy = -float(np.sum(hist * np.log(np.maximum(hist, 1e-8))))
    texture = np.asarray(
        [
            gray.mean(),
            gray.std(),
            *np.quantile(gray, [0.1, 0.5, 0.9]),
            magnitude.mean(),
            magnitude.std(),
            np.quantile(magnitude, 0.9),
            laplacian.var(),
            entropy,
        ],
        dtype=np.float32,
    )
    thumbnail = np.asarray(
        Image.fromarray(patch).convert("L").resize((4, 4), Image.Resampling.BILINEAR),
        dtype=np.float32,
    ).reshape(-1) / 255.0
    return np.concatenate([rgb, hsv, texture, hist, thumbnail]).astype(np.float32)


def extract_handcrafted(image: np.ndarray, xy: np.ndarray, radii: list[int]) -> np.ndarray:
    rows = []
    for x, y in xy:
        rows.append(np.concatenate([handcrafted_one(crop_rgb(image, x, y, radius)) for radius in radii]))
    return np.asarray(rows, dtype=np.float32)


def extract_resnet(
    image: np.ndarray,
    xy: np.ndarray,
    radii: list[int],
    batch_size: int,
) -> tuple[np.ndarray, dict]:
    import torch
    from torch import nn
    from torchvision.models import ResNet18_Weights, resnet18

    torch.manual_seed(20260824)
    torch.cuda.manual_seed_all(20260824)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = nn.Identity()
    model.eval().cuda()
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device="cuda")[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device="cuda")[None, :, None, None]
    scale_outputs = []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for radius in radii:
        outputs = []
        for begin in range(0, len(xy), batch_size):
            patches = []
            for x, y in xy[begin : begin + batch_size]:
                patch = crop_rgb(image, x, y, radius)
                patch = np.asarray(
                    Image.fromarray(patch).resize((224, 224), Image.Resampling.BILINEAR),
                    dtype=np.float32,
                )
                patches.append(np.transpose(patch / 255.0, (2, 0, 1)))
            tensor = torch.from_numpy(np.asarray(patches, dtype=np.float32)).cuda(non_blocking=False)
            tensor = (tensor - mean) / std
            with torch.inference_mode():
                outputs.append(model(tensor).cpu().numpy().astype(np.float32))
        scale_outputs.append(np.concatenate(outputs, axis=0))
    torch.cuda.synchronize()
    output = np.concatenate(scale_outputs, axis=1).astype(np.float32)
    cache = Path(torch.hub.get_dir()) / "checkpoints" / "resnet18-f37072fd.pth"
    resource_record = {
        "wall_seconds": time.perf_counter() - started,
        "peak_gpu_mib": torch.cuda.max_memory_allocated() / (1024.0 * 1024.0),
        "weight_path": str(cache),
        "weight_sha256": sha256_file(cache),
        "weight_enum": "torchvision.models.ResNet18_Weights.IMAGENET1K_V1",
        "source_license": "torchvision BSD-3-Clause",
    }
    return output, resource_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output_root.mkdir(parents=True, exist_ok=True)
    run_records = []
    for record in manifest["units"]:
        started = time.perf_counter()
        if record["image_source"] == "geo_gzip":
            ordered_ids, xy, image, audit = load_geo_unit(record)
        elif record["image_source"] == "h5ad_embedded":
            ordered_ids, xy, image, audit = load_embedded_unit(record)
        else:
            raise ValueError("unsupported image source")
        radii = [int(value) for value in record["patch_radii_hires_pixels"]]
        boundary_shift = {}
        for radius in radii:
            rounded = np.rint(xy).astype(np.int64)
            clamped = rounded.copy()
            clamped[:, 0] = np.clip(clamped[:, 0], radius, image.shape[1] - radius)
            clamped[:, 1] = np.clip(clamped[:, 1], radius, image.shape[0] - radius)
            shift = np.sqrt(np.sum((clamped - rounded) ** 2, axis=1))
            boundary_shift[str(radius)] = {
                "shifted_patch_count": int(np.sum(shift > 0)),
                "maximum_center_shift_pixels": float(shift.max()),
                "mean_center_shift_pixels": float(shift.mean()),
            }
        handcrafted = extract_handcrafted(image, xy, radii)
        resnet, gpu_record = extract_resnet(image, xy, radii, args.batch_size)
        output_path = args.output_root / f"{record['unit_id']}_morphology_views.npz"
        np.savez_compressed(
            output_path,
            ordered_ids=np.asarray(ordered_ids),
            coordinates_xy_hires=xy,
            handcrafted=handcrafted,
            resnet18=resnet,
            patch_radii_hires_pixels=np.asarray(radii, dtype=np.int32),
            presence_mask=np.ones(len(ordered_ids), dtype=np.uint8),
        )
        run_records.append(
            {
                "unit_id": record["unit_id"],
                "observations": len(ordered_ids),
                "ordered_id_sha256": ordered_id_sha256(ordered_ids),
                "image_shape": list(image.shape),
                "coordinate_xy_min": xy.min(axis=0).tolist(),
                "coordinate_xy_max": xy.max(axis=0).tolist(),
                "patch_radii_hires_pixels": radii,
                "all_patches_in_bounds": True,
                "patch_boundary_rule": "shift_window_inward_without_synthetic_pixels",
                "patch_boundary_shift_by_radius": boundary_shift,
                "handcrafted_shape": list(handcrafted.shape),
                "handcrafted_finite": bool(np.isfinite(handcrafted).all()),
                "resnet18_shape": list(resnet.shape),
                "resnet18_finite": bool(np.isfinite(resnet).all()),
                "output_path": str(output_path),
                "output_sha256": sha256_file(output_path),
                "wall_seconds_total": time.perf_counter() - started,
                "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
                **audit,
                **gpu_record,
            }
        )
    audit_path = args.output_root / "morphology_extraction_audit.json"
    audit_path.write_text(json.dumps({"units": run_records}, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", "units": run_records, "audit": str(audit_path)}, indent=2))


if __name__ == "__main__":
    main()
