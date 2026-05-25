"""
Tier-3: real measured roof surfaces from LIDAR + clean DTM terrain.

Inputs: DSM and DTM GeoTIFFs covering your bbox (download from EA portal).
Output: print-ready STL.

Pipeline:
  1. Crop DSM and DTM to the same bbox
  2. Compute DSM - DTM = above-ground height
  3. Build building mask: hag > 2.5m + morphological opening + size filter
       + dilate-back to recover edges
  4. Top raster: DSM where building, DTM elsewhere (this is the key step —
     keeps measured roof surface, throws away tree/vegetation noise)
  5. Max-pool to 2m grid (keeps building tops sharp at boundaries)
  6. Build watertight mesh: top + bottom + 4 side walls
  7. Write binary STL

Usage:
    python tier3_measured_roofs.py \\
        --dsm path/to/DSM_1m.tif --dtm path/to/DTM_1m.tif \\
        --centre-east 532500 --centre-north 181000 \\
        --size 1200 --print-mm 90 --out london.stl
"""

import argparse
import struct
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds
from scipy import ndimage


def crop_raster(path, bbox):
    with rasterio.open(path) as src:
        win = from_bounds(*bbox, transform=src.transform)
        arr = src.read(1, window=win)
        if src.nodata is not None:
            arr = np.where(arr == src.nodata, np.nan, arr)
    return arr


def write_stl(triangles: np.ndarray, output_path: Path, header: str):
    v0 = triangles[:, 0]; v1 = triangles[:, 1]; v2 = triangles[:, 2]
    normals = np.cross(v1 - v0, v2 - v0)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = (normals / np.where(lengths > 0, lengths, 1.0)).astype(np.float32)

    record_dtype = np.dtype([
        ("normal", np.float32, 3),
        ("v0", np.float32, 3),
        ("v1", np.float32, 3),
        ("v2", np.float32, 3),
        ("attr", np.uint16),
    ])
    records = np.empty(len(triangles), dtype=record_dtype)
    records["normal"] = normals
    records["v0"] = triangles[:, 0]
    records["v1"] = triangles[:, 1]
    records["v2"] = triangles[:, 2]
    records["attr"] = 0

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        h = header.encode("ascii", errors="replace")[:80]
        f.write(h + b"\x00" * (80 - len(h)))
        f.write(struct.pack("<I", len(triangles)))
        f.write(records.tobytes())


def build_tier3_stl(
    *, dsm_path, dtm_path, centre_east, centre_north,
    size_m=1200, print_w_mm=90.0, plinth_mm=4.0,
    z_exaggeration=1.0, out_path="output.stl",
):
    half = size_m / 2
    bbox = (centre_east - half, centre_north - half,
            centre_east + half, centre_north + half)

    print(f"Cropping bbox {bbox}…")
    dsm = crop_raster(dsm_path, bbox)
    dtm = crop_raster(dtm_path, bbox)
    if dsm.shape != dtm.shape:
        raise SystemExit(f"DSM and DTM shapes differ: {dsm.shape} vs {dtm.shape}")

    # Clean outliers and fill any nodata
    dsm = np.clip(np.nan_to_num(dsm, nan=np.nanmedian(dsm)), -10, 400)
    dtm = np.clip(np.nan_to_num(dtm, nan=np.nanmedian(dtm)), -50, 1500)

    print("Detecting buildings…")
    hag = np.maximum(dsm - dtm, 0)
    mask_init = hag > 2.5
    mask_opened = ndimage.binary_opening(mask_init, iterations=2)
    labels, n = ndimage.label(mask_opened)
    sizes = ndimage.sum(mask_opened, labels, range(1, n + 1))
    keep = np.where(sizes >= 25)[0] + 1
    mask_filtered = np.isin(labels, keep)
    mask_dilated = ndimage.binary_dilation(mask_filtered, iterations=3)
    mask_final = ndimage.binary_closing(mask_dilated & mask_init, iterations=1)
    print(f"  building coverage: {100 * mask_final.mean():.1f}%")

    print("Building top raster (DSM in buildings, DTM elsewhere)…")
    top = np.where(mask_final, dsm, dtm).astype(np.float32)

    DEC = 2
    H_orig, W_orig = top.shape
    H = H_orig // DEC
    W = W_orig // DEC
    top_cropped = top[:H * DEC, :W * DEC]
    heights = top_cropped.reshape(H, DEC, W, DEC).max(axis=(1, 3))

    SCALE = print_w_mm / size_m
    z_floor = float(heights.min())
    z_world = heights - z_floor
    relief_mm = float(z_world.max() * SCALE * z_exaggeration)
    print(f"  relief: {z_world.max():.1f} m → {relief_mm:.2f} mm at print scale")

    xs = np.arange(W) * DEC * SCALE
    ys = (H - 1 - np.arange(H)) * DEC * SCALE
    xx, yy = np.meshgrid(xs, ys)
    zz = z_world * SCALE * z_exaggeration + plinth_mm
    top_pts = np.stack([xx, yy, zz], axis=-1).astype(np.float32)
    bot_pts = top_pts.copy()
    bot_pts[..., 2] = 0.0

    print("Building mesh…")
    triangles = []
    for j in range(H - 1):
        for i in range(W - 1):
            v00, v10 = top_pts[j, i], top_pts[j, i + 1]
            v01, v11 = top_pts[j + 1, i], top_pts[j + 1, i + 1]
            triangles.append((v00, v10, v11))
            triangles.append((v00, v11, v01))
    for j in range(H - 1):
        for i in range(W - 1):
            v00, v10 = bot_pts[j, i], bot_pts[j, i + 1]
            v01, v11 = bot_pts[j + 1, i], bot_pts[j + 1, i + 1]
            triangles.append((v00, v11, v10))
            triangles.append((v00, v01, v11))
    for i in range(W - 1):
        t0, t1 = top_pts[0, i], top_pts[0, i + 1]
        b0, b1 = bot_pts[0, i], bot_pts[0, i + 1]
        triangles.append((t0, b0, b1)); triangles.append((t0, b1, t1))
    for i in range(W - 1):
        t0, t1 = top_pts[H - 1, i], top_pts[H - 1, i + 1]
        b0, b1 = bot_pts[H - 1, i], bot_pts[H - 1, i + 1]
        triangles.append((t1, b1, b0)); triangles.append((t1, b0, t0))
    for j in range(H - 1):
        t0, t1 = top_pts[j, 0], top_pts[j + 1, 0]
        b0, b1 = bot_pts[j, 0], bot_pts[j + 1, 0]
        triangles.append((t1, b1, b0)); triangles.append((t1, b0, t0))
    for j in range(H - 1):
        t0, t1 = top_pts[j, W - 1], top_pts[j + 1, W - 1]
        b0, b1 = bot_pts[j, W - 1], bot_pts[j + 1, W - 1]
        triangles.append((t0, b0, b1)); triangles.append((t0, b1, t1))

    triangles = np.array(triangles, dtype=np.float32)
    print(f"  {len(triangles):,} triangles")

    print(f"Writing {out_path}…")
    write_stl(triangles, Path(out_path), header="cityform tier3")
    size_mb = Path(out_path).stat().st_size / 1e6
    print(f"  done — {size_mb:.1f} MB")


def main():
    p = argparse.ArgumentParser(description="Tier-3 STL builder (LIDAR-only)")
    p.add_argument("--dsm", required=True, help="DSM GeoTIFF path")
    p.add_argument("--dtm", required=True, help="DTM GeoTIFF path")
    p.add_argument("--centre-east", type=float, required=True, help="Crop centre BNG E")
    p.add_argument("--centre-north", type=float, required=True, help="Crop centre BNG N")
    p.add_argument("--size", type=int, default=1200, help="Bbox size in metres (default 1200)")
    p.add_argument("--print-mm", type=float, default=90.0, help="Print width in mm")
    p.add_argument("--plinth-mm", type=float, default=4.0, help="Plinth thickness")
    p.add_argument("--z-exag", type=float, default=1.0, help="Vertical exaggeration")
    p.add_argument("--out", default="cityform_tier3.stl", help="Output STL path")
    args = p.parse_args()

    build_tier3_stl(
        dsm_path=args.dsm, dtm_path=args.dtm,
        centre_east=args.centre_east, centre_north=args.centre_north,
        size_m=args.size, print_w_mm=args.print_mm, plinth_mm=args.plinth_mm,
        z_exaggeration=args.z_exag, out_path=args.out,
    )


if __name__ == "__main__":
    main()
