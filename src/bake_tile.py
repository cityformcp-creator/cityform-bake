"""Bake ONE 1 km × 1 km tile into STL + GLB + preview PNG.

Inputs: tile_id (OS grid ref like "SK3587") + centre lat/lng.
Outputs: <out_dir>/{city.stl, city.glb, preview.png, meta.json}

Wraps the existing cityform-tool pipeline (no logic duplicated):
  pipeline.wcs.WCSFetcher    — pulls EA LIDAR DTM + DSM tiles
  tier3_measured_roofs       — builds the print STL from DTM + DSM
  pipeline.preview_mesh      — STL → compact GLB for the storefront viewer
  render                     — STL → top-down preview PNG

Designed to run anywhere with the cityform-tool source available, set
the CITYFORM_TOOL env var to point at it (defaults to
"../../cityform-offline/cityform-tool" relative to this file).

The OUTPUT of this script is what gets uploaded to Cloudflare R2 by the
GitHub Actions workflow. No web/Shopify/Etsy state is touched here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


# Use the vendored cityform-tool subset that ships with this repo. The
# four files in vendor/ are sufficient for the bake — see
# vendor/README.md for the snapshot date + how to re-sync. Falling back
# to the full cityform-tool checkout via the CITYFORM_TOOL env var is
# still supported for ad-hoc local runs against an unvendored copy.
VENDOR = Path(__file__).resolve().parent.parent / "vendor"
CITYFORM_TOOL_OVERRIDE = os.environ.get("CITYFORM_TOOL")

if CITYFORM_TOOL_OVERRIDE:
    source = Path(CITYFORM_TOOL_OVERRIDE)
    if not source.exists():
        sys.exit(f"CITYFORM_TOOL={source} does not exist")
    sys.path.insert(0, str(source))
elif VENDOR.exists():
    sys.path.insert(0, str(VENDOR))
else:
    sys.exit(f"no vendor/ dir at {VENDOR} and CITYFORM_TOOL env var unset")

# Imports below depend on sys.path patched above.
from pyproj import Transformer    # noqa: E402

from pipeline.wcs import WCSFetcher                      # noqa: E402
from pipeline.preview_mesh import generate_preview_glb   # noqa: E402
import tier3_measured_roofs                              # noqa: E402

try:
    import render as _render
    _HAS_RENDER = True
except Exception as exc:    # noqa: BLE001
    print(f"[warn] render module not importable: {exc} — preview PNG will be skipped",
          file=sys.stderr)
    _HAS_RENDER = False


# Geometry constants — match what the catalogue cities use so the
# picker output is visually consistent with hand-curated products.
SIZE_M = 1000           # 1 km × 1 km world tile
PRINT_MM = 90.0         # 9 cm physical print width
PLINTH_MM = 2.0         # 2 mm raised plinth border
Z_EXAGGERATION = 1.0    # no vertical scaling (matches catalogue default)


_WGS84_TO_BNG = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)


def bake(
    tile_id: str,
    centre_lat: float,
    centre_lng: float,
    out_dir: Path,
    *,
    place_name: str = "",
    cache_dir: Path | None = None,
    skip_glb: bool = False,
    skip_png: bool = False,
) -> dict:
    """Run the full STL → GLB → PNG pipeline for one tile.

    Returns a dict of timings + output paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if cache_dir is None:
        # WCSFetcher caches downloaded LIDAR GeoTIFFs here. Sharing the
        # cache across tile bakes saves bandwidth — neighbouring tiles
        # often share LIDAR cells. Repo-local so it's per-bake-run on
        # GH Actions, persistent on local dev.
        cache_dir = VENDOR.parent / ".lidar-cache"

    timings: dict[str, float] = {}
    t_start = time.time()

    # 1. Project WGS84 → BNG, compute bbox.
    centre_e, centre_n = _WGS84_TO_BNG.transform(centre_lng, centre_lat)
    half = SIZE_M / 2
    bng_e_min = centre_e - half
    bng_n_min = centre_n - half
    bng_e_max = centre_e + half
    bng_n_max = centre_n + half
    print(f"[bake] {tile_id} centre={centre_lat:.4f},{centre_lng:.4f} "
          f"BNG=({centre_e:.0f},{centre_n:.0f})")

    # 2. Fetch DTM + DSM. WCSFetcher caches under cache_dir so repeated
    # bakes of adjacent tiles share data — important for the GH Actions
    # workflow where neighbours often share LIDAR cells.
    t0 = time.time()
    fetcher = WCSFetcher(cache_dir=cache_dir)
    dtm_path = fetcher.fetch_geotiff("dtm_1m", bng_e_min, bng_n_min, bng_e_max, bng_n_max)
    dsm_path = fetcher.fetch_geotiff("dsm_1m_first", bng_e_min, bng_n_min, bng_e_max, bng_n_max)
    timings["fetch_lidar_s"] = round(time.time() - t0, 2)
    print(f"[bake] LIDAR fetched in {timings['fetch_lidar_s']}s")

    # 3. Build STL.
    stl_path = out_dir / "city.stl"
    t0 = time.time()
    tier3_measured_roofs.build_tier3_stl(
        dsm_path=str(dsm_path), dtm_path=str(dtm_path),
        centre_east=centre_e, centre_north=centre_n,
        size_m=SIZE_M, print_w_mm=PRINT_MM, plinth_mm=PLINTH_MM,
        z_exaggeration=Z_EXAGGERATION, out_path=str(stl_path),
    )
    timings["build_stl_s"] = round(time.time() - t0, 2)
    stl_mb = stl_path.stat().st_size / 1e6
    print(f"[bake] STL built in {timings['build_stl_s']}s ({stl_mb:.1f} MB)")

    # 4. STL → GLB for the storefront 3D viewer.
    glb_path = out_dir / "city.glb"
    if not skip_glb:
        t0 = time.time()
        generate_preview_glb(stl_path, glb_path)
        timings["build_glb_s"] = round(time.time() - t0, 2)
        glb_mb = glb_path.stat().st_size / 1e6
        print(f"[bake] GLB built in {timings['build_glb_s']}s ({glb_mb:.2f} MB)")

    # 5. STL → preview PNG for the picker hover state.
    preview_path = out_dir / "preview.png"
    wireframe_path = out_dir / "wireframe.png"
    if not skip_png and _HAS_RENDER:
        t0 = time.time()
        try:
            _render.render_both(str(stl_path), str(preview_path), str(wireframe_path))
            timings["render_png_s"] = round(time.time() - t0, 2)
            print(f"[bake] PNG rendered in {timings['render_png_s']}s")
        except Exception as exc:    # noqa: BLE001
            print(f"[warn] PNG render failed: {exc}", file=sys.stderr)
            timings["render_png_error"] = str(exc)

    # 6. Write meta.json — the per-tile metadata the storefront picker
    # reads to populate the customise-label defaults + cart properties.
    meta = {
        "tile_id": tile_id,
        "place_name": place_name,
        "centre_lat": round(centre_lat, 6),
        "centre_lng": round(centre_lng, 6),
        "bng_easting": round(centre_e, 1),
        "bng_northing": round(centre_n, 1),
        "size_m": SIZE_M,
        "print_mm": PRINT_MM,
        "plinth_mm": PLINTH_MM,
        "z_exaggeration": Z_EXAGGERATION,
        "outputs": {
            "stl": "city.stl",
            "glb": "city.glb" if glb_path.exists() else None,
            "preview": "preview.png" if preview_path.exists() else None,
        },
        "timings_s": timings,
        "baked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    total_s = round(time.time() - t_start, 2)
    print(f"[bake] {tile_id} DONE in {total_s}s → {out_dir}")

    return {
        "tile_id": tile_id,
        "out_dir": str(out_dir),
        "total_s": total_s,
        "timings": timings,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tile-id", required=True, help="OS grid ref, e.g. SK3587")
    p.add_argument("--lat", type=float, required=True, help="Tile centre latitude (WGS84)")
    p.add_argument("--lng", type=float, required=True, help="Tile centre longitude (WGS84)")
    p.add_argument("--place", default="", help="Place name (label)")
    p.add_argument("--out", required=True, help="Output directory for this tile")
    p.add_argument("--skip-glb", action="store_true", help="Skip GLB generation (faster, STL only)")
    p.add_argument("--skip-png", action="store_true", help="Skip preview PNG generation")
    args = p.parse_args()

    try:
        bake(
            tile_id=args.tile_id,
            centre_lat=args.lat,
            centre_lng=args.lng,
            out_dir=Path(args.out),
            place_name=args.place,
            skip_glb=args.skip_glb,
            skip_png=args.skip_png,
        )
    except Exception as exc:    # noqa: BLE001
        print(f"[bake] ERROR baking {args.tile_id}: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
