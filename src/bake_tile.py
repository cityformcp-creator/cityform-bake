"""Bake ONE 1 km × 1 km tile into STL + GLB + preview PNG.

Inputs: tile_id (OS grid ref like "SK3587") + centre lat/lng.
Outputs: <out_dir>/{city.stl, city.glb, preview.png, meta.json}

Wraps the existing cityform-tool pipeline (no logic duplicated):
  pipeline.wcs.WCSFetcher          — pulls EA LIDAR DTM + DSM tiles
  pipeline.overpass.OverpassFetcher — pulls OSM water/bridges/landmarks/roads
  tier3_with_water                  — builds the print STL with full finish:
                                       water flattened, roads engraved,
                                       bridges as separate decks, landmark
                                       heights overridden by OSM 3D tags
  pipeline.preview_mesh             — STL → compact GLB for storefront viewer
  render                            — STL → top-down preview PNG

This matches the Flask app's /api/generate pipeline byte-for-byte, so
the picker's preview models are catalogue-quality (not the noisy
LIDAR-only output of the original `tier3_measured_roofs` path).

OSM fetches are best-effort: any individual failure falls back silently
to "no polygons of that type" rather than aborting the bake. A tile
with no water fetched still produces a valid STL — just without the
water cuts. The bake never aborts on transient network issues.

Designed to run anywhere with cityform-tool source available, set the
CITYFORM_TOOL env var to point at it (defaults to vendored copy in
../vendor/ shipped with this repo).

OUTPUT of this script is what gets uploaded to GitHub Releases by the
GH Actions workflow. No web/Shopify/Etsy state is touched here.
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

from pipeline.sources import pick_source                 # noqa: E402
from pipeline.overpass import OverpassFetcher            # noqa: E402
from pipeline.preview_mesh import generate_preview_glb   # noqa: E402
import tier3_with_water                                  # noqa: E402

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
_BNG_TO_WGS84 = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)


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
    admin1: str | None = None,
    roof_detail: str = "detailed",
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

    # 2. Fetch DTM + DSM. pick_source routes by lat/lng — EA WCS for
    # England, NRW WFS+Azure for Wales, Scottish Remote Sensing Portal
    # for Scotland. Cache shared across tiles so adjacent bakes reuse
    # LIDAR cells.
    t0 = time.time()
    # WGS84 bbox for source routing.
    lat_min_wgs, _  = _BNG_TO_WGS84.transform(bng_e_min, bng_n_min)[::-1]
    lat_max_wgs, _  = _BNG_TO_WGS84.transform(bng_e_max, bng_n_max)[::-1]
    lng_min_wgs = _BNG_TO_WGS84.transform(bng_e_min, bng_n_min)[0]
    lng_max_wgs = _BNG_TO_WGS84.transform(bng_e_max, bng_n_max)[0]
    source = pick_source(
        lat_min=lat_min_wgs, lng_min=lng_min_wgs,
        lat_max=lat_max_wgs, lng_max=lng_max_wgs,
        cache_root=cache_dir,
        admin1=admin1,
    )
    print(f"[bake] DEM source: {source.region_name} ({source.expected_resolution_m:g} m)")
    dtm_path = source.fetch_dtm(bng_e_min, bng_n_min, bng_e_max, bng_n_max)
    dsm_path = source.fetch_dsm(bng_e_min, bng_n_min, bng_e_max, bng_n_max)
    timings["fetch_lidar_s"] = round(time.time() - t0, 2)
    print(f"[bake] LIDAR fetched in {timings['fetch_lidar_s']}s ({source.region_name})")

    # 3. Fetch OSM water + bridges + landmarks + roads. Each fetch is
    # best-effort: a failure → empty list, bake continues with whatever
    # polygons made it back.
    lat_min, lng_min = _BNG_TO_WGS84.transform(bng_e_min, bng_n_min)[::-1]
    lat_max, lng_max = _BNG_TO_WGS84.transform(bng_e_max, bng_n_max)[::-1]
    # WGS84 transform returns (lng, lat) — reorder above flipped that.

    t0 = time.time()
    osm_cache = cache_dir / "osm"
    osm_cache.mkdir(parents=True, exist_ok=True)
    osm_fetcher = OverpassFetcher(cache_dir=osm_cache)

    def _safe_fetch(name, fn, *args):
        try:
            return fn(*args)
        except Exception as exc:    # noqa: BLE001
            print(f"[warn] OSM fetch '{name}' failed: {exc} — continuing without",
                  file=sys.stderr)
            return {"features": []}

    water_geojson    = _safe_fetch("water",     osm_fetcher.fetch_water,     lat_min, lng_min, lat_max, lng_max)
    coast_geojson    = _safe_fetch("coastline", osm_fetcher.fetch_coastline, lat_min, lng_min, lat_max, lng_max)
    bridge_geojson   = _safe_fetch("bridges",   osm_fetcher.fetch_bridges,   lat_min, lng_min, lat_max, lng_max)
    landmark_geojson = _safe_fetch("landmarks", osm_fetcher.fetch_landmarks, lat_min, lng_min, lat_max, lng_max)
    road_geojson     = _safe_fetch("roads",     osm_fetcher.fetch_roads,     lat_min, lng_min, lat_max, lng_max)
    # OSM Simple-3D-Buildings (`building:part`) — overrides LIDAR DSM for
    # tall complex buildings (Shard, Gherkin, Walkie-Talkie, cathedral
    # spires). Without this, LIDAR scan-line noise on tall roofs renders
    # as a forest of thin vertical lines. Coverage is concentrated in
    # London + a few major cities; outside those it's an empty fetch
    # and the LIDAR fallback runs unchanged.
    building_parts_geojson = _safe_fetch(
        "building_parts", osm_fetcher.fetch_building_parts,
        lat_min, lng_min, lat_max, lng_max,
    )
    timings["fetch_osm_s"] = round(time.time() - t0, 2)

    # Convert each GeoJSON feature set to BNG polygons via the helpers
    # vendored from tier3_with_water.
    # NOTE: use the waterway-aware helper so linestring rivers
    # (waterway=river|stream|canal|drain) get buffered into polygons too.
    # Polygon-only `osm_features_to_bng_polygons` silently misses them.
    water_polys = tier3_with_water.osm_waterway_features_to_bng_polygons(
        water_geojson.get("features", []) or [])
    # Coastlines need separate conversion — they're LineStrings that get
    # polygonised against the bbox into sea polygons.
    coast_feats = coast_geojson.get("features", []) or []
    if coast_feats:
        sea_polys = tier3_with_water.coastline_features_to_sea_polygons(
            coast_feats, centre_e, centre_n, SIZE_M)
        water_polys = list(water_polys) + list(sea_polys)
    bridge_polys = tier3_with_water.osm_bridge_features_to_bng_geoms(
        bridge_geojson.get("features", []) or [], line_buffer_m=4.0)
    landmark_polys = tier3_with_water.osm_features_to_bng_polygons(
        landmark_geojson.get("features", []) or [])
    road_polys = tier3_with_water.osm_road_features_to_bng_geoms(
        road_geojson.get("features", []) or [])
    # Parse S3DB building:part features into the dict-of-parts the builder
    # expects: {polygon, height_m, min_height_m, roof_shape, …}. Empty
    # outside London / major cities — bake then falls back to LIDAR.
    building_part_dicts = tier3_with_water.osm_building_parts_to_bng(
        building_parts_geojson.get("features", []) or [])
    print(f"[bake] OSM fetched in {timings['fetch_osm_s']}s "
          f"(water={len(water_polys)}, bridges={len(bridge_polys)}, "
          f"landmarks={len(landmark_polys)}, roads={len(road_polys)}, "
          f"s3db_parts={len(building_part_dicts)})")

    # 4. Build STL with the full pipeline — water flattened, roads
    # engraved, bridges as separate decks, landmark heights honoured.
    stl_path = out_dir / "city.stl"
    t0 = time.time()
    tier3_with_water.build_tier3_with_water_stl(
        dsm_path=str(dsm_path), dtm_path=str(dtm_path),
        centre_east=centre_e, centre_north=centre_n,
        size_m=SIZE_M, print_w_mm=PRINT_MM, plinth_mm=PLINTH_MM,
        z_exaggeration=Z_EXAGGERATION,
        water_polygons_bng=water_polys,
        bridge_polygons_bng=bridge_polys,
        landmark_polygons_bng=landmark_polys,
        road_polygons_bng=road_polys,
        building_part_dicts=building_part_dicts,
        # roof_detail: "smoothed" (5×5 median) is the bake default — kills
        # LIDAR scan-line striping that gives modern flat office roofs a
        # corrugated/vertical-lined appearance on the print. "detailed"
        # (3×3) preserves more roof texture but lets stripe artefacts
        # through on tall buildings. Pitched roofs survive both modes.
        roof_detail=roof_detail,
        out_path=str(stl_path),
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
    p.add_argument("--admin1", default="",
                   help="Geonames country code (ENG/WLS/SCT) — authoritative "
                        "source routing, bypasses bbox heuristic that mis-routes "
                        "border-region tiles. Empty = use bbox fallback.")
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
            admin1=args.admin1 or None,
        )
    except Exception as exc:    # noqa: BLE001
        print(f"[bake] ERROR baking {args.tile_id}: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
