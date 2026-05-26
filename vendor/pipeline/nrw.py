"""
Natural Resources Wales / Welsh Government LIDAR fetcher.

NRW publishes LIDAR via DataMapWales (GeoServer-backed). Unlike the EA
WCS, there is no OGC Web Coverage Service exposing the rasters directly
— the supported pattern is:

  1. WFS query the tile catalogue layer for tiles intersecting the bbox.
  2. Each tile record carries `dtm_link` / `dsm_link` fields pointing
     at a raw GeoTIFF on Azure Blob Storage (public, no auth).
  3. Download those tiles (1 km × 1 km, EPSG:27700, float32, 1 m res
     for the Welsh Government 2020-23 dataset).
  4. Mosaic + crop to the requested bbox.

Two catalogue layers exist:
  - geonode:welsh_government_lidar_tile_catalogue_2020_2023  (PRIMARY)
    1 m resolution, ~Wales-wide for areas flown 2020-22, direct GeoTIFFs,
    fields `dtm_link` + `dsm_link`.
  - geonode:nrw_lidar_tile_catalogue_archive                  (FALLBACK)
    Historic captures 1998-2019, mixed 1 m / 2 m resolution, ZIP-bundled
    per 10 km square, fields `dtm_url` + `dsm_url`. Not implemented in
    this MVP — covered tiles route to the 2020-23 dataset first; uncovered
    bboxes raise so the caller can surface a "request via studio@" path.

Licence: Open Government Licence v3.0 (DataMapWales). Attribution:
  "Contains Natural Resources Wales information © Natural Resources Wales
  and Database Right. Contains data from Welsh Government LiDAR Programme."

Output GeoTIFFs are already in EPSG:27700 — no reprojection needed; the
mesh builder consumes them identically to EA outputs.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import requests


# Welsh Government LiDAR 2020-2023 (the main 1 m dataset). Verified
# 2026-05-26 via WFS GetCapabilities on the lidar-viewer endpoint.
CATALOGUE_LAYER = "geonode:welsh_government_lidar_tile_catalogue_2020_2023"

# GeoServer OWS endpoint for DataMapWales. WFS GetFeature for the
# catalogue layer returns tile records with download links.
WFS_BASE = "https://datamap.gov.wales/geoserver/ows"

# Per-tile GeoTIFFs live on Azure Blob Storage. The WFS response stores
# the URL without scheme (e.g. "dmwproductionblob.blob.core.windows.net/...").
_AZURE_SCHEME = "https://"

# Retry policy for WFS + Azure downloads. NRW's GeoServer occasionally
# returns 502/504 under load, mirroring the EA WCS behaviour.
_RETRY_STATUS = {429, 500, 502, 503, 504}
_TIMEOUT_SEC = 120


def _get_with_retries(url, *, params=None, stream=False, timeout=_TIMEOUT_SEC,
                       attempts=4, base_delay=2.0):
    """GET with exponential backoff on transient upstream failures.

    Mirrors pipeline/wcs.py's helper — kept here as a local copy so this
    module is independently importable (vendoring story for cityform-bake).
    """
    for attempt in range(attempts):
        last = attempt == attempts - 1
        try:
            resp = requests.get(url, params=params, timeout=timeout, stream=stream)
        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError) as exc:
            if last:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"  NRW request error ({exc.__class__.__name__}); "
                  f"attempt {attempt + 1}/{attempts}, retrying in {delay:.0f}s…")
            time.sleep(delay)
            continue
        if resp.status_code not in _RETRY_STATUS or last:
            return resp
        status = resp.status_code
        retry_after = resp.headers.get("Retry-After", "")
        resp.close()
        delay = base_delay * (2 ** attempt)
        if retry_after.isdigit():
            delay = max(delay, float(retry_after))
        print(f"  NRW HTTP {status}; attempt {attempt + 1}/{attempts}, "
              f"retrying in {delay:.0f}s…")
        time.sleep(delay)
    return resp  # pragma: no cover


class NrwLidarFetcher:
    """Fetches Welsh Gov 2020-23 LIDAR tiles via DataMapWales WFS + Azure CDN.

    Public API mirrors WCSFetcher.fetch_geotiff so the two are
    interchangeable behind `pipeline/sources.py` (the EA WCS source and
    the NRW source both produce EPSG:27700 GeoTIFFs).

    Caching:
      - per-tile GeoTIFFs cached under cache_dir/tiles/<filename>
      - mosaic results cached under cache_dir/<bbox>_<product>.tif
      - WFS query results NOT cached — they're cheap and the catalogue
        may be updated as new captures land.
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._tile_cache = self.cache_dir / "tiles"
        self._tile_cache.mkdir(parents=True, exist_ok=True)

    # ── public API ──────────────────────────────────────────────────────

    def fetch_geotiff(
        self,
        product: str,
        bng_e_min: float, bng_n_min: float,
        bng_e_max: float, bng_n_max: float,
    ) -> Path:
        """Return path to a mosaicked GeoTIFF covering the bbox.

        ``product`` is either "dtm_1m" or "dsm_1m" — names match WCSFetcher's
        product IDs so the two fetchers are drop-in interchangeable from
        the caller's side.
        """
        if product == "dtm_1m":
            link_field, kind = "dtm_link", "dtm"
        elif product in ("dsm_1m_first", "dsm_1m"):
            link_field, kind = "dsm_link", "dsm"
        else:
            raise ValueError(
                f"NrwLidarFetcher supports 'dtm_1m' / 'dsm_1m_first' only; got {product!r}"
            )

        # Mosaic cache check first — most repeat-bbox calls short-circuit here.
        e1, n1, e2, n2 = (int(round(v)) for v in
                          (bng_e_min, bng_n_min, bng_e_max, bng_n_max))
        mosaic_path = self.cache_dir / f"nrw_{kind}_{e1}_{n1}_{e2}_{n2}.tif"
        if mosaic_path.exists() and mosaic_path.stat().st_size > 1024:
            return mosaic_path

        # WFS bbox query expects WGS84 lat/lng. Convert from BNG.
        from pyproj import Transformer
        to_wgs84 = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
        lng_min, lat_min = to_wgs84.transform(bng_e_min, bng_n_min)
        lng_max, lat_max = to_wgs84.transform(bng_e_max, bng_n_max)

        tile_records = self._query_catalogue(lat_min, lng_min, lat_max, lng_max)
        if not tile_records:
            raise FileNotFoundError(
                f"No NRW LiDAR tiles cover bbox ({lat_min:.4f},{lng_min:.4f}) → "
                f"({lat_max:.4f},{lng_max:.4f}). This area is outside the Welsh "
                f"Government 2020-23 dataset; route customer to email fulfilment."
            )

        # Download each tile (cached by URL hash → filename).
        tile_paths: list[Path] = []
        for rec in tile_records:
            link = rec.get(link_field) or ""
            if not link:
                # Older captures sometimes have DSM but no DTM (or vice versa).
                continue
            tile_path = self._download_tile(link)
            tile_paths.append(tile_path)

        if not tile_paths:
            raise FileNotFoundError(
                f"NRW catalogue returned {len(tile_records)} tiles for the bbox but "
                f"none have a {link_field} (product={product}). The area may only "
                f"have the complementary surface — try the other product."
            )

        # Mosaic + crop to requested bbox.
        self._mosaic_and_crop(
            tile_paths, mosaic_path,
            bng_e_min, bng_n_min, bng_e_max, bng_n_max,
        )
        return mosaic_path

    # ── internals ───────────────────────────────────────────────────────

    def _query_catalogue(
        self, lat_min: float, lng_min: float, lat_max: float, lng_max: float,
    ) -> list[dict]:
        """WFS GetFeature against the Welsh Gov 2020-23 catalogue layer.

        GeoServer's bbox parameter takes (lng_min, lat_min, lng_max, lat_max)
        in CRS-axis order for the named SRS. EPSG:4326 in WFS 2.0 is
        latitude-first by default, but GeoServer accepts the order
        explicitly when you append the CRS to the bbox param. We use
        lng_min,lat_min,lng_max,lat_max — observed to work in the
        2026-05-26 probe.
        """
        params = {
            "service": "WFS",
            "version": "2.0.0",
            "request": "GetFeature",
            "typeNames": CATALOGUE_LAYER,
            "srsName": "EPSG:4326",
            "outputFormat": "application/json",
            "bbox": f"{lng_min},{lat_min},{lng_max},{lat_max},EPSG:4326",
        }
        resp = _get_with_retries(WFS_BASE, params=params)
        if resp.status_code != 200:
            raise RuntimeError(
                f"NRW WFS GetFeature failed: HTTP {resp.status_code}. "
                f"Body preview: {resp.text[:300]}"
            )
        try:
            data = resp.json()
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"NRW WFS returned non-JSON body: {resp.text[:300]}"
            ) from exc

        features = data.get("features", []) or []
        # Sort by year desc so newer captures win when we have multiple
        # coverages of the same OS grid cell. Keeps the mosaic consistent
        # across re-runs even if the catalogue order shifts.
        features.sort(
            key=lambda f: -(int(f.get("properties", {}).get("year") or 0))
        )
        return [f.get("properties", {}) for f in features]

    def _download_tile(self, link_field_value: str) -> Path:
        """Download one Azure-hosted GeoTIFF to the tile cache.

        link_field_value is typically scheme-less like
        "dmwproductionblob.blob.core.windows.net/lidar-zips/.../foo.tif".
        Cache filename = basename of the URL (unique enough — Welsh Gov
        embeds the OS grid + date in the filename).
        """
        if "://" not in link_field_value:
            url = _AZURE_SCHEME + link_field_value
        else:
            url = link_field_value

        filename = url.rsplit("/", 1)[-1]
        # Sanity-check: only .tif allowed for now. ZIP-bundled historic
        # archive needs separate handling — return a clear error.
        if not filename.lower().endswith((".tif", ".tiff")):
            raise NotImplementedError(
                f"NRW tile {filename!r} is not a GeoTIFF (looks like a ZIP "
                f"from the historic archive). Historic-archive fallback not "
                f"yet implemented — see pipeline/nrw.py docstring."
            )

        out_path = self._tile_cache / filename
        if out_path.exists() and out_path.stat().st_size > 1024:
            return out_path

        resp = _get_with_retries(url, stream=True, timeout=300)
        if resp.status_code != 200:
            raise RuntimeError(
                f"NRW tile download failed: HTTP {resp.status_code} from {url}. "
                f"Body preview: {resp.text[:300] if resp.text else '(empty)'}"
            )
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=64 * 1024):
                f.write(chunk)
        tmp.rename(out_path)
        return out_path

    def _mosaic_and_crop(
        self,
        tile_paths: list[Path], out_path: Path,
        bng_e_min: float, bng_n_min: float,
        bng_e_max: float, bng_n_max: float,
    ) -> None:
        """Mosaic the per-tile GeoTIFFs and crop to the requested bbox.

        Uses rasterio.merge — same library and code shape as WCSFetcher's
        mosaic path. Crop happens via the `bounds` parameter on merge,
        which clips during the mosaic in one pass.
        """
        import rasterio
        from rasterio.merge import merge

        sources = [rasterio.open(p) for p in tile_paths]
        try:
            mosaic_arr, out_transform = merge(
                sources,
                bounds=(bng_e_min, bng_n_min, bng_e_max, bng_n_max),
            )
            out_meta = sources[0].meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": mosaic_arr.shape[1],
                "width": mosaic_arr.shape[2],
                "transform": out_transform,
                "compress": "lzw",
            })
        finally:
            for src in sources:
                src.close()

        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        with rasterio.open(tmp, "w", **out_meta) as dest:
            dest.write(mosaic_arr)
        tmp.rename(out_path)
        print(
            f"  NRW mosaicked {len(tile_paths)} tiles → {out_path.name} "
            f"({out_path.stat().st_size / 1e6:.1f} MB)"
        )
