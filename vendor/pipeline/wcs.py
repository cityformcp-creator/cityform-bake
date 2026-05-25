"""
Environment Agency LIDAR fetcher (WCS 2.0.1).

Endpoints documented at:
  https://environment.data.gov.uk/support/announcements/275811447/275811543

The portal exposes per-product OGC Web Coverage Services. We hit GetCapabilities
once per endpoint to discover the actual coverageId (cached on disk so subsequent
runs skip the discovery), then GetCoverage for each bbox.

GeoTIFFs returned are in EPSG:27700 (British National Grid).
"""

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Optional

import requests

# EA's WCS gateway intermittently 504s (and occasionally 502/503) under
# load, and slow GetCoverage calls sometimes read-timeout. These are
# transient — retry with exponential backoff before surfacing the error.
_RETRY_STATUS = {429, 500, 502, 503, 504}


def _get_with_retries(url, params, *, stream=False, timeout=None,
                       attempts=4, base_delay=2.0):
    """GET with exponential backoff on transient upstream failures.

    Retries on connection/read timeouts and HTTP 429/5xx. Non-retriable
    responses (e.g. 400/404) and the final attempt are returned as-is so
    callers keep their existing detailed error handling.
    """
    if timeout is None:
        timeout = TIMEOUT_SEC
    for attempt in range(attempts):
        last = attempt == attempts - 1
        try:
            resp = requests.get(url, params=params, timeout=timeout,
                                 stream=stream)
        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError) as exc:
            if last:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"  WCS request error ({exc.__class__.__name__}); "
                  f"attempt {attempt + 1}/{attempts}, retrying in "
                  f"{delay:.0f}s…")
            time.sleep(delay)
            continue
        if resp.status_code not in _RETRY_STATUS or last:
            return resp
        status = resp.status_code
        retry_after = resp.headers.get("Retry-After", "")
        resp.close()  # don't leak the streamed connection before retrying
        delay = base_delay * (2 ** attempt)
        if retry_after.isdigit():
            delay = max(delay, float(retry_after))
        print(f"  WCS HTTP {status}; attempt {attempt + 1}/{attempts}, "
              f"retrying in {delay:.0f}s…")
        time.sleep(delay)
    return resp  # pragma: no cover — loop always returns/raises above

# Per-product WCS service URLs
ENDPOINTS = {
    "dtm_1m": "https://environment.data.gov.uk/spatialdata/lidar-composite-digital-terrain-model-dtm-1m/wcs",
    "dsm_1m_first": "https://environment.data.gov.uk/spatialdata/lidar-composite-digital-surface-model-first-return-dsm-1m/wcs",
    "dsm_1m_last":  "https://environment.data.gov.uk/spatialdata/lidar-composite-digital-surface-model-last-return-dsm-1m/wcs",
    "dtm_2m": "https://environment.data.gov.uk/spatialdata/lidar-composite-digital-terrain-model-dtm-2m/wcs",
    "dsm_2m_first": "https://environment.data.gov.uk/spatialdata/lidar-composite-digital-surface-model-first-return-dsm-2m/wcs",
    "dsm_2m_last":  "https://environment.data.gov.uk/spatialdata/lidar-composite-digital-surface-model-last-return-dsm-2m/wcs",
}

# Default request timeout. WCS GetCoverage for a 1.2 km bbox typically
# responds in 5-30s; allow generous headroom for cold-cache cases.
TIMEOUT_SEC = 180


class WCSFetcher:
    """Fetches LIDAR rasters via OGC Web Coverage Service.

    Discoveries (coverage IDs) and downloaded GeoTIFFs are both cached on
    disk under cache_dir, so repeat runs for the same bbox are instant.
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._coverage_cache_path = self.cache_dir / "wcs_coverage_ids.json"
        self._coverage_ids: dict[str, str] = self._load_coverage_cache()

    def _load_coverage_cache(self) -> dict[str, str]:
        if self._coverage_cache_path.exists():
            try:
                return json.loads(self._coverage_cache_path.read_text())
            except json.JSONDecodeError:
                return {}
        return {}

    def _save_coverage_cache(self) -> None:
        self._coverage_cache_path.write_text(json.dumps(self._coverage_ids, indent=2))

    def discover_coverage_id(self, product: str) -> str:
        """Return the coverage ID for a product, fetching GetCapabilities if needed.

        WCS GetCapabilities returns XML with one or more <wcs:CoverageSummary>
        blocks. For these EA endpoints there's exactly one coverage per URL,
        but we don't hardcode the ID because EA renames them periodically.
        """
        if product in self._coverage_ids:
            return self._coverage_ids[product]
        if product not in ENDPOINTS:
            raise ValueError(f"Unknown product '{product}'. Options: {list(ENDPOINTS)}")

        url = ENDPOINTS[product]
        params = {"service": "WCS", "version": "2.0.1", "request": "GetCapabilities"}
        resp = _get_with_retries(url, params)
        resp.raise_for_status()
        xml = resp.text
        # Extract first CoverageId. Be permissive about XML namespacing.
        match = re.search(r"<(?:\w+:)?CoverageId>([^<]+)</(?:\w+:)?CoverageId>", xml)
        if not match:
            raise RuntimeError(
                f"No CoverageId found in GetCapabilities response from {url}. "
                f"First 500 chars: {xml[:500]}"
            )
        coverage_id = match.group(1).strip()
        self._coverage_ids[product] = coverage_id
        self._save_coverage_cache()
        return coverage_id

    # WCS endpoints typically cap a single GetCoverage at ~4096×4096 pixels.
    # 3500 leaves a safety margin against off-by-one rounding inside the
    # service. Larger bboxes get split into a sub-tile grid + mosaicked.
    MAX_PIXELS_PER_SIDE = 3500

    def fetch_geotiff(
        self,
        product: str,
        bng_e_min: float, bng_n_min: float,
        bng_e_max: float, bng_n_max: float,
    ) -> Path:
        """Fetch a GeoTIFF for the given BNG bbox. Returns path to cached file.

        Auto-tiles the request if the bbox is too big for a single WCS
        GetCoverage. Tiles cache individually, the mosaic caches under a
        ``_mosaic`` suffix so re-runs are instant.
        """
        # Resolution in metres per pixel — drive the tile-grid math.
        res_m = 2.0 if product.endswith("_2m") or product.endswith("_2m_first") or product.endswith("_2m_last") else 1.0
        width_px = (bng_e_max - bng_e_min) / res_m
        height_px = (bng_n_max - bng_n_min) / res_m
        if max(width_px, height_px) <= self.MAX_PIXELS_PER_SIDE:
            return self._fetch_single(product, bng_e_min, bng_n_min, bng_e_max, bng_n_max)

        # Mosaic cache key — same shape as a single fetch but suffixed.
        e1, n1, e2, n2 = (int(round(v)) for v in (bng_e_min, bng_n_min, bng_e_max, bng_n_max))
        cache_path = self.cache_dir / f"{product}_{e1}_{n1}_{e2}_{n2}_mosaic.tif"
        if cache_path.exists() and cache_path.stat().st_size > 1024:
            return cache_path

        # Compute grid size that keeps each tile under MAX_PIXELS_PER_SIDE.
        import math
        n_x = max(1, math.ceil(width_px / self.MAX_PIXELS_PER_SIDE))
        n_y = max(1, math.ceil(height_px / self.MAX_PIXELS_PER_SIDE))
        print(f"  WCS bbox {width_px:.0f}×{height_px:.0f} px exceeds {self.MAX_PIXELS_PER_SIDE} cap "
              f"— splitting into {n_x}×{n_y} tiles for {product}…")

        # Fetch each sub-tile (cached individually so a partial failure can resume).
        tile_paths: list[Path] = []
        dx = (bng_e_max - bng_e_min) / n_x
        dy = (bng_n_max - bng_n_min) / n_y
        for i in range(n_x):
            for j in range(n_y):
                sub_e_min = bng_e_min + i * dx
                sub_e_max = bng_e_min + (i + 1) * dx
                sub_n_min = bng_n_min + j * dy
                sub_n_max = bng_n_min + (j + 1) * dy
                p = self._fetch_single(product, sub_e_min, sub_n_min, sub_e_max, sub_n_max)
                tile_paths.append(p)
                print(f"    tile {i + 1 + j * n_x}/{n_x * n_y}: {p.name} ({p.stat().st_size // 1024} KB)")

        # Mosaic via rasterio.merge — handles overlap + georef alignment.
        import rasterio
        from rasterio.merge import merge
        sources = [rasterio.open(p) for p in tile_paths]
        try:
            mosaic_arr, out_transform = merge(sources)
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

        tmp_path = cache_path.with_suffix(".tif.tmp")
        with rasterio.open(tmp_path, "w", **out_meta) as dest:
            dest.write(mosaic_arr)
        tmp_path.rename(cache_path)
        print(f"  mosaicked → {cache_path.name} ({cache_path.stat().st_size / 1e6:.1f} MB)")
        return cache_path

    def _fetch_single(
        self,
        product: str,
        bng_e_min: float, bng_n_min: float,
        bng_e_max: float, bng_n_max: float,
    ) -> Path:
        """Fetch a single WCS tile (no auto-tiling). Caller is responsible
        for sizing the bbox under MAX_PIXELS_PER_SIDE."""
        # Cache key: product + bbox snapped to integer metres
        e1, n1, e2, n2 = (int(round(v)) for v in (bng_e_min, bng_n_min, bng_e_max, bng_n_max))
        key = f"{product}_{e1}_{n1}_{e2}_{n2}"
        cache_path = self.cache_dir / f"{key}.tif"
        if cache_path.exists() and cache_path.stat().st_size > 1024:
            return cache_path

        coverage_id = self.discover_coverage_id(product)
        endpoint = ENDPOINTS[product]

        # WCS 2.0 'subset' parameters use axis labels. EA's coverages use
        # E (easting) and N (northing) for OSGB. requests will percent-encode
        # the parentheses.
        params = [
            ("service", "WCS"),
            ("version", "2.0.1"),
            ("request", "GetCoverage"),
            ("coverageId", coverage_id),
            ("subset", f"E({e1},{e2})"),
            ("subset", f"N({n1},{n2})"),
            ("format", "image/tiff"),
        ]
        resp = _get_with_retries(endpoint, params, stream=True)
        if resp.status_code != 200:
            # WCS exception responses are XML — surface a useful error
            preview = resp.text[:500] if resp.text else "(no body)"
            raise RuntimeError(
                f"WCS GetCoverage failed: HTTP {resp.status_code}. "
                f"Endpoint: {endpoint}. Body preview: {preview}"
            )
        # Verify content-type is actually TIFF, not an XML exception
        ct = resp.headers.get("content-type", "")
        if "tiff" not in ct.lower() and "image" not in ct.lower():
            preview = resp.text[:500] if hasattr(resp, "text") else "(non-text)"
            raise RuntimeError(
                f"WCS returned non-image content-type '{ct}'. Body preview: {preview}"
            )

        # Stream to disk
        tmp_path = cache_path.with_suffix(".tif.tmp")
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=64 * 1024):
                f.write(chunk)
        tmp_path.rename(cache_path)
        return cache_path
