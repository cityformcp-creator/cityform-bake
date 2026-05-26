"""
DEM source abstraction + region router.

Cityform v1 used a hard-coded Environment Agency WCS path that only worked
inside England. v2 generalises this so different regions can plug in
different DEM providers behind a common interface, without the rest of the
pipeline (mesh build, OSM extraction, mesh emission) needing to know.

Status of each source as of 2026-05:

    EaWcsSource        — England, EA National LIDAR, 1 m + 2 m. PRODUCTION.
    WalesLidarSource   — DataMapWales WMS/WCS, 1 m. NOT YET IMPLEMENTED.
    ScotlandRsSource   — Scottish Remote Sensing Portal, multi-res. NOT YET.
    UsgsThreedepSource — USGS 3DEP, 1 m where available. NOT YET.
    CopernicusDemSource — Global GLO-30, 30 m, no DSM. NOT YET.

When a non-England bbox arrives today, ``pick_source`` raises
``NotImplementedError`` with a clear message; the auto_server endpoint
catches that and surfaces it as a user-friendly error event. The picker UI
also no longer hard-disables non-England bboxes — it shows a "limited
support" warning so the user can still make a UK-bounded selection.

Licences (all confirmed free for commercial use with attribution):
  EA          → OGL v3
  Wales       → OGL v3 (DataMapWales)
  Scotland    → OGL v3 (Scottish public sector LIDAR)
  USGS 3DEP   → US public domain
  Copernicus  → CC BY 4.0 (Copernicus open data)

Implementing a new source: subclass ``DemSource``, set ``target_crs``,
``has_dsm``, ``expected_resolution_m``, ``region_name``; implement
``fetch_dtm`` and ``fetch_dsm`` (return paths to GeoTIFFs already in
``target_crs``). Then add the source to ``pick_source``'s region routing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

# Approximate country bounding boxes (WGS84 lat/lng) — used for cheap region
# routing without a full GIS lookup. Overlaps at edges are resolved by the
# order in `pick_source` (EA tried first, etc.).
ENGLAND_BBOX = {"lat_min": 49.85, "lat_max": 55.85, "lng_min": -6.50, "lng_max":  1.85}
WALES_BBOX   = {"lat_min": 51.30, "lat_max": 53.50, "lng_min": -5.40, "lng_max": -2.65}
# Scotland southern boundary tightened to ~55.05 so Newcastle (54.97), Carlisle
# (54.89), and other northern English towns route to EA, not the (yet-unbuilt)
# Scottish source. Real border is squiggly; lat 55.05 is conservative.
SCOTLAND_BBOX = {"lat_min": 55.05, "lat_max": 60.90, "lng_min": -8.70, "lng_max": -0.70}
# Lower 48 (Alaska/Hawaii excluded for now — separate routing if/when added)
USA_BBOX     = {"lat_min": 24.40, "lat_max": 49.40, "lng_min": -125.00, "lng_max": -66.90}


def _bbox_inside(test: dict, region: dict) -> bool:
    """True iff every corner of `test` falls within `region`. Both WGS84.

    Conservative — partial overlaps fail. The user-facing picker can still
    show a warning for the partial-overlap case; pick_source needs a
    definitive answer.
    """
    return (
        test["lat_min"] >= region["lat_min"] and
        test["lat_max"] <= region["lat_max"] and
        test["lng_min"] >= region["lng_min"] and
        test["lng_max"] <= region["lng_max"]
    )


@runtime_checkable
class DemSource(Protocol):
    """Common interface for any DEM provider.

    Concrete sources must return GeoTIFFs already in ``target_crs`` (a
    metric projected CRS). The mesh builder works in metres regardless of
    which CRS — it just reads the raster's affine transform and the
    `centre_east` / `centre_north` it was given. So as long as the source
    pre-reprojects to a metric CRS, the rest of the pipeline keeps working
    unchanged.
    """

    target_crs: str               # e.g. "EPSG:27700" for BNG, "EPSG:32630" for UTM 30N
    region_name: str              # human-readable, used for UI badges + attribution
    has_dsm: bool                 # False for Copernicus (no DSM globally available)
    expected_resolution_m: float  # 1.0 for LIDAR, 30.0 for Copernicus

    def fetch_dtm(self, bng_e_min, bng_n_min, bng_e_max, bng_n_max) -> Path: ...
    def fetch_dsm(self, bng_e_min, bng_n_min, bng_e_max, bng_n_max) -> Optional[Path]: ...


class EaWcsSource:
    """Environment Agency LIDAR via OGC WCS. England only.

    Thin wrapper around the existing WCSFetcher in pipeline/wcs.py — the
    legacy code path is unchanged. New work routes through this class so
    auto_server can ask "give me a source for this bbox" without caring.

    For bboxes wider than ``LIDAR_2M_THRESHOLD_M`` (default 4 km) the
    1 m WCS endpoint typically rejects requests with HTTP 413/4096-pixel
    cap errors. The constructor accepts ``resolution_m=2.0`` to switch
    to the 2 m product family instead. auto_server picks this for big
    bboxes so the fetch actually succeeds.
    """

    target_crs = "EPSG:27700"
    region_name = "England (EA LIDAR)"
    has_dsm = True

    def __init__(self, cache_dir: Path, resolution_m: float = 1.0):
        # Import locally so `from pipeline.sources import DemSource` works
        # even without rasterio/requests installed (e.g. during tests).
        from .wcs import WCSFetcher
        if resolution_m not in (1.0, 2.0):
            raise ValueError(f"resolution_m must be 1.0 or 2.0, got {resolution_m}")
        self._fetcher = WCSFetcher(cache_dir=cache_dir)
        self.expected_resolution_m = resolution_m
        # Pick the matching WCS product IDs (defined in pipeline/wcs.py).
        self._dtm_product = "dtm_2m" if resolution_m == 2.0 else "dtm_1m"
        self._dsm_product = "dsm_2m_first" if resolution_m == 2.0 else "dsm_1m_first"

    def fetch_dtm(self, e_min, n_min, e_max, n_max) -> Path:
        return self._fetcher.fetch_geotiff(self._dtm_product, e_min, n_min, e_max, n_max)

    def fetch_dsm(self, e_min, n_min, e_max, n_max) -> Optional[Path]:
        return self._fetcher.fetch_geotiff(self._dsm_product, e_min, n_min, e_max, n_max)


# Stub classes — kept here so the routing function can reference them and
# raise a useful "not yet supported" error rather than silently mis-routing.
# To implement: replace `_NotYetImplementedSource` with a real class that
# does fetch+reproject for the given region's API.

class _NotYetImplementedSource:
    """Sentinel that raises with a helpful message when actually used.

    Concrete subclasses (Wales/Scotland/USGS/Copernicus) replace this when
    their fetch + reproject pipeline is implemented. Metadata attributes
    (region_name, has_dsm, expected_resolution_m) are set so log lines /
    UI badges can introspect the source without triggering the not-yet
    NotImplementedError — that fires only when a real fetch is attempted.
    """

    target_crs = "EPSG:27700"   # most concrete sources will be projected metric

    def __init__(self, region_name: str, license_text: str,
                 has_dsm: bool = True, expected_resolution_m: float = 1.0):
        self.region_name = region_name
        self.license_text = license_text
        self.has_dsm = has_dsm
        self.expected_resolution_m = expected_resolution_m

    def fetch_dtm(self, e_min, n_min, e_max, n_max) -> Path:
        raise NotImplementedError(
            f"{self.region_name} DEM source not yet implemented. "
            f"Planned licence: {self.license_text}. "
            f"Tracker: see Phase 8 in the project plan."
        )

    def fetch_dsm(self, e_min, n_min, e_max, n_max) -> Optional[Path]:
        raise NotImplementedError(
            f"{self.region_name} DSM source not yet implemented. "
            f"Planned licence: {self.license_text}."
        )


class WalesLidarSource:
    """Welsh Government / NRW LIDAR via DataMapWales WFS + Azure CDN.

    Backed by the Welsh Government LiDAR 2020-23 dataset (the post-Lle
    successor). 1 m resolution, EPSG:27700 native — drop-in compatible
    with EaWcsSource downstream of the pipeline. See pipeline/nrw.py for
    the WFS-catalogue → Azure-blob fetch pattern.

    Historic-archive fallback (1998-2019 captures, ZIP-bundled) is not
    yet implemented; bboxes outside the 2020-23 coverage raise
    FileNotFoundError so auto_server can surface a friendly
    "request via studio@cityform.co.uk" message.
    """

    target_crs = "EPSG:27700"
    region_name = "Wales (DataMapWales LIDAR)"
    has_dsm = True
    expected_resolution_m = 1.0

    def __init__(self, cache_dir: Path):
        from .nrw import NrwLidarFetcher
        self._fetcher = NrwLidarFetcher(cache_dir=cache_dir)

    def fetch_dtm(self, e_min, n_min, e_max, n_max) -> Path:
        return self._fetcher.fetch_geotiff("dtm_1m", e_min, n_min, e_max, n_max)

    def fetch_dsm(self, e_min, n_min, e_max, n_max) -> Optional[Path]:
        return self._fetcher.fetch_geotiff("dsm_1m_first", e_min, n_min, e_max, n_max)


class ScotlandRsSource:
    """Scottish Public Sector LIDAR via the AWS Open Data mirror of the
    Scottish Remote Sensing Portal (s3://srsp-open-data, eu-west-2).

    Walks a preference list of datasets per grid cell — national-lidar-
    programme (new 2025-27 50 cm) → phase 6/5/4/3 (50 cm) → phase 1/2
    (1 m) — and picks the first hit. EPSG:27700 native; drop-in
    compatible with EaWcsSource and WalesLidarSource downstream.

    Bboxes outside Scottish coverage raise FileNotFoundError. The picker
    coverage overlay should keep customers inside-bounds in practice.
    """

    target_crs = "EPSG:27700"
    region_name = "Scotland (Remote Sensing Portal)"
    has_dsm = True
    expected_resolution_m = 1.0  # nominal; actual varies per dataset (0.5-1 m)

    def __init__(self, cache_dir: Path):
        from .scotland import ScotlandRsFetcher
        self._fetcher = ScotlandRsFetcher(cache_dir=cache_dir)

    def fetch_dtm(self, e_min, n_min, e_max, n_max) -> Path:
        return self._fetcher.fetch_geotiff("dtm_1m", e_min, n_min, e_max, n_max)

    def fetch_dsm(self, e_min, n_min, e_max, n_max) -> Optional[Path]:
        return self._fetcher.fetch_geotiff("dsm_1m_first", e_min, n_min, e_max, n_max)


class UsgsThreedepSource(_NotYetImplementedSource):
    def __init__(self, cache_dir: Path):
        super().__init__("USA (USGS 3DEP)", "US public domain",
                         has_dsm=True, expected_resolution_m=1.0)


class CopernicusDemSource(_NotYetImplementedSource):
    def __init__(self, cache_dir: Path):
        # Copernicus is global but coarse and lacks DSM — measured roofs
        # aren't possible. The fallback path (OSM building heights → flat
        # prisms) is itself unimplemented; gating on has_dsm in
        # auto_server's _run_pipeline surfaces a friendly error instead.
        super().__init__("Global (Copernicus DEM 30 m)", "CC BY 4.0",
                         has_dsm=False, expected_resolution_m=30.0)


def pick_source(
    *, lat_min: float, lng_min: float, lat_max: float, lng_max: float,
    cache_root: Path,
    prefer_resolution_m: float = 1.0,
):
    """Pick the best DEM source for a WGS84 bbox.

    Returns a DemSource-compatible object (or raises NotImplementedError
    when the matching region isn't implemented yet). For partial-overlap
    bboxes (e.g. straddling the Severn estuary), preference order is
    England → Wales → Scotland → USA → Copernicus. Picker UI surfaces a
    warning for these edge cases.

    ``prefer_resolution_m`` lets the caller request 2 m LIDAR for big
    bboxes that the 1 m WCS endpoint won't serve. Currently honoured by
    EaWcsSource only — the other regions ignore it.
    """
    bbox = {"lat_min": lat_min, "lat_max": lat_max, "lng_min": lng_min, "lng_max": lng_max}

    # Per-source cache subdirectory keeps coverage IDs / GeoTIFFs from
    # different regions out of each other's way.
    cache_root = Path(cache_root)

    # Order matters: tighter bboxes first so a Cardiff bbox doesn't match
    # ENGLAND_BBOX before WALES_BBOX. England is the catch-all for the GB
    # mainland portion that isn't Wales/Scotland.
    if _bbox_inside(bbox, WALES_BBOX):
        return WalesLidarSource(cache_dir=cache_root / "wales")
    if _bbox_inside(bbox, SCOTLAND_BBOX):
        return ScotlandRsSource(cache_dir=cache_root / "scotland")
    if _bbox_inside(bbox, ENGLAND_BBOX):
        return EaWcsSource(cache_dir=cache_root / "ea",
                           resolution_m=prefer_resolution_m)
    if _bbox_inside(bbox, USA_BBOX):
        return UsgsThreedepSource(cache_dir=cache_root / "usgs")
    return CopernicusDemSource(cache_dir=cache_root / "copernicus")


# Per-source attribution snippets. Concatenate on Etsy listings / inserts
# alongside the existing OSM (ODbL) attribution.
ATTRIBUTION_LINES = {
    "England (EA LIDAR)":            "Contains Environment Agency data licensed under the Open Government Licence v3.0",
    "Wales (DataMapWales LIDAR)":    "Contains DataMapWales © Crown copyright and database right",
    "Scotland (Remote Sensing Portal)": "Contains Scottish public sector LIDAR © Crown copyright (OGL v3)",
    "USA (USGS 3DEP)":               "Contains USGS 3DEP elevation data (US public domain)",
    "Global (Copernicus DEM 30 m)":  "Contains modified Copernicus DEM data (CC BY 4.0)",
}
