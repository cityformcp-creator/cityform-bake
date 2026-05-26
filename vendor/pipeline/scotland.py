"""
Scottish Public Sector LIDAR fetcher.

Backed by the AWS Open Data mirror of the Scottish Remote Sensing Portal
at s3://srsp-open-data (eu-west-2). The bucket organises LIDAR by
delivery phase:

  national-lidar-programme/   The new 2025-27 BlueSky programme. 50 cm
                              res, 1 km × 1 km tiles. Highest quality
                              ("one of the highest resolution public
                              lidar in Europe"). Coverage building out
                              quarterly — ~15% of Scotland as of Jan 2026,
                              full coverage targeted by July 2027.
  phase-6                     50 cm res, 5 km quadrant tiles. Newer captures.
  phase-5, 4, 3               50 cm res, 5 km quadrant tiles.
  phase-2                     1 m res, 10 km tiles. DTM/DSM rasters OGL
                              (note: phase-2 LAZ point clouds are non-
                              commercial — we don't touch those here).
  phase-1                     1 m res, 10 km tiles. Oldest captures.
  hes, orkney-islands-council-23, outer-hebrides
                              Specialist datasets. Not consulted in this
                              MVP — main phases cover their footprints.

Fetch strategy: for each grid cell intersecting the requested bbox, walk
the dataset preference list (newest+highest-res first) and pick the first
one that exists on S3. HEAD probes are cached on disk per cell so re-runs
of adjacent bboxes are near-free.

Output GeoTIFFs are already in EPSG:27700 — drop-in compatible with EA
and NRW outputs. The pipeline downstream consumes them identically.

Licence: Open Government Licence v3.0. Attribution:
  "Contains Scottish public sector LIDAR © Crown copyright (OGL v3)"
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import requests


S3_BASE = "https://srsp-open-data.s3.eu-west-2.amazonaws.com"

# Dataset preference order — first hit wins per cell. Each entry is a
# tuple of (s3_prefix, tile_kind, filename_template, resolution_m).
# Tile kinds:
#   1km     — 4-digit grid ref (e.g. NT2773)
#   5km     — 4-digit grid ref + NE/NW/SE/SW (e.g. NT27NE)
#   10km    — 4-digit grid ref (letters + 2 digits, e.g. NT27)
DATASETS = [
    ("national-lidar-programme", "1km",  "{ref}_50cm_{kind}_ScotlandNationalLiDAR.tif", 0.5),
    ("phase-6",                  "5km",  "{ref}_50CM_{kind}_PHASE6.tif",                0.5),
    ("phase-5",                  "5km",  "{ref}_50CM_{kind}_PHASE5.tif",                0.5),
    ("phase-4",                  "5km",  "{ref}_50CM_{kind}_PHASE4.tif",                0.5),
    ("phase-3",                  "5km",  "{ref}_50CM_{kind}_PHASE3.tif",                0.5),
    ("phase-1",                  "10km", "{ref}_1M_{kind}_PHASE1.tif",                  1.0),
    ("phase-2",                  "10km", "{ref}_1M_{kind}_PHASE2.tif",                  1.0),
]

_TIMEOUT_SEC = 120

# Reuse the OS National Grid letter table from select_tiles. Defined
# here too so this module is independently importable for vendoring.
_NG_LETTERS = [
    ["SV", "SW", "SX", "SY", "SZ", "TV", "TW"],
    ["SQ", "SR", "SS", "ST", "SU", "TQ", "TR"],
    ["SL", "SM", "SN", "SO", "SP", "TL", "TM"],
    ["SF", "SG", "SH", "SJ", "SK", "TF", "TG"],
    ["SA", "SB", "SC", "SD", "SE", "TA", "TB"],
    ["NV", "NW", "NX", "NY", "NZ", "OV", "OW"],
    ["NQ", "NR", "NS", "NT", "NU", "OQ", "OR"],
    ["NL", "NM", "NN", "NO", "NP", "OL", "OM"],
    ["NF", "NG", "NH", "NJ", "NK", "OF", "OG"],
    ["NA", "NB", "NC", "ND", "NE", "OA", "OB"],
    ["HV", "HW", "HX", "HY", "HZ", "JV", "JW"],
    ["HQ", "HR", "HS", "HT", "HU", "JQ", "JR"],
    ["HL", "HM", "HN", "HO", "HP", "JL", "JM"],
]


def _bng_letters(easting: float, northing: float) -> Optional[str]:
    e_idx = int(easting // 100_000)
    n_idx = int(northing // 100_000)
    if not (0 <= e_idx < 7 and 0 <= n_idx < len(_NG_LETTERS)):
        return None
    return _NG_LETTERS[n_idx][e_idx]


def _ref_1km(easting: float, northing: float) -> Optional[str]:
    """Like 'NT2773' — 4 digits after the 100 km letters."""
    letters = _bng_letters(easting, northing)
    if not letters:
        return None
    e_km = int((easting % 100_000) // 1000)
    n_km = int((northing % 100_000) // 1000)
    return f"{letters}{e_km:02d}{n_km:02d}"


def _ref_5km_quadrant(easting: float, northing: float) -> Optional[str]:
    """Like 'NT27NE' — 10 km tile ref + NE/NW/SE/SW quadrant."""
    letters = _bng_letters(easting, northing)
    if not letters:
        return None
    e_10km = int((easting % 100_000) // 10_000)
    n_10km = int((northing % 100_000) // 10_000)
    # Position within the 10 km tile.
    e_off = (easting % 10_000)
    n_off = (northing % 10_000)
    ns = "N" if n_off >= 5000 else "S"
    ew = "E" if e_off >= 5000 else "W"
    return f"{letters}{e_10km}{n_10km}{ns}{ew}"


def _ref_10km(easting: float, northing: float) -> Optional[str]:
    """Like 'NT27' — letters + 2 digits."""
    letters = _bng_letters(easting, northing)
    if not letters:
        return None
    e_10km = int((easting % 100_000) // 10_000)
    n_10km = int((northing % 100_000) // 10_000)
    return f"{letters}{e_10km}{n_10km}"


def _cells_covering_bbox(
    e_min: float, n_min: float, e_max: float, n_max: float, kind: str,
) -> list[str]:
    """Enumerate grid-cell refs (1 km / 5 km / 10 km) covering the bbox.

    For a 1 km × 1 km bbox we usually get 1-4 cells per kind. The bake
    pipeline rarely asks for anything bigger.
    """
    if kind == "1km":
        step = 1000
        ref_fn = _ref_1km
    elif kind == "5km":
        step = 5000
        ref_fn = _ref_5km_quadrant
    elif kind == "10km":
        step = 10000
        ref_fn = _ref_10km
    else:
        raise ValueError(f"unknown tile kind {kind!r}")

    cells: set[str] = set()
    # Snap the bbox to the cell grid, then step through.
    e0 = (e_min // step) * step + step / 2
    n0 = (n_min // step) * step + step / 2
    e = e0
    while e <= e_max + step:
        n = n0
        while n <= n_max + step:
            r = ref_fn(e, n)
            if r:
                cells.add(r)
            n += step
        e += step
    return sorted(cells)


class ScotlandRsFetcher:
    """Fetches Scottish LiDAR from the AWS Open Data mirror.

    Public API mirrors WCSFetcher.fetch_geotiff / NrwLidarFetcher.fetch_geotiff
    — drop-in interchangeable behind pipeline/sources.py.
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._tile_cache = self.cache_dir / "tiles"
        self._tile_cache.mkdir(parents=True, exist_ok=True)
        # On-disk cache: {cell_ref → (dataset_prefix, filename)}, populated
        # by HEAD probes. Avoids re-probing on subsequent bakes.
        self._probe_cache_path = self.cache_dir / "_probe_cache.json"
        self._probe_cache: dict[str, list] = self._load_probe_cache()

    def _load_probe_cache(self) -> dict:
        if self._probe_cache_path.exists():
            try:
                return json.loads(self._probe_cache_path.read_text())
            except json.JSONDecodeError:
                return {}
        return {}

    def _save_probe_cache(self) -> None:
        self._probe_cache_path.write_text(json.dumps(self._probe_cache, indent=2))

    # ── public API ──────────────────────────────────────────────────────

    def fetch_geotiff(
        self,
        product: str,
        bng_e_min: float, bng_n_min: float,
        bng_e_max: float, bng_n_max: float,
    ) -> Path:
        """Return path to a mosaicked GeoTIFF covering the bbox.

        Walks the DATASETS preference list per grid cell and picks the
        first hit. Different cells within the same bbox may resolve to
        different datasets — rasterio.merge handles the resolution
        mismatch (resamples to the target's transform).
        """
        if product == "dtm_1m":
            kind = "DTM"
        elif product in ("dsm_1m_first", "dsm_1m"):
            kind = "DSM"
        else:
            raise ValueError(
                f"ScotlandRsFetcher supports 'dtm_1m' / 'dsm_1m_first' only; "
                f"got {product!r}"
            )

        e1, n1, e2, n2 = (int(round(v)) for v in
                          (bng_e_min, bng_n_min, bng_e_max, bng_n_max))
        mosaic_path = self.cache_dir / f"srsp_{kind.lower()}_{e1}_{n1}_{e2}_{n2}.tif"
        if mosaic_path.exists() and mosaic_path.stat().st_size > 1024:
            return mosaic_path

        # For each dataset (in preference order), download all tiles of its
        # kind that cover the bbox AND exist on S3. Stop once we have
        # enough tiles to cover the bbox (i.e. no further dataset is
        # needed for the remaining gaps).
        downloaded: list[Path] = []
        covered_cells_1km: set[str] = set()
        target_cells_1km = set(_cells_covering_bbox(bng_e_min, bng_n_min, bng_e_max, bng_n_max, "1km"))

        for prefix, dataset_kind, template, _res_m in DATASETS:
            if covered_cells_1km >= target_cells_1km:
                break
            cells = _cells_covering_bbox(
                bng_e_min, bng_n_min, bng_e_max, bng_n_max, dataset_kind,
            )
            for cell in cells:
                filename = template.format(ref=cell, kind=kind)
                url = f"{S3_BASE}/lidar/{prefix}/{kind.lower()}/27700/gridded/{filename}"
                tile_path = self._fetch_if_exists(url, cell, prefix, kind)
                if tile_path is None:
                    continue
                downloaded.append(tile_path)
                # Crude coverage tracking — mark all 1km cells within the
                # tile's footprint as covered. Approximate but good enough
                # to stop us pulling redundant lower-resolution tiles for
                # the same area.
                covered_cells_1km |= self._cells_in_tile(cell, dataset_kind)

        if not downloaded:
            raise FileNotFoundError(
                f"No Scottish LIDAR found for bbox ({bng_e_min:.0f},{bng_n_min:.0f}) → "
                f"({bng_e_max:.0f},{bng_n_max:.0f}) across {len(DATASETS)} datasets. "
                f"Likely outside Scottish coverage entirely. Route to email fulfilment."
            )

        self._mosaic_and_crop(
            downloaded, mosaic_path,
            bng_e_min, bng_n_min, bng_e_max, bng_n_max,
        )
        return mosaic_path

    # ── internals ───────────────────────────────────────────────────────

    def _cells_in_tile(self, ref: str, kind: str) -> set[str]:
        """Return the 1 km cells covered by a given tile ref."""
        # Parse ref back to BNG corner. Letters are 2 chars; the rest is
        # numeric (with optional NE/NW/SE/SW for 5 km).
        letters = ref[:2]
        e_idx, n_idx = None, None
        for ni, row in enumerate(_NG_LETTERS):
            for ei, lt in enumerate(row):
                if lt == letters:
                    e_idx, n_idx = ei, ni
                    break
            if e_idx is not None:
                break
        if e_idx is None:
            return set()
        base_e = e_idx * 100_000
        base_n = n_idx * 100_000
        rest = ref[2:]
        if kind == "10km":
            e_km = int(rest[:1]) * 10
            n_km = int(rest[1:2]) * 10
            tile_size_km = 10
        elif kind == "5km":
            e_km = int(rest[:1]) * 10
            n_km = int(rest[1:2]) * 10
            quad = rest[2:]
            if "E" in quad:
                e_km += 5
            if "N" in quad:
                n_km += 5
            tile_size_km = 5
        else:  # 1 km
            e_km = int(rest[:2])
            n_km = int(rest[2:4])
            tile_size_km = 1
        out: set[str] = set()
        for de in range(tile_size_km):
            for dn in range(tile_size_km):
                e = base_e + (e_km + de) * 1000 + 500
                n = base_n + (n_km + dn) * 1000 + 500
                r = _ref_1km(e, n)
                if r:
                    out.add(r)
        return out

    def _fetch_if_exists(
        self, url: str, cell: str, prefix: str, kind: str,
    ) -> Optional[Path]:
        """Download URL if it exists on S3; return None on 404. Cached
        per-URL on disk so re-runs skip the round-trip entirely.

        Probe-cache stores a per-cell hit/miss across datasets so adjacent
        bbox queries don't re-probe every dataset for known-missing cells.
        """
        # Negative cache — known 404 from a prior probe.
        miss_key = f"{prefix}|{kind}|{cell}|miss"
        if miss_key in self._probe_cache:
            return None

        filename = url.rsplit("/", 1)[-1]
        local = self._tile_cache / filename
        if local.exists() and local.stat().st_size > 1024:
            return local

        # Stream-download; rely on S3 returning 404 for nonexistent keys.
        resp = requests.get(url, stream=True, timeout=_TIMEOUT_SEC)
        if resp.status_code == 404:
            self._probe_cache[miss_key] = True
            # Save occasionally — every 25 misses keeps I/O bounded.
            if len(self._probe_cache) % 25 == 0:
                self._save_probe_cache()
            resp.close()
            return None
        if resp.status_code != 200:
            resp.close()
            raise RuntimeError(
                f"Scottish LiDAR S3 returned HTTP {resp.status_code} for {url}"
            )

        tmp = local.with_suffix(local.suffix + ".tmp")
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=64 * 1024):
                f.write(chunk)
        tmp.rename(local)
        print(f"  SRSP downloaded {prefix}/{filename} ({local.stat().st_size // 1024} KB)")
        return local

    def _mosaic_and_crop(
        self, tile_paths: list[Path], out_path: Path,
        bng_e_min: float, bng_n_min: float,
        bng_e_max: float, bng_n_max: float,
    ) -> None:
        """Mosaic + crop to requested bbox. Mixed-resolution inputs are
        resampled to the first input's transform (rasterio.merge default
        behaviour — fine for our use case where tiles overlap rather than
        tile-mismatch)."""
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
        # Persist probe cache after a successful mosaic run.
        self._save_probe_cache()

        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        with rasterio.open(tmp, "w", **out_meta) as dest:
            dest.write(mosaic_arr)
        tmp.rename(out_path)
        print(
            f"  SRSP mosaicked {len(tile_paths)} tiles → {out_path.name} "
            f"({out_path.stat().st_size / 1e6:.1f} MB)"
        )
