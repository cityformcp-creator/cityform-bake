"""
Tier-3 STL builder with hollow water cutouts.

Variant of tier3_measured_roofs.build_tier3_stl that takes a list of OSM
water polygons (in BNG) and punches them clean through the model:
top quads, bottom quads, and the four outer walls are skipped where they
fall over water; new inner walls are emitted along every land/water cell
boundary so the model stays solid where it should be solid.

A cell is classified as "water" when at least 3 of its 4 corner vertices
fall inside any water polygon — conservative, so narrow rivers may not cut
at the default 2 m mesh resolution.

Public API:

    build_tier3_with_water_stl(
        dsm_path, dtm_path,
        centre_east, centre_north, size_m,
        print_w_mm=90.0, plinth_mm=2.0, z_exaggeration=1.0,
        print_margin_mm=0.5,        # actual XY footprint = print_w_mm - this
        water_polygons_bng=None,    # list[shapely.Polygon] in EPSG:27700
        out_path="output.stl",
    ) -> dict        # summary stats: triangles, water_cells, size_mb
"""

import math
import re
import struct
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds as window_from_bounds
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.features import geometry_mask
from scipy import ndimage
from shapely.affinity import rotate as shp_rotate, translate as shp_translate
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon, box, shape
from shapely.ops import polygonize, unary_union
from pyproj import Transformer


_TX_WGS84_TO_BNG = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)


# ─── Public helpers for OSM → BNG polygon conversion ─────────────────────

def osm_features_to_bng_polygons(features: list[dict]) -> list[Polygon]:
    """Flatten an Overpass GeoJSON FeatureCollection's features (WGS84) into
    a list of shapely Polygons in BNG (EPSG:27700)."""
    out: list[Polygon] = []
    for f in features:
        geom_dict = f.get("geometry")
        if not geom_dict:
            continue
        try:
            geom = shape(geom_dict)
        except Exception:
            continue
        if geom.is_empty:
            continue
        if isinstance(geom, Polygon):
            out.append(_polygon_wgs_to_bng(geom))
        elif isinstance(geom, MultiPolygon):
            out.extend(_polygon_wgs_to_bng(p) for p in geom.geoms)
    # Drop invalid/empty after reprojection
    return [p for p in out if not p.is_empty and p.is_valid]


def _polygon_wgs_to_bng(p: Polygon) -> Polygon:
    def tx(coords):
        return [_TX_WGS84_TO_BNG.transform(x, y) for x, y in coords]
    try:
        return Polygon(tx(p.exterior.coords), [tx(r.coords) for r in p.interiors])
    except Exception:
        return Polygon()


# ─── OSM building:part / S3DB parsing ────────────────────────────────────

_HEIGHT_RE = re.compile(r"^\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*(.*?)\s*$")
_VALID_ROOF_SHAPES = {"flat", "pyramidal", "skillion", "hipped", "gabled"}
_KNOWN_ROOF_SHAPES = _VALID_ROOF_SHAPES | {
    "dome", "round", "onion", "half-hipped", "gambrel", "mansard", "saltbox",
}


def _parse_osm_height_m(raw) -> float | None:
    """Parse an OSM height tag string into metres.

    Handles ``"50"``, ``"50.5"``, ``"50 m"``, ``"50m"``, ``"395'"``,
    ``"395 ft"``, ``"-.2"``. Returns None for unparseable, missing, or
    non-finite values. Negative or zero heights are returned as-is so the
    caller can decide whether to drop them (basement parts).
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    m = _HEIGHT_RE.match(s)
    if not m:
        return None
    try:
        n = float(m.group(1))
    except ValueError:
        return None
    if not math.isfinite(n):
        return None
    unit = m.group(2).lower().strip()
    if unit in ("", "m", "metre", "metres", "meter", "meters"):
        return n
    if unit in ("ft", "feet", "foot", "'"):
        return n * 0.3048
    # Unknown unit — treat as metres rather than dropping (most OSM
    # tags omit units, and the few odd ones are usually metric anyway).
    return n


def osm_building_parts_to_bng(features: list[dict]) -> list[dict]:
    """Parse OSM building:part features into BNG polygons with metadata.

    Returns a list of dicts: ``{polygon, height_m, min_height_m,
    roof_shape, roof_height_m, roof_direction_deg, raw_shape}``.
    Polygons with non-positive ``height``, unparseable tags, or invalid
    geometry are dropped silently. ``raw_shape`` carries the original
    tag (or empty string) so callers can log unsupported shapes.
    """
    out: list[dict] = []
    for f in features:
        geom_dict = f.get("geometry")
        if not geom_dict:
            continue
        try:
            geom = shape(geom_dict)
        except Exception:
            continue
        if geom.is_empty:
            continue
        props = f.get("properties") or {}
        height_m = _parse_osm_height_m(props.get("height"))
        if height_m is None or height_m <= 0:
            continue
        min_height_m = _parse_osm_height_m(props.get("min_height")) or 0.0
        if min_height_m < 0:
            min_height_m = 0.0
        # Cap min_height at height (defensive, avoids inverted parts).
        if min_height_m >= height_m:
            continue
        raw_shape = str(props.get("roof:shape") or "").strip().lower()
        roof_shape = raw_shape if raw_shape in _VALID_ROOF_SHAPES else "flat"
        roof_height_m = _parse_osm_height_m(props.get("roof:height"))
        if roof_height_m is None or roof_height_m <= 0:
            # Default roof_height depends on the shape and whether the
            # part has a min_height. With min_height>0, the part is
            # "stacked on" something below it and the standard reading
            # applies (vertical walls, then a small sloped cap). With
            # min_height=0 AND a non-flat shape, the polygon often
            # represents an architectural lean (Shard-style faceted
            # skillion or pyramidal spire) — treat the entire extent
            # as the slope so the print captures the taper.
            if roof_shape in ("pyramidal", "skillion", "hipped", "gabled") and min_height_m == 0:
                roof_height_m = height_m
            else:
                roof_height_m = min(5.0, max(0.5, (height_m - min_height_m) / 8.0))
        # Clamp roof_height to the available headroom above min_height.
        roof_height_m = min(roof_height_m, height_m - min_height_m)
        try:
            rd = props.get("roof:direction")
            roof_direction_deg = float(rd) if rd not in (None, "") else None
        except (TypeError, ValueError):
            roof_direction_deg = None
        polys: list[Polygon] = []
        if isinstance(geom, Polygon):
            polys.append(_polygon_wgs_to_bng(geom))
        elif isinstance(geom, MultiPolygon):
            polys.extend(_polygon_wgs_to_bng(p) for p in geom.geoms)
        for p in polys:
            if p.is_empty or not p.is_valid:
                continue
            out.append({
                "polygon": p,
                "height_m": float(height_m),
                "min_height_m": float(min_height_m),
                "roof_shape": roof_shape,
                "roof_height_m": float(roof_height_m),
                "roof_direction_deg": roof_direction_deg,
                "raw_shape": raw_shape,
            })
    return out


def promote_envelope_parts(parts: list[dict], coverage: float = 0.5) -> tuple[list[dict], list[dict]]:
    """Promote untagged-roof-shape "envelope" parts to inherit the shape
    of the most prominent shape-tagged part inside them.

    OSM S3DB convention often includes a coarse flat-topped polygon for
    the whole building at full height as a fallback for renderers that
    don't understand parts. The actual taper is encoded in smaller
    skillion/pyramidal/etc. polygons covering sub-regions. With
    max-z-per-pixel rendering and a 2D height field, those small
    sub-parts can't reach far enough out to taper the envelope's outer
    edge — so the envelope renders as a rectangular tower.

    Solution: when an envelope's footprint is mostly covered by a single
    shape-tagged sub-part (pyramidal/skillion/hipped/gabled) at the same
    height, copy that sub-part's shape onto the envelope and switch to
    full-extent roof_height. The envelope then renders with the proper
    taper across its FULL footprint, not just the sub-part's small
    region. The original sub-parts are kept too — max-z lets them add
    further detail wherever they reach.

    A part is a promotion candidate iff it has ``raw_shape == ""`` (the
    OSM `roof:shape` tag was missing). The donor is the largest non-flat
    sub-part at the same height that intersects ≥``coverage`` of the
    envelope's footprint.

    Returns (parts_after_promotion, list_of_promoted_dicts_for_logging).
    """
    out: list[dict] = []
    promoted_log: list[dict] = []
    for p in parts:
        if p["raw_shape"] != "":
            out.append(p)
            continue
        try:
            p_area = p["polygon"].area
        except Exception:
            p_area = 0.0
        if p_area <= 0:
            out.append(p)
            continue
        best_donor = None
        best_inter_area = 0.0
        for q in parts:
            if q is p or q["raw_shape"] == "":
                continue
            if q["roof_shape"] not in ("pyramidal", "skillion", "hipped", "gabled"):
                continue
            if abs(q["height_m"] - p["height_m"]) > 1.0:
                continue
            try:
                inter = p["polygon"].intersection(q["polygon"])
            except Exception:
                continue
            if inter.is_empty:
                continue
            if inter.area > best_inter_area:
                best_inter_area = inter.area
                best_donor = q
        if best_donor is not None and best_inter_area >= coverage * p_area:
            promoted = dict(p)
            promoted["roof_shape"] = best_donor["roof_shape"]
            promoted["raw_shape"] = f"(promoted from envelope → {best_donor['roof_shape']})"
            promoted["roof_height_m"] = promoted["height_m"] - promoted["min_height_m"]
            promoted["roof_direction_deg"] = best_donor.get("roof_direction_deg")
            out.append(promoted)
            promoted_log.append({
                "area": p_area, "height": p["height_m"],
                "donor_shape": best_donor["roof_shape"],
                "donor_area": best_donor["polygon"].area,
            })
        else:
            out.append(p)
    return out, promoted_log


def drop_envelope_parts(parts: list[dict], coverage: float = 0.5) -> tuple[list[dict], list[dict]]:
    """Legacy alternative to promote_envelope_parts: drop the untagged
    envelope rather than promoting its shape. Kept as a strict dropping
    helper for callers that prefer to see only shape-tagged geometry.
    """
    kept: list[dict] = []
    dropped: list[dict] = []
    for p in parts:
        if p["raw_shape"] != "":
            kept.append(p)
            continue
        is_envelope = False
        try:
            p_area = p["polygon"].area
        except Exception:
            p_area = 0.0
        if p_area <= 0:
            kept.append(p)
            continue
        for q in parts:
            if q is p or q["raw_shape"] == "":
                continue
            if q["height_m"] < p["height_m"] - 1.0:
                continue
            try:
                inter = p["polygon"].intersection(q["polygon"])
            except Exception:
                continue
            if inter.is_empty:
                continue
            if inter.area > coverage * p_area:
                is_envelope = True
                break
        if is_envelope:
            dropped.append(p)
        else:
            kept.append(p)
    return kept, dropped


def detect_overhanging_parts(parts: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split parts into (kept, dropped). A part with min_height>0 whose
    footprint isn't fully covered by the union of lower parts (any part
    with height>=this.min_height) is considered overhanging — the
    height-field mesh can't represent the empty space beneath it without
    artefacts, so it's dropped.

    Walkie-Talkie's bulge-top polygon is the canonical case.
    """
    kept: list[dict] = []
    dropped: list[dict] = []
    for p in parts:
        mh = p["min_height_m"]
        if mh <= 0:
            kept.append(p)
            continue
        # Candidates: every other part whose top reaches at least this
        # part's base AND whose footprint touches this part's footprint.
        supporters = []
        for q in parts:
            if q is p:
                continue
            if q["height_m"] < mh - 1e-3:
                continue
            if q["polygon"].intersects(p["polygon"]):
                supporters.append(q["polygon"])
        if not supporters:
            dropped.append(p)
            continue
        try:
            support_union = unary_union(supporters)
        except Exception:
            dropped.append(p)
            continue
        # Use buffer(0) on this part's polygon to clean any tiny
        # self-intersection issues before the within() test.
        try:
            this_poly = p["polygon"].buffer(0)
        except Exception:
            this_poly = p["polygon"]
        if this_poly.within(support_union.buffer(0.5)):
            kept.append(p)
        else:
            dropped.append(p)
    return kept, dropped


def render_part_roof_z(
    part_mask: np.ndarray,
    dtm: np.ndarray,
    height_m: float,
    min_height_m: float,
    roof_shape: str,
    roof_height_m: float,
    roof_direction_deg: float | None,
    pixel_xy_local: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Compute per-pixel target z (absolute elevation, metres) for a
    building:part footprint. Returns an array shaped like ``part_mask``;
    only values inside the mask are meaningful (others are 0).

    ``dtm`` is the raw terrain model in absolute metres. Roof shapes
    other than ``flat`` taper between ``height - roof_height`` (the eave
    line) and ``height`` (the peak) according to a per-shape formula.
    """
    base_eave_z = dtm + (height_m - roof_height_m)
    target = np.zeros_like(dtm, dtype=np.float32)
    if not part_mask.any():
        return target

    def _finalise(arr):
        return np.where(part_mask, arr, 0.0).astype(np.float32)

    if roof_shape == "flat" or roof_height_m <= 0:
        return _finalise(dtm + height_m)

    if roof_shape == "pyramidal":
        # Distance from each pixel to the nearest non-mask pixel.
        # Peak is the pixel furthest from the edge.
        dist = ndimage.distance_transform_edt(part_mask).astype(np.float32)
        max_d = float(dist[part_mask].max()) if part_mask.any() else 0.0
        if max_d <= 0:
            return _finalise(dtm + height_m)
        ratio = np.where(part_mask, dist / max_d, 0.0)
        return _finalise(base_eave_z + roof_height_m * ratio)

    if roof_shape == "hipped":
        # Same shape as pyramidal but the "peak" is a ridge along the
        # part's longest axis. Approximated as: skeleton-like distance
        # transform along the part's ridge gives a high plateau, falling
        # off equally on all sides. For a simple v1, we re-use the
        # pyramidal taper but cap the top to a flat ridge of width = half
        # the inscribed-circle radius.
        dist = ndimage.distance_transform_edt(part_mask).astype(np.float32)
        max_d = float(dist[part_mask].max()) if part_mask.any() else 0.0
        if max_d <= 0:
            return _finalise(dtm + height_m)
        ridge_width_d = max_d * 0.5
        # Below ridge: linear taper to the eave; above: flat at peak.
        ratio = np.where(
            dist >= ridge_width_d,
            1.0,
            dist / ridge_width_d,
        )
        ratio = np.where(part_mask, ratio, 0.0).astype(np.float32)
        return _finalise(base_eave_z + roof_height_m * ratio)

    if roof_shape in ("skillion", "gabled"):
        # Need a per-pixel position along the slope direction.
        if pixel_xy_local is None:
            return _finalise(dtm + height_m)
        xs_local, ys_local = pixel_xy_local
        # Default direction: 0° = north (+y axis); mirrors OSM convention.
        # If untagged, use the part's longest extent direction.
        if roof_direction_deg is None:
            ys_p = ys_local[part_mask]
            xs_p = xs_local[part_mask]
            if ys_p.size == 0:
                return _finalise(dtm + height_m)
            span_y = float(ys_p.max() - ys_p.min())
            span_x = float(xs_p.max() - xs_p.min())
            roof_direction_deg = 0.0 if span_y >= span_x else 90.0
        rad = math.radians(roof_direction_deg)
        # Project each pixel onto the slope axis (unit vector pointing to
        # the high side of a skillion, or perpendicular to the gabled ridge).
        # OSM convention: roof:direction is the compass bearing FROM
        # which the high edge is visible — i.e., direction the slope FACES.
        ux, uy = math.sin(rad), math.cos(rad)
        proj = xs_local * ux + ys_local * uy
        proj_in = proj[part_mask]
        if proj_in.size == 0:
            return _finalise(dtm + height_m)
        lo, hi = float(proj_in.min()), float(proj_in.max())
        span = max(hi - lo, 1e-6)
        if roof_shape == "skillion":
            ratio = np.where(part_mask, (proj - lo) / span, 0.0).astype(np.float32)
        else:  # gabled
            ridge = (lo + hi) * 0.5
            d_from_ridge = np.abs(proj - ridge)
            half_span = span * 0.5
            ratio = np.where(
                part_mask,
                1.0 - d_from_ridge / max(half_span, 1e-6),
                0.0,
            ).astype(np.float32)
        ratio = np.clip(ratio, 0.0, 1.0)
        return _finalise(base_eave_z + roof_height_m * ratio)

    # Unrecognised shape — treat as flat (callers log the count).
    return _finalise(dtm + height_m)


def _rotate_part_polygons_to_local(
    parts: list[dict], centre_east: float, centre_north: float, angle_deg: float,
) -> list[dict]:
    """Apply the same local-frame rotation used by `_rotate_polygons_to_local`
    to a list of building:part dicts (preserving all metadata).
    """
    if abs(angle_deg) < 1e-6 or not parts:
        return parts
    out = []
    for p in parts:
        try:
            translated = shp_translate(p["polygon"], xoff=-centre_east, yoff=-centre_north)
            rotated = shp_rotate(translated, -angle_deg, origin=(0, 0))
        except Exception:
            continue
        if rotated.is_empty or not rotated.is_valid:
            continue
        q = dict(p)
        q["polygon"] = rotated
        # Rotate roof:direction too (it's a compass bearing).
        if q.get("roof_direction_deg") is not None:
            q["roof_direction_deg"] = (q["roof_direction_deg"] - angle_deg) % 360.0
        out.append(q)
    return out


def osm_road_features_to_bng_geoms(
    features: list[dict], default_buffer_m: float = 2.0
) -> list[Polygon]:
    """Flatten OSM road features (WGS84) into BNG buffered Polygons.

    Roads are LineStrings that need a width to rasterise. Major roads are
    buffered wider so the road hierarchy reads on the print. Tags considered:
    motorway / trunk / primary → 3.0 m, secondary / tertiary → 2.5 m,
    everything else → ``default_buffer_m`` (default 2.0 m).
    """
    out: list[Polygon] = []
    for f in features:
        geom_dict = f.get("geometry")
        if not geom_dict:
            continue
        try:
            geom = shape(geom_dict)
        except Exception:
            continue
        if geom.is_empty:
            continue
        hwy = (f.get("properties") or {}).get("highway") or ""
        if hwy in ("motorway", "trunk", "primary"):
            buf = 3.0
        elif hwy in ("secondary", "tertiary"):
            buf = 2.5
        else:
            buf = default_buffer_m
        if isinstance(geom, LineString):
            out.append(_linestring_wgs_to_bng_buffered(geom, buf))
        elif isinstance(geom, MultiLineString):
            out.extend(_linestring_wgs_to_bng_buffered(ls, buf) for ls in geom.geoms)
        # Ignore Polygon/MultiPolygon — roads don't ship as areas in OSM
    return [p for p in out if p is not None and not p.is_empty and p.is_valid
            and isinstance(p, Polygon)]


def osm_bridge_features_to_bng_geoms(
    features: list[dict], line_buffer_m: float = 4.0
) -> list[Polygon]:
    """Flatten OSM bridge features (WGS84) into BNG polygons.

    Bridges are usually mapped as LineStrings (the way that carries the road
    over a feature). They have no rasterisable area, so each LineString is
    buffered by ``line_buffer_m`` (≈ deck half-width) before being returned.
    Closed Polygons / MultiPolygons (rare for bridges, e.g. ``man_made=bridge``
    deck outlines) pass through unchanged.
    """
    out: list[Polygon] = []
    for f in features:
        geom_dict = f.get("geometry")
        if not geom_dict:
            continue
        try:
            geom = shape(geom_dict)
        except Exception:
            continue
        if geom.is_empty:
            continue
        if isinstance(geom, Polygon):
            out.append(_polygon_wgs_to_bng(geom))
        elif isinstance(geom, MultiPolygon):
            out.extend(_polygon_wgs_to_bng(p) for p in geom.geoms)
        elif isinstance(geom, LineString):
            out.append(_linestring_wgs_to_bng_buffered(geom, line_buffer_m))
        elif isinstance(geom, MultiLineString):
            out.extend(
                _linestring_wgs_to_bng_buffered(ls, line_buffer_m) for ls in geom.geoms
            )
    return [p for p in out if p is not None and not p.is_empty and p.is_valid
            and isinstance(p, Polygon)]


def osm_railway_features_to_bng_geoms(
    features: list[dict], default_buffer_m: float = 2.5
) -> list[Polygon]:
    """Flatten OSM railway features (WGS84) into BNG buffered Polygons.

    Railways are LineStrings buffered to a uniform width. The default
    ``default_buffer_m`` of 2.5 m produces a 5 m corridor — slightly wider
    than residential roads so rail lines read distinctly on the print.
    """
    out: list[Polygon] = []
    for f in features:
        geom_dict = f.get("geometry")
        if not geom_dict:
            continue
        try:
            geom = shape(geom_dict)
        except Exception:
            continue
        if geom.is_empty:
            continue
        if isinstance(geom, LineString):
            out.append(_linestring_wgs_to_bng_buffered(geom, default_buffer_m))
        elif isinstance(geom, MultiLineString):
            out.extend(
                _linestring_wgs_to_bng_buffered(ls, default_buffer_m) for ls in geom.geoms
            )
    return [p for p in out if p is not None and not p.is_empty and p.is_valid
            and isinstance(p, Polygon)]


def osm_park_features_to_bng_geoms(
    features: list[dict], boundary_buffer_m: float = 1.5,
    min_area_m2: float = 500.0,
) -> list[Polygon]:
    """Flatten OSM park/green-space features into BNG boundary-outline Polygons.

    Only the polygon *perimeter* is returned (buffered to ``boundary_buffer_m``),
    not the filled area — engraving the entire park would flatten all terrain
    inside. Features smaller than ``min_area_m2`` are dropped to avoid clutter
    on small prints.
    """
    out: list[Polygon] = []
    for f in features:
        geom_dict = f.get("geometry")
        if not geom_dict:
            continue
        try:
            geom = shape(geom_dict)
        except Exception:
            continue
        if geom.is_empty or not isinstance(geom, (Polygon, MultiPolygon)):
            continue
        polys = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
        for poly in polys:
            bng_poly = _polygon_wgs_to_bng(poly)
            if bng_poly is None or bng_poly.is_empty or not bng_poly.is_valid:
                continue
            if bng_poly.area < min_area_m2:
                continue
            # Buffer just the boundary ring into a thin outline polygon
            boundary_line = LineString(bng_poly.exterior.coords)
            outline = boundary_line.buffer(boundary_buffer_m, cap_style=2)
            if outline.is_valid and not outline.is_empty and isinstance(outline, Polygon):
                out.append(outline)
    return out


def _linestring_wgs_to_bng_buffered(ls: LineString, buffer_m: float):
    """LineString in WGS84 → BNG, then buffered into a Polygon."""
    try:
        coords_bng = [_TX_WGS84_TO_BNG.transform(x, y) for x, y in ls.coords]
        if len(coords_bng) < 2:
            return None
        line_bng = LineString(coords_bng)
        return line_bng.buffer(buffer_m, cap_style="flat")
    except Exception:
        return None


def _linestring_wgs_to_bng(ls: LineString):
    """LineString in WGS84 → LineString in BNG (no buffering)."""
    try:
        coords_bng = [_TX_WGS84_TO_BNG.transform(x, y) for x, y in ls.coords]
        if len(coords_bng) < 2:
            return None
        return LineString(coords_bng)
    except Exception:
        return None


def osm_coastline_features_to_bng_lines(features: list[dict]) -> list[LineString]:
    """OSM `natural=coastline` features (WGS84 LineStrings) → BNG LineStrings.

    Coastlines are unique in OSM: they're stored as LineStrings, not closed
    Polygons. The convention is `water on the right of the line`. The
    polygonize step (below) turns them into sea Polygons by closing each
    line against the bbox boundary and classifying each enclosed region.
    """
    out: list[LineString] = []
    for f in features or []:
        geom_dict = f.get("geometry")
        if not geom_dict:
            continue
        try:
            geom = shape(geom_dict)
        except Exception:
            continue
        if geom.is_empty:
            continue
        if isinstance(geom, LineString):
            ls = _linestring_wgs_to_bng(geom)
            if ls is not None and not ls.is_empty:
                out.append(ls)
        elif isinstance(geom, MultiLineString):
            for g in geom.geoms:
                ls = _linestring_wgs_to_bng(g)
                if ls is not None and not ls.is_empty:
                    out.append(ls)
    return out


def polygonize_coastline_to_sea(
    lines: list[LineString],
    bbox: tuple[float, float, float, float],
) -> list[Polygon]:
    """Build sea Polygons from a set of OSM coastline LineStrings.

    `lines`: LineStrings in any planar CRS (typically BNG for the builder,
    or WGS84 for the picker's water preview).
    `bbox` : (west, south, east, north) in the same CRS as `lines`.

    Algorithm (proven by `test_coastline_polygonize.py`):
      1. Clip every line to the bbox.
      2. `unary_union` the clipped lines with the bbox boundary so every
         enclosed region has a closed boundary.
      3. `shapely.ops.polygonize` → enclosed regions.
      4. For each region, walk its CCW exterior. For every boundary segment
         that lies on a coastline (rather than the bbox edge), step a tiny
         distance into the polygon interior and run the standard
         cross-product side test against the nearest original coastline
         segment. Per OSM wiki: water-on-right means cross<0 → SEA, cross>0
         → LAND. Tally votes across all coastline-edge segments — works
         even for thin slivers near the bbox boundary.

    Returns a list of sea-side Polygons in the input CRS. Land-side regions
    are dropped (the builder doesn't need them — the LIDAR DTM already
    represents land).
    """
    if not lines:
        return []
    bbox_poly = box(*bbox)

    clipped: list[LineString] = []
    for ls in lines:
        try:
            cut = ls.intersection(bbox_poly)
        except Exception:
            continue
        if cut.is_empty:
            continue
        if isinstance(cut, LineString):
            clipped.append(cut)
        elif isinstance(cut, MultiLineString):
            clipped.extend(g for g in cut.geoms if not g.is_empty)
    if not clipped:
        return []

    # Pre-explode coastline segments for the side test
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for ls in clipped:
        coords = list(ls.coords)
        for i in range(len(coords) - 1):
            segments.append((coords[i], coords[i + 1]))
    if not segments:
        return []

    coast_geom = unary_union(clipped)
    all_lines = unary_union([*clipped, bbox_poly.boundary])
    regions = list(polygonize(all_lines))
    if not regions:
        return []

    # Tolerance for "this boundary segment lies on a coastline (vs the
    # bbox edge)". Relative to the bbox dimension so it works in metres
    # (BNG) and degrees (WGS84) alike.
    coord_scale = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    coast_eps = coord_scale * 1e-7

    def _nearest_segment_cross(sx: float, sy: float) -> float:
        best = None
        best_d2 = float("inf")
        for (ax, ay), (bx, by) in segments:
            dx, dy = bx - ax, by - ay
            seg_len2 = dx * dx + dy * dy
            if seg_len2 < 1e-30:
                t = 0.0
            else:
                t = max(0.0, min(1.0, ((sx - ax) * dx + (sy - ay) * dy) / seg_len2))
            cx, cy = ax + t * dx, ay + t * dy
            d2 = (sx - cx) ** 2 + (sy - cy) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best = ((ax, ay), (bx, by))
        if best is None:
            return 0.0
        (ax, ay), (bx, by) = best
        return (bx - ax) * (sy - ay) - (by - ay) * (sx - ax)

    sea_polys: list[Polygon] = []
    for region in regions:
        if region.is_empty or region.area < 1e-12:
            continue
        ring = (region.exterior if region.exterior.is_ccw
                else LineString(list(region.exterior.coords)[::-1]))
        coords = list(ring.coords)
        sea_votes = 0
        land_votes = 0
        for i in range(len(coords) - 1):
            p0, p1 = coords[i], coords[i + 1]
            mid = Point((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
            if mid.distance(coast_geom) > coast_eps:
                continue
            dx = p1[0] - p0[0]
            dy = p1[1] - p0[1]
            mag = (dx * dx + dy * dy) ** 0.5
            if mag < 1e-12:
                continue
            eps_step = mag * 0.001
            sx = mid.x - dy * eps_step / mag
            sy = mid.y + dx * eps_step / mag
            cross = _nearest_segment_cross(sx, sy)
            if cross > 0:
                land_votes += 1
            elif cross < 0:
                sea_votes += 1

        if sea_votes == 0 and land_votes == 0:
            sample = region.representative_point()
            if _nearest_segment_cross(sample.x, sample.y) < 0:
                sea_polys.append(region)
        elif sea_votes > land_votes:
            sea_polys.append(region)
        # else: land — drop

    return sea_polys


def coastline_features_to_sea_polygons(
    features: list[dict],
    centre_east: float,
    centre_north: float,
    aabb_size_m: float,
) -> list[Polygon]:
    """Convenience wrapper for the auto_server pipeline. Takes Overpass
    coastline features (WGS84) + the BNG-aligned AABB the builder will
    fetch LIDAR for, returns sea Polygons in BNG ready to union into
    `water_polygons_bng`.
    """
    lines = osm_coastline_features_to_bng_lines(features)
    if not lines:
        return []
    half = aabb_size_m / 2.0
    bbox = (
        centre_east - half, centre_north - half,
        centre_east + half, centre_north + half,
    )
    return polygonize_coastline_to_sea(lines, bbox)


# ─── Internal raster helpers ─────────────────────────────────────────────

def _crop_raster(path, bbox):
    with rasterio.open(path) as src:
        win = window_from_bounds(*bbox, transform=src.transform)
        arr = src.read(1, window=win)
        if src.nodata is not None:
            arr = np.where(arr == src.nodata, np.nan, arr)
    return arr


def _crop_raster_rotated(path, centre_east, centre_north, size_m, angle_deg, out_size):
    """Crop and resample a raster onto a rotated square grid centred at
    (centre_east, centre_north) with side length ``size_m`` and clockwise
    rotation ``angle_deg`` from north. Returns a (out_size, out_size) array
    where row 0 is the top edge of the rotated square (high local v) and
    column 0 is the left edge (low local u).

    The rotated square is sampled from the source raster via bilinear
    interpolation (scipy.ndimage.map_coordinates). The fetched AABB always
    contains the rotated square plus a 2 m margin.
    """
    half = size_m / 2.0
    theta = math.radians(angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    aabb_half = half * (abs(cos_t) + abs(sin_t)) + 2.0
    src_bbox = (
        centre_east - aabb_half, centre_north - aabb_half,
        centre_east + aabb_half, centre_north + aabb_half,
    )
    with rasterio.open(path) as src:
        win = window_from_bounds(*src_bbox, transform=src.transform).round_offsets().round_lengths()
        arr = src.read(1, window=win)
        if src.nodata is not None:
            arr = np.where(arr == src.nodata, np.nan, arr)
        win_t = src.window_transform(win)

    src_h, src_w = arr.shape
    if src_h == 0 or src_w == 0:
        raise SystemExit(f"Empty raster crop for rotated bbox around ({centre_east}, {centre_north})")

    a, b, c = win_t.a, win_t.b, win_t.c
    d, e, f = win_t.d, win_t.e, win_t.f
    det = a * e - b * d  # for north-up rasters: a>0, e<0, so det<0

    out_pix_m = size_m / float(out_size)
    j_idx, i_idx = np.indices((out_size, out_size), dtype=np.float64)
    u = (i_idx + 0.5 - out_size / 2.0) * out_pix_m
    v = (out_size / 2.0 - j_idx - 0.5) * out_pix_m
    bng_e = centre_east + u * cos_t + v * sin_t
    bng_n = centre_north - u * sin_t + v * cos_t

    # Affine inverse: world (x, y) → source pixel (col, row)
    src_col = (e * (bng_e - c) - b * (bng_n - f)) / det - 0.5
    src_row = (-d * (bng_e - c) + a * (bng_n - f)) / det - 0.5

    out = ndimage.map_coordinates(
        arr, np.stack([src_row, src_col]), order=1, mode="constant", cval=np.nan
    )
    return out


def _flatten_building_roofs(building_mask: np.ndarray, dsm: np.ndarray,
                             mode: str = "detailed",
                             percentile: float = 95.0,
                             erode_px: int = 2,
                             height_gap_m: float = 5.0,
                             min_cluster_px: int = 10) -> np.ndarray:
    """Suppress DSM noise inside building footprints.

    Three modes trading roof shape for noise suppression:

    ``"detailed"`` (default) — apply a 3×3 mask-aware median filter to
        DSM pixels inside ``building_mask``. Single-pixel HVAC/antenna
        spikes are killed; pitched roofs, dormers, set-back upper
        floors, and stepped terraces keep their measured shape.

    ``"smoothed"`` — same idea with a 5×5 kernel, for cities with very
        cluttered modern rooftops where 3×3 isn't enough to clean up.
        Still preserves gross roof gradient.

    ``"flat"`` — legacy behaviour. Erode → label cores → expand → flatten
        each component to one ``percentile`` height (with gap-clustered
        multi-band splitting for stepped terraces). Every roof becomes
        a flat slab.

    The mask-aware median in detailed/smoothed mode treats non-building
    neighbours as ``+inf`` so they sort to the high end and are excluded
    from the median as long as the kernel still has a building-pixel
    majority. Building-edge pixels with mostly-ground neighbours fall
    back to ``np.median`` over building-only neighbours.

    Trees and other vegetation should NOT be in ``building_mask``. Run
    after landmark rescue but before tree rescue.
    """
    if not building_mask.any():
        return dsm

    if mode in ("detailed", "smoothed"):
        return _median_filter_within_mask(
            building_mask, dsm, kernel=5 if mode == "smoothed" else 3,
        )

    if mode != "flat":
        raise ValueError(f"unknown roof_detail mode: {mode!r}")

    # Step 1: erode → label cores → expand back to original mask.
    if erode_px > 0:
        core = ndimage.binary_erosion(building_mask, iterations=erode_px)
        if core.any():
            labels_core, n = ndimage.label(core)
            inv = labels_core == 0
            indices = ndimage.distance_transform_edt(
                inv, return_distances=False, return_indices=True
            )
            expanded = labels_core[indices[0], indices[1]]
            labels = np.where(building_mask, expanded, 0).astype(np.int32)
        else:
            labels, n = ndimage.label(building_mask)
    else:
        labels, n = ndimage.label(building_mask)

    if n == 0:
        return dsm

    out = dsm.astype(np.float32, copy=True)
    flat_labels = labels.ravel()
    flat_dsm = dsm.ravel()
    flat_out = out.ravel()

    # Sort all pixels by their component label so we can slice per-component.
    order = np.argsort(flat_labels, kind="stable")
    sorted_labels = flat_labels[order]
    boundaries = np.searchsorted(sorted_labels, np.arange(n + 2))

    n_split_components = 0
    n_total_bands = 0
    for k in range(1, n + 1):
        idx = order[boundaries[k]:boundaries[k + 1]]
        if idx.size == 0:
            continue
        vals = flat_dsm[idx]

        # Too small to bother sub-clustering → single percentile.
        if idx.size < 2 * min_cluster_px:
            flat_out[idx] = float(np.percentile(vals, percentile))
            n_total_bands += 1
            continue

        # Step 2: gap-based height clustering.
        sort_inner = np.argsort(vals, kind="stable")
        sorted_vals = vals[sort_inner]
        gaps = np.diff(sorted_vals)
        candidate_splits = np.where(gaps > height_gap_m)[0]

        # Each side of a split must have >= min_cluster_px pixels.
        valid_splits = []
        prev = 0
        for s in candidate_splits:
            left = s + 1 - prev
            right = idx.size - (s + 1)
            if left >= min_cluster_px and right >= min_cluster_px:
                valid_splits.append(s + 1)
                prev = s + 1

        if not valid_splits:
            flat_out[idx] = float(np.percentile(vals, percentile))
            n_total_bands += 1
            continue

        # Multi-band component — flatten each band independently.
        n_split_components += 1
        bounds = [0] + valid_splits + [idx.size]
        cluster_max = np.array(
            [sorted_vals[bounds[i + 1] - 1] for i in range(len(bounds) - 1)],
            dtype=np.float32,
        )
        cluster_height = np.array(
            [float(np.percentile(sorted_vals[bounds[i]:bounds[i + 1]], percentile))
             for i in range(len(bounds) - 1)],
            dtype=np.float32,
        )
        which = np.searchsorted(cluster_max, vals, side="left")
        which = np.clip(which, 0, len(cluster_height) - 1)
        flat_out[idx] = cluster_height[which]
        n_total_bands += len(cluster_height)

    if n_split_components:
        print(f"  height-split: {n_split_components} merged components → "
              f"{n_total_bands} distinct roof bands")

    return out


def _median_filter_within_mask(mask: np.ndarray, dsm: np.ndarray,
                                kernel: int = 3) -> np.ndarray:
    """Median-filter only the masked region. Non-mask neighbours don't
    participate in the median, so a roof edge isn't blended with adjacent
    ground.

    Implementation: float-cast the DSM, replace non-mask pixels with NaN,
    then for each mask pixel take ``np.nanmedian`` over its k×k window.
    Vectorised via a small stack of shifted views (kernel² copies, fine
    for kernels ≤ 7). Pure-building pixels see normal median behaviour
    (HVAC spikes drop out); edge pixels see fewer neighbours and fall
    back to whatever building-pixel data is available.
    """
    if kernel < 3 or kernel % 2 == 0:
        raise ValueError(f"kernel must be an odd integer >= 3, got {kernel}")

    out = dsm.astype(np.float32, copy=True)
    masked = np.where(mask, dsm.astype(np.float32), np.nan)
    pad = kernel // 2
    padded = np.pad(masked, pad, mode="constant", constant_values=np.nan)

    H, W = dsm.shape
    stack = np.empty((kernel * kernel, H, W), dtype=np.float32)
    i = 0
    for dy in range(kernel):
        for dx in range(kernel):
            stack[i] = padded[dy:dy + H, dx:dx + W]
            i += 1

    # All-NaN columns happen wherever the kernel doesn't overlap any
    # building pixel. Those positions aren't written back to ``out``
    # (they fall outside ``mask``) so the warning is noise.
    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.filterwarnings("ignore", r"All-NaN slice encountered",
                                 category=RuntimeWarning)
        med = np.nanmedian(stack, axis=0)

    fill = np.where(mask & np.isfinite(med), med, dsm.astype(np.float32))
    out = np.where(mask, fill, dsm.astype(np.float32))
    return out

    return flat_out.reshape(dsm.shape).astype(dsm.dtype, copy=False)


def _rotate_polygons_to_local(polygons, centre_east, centre_north, angle_deg):
    """Translate and counter-rotate BNG polygons so the rotated square's
    local frame becomes axis-aligned: u = local x ∈ [-half, +half],
    v = local y ∈ [-half, +half]. Used so we can rasterise water/bridge
    polygons against a local-frame transform."""
    out = []
    for p in polygons or []:
        if p is None or p.is_empty or not p.is_valid:
            continue
        try:
            local = shp_translate(p, xoff=-centre_east, yoff=-centre_north)
            local = shp_rotate(local, angle_deg, origin=(0.0, 0.0), use_radians=False)
        except Exception:
            continue
        if not local.is_empty and local.is_valid:
            out.append(local)
    return out


def _smooth_z_along_path(z: np.ndarray, window: int, is_loop: bool) -> np.ndarray:
    """Moving-average filter applied along the route. Window is forced odd
    (and clamped to ≤ len(z)) so the centre sample is well-defined. For
    loops the padding wraps around; otherwise the edges replicate so the
    start/end of the path don't get pulled toward zero."""
    n = len(z)
    if window <= 1 or n < 3:
        return z.astype(np.float32, copy=True)
    w = int(min(window, n))
    if w % 2 == 0:
        w -= 1
    if w < 3:
        return z.astype(np.float32, copy=True)
    half = w // 2
    if is_loop:
        padded = np.concatenate([z[-half:], z, z[:half]])
    else:
        padded = np.pad(z, half, mode="edge")
    kernel = np.ones(w, dtype=np.float64) / w
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed.astype(np.float32)


def _build_route_insert_mesh(
    polyline_local,
    dsm: np.ndarray,
    bbox: tuple[float, float, float, float],
    *,
    z_floor: float,
    SCALE: float,
    z_exaggeration: float,
    plinth_mm: float,
    width_mm: float,
    tolerance_mm: float,
    raised_mm: float,
    insert_tail_mm: float,
    insert_floor_z_mm: float | None = None,
    top_smooth_window: int = 11,
    is_loop: bool = False,
    sample_spacing_m: float = 5.0,
) -> np.ndarray:
    """Sweep a rectangular ribbon along ``polyline_local`` and return triangles.

    The cross-section is a flat-bottomed bar so the insert prints directly
    on the bed without supports:
      bottom z = ``insert_floor_z_mm`` (constant — the slot floor matches it)
      top z    = terrain_print_z + raised_mm  (varies, follows the contour)

    When ``insert_floor_z_mm`` is None we fall back to the legacy
    contour-following bottom (kept for parity with older callers).

    When ``is_loop`` is true the polyline closes on itself: end caps are
    omitted and the sweep wraps from sample N-1 back to sample 0.
    """
    from scipy.ndimage import map_coordinates

    L = polyline_local.length
    if L < sample_spacing_m * 2:
        return np.zeros((0, 3, 3), dtype=np.float32)
    n_samples = max(2, int(L / sample_spacing_m) + 1)
    samples = [polyline_local.interpolate(i * L / (n_samples - 1)) for i in range(n_samples)]
    x_world = np.asarray([p.x for p in samples], dtype=np.float64)
    y_world = np.asarray([p.y for p in samples], dtype=np.float64)

    # Pixel coordinates for DSM interpolation. Row 0 is north (top of bbox).
    H_orig, W_orig = dsm.shape
    cols = (x_world - bbox[0]) / max(bbox[2] - bbox[0], 1e-9) * W_orig
    rows = (bbox[3] - y_world) / max(bbox[3] - bbox[1], 1e-9) * H_orig
    rows = np.clip(rows, 0, H_orig - 1)
    cols = np.clip(cols, 0, W_orig - 1)
    z_world_samples = map_coordinates(dsm, [rows, cols], order=1, mode="nearest")

    # Convert terrain world z → print mm z (matches main mesh's zz formula)
    z_print = (z_world_samples - z_floor) * SCALE * z_exaggeration + plinth_mm

    # Print-frame x, y for each sample (matches xx, yy = (world - bbox.left) * SCALE)
    x_print = (x_world - bbox[0]) * SCALE
    y_print = (y_world - bbox[1]) * SCALE

    # Tangent + perpendicular at each sample (centred difference). For
    # closed loops we wrap the gradient so the start/end tangents match.
    if is_loop:
        x_ext = np.concatenate([[x_print[-1]], x_print, [x_print[0]]])
        y_ext = np.concatenate([[y_print[-1]], y_print, [y_print[0]]])
        dx = (x_ext[2:] - x_ext[:-2]) / 2.0
        dy = (y_ext[2:] - y_ext[:-2]) / 2.0
    else:
        dx = np.gradient(x_print)
        dy = np.gradient(y_print)
    seg_len = np.hypot(dx, dy)
    seg_len = np.where(seg_len < 1e-6, 1.0, seg_len)
    tx, ty = dx / seg_len, dy / seg_len
    nx, ny = -ty, tx   # left-perpendicular

    half_w_mm = (width_mm - 2 * tolerance_mm) / 2.0
    # Flat bottom by default — sits flush on the print bed and exactly
    # matches the slot floor in the main map.
    if insert_floor_z_mm is not None:
        z_bot = np.full_like(z_print, float(insert_floor_z_mm))
    else:
        z_bot = z_print - insert_tail_mm + 0.05
    # Smooth the terrain z along the path before forming the top, so the
    # insert reads as a flowing ribbon rather than a bumpy contour. The
    # window is in samples (default 11 ≈ 50 m at 5 m sampling).
    z_print_smoothed = _smooth_z_along_path(z_print, top_smooth_window, is_loop)
    z_top = z_print_smoothed + raised_mm
    # Hard floor: never let the smoothed top dip below the bottom + a
    # small printable thickness, even if smoothing pulls a peak down.
    z_top = np.maximum(z_top, z_bot + 0.4)

    lx = x_print + nx * half_w_mm
    ly = y_print + ny * half_w_mm
    rx = x_print - nx * half_w_mm
    ry = y_print - ny * half_w_mm

    triangles: list[tuple] = []
    n = n_samples
    # For loop mode, also emit the wrap-around segment (i=n-1 → i=0).
    pair_count = n if is_loop else (n - 1)
    for k in range(pair_count):
        i = k
        j = (k + 1) % n
        l0_t = (lx[i], ly[i], z_top[i]); r0_t = (rx[i], ry[i], z_top[i])
        l1_t = (lx[j], ly[j], z_top[j]); r1_t = (rx[j], ry[j], z_top[j])
        l0_b = (lx[i], ly[i], z_bot[i]); r0_b = (rx[i], ry[i], z_bot[i])
        l1_b = (lx[j], ly[j], z_bot[j]); r1_b = (rx[j], ry[j], z_bot[j])
        # Top face (CCW from above → +z normal)
        triangles.append((l0_t, r0_t, r1_t))
        triangles.append((l0_t, r1_t, l1_t))
        # Bottom face (CCW from below → −z normal, i.e. CW from above)
        triangles.append((l0_b, l1_b, r1_b))
        triangles.append((l0_b, r1_b, r0_b))
        # Left side wall (outward = +nx, +ny direction)
        triangles.append((l0_t, l0_b, l1_b))
        triangles.append((l0_t, l1_b, l1_t))
        # Right side wall (outward = −nx, −ny)
        triangles.append((r0_t, r1_t, r1_b))
        triangles.append((r0_t, r1_b, r0_b))

    if is_loop:
        # Closed loop: no end caps.
        return np.asarray(triangles, dtype=np.float32)

    # End cap at start (i=0). Outward = −tangent.
    l0_t = (lx[0], ly[0], z_top[0]); r0_t = (rx[0], ry[0], z_top[0])
    l0_b = (lx[0], ly[0], z_bot[0]); r0_b = (rx[0], ry[0], z_bot[0])
    triangles.append((l0_t, r0_t, r0_b))
    triangles.append((l0_t, r0_b, l0_b))

    # End cap at end (i=N-1). Outward = +tangent.
    lN_t = (lx[-1], ly[-1], z_top[-1]); rN_t = (rx[-1], ry[-1], z_top[-1])
    lN_b = (lx[-1], ly[-1], z_bot[-1]); rN_b = (rx[-1], ry[-1], z_bot[-1])
    triangles.append((lN_t, lN_b, rN_b))
    triangles.append((lN_t, rN_b, rN_t))

    return np.asarray(triangles, dtype=np.float32)


def _make_preview_triangles(tri: np.ndarray) -> np.ndarray:
    """Derive a deliberately degraded preview mesh from the clean one.

    Coarse vertex quantisation snaps every vertex to a ~1 mm grid: the
    surface picks up visible stepping and ~±0.5 mm dimensional error, so a
    grabbed copy prints as a lumpy, inaccurate approximation — clearly not
    the crisp dec=2 deliverable — yet the city still reads fine in the
    small on-page 3D viewer (the customer's sales preview stays attractive).

    Combined with the caller's 'NOT FOR PRINT' STL-header stamp and the
    clean print mesh being written to a server-only path, a console /
    Network-tab grab yields something that is provably not the product.
    A raised frame was tried and removed — it walled the cityscape and
    wrecked the sales preview for no extra protection."""
    pts = tri.reshape(-1, 3)
    mn = pts.min(axis=0); mx = pts.max(axis=0)
    ext = float(max(mx[0] - mn[0], mx[1] - mn[1])) or 1.0
    grid = ext / 90.0  # ~1 mm on a ~90 mm model — print-inferior, viewer-fine
    return (np.round(tri / grid) * grid).astype(np.float32)


def _write_stl(triangles: np.ndarray, output_path: Path, header: str):
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


# ─── Main builder ────────────────────────────────────────────────────────

def build_tier3_with_water_stl(
    *, dsm_path, dtm_path, centre_east, centre_north,
    size_m=1200, print_w_mm=90.0, plinth_mm=2.0,
    z_exaggeration=1.0,
    angle_deg: float = 0.0,
    print_margin_mm: float = 0.5,
    water_polygons_bng=None,
    bridge_polygons_bng=None,
    landmark_polygons_bng=None,
    road_polygons_bng=None,
    road_depth_mm: float = 0.6,
    include_trees: bool = True,
    tree_min_height_m: float = 3.0,
    tree_max_height_m: float = 25.0,
    tree_min_area_px: int = 4,
    z_floor_override: float | None = None,
    # Phase 9 — route insert. polyline is a list[(e_bng, n_bng)] already
    # densified by the caller. width/raised/tolerance are in print mm.
    route_polyline_bng=None,
    route_width_mm: float = 1.5,
    route_raised_mm: float = 0.5,
    route_tolerance_mm: float = 0.1,
    route_insert_tail_mm: float = 1.5,
    route_top_smooth_window: int = 11,
    route_loop: bool = False,
    route_out_path=None,
    dec: int = 1,
    roof_detail: str = "detailed",
    wall_style: str = "sloped",
    building_part_dicts: list[dict] | None = None,
    osm_env_mode: str = "promote",
    out_path="output.stl",
    preview: bool = False,
    clean_out_path=None,
) -> dict:
    if osm_env_mode not in ("promote", "keep", "drop"):
        raise ValueError(
            f"osm_env_mode must be 'promote', 'keep', or 'drop', got {osm_env_mode!r}"
        )
    if roof_detail not in ("detailed", "smoothed", "flat"):
        raise ValueError(
            f"roof_detail must be 'detailed', 'smoothed', or 'flat', got {roof_detail!r}"
        )
    if wall_style not in ("sloped", "vertical"):
        raise ValueError(
            f"wall_style must be 'sloped' or 'vertical', got {wall_style!r}"
        )
    half = size_m / 2
    rotated = abs(angle_deg) > 1e-6

    if rotated:
        # Sample the rotated square at ~1m to match the unrotated path's pixel density.
        out_size = int(round(size_m))
        print(f"Cropping rotated square at {angle_deg:+.1f}° around ({centre_east:.0f}, {centre_north:.0f}), {out_size}×{out_size} px…")
        dsm = _crop_raster_rotated(dsm_path, centre_east, centre_north, size_m, angle_deg, out_size)
        dtm = _crop_raster_rotated(dtm_path, centre_east, centre_north, size_m, angle_deg, out_size)
        # Local-frame bbox used by every transform_from_bounds call below.
        bbox = (-half, -half, half, half)
    else:
        bbox = (centre_east - half, centre_north - half,
                centre_east + half, centre_north + half)
        print(f"Cropping bbox {bbox}…")
        dsm = _crop_raster(dsm_path, bbox)
        dtm = _crop_raster(dtm_path, bbox)
    if dsm.shape != dtm.shape:
        raise SystemExit(f"DSM and DTM shapes differ: {dsm.shape} vs {dtm.shape}")

    dsm = np.clip(np.nan_to_num(dsm, nan=np.nanmedian(dsm)), -10, 400)
    dtm = np.clip(np.nan_to_num(dtm, nan=np.nanmedian(dtm)), -50, 1500)

    # ─── OSM building:part / S3DB DSM override ────────────────────────
    # If the caller supplied parsed building_part dicts (height, min_height,
    # roof_shape, etc.), rasterise each footprint and overwrite the DSM
    # with `dtm + tagged_height` (with a per-shape roof contour). The
    # `> dsm` guard means the override only ever raises the surface, so
    # LIDAR-detected geometry that's already taller stays. The accumulated
    # `bp_mask` is OR'd into mask_final after the landmark rescue so
    # building:part footprints survive morphological filtering regardless
    # of what LIDAR read.
    bp_mask = np.zeros(dsm.shape, dtype=bool)
    parts_for_render = list(building_part_dicts or [])
    if parts_for_render:
        # Rotate to local frame if the bbox itself is rotated.
        if rotated:
            parts_for_render = _rotate_part_polygons_to_local(
                parts_for_render, centre_east, centre_north, angle_deg,
            )
        # Promote "envelope" parts — untagged-roof-shape polygons that wrap
        # a shape-tagged sub-part at the same height (OSM fallback geometry).
        # Instead of dropping, copy the sub-part's roof_shape onto the
        # envelope so the WHOLE building footprint renders with that
        # taper, not just the small sub-part's region. This is the only
        # way a 2D height-field can recover the iconic shape of buildings
        # like the Shard, where the OSM data has a 310 m flat envelope +
        # tiny pyramidal+skillion details on top, with the middle 200 m
        # of the tower entirely untagged.
        if osm_env_mode == "promote":
            parts_for_render, env_promoted = promote_envelope_parts(parts_for_render)
            if env_promoted:
                for ep in env_promoted:
                    print(f"  OSM 3D: promoted envelope at h={ep['height']:.0f}m "
                          f"({ep['area']:.0f}m²) to '{ep['donor_shape']}' "
                          f"(donor {ep['donor_area']:.0f}m²) — full-footprint taper")
        elif osm_env_mode == "drop":
            parts_for_render, env_dropped = drop_envelope_parts(parts_for_render)
            if env_dropped:
                print(f"  OSM 3D: dropped {len(env_dropped)} untagged envelope parts "
                      f"(rectangular fallbacks; only shape-tagged geometry remains)")
        # else osm_env_mode == "keep": preserve envelope as flat block — gives a
        # full-height rectangular tower (shape-tagged sub-parts may add a small
        # cap on top, depending on roof_height). Suits buildings where the user
        # wants the building's outline volume preserved (e.g. modern blocks).
        # Overhang detection on rotated polygons (geometry-preserving).
        kept_parts, dropped_parts = detect_overhanging_parts(parts_for_render)
        if dropped_parts:
            print(f"  OSM 3D: dropped {len(dropped_parts)} overhanging parts "
                  f"(e.g. Walkie-Talkie-style top-of-bulge)")
        unsupported_shapes = sorted({
            p["raw_shape"] for p in kept_parts
            if p["raw_shape"] and p["raw_shape"] not in _VALID_ROOF_SHAPES
        })
        if unsupported_shapes:
            print(f"  OSM 3D: roof shapes treated as flat (not yet supported): "
                  f"{', '.join(unsupported_shapes)}")
        # Build per-pixel local-frame xy in metres (for skillion/gabled
        # projection). Same pixel resolution as dsm.
        _h_pix, _w_pix = dsm.shape
        bp_transform = transform_from_bounds(*bbox, _w_pix, _h_pix)
        bxs = bbox[0] + (np.arange(_w_pix) + 0.5) * (bbox[2] - bbox[0]) / _w_pix
        bys = bbox[3] - (np.arange(_h_pix) + 0.5) * (bbox[3] - bbox[1]) / _h_pix
        bxx, byy = np.meshgrid(bxs, bys)
        bxx = bxx.astype(np.float32); byy = byy.astype(np.float32)
        applied = 0
        for p in kept_parts:
            try:
                part_mask = geometry_mask(
                    [p["polygon"].__geo_interface__],
                    out_shape=(_h_pix, _w_pix),
                    transform=bp_transform,
                    invert=True,
                )
            except Exception:
                continue
            if not part_mask.any():
                continue
            target_z = render_part_roof_z(
                part_mask=part_mask,
                dtm=dtm.astype(np.float32),
                height_m=p["height_m"],
                min_height_m=p["min_height_m"],
                roof_shape=p["roof_shape"],
                roof_height_m=p["roof_height_m"],
                roof_direction_deg=p["roof_direction_deg"],
                pixel_xy_local=(bxx, byy),
            )
            override = part_mask & (target_z > dsm)
            if override.any():
                dsm = np.where(override, target_z, dsm).astype(np.float32)
                bp_mask = bp_mask | part_mask
                applied += 1
        print(f"  OSM 3D: applied {applied} of {len(kept_parts)} kept parts "
              f"(of {len(parts_for_render)} total)")

    # Buildings (same logic as tier3_measured_roofs).
    # iterations=1 keeps narrow features (chimneys, narrow turrets, slim
    # spires) that iterations=2 was erasing; the >=25 px size filter
    # still drops single-pixel noise.
    print("Detecting buildings…")
    hag = np.maximum(dsm - dtm, 0)
    mask_init = hag > 2.5
    mask_opened = ndimage.binary_opening(mask_init, iterations=1)
    labels, n = ndimage.label(mask_opened)
    sizes = ndimage.sum(mask_opened, labels, range(1, n + 1))
    keep = np.where(sizes >= 25)[0] + 1
    mask_filtered = np.isin(labels, keep)
    mask_dilated = ndimage.binary_dilation(mask_filtered, iterations=3)
    mask_final = ndimage.binary_closing(mask_dilated & mask_init, iterations=1)
    print(f"  building coverage: {100 * mask_final.mean():.1f}%")

    # Landmark preservation. Narrow tall features (church spires, monument
    # towers, slim chimneys) are typically <5 m wide and get erased by
    # `binary_opening(iterations=2)` above. We rasterise OSM-tagged landmark
    # polygons and OR them back into the mask, gated by `mask_init` so we
    # only re-include pixels that are physically elevated (>2.5 m above
    # ground). This way the landmark's actual DSM height drives the print.
    landmark_polygons_bng = [p for p in (landmark_polygons_bng or [])
                             if isinstance(p, Polygon) and not p.is_empty and p.is_valid]
    if rotated and landmark_polygons_bng:
        landmark_polygons_local = _rotate_polygons_to_local(
            landmark_polygons_bng, centre_east, centre_north, angle_deg
        )
    else:
        landmark_polygons_local = landmark_polygons_bng
    if landmark_polygons_local:
        print(f"Rasterising {len(landmark_polygons_local)} OSM landmark polygons…")
        # `top` (and therefore H_orig/W_orig) hasn't been built yet at this
        # point — landmarks have to amend `mask_final` BEFORE `top` is
        # constructed. Use dsm.shape directly; it's identical to top.shape
        # since top is just np.where(mask_final, dsm, dtm).
        _h_pix, _w_pix = dsm.shape
        crop_transform_landmark = transform_from_bounds(*bbox, _w_pix, _h_pix)
        landmark_mask = geometry_mask(
            [p.__geo_interface__ for p in landmark_polygons_local],
            out_shape=(_h_pix, _w_pix),
            transform=crop_transform_landmark,
            invert=True,
        )
        rescued = (landmark_mask & mask_init) & ~mask_final
        n_rescued = int(rescued.sum())
        if n_rescued:
            mask_final = mask_final | (landmark_mask & mask_init)
            print(f"  landmarks: rescued {n_rescued:,} pixels from morphological filter")

    # OSM building:part footprints get force-merged into mask_final so the
    # tagged geometry survives the morphological filter regardless of LIDAR.
    if bp_mask.any():
        bp_added = int(bp_mask.sum() - (bp_mask & mask_final).sum())
        if bp_added:
            mask_final = mask_final | bp_mask
            print(f"  OSM 3D: merged {bp_added:,} building:part pixels into mask")

    # Suppress DSM noise inside building footprints. The default "detailed"
    # mode runs a 3×3 mask-aware median filter that kills HVAC/antenna
    # spikes while preserving pitched roofs, dormers, set-back upper
    # floors, and stepped terraces. "smoothed" uses a 5×5 kernel for
    # cluttered modern rooftops. "flat" is the legacy 95th-percentile
    # per-footprint behaviour. Run after landmarks are merged but
    # before trees are added.
    #
    # OSM 3D pixels are protected from the flatten — the per-part heights
    # are already the intended geometry, and percentile-flattening across
    # the whole building's connected component (which now includes 12 m
    # lower-mass parts AND 310 m skillion peaks) drags every pixel to
    # the 95th percentile and erases the taper.
    if mask_final.any():
        n_roof_px = int(mask_final.sum())
        if bp_mask.any():
            dsm_osm_snapshot = dsm.copy()
        dsm = _flatten_building_roofs(mask_final, dsm, mode=roof_detail)
        if bp_mask.any():
            dsm = np.where(bp_mask, dsm_osm_snapshot, dsm).astype(np.float32)
            print(f"  filtered {n_roof_px:,} rooftop pixels (mode={roof_detail}; "
                  f"{int(bp_mask.sum()):,} OSM 3D pixels protected)")
        else:
            print(f"  filtered {n_roof_px:,} rooftop pixels (mode={roof_detail})")

    # Trees / vegetation. The DSM already contains tree heights — they just
    # get erased by `binary_opening(iterations=2)` because trees in parks
    # and avenues form clusters too small/diffuse to survive morphology.
    # We OR back any cluster of `mask_init` pixels (>2.5 m above ground)
    # that's at most `tree_max_height_m` tall and at least `tree_min_area_px`
    # in size. Buildings are already in `mask_final`; whatever was in
    # `mask_init` but excluded is mostly trees, scaffolding, and noise.
    # Lower bound on height filters out cars/walls; upper bound filters out
    # construction cranes / pylons. min_area filters single-pixel noise.
    if include_trees:
        tree_candidates = mask_init & ~mask_final
        tree_height_band = (hag >= tree_min_height_m) & (hag <= tree_max_height_m)
        tree_pixels = tree_candidates & tree_height_band
        if tree_pixels.any():
            tree_labels, n_tree = ndimage.label(tree_pixels)
            if n_tree > 0:
                tree_sizes = ndimage.sum(tree_pixels, tree_labels, range(1, n_tree + 1))
                keep_trees = np.where(tree_sizes >= tree_min_area_px)[0] + 1
                if len(keep_trees):
                    tree_mask = np.isin(tree_labels, keep_trees)
                    n_tree_px = int(tree_mask.sum())
                    mask_final = mask_final | tree_mask
                    print(f"  trees: rescued {n_tree_px:,} pixels in "
                          f"{len(keep_trees):,} clusters")

    print("Building top raster…")
    top = np.where(mask_final, dsm, dtm).astype(np.float32)

    DEC = int(dec)
    if DEC < 1:
        raise ValueError(f"dec must be >= 1, got {dec}")
    H_orig, W_orig = top.shape
    H = H_orig // DEC
    W = W_orig // DEC
    top_cropped = top[:H * DEC, :W * DEC]
    heights = top_cropped.reshape(H, DEC, W, DEC).max(axis=(1, 3))

    # Combine LIDAR-detected structures with OSM bridge polygons. OSM picks
    # up thin footbridges that LIDAR can miss; LIDAR catches things OSM
    # hasn't been mapped for. Both must live on the same H_orig × W_orig
    # raster grid that we then max-pool down to per-vertex / per-cell.
    bridge_polygons_bng = [p for p in (bridge_polygons_bng or [])
                           if isinstance(p, Polygon) and not p.is_empty and p.is_valid]
    if rotated and bridge_polygons_bng:
        bridge_polygons_local = _rotate_polygons_to_local(
            bridge_polygons_bng, centre_east, centre_north, angle_deg
        )
    else:
        bridge_polygons_local = bridge_polygons_bng
    if bridge_polygons_local:
        print(f"Rasterising {len(bridge_polygons_local)} OSM bridge polygons…")
        crop_transform_bridge = transform_from_bounds(*bbox, W_orig, H_orig)
        osm_bridge_mask = geometry_mask(
            [p.__geo_interface__ for p in bridge_polygons_local],
            out_shape=(H_orig, W_orig),
            transform=crop_transform_bridge,
            invert=True,
        )
        combined_struct = mask_final | osm_bridge_mask
    else:
        osm_bridge_mask = np.zeros_like(mask_final)
        combined_struct = mask_final

    # Bridge mask at vertex resolution; used a few lines below to avoid
    # lifting heights over bridges (would lower them by accident).
    struct_block_vertex = combined_struct[:H * DEC, :W * DEC].astype(np.float32).reshape(
        H, DEC, W, DEC).mean(axis=(1, 3)) > 0.1

    # ─── Water mask ──────────────────────────────────────────────────────
    water_polygons_bng = [p for p in (water_polygons_bng or [])
                          if isinstance(p, Polygon) and not p.is_empty and p.is_valid]
    if rotated and water_polygons_bng:
        water_polygons_local = _rotate_polygons_to_local(
            water_polygons_bng, centre_east, centre_north, angle_deg
        )
    else:
        water_polygons_local = water_polygons_bng
    if water_polygons_local:
        print(f"Rasterising {len(water_polygons_local)} water polygons…")
        crop_transform = transform_from_bounds(*bbox, W_orig, H_orig)
        water_mask_orig = geometry_mask(
            [p.__geo_interface__ for p in water_polygons_local],
            out_shape=(H_orig, W_orig),
            transform=crop_transform,
            invert=True,   # True INSIDE polygons
        )

        # Per-cell water coverage: each mesh cell (j, i) covers the DEC×DEC
        # pixel block (j*DEC..(j+1)*DEC-1, i*DEC..(i+1)*DEC-1). A cell is cut
        # only if MOST of its area is inside the polygon — at the default
        # threshold of 0.7, the cut hugs the actual water surface and leaves
        # the riverbank intact. Increase to 0.85+ for very tight cuts;
        # lower toward 0.5 to catch narrower rivers at the cost of trimming
        # shores.
        WATER_CUT_THRESHOLD = 0.7
        trimmed = water_mask_orig[:(H - 1) * DEC, :(W - 1) * DEC].astype(np.float32)
        blocks = trimmed.reshape(H - 1, DEC, W - 1, DEC)
        water_frac = blocks.mean(axis=(1, 3))
        water_cell = water_frac > WATER_CUT_THRESHOLD

        # Bridge / structure protection.
        # `combined_struct` = LIDAR `mask_final` ∪ OSM bridge polygons. A
        # cell over water that also contains a meaningful patch of structure
        # is almost certainly a bridge or similar elevated structure — keep
        # it solid so it stays visible in the model instead of being punched
        # through with the water.
        STRUCTURE_PROTECT_THRESHOLD = 0.1   # 10 % of cell area is enough
        struct_blocks = combined_struct[:(H - 1) * DEC, :(W - 1) * DEC].reshape(
            H - 1, DEC, W - 1, DEC)
        structure_cell = struct_blocks.mean(axis=(1, 3)) > STRUCTURE_PROTECT_THRESHOLD
        bridge_cell = water_cell & structure_cell    # cells over water with structure
        preserved = int(bridge_cell.sum())
        if preserved:
            print(f"  preserving {preserved} bridge/structure cells over water")
        water_cell = water_cell & ~structure_cell

        # Manifold-correctness fix.
        # The vertex-resolution `heights` array is built from per-block max-pool
        # of the DSM/DTM. Over open water, those blocks come back at water
        # surface elevation — well below the surrounding land. Without
        # correction, LAND cell top quads sloped DOWN to water level at every
        # land/water boundary, and the inner walls only reached up to that
        # water-surface height. Above water level the cavity was unbounded:
        # slicers see a non-manifold opening and "fill" the river starting
        # from a certain layer up.
        # Fix: at every water-block vertex that ISN'T a bridge/structure,
        # replace the height with the nearest land-block height. Land cell
        # tops then stay flat all the way to the shore, and the cavity is
        # walled from the land surface down to z=0.
        water_block_vertex = (
            water_mask_orig[:H * DEC, :W * DEC].astype(np.float32)
            .reshape(H, DEC, W, DEC).mean(axis=(1, 3)) > 0.5
        )
        needs_lift = water_block_vertex & ~struct_block_vertex
        if needs_lift.any() and not needs_lift.all():
            n_lifted = int(needs_lift.sum())
            print(f"  lifting {n_lifted} water-side vertices to nearest land height")
            indices = ndimage.distance_transform_edt(
                needs_lift, return_distances=False, return_indices=True
            )
            heights = heights.copy()
            heights[needs_lift] = heights[
                indices[0][needs_lift], indices[1][needs_lift]
            ]
    else:
        water_cell = np.zeros((H - 1, W - 1), dtype=bool)
        bridge_cell = np.zeros((H - 1, W - 1), dtype=bool)
        preserved = 0

    cells_total = (H - 1) * (W - 1)
    cells_water = int(water_cell.sum())
    print(f"  water cells: {cells_water}/{cells_total} "
          f"({100 * cells_water / cells_total:.1f}%)")

    # Footprint inset: requested print_w_mm is the bbox the model FITS INSIDE,
    # not the footprint itself. The actual mesh is built at (print_w_mm -
    # print_margin_mm) so a 90 mm request prints as 89.5 mm × 89.5 mm. This
    # leaves room for slicer brims, OS dimensional bias, etc., while keeping
    # the user-visible "print width" number predictable.
    effective_print_w_mm = max(1.0, float(print_w_mm) - float(print_margin_mm))

    # The mesh has W vertices spanning W-1 cells of `m_per_cell` metres, so the
    # vertex grid covers (W-1)*m_per_cell metres in BNG space. Scale that span
    # — not the nominal `size_m` — to exactly fill `effective_print_w_mm` so
    # the printed footprint is bit-precise. Without this correction the model
    # is shy by one cell width (≈0.08 mm at 90 mm print, 1 m LIDAR).
    m_per_cell = (size_m / H_orig) * DEC
    mesh_span_m = (max(W, H) - 1) * m_per_cell
    SCALE = effective_print_w_mm / mesh_span_m
    if print_margin_mm > 0:
        print(f"  footprint inset: {print_w_mm:.1f} mm requested → {effective_print_w_mm:.2f} mm actual ({print_margin_mm:.2f} mm margin)")
    # Default: clamp the floor to this tile's lowest LIDAR sample so the
    # plinth sits at z=0 right under the lowest point. For multi-tile prints
    # the caller passes the SHARED min across every tile so adjacent tiles
    # join flush instead of stepping at the seam.
    z_floor = float(heights.min()) if z_floor_override is None else float(z_floor_override)
    if z_floor_override is not None:
        local_min = float(heights.min())
        if z_floor > local_min + 0.01:
            print(f"  z_floor override ({z_floor:.2f} m) is ABOVE this tile's "
                  f"local min ({local_min:.2f} m); some terrain will clip below "
                  f"the plinth. Override is intended for joinable tile prints.")
    z_world = heights - z_floor
    relief_mm = float(z_world.max() * SCALE * z_exaggeration)
    print(f"  relief: {z_world.max():.1f} m → {relief_mm:.2f} mm at print scale")

    xs = np.arange(W) * m_per_cell * SCALE
    ys = (H - 1 - np.arange(H)) * m_per_cell * SCALE
    xx, yy = np.meshgrid(xs, ys)
    zz = z_world * SCALE * z_exaggeration + plinth_mm

    # ─── Engraved streets ────────────────────────────────────────────────
    # Roads come in as buffered Polygons in BNG (per-tag widths). We
    # rasterise them at DSM resolution, max-pool to mesh cells, mask off
    # water and bridges (engraving over either would punch a hole or
    # thin the deck below MIN_DECK_THICK_MM), then lower the z of every
    # vertex bordering a road cell by `road_depth_mm`. The vertex pool is
    # min'd with the cell's z so we never lift terrain — only carve.
    road_polygons_bng = [p for p in (road_polygons_bng or [])
                         if isinstance(p, Polygon) and not p.is_empty and p.is_valid]
    if rotated and road_polygons_bng:
        road_polygons_local = _rotate_polygons_to_local(
            road_polygons_bng, centre_east, centre_north, angle_deg
        )
    else:
        road_polygons_local = road_polygons_bng
    if road_polygons_local and road_depth_mm > 0:
        print(f"Rasterising {len(road_polygons_local)} OSM road polygons…")
        crop_transform_road = transform_from_bounds(*bbox, W_orig, H_orig)
        road_pixel_mask = geometry_mask(
            [p.__geo_interface__ for p in road_polygons_local],
            out_shape=(H_orig, W_orig),
            transform=crop_transform_road,
            invert=True,
        )
        # Per-cell road coverage. Lower threshold than water (0.3 vs 0.7) so
        # the engraved line shows as a continuous groove on narrow streets,
        # not a dotted scatter.
        ROAD_THRESHOLD = 0.3
        trimmed_r = road_pixel_mask[:(H - 1) * DEC, :(W - 1) * DEC].astype(np.float32)
        blocks_r = trimmed_r.reshape(H - 1, DEC, W - 1, DEC)
        road_frac = blocks_r.mean(axis=(1, 3))
        road_cell = road_frac > ROAD_THRESHOLD
        # Don't engrave water (would punch holes) or bridges (would thin
        # the deck). bridge_cell is computed earlier in the file.
        road_cell &= ~water_cell & ~bridge_cell
        n_road_cells = int(road_cell.sum())
        if n_road_cells:
            # A vertex (j, i) borders cells (j-1,i-1), (j-1,i), (j,i-1), (j,i).
            # If any of those is a road cell, the vertex is a road vertex.
            road_vertex = np.zeros((H, W), dtype=bool)
            road_vertex[:-1, :-1] |= road_cell
            road_vertex[:-1, 1:]  |= road_cell
            road_vertex[1:,  :-1] |= road_cell
            road_vertex[1:,  1:]  |= road_cell
            # Lower zz at road vertices, clamped above the plinth top.
            depth = float(road_depth_mm)
            min_z = float(plinth_mm) + 0.05
            zz = np.where(
                road_vertex,
                np.maximum(zz - depth, min_z),
                zz,
            )
            print(f"  roads: engraved {n_road_cells:,} cells "
                  f"({depth:.2f} mm × {road_frac[road_cell].mean():.2f} avg coverage)")

    # ─── Route slot (Phase 9c) ───────────────────────────────────────
    # Polyline → buffered polygon → rasterise → per-cell mask → carve.
    # Deeper than road grooves so the male insert has enough body to
    # print without warping. Skips water/bridge cells just like roads.
    route_slot_info = None       # populated when a route is requested
    route_polyline_local = None  # cached for the male insert builder
    if route_polyline_bng and len(route_polyline_bng) >= 2:
        # Convert the polyline into the same coordinate frame `bbox` /
        # `crop_transform_*` use: BNG metres in the unrotated path, the
        # local centre-anchored frame in the rotated path.
        from shapely.geometry import LineString
        from shapely.affinity import rotate as _shp_rotate, translate as _shp_translate
        route_line_bng = LineString(list(route_polyline_bng))
        if rotated:
            shifted = _shp_translate(route_line_bng, xoff=-centre_east, yoff=-centre_north)
            route_line_local = _shp_rotate(shifted, -angle_deg, origin=(0, 0), use_radians=False)
        else:
            route_line_local = route_line_bng
        # zz is in print mm, route_width_mm is in print mm; convert width
        # back to metres via SCALE = print_w_mm / mesh_span_m.
        buffer_m = (route_width_mm / 2.0) / SCALE
        route_buffer_local = route_line_local.buffer(
            buffer_m, cap_style="round", join_style="round"
        )
        if route_buffer_local.is_valid and not route_buffer_local.is_empty:
            crop_transform_route = transform_from_bounds(*bbox, W_orig, H_orig)
            route_pixel_mask = geometry_mask(
                [route_buffer_local.__geo_interface__],
                out_shape=(H_orig, W_orig),
                transform=crop_transform_route,
                invert=True,
            )
            ROUTE_THRESHOLD = 0.5
            trimmed_p = route_pixel_mask[:(H - 1) * DEC, :(W - 1) * DEC].astype(np.float32)
            blocks_p = trimmed_p.reshape(H - 1, DEC, W - 1, DEC)
            route_frac = blocks_p.mean(axis=(1, 3))
            route_cell = route_frac > ROUTE_THRESHOLD
            # Skip slot over water + bridges (would punch holes through
            # bridges; would create a slot in the riverbed). The male
            # insert is still emitted continuously across these — the
            # user can manually break it after print if needed.
            route_cell &= ~water_cell & ~bridge_cell
            n_route_cells = int(route_cell.sum())
            if n_route_cells:
                route_vertex = np.zeros((H, W), dtype=bool)
                route_vertex[:-1, :-1] |= route_cell
                route_vertex[:-1, 1:]  |= route_cell
                route_vertex[1:,  :-1] |= route_cell
                route_vertex[1:,  1:]  |= route_cell
                # Flat-bottom slot: instead of subtracting a constant
                # depth (which would make the slot floor follow terrain
                # → curved insert bottom → unprintable without supports),
                # set the slot floor to a single z value derived from the
                # LOWEST terrain along the route. The matching male
                # insert uses the same constant for its bottom, so it
                # sits flat on the print bed yet seats flush in the
                # variable-depth slot.
                min_z = float(plinth_mm) + 0.05
                # Read terrain heights at route vertices BEFORE we modify
                # zz — those values are the print-mm representations of
                # the DSM along the path.
                terrain_zz_route = zz[route_vertex]
                lowest_terrain_zz = float(terrain_zz_route.min())
                highest_terrain_zz = float(terrain_zz_route.max())
                # Bottom of slot = lowest terrain z − tail. Clamp above
                # plinth top so we never punch into the plinth.
                route_floor_z = max(
                    min_z,
                    lowest_terrain_zz - float(route_insert_tail_mm) + 0.05,
                )
                # Carve: replace zz at every route vertex with the
                # constant floor — only where it would LOWER zz (never
                # raise terrain).
                zz = np.where(
                    route_vertex,
                    np.minimum(zz, route_floor_z),
                    zz,
                )
                slot_depth_at_low  = lowest_terrain_zz  - route_floor_z
                slot_depth_at_high = highest_terrain_zz - route_floor_z
                print(f"  route: carved flat-bottom slot through {n_route_cells:,} cells "
                      f"(floor z={route_floor_z:.2f} mm, depth "
                      f"{slot_depth_at_low:.2f}–{slot_depth_at_high:.2f} mm)")
                route_slot_info = {
                    "n_cells": n_route_cells,
                    "floor_z_mm": route_floor_z,
                }
                route_polyline_local = route_line_local

    top_pts = np.stack([xx, yy, zz], axis=-1).astype(np.float32)
    bot_pts = top_pts.copy()
    bot_pts[..., 2] = 0.0

    # ─── Suspended-bridge bottoms ──────────────────────────────────────
    # For each bridge cell, lift the cell's "bottom" off the plinth so the
    # deck reads as a slab spanning empty air rather than a solid block from
    # river bed to deck. Ranges and constants are in print mm because zz is
    # already in print mm.
    TARGET_CLEAR_MM   = 2.0     # target visible air gap under the deck
    MIN_DECK_THICK_MM = 0.6     # minimum printable slab on top of the cavity
    MIN_CLEAR_MM      = 0.4     # minimum printable air gap

    bridge_bottom_z = np.full((H - 1, W - 1), float(plinth_mm), dtype=np.float32)
    if bool(bridge_cell.any()):
        # Mean deck z per cell from its 4 corner zz values (already in print mm).
        zz_cells = (zz[:-1, :-1] + zz[:-1, 1:] + zz[1:, :-1] + zz[1:, 1:]) * 0.25
        headroom = zz_cells - plinth_mm
        eff_clear = np.minimum(TARGET_CLEAR_MM, headroom - MIN_DECK_THICK_MM)
        liftable = bridge_cell & (headroom >= MIN_DECK_THICK_MM + MIN_CLEAR_MM)
        bridge_bottom_z = np.where(
            liftable,
            (plinth_mm + eff_clear).astype(np.float32),
            np.float32(plinth_mm),
        )

        # Smooth bridge_bottom_z within each connected bridge component so
        # adjacent cells share a flush underside (avoids stepped seams that
        # would create non-manifold edges between neighbouring bridge cells).
        labels, n_lab = ndimage.label(bridge_cell)

        # Fall back to solid for any bridge component touching the bbox edge
        # — the cavity would otherwise open out the side of the model.
        boundary = np.zeros_like(bridge_cell)
        boundary[0, :] = boundary[-1, :] = True
        boundary[:, 0] = boundary[:, -1] = True

        # Also fall back to solid for bridge components with no water
        # neighbour (rare LIDAR misclassification): a sealed under-deck
        # pocket would be a closed void inside the solid.
        # Build a 4-connected neighbour test for the water_cell mask.
        wc = water_cell
        water_neighbour = np.zeros_like(bridge_cell)
        water_neighbour[:-1, :] |= wc[1:, :]
        water_neighbour[1:, :]  |= wc[:-1, :]
        water_neighbour[:, :-1] |= wc[:, 1:]
        water_neighbour[:, 1:]  |= wc[:, :-1]

        for k in range(1, n_lab + 1):
            comp = (labels == k)
            if (comp & boundary).any():
                bridge_bottom_z[comp] = plinth_mm
                continue
            if not (comp & water_neighbour).any():
                bridge_bottom_z[comp] = plinth_mm
                continue
            # Flush underside across the component
            bridge_bottom_z[comp] = bridge_bottom_z[comp].min()

    bridges_lifted = int((bridge_cell & (bridge_bottom_z > plinth_mm + 1e-4)).sum())
    bridges_solid  = int((bridge_cell & (bridge_bottom_z <= plinth_mm + 1e-4)).sum())
    if bridges_lifted or bridges_solid:
        print(f"  suspended bridges: {bridges_lifted} cells; "
              f"solid fallback: {bridges_solid} cells")

    # Convenience masks used by the mesh-builder loops below.
    land_cell = ~water_cell & ~bridge_cell
    lifted_cell = bridge_cell & (bridge_bottom_z > plinth_mm + 1e-4)
    # Solid bridge cells (deck too low / bbox-boundary / no water neighbour)
    # are rendered exactly like land cells.
    solid_cell = land_cell | (bridge_cell & ~lifted_cell)

    print("Building mesh…")
    # Build mesh with consistent OUTWARD-facing normals everywhere — that's
    # the convention every slicer expects (positive volume, watertight).
    # Y axis: row 0 is at the highest y (north); row H-1 at y=0 (south).
    # Cell (j, i) corners: NW=(j,i)  NE=(j,i+1)  SW=(j+1,i)  SE=(j+1,i+1).
    triangles: list[tuple] = []

    def add(a, b, c):
        triangles.append((tuple(a), tuple(b), tuple(c)))

    # Helpers: build a vertex at (existing top_pts xy) but at an arbitrary z.
    def at_z(top_vertex, z):
        return (float(top_vertex[0]), float(top_vertex[1]), float(z))

    # Cell-top z for vertical-wall mode. Each cell becomes a flat-topped
    # extrusion at the max of its 4 corner vertex heights (the highest
    # surrounding point, so a building cell next to ground reads at
    # building height, not ground height). Vertical mode duplicates
    # corners between cells with different cell_top_z so the wall step
    # is geometrically vertical; trimesh.merge_vertices() dedupes corners
    # where neighbours agree on z. We snap to 0.05 mm in vertical mode
    # so noise smaller than any printer's z-step doesn't generate
    # sub-resolution micro-walls (each of which leaks a tiny T-junction).
    cell_top_z = np.maximum.reduce([
        zz[:-1, :-1], zz[:-1, 1:],
        zz[1:,  :-1], zz[1:,  1:],
    ]).astype(np.float32)
    if wall_style == "vertical":
        VERT_SNAP_MM = 0.05
        cell_top_z = (np.round(cell_top_z / VERT_SNAP_MM) * VERT_SNAP_MM).astype(np.float32)

    if wall_style == "sloped":
        # ─── Top: outward normal points +z ─────────────────────────────────
        # CCW from +z above: NW → SW → SE → NE.
        for j in range(H - 1):
            for i in range(W - 1):
                if water_cell[j, i]:
                    continue
                nw = top_pts[j, i];     ne = top_pts[j, i + 1]
                sw = top_pts[j + 1, i]; se = top_pts[j + 1, i + 1]
                add(nw, sw, se)
                add(nw, se, ne)
    else:
        # Vertical mode: each cell gets 4 fresh corners all at cell_top_z.
        # Adjacent cells with the same z share corners after merge_vertices.
        for j in range(H - 1):
            for i in range(W - 1):
                if water_cell[j, i]:
                    continue
                z = float(cell_top_z[j, i])
                nw = (float(xx[j,     i]),     float(yy[j,     i]),     z)
                ne = (float(xx[j,     i + 1]), float(yy[j,     i + 1]), z)
                sw = (float(xx[j + 1, i]),     float(yy[j + 1, i]),     z)
                se = (float(xx[j + 1, i + 1]), float(yy[j + 1, i + 1]), z)
                add(nw, sw, se)
                add(nw, se, ne)

    # ─── Bottom: outward normal points −z ──────────────────────────────
    # CCW from -z below: NW → NE → SE → SW.
    # - Solid (land or solid-fallback bridge): bottom at z=0.
    # - Lifted bridge: bottom at bridge_bottom_z (the deck underside).
    # - Water: skip.
    for j in range(H - 1):
        for i in range(W - 1):
            if water_cell[j, i]:
                continue
            if lifted_cell[j, i]:
                bz = float(bridge_bottom_z[j, i])
                nw = at_z(top_pts[j,     i],     bz)
                ne = at_z(top_pts[j,     i + 1], bz)
                sw = at_z(top_pts[j + 1, i],     bz)
                se = at_z(top_pts[j + 1, i + 1], bz)
            else:
                nw = bot_pts[j, i];     ne = bot_pts[j, i + 1]
                sw = bot_pts[j + 1, i]; se = bot_pts[j + 1, i + 1]
            add(nw, ne, se)
            add(nw, se, sw)

    # ─── Inner walls — 3-state classifier (solid / water / lifted) ────
    # For each shared edge between cells A and B we may emit one of:
    #   • full-height wall on the SOLID side (solid ↔ water)
    #   • deck-side wall on the LIFTED side from deck z down to
    #     bridge_bottom_z  (lifted ↔ water)
    #   • under-deck wall on the SOLID side from z=0 up to
    #     bridge_bottom_z[lifted]  (solid ↔ lifted)
    #   • step wall between two solid cells with different cell_top_z
    #     (vertical mode only — sloped mode handles this via shared top
    #     vertices that just slope between adjacent z's)
    #   • nothing  (water↔water, lifted↔lifted, sloped solid↔solid)
    # Outward normals always face away from solid material.

    if wall_style == "sloped":
        # Vertical edges between cells (j, i) and (j, i+1) live at vertex column i+1.
        for j in range(H - 1):
            for i in range(W - 2):
                L_w, L_l, L_s = water_cell[j, i], lifted_cell[j, i], solid_cell[j, i]
                R_w, R_l, R_s = water_cell[j, i + 1], lifted_cell[j, i + 1], solid_cell[j, i + 1]
                if (L_w == R_w) and (L_l == R_l) and (L_s == R_s):
                    continue
                tn = top_pts[j,     i + 1]
                ts = top_pts[j + 1, i + 1]
                bn = bot_pts[j,     i + 1]   # z=0
                bs = bot_pts[j + 1, i + 1]   # z=0

                # ── solid ↔ water (full height) ────────────────────────────
                if L_s and R_w:   # solid on left, water on right → outward +x
                    add(tn, bs, bn)
                    add(tn, ts, bs)
                elif L_w and R_s: # water on left, solid on right → outward -x
                    add(tn, bn, bs)
                    add(tn, bs, ts)

                # ── lifted ↔ water (deck-side wall on the bridge) ─────────
                elif L_l and R_w:   # lifted on left, water on right → outward +x
                    bz = float(bridge_bottom_z[j, i])
                    tn_b = at_z(tn, bz);   ts_b = at_z(ts, bz)
                    add(tn, ts, ts_b)
                    add(tn, ts_b, tn_b)
                elif L_w and R_l:   # water on left, lifted on right → outward -x
                    bz = float(bridge_bottom_z[j, i + 1])
                    tn_b = at_z(tn, bz);   ts_b = at_z(ts, bz)
                    add(tn, ts_b, ts)
                    add(tn, tn_b, ts_b)

                # ── solid ↔ lifted (under-deck wall on solid side) ────────
                elif L_s and R_l:   # outward into bridge cavity = +x
                    bz = float(bridge_bottom_z[j, i + 1])
                    tn_b = at_z(tn, bz);   ts_b = at_z(ts, bz)
                    add(tn_b, ts_b, bs)
                    add(tn_b, bs, bn)
                elif L_l and R_s:   # outward into bridge cavity = -x
                    bz = float(bridge_bottom_z[j, i])
                    tn_b = at_z(tn, bz);   ts_b = at_z(ts, bz)
                    add(tn_b, bs, ts_b)
                    add(tn_b, bn, bs)

        # Horizontal edges between cells (j, i) and (j+1, i) live at vertex row j+1.
        for j in range(H - 2):
            for i in range(W - 1):
                N_w, N_l, N_s = water_cell[j,     i], lifted_cell[j,     i], solid_cell[j,     i]
                S_w, S_l, S_s = water_cell[j + 1, i], lifted_cell[j + 1, i], solid_cell[j + 1, i]
                if (N_w == S_w) and (N_l == S_l) and (N_s == S_s):
                    continue
                tw = top_pts[j + 1, i]
                te = top_pts[j + 1, i + 1]
                bw = bot_pts[j + 1, i]   # z=0
                be = bot_pts[j + 1, i + 1]

                # ── solid ↔ water (full height) ────────────────────────────
                if N_s and S_w:   # solid north, water south → outward -y
                    add(tw, bw, be)
                    add(tw, be, te)
                elif N_w and S_s: # water north, solid south → outward +y
                    add(tw, be, bw)
                    add(tw, te, be)

                # ── lifted ↔ water (deck-side) ────────────────────────────
                elif N_l and S_w:   # lifted north, water south → outward -y
                    bz = float(bridge_bottom_z[j, i])
                    tw_b = at_z(tw, bz);   te_b = at_z(te, bz)
                    add(tw, te_b, te)
                    add(tw, tw_b, te_b)
                elif N_w and S_l:   # water north, lifted south → outward +y
                    bz = float(bridge_bottom_z[j + 1, i])
                    tw_b = at_z(tw, bz);   te_b = at_z(te, bz)
                    add(tw, te, te_b)
                    add(tw, te_b, tw_b)

                # ── solid ↔ lifted (under-deck) ───────────────────────────
                elif N_s and S_l:   # solid north, lifted south. The bridge
                    # cavity sits BELOW the deck on the lifted (south) side
                    # and extends back under the solid soffit at the
                    # boundary. The wall faces the cavity → outward = -y.
                    bz = float(bridge_bottom_z[j + 1, i])
                    tw_b = at_z(tw, bz);   te_b = at_z(te, bz)
                    add(tw_b, bw, be)
                    add(tw_b, be, te_b)
                elif N_l and S_s:   # cavity is on the north side → outward +y
                    bz = float(bridge_bottom_z[j, i])
                    tw_b = at_z(tw, bz);   te_b = at_z(te, bz)
                    add(tw_b, be, bw)
                    add(tw_b, te_b, be)

        # ─── Outer walls (bbox edges) ─────────────────────────────────────
        # Boundary cells are forced to solid by the bridge_bottom_z fallback,
        # so only the existing solid/water dichotomy applies here.
        # NORTH (row 0, outward +y).
        for i in range(W - 1):
            if water_cell[0, i]:
                continue
            tw_, te_ = top_pts[0, i], top_pts[0, i + 1]
            bw_, be_ = bot_pts[0, i], bot_pts[0, i + 1]
            add(tw_, be_, bw_)
            add(tw_, te_, be_)

        # SOUTH (row H-1, outward -y).
        for i in range(W - 1):
            if water_cell[H - 2, i]:
                continue
            tw_, te_ = top_pts[H - 1, i], top_pts[H - 1, i + 1]
            bw_, be_ = bot_pts[H - 1, i], bot_pts[H - 1, i + 1]
            add(tw_, bw_, be_)
            add(tw_, be_, te_)

        # WEST (col 0, outward -x).
        for j in range(H - 1):
            if water_cell[j, 0]:
                continue
            tn_, ts_ = top_pts[j, 0], top_pts[j + 1, 0]
            bn_, bs_ = bot_pts[j, 0], bot_pts[j + 1, 0]
            add(tn_, bn_, bs_)
            add(tn_, bs_, ts_)

        # EAST (col W-1, outward +x).
        for j in range(H - 1):
            if water_cell[j, W - 2]:
                continue
            tn_, ts_ = top_pts[j, W - 1], top_pts[j + 1, W - 1]
            bn_, bs_ = bot_pts[j, W - 1], bot_pts[j + 1, W - 1]
            add(tn_, bs_, bn_)
            add(tn_, ts_, bs_)

    else:
        # ─── Vertical-wall mode ──────────────────────────────────────
        # Each cell has a flat top at cell_top_z[j,i]. Walls fill the
        # height step at every shared edge between cells with different
        # tops, plus the existing solid↔water and bridge cases.
        # All wall corners use the cell's xy + cell_top_z for the top,
        # and z=0 (or bridge_bottom_z) for the bottom. trimesh's
        # merge_vertices() dedupes coincident corners across cells.
        def _corner(j_v, i_v, z):
            return (float(xx[j_v, i_v]), float(yy[j_v, i_v]), float(z))

        # Vertical edges between cells (j, i) and (j, i+1) at vertex column i+1.
        for j in range(H - 1):
            for i in range(W - 2):
                L_w, L_l, L_s = water_cell[j, i], lifted_cell[j, i], solid_cell[j, i]
                R_w, R_l, R_s = water_cell[j, i + 1], lifted_cell[j, i + 1], solid_cell[j, i + 1]

                # Solid ↔ water — full-height wall on the solid side.
                if L_s and R_w:
                    zL = float(cell_top_z[j, i])
                    tn = _corner(j,     i + 1, zL)
                    ts = _corner(j + 1, i + 1, zL)
                    bn = _corner(j,     i + 1, 0.0)
                    bs = _corner(j + 1, i + 1, 0.0)
                    add(tn, bs, bn); add(tn, ts, bs)
                elif L_w and R_s:
                    zR = float(cell_top_z[j, i + 1])
                    tn = _corner(j,     i + 1, zR)
                    ts = _corner(j + 1, i + 1, zR)
                    bn = _corner(j,     i + 1, 0.0)
                    bs = _corner(j + 1, i + 1, 0.0)
                    add(tn, bn, bs); add(tn, bs, ts)

                # Solid ↔ solid — step wall when the two cell tops differ.
                elif L_s and R_s:
                    zL = float(cell_top_z[j, i])
                    zR = float(cell_top_z[j, i + 1])
                    if abs(zL - zR) < 1e-6:
                        continue
                    if zL > zR:   # higher on left → wall faces +x (outward east)
                        tn = _corner(j,     i + 1, zL)
                        ts = _corner(j + 1, i + 1, zL)
                        bn = _corner(j,     i + 1, zR)
                        bs = _corner(j + 1, i + 1, zR)
                        add(tn, bs, bn); add(tn, ts, bs)
                    else:         # higher on right → wall faces -x (outward west)
                        tn = _corner(j,     i + 1, zR)
                        ts = _corner(j + 1, i + 1, zR)
                        bn = _corner(j,     i + 1, zL)
                        bs = _corner(j + 1, i + 1, zL)
                        add(tn, bn, bs); add(tn, bs, ts)

                # Lifted ↔ water (deck-side) and solid ↔ lifted (under-deck) —
                # same logic as sloped, but using cell_top_z for the deck top.
                elif L_l and R_w:
                    bz = float(bridge_bottom_z[j, i]); zL = float(cell_top_z[j, i])
                    tn = _corner(j, i + 1, zL); ts = _corner(j + 1, i + 1, zL)
                    tn_b = _corner(j, i + 1, bz); ts_b = _corner(j + 1, i + 1, bz)
                    add(tn, ts, ts_b); add(tn, ts_b, tn_b)
                elif L_w and R_l:
                    bz = float(bridge_bottom_z[j, i + 1]); zR = float(cell_top_z[j, i + 1])
                    tn = _corner(j, i + 1, zR); ts = _corner(j + 1, i + 1, zR)
                    tn_b = _corner(j, i + 1, bz); ts_b = _corner(j + 1, i + 1, bz)
                    add(tn, ts_b, ts); add(tn, tn_b, ts_b)
                elif L_s and R_l:
                    bz = float(bridge_bottom_z[j, i + 1])
                    tn_b = _corner(j, i + 1, bz); ts_b = _corner(j + 1, i + 1, bz)
                    bn = _corner(j, i + 1, 0.0); bs = _corner(j + 1, i + 1, 0.0)
                    add(tn_b, ts_b, bs); add(tn_b, bs, bn)
                elif L_l and R_s:
                    bz = float(bridge_bottom_z[j, i])
                    tn_b = _corner(j, i + 1, bz); ts_b = _corner(j + 1, i + 1, bz)
                    bn = _corner(j, i + 1, 0.0); bs = _corner(j + 1, i + 1, 0.0)
                    add(tn_b, bs, ts_b); add(tn_b, bn, bs)

        # Horizontal edges between cells (j, i) and (j+1, i) at vertex row j+1.
        for j in range(H - 2):
            for i in range(W - 1):
                N_w, N_l, N_s = water_cell[j,     i], lifted_cell[j,     i], solid_cell[j,     i]
                S_w, S_l, S_s = water_cell[j + 1, i], lifted_cell[j + 1, i], solid_cell[j + 1, i]

                if N_s and S_w:
                    zN = float(cell_top_z[j, i])
                    tw = _corner(j + 1, i,     zN); te = _corner(j + 1, i + 1, zN)
                    bw = _corner(j + 1, i,     0.0); be = _corner(j + 1, i + 1, 0.0)
                    add(tw, bw, be); add(tw, be, te)
                elif N_w and S_s:
                    zS = float(cell_top_z[j + 1, i])
                    tw = _corner(j + 1, i,     zS); te = _corner(j + 1, i + 1, zS)
                    bw = _corner(j + 1, i,     0.0); be = _corner(j + 1, i + 1, 0.0)
                    add(tw, be, bw); add(tw, te, be)

                # Solid ↔ solid step wall.
                elif N_s and S_s:
                    zN = float(cell_top_z[j, i])
                    zS = float(cell_top_z[j + 1, i])
                    if abs(zN - zS) < 1e-6:
                        continue
                    if zN > zS:   # higher on north → wall faces -y (outward south)
                        tw = _corner(j + 1, i,     zN); te = _corner(j + 1, i + 1, zN)
                        bw = _corner(j + 1, i,     zS); be = _corner(j + 1, i + 1, zS)
                        add(tw, bw, be); add(tw, be, te)
                    else:         # higher on south → wall faces +y (outward north)
                        tw = _corner(j + 1, i,     zS); te = _corner(j + 1, i + 1, zS)
                        bw = _corner(j + 1, i,     zN); be = _corner(j + 1, i + 1, zN)
                        add(tw, be, bw); add(tw, te, be)

                elif N_l and S_w:
                    bz = float(bridge_bottom_z[j, i]); zN = float(cell_top_z[j, i])
                    tw = _corner(j + 1, i, zN); te = _corner(j + 1, i + 1, zN)
                    tw_b = _corner(j + 1, i, bz); te_b = _corner(j + 1, i + 1, bz)
                    add(tw, te_b, te); add(tw, tw_b, te_b)
                elif N_w and S_l:
                    bz = float(bridge_bottom_z[j + 1, i]); zS = float(cell_top_z[j + 1, i])
                    tw = _corner(j + 1, i, zS); te = _corner(j + 1, i + 1, zS)
                    tw_b = _corner(j + 1, i, bz); te_b = _corner(j + 1, i + 1, bz)
                    add(tw, te, te_b); add(tw, te_b, tw_b)
                elif N_s and S_l:
                    bz = float(bridge_bottom_z[j + 1, i])
                    tw_b = _corner(j + 1, i, bz); te_b = _corner(j + 1, i + 1, bz)
                    bw = _corner(j + 1, i, 0.0); be = _corner(j + 1, i + 1, 0.0)
                    add(tw_b, bw, be); add(tw_b, be, te_b)
                elif N_l and S_s:
                    bz = float(bridge_bottom_z[j, i])
                    tw_b = _corner(j + 1, i, bz); te_b = _corner(j + 1, i + 1, bz)
                    bw = _corner(j + 1, i, 0.0); be = _corner(j + 1, i + 1, 0.0)
                    add(tw_b, be, bw); add(tw_b, te_b, be)

        # ─── Outer walls (bbox edges) ────────────────────────────────
        # In vertical mode we walk each cell's outer wall as a polygon and
        # insert a vertex at the SHORTER neighbour cell's height on the
        # adjacent edge. Without this, the inner step wall between adjacent
        # outer cells creates a T-junction against the taller cell's wall.
        # ``poly`` is laid out CCW from the *outside* face, so a fan from
        # poly[0] gives the correct outward normal.

        def _fan(poly):
            for k in range(1, len(poly) - 1):
                add(poly[0], poly[k], poly[k + 1])

        # NORTH (row 0, outward +y). Polygon order: BR, BL, [M_L], TL, TR, [M_R].
        for i in range(W - 1):
            if water_cell[0, i]:
                continue
            z = float(cell_top_z[0, i])
            poly = [_corner(0, i + 1, 0.0), _corner(0, i, 0.0)]
            if i > 0 and not water_cell[0, i - 1]:
                z_n = float(cell_top_z[0, i - 1])
                if 0.0 < z_n < z:
                    poly.append(_corner(0, i, z_n))
            poly.append(_corner(0, i, z))
            poly.append(_corner(0, i + 1, z))
            if i < W - 2 and not water_cell[0, i + 1]:
                z_n = float(cell_top_z[0, i + 1])
                if 0.0 < z_n < z:
                    poly.append(_corner(0, i + 1, z_n))
            _fan(poly)

        # SOUTH (row H-1, outward -y). Polygon order: BL, BR, [M_R], TR, TL, [M_L].
        for i in range(W - 1):
            if water_cell[H - 2, i]:
                continue
            z = float(cell_top_z[H - 2, i])
            poly = [_corner(H - 1, i, 0.0), _corner(H - 1, i + 1, 0.0)]
            if i < W - 2 and not water_cell[H - 2, i + 1]:
                z_n = float(cell_top_z[H - 2, i + 1])
                if 0.0 < z_n < z:
                    poly.append(_corner(H - 1, i + 1, z_n))
            poly.append(_corner(H - 1, i + 1, z))
            poly.append(_corner(H - 1, i, z))
            if i > 0 and not water_cell[H - 2, i - 1]:
                z_n = float(cell_top_z[H - 2, i - 1])
                if 0.0 < z_n < z:
                    poly.append(_corner(H - 1, i, z_n))
            _fan(poly)

        # WEST (col 0, outward -x). Polygon order: BN, BS, [M_S], TS, TN, [M_N].
        for j in range(H - 1):
            if water_cell[j, 0]:
                continue
            z = float(cell_top_z[j, 0])
            poly = [_corner(j, 0, 0.0), _corner(j + 1, 0, 0.0)]
            if j < H - 2 and not water_cell[j + 1, 0]:
                z_n = float(cell_top_z[j + 1, 0])
                if 0.0 < z_n < z:
                    poly.append(_corner(j + 1, 0, z_n))
            poly.append(_corner(j + 1, 0, z))
            poly.append(_corner(j, 0, z))
            if j > 0 and not water_cell[j - 1, 0]:
                z_n = float(cell_top_z[j - 1, 0])
                if 0.0 < z_n < z:
                    poly.append(_corner(j, 0, z_n))
            _fan(poly)

        # EAST (col W-1, outward +x). Polygon order: BS, BN, [M_N], TN, TS, [M_S].
        for j in range(H - 1):
            if water_cell[j, W - 2]:
                continue
            z = float(cell_top_z[j, W - 2])
            poly = [_corner(j + 1, W - 1, 0.0), _corner(j, W - 1, 0.0)]
            if j > 0 and not water_cell[j - 1, W - 2]:
                z_n = float(cell_top_z[j - 1, W - 2])
                if 0.0 < z_n < z:
                    poly.append(_corner(j, W - 1, z_n))
            poly.append(_corner(j, W - 1, z))
            poly.append(_corner(j + 1, W - 1, z))
            if j < H - 2 and not water_cell[j + 1, W - 2]:
                z_n = float(cell_top_z[j + 1, W - 2])
                if 0.0 < z_n < z:
                    poly.append(_corner(j + 1, W - 1, z_n))
            _fan(poly)

    triangles_arr = np.array(triangles, dtype=np.float32)
    print(f"  {len(triangles_arr):,} triangles before repair")

    # Auto-repair pass with trimesh: merges duplicated vertices, removes
    # degenerate faces, fills small holes, and re-orients winding so the
    # outward-normal convention we built actually holds end-to-end. Then we
    # drop any disconnected "island" bodies (small landmasses surrounded by
    # water — they are tiny, often non-printable, and confuse slicers that
    # expect a single body).
    try:
        import trimesh
        verts = triangles_arr.reshape(-1, 3)
        faces = np.arange(len(verts)).reshape(-1, 3)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        mesh.merge_vertices()
        mesh.update_faces(mesh.unique_faces())
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.remove_unreferenced_vertices()
        try:
            mesh.fill_holes()
        except Exception:
            pass
        mesh.fix_normals()
        if mesh.volume < 0:
            mesh.invert()

        # Drop tiny disconnected bodies (river islands, isolated bridge
        # cells, etc.). Keep only the largest body — that's the model.
        components = mesh.split(only_watertight=False)
        if len(components) > 1:
            kept = max(components, key=lambda c: abs(c.volume))
            dropped = len(components) - 1
            dropped_vol = sum(abs(c.volume) for c in components if c is not kept)
            print(f"  dropping {dropped} disconnected island(s) "
                  f"({dropped_vol:.0f} mm³ total)")
            mesh = kept

        triangles_arr = mesh.triangles.astype(np.float32)
        print(
            f"  after repair: {len(triangles_arr):,} triangles, "
            f"watertight={mesh.is_watertight}, volume={mesh.volume:.0f} mm³, "
            f"bodies={mesh.body_count}"
        )
    except ImportError:
        print("  (trimesh not available — skipping repair pass)")

    if preview:
        # Clean print-grade mesh → server-only path (withheld from browser).
        if clean_out_path:
            print(f"Writing CLEAN (server-only) {clean_out_path}…")
            _write_stl(triangles_arr, Path(clean_out_path), header="cityform tier3 hollow water")
        # Web-served file is the degraded + framed + header-stamped preview.
        print(f"Writing PREVIEW (web) {out_path}…")
        _write_stl(_make_preview_triangles(triangles_arr), Path(out_path),
                   header="CITYFORM PREVIEW - NOT FOR PRINT (c) cityform")
    else:
        print(f"Writing {out_path}…")
        _write_stl(triangles_arr, Path(out_path), header="cityform tier3 hollow water")
    file_mb = Path(out_path).stat().st_size / 1e6
    print(f"  done — {file_mb:.1f} MB")

    # ─── Route insert (Phase 9d) ────────────────────────────────────
    # Builds the matching male piece using the same polyline + DSM that
    # carved the slot. Skipped silently if no route was supplied or the
    # caller didn't pass a `route_out_path`.
    summary = {
        "triangles": int(len(triangles_arr)),
        "water_cells": cells_water,
        "total_cells": cells_total,
        "bridges_preserved": int(preserved),
        "bridges_lifted": int(bridges_lifted),
        "bridges_solid_fallback": int(bridges_solid),
        "file_mb": round(file_mb, 2),
    }
    if route_polyline_local is not None and route_out_path:
        try:
            # Pass the slot's floor_z so the insert's flat bottom sits
            # at exactly the slot floor — they match by construction.
            insert_floor_z = (route_slot_info or {}).get("floor_z_mm")
            insert_triangles = _build_route_insert_mesh(
                polyline_local=route_polyline_local,
                dsm=dsm,
                bbox=bbox,
                z_floor=z_floor,
                SCALE=SCALE,
                z_exaggeration=z_exaggeration,
                plinth_mm=plinth_mm,
                width_mm=route_width_mm,
                tolerance_mm=route_tolerance_mm,
                raised_mm=route_raised_mm,
                insert_tail_mm=route_insert_tail_mm,
                insert_floor_z_mm=insert_floor_z,
                top_smooth_window=route_top_smooth_window,
                is_loop=bool(route_loop),
            )
            if len(insert_triangles):
                _write_stl(insert_triangles, Path(route_out_path),
                           header="cityform route insert")
                route_file_mb = Path(route_out_path).stat().st_size / 1e6
                print(f"Writing {route_out_path}… {route_file_mb:.2f} MB, "
                      f"{len(insert_triangles):,} triangles")
                summary["route_path"] = str(route_out_path)
                summary["route_triangles"] = int(len(insert_triangles))
                summary["route_file_mb"] = round(route_file_mb, 2)
                summary["route_length_m"] = round(route_polyline_local.length, 1)
            else:
                print("  route insert: too short to build")
        except Exception as exc:    # noqa: BLE001 — main STL must succeed regardless
            print(f"  route insert FAILED: {exc}")

    return summary
