"""Server-side STL preview renderers for the library.

PNGs (and one GIF) per generated city, produced inside `_save_to_library`
after the STL has been copied into the library folder:

  render_hero(stl, png)            → shaded isometric, white background
  render_three_quarter(stl, png)   → low-angle 3/4 view, white background
  render_lifestyle(stl, png)       → hero composited onto a desk-surface scene
  render_packaging(stl, png)       → hero scaled into a kraft-mailer mock
  render_turntable(stl, gif)       → 16-frame rotating GIF, ~800×600
  render_pinterest_pin(stl, png)   → 1000×1500 portrait with brand overlay
  render_wireframe(stl, png)       → top-down line art, transparent
  render_sticker(...)              → SVG sticker that wraps the wireframe
  render_laser_label(...)          → standalone laser-cut label SVG (57.6×7.6 mm)

All raster renders are headless (no GL context). The 3D poses share an
internal `_render_view()` helper that wraps matplotlib + trimesh decimation;
wireframe stream-parses the binary STL into a heightmap and runs Sobel edges
for clean architectural lines without triangulation noise.

CLI smoke test:
    .venv/bin/python cityform-tool/render.py <stl> <hero.png> <wireframe.png>
"""

from __future__ import annotations

import re
import struct
import sys
import time
from pathlib import Path

import numpy as np
import trimesh


def _load_and_decimate(stl_path: Path, target_faces: int) -> trimesh.Trimesh:
    """Load STL via trimesh and reduce to roughly `target_faces` triangles.
    trimesh.simplify_quadric_decimation has no effect when the mesh is
    already smaller, so it's safe to call unconditionally."""
    mesh = trimesh.load(str(stl_path), force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Loaded geometry is not a triangle mesh: {type(mesh)}")
    if len(mesh.faces) > target_faces:
        try:
            mesh = mesh.simplify_quadric_decimation(face_count=target_faces)
        except Exception:
            # Older trimesh: positional name was different. Fall back to a
            # fixed ratio.
            ratio = target_faces / max(len(mesh.faces), 1)
            mesh = mesh.simplify_quadric_decimation(percent=ratio)
    return mesh


def _trimesh_to_polydata(mesh: trimesh.Trimesh):
    """Convert a trimesh.Trimesh to pv.PolyData. PyVista expects faces as
    a flat array prefixed by the per-face vertex count (3 for triangles)."""
    import pyvista as pv
    faces = np.column_stack(
        (np.full(len(mesh.faces), 3, dtype=np.int64), mesh.faces.astype(np.int64))
    ).flatten()
    return pv.PolyData(np.asarray(mesh.vertices, dtype=np.float64), faces)


def _set_camera(plotter, focal, *, elev_deg: float, azim_deg: float, distance: float) -> None:
    """Position the PyVista camera with matplotlib's (elev, azim) convention.

    elev_deg = angle above the XY plane (0 = horizontal, 90 = top-down)
    azim_deg = rotation around +Z (0 looks down +X, 90 looks down +Y)
    """
    import math
    e = math.radians(elev_deg)
    a = math.radians(azim_deg)
    dx = distance * math.cos(e) * math.cos(a)
    dy = distance * math.cos(e) * math.sin(a)
    dz = distance * math.sin(e)
    pos = (focal[0] + dx, focal[1] + dy, focal[2] + dz)
    plotter.camera_position = [pos, focal, (0.0, 0.0, 1.0)]


def _render_view(
    mesh_or_path,
    png_path: Path | str,
    *,
    width: int = 1200,
    height: int = 900,
    elev: float = 35.0,
    azim: float = -50.0,
    target_faces: int = 20_000,    # kept for signature compat — not used
    transparent: bool = False,
    bg_color: str = "white",
    extra_meshes=None,             # iterable of (pv.PolyData, kwargs) tuples
    combined_bbox=None,            # if set, frame the camera around this bbox
) -> Path:
    """Shared 3D renderer used by every shaded pose. PyVista (VTK) backend
    so we can render the FULL-resolution mesh with proper depth-buffered
    GPU shading instead of matplotlib's painter-algorithm decimated view.

    `mesh_or_path` accepts either a path to an STL or a pre-loaded
    `trimesh.Trimesh` (turntable reuses one mesh across frames). Camera
    is parameterised by `(elev, azim)` in matplotlib convention so existing
    pose presets transfer over.
    """
    import pyvista as pv
    del target_faces  # ignored: we render the full mesh now

    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(mesh_or_path, trimesh.Trimesh):
        tm = mesh_or_path
    else:
        tm = trimesh.load(str(mesh_or_path), force="mesh")
        if not isinstance(tm, trimesh.Trimesh):
            raise ValueError(f"Loaded geometry is not a triangle mesh: {type(tm)}")

    pv_mesh = _trimesh_to_polydata(tm)

    if combined_bbox is not None:
        bb_min, bb_max = combined_bbox
    else:
        bb_min, bb_max = tm.bounds
    cx = float((bb_min[0] + bb_max[0]) / 2)
    cy = float((bb_min[1] + bb_max[1]) / 2)
    cz = float((bb_min[2] + bb_max[2]) / 2)
    extent = float(max(bb_max[0] - bb_min[0],
                       bb_max[1] - bb_min[1],
                       bb_max[2] - bb_min[2]))
    distance = extent * 2.4

    pl = pv.Plotter(off_screen=True, window_size=(width, height))
    if transparent:
        pl.set_background([1, 1, 1])    # transparent flag passed at screenshot time
    else:
        pl.set_background(bg_color)

    pl.add_mesh(
        pv_mesh,
        color="#bdc1c4",                # warm grey, close to PLA tone
        smooth_shading=True,
        ambient=0.30,
        diffuse=0.85,
        specular=0.18,
        specular_power=15,
    )
    if extra_meshes:
        for extra, kwargs in extra_meshes:
            pl.add_mesh(extra, **kwargs)

    _set_camera(pl, focal=(cx, cy, cz),
                elev_deg=elev, azim_deg=azim, distance=distance)
    pl.enable_anti_aliasing("ssaa")     # GPU SSAA — kills jaggies on building edges
    try:
        # Screen-space ambient occlusion — soft contact shadows in the gaps
        # between buildings. Makes the model read with proper depth instead
        # of looking like a flat grey paper cut-out.
        pl.enable_ssao(radius=2.0, bias=0.5, kernel_size=64, blur=True)
    except Exception:    # noqa: BLE001 — older VTK / no GL extension; soldier on without it
        pass

    # Use screenshot() directly so we can pass transparent_background.
    pl.screenshot(str(png_path), transparent_background=transparent)
    pl.close()
    return png_path


# Path to the printable presentation base accessory. This is the real
# STL the user prints in black filament — `render_on_base` loads it so
# the rendered preview matches what the customer actually sees, not a
# stand-in geometric box.
BASE_ACCESSORY_STL = Path(__file__).resolve().parent / "assets" / "accessories" / "9x9 base (base only).stl"


def _load_base_accessory_zup() -> trimesh.Trimesh:
    """Load the 9x9 base STL, rotate to Z-up, centre on XY, top at z=0.

    The exported STL is Y-up (the 16 mm thickness sits along Y, not Z).
    +90° rotation about +X maps `(x, y, z) → (x, -z, y)`, so the original
    +Y top of the base becomes the +Z top after rotation — staying right-
    side-up rather than flipping over.
    """
    base = trimesh.load(str(BASE_ACCESSORY_STL), force="mesh")
    if not isinstance(base, trimesh.Trimesh):
        raise RuntimeError(f"Base STL is not a triangle mesh: {type(base)}")
    R = trimesh.transformations.rotation_matrix(
        angle=np.pi / 2, direction=[1.0, 0.0, 0.0],
    )
    base.apply_transform(R)
    bb_min, bb_max = base.bounds
    cx = float((bb_min[0] + bb_max[0]) / 2)
    cy = float((bb_min[1] + bb_max[1]) / 2)
    z_top = float(bb_max[2])
    base.apply_translation([-cx, -cy, -z_top])    # top at z=0, centred on XY
    return base


def render_on_base(
    stl_path: Path | str,
    png_path: Path | str,
    *,
    width: int = 1200,
    height: int = 900,
) -> Path:
    """Hero pose with the actual printable `9x9 base (base only).stl`
    accessory beneath the model — matches the physical printed setup
    where the cityform sits on the black base accessory.

    The base STL is auto-oriented to Z-up and positioned so its top
    surface meets the model's z_min, centred under the model's XY
    footprint. The base is rendered in solid black; the model in grey
    like the regular hero shot.
    """
    stl_path = Path(stl_path)
    tm = trimesh.load(str(stl_path), force="mesh")
    if not isinstance(tm, trimesh.Trimesh):
        raise ValueError(f"Loaded geometry is not a triangle mesh: {type(tm)}")

    base = _load_base_accessory_zup()
    model_min, model_max = tm.bounds
    model_cx = float((model_min[0] + model_max[0]) / 2)
    model_cy = float((model_min[1] + model_max[1]) / 2)
    model_z_min = float(model_min[2])
    # Stick the base's top to the bottom of the model + centre on the
    # model's footprint. Since _load_base_accessory_zup() returns the
    # base centred at (0,0,0) with top at z=0, a single translation
    # parks it correctly.
    base.apply_translation([model_cx, model_cy, model_z_min])

    base_pv = _trimesh_to_polydata(base)
    base_kwargs = dict(
        color="#0d0d10",
        smooth_shading=True,
        ambient=0.18,
        diffuse=0.55,
        specular=0.30,
        specular_power=25,
    )

    # Frame the camera around the union of model + base so neither
    # dominates and we don't crop edges.
    base_min, base_max = base.bounds
    combined = (
        np.minimum(model_min, base_min),
        np.maximum(model_max, base_max),
    )

    return _render_view(
        tm, png_path,
        width=width, height=height,
        elev=35.0, azim=-50.0,
        transparent=False, bg_color="white",
        extra_meshes=[(base_pv, base_kwargs)],
        combined_bbox=combined,
    )


def render_hero(
    stl_path: Path | str,
    png_path: Path | str,
    *,
    width: int = 1200,
    height: int = 900,
    target_faces: int = 20_000,
) -> Path:
    """Shaded isometric render. White background, no axes — Etsy-listing style."""
    return _render_view(
        stl_path, png_path,
        width=width, height=height,
        elev=35.0, azim=-50.0,
        target_faces=target_faces,
        transparent=False, bg_color="white",
    )


def render_three_quarter(
    stl_path: Path | str,
    png_path: Path | str,
    *,
    width: int = 1200,
    height: int = 900,
    target_faces: int = 20_000,
) -> Path:
    """Lower-angle three-quarter view. Reads as 'in the room' rather than
    'top-down map' — complements the hero on Etsy listings."""
    return _render_view(
        stl_path, png_path,
        width=width, height=height,
        elev=22.0, azim=-30.0,
        target_faces=target_faces,
        transparent=False, bg_color="white",
    )


# ── Composite renders (hero + PIL post-processing) ─────────────────────

def _draw_drop_shadow(img, *, blur_radius: float = 14.0, opacity: float = 0.28):
    """Return an RGBA image: a soft elliptical drop-shadow underneath the
    model, sized from the alpha channel of `img`. Used by lifestyle +
    packaging composites so the model doesn't look pasted on."""
    from PIL import Image, ImageDraw, ImageFilter

    w, h = img.size
    # Find horizontal extent of the alpha mask near the bottom of the image —
    # that's where the shadow should sit.
    alpha = img.split()[-1]
    bbox = alpha.getbbox()
    if not bbox:
        return Image.new("RGBA", img.size, (0, 0, 0, 0))
    left, _top, right, bottom = bbox
    cx = (left + right) // 2
    cy = bottom
    ell_w = max(60, int((right - left) * 0.85))
    ell_h = max(14, int(ell_w * 0.18))

    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(shadow)
    draw.ellipse(
        (cx - ell_w // 2, cy - ell_h // 2, cx + ell_w // 2, cy + ell_h // 2),
        fill=(0, 0, 0, int(255 * opacity)),
    )
    return shadow.filter(ImageFilter.GaussianBlur(radius=blur_radius))


def render_lifestyle(
    stl_path: Path | str,
    png_path: Path | str,
    *,
    width: int = 1200,
    height: int = 900,
    target_faces: int = 20_000,
) -> Path:
    """Hero composited onto a procedural desk-surface scene. Vertical gradient
    plus a soft drop-shadow under the model. No external assets — keeps the
    offline guarantee while filling Etsy's lifestyle photo slot."""
    from PIL import Image, ImageDraw

    stl_path = Path(stl_path)
    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Render the hero pose with transparent background.
    tmp_dir = png_path.parent
    transparent_path = tmp_dir / f".{png_path.stem}_lifestyle_tmp.png"
    try:
        _render_view(
            stl_path, transparent_path,
            width=width, height=height,
            elev=22.0, azim=-30.0,
            target_faces=target_faces,
            transparent=True,
        )
        model = Image.open(transparent_path).convert("RGBA")
    finally:
        try:
            transparent_path.unlink(missing_ok=True)
        except OSError:
            pass

    # Resize the rendered model so it sits comfortably inside the canvas
    # with breathing room top + sides.
    canvas_size = (width, height)
    canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))

    target_w = int(width * 0.70)
    aspect = model.height / max(model.width, 1)
    target_h = int(target_w * aspect)
    if target_h > int(height * 0.78):
        target_h = int(height * 0.78)
        target_w = int(target_h / max(aspect, 1e-6))
    model = model.resize((target_w, target_h), Image.LANCZOS)

    # 2. Build the desk-surface gradient. Top half is a soft warm light;
    # bottom half is a deeper neutral that suggests a worktop edge.
    bg = Image.new("RGB", canvas_size, "#efece6")
    pixels = bg.load()
    for y in range(height):
        t = y / max(height - 1, 1)
        # Subtle vertical fade — top: #f4f1ea, bottom: #d6d2c9
        r = int(244 * (1 - t) + 214 * t)
        g = int(241 * (1 - t) + 210 * t)
        b = int(234 * (1 - t) + 201 * t)
        for x in range(width):
            pixels[x, y] = (r, g, b)

    # 3. Drop shadow + paste model.
    paste_x = (width - target_w) // 2
    paste_y = int(height * 0.10)
    canvas.paste(model, (paste_x, paste_y), model)
    shadow = _draw_drop_shadow(canvas, blur_radius=18.0, opacity=0.30)

    composite = Image.new("RGB", canvas_size)
    composite.paste(bg, (0, 0))
    composite.paste(shadow, (0, 0), shadow)
    composite.paste(model, (paste_x, paste_y), model)

    # 4. Hairline horizon line a third up from the bottom — suggests the
    # back edge of a desk without committing to a literal photograph.
    draw = ImageDraw.Draw(composite)
    horizon_y = int(height * 0.62)
    draw.line(
        [(0, horizon_y), (width, horizon_y)],
        fill=(168, 162, 150), width=1,
    )

    composite.save(png_path)
    return png_path


def render_packaging(
    stl_path: Path | str,
    png_path: Path | str,
    *,
    width: int = 1200,
    height: int = 900,
    target_faces: int = 20_000,
) -> Path:
    """Hero scaled into a kraft-mailer mock. Suggests packaging without
    needing photographs of real boxes — fills Etsy's packaging slot."""
    from PIL import Image, ImageDraw, ImageFont

    stl_path = Path(stl_path)
    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_dir = png_path.parent
    transparent_path = tmp_dir / f".{png_path.stem}_pkg_tmp.png"
    try:
        _render_view(
            stl_path, transparent_path,
            width=width, height=height,
            elev=35.0, azim=-50.0,
            target_faces=target_faces,
            transparent=True,
        )
        model = Image.open(transparent_path).convert("RGBA")
    finally:
        try:
            transparent_path.unlink(missing_ok=True)
        except OSError:
            pass

    bg = Image.new("RGB", (width, height), "#f2eee5")

    # Kraft mailer rectangle, centred, ~75% of the canvas.
    mailer_w = int(width * 0.74)
    mailer_h = int(height * 0.78)
    mx = (width - mailer_w) // 2
    my = (height - mailer_h) // 2
    draw = ImageDraw.Draw(bg)
    # Mailer body
    draw.rounded_rectangle(
        (mx, my, mx + mailer_w, my + mailer_h),
        radius=16,
        fill="#bca678",
        outline="#8c7a52",
        width=2,
    )
    # Faint diagonal kraft texture: stippled lines
    for i in range(0, mailer_w + mailer_h, 14):
        draw.line(
            [(mx + i, my), (mx, my + i)],
            fill=(178, 158, 116, 90), width=1,
        )

    # Address label patch in the top-left of the mailer
    label_x0 = mx + 36
    label_y0 = my + 30
    label_w = int(mailer_w * 0.42)
    label_h = 78
    draw.rectangle(
        (label_x0, label_y0, label_x0 + label_w, label_y0 + label_h),
        fill="white", outline="#bbae87", width=1,
    )

    # Try to render brand text. Fall back to default font if no system
    # font path resolves — Pillow always provides a bitmap default.
    def _load_font(size: int):
        for candidate in (
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/HelveticaNeue.ttc",
            "/Library/Fonts/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ):
            try:
                return ImageFont.truetype(candidate, size=size)
            except OSError:
                continue
        return ImageFont.load_default()

    title_font = _load_font(22)
    small_font = _load_font(13)
    draw.text((label_x0 + 12, label_y0 + 10), "cityform", fill="#1a1f26", font=title_font)
    draw.text((label_x0 + 12, label_y0 + 40), "Royal Mail Tracked 48", fill="#5a5f64", font=small_font)
    draw.text((label_x0 + 12, label_y0 + 56), "Made in Sheffield", fill="#5a5f64", font=small_font)

    # Resize and paste model so it sits centred-bottom-ish on the mailer
    target_w = int(mailer_w * 0.55)
    aspect = model.height / max(model.width, 1)
    target_h = int(target_w * aspect)
    if target_h > int(mailer_h * 0.78):
        target_h = int(mailer_h * 0.78)
        target_w = int(target_h / max(aspect, 1e-6))
    model = model.resize((target_w, target_h), Image.LANCZOS)

    paste_x = mx + mailer_w - target_w - 60
    paste_y = my + mailer_h - target_h - 50

    rgba_canvas = Image.new("RGBA", (width, height))
    rgba_canvas.paste(bg, (0, 0))
    # Build a small canvas just for the model + its shadow so the shadow
    # picks up the model's actual silhouette.
    model_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    model_layer.paste(model, (paste_x, paste_y), model)
    shadow = _draw_drop_shadow(model_layer, blur_radius=12.0, opacity=0.25)
    rgba_canvas.paste(shadow, (0, 0), shadow)
    rgba_canvas.paste(model, (paste_x, paste_y), model)

    rgba_canvas.convert("RGB").save(png_path)
    return png_path


# ── Turntable GIF ──────────────────────────────────────────────────────

def render_turntable(
    stl_path: Path | str,
    gif_path: Path | str,
    *,
    width: int = 800,
    height: int = 600,
    frames: int = 16,
    target_faces: int = 12_000,
    elev: float = 28.0,
    duration_ms: int = 110,
) -> Path:
    """Animated GIF rotating the model 360°. Decimates once and reuses the
    mesh across frames so total cost is `frames × _render_view` instead of
    `frames × (load + decimate + render)`. Default 16×800×600 ≈ 8–15 s on
    an M-series Mac. Etsy listings show GIFs as static thumbnails plus an
    animated viewer; ffmpeg can convert to MP4 if/when needed (see README).
    """
    from PIL import Image

    stl_path = Path(stl_path)
    gif_path = Path(gif_path)
    gif_path.parent.mkdir(parents=True, exist_ok=True)

    mesh = _load_and_decimate(stl_path, target_faces)

    tmp_dir = gif_path.parent
    frame_paths: list[Path] = []
    images: list[Image.Image] = []
    try:
        for i in range(frames):
            azim = -50.0 + (360.0 * i / frames)
            frame_path = tmp_dir / f".{gif_path.stem}_frame{i:02d}.png"
            _render_view(
                mesh, frame_path,
                width=width, height=height,
                elev=elev, azim=azim,
                target_faces=target_faces,
                transparent=False, bg_color="white",
            )
            frame_paths.append(frame_path)
            img = Image.open(frame_path).convert("P", palette=Image.ADAPTIVE, colors=128)
            images.append(img)

        if not images:
            raise RuntimeError("turntable produced zero frames")

        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=duration_ms,
            loop=0,
            optimize=True,
            disposal=2,
        )
    finally:
        for p in frame_paths:
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass

    return gif_path


# ── Pinterest pin ──────────────────────────────────────────────────────

def render_pinterest_pin(
    stl_path: Path | str,
    png_path: Path | str,
    *,
    name: str = "",
    width: int = 1000,
    height: int = 1500,
    target_faces: int = 20_000,
) -> Path:
    """1000×1500 portrait pin for Pinterest / social. Hero render at the
    top, brand wordmark + city name in the lower third on a soft band.
    `name` is the same display string used by the sticker (typically
    "Sheffield Park Hill" — underscores replaced with spaces)."""
    from PIL import Image, ImageDraw, ImageFont

    stl_path = Path(stl_path)
    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_dir = png_path.parent
    transparent_path = tmp_dir / f".{png_path.stem}_pin_tmp.png"
    try:
        _render_view(
            stl_path, transparent_path,
            width=int(width * 0.95), height=int(width * 0.95 * 0.8),
            elev=32.0, azim=-50.0,
            target_faces=target_faces,
            transparent=True,
        )
        model = Image.open(transparent_path).convert("RGBA")
    finally:
        try:
            transparent_path.unlink(missing_ok=True)
        except OSError:
            pass

    canvas = Image.new("RGB", (width, height), "#f4f1ea")
    draw = ImageDraw.Draw(canvas)

    # Soft footer band
    band_top = int(height * 0.70)
    draw.rectangle((0, band_top, width, height), fill="#1a1f26")

    # Place model in upper section — fit width with margin
    target_w = int(width * 0.85)
    aspect = model.height / max(model.width, 1)
    target_h = int(target_w * aspect)
    if target_h > int(height * 0.62):
        target_h = int(height * 0.62)
        target_w = int(target_h / max(aspect, 1e-6))
    model = model.resize((target_w, target_h), Image.LANCZOS)
    paste_x = (width - target_w) // 2
    paste_y = int(height * 0.06)
    canvas.paste(model, (paste_x, paste_y), model)

    # Footer typography
    def _load_font(size: int):
        for candidate in (
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/HelveticaNeue.ttc",
            "/Library/Fonts/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ):
            try:
                return ImageFont.truetype(candidate, size=size)
            except OSError:
                continue
        return ImageFont.load_default()

    display_name = (name or "").replace("_", " ").strip() or "British city model"
    name_font = _load_font(72)
    small_font = _load_font(28)
    draw.text((60, band_top + 60), display_name, fill="white", font=name_font)
    draw.text((60, band_top + 160), "cityform · made in Sheffield", fill="#cfd2d6", font=small_font)

    canvas.save(png_path)
    return png_path


_WIRE_W = 4000
_WIRE_CHUNK = 500_000
_WIRE_MEDIAN_SIZE = 5
_WIRE_GAUSS_SIGMA = 1.0
_WIRE_GRAD_FRAC = 0.003
_WIRE_MIN_OBJ_PX = 15
_WIRE_LINE_DILATE = 1
_WIRE_EDGE_BLUR = 0.6

_STL_DTYPE = np.dtype([
    ('normal', '<f4', 3),
    ('v0',     '<f4', 3),
    ('v1',     '<f4', 3),
    ('v2',     '<f4', 3),
    ('attr',   '<u2'),
])


def _stream_stl_bounds(stl_path: Path) -> tuple[int, np.ndarray, np.ndarray]:
    with open(stl_path, "rb") as f:
        f.read(80)
        n = struct.unpack("<I", f.read(4))[0]
        bmin = np.full(3, np.inf, dtype=np.float64)
        bmax = np.full(3, -np.inf, dtype=np.float64)
        read = 0
        while read < n:
            c = min(_WIRE_CHUNK, n - read)
            buf = np.frombuffer(f.read(c * _STL_DTYPE.itemsize), dtype=_STL_DTYPE)
            v = np.concatenate([buf["v0"], buf["v1"], buf["v2"]], axis=0)
            bmin = np.minimum(bmin, v.min(axis=0))
            bmax = np.maximum(bmax, v.max(axis=0))
            read += c
    return n, bmin, bmax


def _build_heightmap(stl_path: Path, n: int, bmin: np.ndarray, bmax: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (heightmap, land_mask). The mask is True where the STL had
    at least one vertex (= solid land), False where there's a hole (= water
    cut, etc). The heightmap itself is NN-filled across holes so the
    median-filter + Sobel pass doesn't see artificial cliffs around them —
    we mask edges back out in `_extract_edges`."""
    from scipy.ndimage import distance_transform_edt

    xmin = bmin[0]
    xmax = bmax[0]
    ymin, ymax = bmin[1], bmax[1]
    H = int(round(_WIRE_W * (ymax - ymin) / (xmax - xmin)))
    H = max(100, min(H, 8000))
    heightmap = np.full((H, _WIRE_W), -1e9, dtype=np.float32)

    with open(stl_path, "rb") as f:
        f.read(80)
        struct.unpack("<I", f.read(4))
        read = 0
        while read < n:
            c = min(_WIRE_CHUNK, n - read)
            buf = np.frombuffer(f.read(c * _STL_DTYPE.itemsize), dtype=_STL_DTYPE)
            v = np.concatenate([
                buf["v0"], buf["v1"], buf["v2"],
                (buf["v0"] + buf["v1"]) * 0.5,
                (buf["v1"] + buf["v2"]) * 0.5,
                (buf["v2"] + buf["v0"]) * 0.5,
                (buf["v0"] + buf["v1"] + buf["v2"]) / 3.0,
            ], axis=0)
            px = (v[:, 0] - xmin) / (xmax - xmin) * (_WIRE_W - 1)
            py = (ymax - v[:, 1]) / (ymax - ymin) * (H - 1)
            ix = np.clip(np.round(px).astype(np.int32), 0, _WIRE_W - 1)
            iy = np.clip(np.round(py).astype(np.int32), 0, H - 1)
            np.maximum.at(heightmap, (iy, ix), v[:, 2].astype(np.float32))
            read += c

    mask = heightmap > -1e8
    ind = distance_transform_edt(~mask, return_distances=False, return_indices=True)
    return heightmap[tuple(ind)], mask


def _extract_edges(heightmap: np.ndarray, land_mask: np.ndarray | None = None) -> np.ndarray:
    """Sobel edges on the smoothed heightmap. When `land_mask` is given,
    edges falling in water (mask=False) are zeroed out — otherwise the
    NN-filled water pixels produce ghost lines from neighbouring land
    fills meeting inside the cut."""
    from scipy.ndimage import median_filter
    from skimage import filters, morphology

    h_med = median_filter(heightmap, size=_WIRE_MEDIAN_SIZE)
    h_smooth = filters.gaussian(h_med, sigma=_WIRE_GAUSS_SIGMA, preserve_range=True)
    gx, gy = filters.sobel_h(h_smooth), filters.sobel_v(h_smooth)
    grad = np.sqrt(gx * gx + gy * gy)
    zrange = heightmap.max() - heightmap.min()
    edges = grad > zrange * _WIRE_GRAD_FRAC
    edges = morphology.skeletonize(edges)
    edges = morphology.remove_small_objects(edges, min_size=_WIRE_MIN_OBJ_PX, connectivity=2)
    if land_mask is not None:
        edges = edges & land_mask
    return edges


def render_wireframe(stl_path: Path | str, png_path: Path | str) -> Path:
    """Top-down balanced wireframe, transparent background.

    Stream-parses the binary STL → builds a top-down heightmap → median-filters
    out triangulation noise → Sobel edges → skeletonize → small-object prune
    → AA dilate. Output is black lines on alpha at ~4000 px wide.
    Binary STL only (the Tier-3 pipeline always emits binary)."""
    from PIL import Image, ImageFilter
    from skimage import morphology

    stl_path = Path(stl_path)
    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)

    n, bmin, bmax = _stream_stl_bounds(stl_path)
    heightmap, land_mask = _build_heightmap(stl_path, n, bmin, bmax)
    edges = _extract_edges(heightmap, land_mask=land_mask)

    em = morphology.binary_dilation(edges, morphology.disk(_WIRE_LINE_DILATE))
    alpha = (em.astype(np.float32) * 255).astype(np.uint8)
    a_img = Image.fromarray(alpha).filter(ImageFilter.GaussianBlur(radius=_WIRE_EDGE_BLUR))
    alpha = np.array(a_img)

    H, W_ = edges.shape
    out = np.zeros((H, W_, 4), dtype=np.uint8)
    out[..., 3] = alpha

    Image.fromarray(out).save(png_path)
    return png_path


def render_both(stl_path: Path | str, hero_path: Path | str, wire_path: Path | str) -> tuple[Path, Path]:
    """Convenience for the library-save flow — renders both, never raises:
    individual failures return None for that path."""
    hero = wire = None
    try:
        hero = render_hero(stl_path, hero_path)
    except Exception as exc:
        print(f"[render] hero failed for {stl_path}: {exc}", file=sys.stderr)
    try:
        wire = render_wireframe(stl_path, wire_path)
    except Exception as exc:
        print(f"[render] wireframe failed for {stl_path}: {exc}", file=sys.stderr)
    return hero, wire


# Names of the per-entry pose files. Centralised here so
# `_save_to_library` and `/api/library/backfill_photos` agree.
PHOTO_FILE_SUFFIXES = {
    "three_quarter_path": "_three_quarter.png",
    "lifestyle_path":     "_lifestyle.png",
    "packaging_path":     "_packaging.png",
    "turntable_path":     "_turntable.gif",
    "pinterest_path":     "_pinterest.png",
    "on_base_path":       "_on_base.png",
}


def render_extra_poses(
    stl_path: Path | str,
    dest_dir: Path | str,
    safe_dir: str,
    *,
    name: str = "",
    skip_existing: bool = True,
) -> dict[str, Path]:
    """Render the four still poses + the turntable GIF into `dest_dir`,
    naming files `<safe_dir>_<suffix>`. Returns the manifest field → path
    dict for paths that landed on disk. Failures only log; never raise.

    `skip_existing=True` lets the backfill flow skip work that's already
    been done; the per-generation save flow can pass False to force a full
    re-render."""
    stl_path = Path(stl_path)
    dest_dir = Path(dest_dir)
    written: dict[str, Path] = {}

    for field, suffix in PHOTO_FILE_SUFFIXES.items():
        out_path = dest_dir / f"{safe_dir}{suffix}"
        if skip_existing and out_path.exists():
            written[field] = out_path
            continue
        try:
            if field == "three_quarter_path":
                render_three_quarter(stl_path, out_path)
            elif field == "lifestyle_path":
                render_lifestyle(stl_path, out_path)
            elif field == "packaging_path":
                render_packaging(stl_path, out_path)
            elif field == "turntable_path":
                render_turntable(stl_path, out_path)
            elif field == "pinterest_path":
                render_pinterest_pin(stl_path, out_path, name=name or safe_dir)
            written[field] = out_path
        except Exception as exc:    # noqa: BLE001
            print(f"[render] {field} failed for {stl_path}: {exc}", file=sys.stderr)
    return written


# --- SVG sticker --------------------------------------------------------
# Built to match the London artboard in cityform_sticker_sheffield1.svg:
# 343.7 × 343.7 viewBox, rounded border, accent dot top-right, title +
# coordinates top-left, square wireframe in the middle, "cityform" + size
# in the footer. The image href is written as a relative filename so the
# SVG sits next to its wireframe PNG inside the entry's library folder.

# Same per-city palette the original sheet uses, rotated by index when
# the caller doesn't pin a colour.
_STICKER_PALETTE = (
    "#d34833", "#2f3bb9", "#e2a439", "#296d56", "#4f3253",
    "#588388", "#3d505f", "#602736", "#c0893d", "#1b2544", "#ad5c70",
)


def _format_coords(lat: float, lng: float) -> str:
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lng >= 0 else "W"
    return f"{abs(lat):.4f}°{ns} · {abs(lng):.4f}°{ew}"


def render_sticker(
    name: str,
    centre_lat: float,
    centre_lng: float,
    wireframe_filename: str,
    out_path: Path | str,
    *,
    color: str | None = None,
    print_label: str = "9 CM x 9 CM",
) -> Path:
    """Write a single-artboard SVG sticker in the London style.

    `wireframe_filename` is referenced relative to the SVG, so the PNG and
    SVG should live in the same folder. `color` accepts any CSS colour;
    if omitted, one is picked from the palette by hashing the name."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if color is None:
        color = _STICKER_PALETTE[hash(name) % len(_STICKER_PALETTE)]
    display_name = name.replace("_", " ")
    coords = _format_coords(centre_lat, centre_lng)

    # Element coordinates mirror the London artboard offsets relative to
    # its rect at x=-3270.8 (so e.g. title (-3244.2, 43.8) → (26.6, 43.8)).
    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
     version="1.1" viewBox="0 0 343.7 343.7">
  <defs>
    <style>
      .title {{ font-family: Helvetica-Bold, Helvetica; font-weight: 700; font-size: 18.4px; fill: {color}; }}
      .coords {{ font-family: Helvetica, Helvetica; font-size: 7.4px; fill: #6e7479; letter-spacing: .2em; }}
      .label {{ font-family: Helvetica-Bold, Helvetica; font-weight: 700; font-size: 9.9px; fill: #1a1f26; }}
      .size {{ font-family: Helvetica, Helvetica; font-size: 7.4px; fill: #6e7479; }}
      .border {{ fill: none; stroke: #000; stroke-linecap: round; stroke-linejoin: round; stroke-width: .7px; }}
      .divider {{ fill: none; stroke: #1a1f26; stroke-miterlimit: 11.3; stroke-opacity: .4; stroke-width: .4px; }}
    </style>
  </defs>
  <rect class="border" x=".3" y=".3" width="343" height="343" rx="9.9" ry="9.9"/>
  <text class="title" transform="translate(26.6 43.8)">{display_name}</text>
  <text class="coords" transform="translate(26.6 55.3)">{coords}</text>
  <ellipse fill="{color}" cx="302.6" cy="34.6" rx="4.6" ry="4.3"/>
  <image x="51.5" y="60" width="240" height="240" preserveAspectRatio="xMidYMid meet"
         xlink:href="{wireframe_filename}"/>
  <line class="divider" x1="28.1" y1="307.9" x2="307.6" y2="307.9"/>
  <text class="label" transform="translate(28.1 319.3)">cityform</text>
  <text class="size" transform="translate(251.3 319.3)">{print_label}</text>
</svg>
"""
    out_path.write_text(svg, encoding="utf-8")
    return out_path


# --- Laser-cut location label (design system v1.1) ---------------------------
# Standalone 57.6 × 7.6 mm SVG per the label spec v1.1: visually centred
# baselines, two-line area stack with locality dominance, single-tone
# Pearl on Carbon. Two Inkscape layers: "engrave" and "cut".


def _format_coords_3dp(lat: float, lng: float) -> str:
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lng >= 0 else "W"
    return f"{abs(lat):.3f}° {ns} · {abs(lng):.3f}° {ew}"


def _split_city_area(name: str) -> tuple[str, str]:
    """Split a library name into (CITY, LOCALITY) for laser labels.

    Returns uppercase strings. Locality is empty for city-only plates.
    """
    prefix = name.split("_", 1)[0]
    prefix = re.sub(r"([a-z])(of|upon)([A-Z])", r"\1 \2 \3", prefix)
    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", prefix)
    words = spaced.split()
    if not words:
        return name.upper(), ""
    # Consume the first capitalized word plus any following join words
    # and the next capitalized word as the city name.
    # "City of London" -> city="CITY OF LONDON", area=""
    # "Newcastle upon Tyne" -> city="NEWCASTLE UPON TYNE", area=""
    _joins = {"of", "upon", "on", "the", "in", "le", "de"}
    city_parts = [words[0]]
    i = 1
    while i < len(words) and words[i].lower() in _joins and i + 1 < len(words):
        city_parts.append(words[i])
        city_parts.append(words[i + 1])
        i += 2
    city = " ".join(w.upper() for w in city_parts)
    area = " ".join(w.upper() for w in words[i:])
    return city, area


_LABEL_STYLES = """\
      .plate { fill: #2C313A; }
      .engraved-stroke { stroke: #C8CCD0; stroke-width: 0.25; }
      .name-1 {
        font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
        font-weight: 600; font-size: 2.4px; letter-spacing: 0.28px;
        fill: #C8CCD0;
      }
      .parent {
        font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
        font-weight: 600; font-size: 1.55px; letter-spacing: 0.45px;
        fill: #C8CCD0;
      }
      .locality {
        font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
        font-weight: 600; font-size: 2.0px; letter-spacing: 0.18px;
        fill: #C8CCD0;
      }
      .coord {
        font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
        font-weight: 400; font-size: 1.8px; letter-spacing: 0.16px;
        fill: #C8CCD0;
      }
      .cut-line { fill: none; stroke: #FF0000; stroke-width: 0.01; }"""


def render_laser_label(
    name: str,
    centre_lat: float,
    centre_lng: float,
    out_path: Path | str,
) -> Path:
    """Write a standalone laser-cut label SVG (57.6 x 7.6 mm, design system v1.1).

    City-only names produce a single-line city plate; names with a locality
    produce a two-line area plate (parent small on top, locality large below).
    Output has two Inkscape layers: *engrave* and *cut* (red hairline).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    city, area = _split_city_area(name)
    coords = _format_coords_3dp(centre_lat, centre_lng)
    title = f"{city} · {area}" if area else city

    if area:
        name_elements = (
            f'    <text class="parent" x="3" y="2.91">{city}</text>\n'
            f'    <text class="locality" x="3" y="5.81">{area}</text>'
        )
    else:
        name_elements = f'    <text class="name-1" x="3" y="4.70">{city}</text>'

    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
     width="57.6mm" height="7.6mm"
     viewBox="0 0 57.6 7.6"
     version="1.1">
  <title>Cityform laser label — {title}</title>
  <desc>57.6 x 7.6 mm. Design system v1.1. Convert text to paths in Inkscape before importing to xTool XCS.</desc>
  <defs>
    <style type="text/css">
{_LABEL_STYLES}
    </style>
  </defs>
  <g id="engrave" inkscape:label="engrave" inkscape:groupmode="layer">
    <rect class="plate" x="0" y="0" width="57.6" height="7.6" rx="3.8" ry="3.8"/>
    <line class="engraved-stroke" x1="33.6" y1="1.9" x2="33.6" y2="5.7"/>
{name_elements}
    <text class="coord" x="54.6" y="4.46" text-anchor="end">{coords}</text>
  </g>
  <g id="cut" inkscape:label="cut" inkscape:groupmode="layer">
    <rect class="cut-line" x="0" y="0" width="57.6" height="7.6" rx="3.8" ry="3.8"/>
  </g>
</svg>
"""
    out_path.write_text(svg, encoding="utf-8")
    return out_path


def _cli():
    """CLI smoke test. Two modes:

    Hero + wireframe (existing 3-arg form):
        render.py <stl> <hero.png> <wireframe.png>

    Full pose pack into a dest folder (used to QA Phase A):
        render.py --all <stl> <dest_dir> [<safe_dir_name>]
    """
    if len(sys.argv) >= 3 and sys.argv[1] == "--all":
        stl = sys.argv[2]
        dest_dir = Path(sys.argv[3]) if len(sys.argv) >= 4 else Path("./pose_pack")
        safe_dir = sys.argv[4] if len(sys.argv) >= 5 else Path(stl).stem
        dest_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.monotonic()
        render_hero(stl, dest_dir / f"{safe_dir}_preview.png")
        print(f"hero            ({time.monotonic() - t0:.1f}s)")
        t0 = time.monotonic()
        render_wireframe(stl, dest_dir / f"{safe_dir}_wireframe.png")
        print(f"wireframe       ({time.monotonic() - t0:.1f}s)")
        for field, suffix in PHOTO_FILE_SUFFIXES.items():
            t = time.monotonic()
            out = dest_dir / f"{safe_dir}{suffix}"
            try:
                if field == "three_quarter_path":
                    render_three_quarter(stl, out)
                elif field == "lifestyle_path":
                    render_lifestyle(stl, out)
                elif field == "packaging_path":
                    render_packaging(stl, out)
                elif field == "turntable_path":
                    render_turntable(stl, out)
                elif field == "pinterest_path":
                    render_pinterest_pin(stl, out, name=safe_dir)
                print(f"{field:<20} ({time.monotonic() - t:.1f}s) → {out.name}")
            except Exception as exc:    # noqa: BLE001
                print(f"{field:<20} FAILED: {exc}")
        return

    if len(sys.argv) != 4:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    stl, hero, wire = sys.argv[1], sys.argv[2], sys.argv[3]
    t0 = time.monotonic()
    h = render_hero(stl, hero)
    print(f"hero      {h} ({time.monotonic() - t0:.1f}s)")
    t1 = time.monotonic()
    w = render_wireframe(stl, wire)
    print(f"wireframe {w} ({time.monotonic() - t1:.1f}s)")


if __name__ == "__main__":
    _cli()
