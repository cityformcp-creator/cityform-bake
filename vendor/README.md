# vendor/

Snapshot of the minimal cityform-tool subset that `src/bake_tile.py`
depends on. Vendored so GitHub Actions can run the bake without needing
the private cityform-offline repo cloned.

## Files

| File | Purpose | Source |
|---|---|---|
| `pipeline/__init__.py` | package marker | `cityform-tool/pipeline/__init__.py` |
| `pipeline/wcs.py` | Environment Agency WCS fetcher (LIDAR DTM/DSM) | `cityform-tool/pipeline/wcs.py` |
| `pipeline/preview_mesh.py` | STL → compact GLB | `cityform-tool/pipeline/preview_mesh.py` |
| `tier3_measured_roofs.py` | DSM+DTM → printable STL | `cityform-tool/tier3_measured_roofs.py` |
| `render.py` | STL → preview PNG | `cityform-tool/render.py` (optional — bake skips PNG if import fails) |

These files have NO cross-imports between themselves — each is
self-contained relative to its PyPI deps (numpy / trimesh / rasterio /
scipy / pyfqmr / pymeshfix). That's why vendoring is feasible without
pulling the entire cityform-tool tree.

## Snapshot date

**2026-05-25** — initial vendoring.

To re-sync after updating any of these files in cityform-tool:

```bash
cd ~/Downloads/cityform-bake
for f in pipeline/wcs.py pipeline/preview_mesh.py tier3_measured_roofs.py render.py; do
  cp ~/Downloads/cityform-offline/cityform-tool/$f vendor/$f
done
git add vendor/ && git commit -m "vendor: re-sync from cityform-tool $(date +%Y-%m-%d)"
git push
```

## Override

The bake driver also accepts `CITYFORM_TOOL=/path/to/cityform-tool`
env var, which takes priority over `vendor/`. Use this for local
testing against an unvendored cityform-tool to confirm a snapshot is
fresh.

```bash
CITYFORM_TOOL=~/Downloads/cityform-offline/cityform-tool \
  python3 src/bake_tile.py --tile-id SK3587 --lat 53.380 --lng -1.464 --out test/
```
