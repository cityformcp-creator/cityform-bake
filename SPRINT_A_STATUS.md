# Sprint A status

**Done locally — 2026-05-25.** Local code is ready + first commit made on
a local git repo. Waiting on a GitHub PAT to push to a public repo.

## What works

- ✅ `src/select_tiles.py` — Geonames → 37,919 GB tiles → `data/tiles.geojson`
- ✅ `src/bake_tile.py` — wraps cityform-tool pipeline, produces STL+GLB+PNG+meta per tile
- ✅ Smoke test: Sheffield (SK3587) baked in 37.3s, all outputs valid (see `bake-sample/SK3587/`)
- ✅ `git init` done, first commit `Sprint A: tile selector + bake driver` made

## Architecture — GitHub Releases as asset host (no R2)

Per user preference for zero card-on-file:

| What | Where | Why |
|---|---|---|
| Bake driver code | `cityform-bake` (this repo, public) | Free unlimited storage on public GH repos |
| Bake compute | GitHub Actions matrix | Free unlimited minutes on public repos |
| **Tile assets (STL / GLB / PNG / meta)** | **GitHub Releases on this repo** | **Truly free, no card, ~$0 forever** |
| Manifest | `releases/latest/download/manifest.json` | GitHub's stable "latest" URL — no version pinning needed |
| Map basemap | OpenStreetMap (free) → OS DataHub later (also free tier) | Free at any traffic scale |
| Storefront | Existing Shopify | You already pay |

Total monthly cost: **£0 / $0 forever, no card on file anywhere.**

## What you need to do (~1 min total now)

### 1. GitHub Personal Access Token

Go to: **https://github.com/settings/tokens/new**

Fill in:
- **Note**: `cityform-bake CI`
- **Expiration**: 90 days (or "No expiration")
- **Scopes**: tick `repo` (full) AND `workflow`
- Click **Generate token**

Paste the `ghp_…` token into `~/.cityform-credentials.env` as:
```
GH_TOKEN=ghp_…
```

Save TextEdit (⌘S). Tell Claude "GH_TOKEN set" — Claude proceeds automatically through tasks 5–10.

## What Claude does automatically once GH_TOKEN is set

1. **#5** `gh auth login --with-token` then `gh repo create cityform-bake --public --source . --push` — repo lands on GitHub at `https://github.com/<your-username>/cityform-bake`
2. **#8** Writes `.github/workflows/bake-all.yml`:
   - Matrix of 20 parallel runners
   - Each runner bakes a batch of ~50 tiles (~25 min wall clock)
   - Each batch uploads outputs to a release named `bake-<run_id>` (or appends to `latest`)
   - Aggregator job builds `manifest.json` mapping tile_id → asset URLs
3. **#9** `gh workflow run bake-all.yml` — ~30 min wall clock for top-1000 tiles
4. **#10** Updates `cf-picker.jsx` to fetch manifest from GH Releases, pushes to LIVE → picker expands 18 → 1000+ tiles

You're hands-off after step 1.

## Repo layout

```
cityform-bake/
├── README.md
├── SPRINT_A_STATUS.md           (this file)
├── .gitignore
├── .github/workflows/
│   └── bake-all.yml             (added by task #8)
├── src/
│   ├── select_tiles.py          ✅ done
│   ├── bake_tile.py             ✅ done
│   └── upload_to_release.py     (added by task #8 — wraps `gh release upload`)
├── data/
│   ├── tiles.geojson            ✅ 37,919 tiles
│   └── _geonames_cache.json     (gitignored)
└── bake-sample/                 (gitignored — local smoke test only)
    └── SK3587/
        ├── city.stl
        ├── city.glb
        ├── preview.png
        ├── wireframe.png
        └── meta.json
```

## Asset URL convention (after bake)

```
https://github.com/<your-username>/cityform-bake/releases/latest/download/<tile_id>__<file>

e.g.
  …/releases/latest/download/SK3587__city.glb
  …/releases/latest/download/SK3587__preview.png
  …/releases/latest/download/SK3587__meta.json
  …/releases/latest/download/manifest.json
```

The picker fetches `manifest.json` once on mount, then lazy-loads previews on
hover and GLBs on click — same UX as the R2 plan, just hosted on GitHub
instead.

## Trade-offs vs the original R2 plan

| Concern | R2 (original) | GH Releases (now) |
|---|---|---|
| Cost | $0 with card on file | $0 no card |
| CDN performance | Cloudflare edge, ~30 ms | GitHub CDN, ~100-200 ms |
| Asset count | Unlimited | Unlimited per release, but UI gets sluggish past ~5k |
| Asset size cap | 5 TB per object | 2 GB per asset (our biggest is 50 MB STL) |
| Re-bake workflow | Overwrite same R2 keys | Create new release tag, picker auto-uses `latest` |
| Browser fetch | `cityform-tiles.r2.dev/…` (CORS configurable) | `github.com/.../releases/download/…` (CORS allow-all by default) |
| Tooling | wrangler / S3 SDK | `gh release upload` (already installed) |

Net: marginal performance hit, zero ops cost, zero monetary cost. Acceptable trade.
