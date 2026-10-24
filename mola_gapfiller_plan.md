# MOLA Gap-Filler — Project Plan

## v1: "We're not dumb" (verify the pipeline works)

**Goal:** prove a super-resolution pipeline on Mars elevation data beats bicubic on held-out terrain, with visible improvement in a hillshade.

### Data (small, free, public)
- MOLA global GeoTIFF (64 ppd, ~463 m) from USGS: `https://planetarymaps.usgs.gov/mosaic/Mars_MGS_MOLA_DEM_mosaic_global_463m.tif`
- 2–3 CTX stereo DEM tiles from USGS Astrogeology (18–50 m), e.g. Lethe Vallis 18 m.
- Work on ONE region of interest (one CTX tile footprint), not the whole planet.

### Pipeline
1. Load MOLA + CTX tile, co-register to same grid (CTX already aligned to MOLA).
2. Downsample CTX → MOLA resolution → (low-res, high-res) training pairs.
3. Baseline: bicubic + Lanczos upsample → compare to CTX (RMSE, MAE, PSNR, SSIM).
4. Tiny SRCNN (3 conv layers, ~8k params) trained on CPU in minutes.
5. Compare CNN vs baseline on held-out tile.
6. Hillshade before/after/error → `mars_elev_plot.svg`.
7. Metrics → `mars_elev.json`. Repro script → `analyze_mars.py`.

### Success criterion
CNN beats bicubic on held-out terrain, visibly better in hillshade.

### v1 artifacts
- `mars_elev_plot.svg`
- `mars_elev.json`
- `analyze_mars.py`
- (optional) `mars_elev_model.pt`

## v2: "Get creative" (after v1 passes)

### Creative levers (do these 3)
1. **Terrain losses** — add slope + curvature + SSIM loss (not just pixel MSE). Biggest quality bump, easy.
2. **Shape-from-shading fusion** — combine MOLA height with HiRISE images (brightness → slope) to recover fine detail. Free data, huge wow factor, Fable-Venus parallel.
3. **Uncertainty map** — ensemble of 3–5 models → per-pixel std. Science-credibility piece. Overlay on hillshade.

### Defer to v3+
- Transfer learning from Moon (LunarDEM-SR) — needs more data wrangling.
- Terrain-type conditioning (separate models per crater/volcano/plain).
- Drainage-network / crater-preservation evaluation.
- Interactive HTML viewer.

### Honest ceiling
MOLA is 463 m. Even perfect SR can only *infer* fine features statistically, not recover ground truth. Output is a plausible prior for mission planning, not a measurement. Set expectations accordingly.

## Key gotchas
- Mars CRS isn't a standard EPSG code — use the CRS embedded in the GeoTIFF WKT.
- HiRISE/CTX cover <1% of Mars → labels are tiny; validate on held-out tiles only.
- AI can hallucinate craters → uncertainty map + held-out validation are the credibility.
- GDAL: install via system `gdal-bin`, not PyPI wheel.

## Data sources
- MOLA: PDS Geosciences Node (`MGS-M-MOLA-5-MEGDR-L3-V1.0`), 4/16/32/64/128 ppd; global max 128 ppd (463 m); 512 ppd poles only.
- CTX/HiRISE DEMs: USGS Astrogeology (GeoTIFF 32-bit float), ASU HiRISE (~1338 tiles, 1–2 m), PDS Imaging Node. Already aligned to MOLA.
- Global MOLA GeoTIFF: `https://planetarymaps.usgs.gov/mosaic/Mars_MGS_MOLA_DEM_mosaic_global_463m.tif`
- CTX Lethe Vallis 18 m: `https://planetarymaps.usgs.gov/mosaic/dundas_flood_lavas_topo/DEM_18m_Lethe_Vallis.tif`
- CTX south-polar 50 m: `https://planetarymaps.usgs.gov/mosaic/Mars/mars_mro_ctx_spoledtmmosaic_mayer_2020/ctx_abs_align_iau_deltaradius_mosaic.tif`
- HiRISE MSR TRN 1 m (zipped): `https://edc-pds.s3.us-west-2.amazonaws.com/mosaic/Mars/MSR_TRN_HiRISE_soc_003_USGS_release_aug2024/MSR_TRN_HiRISE_soc_003_USGS_release_aug2024_mosaics.zip`

## Research references
- D-SRGAN (Demiray et al. 2020): https://arxiv.org/abs/2004.04788
- DSRFB (Kubade et al. 2020): https://arxiv.org/abs/2007.01940
- CraterDiffusion (Mars): https://github.com/Dropslam/CraterDiffusion
- LunarDEM-SR: https://github.com/Bharat-Sharma-007/LunarDEM-SR
- SRCNN: https://arxiv.org/abs/1501.00092
- ESRGAN: https://arxiv.org/abs/1809.00219
- MOLA reference script: https://github.com/theonlyallana/mars-mola-dem
- Fable 5.1 Venus map (the parallel): https://claudebuilds.com/builds/venus-elevation-map/
