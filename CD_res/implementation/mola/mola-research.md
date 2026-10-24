# Implementation research: MOLA gap-filler

## The task

Super-resolve the 463 m Mars elevation model using higher-resolution CTX stereo DEMs as labels. The v1 goal from `mola_gapfiller_plan.md` is to beat bicubic on held-out terrain and show a better hillshade. This pass checks the data, the prior art, and the traps before any code.

Short version: someone already built almost exactly this, the code is public, two numbers in the plan are wrong, and the plan's v1 test cannot prove the claim. The fix is a tighter scope and a test that uses real MOLA input.

## The uncomfortable question first

Why super-resolve MOLA at all when HRSC already gives a measured 50 m global DTM?

Mars Express HRSC has a global DTM mosaic at 50 m per pixel, from nadir coverage that reaches about 96% of the surface. The UCL MSSL release ships 50 m DTM tiles and 12.5 m orthoimages, tied to the MOLA areoid. Handmer's own released product is a global 115 m DEM and 14 m quads. That is worse than HRSC.

So "better than bicubic" is the wrong bar. Bicubic is a strawman. The real bars are:

1. HRSC 50 m, the measured global product that already exists.
2. CTX stereo DEM 18 m, the sparse but real high-resolution product.

A project that beats bicubic but loses to HRSC has not earned its place. State the niche precisely: 18 m topography where only 463 m MOLA and 5 m optical imagery exist. CTX stereo DEMs cover well under 1% of Mars, so that niche is large. But it is a claim about beating HRSC and matching CTX labels, not about beating interpolation.

## What the data actually is

### MOLA global DEM

Verified against the USGS Astropedia record, a live HEAD request, and the TIFF header.

- One GeoTIFF, 2,125,771,142 bytes (2.13 GB), live at `https://planetarymaps.usgs.gov/mosaic/Mars_MGS_MOLA_DEM_mosaic_global_463m.tif`.
- 46,080 by 23,040 pixels, signed 16-bit, one band, GDAL NoData -32768.
- 128 pixels per degree, 463.08 m per pixel at the equator. The TIFF's own pixel scale reads 463.0935 m, so the file and the catalog disagree by about 1 cm.
- Simple Cylindrical, planetocentric latitude, positive east, longitude -180 to 180.
- Heights are above the Mars areoid (GMM-2B gravity, degree 50). About 100 m horizontal and 1 m radial accuracy per point.
- Total vertical uncertainty is at least ±3 m because the areoid itself is uncertain by ±1.8 m and varies by region.
- Coverage above 87° north and below 87° south is sparse. Track gaps of 1 to 2 km are common, some reach 12 km near the equator, and the published DEM filled those gaps by interpolation.
- License is CC0, public domain.
- Mars has no standard EPSG code. The projection lives in the GeoTIFF WKT, or you set a custom CRS. QGIS and GDAL will snap it to Earth coordinates if you let them.

The plan says "64 ppd, ~463 m". That is wrong. 463 m is 128 ppd. 64 ppd would be 926 m.

One file-structure warning: the MOLA GeoTIFF has no tiles and one strip per row, with its image directory at the end of the file, byte 2,125,586,176. A reader that sniffs only the first few KB sees nothing. rasterio handles it. A hand-rolled header parser does not.

### CTX stereo DEMs

The best label source is the UCLA / GALE Lab archive, not the single Lethe Vallis file.

- 1,353 CTX DEMs at 18 m post spacing, each paired with a 1 m HiRISE DEM, orthoimages, and a Good Pixel Map mask. The USGS catalog page says 1,354 while the PDS readme says 1,353. Footnote the conflict instead of picking silently.
- All tied to the MOLA areoid, projected with `GCS_Mars_2000_Sphere` and the IAU D_MARS datum.
- Total archive is about 1.2 TB. Individual products are much smaller.
- Access: `https://astrogeology.usgs.gov/search/map/mars_human_exploration_zone_dem_archive_2023`
- The Good Pixel Map matters. It marks where stereo correlation failed, so you drop bad labels instead of training on interpolation.

Two smaller options already in the plan, both confirmed live:

- Lethe Vallis 18 m DEM, 16,145,119 bytes, 1,691 by 2,384 pixels, float32, 18.21 m pixel scale. `https://planetarymaps.usgs.gov/mosaic/dundas_flood_lavas_topo/DEM_18m_Lethe_Vallis.tif`
- South polar CTX mosaic, 256,048,639 bytes, 8,000 by 8,000 pixels, exactly 50.0 m pixel scale. `https://planetarymaps.usgs.gov/mosaic/Mars/mars_mro_ctx_spoledtmmosaic_mayer_2020/ctx_abs_align_iau_deltaradius_mosaic.tif`

The Lethe tile marks voids as `-3.4028235e38`, not NaN and not zero. Mask it or you train on garbage.

There is also the Hawaii PRPDC CTX DEM collection. The host timed out during this check, but Wayback shows the page alive in December 2024, so treat it as a secondary source.

### CTX optical imagery

If you go the optical-guided route you need the Murray Lab mosaic.

- 5.0 m per pixel, 5.7 terapixel, covers 99.5% of Mars between 88°S and 88°N. The missing 0.5% shows as white strips.
- Free use for all purposes, credit NASA/JPL/MSSS/The Murray Lab.
- v01 is about 11.48 TB unzipped and uncompressed including overviews, 5.6 TB compressed. beta01 is about 3.9 TB in 3,960 four-degree zip tiles.
- The 4x4 degree tiling is convenient: you download one tile, not the planet.
- A seamless mosaic is not feasible, so seams and seam maps exist. Blending artifacts can look like landforms.

## What already exists

This is the part the plan is missing.

### Handmer 2021, ODENet

`Digital Elevation Model enhancement using Deep Learning`, arXiv:2101.04812. Code at `github.com/CHandmer/molaCTXEnhance`. This is an unreviewed draft, marked "Released for External Review" on every page, not a peer-reviewed paper. Treat its numbers as the author's own claims.

This is the same idea, already built and released.

- Two networks. InterpNet upsamples a 16x16 MOLA tile to 32x32 and trains on MOLA data alone. EnhanceNet is pix2pix and turns a 32x32 CTX optical image into the residual between InterpNet and the MOLA truth.
- Supervision is real MOLA height. The optical image is the input, not the label. This matters for how you describe the difference.
- Each step doubles resolution. Applied recursively, the draft claims 90x improvement, limited only by optical resolution.
- Validated by the author against an HRSC 100 m DEM on a 4x enhancement of the Gale Crater quad. He reports slope error consistent with the intrinsic systematic error of DEM data at that resolution, and found a lat/lon mismatch of 0.0139° east and 0.00284° north that had to be corrected first.
- Directional power spectrum score: MOLA 0.0037, CTX 0.027, ODENet 2x 0.018, ODENet 4x 0.013. About half the off-angle high-frequency signal recovered.
- Both networks train in under an hour on a MacBook Pro.
- Released a global 115 m DEM and 14 m quads on request.
- The repo is a single TensorFlow/Keras script, five commits, three stars. It expects the full 2.1 GB MOLA file in memory and the Murray Lab tiles unzipped next to it.

Two errors in that draft to not copy:

- It calls MOLA a "512 ppd" DEM in the same sentence as "462 m/pixel". USGS says 128 ppd. The author's own code indexes at 128 ppd. 512 ppd would be about 116 m per pixel. Only the prose is wrong.
- Its `getMola` samples with a +1 row and +2 column offset, and the author comments that this gives "dramatic improvement" aligning with CTX. That is a registration fudge, not a projection fact.

### DEM super-resolution papers

- D-SRGAN (Demiray, Sit, Demir, 2020, arXiv:2004.04788). GAN, 4x upscale, terrestrial LiDAR from North Carolina. Test MSE 0.861 m versus bicubic 0.946 m and bilinear 1.124 m. The authors say it performs better on flatter terrain and that MSE-trained models miss high-frequency detail.
- DSRFB (Kubade, Sharma, Rajan, IGARSS 2020, arXiv:2007.01940). Feedback network, no aerial imagery, RMSE 0.59 to 1.27 m across four datasets.
- LoGAN (ISPRS Journal of Photogrammetry and Remote Sensing, 2025). Monocular HiRISE image plus a paired low-resolution DTM. The abstract claims pixel-level 0.25 m output over three regions. The paper is closed access, so I could not check the numbers.
- MADNet 2.0, from UCL MSSL (Remote Sensing, 2021). Single-image DTM from HiRISE at pixel scale, with HiRISE input at 25 to 50 cm. MARSGAN is the name of the super-resolution GAN used inside this line of work, not a paper title. Do not cite it as one.
- Subpixel CaSSIS DTM paper (Remote Sensing, 2022). Combined super-resolution restoration and single-image DTM estimation, reported RMSE 1.876 m and SSIM 0.607 against HiRISE reference DTMs at 12.5 cm.
- Guided DEM super-resolution is an active line: Sinkhorn graph transformer work in 2024, multispectral-guided work in 2023, diffusion-guided work in 2026, and Dfilled in 2025 for guided void filling.
- A Neural Process approach for probabilistic lunar DEM gap filling (Park and Choi, 2020, arXiv:2004.09791) already ships per-pixel uncertainty for DEM voids. If uncertainty is your pitch, this is the paper to read first.

Nothing here is a blank slate. A reviewer will ask why this exists, and "I did not know about ODENet or HRSC" is not an answer.

## 1. Common gotchas

**Registration.** CTX DEMs are tied to the MOLA areoid but not pixel-aligned. Handmer needed a +1/+2 pixel offset for a visible improvement, and still found a sub-pixel lat/lon mismatch. Do a cross-correlation shift per tile and record it. "Tied to the same datum" does not mean "aligned".

**Downsampling mismatch.** If you build training pairs by averaging a CTX DEM down to 463 m, you teach the model to invert your downsampler. Real MOLA is sparse laser tracks interpolated onto a grid, not an area average. A model trained only on your downsampler can look great on your test set and fail on real MOLA input. Test on real MOLA patches or the result means nothing.

**Latitude distortion.** 463 m is the pixel size at the equator. In equirectangular projection, east-west ground spacing shrinks with the cosine of latitude. At 60° a pixel is about 231 m wide on the ground but still 463 m tall. Training in raw pixel space mixes ground scales. Either work in a local projection or restrict to one latitude band, and apply the cosine factor to any slope calculation.

**Memory.** This machine has 7.5 GB of RAM, 4 CPUs, and no GPU. Loading the 2.1 GB MOLA file plus float arrays will swap or die. Handmer recommends 32 GB for his naive path. Use windowed reads, or clip a small region. Do not `read_image()` the planet.

**Missing tooling.** `gdalinfo`, `gdal_translate`, numpy, rasterio, torch, scipy, and matplotlib are all absent on this box. Python is 3.14. Install rasterio instead of shelling out to GDAL, and check that the torch wheel supports cp314 before planning around it. rasterio wheels bundle their own GDAL.

**RMSE rewards blur.** A model that outputs the local mean gets a low RMSE and a useless hillshade. Report slope error, hillshade difference, and SSIM next to RMSE.

**Hallucinated terrain.** Super-resolution infers detail, it does not measure it. The output is a plausible prior, not ground truth.

**Ensemble spread is not uncertainty.** The standard deviation across three models is uncalibrated disagreement. Where all models hallucinate the same crater, the spread stays low. If you ship uncertainty, report calibration: prediction interval coverage probability, and check it on held-out tiles.

**Sparse labels.** 1,353 CTX DEMs cover a tiny fraction of the planet, and the full archive is 1.2 TB. You can validate only on held-out tiles.

**Vertical datum.** CTX DEMs in the UCLA archive are tied to the MOLA areoid, which is what you want. DEMs from other pipelines may not be. Check before pairing.

**Signal-to-noise.** Nobody has checked whether the relief you are trying to recover survives the noise. MOLA carries about ±3 m vertical uncertainty and CTX stereo carries its own error. If the target band has only a few metres of signal, the model is fitting noise. Measure the variogram and the MOLA-versus-CTX power spectrum in the target band before training.

## 2. Best practices

**Use real MOLA as the input.** Pair the real 463 m MOLA clip with the CTX DEM resampled to the output grid. No downsampler anywhere in the loop. This is the only way the test measures the real task.

**Split by tile, not by strip.** A held-out strip inside one 31 by 43 km flood lava plain is adjacent pixels. Use at least five tiles and leave one tile out. Lethe alone is only about 67 by 94 MOLA pixels, too small to prove anything.

**Baseline against HRSC and kriging, not just bicubic.** Bicubic and Lanczos are the floor. Add kriging or a Gaussian-process interpolation, and compare against the 50 m HRSC DTM on the same area. If you cannot beat those, stop.

**Start with a learned interpolator.** Handmer's InterpNet is a small CNN trained only on MOLA that beats naive upsampling before any optical data enters. That is a clean first milestone, not a final claim.

**Predict the residual, not the height.** Feed the interpolated DEM and ask the network for the missing high-frequency correction. Easier to train and easier to reason about.

**Normalize per tile.** Heights span kilometres across Mars. Rescale each tile to a fixed range and store the scale.

**Slide with overlap.** Handmer notes the network is worse at tile edges, so he runs a sliding window and keeps the interior. Do the same or you get a visible grid.

**Climb one scale at a time.** Prove 2x, then 4x, then 8x. Recursion compounds error.

**Use the Good Pixel Map.** Drop label pixels where stereo correlation failed.

**Keep the metric set small and honest.** RMSE, MAE, slope error, SSIM, one hillshade figure, and one spectral score. That is enough.

## 3. Pitfalls and quirks

- The plan's "64 ppd" is wrong. It is 128 ppd.
- The plan's "CTX already aligned to MOLA" is true at the datum level only.
- The plan's SRCNN is a 2015 RGB super-resolution network. It works as a baseline, not as the state of the art.
- Handmer used the `libtiff` package (`from libtiff import TIFF`), which is old. Prefer rasterio or tifffile.
- TensorFlow may not have cp314 wheels. Prefer PyTorch, or pin an older Python in a venv.
- MOLA is signed 16-bit. Cast to float32 before computing differences or squares.
- The Lethe tile's NoData is `-3.4028235e38`, not zero.
- The south polar mosaic is 50 m and Lethe is 18 m. Do not mix scales without resampling.
- Equirectangular pixels are not square on the ground away from the equator.
- The MOLA GeoTIFF stores its image directory at the end of the file. Header-only sniffing fails.

## 4. Differentiation

The industry standard is optical-guided photoclinometry (Handmer) or single-image DTM estimation (MADNet, MARSGAN, LoGAN). Both are well covered.

Three candidate differences, ranked by how much they actually buy you:

**Denser height labels.** Handmer supervises against MOLA at 463 m. You would supervise against CTX stereo DEMs at 18 m, which is 25 times denser. This is a data claim, not a method claim. It is real, but it only matters if the 18 m labels are trustworthy where they exist and the model transfers where they do not.

**Multi-modal input with a measured target.** Give the network real MOLA height plus the CTX optical image, and train against the CTX stereo DEM. Handmer's input is optical alone and his target is MOLA. Yours would be both inputs and a denser target. This is the most defensible version.

**Calibrated uncertainty.** An ensemble with a coverage check. The Neural Process paper already does probabilistic gap filling on the Moon, so uncertainty alone is not novel, but calibrated uncertainty on Mars DEM enhancement is thin.

A pure "MOLA in, bigger MOLA out" model with no optical input is the weakest option. It cannot invent detail that is not in the input, so it mostly learns a smoother interpolator. Do not call that the contribution.

Honest conclusion: the useful difference is real MOLA plus optical input, denser stereo labels, and calibration. That is a small extension of Handmer, not a new field. Say so in the README.

## Recommendation

Reframe the project before writing code. The product is not "a better MOLA". It is "18 m topography where only 463 m MOLA and 5 m optical exist". HRSC 50 m is the baseline to beat, not bicubic.

**v1, CPU only, one weekend per step:**

1. Pick five CTX DEM tiles from the UCLA archive with different terrain: a flat plain, a crater, a canyon, a polar site, and a volcanic slope. Download only those tiles.
2. For each tile, clip the matching real MOLA region and co-register by cross-correlation. Record every shift in the JSON.
3. Check the signal first. Compute the variogram and compare MOLA-versus-CTX differences at MOLA support. If the target relief is inside the noise, write that down and stop.
4. Build pairs with real MOLA as input and the CTX DEM resampled to the output grid as target. No downsampler.
5. Baselines: bicubic, Lanczos, kriging, and the matching HRSC 50 m DTM where available.
6. Train a small CNN with residual output on 32x32 patches. Leave one tile out.
7. Evaluate on the held-out tile with RMSE, MAE, slope error, SSIM, hillshade, and the spectral score. Report against HRSC and CTX, not only against bicubic.
8. Ship `analyze_mars.py`, `mars_elev.json`, `mars_elev_plot.svg`, and a README that states where the model beats HRSC and where it does not.

**v2, only if v1 clears HRSC on the held-out tile:**

- Add the CTX optical image as a second input.
- Add slope and curvature loss.
- Add a three-model ensemble with a coverage check on held-out tiles.
- Scale to twenty tiles across latitude bands.

**Do not do yet:** global 90x enhancement, transfer learning from the Moon, terrain-type conditioning, an interactive viewer. None of that is needed to know whether the idea works.

## Sources

- USGS Astropedia, Mars MGS MOLA DEM 463m: https://astrogeology.usgs.gov/search/map/mars_mgs_mola_dem_463m
- MOLA GeoTIFF (live, 2.13 GB): https://planetarymaps.usgs.gov/mosaic/Mars_MGS_MOLA_DEM_mosaic_global_463m.tif
- Handmer 2021 draft, arXiv:2101.04812: https://arxiv.org/abs/2101.04812
- ODENet code: https://github.com/CHandmer/molaCTXEnhance
- D-SRGAN, arXiv:2004.04788: https://arxiv.org/abs/2004.04788
- DSRFB, arXiv:2007.01940: https://arxiv.org/abs/2007.01940
- LoGAN, ISPRS J. 2025: https://www.sciencedirect.com/science/article/pii/S0924271625001492
- MADNet 2.0, Remote Sensing 2021: https://www.mdpi.com/2072-4292/13/21/4220
- Subpixel CaSSIS DTM, Remote Sensing 2022: https://www.mdpi.com/2072-4292/14/2/257
- Ultra-high-res CaSSIS DTM, Remote Sensing 2021: https://www.mdpi.com/2072-4292/13/11/2185
- Neural Process lunar DEM gap filling, arXiv:2004.09791: https://arxiv.org/abs/2004.09791
- UCLA / GALE Lab Mars DEM archive: https://astrogeology.usgs.gov/search/map/mars_human_exploration_zone_dem_archive_2023
- GALE Lab DEM index repo: https://github.com/GALE-Lab/Mars_DEMs
- HRSC 50 m DTM product guide: https://www.cosmos.esa.int/documents/772136/977578/PUG-UCL-HRSC.pdf
- PRPDC CTX DEMs (host timed out during check, Wayback shows it alive Dec 2024): https://www.higp.hawaii.edu/prpdc/CTX_DEMs/toplevel.html
- Murray Lab CTX mosaic: https://murray-lab.caltech.edu/CTX/
- Murray Lab beta01 details: http://murray-lab.caltech.edu/CTX/beta01.html
- Lethe Vallis 18 m DEM (live, 16 MB): https://planetarymaps.usgs.gov/mosaic/dundas_flood_lavas_topo/DEM_18m_Lethe_Vallis.tif
- South polar CTX mosaic (live, 256 MB): https://planetarymaps.usgs.gov/mosaic/Mars/mars_mro_ctx_spoledtmmosaic_mayer_2020/ctx_abs_align_iau_deltaradius_mosaic.tif
- Mars MOLA reference script: https://github.com/theonlyallana/mars-mola-dem

## Adversarial verification

Ran a source pass and a logic pass. Both found real problems, now fixed in this document.

Source pass results:
- URLs verified live: MOLA, Lethe, south polar, arXiv pages, GitHub repos, USGS pages, ESA PDF. MDPI and ScienceDirect return 403 to bots but resolve through Crossref. The Hawaii PRPDC host timed out; not a 404.
- Byte counts verified exactly: MOLA 2,125,771,142; Lethe 16,145,119; south polar 256,048,639.
- MOLA dimensions, int16, 128 ppd, 463.08 m, CC0, areoid numbers, polar gap numbers all match the USGS record.
- Lethe is 1,691 by 2,384 at 18.21 m, float32. South polar is 8,000 by 8,000 at 50.0 m.
- Handmer spectral scores, 90x, 4x Gale, mismatch, training time all match the PDF.
- D-SRGAN MSE table, DSRFB RMSE range match the PDFs.

Fixes applied from the source pass:
- "more than 99.5%" changed to "99.5%" for Murray Lab coverage.
- Murray Lab v01 size changed from "about 10 TB" to "about 11.48 TB uncompressed, 5.6 TB compressed".
- `pytiff` corrected to the `libtiff` package.
- MARSGAN attribution corrected: it is the GAN used inside the UCL work, not a paper title.
- Handmer relabelled as an unreviewed draft, not a peer-reviewed paper.
- LoGAN's 0.25 m and three-region claims marked as unverifiable (closed access).
- MADNet's resolution softened to "pixel scale, HiRISE input at 25 to 50 cm".
- Added a footnote on the 1,353 versus 1,354 DEM count conflict.

Logic pass results:
- The document contradicted itself: it called pure "MOLA in, bigger MOLA out" the weakest option, then made that the v1. Fixed by making real MOLA the input and CTX the target, and by relabelling v1 as a test that must beat HRSC.
- The original "height-label supervision has no lighting bias" claim was false, because Handmer already supervises against MOLA heights. Restated as a label-density claim (18 m versus 463 m).
- The original v1 trained on downsampled CTX and tested on downsampled CTX, which proves only that a CNN can invert averaging. Removed the downsampler from the input path.
- The Lethe tile is too small for a held-out strip. Fixed with five tiles and leave-one-tile-out.
- Added the signal-to-noise check before training.
- Added the calibrated uncertainty requirement (coverage, not raw ensemble spread).
- Added HRSC 50 m as the real baseline and the "why bother" question.
- Added missing literature: Neural Process lunar gap filling, guided DEM super-resolution, Dfilled.
- Added box landmines: Lethe NoData value, MOLA TIFF layout, cp314 wheel risk.

Status: GREEN after fixes.
