# MOLA super-resolution, v1 (Colab)

This folder holds the Colab notebook for the MOLA gap-filler experiment.

- `mola_sr_v1.py` is the source of truth, in percent-cell format.
- `build_notebook.py` turns it into `mola_sr_v1.ipynb`.
- `mola_sr_v1.ipynb` is the file you upload to Colab. Do not edit it by hand.

## Open it

Upload `mola_sr_v1.ipynb` at https://colab.research.google.com (File, Upload notebook), or drag it into Drive and open with Colab.

## Runtime

- Pick the free T4 GPU runtime (Runtime, Change runtime type). CPU works too, just slower.
- Needs about 3 GB of disk for the MOLA file and the five CTX DEMs. Colab gives you that.
- Full run is roughly 15 to 25 minutes, mostly the 2.1 GB MOLA download.
- Set `SMOKE = True` in cell 2 for a two minute check that the pipeline runs at all.

## What it does

1. Downloads the global 463 m MOLA GeoTIFF and five CTX stereo DEMs from the UCLA / GALE Lab archive.
2. Reads a small window of MOLA around each CTX tile, and resamples the CTX DEM down to 58 m (an 8x grid).
3. Co-registers each tile with phase correlation and records the shift.
4. Scores bilinear and bicubic baselines.
5. Trains a small residual CNN, leave one tile out each fold.
6. Writes RMSE, MAE, slope error, SSIM, hillshades, `results.json`, and the per-fold predictions.

Real MOLA is the input. Real CTX is the target. No downsampler sits in the input path, so the test measures the actual task.

## What success looks like

- All five tiles build, shifts stay small, valid pixels above 50 percent.
- The HR minus LR residual std is well above the MOLA noise floor.
- The model beats bicubic on at least three of five held-out tiles, and the hillshade is not a blur.

If it loses to bicubic, that is a real result. Write it down and stop. Do not tune until it passes.

## What this does not prove

It does not compare against HRSC 50 m or kriging, and it does not cover the planet. Beating bicubic is the floor, not the claim. Add HRSC and kriging before calling the method useful.

## Regenerate the notebook

```bash
python3 colab/build_notebook.py
```

Edit `mola_sr_v1.py`, then run that. The builder only splits on `# %%` markers and strips `# ` from markdown cells.
