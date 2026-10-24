# %% [markdown]
# # MOLA super-resolution v1 (Colab)
#
# Goal: take the real 463 m MOLA elevation and add detail up to about 58 m (8x), using CTX stereo DEMs as the target. This is the honest version from the research doc: real MOLA goes in, real CTX comes out, no downsampler in the loop.
#
# What this notebook does:
#
# 1. Downloads the global MOLA GeoTIFF and five CTX stereo DEMs from the UCLA / GALE Lab archive.
# 2. Co-registers each tile and builds a common 58 m grid.
# 3. Runs baselines: bilinear, bicubic.
# 4. Trains a small residual CNN, leave-one-tile-out.
# 5. Reports RMSE, MAE, slope error, SSIM, and hillshade images.
#
# Runtime on a free T4: about 15 to 25 minutes, most of it the 2.1 GB MOLA download. Set `SMOKE = True` for a 2 minute test on one tile.
#
# This is a first pass. It does not prove the method beats HRSC yet. It tells you whether the pipeline works and whether the signal is there.

# %% [markdown]
# ## 1. Install
# Colab already has torch. We only add the geospatial readers.

# %%
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rasterio", "scikit-image"], check=True)
print("deps ok")

# %% [markdown]
# ## 2. Config

# %%
import os, math, json, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates, shift as nd_shift, gaussian_filter
from skimage.transform import resize
from skimage.registration import phase_cross_correlation
from skimage.metrics import structural_similarity as ssim_fn

DATA = Path("/content/data"); DATA.mkdir(parents=True, exist_ok=True)
OUT = Path("/content/out"); OUT.mkdir(parents=True, exist_ok=True)

MOLA_URL = "https://planetarymaps.usgs.gov/mosaic/Mars_MGS_MOLA_DEM_mosaic_global_463m.tif"
MOLA_PATH = DATA / "mola_463m.tif"
MOLA_PPD = 128.0
MARS_R = 3396190.0

S3 = "https://asc-pds-individual-investigations.s3.us-west-2.amazonaws.com/mars_mro_hirise_explorationzones_day_2023/data"

TILES = [
    {"id": "t1_south_mid", "name": "b01_009894_1665_xi_13s042w_b02_010606_1666_xn_13s042w", "lat": -13, "lon": -42},
    {"id": "t2_equator",   "name": "b02_010316_1830_xn_03n043w_p22_009683_1830_xn_03n043w", "lat": 3,   "lon": -43},
    {"id": "t3_north_low", "name": "b01_009929_2019_xn_21n281w_b01_010219_2019_xi_21n281w", "lat": 21,  "lon": -281},
    {"id": "t4_north_high","name": "b01_010192_2189_xn_38n262w_p22_009704_2189_xn_38n262w", "lat": 38,  "lon": -262},
    {"id": "t5_south_high","name": "b02_010326_1560_xn_24s315w_b02_010471_1560_xn_24s315w", "lat": -24, "lon": -315},
]
for t in TILES:
    t["url"] = f"{S3}/{t['name']}_tied-dem.tif"
    t["path"] = DATA / f"{t['id']}_ctx.tif"

TARGET_M = 463.08 / 8.0   # 57.9 m, the 8x grid
PATCH = 64
STRIDE = 32
EPOCHS = 20
BATCH = 32
LR = 1e-3
SMOKE = False             # True: one tile, two epochs, fast check

SEED = 7
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
if DEV == "cuda":
    torch.backends.cudnn.benchmark = True
print("device:", DEV, "| torch:", torch.__version__)

# %% [markdown]
# ## 3. Download
# The MOLA file is 2.1 GB. It lands on the Colab disk, not in RAM.

# %%
def fetch(url, path):
    path = Path(path)
    part = path.with_suffix(path.suffix + ".part")
    if path.exists() and path.stat().st_size > 1_000_000:
        print(f"have {path.name} ({path.stat().st_size/1e6:.0f} MB)")
        return
    print(f"download {path.name}")
    t0 = time.time()
    try:
        subprocess.run(["curl", "-L", "--fail", "--retry", "3", "-C", "-", "-o", str(part), url], check=True)
        part.replace(path)
    finally:
        if part.exists() and not path.exists():
            print(f"  incomplete download left at {part}")
    print(f"  done in {time.time()-t0:.0f}s ({path.stat().st_size/1e6:.0f} MB)")

fetch(MOLA_URL, MOLA_PATH)
for t in TILES:
    fetch(t["url"], t["path"])
    if SMOKE:
        break

# %% [markdown]
# ## 4. Inspect the tiles
# This checks the CRS, pixel size, NoData, and value range before we trust anything.

# %%
def describe(path):
    with rasterio.open(path) as ds:
        print(f"{path.name}: {ds.width}x{ds.height} crs={ds.crs} nodata={ds.nodata} dtype={ds.dtypes[0]}")
        print(f"  bounds={ds.bounds}")
        print(f"  pixel={abs(ds.transform.a):.6f},{abs(ds.transform.e):.6f} deg")
        a = ds.read(1, out_shape=(min(ds.height, 512), min(ds.width, 512))).astype(np.float32)
        nod = ds.nodata
        if nod is not None:
            a[a == nod] = np.nan
        a[a < -1e30] = np.nan
        print(f"  elev min/med/max = {np.nanmin(a):.0f} / {np.nanmedian(a):.0f} / {np.nanmax(a):.0f} m")
        print(f"  valid = {np.isfinite(a).mean()*100:.1f}%")

describe(MOLA_PATH)
for t in TILES:
    if t["path"].exists():
        describe(t["path"])
    if SMOKE:
        break

# %% [markdown]
# ## 5. Grid and sampling
# Both MOLA and the CTX DEMs are equirectangular degrees, so we work in lon/lat directly and skip the missing-EPSG problem.
#
# MOLA is a fixed global grid: 128 pixels per degree, lon -180 to 180, lat 90 to -90. We read a small window and sample it with `map_coordinates`. Order 1 is bilinear, order 3 is bicubic.

# %%
def wrap_lon(x):
    return ((np.asarray(x, dtype=np.float64) + 180.0) % 360.0) - 180.0

def mola_window(lon_min, lon_max, lat_min, lat_max, pad=4):
    c0 = int(math.floor((lon_min + 180.0) * MOLA_PPD)) - pad
    c1 = int(math.ceil((lon_max + 180.0) * MOLA_PPD)) + pad
    r0 = int(math.floor((90.0 - lat_max) * MOLA_PPD)) - pad
    r1 = int(math.ceil((90.0 - lat_min) * MOLA_PPD)) + pad
    c0 = max(c0, 0); r0 = max(r0, 0)
    c1 = min(c1, 46080); r1 = min(r1, 23040)
    return r0, r1, c0, c1

def sample_mola(ds, lon2d, lat2d, order=1):
    """Sample the global MOLA raster at lon/lat arrays. Returns float32, NaN outside."""
    lon = wrap_lon(lon2d)
    lat = np.asarray(lat2d, dtype=np.float64)
    if np.nanmax(lon) - np.nanmin(lon) > 180.0:
        raise ValueError("tile crosses the antimeridian; split it first")
    r0, r1, c0, c1 = mola_window(np.nanmin(lon), np.nanmax(lon), np.nanmin(lat), np.nanmax(lat), pad=4)
    win = ds.read(1, window=Window(c0, r0, c1 - c0, r1 - r0)).astype(np.float32)
    if ds.nodata is not None:
        win[win == ds.nodata] = np.nan
    win[win < -1e30] = np.nan
    col = (lon + 180.0) * MOLA_PPD - 0.5 - c0
    row = (90.0 - lat) * MOLA_PPD - 0.5 - r0
    out = map_coordinates(win, [row, col], order=order, mode="nearest")
    return out.astype(np.float32)

def load_ctx(path, target_m=TARGET_M):
    """Read a CTX DEM, mask NoData, resample to the target ground size, return lon/lat grid."""
    with rasterio.open(path) as ds:
        if not ds.crs.is_geographic:
            raise ValueError(f"{path.name}: CRS is not geographic ({ds.crs}). Handle projection first.")
        if abs(ds.transform.b) > 1e-12 or abs(ds.transform.d) > 1e-12:
            raise ValueError(f"{path.name}: rotated transform, not supported in v1")
        arr = ds.read(1).astype(np.float32)
        nod = ds.nodata
        if nod is not None:
            arr[arr == nod] = np.nan
        arr[arr < -1e30] = np.nan
        h, w = arr.shape
        lon_l = ds.transform.c
        lon_r = ds.transform.c + ds.transform.a * w
        lat_t = ds.transform.f
        lat_b = ds.transform.f + ds.transform.e * h
        px_x = abs(ds.transform.a)
        px_y = abs(ds.transform.e)
        lon_l, lon_r = float(wrap_lon(lon_l)), float(wrap_lon(lon_r))
        if lon_r < lon_l:
            lon_r += 360.0
        lat_b, lat_t = sorted([float(lat_b), float(lat_t)])
    lat_mid = 0.5 * (lat_b + lat_t)
    ctx_dx_m = px_x * math.pi / 180.0 * MARS_R * math.cos(math.radians(lat_mid))
    ctx_dy_m = px_y * math.pi / 180.0 * MARS_R
    fx = max(1.0, target_m / ctx_dx_m)
    fy = max(1.0, target_m / ctx_dy_m)
    new_w = max(8, int(round(w / fx)))
    new_h = max(8, int(round(h / fy)))
    valid = np.isfinite(arr)
    fill = np.where(valid, arr, np.nanmedian(arr)).astype(np.float32)
    hr = resize(fill, (new_h, new_w), anti_aliasing=True, preserve_range=True).astype(np.float32)
    mask = resize(valid.astype(np.float32), (new_h, new_w), anti_aliasing=True, preserve_range=True) > 0.99
    lons = lon_l + (np.arange(new_w) + 0.5) * (lon_r - lon_l) / new_w
    lats = lat_t - (np.arange(new_h) + 0.5) * (lat_t - lat_b) / new_h
    LON, LAT = np.meshgrid(lons, lats, indexing="xy")
    return hr, mask, LON, LAT, (lat_b, lat_t, lon_l, lon_r)

def build_tile(t):
    hr, mask, LON, LAT, bounds = load_ctx(t["path"])
    with rasterio.open(MOLA_PATH) as ds:
        lr_bilinear = sample_mola(ds, LON, LAT, order=1)
        lr_bicubic = sample_mola(ds, LON, LAT, order=3)
    valid = mask & np.isfinite(lr_bilinear) & np.isfinite(hr)
    if valid.mean() < 0.5:
        raise ValueError(f"{t['id']}: only {valid.mean()*100:.0f}% valid pixels")
    # alignment: estimate the shift of MOLA onto CTX on the large-scale signal
    a = np.where(valid, hr, np.nanmedian(hr)).astype(np.float32)
    b = np.where(valid, lr_bilinear, np.nanmedian(lr_bilinear)).astype(np.float32)
    a = gaussian_filter(a - a.mean(), 8); b = gaussian_filter(b - b.mean(), 8)
    try:
        sh, err, _ = phase_cross_correlation(a, b, upsample_factor=10, normalization=None)
    except Exception as e:
        sh, err = (0.0, 0.0), float("nan")
        print("  phase correlation failed:", e)
    sh = np.nan_to_num(np.array(sh, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if np.max(np.abs(sh)) > 12:
        print(f"  warning: large shift {sh}, not applying")
        sh = np.zeros(2, dtype=np.float32)
    lr_bilinear = nd_shift(lr_bilinear, sh, order=1, mode="nearest")
    lr_bicubic = nd_shift(lr_bicubic, sh, order=3, mode="nearest")
    valid = valid & np.isfinite(lr_bilinear)
    t.update(dict(hr=hr, lr=lr_bilinear, lr_bicubic=lr_bicubic, mask=valid,
                  lon=LON, lat=LAT, bounds=bounds, shift=sh.tolist(),
                  lat_mid=float(np.mean(LAT[valid]))))
    print(f"{t['id']}: {hr.shape} valid={valid.mean()*100:.0f}% shift={sh} "
          f"relief={np.nanpercentile(hr[valid],99)-np.nanpercentile(hr[valid],1):.0f} m")
    return t

print("building tiles...")
TILES = [build_tile(t) for t in (TILES[:1] if SMOKE else TILES)]

# %% [markdown]
# ## 6. Signal check
# Before training, check that the detail we want to recover is bigger than the noise. If the relief at the target scale is only a few metres, the model will fit noise.

# %%
for t in TILES:
    v = t["mask"]
    d = (t["hr"] - t["lr"])[v]
    p1, p99 = np.percentile(t["hr"][v], [1, 99])
    print(f"{t['id']}: relief p1-p99 = {p99-p1:.0f} m | "
          f"HR-LR residual std = {d.std():.2f} m | "
          f"residual p99 = {np.percentile(np.abs(d), 99):.2f} m")

# %% [markdown]
# ## 7. Baselines
# Bilinear and bicubic are the floor. If the CNN cannot beat these, stop.

# %%
def grid_spacing_m(t):
    lat_mid = math.radians(t["lat_mid"])
    vb = t["bounds"]; h, w = t["hr"].shape
    dx = (vb[3] - vb[2]) / w * math.pi / 180.0 * MARS_R * math.cos(lat_mid)
    dy = (vb[1] - vb[0]) / h * math.pi / 180.0 * MARS_R
    return dx, dy

def score(pred, target, mask, dx, dy):
    m = mask & np.isfinite(pred) & np.isfinite(target)
    if m.sum() < 100:
        return {}
    fill = float(np.nanmedian(target[m]))
    pred_f = np.where(m, pred, fill).astype(np.float64)
    target_f = np.where(m, target, fill).astype(np.float64)
    d = pred_f[m] - target_f[m]
    rmse = float(np.sqrt(np.mean(d * d)))
    mae = float(np.mean(np.abs(d)))
    # slope error, in m/m
    gy_t, gx_t = np.gradient(target_f, dy, dx)
    gy_p, gx_p = np.gradient(pred_f, dy, dx)
    sm = m[1:-1, 1:-1]
    sx = (gx_p - gx_t)[1:-1, 1:-1][sm]
    sy = (gy_p - gy_t)[1:-1, 1:-1][sm]
    slope_mae = float(np.nanmean(np.sqrt(sx * sx + sy * sy)))
    # SSIM on normalized height
    lo, hi = np.percentile(target_f[m], [1, 99])
    rng = max(hi - lo, 1e-6)
    a = np.clip((pred_f - lo) / rng, 0, 1)
    b = np.clip((target_f - lo) / rng, 0, 1)
    ssim = float(ssim_fn(a, b, data_range=1.0))
    return {"rmse_m": rmse, "mae_m": mae, "slope_mae": slope_mae, "ssim": ssim, "n": int(m.sum())}

BASELINES = {}
for t in TILES:
    dx, dy = grid_spacing_m(t)
    BASELINES[t["id"]] = {
        "bilinear": score(t["lr"], t["hr"], t["mask"], dx, dy),
        "bicubic": score(t["lr_bicubic"], t["hr"], t["mask"], dx, dy),
    }
    print(t["id"], BASELINES[t["id"]])

# %% [markdown]
# ## 8. Model
# A small residual CNN. Input is MOLA at 58 m, output is the correction. Predicting the residual is easier than predicting absolute height.

# %%
class ResidualSR(nn.Module):
    def __init__(self, ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, 1, 3, padding=1),
        )
    def forward(self, x):
        return x + self.net(x)

def norm_stats(t):
    v = t["mask"]
    mu = float(np.nanmean(t["hr"][v]))
    sd = float(np.nanstd(t["hr"][v]))
    return mu, max(sd, 1e-6)

def patches(t, mu, sd):
    lr = (t["lr"] - mu) / sd
    hr = (t["hr"] - mu) / sd
    valid = t["mask"]
    xs, ys = [], []
    H, W = lr.shape
    for y in range(0, H - PATCH + 1, STRIDE):
        for x in range(0, W - PATCH + 1, STRIDE):
            if valid[y:y+PATCH, x:x+PATCH].mean() < 0.95:
                continue
            xs.append(lr[y:y+PATCH, x:x+PATCH])
            ys.append(hr[y:y+PATCH, x:x+PATCH])
    if not xs:
        return np.zeros((0, 1, PATCH, PATCH), np.float32), np.zeros((0, 1, PATCH, PATCH), np.float32)
    X = np.stack(xs)[:, None].astype(np.float32)
    Y = np.stack(ys)[:, None].astype(np.float32)
    return X, Y

def train_fold(train_tiles, epochs=EPOCHS, verbose=True):
    Xs, Ys = [], []
    stats = {}
    for t in train_tiles:
        mu, sd = norm_stats(t)
        stats[t["id"]] = (mu, sd)
        X, Y = patches(t, mu, sd)
        Xs.append(X); Ys.append(Y)
    X = np.concatenate(Xs); Y = np.concatenate(Ys)
    assert len(X) > 0, "no training patches built, check the tile masks and PATCH size"
    order = np.random.permutation(len(X))
    X, Y = X[order], Y[order]
    print(f"train patches: {X.shape}")
    model = ResidualSR().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n = len(X)
    for ep in range(epochs):
        model.train()
        idx = np.random.permutation(n)
        tot = 0.0
        for i in range(0, n, BATCH):
            j = idx[i:i+BATCH]
            xb = torch.from_numpy(X[j]).to(DEV)
            yb = torch.from_numpy(Y[j]).to(DEV)
            pred = model(xb)
            loss = F.l1_loss(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(j)
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            print(f"  epoch {ep+1:02d}/{epochs} loss={tot/n:.5f}")
    return model, stats

@torch.no_grad()
def predict_tile(model, t):
    mu, sd = norm_stats(t)
    lr = (t["lr"] - mu) / sd
    x = torch.from_numpy(lr[None, None]).to(DEV)
    out = model(x)[0, 0].cpu().numpy()
    return out * sd + mu

# %% [markdown]
# ## 9. Leave-one-tile-out
# Train on four tiles, test on the fifth. This is the number that matters.

# %%
results = []
folds = TILES if not SMOKE else TILES
for i, test_tile in enumerate(folds):
    train_tiles = [t for t in TILES if t["id"] != test_tile["id"]]
    if not train_tiles:
        train_tiles = [test_tile]
    print(f"\nfold {i+1}/{len(folds)}: test={test_tile['id']}")
    model, _ = train_fold(train_tiles, epochs=(2 if SMOKE else EPOCHS))
    pred = predict_tile(model, test_tile)
    dx, dy = grid_spacing_m(test_tile)
    s = score(pred, test_tile["hr"], test_tile["mask"], dx, dy)
    s.update({"fold": i + 1, "test": test_tile["id"],
              "bicubic_rmse_m": BASELINES[test_tile["id"]]["bicubic"]["rmse_m"],
              "bilinear_rmse_m": BASELINES[test_tile["id"]]["bilinear"]["rmse_m"],
              "shift": test_tile["shift"]})
    results.append(s)
    print("  model  ", {k: round(v, 4) if isinstance(v, float) else v for k, v in s.items()})
    print("  bicubic", BASELINES[test_tile["id"]]["bicubic"])
    np.save(OUT / f"pred_{test_tile['id']}.npy", pred.astype(np.float32))
    torch.save(model.state_dict(), OUT / f"model_{test_tile['id']}.pt")

# %% [markdown]
# ## 10. Results

# %%
print(f"{'tile':16s} {'model':>8s} {'bicubic':>8s} {'bilinear':>9s} {'slope':>7s} {'ssim':>6s}")
for r in results:
    print(f"{r['test']:16s} {r['rmse_m']:8.2f} {r['bicubic_rmse_m']:8.2f} {r['bilinear_rmse_m']:9.2f} "
          f"{r['slope_mae']:7.4f} {r['ssim']:6.3f}")
wins = sum(1 for r in results if r["rmse_m"] < r["bicubic_rmse_m"])
print(f"\nmodel beats bicubic on {wins}/{len(results)} held-out tiles")

summary = {
    "target_m": TARGET_M,
    "epochs": EPOCHS,
    "smoke": SMOKE,
    "results": results,
    "baselines": BASELINES,
}
(OUT / "results.json").write_text(json.dumps(summary, indent=2))
print("wrote", OUT / "results.json")

# %% [markdown]
# ## 11. Hillshade check
# Numbers can hide a blurry output. Look at the pictures.

# %%
def hillshade(z):
    gy, gx = np.gradient(z)
    slope = np.pi / 2 - np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az, alt = np.radians(315), np.radians(45)
    return np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)

for t in TILES:
    pred_path = OUT / f"pred_{t['id']}.npy"
    if not pred_path.exists():
        continue
    pred = np.load(pred_path)
    v = t["mask"]
    lo = np.percentile(t["hr"][v], 1)
    hi = np.percentile(t["hr"][v], 99)
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    for a, z, name in zip(ax, [t["lr_bicubic"], pred, t["hr"], pred - t["hr"]],
                          ["MOLA bicubic", "model", "CTX truth", "model - truth"]):
        if name.endswith("truth") or "model -" in name:
            a.imshow(z, cmap="gray" if "model -" in name else "terrain")
            if "model -" in name:
                a.imshow(z, cmap="RdBu", vmin=-50, vmax=50)
        else:
            a.imshow(hillshade(z), cmap="gray")
        a.set_title(name); a.axis("off")
    plt.tight_layout()
    p = OUT / f"hillshade_{t['id']}.png"
    plt.savefig(p, dpi=120, bbox_inches="tight"); plt.close()
    print("wrote", p)

# %% [markdown]
# ## 12. Save to Drive (optional)
# Uncomment to keep results after the Colab runtime dies.

# %%
# from google.colab import drive
# drive.mount("/content/drive")
# import shutil
# shutil.copytree(OUT, "/content/drive/MyDrive/mola_sr_v1", dirs_exist_ok=True)
# print("copied to Drive")

# %% [markdown]
# ## What success looks like
#
# - **Pipeline works**: all five tiles build, shifts are small, valid pixels above 50%.
# - **Signal exists**: the HR minus LR residual std is well above the MOLA noise floor.
# - **Model beats bicubic** on at least 3 of 5 held-out tiles, and the hillshade does not look like a blur.
# - **Honest failure**: if it does not beat bicubic, that is a real result. Write it down and stop. Do not tune until it passes.
#
# This notebook does not yet compare against HRSC 50 m or kriging. Add those before claiming the method is useful.
