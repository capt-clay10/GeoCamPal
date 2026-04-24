"""
georef.py — GeoCamPal Unified Georeferencing Module

Workflow:
  1. Choose image
  2. Choose method (Homography / Camera Projection / TPS / Poly1 / Poly2)
  3. Load required inputs (varies by method)
  4. Choose corner preset + crop values (all methods)
  5. Initial Georeferencing → preview
  6. Select AOI (interactive box or manual pixel coords)
  7. Compute Accuracy (LOO cross-validation per GCP)
  8. Optimise GCPs (SA) or manually exclude bad ones
  9. Secondary Georeferencing (refined GCPs + scale)
  10. Batch process (single folder or subfolders) with validation
"""

import os, sys, time, glob, re, pickle, threading, concurrent.futures, random
import numpy as np
import cv2
import pandas as pd
from PIL import Image, ImageTk
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
from osgeo import gdal, osr
gdal.UseExceptions()

from utils import fit_geometry, resource_path, setup_console, restore_console, save_settings_json, load_settings_json

try:
    from csv_utils import read_gcp_csv, normalise_columns
except ImportError:
    def read_gcp_csv(p, verbose=True):
        df = pd.read_csv(p, sep=None, engine="python", encoding="utf-8-sig")
        df.columns = [c.strip() for c in df.columns]
        if "Real_Z" not in df.columns: df["Real_Z"] = 0.0
        if "EPSG" not in df.columns: df["EPSG"] = 0
        return df
    def normalise_columns(df, verbose=True): return df

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")

MAX_BATCH_WORKERS = max(1, min(4, os.cpu_count() or 1))
MIN_GEOTIFF_SIZE  = 10_000
MAX_RETRIES       = 10
METHODS = ["Homography", "Camera Projection", "Thin Plate Spline",
           "Polynomial Ord-1", "Polynomial Ord-2"]
_METHOD_KEY = {"Homography": "homo", "Camera Projection": "proj",
               "Thin Plate Spline": "tps", "Polynomial Ord-1": "poly1",
               "Polynomial Ord-2": "poly2"}

# ── GeoTIFF validation ──
def validate_geotiff(fp, min_size=MIN_GEOTIFF_SIZE):
    if not os.path.exists(fp): return False, "Missing"
    if os.path.getsize(fp) < min_size: return False, "Too small"
    try:
        ds = gdal.Open(fp)
        if ds is None: return False, "GDAL fail"
        w, h = ds.RasterXSize, ds.RasterYSize; ds = None
        return (w > 1 and h > 1), f"dims {w}x{h}"
    except Exception as e: return False, str(e)

def sweep_and_validate(folder):
    tifs = glob.glob(os.path.join(folder, "*.tif"))
    if not tifs: return []
    sizes = [os.path.getsize(f) for f in tifs if os.path.exists(f)]
    if not sizes: return []
    s = sorted(sizes); med = s[len(s)//2]
    thr = max(MIN_GEOTIFF_SIZE, int(med * 0.5))
    return [(fp, r) for fp in tifs for ok, r in [validate_geotiff(fp, thr)] if not ok]

def _estimate_gsd(df):
    px, utm = df[["Pixel_X","Pixel_Y"]].values, df[["Real_X","Real_Y"]].values
    ratios = [np.hypot(*(utm[i]-utm[j]))/np.hypot(*(px[i]-px[j]))
              for i in range(len(df)) for j in range(i+1, len(df))
              if np.hypot(*(px[i]-px[j])) > 10]
    return float(np.median(ratios)) if ratios else 0.25


def _guess_epsg(df):
    """Try to guess EPSG from GCP coordinate ranges (Real_X, Real_Y).
    Returns (epsg_int, description_str) or (None, reason_str)."""
    x_vals = df["Real_X"].values
    y_vals = df["Real_Y"].values
    x_med, y_med = float(np.median(x_vals)), float(np.median(y_vals))

    # Geographic coordinates (lon/lat)
    if -180 <= x_med <= 180 and -90 <= y_med <= 90:
        return 4326, "WGS 84 geographic (lon/lat)"

    # UTM-like coordinates: Easting 100k–900k, Northing 0–10M
    if 100_000 < x_med < 900_000 and 0 < y_med < 10_000_000:
        # Estimate UTM zone from easting is impossible without more info,
        # but if a Real_Z or other context is available we can't determine zone.
        # We can only flag it as likely UTM.
        hemisphere = "north" if y_med < 5_500_000 else "could be north or south"
        return None, f"Coordinates look like UTM (easting~{x_med:.0f}, northing~{y_med:.0f}, {hemisphere}) but zone cannot be determined automatically. Please enter the EPSG code manually."

    return None, f"Could not determine CRS from coordinate ranges (X~{x_med:.0f}, Y~{y_med:.0f}). Please enter the EPSG code manually."

# ── Homography backends ──
def _homo_compute(pixel_pts, utm_pts):
    def _norm(pts):
        c = pts.mean(axis=0); s = pts - c
        d = np.sqrt((s**2).sum(axis=1)).mean()
        sc = np.sqrt(2) / d if d > 0 else 1
        T = np.array([[sc,0,-sc*c[0]],[0,sc,-sc*c[1]],[0,0,1]])
        h = np.hstack([pts, np.ones((len(pts),1))])
        return (T @ h.T).T, T
    pn, Tp = _norm(pixel_pts.astype(np.float32))
    un, Tu = _norm(utm_pts.astype(np.float32))
    Hn, _ = cv2.findHomography(pn[:,:2], un[:,:2], cv2.RANSAC, 0.5)
    if Hn is None: return None
    H = np.linalg.inv(Tu) @ Hn @ Tp
    if abs(H[2,2]) > 1e-6 and abs(H[2,2]-1) > 1e-3: H /= H[2,2]
    return H

def _homo_warp_full(img, H, scale):
    h, w = img.shape[:2]
    corners = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], np.float32).reshape(-1,1,2)
    tc = cv2.perspectiveTransform(corners, H)
    mn = np.floor(tc.min(axis=(0,1))).astype(int)
    mx = np.ceil(tc.max(axis=(0,1))).astype(int)
    ow, oh = int((mx[0]-mn[0])*scale), int((mx[1]-mn[1])*scale)
    if ow <= 0 or oh <= 0: raise ValueError(f"Bad dims {ow}x{oh}")
    T = np.array([[1,0,-mn[0]*scale],[0,1,mx[1]*scale],[0,0,1]], np.float64)
    Hs = H.copy(); Hs[0] *= scale; Hs[1] *= -scale
    Hf = T @ Hs
    warped = cv2.warpPerspective(img, Hf, (ow, oh), flags=cv2.INTER_LANCZOS4)
    gt = (float(mn[0]), 1.0/scale, 0.0, float(mx[1]), 0.0, -1.0/scale)
    return warped, Hf, gt

def _homo_warp_windowed(img, H, scale, source_corners, crop_fracs=(0.,0.,0.,0.)):
    tc = cv2.perspectiveTransform(source_corners.astype(np.float32), H)
    mn = np.floor(tc.min(axis=(0,1))).astype(np.float64)
    mx = np.ceil(tc.max(axis=(0,1))).astype(np.float64)
    min_x, min_y, max_x, max_y = float(mn[0]), float(mn[1]), float(mx[0]), float(mx[1])
    dx, dy = max_x - min_x, max_y - min_y
    cs, cn, ce, cw = crop_fracs
    min_y += dy * cs; max_y += dy * cn; max_x -= dx * ce; min_x -= dx * cw
    ow = int(round((max_x - min_x) * scale))
    oh = int(round((max_y - min_y) * scale))
    if ow <= 0 or oh <= 0: raise ValueError(f"Bad dims {ow}x{oh}")
    T = np.array([[1,0,-min_x*scale],[0,1,max_y*scale],[0,0,1]], np.float64)
    Hs = H.copy(); Hs[0] *= scale; Hs[1] *= -scale
    Hf = T @ Hs
    warped = cv2.warpPerspective(img, Hf, (ow, oh), flags=cv2.INTER_LANCZOS4)
    gt = (float(min_x), 1.0/scale, 0.0, float(max_y), 0.0, -1.0/scale)
    return warped, Hf, gt

# ── Projection (CIRN-style) ──
def _scale_K(K, cal_wh, img_wh):
    Ks = K.astype(np.float64).copy()
    Ks[0,0] *= img_wh[0]/cal_wh[0]; Ks[0,2] *= img_wh[0]/cal_wh[0]
    Ks[1,1] *= img_wh[1]/cal_wh[1]; Ks[1,2] *= img_wh[1]/cal_wh[1]
    return Ks

def _solve_extrinsics(pixel_pts, world_pts, K, dist):
    K64 = K.astype(np.float64); d64 = dist.ravel().astype(np.float64)
    img = pixel_pts.astype(np.float64).reshape(-1,1,2)
    centroid = world_pts.astype(np.float64).mean(axis=0)
    obj = (world_pts.astype(np.float64) - centroid)
    ok = False
    for method in [cv2.SOLVEPNP_ITERATIVE, cv2.SOLVEPNP_SQPNP, cv2.SOLVEPNP_EPNP]:
        try:
            ok, rvec, tvec = cv2.solvePnP(obj, img, K64, d64, flags=method)
            if ok: break
        except cv2.error: continue
    if not ok: raise RuntimeError("solvePnP failed.")
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, K64, d64, rvec, tvec)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K64, d64)
    errors = np.linalg.norm(proj.reshape(-1,2) - pixel_pts.astype(np.float64), axis=1)
    return rvec, tvec, centroid, errors

def _build_grid(x0, x1, y0, y1, res, elev):
    xs = np.arange(x0, x1 + res*0.01, res)
    ys = np.arange(y1, y0 - res*0.01, -res)
    gx, gy = np.meshgrid(xs, ys)
    return gx, gy, np.full_like(gx, elev)

def _rectify_image(img, K, dist, rvec, tvec, centroid, gx, gy, gz):
    hs, ws = img.shape[:2]; oh, ow = gx.shape
    flat = np.column_stack([gx.ravel()-centroid[0], gy.ravel()-centroid[1],
                            gz.ravel()-centroid[2]]).astype(np.float64)
    proj, _ = cv2.projectPoints(flat, rvec, tvec,
                                K.astype(np.float64), dist.ravel().astype(np.float64))
    p = proj.reshape(-1,2)
    mx = p[:,0].reshape(oh, ow).astype(np.float32)
    my = p[:,1].reshape(oh, ow).astype(np.float32)
    R, _ = cv2.Rodrigues(rvec)
    behind = (R @ flat.T + tvec.reshape(3,1))[2,:].reshape(oh, ow) <= 0
    rect = cv2.remap(img, mx, my, cv2.INTER_LANCZOS4,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    valid = (mx>=0)&(mx<ws)&(my>=0)&(my<hs)&~behind
    alpha = (valid*255).astype(np.uint8); rect[~valid] = 0
    return rect, alpha

def _back_project_pixel(px, py, K, dist, rvec, tvec, centroid, elev):
    pts = np.array([[[float(px), float(py)]]], dtype=np.float64)
    p_norm = cv2.undistortPoints(pts, K.astype(np.float64),
                                dist.ravel().astype(np.float64))
    d_cam = np.array([p_norm[0,0,0], p_norm[0,0,1], 1.0])
    R, _ = cv2.Rodrigues(rvec)
    d_world = R.T @ d_cam
    C = (-R.T @ tvec.ravel())
    z_target = elev - centroid[2]
    if abs(d_world[2]) < 1e-10: return None
    s = (z_target - C[2]) / d_world[2]
    if s <= 0: return None
    return C + s * d_world + centroid

# ── GCP Warp (GDAL: TPS / Poly) ──
def _make_gdal_gcps(df):
    gcps = []
    for _, row in df.iterrows():
        g = gdal.GCP()
        g.GCPX = float(row["Real_X"]); g.GCPY = float(row["Real_Y"])
        g.GCPZ = float(row.get("Real_Z", 0))
        g.GCPPixel = float(row["Pixel_X"]); g.GCPLine = float(row["Pixel_Y"])
        g.Id = str(row.get("GCP_ID", ""))
        gcps.append(g)
    epsg = int(df["EPSG"].iloc[0]) if "EPSG" in df.columns else 0
    return gcps, epsg

def _gdal_warp(src_path, dst_path, gcps, srs_wkt, epsg,
               method, pxsize, te, lock=None):
    def _do():
        gdal.UseExceptions()
        src = gdal.Open(src_path, gdal.GA_ReadOnly)
        if src is None: raise RuntimeError(f"Cannot open {src_path}")
        mem = gdal.GetDriverByName("MEM").CreateCopy("", src, 0); src = None
        mem.SetGCPs(gcps, srs_wkt)
        kw = dict(dstSRS=f"EPSG:{epsg}", resampleAlg=gdal.GRA_Lanczos,
                  outputType=gdal.GDT_Byte, xRes=pxsize, yRes=pxsize,
                  outputBounds=te,
                  creationOptions=["ALPHA=YES","COMPRESS=DEFLATE","TILED=YES"],
                  multithread=True)
        if method == "tps": kw["tps"] = True
        elif method == "poly1": kw["polynomialOrder"] = 1
        elif method == "poly2": kw["polynomialOrder"] = 2
        r = gdal.Warp(dst_path, mem, options=gdal.WarpOptions(**kw))
        if r is None: raise RuntimeError("gdal.Warp returned None")
        r.FlushCache(); r = None; mem = None
    if lock:
        with lock: _do()
    else: _do()

def _write_geotiff(path, bgr, alpha, gt, epsg, lock=None):
    h, w = bgr.shape[:2]
    def _do():
        drv = gdal.GetDriverByName("GTiff")
        ds = drv.Create(path, w, h, 4, gdal.GDT_Byte,
                        ["COMPRESS=DEFLATE","TILED=YES","ALPHA=YES"])
        if ds is None: return False
        ds.SetGeoTransform(gt)
        srs = osr.SpatialReference(); srs.ImportFromEPSG(int(epsg))
        ds.SetProjection(srs.ExportToWkt())
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        for i in range(3): ds.GetRasterBand(i+1).WriteArray(rgb[:,:,i])
        ds.GetRasterBand(4).WriteArray(alpha)
        ds.FlushCache(); ds = None; return True
    if lock:
        with lock: return _do()
    return _do()

# ── Corner-to-world helpers (used by all methods) ──
def _corners_to_extent_affine(df, source_corners):
    px = df[["Pixel_X","Pixel_Y"]].values.astype(np.float64)
    utm = df[["Real_X","Real_Y"]].values.astype(np.float64)
    A = np.column_stack([px, np.ones(len(px))])
    cx, *_ = np.linalg.lstsq(A, utm[:,0], rcond=None)
    cy, *_ = np.linalg.lstsq(A, utm[:,1], rcond=None)
    pts = source_corners.reshape(-1, 2).astype(np.float64)
    At = np.column_stack([pts, np.ones(len(pts))])
    wx, wy = At @ cx, At @ cy
    return float(wx.min()), float(wy.min()), float(wx.max()), float(wy.max())

def _apply_crop_to_extent(extent, crop_fracs):
    x0, y0, x1, y1 = extent
    cs, cn, ce, cw = crop_fracs
    dx, dy = x1 - x0, y1 - y0
    x0 -= dx * cw; x1 -= dx * ce
    y0 += dy * cs; y1 += dy * cn
    return x0, y0, x1, y1

# ── LOO Cross-Validation ──
def _loo_homography(df):
    px = df[["Pixel_X","Pixel_Y"]].values.astype(np.float32)
    utm = df[["Real_X","Real_Y"]].values.astype(np.float32)
    n = len(df); errs = np.full(n, np.nan)
    for i in range(n):
        mask = np.ones(n, bool); mask[i] = False
        H = _homo_compute(px[mask], utm[mask])
        if H is None: continue
        pred = cv2.perspectiveTransform(np.array([[px[i]]], np.float32), H).reshape(2)
        errs[i] = np.linalg.norm(pred - utm[i])
    return errs

def _loo_projection(df, K, dist, elev):
    px = df[["Pixel_X", "Pixel_Y"]].values.astype(np.float64)
    wp = df[["Real_X", "Real_Y", "Real_Z"]].values.astype(np.float64)
    n = len(df)
    errs = np.full(n, np.nan)

    for i in range(n):
        mask = np.ones(n, bool)
        mask[i] = False
        try:
            rv, tv, cen, _ = _solve_extrinsics(px[mask], wp[mask], K, dist)

            # use the held-out point's own Z, not the single GUI elevation
            pred = _back_project_pixel(
                px[i, 0], px[i, 1],
                K, dist, rv, tv, cen,
                wp[i, 2]
            )

            if pred is not None:
                errs[i] = np.linalg.norm(pred[:2] - wp[i, :2])
        except:
            pass

    return errs

def _loo_tps_poly(df, method):
    px = df[["Pixel_X","Pixel_Y"]].values.astype(np.float64)
    utm = df[["Real_X","Real_Y"]].values.astype(np.float64)
    n = len(df); errs = np.full(n, np.nan)
    for i in range(n):
        mask = np.ones(n, bool); mask[i] = False
        tr_px, tr_utm = px[mask], utm[mask]
        try:
            if method == "tps":
                from scipy.interpolate import RBFInterpolator
                rx = RBFInterpolator(tr_px, tr_utm[:,0], kernel="thin_plate_spline")
                ry = RBFInterpolator(tr_px, tr_utm[:,1], kernel="thin_plate_spline")
                pred = np.array([rx(px[[i]])[0], ry(px[[i]])[0]])
            else:
                order = 1 if method == "poly1" else 2
                x, y = tr_px[:,0], tr_px[:,1]
                F = np.c_[np.ones(len(x)), x, y] if order == 1 else np.c_[np.ones(len(x)), x, y, x*y, x**2, y**2]
                xt, yt = px[i,0], px[i,1]
                Ft = np.array([[1, xt, yt]]) if order == 1 else np.array([[1, xt, yt, xt*yt, xt**2, yt**2]])
                cx, *_ = np.linalg.lstsq(F, tr_utm[:,0], rcond=None)
                cy, *_ = np.linalg.lstsq(F, tr_utm[:,1], rcond=None)
                pred = np.array([(Ft @ cx)[0], (Ft @ cy)[0]])
            errs[i] = np.linalg.norm(pred - utm[i])
        except: pass
    return errs

def compute_loo(df, method_key, K=None, dist=None, elev=0.0):
    if method_key == "homo": return _loo_homography(df)
    elif method_key == "proj": return _loo_projection(df, K, dist, elev)
    else: return _loo_tps_poly(df, method_key)

# ── Simulated Annealing ──
def run_sa_optimisation(df, method_key, K=None, dist=None, elev=0.0,
                        n_keep=None, max_iter=100000, temp=10000,
                        cooling=0.9999, log_fn=print):
    n = len(df)
    if n_keep is None: n_keep = max(4, n - 2)
    n_keep = min(n_keep, n)
    if n_keep < 4: log_fn("[SA] Need >= 4 GCPs."); return np.arange(n), np.inf
    idx_all = np.arange(n)
    current_sub = np.sort(np.random.choice(idx_all, n_keep, replace=False))
    def _cost(sub):
        errs = compute_loo(df.iloc[sub].reset_index(drop=True), method_key, K, dist, elev)
        valid = errs[~np.isnan(errs)]
        return valid.mean() if len(valid) > 0 else 1e9
    current_cost = _cost(current_sub)
    best_sub, best_cost = current_sub.copy(), current_cost
    log_fn(f"[SA] Starting: {n_keep}/{n} GCPs, cost={current_cost:.2f} m")
    t0 = time.time()
    for it in range(max_iter):
        new_sub = current_sub.copy()
        outside = np.setdiff1d(idx_all, new_sub)
        if len(outside) == 0: break
        new_sub[random.randrange(len(new_sub))] = random.choice(list(outside))
        new_sub = np.sort(new_sub)
        new_cost = _cost(new_sub)
        if new_cost < best_cost: best_sub, best_cost = new_sub.copy(), new_cost
        dc = new_cost - current_cost
        if dc < 0 or random.random() < np.exp(-dc / max(temp, 1e-10)):
            current_sub, current_cost = new_sub, new_cost
        temp *= cooling
        if (it+1) % 20000 == 0:
            log_fn(f"  [SA] iter {it+1}  temp={temp:.2f}  best={best_cost:.2f} m")
    log_fn(f"[SA] Done in {time.time()-t0:.1f}s — best: {best_cost:.2f} m")
    ids = df["GCP_ID"].values if "GCP_ID" in df.columns else np.arange(n)
    excluded = np.setdiff1d(idx_all, best_sub)
    if len(excluded) > 0: log_fn(f"[SA] Excluded: {[str(ids[i]) for i in excluded]}")
    return best_sub, best_cost

# ── Interactive AOI Selector ──
class InteractiveAOISelector(ctk.CTkToplevel):
    _MZ = 0.05; _XZ = 20.0; _ZS = 1.15
    def __init__(self, master, preview_rgb, on_done):
        super().__init__(master)
        self.title("Select AOI — draw a rectangle, then Confirm")
        self.resizable(True, True)
        try: self.after(200, lambda: self.iconphoto(False, tk.PhotoImage(file=resource_path("launch_logo.png"))))
        except: pass
        self._rgb = preview_rgb; self._cb = on_done
        ih, iw = preview_rgb.shape[:2]; self._iw, self._ih = iw, ih
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        iz = min(0.85*sw/iw, 0.80*sh/ih, 1.0)
        self._z = iz; self._ox = 0.0; self._oy = 0.0
        self._sel = None; self._ds = None
        ww = min(int(iw*iz)+20, int(sw*0.9))
        wh = min(int(ih*iz)+80, int(sh*0.88))
        self.geometry(f"{ww}x{wh}+{(sw-ww)//2}+{max(0,(sh-wh)//2)}")
        tk.Label(self, text="L-drag: select | Wheel: zoom | Mid-drag: pan | R-click: clear",
                 bg="#1a1a2e", fg="#aac", font=("Consolas", 9)).pack(side="top", fill="x")
        self._c = tk.Canvas(self, bg="#0a0a14", cursor="crosshair", highlightthickness=0)
        self._c.pack(side="top", fill="both", expand=True)
        bf = ctk.CTkFrame(self); bf.pack(side="bottom", fill="x", padx=4, pady=4)
        self._sl = ctk.CTkLabel(bf, text="No selection", text_color="gray")
        self._sl.pack(side="left", padx=10)
        ctk.CTkButton(bf, text="Clear", width=80, fg_color="#555", command=self._clr).pack(side="right", padx=6)
        ctk.CTkButton(bf, text="Confirm", width=120, fg_color="#2E7D32", command=self._cfm).pack(side="right", padx=6)
        self._pil = Image.fromarray(preview_rgb)
        c = self._c
        c.bind("<ButtonPress-1>", self._lp); c.bind("<B1-Motion>", self._ld); c.bind("<ButtonRelease-1>", self._lr)
        c.bind("<ButtonPress-2>", self._mp); c.bind("<B2-Motion>", self._mm)
        c.bind("<ButtonPress-3>", lambda e: self._clr())
        for ev in ("<MouseWheel>","<Button-4>","<Button-5>"): c.bind(ev, self._wh)
        c.bind("<Configure>", lambda e: self._dr())
        self.bind("<Return>", lambda e: self._cfm()); self.bind("<Escape>", lambda e: self.destroy())
        self._dr(); self.focus_set()
    def _i2c(self, x, y): return (x-self._ox)*self._z, (y-self._oy)*self._z
    def _c2i(self, x, y): return x/self._z+self._ox, y/self._z+self._oy
    def _dr(self):
        c = self._c; cw, ch = c.winfo_width(), c.winfo_height()
        if cw < 2: return
        a, b = self._c2i(0, 0); d, e = self._c2i(cw, ch)
        sa, sb = max(0, int(a)), max(0, int(b)); sd, se = min(self._iw, int(d)+1), min(self._ih, int(e)+1)
        if sd <= sa or se <= sb: return
        cr = self._pil.crop((sa, sb, sd, se))
        dw, dh = max(1, int((sd-sa)*self._z)), max(1, int((se-sb)*self._z))
        r = cr.resize((dw, dh), Image.NEAREST if self._z >= 2 else Image.BILINEAR)
        cx, cy = self._i2c(sa, sb); self._tk = ImageTk.PhotoImage(r); c.delete("all")
        c.create_image(int(cx), int(cy), anchor="nw", image=self._tk)
        ax, ay = cw-36, 48; nr = 18
        c.create_oval(ax-nr, ay-nr, ax+nr, ay+nr, fill="#1a1a2e", outline="#aac")
        c.create_polygon(ax, ay-nr+4, ax-5, ay+4, ax+5, ay+4, fill="#e0e0ff", outline="#aac")
        c.create_text(ax, ay-nr-8, text="N", fill="#e0e0ff", font=("Arial", 9, "bold"))
        if self._sel:
            a2, b2, d2, e2 = self._sel; ca, cb2 = self._i2c(a2, b2); cd2, ce2 = self._i2c(d2, e2)
            c.create_rectangle(ca, cb2, cd2, ce2, outline="#FFD700", width=2, fill="#FFD700", stipple="gray25")
            for hx, hy in [(ca,cb2),(cd2,cb2),(cd2,ce2),(ca,ce2)]:
                c.create_oval(hx-4, hy-4, hx+4, hy+4, fill="#FFD700", outline="#000")
    def _lp(self, e): self._ds = self._c2i(e.x, e.y); self._sel = None; self._dr()
    def _ld(self, e):
        if not self._ds: return
        ix, iy = self._c2i(e.x, e.y); sx, sy = self._ds
        self._sel = (min(sx,ix), min(sy,iy), max(sx,ix), max(sy,iy)); self._dr(); self._ul()
    def _lr(self, e):
        self._ds = None
        if self._sel:
            a, b, d, e2 = self._sel
            if abs(d-a) < 2 or abs(e2-b) < 2: self._sel = None
        self._ul()
    _pl = None
    def _mp(self, e): self._pl = (e.x, e.y)
    def _mm(self, e):
        if not self._pl: return
        self._ox -= (e.x-self._pl[0])/self._z; self._oy -= (e.y-self._pl[1])/self._z
        self._pl = (e.x, e.y); self._dr()
    def _wh(self, e):
        f = self._ZS if (e.num == 4 or e.delta > 0) else 1/self._ZS
        nz = max(self._MZ, min(self._XZ, self._z*f))
        if nz == self._z: return
        ix, iy = self._c2i(e.x, e.y); self._z = nz
        self._ox = ix - e.x/nz; self._oy = iy - e.y/nz; self._dr()
    def _clr(self): self._sel = None; self._dr(); self._sl.configure(text="No selection", text_color="gray")
    def _ul(self):
        if not self._sel: self._sl.configure(text="No selection", text_color="gray"); return
        a, b, d, e = self._sel
        self._sl.configure(text=f"({a:.0f},{b:.0f})->({d:.0f},{e:.0f}) [{d-a:.0f}x{e-b:.0f}px]", text_color="#FFD700")
    def _cfm(self):
        if not self._sel: messagebox.showwarning("", "Draw a rectangle first.", parent=self); return
        a, b, d, e = self._sel
        a = max(0, min(a, self._iw-1)); b = max(0, min(b, self._ih-1))
        d = max(0, min(d, self._iw-1)); e = max(0, min(e, self._ih-1))
        print(f"[AOI] Preview pixels: ({a:.0f},{b:.0f}) -> ({d:.0f},{e:.0f})")
        self._cb(a, b, d, e); self.destroy()


# %%
#  Main GUI


class GeoReferenceModule(ctk.CTkToplevel):

    def __init__(self, master=None, **kw):
        super().__init__(master=master, **kw)
        self.title("Georeferencing Tool")
        fit_geometry(self, 1250, 900, resizable=True)
        try: self.iconbitmap(resource_path("launch_logo.ico"))
        except: pass
        self._img_path = ""
        self._method_key = "homo"
        self._H = None; self._homo_epsg = None; self._homo_path = ""
        self._K = None; self._dist = None; self._cal_img_size = None; self._cal_path = ""
        self._gcp_df = None; self._gcp_epsg = None; self._elev = 0.0; self._gcp_path = ""
        self._rvec = None; self._tvec = None; self._centroid = None
        self._grid_x = self._grid_y = self._grid_z = None
        self._preview_bgr = None; self._preview_gt = None; self._preview_Hf = None
        self._aoi = None; self._aoi_preview_shape = None; self._excluded_gcps = []; self._scale = 4.0
        self._corner_var = tk.StringVar(value="Full Image")
        self.image_list = []; self.output_folder = ""; self.input_folder = ""
        self.user_epsg = None; self._lock = threading.Lock(); self.executor = None
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._close)

    def _close(self):
        if self.executor: self.executor.shutdown(wait=False, cancel_futures=True)
        restore_console(getattr(self, "_console_redir", None)); self.destroy()

    @staticmethod
    def _eta(t0, f):
        if f <= 0: return "ETA: ~"
        r = (time.time()-t0)*(1/f-1)
        if r >= 3600: return f"ETA: {r/3600:.1f}h"
        if r >= 60: return f"ETA: {r/60:.1f}min"
        return f"ETA: {int(r)}s"

    def _show(self, src, lbl):
        img = Image.fromarray(src) if isinstance(src, np.ndarray) else Image.open(src)
        img.thumbnail((500, 380))
        ci = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
        lbl.configure(image=ci, text=""); lbl.image = ci

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1); self.grid_rowconfigure(0, weight=1)
        for r in range(1, 8): self.grid_rowconfigure(r, weight=0)

        pf = ctk.CTkFrame(self, fg_color="black")
        pf.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self._orig_lbl = self._mkp(pf, "Original")
        self._init_lbl = self._mkp(pf, "Initial Georef")
        self._sec_lbl  = self._mkp(pf, "Secondary Georef")

        # Method + inputs
        r1 = ctk.CTkFrame(self); r1.grid(row=2, column=0, sticky="ew", padx=5, pady=2)
        ctk.CTkLabel(r1, text="Method:", font=("Arial", 12, "bold")).pack(side="left", padx=(8,2))
        self._method_var = ctk.StringVar(value=METHODS[0])
        ctk.CTkOptionMenu(r1, values=METHODS, variable=self._method_var,
            command=self._on_method_change, width=180,
            fg_color="white", text_color="black", button_color="#d0d0d0",
            button_hover_color="#a0a0a0", dropdown_fg_color="white",
            dropdown_text_color="black", dropdown_hover_color="#d0d0d0").pack(side="left", padx=4)
        ctk.CTkButton(r1, text="Browse Image", command=self._browse_img).pack(side="left", padx=6)
        self._img_lbl = ctk.CTkLabel(r1, text="No image", text_color="gray"); self._img_lbl.pack(side="left", padx=4)

        self._inp_homo = ctk.CTkFrame(r1)
        ctk.CTkButton(self._inp_homo, text="Load Homography (.txt)", command=self._load_homo).pack(side="left", padx=4)
        self._homo_lbl = ctk.CTkLabel(self._inp_homo, text="\u2014", text_color="gray"); self._homo_lbl.pack(side="left", padx=4)

        self._inp_gcp = ctk.CTkFrame(r1)
        ctk.CTkButton(self._inp_gcp, text="Load GCP CSV", command=self._load_gcp, fg_color="#2E7D32").pack(side="left", padx=4)
        self._gcp_lbl = ctk.CTkLabel(self._inp_gcp, text="\u2014", text_color="gray"); self._gcp_lbl.pack(side="left", padx=4)

        self._inp_proj = ctk.CTkFrame(r1)
        ctk.CTkButton(self._inp_proj, text="Load Calibration (.pkl)", command=self._load_cal).pack(side="left", padx=4)
        self._cal_lbl = ctk.CTkLabel(self._inp_proj, text="\u2014", text_color="gray"); self._cal_lbl.pack(side="left", padx=4)
        ctk.CTkLabel(self._inp_proj, text="Elev (m):").pack(side="left", padx=(8,2))
        self._elev_ent = ctk.CTkEntry(self._inp_proj, width=60, placeholder_text="0.0"); self._elev_ent.pack(side="left", padx=2)

        # Corner + crop (ALL methods)
        r2 = ctk.CTkFrame(self); r2.grid(row=3, column=0, sticky="ew", padx=5, pady=2)
        self._corner_tools = ctk.CTkFrame(r2, fg_color="transparent"); self._corner_tools.pack(side="left", padx=4)
        ctk.CTkLabel(self._corner_tools, text="Corner:").pack(side="left", padx=(4,2))
        ctk.CTkOptionMenu(self._corner_tools, values=["Full Image","Bottom Left","Bottom Right","Top Left","Top Right","Manual"],
            variable=self._corner_var, command=self._on_corner_change, width=115,
            fg_color="white", text_color="black", button_color="#d0d0d0",
            button_hover_color="#a0a0a0", dropdown_fg_color="white",
            dropdown_text_color="black", dropdown_hover_color="#d0d0d0").pack(side="left", padx=2)
        self._manual_corner_ent = ctk.CTkEntry(self._corner_tools, width=190, placeholder_text="x1,y1,x2,y2,x3,y3,x4,y4")
        self._crop_entries = {}
        for tag in ["S","N","E","W"]:
            ctk.CTkLabel(self._corner_tools, text=tag).pack(side="left", padx=(8 if tag=="S" else 4, 1))
            ent = ctk.CTkEntry(self._corner_tools, width=45); ent.pack(side="left", padx=1); ent.insert(0,"0")
            self._crop_entries[tag] = ent

        # Initial georef + AOI
        r2b = ctk.CTkFrame(self); r2b.grid(row=4, column=0, sticky="ew", padx=5, pady=2)
        ctk.CTkButton(r2b, text="Initial Georeferencing", command=self._initial_georef, fg_color="#0F52BA", hover_color="#2A6BD1",
                      font=("Arial", 12, "bold")).pack(side="left", padx=6)
        ctk.CTkButton(r2b, text="Select AOI Interactively", command=self._select_aoi, fg_color="#2E7D32").pack(side="left", padx=6)
        ctk.CTkLabel(r2b, text="or manual (x1,y1,x2,y2):").pack(side="left", padx=(8,2))
        self._aoi_ent = ctk.CTkEntry(r2b, width=200, placeholder_text="preview pixel coords"); self._aoi_ent.pack(side="left", padx=2)
        ctk.CTkButton(r2b, text="Set", command=self._set_manual_aoi, width=50, fg_color="#0F52BA", hover_color="#2A6BD1").pack(side="left", padx=2)
        self._aoi_status = ctk.CTkLabel(r2b, text="AOI: not set", text_color="gray"); self._aoi_status.pack(side="left", padx=8)

        # Accuracy + Secondary
        r3 = ctk.CTkFrame(self); r3.grid(row=5, column=0, sticky="ew", padx=5, pady=2)
        ctk.CTkButton(r3, text="Compute Accuracy", command=self._compute_accuracy, fg_color="#6693F5", hover_color="#7DA5F7").pack(side="left", padx=6)
        ctk.CTkButton(r3, text="Optimise GCPs (SA)", command=self._run_sa, fg_color="#37474F").pack(side="left", padx=4)
        ctk.CTkLabel(r3, text="Exclude GCPs:").pack(side="left", padx=(10,2))
        self._excl_ent = ctk.CTkEntry(r3, width=140, placeholder_text="e.g. 5,7,11"); self._excl_ent.pack(side="left", padx=2)
        ctk.CTkLabel(r3, text="|", text_color="gray").pack(side="left", padx=6)
        ctk.CTkLabel(r3, text="Scale:").pack(side="left", padx=(4,2))
        self._scale_ent = ctk.CTkEntry(r3, width=55); self._scale_ent.insert(0, "4"); self._scale_ent.pack(side="left", padx=2)
        self._rec_scale_lbl = ctk.CTkLabel(r3, text="", text_color="gray", font=("Arial", 10)); self._rec_scale_lbl.pack(side="left", padx=4)
        ctk.CTkButton(r3, text="Secondary Georeferencing", command=self._secondary_georef, fg_color="#0F52BA",
                      font=("Arial", 12, "bold")).pack(side="left", padx=8)

        # Batch
        r4 = ctk.CTkFrame(self); r4.grid(row=6, column=0, sticky="ew", padx=5, pady=2)
        ctk.CTkButton(r4, text="Browse Input Folder", command=self._browse_in).pack(side="left", padx=4)
        self._in_lbl = ctk.CTkLabel(r4, text="\u2014", text_color="gray"); self._in_lbl.pack(side="left", padx=4)
        self._subfolder_var = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(r4, text="Include subfolders", variable=self._subfolder_var).pack(side="left", padx=8)
        ctk.CTkButton(r4, text="Browse Output Folder", command=self._browse_out, fg_color="#8C7738", hover_color="#A18A45").pack(side="left", padx=4)
        self._out_lbl = ctk.CTkLabel(r4, text="\u2014", text_color="gray"); self._out_lbl.pack(side="left", padx=4)
        self._epsg_ent = ctk.CTkEntry(r4, width=80, placeholder_text="EPSG"); self._epsg_ent.pack(side="left", padx=4)
        ctk.CTkButton(r4, text="Process All", fg_color="#0F52BA", hover_color="#2A6BD1", command=self._process_all,
                      font=("Arial", 12, "bold")).pack(side="left", padx=6)
        self._prog = ctk.CTkProgressBar(r4, width=160); self._prog.set(0); self._prog.pack(side="left", padx=6)
        self._eta_lbl = ctk.CTkLabel(r4, text="ETA: ~"); self._eta_lbl.pack(side="left", padx=4)


        # Settings row
        r5 = ctk.CTkFrame(self); r5.grid(row=7, column=0, sticky="ew", padx=5, pady=2)
        ctk.CTkButton(r5, text="Save Settings", command=self._save_settings, width=100, fg_color="#4F5D75",hover_color="#61708A").pack(side="left", padx=4)
        ctk.CTkButton(r5, text="Load Settings", command=self._load_settings, width=100, fg_color="#4F5D75",hover_color="#61708A").pack(side="left", padx=4)
        ctk.CTkButton(r5, text="Reset", command=self._reset, width=60, fg_color="#8B0000", hover_color="#B22222").pack(side="left", padx=8)
        
        # Console
        cf = ctk.CTkFrame(self); cf.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        ct = tk.Text(cf, wrap="word", height=10); ct.pack(fill="both", expand=True, padx=5, pady=5)
        self._console_redir = setup_console(ct)
        print("GeoCamPal \u2014 Georeferencing Tool")
        print("="*50)
        print("\nWorkflow:")
        print("  1. Choose georef method + load inputs")
        print("  2. Choose corner preset + crop (all methods)")
        print("  3. Click \u2018Initial Georeferencing\u2019 for preview")
        print("  4. Optionally select AOI for fine-cropping")
        print("  5. Compute Accuracy / Optimise GCPs")
        print("  6. Secondary Georeferencing for final preview")
        print("  7. Process All for batch")
        print("="*50 + "\n")
        self._on_method_change(METHODS[0])

    @staticmethod
    def _mkp(parent, text):
        fr = ctk.CTkFrame(parent, width=370, height=340, fg_color="black")
        fr.pack_propagate(False); fr.pack(side="left", fill="both", expand=True, padx=4, pady=4)
        lbl = ctk.CTkLabel(fr, text=text, fg_color="black"); lbl.pack(fill="both", expand=True)
        return lbl

    def _on_method_change(self, choice):
        self._method_key = _METHOD_KEY[choice]
        for w in (self._inp_homo, self._inp_gcp, self._inp_proj): w.pack_forget()
        if self._method_key == "homo":
            self._inp_homo.pack(side="left", padx=4)
            print(f"\n[Method] Homography selected")
            print(f"  Required: Homography matrix (.txt)")
        elif self._method_key == "proj":
            self._inp_proj.pack(side="left", padx=4); self._inp_gcp.pack(side="left", padx=4)
            print(f"\n[Method] Camera Projection selected")
            print(f"  Required: Calibration (.pkl) + GCP CSV + elevation")
        else:
            self._inp_gcp.pack(side="left", padx=4)
            names = {"tps":"Thin Plate Spline","poly1":"Polynomial Ord-1","poly2":"Polynomial Ord-2"}
            print(f"\n[Method] {names[self._method_key]} selected")
            print(f"  Required: GCP CSV")

    def _on_corner_change(self, *_):
        if self._corner_var.get() == "Manual":
            if not self._manual_corner_ent.winfo_manager(): self._manual_corner_ent.pack(side="left", padx=4)
        else:
            if self._manual_corner_ent.winfo_manager(): self._manual_corner_ent.pack_forget()

    def _get_source_corners(self, h, w):
        choice = self._corner_var.get()
        if choice == "Manual":
            vals = [float(v.strip()) for v in self._manual_corner_ent.get().split(",") if v.strip()]
            if len(vals) != 8: raise ValueError("Manual corners need 8 numbers")
            return np.array(vals, dtype=np.float32).reshape(4, 1, 2)
        lut = {"Bottom Left": [[0,int(h*0.5)],[w-1,int(h*0.5)],[w-1,h-1],[0,h-1]],
               "Bottom Right": [[int(w*0.5),int(h*0.5)],[w-1,int(h*0.5)],[w-1,h-1],[int(w*0.5),h-1]],
               "Top Left": [[0,0],[int(w*0.5),0],[int(w*0.5),int(h*0.5)],[0,int(h*0.5)]],
               "Top Right": [[int(w*0.5),0],[w-1,0],[w-1,int(h*0.5)],[int(w*0.5),int(h*0.5)]]}
        pts = lut.get(choice, [[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
        return np.array(pts, dtype=np.float32).reshape(4, 1, 2)

    def _get_crop_fracs(self):
        return tuple(float(self._crop_entries[t].get().strip() or 0) for t in ["S","N","E","W"])

    def _corners_to_world_extent(self, source_corners, h, w):
        mk = self._method_key
        if mk == "homo":
            tc = cv2.perspectiveTransform(source_corners, self._H)
            mn = tc.min(axis=(0,1)); mx = tc.max(axis=(0,1))
            return float(mn[0]), float(mn[1]), float(mx[0]), float(mx[1])
        elif mk == "proj":
            K = self._get_K_for((w, h)); elev = self._get_elev()
            wpts = []
            for pt in source_corners.reshape(-1, 2):
                wp = _back_project_pixel(float(pt[0]), float(pt[1]), K, self._dist, self._rvec, self._tvec, self._centroid, elev)
                if wp is not None: wpts.append(wp[:2])
            if len(wpts) < 2: return None
            wpts = np.array(wpts)
            return wpts[:,0].min(), wpts[:,1].min(), wpts[:,0].max(), wpts[:,1].max()
        else:
            df = self._get_working_df()
            if df is None or len(df) < 3: return None
            return _corners_to_extent_affine(df, source_corners)

    def _browse_img(self):
        p = filedialog.askopenfilename(parent=self,filetypes=[("Images","*.jpg *.jpeg *.png *.bmp *.tif *.tiff")])
        if p:
            self._img_path = p; self._img_lbl.configure(text=os.path.basename(p), text_color="#81C784")
            self._show(p, self._orig_lbl)

    def _load_homo(self, filepath=None):
        p = filepath or filedialog.askopenfilename(parent=self,filetypes=[("Text","*.txt")])
        if not p: return
        if not os.path.exists(p):
            print(f"[Warning] Homography file not found: {p}"); return
        try:
            self._homo_epsg = None
            with open(p) as f:
                line = f.readline()
                if "#" in line and "EPSG:" in line:
                    m = re.search(r"EPSG:(\d+)", line)
                    if m: self._homo_epsg = int(m.group(1))
            self._H = np.loadtxt(p).reshape(3, 3)
            self._homo_path = p
            self._homo_lbl.configure(text=os.path.basename(p), text_color="#81C784")
            if self._homo_epsg:
                if not self._epsg_ent.get().strip(): self._epsg_ent.insert(0, str(self._homo_epsg))
                print(f"[CRS] EPSG:{self._homo_epsg} from homography")
        except Exception as e: messagebox.showerror("Error", f"Bad homography: {e}", parent=self)

    def _load_gcp(self, filepath=None):
        p = filepath or filedialog.askopenfilename(parent=self,filetypes=[("CSV","*.csv"),("All","*.*")])
        if not p: return
        if not os.path.exists(p):
            print(f"[Warning] GCP file not found: {p}"); return
        try:
            df = read_gcp_csv(p)
            for c in ("Pixel_X","Pixel_Y","Real_X","Real_Y"):
                if c not in df.columns: messagebox.showerror("Error", f"Missing \u2018{c}\u2019", parent=self); return
            self._gcp_df = df; self._gcp_path = p
            self._gcp_epsg = int(df["EPSG"].iloc[0]) if "EPSG" in df.columns and df["EPSG"].iloc[0] > 0 else None
            gsd = _estimate_gsd(df)
            if self._gcp_epsg:
                if not self._epsg_ent.get().strip(): self._epsg_ent.insert(0, str(self._gcp_epsg))
                print(f"[CRS] EPSG:{self._gcp_epsg} detected from GCP CSV")
            elif not self._epsg_ent.get().strip():
                # Try auto-detection from coordinate ranges
                guessed, msg = _guess_epsg(df)
                if guessed:
                    self._gcp_epsg = guessed
                    self._epsg_ent.insert(0, str(guessed))
                    print(f"[CRS] Auto-detected EPSG:{guessed} — {msg}")
                else:
                    print(f"[CRS] {msg}")
            mz = df["Real_Z"].mean()
            if not self._elev_ent.get().strip(): self._elev_ent.insert(0, f"{mz:.1f}")
            self._gcp_lbl.configure(text=f"{len(df)} GCPs (GSD~{gsd:.3f})", text_color="#81C784")
            print(f"[GCP] {len(df)} GCPs, EPSG:{self._gcp_epsg}, GSD~{gsd:.4f}")
        except Exception as e: messagebox.showerror("GCP Error", str(e), parent=self)

    def _load_cal(self, filepath=None):
        p = filepath or filedialog.askopenfilename(parent=self,filetypes=[("Pickle","*.pkl"),("All","*.*")])
        if not p: return
        if not os.path.exists(p):
            print(f"[Warning] Calibration file not found: {p}"); return
        try:
            with open(p, "rb") as f: d = pickle.load(f)
            self._K = d["camera_matrix"].astype(np.float64)
            self._dist = d["dist_coeff"].ravel().astype(np.float64)
            self._cal_img_size = d.get("image_size")
            self._cal_path = p
            self._cal_lbl.configure(text=f"{os.path.basename(p)} (fx={self._K[0,0]:.0f})", text_color="#81C784")
        except Exception as e: messagebox.showerror("Calibration Error", str(e), parent=self)

    def _get_K_for(self, wh):
        K = self._K.copy()
        if self._cal_img_size:
            cal = self._cal_img_size
            if isinstance(cal, (list, tuple)) and len(cal) == 2:
                # calibration image_size is stored as (width, height)
                cwh = (int(cal[0]), int(cal[1]))
                if cwh != wh:
                    K = _scale_K(K, cwh, wh)
            return K
        return K

    def _get_working_df(self):
        if self._gcp_df is None: return None
        df = self._gcp_df.copy()
        excl = self._excl_ent.get().strip()
        if excl:
            try:
                nums = [int(x.strip()) for x in excl.split(",") if x.strip()]
                df = df[~df["GCP_ID"].isin([f"GCP_{n}" for n in nums])].reset_index(drop=True)
            except: pass
        return df

    def _get_elev(self):
        try: return float(self._elev_ent.get() or 0)
        except: return 0.0

    def _get_scale(self):
        try: return float(self._scale_ent.get() or 4)
        except: return 4.0

    def _resolve_epsg(self):
        try: self.user_epsg = int(self._epsg_ent.get()); return True
        except: pass
        if self._gcp_epsg: self.user_epsg = self._gcp_epsg; return True
        if self._homo_epsg: self.user_epsg = self._homo_epsg; return True
        messagebox.showerror("Error",
            "No EPSG code found.\n\n"
            "The EPSG could not be auto-detected from the GCP CSV or homography file. "
            "Please enter the EPSG code manually in the EPSG field.\n\n"
            "Tip: include an 'EPSG' column in your GCP CSV, or add '# EPSG:XXXXX' "
            "as the first line of your homography .txt file.", parent=self); return False

    # ── Initial Georeferencing ──
    def _initial_georef(self):
        if not self._img_path: messagebox.showerror("Error", "Browse an image first.", parent=self); return
        img = cv2.imread(self._img_path)
        if img is None: messagebox.showerror("Error", "Cannot read image.", parent=self); return
        self._show(self._img_path, self._orig_lbl)
        mk = self._method_key; scale = self._get_scale()
        try:
            hi, wi = img.shape[:2]
            source_corners = self._get_source_corners(hi, wi)
            crop_fracs = self._get_crop_fracs()

            if mk == "homo":
                if self._H is None: messagebox.showerror("Error", "Load homography.", parent=self); return
                self._aoi = None; self._aoi_preview_shape = None; self._aoi_ent.delete(0, "end")
                self._aoi_status.configure(text="AOI: not set", text_color="gray")
                warped, Hf, gt = _homo_warp_windowed(img, self._H, scale, source_corners, crop_fracs)
                self._preview_bgr = warped; self._preview_gt = gt; self._preview_Hf = Hf
                tc = cv2.perspectiveTransform(source_corners, self._H)
                mn = tc.min(axis=(0,1)); mx = tc.max(axis=(0,1))
                ns = max(wi/(mx[0]-mn[0]), hi/(mx[1]-mn[1]))
                self._rec_scale_lbl.configure(text=f"(recommended ~{ns:.1f})")

            elif mk == "proj":
                if self._K is None: messagebox.showerror("Error", "Load calibration.", parent=self); return
                df = self._get_working_df()
                if df is None or len(df) < 4: messagebox.showerror("Error", "Need >= 4 GCPs.", parent=self); return
                K = self._get_K_for((wi, hi)); elev = self._get_elev()
                px = df[["Pixel_X","Pixel_Y"]].values; wp = df[["Real_X","Real_Y","Real_Z"]].values
                rv, tv, cen, errs = _solve_extrinsics(px, wp, K, self._dist)
                self._rvec, self._tvec, self._centroid = rv, tv, cen
                print(f"[Solve] RMS reproj: {np.sqrt((errs**2).mean()):.1f} px")
                extent = self._corners_to_world_extent(source_corners, hi, wi)
                if extent:
                    x0, y0, x1, y1 = _apply_crop_to_extent(extent, crop_fracs)
                else:
                    ex, ey = df["Real_X"].values, df["Real_Y"].values; mg = 0.20
                    x0, y0 = ex.min()-(ex.max()-ex.min())*mg, ey.min()-(ey.max()-ey.min())*mg
                    x1, y1 = ex.max()+(ex.max()-ex.min())*mg, ey.max()+(ey.max()-ey.min())*mg
                gsd = _estimate_gsd(df)
                gx, gy, gz = _build_grid(x0, x1, y0, y1, gsd, elev)
                self._grid_x, self._grid_y, self._grid_z = gx, gy, gz
                rect, alpha = _rectify_image(img, K, self._dist, rv, tv, cen, gx, gy, gz)
                self._preview_bgr = rect
                self._preview_gt = (float(gx[0,0]), gsd, 0, float(gy[0,0]), 0, -gsd)
                self._rec_scale_lbl.configure(text=f"(GSD={gsd:.4f} m/px)")

            else:  # tps, poly1, poly2
                df = self._get_working_df()
                if df is None or len(df) < 4: messagebox.showerror("Error", "Need >= 4 GCPs.", parent=self); return
                if mk == "poly2" and len(df) < 6: messagebox.showerror("Error", "Poly-2 needs >= 6 GCPs.", parent=self); return
                if not self._resolve_epsg(): return
                extent = self._corners_to_world_extent(source_corners, hi, wi)
                if extent:
                    x0, y0, x1, y1 = _apply_crop_to_extent(extent, crop_fracs)
                    te = [x0, y0, x1, y1]
                else:
                    ex, ey = df["Real_X"].values, df["Real_Y"].values
                    mg = max(50, (ex.max()-ex.min())*0.05)
                    te = [ex.min()-mg, ey.min()-mg, ex.max()+mg, ey.max()+mg]
                gcps, epsg = _make_gdal_gcps(df)
                srs = osr.SpatialReference(); srs.ImportFromEPSG(epsg)
                gsd = _estimate_gsd(df)
                import tempfile
                tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False); tmp.close()
                try:
                    _gdal_warp(self._img_path, tmp.name, gcps, srs.ExportToWkt(), epsg, mk, gsd, te)
                    ds = gdal.Open(tmp.name); self._preview_gt = ds.GetGeoTransform()
                    arr = np.zeros((ds.RasterYSize, ds.RasterXSize, 3), np.uint8)
                    for i in range(min(3, ds.RasterCount)): arr[:,:,i] = ds.GetRasterBand(i+1).ReadAsArray()
                    self._preview_bgr = arr[:,:,::-1].copy(); ds = None
                finally:
                    if os.path.exists(tmp.name): os.unlink(tmp.name)
                self._rec_scale_lbl.configure(text=f"(GSD={gsd:.4f} m/px)")

            rgb = cv2.cvtColor(self._preview_bgr, cv2.COLOR_BGR2RGB)
            self._show(rgb, self._init_lbl)
            oh, ow = self._preview_bgr.shape[:2]
            print(f"[Initial] Preview: {ow}x{oh}")
            print(f"[Next] Select AOI or click Secondary Georeferencing")
        except Exception as e:
            messagebox.showerror("Error", str(e), parent=self)
            import traceback; traceback.print_exc()

    # ── AOI ──
    def _select_aoi(self):
        if self._preview_bgr is None: messagebox.showerror("Error", "Run Initial Georeferencing first.", parent=self); return
        rgb = cv2.cvtColor(self._preview_bgr, cv2.COLOR_BGR2RGB)
        def _on_aoi(x1, y1, x2, y2):
            self._aoi = (x1, y1, x2, y2)
            self._aoi_preview_shape = self._preview_bgr.shape[:2]
            self._aoi_ent.delete(0, "end")
            self._aoi_ent.insert(0, f"{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}")
            self._aoi_status.configure(text=f"AOI: {x2-x1:.0f}x{y2-y1:.0f} px", text_color="#81C784")
            self._show_aoi_crop()
        InteractiveAOISelector(self, rgb, _on_aoi)

    def _set_manual_aoi(self):
        try:
            vals = [float(v.strip()) for v in self._aoi_ent.get().split(",")]
            if len(vals) != 4: raise ValueError("Need 4 values")
            self._aoi = tuple(vals)
            self._aoi_preview_shape = self._preview_bgr.shape[:2] if self._preview_bgr is not None else None
            self._aoi_status.configure(text=f"AOI: {vals[2]-vals[0]:.0f}x{vals[3]-vals[1]:.0f} px", text_color="#81C784")
            self._show_aoi_crop()
        except Exception as e: messagebox.showerror("Error", f"Bad AOI: {e}", parent=self)

    def _show_aoi_crop(self):
        if self._preview_bgr is None or self._aoi is None: return
        x1, y1, x2, y2 = [int(v) for v in self._aoi]; h, w = self._preview_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1: return
        self._show(cv2.cvtColor(self._preview_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB), self._init_lbl)

    # ── Accuracy ──
    def _compute_accuracy(self):
        mk = self._method_key
        if mk == "homo" and self._H is None: messagebox.showerror("Error", "Load homography.", parent=self); return
        df = self._get_working_df()
        if df is None or (mk != "homo" and len(df) < 4): messagebox.showerror("Error", "Need GCPs, if testing for Homography use the dedicated Homography module", parent=self); return
        if mk == "proj" and self._K is None: messagebox.showerror("Error", "Load calibration.", parent=self); return
        def _run():
            K = self._K if mk == "proj" else None; d = self._dist if mk == "proj" else None
            if mk == "proj":
                img = cv2.imread(self._img_path)
                if img is not None: K = self._get_K_for((img.shape[1], img.shape[0]))
            errs = compute_loo(df, mk, K, d, self._get_elev())
            ids = df["GCP_ID"].values if "GCP_ID" in df.columns else np.arange(len(df))
            sep = "=" * 55
            print(f"\n{sep}")
            print(f"  LOO Cross-Validation  ({self._method_var.get()})")
            hdr = f"  {'GCP':<12} {'Error (m)':>10}"
            print(f"\n{hdr}")
            tilde = "~" * 24
            print(f"  {tilde}")
            for gid, e in zip(ids, errs):
                flag = "  <-- outlier" if (not np.isnan(e) and e > 50) else ""
                e_str = f"{e:.2f}" if not np.isnan(e) else "n/a"
                print(f"  {str(gid):<12} {e_str:>10}{flag}")
            valid = errs[~np.isnan(errs)]
            if len(valid):
                print(f"  {tilde}")
                print(f"  Mean: {valid.mean():.2f}m  RMS: {np.sqrt((valid**2).mean()):.2f}m")
            print(f"{sep}\n")
        threading.Thread(target=_run, daemon=True).start()

    # ── SA ──
    def _run_sa(self):
        df = self._get_working_df()
        if df is None or len(df) < 5: messagebox.showerror("Error", "Need >= 5 GCPs.", parent=self); return
        mk = self._method_key; K = self._K if mk == "proj" else None; d = self._dist if mk == "proj" else None
        if mk == "proj" and self._img_path:
            img = cv2.imread(self._img_path)
            if img is not None: K = self._get_K_for((img.shape[1], img.shape[0]))
        def _run():
            best_idx, _ = run_sa_optimisation(df, mk, K, d, self._get_elev(), log_fn=print)
            dropped = set(range(len(df))) - set(best_idx.tolist())
            if dropped:
                ids = df["GCP_ID"].values if "GCP_ID" in df.columns else np.arange(len(df))
                excl = ",".join(str(ids[i]).split("_")[-1] for i in dropped)
                self.after(0, lambda: (self._excl_ent.delete(0, "end"), self._excl_ent.insert(0, excl)))
        threading.Thread(target=_run, daemon=True).start()

    # ── Secondary Georeferencing ──
    def _secondary_georef(self):
        if self._preview_bgr is None: messagebox.showerror("Error", "Run Initial Georeferencing first.", parent=self); return
        if not self._img_path: messagebox.showerror("Error", "No image.", parent=self); return
        img = cv2.imread(self._img_path)
        if img is None: messagebox.showerror("Error", "Cannot read image.", parent=self); return
        mk = self._method_key; scale = self._get_scale(); aoi = self._aoi
        try:
            hi, wi = img.shape[:2]
            source_corners = self._get_source_corners(hi, wi)
            crop_fracs = self._get_crop_fracs()

            if mk == "homo":
                if self._H is None: messagebox.showerror("Error", "Load homography.", parent=self); return
                warped, Hf, gt = _homo_warp_windowed(img, self._H, scale, source_corners, crop_fracs)
                if aoi and self._aoi_preview_shape is not None:
                    x1, y1, x2, y2 = [int(v) for v in aoi]
                    oh, ow = self._aoi_preview_shape; nh, nw = warped.shape[:2]
                    sx, sy = nw/ow, nh/oh
                    x1, y1 = max(0, int(x1*sx)), max(0, int(y1*sy))
                    x2, y2 = min(nw, int(x2*sx)), min(nh, int(y2*sy))
                    if x2 <= x1 or y2 <= y1:
                        raise ValueError(f"AOI out of bounds after scaling ({x1},{y1})->({x2},{y2}). Re-select AOI.")
                    warped = warped[y1:y2, x1:x2]
                    gt = (gt[0]+x1*gt[1], gt[1], 0, gt[3]+y1*gt[5], 0, gt[5])
                self._preview_bgr = warped; self._preview_gt = gt; self._preview_Hf = Hf

            elif mk == "proj":
                df = self._get_working_df()
                if df is None or len(df) < 4: messagebox.showerror("Error", "Need >= 4 GCPs.", parent=self); return
                K = self._get_K_for((wi, hi)); elev = self._get_elev()
                rv, tv, cen, errs = _solve_extrinsics(df[["Pixel_X","Pixel_Y"]].values, df[["Real_X","Real_Y","Real_Z"]].values, K, self._dist)
                self._rvec, self._tvec, self._centroid = rv, tv, cen
                if aoi and self._preview_gt:
                    gt = self._preview_gt; x1, y1, x2, y2 = aoi
                    utm_x0, utm_x1 = gt[0]+x1*gt[1], gt[0]+x2*gt[1]
                    utm_y0, utm_y1 = gt[3]+y2*gt[5], gt[3]+y1*gt[5]
                else:
                    extent = self._corners_to_world_extent(source_corners, hi, wi)
                    if extent: utm_x0, utm_y0, utm_x1, utm_y1 = _apply_crop_to_extent(extent, crop_fracs)
                    else:
                        ex, ey = df["Real_X"].values, df["Real_Y"].values; mg = 0.20
                        utm_x0, utm_y0 = ex.min()-(ex.max()-ex.min())*mg, ey.min()-(ey.max()-ey.min())*mg
                        utm_x1, utm_y1 = ex.max()+(ex.max()-ex.min())*mg, ey.max()+(ey.max()-ey.min())*mg
                gsd = _estimate_gsd(df) / (scale/4.0) if scale != 4 else _estimate_gsd(df)
                gx, gy, gz = _build_grid(utm_x0, utm_x1, utm_y0, utm_y1, gsd, elev)
                self._grid_x, self._grid_y, self._grid_z = gx, gy, gz
                rect, alpha = _rectify_image(img, K, self._dist, rv, tv, cen, gx, gy, gz)
                self._preview_bgr = rect; self._preview_gt = (float(gx[0,0]), gsd, 0, float(gy[0,0]), 0, -gsd)

            else:
                df = self._get_working_df()
                if df is None or len(df) < 4: messagebox.showerror("Error", "Need >= 4 GCPs.", parent=self); return
                if not self._resolve_epsg(): return
                gcps, epsg = _make_gdal_gcps(df); srs = osr.SpatialReference(); srs.ImportFromEPSG(epsg)
                gsd = _estimate_gsd(df)
                if aoi and self._preview_gt:
                    gt = self._preview_gt; x1, y1, x2, y2 = aoi
                    te = [gt[0]+x1*gt[1], gt[3]+max(y1,y2)*gt[5], gt[0]+x2*gt[1], gt[3]+min(y1,y2)*gt[5]]
                else:
                    extent = self._corners_to_world_extent(source_corners, hi, wi)
                    if extent:
                        x0, y0, x1, y1 = _apply_crop_to_extent(extent, crop_fracs); te = [x0, y0, x1, y1]
                    else:
                        ex, ey = df["Real_X"].values, df["Real_Y"].values; mg = max(50, (ex.max()-ex.min())*0.05)
                        te = [ex.min()-mg, ey.min()-mg, ex.max()+mg, ey.max()+mg]
                import tempfile; tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False); tmp.close()
                try:
                    _gdal_warp(self._img_path, tmp.name, gcps, srs.ExportToWkt(), epsg, mk, gsd, te)
                    ds = gdal.Open(tmp.name); self._preview_gt = ds.GetGeoTransform()
                    arr = np.zeros((ds.RasterYSize, ds.RasterXSize, 3), np.uint8)
                    for i in range(min(3, ds.RasterCount)): arr[:,:,i] = ds.GetRasterBand(i+1).ReadAsArray()
                    self._preview_bgr = arr[:,:,::-1].copy(); ds = None
                finally:
                    if os.path.exists(tmp.name): os.unlink(tmp.name)
            self._show(cv2.cvtColor(self._preview_bgr, cv2.COLOR_BGR2RGB), self._sec_lbl)
            print(f"[Secondary] Preview: {self._preview_bgr.shape[1]}x{self._preview_bgr.shape[0]}")
        except Exception as e:
            messagebox.showerror("Error", str(e), parent=self); import traceback; traceback.print_exc()

    # ── Batch save single ──
    def _save_single(self, img_path, out_path):
        img = cv2.imread(img_path)
        if img is None: return False
        mk = self._method_key; hi, wi = img.shape[:2]
        source_corners = self._get_source_corners(hi, wi)
        crop_fracs = self._get_crop_fracs()
        if mk == "homo":
            warped, Hf, gt = _homo_warp_windowed(img, self._H, self._get_scale(), source_corners, crop_fracs)
            if self._aoi and self._aoi_preview_shape is not None:
                oh, ow = self._aoi_preview_shape; nh, nw = warped.shape[:2]
                sx, sy = nw/ow, nh/oh; x1, y1, x2, y2 = self._aoi
                x1, y1 = max(0, int(x1*sx)), max(0, int(y1*sy)); x2, y2 = min(nw, int(x2*sx)), min(nh, int(y2*sy))
                if x2 <= x1 or y2 <= y1:
                    raise ValueError(f"AOI out of bounds after scaling ({x1},{y1})->({x2},{y2})")
                warped = warped[y1:y2, x1:x2]; gt = (gt[0]+x1*gt[1], gt[1], 0, gt[3]+y1*gt[5], 0, gt[5])
            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            return _write_geotiff(out_path, warped, np.where(gray > 1, 255, 0).astype(np.uint8), gt, self.user_epsg, self._lock)
        elif mk == "proj":
            K = self._get_K_for((wi, hi))
            rect, alpha = _rectify_image(img, K, self._dist, self._rvec, self._tvec, self._centroid, self._grid_x, self._grid_y, self._grid_z)
            gt = (float(self._grid_x[0,0]), self._preview_gt[1], 0, float(self._grid_y[0,0]), 0, self._preview_gt[5])
            return _write_geotiff(out_path, rect, alpha, gt, self.user_epsg, self._lock)
        else:
            df = self._get_working_df(); gcps, epsg = _make_gdal_gcps(df)
            srs = osr.SpatialReference(); srs.ImportFromEPSG(epsg)
            gsd = abs(self._preview_gt[1]) if self._preview_gt else _estimate_gsd(df)
            if self._aoi and self._preview_gt:
                gt = self._preview_gt; x1, y1, x2, y2 = self._aoi
                te = [gt[0]+x1*gt[1], gt[3]+max(y1,y2)*gt[5], gt[0]+x2*gt[1], gt[3]+min(y1,y2)*gt[5]]
            else:
                extent = self._corners_to_world_extent(source_corners, hi, wi)
                if extent: x0, y0, x1, y1 = _apply_crop_to_extent(extent, crop_fracs); te = [x0, y0, x1, y1]
                else:
                    ex, ey = df["Real_X"].values, df["Real_Y"].values; mg = max(50, (ex.max()-ex.min())*0.05)
                    te = [ex.min()-mg, ey.min()-mg, ex.max()+mg, ey.max()+mg]
            _gdal_warp(img_path, out_path, gcps, srs.ExportToWkt(), epsg, mk, gsd, te, self._lock)
            return True

    # ── Batch ──
    def _browse_in(self):
        f = filedialog.askdirectory(parent=self)
        if f: self.input_folder = f; self._in_lbl.configure(text=os.path.basename(f), text_color="#81C784")
    def _browse_out(self):
        f = filedialog.askdirectory(parent=self)
        if f: self.output_folder = f; self._out_lbl.configure(text=os.path.basename(f), text_color="#81C784")
    def _collect_images(self, folder):
        imgs = []
        for e in ["*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff"]: imgs.extend(glob.glob(os.path.join(folder, e)))
        return sorted(set(imgs))

    def _process_all(self):
        if not self.input_folder: messagebox.showerror("Error", "Select input folder.", parent=self); return
        if not self.output_folder: messagebox.showerror("Error", "Select output folder.", parent=self); return
        if not self._resolve_epsg(): return
        mk = self._method_key
        if mk == "homo" and self._H is None: messagebox.showerror("Error", "Load homography.", parent=self); return
        if mk == "proj" and self._rvec is None: messagebox.showerror("Error", "Run georeferencing first.", parent=self); return
        if mk in ("tps","poly1","poly2") and self._gcp_df is None: messagebox.showerror("Error", "Load GCPs.", parent=self); return
        use_sub = self._subfolder_var.get()
        def _worker():
            # Collect all directories to process (recursive with os.walk)
            if use_sub:
                subs = sorted(
                    dirpath
                    for dirpath, _, _ in os.walk(self.input_folder)
                )
            else:
                subs = [self.input_folder]
            all_jobs = []
            for sub in subs:
                imgs = self._collect_images(sub)
                if not imgs: continue
                # Preserve full relative directory structure in output
                if use_sub and sub != self.input_folder:
                    rel = os.path.relpath(sub, self.input_folder)
                    out_sub = os.path.join(self.output_folder, rel)
                    os.makedirs(out_sub, exist_ok=True)
                else:
                    out_sub = self.output_folder
                for ip in imgs: all_jobs.append((ip, os.path.join(out_sub, os.path.splitext(os.path.basename(ip))[0]+".tif")))
            if not all_jobs: self.after(0, lambda: messagebox.showerror("Error", "No images.", parent=self)); return
            n = len(all_jobs); t0 = time.time(); m2i = {}
            print(f"\n[Batch] Processing {n} image(s)...")
            self.after(0, lambda: self._prog.set(0))
            for i, (ip, op) in enumerate(all_jobs, 1):
                try: self._save_single(ip, op); m2i[op] = ip
                except Exception as e: print(f"  Error: {os.path.basename(ip)}: {e}")
                f = i/n; self.after(0, lambda f=f: self._prog.set(f))
                self.after(0, lambda t=self._eta(t0, f): self._eta_lbl.configure(text=t))
            print("\n[Validation] Checking...")
            val_t0 = time.time()
            self.after(0, lambda: self._eta_lbl.configure(text="Validating..."))
            rc = {}; pf = []; pfs = set()
            out_folders = set(os.path.dirname(op) for _, op in all_jobs)
            max_passes = 20
            for vpass in range(max_passes):
                bad = []
                for folder in out_folders: bad.extend(sweep_and_validate(folder))
                bad = [(p, r) for p, r in bad if p not in pfs]
                if not bad: print("[Validation] All OK!"); break
                vfrac = (vpass + 1) / max_passes
                self.after(0, lambda t=self._eta(val_t0, vfrac): self._eta_lbl.configure(text=f"Validating… {t}"))
                for cp, reason in bad:
                    if rc.get(cp, 0) >= MAX_RETRIES:
                        if cp not in pfs: pf.append((cp, reason)); pfs.add(cp)
                    else:
                        inp = m2i.get(cp)
                        if inp and os.path.exists(inp):
                            try:
                                if os.path.exists(cp): os.remove(cp)
                                self._save_single(inp, cp)
                            except: pass
                            rc[cp] = rc.get(cp, 0) + 1
                        else: pf.append((cp, "Input not found")); pfs.add(cp)
            m, s = divmod(int(time.time()-t0), 60); print(f"\n[Batch] Complete in {m}m {s}s")
            self.after(0, lambda: self._prog.set(1.0)); self.after(0, lambda: self._eta_lbl.configure(text="Done"))
            if pf:
                ns = "\n".join(os.path.basename(p) for p, _ in pf[:10])
                self.after(0, lambda: messagebox.showwarning("Some Failed", f"{len(pf)} file(s) failed:\n\n{ns}", parent=self))
            else:
                self.after(0, lambda: messagebox.showinfo("Complete", f"All {n} files processed!", parent=self))
        threading.Thread(target=_worker, daemon=True).start()

    # ── Save / Load Settings ──
    def _save_settings(self):
        data = {
            "paths": {
                "image_path": self._img_path or "",
                "input_folder": self.input_folder or "",
                "output_folder": self.output_folder or "",
                "homography_path": self._homo_path or "",
                "gcp_path": self._gcp_path or "",
                "calibration_path": self._cal_path or "",
            },
            "method": self._method_var.get(),
            "corner": self._corner_var.get(),
            "manual_corners": self._manual_corner_ent.get().strip(),
            "crop": {tag: ent.get().strip() for tag, ent in self._crop_entries.items()},
            "scale": self._scale_ent.get().strip(),
            "epsg": self._epsg_ent.get().strip(),
            "elevation": self._elev_ent.get().strip(),
            "excluded_gcps": self._excl_ent.get().strip(),
            "aoi_manual": self._aoi_ent.get().strip(),
            "include_subfolders": bool(self._subfolder_var.get()),
        }
        try:
            initialdir = self.output_folder or None
            path = save_settings_json(self, "georef", data, initialdir=initialdir)
            if path:
                print(f"Settings saved: {path}")
        except Exception as e:
            messagebox.showerror("Save Settings", f"Could not save:\n{e}", parent=self)

    def _load_settings(self):
        try:
            initialdir = self.output_folder or None
            data, path = load_settings_json(self, "georef", initialdir=initialdir)
            if not data:
                return

            paths = data.get("paths", {})
            if paths.get("image_path"):
                self._img_path = paths["image_path"]
                self._img_lbl.configure(text=os.path.basename(self._img_path), text_color="#81C784")
                if os.path.exists(self._img_path):
                    self._show(self._img_path, self._orig_lbl)
            if paths.get("input_folder"):
                self.input_folder = paths["input_folder"]
                self._in_lbl.configure(text=os.path.basename(self.input_folder), text_color="#81C784")
            if paths.get("output_folder"):
                self.output_folder = paths["output_folder"]
                self._out_lbl.configure(text=os.path.basename(self.output_folder), text_color="#81C784")

            method = data.get("method")
            if method and method in METHODS:
                self._method_var.set(method)
                self._on_method_change(method)

            # Reload data files (homography, GCPs, calibration)
            if paths.get("homography_path"):
                self._load_homo(paths["homography_path"])
            if paths.get("gcp_path"):
                self._load_gcp(paths["gcp_path"])
            if paths.get("calibration_path"):
                self._load_cal(paths["calibration_path"])

            corner = data.get("corner")
            if corner:
                self._corner_var.set(corner)
                self._on_corner_change()
            manual = data.get("manual_corners", "")
            if manual:
                self._manual_corner_ent.delete(0, "end")
                self._manual_corner_ent.insert(0, manual)

            for tag, val in data.get("crop", {}).items():
                if tag in self._crop_entries:
                    self._crop_entries[tag].delete(0, "end")
                    self._crop_entries[tag].insert(0, val or "0")

            scale = data.get("scale", "")
            if scale:
                self._scale_ent.delete(0, "end")
                self._scale_ent.insert(0, scale)

            epsg = data.get("epsg", "")
            if epsg:
                self._epsg_ent.delete(0, "end")
                self._epsg_ent.insert(0, epsg)

            elev = data.get("elevation", "")
            if elev:
                self._elev_ent.delete(0, "end")
                self._elev_ent.insert(0, elev)

            excl = data.get("excluded_gcps", "")
            if excl:
                self._excl_ent.delete(0, "end")
                self._excl_ent.insert(0, excl)

            aoi = data.get("aoi_manual", "")
            if aoi:
                self._aoi_ent.delete(0, "end")
                self._aoi_ent.insert(0, aoi)

            self._subfolder_var.set(bool(data.get("include_subfolders", False)))

            print(f"Settings loaded: {path}")
        except Exception as e:
            messagebox.showerror("Load Settings", f"Could not load:\n{e}", parent=self)

    # ── Reset ──
    def _reset(self):
        if not messagebox.askyesno("Reset", "Clear everything?", parent=self): return
        self._img_path = ""; self._H = None; self._homo_epsg = None; self._homo_path = ""
        self._K = None; self._dist = None; self._cal_img_size = None; self._cal_path = ""
        self._gcp_df = None; self._gcp_epsg = None; self._gcp_path = ""
        self._rvec = self._tvec = self._centroid = None
        self._grid_x = self._grid_y = self._grid_z = None
        self._preview_bgr = None; self._preview_gt = None; self._preview_Hf = None
        self._aoi = None; self._aoi_preview_shape = None; self._excluded_gcps = []; self._scale = 4.0
        self.image_list = []; self.output_folder = ""; self.input_folder = ""; self.user_epsg = None
        for lbl, txt in [(self._orig_lbl,"Original"),(self._init_lbl,"Initial Georef"),(self._sec_lbl,"Secondary Georef")]:
            lbl.configure(image=None, text=txt); lbl.image = None
        for lbl in [self._img_lbl, self._homo_lbl, self._gcp_lbl, self._cal_lbl, self._in_lbl, self._out_lbl]:
            lbl.configure(text="~", text_color="gray")
        for ent in [self._aoi_ent, self._excl_ent, self._epsg_ent, self._elev_ent, self._manual_corner_ent]:
            ent.delete(0, "end")
        for ent in self._crop_entries.values(): ent.delete(0, "end"); ent.insert(0, "0")
        self._corner_var.set("Full Image"); self._on_corner_change()
        self._scale_ent.delete(0, "end"); self._scale_ent.insert(0, "4")
        self._aoi_status.configure(text="AOI: not set", text_color="gray")
        self._rec_scale_lbl.configure(text="")
        self._prog.set(0); self._eta_lbl.configure(text="ETA: ~"); self._subfolder_var.set(False)
        print("\n" + "="*50 + "\n  Reset complete\n" + "="*50 + "\n")


def main():
    root = ctk.CTk(); root.withdraw()
    GeoReferenceModule(master=root).mainloop()

if __name__ == "__main__":
    main()