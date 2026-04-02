"""
Field of View (FOV) Generator

Visualises single‑ or multi‑camera FOV footprints on an optional basemap
and/or DEM.  When a DEM is supplied the tool performs a line‑of‑sight
viewshed so that terrain obstructions mask the FOV correctly.

Basemap priority:
  1. User‑supplied GeoTIFF            → rendered behind FOV
  2. Distance rings + coordinate grid  → clean standalone plot (no internet)
"""

# %% ————————————————————————————— imports —————————————————————————————
import math
import os
import threading
import re
from pathlib import Path
from typing import List, Dict, Any
from shapely.ops import unary_union
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import shapely.geometry as geom
import pyproj
import geopandas as gpd

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox

from utils import fit_geometry, resource_path, setup_console, restore_console

try:
    import rasterio
    from rasterio.warp import transform_bounds
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")


# %% ————————————————————————————— optics helpers ——————————————————————



def utm_crs_from_lonlat(lon_deg: float, lat_deg: float) -> pyproj.CRS:
    zone = int((lon_deg + 180) // 6) + 1
    is_northern = lat_deg >= 0
    epsg = 32600 + zone if is_northern else 32700 + zone
    return pyproj.CRS.from_epsg(epsg)


def project_xy_from_lonlat(lon_deg: float, lat_deg: float, crs_to: pyproj.CRS):
    crs_from = pyproj.CRS.from_epsg(4326)
    transformer = pyproj.Transformer.from_crs(crs_from, crs_to, always_xy=True)
    return transformer.transform(lon_deg, lat_deg)


def offset_point(x, y, dist_m, azimuth_rad):
    dx = dist_m * math.sin(azimuth_rad)
    dy = dist_m * math.cos(azimuth_rad)
    return (x + dx, y + dy)


def fov_radians(sensor_mm, focal_mm):
    return 2.0 * math.atan(sensor_mm / (2.0 * focal_mm))


def dof_near_far_m(f_mm, N, coc_mm, focus_m):
    H_hyper_mm = (f_mm * f_mm) / (N * coc_mm) + f_mm
    s_mm = focus_m * 1000.0
    near_mm = (H_hyper_mm * s_mm) / (H_hyper_mm + (s_mm - f_mm))
    if s_mm >= H_hyper_mm:
        far_mm = math.inf
    else:
        far_mm = (H_hyper_mm * s_mm) / (H_hyper_mm - (s_mm - f_mm))
    return (H_hyper_mm / 1000.0, near_mm / 1000.0, far_mm / 1000.0)


def ground_radius_from_slant(D_m, H_m):
    if math.isinf(D_m):
        return math.inf
    return math.sqrt(max(0.0, D_m * D_m - H_m * H_m))


def build_wedge_polygon(x0, y0, f_mm, sw_mm, sh_mm, H_m,
                        heading_deg, depress_deg, range_m):
    head = math.radians(heading_deg)
    delta = math.radians(depress_deg)
    hfov = fov_radians(sw_mm, f_mm)
    vfov = fov_radians(sh_mm, f_mm)
    h = hfov / 2.0
    v = vfov / 2.0
    beta_near = delta + v
    beta_far = delta - v
    if beta_near <= 0:
        return None, hfov, vfov
    R0 = H_m / math.tan(beta_near)
    if beta_far <= 0:
        R1 = range_m
    else:
        R1 = min(H_m / math.tan(beta_far), range_m)
    R0 = max(0.0, R0)
    R1 = max(R0, R1)
    L0 = R0 * math.tan(h)
    L1 = R1 * math.tan(h)
    C0 = offset_point(x0, y0, R0, head)
    C1 = offset_point(x0, y0, R1, head)
    NL = offset_point(C0[0], C0[1], L0, head - math.pi / 2)
    NR = offset_point(C0[0], C0[1], L0, head + math.pi / 2)
    FL = offset_point(C1[0], C1[1], L1, head - math.pi / 2)
    FR = offset_point(C1[0], C1[1], L1, head + math.pi / 2)
    poly = geom.Polygon([NL, NR, FR, FL, NL])
    return poly, hfov, vfov


def build_dof_zone(x0, y0, H_m, f_mm, N, coc_mm, focus_m, range_m):
    Hhyper_m, near_m, far_m = dof_near_far_m(f_mm, N, coc_mm, focus_m)
    r0 = ground_radius_from_slant(near_m, H_m)
    r1 = ground_radius_from_slant(far_m, H_m)
    if math.isinf(r1):
        r1 = range_m
    else:
        r1 = min(r1, range_m)
    r0 = max(0.0, r0)
    r1 = max(r0, r1)
    center = geom.Point(x0, y0)
    outer = center.buffer(r1, resolution=24)
    if r0 <= 0:
        ring = outer
    else:
        inner = center.buffer(r0, resolution=24)
        ring = outer.difference(inner)
    return ring, Hhyper_m, near_m, far_m, r0, r1


def build_best_focus_band(x0, y0, H_m, focus_m, halfwidth_m, range_m):
    if halfwidth_m <= 0:
        return None
    r = math.sqrt(max(0.0, focus_m * focus_m - H_m * H_m))
    r_in = max(0.0, r - halfwidth_m)
    r_out = min(range_m, r + halfwidth_m)
    center = geom.Point(x0, y0)
    band = center.buffer(r_out, resolution=24).difference(
        center.buffer(r_in, resolution=24))
    return band


def clamp01(x):
    return max(0.0, min(1.0, x))


def compute_headings_for_overlap(cam_params, base_heading_deg, overlap_pct):
    overlap_frac = clamp01(overlap_pct / 100.0)
    hfov_degs = [math.degrees(fov_radians(p["sw_mm"], p["f_mm"]))
                 for p in cam_params]
    seps = []
    for i in range(len(cam_params) - 1):
        h1, h2 = hfov_degs[i], hfov_degs[i + 1]
        overlap_angle = overlap_frac * min(h1, h2)
        seps.append(max(0.0, (h1 / 2.0 + h2 / 2.0) - overlap_angle))
    rel = [0.0]
    for s in seps:
        rel.append(rel[-1] + s)
    mid = len(cam_params) // 2
    shift = base_heading_deg - rel[mid]
    return [r + shift for r in rel]


# ——— bilinear interpolation helper ———

def _bilinear_sample(grid_z, xs, ys, query_x, query_y):
    """
    Bilinear interpolation on a regular grid.

    Parameters
    ----------
    grid_z : np.ndarray (H, W)  — elevation grid
    xs : np.ndarray (W,)        — x coordinates of grid columns
    ys : np.ndarray (H,)        — y coordinates of grid rows (top→bottom)
    query_x, query_y : np.ndarray — query coordinates (same shape)

    Returns
    -------
    np.ndarray — interpolated values, same shape as query_x
    """
    # compute fractional indices
    fx = (query_x - xs[0]) / (xs[-1] - xs[0]) * (len(xs) - 1)
    fy = (query_y - ys[0]) / (ys[-1] - ys[0]) * (len(ys) - 1)

    ix0 = np.clip(np.floor(fx).astype(int), 0, len(xs) - 2)
    iy0 = np.clip(np.floor(fy).astype(int), 0, len(ys) - 2)
    ix1 = ix0 + 1
    iy1 = iy0 + 1

    # fractional part
    wx = np.clip(fx - ix0, 0.0, 1.0)
    wy = np.clip(fy - iy0, 0.0, 1.0)

    # four corner values
    v00 = grid_z[iy0, ix0]
    v01 = grid_z[iy0, ix1]
    v10 = grid_z[iy1, ix0]
    v11 = grid_z[iy1, ix1]

    # bilinear blend
    return (v00 * (1 - wx) * (1 - wy) +
            v01 * wx * (1 - wy) +
            v10 * (1 - wx) * wy +
            v11 * wx * wy)


# ——— vectorised line‑of‑sight viewshed on a DEM ———

def compute_viewshed_mask(dem_path, x0, y0, cam_height_m, utm_crs,
                          bounds, width_px=600, height_px=600):
    """
    Return (bool_mask, cam_ground_z) where True = visible from camera.
    Uses vectorised ray marching along radial lines sampled from the DEM
    with bilinear interpolation for terrain lookups.
    Returns (None, 0.0) on failure.
    """
    if not HAS_RASTERIO:
        return None, 0.0
    try:
        with rasterio.open(dem_path) as src:
            dem_crs = src.crs
            transformer_to_dem = pyproj.Transformer.from_crs(
                utm_crs, dem_crs, always_xy=True)

            dem_data = src.read(1).astype(np.float32)
            dem_transform = src.transform
            dem_nodata = src.nodata

            # camera position in DEM pixel space
            x0_dem, y0_dem = transformer_to_dem.transform(x0, y0)
            cam_col, cam_row = ~dem_transform * (x0_dem, y0_dem)
            cam_row, cam_col = int(round(cam_row)), int(round(cam_col))

            # camera ground elevation from DEM
            if (0 <= cam_row < dem_data.shape[0] and
                    0 <= cam_col < dem_data.shape[1]):
                cam_ground_z = float(dem_data[cam_row, cam_col])
                if dem_nodata is not None and cam_ground_z == dem_nodata:
                    cam_ground_z = 0.0
            else:
                cam_ground_z = 0.0
            cam_z = cam_ground_z + cam_height_m

            # sample DEM elevations on the output grid
            left, bottom, right, top = bounds
            xs = np.linspace(left, right, width_px)
            ys = np.linspace(top, bottom, height_px)  # top→bottom

            grid_x, grid_y = np.meshgrid(xs, ys)
            gx_dem, gy_dem = transformer_to_dem.transform(
                grid_x.ravel(), grid_y.ravel())
            gx_dem = np.array(gx_dem).reshape(grid_x.shape)
            gy_dem = np.array(gy_dem).reshape(grid_y.shape)

            rows_f = (gy_dem - dem_transform.f) / dem_transform.e
            cols_f = (gx_dem - dem_transform.c) / dem_transform.a

            # bilinear sample from DEM
            r0 = np.clip(np.floor(rows_f).astype(int), 0, dem_data.shape[0] - 2)
            c0 = np.clip(np.floor(cols_f).astype(int), 0, dem_data.shape[1] - 2)
            r1 = r0 + 1
            c1 = c0 + 1
            wr = np.clip(rows_f - r0, 0.0, 1.0)
            wc = np.clip(cols_f - c0, 0.0, 1.0)

            grid_z = (dem_data[r0, c0] * (1 - wr) * (1 - wc) +
                      dem_data[r0, c1] * (1 - wr) * wc +
                      dem_data[r1, c0] * wr * (1 - wc) +
                      dem_data[r1, c1] * wr * wc)

            if dem_nodata is not None:
                grid_z[grid_z == dem_nodata] = 0.0

        # vectorised ray march: for each output pixel, check LOS
        # compute distances from camera to each grid point
        dx = grid_x - x0
        dy = grid_y - y0
        dist = np.sqrt(dx * dx + dy * dy)

        # target elevations
        target_z = grid_z.copy()

        # number of steps scales with distance (every ~3 m for finer sampling)
        max_dist = float(np.max(dist))
        n_steps = max(int(max_dist / 3.0), 10)

        vis = np.ones((height_px, width_px), dtype=bool)

        # vectorised: check all pixels at each fractional step
        for step in range(1, n_steps):
            frac = step / n_steps
            # intermediate positions along each ray
            sx = x0 + dx * frac
            sy = y0 + dy * frac

            # LOS height at this fraction
            los_z = cam_z + (target_z - cam_z) * frac

            # terrain height at intermediate positions (bilinear)
            terrain_z = _bilinear_sample(grid_z, xs, ys, sx, sy)

            # mark blocked where terrain exceeds LOS
            newly_blocked = (terrain_z > los_z) & (dist > 1.0)
            vis[newly_blocked] = False

        # pixels very close to camera are always visible
        vis[dist < 1.0] = True

        return vis, cam_ground_z

    except Exception as e:
        print(f"[WARN] Viewshed computation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, 0.0


# ——— load basemap helpers ———

def _load_geotiff_basemap(path, ax, bounds_utm, utm_crs):
    """Render a user GeoTIFF behind the FOV plot."""
    if not HAS_RASTERIO:
        return False
    try:
        with rasterio.open(path) as src:
            left, bottom, right, top = bounds_utm
            # transform bounds to basemap CRS
            bm_bounds = transform_bounds(utm_crs, src.crs,
                                         left, bottom, right, top,
                                         densify_pts=21)
            # read a window
            from rasterio.windows import from_bounds
            window = from_bounds(*bm_bounds, transform=src.transform)
            if src.count >= 3:
                r = src.read(1, window=window)
                g = src.read(2, window=window)
                b = src.read(3, window=window)
                img = np.dstack([r, g, b])
                # scale to 0-1
                if img.max() > 1:
                    img = img.astype(np.float32) / img.max()
            else:
                gray = src.read(1, window=window).astype(np.float32)
                gray = gray / max(gray.max(), 1e-6)
                img = np.dstack([gray, gray, gray])

            ax.imshow(img, extent=[left, right, bottom, top],
                      origin="upper", aspect="equal", zorder=0, alpha=0.8)
            return True
    except Exception as e:
        print(f"[WARN] Could not load basemap: {e}")
        return False


# ——— DEM hillshade fallback ———

def _compute_hillshade(elevation, azimuth_deg=315, altitude_deg=45,
                       cell_size=1.0):
    """Return a 0–1 hillshade array from an elevation grid."""
    az = math.radians(azimuth_deg)
    alt = math.radians(altitude_deg)
    dy, dx = np.gradient(elevation, cell_size)
    slope = np.arctan(np.sqrt(dx * dx + dy * dy))
    aspect = np.arctan2(-dy, dx)
    hs = (np.sin(alt) * np.cos(slope) +
          np.cos(alt) * np.sin(slope) * np.cos(az - aspect))
    hs = np.clip(hs, 0, 1)
    return hs


def _load_dem_hillshade(dem_path, ax, bounds_utm, utm_crs):
    """Render a hillshade derived from the DEM as background.

    Returns (True, e_min, e_max) on success, (False, None, None) on failure.
    """
    if not HAS_RASTERIO:
        return False, None, None
    try:
        with rasterio.open(dem_path) as src:
            left, bottom, right, top = bounds_utm
            bm_bounds = transform_bounds(utm_crs, src.crs,
                                         left, bottom, right, top,
                                         densify_pts=21)
            from rasterio.windows import from_bounds
            window = from_bounds(*bm_bounds, transform=src.transform)
            elev = src.read(1, window=window).astype(np.float32)
            nodata = src.nodata
            if nodata is not None:
                elev[elev == nodata] = np.nanmin(
                    elev[elev != nodata]) if np.any(elev != nodata) else 0.0
            # estimate cell size in metres from the window transform
            win_transform = src.window_transform(window)
            cell_m = abs(win_transform.a)
            # if DEM CRS is geographic (degrees), approximate metres
            if src.crs and src.crs.is_geographic:
                cell_m = cell_m * 111_000.0

        hs = _compute_hillshade(elev, cell_size=max(cell_m, 0.01))
        # colour the hillshade with a terrain tint
        terrain_cmap = plt.cm.terrain
        # normalise elevation for colour
        e_min, e_max = float(np.nanmin(elev)), float(np.nanmax(elev))
        if e_max - e_min < 0.01:
            e_norm = np.zeros_like(elev)
        else:
            e_norm = (elev - e_min) / (e_max - e_min)
        rgb = terrain_cmap(e_norm)[..., :3]
        # blend hillshade with terrain colour
        hs3 = np.dstack([hs, hs, hs])
        blended = np.clip(rgb * 0.5 + hs3 * 0.5, 0, 1)

        ax.imshow(blended, extent=[left, right, bottom, top],
                  origin="upper", aspect="equal", zorder=0, alpha=0.85)
        return True, e_min, e_max
    except Exception as e:
        print(f"[WARN] Could not render DEM hillshade: {e}")
        return False, None, None


# ——— query DEM elevation at a point ———

def _query_dem_elevation(dem_path, x_utm, y_utm, utm_crs):
    """Return ground elevation (m) at a UTM point from the DEM."""
    if not HAS_RASTERIO:
        return None
    try:
        with rasterio.open(dem_path) as src:
            transformer = pyproj.Transformer.from_crs(
                utm_crs, src.crs, always_xy=True)
            x_dem, y_dem = transformer.transform(x_utm, y_utm)
            row, col = ~src.transform * (x_dem, y_dem)
            row, col = int(round(row)), int(round(col))
            data = src.read(1)
            if 0 <= row < data.shape[0] and 0 <= col < data.shape[1]:
                val = float(data[row, col])
                if src.nodata is not None and val == src.nodata:
                    return 0.0
                return val
        return 0.0
    except Exception:
        return None


# ——— get DEM elevation range within bounds (for colorbar) ———

def _get_dem_elevation_range(dem_path, bounds_utm, utm_crs):
    """Return (e_min, e_max) from the DEM within the given UTM bounds."""
    if not HAS_RASTERIO:
        return None, None
    try:
        with rasterio.open(dem_path) as src:
            left, bottom, right, top = bounds_utm
            bm_bounds = transform_bounds(utm_crs, src.crs,
                                         left, bottom, right, top,
                                         densify_pts=21)
            from rasterio.windows import from_bounds
            window = from_bounds(*bm_bounds, transform=src.transform)
            elev = src.read(1, window=window).astype(np.float32)
            nodata = src.nodata
            if nodata is not None:
                elev = elev[elev != nodata]
            if elev.size == 0:
                return None, None
            return float(np.nanmin(elev)), float(np.nanmax(elev))
    except Exception:
        return None, None


# ——— vectorised FOV mask rasterisation ———

def _rasterise_fov_mask(wedge_union, bounds, width_px, height_px):
    """
    Return a bool mask (height_px × width_px) True inside the FOV union.
    Uses vectorised approach with Shapely's prepared geometry and chunked
    point-in-polygon testing.
    """
    from shapely.prepared import prep
    left, bottom, right, top = bounds
    xs = np.linspace(left, right, width_px)
    ys = np.linspace(top, bottom, height_px)

    grid_x, grid_y = np.meshgrid(xs, ys)
    points_flat = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    # use Shapely's vectorised contains if available (shapely >= 2.0)
    try:
        from shapely import contains_xy
        mask_flat = contains_xy(wedge_union, points_flat[:, 0], points_flat[:, 1])
        return mask_flat.reshape(height_px, width_px)
    except ImportError:
        pass

    # fallback: use prepared geometry with MultiPoint chunks
    prepared = prep(wedge_union)
    mask = np.zeros(height_px * width_px, dtype=bool)

    # process in row chunks for efficiency
    chunk_size = width_px
    for i in range(0, len(points_flat), chunk_size):
        chunk = points_flat[i:i + chunk_size]
        mp = geom.MultiPoint(chunk.tolist())
        result = prepared.intersects(mp)
        if result:
            # need per-point test for this chunk
            for j, pt in enumerate(chunk):
                if prepared.contains(geom.Point(pt[0], pt[1])):
                    mask[i + j] = True

    return mask.reshape(height_px, width_px)


# ——— auto-detect DEM native resolution ———

def _get_dem_native_res_m(dem_path, utm_crs):
    """
    Return approximate DEM cell size in metres.
    Used to auto-scale viewshed grid resolution.
    """
    if not HAS_RASTERIO:
        return None
    try:
        with rasterio.open(dem_path) as src:
            cell_x = abs(src.transform.a)
            cell_y = abs(src.transform.e)
            cell = (cell_x + cell_y) / 2.0
            if src.crs and src.crs.is_geographic:
                cell = cell * 111_000.0
            return cell
    except Exception:
        return None


# ——— scale bar ———

def _nice_number(x):
    if x <= 0:
        return 0
    exp = 10 ** math.floor(math.log10(x))
    for m in (1, 2, 5, 10):
        if m * exp >= x:
            return m * exp
    return 10 * exp


def add_scalebar(ax, frac=0.20, x_frac=0.50, y_frac=0.06, align="center"):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    w_map = xmax - xmin
    h_map = ymax - ymin
    target_ground = w_map * frac
    L_ground = _nice_number(target_ground)
    y0 = ymin + y_frac * h_map
    if align == "center":
        x0 = xmin + x_frac * w_map - L_ground / 2
    elif align == "right":
        x0 = xmin + x_frac * w_map - L_ground
    else:
        x0 = xmin + x_frac * w_map

    ax.plot([x0, x0 + L_ground], [y0, y0], color="white",
            linewidth=6, solid_capstyle="butt", zorder=20)
    ax.plot([x0, x0 + L_ground], [y0, y0], color="black",
            linewidth=2, solid_capstyle="butt", zorder=21)
    tick_h = 0.01 * h_map
    for xp in (x0, x0 + L_ground):
        ax.plot([xp, xp], [y0 - tick_h, y0 + tick_h],
                color="black", linewidth=2, zorder=21)
    ax.text(x0 + L_ground / 2, y0 + 0.02 * h_map, f"{int(L_ground)} m",
            ha="center", va="bottom", zorder=22,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1))


# %% ————————————————————————————— main GUI ————————————————————————————
class FOVGeneratorWindow(ctk.CTkToplevel):
    """Camera Field‑of‑View planning tool."""

    def __init__(self, master=None, **kw):
        super().__init__(master=master, **kw)
        self.title("Field of View Generator")
        fit_geometry(self, 1400, 1050, resizable=True)
        try:
            self.iconbitmap(resource_path("launch_logo.ico"))
        except Exception:
            pass

        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # ——— state ———
        self.basemap_path = None
        self.dem_path = None
        self.output_folder = None
        self.ax_vs = None

        # ——— layout: top display, console, bottom controls ———
        self.grid_rowconfigure(0, weight=4)
        self.grid_rowconfigure(1, weight=0, minsize=160)
        self.grid_rowconfigure(2, weight=0)
        self.grid_columnconfigure(0, weight=1)

        # ---- TOP: display panel (plot left, legend right) ----
        self.top_panel = ctk.CTkFrame(self, fg_color="black")
        self.top_panel.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.top_panel.grid_columnconfigure(0, weight=4)
        self.top_panel.grid_columnconfigure(1, weight=1)
        self.top_panel.grid_rowconfigure(0, weight=1)

        # plot (left)
        self.plot_frame = ctk.CTkFrame(self.top_panel, fg_color="black")
        self.plot_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 2))

        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.ax.set_title("FOV Map — Generate to display")
        self.ax.set_xlabel("Easting (m)")
        self.ax.set_ylabel("Northing (m)")
        self.ax.grid(True, alpha=0.3)
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas_plot.get_tk_widget().pack(fill="both", expand=True)

        # legend panel (right)
        self.legend_frame = ctk.CTkFrame(self.top_panel, width=220, fg_color="black")
        self.legend_frame.grid(row=0, column=1, sticky="nsew", padx=(2, 0))
        self.legend_frame.grid_propagate(False)
        ctk.CTkLabel(self.legend_frame, text="Legend",
                     font=("Arial", 13, "bold")).pack(
            anchor="w", padx=8, pady=(8, 4))
        self.legend_content = ctk.CTkFrame(self.legend_frame,
                                            fg_color="transparent")
        self.legend_content.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        # ---- BOTTOM: controls ----
        self.bottom_panel = ctk.CTkFrame(self)
        self.bottom_panel.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)

        # Row 1 — basemap & DEM
        row1 = ctk.CTkFrame(self.bottom_panel)
        row1.pack(fill="x", padx=5, pady=2)

        ctk.CTkButton(row1, text="Browse Basemap (GeoTIFF)",
                      command=self._browse_basemap).grid(row=0, column=0, padx=5, pady=5)
        self.basemap_label = ctk.CTkLabel(row1, text="No basemap (distance grid)")
        self.basemap_label.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        ctk.CTkButton(row1, text="Browse DEM (GeoTIFF)",
                      command=self._browse_dem).grid(row=0, column=2, padx=5, pady=5)
        self.dem_label = ctk.CTkLabel(row1, text="No DEM (flat ground)")
        self.dem_label.grid(row=0, column=3, padx=5, pady=5, sticky="w")

        # Row 2 — location
        row2 = ctk.CTkFrame(self.bottom_panel)
        row2.pack(fill="x", padx=5, pady=2)

        entries_r2 = [
            ("Latitude", "54.7900"),
            ("Longitude", "8.2834"),
            ("Height above ground (m)", "10"),
            ("Heading (° from N)", "110"),
            ("Depression angle (°)", "9.6"),
        ]
        self.loc_entries = {}
        for col, (lbl, default) in enumerate(entries_r2):
            ctk.CTkLabel(row2, text=lbl).grid(row=0, column=col * 2, padx=3, pady=3)
            e = ctk.CTkEntry(row2, width=80)
            e.insert(0, default)
            e.grid(row=0, column=col * 2 + 1, padx=3, pady=3)
            self.loc_entries[lbl] = e

        # Row 3 — focus & range
        row3 = ctk.CTkFrame(self.bottom_panel)
        row3.pack(fill="x", padx=5, pady=2)

        entries_r3 = [
            ("Focus distance (m)", "300"),
            ("Max display range (m)", "1000"),
        ]
        self.range_entries = {}
        for col, (lbl, default) in enumerate(entries_r3):
            ctk.CTkLabel(row3, text=lbl).grid(row=0, column=col * 2, padx=3, pady=3)
            e = ctk.CTkEntry(row3, width=80)
            e.insert(0, default)
            e.grid(row=0, column=col * 2 + 1, padx=3, pady=3)
            self.range_entries[lbl] = e

        # Row 4 — cameras & overlap & viewshed resolution
        row4 = ctk.CTkFrame(self.bottom_panel)
        row4.pack(fill="x", padx=5, pady=2)

        ctk.CTkLabel(row4, text="Number of cameras").grid(row=0, column=0, padx=3, pady=3)
        self.num_cams_entry = ctk.CTkEntry(row4, width=60)
        self.num_cams_entry.insert(0, "4")
        self.num_cams_entry.grid(row=0, column=1, padx=3, pady=3)

        ctk.CTkLabel(row4, text="Overlap (%)").grid(row=0, column=2, padx=3, pady=3)
        self.overlap_entry = ctk.CTkEntry(row4, width=60)
        self.overlap_entry.insert(0, "20")
        self.overlap_entry.grid(row=0, column=3, padx=3, pady=3)

        ctk.CTkLabel(row4, text="Viewshed resolution (px)").grid(
            row=0, column=4, padx=3, pady=3)
        self.vs_res_entry = ctk.CTkEntry(row4, width=60)
        self.vs_res_entry.insert(0, "auto")
        self.vs_res_entry.grid(row=0, column=5, padx=3, pady=3)

        # Row 5 — sensor settings
        row5 = ctk.CTkFrame(self.bottom_panel)
        row5.pack(fill="x", padx=5, pady=2)

        entries_r5 = [
            ("Focal length (mm)", "16"),
            ("Aperture (f-number)", "4"),
            ("Sensor width (mm)", "7.37"),
            ("Sensor height (mm)", "4.92"),
        ]
        self.sensor_entries = {}
        for col, (lbl, default) in enumerate(entries_r5):
            ctk.CTkLabel(row5, text=lbl).grid(row=0, column=col * 2, padx=3, pady=3)
            e = ctk.CTkEntry(row5, width=80)
            e.insert(0, default)
            e.grid(row=0, column=col * 2 + 1, padx=3, pady=3)
            self.sensor_entries[lbl] = e

        # Row 6 — output & generate
        row6 = ctk.CTkFrame(self.bottom_panel)
        row6.pack(fill="x", padx=5, pady=2)

        ctk.CTkButton(row6, text="Browse Output Folder",
                      command=self._browse_output, fg_color="#8C7738").grid(row=0, column=0, padx=5, pady=5)
        self.output_label = ctk.CTkLabel(row6, text="No output folder selected")
        self.output_label.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        ctk.CTkButton(row6, text="Generate FOV Map",
                      command=self._generate_threaded, fg_color="#0F52BA").grid(row=0, column=2, padx=10, pady=5)

        self.btn_reset = ctk.CTkButton(
            row6, text="Reset", command=self._reset,
            width=80, fg_color="#8B0000", hover_color="#A52A2A")
        self.btn_reset.grid(row=0, column=3, padx=5, pady=5, sticky="e")

        # ---- CONSOLE ----
        self.console_frame = ctk.CTkFrame(self)
        self.console_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        console_scroll = tk.Scrollbar(self.console_frame)
        console_scroll.pack(side="right", fill="y")
        self.console_text = tk.Text(
            self.console_frame, wrap="word", height=10,
            yscrollcommand=console_scroll.set)
        self.console_text.pack(fill="both", expand=True, padx=5, pady=5)
        console_scroll.config(command=self.console_text.yview)
        self._console_redir = setup_console(
            self.console_text,
            "FOV Generator ready.\n"
            "Configure camera parameters below, then click Generate FOV Map.\n"
            "Viewshed resolution: 'auto' = match DEM native resolution,\n"
            "  or enter a number (e.g. 600) for manual control.\n"
            "--------------------------------"
        )

    def _on_close(self):
        restore_console(getattr(self, "_console_redir", None))
        self.destroy()

    def _ui_call(self, func, *args, **kwargs):
        try:
            self.after(0, lambda: func(*args, **kwargs))
        except Exception:
            pass

    def _ui_message(self, kind, title, message):
        def _show():
            fn = getattr(messagebox, f"show{kind}", None)
            if fn is not None:
                fn(title, message)
        self._ui_call(_show)

    def _get_float_value(self, raw, name):
        try:
            return float(raw)
        except ValueError:
            raise ValueError(f"Invalid value for '{name}'")

    def _collect_generate_config(self):
        return {
            "lat": self.loc_entries["Latitude"].get().strip(),
            "lon": self.loc_entries["Longitude"].get().strip(),
            "height_m": self.loc_entries["Height above ground (m)"].get().strip(),
            "heading": self.loc_entries["Heading (° from N)"].get().strip(),
            "depress": self.loc_entries["Depression angle (°)"].get().strip(),
            "focus_m": self.range_entries["Focus distance (m)"].get().strip(),
            "range_m": self.range_entries["Max display range (m)"].get().strip(),
            "n_cams": self.num_cams_entry.get().strip(),
            "overlap_pct": self.overlap_entry.get().strip(),
            "viewshed_px": self.vs_res_entry.get().strip(),
            "f_mm": self.sensor_entries["Focal length (mm)"].get().strip(),
            "aperture": self.sensor_entries["Aperture (f-number)"].get().strip(),
            "sensor_w": self.sensor_entries["Sensor width (mm)"].get().strip(),
            "sensor_h": self.sensor_entries["Sensor height (mm)"].get().strip(),
            "basemap_path": self.basemap_path,
            "dem_path": self.dem_path,
            "output_folder": self.output_folder,
        }

    # ——— browse callbacks ———

    def _browse_basemap(self):
        p = filedialog.askopenfilename(parent=self,
            title="Select Basemap GeoTIFF",
            filetypes=[("GeoTIFF", "*.tif *.tiff")])
        if p:
            self.basemap_path = p
            self.basemap_label.configure(text=os.path.basename(p))

    def _browse_dem(self):
        p = filedialog.askopenfilename(parent=self,
            title="Select DEM GeoTIFF",
            filetypes=[("GeoTIFF", "*.tif *.tiff")])
        if p:
            self.dem_path = p
            self.dem_label.configure(text=os.path.basename(p))

    def _browse_output(self):
        d = filedialog.askdirectory(parent=self,title="Select Output Folder")
        if d:
            self.output_folder = d
            self.output_label.configure(text=d)

    # ——— reset ———

    def _reset(self):
        self.basemap_path = None
        self.dem_path = None
        self.output_folder = None
        self.basemap_label.configure(text="No basemap (distance grid)")
        self.dem_label.configure(text="No DEM (flat ground)")
        self.output_label.configure(text="No output folder selected")

        for d in (self.loc_entries, self.range_entries, self.sensor_entries):
            for e in d.values():
                e.delete(0, tk.END)
        self.num_cams_entry.delete(0, tk.END)
        self.overlap_entry.delete(0, tk.END)
        self.vs_res_entry.delete(0, tk.END)
        self.vs_res_entry.insert(0, "auto")

        self.fig.clear()
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax_vs = None
        self.ax.set_title("FOV Map — Generate to display")
        self.ax.set_xlabel("Easting (m)")
        self.ax.set_ylabel("Northing (m)")
        self.ax.grid(True, alpha=0.3)
        self.canvas_plot.draw()

        # clear legend panel
        for w in self.legend_content.winfo_children():
            w.destroy()

        self.console_text.delete("1.0", tk.END)
        print("Session reset.\n--------------------------------")

    # ——— generation ———

    def _get_float(self, entry, name):
        try:
            return float(entry.get())
        except ValueError:
            raise ValueError(f"Invalid value for '{name}'")

    def _resolve_viewshed_px(self, raw_value, range_m, lon, lat, dem_path):
        """
        Determine viewshed grid resolution in pixels.

        'auto' → scale to DEM native resolution (capped at 1200 px).
        A number → use directly (clamped 100–2000).
        """
        raw = str(raw_value).strip().lower()
        pad = range_m * 1.3
        extent_m = pad * 2.0  # total width/height in metres

        if raw == "auto":
            if dem_path and os.path.exists(dem_path):
                utm = utm_crs_from_lonlat(float(lon), float(lat))
                cell_m = _get_dem_native_res_m(dem_path, utm)
                if cell_m and cell_m > 0:
                    px = int(extent_m / cell_m)
                    px = max(200, min(px, 1200))
                    print(f"[INFO] DEM native resolution: ~{cell_m:.2f} m/px"
                          f" → viewshed grid: {px}×{px} px")
                    return px
            return 600
        else:
            try:
                px = int(float(raw))
                return max(100, min(px, 2000))
            except ValueError:
                print("[WARN] Invalid viewshed resolution, using 600.")
                return 600

    def _generate_threaded(self):
        cfg = self._collect_generate_config()
        threading.Thread(target=self._generate, args=(cfg,), daemon=True).start()

    def _generate(self, cfg):
        try:
            # ---- read parameters (snapshot collected on main thread) ----
            lat = self._get_float_value(cfg["lat"], "Latitude")
            lon = self._get_float_value(cfg["lon"], "Longitude")
            H_m = self._get_float_value(cfg["height_m"], "Height above ground")
            heading = self._get_float_value(cfg["heading"], "Heading")
            depress = self._get_float_value(cfg["depress"], "Depression angle")
            focus_m = self._get_float_value(cfg["focus_m"], "Focus distance")
            range_m = self._get_float_value(cfg["range_m"], "Max range")
            n_cams = int(self._get_float_value(cfg["n_cams"], "Number of cameras"))
            overlap_pct = self._get_float_value(cfg["overlap_pct"], "Overlap")
            f_mm = self._get_float_value(cfg["f_mm"], "Focal length")
            N_ap = self._get_float_value(cfg["aperture"], "Aperture")
            sw_mm = self._get_float_value(cfg["sensor_w"], "Sensor width")
            sh_mm = self._get_float_value(cfg["sensor_h"], "Sensor height")
            basemap_path = cfg.get("basemap_path")
            dem_path = cfg.get("dem_path")
            output_folder = cfg.get("output_folder")

            coc_mm = 0.00024  # standard CoC for small sensors
            best_band_hw = 2.0

            print(f"\nGenerating FOV for {n_cams} camera(s) at "
                  f"({lat:.6f}, {lon:.6f}) …")

            # ---- build camera parameter list ----
            DEFAULT = dict(
                f_mm=f_mm, N=N_ap, coc_mm=coc_mm, H_m=H_m,
                sw_mm=sw_mm, sh_mm=sh_mm, focus_m=focus_m,
                heading_deg=heading, depress_deg=depress,
                range_m=range_m, best_band_halfwidth_m=best_band_hw,
            )
            cam_params: List[Dict[str, Any]] = []
            for i in range(n_cams):
                p = DEFAULT.copy()
                p["name"] = f"Cam {i + 1}"
                cam_params.append(p)

            # auto-headings
            if n_cams > 1:
                auto_h = compute_headings_for_overlap(
                    cam_params, heading, overlap_pct)
                for i, h in enumerate(auto_h):
                    cam_params[i]["heading_deg"] = h

            # ---- project ----
            utm = utm_crs_from_lonlat(lon, lat)
            x0, y0 = project_xy_from_lonlat(lon, lat, utm)

            # ---- query ground elevation at camera from DEM ----
            cam_ground_z = None
            if dem_path and os.path.exists(dem_path):
                cam_ground_z = _query_dem_elevation(
                    dem_path, x0, y0, utm)
                if cam_ground_z is not None:
                    cam_abs_z = cam_ground_z + H_m
                    print(f"DEM ground elevation at camera: "
                          f"{cam_ground_z:.2f} m")
                    print(f"Camera absolute elevation: "
                          f"{cam_abs_z:.2f} m  "
                          f"(ground {cam_ground_z:.2f} m + "
                          f"height {H_m:.1f} m)\n")

            # ---- build geometry ----
            records = []
            info_lines = []
            info_lines.append("=== Camera FOV Report ===\n")
            info_lines.append(f"Location: {lat:.6f}°N, {lon:.6f}°E\n")
            info_lines.append(f"UTM CRS: {utm}\n")
            info_lines.append(f"Camera height above ground: {H_m} m\n")
            if cam_ground_z is not None:
                info_lines.append(
                    f"DEM ground elevation: {cam_ground_z:.2f} m\n")
                info_lines.append(
                    f"Camera absolute elevation: "
                    f"{cam_ground_z + H_m:.2f} m\n\n")
            else:
                info_lines.append("\n")

            for cam_id, p in enumerate(cam_params, start=1):
                wedge, hfov_rad, vfov_rad = build_wedge_polygon(
                    x0, y0, p["f_mm"], p["sw_mm"], p["sh_mm"],
                    p["H_m"], p["heading_deg"], p["depress_deg"],
                    p["range_m"])
                if wedge is None:
                    print(f"[WARN] {p['name']}: ground not visible.")
                    continue

                dof_zone_full, Hhyper_m, near_m, far_m, r0, r1 = build_dof_zone(
                    x0, y0, p["H_m"], p["f_mm"], p["N"],
                    p["coc_mm"], p["focus_m"], p["range_m"])
                dof_zone = dof_zone_full.intersection(wedge)
                visible_and_in_focus = wedge.intersection(dof_zone_full)
                best_band = build_best_focus_band(
                    x0, y0, p["H_m"], p["focus_m"],
                    p["best_band_halfwidth_m"], p["range_m"])

                p["_hfov_deg"] = math.degrees(hfov_rad)
                p["_vfov_deg"] = math.degrees(vfov_rad)
                p["_near_m"] = near_m
                p["_far_m"] = far_m

                far_txt = "∞" if math.isinf(far_m) else f"{far_m:.2f} m"
                info = (f"--- {p['name']} ---\n"
                        f"Heading: {p['heading_deg']:.2f}°  |  "
                        f"Depression: {p['depress_deg']:.2f}°\n"
                        f"HFOV: {math.degrees(hfov_rad):.2f}°  |  "
                        f"VFOV: {math.degrees(vfov_rad):.2f}°\n"
                        f"Hyperfocal: {Hhyper_m:.2f} m\n"
                        f"DoF near: {near_m:.2f} m  |  DoF far: {far_txt}\n"
                        f"Ground DoF radii: r0={r0:.2f} m, r1={r1:.2f} m\n\n")
                info_lines.append(info)
                print(info)

                records.extend([
                    {"cam_id": cam_id, "layer": "camera",
                     "name": p["name"], "geometry": geom.Point(x0, y0)},
                    {"cam_id": cam_id, "layer": "wedge",
                     "name": p["name"], "geometry": wedge},
                    {"cam_id": cam_id, "layer": "dof_zone",
                     "name": p["name"], "geometry": dof_zone},
                    {"cam_id": cam_id, "layer": "visible_in_focus",
                     "name": p["name"], "geometry": visible_and_in_focus},
                ])
                if best_band is not None:
                    records.append({"cam_id": cam_id, "layer": "best_band",
                                    "name": p["name"], "geometry": best_band})

            if not records:
                self._ui_message("warning", "Warning",
                                 "No visible ground for any camera.")
                return

            # overlap info
            if n_cams > 1:
                info_lines.append("=== Adjacent Overlap ===\n")
                overlap_frac = clamp01(overlap_pct / 100.0)
                for i in range(len(cam_params) - 1):
                    h1 = math.degrees(fov_radians(
                        cam_params[i]["sw_mm"], cam_params[i]["f_mm"]))
                    h2 = math.degrees(fov_radians(
                        cam_params[i + 1]["sw_mm"], cam_params[i + 1]["f_mm"]))
                    oa = overlap_frac * min(h1, h2)
                    sep = (h1 / 2 + h2 / 2) - oa
                    line = (f"{cam_params[i]['name']} ↔ "
                            f"{cam_params[i + 1]['name']}: "
                            f"overlap≈{oa:.2f}° | sep≈{sep:.2f}°\n")
                    info_lines.append(line)
                    print(line)

            gdf = gpd.GeoDataFrame(records, crs=utm)

            # ---- collect wedge geometries for FOV-restricted viewshed ----
            from shapely.ops import unary_union
            wedge_geoms = [r["geometry"] for r in records
                           if r["layer"] == "wedge"]
            wedge_union = unary_union(wedge_geoms) if wedge_geoms else None

            # ---- optional viewshed ----
            vis_mask = None
            fov_mask = None
            has_dem = dem_path and os.path.exists(dem_path)
            pad = range_m * 1.3
            bounds = (x0 - pad, y0 - pad, x0 + pad, y0 + pad)

            # resolve viewshed pixel resolution
            vs_px = self._resolve_viewshed_px(cfg.get("viewshed_px", "auto"), range_m, lon, lat, dem_path)

            if has_dem:
                print(f"Computing viewshed from DEM ({vs_px}×{vs_px} grid, "
                      f"this may take a moment)…")
                vis_mask, cam_ground_z_vs = compute_viewshed_mask(
                    dem_path, x0, y0, H_m, utm, bounds,
                    width_px=vs_px, height_px=vs_px)
                # use DEM-derived ground elevation if not already set
                if cam_ground_z is None and cam_ground_z_vs > 0:
                    cam_ground_z = cam_ground_z_vs
                if vis_mask is not None:
                    cam_eye = (cam_ground_z or 0.0) + H_m
                    print(f"DEM ground elevation at camera: "
                          f"{cam_ground_z or 0:.1f} m"
                          f"  →  camera eye: {cam_eye:.1f} m")
                    # restrict viewshed to within camera FOV wedges
                    if wedge_union is not None:
                        print("Restricting viewshed to camera FOV…")
                        fov_mask = _rasterise_fov_mask(
                            wedge_union, bounds, vs_px, vs_px)
                        vis_mask_fov = vis_mask & fov_mask
                    else:
                        vis_mask_fov = vis_mask
                    print("Viewshed computed — masked areas are not visible.")
                else:
                    vis_mask_fov = None
                    print("[WARN] Viewshed returned None; using flat ground.")
            else:
                vis_mask_fov = None

            # ---- get DEM elevation range for colorbar (always, if DEM exists) ----
            _dem_elev_range = [None, None]
            if has_dem:
                e_lo, e_hi = _get_dem_elevation_range(
                    dem_path, bounds, utm)
                if e_lo is not None:
                    _dem_elev_range = [e_lo, e_hi]

            result = {
                "gdf": gdf,
                "utm": utm,
                "bounds": bounds,
                "x0": x0,
                "y0": y0,
                "pad": pad,
                "range_m": range_m,
                "cam_params": cam_params,
                "f_mm": f_mm,
                "N_ap": N_ap,
                "H_m": H_m,
                "cam_ground_z": cam_ground_z,
                "has_dem": has_dem,
                "vis_mask": vis_mask,
                "vis_mask_fov": vis_mask_fov,
                "fov_mask": fov_mask,
                "vs_px": vs_px,
                "dem_elev_range": _dem_elev_range,
                "info_lines": info_lines,
                "basemap_path": basemap_path,
                "dem_path": dem_path,
                "output_folder": output_folder,
            }
            self._ui_call(self._apply_generate_results, result)

        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback
            traceback.print_exc()
            self._ui_message("error", "Error", str(e))

    def _apply_generate_results(self, result):
        gdf = result["gdf"]
        utm = result["utm"]
        bounds = result["bounds"]
        x0 = result["x0"]
        y0 = result["y0"]
        pad = result["pad"]
        range_m = result["range_m"]
        cam_params = result["cam_params"]
        f_mm = result["f_mm"]
        N_ap = result["N_ap"]
        H_m = result["H_m"]
        cam_ground_z = result["cam_ground_z"]
        has_dem = result["has_dem"]
        vis_mask = result["vis_mask"]
        vis_mask_fov = result["vis_mask_fov"]
        fov_mask = result["fov_mask"]
        vs_px = result["vs_px"]
        _dem_elev_range = list(result["dem_elev_range"])
        info_lines = result["info_lines"]
        basemap_path = result["basemap_path"]
        dem_path = result["dem_path"]
        output_folder = result["output_folder"]

        self.fig.clear()
        if has_dem and vis_mask is not None:
            self.ax = self.fig.add_subplot(1, 2, 1)
            self.ax_vs = self.fig.add_subplot(1, 2, 2)
        else:
            self.ax = self.fig.add_subplot(1, 1, 1)
            self.ax_vs = None

        cycle_colors = plt.rcParams["axes.prop_cycle"].by_key().get(
            "color", ["C0", "C1", "C2", "C3", "C4"])

        def cam_color(cid):
            return cycle_colors[(cid - 1) % len(cycle_colors)]

        COL_IN_FOCUS = "#ff7f0e"
        IN_FOCUS_HATCH = "///"
        _bg_msg_printed = [False]

        def _draw_background(ax_target):
            bg_loaded = False
            if basemap_path and os.path.exists(basemap_path):
                bg_loaded = _load_geotiff_basemap(basemap_path, ax_target, bounds, utm)
            if not bg_loaded and has_dem:
                bg_loaded, e_lo, e_hi = _load_dem_hillshade(dem_path, ax_target, bounds, utm)
                if bg_loaded:
                    if e_lo is not None and _dem_elev_range[0] is None:
                        _dem_elev_range[0] = e_lo
                        _dem_elev_range[1] = e_hi
                    if not _bg_msg_printed[0]:
                        print("[INFO] Using DEM hillshade as background.")
                        _bg_msg_printed[0] = True
            if not bg_loaded:
                ax_target.set_facecolor("#f0f0f0")
                ring_distances = [d for d in [50, 100, 200, 500, 1000, 2000] if d <= range_m * 1.2]
                for rd in ring_distances:
                    circle = plt.Circle((x0, y0), rd, fill=False, edgecolor="#b0b0b0", linewidth=0.6, linestyle="--", zorder=0)
                    ax_target.add_patch(circle)
                    ax_target.text(x0 + rd * 0.71, y0 + rd * 0.71, f"{rd} m", fontsize=7, color="#888888", ha="left", va="bottom", zorder=0)
                if not _bg_msg_printed[0]:
                    print("[INFO] No basemap loaded — showing distance grid.\n       Provide a GeoTIFF orthoimage for satellite background.")
                    _bg_msg_printed[0] = True
            return bg_loaded

        def _draw_decorations(ax_target, title):
            arr_x = x0 + pad * 0.85
            arr_y = y0 - pad * 0.7
            arr_len = pad * 0.15
            ax_target.annotate("N", xy=(arr_x, arr_y + arr_len),
                               xytext=(arr_x, arr_y),
                               arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
                               fontsize=10, ha="center", va="bottom",
                               fontweight="bold", zorder=7)
            ax_target.set_xlim(x0 - pad, x0 + pad)
            ax_target.set_ylim(y0 - pad, y0 + pad)
            ax_target.set_xlabel("Easting (m)")
            ax_target.set_ylabel("Northing (m)")
            ax_target.set_title(title)
            add_scalebar(ax_target, x_frac=0.60, y_frac=0.06, align="left")

        _draw_background(self.ax)
        if vis_mask_fov is not None:
            left_b, bottom_b, right_b, top_b = bounds
            mask_rgba = np.zeros((*vis_mask_fov.shape, 4), dtype=np.float32)
            mask_rgba[~vis_mask_fov & (fov_mask if fov_mask is not None else np.ones_like(vis_mask_fov, dtype=bool))] = [0, 0, 0, 0.45]
            self.ax.imshow(mask_rgba, extent=[left_b, right_b, bottom_b, top_b], origin="upper", zorder=2, aspect="equal")

        in_focus_geoms = []
        for cam_id in sorted(gdf["cam_id"].unique()):
            col = cam_color(cam_id)
            sub = gdf[gdf.cam_id == cam_id]
            sub[sub.layer == "dof_zone"].plot(ax=self.ax, color=col, alpha=0.15, edgecolor="none", zorder=3)
            sub[sub.layer == "wedge"].plot(ax=self.ax, facecolor="none", edgecolor=col, linewidth=1.0, zorder=5)
            best = sub[sub.layer == "best_band"]
            if len(best) > 0:
                best.plot(ax=self.ax, color=col, alpha=0.6, edgecolor="none", zorder=4)
            vif = sub[sub.layer == "visible_in_focus"]
            if len(vif) > 0:
                in_focus_geoms.extend(vif.geometry.tolist())

        if in_focus_geoms:
            union_geom = unary_union(in_focus_geoms)
            gs = gpd.GeoSeries([union_geom], crs=utm)
            gs.plot(ax=self.ax, color=COL_IN_FOCUS, alpha=0.18, edgecolor=COL_IN_FOCUS, linewidth=0.0, hatch=IN_FOCUS_HATCH, zorder=3)

        self.ax.scatter([x0], [y0], marker="x", s=220, linewidths=3.0, color="red", zorder=6)
        _draw_decorations(self.ax, "Camera Field of View")

        if self.ax_vs is not None and vis_mask is not None:
            _draw_background(self.ax_vs)
            left_b, bottom_b, right_b, top_b = bounds
            vs_rgba = np.zeros((*vis_mask.shape, 4), dtype=np.float32)
            fov = fov_mask if fov_mask is not None else np.ones_like(vis_mask, dtype=bool)
            vs_rgba[vis_mask & fov] = [0.2, 0.8, 0.2, 0.35]
            vs_rgba[~vis_mask & fov] = [0.8, 0.1, 0.1, 0.40]
            self.ax_vs.imshow(vs_rgba, extent=[left_b, right_b, bottom_b, top_b], origin="upper", zorder=2, aspect="equal")
            for cam_id in sorted(gdf["cam_id"].unique()):
                col = cam_color(cam_id)
                sub = gdf[(gdf.cam_id == cam_id) & (gdf.layer == "wedge")]
                sub.plot(ax=self.ax_vs, facecolor="none", edgecolor=col, linewidth=0.6, linestyle="--", alpha=0.5, zorder=4)
            self.ax_vs.scatter([x0], [y0], marker="x", s=220, linewidths=3.0, color="red", zorder=6)
            _draw_decorations(self.ax_vs, "Viewshed (within FOV)")

        layer_handles = [
            Patch(facecolor=COL_IN_FOCUS, edgecolor=COL_IN_FOCUS, hatch=IN_FOCUS_HATCH, alpha=0.18, label="Visible & in focus"),
            Line2D([0], [0], marker="x", color="red", linestyle="None", markersize=10, label="Camera location"),
        ]
        if self.ax_vs is not None and vis_mask is not None:
            layer_handles.extend([
                Patch(facecolor=(0.2, 0.8, 0.2, 0.35), edgecolor="none", label="Viewshed: visible"),
                Patch(facecolor=(0.8, 0.1, 0.1, 0.40), edgecolor="none", label="Viewshed: blocked"),
            ])
        camera_handles = []
        for cam_id in sorted(gdf["cam_id"].unique()):
            p = cam_params[cam_id - 1]
            label = f"{p['name']} | Head={p['heading_deg']:.1f}° | F={f_mm}mm | f/{N_ap}"
            camera_handles.append(Line2D([0], [0], color=cam_color(cam_id), linewidth=3, label=label))
        all_handles = layer_handles + camera_handles

        if _dem_elev_range[0] is not None and _dem_elev_range[1] is not None:
            import matplotlib.cm as cm
            from matplotlib.colors import Normalize
            e_lo, e_hi = _dem_elev_range
            norm = Normalize(vmin=e_lo, vmax=e_hi)
            sm = cm.ScalarMappable(cmap=plt.cm.terrain, norm=norm)
            sm.set_array([])
            cbar = self.fig.colorbar(sm, ax=self.ax, orientation="vertical", fraction=0.03, pad=0.02, shrink=0.6)
            cbar.set_label("Elevation (m)", fontsize=8)
            cbar.ax.tick_params(labelsize=7)

        if self.ax_vs is not None:
            self.fig.subplots_adjust(left=0.07, right=0.97, bottom=0.10, top=0.90, wspace=0.22)
        else:
            self.fig.subplots_adjust(left=0.08, right=0.97, bottom=0.10, top=0.90)

        self.canvas_plot.draw()

        for w in self.legend_content.winfo_children():
            w.destroy()

        legend_items = [("■ Visible & in focus", COL_IN_FOCUS), ("✕ Camera location", "#ff0000")]
        if self.ax_vs is not None:
            legend_items.append(("■ Viewshed: visible", "#33cc33"))
            legend_items.append(("■ Viewshed: blocked", "#cc1a1a"))
        if _dem_elev_range[0] is not None:
            e_lo, e_hi = _dem_elev_range
            legend_items.append((f"Elevation: {e_lo:.1f} – {e_hi:.1f} m", "#aaaaaa"))
        if cam_ground_z is not None:
            legend_items.append((f"Cam ground: {cam_ground_z:.1f} m\nCam absolute: {cam_ground_z + H_m:.1f} m", "#dddddd"))
        legend_items.append((f"Viewshed grid: {vs_px}×{vs_px} px", "#888888"))
        for cam_id in sorted(gdf["cam_id"].unique()):
            p = cam_params[cam_id - 1]
            col = cam_color(cam_id)
            txt = f"━ {p['name']}\n   Head={p['heading_deg']:.1f}°  F={f_mm}mm  f/{N_ap}"
            legend_items.append((txt, col))

        for txt, color in legend_items:
            ctk.CTkLabel(self.legend_content, text=txt, text_color=color, font=("Arial", 11), justify="left", anchor="w").pack(anchor="w", pady=2)

        if output_folder:
            plot_path = os.path.join(output_folder, "fov_map.png")
            leg = self.fig.legend(handles=all_handles, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=min(3, len(all_handles)), frameon=True, framealpha=0.95, fontsize="small")
            self.fig.savefig(plot_path, dpi=200, bbox_extra_artists=(leg,), bbox_inches="tight")
            leg.remove()
            self.canvas_plot.draw()

            txt_path = os.path.join(output_folder, "fov_report.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.writelines(info_lines)

            print(f"\nPlot saved to: {plot_path}")
            print(f"Report saved to: {txt_path}")

            if HAS_RASTERIO and vis_mask is not None and fov_mask is not None:
                from rasterio.transform import from_bounds as tf_from_bounds
                vs_path = os.path.join(output_folder, "viewshed_mask.tif")
                left_b, bottom_b, right_b, top_b = bounds
                h_px, w_px = vis_mask.shape
                out_arr = np.full((h_px, w_px), 255, dtype=np.uint8)
                out_arr[fov_mask & vis_mask] = 1
                out_arr[fov_mask & ~vis_mask] = 0
                vs_transform = tf_from_bounds(left_b, bottom_b, right_b, top_b, w_px, h_px)
                with rasterio.open(vs_path, "w", driver="GTiff", height=h_px, width=w_px, count=1, dtype="uint8", crs=utm, transform=vs_transform, nodata=255, compress="deflate") as dst:
                    dst.write(out_arr, 1)
                    dst.update_tags(DESCRIPTION="Viewshed mask from GeoCamPal FOV Generator", ENCODING="1=visible, 0=blocked, 255=outside_FOV")
                print(f"Viewshed GeoTIFF saved to: {vs_path}")
                print(f"  CRS: {utm}")
                print(f"  Grid: {w_px}×{h_px} px")
                print(f"  Values: 1=visible, 0=blocked, 255=outside FOV (nodata)")
            else:
                vs_path = None

            msg = f"FOV map saved to:\n{plot_path}\n\nReport saved to:\n{txt_path}"
            if vs_path:
                msg += f"\n\nViewshed GeoTIFF saved to:\n{vs_path}"
            self._ui_message("info", "Done", msg)
        else:
            print("\n[INFO] No output folder selected — map displayed but not saved.")



# ——— standalone ———
if __name__ == "__main__":
    root = ctk.CTk()
    root.withdraw()
    FOVGeneratorWindow(master=root)
    root.mainloop()