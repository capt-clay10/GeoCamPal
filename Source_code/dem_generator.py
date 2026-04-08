"""
GeoCamPal — Create DEMs from shoreline GeoJSON files and water‑level CSV.

Uses PCA‑aligned cross‑shore transect interpolation (waterline method)
to avoid Delaunay triangulation artefacts.
"""

# %% ————————————————————————————— imports —————————————————————————————
from rasterio.transform import from_origin
import rasterio
from rasterio import features
from shapely.geometry import LineString, Polygon, MultiLineString
from shapely.ops import unary_union
from pathlib import Path
import re
import sys
import os
import time
import threading
import pandas as pd
import geopandas as gpd
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from scipy.spatial import ConvexHull
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.dates as mdates
import matplotlib

from utils import (
    fit_geometry, resource_path, setup_console, restore_console,
    save_settings_json, load_settings_json, compute_eta, format_eta,
)
matplotlib.use("Agg")
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")

# %% ————————————————————————————— main window ——————————————————————————
class CreateDemWindow(ctk.CTkToplevel):
    """
    GUI window that creates Digital Elevation Models (DEMs)
    from shoreline GeoJSON files and a water‑level CSV.

    Interpolation uses PCA‑aligned cross‑shore transects to
    avoid Delaunay triangulation artefacts at contour edges.
    """
    # —————————————————————————— init & UI ——————————————————————————
    def __init__(self, master=None, *args, **kwargs):
        super().__init__(master=master, *args, **kwargs)
        self.title("Create DEM")
        #self.geometry("1200x700")
        ctk.set_widget_scaling(0.9)
        #self.resizable(True, True)
        fit_geometry(self,1200,700,resizable=True)
        try:
            self.after(200, lambda: self.iconphoto(False, tk.PhotoImage(file=resource_path("launch_logo.png"))))
        except Exception:
            pass  # .ico may not exist on all platforms

        # ——— close handler ———
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # cached data
        self.df_wl_1min: pd.DataFrame | None = None
        self.shorelines_gdf: gpd.GeoDataFrame | None = None
        self.daily_dates: list = []
        self.current_day_index: int = 0

        # PCA cache (computed once per dataset)
        self._pca_center: np.ndarray | None = None
        self._pca_along: np.ndarray | None = None
        self._pca_cross: np.ndarray | None = None
        self._cancel_flag = False
        self._batch_thread = None

        # UI state variables
        self.export_xyz_var = tk.BooleanVar(value=False)
        self.beach_shape_var = tk.StringVar(value="Straight")

        # default filename pattern
        DEFAULT_PATTERN = (
            r"(?P<year>\d{4})_(?P<month>\d{2})_(?P<day>\d{2})_"
            r"(?P<hour1>\d{2})_(?P<min1>\d{2})-(?P<hour2>\d{2})_(?P<min2>\d{2})_"
            r"cam[A-Za-z0-9]+_v\d+_HDR\.geojson"
        )
        self.regex_var = tk.StringVar(value=DEFAULT_PATTERN)

        self._create_top_panel()
        self._create_bottom_panel()

        # console
        console_frame = ctk.CTkFrame(self)
        console_frame.pack(side="bottom", fill="both", expand=False, padx=5, pady=5)
        self.console_text = tk.Text(console_frame, wrap="word", height=10)
        self.console_text.pack(fill="both", expand=True, padx=5, pady=5)
        self._console_redir = setup_console(
            self.console_text,
            "Here you may see console outputs\n--------------------------------\n",
        )

    # ————————————————————————— UI helpers ——————————————————————————————
    def _create_top_panel(self):
        self.top_frame = ctk.CTkFrame(self, fg_color="black")
        self.top_frame.pack(side="top", fill="both", expand=True)

        self.top_left_container = ctk.CTkFrame(self.top_frame)
        self.top_left_container.pack(side="left", fill="both", expand=True, padx=5, pady=5)

        self.top_left_up_frame = ctk.CTkFrame(self.top_left_container)
        self.top_left_up_frame.pack(side="top", fill="both", expand=True)
        self.top_left_up_label = ctk.CTkLabel(
            self.top_left_up_frame,
            text="Water‑level timeseries + shoreline availability")
        self.top_left_up_label.pack(fill="both", expand=True)

        self.top_left_down_frame = ctk.CTkFrame(self.top_left_container)
        self.top_left_down_frame.pack(side="top", fill="both", expand=True)
        self.top_left_down_label = ctk.CTkLabel(
            self.top_left_down_frame,
            text="Daily water‑level and shoreline availability")
        self.top_left_down_label.pack(fill="both", expand=True)

        self.top_right_frame = ctk.CTkFrame(self.top_frame)
        self.top_right_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        self.top_right_label = ctk.CTkLabel(self.top_right_frame, text="Daily DEM")
        self.top_right_label.pack(fill="both", expand=True)

        self.progress_bar = None
        self.progress_label = None

    def _create_bottom_panel(self):
        self.bottom_frame = ctk.CTkFrame(self)
        self.bottom_frame.pack(side="bottom", fill="x", expand=False, padx=5, pady=15)
    
        def left_row(parent, pady=4):
            outer = ctk.CTkFrame(parent, fg_color="transparent")
            outer.pack(side="top", fill="x", padx=5, pady=pady)
    
            row = ctk.CTkFrame(outer, fg_color="transparent")
            row.pack(side="left", anchor="w")
            return row
    
        # ————————————— inputs row —————————————
        inputs = left_row(self.bottom_frame)
    
        self.wl_csv_var = tk.StringVar()
        ctk.CTkEntry(inputs, textvariable=self.wl_csv_var, width=240).pack(side="left", padx=5)
        ctk.CTkButton(
            inputs,
            text="Browse Water‑Level CSV",
            command=self.browse_wl_csv
        ).pack(side="left", padx=5)
    
        self.geojson_dir_var = tk.StringVar()
        ctk.CTkEntry(inputs, textvariable=self.geojson_dir_var, width=240).pack(side="left", padx=5)
        ctk.CTkButton(
            inputs,
            text="Browse GeoJSON Folder",
            command=self.browse_geojson_dir
        ).pack(side="left", padx=5)
    
        ctk.CTkLabel(inputs, text="Filename pattern:").pack(side="left", padx=(15, 5))
        ctk.CTkEntry(inputs, textvariable=self.regex_var, width=320).pack(side="left", padx=5)
    
        # ————————————— DEM settings row —————————————
        dem = left_row(self.bottom_frame)
    
        ctk.CTkLabel(dem, text="Resolution (m):").pack(side="left", padx=5)
        self.resolution_var = tk.StringVar(value="1")
        ctk.CTkEntry(dem, textvariable=self.resolution_var, width=60).pack(side="left", padx=5)
    
        ctk.CTkLabel(dem, text="Vertex spacing (m):").pack(side="left", padx=(10, 5))
        self.spacing_var = tk.StringVar(value="1")
        ctk.CTkEntry(dem, textvariable=self.spacing_var, width=60).pack(side="left", padx=5)
    
        ctk.CTkLabel(dem, text="Smoothing σ:").pack(side="left", padx=(10, 5))
        self.sigma_var = tk.StringVar(value="1.5")
        ctk.CTkEntry(dem, textvariable=self.sigma_var, width=60).pack(side="left", padx=5)
    
        ctk.CTkLabel(dem, text="Beach shape:").pack(side="left", padx=(10, 5))
        ctk.CTkOptionMenu(
            dem,
            variable=self.beach_shape_var,
            values=["Straight", "Curved"],
            width=110,
        ).pack(side="left", padx=5)
    
        ctk.CTkButton(
            dem,
            text="Generate next DEM",
            command=self.on_generate_next_dem,
            fg_color="#0F52BA", hover_color="#3A7AE0",
        ).pack(side="left", padx=(15, 5))
    
        # ————————————— output row —————————————
        out = left_row(self.bottom_frame)
    
        self.out_dir_var = tk.StringVar()
        ctk.CTkEntry(out, textvariable=self.out_dir_var, width=240).pack(side="left", padx=5)
        ctk.CTkButton(
            out,
            text="Browse Output Folder",
            command=self.browse_out_dir,
            fg_color="#8C7738", hover_color="#B19749"
        ).pack(side="left", padx=5)
    
        self.out_dir_display_label = ctk.CTkLabel(out, text="", width=240, anchor="w")
        self.out_dir_display_label.pack(side="left", padx=5)
    
        ctk.CTkCheckBox(
            out,
            text="Export XYZ?",
            variable=self.export_xyz_var
        ).pack(side="left", padx=(20, 5))
    
        ctk.CTkButton(
            out,
            text="Batch process",
            command=self.on_batch_process,
            fg_color="#0F52BA", hover_color="#3A7AE0"
        ).pack(side="left", padx=(25, 5))

        ctk.CTkButton(
            out,
            text="Cancel batch",
            command=self._cancel_batch,
            fg_color="#8B0000",
            hover_color="#A52A2A",
        ).pack(side="left", padx=5)
    
        ctk.CTkButton(
            out,
            text="Reset",
            command=self.reset_to_initial,
            fg_color="red",
            hover_color="#A52A2A",
            text_color="white"
        ).pack(side="left", padx=10)

        ctk.CTkButton(
            out, text="Save Settings", fg_color="#4F5D75",hover_color="#6C7C97",
            command=self._save_settings,
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            out, text="Load Settings", fg_color="#4F5D75",hover_color="#6C7C97",
            command=self._load_settings,
        ).pack(side="left", padx=5)

    # ————————————————————————— file‑dialog callbacks ——————————————————
    def browse_wl_csv(self):
        path = filedialog.askopenfilename(parent = self,filetypes=[("CSV files", "*.csv")])
        if path:
            self.wl_csv_var.set(path)

    def browse_geojson_dir(self):
        path = filedialog.askdirectory(parent = self)
        if path:
            self.geojson_dir_var.set(path)
            self._invalidate_caches()

    def browse_out_dir(self):
        path = filedialog.askdirectory(parent = self)
        if path:
            self.out_dir_var.set(path)
            self.out_dir_display_label.configure(text=path)



    # ————————————————————————— reset ———————————————————————————————————
    def _invalidate_caches(self):
        """Clear all cached data so the next run reloads everything."""
        self.shorelines_gdf = None
        self.daily_dates = []
        self.current_day_index = 0
        self.df_wl_1min = None
        self._pca_center = None
        self._pca_along = None
        self._pca_cross = None

    def reset_to_initial(self):
        """Full reset: cancel batch, clear caches, restore plots to placeholders."""
        self._cancel_flag = True
        self._invalidate_caches()

        # restore plot labels to defaults
        self.top_left_up_label.configure(
            image=None,
            text="Water‑level timeseries + shoreline availability")
        self.top_left_down_label.configure(
            image=None,
            text="Daily water‑level and shoreline availability")
        self.top_right_label.configure(
            image=None,
            text="Daily DEM")

        # clear path displays and entries
        self.wl_csv_var.set("")
        self.geojson_dir_var.set("")
        self.out_dir_var.set("")
        self.out_dir_display_label.configure(text="")
        self.export_xyz_var.set(False)
        self.resolution_var.set("1")
        self.spacing_var.set("1")
        self.sigma_var.set("1.5")
        self.beach_shape_var.set("Straight")

        self.console_text.delete("1.0", tk.END)
        print("--- Session reset. All caches cleared. ---\n")

    # ————————————————————————— data loading & plotting ————————————————
    def load_data_if_needed(self):
        if self.shorelines_gdf is not None:
            return  # already loaded

        # water‑level CSV
        csv_path = self.wl_csv_var.get().strip()
        if not csv_path or not os.path.isfile(csv_path):
            messagebox.showerror("Error", "Please specify a valid water‑level CSV.")
            raise ValueError("No CSV")

        df = pd.read_csv(csv_path)
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)
        self.df_wl_1min = df.resample("1min").interpolate()
        self.df_wl_1min.index = self.df_wl_1min.index.tz_localize(None)

        # GeoJSON folder
        folder = self.geojson_dir_var.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showerror("Error", "Please specify a valid GeoJSON folder.")
            raise ValueError("No folder")

        try:
            pattern = re.compile(self.regex_var.get().strip())
        except re.error as err:
            messagebox.showerror("Regex error", f"Invalid filename pattern:\n{err}")
            raise

        shore_gdfs = []
        for gj in sorted(Path(folder).glob("*.geojson")):
            m = pattern.search(gj.name)
            if not m:
                continue
            gd = m.groupdict()
            year = int(gd.get("year"))
            month = int(gd.get("month"))
            day = int(gd.get("day"))
            h1 = int(gd.get("hour1"))
            min1 = int(gd.get("min1"))
            h2 = int(gd.get("hour2", h1))
            min2 = int(gd.get("min2", min1))
            mid_total_min = (h1*60 + min1 + h2*60 + min2) // 2
            mid_hour, mid_min = divmod(mid_total_min, 60)
            ts = pd.to_datetime(f"{year}-{month}-{day} {mid_hour}:{mid_min}:00")
            gdf = gpd.read_file(gj)
            if gdf.empty:
                continue
            gdf["time"] = ts.tz_localize(None)
            gdf["filename"] = gj.name
            shore_gdfs.append(gdf)

        if not shore_gdfs:
            messagebox.showwarning("Warning", "No GeoJSONs matched the pattern.")
            raise ValueError("No geojson data")

        combined = gpd.GeoDataFrame(pd.concat(shore_gdfs, ignore_index=True),
                                    crs=shore_gdfs[0].crs)
        combined["date"] = combined["time"].dt.date
        self.shorelines_gdf = combined
        self.daily_dates = sorted(combined["date"].unique())
        self.current_day_index = 0

        # compute PCA directions from the full dataset
        self._compute_pca_directions()

        self.plot_waterlevel_overlay()

    # ————————————————— PCA shore orientation ——————————————————————————
    def _pca_from_xy(self, xy: np.ndarray):
        """Return (center, along, cross) from an Nx2 array of XY points."""
        if xy is None or len(xy) < 3:
            return (np.array([0.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0]))

        xy = np.asarray(xy, dtype=float)[:, :2]
        center = xy.mean(axis=0)
        cov = np.cov(xy.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx_sort = np.argsort(eigvals)[::-1]

        along = eigvecs[:, idx_sort[0]].astype(float)
        cross = eigvecs[:, idx_sort[1]].astype(float)

        along_norm = np.linalg.norm(along)
        cross_norm = np.linalg.norm(cross)
        if along_norm < 1e-12 or cross_norm < 1e-12:
            return (center, np.array([0.0, 1.0]), np.array([1.0, 0.0]))

        along /= along_norm
        cross /= cross_norm
        return center, along, cross

    def _align_local_axes(self, along: np.ndarray, cross: np.ndarray):
        """Keep local PCA axes sign-consistent with the dataset-wide PCA axes."""
        if self._pca_along is not None and float(np.dot(along, self._pca_along)) < 0:
            along = -along
        if self._pca_cross is not None and float(np.dot(cross, self._pca_cross)) < 0:
            cross = -cross
        return along, cross

    def _compute_pca_directions(self):
        """
        Determine the along‑shore / cross‑shore principal directions
        from all shoreline vertices using PCA.  Computed once per dataset.
        """
        all_xy = []
        for _, row in self.shorelines_gdf.iterrows():
            geom = row.geometry
            if geom is None:
                continue
            if geom.geom_type == "LineString":
                all_xy.extend(geom.coords)
            elif geom.geom_type == "MultiLineString":
                for part in geom.geoms:
                    all_xy.extend(part.coords)
        if len(all_xy) < 3:
            print("Warning: too few shoreline points for PCA, "
                  "falling back to X = cross‑shore, Y = along‑shore.")
            self._pca_center = np.array([0.0, 0.0])
            self._pca_along = np.array([0.0, 1.0])
            self._pca_cross = np.array([1.0, 0.0])
            return

        self._pca_center, self._pca_along, self._pca_cross = self._pca_from_xy(
            np.array(all_xy)[:, :2]
        )
        print(f"PCA along‑shore direction: {self._pca_along}")
        print(f"PCA cross‑shore direction: {self._pca_cross}")

    # ————————————————— plotting helpers ———————————————————————————————
    def update_figure_in_label(self, label: ctk.CTkLabel, fig):
        import io
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        im = Image.open(buf)
        label.configure(image=ctk.CTkImage(light_image=im, dark_image=im,
                                           size=(im.width, im.height)))
        label.image = label.cget("image")

    def plot_waterlevel_overlay(self):
        if self.df_wl_1min is None or self.shorelines_gdf is None:
            return
        wl = self.df_wl_1min
        times = sorted(self.shorelines_gdf["time"].unique())
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(wl.index, wl["wl"], label="Water level")
        ax.scatter(times, [wl["wl"].asof(t) for t in times],
                   c="r", s=10, label="Shorelines")
        ax.set_title("WL timeseries + shoreline availability")
        ax.set_xlabel("Time"); ax.set_ylabel("WL (m)")
        ax.tick_params(axis="x", labelrotation=45)
        ax.legend(); fig.tight_layout()
        self.update_figure_in_label(self.top_left_up_label, fig)
        plt.close(fig)

    def plot_daily_view(self, date_val):
        wl = self.df_wl_1min; gdf = self.shorelines_gdf
        if wl is None or gdf is None:
            return
        gdf_day = gdf[gdf["date"] == date_val]
        subset = wl[gdf_day["time"].min(): gdf_day["time"].max()]
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(subset.index, subset["wl"], label="Daily WL")
        ax.scatter(gdf_day["time"],
                   [wl["wl"].asof(t) for t in gdf_day["time"]],
                   c="m", s=15, label="Shorelines")
        idx = self.daily_dates.index(date_val)
        ax.set_title(f"Day {idx+1} of {len(self.daily_dates)} — {date_val}")
        ax.set_xlabel("Hour"); ax.set_ylabel("WL (m)")
        ax.legend(); ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        fig.tight_layout()
        self.update_figure_in_label(self.top_left_down_label, fig)
        plt.close(fig)

    def plot_dem(self, grid_z):
        if grid_z is None or np.all(np.isnan(grid_z)):
            return
        fig, ax = plt.subplots(figsize=(6, 8))
        im = ax.imshow(grid_z, cmap="viridis", aspect="auto")
        fig.colorbar(im, ax=ax, label="Elevation (m)")
        ax.set_title("Created DEM"); fig.tight_layout()
        self.update_figure_in_label(self.top_right_label, fig)
        plt.close(fig)

    # ————————————————————— densification helper ———————————————————————
    def densify_geometry(self, geom, spacing: float):
        """Return equally spaced coordinates along a geometry."""
        coords = []

        def _densify_line(line: LineString):
            length = line.length
            if length == 0:
                return []
            n = max(int(np.floor(length / spacing)), 1)
            dists = np.linspace(0, length, n + 1)
            return [line.interpolate(d).coords[0] for d in dists]

        if geom.geom_type == "LineString":
            coords.extend(_densify_line(geom))
        elif geom.geom_type == "MultiLineString":
            for part in geom.geoms:
                coords.extend(_densify_line(part))
        elif geom.geom_type == "Polygon":
            coords.extend(_densify_line(LineString(geom.exterior.coords)))
        return coords

    # ————————————————————— transect interpolation —————————————————————
    def _interpolate_transects(self, dense_shorelines, grid_x, grid_y,
                               grid_x_vals, grid_y_vals, res):
        """Dispatch DEM interpolation according to the selected beach shape."""
        beach_shape = self.beach_shape_var.get().strip().lower()
        if beach_shape == "curved":
            return self._interpolate_transects_curved(
                dense_shorelines, grid_x, grid_y, grid_x_vals, grid_y_vals, res
            )
        return self._interpolate_transects_straight(
            dense_shorelines, grid_x, grid_y, grid_x_vals, grid_y_vals, res
        )

    def _interpolate_transects_straight(self, dense_shorelines, grid_x, grid_y,
                                        grid_x_vals, grid_y_vals, res):
        """
        Cross‑shore transect interpolation aligned to one global PCA direction.
        Best suited to beaches whose shoreline direction is approximately constant.
        """
        center = self._pca_center
        along_dir = self._pca_along
        cross_dir = self._pca_cross

        # project all dense shoreline coordinates into rotated frame
        all_along = []
        for sl in dense_shorelines:
            proj = (sl["coords"] - center) @ along_dir
            all_along.append(proj)
        all_along_flat = np.concatenate(all_along)
        a_min, a_max = all_along_flat.min(), all_along_flat.max()

        # step along‑shore at the grid resolution
        along_vals = np.arange(a_min, a_max + res, res)

        # pre‑project each shoreline into along/cross coords
        sl_projections = []
        for sl in dense_shorelines:
            diff = sl["coords"] - center
            sl_along = diff @ along_dir
            sl_cross = diff @ cross_dir
            sl_projections.append((sl_along, sl_cross, sl["wl"]))

        # determine cross‑shore extent
        all_cross = np.concatenate([p[1] for p in sl_projections])
        c_min, c_max = all_cross.min() - 2, all_cross.max() + 2
        cross_vals = np.arange(c_min, c_max + res, res)

        dem_rotated = np.full((len(along_vals), len(cross_vals)), np.nan)

        for ia, a_target in enumerate(along_vals):
            # find where each shoreline crosses this along‑shore position
            xz_pairs = []
            for sl_along, sl_cross, wl in sl_projections:
                for k in range(len(sl_along) - 1):
                    a0, a1 = sl_along[k], sl_along[k + 1]
                    if (a0 <= a_target <= a1) or (a1 <= a_target <= a0):
                        denom = a1 - a0
                        if abs(denom) < 1e-10:
                            c_interp = (sl_cross[k] + sl_cross[k + 1]) / 2
                        else:
                            t = (a_target - a0) / denom
                            c_interp = sl_cross[k] + t * (sl_cross[k + 1] - sl_cross[k])
                        xz_pairs.append((c_interp, wl))

            if len(xz_pairs) < 2:
                continue

            xz_pairs.sort(key=lambda p: p[0])
            cs = np.array([p[0] for p in xz_pairs])
            zs = np.array([p[1] for p in xz_pairs])

            # remove exact duplicates at the same cross‑shore position
            _, uidx = np.unique(cs, return_index=True)
            cs, zs = cs[uidx], zs[uidx]
            if len(cs) < 2:
                continue

            f = interp1d(cs, zs, kind="linear",
                         bounds_error=False, fill_value=np.nan)
            dem_rotated[ia, :] = f(cross_vals)

        # map rotated DEM back to geographic grid
        grid_diff_x = grid_x - center[0]
        grid_diff_y = grid_y - center[1]
        grid_a = grid_diff_x * along_dir[0] + grid_diff_y * along_dir[1]
        grid_c = grid_diff_x * cross_dir[0] + grid_diff_y * cross_dir[1]

        ia_idx = np.round((grid_a - along_vals[0]) / res).astype(int)
        ic_idx = np.round((grid_c - cross_vals[0]) / res).astype(int)

        valid = ((ia_idx >= 0) & (ia_idx < len(along_vals)) &
                 (ic_idx >= 0) & (ic_idx < len(cross_vals)))

        dem = np.full(grid_x.shape, np.nan)
        dem[valid] = dem_rotated[ia_idx[valid], ic_idx[valid]]

        return dem

    def _interpolate_transects_curved(self, dense_shorelines, grid_x, grid_y,
                                      grid_x_vals, grid_y_vals, res):
        """
        Curved-beach mode: estimate a local PCA direction in a moving window
        for each along-shore step, then cast the cross-shore transect in that
        local frame. This preserves the current waterline method, but lets the
        orientation bend with the shoreline instead of remaining globally fixed.
        """
        global_center = self._pca_center
        global_along = self._pca_along

        all_xy = np.vstack([sl["coords"] for sl in dense_shorelines])
        all_along_global = (all_xy - global_center) @ global_along
        a_min, a_max = all_along_global.min(), all_along_global.max()
        along_vals = np.arange(a_min, a_max + res, res)
        extent_m = max(float(a_max - a_min), res)
        window_m = max(25.0, 20.0 * res, 0.08 * extent_m)

        z_sum = np.zeros(grid_x.shape, dtype=float)
        z_count = np.zeros(grid_x.shape, dtype=float)

        for a_target in along_vals:
            local_mask = np.abs(all_along_global - a_target) <= (window_m / 2.0)
            local_xy = all_xy[local_mask]
            if len(local_xy) < 6:
                continue

            local_center, local_along, local_cross = self._pca_from_xy(local_xy)
            local_along, local_cross = self._align_local_axes(local_along, local_cross)

            xz_pairs = []
            local_cross_cloud = (local_xy - local_center) @ local_cross

            for sl in dense_shorelines:
                diff = sl["coords"] - local_center
                sl_along = diff @ local_along
                sl_cross = diff @ local_cross
                for k in range(len(sl_along) - 1):
                    a0, a1 = sl_along[k], sl_along[k + 1]
                    if (a0 <= 0.0 <= a1) or (a1 <= 0.0 <= a0):
                        denom = a1 - a0
                        if abs(denom) < 1e-10:
                            c_interp = (sl_cross[k] + sl_cross[k + 1]) / 2.0
                        else:
                            t = -a0 / denom
                            c_interp = sl_cross[k] + t * (sl_cross[k + 1] - sl_cross[k])
                        xz_pairs.append((c_interp, sl["wl"]))

            if len(xz_pairs) < 2:
                continue

            xz_pairs.sort(key=lambda p: p[0])
            cs = np.array([p[0] for p in xz_pairs], dtype=float)
            zs = np.array([p[1] for p in xz_pairs], dtype=float)
            _, uidx = np.unique(cs, return_index=True)
            cs, zs = cs[uidx], zs[uidx]
            if len(cs) < 2:
                continue

            c_min = min(local_cross_cloud.min(), cs.min()) - 2.0
            c_max = max(local_cross_cloud.max(), cs.max()) + 2.0
            cross_vals = np.arange(c_min, c_max + res, res)
            z_profile = interp1d(cs, zs, kind="linear", bounds_error=False, fill_value=np.nan)(cross_vals)

            line_xy = local_center[None, :] + cross_vals[:, None] * local_cross[None, :]
            x_world = line_xy[:, 0]
            y_world = line_xy[:, 1]

            ix = np.round((x_world - grid_x_vals[0]) / res).astype(int)
            iy = np.round((grid_y_vals[0] - y_world) / res).astype(int)

            valid = (~np.isnan(z_profile) &
                     (ix >= 0) & (ix < grid_x.shape[1]) &
                     (iy >= 0) & (iy < grid_x.shape[0]))
            if not np.any(valid):
                continue

            z_sum[iy[valid], ix[valid]] += z_profile[valid]
            z_count[iy[valid], ix[valid]] += 1.0

        dem = np.full(grid_x.shape, np.nan)
        valid = z_count > 0
        dem[valid] = z_sum[valid] / z_count[valid]

        print(f"Curved-beach mode used moving-window PCA (window ≈ {window_m:.1f} m).")
        return dem

    def _smooth_dem(self, dem, sigma):
        """
        Gaussian smooth the DEM while respecting NaN gaps.

        Uses a normalised‑convolution approach so that NaN cells
        do not bleed into valid data.
        """
        if sigma <= 0:
            return dem

        valid_mask = ~np.isnan(dem)
        dem_filled = np.where(valid_mask, dem, 0.0)
        smoothed = gaussian_filter(dem_filled, sigma=sigma)
        weight = gaussian_filter(valid_mask.astype(float), sigma=sigma)
        weight[weight < 0.05] = np.nan
        result = smoothed / weight
        # keep original NaN footprint
        result[~valid_mask & np.isnan(dem)] = np.nan
        return result

    # ————————————————————— DEM generation —————————————————————————————
    def create_dem_for_day(self, date_val):
        out_dir = self.out_dir_var.get().strip()
        if not out_dir:
            messagebox.showerror("Error", "Please specify an output directory.")
            return None

        gdf_day = self.shorelines_gdf[self.shorelines_gdf["date"] == date_val].copy()
        gdf_day["z"] = gdf_day["time"].apply(lambda t: self.df_wl_1min["wl"].asof(t))

        # --- gather densified shoreline data ---
        try:
            spacing = float(self.spacing_var.get())
            if spacing <= 0:
                raise ValueError
        except Exception:
            spacing = float(self.resolution_var.get() or 1)
            print(f"Invalid spacing, using {spacing} m.")

        dense_shorelines = []
        xyz_all = []
        for _, row in gdf_day.iterrows():
            geom = row.geometry
            zval = row.z
            if pd.isna(zval) or geom is None:
                continue
            pts = self.densify_geometry(geom, spacing)
            if not pts:
                continue
            coords = np.array(pts)[:, :2]
            dense_shorelines.append({"wl": zval, "coords": coords})
            for x, y in coords:
                xyz_all.append((x, y, zval))

        if not xyz_all:
            print(f"No valid shoreline points for {date_val}")
            return None

        # DEM grid set‑up
        try:
            res = float(self.resolution_var.get())
        except Exception:
            res = 1.0

        try:
            sigma = float(self.sigma_var.get())
        except Exception:
            sigma = 1.5

        xyz_arr = np.array(xyz_all)
        x_min, x_max = xyz_arr[:, 0].min(), xyz_arr[:, 0].max()
        y_min, y_max = xyz_arr[:, 1].min(), xyz_arr[:, 1].max()
        grid_x_vals = np.arange(x_min, x_max + res, res)
        grid_y_vals = np.arange(y_max, y_min - res, -res)
        if len(grid_x_vals) < 2 or len(grid_y_vals) < 2:
            print(f"Skipping day {date_val}: bbox too small.")
            return None
        grid_x, grid_y = np.meshgrid(grid_x_vals, grid_y_vals)

        # --- transect interpolation ---
        print(f"Beach shape mode: {self.beach_shape_var.get()}")
        grid_z = self._interpolate_transects(
            dense_shorelines, grid_x, grid_y, grid_x_vals, grid_y_vals, res)

        if grid_z is None or np.all(np.isnan(grid_z)):
            print(f"Interpolation failed for {date_val}")
            return None

        # --- along‑shore smoothing ---
        grid_z = self._smooth_dem(grid_z, sigma)

        # --- mask to convex hull of data points ---
        if len(xyz_all) >= 3:
            try:
                hull = ConvexHull(xyz_arr[:, :2])
                hull_path = MPath(xyz_arr[hull.vertices, :2])
                grid_pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])
                inside = hull_path.contains_points(grid_pts).reshape(grid_x.shape)
                grid_z = np.where(inside, grid_z, np.nan)
            except Exception:
                pass  # degenerate hull, skip masking

        transform = from_origin(x_min, y_max, res, res)

        dem_masked = grid_z

        # save GeoTIFF
        out_path = Path(out_dir) / f"DEM_{date_val}_transect.tif"
        with rasterio.open(
            out_path, "w", driver="GTiff",
            height=dem_masked.shape[0], width=dem_masked.shape[1],
            count=1, dtype="float64",
            crs=self.shorelines_gdf.crs,
            transform=transform) as dst:
            dst.write(dem_masked, 1)
        print(f"Saved DEM → {out_path}")

        if self.export_xyz_var.get():
            xyz_df = pd.DataFrame(xyz_all, columns=["x", "y", "z"])
            csv_path = Path(out_dir) / f"shoreline_xyz_{date_val}.csv"
            xyz_df.to_csv(csv_path, index=False)
            print(f"Exported XYZ → {csv_path}")
        return dem_masked

    # ————————————————————— UI actions —————————————————————————————————
    def on_generate_next_dem(self):
        try:
            self.load_data_if_needed()
        except Exception:
            return
        if self.current_day_index >= len(self.daily_dates):
            messagebox.showinfo("Done", "No more days left to process.")
            return
        date_val = self.daily_dates[self.current_day_index]
        self.plot_daily_view(date_val)
        dem = self.create_dem_for_day(date_val)
        self.plot_dem(dem)
        self.current_day_index += 1

    def on_batch_process(self):
        try:
            self.load_data_if_needed()
        except Exception:
            return
        out_dir = self.out_dir_var.get().strip()
        if not out_dir:
            messagebox.showerror("Error", "Please specify output folder.")
            return
        self._cancel_flag = False
        self.top_left_container.pack_forget()
        self.top_right_frame.pack_forget()
        self.show_progress_bar()
        self._batch_thread = threading.Thread(
            target=self._batch_worker, daemon=True)
        self._batch_thread.start()

    def _batch_worker(self):
        total = len(self.daily_dates)
        t0 = time.time()
        print("Batch process has started")
        for i, date_val in enumerate(self.daily_dates, 1):
            if self._cancel_flag:
                print("Batch cancelled by user.")
                self.after(0, self.hide_progress_bar)
                return
            self.after(0, self.progress_bar.set, i / total)
            eta = compute_eta(t0, i, total)
            eta_str = format_eta(eta)
            self.after(0, self.progress_label.configure,
                       {"text": f"Batch {i}/{total} — ETA {eta_str}"})
            self.create_dem_for_day(date_val)
        self.after(0, self.hide_progress_bar)
        elapsed = format_eta(time.time() - t0)
        self.after(0, lambda: messagebox.showinfo(
            "Done", f"Batch DEM creation finished ({total} days, {elapsed})."))

    def _cancel_batch(self):
        self._cancel_flag = True
        print("Cancelling batch…")

    # progress helpers
    def show_progress_bar(self):
        if self.progress_bar is None:
            self.progress_label = ctk.CTkLabel(self.top_left_container, text="Starting batch…")
            self.progress_label.pack(pady=10)
            self.progress_bar = ctk.CTkProgressBar(self.top_left_container, width=400)
            self.progress_bar.pack()
            self.progress_bar.set(0.0)

    def hide_progress_bar(self):
        if self.progress_bar:
            self.progress_bar.destroy()
            self.progress_label.destroy()
            self.progress_bar = self.progress_label = None
        self.top_left_container.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        self.top_right_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)

    # ————————————————————— save / load settings ——————————————————————————
    def _settings_payload(self):
        return {
            "paths": {
                "wl_csv": self.wl_csv_var.get(),
                "geojson_dir": self.geojson_dir_var.get(),
                "output_dir": self.out_dir_var.get(),
            },
            "parameters": {
                "resolution": self.resolution_var.get(),
                "vertex_spacing": self.spacing_var.get(),
                "smoothing_sigma": self.sigma_var.get(),
                "beach_shape": self.beach_shape_var.get(),
                "filename_pattern": self.regex_var.get(),
                "export_xyz": bool(self.export_xyz_var.get()),
            },
        }

    def _save_settings(self):
        try:
            initialdir = self.out_dir_var.get().strip() or None
            path = save_settings_json(
                self, "dem_generator", self._settings_payload(),
                initialdir=initialdir)
            if path:
                print(f"Settings saved: {path}")
        except Exception as e:
            messagebox.showerror("Save Settings",
                                 f"Could not save settings:\n{e}", parent=self)

    def _load_settings(self):
        try:
            initialdir = self.out_dir_var.get().strip() or None
            data, path = load_settings_json(
                self, "dem_generator", initialdir=initialdir)
            if not data:
                return

            paths = data.get("paths", {})
            params = data.get("parameters", {})

            if paths.get("wl_csv"):
                self.wl_csv_var.set(paths["wl_csv"])
            if paths.get("geojson_dir"):
                self.geojson_dir_var.set(paths["geojson_dir"])
            if paths.get("output_dir"):
                self.out_dir_var.set(paths["output_dir"])
                self.out_dir_display_label.configure(text=paths["output_dir"])

            if params.get("resolution"):
                self.resolution_var.set(params["resolution"])
            if params.get("vertex_spacing"):
                self.spacing_var.set(params["vertex_spacing"])
            if params.get("smoothing_sigma"):
                self.sigma_var.set(params["smoothing_sigma"])
            if params.get("beach_shape"):
                self.beach_shape_var.set(params["beach_shape"])
            if params.get("filename_pattern"):
                self.regex_var.set(params["filename_pattern"])
            self.export_xyz_var.set(bool(params.get("export_xyz", False)))

            # invalidate caches so data reloads with new paths
            self._invalidate_caches()

            print(f"Settings loaded: {path}")
        except Exception as e:
            messagebox.showerror("Load Settings",
                                 f"Could not load settings:\n{e}", parent=self)

    def _on_close(self):
        self._cancel_flag = True
        restore_console(getattr(self, "_console_redir", None))
        self.destroy()


# —————————————————————————————— entry point ———————————————————————————
def main():
    root = ctk.CTk()
    root.withdraw()
    CreateDemWindow(master=root).mainloop()


if __name__ == "__main__":
    main()