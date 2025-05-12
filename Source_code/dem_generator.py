"""
GeoCamPal – Create DEMs from shoreline GeoJSON files and water‑level CSV.

NEW 2025‑05‑12
• Adds uniform‑spacing densification so DEMs are based on evenly spaced
  sample points instead of the hand‑drawn vertices in the GeoJSON.
• Adds a "Vertex spacing (m)" entry to let the user pick the spacing.
"""

# ───────────────────────────── imports ─────────────────────────────
from rasterio.transform import from_origin
import rasterio
from rasterio import features
from shapely.geometry import LineString, Polygon, MultiLineString
from shapely.ops import unary_union
from pathlib import Path
import re
import sys
import os
import pandas as pd
import geopandas as gpd
from scipy.interpolate import griddata
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.dates as mdates
import matplotlib
from osgeo import osr
osr.DontUseExceptions()
matplotlib.use("Agg")
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")

# ───────────────────────── util helpers ────────────────────────────
def resource_path(relative_path: str) -> str:
    try:
        base_path = sys._MEIPASS  # type: ignore
    except Exception:
        base_path = os.path.dirname(__file__)
    return os.path.join(base_path, relative_path)


class StdoutRedirector:
    def __init__(self, text_widget: tk.Text):
        self.text_widget = text_widget

    def write(self, message: str):
        self.text_widget.insert(tk.END, message)
        self.text_widget.see(tk.END)

    def flush(self):
        pass


# ──────────────────────── main window ──────────────────────────────
class CreateDemWindow(ctk.CTkToplevel):
    """
    GUI window that creates Digital Elevation Models (DEMs)
    from shoreline GeoJSON files and a water‑level CSV.
    """
    # ────────────────────────── init & UI ──────────────────────────
    def __init__(self, master=None, *args, **kwargs):
        super().__init__(master=master, *args, **kwargs)
        self.title("Create DEM")
        self.geometry("1200x700")
        ctk.set_widget_scaling(0.9)
        self.resizable(True, True)
        try:
            self.iconbitmap(resource_path("launch_logo.ico"))
        except Exception as err:
            print("Warning: could not load window icon:", err)

        # cached data
        self.df_wl_1min: pd.DataFrame | None = None
        self.shorelines_gdf: gpd.GeoDataFrame | None = None
        self.daily_dates: list = []
        self.current_day_index: int = 0

        # UI state variables
        self.export_xyz_var = tk.BooleanVar(value=False)

        # default filename pattern (same as previous version)
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
        console_text = tk.Text(console_frame, wrap="word", height=10)
        console_text.pack(fill="both", expand=True, padx=5, pady=5)
        sys.stdout = StdoutRedirector(console_text)
        sys.stderr = sys.stdout
        print("Here you may see console outputs\n")

    # ───────────────────── UI helpers ───────────────────────────────
    def _create_top_panel(self):
        self.top_frame = ctk.CTkFrame(self)
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
        self.bottom_frame.pack(side="bottom", fill="x", expand=False, padx=5, pady=25, ipady=10)

        # ───────────── inputs row ─────────────
        inputs = ctk.CTkFrame(self.bottom_frame)
        inputs.pack(side="top", fill="x", padx=5, pady=5)

        ctk.CTkLabel(inputs, text="Water‑level CSV:").pack(side="left", padx=5)
        self.wl_csv_var = tk.StringVar()
        ctk.CTkEntry(inputs, textvariable=self.wl_csv_var, width=240).pack(side="left", padx=5)
        ctk.CTkButton(inputs, text="Browse", command=self.browse_wl_csv).pack(side="left", padx=5)

        ctk.CTkLabel(inputs, text="GeoJSON folder:").pack(side="left", padx=15)
        self.geojson_dir_var = tk.StringVar()
        ctk.CTkEntry(inputs, textvariable=self.geojson_dir_var, width=240).pack(side="left", padx=5)
        ctk.CTkButton(inputs, text="Browse", command=self.browse_geojson_dir).pack(side="left", padx=5)

        ctk.CTkLabel(inputs, text="Filename pattern:").pack(side="left", padx=15)
        ctk.CTkEntry(inputs, textvariable=self.regex_var, width=320).pack(side="left", padx=5)

        ctk.CTkLabel(inputs, text="Output folder:").pack(side="left", padx=15)
        self.out_dir_var = tk.StringVar()
        out_entry = ctk.CTkEntry(inputs, textvariable=self.out_dir_var, width=180)
        out_entry.pack(side="left", padx=5)
        ctk.CTkButton(inputs, text="Browse", command=self.browse_out_dir).pack(side="left", padx=5)
        self.out_dir_display_label = ctk.CTkLabel(inputs, text="", width=240)
        self.out_dir_display_label.pack(side="left", padx=5)

        # ───────────── DEM settings row ─────────────
        dem = ctk.CTkFrame(self.bottom_frame)
        dem.pack(side="top", fill="x", padx=5, pady=5)

        ctk.CTkLabel(dem, text="Resolution (m):").pack(side="left", padx=5)
        self.resolution_var = tk.StringVar(value="1")
        ctk.CTkEntry(dem, textvariable=self.resolution_var, width=60).pack(side="left", padx=5)

        ctk.CTkLabel(dem, text="Vertex spacing (m):").pack(side="left", padx=10)  # NEW
        self.spacing_var = tk.StringVar(value="1")                                 # NEW
        ctk.CTkEntry(dem, textvariable=self.spacing_var, width=60).pack(side="left", padx=5)  # NEW

        ctk.CTkLabel(dem, text="Interpolation:").pack(side="left", padx=10)
        self.interp_var = tk.StringVar(value="nearest")
        ctk.CTkOptionMenu(dem, variable=self.interp_var, values=["linear", "cubic","nearest"]).pack(side="left", padx=5)

        ctk.CTkButton(dem, text="Generate next DEM", command=self.on_generate_next_dem).pack(side="left", padx=15)

        # ───────────── output row ─────────────
        out = ctk.CTkFrame(self.bottom_frame)
        out.pack(side="top", fill="x", padx=5, pady=5)

        ctk.CTkCheckBox(out, text="Export XYZ?", variable=self.export_xyz_var,
                        command=self.on_toggle_export_xyz).pack(side="left", padx=20)
        self.xyz_folder_var = tk.StringVar()
        self.xyz_btn = ctk.CTkButton(out, text="Select XYZ folder", command=self.browse_xyz_folder)
        self.xyz_btn.pack(side="left", padx=5)
        self.xyz_display = ctk.CTkLabel(out, text="", width=240)
        self.xyz_display.pack(side="left", padx=5)
        self.xyz_btn.configure(state="disabled")

        ctk.CTkButton(out, text="Batch process", command=self.on_batch_process).pack(side="left", padx=25)

    # ───────────────────── file‑dialog callbacks ─────────────────────
    def browse_wl_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if path:
            self.wl_csv_var.set(path)

    def browse_geojson_dir(self):
        path = filedialog.askdirectory()
        if path:
            self.geojson_dir_var.set(path)
            # reset caches
            self.shorelines_gdf = None
            self.daily_dates = []
            self.current_day_index = 0

    def browse_out_dir(self):
        path = filedialog.askdirectory()
        if path:
            self.out_dir_var.set(path)
            self.out_dir_display_label.configure(text=path)

    def browse_xyz_folder(self):
        path = filedialog.askdirectory()
        if path:
            self.xyz_folder_var.set(path)
            self.xyz_display.configure(text=path)

    def on_toggle_export_xyz(self):
        state = "normal" if self.export_xyz_var.get() else "disabled"
        self.xyz_btn.configure(state=state)
        if state == "disabled":
            self.xyz_display.configure(text="")

    # ───────────────────── data loading & plotting ───────────────────
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
        self.plot_waterlevel_overlay()

    # ---------- plotting helpers (unchanged logic, condensed for brevity) ----------
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
        ax.set_title(f"Day {idx+1} of {len(self.daily_dates)} – {date_val}")
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
        fig.colorbar(im, ax=ax, label="WL (m)")
        ax.set_title("Created DEM"); fig.tight_layout()
        self.update_figure_in_label(self.top_right_label, fig)
        plt.close(fig)

    # ─────────────────── densification helper (NEW) ──────────────────
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

    # ───────────────────── DEM generation ────────────────────────────
    def create_dem_for_day(self, date_val):
        out_dir = self.out_dir_var.get().strip()
        if not out_dir:
            messagebox.showerror("Error", "Please specify an output directory.")
            return None

        gdf_day = self.shorelines_gdf[self.shorelines_gdf["date"] == date_val].copy()
        gdf_day["z"] = gdf_day["time"].apply(lambda t: self.df_wl_1min["wl"].asof(t))

        # --- gather XYZ from densified shorelines ---
        xyz = []
        try:
            spacing = float(self.spacing_var.get())
            if spacing <= 0:
                raise ValueError
        except Exception:
            spacing = float(self.resolution_var.get() or 1)
            print(f"Invalid spacing, using {spacing} m.")

        for _, row in gdf_day.iterrows():
            geom = row.geometry
            zval = row.z
            if pd.isna(zval) or geom is None:
                continue
            for x, y in self.densify_geometry(geom, spacing):
                xyz.append((x, y, zval))

        if not xyz:
            print(f"No valid shoreline points for {date_val}")
            return None

        # DEM interpolation set‑up
        try:
            res = float(self.resolution_var.get())
        except Exception:
            res = 1.0
        method = self.interp_var.get().strip().lower()
        if method not in ("linear", "cubic"):
            method = "linear"

        xyz_df = pd.DataFrame(xyz, columns=["x", "y", "z"])
        x_min, x_max = xyz_df["x"].min(), xyz_df["x"].max()
        y_min, y_max = xyz_df["y"].min(), xyz_df["y"].max()
        grid_x_vals = np.arange(x_min, x_max + res, res)
        grid_y_vals = np.arange(y_max, y_min - res, -res)
        if len(grid_x_vals) < 2 or len(grid_y_vals) < 2:
            print(f"Skipping day {date_val}: bbox too small.")
            return None
        grid_x, grid_y = np.meshgrid(grid_x_vals, grid_y_vals)
        grid_z = griddata(xyz_df[["x", "y"]].values, xyz_df["z"].values,
                          (grid_x, grid_y), method=method)

        if grid_z is None or np.all(np.isnan(grid_z)):
            print(f"Interpolation failed for {date_val}")
            return None

        transform = from_origin(x_min, y_max, res, res)

        # --- concave mask from densified coords ---
        try:
            import alphashape
        except ImportError:
            messagebox.showerror("Error", "Install alphashape: pip install alphashape")
            return None

        all_points = [(x, y) for x, y, _ in xyz]  # densified already
        if len(all_points) < 3:
            shoreline_poly = unary_union(gdf_day.geometry).convex_hull
        else:
            alpha_val = 0.02
            shoreline_poly = alphashape.alphashape(all_points, alpha_val)
            if shoreline_poly.geom_type != "Polygon":
                shoreline_poly = shoreline_poly.convex_hull

        mask = features.geometry_mask(
            [shoreline_poly.__geo_interface__],
            out_shape=grid_z.shape,
            transform=transform,
            invert=True)

        dem_masked = np.where(mask, grid_z, np.nan)

        # save GeoTIFF
        out_path = Path(out_dir) / f"DEM_{date_val}_{method}.tif"
        with rasterio.open(
            out_path, "w", driver="GTiff",
            height=dem_masked.shape[0], width=dem_masked.shape[1],
            count=1, dtype=dem_masked.dtype,
            crs=self.shorelines_gdf.crs,
            transform=transform) as dst:
            dst.write(dem_masked, 1)
        print(f"Saved DEM → {out_path}")

        if self.export_xyz_var.get():
            folder = self.xyz_folder_var.get().strip()
            if folder:
                csv_path = Path(folder) / f"shoreline_xyz_{date_val}.csv"
                xyz_df.to_csv(csv_path, index=False)
                print(f"Exported XYZ → {csv_path}")
        return dem_masked

    # ───────────────────── UI actions ───────────────────────────────
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
        self.top_left_container.pack_forget()
        self.top_right_frame.pack_forget()
        self.show_progress_bar()
        out_dir = self.out_dir_var.get().strip()
        if not out_dir:
            messagebox.showerror("Error", "Please specify output folder.")
            self.hide_progress_bar()
            return
        total = len(self.daily_dates)
        for i, date_val in enumerate(self.daily_dates, 1):
            self.progress_bar.set(i / total)
            self.progress_label.configure(text=f"Batch {i}/{total}")
            self.update_idletasks()
            self.create_dem_for_day(date_val)
        self.hide_progress_bar()
        messagebox.showinfo("Done", f"Batch DEM creation finished ({total} days).")

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


# ────────────────────────── entry point ───────────────────────────
def main():
    root = ctk.CTk()
    root.withdraw()
    CreateDemWindow(master=root).mainloop()


if __name__ == "__main__":
    main()
