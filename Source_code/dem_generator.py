from rasterio.transform import from_origin
import rasterio
from rasterio import features
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union
import json
from pathlib import Path
from scipy.interpolate import griddata
import shutil
from rasterio.transform import from_origin
import rasterio
from rasterio import features
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union
import re
import sys
import pandas as pd
import geopandas as gpd
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import os
import cv2
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import matplotlib.dates as mdates
import matplotlib
from scipy.interpolate import griddata
import threading
from pathlib import Path
from osgeo import gdal, osr
matplotlib.use("Agg")  # Use headless mode for non-interactive environments

def resource_path(relative_path: str) -> str:
    try:
        base_path = sys._MEIPASS  # If running in a PyInstaller .exe
    except Exception:
        base_path = os.path.dirname(__file__)  # Running directly from source
    return os.path.join(base_path, relative_path)


# --- StdoutRedirector class for redirecting console output into the GUI ---
class StdoutRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        self.text_widget.insert(tk.END, message)
        self.text_widget.see(tk.END)  # auto-scroll

    def flush(self):
        pass  # no-op for Python's IO requirements

class CreateDemWindow(ctk.CTk):
    """
    A top-level window for creating Digital Elevation Models (DEMs) from shoreline and water-level data.
    Provides options for single-day processing or batch processing.
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize the Create DEM window.
        """
        super().__init__(*args, **kwargs)
        self.title("Create DEM")
        self.geometry("1200x700")
        ctk.set_widget_scaling(0.9)
        self.resizable(True, True)
        # ctk.set_appearance_mode("System")
        # ctk.set_default_color_theme("blue")
        
        try:
            self.iconbitmap(resource_path("launch_logo.ico"))
        except Exception as e:
            print("Warning: Could not load window icon:", e)

        # Data references
        self.df_wl_1min = None      # 1-min interpolated water-level DataFrame
        self.shorelines_gdf = None  # Combined GeoDataFrame with shoreline data
        self.daily_dates = []       # Sorted list of unique dates from shorelines
        self.current_day_index = 0  # Pointer to next day for processing

        # Figure references for plotting
        self.fig_left = None
        self.fig_center = None
        self.fig_right = None

        self.export_xyz_var = tk.BooleanVar(value=False)

        self._create_top_panel()
        self._create_bottom_panel()
        
        # --- Console Panel at the bottom ---
        self.console_frame = ctk.CTkFrame(self)
        self.console_frame.pack(side="bottom", fill="both", expand=False, padx=5, pady=5)
        self.console_text = tk.Text(self.console_frame, wrap="word", height=10)
        self.console_text.pack(fill="both", expand=True, padx=5, pady=5)
        self.stdout_redirector = StdoutRedirector(self.console_text)
        sys.stdout = self.stdout_redirector
        sys.stderr = self.stdout_redirector
        print("Here you may see console outputs\n")

    def _create_top_panel(self):
        """
        Set up the top panel with three sub-frames.
        """
        self.top_frame = ctk.CTkFrame(self)
        self.top_frame.pack(side="top", fill="both", expand=True)

        # Left container for water-level timeseries and daily view
        self.top_left_container = ctk.CTkFrame(self.top_frame)
        self.top_left_container.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        
        self.top_left_up_frame = ctk.CTkFrame(self.top_left_container)
        self.top_left_up_frame.pack(side="top", fill="both", expand=True)
        self.top_left_up_label = ctk.CTkLabel(self.top_left_up_frame, text="Water-level timeseries + shoreline availability here")
        self.top_left_up_label.pack(fill="both", expand=True)
        
        self.top_left_down_frame = ctk.CTkFrame(self.top_left_container)
        self.top_left_down_frame.pack(side="top", fill="both", expand=True)
        self.top_left_down_label = ctk.CTkLabel(self.top_left_down_frame, text="Daily water-level and shoreline availability")
        self.top_left_down_label.pack(fill="both", expand=True)

        # Top-right for DEM preview
        self.top_right_frame = ctk.CTkFrame(self.top_frame)
        self.top_right_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        self.top_right_label = ctk.CTkLabel(self.top_right_frame, text="Daily DEM")
        self.top_right_label.pack(fill="both", expand=True)

        self.progress_bar = None
        self.progress_label = None

    def _create_bottom_panel(self):
        """
        Set up the bottom panel with input fields, DEM settings, and processing buttons.
        """
        self.bottom_frame = ctk.CTkFrame(self)
        self.bottom_frame.pack(side="bottom", fill="x", expand=False, padx=5, pady=25, ipady=10)

        # Inputs section for water-level CSV, GeoJSON folder, and output path
        inputs_frame = ctk.CTkFrame(self.bottom_frame)
        inputs_frame.pack(side="top", fill="x", padx=5, pady=5)
        
        ctk.CTkLabel(inputs_frame, text="Water Level CSV:").pack(side="left", padx=5)
        self.wl_csv_var = tk.StringVar()
        self.wl_csv_entry = ctk.CTkEntry(inputs_frame, textvariable=self.wl_csv_var, width=240)
        self.wl_csv_entry.pack(side="left", padx=5)
        ctk.CTkButton(inputs_frame, text="Browse", command=self.browse_wl_csv).pack(side="left", padx=5)
        
        ctk.CTkLabel(inputs_frame, text="GeoJSON Folder:").pack(side="left", padx=15)
        self.geojson_dir_var = tk.StringVar()
        self.geojson_dir_entry = ctk.CTkEntry(inputs_frame, textvariable=self.geojson_dir_var, width=240)
        self.geojson_dir_entry.pack(side="left", padx=5)
        ctk.CTkButton(inputs_frame, text="Browse", command=self.browse_geojson_dir).pack(side="left", padx=5)
        
        ctk.CTkLabel(inputs_frame, text="Output GeoTIFF Path:").pack(side="left", padx=20)
        self.out_dir_var = tk.StringVar()
        self.out_dir_entry = ctk.CTkEntry(inputs_frame, textvariable=self.out_dir_var, width=180)
        self.out_dir_entry.pack(side="left", padx=5)
        ctk.CTkButton(inputs_frame, text="Browse", command=self.browse_out_dir).pack(side="left", padx=5)
        # Label to display the selected output folder
        self.out_dir_display_label = ctk.CTkLabel(inputs_frame, text="", width=250)
        self.out_dir_display_label.pack(side="left", padx=5)
        
        # DEM settings section for resolution and interpolation method
        dem_frame = ctk.CTkFrame(self.bottom_frame)
        dem_frame.pack(side="top", fill="x", padx=5, pady=5)
        ctk.CTkLabel(dem_frame, text="Resolution (m):").pack(side="left", padx=5)
        self.resolution_var = tk.StringVar(value="1")
        self.resolution_entry = ctk.CTkEntry(dem_frame, textvariable=self.resolution_var, width=60)
        self.resolution_entry.pack(side="left", padx=5)
        ctk.CTkLabel(dem_frame, text="Interpolation:").pack(side="left", padx=10)
        self.interp_var = tk.StringVar(value="linear")
        self.interp_menu = ctk.CTkOptionMenu(dem_frame, variable=self.interp_var, values=["linear", "cubic"])
        self.interp_menu.pack(side="left", padx=5)
        ctk.CTkButton(dem_frame, text="Generate Next DEM", command=self.on_generate_next_dem).pack(side="left", padx=15)

        # Output section with optional export of XYZ
        output_frame = ctk.CTkFrame(self.bottom_frame)
        output_frame.pack(side="top", fill="x", padx=5, pady=5)
        self.xyz_check = ctk.CTkCheckBox(output_frame, text="Export XYZ?", variable=self.export_xyz_var,
                                          command=self.on_toggle_export_xyz)
        self.xyz_check.pack(side="left", padx=20)
        self.xyz_folder_var = tk.StringVar()
        self.xyz_folder_btn = ctk.CTkButton(output_frame, text="Select XYZ Folder", command=self.browse_xyz_folder)
        self.xyz_folder_btn.pack(side="left", padx=5)
        # New label to display the selected XYZ folder next to the button.
        self.xyz_folder_display_label = ctk.CTkLabel(output_frame, text="", width=250)
        self.xyz_folder_display_label.pack(side="left", padx=5)
        self.xyz_folder_btn.configure(state="disabled")
        ctk.CTkButton(output_frame, text="Batch Process", command=self.on_batch_process).pack(side="left", padx=25)

    def browse_wl_csv(self):
        """
        Open a file dialog to select the water-level CSV.
        """
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if path:
            self.wl_csv_var.set(path)

    def browse_geojson_dir(self):
        """
        Open a dialog to select the GeoJSON folder.
        """
        path = filedialog.askdirectory()
        if path:
            self.geojson_dir_var.set(path)

    def browse_out_dir(self):
        """
        Open a dialog to select the output directory and update the adjacent label.
        """
        path = filedialog.askdirectory()
        if path:
            self.out_dir_var.set(path)
            self.out_dir_display_label.configure(text=path)

    def browse_xyz_folder(self):
        """
        Open a dialog to select the folder for exporting XYZ data and update the adjacent label.
        """
        path = filedialog.askdirectory()
        if path:
            self.xyz_folder_var.set(path)
            self.xyz_folder_display_label.configure(text=path)

    def on_toggle_export_xyz(self):
        """
        Enable or disable the XYZ folder selection based on user choice.
        """
        if self.export_xyz_var.get():
            self.xyz_folder_btn.configure(state="normal")
        else:
            self.xyz_folder_btn.configure(state="disabled")
            self.xyz_folder_display_label.configure(text="")

    def load_data_if_needed(self):
        """
        Lazy-load the water-level and shoreline data.
        Parse daily shorelines and store daily dates.
        """
        if self.shorelines_gdf is not None:
            return
        wl_csv = self.wl_csv_var.get().strip()
        if not wl_csv or not os.path.isfile(wl_csv):
            messagebox.showerror("Error", "Please specify a valid Water Level CSV.")
            raise ValueError("Invalid WL CSV")
        df_wl = pd.read_csv(wl_csv)
        df_wl["time"] = pd.to_datetime(df_wl["time"])
        df_wl.set_index("time", inplace=True)
        df_wl_1min = df_wl.resample("1min").interpolate()
        df_wl_1min.index = df_wl_1min.index.tz_localize(None)
        self.df_wl_1min = df_wl_1min
        geojson_dir = self.geojson_dir_var.get().strip()
        if not geojson_dir or not os.path.isdir(geojson_dir):
            messagebox.showerror("Error", "Please specify a valid GeoJSON folder.")
            raise ValueError("Invalid GeoJSON folder")
        pattern = re.compile(r"(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})-(\d{2})_(\d{2})_cam\d+_v\d+_HDR\.geojson")
        all_files = sorted(Path(geojson_dir).glob("*.geojson"))
        if not all_files:
            messagebox.showwarning("Warning", "No .geojson files found in the folder.")
            raise ValueError("No geojson files")
        shoreline_list = []
        for gj in all_files:
            match = pattern.search(gj.name)
            if not match:
                continue
            year, mo, da, hr1, mn1, hr2, mn2 = map(int, match.groups())
            mid_min = (hr1*60 + mn1 + hr2*60 + mn2) // 2
            mid_hour, mid_minute = divmod(mid_min, 60)
            timestamp = pd.to_datetime(f"{year}-{mo}-{da} {mid_hour}:{mid_minute}:00")
            gdf = gpd.read_file(gj)
            if gdf.empty:
                continue
            gdf["time"] = timestamp.tz_localize(None)
            gdf["filename"] = gj.name
            shoreline_list.append(gdf)
        if not shoreline_list:
            messagebox.showwarning("Warning", "No valid geojson data loaded.")
            raise ValueError("No valid geojson data")
        combined = gpd.GeoDataFrame(pd.concat(shoreline_list, ignore_index=True), crs=shoreline_list[0].crs)
        combined["date"] = combined["time"].dt.date
        self.shorelines_gdf = combined
        self.daily_dates = sorted(combined["date"].unique())
        self.current_day_index = 0
        self.plot_waterlevel_overlay()

    def plot_waterlevel_overlay(self):
        """
        Plot the water-level timeseries and indicate shoreline availability.
        """
        if self.df_wl_1min is None or self.shorelines_gdf is None:
            return
        wl = self.df_wl_1min
        times = sorted(self.shorelines_gdf["time"].unique())
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(wl.index, wl["wl"], label="Water Level")
        yvals = [wl["wl"].asof(t) for t in times]
        ax.scatter(times, yvals, marker='o', c='r', label="Shoreline times", s=10)
        ax.set_title(f"Water-level Timeseries + Shoreline Availability\n({len(self.daily_dates)} days total)")
        ax.set_xlabel("Time")
        ax.set_ylabel("WL (m)")
        ax.tick_params(axis='x', labelrotation=45)
        ax.legend()
        fig.tight_layout()
        self.update_figure_in_label(self.top_left_up_label, fig)
        plt.close(fig)

    def plot_daily_view(self, date_val):
        """
        Plot the daily water-level and shoreline times for a specific day.
        """
        if self.df_wl_1min is None or self.shorelines_gdf is None:
            return
        fig, ax = plt.subplots(figsize=(6, 3))
        gdf_day = self.shorelines_gdf[self.shorelines_gdf["date"] == date_val]
        day_min = gdf_day["time"].min()
        day_max = gdf_day["time"].max()
        mask = (self.df_wl_1min.index >= day_min) & (self.df_wl_1min.index <= day_max)
        subset_wl = self.df_wl_1min[mask]
        ax.plot(subset_wl.index, subset_wl["wl"], label="Daily WL")
        daily_times = gdf_day["time"]
        yvals = [self.df_wl_1min["wl"].asof(t) for t in daily_times]
        ax.scatter(daily_times, yvals, marker='o', c='m', label="This day's shorelines")
        idx = self.daily_dates.index(date_val)
        total = len(self.daily_dates)
        ax.set_title(f"Day {idx+1} of {total}\nDaily WL + Shorelines for {date_val}")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("WL (m)")
        ax.legend()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        fig.tight_layout()
        self.update_figure_in_label(self.top_left_down_label, fig)
        plt.close(fig)

    def plot_dem(self, grid_z):
        """
        Plot the generated DEM in the top-right panel.
        """
        if grid_z is None or np.all(np.isnan(grid_z)):
            return
        fig, ax = plt.subplots(figsize=(6, 8))
        z_min, z_max = np.nanmin(grid_z), np.nanmax(grid_z)
        im = ax.imshow(grid_z, cmap="viridis", vmin=z_min, vmax=z_max, aspect="auto")
        ax.text(10, 200, "Daily DEM", fontsize=14, color="black", zorder=0)
        cbar = fig.colorbar(im, ax=ax, label="Water Level (m)")
        ax.set_title("Created DEM")
        fig.tight_layout()
        self.update_figure_in_label(self.top_right_label, fig)
        plt.close(fig)

    def update_figure_in_label(self, ctk_label, fig):
        """
        Render a Matplotlib figure into a Label widget.
        """
        import io
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        im = Image.open(buf)
        ctk_img = ctk.CTkImage(light_image=im, dark_image=im, size=(im.width, im.height))
        ctk_label.configure(image=ctk_img)
        ctk_label.image = ctk_img

    def on_generate_next_dem(self):
        """
        Generate a DEM for the next day.
        """
        try:
            self.load_data_if_needed()
        except:
            return
        if self.current_day_index >= len(self.daily_dates):
            messagebox.showinfo("Done", "No more days left to process.")
            return
        date_val = self.daily_dates[self.current_day_index]
        self.plot_daily_view(date_val)
        grid_z = self.create_dem_for_day(date_val)
        self.plot_dem(grid_z)
        self.current_day_index += 1

    def on_batch_process(self):
        """
        Generate DEMs for all days in batch mode.
        """
        try:
            self.load_data_if_needed()
        except:
            return
        # Hide certain frames during batch processing (if desired)
        self.top_center_frame.pack_forget()
        self.top_right_frame.pack_forget()
        self.show_progress_bar()
        out_dir = self.out_dir_var.get().strip()
        if not out_dir:
            messagebox.showerror("Error", "Please specify Output GeoTIFF path.")
            self.hide_progress_bar()
            return
        total = len(self.daily_dates)
        for i, date_val in enumerate(self.daily_dates, start=1):
            frac = i / total
            self.progress_bar.set(frac)
            self.progress_label.configure(text=f"Batch: Creating DEM {i}/{total}")
            self.update_idletasks()
            self.create_dem_for_day(date_val)
        self.hide_progress_bar()
        messagebox.showinfo("Done", f"Batch DEM creation finished for {total} days.")

    def show_progress_bar(self):
        """
        Display a progress bar during batch processing.
        """
        if self.progress_bar is None:
            self.progress_label = ctk.CTkLabel(self.top_left_frame, text="Starting batch...")
            self.progress_label.pack(pady=10)
            self.progress_bar = ctk.CTkProgressBar(self.top_left_frame, width=400)
            self.progress_bar.pack()
            self.progress_bar.set(0.0)

    def hide_progress_bar(self):
        """
        Hide and remove the progress bar.
        """
        if self.progress_bar is not None:
            self.progress_bar.destroy()
            self.progress_label.destroy()
            self.progress_bar = None
            self.progress_label = None
        self.top_center_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        self.top_right_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)

    def create_dem_for_day(self, date_val):
        """
        Create a DEM for a specific day using shoreline and water-level data.
        """
        out_dir = self.out_dir_var.get().strip()
        if not out_dir:
            messagebox.showerror("Error", "Please specify an output directory.")
            return None
        gdf_day = self.shorelines_gdf[self.shorelines_gdf["date"] == date_val].copy()
        gdf_day["z"] = gdf_day["time"].apply(lambda t: self.df_wl_1min["wl"].asof(t))
        xyz = []
        for _, row in gdf_day.iterrows():
            geom = row.geometry
            zval = row.z
            if pd.isna(zval) or geom is None:
                continue
            if geom.geom_type == "LineString":
                coords = geom.coords
            elif geom.geom_type == "MultiLineString":
                coords = []
                for part in geom.geoms:
                    coords.extend(part.coords)
            else:
                continue
            for x, y in coords:
                xyz.append((x, y, zval))
        if not xyz:
            print(f"No valid shoreline points for {date_val}")
            return None
        try:
            resolution = float(self.resolution_var.get())
        except Exception:
            messagebox.showwarning("Warning", "Invalid resolution, defaulting to 1m.")
            resolution = 1.0
        method = self.interp_var.get().strip()
        if method not in ["linear", "cubic"]:
            messagebox.showwarning("Warning", "Interpolation must be 'linear' or 'cubic'. Using 'linear'.")
            method = "linear"
        xyz_df = pd.DataFrame(xyz, columns=["x", "y", "z"])
        x_min, x_max = xyz_df["x"].min(), xyz_df["x"].max()
        y_min, y_max = xyz_df["y"].min(), xyz_df["y"].max()
        # Create grid_x in ascending order.
        grid_x_vals = np.arange(x_min, x_max, resolution)
        # Create grid_y in descending order (so first row corresponds to y_max).
        grid_y_vals = np.arange(y_max, y_min, -resolution)
        if len(grid_x_vals) < 2 or len(grid_y_vals) < 2:
            print(f"Skipping day {date_val}: bounding box too small or invalid.")
            return None
        grid_x, grid_y = np.meshgrid(grid_x_vals, grid_y_vals)
        grid_z = griddata(xyz_df[["x", "y"]].values, xyz_df["z"].values, (grid_x, grid_y), method=method)
        if grid_z is None or np.all(np.isnan(grid_z)):
            print(f"Interpolation failed for {date_val}")
            return None
    
        # Create DEM affine transformation parameters.
        transform = from_origin(x_min, y_max, resolution, resolution)
    
        # --- New Code: Generate a concave (alpha shape) polygon from shoreline data ---
        try:
            import alphashape
        except ImportError:
            messagebox.showerror("Error", "The alphashape module is required. Please install it via 'pip install alphashape'.")
            return None
    
        # Collect all shoreline points from the geometries.
        all_points = []
        for geom in gdf_day.geometry:
            if geom is None:
                continue
            if geom.geom_type == "LineString":
                all_points.extend(list(geom.coords))
            elif geom.geom_type == "MultiLineString":
                for part in geom.geoms:
                    all_points.extend(list(part.coords))
            elif geom.geom_type == "Polygon":
                all_points.extend(list(geom.exterior.coords))
        
        if len(all_points) < 3:
            messagebox.showwarning("Warning", "Not enough shoreline points for concave polygon. Using convex hull.")
            shoreline_poly = unary_union(gdf_day.geometry).convex_hull
        else:
            # Adjust alpha as needed based on the spatial scale.
            alpha_val = 0.02  
            shoreline_poly = alphashape.alphashape(all_points, alpha_val)
            if shoreline_poly.geom_type != "Polygon":
                shoreline_poly = shoreline_poly.convex_hull
    
        # Rasterize the polygon to create a mask matching the DEM grid.
        from shapely.geometry import mapping
        mask = features.geometry_mask(
            [mapping(shoreline_poly)],
            out_shape=grid_z.shape,
            transform=transform,
            invert=True
        )
    
        # Apply the mask: keep values within the polygon and set values outside to NaN.
        dem_masked = np.where(mask, grid_z, np.nan)
        # --- End New Code ---
    
        out_path = Path(out_dir) / f"DEM_{date_val}_{method}.tif"
        with rasterio.open(
            out_path, "w", driver="GTiff",
            height=dem_masked.shape[0], width=dem_masked.shape[1],
            count=1, dtype=dem_masked.dtype,
            crs=self.shorelines_gdf.crs,
            transform=transform
        ) as dst:
            dst.write(dem_masked, 1)
        print(f"Saved DEM => {out_path}")
    
        if self.export_xyz_var.get():
            folder = self.xyz_folder_var.get().strip()
            if folder:
                out_xyz = Path(folder) / f"shoreline_xyz_{date_val}.csv"
                xyz_df.to_csv(out_xyz, index=False)
                print(f"Exported XYZ => {out_xyz}")
        return dem_masked
