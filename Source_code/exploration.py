"""
 Data Exploration: Time Series Image Selector

Matches images to hydrodynamic time series (time series, waves, currents)
based on timestamps extracted from filenames.

Analysis modes:
  • High water          — images closest to local maxima
  • Low water           — images closest to local minima
  • Threshold-based     — images near user-defined extreme events
  • User-defined value  — images nearest a specific level value
  • Tidal-range class.  — classify images into spring / neap cycles

Outputs:
  • A tab-separated .txt file with columns:
      image_filename | analysis_mode | time_offset_minutes
  • Optionally copies selected images to a user-defined output folder
    (tidal-range mode creates spring/ and neap/ sub-folders).
"""

# %% ————————————————————————————— imports —————————————————————————————
import sys
import os
import re
import shutil
import threading
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.dates as mdates

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# %% ————————————————————————————— util helpers ————————————————————————
def resource_path(relative_path: str) -> str:
    try:
        base_path = sys._MEIPASS
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


# %% ————————————————————————————— timestamp parsing ———————————————————

# common patterns found in coastal camera filenames
_COMMON_PATTERNS = [
    # YYYY_MM_DD_HH_MM  or  YYYY_MM_DD_HH_MM_SS
    (r"(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})(?:_(\d{2}))?",
     "%Y_%m_%d_%H_%M_%S"),
    # YYYYMMDD_HHMMSS
    (r"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})",
     "%Y%m%d_%H%M%S"),
    # YYYYMMDDTHHMMSS
    (r"(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})",
     "%Y%m%dT%H%M%S"),
    # YYYY-MM-DD_HH-MM-SS
    (r"(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})",
     "%Y-%m-%d_%H-%M-%S"),
    # YYYY-MM-DDTHH:MM:SS
    (r"(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})",
     "%Y-%m-%dT%H:%M:%S"),
]


def parse_datetime_from_filename(filename, user_format=None):
    """
    Extract a datetime from a filename.
    If *user_format* is given (e.g. '%Y_%m_%d_%H_%M'), try that first.
    Falls back to common coastal camera patterns.
    """
    stem = Path(filename).stem

    if user_format:
        # build a regex from the strftime format
        rx = user_format
        rx = rx.replace("%Y", r"(\d{4})")
        rx = rx.replace("%m", r"(\d{2})")
        rx = rx.replace("%d", r"(\d{2})")
        rx = rx.replace("%H", r"(\d{2})")
        rx = rx.replace("%M", r"(\d{2})")
        rx = rx.replace("%S", r"(\d{2})")
        m = re.search(rx, stem)
        if m:
            try:
                return datetime.strptime(m.group(0), user_format)
            except ValueError:
                pass

    for pattern, fmt in _COMMON_PATTERNS:
        m = re.search(pattern, stem)
        if m:
            groups = m.groups()
            # pad missing seconds
            if len(groups) == 5 or (len(groups) == 6 and groups[5] is None):
                groups = groups[:5] + ("00",)
            date_str = "_".join(groups[:6])
            try:
                return datetime.strptime(date_str, "%Y_%m_%d_%H_%M_%S")
            except ValueError:
                continue
    return None


def collect_dated_images(folder, user_format=None, recursive=False):
    """
    Return sorted list of (filepath, datetime) tuples.
    """
    results = []
    if recursive:
        for root, _, files in os.walk(folder):
            for f in files:
                if Path(f).suffix.lower() in IMAGE_EXTS:
                    fp = Path(root) / f
                    dt = parse_datetime_from_filename(f, user_format)
                    if dt is not None:
                        results.append((fp, dt))
    else:
        for p in Path(folder).iterdir():
            if p.suffix.lower() in IMAGE_EXTS:
                dt = parse_datetime_from_filename(p.name, user_format)
                if dt is not None:
                    results.append((p, dt))
    results.sort(key=lambda x: x[1])
    return results


# %% ————————————————————————————— time series analysis ————————————————

def load_timeseries(csv_path, dt_col=0, val_col=1, sep=None):
    """
    Load a time series CSV (water level, waves, wind, currents, etc.).
    Auto-detects delimiter.  Strips timezone info to avoid naive/aware
    datetime conflicts with filename-derived timestamps.
    Returns a DataFrame with columns ['datetime', 'value'].
    """
    # try to auto-detect separator
    if sep is None:
        with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
            sample = f.read(4096)
        for s in [",", ";", "\t", " "]:
            if s in sample:
                sep = s
                break
        if sep is None:
            sep = ","

    df = pd.read_csv(csv_path, sep=sep, header=0, encoding="utf-8",
                      engine="python")
    # use positional columns
    dt_series = pd.to_datetime(df.iloc[:, dt_col], dayfirst=True,
                                errors="coerce")
    # strip timezone to keep everything as naive (filenames are naive)
    try:
        if dt_series.dt.tz is not None:
            dt_series = dt_series.dt.tz_localize(None)
    except TypeError:
        # some pandas versions need tz_convert first
        dt_series = dt_series.dt.tz_convert("UTC").dt.tz_localize(None)
    val_series = pd.to_numeric(df.iloc[:, val_col], errors="coerce")
    out = pd.DataFrame({"datetime": dt_series, "value": val_series})
    out = out.dropna().sort_values("datetime").reset_index(drop=True)
    return out


def find_extrema(wl_df, kind="max", min_sep_hours=5.0, min_prominence=0.2):
    """
    Find true tidal/hydrodynamic peaks or troughs using scipy.signal.find_peaks.

    Parameters
    ----------
    wl_df : DataFrame with 'datetime' and 'value' columns.
    kind  : "max" for peaks, "min" for troughs.
    min_sep_hours  : Minimum hours between consecutive peaks.
                     Prevents noise from being detected as separate events.
    min_prominence : Minimum vertical rise a peak must have above its
                     surrounding baseline to be considered a real event.

    The sampling interval is auto-detected from the data.
    """
    from scipy.signal import find_peaks

    values = wl_df["value"].values.copy()

    # auto-detect sampling interval in minutes
    dt_diff = wl_df["datetime"].diff().median()
    if hasattr(dt_diff, 'total_seconds'):
        interval_min = dt_diff.total_seconds() / 60.0
    else:
        # numpy timedelta
        interval_min = dt_diff / np.timedelta64(1, 'm')
    interval_min = max(interval_min, 0.1)  # safety

    # convert min_sep_hours to number of samples
    distance = max(1, int((min_sep_hours * 60.0) / interval_min))

    if kind == "min":
        # invert to find troughs as peaks
        values = -values

    peak_idx, properties = find_peaks(
        values, distance=distance, prominence=min_prominence)

    return wl_df.iloc[peak_idx].reset_index(drop=True)


def find_threshold_events(wl_df, threshold, direction="above"):
    """Find times where level crosses above/below a threshold."""
    if direction == "above":
        mask = wl_df["value"] >= threshold
    else:
        mask = wl_df["value"] <= threshold
    events = wl_df[mask].copy()
    # group consecutive True into events, take peak of each
    events["group"] = (~mask).cumsum()[mask]
    if direction == "above":
        peaks = events.groupby("group").apply(
            lambda g: g.loc[g["value"].idxmax()])
    else:
        peaks = events.groupby("group").apply(
            lambda g: g.loc[g["value"].idxmin()])
    return peaks[["datetime", "value"]].reset_index(drop=True)


def find_closest_value(wl_df, target_value, n_results=20):
    """Find the N times where level is closest to a target value."""
    wl_df = wl_df.copy()
    wl_df["_diff"] = (wl_df["value"] - target_value).abs()
    return wl_df.nsmallest(n_results, "_diff")[["datetime", "value"]
                                                ].reset_index(drop=True)


def find_images_closest_to_value(wl_df, image_list, target_value,
                                 value_tolerance=None, n_results=100):
    """
    Rank images by how close the time-series value at the image timestamp is
    to *target_value*.

    The image timestamp is used only to look up/interpolate the series value;
    selection is based on value difference, not a time buffer.
    """
    if wl_df.empty or not image_list:
        return []

    ts = wl_df["datetime"].astype("int64").to_numpy(dtype=np.int64)
    vals = wl_df["value"].to_numpy(dtype=float)
    if len(ts) < 2:
        return []

    results = []
    for img_path, img_dt in image_list:
        img_ns = pd.Timestamp(img_dt).value
        if img_ns < ts[0] or img_ns > ts[-1]:
            continue

        matched_value = float(np.interp(img_ns, ts, vals))
        value_diff = abs(matched_value - target_value)

        if value_tolerance is not None and value_diff > value_tolerance:
            continue

        ins = int(np.searchsorted(ts, img_ns))
        left_idx = max(0, ins - 1)
        right_idx = min(len(ts) - 1, ins)
        if abs(ts[right_idx] - img_ns) < abs(ts[left_idx] - img_ns):
            nearest_idx = right_idx
        else:
            nearest_idx = left_idx

        nearest_dt = pd.Timestamp(ts[nearest_idx]).to_pydatetime()
        series_offset_min = (img_dt - nearest_dt).total_seconds() / 60.0

        results.append({
            "image_path": img_path,
            "image_dt": img_dt,
            "event_time": img_dt,
            "offset_min": series_offset_min,
            "matched_value": matched_value,
            "event_value": matched_value,
            "value_diff": value_diff,
            "nearest_series_dt": nearest_dt,
            "target_value": target_value,
        })

    results.sort(key=lambda r: (r["value_diff"], r["image_dt"]))
    return results[:n_results]


def classify_tidal_range(wl_df, min_sep_hours=5.0, min_prominence=0.2):
    """
    Classify each tidal cycle as spring or neap based on range.
    Returns a DataFrame with columns:
      hw_time, lw_time, range_m, classification
    """
    hw = find_extrema(wl_df, "max", min_sep_hours, min_prominence)
    lw = find_extrema(wl_df, "min", min_sep_hours, min_prominence)
    if hw.empty or lw.empty:
        return pd.DataFrame()

    cycles = []
    for i in range(len(hw)):
        hw_time = hw.iloc[i]["datetime"]
        hw_level = hw.iloc[i]["value"]
        # find nearest LW before this HW
        lw_before = lw[lw["datetime"] < hw_time]
        if lw_before.empty:
            continue
        lw_row = lw_before.iloc[-1]
        tidal_range = hw_level - lw_row["value"]
        cycles.append({
            "hw_time": hw_time,
            "lw_time": lw_row["datetime"],
            "hw_level": hw_level,
            "lw_level": lw_row["value"],
            "range_m": tidal_range,
        })

    if not cycles:
        return pd.DataFrame()

    cdf = pd.DataFrame(cycles)
    median_range = cdf["range_m"].median()
    cdf["classification"] = np.where(
        cdf["range_m"] >= median_range, "spring", "neap")
    return cdf


def match_images_to_events(event_times, image_list, buffer_minutes,
                           event_values=None):
    """
    For each event time, find the closest image within the buffer.
    Returns list of dicts: {image_path, image_dt, event_time, offset_min}.
    """
    results = []
    img_times = np.array([dt for _, dt in image_list])
    img_paths = [p for p, _ in image_list]

    for i, evt_time in enumerate(event_times):
        if isinstance(evt_time, np.datetime64):
            evt_time = pd.Timestamp(evt_time).to_pydatetime()
        elif isinstance(evt_time, pd.Timestamp):
            evt_time = evt_time.to_pydatetime()

        diffs = np.array([(t - evt_time).total_seconds() / 60.0
                          for t in img_times])
        abs_diffs = np.abs(diffs)
        best_idx = int(np.argmin(abs_diffs))
        offset = diffs[best_idx]

        if abs(offset) <= buffer_minutes:
            result = {
                "image_path": img_paths[best_idx],
                "image_dt": img_times[best_idx],
                "event_time": evt_time,
                "offset_min": offset,
            }
            if event_values is not None and i < len(event_values):
                result["event_value"] = float(event_values[i])
            results.append(result)
    return results


# %% ————————————————————————————— main GUI ————————————————————————————
class TimeSeriesExplorerWindow(ctk.CTkToplevel):
    """Match images to any hydrodynamic time series (water level, waves, wind, currents, etc.)."""

    ANALYSIS_MODES = [
        "High Water",
        "Low Water",
        "Threshold-based Extreme Event",
        "User-defined Value Targeting",
        "Tidal-range Classification",
    ]

    def __init__(self, master=None, **kw):
        super().__init__(master=master, **kw)
        self.title("Time Series Image Explorer")
        self.geometry("1350x900")
        try:
            self.iconbitmap(resource_path("launch_logo.ico"))
        except Exception:
            pass

        # ——— state ———
        self.wl_path = None
        self.image_folder = None
        self.output_folder = None
        self.wl_df = None
        self.matched_results = []
        self.recursive_var = tk.BooleanVar(value=False)
        self.copy_images_var = tk.BooleanVar(value=False)
        self.plot_scatter = None
        self.scatter_meta = []
        self.hover_annotation = None

        # ——— layout ———
        self.grid_rowconfigure(0, weight=3)
        self.grid_rowconfigure(1, weight=0)
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # ---- TOP: plot ----
        self.top_panel = ctk.CTkFrame(self)
        self.top_panel.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.plot_canvas_frame = tk.Frame(self.top_panel)
        self.plot_canvas_frame.pack(fill="both", expand=True)

        self.fig, self.ax = plt.subplots(figsize=(12, 4))
        self.ax.set_title("Time Series")
        self.ax.set_xlabel("Date")
        self.ax.set_ylabel("Value")
        self.ax.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=self.plot_canvas_frame)
        self.canvas_plot.get_tk_widget().pack(fill="both", expand=True)

        self.toolbar_frame = tk.Frame(self.top_panel)
        self.toolbar_frame.pack(fill="x")
        self.toolbar = NavigationToolbar2Tk(self.canvas_plot, self.toolbar_frame)
        self.toolbar.update()

        self.hover_annotation = self.ax.annotate(
            "", xy=(0, 0), xytext=(12, 12), textcoords="offset points",
            bbox=dict(boxstyle="round", fc="white", alpha=0.9),
            arrowprops=dict(arrowstyle="->", alpha=0.7)
        )
        self.hover_annotation.set_visible(False)
        self.canvas_plot.mpl_connect("motion_notify_event", self._on_plot_hover)
        self.canvas_plot.mpl_connect("scroll_event", self._on_plot_scroll)

        # ---- BOTTOM: controls ----
        self.bottom_panel = ctk.CTkFrame(self)
        self.bottom_panel.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        # Row 1 — data inputs
        row1 = ctk.CTkFrame(self.bottom_panel)
        row1.pack(fill="x", padx=5, pady=2)

        ctk.CTkButton(row1, text="Browse Time Series CSV",
                      command=self._browse_wl).grid(
            row=0, column=0, padx=5, pady=5)
        self.wl_label = ctk.CTkLabel(row1, text="No file selected")
        self.wl_label.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        ctk.CTkButton(row1, text="Browse Image Folder",
                      command=self._browse_images).grid(
            row=0, column=2, padx=5, pady=5)
        self.img_label = ctk.CTkLabel(row1, text="No folder selected")
        self.img_label.grid(row=0, column=3, padx=5, pady=5, sticky="w")

        ctk.CTkCheckBox(row1, text="Include sub-folders",
                        variable=self.recursive_var).grid(
            row=0, column=4, padx=10, pady=5)

        # datetime format
        ctk.CTkLabel(row1, text="Filename datetime format:").grid(
            row=0, column=5, padx=3, pady=5)
        self.fmt_entry = ctk.CTkEntry(row1, width=160,
                                       placeholder_text="%Y_%m_%d_%H_%M")
        self.fmt_entry.grid(row=0, column=6, padx=3, pady=5)

        # Row 2 — analysis mode & parameters
        row2 = ctk.CTkFrame(self.bottom_panel)
        row2.pack(fill="x", padx=5, pady=2)

        ctk.CTkLabel(row2, text="Analysis mode:").grid(
            row=0, column=0, padx=5, pady=3)
        self.mode_var = ctk.StringVar(value=self.ANALYSIS_MODES[0])
        self.mode_menu = ctk.CTkOptionMenu(
            row2, variable=self.mode_var,
            values=self.ANALYSIS_MODES,
            command=self._on_mode_change)
        self.mode_menu.grid(row=0, column=1, padx=5, pady=3)

        self.buffer_label = ctk.CTkLabel(row2, text="Search buffer (min):")
        self.buffer_label.grid(row=0, column=2, padx=5, pady=3)
        self.buffer_entry = ctk.CTkEntry(row2, width=60)
        self.buffer_entry.insert(0, "30")
        self.buffer_entry.grid(row=0, column=3, padx=3, pady=3)

        # peak detection params (visible for HW/LW/tidal-range)
        self.sep_label = ctk.CTkLabel(row2, text="Min peak sep. (hrs):")
        self.sep_label.grid(row=0, column=4, padx=5, pady=3)
        self.sep_entry = ctk.CTkEntry(row2, width=50)
        self.sep_entry.insert(0, "5")
        self.sep_entry.grid(row=0, column=5, padx=3, pady=3)

        self.prom_label = ctk.CTkLabel(row2, text="Min prominence:")
        self.prom_label.grid(row=0, column=6, padx=5, pady=3)
        self.prom_entry = ctk.CTkEntry(row2, width=50)
        self.prom_entry.insert(0, "0.2")
        self.prom_entry.grid(row=0, column=7, padx=3, pady=3)

        # threshold / value entry (shown/hidden by mode)
        self.param_label = ctk.CTkLabel(row2, text="Threshold:")
        self.param_label.grid(row=0, column=8, padx=5, pady=3)
        self.param_entry = ctk.CTkEntry(row2, width=80)
        self.param_entry.grid(row=0, column=9, padx=3, pady=3)

        self.direction_var = ctk.StringVar(value="above")
        self.direction_menu = ctk.CTkOptionMenu(
            row2, variable=self.direction_var,
            values=["above", "below"], width=80)
        self.direction_menu.grid(row=0, column=10, padx=3, pady=3)

        self.value_tol_label = ctk.CTkLabel(row2, text="Value tolerance:")
        self.value_tol_label.grid(row=0, column=11, padx=5, pady=3)
        self.value_tol_entry = ctk.CTkEntry(row2, width=80)
        self.value_tol_entry.insert(0, "0.5")
        self.value_tol_entry.grid(row=0, column=12, padx=3, pady=3)

        # initially hide threshold-specific widgets
        self._on_mode_change(self.mode_var.get())

        # Row 3 — actions
        row3 = ctk.CTkFrame(self.bottom_panel)
        row3.pack(fill="x", padx=5, pady=2)

        ctk.CTkCheckBox(row3, text="Copy identified images to output",
                        variable=self.copy_images_var).grid(
            row=0, column=0, padx=5, pady=5)

        ctk.CTkButton(row3, text="Browse Output Folder",
                      command=self._browse_output).grid(
            row=0, column=1, padx=5, pady=5)
        self.output_label = ctk.CTkLabel(row3, text="No output folder selected")
        self.output_label.grid(row=0, column=2, padx=5, pady=5, sticky="w")

        ctk.CTkButton(row3, text="Run Analysis",
                      command=self._run_threaded).grid(
            row=0, column=3, padx=10, pady=5)

        self.progress_bar = ctk.CTkProgressBar(row3, width=200)
        self.progress_bar.grid(row=0, column=4, padx=5, pady=5)
        self.progress_bar.set(0)

        self.btn_reset = ctk.CTkButton(
            row3, text="Reset", command=self._reset,
            width=80, fg_color="#8B0000", hover_color="#A52A2A")
        self.btn_reset.grid(row=0, column=5, padx=5, pady=5, sticky="e")

        # ---- CONSOLE ----
        self.console_frame = ctk.CTkFrame(self)
        self.console_frame.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        self.console_text = tk.Text(self.console_frame, wrap="word", height=8)
        self.console_text.pack(fill="both", expand=True, padx=5, pady=5)
        self.stdout_redirector = StdoutRedirector(self.console_text)
        sys.stdout = self.stdout_redirector
        sys.stderr = self.stdout_redirector
        print("Time Series Image Explorer ready.\n"
              "Provide a time series CSV (water level, waves, wind, currents,\n"
              "  etc.) and a folder of images with timestamps in filenames.\n"
              "\n"
              "Peak detection (High Water / Low Water / Tidal-range modes):\n"
              "  Min peak sep. (hrs) — Minimum hours between consecutive peaks.\n"
              "    Prevents noise from being detected as separate events.\n"
              "    For tides: 5-6 hrs.  For wind gusts: 1-2 hrs.\n"
              "  Min prominence — Minimum rise above surrounding baseline.\n"
              "    For tides: 0.2-0.5 m.  Increase if too many false peaks.\n"
              "\n"
              "Filename datetime format (optional):\n"
              "  Leave blank for auto-detection — works with patterns like:\n"
              "    2025_05_01_04_02-04_12_camPANO_v2_HDR.tif\n"
              "  Or enter a Python strftime pattern, e.g. %Y_%m_%d_%H_%M\n"
              "--------------------------------")

    # ——— mode change visibility ———

    def _on_mode_change(self, mode):
        # peak detection params — visible for HW, LW, tidal-range
        if mode in ("High Water", "Low Water", "Tidal-range Classification"):
            self.sep_label.grid()
            self.sep_entry.grid()
            self.prom_label.grid()
            self.prom_entry.grid()
        else:
            self.sep_label.grid_remove()
            self.sep_entry.grid_remove()
            self.prom_label.grid_remove()
            self.prom_entry.grid_remove()

        if mode == "User-defined Value Targeting":
            self.buffer_label.grid_remove()
            self.buffer_entry.grid_remove()
            self.value_tol_label.grid()
            self.value_tol_entry.grid()
        else:
            self.buffer_label.grid()
            self.buffer_entry.grid()
            self.value_tol_label.grid_remove()
            self.value_tol_entry.grid_remove()

        # threshold / value / direction — mode-specific
        if mode == "Threshold-based Extreme Event":
            self.param_label.configure(text="Threshold:")
            self.param_entry.grid()
            self.direction_menu.grid()
        elif mode == "User-defined Value Targeting":
            self.param_label.configure(text="Target value:")
            self.param_entry.grid()
            self.direction_menu.grid_remove()
        else:
            self.param_label.configure(text="")
            self.param_entry.grid_remove()
            self.direction_menu.grid_remove()

    # ——— browse callbacks ———

    def _browse_wl(self):
        p = filedialog.askopenfilename(
            title="Select Time Series CSV",
            filetypes=[("CSV / TXT", "*.csv *.txt *.dat")])
        if p:
            self.wl_path = p
            self.wl_label.configure(text=os.path.basename(p))
            self._load_and_plot_wl()

    def _browse_images(self):
        d = filedialog.askdirectory(title="Select Image Folder")
        if d:
            self.image_folder = d
            self.img_label.configure(text=d)

    def _browse_output(self):
        d = filedialog.askdirectory(title="Select Output Folder")
        if d:
            self.output_folder = d
            self.output_label.configure(text=d)

    # ——— load & plot time series ———

    def _load_and_plot_wl(self):
        try:
            self.wl_df = load_timeseries(self.wl_path)
            print(f"Loaded {len(self.wl_df)} time series records.")
            print(f"  Range: {self.wl_df['datetime'].min()} → "
                  f"{self.wl_df['datetime'].max()}")
            print(f"  Value range: {self.wl_df['value'].min():.3f} – "
                  f"{self.wl_df['value'].max():.3f}")
            self._refresh_plot("Time Series", [])
        except Exception as e:
            print(f"[ERROR] Loading time series: {e}")

    # ——— reset ———

    def _reset(self):
        self.wl_path = None
        self.image_folder = None
        self.output_folder = None
        self.wl_df = None
        self.matched_results = []
        self.wl_label.configure(text="No file selected")
        self.img_label.configure(text="No folder selected")
        self.output_label.configure(text="No output folder selected")
        self.progress_bar.set(0)
        self.recursive_var.set(False)
        self.copy_images_var.set(False)
        self.mode_var.set(self.ANALYSIS_MODES[0])
        self._on_mode_change(self.ANALYSIS_MODES[0])
        self.buffer_entry.delete(0, tk.END)
        self.buffer_entry.insert(0, "30")
        self.sep_entry.delete(0, tk.END)
        self.sep_entry.insert(0, "5")
        self.prom_entry.delete(0, tk.END)
        self.prom_entry.insert(0, "0.2")
        self.param_entry.delete(0, tk.END)
        self.value_tol_entry.delete(0, tk.END)
        self.value_tol_entry.insert(0, "0.5")
        self.fmt_entry.delete(0, tk.END)

        self._refresh_plot("Time Series", [])
        self.console_text.delete("1.0", tk.END)
        print("Session reset.\n--------------------------------")

    # ——— analysis ———

    def _refresh_plot(self, title, results, mode=None):
        self.ax.clear()

        if self.wl_df is not None and not self.wl_df.empty:
            self.ax.plot(self.wl_df["datetime"], self.wl_df["value"],
                         color="#3498db", linewidth=0.5, alpha=0.6,
                         label="Time series")

        self.plot_scatter = None
        self.scatter_meta = []
        if self.hover_annotation is not None:
            self.hover_annotation.set_visible(False)

        if results:
            xs = []
            ys = []
            colors = []

            for r in results:
                if mode == "User-defined Value Targeting":
                    x = r["image_dt"]
                    y = r["matched_value"]
                    color = "#e67e22"
                    text = (
                        f"Image: {r['image_path'].name}\n"
                        f"Image time: {r['image_dt']}\n"
                        f"Series value: {r['matched_value']:.3f}\n"
                        f"Target: {r['target_value']:.3f}\n"
                        f"Value diff: {r['value_diff']:.3f}\n"
                        f"Nearest series time: {r['nearest_series_dt']}\n"
                        f"Series time offset: {r['offset_min']:.2f} min"
                    )
                else:
                    x = r["event_time"]
                    y = r.get("event_value")
                    if y is None and self.wl_df is not None:
                        idx = (self.wl_df["datetime"] - x).abs().idxmin()
                        y = float(self.wl_df.loc[idx, "value"])
                    if mode == "Tidal-range Classification":
                        color = "#e74c3c" if r.get("classification") == "spring" else "#2ecc71"
                        text = (
                            f"Image: {r['image_path'].name}\n"
                            f"Event time: {r['event_time']}\n"
                            f"Image time: {r['image_dt']}\n"
                            f"Value: {float(y):.3f}\n"
                            f"Time offset: {r['offset_min']:.2f} min\n"
                            f"Class: {r.get('classification', '')}\n"
                            f"Tidal range: {r.get('tidal_range', float('nan')):.3f}"
                        )
                    else:
                        color = "#e74c3c"
                        text = (
                            f"Image: {r['image_path'].name}\n"
                            f"Event time: {r['event_time']}\n"
                            f"Image time: {r['image_dt']}\n"
                            f"Value: {float(y):.3f}\n"
                            f"Time offset: {r['offset_min']:.2f} min"
                        )
                xs.append(x)
                ys.append(y)
                colors.append(color)
                self.scatter_meta.append({"x": x, "y": y, "text": text})

            self.plot_scatter = self.ax.scatter(xs, ys, c=colors, s=36,
                                                zorder=5, picker=True)

            if mode == "Tidal-range Classification":
                from matplotlib.patches import Patch as MPatch
                self.ax.legend(handles=[
                    MPatch(color="#e74c3c", label="Spring"),
                    MPatch(color="#2ecc71", label="Neap"),
                ], fontsize="small")
            else:
                self.ax.legend([self.plot_scatter], [f"Matched ({mode})"],
                               fontsize="small")

            img_dts = sorted([r["image_dt"] for r in results])
            dt_first, dt_last = img_dts[0], img_dts[-1]
            span = (dt_last - dt_first) if dt_last != dt_first else timedelta(hours=6)
            pad = span * 0.05 + timedelta(hours=1)
            self.ax.set_xlim(dt_first - pad, dt_last + pad)

        self.ax.set_title(title)
        self.ax.set_xlabel("Date")
        self.ax.set_ylabel("Value")
        self.ax.grid(True, alpha=0.3)
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter(
            "%Y-%m-%d %H:%M" if results else "%Y-%m-%d"))
        self.fig.autofmt_xdate()
        self.fig.tight_layout()
        self.canvas_plot.draw_idle()

    def _on_plot_hover(self, event):
        if self.plot_scatter is None or self.hover_annotation is None:
            return
        if event.inaxes != self.ax:
            if self.hover_annotation.get_visible():
                self.hover_annotation.set_visible(False)
                self.canvas_plot.draw_idle()
            return

        contains, info = self.plot_scatter.contains(event)
        if contains and info.get("ind"):
            idx = info["ind"][0]
            meta = self.scatter_meta[idx]
            self.hover_annotation.xy = (meta["x"], meta["y"])
            self.hover_annotation.set_text(meta["text"])
            self.hover_annotation.set_visible(True)
            self.canvas_plot.draw_idle()
        elif self.hover_annotation.get_visible():
            self.hover_annotation.set_visible(False)
            self.canvas_plot.draw_idle()

    def _on_plot_scroll(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        scale_factor = 1 / 1.2 if event.button == "up" else 1.2
        x_min, x_max = self.ax.get_xlim()
        y_min, y_max = self.ax.get_ylim()
        x_range = (x_max - x_min) * scale_factor
        y_range = (y_max - y_min) * scale_factor

        rel_x = (x_max - event.xdata) / (x_max - x_min) if x_max != x_min else 0.5
        rel_y = (y_max - event.ydata) / (y_max - y_min) if y_max != y_min else 0.5

        self.ax.set_xlim([event.xdata - x_range * (1 - rel_x),
                          event.xdata + x_range * rel_x])
        self.ax.set_ylim([event.ydata - y_range * (1 - rel_y),
                          event.ydata + y_range * rel_y])
        self.canvas_plot.draw_idle()

    def _run_threaded(self):
        threading.Thread(target=self._run_analysis, daemon=True).start()

    def _run_analysis(self):
        try:
            if self.wl_df is None:
                messagebox.showwarning("Warning",
                                       "Load a time series file first.")
                return
            if not self.image_folder:
                messagebox.showwarning("Warning",
                                       "Select an image folder first.")
                return
            if not self.output_folder:
                messagebox.showwarning("Warning",
                                       "Select an output folder first.")
                return

            mode = self.mode_var.get()
            user_fmt = self.fmt_entry.get().strip() or None
            buffer_min = (float(self.buffer_entry.get())
                          if mode != "User-defined Value Targeting" else None)

            # peak detection params (used by HW/LW/tidal-range)
            min_sep_h = float(self.sep_entry.get())
            min_prom = float(self.prom_entry.get())

            print(f"\n=== Analysis: {mode} ===")
            value_tol = None
            target = None
            if mode == "User-defined Value Targeting":
                target = float(self.param_entry.get())
                tol_text = self.value_tol_entry.get().strip()
                value_tol = float(tol_text) if tol_text else None
                if value_tol is None:
                    print(f"Target value: {target} (no tolerance filter)")
                else:
                    print(f"Target value: {target} ± {value_tol}")
            else:
                print(f"Search buffer: ±{buffer_min} min")
            if mode in ("High Water", "Low Water",
                        "Tidal-range Classification"):
                print(f"Peak detection: min separation={min_sep_h} hrs, "
                      f"min prominence={min_prom}")

            print("Scanning images for timestamps …")
            image_list = collect_dated_images(
                self.image_folder, user_fmt, self.recursive_var.get())
            if not image_list:
                messagebox.showwarning(
                    "Warning",
                    "No images with parseable timestamps found.\n"
                    "Check your filename format.")
                return
            print(f"Found {len(image_list)} dated images.")
            self.progress_bar.set(0.1)

            results = []
            mode_tag = mode.lower().replace(" ", "_").replace("-", "_")

            if mode == "High Water":
                events = find_extrema(self.wl_df, "max",
                                      min_sep_h, min_prom)
                print(f"Detected {len(events)} high-water events.")
                results = match_images_to_events(
                    events["datetime"].tolist(), image_list, buffer_min,
                    event_values=events["value"].tolist())

            elif mode == "Low Water":
                events = find_extrema(self.wl_df, "min",
                                      min_sep_h, min_prom)
                print(f"Detected {len(events)} low-water events.")
                results = match_images_to_events(
                    events["datetime"].tolist(), image_list, buffer_min,
                    event_values=events["value"].tolist())

            elif mode == "Threshold-based Extreme Event":
                thresh = float(self.param_entry.get())
                direction = self.direction_var.get()
                events = find_threshold_events(
                    self.wl_df, thresh, direction)
                print(f"Detected {len(events)} events ({direction} {thresh}).")
                results = match_images_to_events(
                    events["datetime"].tolist(), image_list, buffer_min,
                    event_values=events["value"].tolist())

            elif mode == "User-defined Value Targeting":
                results = find_images_closest_to_value(
                    self.wl_df, image_list, target,
                    value_tolerance=value_tol, n_results=100)
                if value_tol is None:
                    print(f"Ranked images by closeness to {target}.")
                else:
                    print(f"Found {len(results)} images within {target} ± {value_tol}.")

            elif mode == "Tidal-range Classification":
                cycle_df = classify_tidal_range(
                    self.wl_df, min_sep_h, min_prom)
                if cycle_df.empty:
                    messagebox.showwarning("Warning",
                                           "Could not classify tidal ranges.")
                    return
                n_spring = (cycle_df["classification"] == "spring").sum()
                n_neap = (cycle_df["classification"] == "neap").sum()
                print(f"Classified {len(cycle_df)} tidal cycles: "
                      f"{n_spring} spring, {n_neap} neap.")

                for _, row in cycle_df.iterrows():
                    hw_time = row["hw_time"]
                    if isinstance(hw_time, pd.Timestamp):
                        hw_time = hw_time.to_pydatetime()
                    diffs = [(abs((dt - hw_time).total_seconds() / 60.0),
                              p, dt)
                             for p, dt in image_list]
                    diffs.sort(key=lambda x: x[0])
                    if diffs and diffs[0][0] <= buffer_min:
                        best = diffs[0]
                        results.append({
                            "image_path": best[1],
                            "image_dt": best[2],
                            "event_time": hw_time,
                            "offset_min": (best[2] - hw_time).total_seconds() / 60.0,
                            "classification": row["classification"],
                            "tidal_range": row["range_m"],
                            "event_value": row["hw_level"],
                        })

            self.matched_results = results
            self.progress_bar.set(0.6)

            if not results:
                if mode == "User-defined Value Targeting":
                    print("No images matched the requested value tolerance.")
                else:
                    print("No matching images found within the buffer window.")
                messagebox.showinfo("Result", "No matching images found.")
                self._refresh_plot("Time Series", [], mode=mode)
                return

            print(f"\nMatched {len(results)} images.")

            txt_name = f"matched_images_{mode_tag}.txt"
            txt_path = os.path.join(self.output_folder, txt_name)
            with open(txt_path, "w", encoding="utf-8") as f:
                if mode == "Tidal-range Classification":
                    f.write("image_filename\tanalysis_mode\t"
                            "time_offset_minutes\tclassification\t"
                            "tidal_range_m\n")
                    for r in results:
                        f.write(f"{r['image_path'].name}\t{mode}\t"
                                f"{r['offset_min']:.1f}\t"
                                f"{r.get('classification', '')}\t"
                                f"{r.get('tidal_range', ''):.3f}\n")
                elif mode == "User-defined Value Targeting":
                    f.write("image_filename\tanalysis_mode\t"
                            "matched_value\tvalue_difference\t"
                            "series_time_offset_minutes\n")
                    for r in results:
                        f.write(f"{r['image_path'].name}\t{mode}\t"
                                f"{r['matched_value']:.3f}\t"
                                f"{r['value_diff']:.3f}\t"
                                f"{r['offset_min']:.1f}\n")
                else:
                    f.write("image_filename\tanalysis_mode\t"
                            "time_offset_minutes\n")
                    for r in results:
                        f.write(f"{r['image_path'].name}\t{mode}\t"
                                f"{r['offset_min']:.1f}\n")
            print(f"Results saved: {txt_path}")

            if self.copy_images_var.get():
                print("Copying images …")
                if mode == "Tidal-range Classification":
                    spring_dir = os.path.join(self.output_folder, "spring")
                    neap_dir = os.path.join(self.output_folder, "neap")
                    os.makedirs(spring_dir, exist_ok=True)
                    os.makedirs(neap_dir, exist_ok=True)
                    for i, r in enumerate(results):
                        dest = spring_dir if r.get(
                            "classification") == "spring" else neap_dir
                        shutil.copy2(str(r["image_path"]),
                                     os.path.join(dest, r["image_path"].name))
                        self.progress_bar.set(0.6 + 0.35 * (i + 1) / len(results))
                    print(f"  Spring: {spring_dir}")
                    print(f"  Neap: {neap_dir}")
                else:
                    copy_dir = os.path.join(self.output_folder,
                                            f"images_{mode_tag}")
                    os.makedirs(copy_dir, exist_ok=True)
                    for i, r in enumerate(results):
                        shutil.copy2(str(r["image_path"]),
                                     os.path.join(copy_dir,
                                                   r["image_path"].name))
                        self.progress_bar.set(0.6 + 0.35 * (i + 1) / len(results))
                    print(f"  Copied to: {copy_dir}")

            self._refresh_plot(f"Time Series — {mode}", results, mode=mode)

            self.progress_bar.set(1.0)
            messagebox.showinfo(
                "Done",
                f"Matched {len(results)} images.\n"
                f"Results saved to:\n{txt_path}")

        except Exception as e:
            print(f"[ERROR] {e}")
            messagebox.showerror("Error", str(e))


# ——— standalone ———
if __name__ == "__main__":
    root = ctk.CTk()
    root.withdraw()
    TimeSeriesExplorerWindow(master=root)
    root.mainloop()