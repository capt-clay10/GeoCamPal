"""
 Data Exploration: Multi-Time-Series Image Selector

Matches images to one or more hydrodynamic time series
(water level, waves, wind, currents, …) based on timestamps
extracted from filenames.

Per-series criteria (combined with AND logic):
  • Peaks (Maxima)          — images closest to local maxima
  • Troughs (Minima)        — images closest to local minima
  • Above Threshold         — value at image time ≥ threshold
  • Below Threshold         — value at image time ≤ threshold
  • Near Target Value       — value at image time ≈ target ± tolerance
  • Spring Tide Peaks       — images near spring high-water events
  • Neap Tide Peaks         — images near neap high-water events
  • No Filter               — always passes (records value only)

Outputs:
  • A tab-separated .txt file with matched images and values
    from every loaded time series.
  • Optionally copies selected images to a user-defined output folder.
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

# Maximum number of simultaneous time series
MAX_SERIES = 5

# Criteria that each series can be filtered by
CRITERIA = [
    "Peaks (Maxima)",
    "Troughs (Minima)",
    "Above Threshold",
    "Below Threshold",
    "Near Target Value",
    "Spring Tide Peaks",
    "Neap Tide Peaks",
    "No Filter",
]

# Common no-data sentinel values (checked in order)
COMMON_NODATA = [
    -9999, -9999.0, -999, -999.0, -99.9, -99.99,
    9999, 9999.0, 999, 999.0, 99.9, 99.99,
    -1e30, 1e30, -1e10, 1e10,
]


# %% ————————————————————————————— util helpers ————————————————————————
def resource_path(relative_path: str) -> str:
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.dirname(__file__)
    return os.path.join(base_path, relative_path)


def fit_geometry(window, design_w, design_h, resizable=True, margin=0.90):
    """
    Scale a window to fit the current screen while preserving
    the aspect ratio of the original design size.
    Centers the result on screen.  Never upscales beyond the design size.
    """
    screen_w = window.winfo_screenwidth()
    screen_h = window.winfo_screenheight()

    max_w = int(screen_w * margin)
    max_h = int(screen_h * margin)

    scale = min(max_w / design_w, max_h / design_h, 1.0)

    final_w = int(design_w * scale)
    final_h = int(design_h * scale)

    x = (screen_w - final_w) // 2
    y = max(0, (screen_h - final_h) // 2)

    window.geometry(f"{final_w}x{final_h}+{x}+{y}")
    window.resizable(resizable, resizable)


class StdoutRedirector:
    def __init__(self, text_widget: tk.Text):
        self.text_widget = text_widget

    def write(self, message: str):
        self.text_widget.insert(tk.END, message)
        self.text_widget.see(tk.END)

    def flush(self):
        pass


# %% ————————————————————————————— no-data detection ———————————————————

def detect_nodata(values, custom_nodata=None):
    """
    Detect no-data sentinel values in a numeric array.

    Parameters
    ----------
    values : array-like
        Raw numeric values (may contain NaN and sentinels).
    custom_nodata : float or None
        If provided, ONLY this value is treated as no-data
        (auto-detection is skipped).

    Returns
    -------
    nodata_vals : set of float
        Detected (or user-specified) sentinel values.
    report : str
        Human-readable summary of what was found.
    """
    if custom_nodata is not None:
        count = int(np.sum(np.isclose(values, custom_nodata,
                                       atol=1e-6, equal_nan=False)))
        report = (f"Using user-specified no-data value: {custom_nodata} "
                  f"({count} occurrences)")
        return {custom_nodata}, report

    # auto-detect
    detected = {}
    for sentinel in COMMON_NODATA:
        count = int(np.sum(np.isclose(values, sentinel,
                                       atol=1e-6, equal_nan=False)))
        if count > 0:
            detected[sentinel] = count

    if detected:
        parts = [f"{v} ({n}×)" for v, n in detected.items()]
        report = "Auto-detected no-data values: " + ", ".join(parts)
    else:
        report = "No common no-data sentinels detected."
    return set(detected.keys()), report


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


# %% ————————————————————————————— time series loading —————————————————

def load_timeseries(csv_path, dt_col=0, val_col=1, sep=None,
                    nodata_value=None):
    """
    Load a time series CSV (water level, waves, wind, currents, etc.).
    Auto-detects delimiter and no-data sentinels.
    Strips timezone info to avoid naive/aware datetime conflicts.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.
    dt_col : int
        Column index for datetime (default 0).
    val_col : int
        Column index for values (default 1).
    sep : str or None
        Delimiter (auto-detected if None).
    nodata_value : float or None
        If provided, only this value is treated as no-data.
        Otherwise, common sentinels are auto-detected.

    Returns
    -------
    df : DataFrame
        Columns ['datetime', 'value'], cleaned and sorted.
    nodata_report : str
        Description of no-data handling applied.
    """
    # auto-detect separator
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

    # datetime column
    dt_series = pd.to_datetime(df.iloc[:, dt_col], dayfirst=True,
                                errors="coerce")
    # strip timezone
    try:
        if dt_series.dt.tz is not None:
            dt_series = dt_series.dt.tz_localize(None)
    except TypeError:
        dt_series = dt_series.dt.tz_convert("UTC").dt.tz_localize(None)

    val_series = pd.to_numeric(df.iloc[:, val_col], errors="coerce")

    # --- no-data detection and removal ---
    raw_vals = val_series.dropna().values
    nodata_vals, nodata_report = detect_nodata(raw_vals, nodata_value)

    if nodata_vals:
        mask = pd.Series(False, index=val_series.index)
        for nd in nodata_vals:
            mask |= np.isclose(val_series.values, nd,
                               atol=1e-6, equal_nan=False)
        val_series[mask] = np.nan

    out = pd.DataFrame({"datetime": dt_series, "value": val_series})
    out = out.dropna().sort_values("datetime").reset_index(drop=True)
    return out, nodata_report


# %% ————————————————————————————— analysis functions ——————————————————

def find_extrema(wl_df, kind="max", min_sep_hours=5.0, min_prominence=0.2):
    """
    Find true tidal/hydrodynamic peaks or troughs using
    scipy.signal.find_peaks.
    """
    from scipy.signal import find_peaks

    values = wl_df["value"].values.copy()

    # auto-detect sampling interval in minutes
    dt_diff = wl_df["datetime"].diff().median()
    if hasattr(dt_diff, 'total_seconds'):
        interval_min = dt_diff.total_seconds() / 60.0
    else:
        interval_min = dt_diff / np.timedelta64(1, 'm')
    interval_min = max(interval_min, 0.1)

    distance = max(1, int((min_sep_hours * 60.0) / interval_min))

    if kind == "min":
        values = -values

    peak_idx, _props = find_peaks(
        values, distance=distance, prominence=min_prominence)

    return wl_df.iloc[peak_idx].reset_index(drop=True)


def find_threshold_events(wl_df, threshold, direction="above"):
    """Find times where level crosses above/below a threshold."""
    if direction == "above":
        mask = wl_df["value"] >= threshold
    else:
        mask = wl_df["value"] <= threshold
    events = wl_df[mask].copy()
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
    Rank images by how close the time-series value at the image timestamp
    is to *target_value*.
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
      hw_time, lw_time, range_m, classification, hw_level, lw_level
    """
    hw = find_extrema(wl_df, "max", min_sep_hours, min_prominence)
    lw = find_extrema(wl_df, "min", min_sep_hours, min_prominence)
    if hw.empty or lw.empty:
        return pd.DataFrame()

    cycles = []
    for i in range(len(hw)):
        hw_time = hw.iloc[i]["datetime"]
        hw_level = hw.iloc[i]["value"]
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


# %% ————————————————————— multi-series evaluation —————————————————————

def interpolate_value(series_df, img_dt):
    """
    Interpolate the series value at a given image datetime.
    Returns float value or None if out of range.
    """
    if series_df.empty or len(series_df) < 2:
        return None
    ts = series_df["datetime"].astype("int64").to_numpy(dtype=np.int64)
    vals = series_df["value"].to_numpy(dtype=float)
    img_ns = pd.Timestamp(img_dt).value
    if img_ns < ts[0] or img_ns > ts[-1]:
        return None
    return float(np.interp(img_ns, ts, vals))


def evaluate_criterion(series_df, img_dt, criterion, params,
                       precomputed=None, buffer_minutes=30.0):
    """
    Evaluate whether a single image timestamp passes the criterion
    for one time series.

    Parameters
    ----------
    series_df : DataFrame
        Time series with 'datetime' and 'value' columns.
    img_dt : datetime
        Image timestamp.
    criterion : str
        One of CRITERIA.
    params : dict
        Criterion-specific parameters:
          threshold, target, tolerance,
          min_sep_hours, min_prominence.
    precomputed : dict or None
        Pre-computed events (peaks, troughs, tidal classification).
    buffer_minutes : float
        Max offset for event-based criteria.

    Returns
    -------
    passed : bool
    info : dict
        Contains 'value', 'offset_min', 'event_time', etc.
    """
    value = interpolate_value(series_df, img_dt)
    if value is None:
        return False, {"value": None, "reason": "out_of_range"}

    info = {"value": value}

    if criterion == "No Filter":
        return True, info

    elif criterion == "Above Threshold":
        threshold = params.get("threshold", 0.0)
        passed = value >= threshold
        info["threshold"] = threshold
        return passed, info

    elif criterion == "Below Threshold":
        threshold = params.get("threshold", 0.0)
        passed = value <= threshold
        info["threshold"] = threshold
        return passed, info

    elif criterion == "Near Target Value":
        target = params.get("target", 0.0)
        tolerance = params.get("tolerance", 0.5)
        diff = abs(value - target)
        passed = diff <= tolerance
        info["target"] = target
        info["tolerance"] = tolerance
        info["value_diff"] = diff
        return passed, info

    elif criterion in ("Peaks (Maxima)", "Troughs (Minima)"):
        kind = "max" if criterion == "Peaks (Maxima)" else "min"
        events = precomputed.get(f"events_{kind}")
        if events is None or events.empty:
            return False, {**info, "reason": "no_events"}
        return _check_event_proximity(events, img_dt, buffer_minutes, info)

    elif criterion in ("Spring Tide Peaks", "Neap Tide Peaks"):
        target_class = ("spring" if criterion == "Spring Tide Peaks"
                        else "neap")
        tidal_df = precomputed.get("tidal_classification")
        if tidal_df is None or tidal_df.empty:
            return False, {**info, "reason": "no_tidal_data"}
        filtered = tidal_df[tidal_df["classification"] == target_class]
        if filtered.empty:
            return False, {**info, "reason": f"no_{target_class}_events"}
        # build an events-like DataFrame from hw_time / hw_level
        events = pd.DataFrame({
            "datetime": filtered["hw_time"].values,
            "value": filtered["hw_level"].values,
        })
        passed, evt_info = _check_event_proximity(
            events, img_dt, buffer_minutes, info)
        if passed:
            # look up tidal range for matched event
            evt_t = evt_info.get("event_time")
            if evt_t is not None:
                match_rows = filtered[filtered["hw_time"] == evt_t]
                if not match_rows.empty:
                    evt_info["classification"] = target_class
                    evt_info["tidal_range"] = float(
                        match_rows.iloc[0]["range_m"])
        return passed, evt_info

    return False, info


def _check_event_proximity(events_df, img_dt, buffer_minutes, info):
    """Check if img_dt is within buffer_minutes of any event."""
    evt_times = events_df["datetime"].tolist()
    evt_vals = events_df["value"].tolist()

    best_offset = None
    best_evt_time = None
    best_evt_val = None

    for et, ev in zip(evt_times, evt_vals):
        if isinstance(et, np.datetime64):
            et = pd.Timestamp(et).to_pydatetime()
        elif isinstance(et, pd.Timestamp):
            et = et.to_pydatetime()
        offset = (img_dt - et).total_seconds() / 60.0
        if abs(offset) <= buffer_minutes:
            if best_offset is None or abs(offset) < abs(best_offset):
                best_offset = offset
                best_evt_time = et
                best_evt_val = ev

    if best_offset is not None:
        info["offset_min"] = best_offset
        info["event_time"] = best_evt_time
        info["event_value"] = best_evt_val
        return True, info
    return False, info


def precompute_events(series_df, criterion, params):
    """
    Pre-compute events for a series + criterion combination.
    Returns a dict that can be passed to evaluate_criterion().
    """
    result = {}
    min_sep = params.get("min_sep_hours", 5.0)
    min_prom = params.get("min_prominence", 0.2)

    if criterion == "Peaks (Maxima)":
        result["events_max"] = find_extrema(
            series_df, "max", min_sep, min_prom)
    elif criterion == "Troughs (Minima)":
        result["events_min"] = find_extrema(
            series_df, "min", min_sep, min_prom)
    elif criterion in ("Spring Tide Peaks", "Neap Tide Peaks"):
        result["tidal_classification"] = classify_tidal_range(
            series_df, min_sep, min_prom)

    return result


def run_multi_series_analysis(series_configs, image_list,
                              buffer_minutes, print_fn=print):
    """
    Run combined multi-series analysis with AND logic.

    Parameters
    ----------
    series_configs : list of dict
        Each dict has keys: 'df', 'label', 'criterion', 'params'.
    image_list : list of (Path, datetime) tuples
    buffer_minutes : float
    print_fn : callable

    Returns
    -------
    results : list of dict
        Each dict has: image_path, image_dt, and per-series info.
    """
    if not series_configs or not image_list:
        return []

    # pre-compute events for each series
    precomputed_list = []
    for sc in series_configs:
        pc = precompute_events(sc["df"], sc["criterion"], sc["params"])
        precomputed_list.append(pc)

        # report pre-computed events
        crit = sc["criterion"]
        label = sc["label"]
        if crit == "Peaks (Maxima)" and "events_max" in pc:
            n = len(pc["events_max"])
            print_fn(f"  [{label}] Detected {n} peak events.")
        elif crit == "Troughs (Minima)" and "events_min" in pc:
            n = len(pc["events_min"])
            print_fn(f"  [{label}] Detected {n} trough events.")
        elif crit in ("Spring Tide Peaks", "Neap Tide Peaks"):
            tdf = pc.get("tidal_classification", pd.DataFrame())
            if not tdf.empty:
                ns = (tdf["classification"] == "spring").sum()
                nn = (tdf["classification"] == "neap").sum()
                print_fn(f"  [{label}] Tidal cycles: {ns} spring, "
                         f"{nn} neap.")
            else:
                print_fn(f"  [{label}] Warning: could not classify "
                         f"tidal ranges.")

    # evaluate each image against ALL series criteria
    results = []
    for img_path, img_dt in image_list:
        all_pass = True
        row = {"image_path": img_path, "image_dt": img_dt}

        for i, sc in enumerate(series_configs):
            passed, info = evaluate_criterion(
                sc["df"], img_dt, sc["criterion"], sc["params"],
                precomputed=precomputed_list[i],
                buffer_minutes=buffer_minutes,
            )
            label = sc["label"]
            row[f"{label}_value"] = info.get("value")
            row[f"{label}_offset_min"] = info.get("offset_min", 0.0)
            row[f"{label}_event_time"] = info.get("event_time")
            row[f"{label}_event_value"] = info.get("event_value")
            row[f"{label}_criterion"] = sc["criterion"]
            row[f"{label}_passed"] = passed

            # carry extra tidal info
            if "classification" in info:
                row[f"{label}_classification"] = info["classification"]
            if "tidal_range" in info:
                row[f"{label}_tidal_range"] = info["tidal_range"]
            if "value_diff" in info:
                row[f"{label}_value_diff"] = info["value_diff"]

            if not passed:
                all_pass = False
                break  # short-circuit AND

        if all_pass:
            results.append(row)

    return results


# %% ————————————————————————————— main GUI ————————————————————————————
class TimeSeriesExplorerWindow(ctk.CTkToplevel):
    """
    Multi-time-series image explorer.

    Load 1–5 time series (each from a separate CSV) and assign a
    criterion to each.  Images that satisfy ALL criteria simultaneously
    are selected.
    """

    def __init__(self, master=None, **kw):
        super().__init__(master=master, **kw)
        self.title("Multi-Time-Series Image Explorer")
        fit_geometry(self, 1400, 900, resizable=True)
        try:
            self.iconbitmap(resource_path("launch_logo.ico"))
        except Exception:
            pass

        # ——— state ———
        self.image_folder = None
        self.output_folder = None
        self.matched_results = []
        self.recursive_var = tk.BooleanVar(value=False)
        self.copy_images_var = tk.BooleanVar(value=False)
        self.plot_scatter_list = []   # one scatter per subplot
        self.scatter_meta = []
        self.hover_annotations = []

        # per-series state: list of dicts
        self.series_state = []
        for _ in range(MAX_SERIES):
            self.series_state.append({
                "csv_path": None,
                "df": None,
                "nodata_report": "",
            })

        # ——— main layout (3 rows) ———
        self.grid_rowconfigure(0, weight=3)     # plot
        self.grid_rowconfigure(1, weight=0)     # controls
        self.grid_rowconfigure(2, weight=1)     # console
        self.grid_columnconfigure(0, weight=1)

        # ---- TOP: plot ----
        self._build_plot_area()

        # ---- MIDDLE: controls ----
        self._build_controls()

        # ---- BOTTOM: console ----
        self._build_console()

        # initial visibility
        self._on_num_series_change(1)
        self._print_welcome()

    # ═══════════════════════════════════════════════════════════════════
    # BUILD UI SECTIONS
    # ═══════════════════════════════════════════════════════════════════

    def _build_plot_area(self):
        self.top_panel = ctk.CTkFrame(self)
        self.top_panel.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.plot_canvas_frame = tk.Frame(self.top_panel)
        self.plot_canvas_frame.pack(fill="both", expand=True)

        # start with a single subplot — rebuilt dynamically
        self.fig, self.axes = plt.subplots(1, 1, figsize=(12, 4))
        if not isinstance(self.axes, np.ndarray):
            self.axes = np.array([self.axes])
        self.fig.tight_layout()

        self.canvas_plot = FigureCanvasTkAgg(
            self.fig, master=self.plot_canvas_frame)
        self.canvas_plot.get_tk_widget().pack(fill="both", expand=True)

        self.toolbar_frame = tk.Frame(self.top_panel)
        self.toolbar_frame.pack(fill="x")
        self.toolbar = NavigationToolbar2Tk(
            self.canvas_plot, self.toolbar_frame)
        self.toolbar.update()

        self.canvas_plot.mpl_connect(
            "motion_notify_event", self._on_plot_hover)
        self.canvas_plot.mpl_connect(
            "scroll_event", self._on_plot_scroll)

    def _build_controls(self):
        self.bottom_panel = ctk.CTkFrame(self)
        self.bottom_panel.grid(
            row=1, column=0, sticky="nsew", padx=5, pady=5)

        # ── Row 0: global top bar ──
        top_bar = ctk.CTkFrame(self.bottom_panel)
        top_bar.pack(fill="x", padx=5, pady=2)

        ctk.CTkLabel(top_bar, text="Number of time series:"
                     ).grid(row=0, column=0, padx=5, pady=3)
        self.num_series_var = tk.IntVar(value=1)
        self.num_series_menu = ctk.CTkOptionMenu(
            top_bar,
            variable=self.num_series_var,
            values=[str(i) for i in range(1, MAX_SERIES + 1)],
            command=lambda v: self._on_num_series_change(int(v)),
            width=60,
        )
        self.num_series_menu.grid(row=0, column=1, padx=5, pady=3)

        ctk.CTkLabel(top_bar, text="Search buffer (min):"
                     ).grid(row=0, column=2, padx=(20, 5), pady=3)
        self.buffer_entry = ctk.CTkEntry(top_bar, width=60)
        self.buffer_entry.insert(0, "30")
        self.buffer_entry.grid(row=0, column=3, padx=3, pady=3)

        ctk.CTkLabel(top_bar, text="Filename datetime format:"
                     ).grid(row=0, column=4, padx=(20, 3), pady=3)
        self.fmt_entry = ctk.CTkEntry(
            top_bar, width=160,
            placeholder_text="%Y_%m_%d_%H_%M")
        self.fmt_entry.grid(row=0, column=5, padx=3, pady=3)

        # ── Series panels ──
        self.series_container = ctk.CTkFrame(
            self.bottom_panel, fg_color="transparent")
        self.series_container.pack(fill="x", padx=5, pady=2)

        self.series_panels = []
        self.series_widgets = []

        for idx in range(MAX_SERIES):
            self._create_series_panel(idx)

        # ── Row: image folder, output, actions ──
        actions_bar = ctk.CTkFrame(self.bottom_panel)
        actions_bar.pack(fill="x", padx=5, pady=2)

        ctk.CTkButton(actions_bar, text="Browse Image Folder",
                      command=self._browse_images
                      ).grid(row=0, column=0, padx=5, pady=5)
        self.img_label = ctk.CTkLabel(
            actions_bar, text="No folder selected")
        self.img_label.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        ctk.CTkCheckBox(actions_bar, text="Include sub-folders",
                        variable=self.recursive_var
                        ).grid(row=0, column=2, padx=10, pady=5)

        ctk.CTkCheckBox(actions_bar, text="Copy images to output",
                        variable=self.copy_images_var
                        ).grid(row=0, column=3, padx=10, pady=5)

        ctk.CTkButton(actions_bar, text="Browse Output Folder",
                      command=self._browse_output,
                      fg_color="#8C7738"
                      ).grid(row=0, column=4, padx=5, pady=5)
        self.output_label = ctk.CTkLabel(
            actions_bar, text="No output folder selected")
        self.output_label.grid(
            row=0, column=5, padx=5, pady=5, sticky="w")

        ctk.CTkButton(actions_bar, text="Run Analysis",
                      command=self._run_threaded,
                      fg_color="#0F52BA"
                      ).grid(row=0, column=6, padx=10, pady=5)

        self.progress_bar = ctk.CTkProgressBar(actions_bar, width=160)
        self.progress_bar.grid(row=0, column=7, padx=5, pady=5)
        self.progress_bar.set(0)

        ctk.CTkButton(actions_bar, text="Reset",
                      command=self._reset,
                      width=80, fg_color="#8B0000",
                      hover_color="#A52A2A"
                      ).grid(row=0, column=8, padx=5, pady=5)

    def _create_series_panel(self, idx):
        """Build the widgets for one series input row."""
        frame = ctk.CTkFrame(self.series_container)
        frame.pack(fill="x", padx=2, pady=3)

        w = {}  # widget references

        # ── Row 0: file & label ──
        ctk.CTkLabel(frame, text=f"Series {idx + 1}:",
                     font=("Serif", 13, "bold")
                     ).grid(row=0, column=0, padx=5, pady=2)

        w["browse_btn"] = ctk.CTkButton(
            frame, text="Browse CSV", width=100,
            command=lambda i=idx: self._browse_series_csv(i))
        w["browse_btn"].grid(row=0, column=1, padx=3, pady=2)

        w["file_label"] = ctk.CTkLabel(frame, text="No file",
                                        width=150, anchor="w")
        w["file_label"].grid(row=0, column=2, padx=3, pady=2, sticky="w")

        ctk.CTkLabel(frame, text="Label:"
                     ).grid(row=0, column=3, padx=(10, 3), pady=2)
        default_labels = ["Water_Level", "Wave_Height", "Wind_Speed",
                          "Current", "Series_5"]
        w["label_entry"] = ctk.CTkEntry(frame, width=120)
        w["label_entry"].insert(0, default_labels[idx])
        w["label_entry"].grid(row=0, column=4, padx=3, pady=2)

        ctk.CTkLabel(frame, text="No-data:"
                     ).grid(row=0, column=5, padx=(10, 3), pady=2)
        w["nodata_entry"] = ctk.CTkEntry(
            frame, width=80,
            placeholder_text="auto")
        w["nodata_entry"].grid(row=0, column=6, padx=3, pady=2)

        # ── Row 1: criterion & parameters ──
        ctk.CTkLabel(frame, text="Criterion:"
                     ).grid(row=1, column=0, padx=5, pady=2)
        w["criterion_var"] = ctk.StringVar(value="No Filter")
        w["criterion_menu"] = ctk.CTkOptionMenu(
            frame, variable=w["criterion_var"],
            values=CRITERIA, width=160,
            command=lambda v, i=idx: self._on_criterion_change(i, v))
        w["criterion_menu"].grid(row=1, column=1, padx=3, pady=2,
                                  columnspan=2)

        # threshold / target / tolerance
        w["thresh_label"] = ctk.CTkLabel(frame, text="Threshold:")
        w["thresh_label"].grid(row=1, column=3, padx=3, pady=2)
        w["thresh_entry"] = ctk.CTkEntry(frame, width=70)
        w["thresh_entry"].grid(row=1, column=4, padx=3, pady=2)

        w["tol_label"] = ctk.CTkLabel(frame, text="Tolerance:")
        w["tol_label"].grid(row=1, column=5, padx=3, pady=2)
        w["tol_entry"] = ctk.CTkEntry(frame, width=70)
        w["tol_entry"].insert(0, "0.5")
        w["tol_entry"].grid(row=1, column=6, padx=3, pady=2)

        # peak detection params
        w["sep_label"] = ctk.CTkLabel(frame, text="Peak sep. (hrs):")
        w["sep_label"].grid(row=1, column=7, padx=3, pady=2)
        w["sep_entry"] = ctk.CTkEntry(frame, width=50)
        w["sep_entry"].insert(0, "5")
        w["sep_entry"].grid(row=1, column=8, padx=3, pady=2)

        w["prom_label"] = ctk.CTkLabel(frame, text="Prominence:")
        w["prom_label"].grid(row=1, column=9, padx=3, pady=2)
        w["prom_entry"] = ctk.CTkEntry(frame, width=50)
        w["prom_entry"].insert(0, "0.2")
        w["prom_entry"].grid(row=1, column=10, padx=3, pady=2)

        self.series_panels.append(frame)
        self.series_widgets.append(w)

        # set initial visibility
        self._on_criterion_change(idx, "No Filter")

    def _build_console(self):
        self.console_frame = ctk.CTkFrame(self)
        self.console_frame.grid(
            row=2, column=0, sticky="nsew", padx=5, pady=5)
        self.console_text = tk.Text(
            self.console_frame, wrap="word", height=8)
        self.console_text.pack(fill="both", expand=True, padx=5, pady=5)
        self.stdout_redirector = StdoutRedirector(self.console_text)
        sys.stdout = self.stdout_redirector
        sys.stderr = self.stdout_redirector

    def _print_welcome(self):
        print(
            "Multi-Time-Series Image Explorer ready.\n"
            "\n"
            "Load 1–5 time series (water level, waves, wind, …) from\n"
            "separate CSV files.  Assign a criterion to each series.\n"
            "Images satisfying ALL criteria simultaneously are selected.\n"
            "\n"
            "Criteria per series:\n"
            "  Peaks / Troughs — images near detected extrema\n"
            "  Above / Below Threshold — value at image time vs threshold\n"
            "  Near Target Value — value at image time ≈ target ± tolerance\n"
            "  Spring / Neap Tide Peaks — tidal classification + peaks\n"
            "  No Filter — records value but does not filter\n"
            "\n"
            "Peak detection parameters (per series):\n"
            "  Peak sep. (hrs) — min hours between consecutive peaks\n"
            "    Tides: 5–6 hrs.  Wind gusts: 1–2 hrs.\n"
            "  Prominence — min rise above surrounding baseline\n"
            "    Tides: 0.2–0.5 m.  Increase if too many false peaks.\n"
            "\n"
            "No-data handling:\n"
            "  Leave the no-data field as 'auto' to auto-detect common\n"
            "  sentinels (-9999, -999, 9999, etc.).\n"
            "  Enter a custom value to override auto-detection.\n"
            "\n"
            "Filename datetime format (optional):\n"
            "  Leave blank for auto-detection, or enter strftime pattern.\n"
            "--------------------------------"
        )

    # ═══════════════════════════════════════════════════════════════════
    # VISIBILITY MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════

    def _on_num_series_change(self, n):
        """Show/hide series panels based on selected count."""
        if isinstance(n, str):
            n = int(n)
        for idx in range(MAX_SERIES):
            if idx < n:
                self.series_panels[idx].pack(fill="x", padx=2, pady=3)
            else:
                self.series_panels[idx].pack_forget()

    def _on_criterion_change(self, idx, criterion):
        """Show/hide criterion-specific parameter fields."""
        w = self.series_widgets[idx]

        # peak detection params — only for event-based criteria
        needs_peak = criterion in (
            "Peaks (Maxima)", "Troughs (Minima)",
            "Spring Tide Peaks", "Neap Tide Peaks")
        for key in ("sep_label", "sep_entry", "prom_label", "prom_entry"):
            if needs_peak:
                w[key].grid()
            else:
                w[key].grid_remove()

        # threshold — for Above/Below Threshold
        needs_thresh = criterion in ("Above Threshold", "Below Threshold")
        if needs_thresh:
            w["thresh_label"].configure(text="Threshold:")
            w["thresh_label"].grid()
            w["thresh_entry"].grid()
        elif criterion == "Near Target Value":
            w["thresh_label"].configure(text="Target value:")
            w["thresh_label"].grid()
            w["thresh_entry"].grid()
        else:
            w["thresh_label"].grid_remove()
            w["thresh_entry"].grid_remove()

        # tolerance — only for Near Target Value
        if criterion == "Near Target Value":
            w["tol_label"].grid()
            w["tol_entry"].grid()
        else:
            w["tol_label"].grid_remove()
            w["tol_entry"].grid_remove()

    # ═══════════════════════════════════════════════════════════════════
    # BROWSE CALLBACKS
    # ═══════════════════════════════════════════════════════════════════

    def _browse_series_csv(self, idx):
        p = filedialog.askopenfilename(
            title=f"Select CSV for Series {idx + 1}",
            filetypes=[("CSV / TXT", "*.csv *.txt *.dat")])
        if p:
            self.series_state[idx]["csv_path"] = p
            self.series_widgets[idx]["file_label"].configure(
                text=os.path.basename(p))
            self._load_series(idx)

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

    # ═══════════════════════════════════════════════════════════════════
    # LOAD & PLOT
    # ═══════════════════════════════════════════════════════════════════

    def _load_series(self, idx):
        """Load a time series CSV and plot it."""
        try:
            w = self.series_widgets[idx]
            csv_path = self.series_state[idx]["csv_path"]
            label = w["label_entry"].get().strip() or f"Series_{idx + 1}"

            # parse user no-data value
            nodata_text = w["nodata_entry"].get().strip()
            nodata_val = None
            if nodata_text and nodata_text.lower() != "auto":
                try:
                    nodata_val = float(nodata_text)
                except ValueError:
                    print(f"[WARNING] Invalid no-data value '{nodata_text}'"
                          f" for {label}, falling back to auto-detect.")

            df, nodata_report = load_timeseries(
                csv_path, nodata_value=nodata_val)
            self.series_state[idx]["df"] = df
            self.series_state[idx]["nodata_report"] = nodata_report

            print(f"\n[{label}] Loaded {len(df)} records from "
                  f"{os.path.basename(csv_path)}")
            print(f"  Range: {df['datetime'].min()} → "
                  f"{df['datetime'].max()}")
            print(f"  Value range: {df['value'].min():.3f} – "
                  f"{df['value'].max():.3f}")
            print(f"  {nodata_report}")

            self._refresh_plot(results=[])

        except Exception as e:
            print(f"[ERROR] Loading series {idx + 1}: {e}")

    def _get_active_count(self):
        """Return the number of series panels currently visible."""
        return self.num_series_var.get()

    def _get_active_series(self):
        """Return list of (idx, series_state, widgets) for loaded series."""
        n = self._get_active_count()
        active = []
        for idx in range(n):
            if self.series_state[idx]["df"] is not None:
                active.append((idx, self.series_state[idx],
                                self.series_widgets[idx]))
        return active

    # ═══════════════════════════════════════════════════════════════════
    # PLOT
    # ═══════════════════════════════════════════════════════════════════

    def _rebuild_subplots(self, n_plots):
        """Recreate figure with n_plots stacked subplots."""
        self.fig.clear()
        if n_plots <= 0:
            n_plots = 1
        height_per = max(2.5, 10.0 / n_plots)
        self.fig.set_size_inches(12, height_per * n_plots)
        self.axes = self.fig.subplots(
            n_plots, 1, squeeze=False, sharex=True)[:, 0]
        self.hover_annotations = []
        for ax in self.axes:
            ann = ax.annotate(
                "", xy=(0, 0), xytext=(12, 12),
                textcoords="offset points",
                bbox=dict(boxstyle="round", fc="white", alpha=0.9),
                arrowprops=dict(arrowstyle="->", alpha=0.7))
            ann.set_visible(False)
            self.hover_annotations.append(ann)

    def _refresh_plot(self, results=None, title=None):
        """Redraw all subplots with loaded series and matched results."""
        active = self._get_active_series()
        n_plots = max(1, len(active))
        self._rebuild_subplots(n_plots)

        self.plot_scatter_list = []
        self.scatter_meta = []

        series_colors = ["#3498db", "#e67e22", "#2ecc71",
                         "#9b59b6", "#e74c3c"]

        for ax_idx, ax in enumerate(self.axes):
            if ax_idx < len(active):
                idx, state, widgets = active[ax_idx]
                df = state["df"]
                label = widgets["label_entry"].get().strip() or \
                    f"Series_{idx + 1}"
                color = series_colors[ax_idx % len(series_colors)]

                ax.plot(df["datetime"], df["value"],
                        color=color, linewidth=0.5, alpha=0.6,
                        label=label)
                ax.set_ylabel(label, fontsize=9)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize="small", loc="upper right")

                # overlay matched results on this series
                if results:
                    xs, ys, cs, meta_batch = [], [], [], []
                    for r in results:
                        val_key = f"{label}_value"
                        val = r.get(val_key)
                        if val is None:
                            continue

                        evt_t = r.get(f"{label}_event_time") or \
                            r["image_dt"]
                        xs.append(evt_t)
                        ys.append(val)

                        # colour by classification if present
                        cls = r.get(f"{label}_classification")
                        if cls == "spring":
                            cs.append("#e74c3c")
                        elif cls == "neap":
                            cs.append("#2ecc71")
                        else:
                            cs.append("#e74c3c")

                        crit = r.get(f"{label}_criterion", "")
                        offset = r.get(f"{label}_offset_min", 0.0)
                        text = (
                            f"Image: {r['image_path'].name}\n"
                            f"Image time: {r['image_dt']}\n"
                            f"[{label}] value: {val:.3f}\n"
                            f"[{label}] criterion: {crit}\n"
                            f"[{label}] offset: {offset:.1f} min"
                        )
                        meta_batch.append({
                            "x": evt_t, "y": val, "text": text,
                            "ax_idx": ax_idx,
                        })

                    if xs:
                        sc = ax.scatter(xs, ys, c=cs, s=36,
                                        zorder=5, picker=True)
                        self.plot_scatter_list.append(
                            (ax_idx, sc, meta_batch))
                        self.scatter_meta.extend(meta_batch)

                        # auto-zoom to matched range
                        dt_sorted = sorted(xs)
                        dt_first, dt_last = dt_sorted[0], dt_sorted[-1]
                        span = ((dt_last - dt_first)
                                if dt_last != dt_first
                                else timedelta(hours=6))
                        pad = span * 0.05 + timedelta(hours=1)
                        ax.set_xlim(dt_first - pad, dt_last + pad)
            else:
                ax.set_visible(False)

        # format x axis on the bottom-most visible axis
        for ax in self.axes:
            if ax.get_visible():
                last_visible = ax
        last_visible.set_xlabel("Date")
        last_visible.xaxis.set_major_formatter(
            mdates.DateFormatter(
                "%Y-%m-%d %H:%M" if results else "%Y-%m-%d"))

        if title:
            self.axes[0].set_title(title, fontsize=11)

        self.fig.autofmt_xdate()
        self.fig.tight_layout()
        self.canvas_plot.draw_idle()

    def _on_plot_hover(self, event):
        if not self.scatter_meta:
            return

        for ax_idx, scatter, meta_batch in self.plot_scatter_list:
            ax = self.axes[ax_idx]
            ann = self.hover_annotations[ax_idx]
            if event.inaxes != ax:
                if ann.get_visible():
                    ann.set_visible(False)
                    self.canvas_plot.draw_idle()
                continue

            contains, info = scatter.contains(event)
            if contains and info.get("ind"):
                i = info["ind"][0]
                meta = meta_batch[i]
                ann.xy = (mdates.date2num(meta["x"]), meta["y"])
                ann.set_text(meta["text"])
                ann.set_visible(True)
                self.canvas_plot.draw_idle()
            elif ann.get_visible():
                ann.set_visible(False)
                self.canvas_plot.draw_idle()

    def _on_plot_scroll(self, event):
        for ax in self.axes:
            if event.inaxes == ax:
                if event.xdata is None or event.ydata is None:
                    return
                scale = 1 / 1.2 if event.button == "up" else 1.2
                x0, x1 = ax.get_xlim()
                y0, y1 = ax.get_ylim()
                xr = (x1 - x0) * scale
                yr = (y1 - y0) * scale
                rx = ((x1 - event.xdata) / (x1 - x0)
                      if x1 != x0 else 0.5)
                ry = ((y1 - event.ydata) / (y1 - y0)
                      if y1 != y0 else 0.5)
                ax.set_xlim([event.xdata - xr * (1 - rx),
                             event.xdata + xr * rx])
                ax.set_ylim([event.ydata - yr * (1 - ry),
                             event.ydata + yr * ry])
                self.canvas_plot.draw_idle()
                break

    # ═══════════════════════════════════════════════════════════════════
    # RESET
    # ═══════════════════════════════════════════════════════════════════

    def _reset(self):
        self.image_folder = None
        self.output_folder = None
        self.matched_results = []
        self.img_label.configure(text="No folder selected")
        self.output_label.configure(text="No output folder selected")
        self.progress_bar.set(0)
        self.recursive_var.set(False)
        self.copy_images_var.set(False)
        self.num_series_var.set(1)
        self._on_num_series_change(1)

        self.buffer_entry.delete(0, tk.END)
        self.buffer_entry.insert(0, "30")
        self.fmt_entry.delete(0, tk.END)

        for idx in range(MAX_SERIES):
            self.series_state[idx] = {
                "csv_path": None, "df": None, "nodata_report": ""}
            w = self.series_widgets[idx]
            w["file_label"].configure(text="No file")
            w["criterion_var"].set("No Filter")
            self._on_criterion_change(idx, "No Filter")
            w["thresh_entry"].delete(0, tk.END)
            w["tol_entry"].delete(0, tk.END)
            w["tol_entry"].insert(0, "0.5")
            w["sep_entry"].delete(0, tk.END)
            w["sep_entry"].insert(0, "5")
            w["prom_entry"].delete(0, tk.END)
            w["prom_entry"].insert(0, "0.2")
            w["nodata_entry"].delete(0, tk.END)

        self._refresh_plot(results=[])
        self.console_text.delete("1.0", tk.END)
        print("Session reset.\n--------------------------------")

    # ═══════════════════════════════════════════════════════════════════
    # RUN ANALYSIS
    # ═══════════════════════════════════════════════════════════════════

    def _run_threaded(self):
        threading.Thread(target=self._run_analysis, daemon=True).start()

    def _run_analysis(self):
        try:
            # --- validation ---
            n = self._get_active_count()
            loaded = []
            for idx in range(n):
                if self.series_state[idx]["df"] is None:
                    messagebox.showwarning(
                        "Warning",
                        f"Series {idx + 1} has no CSV loaded.")
                    return
                loaded.append(idx)

            if not self.image_folder:
                messagebox.showwarning("Warning",
                                       "Select an image folder first.")
                return
            if not self.output_folder:
                messagebox.showwarning("Warning",
                                       "Select an output folder first.")
                return

            buffer_min = float(self.buffer_entry.get())
            user_fmt = self.fmt_entry.get().strip() or None

            # --- build series configs ---
            series_configs = []
            for idx in loaded:
                w = self.series_widgets[idx]
                label = w["label_entry"].get().strip() or \
                    f"Series_{idx + 1}"
                criterion = w["criterion_var"].get()

                # re-load with current no-data setting in case user
                # changed it after initial load
                nodata_text = w["nodata_entry"].get().strip()
                nodata_val = None
                if nodata_text and nodata_text.lower() != "auto":
                    try:
                        nodata_val = float(nodata_text)
                    except ValueError:
                        pass

                csv_path = self.series_state[idx]["csv_path"]
                df, nodata_report = load_timeseries(
                    csv_path, nodata_value=nodata_val)
                self.series_state[idx]["df"] = df
                self.series_state[idx]["nodata_report"] = nodata_report

                params = {}
                if criterion in ("Above Threshold", "Below Threshold"):
                    params["threshold"] = float(
                        w["thresh_entry"].get())
                elif criterion == "Near Target Value":
                    params["target"] = float(
                        w["thresh_entry"].get())
                    params["tolerance"] = float(
                        w["tol_entry"].get())

                if criterion in ("Peaks (Maxima)", "Troughs (Minima)",
                                 "Spring Tide Peaks", "Neap Tide Peaks"):
                    params["min_sep_hours"] = float(
                        w["sep_entry"].get())
                    params["min_prominence"] = float(
                        w["prom_entry"].get())

                series_configs.append({
                    "df": df,
                    "label": label,
                    "criterion": criterion,
                    "params": params,
                })

            # --- summarise ---
            crit_desc = " AND ".join(
                f"[{sc['label']}: {sc['criterion']}]"
                for sc in series_configs)
            print(f"\n=== Multi-Series Analysis ===")
            print(f"Criteria (AND): {crit_desc}")
            print(f"Search buffer: ±{buffer_min} min")

            for sc in series_configs:
                nd_rep = ""
                for idx in loaded:
                    lbl = self.series_widgets[idx][
                        "label_entry"].get().strip()
                    if lbl == sc["label"]:
                        nd_rep = self.series_state[idx]["nodata_report"]
                        break
                print(f"  [{sc['label']}] {nd_rep}")

            # --- collect images ---
            print("Scanning images for timestamps …")
            image_list = collect_dated_images(
                self.image_folder, user_fmt,
                self.recursive_var.get())
            if not image_list:
                messagebox.showwarning(
                    "Warning",
                    "No images with parseable timestamps found.\n"
                    "Check your filename format.")
                return
            print(f"Found {len(image_list)} dated images.")
            self.progress_bar.set(0.1)

            # --- run combined analysis ---
            print("Evaluating criteria …")
            results = run_multi_series_analysis(
                series_configs, image_list, buffer_min,
                print_fn=print)
            self.matched_results = results
            self.progress_bar.set(0.6)

            if not results:
                print("No images matched ALL criteria.")
                messagebox.showinfo("Result",
                                    "No matching images found.")
                self._refresh_plot(results=[])
                return

            print(f"\nMatched {len(results)} images (all criteria).")

            # --- save results ---
            labels = [sc["label"] for sc in series_configs]
            mode_tag = "_".join(labels).lower().replace(" ", "_")
            txt_name = f"matched_images_{mode_tag}.txt"
            txt_path = os.path.join(self.output_folder, txt_name)

            with open(txt_path, "w", encoding="utf-8") as f:
                # header
                cols = ["image_filename", "image_time"]
                for lb in labels:
                    cols.extend([f"{lb}_value", f"{lb}_criterion",
                                 f"{lb}_offset_min"])
                f.write("\t".join(cols) + "\n")

                for r in results:
                    row_parts = [
                        r["image_path"].name,
                        str(r["image_dt"]),
                    ]
                    for lb in labels:
                        val = r.get(f"{lb}_value")
                        val_str = f"{val:.3f}" if val is not None else ""
                        crit = r.get(f"{lb}_criterion", "")
                        off = r.get(f"{lb}_offset_min", 0.0)
                        row_parts.extend([val_str, crit, f"{off:.1f}"])
                    f.write("\t".join(row_parts) + "\n")

            print(f"Results saved: {txt_path}")

            # --- copy images ---
            if self.copy_images_var.get():
                print("Copying images …")
                copy_dir = os.path.join(
                    self.output_folder, f"images_{mode_tag}")
                os.makedirs(copy_dir, exist_ok=True)
                for i, r in enumerate(results):
                    shutil.copy2(
                        str(r["image_path"]),
                        os.path.join(copy_dir,
                                     r["image_path"].name))
                    self.progress_bar.set(
                        0.6 + 0.35 * (i + 1) / len(results))
                print(f"  Copied to: {copy_dir}")

            # --- update plot ---
            self._refresh_plot(
                results=results,
                title=f"Matched: {crit_desc}")

            self.progress_bar.set(1.0)
            messagebox.showinfo(
                "Done",
                f"Matched {len(results)} images.\n"
                f"Results saved to:\n{txt_path}")

        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", str(e))


# ——— standalone ———
if __name__ == "__main__":
    root = ctk.CTk()
    root.withdraw()
    TimeSeriesExplorerWindow(master=root)
    root.mainloop()