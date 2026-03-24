"""
Color Space Explorer

Exploratory analysis of color distributions across a folder of images.
Treats images as data — visualises color-channel distributions, scatter
density clouds, and temporal color signatures.

Analysis modes:
  • Channel histograms  — per-channel distribution across all images
  • 2-D scatter density — joint distribution of any two color channels
  • Color timeline      — channel statistics vs. image timestamp
  • Outlier detection   — flag images with anomalous color profiles

Supported color spaces:  RGB · HSV · LAB · Normalised RGB (r, g, b)

Outputs:
  • color_stats.csv     — per-image channel statistics
  • outliers.txt        — images > N σ from dataset mean (optional)
  • Plots exportable as PNG from the matplotlib toolbar
"""

# %% ————————————————————————————— imports —————————————————————————————
import sys
import os
import re
import time
import threading
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.dates as mdates
import matplotlib.colors as mcolors

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

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
# (Duplicated from exploration.py so this module stays standalone.)

_COMMON_PATTERNS = [
    (r"(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})(?:_(\d{2}))?",
     "%Y_%m_%d_%H_%M_%S"),
    (r"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})",
     "%Y%m%d_%H%M%S"),
    (r"(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})",
     "%Y%m%dT%H%M%S"),
    (r"(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})",
     "%Y-%m-%d_%H-%M-%S"),
    (r"(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})",
     "%Y-%m-%dT%H:%M:%S"),
]


def parse_datetime_from_filename(filename, user_format=None):
    """Extract a datetime from a filename (common coastal camera patterns)."""
    stem = Path(filename).stem
    if user_format:
        rx = user_format
        for tok, pat in [("%Y", r"(\d{4})"), ("%m", r"(\d{2})"),
                         ("%d", r"(\d{2})"), ("%H", r"(\d{2})"),
                         ("%M", r"(\d{2})"), ("%S", r"(\d{2})")]:
            rx = rx.replace(tok, pat)
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
            if len(groups) == 5 or (len(groups) == 6 and groups[5] is None):
                groups = groups[:5] + ("00",)
            date_str = "_".join(groups[:6])
            try:
                return datetime.strptime(date_str, "%Y_%m_%d_%H_%M_%S")
            except ValueError:
                continue
    return None


# %% ————————————————————————————— file helpers ————————————————————————

def collect_images(folder, recursive=False):
    """Collect image paths, optionally recursing into sub-folders."""
    if recursive:
        results = []
        for root, _, files in os.walk(folder):
            for f in sorted(files):
                if Path(f).suffix.lower() in IMAGE_EXTS:
                    results.append(Path(root) / f)
        return results
    return sorted(
        p for p in Path(folder).iterdir()
        if p.suffix.lower() in IMAGE_EXTS
    )


# %% ————————————————————————————— color-space conversion ——————————————

# Channel name tuples for each color space
CS_CHANNELS = {
    "RGB":            ("Red", "Green", "Blue"),
    "HSV":            ("Hue", "Saturation", "Value"),
    "LAB":            ("L*", "a*", "b*"),
    "Normalised RGB": ("r (R/sum)", "g (G/sum)", "b (B/sum)"),
}

# Channel value ranges (for axis limits)
CS_RANGES = {
    "RGB":            [(0, 255), (0, 255), (0, 255)],
    "HSV":            [(0, 180), (0, 255), (0, 255)],
    "LAB":            [(0, 255), (0, 255), (0, 255)],
    "Normalised RGB": [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
}


def convert_pixels(img_bgr, color_space, sky_frac=0.0, ground_frac=0.0,
                   aoi_mask=None):
    """
    Convert a BGR image to the requested color space.
    Returns an (N, 3) float32 array of pixel values.
    Optionally crops top *sky_frac* and bottom *ground_frac* of rows.
    If *aoi_mask* is provided (uint8 H×W, 255 = inside), only those
    pixels are returned.
    """
    h, w = img_bgr.shape[:2]
    top = int(h * sky_frac)
    bot = h - int(h * ground_frac)
    if bot <= top:
        bot = h
    roi = img_bgr[top:bot, :]

    if color_space == "RGB":
        converted = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    elif color_space == "HSV":
        converted = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    elif color_space == "LAB":
        converted = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    elif color_space == "Normalised RGB":
        converted = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    else:
        converted = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    # Apply AOI mask (crop to same vertical slice first)
    if aoi_mask is not None:
        mask_roi = aoi_mask[top:bot, :]
        # Resize mask if image dimensions don't match (different image sizes)
        if mask_roi.shape[:2] != converted.shape[:2]:
            mask_roi = cv2.resize(mask_roi, (converted.shape[1], converted.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
        flat_mask = mask_roi.reshape(-1) > 0
        out = converted.reshape(-1, 3).astype(np.float32)
        out = out[flat_mask]
    else:
        out = converted.reshape(-1, 3).astype(np.float32)

    if color_space == "Normalised RGB" and out.shape[0] > 0:
        total = out.sum(axis=1, keepdims=True)
        total[total == 0] = 1.0
        out = out / total

    return out


def image_channel_stats(pixels_3col):
    """
    Return dict of per-channel statistics from an (N, 3) pixel array.
    Keys: mean_0, mean_1, mean_2, std_0, std_1, std_2, med_0, …
    """
    stats = {}
    for ch in range(3):
        col = pixels_3col[:, ch]
        stats[f"mean_{ch}"] = float(np.mean(col))
        stats[f"std_{ch}"]  = float(np.std(col))
        stats[f"med_{ch}"]  = float(np.median(col))
        stats[f"p05_{ch}"]  = float(np.percentile(col, 5))
        stats[f"p95_{ch}"]  = float(np.percentile(col, 95))
    return stats


# %% ————————————————————————————— main GUI ————————————————————————————

class ColorSpaceExplorerWindow(ctk.CTkToplevel):
    """Explore color distributions across an image folder."""

    def __init__(self, master=None, **kw):
        super().__init__(master=master, **kw)
        self.title("Color Space Explorer")
        self.geometry("1400x920")
        try:
            self.iconbitmap(resource_path("launch_logo.ico"))
        except Exception:
            pass

        # ——— state ———
        self.input_folder = None
        self.output_folder = None
        self.recursive_var = tk.BooleanVar(value=False)
        self.all_stats = []          # list of dicts (per-image stats)
        self.sampled_pixels = None   # (M, 3) combined sample for scatter
        self.aoi_polygon_pts = None  # list of (x, y) in full-image coords
        self.aoi_mask = None         # uint8 H×W mask (255 = inside AOI)

        # ——— layout: 3 rows — plots | controls | console ———
        self.grid_rowconfigure(0, weight=3)
        self.grid_rowconfigure(1, weight=0)
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # ———————————— TOP: matplotlib plot area ————————————————
        self.top_panel = ctk.CTkFrame(self)
        self.top_panel.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.fig, self.axes = plt.subplots(1, 3, figsize=(15, 4.5))
        self.fig.subplots_adjust(wspace=0.35)
        self.axes[0].set_title("Channel Distributions")
        self.axes[0].axis("off")
        self.axes[1].set_title("2-D Scatter Density")
        self.axes[1].axis("off")
        self.axes[2].set_title("Color Timeline")
        self.axes[2].axis("off")
        self.fig.tight_layout()

        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=self.top_panel)
        self.canvas_plot.get_tk_widget().pack(fill="both", expand=True)

        toolbar_frame = ctk.CTkFrame(self.top_panel, fg_color="transparent")
        toolbar_frame.pack(fill="x")
        self.toolbar = NavigationToolbar2Tk(self.canvas_plot, toolbar_frame)
        self.toolbar.update()

        # ———————————— MIDDLE: controls ————————————————————————
        self.ctrl_panel = ctk.CTkFrame(self)
        self.ctrl_panel.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        # -- Row 1: input folder --
        r1 = ctk.CTkFrame(self.ctrl_panel)
        r1.pack(fill="x", padx=5, pady=2)

        ctk.CTkButton(r1, text="Browse Image Folder",
                      command=self._browse_input).grid(
            row=0, column=0, padx=5, pady=5)
        self.input_label = ctk.CTkLabel(r1, text="No folder selected")
        self.input_label.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        ctk.CTkCheckBox(r1, text="Include sub-folders",
                        variable=self.recursive_var).grid(
            row=0, column=2, padx=10, pady=5)

        # -- Row 2: analysis parameters --
        r2 = ctk.CTkFrame(self.ctrl_panel)
        r2.pack(fill="x", padx=5, pady=2)

        ctk.CTkLabel(r2, text="Analysis Settings",
                     font=("Arial", 12, "bold")).grid(
            row=0, column=0, padx=5, pady=3, sticky="w", columnspan=2)

        # Color space
        ctk.CTkLabel(r2, text="Color space:").grid(
            row=1, column=0, padx=5, pady=3, sticky="e")
        self.cs_var = tk.StringVar(value="RGB")
        self.cs_menu = ctk.CTkOptionMenu(
            r2, variable=self.cs_var,
            values=["RGB", "HSV", "LAB", "Normalised RGB"],
            width=140)
        self.cs_menu.grid(row=1, column=1, padx=5, pady=3, sticky="w")

        # Scatter axes
        ctk.CTkLabel(r2, text="Scatter X:").grid(
            row=1, column=2, padx=5, pady=3, sticky="e")
        self.scatter_x_var = tk.StringVar(value="Ch 0")
        self.scatter_x_menu = ctk.CTkOptionMenu(
            r2, variable=self.scatter_x_var,
            values=["Ch 0", "Ch 1", "Ch 2"], width=80)
        self.scatter_x_menu.grid(row=1, column=3, padx=3, pady=3, sticky="w")

        ctk.CTkLabel(r2, text="Y:").grid(
            row=1, column=4, padx=2, pady=3, sticky="e")
        self.scatter_y_var = tk.StringVar(value="Ch 1")
        self.scatter_y_menu = ctk.CTkOptionMenu(
            r2, variable=self.scatter_y_var,
            values=["Ch 0", "Ch 1", "Ch 2"], width=80)
        self.scatter_y_menu.grid(row=1, column=5, padx=3, pady=3, sticky="w")

        # Sky / ground crop
        ctk.CTkLabel(r2, text="Skip sky (%):").grid(
            row=1, column=6, padx=5, pady=3, sticky="e")
        self.sky_entry = ctk.CTkEntry(r2, width=45)
        self.sky_entry.insert(0, "0")
        self.sky_entry.grid(row=1, column=7, padx=3, pady=3)

        ctk.CTkLabel(r2, text="Skip ground (%):").grid(
            row=1, column=8, padx=5, pady=3, sticky="e")
        self.ground_entry = ctk.CTkEntry(r2, width=45)
        self.ground_entry.insert(0, "0")
        self.ground_entry.grid(row=1, column=9, padx=3, pady=3)

        # Pixel sample size per image (to keep scatter manageable)
        ctk.CTkLabel(r2, text="Pixels/image:").grid(
            row=1, column=10, padx=5, pady=3, sticky="e")
        self.sample_entry = ctk.CTkEntry(r2, width=60)
        self.sample_entry.insert(0, "2000")
        self.sample_entry.grid(row=1, column=11, padx=3, pady=3)

        # Outlier threshold
        ctk.CTkLabel(r2, text="Outlier σ:").grid(
            row=1, column=12, padx=5, pady=3, sticky="e")
        self.sigma_entry = ctk.CTkEntry(r2, width=45)
        self.sigma_entry.insert(0, "2.5")
        self.sigma_entry.grid(row=1, column=13, padx=3, pady=3)

        # Timeline statistic
        ctk.CTkLabel(r2, text="Timeline stat:").grid(
            row=0, column=2, padx=5, pady=3, sticky="e")
        self.timeline_stat_var = tk.StringVar(value="mean")
        ctk.CTkOptionMenu(
            r2, variable=self.timeline_stat_var,
            values=["mean", "median", "std", "p05-p95 range"],
            width=120).grid(row=0, column=3, padx=3, pady=3, sticky="w")

        # Timestamp format (optional)
        ctk.CTkLabel(r2, text="Timestamp fmt (opt):").grid(
            row=0, column=4, padx=5, pady=3, sticky="e")
        self.ts_fmt_entry = ctk.CTkEntry(r2, width=140,
                                         placeholder_text="%Y_%m_%d_%H_%M")
        self.ts_fmt_entry.grid(row=0, column=5, padx=3, pady=3,
                               sticky="w", columnspan=2)

        # -- Row 2b: AOI polygon & Feature class --
        r2b = ctk.CTkFrame(self.ctrl_panel)
        r2b.pack(fill="x", padx=5, pady=2)

        ctk.CTkLabel(r2b, text="AOI & Label",
                     font=("Arial", 12, "bold")).grid(
            row=0, column=0, padx=5, pady=3, sticky="w")

        ctk.CTkButton(r2b, text="Draw AOI Polygon",
                      command=self._draw_aoi_polygon,
                      width=130).grid(
            row=0, column=1, padx=5, pady=3)

        ctk.CTkButton(r2b, text="Clear AOI",
                      command=self._clear_aoi,
                      width=80, fg_color="#8B0000",
                      hover_color="#A52A2A").grid(
            row=0, column=2, padx=5, pady=3)

        self.aoi_status_label = ctk.CTkLabel(
            r2b, text="No AOI set (full image)",
            font=("Arial", 10), text_color="gray")
        self.aoi_status_label.grid(row=0, column=3, padx=5, pady=3,
                                   sticky="w")

        # Separator
        ctk.CTkLabel(r2b, text="│", text_color="gray").grid(
            row=0, column=4, padx=8, pady=3)

        # Feature class label
        ctk.CTkLabel(r2b, text="Feature class:").grid(
            row=0, column=5, padx=5, pady=3, sticky="e")
        self.feature_class_entry = ctk.CTkEntry(
            r2b, width=150,
            placeholder_text="e.g. water, sand, vegetation")
        self.feature_class_entry.grid(row=0, column=6, padx=5, pady=3,
                                      sticky="w")

        ctk.CTkLabel(
            r2b,
            text="→ Draw polygon on first image; label appears in CSV output",
            font=("Arial", 10), text_color="gray",
        ).grid(row=0, column=7, padx=5, pady=3, sticky="w")

        # -- Row 3: output, run, reset --
        r3 = ctk.CTkFrame(self.ctrl_panel)
        r3.pack(fill="x", padx=5, pady=2)

        ctk.CTkButton(r3, text="Browse Output Folder",
                      command=self._browse_output, fg_color="#8C7738").grid(
            row=0, column=0, padx=5, pady=5)
        self.output_label = ctk.CTkLabel(r3, text="No output folder selected")
        self.output_label.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        self.progress_bar = ctk.CTkProgressBar(r3, width=200)
        self.progress_bar.grid(row=0, column=2, padx=10, pady=5)
        self.progress_bar.set(0)

        self.eta_label = ctk.CTkLabel(r3, text="", font=("Arial", 10),
                                      text_color="gray")
        self.eta_label.grid(row=0, column=3, padx=3, pady=5, sticky="w")

        ctk.CTkButton(r3, text="Run Analysis",
                      command=self._run_threaded,
                      fg_color="#0F52BA").grid(
            row=0, column=4, padx=10, pady=5)

        ctk.CTkButton(r3, text="Reset", command=self._reset,
                      width=80, fg_color="#8B0000",
                      hover_color="#A52A2A").grid(
            row=0, column=5, padx=5, pady=5)

        ctk.CTkLabel(
            r3,
            text="→ Generates histograms, scatter density, timeline & outlier report",
            font=("Arial", 10), text_color="gray",
        ).grid(row=0, column=6, padx=5, pady=5, sticky="w")

        # ———————————— BOTTOM: console ————————————————————————
        self.console_frame = ctk.CTkFrame(self)
        self.console_frame.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        self.console_text = tk.Text(self.console_frame, wrap="word", height=8)
        self.console_text.pack(fill="both", expand=True, padx=5, pady=5)
        self.stdout_redirector = StdoutRedirector(self.console_text)
        sys.stdout = self.stdout_redirector
        sys.stderr = self.stdout_redirector

        self._print_guide()

    # ——————————————————————————— guide text ———————————————————————————
    def _print_guide(self):
        print(
            "Color Space Explorer — Tool Guide\n"
            "====================================\n"
            "\n"
            "DISPLAY PANEL (top — three plots)\n"
            "  Left:   Smoothed channel distribution curves across ALL images.\n"
            "  Centre: 2-D scatter density of two selected channels (sampled pixels).\n"
            "  Right:  Color timeline — channel statistic vs. image timestamp.\n"
            "\n"
            "ANALYSIS SETTINGS\n"
            "  Color space       — RGB, HSV, LAB, or Normalised RGB (r=R/sum).\n"
            "  Scatter X / Y     — Which channels to plot on the 2-D scatter.\n"
            "                      Ch 0/1/2 map to the selected color space channels.\n"
            "  Skip sky (%)      — Ignore the top N% of the image (bright sky).\n"
            "  Skip ground (%)   — Ignore the bottom N% of the image.\n"
            "  Pixels/image      — Random pixel sample per image for the scatter plot.\n"
            "                      Higher = denser scatter, slower processing.\n"
            "  Outlier σ         — Images whose mean channel value deviates by more\n"
            "                      than this many standard deviations are flagged.\n"
            "  Timeline stat     — Which statistic to plot over time:\n"
            "                        mean, median, std, or p05–p95 range.\n"
            "  Timestamp fmt     — Optional strftime format for filenames.\n"
            "                      Leave blank to auto-detect (common patterns).\n"
            "\n"
            "AOI & LABEL\n"
            "  Draw AOI Polygon  — Click vertices on the first image to define\n"
            "                      a region of interest.  Only pixels inside the\n"
            "                      polygon are analysed (applied to ALL images).\n"
            "  Clear AOI         — Revert to full-image analysis.\n"
            "  Feature class     — Free-text label (e.g. 'water', 'sand').\n"
            "                      Written to every row of color_stats.csv so\n"
            "                      you can run the tool multiple times with\n"
            "                      different AOIs and merge the CSVs.\n"
            "\n"
            "OUTPUTS (saved to the output folder)\n"
            "  color_stats.csv   — Per-image channel statistics + class label.\n"
            "  outliers.txt      — Images flagged as color outliers.\n"
            "\n"
            "WORKFLOW TIPS\n"
            "  • Use RGB or LAB to inspect general brightness / color drift.\n"
            "  • Use HSV to find optimal Hue–Saturation ranges for the HSV Mask tool.\n"
            "  • The timeline reveals day/night cycles, fog events, and seasonal shifts.\n"
            "  • Outlier detection pre-screens before harmonisation.\n"
            "  • Draw an AOI around 'water' → run → change AOI to 'sand' → run again\n"
            "    → merge the two CSVs to compare color signatures of different features.\n"
            "====================================\n"
        )

    # ——————————————————————————— browse / reset ———————————————————————
    def _browse_input(self):
        d = filedialog.askdirectory(title="Select Image Folder")
        if d:
            self.input_folder = d
            self.input_label.configure(text=d)

    def _browse_output(self):
        d = filedialog.askdirectory(title="Select Output Folder")
        if d:
            self.output_folder = d
            self.output_label.configure(text=d)

    def _reset(self):
        self.input_folder = None
        self.output_folder = None
        self.all_stats = []
        self.sampled_pixels = None
        self.aoi_polygon_pts = None
        self.aoi_mask = None
        self.input_label.configure(text="No folder selected")
        self.output_label.configure(text="No output folder selected")
        self.aoi_status_label.configure(text="No AOI set (full image)")
        self.feature_class_entry.delete(0, tk.END)
        self.progress_bar.set(0)
        self.eta_label.configure(text="")
        self.recursive_var.set(False)

        for ax in self.axes:
            ax.clear()
            ax.axis("off")
        self.axes[0].set_title("Channel Distributions")
        self.axes[1].set_title("2-D Scatter Density")
        self.axes[2].set_title("Color Timeline")
        self.fig.tight_layout()
        self.canvas_plot.draw()
        self.console_text.delete("1.0", tk.END)
        print("Session reset.\n--------------------------------")

    # ——————————————————————————— AOI polygon drawing ——————————————————
    def _draw_aoi_polygon(self):
        """Open a popup showing the first image in the folder; user clicks
        vertices to define a polygon AOI.  Double-click to close."""
        if not self.input_folder:
            messagebox.showwarning("AOI", "Select an image folder first.")
            return

        images = collect_images(self.input_folder, self.recursive_var.get())
        if not images:
            messagebox.showwarning("AOI", "No images found in the folder.")
            return

        # Load the first image for reference
        ref_path = images[0]
        ref_bgr = cv2.imread(str(ref_path))
        if ref_bgr is None:
            messagebox.showerror("AOI", f"Cannot read: {ref_path.name}")
            return

        ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(ref_rgb)

        self.aoi_polygon_pts = []

        win = tk.Toplevel(self)
        win.title("Draw AOI Polygon — click vertices, double-click to close")
        win.geometry("920x680")
        win.grab_set()

        canvas = tk.Canvas(win, cursor="cross", bg="black")
        canvas.pack(fill="both", expand=True)

        info_frame = tk.Frame(win)
        info_frame.pack(fill="x")
        info_label = tk.Label(
            info_frame,
            text="Click to add vertices.  Double-click to close polygon.  "
                 f"Image: {ref_path.name}")
        info_label.pack(side="left", padx=10)
        coord_label = tk.Label(info_frame, text="Vertices: 0")
        coord_label.pack(side="right", padx=10)

        # Scale image to fit the popup
        max_w, max_h = 900, 620
        iw, ih = pil_img.size
        scale = min(max_w / iw, max_h / ih, 1.0)
        disp_w = max(1, int(iw * scale))
        disp_h = max(1, int(ih * scale))
        disp_img = pil_img.resize((disp_w, disp_h), Image.BILINEAR)
        tk_img = ImageTk.PhotoImage(disp_img)
        canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
        canvas.image = tk_img  # keep reference

        # Store full-image dimensions for the mask
        self._aoi_ref_shape = ref_bgr.shape[:2]  # (H, W)

        def on_click(event):
            # Convert display coords → full-image coords
            x_orig = int(event.x / scale)
            y_orig = int(event.y / scale)
            self.aoi_polygon_pts.append((x_orig, y_orig))
            # Draw vertex dot
            r = 4
            canvas.create_oval(event.x - r, event.y - r,
                               event.x + r, event.y + r,
                               fill="lime", outline="lime")
            # Draw edge to previous vertex
            pts = self.aoi_polygon_pts
            if len(pts) >= 2:
                p1, p2 = pts[-2], pts[-1]
                canvas.create_line(
                    p1[0] * scale, p1[1] * scale,
                    p2[0] * scale, p2[1] * scale,
                    fill="lime", width=2)
            coord_label.config(text=f"Vertices: {len(pts)}")

        def on_double_click(event):
            pts = self.aoi_polygon_pts
            if len(pts) < 3:
                messagebox.showwarning("AOI",
                                       "Need at least 3 vertices.")
                return
            # Close polygon visually
            p1, p2 = pts[-1], pts[0]
            canvas.create_line(
                p1[0] * scale, p1[1] * scale,
                p2[0] * scale, p2[1] * scale,
                fill="lime", width=2, dash=(4, 2))

            # Build the binary mask
            h, w = self._aoi_ref_shape
            mask = np.zeros((h, w), dtype=np.uint8)
            np_pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [np_pts], 255)
            self.aoi_mask = mask

            n_inside = int(cv2.countNonZero(mask))
            total = h * w
            pct = 100.0 * n_inside / total
            self.aoi_status_label.configure(
                text=f"AOI set: {len(pts)} vertices, "
                     f"{pct:.1f}% of image")
            print(f"[AOI] Polygon closed: {len(pts)} vertices, "
                  f"{n_inside}/{total} pixels ({pct:.1f}%)")

            win.after(400, win.destroy)

        canvas.bind("<Button-1>", on_click)
        canvas.bind("<Double-Button-1>", on_double_click)

    def _clear_aoi(self):
        """Remove the AOI polygon and revert to full-image analysis."""
        self.aoi_polygon_pts = None
        self.aoi_mask = None
        self.aoi_status_label.configure(text="No AOI set (full image)")
        print("[AOI] Cleared — will use full image.")

    # ——————————————————————————— parameter readers ————————————————————
    def _read_params(self):
        """Read and validate all user parameters.  Returns dict or None."""
        cs = self.cs_var.get()

        try:
            sky_pct = float(self.sky_entry.get())
        except ValueError:
            sky_pct = 0.0
        try:
            ground_pct = float(self.ground_entry.get())
        except ValueError:
            ground_pct = 0.0
        sky_frac = max(0.0, min(sky_pct / 100.0, 0.9))
        ground_frac = max(0.0, min(ground_pct / 100.0, 0.9))
        if sky_frac + ground_frac >= 1.0:
            messagebox.showwarning("Warning",
                                   "Sky + ground crop must be < 100%.")
            return None

        try:
            n_sample = max(100, int(self.sample_entry.get()))
        except ValueError:
            n_sample = 2000

        try:
            sigma_thr = float(self.sigma_entry.get())
        except ValueError:
            sigma_thr = 2.5

        scatter_x = int(self.scatter_x_var.get().split()[-1])
        scatter_y = int(self.scatter_y_var.get().split()[-1])

        ts_fmt = self.ts_fmt_entry.get().strip() or None

        return {
            "color_space": cs,
            "sky_frac": sky_frac,
            "ground_frac": ground_frac,
            "n_sample": n_sample,
            "sigma_thr": sigma_thr,
            "scatter_x": scatter_x,
            "scatter_y": scatter_y,
            "timeline_stat": self.timeline_stat_var.get(),
            "ts_fmt": ts_fmt,
        }

    # ——————————————————————————— run analysis —————————————————————————
    def _run_threaded(self):
        threading.Thread(target=self._run_analysis, daemon=True).start()

    def _format_eta(self, elapsed, done, total):
        """Return a human-readable ETA string."""
        if done <= 0:
            return ""
        rate = elapsed / done
        remaining = rate * (total - done)
        if remaining < 60:
            return f"{remaining:.0f}s left"
        elif remaining < 3600:
            m, s = divmod(remaining, 60)
            return f"{int(m)}m {int(s)}s left"
        else:
            h, rem = divmod(remaining, 3600)
            m = rem // 60
            return f"{int(h)}h {int(m)}m left"

    def _update_eta(self, text):
        """Thread-safe update of the ETA label."""
        try:
            self.eta_label.configure(text=text)
        except Exception:
            pass

    def _run_analysis(self):
        try:
            self.progress_bar.set(0)
            self._update_eta("")

            if not self.input_folder:
                messagebox.showwarning("Warning",
                                       "Select an image folder first.")
                return

            params = self._read_params()
            if params is None:
                return

            cs = params["color_space"]
            ch_names = CS_CHANNELS[cs]
            ch_ranges = CS_RANGES[cs]

            images = collect_images(self.input_folder,
                                    self.recursive_var.get())
            if not images:
                messagebox.showwarning("Warning",
                                       "No images found in the folder.")
                return

            n_img = len(images)
            print(f"Found {n_img} images.  Color space: {cs}")
            print(f"ROI crop — skip sky: {params['sky_frac']*100:.0f}%, "
                  f"skip ground: {params['ground_frac']*100:.0f}%")
            print(f"Pixel sample per image: {params['n_sample']}")

            # AOI info
            aoi = self.aoi_mask
            if aoi is not None:
                n_aoi = int(cv2.countNonZero(aoi))
                print(f"AOI polygon active: {n_aoi} pixels "
                      f"({100*n_aoi/(aoi.shape[0]*aoi.shape[1]):.1f}%)")
            else:
                print("AOI: full image (no polygon set)")

            # Feature class
            feature_class = self.feature_class_entry.get().strip()
            if feature_class:
                print(f"Feature class label: \"{feature_class}\"")

            # ---- pass 1: compute stats + collect pixel samples --------
            all_stats = []
            all_samples = []

            # For the dataset-wide histogram, accumulate per-channel arrays
            hist_data = [[], [], []]  # one list per channel

            t_start = time.time()

            for idx, img_path in enumerate(images):
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"  [SKIP] Cannot read: {img_path.name}")
                    continue

                pixels = convert_pixels(img, cs,
                                        params["sky_frac"],
                                        params["ground_frac"],
                                        aoi_mask=aoi)
                n_px = pixels.shape[0]

                if n_px == 0:
                    print(f"  [SKIP] No pixels in AOI: {img_path.name}")
                    continue

                # per-image stats
                stats = image_channel_stats(pixels)
                stats["filename"] = img_path.name
                stats["filepath"] = str(img_path)

                # parse timestamp
                dt = parse_datetime_from_filename(img_path.name,
                                                  params["ts_fmt"])
                stats["datetime"] = dt
                all_stats.append(stats)

                # random pixel sample for scatter
                n_take = min(params["n_sample"], n_px)
                indices = np.random.choice(n_px, size=n_take, replace=False)
                sample = pixels[indices]
                all_samples.append(sample)

                # subsample for histograms (cap at 5000 per image to
                # keep memory bounded for very large datasets)
                n_hist = min(5000, n_px)
                h_idx = np.random.choice(n_px, size=n_hist, replace=False)
                for ch in range(3):
                    hist_data[ch].append(pixels[h_idx, ch])

                done = idx + 1
                frac = done / n_img * 0.6
                self.progress_bar.set(frac)
                elapsed = time.time() - t_start
                eta_str = self._format_eta(elapsed, done, n_img)
                if (done) % 5 == 0 or done == 1 or done == n_img:
                    self._update_eta(f"{done}/{n_img}  ~{eta_str}")
                if done % 20 == 0 or done == 1 or done == n_img:
                    print(f"  {done}/{n_img} | {img_path.name}  "
                          f"[{eta_str}]")

            if not all_stats:
                messagebox.showwarning("Warning", "No readable images found.")
                return

            self.all_stats = all_stats
            self.sampled_pixels = np.vstack(all_samples) if all_samples else None

            print(f"\nProcessed {len(all_stats)} images, "
                  f"{self.sampled_pixels.shape[0]} sampled pixels total.")

            # ---- Plot 1: channel histograms (smooth KDE curves) ---------
            print("Generating channel histograms …")
            ax0 = self.axes[0]
            ax0.clear()
            colors_hist = ["#e74c3c", "#2ecc71", "#3498db"]

            for ch in range(3):
                data = np.concatenate(hist_data[ch])
                lo, hi = ch_ranges[ch]
                n_bins = 120
                bins = np.linspace(lo, hi, n_bins + 1)
                counts, edges = np.histogram(data, bins=bins, density=True)
                # bin centres for the curve
                centres = 0.5 * (edges[:-1] + edges[1:])
                # simple moving-average smooth (window=5) to get a
                # clean curve without requiring scipy
                kernel_size = 5
                kernel = np.ones(kernel_size) / kernel_size
                smooth = np.convolve(counts, kernel, mode="same")
                # thick curve on top, light transparent fill underneath
                ax0.fill_between(centres, smooth, alpha=0.15,
                                 color=colors_hist[ch])
                ax0.plot(centres, smooth, color=colors_hist[ch],
                         linewidth=2.2, label=ch_names[ch])
            ax0.set_title(f"Channel Distributions ({cs})")
            ax0.set_xlabel("Value")
            ax0.set_ylabel("Density")
            ax0.legend(fontsize="small")
            ax0.grid(True, alpha=0.2)
            self.progress_bar.set(0.7)

            # ---- Plot 2: 2-D scatter density --------------------------
            print("Generating 2-D scatter density …")
            ax1 = self.axes[1]
            ax1.clear()
            sx = params["scatter_x"]
            sy = params["scatter_y"]

            if self.sampled_pixels is not None and len(self.sampled_pixels) > 0:
                x_data = self.sampled_pixels[:, sx]
                y_data = self.sampled_pixels[:, sy]

                # Use hexbin for large point clouds — much faster and
                # more readable than individual scatter points
                hb = ax1.hexbin(x_data, y_data,
                                gridsize=80, cmap="inferno",
                                mincnt=1, linewidths=0.1)
                self.fig.colorbar(hb, ax=ax1, label="count", shrink=0.8)
                ax1.set_xlim(ch_ranges[sx])
                ax1.set_ylim(ch_ranges[sy])

            ax1.set_title(f"Scatter: {ch_names[sx]} vs {ch_names[sy]}")
            ax1.set_xlabel(ch_names[sx])
            ax1.set_ylabel(ch_names[sy])
            ax1.grid(True, alpha=0.2)
            self.progress_bar.set(0.8)

            # ---- Plot 3: color timeline -------------------------------
            print("Generating color timeline …")
            ax2 = self.axes[2]
            ax2.clear()

            # filter entries with valid datetimes
            dated = [s for s in all_stats if s["datetime"] is not None]
            dated.sort(key=lambda s: s["datetime"])

            if dated:
                dts = [s["datetime"] for s in dated]
                stat_key = params["timeline_stat"]

                for ch in range(3):
                    if stat_key == "mean":
                        vals = [s[f"mean_{ch}"] for s in dated]
                        ax2.plot(dts, vals, color=colors_hist[ch],
                                 label=ch_names[ch], alpha=0.8, linewidth=0.8)
                    elif stat_key == "median":
                        vals = [s[f"med_{ch}"] for s in dated]
                        ax2.plot(dts, vals, color=colors_hist[ch],
                                 label=ch_names[ch], alpha=0.8, linewidth=0.8)
                    elif stat_key == "std":
                        vals = [s[f"std_{ch}"] for s in dated]
                        ax2.plot(dts, vals, color=colors_hist[ch],
                                 label=ch_names[ch], alpha=0.8, linewidth=0.8)
                    elif stat_key == "p05-p95 range":
                        lo_vals = [s[f"p05_{ch}"] for s in dated]
                        hi_vals = [s[f"p95_{ch}"] for s in dated]
                        ax2.fill_between(dts, lo_vals, hi_vals,
                                         color=colors_hist[ch],
                                         alpha=0.25)
                        mid = [s[f"med_{ch}"] for s in dated]
                        ax2.plot(dts, mid, color=colors_hist[ch],
                                 label=ch_names[ch], alpha=0.8,
                                 linewidth=0.8)

                ax2.set_title(f"Timeline: {stat_key} ({cs})")
                ax2.set_xlabel("Date")
                ax2.set_ylabel("Value")
                ax2.legend(fontsize="small")
                ax2.grid(True, alpha=0.2)
                ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
                self.fig.autofmt_xdate()
                print(f"  {len(dated)} images with parseable timestamps.")
            else:
                ax2.set_title("Timeline (no timestamps detected)")
                ax2.text(0.5, 0.5,
                         "Could not parse timestamps\nfrom filenames.",
                         ha="center", va="center",
                         transform=ax2.transAxes, fontsize=10,
                         color="gray")
                print("  No timestamps could be parsed — timeline skipped.")

            self.fig.tight_layout()
            self.canvas_plot.draw()
            self.progress_bar.set(0.9)

            # ---- Outlier detection ------------------------------------
            print("\nOutlier detection …")
            sigma = params["sigma_thr"]
            means_arr = np.array([
                [s["mean_0"], s["mean_1"], s["mean_2"]]
                for s in all_stats
            ])
            global_mean = means_arr.mean(axis=0)
            global_std = means_arr.std(axis=0)
            global_std[global_std == 0] = 1.0  # avoid div-by-zero

            outliers = []
            for i, s in enumerate(all_stats):
                z_scores = np.abs(
                    (means_arr[i] - global_mean) / global_std)
                max_z = float(z_scores.max())
                if max_z > sigma:
                    ch_worst = int(z_scores.argmax())
                    outliers.append({
                        "filename": s["filename"],
                        "max_z": max_z,
                        "channel": ch_names[ch_worst],
                        "value": means_arr[i, ch_worst],
                    })

            if outliers:
                outliers.sort(key=lambda o: o["max_z"], reverse=True)
                print(f"  Found {len(outliers)} outlier images "
                      f"(>{sigma:.1f}σ):")
                for o in outliers[:15]:
                    print(f"    {o['filename']}  z={o['max_z']:.2f}  "
                          f"({o['channel']}={o['value']:.1f})")
                if len(outliers) > 15:
                    print(f"    … and {len(outliers)-15} more.")
            else:
                print(f"  No outliers found (threshold: {sigma:.1f}σ).")

            # ---- Save outputs -----------------------------------------
            if self.output_folder:
                aoi_active = "yes" if aoi is not None else "no"

                # CSV
                csv_path = os.path.join(self.output_folder,
                                        "color_stats.csv")
                with open(csv_path, "w", encoding="utf-8") as f:
                    header = (
                        "filename,datetime,feature_class,aoi,"
                        f"{ch_names[0]}_mean,{ch_names[1]}_mean,"
                        f"{ch_names[2]}_mean,"
                        f"{ch_names[0]}_std,{ch_names[1]}_std,"
                        f"{ch_names[2]}_std,"
                        f"{ch_names[0]}_median,{ch_names[1]}_median,"
                        f"{ch_names[2]}_median,"
                        f"{ch_names[0]}_p05,{ch_names[1]}_p05,"
                        f"{ch_names[2]}_p05,"
                        f"{ch_names[0]}_p95,{ch_names[1]}_p95,"
                        f"{ch_names[2]}_p95\n"
                    )
                    f.write(header)
                    for s in all_stats:
                        dt_str = (s["datetime"].strftime("%Y-%m-%d %H:%M:%S")
                                  if s["datetime"] else "")
                        fc_safe = feature_class.replace(",", ";")
                        row = (
                            f"{s['filename']},{dt_str},"
                            f"{fc_safe},{aoi_active},"
                            f"{s['mean_0']:.3f},{s['mean_1']:.3f},"
                            f"{s['mean_2']:.3f},"
                            f"{s['std_0']:.3f},{s['std_1']:.3f},"
                            f"{s['std_2']:.3f},"
                            f"{s['med_0']:.3f},{s['med_1']:.3f},"
                            f"{s['med_2']:.3f},"
                            f"{s['p05_0']:.3f},{s['p05_1']:.3f},"
                            f"{s['p05_2']:.3f},"
                            f"{s['p95_0']:.3f},{s['p95_1']:.3f},"
                            f"{s['p95_2']:.3f}\n"
                        )
                        f.write(row)
                print(f"\nSaved: {csv_path}")

                # Outlier list
                if outliers:
                    out_path = os.path.join(self.output_folder,
                                            "outliers.txt")
                    with open(out_path, "w", encoding="utf-8") as f:
                        f.write(f"# Outlier images (>{sigma:.1f}σ "
                                f"from dataset mean)\n")
                        f.write(f"# Color space: {cs}\n")
                        if feature_class:
                            f.write(f"# Feature class: {feature_class}\n")
                        if aoi is not None:
                            f.write("# AOI polygon was active\n")
                        f.write("filename\tmax_z_score\tchannel\tvalue\n")
                        for o in outliers:
                            f.write(f"{o['filename']}\t{o['max_z']:.3f}\t"
                                    f"{o['channel']}\t{o['value']:.3f}\n")
                    print(f"Saved: {out_path}")
            else:
                print("\nNo output folder selected — CSV/outlier "
                      "files not saved.")

            self.progress_bar.set(1.0)
            total_elapsed = time.time() - t_start
            if total_elapsed < 60:
                elapsed_str = f"{total_elapsed:.1f}s"
            else:
                em, es = divmod(total_elapsed, 60)
                elapsed_str = f"{int(em)}m {int(es)}s"
            self._update_eta(f"Done in {elapsed_str}")
            print(f"\n✓ Analysis complete ({elapsed_str}).")
            messagebox.showinfo("Done",
                                f"Analysed {len(all_stats)} images "
                                f"in {elapsed_str}.\n"
                                f"Outliers: {len(outliers)}")

        except Exception as e:
            print(f"\n[ERROR] {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", str(e))


# ——— standalone ———
if __name__ == "__main__":
    root = ctk.CTk()
    root.withdraw()
    ColorSpaceExplorerWindow(master=root)
    root.mainloop()