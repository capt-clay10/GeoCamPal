"""
Harmonise Images

Three sub‑tasks in one window:
  1. **Filter bad images** — robust blur, clipping‑based over‑exposure,
     under‑exposure, darkness, low‑information (entropy), rain/droplets
     on lens, and partial obstruction detection.
  2. **Harmonise brightness** — luminance‑based gain (with soft‑knee) or
     L‑channel histogram matching, optional sky masking for reference
     computation, and post‑correction sanity check.
  3. **Harmonise colour** — full‑colour transfer to a user‑selected
     reference image using Reinhard, per‑channel LAB histogram matching,
     or iterative distribution transfer (Pitié et al., 2007).

Both harmonisation stages include a **preview** system: a random 5 %
sample (minimum 1) is processed first and shown as before/after pairs
with Next/Previous navigation.  The user can inspect and then commit
with "Process All".

Outputs:
  • bad_images.txt / bad_images.json  — list of identified bad images and reasons
  • _filtered_good/                   — optional export of images that passed filtering
  • _brightness_harmonised/           — brightness‑corrected images
  • _colour_harmonised/               — colour‑corrected images

Originals are NEVER modified or overwritten.
"""

# %% ————————————————————————————— imports —————————————————————————————
import sys
import os
import glob
import shutil
import threading
import time
import random
import json
from pathlib import Path

import numpy as np
import cv2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox

from utils import (
    fit_geometry,
    resource_path,
    setup_console,
    restore_console,
    save_settings_json,
    load_settings_json,
)

try:
    from skimage import exposure as sk_exposure
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# %% ————————————————————————————— image quality filters ———————————————

def is_blurry(img_bgr, lap_threshold, uniform_fraction=0.85):
    """
    Robust blur detector.
    Uses Laplacian variance, but guards against false positives on
    genuinely textureless scenes (calm water, overcast sky) by checking
    whether the image is mostly uniform *before* flagging it as blurry.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var < lap_threshold:
        # check if the scene is genuinely textureless (not blurry)
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
        low_gradient_frac = np.sum(np.abs(sobel) < 5) / sobel.size
        if low_gradient_frac > uniform_fraction:
            return False  # uniform scene, not actually blurry
        return True
    return False


def is_clipped_overexposed(img_bgr, clip_fraction=0.15):
    """
    Saturation‑based overexposure: checks for pixels clipped at 254–255
    across ALL three channels (no recoverable information), rather than
    just "bright" single‑channel pixels.  This avoids false positives
    on naturally bright scenes like white sand.
    """
    saturated = np.all(img_bgr >= 254, axis=2)
    return (saturated.sum() / saturated.size) > clip_fraction


def is_underexposed(img_bgr, dark_px_value, fraction):
    """Too many dark pixels (single‑channel check on grayscale)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    n_dark = np.sum(gray <= dark_px_value)
    return (n_dark / gray.size) > fraction


def is_too_dark(img_bgr, max_threshold):
    """Entire image maximum below threshold → near-black / nighttime."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return int(gray.max()) < max_threshold


def is_low_information(img_bgr, entropy_threshold=5.0):
    """
    Entropy‑based low‑information detector.  Catches fog, haze,
    washed‑out images, and lens condensation better than simple std.
    Typical good outdoor images have entropy 6–7.5; fog/haze drops
    below 5.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    return entropy < entropy_threshold


def has_lens_droplets(img_bgr, blob_fraction=0.08):
    """
    Rain / condensation droplet detector.
    Droplets create localised bright blobs surrounded by blur halos.
    We detect them as bright residuals after heavy Gaussian smoothing.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (31, 31), 0)
    diff = cv2.absdiff(gray, blurred)
    # droplets produce concentrated bright spots in the diff image
    bright_local = np.sum(diff > 40) / diff.size
    return bright_local > blob_fraction


def has_obstruction(img_bgr, grid_size=4, dark_mean_thresh=15,
                    block_fraction=0.35):
    """
    Partial obstruction detector (bird on housing, lens cap edge,
    foreign object).  Divides the image into a grid and checks whether
    a large contiguous region is near‑black.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    bh, bw = h // grid_size, w // grid_size
    dark_blocks = 0
    total_blocks = grid_size * grid_size
    for i in range(grid_size):
        for j in range(grid_size):
            block = gray[i * bh:(i + 1) * bh, j * bw:(j + 1) * bw]
            if block.mean() < dark_mean_thresh:
                dark_blocks += 1
    return (dark_blocks / total_blocks) > block_fraction


def filter_image(img_bgr,
                 blur_thresh,
                 clip_frac,
                 underexp_val, underexp_frac,
                 dark_thresh,
                 entropy_thresh,
                 droplet_frac,
                 obstruction_frac):
    """
    Run all filters.  Returns a list of reason strings; empty = passes.
    """
    reasons = []
    if is_blurry(img_bgr, blur_thresh):
        reasons.append("blurry")
    if is_clipped_overexposed(img_bgr, clip_frac):
        reasons.append("clipped_overexposed")
    if is_underexposed(img_bgr, underexp_val, underexp_frac):
        reasons.append("underexposed")
    if is_too_dark(img_bgr, dark_thresh):
        reasons.append("too_dark")
    if is_low_information(img_bgr, entropy_thresh):
        reasons.append("low_information")
    if has_lens_droplets(img_bgr, droplet_frac):
        reasons.append("lens_droplets")
    if has_obstruction(img_bgr, block_fraction=obstruction_frac):
        reasons.append("obstruction")
    return reasons


# %% ————————————————————————————— brightness harmonisation helpers ————

def mean_luminance(img_bgr, sky_mask_frac=0.0):
    """
    Mean L‑channel value (CIE LAB).
    If sky_mask_frac > 0, the top N% of the image is excluded so that
    bright skies don't skew the reference luminance.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L = lab[..., 0]
    if sky_mask_frac > 0:
        h = L.shape[0]
        cut = int(h * sky_mask_frac)
        if cut > 0:
            L = L[cut:, :]
    return float(L.mean())


def _preserve_saturation(original_bgr, corrected_bgr):
    """
    Restore the original per‑pixel HSV saturation after a brightness
    correction.

    Modifying only the L channel in LAB and converting back to BGR does
    NOT perfectly preserve perceived colour saturation — the LAB→BGR
    gamut mapping clips some channel combinations, especially on images
    that are already low‑chroma (coastal, overcast, dawn/dusk scenes).
    Even a modest L shift can visibly desaturate these images.

    Fix: replace the corrected image's HSV‑S channel with the original's
    while keeping the corrected H and V.  This ensures the brightness
    change takes effect without washing out colour.
    """
    orig_hsv = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2HSV)
    corr_hsv = cv2.cvtColor(corrected_bgr, cv2.COLOR_BGR2HSV)
    # keep corrected brightness (V) and hue (H), restore original saturation (S)
    corr_hsv[..., 1] = orig_hsv[..., 1]
    return cv2.cvtColor(corr_hsv, cv2.COLOR_HSV2BGR)


def soft_gain_match(img_bgr, ref_mean_L, knee=200):
    """
    Luminance gain with soft‑knee rolloff above *knee* to avoid hard
    highlight clipping.  Pixels below the knee are scaled linearly;
    above the knee the gain tapers off smoothly toward 255.

    Includes saturation preservation to prevent colour washout on
    low‑chroma scenes (coastal, overcast, HDR panoramas).
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0]
    current_mean = max(L.mean(), 1e-6)
    gain = ref_mean_L / current_mean

    result = L * gain
    # soft rolloff above knee
    above = result > knee
    if above.any():
        excess = result[above] - knee
        headroom = 255.0 - knee
        result[above] = knee + headroom * excess / (excess + headroom)

    lab[..., 0] = np.clip(result, 0, 255)
    corrected = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return _preserve_saturation(img_bgr, corrected)


def hist_match(img_bgr, ref_bgr):
    """
    Full L‑channel histogram matching using scikit‑image.
    Falls back to soft_gain if skimage is not available.

    Includes saturation preservation to prevent colour washout.
    """
    if not HAS_SKIMAGE:
        return soft_gain_match(img_bgr, mean_luminance(ref_bgr))
    ref_lab = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB)
    tgt_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    matched_L = sk_exposure.match_histograms(
        tgt_lab[..., 0], ref_lab[..., 0], channel_axis=None)
    tgt_lab[..., 0] = np.clip(matched_L, 0, 255).astype(np.uint8)
    corrected = cv2.cvtColor(tgt_lab, cv2.COLOR_LAB2BGR)
    return _preserve_saturation(img_bgr, corrected)


def correction_sanity_check(original_bgr, corrected_bgr, ref_mean_L,
                            sky_mask_frac=0.0):
    """
    Verify correction actually improved things.  If the corrected image's
    mean L is *further* from the reference than the original, something
    went wrong (can happen with histogram matching on unusual distributions).
    Returns (image, reverted_bool).
    """
    orig_L = mean_luminance(original_bgr, sky_mask_frac)
    corr_L = mean_luminance(corrected_bgr, sky_mask_frac)
    if abs(corr_L - ref_mean_L) > abs(orig_L - ref_mean_L):
        return original_bgr, True  # reverted
    return corrected_bgr, False


# %% ————————————————————————————— colour harmonisation helpers ————————

def reinhard_colour_transfer(src_bgr, ref_bgr):
    """
    Reinhard et al. (2001) colour transfer.
    Converts both images to CIE LAB, then for each channel independently
    shifts the mean and scales the standard deviation of the source to
    match the reference.  Fast, effective for same‑scene time‑lapse.
    """
    src_lab = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    ref_lab = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)

    for ch in range(3):
        src_ch = src_lab[..., ch]
        ref_ch = ref_lab[..., ch]

        src_mean, src_std = src_ch.mean(), max(src_ch.std(), 1e-6)
        ref_mean, ref_std = ref_ch.mean(), max(ref_ch.std(), 1e-6)

        # shift and scale
        src_lab[..., ch] = (src_ch - src_mean) * (ref_std / src_std) + ref_mean

    src_lab = np.clip(src_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(src_lab, cv2.COLOR_LAB2BGR)


def lab_histogram_colour_match(src_bgr, ref_bgr):
    """
    Per‑channel histogram matching in CIE LAB.
    Matches the full cumulative distribution of each LAB channel (L, a, b)
    independently using scikit‑image.  Handles larger colour shifts than
    Reinhard but treats channels independently.
    Falls back to Reinhard if scikit‑image is not available.
    """
    if not HAS_SKIMAGE:
        return reinhard_colour_transfer(src_bgr, ref_bgr)

    src_lab = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB)
    ref_lab = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB)

    matched_lab = np.empty_like(src_lab)
    for ch in range(3):
        matched_lab[..., ch] = np.clip(
            sk_exposure.match_histograms(
                src_lab[..., ch].astype(np.float64),
                ref_lab[..., ch].astype(np.float64),
                channel_axis=None),
            0, 255).astype(np.uint8)

    return cv2.cvtColor(matched_lab, cv2.COLOR_LAB2BGR)


def _random_rotation_matrix_3d():
    """Generate a random 3x3 orthogonal rotation matrix via QR decomposition."""
    H = np.random.randn(3, 3)
    Q, R = np.linalg.qr(H)
    # ensure proper rotation (det = +1)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def iterative_distribution_transfer(src_bgr, ref_bgr, iterations=20,
                                     relaxation=0.5):
    """
    Iterative Distribution Transfer (Pitie et al., 2007).

    Treats colour as a 3D distribution in LAB space.  Each iteration:
      1. Generate a random 3D rotation matrix.
      2. Project both source and reference point clouds onto the
         three rotated axes.
      3. Match the 1D histogram along each rotated axis.
      4. Rotate back to LAB space.

    The *relaxation* parameter (0-1) blends each iteration's result
    with the previous to improve convergence stability.  Lower values
    are more conservative; 0.5 works well in practice.

    For performance, the matching statistics are computed on a sub-sample
    of ~300 000 pixels (if the image is larger), but the resulting
    transfer is applied to every pixel via sorted‑rank mapping.
    """
    src_lab = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    ref_lab = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)

    h, w, _ = src_lab.shape
    src_flat = src_lab.reshape(-1, 3).copy()
    ref_flat = ref_lab.reshape(-1, 3)

    n_src = src_flat.shape[0]
    n_ref = ref_flat.shape[0]

    # sub-sample reference for speed (matching target)
    max_pts = 300_000
    if n_ref > max_pts:
        idx_ref = np.random.choice(n_ref, max_pts, replace=False)
        ref_sub = ref_flat[idx_ref]
    else:
        ref_sub = ref_flat

    for it in range(iterations):
        R = _random_rotation_matrix_3d()

        # project into rotated space
        src_rot = src_flat @ R.T
        ref_rot = ref_sub @ R.T

        new_src_rot = src_rot.copy()

        for ch in range(3):
            # sort source pixels along this rotated axis
            src_order = np.argsort(src_rot[:, ch])
            src_sorted = src_rot[src_order, ch]

            # build reference CDF target
            ref_sorted = np.sort(ref_rot[:, ch])

            # interpolate reference quantiles to source length
            ref_quantiles = np.interp(
                np.linspace(0, 1, n_src),
                np.linspace(0, 1, len(ref_sorted)),
                ref_sorted
            )

            # assign matched values
            matched = np.empty(n_src)
            matched[src_order] = ref_quantiles
            new_src_rot[:, ch] = matched

        # rotate back to LAB
        new_src_lab = new_src_rot @ R

        # relaxation: blend with previous iteration
        src_flat = src_flat * (1.0 - relaxation) + new_src_lab * relaxation

    result = src_flat.reshape(h, w, 3)
    result = np.clip(result, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)


def colour_sanity_check(original_bgr, corrected_bgr, ref_bgr):
    """
    Verify colour correction improved similarity to the reference.
    Compares mean LAB distance before and after.  If worse, revert.
    Returns (image, reverted_bool).
    """
    ref_lab = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    orig_lab = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    corr_lab = cv2.cvtColor(corrected_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)

    ref_mean = ref_lab.mean(axis=(0, 1))
    orig_dist = np.linalg.norm(orig_lab.mean(axis=(0, 1)) - ref_mean)
    corr_dist = np.linalg.norm(corr_lab.mean(axis=(0, 1)) - ref_mean)

    if corr_dist > orig_dist:
        return original_bgr, True
    return corrected_bgr, False


COLOUR_ALGORITHMS = {
    "Reinhard (fast)": reinhard_colour_transfer,
    "Histogram Match (LAB)": lab_histogram_colour_match,
    "Iterative Transfer (best)": iterative_distribution_transfer,
}


# %% ————————————————————————————— file helpers ————————————————————————

def collect_images(folder):
    """Collect image paths from a folder (non-recursive)."""
    return sorted(
        p for p in Path(folder).iterdir()
        if p.suffix.lower() in IMAGE_EXTS
    )


def collect_images_recursive(folder):
    """Collect image paths from a folder and all sub-folders."""
    results = []
    for root, _, files in os.walk(folder):
        for f in sorted(files):
            if Path(f).suffix.lower() in IMAGE_EXTS:
                results.append(Path(root) / f)
    return results


def preview_sample_count(n_total):
    """5 % of total, rounded to nearest integer, minimum 1."""
    return max(1, round(n_total * 0.05))


def format_eta(seconds):
    """Format seconds into a human-readable ETA string."""
    if seconds < 0 or not np.isfinite(seconds):
        return "estimating..."
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    elif m > 0:
        return f"{m}m {s:02d}s"
    else:
        return f"{s}s"


# %% ————————————————————————————— main GUI ————————————————————————————
class HarmoniseImagesWindow(ctk.CTkToplevel):
    """Filter bad images, harmonise brightness & harmonise colour."""

    def __init__(self, master=None, **kw):
        super().__init__(master=master, **kw)
        self.title("Harmonise Images")
        #self.geometry("1400x1020")
        fit_geometry(self, 1400, 1020, resizable=True)
        try:
            self.after(200, lambda: self.iconphoto(False, tk.PhotoImage(file=resource_path("launch_logo.png"))))
        except Exception:
            pass

        # ——— close handler ———
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # ——— state ———
        self.input_folder = None
        self.output_folder = None
        self.bad_list = []
        self.bad_details = []
        self.recursive_var = tk.BooleanVar(value=False)
        self.export_filtered_good_var = tk.BooleanVar(value=False)
        self._cancel_requested = False

        # colour harmonisation state
        self.ref_colour_path = None
        self.ref_colour_bgr = None

        # preview state
        self.preview_mode = None        # None | "brightness" | "colour"
        self.preview_pairs = []         # list of (original, corrected, name)
        self.preview_idx = 0
        self._processing = False        # guard against double-clicks

        # ——— layout ———
        self.grid_rowconfigure(0, weight=3)
        self.grid_rowconfigure(1, weight=0)  # console
        self.grid_rowconfigure(2, weight=0)  # preview nav (hidden)
        self.grid_rowconfigure(3, weight=0)  # controls
        self.grid_columnconfigure(0, weight=1)

        # ---- TOP: plot panel ----
        self.top_panel = ctk.CTkFrame(self, fg_color="black")
        self.top_panel.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.fig, self.axes = plt.subplots(1, 3, figsize=(14, 4))
        self.axes[0].set_title("Luminance Distribution")
        self.axes[1].set_title("Filter Results")
        self.axes[1].axis("off")
        self.axes[2].set_title("Harmonisation Results")
        self.axes[2].axis("off")
        self.fig.tight_layout()
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=self.top_panel)
        self.canvas_plot.get_tk_widget().pack(fill="both", expand=True)

        # ---- CONTROLS ----
        self.bottom_panel = ctk.CTkFrame(self)
        self.bottom_panel.grid(row=3, column=0, sticky="nsew", padx=5, pady=5)

        # Row 1 — input folder
        row1 = ctk.CTkFrame(self.bottom_panel)
        row1.pack(fill="x", padx=5, pady=2)

        ctk.CTkButton(row1, text="Browse Image Folder",
                      command=self._browse_input).grid(
            row=0, column=0, padx=5, pady=5)
        self.input_label = ctk.CTkLabel(row1, text="No folder selected")
        self.input_label.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        ctk.CTkCheckBox(row1, text="Include sub-folders",
                        variable=self.recursive_var).grid(
            row=0, column=2, padx=10, pady=5)

        # Row 1b — task selection / collapsible sections
        row1b = ctk.CTkFrame(self.bottom_panel)
        row1b.pack(fill="x", padx=5, pady=2)

        ctk.CTkLabel(row1b, text="Choose Tasks",
                     font=("Arial", 12, "bold")).pack(side="left", padx=5)

        self.run_filter_var = tk.BooleanVar(value=True)
        self.run_brightness_var = tk.BooleanVar(value=True)
        self.run_colour_var = tk.BooleanVar(value=True)
        self.run_average_var = tk.BooleanVar(value=False)

        ctk.CTkCheckBox(row1b, text="Filter bad images",
                        variable=self.run_filter_var,
                        command=self._toggle_task_sections).pack(side="left", padx=8)
        ctk.CTkCheckBox(row1b, text="Harmonise brightness",
                        variable=self.run_brightness_var,
                        command=self._toggle_task_sections).pack(side="left", padx=8)
        ctk.CTkCheckBox(row1b, text="Harmonise colour",
                        variable=self.run_colour_var,
                        command=self._toggle_task_sections).pack(side="left", padx=8)
        ctk.CTkCheckBox(row1b, text="Average per folder",
                        variable=self.run_average_var,
                        command=self._toggle_task_sections).pack(side="left", padx=8)

        # Row 2 — filter parameters (collapsible)
        self.filter_section_frame = ctk.CTkFrame(self.bottom_panel)
        row2 = self.filter_section_frame

        ctk.CTkLabel(row2, text="Filter Bad Images",
                     font=("Arial", 12, "bold")).grid(
            row=0, column=0, padx=5, pady=3, sticky="w", columnspan=2)

        # Each filter gets one checkbox + its parameter entries
        self.filter_entries = {}
        self.filter_enabled = {}

        filters_a = [
            ("Blur", "Blur threshold", "20"),
            ("Overexposure", "Clip overexp fraction", "0.15"),
            ("Underexposure", "Dark pixel value", "20",
                              "Under-exp fraction", "0.25"),
            ("Darkness", "Dark max threshold", "25"),
        ]
        col = 0
        for fdef in filters_a:
            name = fdef[0]
            var = tk.BooleanVar(value=True)
            ctk.CTkCheckBox(row2, text=name, variable=var,
                            width=20).grid(row=1, column=col, padx=2, pady=2, sticky="w")
            self.filter_enabled[name] = var
            col += 1
            params = fdef[1:]
            for i in range(0, len(params), 2):
                plbl, pdef = params[i], params[i + 1]
                e = ctk.CTkEntry(row2, width=50, placeholder_text=plbl)
                e.insert(0, pdef)
                e.grid(row=1, column=col, padx=1, pady=2, sticky="w")
                self.filter_entries[plbl] = e
                col += 1
        max_col = col

        filters_b = [
            ("Low information", "Entropy threshold", "5.0"),
            ("Lens droplets", "Droplet fraction", "0.08"),
            ("Obstruction", "Obstruction fraction", "0.35"),
        ]
        col = 0
        for fdef in filters_b:
            name = fdef[0]
            var = tk.BooleanVar(value=True)
            ctk.CTkCheckBox(row2, text=name, variable=var,
                            width=20).grid(row=2, column=col, padx=2, pady=2, sticky="w")
            self.filter_enabled[name] = var
            col += 1
            params = fdef[1:]
            for i in range(0, len(params), 2):
                plbl, pdef = params[i], params[i + 1]
                e = ctk.CTkEntry(row2, width=50, placeholder_text=plbl)
                e.insert(0, pdef)
                e.grid(row=2, column=col, padx=1, pady=2, sticky="w")
                self.filter_entries[plbl] = e
                col += 1
        max_col = max(max_col, col)

        for c in range(max_col + 2):
            row2.grid_columnconfigure(c, weight=0)
        row2.grid_columnconfigure(max_col + 2, weight=1)

        ctk.CTkButton(row2, text="Run Filter",
                      command=self._filter_threaded, fg_color="#0F52BA", hover_color="#2A6BD1").grid(
            row=3, column=0, padx=5, pady=(6, 3), sticky="w")

        ctk.CTkCheckBox(
            row2, text="Export good filtered images",
            variable=self.export_filtered_good_var
        ).grid(row=3, column=1, padx=5, pady=(6, 3), sticky="w")

        ctk.CTkLabel(
            row2, text="-> Flags blurry, overexposed, dark,\n"
                       "   foggy, rain-on-lens & blocked images",
            font=("Arial", 10), text_color="gray", justify="left",
        ).grid(row=3, column=2, columnspan=max_col + 1, padx=5, pady=(6, 3),
               sticky="w")

        # Row 3 — harmonise BRIGHTNESS (collapsible)
        self.brightness_section_frame = ctk.CTkFrame(self.bottom_panel)
        row3 = self.brightness_section_frame

        ctk.CTkLabel(row3, text="Harmonise Brightness",
                     font=("Arial", 12, "bold")).grid(
            row=0, column=0, padx=5, pady=3, sticky="w")

        ctk.CTkLabel(row3, text="L-tolerance (+/-)").grid(
            row=0, column=1, padx=3, pady=3)
        self.tol_entry = ctk.CTkEntry(row3, width=55)
        self.tol_entry.insert(0, "5")
        self.tol_entry.grid(row=0, column=2, padx=3, pady=3)

        ctk.CTkLabel(row3, text="Sky mask (%)").grid(
            row=0, column=3, padx=3, pady=3)
        self.sky_entry = ctk.CTkEntry(row3, width=55)
        self.sky_entry.insert(0, "0")
        self.sky_entry.grid(row=0, column=4, padx=3, pady=3)

        self.exclude_bad_var = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(row3, text="Exclude filtered bad images",
                        variable=self.exclude_bad_var).grid(
            row=0, column=5, padx=10, pady=3)

        ctk.CTkButton(row3, text="Preview",
                      command=self._preview_brightness_threaded,
                      fg_color="#2E8B57", width=80).grid(
            row=0, column=6, padx=3, pady=3)

        ctk.CTkButton(row3, text="Run Brightness",
                      command=self._harmonise_brightness_threaded,
                      fg_color="#0F52BA", hover_color="#2A6BD1").grid(
            row=0, column=7, padx=5, pady=3)

        ctk.CTkLabel(
            row3, text="-> Matches luminance across images\n"
                       "   (L-channel only, colour unchanged)",
            font=("Arial", 10), text_color="gray", justify="left",
        ).grid(row=0, column=8, padx=5, pady=3, sticky="w")

        # Row 4 — harmonise COLOUR (collapsible)
        self.colour_section_frame = ctk.CTkFrame(self.bottom_panel)
        row4 = self.colour_section_frame

        ctk.CTkLabel(row4, text="Harmonise Colour",
                     font=("Arial", 12, "bold")).grid(
            row=0, column=0, padx=5, pady=3, sticky="w")

        ctk.CTkButton(row4, text="Select Reference Image",
                      command=self._browse_ref_colour, width=160).grid(
            row=0, column=1, padx=5, pady=3)
        self.ref_colour_label = ctk.CTkLabel(row4,
                                              text="No reference selected")
        self.ref_colour_label.grid(row=0, column=2, padx=5, pady=3,
                                    sticky="w")

        ctk.CTkLabel(row4, text="Algorithm:").grid(
            row=0, column=3, padx=3, pady=3)
        self.colour_algo_var = tk.StringVar(
            value="Iterative Transfer (best)")
        self.colour_algo_menu = ctk.CTkOptionMenu(
            row4, variable=self.colour_algo_var,
            values=list(COLOUR_ALGORITHMS.keys()), width=190)
        self.colour_algo_menu.grid(row=0, column=4, padx=3, pady=3)

        self.exclude_bad_colour_var = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(row4, text="Exclude bad",
                        variable=self.exclude_bad_colour_var).grid(
            row=0, column=5, padx=5, pady=3)

        ctk.CTkButton(row4, text="Preview",
                      command=self._preview_colour_threaded,
                      fg_color="#2E8B57", width=80).grid(
            row=0, column=6, padx=3, pady=3)

        ctk.CTkButton(row4, text="Run Colour",
                      command=self._harmonise_colour_threaded,
                      fg_color="#0F52BA", hover_color="#2A6BD1").grid(
            row=0, column=7, padx=5, pady=3)

        ctk.CTkLabel(
            row4, text="-> Transfers colour from reference\n"
                       "   to all images (full LAB match)",
            font=("Arial", 10), text_color="gray", justify="left",
        ).grid(row=0, column=8, padx=5, pady=3, sticky="w")

        # Row 4b — average images per folder (collapsible)
        self.average_section_frame = ctk.CTkFrame(self.bottom_panel)
        row4b = self.average_section_frame

        ctk.CTkLabel(row4b, text="Average Images Per Folder",
                     font=("Arial", 12, "bold")).grid(
            row=0, column=0, padx=5, pady=3, sticky="w")
        ctk.CTkLabel(
            row4b,
            text="-> Averages all readable images in each folder.\n"
                 "   With sub-folders enabled, each image-containing folder is averaged separately.",
            font=("Arial", 10), text_color="gray", justify="left",
        ).grid(row=0, column=1, padx=5, pady=3, sticky="w")
        ctk.CTkButton(row4b, text="Run Averaging",
                      command=self._average_images_threaded,
                      fg_color="#0F52BA", hover_color="#2A6BD1").grid(
            row=0, column=2, padx=10, pady=3)

        # Row 5 — output, progress, ETA, reset
        row5 = ctk.CTkFrame(self.bottom_panel)
        row5.pack(fill="x", padx=5, pady=2)

        ctk.CTkButton(row5, text="Browse Output Folder",
                      command=self._browse_output, fg_color="#8C7738", hover_color="#A18A45").grid(
            row=0, column=0, padx=5, pady=5)
        self.output_label = ctk.CTkLabel(row5,
                                          text="No output folder selected")
        self.output_label.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        self.progress_bar = ctk.CTkProgressBar(row5, width=200)
        self.progress_bar.grid(row=0, column=2, padx=10, pady=5)
        self.progress_bar.set(0)

        self.eta_label = ctk.CTkLabel(row5, text="ETA: --",
                                       font=("Arial", 10))
        self.eta_label.grid(row=0, column=3, padx=5, pady=5)

        self.btn_save_settings = ctk.CTkButton(
            row5, text="Save Settings",fg_color="#4F5D75",hover_color="#61708A", command=self._save_settings, width=110)
        self.btn_save_settings.grid(row=0, column=4, padx=5, pady=5)

        self.btn_load_settings = ctk.CTkButton(
            row5, text="Load Settings",fg_color="#4F5D75",hover_color="#61708A", command=self._load_settings, width=110)
        self.btn_load_settings.grid(row=0, column=5, padx=5, pady=5)

        self.btn_reset = ctk.CTkButton(
            row5, text="Reset", command=self._reset,
            width=80, fg_color="#8B0000", hover_color="#A52A2A")
        self.btn_reset.grid(row=0, column=6, padx=5, pady=5, sticky="e")

        # ---- PREVIEW NAVIGATION (hidden by default) ----
        self.preview_nav_frame = ctk.CTkFrame(self)
        # NOT gridded initially — shown only during preview

        self.btn_prev = ctk.CTkButton(
            self.preview_nav_frame, text="< Previous",
            command=self._preview_prev, width=100)
        self.btn_prev.pack(side="left", padx=10, pady=5)

        self.preview_counter_label = ctk.CTkLabel(
            self.preview_nav_frame, text="0 / 0",
            font=("Arial", 12, "bold"))
        self.preview_counter_label.pack(side="left", padx=15, pady=5)

        self.btn_next = ctk.CTkButton(
            self.preview_nav_frame, text="Next >",
            command=self._preview_next, width=100)
        self.btn_next.pack(side="left", padx=10, pady=5)

        self.btn_accept = ctk.CTkButton(
            self.preview_nav_frame, text="Accept & Process All",
            command=self._accept_preview,
            fg_color="#2E8B57", hover_color="#3CB371", width=180)
        self.btn_accept.pack(side="left", padx=20, pady=5)

        self.btn_cancel_preview = ctk.CTkButton(
            self.preview_nav_frame, text="Cancel Preview",
            command=self._cancel_preview,
            fg_color="#8B0000", hover_color="#A52A2A", width=130)
        self.btn_cancel_preview.pack(side="left", padx=10, pady=5)

        self._toggle_task_sections(initial=True)

        # ---- CONSOLE ----
        self.console_frame = ctk.CTkFrame(self)
        self.console_frame.grid(row=1, column=0, sticky="nsew",
                                padx=5, pady=5)
        self.console_text = tk.Text(self.console_frame, wrap="word",
                                     height=10)
        self.console_text.pack(fill="both", expand=True, padx=5, pady=5)
        self._console_redir = setup_console(self.console_text, "")
        print("Harmonise Images - Tool Guide\n"
              "================================\n"
              "\n"
              "DISPLAY PANEL (top)\n"
              "  Left plot:   Luminance distribution of all images.\n"
              "  Centre plot: Number of images flagged by each filter.\n"
              "  Right plot:  Harmonisation action breakdown.\n"
              "\n"
              "FILTER BAD IMAGES  (uncheck a filter to disable it)\n"
              "  Blur            - Laplacian variance; images below "
              "threshold are blurry.\n"
              "                    Guards against false positives on "
              "calm water / clear sky.\n"
              "  Overexposure    - Fraction of pixels saturated at "
              "254-255 in ALL channels.\n"
              "                    Detects blown highlights (not just "
              "bright sand).\n"
              "  Underexposure   - Two parameters: dark pixel value + "
              "fraction.\n"
              "                    If more than 'fraction' of pixels "
              "are below 'value' -> bad.\n"
              "  Darkness        - If the brightest pixel in the image "
              "is below threshold\n"
              "                    -> nighttime / lens cap on.\n"
              "  Low information - Histogram entropy; low values = "
              "fog, haze, condensation.\n"
              "                    Good outdoor images: 6-7.5; "
              "fog: below 5.\n"
              "  Lens droplets   - Detects rain or condensation "
              "droplets on the lens\n"
              "                    (bright blobs after Gaussian "
              "subtraction).\n"
              "  Obstruction     - Detects partial blockage (bird, "
              "lens cap, foreign object)\n"
              "                    by checking for large dark regions "
              "in a 4x4 grid.\n"
              "\n"
              "HARMONISE BRIGHTNESS\n"
              "  L-tolerance (+/-)  - Images within this luminance "
              "range of the median are\n"
              "                       kept unchanged. Outside:\n"
              "                         Very dark  -> histogram matching\n"
              "                         Very bright -> soft-gain\n"
              "                         Moderate   -> soft-gain\n"
              "                       Sanity check reverts if correction "
              "makes things worse.\n"
              "  Sky mask (%)       - Ignore the top N% of the image "
              "when computing mean\n"
              "                       brightness.\n"
              "\n"
              "HARMONISE COLOUR\n"
              "  Select a reference image that represents the desired\n"
              "  colour appearance.  Three algorithms available:\n"
              "    Reinhard (fast)           - mean+std transfer per "
              "LAB channel\n"
              "    Histogram Match (LAB)     - per-channel CDF matching "
              "in LAB\n"
              "    Iterative Transfer (best) - 3D distribution transfer\n"
              "                                (Pitie et al. 2007)\n"
              "\n"
              "PREVIEW\n"
              "  Both brightness and colour offer a 'Preview' button.\n"
              "  A random 5% sample (min 1) is processed and shown as\n"
              "  before/after pairs.  Use </> to navigate, then\n"
              "  'Accept & Process All' or 'Cancel Preview'.\n"
              "\n"
              "  Originals are NEVER modified - all output goes to\n"
              "  *_filtered_good/, *_brightness_harmonised/, or\n"
              "  *_colour_harmonised/ folders under the chosen output path.\n"
              "================================\n")

    # ——— UI/thread helpers ———

    def _ui_call(self, func, *args, **kwargs):
        try:
            self.after(0, lambda: func(*args, **kwargs))
        except Exception:
            pass

    def _ui_message(self, level, title, text):
        fn = getattr(messagebox, f"show{level}", None)
        if fn is not None:
            self._ui_call(fn, title, text)

    def _ui_set_progress_fraction(self, frac):
        self._ui_call(lambda: self.progress_bar.set(frac))

    def _ui_set_eta(self, text):
        self._ui_call(lambda: self.eta_label.configure(text=text))

    def _toggle_task_sections(self, initial=False):
        sections = [
            (self.run_filter_var.get(), self.filter_section_frame),
            (self.run_brightness_var.get(), self.brightness_section_frame),
            (self.run_colour_var.get(), self.colour_section_frame),
            (self.run_average_var.get(), self.average_section_frame),
        ]
        for enabled, frame in sections:
            if enabled:
                if not frame.winfo_manager():
                    frame.pack(fill="x", padx=5, pady=2)
            else:
                if frame.winfo_manager():
                    frame.pack_forget()

    def _check_folders_ui(self):
        if not self.input_folder:
            messagebox.showwarning("Warning", "Select an image folder first.")
            return False
        if not self.output_folder:
            messagebox.showwarning("Warning", "Select an output folder first.")
            return False
        return True

    def _collect_filter_config(self):
        if not self._check_folders_ui():
            return None
        return {
            "input_folder": self.input_folder,
            "output_folder": self.output_folder,
            "recursive": bool(self.recursive_var.get()),
            "blur_t": float(self.filter_entries["Blur threshold"].get()),
            "clip_f": float(self.filter_entries["Clip overexp fraction"].get()),
            "dark_v": int(self.filter_entries["Dark pixel value"].get()),
            "under_f": float(self.filter_entries["Under-exp fraction"].get()),
            "dark_max": int(self.filter_entries["Dark max threshold"].get()),
            "entropy_t": float(self.filter_entries["Entropy threshold"].get()),
            "droplet_f": float(self.filter_entries["Droplet fraction"].get()),
            "obstruct_f": float(self.filter_entries["Obstruction fraction"].get()),
            "enabled": {k: v.get() for k, v in self.filter_enabled.items()},
            "export_good": bool(self.export_filtered_good_var.get()),
        }

    def _collect_brightness_config(self):
        if not self._check_folders_ui():
            return None
        sky_pct = float(self.sky_entry.get())
        return {
            "input_folder": self.input_folder,
            "output_folder": self.output_folder,
            "recursive": bool(self.recursive_var.get()),
            "tol": float(self.tol_entry.get()),
            "sky_pct": sky_pct,
            "sky_frac": max(0.0, min(0.9, sky_pct / 100.0)),
            "exclude_bad": bool(self.exclude_bad_var.get()),
            "bad_list": list(self.bad_list),
        }

    def _collect_colour_config(self):
        if not self._check_folders_ui():
            return None
        if self.ref_colour_bgr is None:
            messagebox.showwarning("Warning", "Select a reference image for colour first.")
            return None
        return {
            "input_folder": self.input_folder,
            "output_folder": self.output_folder,
            "recursive": bool(self.recursive_var.get()),
            "exclude_bad": bool(self.exclude_bad_colour_var.get()),
            "bad_list": list(self.bad_list),
            "ref_colour_path": self.ref_colour_path,
            "ref_colour_bgr": self.ref_colour_bgr.copy(),
            "algo_name": self.colour_algo_var.get(),
        }

    def _collect_average_config(self):
        if not self._check_folders_ui():
            return None
        return {
            "input_folder": self.input_folder,
            "output_folder": self.output_folder,
            "recursive": bool(self.recursive_var.get()),
        }

    # ——— browse callbacks ———

    def _browse_input(self):
        d = filedialog.askdirectory(parent =self, title="Select Image Folder")
        if d:
            self.input_folder = d
            self.input_label.configure(text=d)

    def _browse_output(self):
        d = filedialog.askdirectory(parent =self, title="Select Output Folder")
        if d:
            self.output_folder = d
            self.output_label.configure(text=d)

    def _browse_ref_colour(self):
        """Let user pick one reference image for colour harmonisation."""
        init_dir = self.input_folder if self.input_folder else None
        f = filedialog.askopenfilename(parent =self, 
            title="Select Reference Image for Colour",
            initialdir=init_dir,
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                       ("All files", "*.*")])
        if f:
            img = cv2.imread(f)
            if img is None:
                messagebox.showerror("Error",
                                      f"Cannot read image:\n{f}")
                return
            self.ref_colour_path = f
            self.ref_colour_bgr = img
            name = os.path.basename(f)
            self.ref_colour_label.configure(
                text=f"{name}  ({img.shape[1]}x{img.shape[0]})")
            print(f"Colour reference image: {name}")

            # show thumbnail in axes[2]
            self._restore_normal_plots()
            self.axes[2].clear()
            self.axes[2].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            self.axes[2].set_title(f"Colour Reference: {name}",
                                    fontsize=9)
            self.axes[2].axis("off")

            self.fig.tight_layout()
            self.canvas_plot.draw()

    # ——— settings / cancel helpers ———

    def _request_cancel(self):
        if self._processing:
            self._cancel_requested = True
            self._ui_set_eta("Cancelling...")
            print("Cancellation requested...")

    def _poll_reset_after_cancel(self):
        if self._processing:
            self.after(150, self._poll_reset_after_cancel)
            return
        self._apply_reset_state()

    def _poll_close_after_cancel(self):
        if self._processing:
            self.after(150, self._poll_close_after_cancel)
            return
        restore_console(getattr(self, "_console_redir", None))
        self.destroy()

    def _settings_payload(self):
        return {
            "module": "harmonise_images",
            "settings_version": 1,
            "paths": {
                "input_folder": self.input_folder or "",
                "output_folder": self.output_folder or "",
                "ref_colour_path": self.ref_colour_path or "",
            },
            "ui_state": {
                "recursive": bool(self.recursive_var.get()),
                "run_filter": bool(self.run_filter_var.get()),
                "run_brightness": bool(self.run_brightness_var.get()),
                "run_colour": bool(self.run_colour_var.get()),
                "run_average": bool(self.run_average_var.get()),
                "exclude_bad_brightness": bool(self.exclude_bad_var.get()),
                "exclude_bad_colour": bool(self.exclude_bad_colour_var.get()),
                "export_filtered_good": bool(self.export_filtered_good_var.get()),
                "tol": self.tol_entry.get().strip(),
                "sky_pct": self.sky_entry.get().strip(),
                "colour_algorithm": self.colour_algo_var.get(),
                "filters_enabled": {k: bool(v.get()) for k, v in self.filter_enabled.items()},
                "filter_values": {k: e.get().strip() for k, e in self.filter_entries.items()},
            },
        }

    def _save_settings(self):
        try:
            initialdir = self.output_folder or self.input_folder or None
            path = save_settings_json(
                self,
                "harmonise_images",
                self._settings_payload(),
                initialdir=initialdir,
            )
            if path:
                print(f"Settings saved: {path}")
        except Exception as e:
            messagebox.showerror("Save Settings", f"Could not save settings:\n{e}", parent=self)

    def _load_settings(self):
        try:
            initialdir = self.output_folder or self.input_folder or None
            data, path = load_settings_json(self, "harmonise_images", initialdir=initialdir)
            if not data:
                return

            paths = data.get("paths", {})
            ui = data.get("ui_state", {})

            self.input_folder = paths.get("input_folder") or None
            self.output_folder = paths.get("output_folder") or None
            self.input_label.configure(text=self.input_folder or "No folder selected")
            self.output_label.configure(text=self.output_folder or "No output folder selected")

            self.recursive_var.set(bool(ui.get("recursive", False)))
            self.run_filter_var.set(bool(ui.get("run_filter", True)))
            self.run_brightness_var.set(bool(ui.get("run_brightness", True)))
            self.run_colour_var.set(bool(ui.get("run_colour", True)))
            self.run_average_var.set(bool(ui.get("run_average", False)))
            self.exclude_bad_var.set(bool(ui.get("exclude_bad_brightness", True)))
            self.exclude_bad_colour_var.set(bool(ui.get("exclude_bad_colour", True)))
            self.export_filtered_good_var.set(bool(ui.get("export_filtered_good", False)))

            self.tol_entry.delete(0, tk.END)
            self.tol_entry.insert(0, str(ui.get("tol", "5")))
            self.sky_entry.delete(0, tk.END)
            self.sky_entry.insert(0, str(ui.get("sky_pct", "0")))

            algo = ui.get("colour_algorithm", "Iterative Transfer (best)")
            if algo in COLOUR_ALGORITHMS:
                self.colour_algo_var.set(algo)

            saved_enabled = ui.get("filters_enabled", {})
            for k, var in self.filter_enabled.items():
                if k in saved_enabled:
                    var.set(bool(saved_enabled[k]))

            saved_values = ui.get("filter_values", {})
            for k, entry in self.filter_entries.items():
                if k in saved_values:
                    entry.delete(0, tk.END)
                    entry.insert(0, str(saved_values[k]))

            ref_path = paths.get("ref_colour_path") or ""
            self.ref_colour_path = ref_path or None
            self.ref_colour_bgr = None
            if self.ref_colour_path and os.path.isfile(self.ref_colour_path):
                img = cv2.imread(self.ref_colour_path)
                if img is not None:
                    self.ref_colour_bgr = img
                    self.ref_colour_label.configure(
                        text=f"{os.path.basename(self.ref_colour_path)}  ({img.shape[1]}x{img.shape[0]})"
                    )
                else:
                    self.ref_colour_label.configure(text="Saved reference missing/unreadable")
            else:
                self.ref_colour_label.configure(text="No reference selected")

            self._toggle_task_sections()
            print(f"Settings loaded: {path}")
        except Exception as e:
            messagebox.showerror("Load Settings", f"Could not load settings:\n{e}", parent=self)

    def _build_output_root(self, output_folder, input_root, suffix):
        return Path(output_folder) / f"{input_root.name}_{suffix}"

    def _build_preserved_output_path(self, source_path, input_root, output_folder, suffix):
        rel = Path(source_path).relative_to(input_root)
        out_root = self._build_output_root(output_folder, input_root, suffix)
        out_path = out_root / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        return out_path

    def _apply_reset_state(self):
        self._cancel_preview()
        self._cancel_requested = False
        self.input_folder = None
        self.output_folder = None
        self.bad_list = []
        self.bad_details = []
        self.ref_colour_path = None
        self.ref_colour_bgr = None
        self.input_label.configure(text="No folder selected")
        self.output_label.configure(text="No output folder selected")
        self.ref_colour_label.configure(text="No reference selected")
        self.progress_bar.set(0)
        self.eta_label.configure(text="ETA: --")
        self.recursive_var.set(False)
        self.run_filter_var.set(True)
        self.run_brightness_var.set(True)
        self.run_colour_var.set(True)
        self.run_average_var.set(False)
        self.exclude_bad_var.set(True)
        self.exclude_bad_colour_var.set(True)
        self.export_filtered_good_var.set(False)
        for var in self.filter_enabled.values():
            var.set(True)

        self._restore_normal_plots()
        self._toggle_task_sections()
        self.console_text.delete("1.0", tk.END)
        print("Session reset.\n--------------------------------")

    # ——— reset ———

    def _on_close(self):
        self._cancel_preview()
        if self._processing:
            self._request_cancel()
            self.after(150, self._poll_close_after_cancel)
            return
        restore_console(getattr(self, "_console_redir", None))
        self.destroy()

    def _reset(self):
        self._cancel_preview()
        if self._processing:
            self._request_cancel()
            self.after(150, self._poll_reset_after_cancel)
            return
        self._apply_reset_state()

    # ——— plot management ———

    def _restore_normal_plots(self):
        """Restore the 3-subplot layout."""
        self.fig.clear()
        self.axes = self.fig.subplots(1, 3)
        self.axes[0].set_title("Luminance Distribution")
        self.axes[0].axis("off")
        self.axes[1].set_title("Filter Results")
        self.axes[1].axis("off")
        self.axes[2].set_title("Harmonisation Results")
        self.axes[2].axis("off")
        self.fig.tight_layout()
        self.canvas_plot.draw()

    def _switch_to_preview_plots(self, n_cols):
        """Switch the figure to preview layout (2 or 3 columns)."""
        self.fig.clear()
        self.axes = self.fig.subplots(1, n_cols)
        if n_cols == 1:
            self.axes = [self.axes]
        self.fig.tight_layout()
        self.canvas_plot.draw()

    def _show_preview_nav(self):
        """Show the preview navigation bar."""
        self.preview_nav_frame.grid(row=2, column=0, sticky="ew",
                                     padx=5, pady=2)

    def _hide_preview_nav(self):
        """Hide the preview navigation bar."""
        self.preview_nav_frame.grid_forget()

    # ——— preview system ———

    def _update_preview_display(self):
        """Show current preview pair in the plot panel."""
        if not self.preview_pairs:
            return

        idx = self.preview_idx
        original, corrected, name = self.preview_pairs[idx]

        is_colour = (self.preview_mode == "colour")
        n_cols = 3 if (is_colour and self.ref_colour_bgr is not None) else 2

        self._switch_to_preview_plots(n_cols)

        col = 0
        if is_colour and self.ref_colour_bgr is not None:
            self.axes[col].imshow(
                cv2.cvtColor(self.ref_colour_bgr, cv2.COLOR_BGR2RGB))
            self.axes[col].set_title("Reference", fontsize=10)
            self.axes[col].axis("off")
            col += 1

        self.axes[col].imshow(
            cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        self.axes[col].set_title("Original", fontsize=10)
        self.axes[col].axis("off")

        self.axes[col + 1].imshow(
            cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
        self.axes[col + 1].set_title("Corrected", fontsize=10)
        self.axes[col + 1].axis("off")

        self.fig.suptitle(
            f"Preview {idx + 1}/{len(self.preview_pairs)}: {name}",
            fontsize=11, fontweight="bold")
        self.fig.tight_layout(rect=[0, 0, 1, 0.94])
        self.canvas_plot.draw()

        self.preview_counter_label.configure(
            text=f"{idx + 1} / {len(self.preview_pairs)}")

    def _preview_next(self):
        if self.preview_pairs:
            self.preview_idx = (
                (self.preview_idx + 1) % len(self.preview_pairs))
            self._update_preview_display()

    def _preview_prev(self):
        if self.preview_pairs:
            self.preview_idx = (
                (self.preview_idx - 1) % len(self.preview_pairs))
            self._update_preview_display()

    def _cancel_preview(self):
        """Exit preview mode, restore normal plots."""
        self.preview_mode = None
        self.preview_pairs = []
        self.preview_idx = 0
        self._hide_preview_nav()
        self._restore_normal_plots()

    def _accept_preview(self):
        """Accept preview and process all images."""
        mode = self.preview_mode
        self._cancel_preview()
        if mode == "brightness":
            self._harmonise_brightness_threaded()
        elif mode == "colour":
            self._harmonise_colour_threaded()

    # ——— gather images helper ———

    @staticmethod
    def _gather_images_from(folder, recursive=False, exclude_bad=False, bad_list=None):
        if recursive:
            images = collect_images_recursive(folder)
        else:
            images = collect_images(folder)
        if exclude_bad and bad_list:
            bad_set = set(map(str, bad_list))
            images = [p for p in images if str(p) not in bad_set]
        return images

    def _gather_images(self, exclude_bad=True):
        return self._gather_images_from(
            self.input_folder,
            recursive=bool(self.recursive_var.get()),
            exclude_bad=exclude_bad,
            bad_list=self.bad_list,
        )

    # ——— ETA helper ———

    def _update_progress(self, idx, total, start_time):
        frac = (idx + 1) / max(total, 1)
        elapsed = time.time() - start_time
        if idx > 0:
            per_image = elapsed / (idx + 1)
            remaining = per_image * (total - idx - 1)
            eta_text = f"ETA: ~{format_eta(remaining)}  ({idx + 1}/{total})"
        else:
            eta_text = f"ETA: estimating...  ({idx + 1}/{total})"
        self._ui_set_progress_fraction(frac)
        self._ui_set_eta(eta_text)

    def _cancelled(self):
        if self._cancel_requested:
            print("Processing cancelled.")
            self._ui_set_eta("Cancelled")
            return True
        return False

    def _render_filter_plot(self, reason_counts, n_bad, n_good):
        self._restore_normal_plots()
        self.axes[1].clear()
        labels = [k for k, v in reason_counts.items() if v > 0]
        values = [reason_counts[k] for k in labels]
        if not labels:
            labels = list(reason_counts.keys())
            values = [0] * len(labels)
        colors = ["#e74c3c", "#e67e22", "#3498db", "#2c3e50",
                  "#95a5a6", "#8e44ad", "#1abc9c"]
        self.axes[1].barh(labels, values, color=colors[:len(labels)])
        self.axes[1].set_title(f"Filter Results ({n_bad} bad / {n_good} good)")
        self.axes[1].set_xlabel("Count")
        self.fig.tight_layout()
        self.canvas_plot.draw()

    def _render_brightness_results(self, means_arr, valid, ref_mean, tol, counts):
        self._restore_normal_plots()
        self.axes[0].clear()
        self.axes[0].hist(means_arr[valid], bins=30, color="#3498db",
                          edgecolor="white", alpha=0.8)
        self.axes[0].axvline(ref_mean, color="red", linestyle="--",
                             label=f"Ref={ref_mean:.1f}")
        self.axes[0].axvspan(ref_mean - tol, ref_mean + tol,
                             alpha=0.15, color="green",
                             label=f"+/-{tol} tolerance")
        self.axes[0].set_title("Luminance Distribution")
        self.axes[0].set_xlabel("Mean L-channel")
        self.axes[0].set_ylabel("Count")
        self.axes[0].legend(fontsize="small")

        self.axes[2].clear()
        plot_labels = [k for k in counts if counts[k] > 0]
        plot_values = [counts[k] for k in plot_labels]
        bar_colors = {
            "copy": "#2ecc71", "histogram": "#9b59b6",
            "soft-gain": "#e67e22", "bright-gain": "#e74c3c",
            "reverted": "#7f8c8d",
        }
        self.axes[2].bar(plot_labels, plot_values,
                         color=[bar_colors.get(k, "#3498db") for k in plot_labels])
        self.axes[2].set_title("Brightness Results")
        self.axes[2].set_ylabel("Count")
        self.fig.tight_layout()
        self.canvas_plot.draw()

    def _render_colour_results(self, ref_bgr, counts, elapsed, algo_name):
        self._restore_normal_plots()
        self.axes[0].imshow(cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB))
        self.axes[0].set_title("Colour Reference", fontsize=9)
        self.axes[0].axis("off")

        self.axes[1].clear()
        plot_labels = [k for k in counts if counts[k] > 0]
        plot_values = [counts[k] for k in plot_labels]
        bar_colors = {"corrected": "#2ecc71", "reverted": "#7f8c8d"}
        self.axes[1].bar(plot_labels, plot_values,
                         color=[bar_colors.get(k, "#3498db") for k in plot_labels])
        self.axes[1].set_title("Colour Results")
        self.axes[1].set_ylabel("Count")

        self.axes[2].set_title(f"Algorithm: {algo_name}", fontsize=9)
        self.axes[2].text(
            0.5, 0.5,
            f"{counts['corrected']} corrected\n"
            f"{counts['reverted']} reverted\n"
            f"{format_eta(elapsed)} elapsed",
            ha="center", va="center", fontsize=12,
            transform=self.axes[2].transAxes)
        self.axes[2].axis("off")
        self.fig.tight_layout()
        self.canvas_plot.draw()

    def _render_average_results(self, folder_count, written_count, skipped_count):
        self._restore_normal_plots()
        self.axes[2].clear()
        labels = ["Folders found", "Averages written", "Folders skipped"]
        values = [folder_count, written_count, skipped_count]
        self.axes[2].bar(labels, values)
        self.axes[2].set_title("Averaging Results")
        self.axes[2].set_ylabel("Count")
        self.fig.tight_layout()
        self.canvas_plot.draw()

    # ——— filtering ———

    def _filter_threaded(self):
        if self._processing:
            return
        try:
            cfg = self._collect_filter_config()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return
        if cfg is None:
            return
        threading.Thread(target=self._run_filter, args=(cfg,), daemon=True).start()

    def _run_filter(self, cfg):
        try:
            self._processing = True
            self._cancel_requested = False
            self._ui_set_progress_fraction(0)
            self._ui_set_eta("ETA: --")

            images = self._gather_images_from(cfg["input_folder"], recursive=cfg["recursive"])
            if not images:
                self._ui_message("warning", "Warning", "No images found.")
                return

            en = cfg["enabled"]
            active = [k for k, v in en.items() if v]
            print(f"\nFiltering {len(images)} images with {len(active)} active filter(s): {', '.join(active)} ...")
            bad_list = []
            bad_reasons = {}
            reason_counts = {
                "blurry": 0, "clipped_overexposed": 0,
                "underexposed": 0, "too_dark": 0,
                "low_information": 0, "lens_droplets": 0,
                "obstruction": 0,
            }
            start_time = time.time()

            for idx, p in enumerate(images):
                if self._cancel_requested:
                    self._ui_set_eta("Cancelled")
                    print("Filtering cancelled.")
                    return
                img = cv2.imread(str(p))
                if img is None:
                    print(f"[WARN] Cannot read: {p.name}")
                    continue
                reasons = []
                if en.get("Blur") and is_blurry(img, cfg["blur_t"]):
                    reasons.append("blurry")
                if en.get("Overexposure") and is_clipped_overexposed(img, cfg["clip_f"]):
                    reasons.append("clipped_overexposed")
                if en.get("Underexposure") and is_underexposed(img, cfg["dark_v"], cfg["under_f"]):
                    reasons.append("underexposed")
                if en.get("Darkness") and is_too_dark(img, cfg["dark_max"]):
                    reasons.append("too_dark")
                if en.get("Low information") and is_low_information(img, cfg["entropy_t"]):
                    reasons.append("low_information")
                if en.get("Lens droplets") and has_lens_droplets(img, cfg["droplet_f"]):
                    reasons.append("lens_droplets")
                if en.get("Obstruction") and has_obstruction(img, block_fraction=cfg["obstruct_f"]):
                    reasons.append("obstruction")

                if reasons:
                    bad_list.append(str(p))
                    bad_reasons[p.name] = reasons
                    for r in reasons:
                        reason_counts[r] = reason_counts.get(r, 0) + 1
                    print(f"  x {p.name}: {', '.join(reasons)}")

                self._update_progress(idx, len(images), start_time)

            self.bad_list = bad_list
            details = []
            input_root = Path(cfg["input_folder"])
            for bp in bad_list:
                bp_path = Path(bp)
                details.append({
                    "absolute_path": str(bp_path),
                    "relative_path": str(bp_path.relative_to(input_root)),
                    "filename": bp_path.name,
                    "reasons": bad_reasons.get(bp_path.name, []),
                })
            self.bad_details = details

            n_bad = len(bad_list)
            n_good = len(images) - n_bad
            print(f"\nFilter complete: {n_bad} bad / {n_good} good out of {len(images)} images.")

            txt_path = os.path.join(cfg["output_folder"], "bad_images.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("relative_path\tfilename\treason\n")
                for item in details:
                    f.write(f"{item['relative_path']}\t{item['filename']}\t{','.join(item['reasons'])}\n")
            print(f"Bad image list saved: {txt_path}")

            json_path = os.path.join(cfg["output_folder"], "bad_images.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({
                    "input_folder": cfg["input_folder"],
                    "recursive": bool(cfg["recursive"]),
                    "bad_images": details,
                }, f, indent=2, ensure_ascii=False)
            print(f"Bad image JSON saved: {json_path}")

            if cfg.get("export_good"):
                out_root = self._build_output_root(cfg["output_folder"], input_root, "filtered_good")
                copied = 0
                bad_set = set(bad_list)
                for p in images:
                    if self._cancel_requested:
                        self._ui_set_eta("Cancelled")
                        print("Filtered-good export cancelled.")
                        return
                    if str(p) in bad_set:
                        continue
                    out_path = self._build_preserved_output_path(p, input_root, cfg["output_folder"], "filtered_good")
                    shutil.copy2(str(p), str(out_path))
                    copied += 1
                print(f"Good filtered images exported: {copied} -> {out_root}")

            self._ui_call(self._render_filter_plot, reason_counts, n_bad, n_good)
            self._ui_set_eta("Done")

        except Exception as e:
            print(f"[ERROR] {e}")
            self._ui_message("error", "Error", str(e))
        finally:
            self._processing = False

    # ——— brightness harmonisation ———

    def _preview_brightness_threaded(self):
        if self._processing:
            return
        try:
            cfg = self._collect_brightness_config()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return
        if cfg is None:
            return
        threading.Thread(target=self._run_preview_brightness, args=(cfg,), daemon=True).start()

    def _run_preview_brightness(self, cfg):
        try:
            self._processing = True
            self._cancel_requested = False
            self._ui_set_progress_fraction(0)

            images = self._gather_images_from(cfg["input_folder"], recursive=cfg["recursive"],
                                              exclude_bad=cfg["exclude_bad"], bad_list=cfg["bad_list"])
            if not images:
                self._ui_message("warning", "Warning", "No images remaining.")
                return

            n_preview = preview_sample_count(len(images))
            sample_indices = sorted(random.sample(range(len(images)), min(n_preview, len(images))))
            print(f"\nBrightness preview: processing {len(sample_indices)} of {len(images)} images...")
            print("  Computing luminance statistics...")

            means = []
            for p in images:
                if self._cancel_requested:
                    self._ui_set_eta("Cancelled")
                    print("Brightness preview cancelled.")
                    return
                img = cv2.imread(str(p))
                if img is None:
                    means.append(np.nan)
                else:
                    means.append(mean_luminance(img, cfg["sky_frac"]))
            means_arr = np.array(means)
            ref_mean = float(np.nanmedian(means_arr))
            ref_idx = int(np.nanargmin(np.abs(means_arr - ref_mean)))
            ref_img = cv2.imread(str(images[ref_idx]))
            if ref_img is None:
                print(f"  ERROR: could not load reference image {images[ref_idx].name}")
                return
            print(f"  Reference luminance: {ref_mean:.1f} (from {images[ref_idx].name})")

            preview_pairs = []
            start_time = time.time()
            for i, si in enumerate(sample_indices):
                if self._cancel_requested:
                    self._ui_set_eta("Cancelled")
                    print("Preview cancelled.")
                    return
                p = images[si]
                img = cv2.imread(str(p))
                if img is None:
                    continue
                m = means_arr[si]
                delta = m - ref_mean
                if abs(delta) <= cfg["tol"]:
                    corrected = img.copy()
                elif delta <= -2 * cfg["tol"]:
                    corrected = hist_match(img, ref_img)
                else:
                    corrected = soft_gain_match(img, ref_mean)
                if abs(delta) > cfg["tol"]:
                    corrected, reverted = correction_sanity_check(img, corrected, ref_mean, cfg["sky_frac"])
                    if reverted:
                        print(f"  [REVERT] {p.name}: correction worsened luminance")
                preview_pairs.append((img, corrected, p.name))
                self._update_progress(i, len(sample_indices), start_time)

            self.preview_pairs = preview_pairs
            self.preview_mode = "brightness"
            self.preview_idx = 0
            self._ui_call(self._show_preview_nav)
            self._ui_call(self._update_preview_display)
            self._ui_set_eta("Preview ready")
            print(f"  Preview ready: {len(self.preview_pairs)} images. Use </> to navigate.")

        except Exception as e:
            print(f"[ERROR] {e}")
            self._ui_message("error", "Error", str(e))
        finally:
            self._processing = False

    def _harmonise_brightness_threaded(self):
        if self._processing:
            return
        try:
            cfg = self._collect_brightness_config()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return
        if cfg is None:
            return
        threading.Thread(target=self._run_harmonise_brightness, args=(cfg,), daemon=True).start()

    def _run_harmonise_brightness(self, cfg):
        try:
            self._processing = True
            self._cancel_requested = False
            self._ui_set_progress_fraction(0)
            self._ui_set_eta("ETA: --")

            images = self._gather_images_from(cfg["input_folder"], recursive=cfg["recursive"],
                                              exclude_bad=cfg["exclude_bad"], bad_list=cfg["bad_list"])
            if not images:
                self._ui_message("warning", "Warning", "No images remaining after filtering.")
                return

            sky_msg = (f", sky mask top {cfg['sky_pct']:.0f}%" if cfg["sky_frac"] > 0 else "")
            print(f"\nHarmonising brightness for {len(images)} images (tolerance +/-{cfg['tol']} L-units{sky_msg})...")

            means = []
            loaded = []
            for p in images:
                if self._cancel_requested:
                    self._ui_set_eta("Cancelled")
                    print("Brightness harmonisation cancelled.")
                    return
                img = cv2.imread(str(p))
                if img is None:
                    means.append(np.nan)
                    loaded.append(None)
                    continue
                loaded.append(img)
                means.append(mean_luminance(img, cfg["sky_frac"]))

            means_arr = np.array(means)
            valid = ~np.isnan(means_arr)
            ref_mean = float(np.nanmedian(means_arr))
            ref_idx = int(np.nanargmin(np.abs(means_arr - ref_mean)))
            ref_img = loaded[ref_idx]
            if ref_img is None:
                print(f"ERROR: reference image {images[ref_idx].name} could not be loaded.")
                return
            print(f"Reference luminance (median): {ref_mean:.1f}  (image: {images[ref_idx].name})")

            input_root = Path(cfg["input_folder"])
            counts = {"copy": 0, "histogram": 0, "soft-gain": 0, "bright-gain": 0, "reverted": 0}
            start_time = time.time()

            for idx, (p, img, m) in enumerate(zip(images, loaded, means)):
                if self._cancel_requested:
                    self._ui_set_eta("Cancelled")
                    print("Brightness harmonisation cancelled.")
                    return
                if img is None:
                    continue
                delta = m - ref_mean
                if abs(delta) <= cfg["tol"]:
                    mode, corrected = "copy", img
                elif delta <= -2 * cfg["tol"]:
                    mode = "histogram"
                    corrected = hist_match(img, ref_img)
                elif delta >= 2 * cfg["tol"]:
                    mode = "bright-gain"
                    corrected = soft_gain_match(img, ref_mean)
                else:
                    mode = "soft-gain"
                    corrected = soft_gain_match(img, ref_mean)
                if mode != "copy":
                    corrected, reverted = correction_sanity_check(img, corrected, ref_mean, cfg["sky_frac"])
                    if reverted:
                        counts["reverted"] += 1
                        print(f"  [REVERT] {p.name}: correction made luminance worse - kept original")
                counts[mode] = counts.get(mode, 0) + 1

                out_path = self._build_preserved_output_path(p, input_root, cfg["output_folder"], "brightness_harmonised")
                cv2.imwrite(str(out_path), corrected)

                if (idx + 1) % 20 == 0 or idx == 0:
                    print(f"  {idx + 1}/{len(images)} | {p.name} | dL={delta:+.1f} -> {mode}")
                self._update_progress(idx, len(images), start_time)

            elapsed = time.time() - start_time
            print(f"\nBrightness harmonisation complete ({format_eta(elapsed)} elapsed):")
            for k, v in counts.items():
                if v > 0:
                    print(f"  {k}: {v}")

            self._ui_call(self._render_brightness_results, means_arr, valid, ref_mean, cfg["tol"], counts)
            self._ui_set_eta("Done")
            self._ui_message("info", "Done", f"Brightness-harmonised images saved to:\n{cfg['output_folder']}")

        except Exception as e:
            print(f"[ERROR] {e}")
            self._ui_message("error", "Error", str(e))
        finally:
            self._processing = False

    # ——— colour harmonisation ———

    def _preview_colour_threaded(self):
        if self._processing:
            return
        cfg = self._collect_colour_config()
        if cfg is None:
            return
        threading.Thread(target=self._run_preview_colour, args=(cfg,), daemon=True).start()

    def _run_preview_colour(self, cfg):
        try:
            self._processing = True
            self._cancel_requested = False
            self._ui_set_progress_fraction(0)

            images = self._gather_images_from(cfg["input_folder"], recursive=cfg["recursive"],
                                              exclude_bad=cfg["exclude_bad"], bad_list=cfg["bad_list"])
            ref_abs = os.path.abspath(cfg["ref_colour_path"])
            images = [p for p in images if os.path.abspath(str(p)) != ref_abs]
            if not images:
                self._ui_message("warning", "Warning", "No images remaining.")
                return

            algo_name = cfg["algo_name"]
            algo_fn = COLOUR_ALGORITHMS[algo_name]
            n_preview = preview_sample_count(len(images))
            sample_indices = sorted(random.sample(range(len(images)), min(n_preview, len(images))))
            print(f"\nColour preview ({algo_name}): processing {len(sample_indices)} of {len(images)} images...")

            preview_pairs = []
            start_time = time.time()
            for i, si in enumerate(sample_indices):
                if self._cancel_requested:
                    self._ui_set_eta("Cancelled")
                    print("Preview cancelled.")
                    return
                p = images[si]
                img = cv2.imread(str(p))
                if img is None:
                    continue
                corrected = algo_fn(img, cfg["ref_colour_bgr"])
                corrected, reverted = colour_sanity_check(img, corrected, cfg["ref_colour_bgr"])
                if reverted:
                    print(f"  [REVERT] {p.name}: colour correction worsened - kept original")
                preview_pairs.append((img, corrected, p.name))
                self._update_progress(i, len(sample_indices), start_time)

            self.preview_pairs = preview_pairs
            self.preview_mode = "colour"
            self.preview_idx = 0
            self._ui_call(self._show_preview_nav)
            self._ui_call(self._update_preview_display)
            self._ui_set_eta("Preview ready")
            print(f"  Preview ready: {len(self.preview_pairs)} images. Use </> to navigate.")

        except Exception as e:
            print(f"[ERROR] {e}")
            self._ui_message("error", "Error", str(e))
        finally:
            self._processing = False

    def _harmonise_colour_threaded(self):
        if self._processing:
            return
        cfg = self._collect_colour_config()
        if cfg is None:
            return
        threading.Thread(target=self._run_harmonise_colour, args=(cfg,), daemon=True).start()

    def _run_harmonise_colour(self, cfg):
        try:
            self._processing = True
            self._cancel_requested = False
            self._ui_set_progress_fraction(0)
            self._ui_set_eta("ETA: --")

            images = self._gather_images_from(cfg["input_folder"], recursive=cfg["recursive"],
                                              exclude_bad=cfg["exclude_bad"], bad_list=cfg["bad_list"])
            ref_abs = os.path.abspath(cfg["ref_colour_path"])
            images = [p for p in images if os.path.abspath(str(p)) != ref_abs]
            if not images:
                self._ui_message("warning", "Warning", "No images remaining.")
                return

            algo_name = cfg["algo_name"]
            algo_fn = COLOUR_ALGORITHMS[algo_name]
            print(f"\nColour harmonisation ({algo_name}): {len(images)} images...")
            print(f"  Reference: {os.path.basename(cfg['ref_colour_path'])}")

            input_root = Path(cfg["input_folder"])
            counts = {"corrected": 0, "reverted": 0}
            start_time = time.time()

            for idx, p in enumerate(images):
                if self._cancel_requested:
                    self._ui_set_eta("Cancelled")
                    print("Colour harmonisation cancelled.")
                    return
                img = cv2.imread(str(p))
                if img is None:
                    print(f"[WARN] Cannot read: {p.name}")
                    continue
                corrected = algo_fn(img, cfg["ref_colour_bgr"])
                corrected, reverted = colour_sanity_check(img, corrected, cfg["ref_colour_bgr"])
                if reverted:
                    counts["reverted"] += 1
                    print(f"  [REVERT] {p.name}: colour correction worsened - kept original")
                else:
                    counts["corrected"] += 1

                out_path = self._build_preserved_output_path(p, input_root, cfg["output_folder"], "colour_harmonised")
                cv2.imwrite(str(out_path), corrected)

                if (idx + 1) % 10 == 0 or idx == 0:
                    print(f"  {idx + 1}/{len(images)} | {p.name}")
                self._update_progress(idx, len(images), start_time)

            elapsed = time.time() - start_time
            print(f"\nColour harmonisation complete ({format_eta(elapsed)} elapsed):")
            print(f"  corrected: {counts['corrected']}")
            if counts["reverted"] > 0:
                print(f"  reverted:  {counts['reverted']}")

            self._ui_call(self._render_colour_results, cfg["ref_colour_bgr"], counts, elapsed, algo_name)
            self._ui_set_eta("Done")
            self._ui_message("info", "Done", f"Colour-harmonised images saved to:\n{cfg['output_folder']}")

        except Exception as e:
            print(f"[ERROR] {e}")
            self._ui_message("error", "Error", str(e))
        finally:
            self._processing = False

    # ——— averaging ———

    def _average_images_threaded(self):
        if self._processing:
            return
        cfg = self._collect_average_config()
        if cfg is None:
            return
        threading.Thread(target=self._run_average_images, args=(cfg,), daemon=True).start()

    def _run_average_images(self, cfg):
        try:
            self._processing = True
            self._cancel_requested = False
            self._ui_set_progress_fraction(0)
            self._ui_set_eta("ETA: --")

            input_root = Path(cfg["input_folder"])
            jobs = []
            if cfg["recursive"]:
                for root, _, files in os.walk(cfg["input_folder"]):
                    imgs = [Path(root) / f for f in sorted(files) if Path(f).suffix.lower() in IMAGE_EXTS]
                    if imgs:
                        jobs.append((Path(root), imgs))
            else:
                imgs = collect_images(cfg["input_folder"])
                if imgs:
                    jobs.append((input_root, imgs))

            if not jobs:
                self._ui_message("warning", "Warning", "No images found.")
                return

            written_count = 0
            skipped_count = 0
            start_time = time.time()
            print(f"\nAveraging images for {len(jobs)} folder(s)...")

            for idx, (folder_path, imgs) in enumerate(jobs):
                if self._cancel_requested:
                    self._ui_set_eta("Cancelled")
                    print("Averaging cancelled.")
                    return
                accum = None
                ref_shape = None
                valid_count = 0
                skipped_local = 0
                for p in imgs:
                    if self._cancel_requested:
                        self._ui_set_eta("Cancelled")
                        print("Averaging cancelled.")
                        return
                    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                    if img is None:
                        skipped_local += 1
                        continue
                    if ref_shape is None:
                        ref_shape = img.shape
                        accum = img.astype(np.float64)
                        valid_count = 1
                    elif img.shape == ref_shape:
                        accum += img.astype(np.float64)
                        valid_count += 1
                    else:
                        skipped_local += 1
                        print(f"  [SKIP] {p.name}: size mismatch in {folder_path.name}")

                if valid_count == 0 or accum is None:
                    skipped_count += 1
                    print(f"  [SKIP] {folder_path}: no valid readable images")
                    self._update_progress(idx, len(jobs), start_time)
                    continue

                avg = np.clip(accum / valid_count, 0, 255).astype(np.uint8)
                if cfg["recursive"]:
                    rel = folder_path.relative_to(input_root)
                    out_dir = Path(cfg["output_folder"]) / rel
                else:
                    out_dir = Path(cfg["output_folder"]) / input_root.name
                out_dir.mkdir(parents=True, exist_ok=True)
                out_name = f"{folder_path.name}_average.png"
                out_path = out_dir / out_name
                cv2.imwrite(str(out_path), avg)
                written_count += 1
                print(f"  Wrote average for {folder_path}: {out_path.name} ({valid_count} images, {skipped_local} skipped)")
                self._update_progress(idx, len(jobs), start_time)

            elapsed = time.time() - start_time
            print(f"\nAveraging complete ({format_eta(elapsed)} elapsed): {written_count} written, {skipped_count} skipped.")
            self._ui_call(self._render_average_results, len(jobs), written_count, skipped_count)
            self._ui_set_eta("Done")
            self._ui_message("info", "Done", f"Folder averages saved to:\n{cfg['output_folder']}")

        except Exception as e:
            print(f"[ERROR] {e}")
            self._ui_message("error", "Error", str(e))
        finally:
            self._processing = False


# ——— standalone ———
if __name__ == "__main__":
    root = ctk.CTk()
    root.withdraw()
    HarmoniseImagesWindow(master=root)
    root.mainloop()