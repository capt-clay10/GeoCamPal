"""
lens_correction.py  —  GeoCamPal Intrinsic Lens Calibration
============================================================

Purpose
-------
Estimates camera intrinsic parameters (camera matrix K and distortion
coefficients D) from a set of checkerboard calibration images using
OpenCV's camera calibration pipeline.  The result is saved as a Python
pickle file that is consumed by the Georeferencing module (Camera
Projection method) and the Harmonise Images lens-correction sub-task.

Method
------
For each image in the input folder, OpenCV's findChessboardCorners
locates the inner corner grid of the checkerboard.  Corners are refined
to sub-pixel accuracy with cornerSubPix.  The full set of detected
corner configurations is then passed to calibrateCamera, which solves
for K (3×3 camera matrix: fx, fy, cx, cy) and D (distortion vector)
by minimising the total reprojection error across all images.

The complexity of the distortion model is user-selectable (GUI
dropdown).  The default — k1 + k2, with k3 and the tangential terms
fixed to zero — suits most conventional lenses.  Freeing all five
coefficients ("Full 5-term (legacy)") reproduces the historic
behaviour, but can overfit badly when the reprojection residuals are
dominated by target defects (e.g. a non-flat board) rather than real
lens distortion, producing a non-physical, wavy correction that makes
images WORSE when applied.

After solving, three quality diagnostics run automatically and a
GOOD / MARGINAL / POOR verdict is issued:
  • true per-image RMS errors, consistent with the global RMS
    (‖residuals‖₂ / √N — earlier releases divided by N, understating
    the per-image error by √N, ≈10× for a typical board);
  • a target-planarity check: per-image homography fit residual,
    also expressed in approximate millimetres on the board — a rigid
    plane must fit a homography, so large residuals expose warped or
    mis-assembled targets;
  • frame coverage of all detected corners — distortion outside the
    covered region is extrapolation and cannot be trusted.
Parameter standard deviations (calibrateCameraExtended) are reported
so overfitted, non-significant coefficients are visible at a glance.
An evidence-based "Model guidance" line then recommends whether to
keep, simplify, or (for a near-zero result) skip correction — based on
this lens's own coefficient significance and the RMS cost of dropping
the highest term, not on a focal-length rule of thumb.

A minimum of 3 images with successfully detected corners is required
for a reliable calibration.  More images with diverse checkerboard
poses (different angles, distances, positions in the frame) give a
more robust result.

Checkerboard input convention
------------------------------
The user enters the number of SQUARES (not inner corners) in each
direction.  The tool subtracts 1 internally before passing to OpenCV,
which expects the number of inner corner intersections.

    Example: a board with 10 columns × 7 rows of squares
             → 9 × 6 inner corners passed to findChessboardCorners

Rectangular cells are supported: cell width and height can differ
(e.g. 30 mm × 25 mm).  Object point coordinates are scaled accordingly
before calibration.  Square cells are the common case; set both
dimensions to the same value.

Outputs  (saved to the user-selected output folder)
-------
    lens_calibration.pkl     — Python pickle with the following keys:
                                 camera_matrix   — 3×3 float64 ndarray
                                 dist_coeff      — distortion vector
                                 rms_error       — RMS reprojection error (px)
                                 image_size      — (W, H) in pixels
                                 n_images_used   — images with corners found
                                 pattern_size    — (n_cols, n_rows) inner corners
                                 board_squares   — (sq_cols, sq_rows) as entered
                                 cell_width_m    — cell width in metres
                                 cell_height_m   — cell height in metres
                                 square_size_m   — alias for cell_width_m
                                 dist_model      — distortion model label
                                 calib_flags     — cv2.calibrateCamera flags used
                                 per_image_rms   — true per-image RMS errors (px)
                                 quality         — "GOOD" / "MARGINAL" / "POOR"
                                 quality_warnings— diagnostic warnings (list)

    calibration_report.txt   — human-readable summary: board dimensions,
                                 distortion model, RMS error and quality
                                 verdict with warnings, K matrix, focal
                                 length, principal point, distortion
                                 coefficients (each ± 1σ), frame coverage,
                                 and per-image RMS + planarity diagnostics

Preview
-------
After calibration the top panel displays two plots: the first image
where corners were successfully detected (with the detected corner
overlay drawn by drawChessboardCorners) alongside the same image after
undistortion.  The undistorted preview uses alpha = 0 ("crop to valid
pixels") to match the Harmonise Images lens-correction default, and
overlays a straight reference grid so correction quality can be judged
against genuinely straight lines.

Dependencies
------------
    numpy, opencv-python (cv2), matplotlib, customtkinter, Pillow
"""

# %% ————————————————————————————— imports —————————————————————————————
import sys
import os
import glob
import pickle
import threading
import time

import numpy as np
import cv2
from PIL import Image, ImageTk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox

from utils import (
    fit_geometry, resource_path, setup_console, restore_console,
    save_settings_json, load_settings_json, compute_eta, format_eta,
    imread_safe, __version__, save_lens_calibration,
)

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")


# %% ——————————— distortion models & quality diagnostics ———————————
#
# Distortion-model presets offered in the GUI.  Values are flag
# combinations for cv2.calibrateCamera; fixed coefficients stay 0.
DIST_MODEL_OPTIONS = {
    "k1 only (simplest)":
        cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST,
    "k1 + k2 (recommended)":
        cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST,
    "k1 + k2 + tangential":
        cv2.CALIB_FIX_K3,
    "Full 5-term (legacy)":
        0,
}
DIST_MODEL_DEFAULT = "k1 + k2 (recommended)"

# Simplicity order (simplest → richest).  Used only to compute the RMS of
# the next-simpler model for model-guidance messaging.
DIST_MODEL_ORDER = [
    "k1 only (simplest)",
    "k1 + k2 (recommended)",
    "k1 + k2 + tangential",
    "Full 5-term (legacy)",
]


def _simpler_flags(flags):
    """Return the flags of the next-simpler model, or None if already
    simplest.  Matches on the flag value used in DIST_MODEL_OPTIONS."""
    for label, f in DIST_MODEL_OPTIONS.items():
        if f == flags:
            i = DIST_MODEL_ORDER.index(label)
            if i > 0:
                return DIST_MODEL_OPTIONS[DIST_MODEL_ORDER[i - 1]]
            return None
    return None

# Verdict thresholds on the global RMS reprojection error (px).
RMS_GOOD = 0.5        # below: calibration usable as-is
RMS_MARGINAL = 1.0    # below: usable with caution; above: do not use

# Diagnostic warning thresholds.
PLANARITY_WARN_MM = 1.0         # target deviates from a plane by more
COVERAGE_WARN_FRACTION = 0.60   # corners cover less than this frame share


def compute_per_image_rms(obj_points, img_points, rvecs, tvecs,
                          camera_matrix, dist_coeffs):
    """True per-image RMS reprojection error in pixels.

    Uses ‖residuals‖₂ / √N, which is consistent with the global RMS
    returned by cv2.calibrateCamera.  (Dividing by N instead — as the
    popular OpenCV tutorial snippet does — understates the error by a
    factor of √N: ≈10.8× for a 13 × 9 corner board.)
    """
    errors = []
    for i in range(len(obj_points)):
        proj, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i],
                                    camera_matrix, dist_coeffs)
        diff = (np.asarray(img_points[i], dtype=np.float64).reshape(-1, 2)
                - proj.reshape(-1, 2))
        errors.append(float(np.sqrt(np.mean(np.sum(diff * diff, axis=1)))))
    return errors


def planarity_check(objp, img_points, pattern_size, cell_w_m):
    """Per-image deviation of the detected corner grid from a flat plane.

    A rigid planar board must fit a single homography up to lens
    distortion (small for conventional lenses), so a large residual
    exposes a non-flat or mis-assembled target — the dominant failure
    mode for paper checkerboards.  Returns [(rms_px, approx_mm), …];
    the millimetre estimate converts via the mean corner pitch.
    """
    n_cols, n_rows = pattern_size
    src = objp[:, :2].astype(np.float64).reshape(-1, 1, 2)
    results = []
    for pts in img_points:
        dst = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        H, _ = cv2.findHomography(src.reshape(-1, 2), dst, 0)
        if H is None:
            results.append((float("nan"), float("nan")))
            continue
        proj = cv2.perspectiveTransform(src, H).reshape(-1, 2)
        diff = dst - proj
        rms_px = float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))
        grid = dst.reshape(n_rows, n_cols, 2)
        pitch_px = float(np.mean(np.linalg.norm(np.diff(grid, axis=1),
                                                axis=2)))
        mm_per_px = ((cell_w_m * 1000.0) / pitch_px if pitch_px > 0
                     else float("nan"))
        results.append((rms_px, rms_px * mm_per_px))
    return results


def coverage_fraction(img_points, img_shape):
    """Fraction of the frame area covered by the convex hull of all
    detected corners, 0..1."""
    W, H = img_shape
    pts = np.vstack([np.asarray(p).reshape(-1, 2)
                     for p in img_points]).astype(np.float32)
    hull = cv2.convexHull(pts)
    return float(cv2.contourArea(hull)) / float(W * H)


def quality_verdict(rms):
    """GOOD / MARGINAL / POOR verdict from the global RMS (px)."""
    if rms < RMS_GOOD:
        return "GOOD"
    if rms < RMS_MARGINAL:
        return "MARGINAL"
    return "POOR"


def model_guidance(flags, dvals, std, rms, rms_simpler):
    """Recommend a distortion model based on this lens's actual fit.

    Evidence-based, not a focal-length rule of thumb: it looks at whether
    the highest freed coefficient is statistically significant (|value| vs
    2σ) and how much the global RMS would change if that coefficient were
    dropped (``rms_simpler`` — the RMS of the next-simpler model, or None
    if there is no simpler model / it could not be computed).

    Returns a single plain-language sentence, or None if no clear call.
    """
    freed_k2 = not (flags & cv2.CALIB_FIX_K2)
    freed_k3 = not (flags & cv2.CALIB_FIX_K3)
    freed_tang = not (flags & cv2.CALIB_ZERO_TANGENT_DIST)

    # Identify the highest-order freed radial term and its ±sigma.
    if freed_k3:
        name, val, sig = "k3", dvals[4], std[8]
    elif freed_k2:
        name, val, sig = "k2", dvals[1], std[5]
    else:
        name, val, sig = "k1", dvals[0], std[4]

    significant = abs(val) > 2.0 * sig if sig > 0 else abs(val) > 0
    # A radial term can be statistically non-zero yet physically absurd
    # (huge, alternating-sign k2/k3 that fit target warp, not the lens).
    # Treat |k2|>1 or |k3|>1 as overfit regardless of significance.
    overfit_large = name in ("k2", "k3") and abs(val) > 1.0
    d_rms = (rms_simpler - rms) if rms_simpler is not None else None

    # Currently at the simplest model (k1 only): only advise going simpler
    # if even k1 is doing nothing.
    if not freed_k2 and not freed_k3 and not freed_tang:
        if not significant:
            return (f"Model guidance: even k1 is not significant "
                    f"({val:+.3g} ± {sig:.2g}) — this lens is close to "
                    "distortion-free; near-zero correction is expected, "
                    "and applying no correction is also reasonable.")
        return (f"Model guidance: k1 is significant ({val:+.3g} ± "
                f"{sig:.2g}) and the model is already minimal — keep "
                "k1 only.")

    # A simpler model exists.  Recommend simplifying when the top term is
    # insignificant, OR when it is implausibly large (overfitting).
    if overfit_large:
        return (f"Model guidance: {name} = {val:+.3g} is implausibly large "
                "for real lens distortion and likely fits target defects "
                f"rather than the lens (RMS barely changes: {d_rms:+.3f} px "
                "simpler)" if d_rms is not None else
                f"Model guidance: {name} = {val:+.3g} is implausibly large "
                "for real lens distortion and likely fits target defects "
                "rather than the lens") + " — prefer a simpler model."

    if not significant and (d_rms is None or d_rms < 0.1):
        extra = (f" (dropping it changes RMS by only {d_rms:+.3f} px)"
                 if d_rms is not None else "")
        return (f"Model guidance: {name} is not significant ({val:+.3g} "
                f"± {sig:.2g}){extra} — prefer the next-simpler model.")

    if significant:
        extra = (f"; dropping it would raise RMS by {d_rms:.2f} px"
                 if d_rms is not None and d_rms >= 0.1 else "")
        return (f"Model guidance: {name} is significant ({val:+.3g} ± "
                f"{sig:.2g}){extra} — keep the current model for this lens.")

    # Insignificant but expensive to drop — ambiguous, stay put quietly.
    return (f"Model guidance: {name} is marginal ({val:+.3g} ± {sig:.2g}); "
            "the current model is a reasonable choice.")


def calibrate_and_diagnose(obj_points, img_points, img_shape, flags,
                           pattern_size, cell_w_m):
    """cv2.calibrateCameraExtended plus quality diagnostics.

    Returns a dict with the calibration result, parameter standard
    deviations, true per-image RMS errors, target-planarity residuals,
    frame coverage, a GOOD/MARGINAL/POOR verdict and a list of warning
    strings.  Pure computation — no GUI dependencies.
    """
    (rms, camera_matrix, dist_coeffs, rvecs, tvecs,
     std_intrinsics, _std_extr, _per_view) = cv2.calibrateCameraExtended(
        obj_points, img_points, img_shape, None, None, flags=flags)

    per_rms = compute_per_image_rms(obj_points, img_points, rvecs, tvecs,
                                    camera_matrix, dist_coeffs)
    planarity = planarity_check(obj_points[0], img_points,
                                pattern_size, cell_w_m)
    coverage = coverage_fraction(img_points, img_shape)
    verdict = quality_verdict(rms)

    std = np.asarray(std_intrinsics, dtype=np.float64).ravel()
    if std.size < 9:                      # defensive: pad to k3 slot
        std = np.pad(std, (0, 9 - std.size))
    dvals = np.zeros(5)
    dr = dist_coeffs.ravel()
    dvals[:min(5, dr.size)] = dr[:5]

    warnings = []
    if verdict == "POOR":
        warnings.append(
            f"RMS reprojection error {rms:.2f} px is far above the "
            f"~{RMS_GOOD} px expected of a sound calibration. "
            "DO NOT apply this file to imagery — fix the causes below "
            "and recalibrate.")
    elif verdict == "MARGINAL":
        warnings.append(
            f"RMS reprojection error {rms:.2f} px is higher than expected "
            f"(good calibrations are typically < {RMS_GOOD} px). "
            "Use with caution.")

    finite_mm = [mm for _, mm in planarity if np.isfinite(mm)]
    if finite_mm and max(finite_mm) > PLANARITY_WARN_MM:
        worst = int(np.nanargmax([mm for _, mm in planarity])) + 1
        warnings.append(
            f"Target planarity: the corner grid deviates from a flat plane "
            f"by up to ~{max(finite_mm):.1f} mm (image {worst}). "
            "calibrateCamera assumes a rigid plane — mount the pattern on "
            "a flat, stiff backing (foam board / dibond / glass) and "
            "re-shoot.")

    if coverage < COVERAGE_WARN_FRACTION:
        warnings.append(
            f"Detected corners cover only {coverage * 100:.0f}% of the "
            "frame. Distortion outside the covered region is extrapolated "
            "and unreliable — include poses where the board reaches the "
            "frame edges and corners.")

    med = float(np.median(per_rms))
    outliers = [i for i, e in enumerate(per_rms)
                if med > 0 and e > 2.0 * med]
    if outliers:
        warnings.append(
            "Images with per-image RMS above 2× the median (consider "
            "removing them and recalibrating): "
            + ", ".join(f"#{i + 1} ({per_rms[i]:.2f} px)" for i in outliers))

    freed_k2 = not (flags & cv2.CALIB_FIX_K2)
    freed_k3 = not (flags & cv2.CALIB_FIX_K3)
    suspect = []
    if freed_k2 and abs(dvals[1]) < 2.0 * std[5]:
        suspect.append(f"k2 = {dvals[1]:.3g} ± {std[5]:.2g} (not significant)")
    if freed_k3 and abs(dvals[4]) < 2.0 * std[8]:
        suspect.append(f"k3 = {dvals[4]:.3g} ± {std[8]:.2g} (not significant)")
    if freed_k2 and abs(dvals[1]) > 1.0:
        suspect.append(f"k2 = {dvals[1]:.3g} (unusually large)")
    if freed_k3 and abs(dvals[4]) > 1.0:
        suspect.append(f"k3 = {dvals[4]:.3g} (unusually large)")
    if suspect:
        warnings.append(
            "Suspect distortion coefficients — likely overfitting to "
            "target defects rather than real lens distortion: "
            + "; ".join(suspect) + ". Prefer a simpler distortion model.")

    # Evidence-based model recommendation.  Fit the next-simpler model
    # once (cheap: one extra calibrateCamera) to report the RMS cost of
    # dropping the highest freed term.  Best-effort — never fatal.
    rms_simpler = None
    simpler = _simpler_flags(flags)
    if simpler is not None:
        try:
            rms_simpler = float(cv2.calibrateCamera(
                obj_points, img_points, img_shape, None, None,
                flags=simpler)[0])
        except Exception:            # noqa: BLE001 — guidance is optional
            rms_simpler = None
    guidance = model_guidance(flags, dvals, std, rms, rms_simpler)

    return {
        "rms": float(rms),
        "camera_matrix": camera_matrix,
        "dist_coeff": dist_coeffs,
        "rvecs": rvecs,
        "tvecs": tvecs,
        "std_intrinsics": std,
        "per_image_rms": per_rms,
        "planarity": planarity,
        "coverage": coverage,
        "verdict": verdict,
        "warnings": warnings,
        "guidance": guidance,
    }


# %% ————————————————————————————— main GUI ————————————————————————————
class LensCorrectionWindow(ctk.CTkToplevel):
    """Intrinsic lens calibration using checkerboard images."""

    def __init__(self, master=None, **kw):
        super().__init__(master=master, **kw)
        self.title("Intrinsic Lens Correction")
        #self.geometry("1200x800")
        fit_geometry(self, 1200, 800, resizable = True)
        try:
            self.after(200, lambda: self.iconphoto(False, tk.PhotoImage(file=resource_path("launch_logo.png"))))
        except Exception:
            pass

        # ——— close handler ———
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # ——— state ———
        self.input_folder = None
        self.output_folder = None
        self.calibration_data = None
        self._cancel_requested = False
        self._job_running = False
        self._worker_thread = None
        self._job_start_time = None

        # ——— layout ———
        self.grid_rowconfigure(0, weight=3)
        self.grid_rowconfigure(1, weight=0)
        self.grid_rowconfigure(2, weight=0)
        self.grid_columnconfigure(0, weight=1)

        # ---- TOP: preview panel ----
        self.top_panel = ctk.CTkFrame(self, fg_color="black")
        self.top_panel.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.fig, self.axes = plt.subplots(1, 2, figsize=(12, 5))
        self.axes[0].set_title("Detected Corners (sample)")
        self.axes[0].axis("off")
        self.axes[1].set_title("Undistorted Result")
        self.axes[1].axis("off")
        self.fig.tight_layout()
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=self.top_panel)
        self.canvas_plot.get_tk_widget().pack(fill="both", expand=True)

        # ---- BOTTOM: controls ----
        self.bottom_panel = ctk.CTkFrame(self)
        self.bottom_panel.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)

        # Row 1 — input folder
        row1 = ctk.CTkFrame(self.bottom_panel)
        row1.pack(fill="x", padx=5, pady=2)

        ctk.CTkButton(row1, text="Browse Checkerboard Folder",
                      command=self._browse_input).grid(
            row=0, column=0, padx=5, pady=5)
        self.input_label = ctk.CTkLabel(row1, text="No folder selected")
        self.input_label.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        # Row 2 — checkerboard parameters
        row2 = ctk.CTkFrame(self.bottom_panel)
        row2.pack(fill="x", padx=5, pady=2)

        ctk.CTkLabel(row2, text="Squares across (columns):").grid(
            row=0, column=0, padx=3, pady=3)
        self.cols_entry = ctk.CTkEntry(row2, width=60)
        self.cols_entry.insert(0, "10")
        self.cols_entry.grid(row=0, column=1, padx=3, pady=3)

        ctk.CTkLabel(row2, text="Squares down (rows):").grid(
            row=0, column=2, padx=3, pady=3)
        self.rows_entry = ctk.CTkEntry(row2, width=60)
        self.rows_entry.insert(0, "7")
        self.rows_entry.grid(row=0, column=3, padx=3, pady=3)

        ctk.CTkLabel(row2, text="Cell width (mm):").grid(
            row=0, column=4, padx=3, pady=3)
        self.cell_w_entry = ctk.CTkEntry(row2, width=60)
        self.cell_w_entry.insert(0, "25")
        self.cell_w_entry.grid(row=0, column=5, padx=3, pady=3)

        ctk.CTkLabel(row2, text="Cell height (mm):").grid(
            row=0, column=6, padx=3, pady=3)
        self.cell_h_entry = ctk.CTkEntry(row2, width=60)
        self.cell_h_entry.insert(0, "25")
        self.cell_h_entry.grid(row=0, column=7, padx=3, pady=3)

        ctk.CTkLabel(row2, text="Distortion model:").grid(
            row=0, column=8, padx=(12, 3), pady=3)
        self.dist_model_var = tk.StringVar(value=DIST_MODEL_DEFAULT)
        self.dist_model_menu = ctk.CTkOptionMenu(
            row2, variable=self.dist_model_var,
            values=list(DIST_MODEL_OPTIONS.keys()), width=200)
        self.dist_model_menu.grid(row=0, column=9, padx=3, pady=3)

        # Row 3 — output & generate
        row3 = ctk.CTkFrame(self.bottom_panel)
        row3.pack(fill="x", padx=5, pady=2)

        ctk.CTkButton(row3, text="Browse Output Folder",
                      command=self._browse_output, fg_color="#8C7738", hover_color="#A18A45").grid(
            row=0, column=0, padx=5, pady=5)
        self.output_label = ctk.CTkLabel(row3, text="No output folder selected")
        self.output_label.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        ctk.CTkButton(row3, text="Generate Lens Correction File",
                      command=self._calibrate_threaded, fg_color="#0F52BA", hover_color="#2A6BD1").grid(
            row=0, column=2, padx=10, pady=5)

        self.progress_bar = ctk.CTkProgressBar(row3, width=200)
        self.progress_bar.grid(row=0, column=3, padx=5, pady=5)
        self.progress_bar.set(0)

        self.eta_label = ctk.CTkLabel(row3, text="ETA: --")
        self.eta_label.grid(row=0, column=4, padx=5, pady=5, sticky="w")

        self.btn_reset = ctk.CTkButton(
            row3, text="Reset", command=self._reset,
            width=80, fg_color="#8B0000", hover_color="#A52A2A")
        self.btn_reset.grid(row=0, column=5, padx=5, pady=5, sticky="e")

        row4 = ctk.CTkFrame(self.bottom_panel)
        row4.pack(fill="x", padx=5, pady=2)

        ctk.CTkButton(
            row4, text="Save Settings",fg_color="#4F5D75",hover_color="#61708A", command=self._save_settings
        ).grid(row=0, column=0, padx=5, pady=5, sticky="w")

        ctk.CTkButton(
            row4, text="Load Settings",fg_color="#4F5D75",hover_color="#61708A", command=self._load_settings
        ).grid(row=0, column=1, padx=5, pady=5, sticky="w")

        # ---- CONSOLE ----
        self.console_frame = ctk.CTkFrame(self)
        self.console_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.console_text = tk.Text(self.console_frame, wrap="word", height=10)
        self.console_text.pack(fill="both", expand=True, padx=5, pady=5)
        self._console_redir = setup_console(self.console_text,
            "Provide a folder of checkerboard images.\n"
            "Enter the number of SQUARES (not inner corners) — the\n"
            "Module automatically converts to inner corners (squares − 1).\n"
            "  e.g. 17 squares across × 19 down → 16 × 18 inner corners\n"
            "Cell width/height = physical size of each cell in mm.\n"
            "For square checkerboards, set both to the same value.\n\n"
            "Distortion model: 'k1 + k2' suits most lenses;\nChoose"
            "'Full 5-term' only for strong wide-angle distortion.\n"
            "Output: lens_calibration.pkl  +  calibration_report.txt\n"
            "--------------------------------")

    # ——————————————————————————— close handler —————————————————————————
    def _on_close(self):
        """Clean up and close the window."""
        self._request_cancel(quiet=True)
        restore_console(self._console_redir)
        self.destroy()

    # ——— browse callbacks ———

    def _browse_input(self):
        d = filedialog.askdirectory(parent= self,title="Select Checkerboard Image Folder")
        if d:
            self.input_folder = d
            n = len(self._collect_images(d))
            self.input_label.configure(text=f"{d}  ({n} images)")

    def _browse_output(self):
        d = filedialog.askdirectory(parent= self,title="Select Output Folder")
        if d:
            self.output_folder = d
            self.output_label.configure(text=d)

    def _get_settings_dict(self):
        return {
            "module": "lens_correction",
            "settings_version": 1,
            "paths": {
                "input_folder": self.input_folder or "",
                "output_folder": self.output_folder or "",
            },
            "ui_state": {
                "sq_cols": self.cols_entry.get().strip(),
                "sq_rows": self.rows_entry.get().strip(),
                "cell_w_mm": self.cell_w_entry.get().strip(),
                "cell_h_mm": self.cell_h_entry.get().strip(),
                "dist_model": self.dist_model_var.get(),
            },
        }

    def _save_settings(self):
        data = self._get_settings_dict()
        initialdir = self.output_folder or self.input_folder or None
        path = save_settings_json(self, "lens_correction", data, initialdir=initialdir)
        if path:
            print(f"Saved settings: {path}")

    def _load_settings(self):
        initialdir = self.output_folder or self.input_folder or None
        data, path = load_settings_json(self, "lens_correction", initialdir=initialdir)
        if not data:
            return

        paths = data.get("paths", {})
        ui_state = data.get("ui_state", {})

        self.input_folder = paths.get("input_folder") or None
        self.output_folder = paths.get("output_folder") or None

        if self.input_folder:
            n = len(self._collect_images(self.input_folder)) if os.path.isdir(self.input_folder) else 0
            label_text = f"{self.input_folder}  ({n} images)" if n else self.input_folder
            self.input_label.configure(text=label_text)
        else:
            self.input_label.configure(text="No folder selected")

        if self.output_folder:
            self.output_label.configure(text=self.output_folder)
        else:
            self.output_label.configure(text="No output folder selected")

        for entry, key, default in (
            (self.cols_entry, "sq_cols", "10"),
            (self.rows_entry, "sq_rows", "7"),
            (self.cell_w_entry, "cell_w_mm", "25"),
            (self.cell_h_entry, "cell_h_mm", "25"),
        ):
            entry.delete(0, tk.END)
            entry.insert(0, str(ui_state.get(key, default)))

        model = ui_state.get("dist_model", DIST_MODEL_DEFAULT)
        if model not in DIST_MODEL_OPTIONS:
            model = DIST_MODEL_DEFAULT
        self.dist_model_var.set(model)

        print(f"Loaded settings: {path}")

    # ——— helpers ———

    @staticmethod
    def _collect_images(folder):
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
        imgs = []
        for ext in exts:
            imgs.extend(glob.glob(os.path.join(folder, ext)))
        return sorted(imgs)

    def _request_cancel(self, quiet=False):
        if self._job_running:
            self._cancel_requested = True
            if not quiet:
                print("Cancellation requested…")

    def _update_progress_ui(self, value=None, eta_text=None):
        if value is not None:
            self.progress_bar.set(value)
        if eta_text is not None:
            self.eta_label.configure(text=eta_text)

    # ——— reset ———

    def _reset(self):
        self._request_cancel(quiet=True)
        self.input_folder = None
        self.output_folder = None
        self.calibration_data = None
        self.input_label.configure(text="No folder selected")
        self.output_label.configure(text="No output folder selected")
        self._update_progress_ui(0, "ETA: --")
        self.cols_entry.delete(0, tk.END)
        self.cols_entry.insert(0, "10")
        self.rows_entry.delete(0, tk.END)
        self.rows_entry.insert(0, "7")
        self.cell_w_entry.delete(0, tk.END)
        self.cell_w_entry.insert(0, "25")
        self.cell_h_entry.delete(0, tk.END)
        self.cell_h_entry.insert(0, "25")
        self.dist_model_var.set(DIST_MODEL_DEFAULT)

        for ax in self.axes:
            ax.clear()
            ax.axis("off")
        self.axes[0].set_title("Detected Corners (sample)")
        self.axes[1].set_title("Undistorted Result")
        self.fig.tight_layout()
        self.canvas_plot.draw()
        self.console_text.delete("1.0", tk.END)
        print("Session reset.\n--------------------------------")

    # ——— calibration ———

    def _ui_call(self, func, *args, **kwargs):
        self.after(0, lambda: func(*args, **kwargs))

    def _ui_message(self, kind, title, message):
        fn = getattr(messagebox, f"show{kind}")
        self._ui_call(fn, title, message, parent=self)

    def _ui_progress(self, value, eta_text=None):
        self._ui_call(self._update_progress_ui, value, eta_text)

    def _apply_preview(self, vis_bgr, undist_bgr):
        self.axes[0].clear()
        self.axes[0].imshow(cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB))
        self.axes[0].set_title("Detected Corners (sample)")
        self.axes[0].axis("off")

        self.axes[1].clear()
        self.axes[1].imshow(cv2.cvtColor(undist_bgr, cv2.COLOR_BGR2RGB))
        self.axes[1].set_title("Undistorted (alpha=0) + straight reference grid")
        self.axes[1].axis("off")
        self.fig.tight_layout()
        self.canvas_plot.draw()

    def _collect_calibration_config(self):
        return {
            "input_folder": self.input_folder,
            "output_folder": self.output_folder,
            "sq_cols": int(self.cols_entry.get()),
            "sq_rows": int(self.rows_entry.get()),
            "cell_w_mm": float(self.cell_w_entry.get()),
            "cell_h_mm": float(self.cell_h_entry.get()),
            "dist_model": self.dist_model_var.get(),
        }

    def _calibrate_threaded(self):
        if self._job_running:
            messagebox.showinfo("Busy", "Calibration is already running.", parent=self)
            return
        try:
            cfg = self._collect_calibration_config()
        except Exception as e:
            messagebox.showerror("Error", str(e), parent=self)
            return
        self._cancel_requested = False
        self._job_running = True
        self._job_start_time = time.time()
        self._update_progress_ui(0, "ETA: estimating...")
        self._worker_thread = threading.Thread(target=self._calibrate, args=(cfg,), daemon=True)
        self._worker_thread.start()

    def _calibrate(self, cfg):
        try:
            input_folder = cfg["input_folder"]
            output_folder = cfg["output_folder"]
            if not input_folder:
                self._ui_message("warning", "Warning",
                                 "Select a checkerboard image folder first.")
                return
            if not output_folder:
                self._ui_message("warning", "Warning",
                                 "Select an output folder first.")
                return

            sq_cols = cfg["sq_cols"]
            sq_rows = cfg["sq_rows"]
            # OpenCV needs inner corners = squares − 1 in each direction
            n_cols = sq_cols - 1
            n_rows = sq_rows - 1
            if n_cols < 2 or n_rows < 2:
                self._ui_message(
                    "error",
                    "Error",
                    "Need at least 3 squares in each direction (2 inner corners).")
                return
            cell_w_mm = cfg["cell_w_mm"]
            cell_h_mm = cfg["cell_h_mm"]
            cell_w_m = cell_w_mm / 1000.0
            cell_h_m = cell_h_mm / 1000.0

            is_square = abs(cell_w_mm - cell_h_mm) < 0.001
            cell_desc = (f"{cell_w_mm} mm" if is_square
                         else f"{cell_w_mm} × {cell_h_mm} mm")

            pattern_size = (n_cols, n_rows)
            images = self._collect_images(input_folder)
            if not images:
                self._ui_message("warning", "Warning",
                                 "No images found in the selected folder.")
                return

            print(f"Found {len(images)} images.")
            print(f"Board: {sq_cols}×{sq_rows} squares → "
                  f"{n_cols}×{n_rows} inner corners, "
                  f"cell size {cell_desc} …\n")

            # ---- prepare object points ----
            objp = np.zeros((n_cols * n_rows, 3), np.float32)
            for r in range(n_rows):
                for c in range(n_cols):
                    idx = r * n_cols + c
                    objp[idx, 0] = c * cell_w_m
                    objp[idx, 1] = r * cell_h_m

            obj_points = []
            img_points = []
            img_shape = None
            sample_img = None
            sample_corners = None
            detected_count = 0

            criteria = (cv2.TERM_CRITERIA_EPS +
                        cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

            for idx, img_path in enumerate(images):
                if self._cancel_requested:
                    print("Calibration cancelled.")
                    self._ui_progress(0, "ETA: Cancelled")
                    return

                img = imread_safe(img_path)
                if img is None:
                    print(f"[WARN] Cannot read: {os.path.basename(img_path)}")
                    continue

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if img_shape is None:
                    img_shape = gray.shape[::-1]
                elif gray.shape[::-1] != img_shape:
                    # calibrateCamera assumes a single image size, and the
                    # planarity / coverage diagnostics stack all img_points
                    # together — corners from a differently-sized image would
                    # corrupt K/D and the diagnostics with no warning. Skip it.
                    print(f"  ✗ {os.path.basename(img_path)} — size "
                          f"{gray.shape[::-1]} differs from {img_shape}; "
                          f"skipped (mixed resolutions corrupt calibration).")
                    frac = (idx + 1) / len(images)
                    eta = compute_eta(self._job_start_time or time.time(),
                                      idx + 1, len(images))
                    self._ui_progress(frac, f"ETA: {format_eta(eta)}")
                    continue

                found, corners = cv2.findChessboardCorners(
                    gray, pattern_size,
                    cv2.CALIB_CB_ADAPTIVE_THRESH +
                    cv2.CALIB_CB_FAST_CHECK +
                    cv2.CALIB_CB_NORMALIZE_IMAGE)

                if found:
                    corners_refined = cv2.cornerSubPix(
                        gray, corners, (11, 11), (-1, -1), criteria)
                    obj_points.append(objp)
                    img_points.append(corners_refined)
                    detected_count += 1
                    print(f"  ✓ {os.path.basename(img_path)} — corners found")

                    if sample_img is None:
                        sample_img = img.copy()
                        sample_corners = corners_refined.copy()
                else:
                    print(f"  ✗ {os.path.basename(img_path)} — corners NOT found")

                frac = (idx + 1) / len(images)
                eta = compute_eta(self._job_start_time or time.time(), idx + 1, len(images))
                self._ui_progress(frac, f"ETA: {format_eta(eta)}")

            print(f"\nDetected corners in {detected_count}/{len(images)} images.")

            if detected_count < 3:
                self._ui_message(
                    "error",
                    "Error",
                    f"Only {detected_count} images had detectable corners.\n"
                    "Need at least 3 for reliable calibration.\n"
                    "Check your checkerboard dimensions (cols × rows).")
                return

            model_label = cfg.get("dist_model") or DIST_MODEL_DEFAULT
            if model_label not in DIST_MODEL_OPTIONS:
                model_label = DIST_MODEL_DEFAULT
            calib_flags = DIST_MODEL_OPTIONS[model_label]

            print(f"Running calibration …  (distortion model: {model_label})")
            diag = calibrate_and_diagnose(
                obj_points, img_points, img_shape, calib_flags,
                pattern_size, cell_w_m)
            ret = diag["rms"]
            camera_matrix = diag["camera_matrix"]
            dist_coeffs = diag["dist_coeff"]
            std_i = diag["std_intrinsics"]
            per_image_rms = diag["per_image_rms"]
            planarity = diag["planarity"]
            coverage = diag["coverage"]
            verdict = diag["verdict"]
            q_warnings = diag["warnings"]
            guidance = diag.get("guidance")

            print(f"\nCalibration RMS reprojection error: {ret:.4f} px")
            print(f"Calibration quality: {verdict}  "
                  f"(GOOD < {RMS_GOOD} px, MARGINAL < {RMS_MARGINAL} px)")
            print(f"\nCamera matrix:\n{camera_matrix}")
            print(f"\nDistortion coefficients:\n{dist_coeffs.ravel()}")

            fx = camera_matrix[0, 0]
            fy = camera_matrix[1, 1]
            cx = camera_matrix[0, 2]
            cy = camera_matrix[1, 2]
            W, H = img_shape
            print(f"\nFocal length:  fx={fx:.2f} ± {std_i[0]:.2f} px,  "
                  f"fy={fy:.2f} ± {std_i[1]:.2f} px")
            print(f"Principal point:  cx={cx:.2f} ± {std_i[2]:.2f} px,  "
                  f"cy={cy:.2f} ± {std_i[3]:.2f} px")
            print(f"Image size: {W} × {H} px")
            print(f"Corner coverage of frame: {coverage * 100:.0f}%")

            if not is_square:
                print(f"\n[NOTE] Rectangular cells used ({cell_w_mm} × {cell_h_mm} mm). Object points have been scaled accordingly.")

            for w in q_warnings:
                print(f"\n[WARNING] {w}")

            if guidance:
                print(f"\n{guidance}")

            cal_data = {
                "camera_matrix": camera_matrix,
                "dist_coeff": dist_coeffs,
                "rms_error": ret,
                "image_size": img_shape,
                "n_images_used": detected_count,
                "pattern_size": pattern_size,
                "board_squares": (sq_cols, sq_rows),
                "cell_width_m": cell_w_m,
                "cell_height_m": cell_h_m,
                "square_size_m": cell_w_m,
                # — added by the quality-diagnostics update; all previous
                #   keys are unchanged, so downstream consumers are safe —
                "dist_model": model_label,
                "calib_flags": int(calib_flags),
                "per_image_rms": [float(e) for e in per_image_rms],
                "quality": verdict,
                "quality_warnings": list(q_warnings),
                "model_guidance": guidance or "",
            }
            self.calibration_data = cal_data
            pkl_path = os.path.join(output_folder, "lens_calibration.pkl")
            pkl_path, json_path = save_lens_calibration(cal_data, pkl_path)
            print(f"\nSaved: {pkl_path}")
            if json_path:
                print(f"Saved: {json_path}  (plain-text sidecar)")

            txt_path = os.path.join(output_folder, "calibration_report.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("=== Lens Calibration Report ===\n\n")
                f.write(f"GeoCamPal version: {__version__}\n")
                f.write(f"Input folder: {input_folder}\n")
                f.write(f"Images found: {len(images)}\n")
                f.write(f"Images with corners: {detected_count}\n")
                f.write(f"Board: {sq_cols} × {sq_rows} squares\n")
                f.write(f"Inner corners: {n_cols} × {n_rows}\n")
                if is_square:
                    f.write(f"Cell size: {cell_w_mm} mm (square)\n")
                else:
                    f.write(f"Cell width: {cell_w_mm} mm\n")
                    f.write(f"Cell height: {cell_h_mm} mm\n")
                f.write(f"Distortion model: {model_label}\n\n")
                f.write(f"RMS reprojection error: {ret:.4f} px\n")
                f.write(f"Calibration quality: {verdict}  "
                        f"(GOOD < {RMS_GOOD} px, MARGINAL < {RMS_MARGINAL} px)\n\n")
                if q_warnings:
                    f.write("Warnings:\n")
                    for w in q_warnings:
                        f.write(f"  ! {w}\n")
                    f.write("\n")
                if guidance:
                    f.write(f"{guidance}\n\n")
                f.write(f"Camera matrix:\n{camera_matrix}\n\n")
                f.write(f"Focal length: fx={fx:.2f} ± {std_i[0]:.2f} px, "
                        f"fy={fy:.2f} ± {std_i[1]:.2f} px\n")
                f.write(f"Principal point: cx={cx:.2f} ± {std_i[2]:.2f} px, "
                        f"cy={cy:.2f} ± {std_i[3]:.2f} px\n")
                f.write(f"Image size: {W} × {H} px\n")
                f.write(f"Corner coverage of frame: {coverage * 100:.0f}%\n\n")
                f.write(f"Distortion coefficients:\n{dist_coeffs.ravel()}\n")
                dvals = np.zeros(5)
                dr = dist_coeffs.ravel()
                dvals[:min(5, dr.size)] = dr[:5]
                f.write(f"  k1 = {dvals[0]: .6f} ± {std_i[4]:.6f}\n")
                f.write(f"  k2 = {dvals[1]: .6f} ± {std_i[5]:.6f}\n")
                f.write(f"  p1 = {dvals[2]: .6f} ± {std_i[6]:.6f}\n")
                f.write(f"  p2 = {dvals[3]: .6f} ± {std_i[7]:.6f}\n")
                f.write(f"  k3 = {dvals[4]: .6f} ± {std_i[8]:.6f}\n")
                f.write("  (fixed coefficients have sigma = 0)\n\n")
                f.write("Per-image diagnostics\n")
                f.write("  (RMS is the true per-image RMS, |residuals|/sqrt(N);\n")
                f.write("   plane-fit is the deviation of the detected grid "
                        "from a flat plane):\n")
                for i, err in enumerate(per_image_rms):
                    p_px, p_mm = planarity[i]
                    f.write(f"  Image {i + 1}: RMS {err:.4f} px | "
                            f"plane-fit {p_px:.2f} px (~{p_mm:.2f} mm)\n")
            print(f"Saved: {txt_path}")

            if sample_img is not None:
                vis = cv2.drawChessboardCorners(sample_img.copy(), pattern_size, sample_corners, True)
                Himg, Wimg = sample_img.shape[:2]
                # alpha=0 matches the Harmonise Images default ("crop to
                # valid pixels"), so this preview shows what the production
                # lens-correction output will actually look like.
                new_mtx, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (Wimg, Himg), 0, (Wimg, Himg))
                undist = cv2.undistort(sample_img, camera_matrix, dist_coeffs, None, new_mtx)
                # straight reference grid for judging correction quality
                step = max(60, Wimg // 16)
                for gx in range(step, Wimg, step):
                    cv2.line(undist, (gx, 0), (gx, Himg - 1), (0, 255, 0), 1)
                for gy in range(step, Himg, step):
                    cv2.line(undist, (0, gy), (Wimg - 1, gy), (0, 255, 0), 1)
                self._ui_call(self._apply_preview, vis, undist)

            self._ui_progress(1.0, "ETA: Completed")
            summary = (f"Calibration complete — quality: {verdict}\n"
                       f"RMS error: {ret:.4f} px\n")
            if q_warnings:
                summary += (f"\n{len(q_warnings)} warning(s) — "
                            "see console / report for details.\n")
            summary += f"\nFiles saved to:\n{output_folder}"
            self._ui_message(
                "warning" if verdict == "POOR" else "info",
                "Done", summary)

        except Exception as e:
            print(f"[ERROR] {e}")
            self._ui_progress(0, "ETA: Error")
            self._ui_message("error", "Error", str(e))
        finally:
            self._job_running = False
            self._worker_thread = None


# ——— standalone ———
if __name__ == "__main__":
    root = ctk.CTk()
    root.withdraw()
    LensCorrectionWindow(master=root)
    root.mainloop()