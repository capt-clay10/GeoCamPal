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

The console and calibration_report.txt open with a Result block that
keeps two things apart:
  • the lens distortion, given as the distance the correction moves
    pixels: at the image centre, for half of all pixels, and the
    largest movement with its place in the image.  Each value carries
    a ±: the calibration is repeated once per photo, each time without
    that photo, and the ± is the standard error of the value over these
    runs (it reflects errors that differ from photo to photo);
  • the checkerboard fit — the GOOD / MARGINAL / POOR grade of the RMS
    reprojection error — with the RMS obtained without any distortion
    model, the reduction by the model, and the RMS as a share of the
    image width (which, unlike the value in pixels, does not change
    with the image size).  The grade says how well the detected corners
    match the camera model; it is not the amount of lens distortion.

calibration_report.txt ends with an Interpretation section, marked as
such.  It compares the largest movement with its ± and says whether the
photos show a lens distortion, do not show one, or only point to one;
it gives no interpretation where that comparison was not reliable in
tests (fewer than 10 photos, corners covering less than 60 % of the
frame, a tangential or 5-term model, contradictory values).  The console
holds values and definitions only.

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
                                 distortion model, the Result block (pixel
                                 movement of the correction with its ±,
                                 checkerboard fit) with warnings, K matrix,
                                 focal length, principal point, distortion
                                 coefficients (each ± 1σ), frame coverage,
                                 per-image RMS + planarity diagnostics, and
                                 the Interpretation section

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
import textwrap
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
    show_path, short_path, PATH_INPUT_COLOR,
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


# %% ——————————————— result block: distortion in pixels ———————————————
#
# The pixel movement of a distortion model is evaluated at the centres of
# this many equal cells (across, down), at every pixel along the four
# image edges, and at the image centre.
MOVEMENT_GRID = (301, 201)

# Limits of the Interpretation section of the report.  The three verdicts
# compare the largest pixel movement with its leave-one-photo-out standard
# error (±).  The limits come from tests on calibrations with a known lens
# (about 2,000 real, modified and simulated calibrations, 2026-11-16); no
# verdict is given outside the conditions in which the comparison proved
# reliable there.
INTERPRET_NOT_SHOWN = 2.0            # movement below this many ±: not shown
INTERPRET_SHOWN = 3.0                # movement of this many ± or more: shown
INTERPRET_MIN_PHOTOS = 10            # fewer photos: no interpretation
INTERPRET_CONTRADICTION_PCT = 10.0   # "not shown", yet the model removes
                                     # this share of the misfit or more
EXAMPLE_PIXEL_CM = 10.0              # ground size of one pixel in the example


def no_distortion_rms(obj_points, img_points, img_shape):
    """RMS reprojection error (px) of the same corners when every
    distortion coefficient is fixed at zero, i.e. for a camera without
    lens distortion.  Compared with the RMS of the chosen model it shows
    how much of the misfit the distortion model accounts for."""
    flags = (cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3
             | cv2.CALIB_ZERO_TANGENT_DIST)
    return float(cv2.calibrateCamera(obj_points, img_points, img_shape,
                                     None, None, flags=flags)[0])


def _distort_pixels(pix, K, D):
    """Where the distortion model puts pixels whose distortion-free
    positions are ``pix`` (N × 2): the model itself, no iteration."""
    rays = np.column_stack([(pix[:, 0] - K[0, 2]) / K[0, 0],
                            (pix[:, 1] - K[1, 2]) / K[1, 1],
                            np.ones(len(pix))])
    # Identity pose: projectPoints then applies the distortion model and K.
    out, _ = cv2.projectPoints(rays.reshape(-1, 1, 3), np.zeros(3),
                               np.zeros(3), K, D)
    return out.reshape(-1, 2)


def _movement_positions(img_shape):
    """The pixels at which the movement is evaluated: the centres of
    MOVEMENT_GRID equal cells, every pixel along the four image edges,
    and the image centre (last).  Returns (positions N × 2, number of
    cell centres)."""
    W, H = int(img_shape[0]), int(img_shape[1])
    nx, ny = min(W, MOVEMENT_GRID[0]), min(H, MOVEMENT_GRID[1])
    gx, gy = np.meshgrid((np.arange(nx) + 0.5) * W / nx - 0.5,
                         (np.arange(ny) + 0.5) * H / ny - 0.5)
    cells = np.column_stack([gx.ravel(), gy.ravel()])
    ex = np.arange(W, dtype=np.float64)
    ey = np.arange(H, dtype=np.float64)
    edges = np.vstack([
        np.column_stack([ex, np.zeros(W)]),
        np.column_stack([ex, np.full(W, H - 1.0)]),
        np.column_stack([np.zeros(H), ey]),
        np.column_stack([np.full(H, W - 1.0), ey])])
    centre = np.array([[(W - 1) / 2.0, (H - 1) / 2.0]])
    return np.vstack([cells, edges, centre]), len(cells)


def _movement_vectors(K, D, pix):
    """Corrected position minus original position of the pixels ``pix``.

    The model gives the original position from the corrected one, so it
    has to be inverted; OpenCV does that by iteration.  Putting the result
    back through the model must return the starting pixel.  If it does not
    (coefficients too strong for the iteration), the model is used the
    other way round: minus the movement of the pixel that ends up at each
    position.
    """
    other = None                     # the corrected position of each pixel
    try:
        other = cv2.undistortPointsIter(
            pix.reshape(-1, 1, 2), K, D, None, K,
            (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 100, 1e-10)
        ).reshape(-1, 2)
        if not (np.all(np.isfinite(other)) and
                np.abs(_distort_pixels(other, K, D) - pix).max() < 1e-3):
            other = None
    except cv2.error:
        other = None
    if other is None:
        return pix - _distort_pixels(pix, K, D)
    return other - pix


def pixel_movement(camera_matrix, dist_coeffs, img_shape):
    """How far the correction moves the pixels of a W × H image.

    For pixels spread over the whole frame, the place a pixel has in the
    original image is compared with the place the correction gives it,
    with the same camera matrix on both sides.  The distance between the
    two is the lens distortion at that pixel, in pixels.  It is the
    movement by the distortion model itself: the rescaling that "Crop to
    valid pixels" adds when a correction is applied in Harmonise Images
    is not part of it.

    The pixels looked at are the centres of MOVEMENT_GRID equal cells
    (they give the median), every pixel along the four image edges, and
    the image centre.

    Returns a dict with
        centre         movement at the image centre (px)
        median         half of the pixels move no more than this (px)
        largest        largest movement (px)
        largest_xy     pixel position (x, y) of the largest movement
        largest_index  its index among the pixels looked at
    """
    K = np.asarray(camera_matrix, dtype=np.float64)
    D = np.asarray(dist_coeffs, dtype=np.float64).ravel()
    pix, n_cells = _movement_positions(img_shape)
    moved = np.linalg.norm(_movement_vectors(K, D, pix), axis=1)
    i = int(np.argmax(moved[:-1]))
    return {
        "centre": float(moved[-1]),
        "median": float(np.median(moved[:n_cells])),
        "largest": float(moved[i]),
        "largest_xy": (float(pix[i, 0]), float(pix[i, 1])),
        "largest_index": i,
    }


def movement_uncertainty(obj_points, img_points, img_shape, flags, movement,
                         progress=None):
    """Leave-one-photo-out standard errors (±) of the pixel movement.

    The calibration is repeated once for every photo, each time without
    that photo and with the same distortion model.  How much the movement
    of a pixel differs between these runs gives its standard error (the
    jackknife).  It reflects errors that differ from photo to photo; an
    error that is the same in every photo, such as an inexact printed
    pattern, is not part of it.

    ``progress(i, n)`` is called before each run; if it returns False the
    work stops.  Returns None with fewer than 4 photos, when one of the
    runs fails, or when the work was stopped; otherwise a dict with
        n        number of repeated calibrations
        centre   ± of the movement at the image centre (px)
        median   half of the pixels have a ± no larger than this (px)
        largest  ± at the pixel of the largest movement (px)
    """
    n = len(img_points)
    if n < 4:
        return None
    obj_points, img_points = list(obj_points), list(img_points)
    pix, n_cells = _movement_positions(img_shape)
    total = np.zeros_like(pix)
    total_sq = np.zeros(len(pix))
    for i in range(n):
        if progress is not None and progress(i, n) is False:
            return None
        try:
            _rms, K, D, _rv, _tv = cv2.calibrateCamera(
                obj_points[:i] + obj_points[i + 1:],
                img_points[:i] + img_points[i + 1:],
                img_shape, None, None, flags=flags)
        except cv2.error:
            return None
        vec = _movement_vectors(np.asarray(K, dtype=np.float64),
                                np.asarray(D, dtype=np.float64).ravel(), pix)
        if not np.all(np.isfinite(vec)):
            return None
        total += vec
        total_sq += (vec * vec).sum(axis=1)
    spread = np.maximum(total_sq - (total * total).sum(axis=1) / n, 0.0)
    se = np.sqrt((n - 1) / n * spread)
    return {
        "n": n,
        "centre": float(se[-1]),
        "median": float(np.median(se[:n_cells])),
        "largest": float(se[movement["largest_index"]]),
    }


def board_error_split(objp, img_points, rvecs, tvecs, camera_matrix,
                      dist_coeffs):
    """Where the remaining misfit sits on the board, in millimetres.

    The reprojection residual of every corner (detected minus fitted
    position) is taken back from pixels to the plane of the board.  The
    part of it that is the same in every photo belongs to the pattern
    itself (printing, assembly); the part that changes from photo to
    photo comes from bending, blur or corner finding.  The chance share
    that the changing part leaves in the mean over the photos is taken
    out.  Object points are expected in metres.

    Returns a dict with total, fixed and changing (RMS, mm), or None with
    fewer than 2 photos.
    """
    n = len(img_points)
    if n < 2:
        return None
    base = np.asarray(objp, dtype=np.float64).reshape(-1, 3)
    K = np.asarray(camera_matrix, dtype=np.float64)
    D = np.asarray(dist_coeffs, dtype=np.float64)
    step = 0.001                                         # 1 mm on the board
    res = []
    for pts, rvec, tvec in zip(img_points, rvecs, tvecs):
        def project(b):
            return cv2.projectPoints(b, rvec, tvec, K, D)[0].reshape(-1, 2)
        p0 = project(base)
        jx = project(base + [step, 0.0, 0.0]) - p0       # px per mm, along x
        jy = project(base + [0.0, step, 0.0]) - p0       # px per mm, along y
        r = np.asarray(pts, dtype=np.float64).reshape(-1, 2) - p0
        det = jx[:, 0] * jy[:, 1] - jx[:, 1] * jy[:, 0]
        res.append(np.column_stack([
            (r[:, 0] * jy[:, 1] - r[:, 1] * jy[:, 0]) / det,
            (-r[:, 0] * jx[:, 1] + r[:, 1] * jx[:, 0]) / det]))
    res = np.array(res)                                  # photo × corner × 2
    mean = res.mean(axis=0)
    changing_sq = float(((res - mean) ** 2).sum(axis=(0, 2)).mean() / (n - 1))
    fixed_sq = float((mean ** 2).sum(axis=1).mean()) - changing_sq / n
    return {
        "total": float(np.sqrt((res ** 2).sum(axis=2).mean())),
        "fixed": float(np.sqrt(max(fixed_sq, 0.0))),
        "changing": float(np.sqrt(changing_sq)),
    }


def movement_place(xy, img_shape):
    """Say where in the image a pixel position is: one of the corners,
    one of the edges (within 2 % of the frame), or its coordinates."""
    W, H = int(img_shape[0]), int(img_shape[1])
    x, y = xy
    tol_x, tol_y = 0.02 * (W - 1), 0.02 * (H - 1)
    side = "left" if x <= tol_x else "right" if x >= (W - 1) - tol_x else ""
    level = "top" if y <= tol_y else "bottom" if y >= (H - 1) - tol_y else ""
    if side and level:
        return f"at the {level}-{side} corner"
    if side or level:
        return f"at the {side or level} edge"
    return f"at pixel ({x:.0f}, {y:.0f})"


def reduction_by_model(rms, rms_no_distortion):
    """How much of the misfit the distortion model removes: the difference
    of the two RMS values in px, and as a percentage of the RMS without a
    distortion model.  (None, None) when that RMS is not available."""
    if rms_no_distortion is None or rms_no_distortion <= 0:
        return None, None
    diff = rms_no_distortion - rms
    if abs(diff) < 5e-5:
        diff = 0.0
    return diff, diff / rms_no_distortion * 100.0


def headline_lines(movement, rms, verdict, uncertainty=None):
    """The two statements a run comes down to: how far the correction
    moves pixels, and how well the checkerboard images fit.  They open
    the Result block and are repeated in the closing pop-up."""
    pm = f" ± {uncertainty['largest']:.2f} px" if uncertainty else ""
    return [
        "Lens distortion:   the correction moves pixels by up to "
        f"{movement['largest']:.2f} px{pm}",
        f"Checkerboard fit:  {verdict}  (RMS reprojection error {rms:.4f} px)",
    ]


def result_lines(model_label, img_shape, movement, rms, verdict,
                 rms_no_distortion, n_warnings, uncertainty=None):
    """The Result block.  The same lines go to the console and to
    calibration_report.txt.  Values and their definitions only.

    Two statements first (``headline_lines``), then each in detail: the
    lens distortion as the distance the correction moves pixels, with the
    leave-one-photo-out ± of every value; and the checkerboard fit, with
    the RMS reprojection error without and with the distortion model, the
    reduction by the model, and the RMS as a share of the image width.
    """
    W, H = int(img_shape[0]), int(img_shape[1])

    def pm(key):
        return f"  ± {uncertainty[key]:.2f} px" if uncertainty else ""

    largest = f"    Largest movement:    {movement['largest']:.2f} px{pm('largest')}"
    if movement["largest"] >= 0.005:
        largest += ", " + movement_place(movement["largest_xy"], img_shape)
    lines = ["=== Result ==="]
    lines += headline_lines(movement, rms, verdict, uncertainty)
    lines += [
        "",
        f"Lens distortion  ({W} × {H} px image, model: {model_label})",
        "  If the correction is applied, each pixel moves by its own amount,",
        "  depending on where it is in the image:",
        f"    Image centre:        {movement['centre']:.2f} px{pm('centre')}",
        f"    Half of all pixels:  no more than {movement['median']:.2f} px"
        f"{pm('median')}",
        largest,
    ]
    if uncertainty:
        lines += [
            "  The ± is how much the value changes when one photo is left out",
            f"  (standard error from {uncertainty['n']} repeated calibrations).",
        ]
    else:
        lines += [
            "  No ± is given: it needs at least 4 photos and a calibration that",
            "  succeeds without each one of them.",
        ]
    lines += [
        "  This is the movement by the distortion model itself. \"Crop to valid",
        "  pixels\" in Harmonise Images also rescales the image; that is not",
        "  included.",
        "",
        f"Checkerboard fit  (GOOD: RMS below {RMS_GOOD} px; MARGINAL: below "
        f"{RMS_MARGINAL} px; POOR: {RMS_MARGINAL} px or more)",
        "  The fit says how well the detected corners match the camera model.",
        "  It is not the amount of lens distortion.",
    ]
    diff, percent = reduction_by_model(rms, rms_no_distortion)
    if rms_no_distortion is not None:
        lines.append("    RMS reprojection error without a distortion model:  "
                     f"{rms_no_distortion:.4f} px")
    lines.append("    RMS reprojection error with the distortion model:   "
                 f"{rms:.4f} px  ({rms / W * 100:.3f} % of the image width)")
    if diff is not None:
        lines.append("    Reduction by the distortion model:                  "
                     f"{diff:.4f} px  ({percent:.2f} %)")
    if n_warnings:
        lines += ["", f"Warnings: {n_warnings} (listed below)"]
    return lines


# %% ——————————————— interpretation (report only) ———————————————
def interpretation_scenario(flags, movement, uncertainty, rms,
                            rms_no_distortion, coverage, n_images):
    """Which case of the Interpretation section applies.

    Returns (scenario, reasons).  scenario is
        "not shown"   largest movement below INTERPRET_NOT_SHOWN × its ±
        "indicated"   between INTERPRET_NOT_SHOWN and INTERPRET_SHOWN × its ±
        "shown"       INTERPRET_SHOWN × its ± or more
        "none"        no interpretation; ``reasons`` then lists why:
                      "no ±", "photos", "coverage", "model", "contradiction"
    """
    reasons = []
    if uncertainty is None:
        reasons.append("no ±")
    if n_images < INTERPRET_MIN_PHOTOS:
        reasons.append("photos")
    if coverage < COVERAGE_WARN_FRACTION:
        reasons.append("coverage")
    if not (flags & cv2.CALIB_ZERO_TANGENT_DIST) or not (flags & cv2.CALIB_FIX_K3):
        reasons.append("model")
    ratio = 0.0
    if uncertainty is not None and movement["largest"] >= 0.005:
        se = uncertainty["largest"]
        ratio = movement["largest"] / se if se > 0 else float("inf")
        _diff, percent = reduction_by_model(rms, rms_no_distortion)
        if (ratio < INTERPRET_NOT_SHOWN and percent is not None
                and percent >= INTERPRET_CONTRADICTION_PCT):
            reasons.append("contradiction")
    if reasons:
        return "none", reasons
    if ratio < INTERPRET_NOT_SHOWN:
        return "not shown", []
    if ratio < INTERPRET_SHOWN:
        return "indicated", []
    return "shown", []


def _bullet(text):
    return textwrap.wrap(text, width=78, initial_indent="  - ",
                         subsequent_indent="    ", break_long_words=False,
                         break_on_hyphens=False)


def _example_cm(px):
    cm = px * EXAMPLE_PIXEL_CM
    return f"{cm:.0f} cm" if cm >= 10 else f"{cm:.1f} cm"


def _short_label(model_label):
    """'k1 + k2 (recommended)' → 'k1 + k2'."""
    return model_label.split(" (")[0]


def interpretation_lines(model_label, flags, img_shape, movement, uncertainty,
                         rms, rms_no_distortion, coverage, n_images,
                         board=None, warned_target=False,
                         warned_coverage=False, simpler_label=None):
    """The Interpretation section at the end of calibration_report.txt.

    Unlike the rest of the report this is not a calculated value but a
    reading of the values, chosen by fixed rules (``interpretation_scenario``)
    and marked as such in its first lines.
    """
    W = int(img_shape[0])
    H = int(img_shape[1])
    scenario, reasons = interpretation_scenario(
        flags, movement, uncertainty, rms, rms_no_distortion, coverage,
        n_images)
    lines = [
        "=== Interpretation ===",
        "Note: this section is an interpretation of the numbers above, written by",
        "GeoCamPal from fixed rules. It is not a measurement. Check it against your",
        "own use before acting on it.",
        "",
    ]
    M = movement["largest"]
    place = (" " + movement_place(movement["largest_xy"], img_shape)
             if M >= 0.005 else "")
    _diff, percent = reduction_by_model(rms, rms_no_distortion)

    if scenario == "none":
        lines.append("No interpretation is given for this calibration.")
        lines.append("Reason:" if len(reasons) == 1 else "Reasons:")
        for reason in reasons:
            if reason == "no ±":
                lines += _bullet(
                    "The ± of the pixel movement could not be calculated. It "
                    "needs at least 4 photos and a calibration that succeeds "
                    "without each one of them.")
            elif reason == "photos":
                lines += _bullet(
                    f"Only {n_images} photos were used. With fewer than "
                    f"{INTERPRET_MIN_PHOTOS} photos the ± is not reliable. "
                    f"Take at least {INTERPRET_MIN_PHOTOS} photos.")
            elif reason == "coverage":
                lines += _bullet(
                    f"The detected corners cover {coverage * 100:.0f} % of "
                    f"the frame. Below {COVERAGE_WARN_FRACTION * 100:.0f} %, "
                    "the movement near the image edges is not measured but "
                    "extended from the middle. Include photos in which the "
                    "board reaches the edges and corners of the frame.")
            elif reason == "model":
                lines += _bullet(
                    f"The model \"{model_label}\" was chosen. With this "
                    "model, an error of the board that is the same in every "
                    "photo can appear as lens distortion, and the ± does not "
                    f"show it. Calibrate again with \"{DIST_MODEL_DEFAULT}\" "
                    "to get an interpretation.")
            elif reason == "contradiction":
                richer = None
                if model_label in DIST_MODEL_ORDER:
                    k = DIST_MODEL_ORDER.index(model_label)
                    if k + 1 < len(DIST_MODEL_ORDER):
                        richer = DIST_MODEL_ORDER[k + 1]
                lines += _bullet(
                    f"The largest movement ({M:.2f} px) is smaller than "
                    f"{INTERPRET_NOT_SHOWN:g} times its ± "
                    f"({uncertainty['largest']:.2f} px), but the distortion "
                    f"model removes {percent:.0f} % of the misfit. These two "
                    "do not fit together; the model may be too simple for "
                    "this lens."
                    + (f" Try \"{richer}\"." if richer else ""))
        lines.append("The values in the Result block are valid as calculated.")
    else:
        se = uncertainty["largest"]
        lines.append("Does the lens distort the image?")
        if scenario == "not shown":
            bound = M + 2.0 * se
            bound_txt = f"{bound:.0f}" if bound >= 10 else f"{bound:.1f}"
            lines += _bullet(
                f"The largest movement ({M:.2f} px) is smaller than "
                f"{INTERPRET_NOT_SHOWN:g} times its ± ({se:.2f} px). These "
                "photos do not show a lens distortion.")
            lines += _bullet(
                "They also cannot rule one out. A distortion of up to about "
                f"{bound_txt} px{place} would not have been noticed (largest "
                "movement + 2 × its ±). For half of all pixels the ± is "
                f"{uncertainty['median']:.2f} px or less.")
        elif scenario == "indicated":
            lines += _bullet(
                f"The largest movement ({M:.2f} px) is between "
                f"{INTERPRET_NOT_SHOWN:g} and {INTERPRET_SHOWN:g} times its ± "
                f"({se:.2f} px). These photos point to a lens distortion of "
                f"about {M:.2f} px{place}, but do not fix it. Better photos "
                "are needed to decide.")
        else:
            lines += _bullet(
                f"The largest movement ({M:.2f} px) is at least "
                f"{INTERPRET_SHOWN:g} times its ± ({se:.2f} px). These photos "
                f"show a lens distortion of up to {M:.2f} px{place}, fixed to "
                f"about ± {se:.2f} px.")

        lines += ["", "Where does the misfit come from?"]
        if percent is not None:
            percent_txt = f"{percent:.0f}" if percent >= 10 else f"{percent:.2f}"
            lines += _bullet(
                f"The distortion model removes {percent_txt} % of the misfit. "
                f"The remaining {rms:.2f} px ({rms / W * 100:.3f} % of the "
                "image width) comes from the board or the photos, not from "
                "lens distortion that this model can describe.")
        if board is not None:
            lines += _bullet(
                f"On the board, the remaining misfit is {board['total']:.2f} "
                f"mm on average. {board['fixed']:.2f} mm of that is the same "
                "in every photo, which points to the pattern itself. "
                f"{board['changing']:.2f} mm changes from photo to photo, "
                "which points to the board bending, to blur, or to how "
                "exactly the corners were found.")

        lines += ["", f"What it means for your images ({W} × {H} px)"]
        lines += _bullet(
            f"If this file is applied, pixels move by at most {M:.2f} px "
            f"({M / W * 100:.2f} % of the image width), and half of all "
            f"pixels by no more than {movement['median']:.2f} px.")
        cm = f"{EXAMPLE_PIXEL_CM:g} cm"
        if scenario == "not shown":
            lines += _bullet(
                "Applying the file and not applying it are both consistent "
                "with these photos.")
            lines += _bullet(
                f"Example: if one pixel covers {cm} on the ground, {M:.2f} px "
                f"is {_example_cm(M)} and the {bound_txt} px that cannot be "
                f"ruled out is {_example_cm(float(bound_txt))}"
                + (f", both{place}" if place else "")
                + ". For half of all pixels the ± is "
                f"{_example_cm(uncertainty['median'])} or less.")
        elif scenario == "indicated":
            lines += _bullet(
                "Whether applying the file improves the images cannot be "
                "decided from these photos.")
            lines += _bullet(
                f"Example: if one pixel covers {cm} on the ground, {M:.2f} px "
                f"is {_example_cm(M)} and its ± of {se:.2f} px is "
                f"{_example_cm(se)}" + (f", both{place}" if place else "")
                + ".")
        else:
            lines += _bullet(
                f"Applying the file removes a lens distortion of up to "
                f"{M:.2f} px; about ± {se:.2f} px of it stays uncertain.")
            lines += _bullet(
                f"Example: if one pixel covers {cm} on the ground, {M:.2f} px "
                f"is {_example_cm(M)} and its ± of {se:.2f} px is "
                f"{_example_cm(se)}" + (f", both{place}" if place else "")
                + ".")
        lines += _bullet(
            "The grade limits are in pixels. A smaller image size gives a "
            "smaller RMS in pixels for the same board, so the grade can "
            "improve without a better calibration. The share of the image "
            f"width ({rms / W * 100:.3f} %) does not change with the image "
            "size; use it to compare calibrations.")

        lines += ["", "Limits of this interpretation"]
        lines += _bullet(
            f"The ± comes from repeating the calibration {uncertainty['n']} "
            "times, each time without one photo. An error that is the same "
            "in every photo, such as an inexact printed pattern, is not part "
            "of it. In tests with such an error, the true error was up to 4 "
            "times the ±.")
        lines += _bullet(
            "More photos make the ± smaller, but not an error of the board.")

    suggestions = []
    if warned_target:
        suggestions += _bullet(
            ("For a more exact result" if scenario == "shown"
             else "To find out whether this lens distorts")
            + ", use a pattern on a flat, stiff backing and check the "
            "printed cell size with a ruler.")
        suggestions += _bullet(
            "Turn the board by 90° and 180° in some of the photos and place "
            "it in all parts of the frame. A board error then no longer "
            "looks like lens distortion.")
    if warned_coverage and "coverage" not in reasons:
        suggestions += _bullet(
            "Include photos in which the board reaches the edges and "
            "corners of the frame.")
    # not next to a reason that already names the model to use
    if simpler_label and not {"model", "contradiction"} & set(reasons):
        suggestions += _bullet(
            f"Use \"{_short_label(simpler_label)}\", as the model guidance "
            "suggests.")
    if scenario != "none":
        suggestions += _bullet(
            "If you apply the file, check that straight lines near the image "
            "edges stay straight in the corrected images.")
    if suggestions:
        lines += ["", "Suggestions"] + suggestions
    return lines


def coefficient_lines(dist_coeffs, std):
    """Each distortion coefficient with its standard deviation (± 1σ).
    ``std`` is the intrinsics vector of cv2.calibrateCameraExtended:
    fx, fy, cx, cy, k1, k2, p1, p2, k3."""
    dvals = np.zeros(5)
    dr = np.asarray(dist_coeffs, dtype=np.float64).ravel()
    dvals[:min(5, dr.size)] = dr[:5]
    return [
        f"  k1 = {dvals[0]: .6f} ± {std[4]:.6f}",
        f"  k2 = {dvals[1]: .6f} ± {std[5]:.6f}",
        f"  p1 = {dvals[2]: .6f} ± {std[6]:.6f}",
        f"  p2 = {dvals[3]: .6f} ± {std[7]:.6f}",
        f"  k3 = {dvals[4]: .6f} ± {std[8]:.6f}",
        "  (fixed coefficients have sigma = 0)",
    ]


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
                           pattern_size, cell_w_m, progress=None):
    """cv2.calibrateCameraExtended plus quality diagnostics.

    Returns a dict with the calibration result, parameter standard
    deviations, true per-image RMS errors, target-planarity residuals,
    frame coverage, a GOOD/MARGINAL/POOR verdict, the pixel movement of
    the correction with its leave-one-photo-out ±, the RMS without a
    distortion model, the split of the remaining misfit on the board and
    a list of warning strings.  Pure computation — no GUI dependencies.

    ``progress(i, n)`` is passed on to ``movement_uncertainty``, which
    repeats the calibration once per photo.

    The verdict is not repeated as a warning: the Result block states it
    together with what it means (see ``result_lines``).  The warnings
    name the specific findings — target accuracy, frame coverage,
    outlying images, suspect coefficients.
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

    # What the correction does to an image, in pixels, and the misfit of
    # the same corners for a camera without distortion.  The second is one
    # more calibrateCamera run; best-effort, never fatal.
    movement = pixel_movement(camera_matrix, dist_coeffs, img_shape)
    try:
        rms_no_dist = no_distortion_rms(obj_points, img_points, img_shape)
    except Exception:                # noqa: BLE001 — an extra, not required
        rms_no_dist = None
    # How firmly the photos fix that movement: the calibration repeated
    # once per photo, each time without it.
    uncertainty = movement_uncertainty(obj_points, img_points, img_shape,
                                       flags, movement, progress)
    try:
        board = board_error_split(obj_points[0], img_points, rvecs, tvecs,
                                  camera_matrix, dist_coeffs)
    except Exception:                # noqa: BLE001 — an extra, not required
        board = None

    warnings = []

    finite_mm = [mm for _, mm in planarity if np.isfinite(mm)]
    warned_target = bool(finite_mm and max(finite_mm) > PLANARITY_WARN_MM)
    if warned_target:
        worst = int(np.nanargmax([mm for _, mm in planarity])) + 1
        # The value is a shift of the corners within the plane of the
        # board.  It cannot tell a sheet that is not flat from a pattern
        # that is not exact, so the warning names both.
        warnings.append(
            f"Target accuracy: the detected corners lie up to "
            f"~{max(finite_mm):.1f} mm from where a flat, exact pattern "
            f"would be seen (image {worst}). The calibration assumes such "
            "a pattern. Mount it on a flat, stiff backing (foam board / "
            "dibond / glass), check the printed cell size, and re-shoot.")

    warned_coverage = coverage < COVERAGE_WARN_FRACTION
    if warned_coverage:
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
    # The model that the guidance points to, when it advises a simpler one.
    simpler_label = None
    if guidance and ("prefer the next-simpler model" in guidance
                     or "prefer a simpler model" in guidance):
        for label, f in DIST_MODEL_OPTIONS.items():
            if f == flags and DIST_MODEL_ORDER.index(label) > 0:
                simpler_label = DIST_MODEL_ORDER[
                    DIST_MODEL_ORDER.index(label) - 1]

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
        "movement": movement,
        "rms_no_distortion": rms_no_dist,
        "uncertainty": uncertainty,
        "board_error": board,
        "warned_target": warned_target,
        "warned_coverage": warned_coverage,
        "simpler_label": simpler_label,
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
            self.input_label.configure(text=f"{short_path(d)}  ({n} images)",
                                       text_color=PATH_INPUT_COLOR)

    def _browse_output(self):
        d = filedialog.askdirectory(parent= self,title="Select Output Folder")
        if d:
            self.output_folder = d
            show_path(self.output_label, d, "output", empty="No output folder selected")

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
            show_path(self.input_label, None, "input")

        if self.output_folder:
            show_path(self.output_label, self.output_folder, "output", empty="No output folder selected")
        else:
            show_path(self.output_label, None, "output", empty="No output folder selected")

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
        show_path(self.input_label, None, "input")
        show_path(self.output_label, None, "output", empty="No output folder selected")
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

    def _finish_run(self, kind, title, message):
        """Closing pop-up of a finished run; afterwards the console is
        brought back to the Result block."""
        getattr(messagebox, f"show{kind}")(title, message, parent=self)
        # after the console has taken in the last lines (it polls every 50 ms)
        self.after(200, self._show_result_in_console)

    def _show_result_in_console(self):
        """Scroll the console so that the latest Result block starts at
        its first visible line.  The console follows the output to the
        last line, which would leave the result above the visible part."""
        try:
            idx = self.console_text.search("=== Result ===", tk.END,
                                           backwards=True)
            if idx:
                self.console_text.yview(idx)
        except tk.TclError:
            pass

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
            # Which images contributed nothing, and why.  The console says so
            # per image; the report kept only the count.
            no_corners = []

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
                    no_corners.append((os.path.basename(img_path), "unreadable"))
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
                    no_corners.append(
                        (os.path.basename(img_path),
                         f"size {gray.shape[::-1]} differs from {img_shape}"))
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
                    no_corners.append((os.path.basename(img_path),
                                       "checkerboard corners not detected"))

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
            if detected_count >= 4:
                print(f"It is then repeated {detected_count} times, each time "
                      "without one photo, for the ± values …")

            def pm_progress(i, n):
                if self._cancel_requested:
                    return False
                self._ui_progress(None, f"± run {i + 1}/{n}")
                return True

            diag = calibrate_and_diagnose(
                obj_points, img_points, img_shape, calib_flags,
                pattern_size, cell_w_m, progress=pm_progress)
            if self._cancel_requested:
                print("Calibration cancelled.")
                self._ui_progress(0, "ETA: Cancelled")
                return
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
            movement = diag["movement"]
            uncertainty = diag["uncertainty"]

            # Result first — what the correction does to an image, then how
            # well the checkerboard images support it.  The details follow.
            result_block = result_lines(
                model_label, img_shape, movement, ret, verdict,
                diag["rms_no_distortion"], len(q_warnings), uncertainty)
            print("\n" + "\n".join(result_block))
            print("An interpretation of these numbers is at the end of "
                  "calibration_report.txt.")
            # The console holds values and definitions; the reading of them
            # is kept to one marked section at the end of the report.
            interpretation = interpretation_lines(
                model_label, calib_flags, img_shape, movement, uncertainty,
                ret, diag["rms_no_distortion"], coverage, detected_count,
                board=diag["board_error"],
                warned_target=diag["warned_target"],
                warned_coverage=diag["warned_coverage"],
                simpler_label=diag["simpler_label"])

            fx = camera_matrix[0, 0]
            fy = camera_matrix[1, 1]
            cx = camera_matrix[0, 2]
            cy = camera_matrix[1, 2]
            W, H = img_shape
            print("\n=== Details ===")
            print(f"Distortion coefficients:\n{dist_coeffs.ravel()}")
            print("\n".join(coefficient_lines(dist_coeffs, std_i)))
            print(f"\nFocal length:  fx={fx:.2f} ± {std_i[0]:.2f} px,  "
                  f"fy={fy:.2f} ± {std_i[1]:.2f} px")
            print(f"Principal point:  cx={cx:.2f} ± {std_i[2]:.2f} px,  "
                  f"cy={cy:.2f} ± {std_i[3]:.2f} px")
            print(f"Image size: {W} × {H} px")
            print(f"Corner coverage of frame: {coverage * 100:.0f}%")
            print(f"\nCamera matrix:\n{camera_matrix}")

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
                if no_corners:
                    f.write(f"Images not used: {len(no_corners)}\n")
                    for nm, why in no_corners:
                        f.write(f"  - {nm}: {why}\n")
                f.write(f"Board: {sq_cols} × {sq_rows} squares\n")
                f.write(f"Inner corners: {n_cols} × {n_rows}\n")
                if is_square:
                    f.write(f"Cell size: {cell_w_mm} mm (square)\n")
                else:
                    f.write(f"Cell width: {cell_w_mm} mm\n")
                    f.write(f"Cell height: {cell_h_mm} mm\n")
                f.write(f"Distortion model: {model_label}\n\n")
                # the same Result block as in the console
                f.write("\n".join(result_block) + "\n\n")
                if q_warnings:
                    f.write("Warnings:\n")
                    for w in q_warnings:
                        f.write(f"  ! {w}\n")
                    f.write("\n")
                if guidance:
                    f.write(f"{guidance}\n\n")
                f.write("=== Details ===\n")
                f.write(f"Camera matrix:\n{camera_matrix}\n\n")
                f.write(f"Focal length: fx={fx:.2f} ± {std_i[0]:.2f} px, "
                        f"fy={fy:.2f} ± {std_i[1]:.2f} px\n")
                f.write(f"Principal point: cx={cx:.2f} ± {std_i[2]:.2f} px, "
                        f"cy={cy:.2f} ± {std_i[3]:.2f} px\n")
                f.write(f"Image size: {W} × {H} px\n")
                f.write(f"Corner coverage of frame: {coverage * 100:.0f}%\n\n")
                f.write(f"Distortion coefficients:\n{dist_coeffs.ravel()}\n")
                f.write("\n".join(coefficient_lines(dist_coeffs, std_i))
                        + "\n\n")
                f.write("Per-image diagnostics\n")
                f.write("  (RMS is the true per-image RMS, |residuals|/sqrt(N);\n")
                f.write("   plane-fit is the deviation of the detected grid "
                        "from a flat plane):\n")
                for i, err in enumerate(per_image_rms):
                    p_px, p_mm = planarity[i]
                    f.write(f"  Image {i + 1}: RMS {err:.4f} px | "
                            f"plane-fit {p_px:.2f} px (~{p_mm:.2f} mm)\n")
                f.write("\n" + "\n".join(interpretation) + "\n")
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
            summary = ("Calibration complete\n\n"
                       + "\n".join(headline_lines(movement, ret, verdict,
                                                 uncertainty))
                       + "\n")
            if q_warnings:
                summary += (f"\n{len(q_warnings)} warning(s) — "
                            "see console / report for details.\n")
            summary += ("\nAn interpretation of these numbers is at the end "
                        "of calibration_report.txt.\n")
            summary += f"\nFiles saved to:\n{output_folder}"
            self._ui_call(
                self._finish_run,
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