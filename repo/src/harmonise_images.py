"""
harmonise_images.py  —  GeoCamPal Harmonise Images
====================================================

Purpose
-------
Pre-processing pipeline for coastal camera image sequences.  Four
sub-tasks are available in a single window; they can be run
independently or in sequence as part of a workflow.

Sub-tasks
---------
1. Filter bad images
   Scans every image and assigns a pass/fail verdict using a
   combination of signal-quality metrics:

       blur             — Laplacian variance, with a uniform-scene
                          guard to avoid false positives on calm water
                          or overcast sky, and an optional block-wise
                          test for partial obstructions (frost, ice).
       clipped_overexposed — pixels saturated in all three channels,
                          distinguishing true overexposure from bright
                          natural features such as white sand.
       underexposed     — excess dark pixels relative to a threshold.
       too_dark         — maximum pixel value below a floor (nighttime,
                          camera failure).
       low_information  — image entropy below threshold; catches fog,
                          haze, and lens condensation.
       lens_droplets    — bright residuals from rain or droplets after
                          Gaussian smoothing.
       obstruction      — grid-based dark-block fraction to catch
                          foreign objects or bird on the housing.

   Failed images are listed in bad_images.txt and bad_images.json.
   The totals of the run (images checked, kept and rejected, the count
   and share per reason, the thresholds used, processing time) are
   written to filter_summary.txt and to the "summary" entry of
   bad_images.json.
   Passing images can optionally be copied to a _filtered_good/ folder.

   The image checks can be limited to an area of interest (AOI): a
   rectangle or a polygon drawn on one of the images.  They then look
   only at the pixels inside it, so a bright sky or a flare at the side
   of the frame does not decide whether an image is kept, and every
   fraction is a share of the AOI instead of the whole image.  The same
   AOI is used for all images of a run (it is scaled to images of
   another size) and is written to the summary.

   Progress is saved to filter_progress.jsonl in the output folder at
   least every 30 s and when the run is cancelled.  If that file is
   found when the filter is started, the run can be continued from the
   last save instead of starting again.  This is offered only when the
   GeoCamPal version, the sub-folder option, the filters and their
   thresholds, the area of interest and the list of images are the same
   as before, so a continued run gives the result of an uninterrupted
   one.  The file is removed when the run finishes.

2. Harmonise brightness
   Corrects inter-image luminance variation relative to a reference
   image or a target mean L value.  Two algorithms are available:

       Soft-gain match  — linear L-channel gain with a soft-knee rolloff
                          above a configurable threshold to prevent hard
                          highlight clipping.
       Histogram match  — full L-channel CDF matching via scikit-image;
                          falls back to soft-gain if skimage is absent.

   Both algorithms include a saturation-preservation step that restores
   the original HSV-S channel after the L correction, preventing colour
   washout on low-chroma scenes (coastal, overcast, dawn/dusk).
   A post-correction sanity check reverts any image whose corrected mean
   L is further from the reference than the original.
   An optional sky mask excludes the top N% of the image from the
   reference luminance calculation.
   Output goes to _brightness_harmonised/.

3. Harmonise colour
   Transfers colour characteristics from a single reference image to
   all others.  Three algorithms are available:

       Reinhard (fast)         — per-channel LAB mean/std shift
                                 (Reinhard et al., 2001).  Fast;
                                 suitable for same-scene time-lapse.
       Histogram Match (LAB)   — per-channel LAB CDF matching via
                                 scikit-image.  Better for larger
                                 colour shifts.
       Iterative Transfer      — 3-D distribution transfer in LAB space
                                 via random rotations and 1-D histogram
                                 matching (Pitié et al., 2007).  Best
                                 quality; slowest.

   A post-correction sanity check reverts images whose mean LAB
   distance to the reference worsened after correction.
   Output goes to _colour_harmonised/.

4. Lens correction
   Applies camera lens undistortion using a calibration file (.pkl)
   produced by the Lens Correction tool.  K is scaled proportionally
   when the image dimensions differ from the calibration resolution
   (same sensor, different output size).  Rectification maps are cached
   per unique (W, H) to avoid redundant computation.  Images with an
   incompatible aspect ratio are skipped with a warning.
   Output goes to _lens_corrected/.

Preview system
--------------
Both harmonisation stages and lens correction generate a before/after
preview from a 5% random sample (minimum 1 image) before committing
to the full batch.  Results are shown as side-by-side pairs with
Next/Previous navigation.

TIFF metadata preservation
--------------------------
When processing georeferenced TIFFs, the CRS, affine transform, nodata
value, extra bands (alpha, NIR, masks), and TIFF tags are read from the
source file and written unchanged to the output.  Only the RGB channels
are modified.  Requires rasterio; plain 3-band TIFFs fall back to
cv2.imwrite.

Originals are never modified or overwritten.

Outputs  (written to the user-selected output folder)
-------
    bad_images.txt             — tab-separated filenames and reason codes
    bad_images.json            — structured list of {filename, reasons},
                                 with the run totals under "summary"
    filter_summary.txt         — totals of the filter run: images checked,
                                 kept, rejected, count and share per
                                 reason, thresholds, processing time
    filter_progress.jsonl      — saved progress of a filter run that has
                                 not finished (removed when it finishes)
    _filtered_good/            — images that passed all filters (optional)
    _brightness_harmonised/    — brightness-corrected images
    _colour_harmonised/        — colour-corrected images
    _lens_corrected/           — lens-undistorted images

Dependencies
------------
    numpy, opencv-python (cv2), matplotlib, customtkinter,
    scikit-image (optional, for histogram matching),
    rasterio (optional, for GeoTIFF metadata preservation)
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
import hashlib
import math
import pickle
from pathlib import Path

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
    show_path,
    fit_geometry,
    resource_path,
    setup_console,
    restore_console,
    save_settings_json,
    load_settings_json,
    imread_safe,
    load_lens_calibration,
    apply_window_icon,
    bring_child_to_front,
    geometry_px,
    DARK_CANVAS_BG,
    __version__ as GEOCAMPAL_VERSION,
)

try:
    from skimage import exposure as sk_exposure
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# %% ————————————————————————————— TIFF preservation helpers ————————————

def _read_tiff_metadata(path):
    """
    Read metadata from a TIFF file that should be preserved through
    harmonisation.

    Returns a dict when the file is a TIF **and** carries data beyond a
    plain 3‑band RGB raster (geo‑referencing, extra bands, tags, etc.).
    Returns ``None`` for non‑TIF formats, plain 3‑band TIFs without any
    geo info, or when rasterio is unavailable — in those cases the caller
    falls back to ``cv2.imwrite`` which is perfectly adequate.

    Checked metadata:
      • CRS and affine transform  (geo‑referencing)
      • Band count               (alpha, masks, NIR, …)
      • Nodata value
      • TIFF tags                 (AREA_OR_POINT, etc.)
      • Compression scheme
      • Per‑band colour interpretation
    """
    if not HAS_RASTERIO:
        return None
    ext = Path(path).suffix.lower()
    if ext not in {".tif", ".tiff"}:
        return None
    try:
        with rasterio.open(str(path)) as src:
            has_geo = not (src.crs is None
                          and src.transform == rasterio.transform.Affine.identity())
            has_extra_bands = src.count > 3

            if not has_geo and not has_extra_bands:
                return None  # plain 3‑band TIFF, nothing special to keep

            return {
                "crs": src.crs,
                "transform": src.transform,
                "nodata": src.nodata,
                "count": src.count,
                "dtype": src.dtypes[0],
                "tags": src.tags(),
                "compress": src.profile.get("compress", "deflate"),
                "colorinterp": [ci.name for ci in src.colorinterp],
                "has_geo": has_geo,
            }
    except Exception:
        return None


def _save_preserving_tiff(out_path, img_bgr, tiff_meta, source_path=None):
    """
    Save *img_bgr* (OpenCV BGR uint8 array) to *out_path*, preserving
    any TIFF metadata and extra bands from the original source file.

    Harmonisation only modifies the RGB channels.  Everything else in the
    source file — alpha, masks, NIR bands, CRS, affine transform, nodata,
    TIFF tags — is read back from *source_path* and written unchanged.

    When *tiff_meta* is ``None`` (non‑TIF input, or a plain 3‑band TIF
    with no geo info), this falls back to ``cv2.imwrite`` — identical to
    the original behaviour.

    Parameters
    ----------
    out_path : Path or str
        Destination file path.
    img_bgr : ndarray
        The harmonised image in OpenCV BGR uint8 format (H, W, 3).
    tiff_meta : dict or None
        Metadata dict from ``_read_tiff_metadata()``, or ``None``.
    source_path : Path or str or None
        Original source file.  Required when the source had extra bands
        (band 4+) so they can be read back and re‑attached.
    """
    if tiff_meta is None or not HAS_RASTERIO:
        cv2.imwrite(str(out_path), img_bgr)
        return

    h, w = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # ── recover extra bands (4+) from the source file ──────────────
    orig_band_count = tiff_meta["count"]
    extra_bands = []  # list of 2‑D arrays, one per extra band
    if orig_band_count > 3 and source_path is not None:
        try:
            with rasterio.open(str(source_path)) as src:
                for b in range(4, orig_band_count + 1):  # 1‑indexed
                    extra_bands.append(src.read(b))
        except Exception:
            extra_bands = []

    out_count = 3 + len(extra_bands)

    # ── build output profile ───────────────────────────────────────
    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": out_count,
        "dtype": "uint8",
        "compress": tiff_meta.get("compress", "deflate"),
    }
    if tiff_meta.get("has_geo"):
        profile["crs"] = tiff_meta["crs"]
        profile["transform"] = tiff_meta["transform"]
    if tiff_meta.get("nodata") is not None:
        profile["nodata"] = tiff_meta["nodata"]

    # ── write ──────────────────────────────────────────────────────
    with rasterio.open(str(out_path), "w", **profile) as dst:
        dst.write(rgb[:, :, 0], 1)
        dst.write(rgb[:, :, 1], 2)
        dst.write(rgb[:, :, 2], 3)
        for i, band in enumerate(extra_bands):
            dst.write(band, 4 + i)
        tags = tiff_meta.get("tags")
        if tags:
            dst.update_tags(**tags)

# %% ————————————————————————————— area of interest for the filter —————
#
# The filter can be limited to an area of interest (AOI): a rectangle or
# a polygon drawn on one image.  The checks then look only at the pixels
# inside it, so a bright sky or a flare at the side of the frame no longer
# decides whether an image is kept.
#
# A rectangle is cut out of the image and the checks run on the cut-out,
# exactly as they run on a whole image.  For a polygon its bounding box is
# cut out and a mask says which pixels of the cut-out lie inside the
# polygon; the checks then count only those pixels.  Their filters (3x3
# Laplacian and Sobel, 31x31 Gaussian) are computed on the cut-out, so the
# outermost few pixels inside a polygon edge still see their neighbours on
# the other side of it.

AOI_SHAPES = ("rectangle", "polygon")
AOI_MIN_SIDE = 64            # smallest width and height of an AOI, in pixels
AOI_BLOCK_MIN_SHARE = 0.10   # share of a grid block that must lie inside a
                             # polygon for the block to count


def _aoi_points(aoi, width, height):
    """Corner points of the AOI as integer pixels of a width x height image.

    Returns (points, scaled).  The AOI is scaled with the image when the
    image has another size than the one the AOI was drawn on.
    """
    ref_w, ref_h = aoi["image_size"]
    pts = np.asarray(aoi["points"], dtype=np.float64).reshape(-1, 2)
    scaled = (int(width), int(height)) != (int(ref_w), int(ref_h))
    if scaled:
        pts = pts * [(width - 1) / max(ref_w - 1, 1), (height - 1) / max(ref_h - 1, 1)]
    pts = np.rint(pts).astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
    return pts, scaled


def aoi_region(aoi, width, height):
    """Where the AOI lies in an image of width x height pixels.

    Returns (x0, y0, x1, y1, mask, scaled).  The checks run on
    img[y0:y1, x0:x1].  mask is None for a rectangle; for a polygon it is a
    bool array of the size of that cut-out, True inside the polygon (None
    as well when the polygon fills its bounding box).  scaled says whether
    the AOI had to be scaled because the image has another size than the
    one it was drawn on.
    """
    pts, scaled = _aoi_points(aoi, width, height)
    x0, y0 = (int(v) for v in pts.min(axis=0))
    x1, y1 = (int(v) + 1 for v in pts.max(axis=0))
    if scaled:
        # An image much smaller than the one the AOI was drawn on: keep at
        # least a 16 x 16 px cut-out so the block-wise checks have pixels.
        if x1 - x0 < 16:
            x0 = max(0, min(x0, width - 16))
            x1 = min(width, x0 + 16)
        if y1 - y0 < 16:
            y0 = max(0, min(y0, height - 16))
            y1 = min(height, y0 + 16)
    mask = None
    if aoi["shape"] == "polygon":
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillPoly(m, [(pts - np.array([x0, y0])).astype(np.int32)], 1)
        if m.any() and not m.all():
            mask = m.astype(bool)
    return x0, y0, x1, y1, mask, scaled


def _grid_share(gray, mask, grid_size, is_bad):
    """Share of a grid_size x grid_size grid of blocks for which
    is_bad(block, block_mask) is true.

    Without a mask (whole image or rectangular AOI) this is the number of
    such blocks divided by the number of blocks.  With a mask (polygon AOI)
    each block counts with the number of its pixels inside the polygon, and
    a block with less than AOI_BLOCK_MIN_SHARE of its pixels inside is left
    out; the result is then the share of the polygon that lies in such
    blocks.  Returns None when no block counts.
    """
    h, w = gray.shape
    bh, bw = h // grid_size, w // grid_size
    bad = total = 0
    for i in range(grid_size):
        for j in range(grid_size):
            rows = slice(i * bh, (i + 1) * bh)
            cols = slice(j * bw, (j + 1) * bw)
            if mask is None:
                weight, block_mask = 1, None
            else:
                block_mask = mask[rows, cols]
                weight = int(np.count_nonzero(block_mask))
                if weight == 0 or weight < AOI_BLOCK_MIN_SHARE * block_mask.size:
                    continue
            total += weight
            if is_bad(gray[rows, cols], block_mask):
                bad += weight
    return bad / total if total else None


def normalise_aoi(aoi):
    """Check an AOI record and return it in the form it is stored in.

    The record is a dict with
        shape       "rectangle" or "polygon"
        points      rectangle: two opposite corners [[x0, y0], [x1, y1]]
                    polygon:   three or more corners [[x, y], ...]
        image_size  [width, height] of the image the AOI was drawn on
    Coordinates are pixel positions in that image (x to the right, y down,
    first pixel 0); the corner pixels belong to the area.

    Raises ValueError with a message for the user when the record cannot
    be used.
    """
    if not isinstance(aoi, dict):
        raise ValueError("The area of interest is not a valid record.")
    shape = str(aoi.get("shape", "")).strip().lower()
    if shape not in AOI_SHAPES:
        raise ValueError("The area of interest must be a rectangle or a polygon.")
    try:
        width, height = (int(v) for v in aoi["image_size"])
        pts = [(int(round(float(x))), int(round(float(y)))) for x, y in aoi["points"]]
    except (KeyError, TypeError, ValueError):
        raise ValueError("The area of interest needs corner points and the size "
                         "of the image it was drawn on.") from None
    if width < 1 or height < 1:
        raise ValueError("The image size of the area of interest is not valid.")
    pts = [(max(0, min(x, width - 1)), max(0, min(y, height - 1))) for x, y in pts]

    if shape == "rectangle":
        if len(pts) != 2:
            raise ValueError("A rectangle needs two opposite corners.")
        (xa, ya), (xb, yb) = pts
        pts = [(min(xa, xb), min(ya, yb)), (max(xa, xb), max(ya, yb))]
    else:
        # drop corners that repeat the one before them (also last == first)
        distinct = []
        for p in pts:
            if not distinct or p != distinct[-1]:
                distinct.append(p)
        if len(distinct) > 1 and distinct[0] == distinct[-1]:
            distinct.pop()
        pts = distinct
        if len(pts) < 3:
            raise ValueError("A polygon needs at least three different corners.")

    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    if max(xs) - min(xs) + 1 < AOI_MIN_SIDE or max(ys) - min(ys) + 1 < AOI_MIN_SIDE:
        raise ValueError(f"The area of interest is too small. It has to be at least "
                         f"{AOI_MIN_SIDE} x {AOI_MIN_SIDE} pixels.")

    record = {"shape": shape, "points": [[x, y] for x, y in pts],
              "image_size": [width, height]}
    if shape == "polygon":
        x0, y0, x1, y1, mask, _ = aoi_region(record, width, height)
        if mask is not None:
            # The Blur and Obstruction checks work on a grid of blocks.  A
            # polygon that is very narrow leaves most of its pixels in blocks
            # that do not count, and those checks would then say little.
            inside = int(np.count_nonzero(mask))
            gray = np.zeros(mask.shape, np.uint8)
            for grid_size in (4, 8):
                counted = [0]

                def count(block, block_mask):
                    counted[0] += int(np.count_nonzero(block_mask))
                    return False

                _grid_share(gray, mask, grid_size, count)
                if counted[0] < 0.5 * inside:
                    raise ValueError("The polygon is too narrow for the block-wise checks "
                                     "(Blur, Obstruction). Draw a wider area.")
    return record


def aoi_share_of_image(aoi):
    """Share (0-1) of the image the AOI was drawn on that lies inside the AOI."""
    width, height = aoi["image_size"]
    x0, y0, x1, y1, mask, _ = aoi_region(aoi, width, height)
    inside = (x1 - x0) * (y1 - y0) if mask is None else int(np.count_nonzero(mask))
    return inside / float(width * height)


def describe_aoi(aoi):
    """One line that says where the AOI is (console, summary file, window)."""
    width, height = aoi["image_size"]
    if aoi["shape"] == "rectangle":
        (x0, y0), (x1, y1) = aoi["points"]
        where = (f"rectangle, x {x0}-{x1}, y {y0}-{y1} "
                 f"({x1 - x0 + 1} x {y1 - y0 + 1} px)")
    else:
        where = f"polygon with {len(aoi['points'])} corners"
    return (f"{where}, {100.0 * aoi_share_of_image(aoi):.1f} % of a "
            f"{width} x {height} px image")


def iter_images(folder, recursive=False):
    """Yield the image files of a folder one by one, without listing them all first."""
    if recursive:
        for root, dirs, files in os.walk(folder):
            dirs.sort()
            for f in sorted(files):
                if Path(f).suffix.lower() in IMAGE_EXTS:
                    yield Path(root) / f
    else:
        with os.scandir(folder) as entries:
            for entry in entries:
                if entry.is_file() and Path(entry.name).suffix.lower() in IMAGE_EXTS:
                    yield Path(entry.path)


# %% ————————————————————————————— image quality filters ———————————————
#
# Every check takes an optional `mask` (bool array, True inside a polygon
# area of interest, see above).  Without it the whole array counts.

def is_blurry(img_bgr, lap_threshold, uniform_fraction=0.85,
              local_blur_frac=0.0, mask=None):
    """
    Robust blur detector.
    Uses Laplacian variance, but guards against false positives on
    genuinely textureless scenes (calm water, overcast sky) by checking
    whether the image is mostly uniform *before* flagging it as blurry.

    Three detection stages (any one triggers a positive):
      1. **Hard floor** — if global Laplacian variance is below one-third
         of ``lap_threshold``, the image is flagged unconditionally.
         Even the most featureless real-world scenes (calm water,
         overcast sky) produce variance > ~15; values below 10 indicate
         a physically obstructed or completely defocused lens.
      2. **Standard check** — global Laplacian variance below threshold
         AND the scene is *not* mostly uniform (Sobel gradient check).
      3. **Local blur fraction** — when the uniform guard *would*
         override the standard check, a secondary block-wise test
         divides the image into an 8×8 grid and measures what fraction
         of blocks are individually blurry.  Partial obstructions
         (frost, ice, condensation covering part of the lens) produce
         60–100 % blurry blocks, while genuine uniform scenes (water,
         sky) typically stay below 25 %.  Set ``local_blur_frac`` to
         0.0 to disable (default, backward-compatible).

    With a ``mask`` every stage looks only at the pixels inside it; the
    block-wise test then counts the blocks as described in _grid_share.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if mask is None:
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    else:
        lap_var = cv2.Laplacian(gray, cv2.CV_64F)[mask].var()
    if lap_var < lap_threshold:
        # Stage 1: hard floor — extremely low variance is always blur
        if lap_var < lap_threshold * 0.33:
            return True
        # Stage 2: check if the scene is genuinely textureless (not blurry)
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
        if mask is None:
            low_gradient_frac = np.sum(np.abs(sobel) < 5) / sobel.size
        else:
            low_gradient_frac = (np.sum(np.abs(sobel[mask]) < 5)
                                 / np.count_nonzero(mask))
        if low_gradient_frac > uniform_fraction:
            # Stage 3: local blur fraction — catch partial obstructions
            # that look "uniform" globally but have many individually
            # blurry blocks (frost, ice, condensation on lens)
            if local_blur_frac > 0:
                def block_is_blurry(block, block_mask):
                    block_lap = cv2.Laplacian(block, cv2.CV_64F)
                    if block_mask is not None:
                        block_lap = block_lap[block_mask]
                    return block_lap.var() < lap_threshold

                blurry_share = _grid_share(gray, mask, 8, block_is_blurry)
                if blurry_share is not None and blurry_share > local_blur_frac:
                    return True
            return False  # uniform scene, not actually blurry
        return True
    return False


def is_clipped_overexposed(img_bgr, clip_fraction=0.15, clip_value=254,
                           mask=None):
    """
    Saturation‑based overexposure: checks for pixels clipped at or above
    clip_value across ALL three channels (no recoverable information),
    rather than just "bright" single‑channel pixels.  This avoids false
    positives on naturally bright scenes like white sand.
    """
    saturated = np.all(img_bgr >= clip_value, axis=2)
    if mask is not None:
        return (np.count_nonzero(saturated[mask])
                / np.count_nonzero(mask)) > clip_fraction
    return (saturated.sum() / saturated.size) > clip_fraction


def is_underexposed(img_bgr, dark_px_value, fraction, mask=None):
    """Too many dark pixels (single‑channel check on grayscale)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if mask is not None:
        return (np.count_nonzero(gray[mask] <= dark_px_value)
                / np.count_nonzero(mask)) > fraction
    n_dark = np.sum(gray <= dark_px_value)
    return (n_dark / gray.size) > fraction


def is_too_dark(img_bgr, max_threshold, mask=None):
    """Entire image maximum below threshold → near-black / nighttime."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if mask is not None:
        return int(gray[mask].max()) < max_threshold
    return int(gray.max()) < max_threshold


def is_low_information(img_bgr, entropy_threshold=5.0, mask=None):
    """
    Entropy‑based low‑information detector.  Catches fog, haze,
    washed‑out images, and lens condensation better than simple std.
    Typical good outdoor images have entropy 6–7.5; fog/haze drops
    below 5.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hist_mask = None if mask is None else mask.astype(np.uint8)
    hist = cv2.calcHist([gray], [0], hist_mask, [256], [0, 256]).ravel()
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    return entropy < entropy_threshold


def has_lens_droplets(img_bgr, blob_fraction=0.08, mask=None):
    """
    Rain / condensation droplet detector.
    Droplets create localised bright blobs surrounded by blur halos.
    We detect them as bright residuals after heavy Gaussian smoothing.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (31, 31), 0)
    diff = cv2.absdiff(gray, blurred)
    # droplets produce concentrated bright spots in the diff image
    if mask is not None:
        return (np.count_nonzero(diff[mask] > 40)
                / np.count_nonzero(mask)) > blob_fraction
    bright_local = np.sum(diff > 40) / diff.size
    return bright_local > blob_fraction


def has_obstruction(img_bgr, grid_size=4, dark_mean_thresh=15,
                    block_fraction=0.35, mask=None):
    """
    Partial obstruction detector (bird on housing, lens cap edge,
    foreign object).  Divides the image into a grid and checks whether
    a large contiguous region is near‑black.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    def block_is_dark(block, block_mask):
        values = block if block_mask is None else block[block_mask]
        return values.mean() < dark_mean_thresh

    dark_share = _grid_share(gray, mask, grid_size, block_is_dark)
    return dark_share is not None and dark_share > block_fraction


def filter_image(img_bgr,
                 blur_thresh,
                 clip_frac,
                 underexp_val, underexp_frac,
                 dark_thresh,
                 entropy_thresh,
                 droplet_frac,
                 obstruction_frac,
                 overexp_px=254,
                 local_blur_frac=0.0,
                 mask=None):
    """
    Run all filters.  Returns a list of reason strings; empty = passes.
    """
    reasons = []
    if is_blurry(img_bgr, blur_thresh, local_blur_frac=local_blur_frac, mask=mask):
        reasons.append("blurry")
    if is_clipped_overexposed(img_bgr, clip_frac, overexp_px, mask=mask):
        reasons.append("clipped_overexposed")
    if is_underexposed(img_bgr, underexp_val, underexp_frac, mask=mask):
        reasons.append("underexposed")
    if is_too_dark(img_bgr, dark_thresh, mask=mask):
        reasons.append("too_dark")
    if is_low_information(img_bgr, entropy_thresh, mask=mask):
        reasons.append("low_information")
    if has_lens_droplets(img_bgr, droplet_frac, mask=mask):
        reasons.append("lens_droplets")
    if has_obstruction(img_bgr, block_fraction=obstruction_frac, mask=mask):
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


def collect_images_recursive(folder, progress_cb=None, cancel_cb=None):
    """Collect image paths from a folder and all sub-folders.

    progress_cb(n_dirs, n_imgs) is called periodically during the walk so the
    caller can show feedback (the crawl itself can take minutes on large or
    networked trees).  cancel_cb() returning True aborts the walk early and
    returns whatever was found so far.  Both are optional and preserve the
    original behaviour when omitted.
    """
    results = []
    for n_dirs, (root, _, files) in enumerate(os.walk(folder), 1):
        if cancel_cb is not None and cancel_cb():
            break
        for f in sorted(files):
            if Path(f).suffix.lower() in IMAGE_EXTS:
                results.append(Path(root) / f)
        if progress_cb is not None and n_dirs % 200 == 0:
            progress_cb(n_dirs, len(results))
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


# %% ————————————————————————————— filter run records ——————————————————
#
# A filter run over a large archive takes many hours.  Its results are
# therefore saved while it runs (FILTER_PROGRESS_FILE in the output
# folder) so that a cancelled or crashed run can be continued, and the
# totals of a finished run are written to FILTER_SUMMARY_FILE.

FILTER_PROGRESS_FILE = "filter_progress.jsonl"
FILTER_SUMMARY_FILE = "filter_summary.txt"
FILTER_CHECKPOINT_SECONDS = 30    # progress is saved at least this often ...
FILTER_CHECKPOINT_IMAGES = 2000   # ... or after this many images, whichever comes first

# filter name in the window -> reason code written to the bad-image list
FILTER_REASONS = {
    "Blur": "blurry",
    "Overexposure": "clipped_overexposed",
    "Underexposure": "underexposed",
    "Darkness": "too_dark",
    "Low information": "low_information",
    "Lens droplets": "lens_droplets",
    "Obstruction": "obstruction",
    "Failed images": "failed_image",
}


def filter_settings_record(cfg):
    """The enabled filters with the thresholds they use.

    Written to the summary, and compared with an unfinished run to decide
    whether that run can be continued.
    """
    en = cfg.get("enabled", {})
    rec = {}
    if en.get("Blur"):
        rec["Blur"] = {"threshold": cfg["blur_t"],
                       "local_blur_fraction": cfg.get("local_blur_f", 0.0)}
    if en.get("Overexposure"):
        rec["Overexposure"] = {"fraction": cfg["clip_f"],
                               "pixel_value": cfg.get("overexp_px", 254)}
    if en.get("Underexposure"):
        rec["Underexposure"] = {"dark_pixel_value": cfg["dark_v"],
                                "fraction": cfg["under_f"]}
    if en.get("Darkness"):
        rec["Darkness"] = {"max_threshold": cfg["dark_max"]}
    if en.get("Low information"):
        rec["Low information"] = {"entropy_threshold": cfg["entropy_t"]}
    if en.get("Lens droplets"):
        rec["Lens droplets"] = {"fraction": cfg["droplet_f"]}
    if en.get("Obstruction"):
        rec["Obstruction"] = {"fraction": cfg["obstruct_f"]}
    if en.get("Failed images"):
        rec["Failed images"] = {"min_percent_of_median_size": cfg["failed_min_pct"]}
    return rec


def image_list_fingerprint(images, input_root):
    """SHA-1 of the image paths relative to the input folder, in processing order.

    Two runs with the same fingerprint check the same images in the same
    order, also when the drive letter of the input folder has changed.
    """
    root = str(Path(input_root))
    h = hashlib.sha1()
    for p in images:
        s = str(p)
        rel = s[len(root):] if s.startswith(root) else s
        h.update(rel.replace("\\", "/").lstrip("/").encode("utf-8", "surrogatepass"))
        h.update(b"\n")
    return h.hexdigest()


def read_filter_progress(path):
    """Read the saved progress of an unfinished filter run.

    Returns None when there is no usable file, otherwise a dict with the
    header, the number of images checked, the processing time so far, the
    counts per reason and the rejected images as (index, reasons) pairs.
    Lines that were cut off while being written are ignored, so the result
    always ends at a complete save.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            header = json.loads(f.readline())
            if not isinstance(header, dict) or header.get("geocampal_filter_progress") != 1:
                return None
            state = {
                "header": header,
                "checked": 0,
                "elapsed_s": 0.0,
                "counts": {},
                "aoi_scaled": 0,
                "saved": str(header.get("started", "")),
            }
            bad = {}
            for line in f:
                try:
                    rec = json.loads(line)
                    checked = int(rec["checked"])
                    elapsed = float(rec.get("elapsed_s", state["elapsed_s"]))
                    counts = {str(k): int(v) for k, v in rec.get("counts", {}).items()}
                    aoi_scaled = int(rec.get("aoi_scaled", state["aoi_scaled"]))
                    entries = [(int(i), [str(r) for r in reasons])
                               for i, reasons in rec.get("bad", [])]
                except (ValueError, KeyError, TypeError, AttributeError):
                    continue
                if checked < state["checked"] or any(i < 0 or i >= checked for i, _ in entries):
                    continue
                state["checked"] = checked
                state["elapsed_s"] = elapsed
                state["counts"] = counts
                state["aoi_scaled"] = aoi_scaled
                state["saved"] = str(rec.get("saved", state["saved"]))
                bad.update(entries)
            state["bad"] = sorted(bad.items())
            return state
    except (OSError, ValueError, TypeError):
        return None


def _checkpoint_record(checked, elapsed_s, counts, bad, aoi_scaled=0):
    rec = {
        "checked": int(checked),
        "elapsed_s": round(float(elapsed_s), 1),
        "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
        "counts": dict(counts),
        "bad": [[int(i), list(reasons)] for i, reasons in bad],
    }
    if aoi_scaled:
        # images so far whose size differed from the one the AOI was drawn on
        rec["aoi_scaled"] = int(aoi_scaled)
    return rec


def start_filter_progress(path, header, state=None):
    """Create the progress file with its header.

    When an unfinished run is continued, `state` (from read_filter_progress)
    is written back as one record, so the new file starts where the old one
    ended.  The file is replaced in one step and is never left half written.
    """
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        # Plain-ASCII JSON, so any folder name can be written.
        f.write(json.dumps(header) + "\n")
        if state is not None:
            rec = _checkpoint_record(state["checked"], state["elapsed_s"],
                                     state["counts"], state["bad"],
                                     state.get("aoi_scaled", 0))
            rec["saved"] = state.get("saved", rec["saved"])
            f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_filter_checkpoint(path, checked, elapsed_s, counts, new_bad, aoi_scaled=0):
    """Add one save to the progress file: the images checked so far and the
    images rejected since the previous save."""
    rec = _checkpoint_record(checked, elapsed_s, counts, new_bad, aoi_scaled)
    with open(path, "a", encoding="utf-8") as f:
        # The leading newline keeps this record on its own line even if an
        # earlier save was cut off part-way.
        f.write("\n" + json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())


def filter_resume_obstacle(header, cfg, version):
    """Why an unfinished run cannot be continued with these settings ('' if it can)."""
    old_version = header.get("geocampal_version")
    if old_version != version:
        return f"it was made with GeoCamPal {old_version} and this is version {version}"
    if bool(header.get("recursive")) != bool(cfg["recursive"]):
        return "the 'Include sub-folders' option was different"
    if header.get("filters") != json.loads(json.dumps(filter_settings_record(cfg))):
        return "the selected filters or their thresholds were different"
    if header.get("aoi") != json.loads(json.dumps(cfg.get("aoi"))):
        return "the area of interest (AOI) was different"
    return ""


def build_filter_summary(cfg, n_images, n_bad, reason_counts, started, finished,
                         elapsed_s, sessions, version, aoi_scaled=0):
    """Totals of a finished filter run (also stored in bad_images.json).

    aoi_scaled: number of images whose size differed from the one the area
    of interest was drawn on (the area was scaled to them).
    """
    def pct(n):
        return round(100.0 * n / n_images, 2) if n_images else 0.0

    filters = filter_settings_record(cfg)
    # Reasons of the filters that ran, plus anything else that was counted
    # (unreadable files are rejected as failed_image even with that filter off).
    reasons = [FILTER_REASONS[name] for name in filters if name in FILTER_REASONS]
    reasons += [r for r, c in reason_counts.items() if c > 0 and r not in reasons]
    aoi = cfg.get("aoi")
    if aoi is not None:
        aoi = {
            "shape": aoi["shape"],
            "points": [list(p) for p in aoi["points"]],
            "image_size": list(aoi["image_size"]),
            "percent_of_image": round(100.0 * aoi_share_of_image(aoi), 1),
            "images_of_other_size": int(aoi_scaled),
        }
    return {
        "geocampal_version": version,
        "input_folder": cfg["input_folder"],
        "subfolders_included": bool(cfg["recursive"]),
        "area_of_interest": aoi,
        "started": started,
        "finished": finished,
        "processing_seconds": round(float(elapsed_s), 1),
        "images_per_second": round(n_images / elapsed_s, 2) if elapsed_s > 0 else None,
        "sessions": int(sessions),
        "images_checked": int(n_images),
        "images_kept": int(n_images - n_bad),
        "images_rejected": int(n_bad),
        "percent_rejected": pct(n_bad),
        "rejected_by_reason": {
            r: {"count": int(reason_counts.get(r, 0)),
                "percent_of_checked": pct(reason_counts.get(r, 0))}
            for r in reasons
        },
        "filters": filters,
        "filters_not_used": [name for name in FILTER_REASONS if name not in filters],
    }


def format_filter_summary(s):
    """The filter summary as plain text."""
    n = s["images_checked"]

    def pct(k):
        return f"{100.0 * k / n:.1f} %" if n else "n/a"

    rate = s.get("images_per_second")
    lines = [
        "GeoCamPal image filter summary",
        "==============================",
        "",
        f"GeoCamPal version : {s['geocampal_version']}",
        f"Input folder      : {s['input_folder']}",
        f"Sub-folders       : {'included' if s['subfolders_included'] else 'not included'}",
    ]
    aoi = s.get("area_of_interest")
    if aoi is None:
        lines.append("Area of interest  : whole image")
    else:
        lines.append(f"Area of interest  : {describe_aoi(aoi)}")
        if aoi["shape"] == "polygon":
            lines.append("                    corners (x, y): "
                         + ", ".join(f"({x}, {y})" for x, y in aoi["points"]))
        other = aoi.get("images_of_other_size", 0)
        if other:
            lines.append(f"                    {other} image{'s' if other != 1 else ''} had "
                         f"another size; the area was scaled to them")
    lines += [
        f"Started           : {s['started']}",
        f"Finished          : {s['finished']}",
        f"Processing time   : {format_eta(s['processing_seconds'])}"
        + (f"  ({rate:.1f} images/s)" if rate else ""),
    ]
    if s["sessions"] > 1:
        lines.append(f"Sessions          : {s['sessions']}  (interrupted and continued; "
                     f"the processing time is their sum)")
    lines += [
        "",
        f"Images checked    : {n}",
        f"Kept              : {s['images_kept']}  ({pct(s['images_kept'])})",
        f"Rejected          : {s['images_rejected']}  ({pct(s['images_rejected'])})",
        "",
        "Rejected by reason  (count, share of the images checked)",
    ]
    by_reason = s["rejected_by_reason"]
    width = max([len(str(v["count"])) for v in by_reason.values()] + [1])
    for reason, v in by_reason.items():
        lines.append(f"  {reason:<22s}{v['count']:>{width}d}  {pct(v['count']):>7s}")
    lines += [
        "  An image can have more than one reason, so these counts can add up",
        "  to more than the number of rejected images.",
        "",
        "Filters and thresholds",
    ]
    for name, params in s["filters"].items():
        txt = ", ".join(f"{k.replace('_', ' ')} = {v}" for k, v in params.items())
        lines.append(f"  {name:<18s}{txt}")
    if s["filters_not_used"]:
        lines.append(f"  {'Not used':<18s}{', '.join(s['filters_not_used'])}")
    if "Failed images" in s["filters_not_used"]:
        lines.append("  Files that cannot be read are rejected as failed_image even when")
        lines.append("  the Failed images filter is not used.")
    if aoi is not None:
        lines.append("  The image checks looked only at the area of interest, so their")
        lines.append("  fractions are shares of that area and not of the whole image.")
        if "Failed images" in s["filters"]:
            lines.append("  Failed images goes by file size and does not use the area.")
    return "\n".join(lines) + "\n"


# %% ————————————————————————————— AOI selector window —————————————————
class AOISelectorWindow(ctk.CTkToplevel):
    """Draw the area of interest (AOI) of the filter on one image.

    A rectangle is dragged.  A polygon is clicked corner by corner and
    closed with a double-click, Enter or Confirm.  The part of the image
    outside the area is dimmed.

    When the user confirms, on_done(aoi, image_path) is called with the
    record made by normalise_aoi and the image it was drawn on.  on_cancel()
    is called when the window is closed without confirming.
    """

    _MIN_ZOOM, _MAX_ZOOM, _ZOOM_STEP = 0.05, 20.0, 1.15
    _OUTLINE = "#FFD700"
    _HINTS = {
        "Rectangle": "Drag: draw the rectangle   |   Wheel: zoom   |   "
                     "Middle-drag: pan   |   Right-click: clear",
        "Polygon": "Click: add a corner   |   Double-click or Enter: close the polygon   |   "
                   "Right-click: remove the last corner   |   Wheel: zoom   |   Middle-drag: pan",
    }

    def __init__(self, master, image_path, on_done, aoi=None, initial_dir=None,
                 on_cancel=None):
        super().__init__(master)
        self.title("Harmonise Images — Select AOI")
        apply_window_icon(self)
        self.protocol("WM_DELETE_WINDOW", self._cancel)

        self._on_done = on_done
        self._on_cancel = on_cancel
        self._initial_dir = initial_dir
        self._image_path = None
        self._rgb = None
        self._iw = self._ih = 0
        self._zoom = 1.0
        self._ox = self._oy = 0.0      # image position shown at the canvas origin
        self._fitted = True            # the view still shows the whole image
        self._rect = None              # edges (left, top, right, bottom), image coordinates
        self._poly = []                # corner pixels [(x, y), ...]
        self._poly_closed = False
        self._drag_start = None
        self._pan_last = None
        self._tk_image = None
        self._view_cache = (None, None)
        self.shape_var = tk.StringVar(master=self, value="Rectangle")

        self.hint_label = ctk.CTkLabel(self, text="", font=("Arial", 11), anchor="w")
        self.hint_label.pack(side="top", fill="x", padx=8, pady=(6, 2))

        # The button bar is packed before the canvas so that the canvas gets
        # the space that is left and never pushes the buttons out of view.
        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.pack(side="bottom", fill="x", padx=6, pady=6)
        ctk.CTkButton(bar, text="Open Another Image", width=150,
                      fg_color="#4F5D75", hover_color="#61708A",
                      command=self._open_image).pack(side="left", padx=4)
        ctk.CTkLabel(bar, text="Shape:").pack(side="left", padx=(12, 3))
        ctk.CTkOptionMenu(bar, variable=self.shape_var,
                          values=["Rectangle", "Polygon"], width=120,
                          command=self._on_shape_change).pack(side="left", padx=3)
        ctk.CTkButton(bar, text="Clear", width=70,
                      fg_color="#555555", hover_color="#6A6A6A",
                      command=self._clear).pack(side="left", padx=(12, 4))
        ctk.CTkButton(bar, text="Confirm", width=110,
                      fg_color="#2E7D32", hover_color="#388E3C",
                      command=self._confirm).pack(side="right", padx=4)
        ctk.CTkButton(bar, text="Cancel", width=80,
                      fg_color="#8B0000", hover_color="#A52A2A",
                      command=self._cancel).pack(side="right", padx=4)
        self.status_label = ctk.CTkLabel(bar, text="No area selected",
                                         text_color="gray", anchor="w")
        self.status_label.pack(side="left", padx=10)

        self.canvas = tk.Canvas(self, bg=DARK_CANVAS_BG, cursor="crosshair",
                                highlightthickness=0)
        self.canvas.pack(side="top", fill="both", expand=True, padx=6)
        c = self.canvas
        c.bind("<ButtonPress-1>", self._press)
        c.bind("<B1-Motion>", self._drag)
        c.bind("<ButtonRelease-1>", self._release)
        c.bind("<Double-Button-1>", self._double)
        c.bind("<ButtonPress-2>", self._pan_start)
        c.bind("<B2-Motion>", self._pan_move)
        c.bind("<ButtonPress-3>", self._right)
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            c.bind(sequence, self._wheel)
        c.bind("<Configure>", self._on_resize)
        self.bind("<Return>", self._enter)
        self.bind("<Escape>", lambda e: self._cancel())

        loaded = self._load_image(image_path)
        if loaded and aoi is not None:
            self._show_record(aoi)
        self._on_shape_change(self.shape_var.get())

        screen_w, screen_h = self.winfo_screenwidth(), self.winfo_screenheight()
        iw, ih = (self._iw, self._ih) if loaded else (1200, 800)
        fit = min(0.80 * screen_w / iw, 0.70 * screen_h / ih, 1.0)
        win_w = int(min(max(iw * fit + 20, 780), 0.92 * screen_w))
        win_h = int(min(max(ih * fit + 110, 480), 0.88 * screen_h))
        geometry_px(self, win_w, win_h,
                    (screen_w - win_w) // 2, max(0, (screen_h - win_h) // 2))
        bring_child_to_front(self, master)
        if not loaded:
            self.status_label.configure(
                text="The image could not be read. Use 'Open Another Image'.",
                text_color="#E57373")

    # ——— image ———

    def _load_image(self, path):
        """Show another image; a shape that is already drawn is scaled to it."""
        bgr = imread_safe(str(path), verbose=False) if path else None
        if bgr is None:
            return False
        before = self._current_record(any_shape=True) if self._iw else []
        old_size, was_closed = (self._iw, self._ih), self._poly_closed
        self._rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self._ih, self._iw = self._rgb.shape[:2]
        self._image_path = str(path)
        self._view_cache = (None, None)
        self._fitted = True
        if old_size not in ((0, 0), (self._iw, self._ih)):
            self._rect, self._poly = None, []       # drawn for another image size
            for record in before:
                self._show_record(record, keep_shape_choice=True)
            self._poly_closed = was_closed and bool(self._poly)
        self._fit()
        self._redraw()
        self._update_status()
        return True

    def _open_image(self, path=None):
        if path is None:
            kwargs = {"parent": self, "title": "Open an image to draw the AOI on",
                      "filetypes": [("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                                    ("All files", "*.*")]}
            start = os.path.dirname(self._image_path) if self._image_path else self._initial_dir
            if start and os.path.isdir(start):
                kwargs["initialdir"] = start
            path = filedialog.askopenfilename(**kwargs)
        if not path:
            return
        if not self._load_image(path):
            messagebox.showerror("Select AOI", "The image could not be read.", parent=self)

    # ——— shapes and records ———

    def _current_record(self, any_shape=False):
        """The shape on screen as an AOI record that has not been checked yet.

        Returns None when nothing usable is drawn.  With any_shape=True it
        returns a list with the rectangle and the polygon that exist, whatever
        the Shape choice is.
        """
        size = [self._iw, self._ih]
        rect = poly = None
        if self._rect is not None:
            left, top, right, bottom = self._rect
            x0, y0 = int(math.floor(left)), int(math.floor(top))
            x1 = max(x0, int(math.ceil(right)) - 1)
            y1 = max(y0, int(math.ceil(bottom)) - 1)
            rect = {"shape": "rectangle", "points": [[x0, y0], [x1, y1]], "image_size": size}
        if len(self._poly) >= 3:
            poly = {"shape": "polygon", "points": [list(p) for p in self._poly],
                    "image_size": size}
        if any_shape:
            return [r for r in (rect, poly) if r is not None]
        return rect if self.shape_var.get() == "Rectangle" else poly

    def _show_record(self, aoi, keep_shape_choice=False):
        """Put an AOI record on screen, scaled to the image that is shown."""
        pts, _ = _aoi_points(aoi, self._iw, self._ih)
        pts = [(int(x), int(y)) for x, y in pts.tolist()]
        if aoi["shape"] == "rectangle":
            (xa, ya), (xb, yb) = pts
            self._rect = (min(xa, xb), min(ya, yb), max(xa, xb) + 1, max(ya, yb) + 1)
        else:
            self._poly, self._poly_closed = pts, True
        if not keep_shape_choice:
            self.shape_var.set("Rectangle" if aoi["shape"] == "rectangle" else "Polygon")

    def _outline(self):
        """Corners of the shape in image coordinates, and whether it is closed."""
        if self.shape_var.get() == "Rectangle":
            if self._rect is None:
                return [], False
            left, top, right, bottom = self._rect
            return [(left, top), (right, top), (right, bottom), (left, bottom)], True
        return [(x + 0.5, y + 0.5) for x, y in self._poly], self._poly_closed

    def _on_shape_change(self, value):
        self.hint_label.configure(text=self._HINTS.get(value, ""))
        self._drag_start = None
        self._redraw()
        self._update_status()

    def _clear(self):
        if self.shape_var.get() == "Rectangle":
            self._rect = None
        else:
            self._poly, self._poly_closed = [], False
        self._redraw()
        self._update_status()

    def _close_polygon(self):
        if len(self._poly) >= 3:
            self._poly_closed = True
            self._redraw()
            self._update_status()
        return self._poly_closed

    def _update_status(self):
        shape = self.shape_var.get()
        if shape == "Polygon" and self._poly and not self._poly_closed:
            n = len(self._poly)
            self.status_label.configure(
                text=f"Polygon: {n} corner{'s' if n != 1 else ''}, not closed yet",
                text_color="gray")
            return
        record = self._current_record()
        if record is None:
            self.status_label.configure(text="No area selected", text_color="gray")
            return
        try:
            text, colour = describe_aoi(normalise_aoi(record)), self._OUTLINE
        except ValueError as e:
            text, colour = str(e), "#E57373"
        self.status_label.configure(text=text[0].upper() + text[1:], text_color=colour)

    # ——— view ———

    def _to_canvas(self, x, y):
        return (x - self._ox) * self._zoom, (y - self._oy) * self._zoom

    def _to_image(self, cx, cy):
        return cx / self._zoom + self._ox, cy / self._zoom + self._oy

    def _clamped(self, x, y):
        return max(0.0, min(float(x), self._iw)), max(0.0, min(float(y), self._ih))

    def _fit(self):
        """Show the whole image, centred."""
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 2 or ch < 2 or not self._iw:
            return
        self._zoom = max(self._MIN_ZOOM, min(cw / self._iw, ch / self._ih, self._MAX_ZOOM))
        self._ox = (self._iw - cw / self._zoom) / 2.0
        self._oy = (self._ih - ch / self._zoom) / 2.0

    def _on_resize(self, _event=None):
        if self._fitted:
            self._fit()
        self._redraw()

    def _redraw(self):
        c = self.canvas
        c.delete("all")
        cw, ch = c.winfo_width(), c.winfo_height()
        if self._rgb is None or cw < 2 or ch < 2:
            return
        ax, ay = self._to_image(0, 0)
        bx, by = self._to_image(cw, ch)
        sx0, sy0 = max(0, int(math.floor(ax))), max(0, int(math.floor(ay)))
        sx1, sy1 = min(self._iw, int(math.ceil(bx))), min(self._ih, int(math.ceil(by)))
        if sx1 <= sx0 or sy1 <= sy0:
            return
        dw = max(1, int(round((sx1 - sx0) * self._zoom)))
        dh = max(1, int(round((sy1 - sy0) * self._zoom)))
        key = (sx0, sy0, sx1, sy1, dw, dh)
        if self._view_cache[0] != key:
            if self._zoom >= 2:
                interpolation = cv2.INTER_NEAREST
            elif self._zoom < 1:
                interpolation = cv2.INTER_AREA
            else:
                interpolation = cv2.INTER_LINEAR
            self._view_cache = (key, cv2.resize(self._rgb[sy0:sy1, sx0:sx1], (dw, dh),
                                                interpolation=interpolation))
        view = self._view_cache[1]

        corners, closed = self._outline()
        if closed and len(corners) >= 3:
            # dim everything outside the area
            inside = np.zeros((dh, dw), np.uint8)
            pts = np.array([[(x - sx0) * self._zoom, (y - sy0) * self._zoom]
                            for x, y in corners])
            cv2.fillPoly(inside, [np.rint(pts).astype(np.int32)], 1)
            view = np.where(inside[..., None] == 1, view,
                            cv2.convertScaleAbs(view, alpha=0.35))
        self._tk_image = ImageTk.PhotoImage(Image.fromarray(view), master=self)
        ox, oy = self._to_canvas(sx0, sy0)
        c.create_image(int(round(ox)), int(round(oy)), anchor="nw", image=self._tk_image)

        if corners:
            flat = [v for x, y in corners for v in self._to_canvas(x, y)]
            if closed and len(corners) >= 3:
                c.create_polygon(*flat, outline=self._OUTLINE, fill="", width=2)
            elif len(corners) >= 2:
                c.create_line(*flat, fill=self._OUTLINE, width=2)
            for x, y in corners:
                hx, hy = self._to_canvas(x, y)
                c.create_oval(hx - 4, hy - 4, hx + 4, hy + 4,
                              fill=self._OUTLINE, outline="#000000")

    # ——— mouse and keys ———

    def _press(self, event):
        if self._rgb is None:
            return
        x, y = self._clamped(*self._to_image(event.x, event.y))
        if self.shape_var.get() == "Rectangle":
            self._drag_start = (x, y)
            self._rect = None
        else:
            if self._poly_closed:          # a click on a finished polygon starts a new one
                self._poly, self._poly_closed = [], False
            corner = (min(int(x), self._iw - 1), min(int(y), self._ih - 1))
            if not self._poly or self._poly[-1] != corner:
                self._poly.append(corner)
        self._redraw()
        self._update_status()

    def _drag(self, event):
        if self._drag_start is None or self.shape_var.get() != "Rectangle":
            return
        x, y = self._clamped(*self._to_image(event.x, event.y))
        sx, sy = self._drag_start
        self._rect = (min(sx, x), min(sy, y), max(sx, x), max(sy, y))
        self._redraw()
        self._update_status()

    def _release(self, _event=None):
        if self._drag_start is None:
            return
        self._drag_start = None
        if self._rect is not None:
            left, top, right, bottom = self._rect
            if right - left < 2 or bottom - top < 2:
                self._rect = None          # a click without a drag
        self._redraw()
        self._update_status()

    def _double(self, _event=None):
        # The first click of the double-click has already added the corner.
        if self.shape_var.get() == "Polygon":
            self._close_polygon()

    def _right(self, _event=None):
        if self.shape_var.get() == "Rectangle":
            self._rect = None
        elif self._poly:
            self._poly.pop()
            self._poly_closed = False
        self._redraw()
        self._update_status()

    def _pan_start(self, event):
        self._pan_last = (event.x, event.y)

    def _pan_move(self, event):
        if self._pan_last is None:
            return
        self._ox -= (event.x - self._pan_last[0]) / self._zoom
        self._oy -= (event.y - self._pan_last[1]) / self._zoom
        self._pan_last = (event.x, event.y)
        self._fitted = False
        self._redraw()

    def _wheel(self, event):
        if self._rgb is None:
            return
        zoom_in = getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0
        factor = self._ZOOM_STEP if zoom_in else 1.0 / self._ZOOM_STEP
        zoom = max(self._MIN_ZOOM, min(self._MAX_ZOOM, self._zoom * factor))
        if zoom == self._zoom:
            return
        ix, iy = self._to_image(event.x, event.y)      # stays under the pointer
        self._zoom = zoom
        self._ox = ix - event.x / zoom
        self._oy = iy - event.y / zoom
        self._fitted = False
        self._redraw()

    def _enter(self, _event=None):
        if self.shape_var.get() == "Polygon" and not self._poly_closed:
            self._close_polygon()
        else:
            self._confirm()

    # ——— finish ———

    def _confirm(self):
        if self.shape_var.get() == "Polygon" and not self._poly_closed:
            if len(self._poly) < 3:
                messagebox.showwarning(
                    "Select AOI", "A polygon needs at least three corners.", parent=self)
                return
            self._close_polygon()
        record = self._current_record()
        if record is None:
            messagebox.showwarning(
                "Select AOI",
                "Draw the area first: drag a rectangle, or choose the shape "
                "Polygon and click its corners.", parent=self)
            return
        try:
            record = normalise_aoi(record)
        except ValueError as e:
            messagebox.showwarning("Select AOI", str(e), parent=self)
            return
        image_path = self._image_path
        self.destroy()
        self._on_done(record, image_path)

    def _cancel(self):
        self.destroy()
        if self._on_cancel is not None:
            self._on_cancel()


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

        # area of interest of the filter (see normalise_aoi)
        self.filter_aoi = None            # the AOI record, or None
        self.filter_aoi_image = None      # image it was drawn on
        self.aoi_enabled_var = tk.BooleanVar(value=False)
        self._aoi_selector = None

        # averaging state
        self.exclude_bad_avg_var = tk.BooleanVar(value=False)

        # lens correction state
        self.lens_calib_path = None
        self.lens_calib_data = None
        self.exclude_bad_lens_var = tk.BooleanVar(value=True)
        self.lens_crop_var = tk.BooleanVar(value=True)  # alpha=0 when checked

        # manually loaded bad-image JSON (shared across brightness/colour/average)
        self.loaded_bad_json_path = None

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
        self.run_lens_var = tk.BooleanVar(value=False)
        self.run_brightness_var = tk.BooleanVar(value=True)
        self.run_colour_var = tk.BooleanVar(value=True)
        self.run_average_var = tk.BooleanVar(value=False)

        ctk.CTkCheckBox(row1b, text="Filter bad images",
                        variable=self.run_filter_var,
                        command=self._toggle_task_sections).pack(side="left", padx=8)
        ctk.CTkCheckBox(row1b, text="Apply lens corr",
                        variable=self.run_lens_var,
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

        # Shared status label for manually loaded bad-image JSON
        self.bad_json_status_label = ctk.CTkLabel(
            row1b, text="", font=("Arial", 10), text_color="#2ECC71")
        self.bad_json_status_label.pack(side="left", padx=(15, 5))

        # Row 2 — filter parameters (collapsible)
        self.filter_section_frame = ctk.CTkFrame(self.bottom_panel)
        row2 = self.filter_section_frame

        ctk.CTkLabel(row2, text="Filter Bad Images",
                     font=("Arial", 12, "bold")).grid(
            row=0, column=0, padx=5, pady=3, sticky="w")

        # Area of interest: limit the image checks to a part of the frame
        aoi_row = ctk.CTkFrame(row2, fg_color="transparent")
        aoi_row.grid(row=0, column=1, columnspan=3, padx=(4, 5), pady=3, sticky="w")
        ctk.CTkCheckBox(aoi_row, text="Filter inside AOI only",
                        variable=self.aoi_enabled_var,
                        command=self._on_aoi_toggle).pack(side="left", padx=(2, 8))
        ctk.CTkButton(aoi_row, text="Select AOI",
                      command=self._select_filter_aoi,
                      width=90, height=24,
                      fg_color="#4F5D75", hover_color="#61708A").pack(side="left")
        self.aoi_status_label = ctk.CTkLabel(
            aoi_row, text="AOI: not set", font=("Arial", 10), text_color="gray")
        self.aoi_status_label.pack(side="left", padx=(10, 5))

        # Each filter gets one checkbox + labelled parameter entries
        self.filter_entries = {}
        self.filter_enabled = {}

        # --- helper: build a compact filter card (sub-frame) ---
        def _make_filter_card(parent, name, params, grid_row, grid_col):
            """Create a small framed card for one filter.

            *params* is a list of (label, entry_key, default) tuples.
            The checkbox and labelled entries are packed inside.
            """
            card = ctk.CTkFrame(parent, fg_color="transparent")
            card.grid(row=grid_row, column=grid_col, padx=(4,18), pady=4,
                      sticky="nsew")

            var = tk.BooleanVar(value=True)
            ctk.CTkCheckBox(card, text=name, variable=var,
                            font=("Arial", 12, "bold"),
                            width=20).grid(
                row=0, column=0, columnspan=2 * len(params),
                padx=2, pady=(2, 0), sticky="w")
            self.filter_enabled[name] = var

            for idx, (lbl, key, default) in enumerate(params):
                col_base = idx * 2
                ctk.CTkLabel(card, text=lbl,
                             font=("Arial", 10), text_color="gray").grid(
                    row=1, column=col_base, padx=(4, 1), pady=1,
                    sticky="w")
                e = ctk.CTkEntry(card, width=55)
                e.insert(0, default)
                e.grid(row=1, column=col_base + 1, padx=(1, 4), pady=1,
                       sticky="w")
                self.filter_entries[key] = e

        # ---- Row 1 of filter cards ----
        _make_filter_card(row2, "Blur", [
            ("Threshold", "Blur threshold", "20"),
            ("Local blur %", "Local blur fraction", "0.40"),
        ], grid_row=1, grid_col=0)

        _make_filter_card(row2, "Overexposure", [
            ("Fraction", "Clip overexp fraction", "0.15"),
            ("Pixel value", "Overexp pixel value", "254"),
        ], grid_row=1, grid_col=1)

        _make_filter_card(row2, "Underexposure", [
            ("Dark px val", "Dark pixel value", "20"),
            ("Fraction", "Under-exp fraction", "0.25"),
        ], grid_row=1, grid_col=2)

        _make_filter_card(row2, "Darkness", [
            ("Max threshold", "Dark max threshold", "25"),
        ], grid_row=1, grid_col=3)

        # ---- Row 2 of filter cards ----
        _make_filter_card(row2, "Low information", [
            ("Entropy", "Entropy threshold", "5.0"),
        ], grid_row=2, grid_col=0)

        _make_filter_card(row2, "Lens droplets", [
            ("Fraction", "Droplet fraction", "0.08"),
        ], grid_row=2, grid_col=1)

        _make_filter_card(row2, "Obstruction", [
            ("Fraction", "Obstruction fraction", "0.35"),
        ], grid_row=2, grid_col=2)

        _make_filter_card(row2, "Failed images", [
            ("% of median size", "Failed min size pct", "10"),
        ], grid_row=2, grid_col=3)

        # Even column weights so cards share space
        for c in range(4):
            row2.grid_columnconfigure(c, weight=0)

        # ---- Action row ----
        action_row = ctk.CTkFrame(row2, fg_color="transparent")
        action_row.grid(row=3, column=0, columnspan=4,
                        padx=5, pady=(6, 3), sticky="w")

        ctk.CTkButton(action_row, text="Run Filter",
                      command=self._filter_threaded,
                      fg_color="#0F52BA", hover_color="#2A6BD1").pack(
            side="left", padx=(0, 10))

        ctk.CTkCheckBox(
            action_row, text="Export good filtered images",
            variable=self.export_filtered_good_var
        ).pack(side="left", padx=(0, 10))

        ctk.CTkLabel(
            action_row,
            text="-> Flags blurry, overexposed, dark, foggy, "
                 "rain-on-lens, blocked & failed/corrupt images",
            font=("Arial", 10), text_color="gray", justify="left",
        ).pack(side="left", padx=5)

        # Row 2c — lens correction (collapsible)  [NEW]
        self.lens_section_frame = ctk.CTkFrame(self.bottom_panel)
        lens_row = self.lens_section_frame

        ctk.CTkLabel(lens_row, text="Lens Correction",
                     font=("Arial", 12, "bold")).grid(
            row=0, column=0, padx=5, pady=3, sticky="w")

        ctk.CTkButton(lens_row, text="Select Calibration (.pkl)",
                      command=self._browse_lens_calib, width=180).grid(
            row=0, column=1, padx=5, pady=3)
        self.lens_calib_label = ctk.CTkLabel(lens_row,
                                              text="No calibration selected")
        self.lens_calib_label.grid(row=0, column=2, padx=5, pady=3, sticky="w")

        ctk.CTkCheckBox(lens_row, text="Crop to valid pixels",
                        variable=self.lens_crop_var).grid(
            row=0, column=3, padx=10, pady=3)

        lens_excl_frame = ctk.CTkFrame(lens_row, fg_color="transparent")
        lens_excl_frame.grid(row=0, column=4, padx=5, pady=3)
        ctk.CTkCheckBox(lens_excl_frame, text="Exclude bad",
                        variable=self.exclude_bad_lens_var).pack(
            side="left", padx=(0, 4))
        ctk.CTkButton(lens_excl_frame, text="Load JSON",
                      command=self._browse_bad_json,
                      width=75, height=24,
                      fg_color="#4F5D75", hover_color="#61708A",
                      font=("Arial", 10)).pack(side="left")

        ctk.CTkButton(lens_row, text="Run Lens Correction",
                      command=self._lens_threaded,
                      fg_color="#0F52BA", hover_color="#2A6BD1").grid(
            row=0, column=5, padx=5, pady=3)

        ctk.CTkLabel(
            lens_row,
            text="-> Undistorts images using a lens_calibration.pkl\n"
                 "   produced by the Lens Correction module",
            font=("Arial", 10), text_color="gray", justify="left",
        ).grid(row=0, column=6, padx=5, pady=3, sticky="w")

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
        bri_excl_frame = ctk.CTkFrame(row3, fg_color="transparent")
        bri_excl_frame.grid(row=0, column=5, padx=5, pady=3)
        ctk.CTkCheckBox(bri_excl_frame, text="Exclude bad",
                        variable=self.exclude_bad_var).pack(
            side="left", padx=(0, 4))
        ctk.CTkButton(bri_excl_frame, text="Load JSON",
                      command=self._browse_bad_json,
                      width=75, height=24,
                      fg_color="#4F5D75", hover_color="#61708A",
                      font=("Arial", 10)).pack(side="left")

        ctk.CTkButton(row3, text="Preview",
                      command=self._preview_brightness_threaded,
                      fg_color="#2E8B57", hover_color="#3AA86A", width=80).grid(
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
        col_excl_frame = ctk.CTkFrame(row4, fg_color="transparent")
        col_excl_frame.grid(row=0, column=5, padx=5, pady=3)
        ctk.CTkCheckBox(col_excl_frame, text="Exclude bad",
                        variable=self.exclude_bad_colour_var).pack(
            side="left", padx=(0, 4))
        ctk.CTkButton(col_excl_frame, text="Load JSON",
                      command=self._browse_bad_json,
                      width=75, height=24,
                      fg_color="#4F5D75", hover_color="#61708A",
                      font=("Arial", 10)).pack(side="left")

        ctk.CTkButton(row4, text="Preview",
                      command=self._preview_colour_threaded,
                      fg_color="#2E8B57", hover_color="#3AA86A", width=80).grid(
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

        avg_excl_frame = ctk.CTkFrame(row4b, fg_color="transparent")
        avg_excl_frame.grid(row=0, column=1, padx=5, pady=3)
        ctk.CTkCheckBox(avg_excl_frame, text="Use good filtered images only",
                        variable=self.exclude_bad_avg_var).pack(
            side="left", padx=(0, 4))
        ctk.CTkButton(avg_excl_frame, text="Load JSON",
                      command=self._browse_bad_json,
                      width=75, height=24,
                      fg_color="#4F5D75", hover_color="#61708A",
                      font=("Arial", 10)).pack(side="left")

        ctk.CTkLabel(
            row4b,
            text="-> Averages all readable images in each folder.\n"
                 "   With sub-folders enabled, each image-containing folder is averaged separately.",
            font=("Arial", 10), text_color="gray", justify="left",
        ).grid(row=0, column=2, padx=5, pady=3, sticky="w")
        ctk.CTkButton(row4b, text="Run Averaging",
                      command=self._average_images_threaded,
                      fg_color="#0F52BA", hover_color="#2A6BD1").grid(
            row=0, column=3, padx=10, pady=3)

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
            row5, text="Save Settings",fg_color="#4F5D75", hover_color="#61708A", command=self._save_settings, width=110)
        self.btn_save_settings.grid(row=0, column=4, padx=5, pady=5)

        self.btn_load_settings = ctk.CTkButton(
            row5, text="Load Settings",fg_color="#4F5D75", hover_color="#61708A", command=self._load_settings, width=110)
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
            fg_color="#2E8B57", hover_color="#3AA86A", width=180)
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
        self._print_guide()

    def _print_guide(self):
        """Print the tool guide and reference table to the console."""
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
              "  Overexposure    - Two parameters:\n"
              "                    Pixel value: pixels >= this value "
              "in ALL 3 channels = saturated.\n"
              "                    Fraction: if more than this fraction "
              "is saturated -> bad.\n"
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
              "  Failed images   - Flags corrupt, truncated, or empty files.\n"
              "                    Pre-checks file size: 0-byte files are always\n"
              "                    flagged; files below '% of median size' of\n"
              "                    the dataset median (computed from non-zero\n"
              "                    files only) are also flagged. Files that\n"
              "                    cannot be decoded by OpenCV are caught\n"
              "                    unconditionally regardless of this checkbox.\n"
              "  AOI             - Area of interest.  Tick 'Filter inside AOI only'\n"
              "                    to limit the image checks to a part of the frame,\n"
              "                    for example to leave out a bright sky or a flare\n"
              "                    at the side.  'Select AOI' opens one of your\n"
              "                    images: drag a rectangle, or choose Polygon and\n"
              "                    click its corners.  The checks then look only at\n"
              "                    the pixels inside the AOI, so every fraction is a\n"
              "                    share of the AOI and not of the whole image;\n"
              "                    check your thresholds again after setting one.\n"
              "                    The same AOI is used for every image of the run,\n"
              "                    so the camera view has to stay the same.  'Failed\n"
              "                    images' goes by file size and does not use it.\n"
              "  Results         - bad_images.txt and bad_images.json list the\n"
              "                    rejected images; filter_summary.txt gives the\n"
              "                    totals per reason and the thresholds used.\n"
              "  Long runs       - Progress is saved at least every 30 s and when\n"
              "                    the run is cancelled.  To continue an unfinished\n"
              "                    run, start the filter again with the same folders\n"
              "                    and settings and answer Yes.\n"
              "\n"
              "NOTE: The 'Blur' filter includes a 'Local blur fraction'\n"
              "parameter.  When set > 0, an 8x8 grid of blocks is\n"
              "checked for individual blurriness. This catches partial\n"
              "obstructions (frost, ice, condensation) where the scene\n"
              "is visible behind the obstruction.  Obstructed images\n"
              "typically score 60-100%, clean images 10-25%.\n"
              "Set to 0.40 (40%) for a good balance.  Set to 0.0\n"
              "to disable and revert to legacy behaviour.\n"
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
              "\n"
              "AVERAGE IMAGES PER FOLDER\n"
              "  Averages all readable images in each folder into a single\n"
              "  composite image.  With sub-folders enabled, each folder is\n"
              "  averaged separately.\n"
              "  'Use good filtered images only' — excludes images flagged\n"
              "  as bad during filtering.  Folders with no good images are\n"
              "  skipped entirely (no empty output folders).\n"
              "\n"
              "BAD-IMAGE LIST RESOLUTION\n"
              "  All 'exclude bad' options (brightness, colour, averaging)\n"
              "  resolve the bad-image list in this order:\n"
              "    1. In-memory list from a filter run in this session.\n"
              "    2. Manually uploaded JSON via the 'Load JSON' button.\n"
              "    3. bad_images.json in the output folder (cross-session).\n"
              "  If no source is available, a warning is shown and the\n"
              "  task is aborted.  This lets you filter once, then run\n"
              "  brightness / colour / averaging in separate sessions.\n"
              "\n"
              "REFERENCE TABLE — Suggested parameter values\n"
              "┌──────────────────────┬──────────────┬──────────────┬──────────────┐\n"
              "│ Parameter            │ Conservative │   Moderate   │   Extreme    │\n"
              "├──────────────────────┼──────────────┼──────────────┼──────────────┤\n"
              "│ Blur threshold       │      10      │      20      │      >40     │\n"
              "│ Local blur fraction  │     0.60     │     0.40     │     <0.30    │\n"
              "│ Overexp pixel value  │     254      │     245      │     <235     │\n"
              "│ Overexp fraction     │     0.20     │     0.15     │     <0.08    │\n"
              "│ Dark pixel value     │      10      │      20      │      >35     │\n"
              "│ Under-exp fraction   │     0.35     │     0.25     │     <0.15    │\n"
              "│ Dark max threshold   │      15      │      25      │      >40     │\n"
              "│ Entropy threshold    │     4.5      │     5.0      │     >5.5     │\n"
              "│ Droplet fraction     │     0.12     │     0.08     │     <0.04    │\n"
              "│ Obstruction fraction │     0.45     │     0.35     │     <0.20    │\n"
              "│ Failed % of median   │      5       │      10      │      >20     │\n"
              "├──────────────────────┼──────────────┼──────────────┼──────────────┤\n"
              "│ L-tolerance (bright) │      8       │      5       │      <3      │\n"
              "│ Sky mask (%)         │      0       │     15       │     >30      │\n"
              "└──────────────────────┴──────────────┴──────────────┴──────────────┘\n"
              "  Conservative = fewer images rejected / less correction applied\n"
              "  Moderate     = balanced (default values)\n"
              "  Extreme      = aggressive filtering / correction\n"
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
            self._ui_call(fn, title, text, parent=self)

    def _ui_set_progress_fraction(self, frac):
        self._ui_call(lambda: self.progress_bar.set(frac))

    def _ui_set_eta(self, text):
        self._ui_call(lambda: self.eta_label.configure(text=text))

    def _toggle_task_sections(self, initial=False):
        sections = [
            (self.run_filter_var.get(), self.filter_section_frame),
            (self.run_lens_var.get(), self.lens_section_frame),
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
            messagebox.showwarning("Warning", "Select an image folder first.", parent=self)
            return False
        if not self.output_folder:
            messagebox.showwarning("Warning", "Select an output folder first.", parent=self)
            return False
        # In recursive mode, an output folder nested inside the input folder
        # means the next run re-ingests previously written images as new inputs
        # (compounding corrections / double-counting stats).  Warn and let the
        # user decide rather than silently corrupting a re-run.
        try:
            if bool(self.recursive_var.get()):
                in_res = Path(self.input_folder).resolve()
                out_res = Path(self.output_folder).resolve()
                if in_res == out_res or in_res in out_res.parents:
                    proceed = messagebox.askyesno(
                        "Output inside input",
                        "The output folder is inside the input folder while "
                        "'recursive' is enabled.\n\nA later run may re-process "
                        "images written by this one. Continue anyway?",
                        parent=self)
                    if not proceed:
                        return False
        except Exception:
            pass
        return True

    def _collect_filter_config(self):
        if not self._check_folders_ui():
            return None
        return {
            "input_folder": self.input_folder,
            "output_folder": self.output_folder,
            "recursive": bool(self.recursive_var.get()),
            "blur_t": float(self.filter_entries["Blur threshold"].get()),
            "local_blur_f": float(self.filter_entries["Local blur fraction"].get()),
            "clip_f": float(self.filter_entries["Clip overexp fraction"].get()),
            "overexp_px": int(self.filter_entries["Overexp pixel value"].get()),
            "dark_v": int(self.filter_entries["Dark pixel value"].get()),
            "under_f": float(self.filter_entries["Under-exp fraction"].get()),
            "dark_max": int(self.filter_entries["Dark max threshold"].get()),
            "entropy_t": float(self.filter_entries["Entropy threshold"].get()),
            "droplet_f": float(self.filter_entries["Droplet fraction"].get()),
            "obstruct_f": float(self.filter_entries["Obstruction fraction"].get()),
            "failed_min_pct": float(self.filter_entries["Failed min size pct"].get()),
            "enabled": {k: v.get() for k, v in self.filter_enabled.items()},
            "export_good": bool(self.export_filtered_good_var.get()),
            "aoi": self._filter_aoi_for_run(),
        }

    # ——— area of interest of the filter ———

    def _filter_aoi_for_run(self):
        """The AOI the next filter run uses: a copy of the record, or None."""
        if not self.aoi_enabled_var.get():
            return None
        if self.filter_aoi is None:
            raise ValueError("'Filter inside AOI only' is ticked, but no AOI has been "
                             "selected.\n\nSelect an AOI or untick the box.")
        return json.loads(json.dumps(self.filter_aoi))

    def _on_aoi_toggle(self):
        if self.aoi_enabled_var.get() and self.filter_aoi is None:
            self._select_filter_aoi()      # the box is unticked again if nothing is selected
        self._update_aoi_label()

    def _aoi_sample_image(self, max_tries=30):
        """An image of the input folder to draw the AOI on: the one used
        before, otherwise the first one that can be read."""
        if self.filter_aoi_image and os.path.isfile(self.filter_aoi_image):
            return self.filter_aoi_image
        try:
            candidates = iter_images(self.input_folder, bool(self.recursive_var.get()))
            for n, p in enumerate(candidates):
                if n >= max_tries:
                    break
                try:
                    if os.path.getsize(p) == 0:
                        continue
                except OSError:
                    continue
                if imread_safe(str(p), verbose=False) is not None:
                    return str(p)
        except OSError:
            pass
        return None

    def _select_filter_aoi(self):
        """Open the window in which the AOI is drawn on one of the images."""
        selector = self._aoi_selector
        try:
            if selector is not None and selector.winfo_exists():
                bring_child_to_front(selector, self)
                return
        except tk.TclError:
            pass
        if not self.input_folder:
            messagebox.showwarning("Select AOI", "Select an image folder first.", parent=self)
            self._aoi_selection_cancelled()
            return
        path = self._aoi_sample_image()
        if path is None:
            messagebox.showwarning(
                "Select AOI",
                "No image that can be read was found in the image folder.\n\n"
                "If the images are in sub-folders, tick 'Include sub-folders'.",
                parent=self)
            self._aoi_selection_cancelled()
            return
        self._aoi_selector = AOISelectorWindow(
            self, path, self._set_filter_aoi, aoi=self.filter_aoi,
            initial_dir=self.input_folder, on_cancel=self._aoi_selection_cancelled)

    def _set_filter_aoi(self, aoi, image_path=None):
        """Take over the AOI confirmed in the selector window."""
        self._aoi_selector = None
        self.filter_aoi = aoi
        self.filter_aoi_image = image_path
        self.aoi_enabled_var.set(True)
        self._update_aoi_label()
        drawn_on = f", drawn on {os.path.basename(image_path)}" if image_path else ""
        print(f"[AOI] {describe_aoi(aoi)}{drawn_on}")
        if aoi["shape"] == "polygon":
            print("      corners (x, y): "
                  + ", ".join(f"({x}, {y})" for x, y in aoi["points"]))
        print("      The image checks look only inside this area; their fractions "
              "are shares of it.")

    def _aoi_selection_cancelled(self):
        self._aoi_selector = None
        if self.filter_aoi is None:
            self.aoi_enabled_var.set(False)
        self._update_aoi_label()

    def _update_aoi_label(self):
        aoi = self.filter_aoi
        if aoi is None:
            self.aoi_status_label.configure(text="AOI: not set", text_color="gray")
            return
        if aoi["shape"] == "rectangle":
            (x0, y0), (x1, y1) = aoi["points"]
            what = f"rectangle {x1 - x0 + 1} x {y1 - y0 + 1} px"
        else:
            what = f"polygon with {len(aoi['points'])} corners"
        used = bool(self.aoi_enabled_var.get())
        self.aoi_status_label.configure(
            text=(f"AOI: {what}, {100.0 * aoi_share_of_image(aoi):.1f} % of the image"
                  + ("" if used else "  (not used)")),
            text_color="#2ECC71" if used else "gray")

    def _aoi_drawn_on(self):
        """The image the AOI was drawn on, relative to the input folder when
        it lies inside it (kept in the settings file)."""
        path = self.filter_aoi_image
        if not path or self.filter_aoi is None:
            return ""
        if self.input_folder:
            try:
                rel = os.path.relpath(path, self.input_folder)
                if not rel.startswith(".."):
                    return rel.replace("\\", "/")
            except ValueError:
                pass
        return os.path.basename(path)

    def _collect_brightness_config(self):
        if not self._check_folders_ui():
            return None
        exclude_bad = bool(self.exclude_bad_var.get())
        bad_list = []
        if exclude_bad:
            bad_list = self._resolve_bad_list("brightness")
            if bad_list is None:
                return None  # user was warned, abort
        sky_pct = float(self.sky_entry.get())
        return {
            "input_folder": self.input_folder,
            "output_folder": self.output_folder,
            "recursive": bool(self.recursive_var.get()),
            "tol": float(self.tol_entry.get()),
            "sky_pct": sky_pct,
            "sky_frac": max(0.0, min(0.9, sky_pct / 100.0)),
            "exclude_bad": exclude_bad,
            "bad_list": bad_list,
        }

    def _collect_colour_config(self):
        if not self._check_folders_ui():
            return None
        if self.ref_colour_bgr is None:
            messagebox.showwarning("Warning", "Select a reference image for colour first.", parent=self)
            return None
        exclude_bad = bool(self.exclude_bad_colour_var.get())
        bad_list = []
        if exclude_bad:
            bad_list = self._resolve_bad_list("colour")
            if bad_list is None:
                return None  # user was warned, abort
        return {
            "input_folder": self.input_folder,
            "output_folder": self.output_folder,
            "recursive": bool(self.recursive_var.get()),
            "exclude_bad": exclude_bad,
            "bad_list": bad_list,
            "ref_colour_path": self.ref_colour_path,
            "ref_colour_bgr": self.ref_colour_bgr.copy(),
            "algo_name": self.colour_algo_var.get(),
        }

    def _collect_average_config(self):
        if not self._check_folders_ui():
            return None
        exclude_bad = bool(self.exclude_bad_avg_var.get())
        bad_list = []
        if exclude_bad:
            bad_list = self._resolve_bad_list("averaging")
            if bad_list is None:
                return None  # user was warned, abort
        return {
            "input_folder": self.input_folder,
            "output_folder": self.output_folder,
            "recursive": bool(self.recursive_var.get()),
            "exclude_bad": exclude_bad,
            "bad_list": bad_list,
        }

    def _collect_lens_config(self):
        if not self._check_folders_ui():
            return None
        if self.lens_calib_data is None:
            messagebox.showwarning(
                "Warning",
                "Select a lens calibration .pkl file first.",
                parent=self)
            return None
        exclude_bad = bool(self.exclude_bad_lens_var.get())
        bad_list = []
        if exclude_bad:
            bad_list = self._resolve_bad_list("lens correction")
            if bad_list is None:
                return None  # user was warned, abort
        return {
            "input_folder": self.input_folder,
            "output_folder": self.output_folder,
            "recursive": bool(self.recursive_var.get()),
            "exclude_bad": exclude_bad,
            "bad_list": bad_list,
            "calib_path": self.lens_calib_path,
            "calib_data": self.lens_calib_data,
            "crop_to_valid": bool(self.lens_crop_var.get()),
        }

    def _resolve_bad_list(self, task_label=""):
        """
        Return a bad-image list suitable for exclusion.

        Priority:
          1. In-memory ``self.bad_list`` (from a filter run in this session).
          2. ``bad_images.json`` found in the output folder.

        Returns a list of absolute path strings, or **None** if exclusion
        was requested but no list could be found (a warning is shown).
        """
        # 1. in-memory list from current session
        if self.bad_list:
            source = "uploaded JSON" if self.loaded_bad_json_path else "filter run"
            print(f"  [{task_label}] Using in-memory bad list ({len(self.bad_list)} images, source: {source})")
            return list(self.bad_list)

        # 2. try loading from JSON in the output folder
        json_path = os.path.join(self.output_folder, "bad_images.json")
        if os.path.isfile(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                loaded = []
                json_input_folder = data.get("input_folder", "")
                for item in data.get("bad_images", []):
                    abs_p = item.get("absolute_path", "")
                    if abs_p and os.path.isfile(abs_p):
                        loaded.append(abs_p)
                    else:
                        # fallback: reconstruct from relative path + current input folder
                        rel_p = item.get("relative_path", "")
                        if rel_p:
                            reconstructed = os.path.join(self.input_folder, rel_p)
                            if os.path.isfile(reconstructed):
                                loaded.append(reconstructed)
                if loaded:
                    print(f"  [{task_label}] Loaded bad list from JSON: {json_path} ({len(loaded)} images)")
                    self.bad_list = loaded  # cache for subsequent tasks
                    return loaded
                else:
                    messagebox.showwarning(
                        "Warning",
                        f"bad_images.json was found but none of the listed files "
                        f"could be resolved.\n\nJSON: {json_path}\n"
                        f"Input folder: {self.input_folder}\n\n"
                        f"Please re-run filtering or check your paths.",
                        parent=self,
                    )
                    return None
            except Exception as e:
                messagebox.showwarning(
                    "Warning",
                    f"Could not parse bad_images.json:\n{json_path}\n\nError: {e}",
                    parent=self,
                )
                return None

        # 3. nothing found
        messagebox.showwarning(
            "Warning",
            f"'Use good filtered images' is enabled but no bad-image list "
            f"was found.\n\n"
            f"Either:\n"
            f"  • Run the Filter step first, or\n"
            f"  • Use the 'Load JSON' button to upload a bad_images.json, or\n"
            f"  • Ensure bad_images.json exists in the output folder:\n"
            f"    {self.output_folder}",
            parent=self,
        )
        return None

    # ——— browse callbacks ———

    def _browse_input(self):
        d = filedialog.askdirectory(parent =self, title="Select Image Folder")
        if d:
            self.input_folder = d
            show_path(self.input_label, d, "input")
            # The AOI stays (the next folder is usually the same camera), but
            # the selector should then open on an image of the new folder.
            image = self.filter_aoi_image
            if image:
                try:
                    outside = os.path.relpath(image, d).startswith("..")
                except ValueError:          # another drive
                    outside = True
                if outside:
                    self.filter_aoi_image = None

    def _browse_output(self):
        d = filedialog.askdirectory(parent =self, title="Select Output Folder")
        if d:
            self.output_folder = d
            show_path(self.output_label, d, "output", empty="No output folder selected")

    def _browse_ref_colour(self):
        """Let user pick one reference image for colour harmonisation."""
        init_dir = self.input_folder if self.input_folder else None
        f = filedialog.askopenfilename(parent =self, 
            title="Select Reference Image for Colour",
            initialdir=init_dir,
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                       ("All files", "*.*")])
        if f:
            img = imread_safe(f)
            if img is None:
                messagebox.showerror("Error",
                                      f"Cannot read image:\n{f}",
                                      parent=self)
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

    def _browse_lens_calib(self):
        """Let user pick a lens_calibration.pkl produced by the
        Lens Correction module."""
        init_dir = self.input_folder if self.input_folder else None
        f = filedialog.askopenfilename(
            parent=self,
            title="Select Lens Calibration File",
            initialdir=init_dir,
            filetypes=[("Calibration", "*.pkl *.json"), ("Pickle files", "*.pkl"), ("JSON files", "*.json"), ("All files", "*.*")])
        if not f:
            return
        try:
            data = load_lens_calibration(f)
            if (not isinstance(data, dict)
                    or "camera_matrix" not in data
                    or "dist_coeff" not in data):
                messagebox.showerror(
                    "Error",
                    "Invalid lens calibration file:\n"
                    "expected a dict with 'camera_matrix' and 'dist_coeff'.",
                    parent=self)
                return
            self.lens_calib_path = f
            self.lens_calib_data = data
            name = os.path.basename(f)
            img_size = data.get("image_size")
            if img_size is not None and len(img_size) >= 2:
                calib_H = int(img_size[0])
                calib_W = int(img_size[1])
                size_str = f"{calib_W}x{calib_H}"
            else:
                calib_H = calib_W = None
                size_str = "?"
            rms = data.get("rms_error")
            rms_str = f", RMS={rms:.3f}px" if rms is not None else ""
            self.lens_calib_label.configure(
                text=f"{name}  ({size_str}{rms_str})")
            print(f"Lens calibration loaded: {name}")
            if calib_W is not None:
                print(f"  Calibrated image size: {calib_W} x {calib_H} px")
            if rms is not None:
                print(f"  RMS reprojection error: {rms:.4f} px")
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Cannot load .pkl file:\n{f}\n\n{e}",
                parent=self)

    def _browse_bad_json(self):
        """Let user upload a bad_images.json from a previous filter run."""
        init_dir = self.output_folder or self.input_folder or None
        f = filedialog.askopenfilename(
            parent=self,
            title="Select Bad Images JSON File",
            initialdir=init_dir,
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if not f:
            return
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            loaded = []
            for item in data.get("bad_images", []):
                abs_p = item.get("absolute_path", "")
                if abs_p and os.path.isfile(abs_p):
                    loaded.append(abs_p)
                else:
                    rel_p = item.get("relative_path", "")
                    if rel_p and self.input_folder:
                        reconstructed = os.path.join(self.input_folder, rel_p)
                        if os.path.isfile(reconstructed):
                            loaded.append(reconstructed)
            if not loaded:
                messagebox.showwarning(
                    "Warning",
                    f"No valid image paths could be resolved from:\n{f}\n\n"
                    f"Ensure the input folder is set and the JSON paths match.",
                    parent=self)
                return
            self.bad_list = loaded
            self.loaded_bad_json_path = f
            self._update_bad_json_label()
            print(f"Loaded bad-image list from JSON: {os.path.basename(f)} "
                  f"({len(loaded)} bad images)")
        except Exception as e:
            messagebox.showerror("Error",
                                  f"Could not parse JSON file:\n{f}\n\nError: {e}",
                                  parent=self)

    def _update_bad_json_label(self):
        """Update the shared label showing loaded bad-list status."""
        if self.loaded_bad_json_path and self.bad_list:
            name = os.path.basename(self.loaded_bad_json_path)
            text = f"Bad list loaded: {name} ({len(self.bad_list)} images)"
        elif self.bad_list:
            text = f"Bad list: {len(self.bad_list)} images (from filter run)"
        else:
            text = ""
        self.bad_json_status_label.configure(text=text)

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
                "loaded_bad_json_path": self.loaded_bad_json_path or "",
                "lens_calib_path": self.lens_calib_path or "",
            },
            "ui_state": {
                "recursive": bool(self.recursive_var.get()),
                "run_filter": bool(self.run_filter_var.get()),
                "run_lens": bool(self.run_lens_var.get()),
                "run_brightness": bool(self.run_brightness_var.get()),
                "run_colour": bool(self.run_colour_var.get()),
                "run_average": bool(self.run_average_var.get()),
                "exclude_bad_brightness": bool(self.exclude_bad_var.get()),
                "exclude_bad_colour": bool(self.exclude_bad_colour_var.get()),
                "exclude_bad_avg": bool(self.exclude_bad_avg_var.get()),
                "exclude_bad_lens": bool(self.exclude_bad_lens_var.get()),
                "lens_crop_to_valid": bool(self.lens_crop_var.get()),
                "export_filtered_good": bool(self.export_filtered_good_var.get()),
                "tol": self.tol_entry.get().strip(),
                "sky_pct": self.sky_entry.get().strip(),
                "colour_algorithm": self.colour_algo_var.get(),
                "filters_enabled": {k: bool(v.get()) for k, v in self.filter_enabled.items()},
                "filter_values": {k: e.get().strip() for k, e in self.filter_entries.items()},
                # area of interest of the filter: pixel coordinates in an
                # image of "image_size" (see normalise_aoi)
                "filter_aoi_enabled": bool(self.aoi_enabled_var.get()),
                "filter_aoi": self.filter_aoi,
                "filter_aoi_drawn_on": self._aoi_drawn_on(),
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
            show_path(self.input_label, self.input_folder, "input")
            show_path(self.output_label, self.output_folder, "output", empty="No output folder selected")

            self.recursive_var.set(bool(ui.get("recursive", False)))
            self.run_filter_var.set(bool(ui.get("run_filter", True)))
            self.run_lens_var.set(bool(ui.get("run_lens", False)))
            self.run_brightness_var.set(bool(ui.get("run_brightness", True)))
            self.run_colour_var.set(bool(ui.get("run_colour", True)))
            self.run_average_var.set(bool(ui.get("run_average", False)))
            self.exclude_bad_var.set(bool(ui.get("exclude_bad_brightness", True)))
            self.exclude_bad_colour_var.set(bool(ui.get("exclude_bad_colour", True)))
            self.exclude_bad_avg_var.set(bool(ui.get("exclude_bad_avg", False)))
            self.exclude_bad_lens_var.set(bool(ui.get("exclude_bad_lens", True)))
            self.lens_crop_var.set(bool(ui.get("lens_crop_to_valid", True)))
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

            # Area of interest of the filter (absent in files of earlier versions)
            self.filter_aoi = None
            self.filter_aoi_image = None
            saved_aoi = ui.get("filter_aoi")
            if saved_aoi:
                try:
                    self.filter_aoi = normalise_aoi(saved_aoi)
                except ValueError as e:
                    print(f"[Settings] The saved AOI was not loaded: {e}")
            drawn_on = str(ui.get("filter_aoi_drawn_on") or "")
            if self.filter_aoi is not None and drawn_on and self.input_folder:
                candidate = os.path.join(self.input_folder, drawn_on)
                if os.path.isfile(candidate):
                    self.filter_aoi_image = candidate
            self.aoi_enabled_var.set(
                bool(ui.get("filter_aoi_enabled", False)) and self.filter_aoi is not None)
            self._update_aoi_label()

            ref_path = paths.get("ref_colour_path") or ""
            self.ref_colour_path = ref_path or None
            self.ref_colour_bgr = None
            if self.ref_colour_path and os.path.isfile(self.ref_colour_path):
                img = imread_safe(self.ref_colour_path)
                if img is not None:
                    self.ref_colour_bgr = img
                    self.ref_colour_label.configure(
                        text=f"{os.path.basename(self.ref_colour_path)}  ({img.shape[1]}x{img.shape[0]})"
                    )
                else:
                    self.ref_colour_label.configure(text="Saved reference missing/unreadable")
            else:
                self.ref_colour_label.configure(text="No reference selected")

            # Restore loaded bad-image JSON path
            bad_json = paths.get("loaded_bad_json_path") or ""
            self.loaded_bad_json_path = bad_json or None
            if self.loaded_bad_json_path and os.path.isfile(self.loaded_bad_json_path):
                # Re-parse the JSON to populate bad_list
                try:
                    with open(self.loaded_bad_json_path, "r", encoding="utf-8") as fh:
                        bdata = json.load(fh)
                    loaded = []
                    for item in bdata.get("bad_images", []):
                        abs_p = item.get("absolute_path", "")
                        if abs_p and os.path.isfile(abs_p):
                            loaded.append(abs_p)
                        else:
                            rel_p = item.get("relative_path", "")
                            if rel_p and self.input_folder:
                                reconstructed = os.path.join(self.input_folder, rel_p)
                                if os.path.isfile(reconstructed):
                                    loaded.append(reconstructed)
                    if loaded:
                        self.bad_list = loaded
                except Exception:
                    self.loaded_bad_json_path = None
            else:
                self.loaded_bad_json_path = None
            self._update_bad_json_label()

            # Restore lens calibration path
            lens_path = paths.get("lens_calib_path") or ""
            self.lens_calib_path = None
            self.lens_calib_data = None
            if lens_path and os.path.isfile(lens_path):
                try:
                    calib = load_lens_calibration(lens_path)
                    if (isinstance(calib, dict)
                            and "camera_matrix" in calib
                            and "dist_coeff" in calib):
                        self.lens_calib_path = lens_path
                        self.lens_calib_data = calib
                        img_size = calib.get("image_size")
                        if img_size is not None and len(img_size) >= 2:
                            size_str = f"{int(img_size[1])}x{int(img_size[0])}"
                        else:
                            size_str = "?"
                        rms = calib.get("rms_error")
                        rms_str = f", RMS={rms:.3f}px" if rms is not None else ""
                        self.lens_calib_label.configure(
                            text=f"{os.path.basename(lens_path)}  ({size_str}{rms_str})")
                    else:
                        self.lens_calib_label.configure(
                            text="Saved calibration invalid")
                except Exception:
                    self.lens_calib_label.configure(
                        text="Saved calibration missing/unreadable")
            else:
                self.lens_calib_label.configure(text="No calibration selected")

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
        self.loaded_bad_json_path = None
        self.ref_colour_path = None
        self.ref_colour_bgr = None
        self.lens_calib_path = None
        self.lens_calib_data = None
        show_path(self.input_label, None, "input")
        show_path(self.output_label, None, "output", empty="No output folder selected")
        self.ref_colour_label.configure(text="No reference selected")
        self.lens_calib_label.configure(text="No calibration selected")
        self.bad_json_status_label.configure(text="")
        self.progress_bar.set(0)
        self.eta_label.configure(text="ETA: --")
        self.recursive_var.set(False)
        self.run_filter_var.set(True)
        self.run_lens_var.set(False)
        self.run_brightness_var.set(True)
        self.run_colour_var.set(True)
        self.run_average_var.set(False)
        self.exclude_bad_var.set(True)
        self.exclude_bad_colour_var.set(True)
        self.exclude_bad_avg_var.set(False)
        self.exclude_bad_lens_var.set(True)
        self.lens_crop_var.set(True)
        self.export_filtered_good_var.set(False)
        for var in self.filter_enabled.values():
            var.set(True)
        selector = self._aoi_selector
        self._aoi_selector = None
        try:
            if selector is not None and selector.winfo_exists():
                selector.destroy()
        except tk.TclError:
            pass
        self.filter_aoi = None
        self.filter_aoi_image = None
        self.aoi_enabled_var.set(False)
        self._update_aoi_label()

        self._restore_normal_plots()
        self._toggle_task_sections()
        self.console_text.delete("1.0", tk.END)
        print("Session reset.\n--------------------------------\n")
        self._print_guide()

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
    def _gather_images_from(folder, recursive=False, exclude_bad=False, bad_list=None,
                            progress_cb=None, cancel_cb=None):
        if recursive:
            images = collect_images_recursive(
                folder, progress_cb=progress_cb, cancel_cb=cancel_cb)
        else:
            images = collect_images(folder)
        if exclude_bad and bad_list:
            # Normalise case/separators so a bad_images.json written on another
            # OS (or with different path casing) still matches reliably.
            def _norm(x):
                return os.path.normcase(os.path.normpath(str(x)))
            bad_set = set(_norm(b) for b in bad_list)
            images = [p for p in images if _norm(p) not in bad_set]
        return images

    def _scan_feedback(self, folder, recursive):
        """Announce a (potentially slow) recursive scan and return
        (progress_cb, cancel_cb).  For a non-recursive gather returns
        (None, None) since that listing is fast."""
        if not recursive:
            return None, None
        print(f"Scanning '{folder}' for images (recursive)… "
              f"this can take a while on large / networked trees.")
        self._ui_set_eta("Scanning folders…")

        def progress_cb(n_dirs, n_imgs):
            print(f"  …scanned {n_dirs} folders, {n_imgs} images so far")
            self._ui_set_eta(f"Scanning… {n_dirs} folders, {n_imgs} imgs")

        def cancel_cb():
            return self._cancel_requested

        return progress_cb, cancel_cb

    def _gather_images(self, exclude_bad=True):
        return self._gather_images_from(
            self.input_folder,
            recursive=bool(self.recursive_var.get()),
            exclude_bad=exclude_bad,
            bad_list=self.bad_list,
        )

    # ——— ETA helper ———

    def _update_progress(self, idx, total, start_time, first_idx=0):
        # first_idx: index this session started at (non-zero when an
        # unfinished filter run is continued), so the ETA uses the images
        # processed since start_time only.
        frac = (idx + 1) / max(total, 1)
        elapsed = time.time() - start_time
        if idx > first_idx:
            per_image = elapsed / (idx + 1 - first_idx)
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
                  "#95a5a6", "#8e44ad", "#1abc9c", "#c0392b"]
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
                             alpha=0.25, color="#56B4E9",
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

    def _render_lens_results(self, counts, elapsed, calib_name):
        self._restore_normal_plots()
        self.axes[0].axis("off")
        self.axes[1].clear()
        labels = ["corrected", "skipped", "failed"]
        values = [counts.get(k, 0) for k in labels]
        bar_colors = ["#2ecc71", "#7f8c8d", "#e74c3c"]
        self.axes[1].bar(labels, values, color=bar_colors)
        self.axes[1].set_title("Lens Correction Results")
        self.axes[1].set_ylabel("Count")

        self.axes[2].clear()
        self.axes[2].text(
            0.5, 0.5,
            f"Calibration:\n{calib_name}\n\n"
            f"{counts.get('corrected', 0)} corrected\n"
            f"{counts.get('skipped', 0)} skipped\n"
            f"{counts.get('failed', 0)} failed\n\n"
            f"{format_eta(elapsed)} elapsed",
            ha="center", va="center", fontsize=11,
            transform=self.axes[2].transAxes)
        self.axes[2].axis("off")
        self.fig.tight_layout()
        self.canvas_plot.draw()

    # ——— filtering ———

    def _filter_threaded(self):
        if self._processing:
            return
        try:
            cfg = self._collect_filter_config()
        except Exception as e:
            messagebox.showerror("Error", str(e), parent=self)
            return
        if cfg is None:
            return
        proceed, cfg["resume"] = self._ask_resume_filter(cfg)
        if not proceed:
            return
        # Set the busy flag synchronously (before the thread starts) so a
        # rapid second click can't slip past the `if self._processing` guard
        # and launch a duplicate worker writing to the same outputs.
        self._processing = True
        self._cancel_requested = False
        threading.Thread(target=self._run_filter, args=(cfg,), daemon=True).start()

    def _ask_resume_filter(self, cfg):
        """Look for an unfinished filter run in the output folder and ask what to do.

        Returns (proceed, state).  `state` is the saved progress to continue
        from, or None for a run that starts at the first image.
        """
        path = os.path.join(cfg["output_folder"], FILTER_PROGRESS_FILE)
        state = read_filter_progress(path)
        if state is None or state["checked"] <= 0:
            return True, None

        header = state["header"]
        done = (f"{state['checked']} of {header.get('n_images', '?')} images checked, "
                f"last saved {state['saved']}")
        obstacle = filter_resume_obstacle(header, cfg, GEOCAMPAL_VERSION)
        if obstacle:
            start_new = messagebox.askyesno(
                "Unfinished filter run",
                "The output folder holds the progress of a filter run that did not "
                f"finish ({done}).\n\n"
                f"It cannot be continued because {obstacle}.\n\n"
                "Starting now discards that progress. Start a new run?",
                parent=self)
            return bool(start_new), None

        answer = messagebox.askyesnocancel(
            "Unfinished filter run",
            "The output folder holds the progress of a filter run that did not finish:\n\n"
            f"{done}\n\n"
            "Continue that run?\n\n"
            "Yes: continue where it stopped\n"
            "No: start again from the first image\n"
            "Cancel: do nothing",
            parent=self)
        if answer is None:
            return False, None
        return True, (state if answer else None)

    def _run_filter(self, cfg):
        try:
            self._processing = True
            self._cancel_requested = False
            self._ui_set_progress_fraction(0)
            self._ui_set_eta("ETA: --")

            pcb, ccb = self._scan_feedback(cfg["input_folder"], cfg["recursive"])
            images = self._gather_images_from(
                cfg["input_folder"], recursive=cfg["recursive"],
                progress_cb=pcb, cancel_cb=ccb)
            if self._cancel_requested:
                self._ui_set_eta("Cancelled")
                print("Cancelled during folder scan.")
                return
            if not images:
                self._ui_message("warning", "Warning", "No images found.")
                return

            # ——— unfinished run to continue? ———
            # The saved progress refers to images by their position in the
            # list, so it is only used when the list is the same as before.
            progress_path = os.path.join(cfg["output_folder"], FILTER_PROGRESS_FILE)
            fingerprint = image_list_fingerprint(images, cfg["input_folder"])
            resume = cfg.get("resume")
            if resume is not None:
                old = resume["header"]
                if (old.get("image_list") != fingerprint
                        or old.get("n_images") != len(images)
                        or resume["checked"] > len(images)):
                    if old.get("n_images") != len(images):
                        what = (f"it started with {old.get('n_images')} images and the "
                                f"input folder now holds {len(images)}")
                    else:
                        what = ("the input folder holds the same number of images, "
                                "but not the same files in the same order")
                    msg = (f"The unfinished filter run cannot be continued: {what}.\n\n"
                           "The saved progress was left as it is. To start from the first "
                           "image, run the filter again and answer No.")
                    print(msg)
                    self._ui_set_eta("Stopped")
                    self._ui_message("error", "Unfinished filter run", msg)
                    return

            en = cfg["enabled"]
            active = [k for k, v in en.items() if v]
            print(f"\nFiltering {len(images)} images with {len(active)} active filter(s): {', '.join(active)} ...")
            print("  Settings:")
            if en.get("Blur"):
                local_b = cfg.get('local_blur_f', 0.0)
                local_str = f", local blur fraction={local_b}" if local_b > 0 else ""
                print(f"    Blur:           threshold={cfg['blur_t']}{local_str}")
            if en.get("Overexposure"):
                print(f"    Overexposure:   fraction={cfg['clip_f']}, pixel value={cfg.get('overexp_px', 254)}")
            if en.get("Underexposure"):
                print(f"    Underexposure:  dark pixel={cfg['dark_v']}, fraction={cfg['under_f']}")
            if en.get("Darkness"):
                print(f"    Darkness:       max threshold={cfg['dark_max']}")
            if en.get("Low information"):
                print(f"    Low info:       entropy={cfg['entropy_t']}")
            if en.get("Lens droplets"):
                print(f"    Lens droplets:  fraction={cfg['droplet_f']}")
            if en.get("Obstruction"):
                print(f"    Obstruction:    fraction={cfg['obstruct_f']}")
            if en.get("Failed images"):
                print(f"    Failed images:  flag if < {cfg['failed_min_pct']}% of median file size")
            aoi = cfg.get("aoi")
            aoi_regions = {}         # (width, height) -> where the AOI lies in such an image
            aoi_scaled = 0           # images of another size than the AOI was drawn on
            if aoi is not None:
                print(f"    AOI:            {describe_aoi(aoi)}")
                print("                    (the image checks look only inside it; "
                      "fractions are shares of the AOI)")
            bad_list = []
            bad_reasons = {}
            reason_counts = {
                "blurry": 0, "clipped_overexposed": 0,
                "underexposed": 0, "too_dark": 0,
                "low_information": 0, "lens_droplets": 0,
                "obstruction": 0, "failed_image": 0,
            }
            start_time = time.time()

            # ——— progress file: results are saved while the run goes on ———
            first_idx = 0            # first image this session has to check
            elapsed_before = 0.0     # processing time of earlier sessions
            sessions = 1
            started_txt = time.strftime("%Y-%m-%d %H:%M:%S")
            if resume is not None:
                first_idx = resume["checked"]
                elapsed_before = resume["elapsed_s"]
                sessions = int(resume["header"].get("sessions", 1)) + 1
                started_txt = resume["header"].get("started", started_txt)
                for i, reasons in resume["bad"]:
                    bad_list.append(str(images[i]))
                    bad_reasons[str(images[i])] = list(reasons)
                reason_counts.update(resume["counts"])
                aoi_scaled = int(resume.get("aoi_scaled", 0))
                print(f"  Continuing the unfinished run: {first_idx} of {len(images)} "
                      f"images were already checked ({len(bad_list)} rejected).")
            start_filter_progress(progress_path, {
                "geocampal_filter_progress": 1,
                "geocampal_version": GEOCAMPAL_VERSION,
                "started": started_txt,
                "sessions": sessions,
                "input_folder": cfg["input_folder"],
                "recursive": bool(cfg["recursive"]),
                "filters": filter_settings_record(cfg),
                "aoi": aoi,
                "n_images": len(images),
                "image_list": fingerprint,
            }, resume)
            pending_bad = []         # rejected since the last save: (index, reasons)
            last_save_idx = first_idx
            last_save_time = time.time()

            def save_progress(n_checked):
                """Save the state after `n_checked` images; a failed save is
                reported and tried again at the next one."""
                try:
                    append_filter_checkpoint(
                        progress_path, n_checked,
                        elapsed_before + (time.time() - start_time),
                        reason_counts, pending_bad, aoi_scaled)
                except OSError as e:
                    print(f"  [Warning] Progress could not be saved ({e}); "
                          f"the run continues.")
                    return False
                pending_bad.clear()
                return True

            # --- Pre-compute file-size threshold for "Failed images" filter ---
            failed_size_threshold = 0  # bytes; 0 = only flag truly empty files
            if en.get("Failed images"):
                file_sizes = []
                for p in images:
                    try:
                        sz = os.path.getsize(str(p))
                    except OSError:
                        sz = 0
                    if sz > 0:
                        file_sizes.append(sz)
                if file_sizes:
                    median_size = float(np.median(file_sizes))
                    failed_size_threshold = median_size * (cfg["failed_min_pct"] / 100.0)
                    print(f"    File size stats: median={median_size/1024:.1f} KB, "
                          f"threshold={failed_size_threshold/1024:.1f} KB "
                          f"({len(file_sizes)} non-zero files)")

            for idx, p in enumerate(images):
                if idx < first_idx:
                    continue  # checked in an earlier session

                if self._cancel_requested:
                    saved = save_progress(idx)
                    self._ui_set_eta("Cancelled")
                    print("Filtering cancelled.")
                    if saved:
                        print(f"  Progress saved: {idx} of {len(images)} images checked. "
                              f"Run the filter again with the same folders and settings "
                              f"to continue.")
                    return

                # --- Save progress every FILTER_CHECKPOINT_SECONDS / _IMAGES ---
                # Done before image `idx` is looked at, so the saved state is
                # "the first idx images are checked".
                if (idx - last_save_idx >= FILTER_CHECKPOINT_IMAGES
                        or time.time() - last_save_time >= FILTER_CHECKPOINT_SECONDS):
                    if idx > last_save_idx and save_progress(idx):
                        last_save_idx = idx
                    last_save_time = time.time()

                # --- File-size pre-check (before attempting cv2.imread) ---
                try:
                    file_sz = os.path.getsize(str(p))
                except OSError:
                    file_sz = 0

                if en.get("Failed images") and file_sz < failed_size_threshold:
                    bad_list.append(str(p))
                    if file_sz == 0:
                        detail = "failed_image (0 bytes)"
                    else:
                        detail = f"failed_image ({file_sz/1024:.1f} KB < {failed_size_threshold/1024:.1f} KB threshold)"
                    bad_reasons[str(p)] = [detail]
                    pending_bad.append((idx, [detail]))
                    reason_counts["failed_image"] += 1
                    print(f"  x {p.name}: {detail}")
                    self._update_progress(idx, len(images), start_time, first_idx)
                    continue

                img = imread_safe(str(p))
                if img is None:
                    # Always flag unreadable files regardless of checkbox state
                    bad_list.append(str(p))
                    detail = f"failed_image (unreadable, {file_sz/1024:.1f} KB)"
                    bad_reasons[str(p)] = [detail]
                    pending_bad.append((idx, [detail]))
                    reason_counts["failed_image"] += 1
                    print(f"  x {p.name}: {detail}")
                    self._update_progress(idx, len(images), start_time, first_idx)
                    continue

                # --- Area of interest: the checks see only that part of the image ---
                mask = None          # set for a polygon: True inside it
                if aoi is not None:
                    size = (img.shape[1], img.shape[0])
                    region = aoi_regions.get(size)
                    if region is None:
                        region = aoi_regions[size] = aoi_region(aoi, *size)
                    x0, y0, x1, y1, mask, scaled = region
                    img = img[y0:y1, x0:x1]
                    if scaled:
                        aoi_scaled += 1
                        if aoi_scaled == 1:
                            print(f"  [Note] {p.name} is {size[0]} x {size[1]} px, not "
                                  f"{aoi['image_size'][0]} x {aoi['image_size'][1]} px like the "
                                  f"image the AOI was drawn on. The AOI is scaled to images of "
                                  f"another size; the summary says how many there were.")

                reasons = []
                if en.get("Blur") and is_blurry(img, cfg["blur_t"], local_blur_frac=cfg["local_blur_f"], mask=mask):
                    reasons.append("blurry")
                if en.get("Overexposure") and is_clipped_overexposed(img, cfg["clip_f"], cfg.get("overexp_px", 254), mask=mask):
                    reasons.append("clipped_overexposed")
                if en.get("Underexposure") and is_underexposed(img, cfg["dark_v"], cfg["under_f"], mask=mask):
                    reasons.append("underexposed")
                if en.get("Darkness") and is_too_dark(img, cfg["dark_max"], mask=mask):
                    reasons.append("too_dark")
                if en.get("Low information") and is_low_information(img, cfg["entropy_t"], mask=mask):
                    reasons.append("low_information")
                if en.get("Lens droplets") and has_lens_droplets(img, cfg["droplet_f"], mask=mask):
                    reasons.append("lens_droplets")
                if en.get("Obstruction") and has_obstruction(img, block_fraction=cfg["obstruct_f"], mask=mask):
                    reasons.append("obstruction")

                if reasons:
                    bad_list.append(str(p))
                    bad_reasons[str(p)] = reasons
                    pending_bad.append((idx, reasons))
                    for r in reasons:
                        reason_counts[r] = reason_counts.get(r, 0) + 1
                    print(f"  x {p.name}: {', '.join(reasons)}")

                self._update_progress(idx, len(images), start_time, first_idx)

            elapsed_total = elapsed_before + (time.time() - start_time)
            self.bad_list = bad_list
            self.loaded_bad_json_path = None  # in-memory list now from filter run
            self._ui_call(self._update_bad_json_label)
            details = []
            input_root = Path(cfg["input_folder"])
            for bp in bad_list:
                bp_path = Path(bp)
                details.append({
                    "absolute_path": str(bp_path),
                    "relative_path": str(bp_path.relative_to(input_root)),
                    "filename": bp_path.name,
                    "reasons": bad_reasons.get(str(bp_path), []),
                })
            self.bad_details = details

            n_bad = len(bad_list)
            n_good = len(images) - n_bad
            print(f"\nFilter complete: {n_bad} bad / {n_good} good out of {len(images)} images.")
            active_counts = {k: v for k, v in reason_counts.items() if v > 0}
            if active_counts:
                print("  Breakdown by reason:")
                for reason, count in active_counts.items():
                    print(f"    {reason:<25s} {count:>5d}")
            else:
                print("  No bad images detected.")

            txt_path = os.path.join(cfg["output_folder"], "bad_images.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("relative_path\tfilename\treason\n")
                for item in details:
                    f.write(f"{item['relative_path']}\t{item['filename']}\t{','.join(item['reasons'])}\n")
            print(f"Bad image list saved: {txt_path}")

            if aoi is not None and aoi_scaled:
                print(f"  AOI: {aoi_scaled} image{'s' if aoi_scaled != 1 else ''} had another "
                      f"size than the image it was drawn on; it was scaled to them.")

            summary = build_filter_summary(
                cfg, len(images), n_bad, reason_counts,
                started=started_txt, finished=time.strftime("%Y-%m-%d %H:%M:%S"),
                elapsed_s=elapsed_total, sessions=sessions, version=GEOCAMPAL_VERSION,
                aoi_scaled=aoi_scaled)

            json_path = os.path.join(cfg["output_folder"], "bad_images.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({
                    "input_folder": cfg["input_folder"],
                    "recursive": bool(cfg["recursive"]),
                    "summary": summary,
                    "bad_images": details,
                }, f, indent=2, ensure_ascii=False)
            print(f"Bad image JSON saved: {json_path}")

            summary_path = os.path.join(cfg["output_folder"], FILTER_SUMMARY_FILE)
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(format_filter_summary(summary))
            print(f"Filter summary saved: {summary_path}")

            # The results are complete, so the progress file is no longer needed.
            # Another program (virus scanner, cloud sync, a viewer) can hold the
            # file open for a moment, and Windows then refuses to delete it, so
            # the removal is tried for up to two seconds.
            for _attempt in range(10):
                try:
                    os.remove(progress_path)
                    break
                except FileNotFoundError:
                    break
                except OSError:
                    time.sleep(0.2)
            else:
                print(f"  [Note] {FILTER_PROGRESS_FILE} could not be removed because another "
                      f"program is using it. The results above are complete; delete the "
                      f"file, or answer No if the next run asks about an unfinished run.")

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
            messagebox.showerror("Error", str(e), parent=self)
            return
        if cfg is None:
            return
        # Set the busy flag synchronously (before the thread starts) so a
        # rapid second click can't slip past the `if self._processing` guard
        # and launch a duplicate worker writing to the same outputs.
        self._processing = True
        self._cancel_requested = False
        threading.Thread(target=self._run_preview_brightness, args=(cfg,), daemon=True).start()

    def _run_preview_brightness(self, cfg):
        try:
            self._processing = True
            self._cancel_requested = False
            self._ui_set_progress_fraction(0)

            pcb, ccb = self._scan_feedback(cfg["input_folder"], cfg["recursive"])
            images = self._gather_images_from(cfg["input_folder"], recursive=cfg["recursive"],
                                              exclude_bad=cfg["exclude_bad"], bad_list=cfg["bad_list"],
                                              progress_cb=pcb, cancel_cb=ccb)
            if self._cancel_requested:
                self._ui_set_eta("Cancelled")
                print("Cancelled during folder scan.")
                return
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
                img = imread_safe(str(p))
                if img is None:
                    means.append(np.nan)
                else:
                    means.append(mean_luminance(img, cfg["sky_frac"]))
            means_arr = np.array(means)
            if not np.any(~np.isnan(means_arr)):
                print("  ERROR: no readable images for brightness preview.")
                self._ui_message("warning", "Warning", "No readable images to preview.")
                return
            ref_mean = float(np.nanmedian(means_arr))
            ref_idx = int(np.nanargmin(np.abs(means_arr - ref_mean)))
            ref_img = imread_safe(str(images[ref_idx]))
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
                img = imread_safe(str(p))
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
            messagebox.showerror("Error", str(e), parent=self)
            return
        if cfg is None:
            return
        # Set the busy flag synchronously (before the thread starts) so a
        # rapid second click can't slip past the `if self._processing` guard
        # and launch a duplicate worker writing to the same outputs.
        self._processing = True
        self._cancel_requested = False
        threading.Thread(target=self._run_harmonise_brightness, args=(cfg,), daemon=True).start()

    def _run_harmonise_brightness(self, cfg):
        try:
            self._processing = True
            self._cancel_requested = False
            self._ui_set_progress_fraction(0)
            self._ui_set_eta("ETA: --")

            pcb, ccb = self._scan_feedback(cfg["input_folder"], cfg["recursive"])
            images = self._gather_images_from(cfg["input_folder"], recursive=cfg["recursive"],
                                              exclude_bad=cfg["exclude_bad"], bad_list=cfg["bad_list"],
                                              progress_cb=pcb, cancel_cb=ccb)
            if self._cancel_requested:
                self._ui_set_eta("Cancelled")
                print("Cancelled during folder scan.")
                return
            if not images:
                self._ui_message("warning", "Warning", "No images remaining after filtering.")
                return

            sky_msg = (f", sky mask top {cfg['sky_pct']:.0f}%" if cfg["sky_frac"] > 0 else "")
            print(f"\nHarmonising brightness for {len(images)} images (tolerance +/-{cfg['tol']} L-units{sky_msg})...")

            # Pass 1: compute per-image luminance ONLY.  Previously every decoded
            # image was held in `loaded` for the whole run, which needs tens of GB
            # (and crashes) on large batches — the exact case recursive mode hits.
            # We now store just the means and re-read images on demand, matching
            # the preview path's memory profile.
            means = []
            for p in images:
                if self._cancel_requested:
                    self._ui_set_eta("Cancelled")
                    print("Brightness harmonisation cancelled.")
                    return
                img = imread_safe(str(p))
                means.append(np.nan if img is None else mean_luminance(img, cfg["sky_frac"]))

            means_arr = np.array(means)
            valid = ~np.isnan(means_arr)
            if not valid.any():
                print("ERROR: no readable images to harmonise.")
                self._ui_message("warning", "Warning", "No readable images to harmonise.")
                return
            ref_mean = float(np.nanmedian(means_arr))
            ref_idx = int(np.nanargmin(np.abs(means_arr - ref_mean)))
            # Re-read just the reference image (kept in memory for hist_match).
            ref_img = imread_safe(str(images[ref_idx]))
            if ref_img is None:
                print(f"ERROR: reference image {images[ref_idx].name} could not be loaded.")
                return
            print(f"Reference luminance (median): {ref_mean:.1f}  (image: {images[ref_idx].name})")

            input_root = Path(cfg["input_folder"])
            counts = {"copy": 0, "histogram": 0, "soft-gain": 0, "bright-gain": 0, "reverted": 0}
            start_time = time.time()

            # Pass 2: re-read each image on demand and apply the correction.
            for idx, (p, m) in enumerate(zip(images, means)):
                if self._cancel_requested:
                    self._ui_set_eta("Cancelled")
                    print("Brightness harmonisation cancelled.")
                    return
                if np.isnan(m):
                    continue
                if idx == ref_idx:
                    img = ref_img
                else:
                    img = imread_safe(str(p))
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
                _save_preserving_tiff(out_path, corrected, _read_tiff_metadata(p), source_path=p)

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
        # Set the busy flag synchronously (before the thread starts) so a
        # rapid second click can't slip past the `if self._processing` guard
        # and launch a duplicate worker writing to the same outputs.
        self._processing = True
        self._cancel_requested = False
        threading.Thread(target=self._run_preview_colour, args=(cfg,), daemon=True).start()

    def _run_preview_colour(self, cfg):
        try:
            self._processing = True
            self._cancel_requested = False
            self._ui_set_progress_fraction(0)

            pcb, ccb = self._scan_feedback(cfg["input_folder"], cfg["recursive"])
            images = self._gather_images_from(cfg["input_folder"], recursive=cfg["recursive"],
                                              exclude_bad=cfg["exclude_bad"], bad_list=cfg["bad_list"],
                                              progress_cb=pcb, cancel_cb=ccb)
            if self._cancel_requested:
                self._ui_set_eta("Cancelled")
                print("Cancelled during folder scan.")
                return
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
                img = imread_safe(str(p))
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
        # Set the busy flag synchronously (before the thread starts) so a
        # rapid second click can't slip past the `if self._processing` guard
        # and launch a duplicate worker writing to the same outputs.
        self._processing = True
        self._cancel_requested = False
        threading.Thread(target=self._run_harmonise_colour, args=(cfg,), daemon=True).start()

    def _run_harmonise_colour(self, cfg):
        try:
            self._processing = True
            self._cancel_requested = False
            self._ui_set_progress_fraction(0)
            self._ui_set_eta("ETA: --")

            pcb, ccb = self._scan_feedback(cfg["input_folder"], cfg["recursive"])
            images = self._gather_images_from(cfg["input_folder"], recursive=cfg["recursive"],
                                              exclude_bad=cfg["exclude_bad"], bad_list=cfg["bad_list"],
                                              progress_cb=pcb, cancel_cb=ccb)
            if self._cancel_requested:
                self._ui_set_eta("Cancelled")
                print("Cancelled during folder scan.")
                return
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
                img = imread_safe(str(p))
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
                _save_preserving_tiff(out_path, corrected, _read_tiff_metadata(p), source_path=p)

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
        # Set the busy flag synchronously (before the thread starts) so a
        # rapid second click can't slip past the `if self._processing` guard
        # and launch a duplicate worker writing to the same outputs.
        self._processing = True
        self._cancel_requested = False
        threading.Thread(target=self._run_average_images, args=(cfg,), daemon=True).start()

    def _run_average_images(self, cfg):
        try:
            self._processing = True
            self._cancel_requested = False
            self._ui_set_progress_fraction(0)
            self._ui_set_eta("ETA: --")

            exclude_bad = cfg.get("exclude_bad", False)
            bad_set = set(map(str, cfg.get("bad_list", []))) if exclude_bad else set()

            input_root = Path(cfg["input_folder"])
            jobs = []
            excluded_total = 0
            if cfg["recursive"]:
                print(f"Scanning '{cfg['input_folder']}' for images (recursive)… "
                      f"this can take a while on large / networked trees.")
                self._ui_set_eta("Scanning folders…")
                for n_dirs, (root, _, files) in enumerate(os.walk(cfg["input_folder"]), 1):
                    if self._cancel_requested:
                        self._ui_set_eta("Cancelled")
                        print("Cancelled during folder scan.")
                        return
                    imgs = [Path(root) / f for f in sorted(files) if Path(f).suffix.lower() in IMAGE_EXTS]
                    if exclude_bad and bad_set:
                        before = len(imgs)
                        imgs = [p for p in imgs if str(p) not in bad_set]
                        excluded_total += before - len(imgs)
                    if imgs:
                        jobs.append((Path(root), imgs))
                    if n_dirs % 200 == 0:
                        print(f"  …scanned {n_dirs} folders, {len(jobs)} with images so far")
                        self._ui_set_eta(f"Scanning… {n_dirs} folders")
            else:
                imgs = collect_images(cfg["input_folder"])
                if exclude_bad and bad_set:
                    before = len(imgs)
                    imgs = [p for p in imgs if str(p) not in bad_set]
                    excluded_total += before - len(imgs)
                if imgs:
                    jobs.append((input_root, imgs))

            if not jobs:
                msg = "No images found."
                if exclude_bad:
                    msg = ("No good images remaining after excluding bad ones.\n"
                           "All images may have been flagged, or the bad list "
                           "does not match the input folder.")
                self._ui_message("warning", "Warning", msg)
                return

            written_count = 0
            skipped_count = 0
            avg_skipped = []   # (path, reason) for images left out of an average
            start_time = time.time()
            filter_msg = f" (excluded {excluded_total} bad images)" if excluded_total > 0 else ""
            print(f"\nAveraging images for {len(jobs)} folder(s){filter_msg}...")

            for idx, (folder_path, imgs) in enumerate(jobs):
                if self._cancel_requested:
                    self._ui_set_eta("Cancelled")
                    print("Averaging cancelled.")
                    return
                accum = None
                ref_shape = None
                valid_count = 0
                skipped_local = 0
                first_valid_path = None
                for p in imgs:
                    if self._cancel_requested:
                        self._ui_set_eta("Cancelled")
                        print("Averaging cancelled.")
                        return
                    img = imread_safe(str(p), cv2.IMREAD_COLOR)
                    if img is None:
                        skipped_local += 1
                        avg_skipped.append((str(p), "unreadable"))
                        print(f"  [SKIP] {p.name}: unreadable")
                        continue
                    if ref_shape is None:
                        ref_shape = img.shape
                        accum = img.astype(np.float64)
                        valid_count = 1
                        first_valid_path = p
                    elif img.shape == ref_shape:
                        accum += img.astype(np.float64)
                        valid_count += 1
                    else:
                        skipped_local += 1
                        avg_skipped.append(
                            (str(p), f"size mismatch (expected {ref_shape})"))
                        print(f"  [SKIP] {p.name}: size mismatch in {folder_path.name}")

                if valid_count == 0 or accum is None:
                    skipped_count += 1
                    avg_skipped.append((str(folder_path),
                                        "folder skipped: no valid readable images"))
                    print(f"  [SKIP] {folder_path}: no valid readable images")
                    self._update_progress(idx, len(jobs), start_time)
                    continue

                avg = np.clip(accum / valid_count, 0, 255).astype(np.uint8)

                # Preserve folder structure: output_folder / inputname_averaged / relative_path
                out_root = self._build_output_root(cfg["output_folder"], input_root, "averaged")
                if cfg["recursive"]:
                    rel = folder_path.relative_to(input_root)
                    out_dir = out_root / rel
                else:
                    out_dir = out_root

                out_dir.mkdir(parents=True, exist_ok=True)
                geo_meta = _read_tiff_metadata(first_valid_path) if first_valid_path else None
                out_ext = ".tif" if geo_meta is not None else ".png"
                out_name = f"{folder_path.name}_average{out_ext}"
                out_path = out_dir / out_name
                _save_preserving_tiff(out_path, avg, geo_meta, source_path=first_valid_path)
                written_count += 1
                print(f"  Wrote average for {folder_path}: {out_path.name} ({valid_count} images, {skipped_local} skipped)")
                self._update_progress(idx, len(jobs), start_time)

            elapsed = time.time() - start_time
            print(f"\nAveraging complete ({format_eta(elapsed)} elapsed): {written_count} written, {skipped_count} skipped.")
            avg_report = None
            if avg_skipped:
                try:
                    avg_report = os.path.join(cfg["output_folder"],
                                              "averaging_skipped.txt")
                    with open(avg_report, "w", encoding="utf-8") as fh:
                        fh.write("Folder averaging - images left out\n")
                        fh.write(f"Run: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                        fh.write(f"Input folder: {cfg['input_folder']}\n")
                        fh.write(f"Averages written: {written_count}   "
                                 f"Folders skipped: {skipped_count}   "
                                 f"Images left out: {len(avg_skipped)}\n\n")
                        for name, reason in avg_skipped:
                            fh.write(f"{name}\t{reason}\n")
                    print(f"  Skipped list saved to: {avg_report}")
                except Exception as e:
                    avg_report = None
                    print(f"  Could not write skipped list: {e}")
            self._ui_call(self._render_average_results, len(jobs), written_count, skipped_count)
            self._ui_set_eta("Done")
            avg_msg = f"Folder averages saved to:\n{cfg['output_folder']}"
            if avg_skipped:
                avg_msg += f"\n\n{len(avg_skipped)} image(s)/folder(s) left out."
                if avg_report:
                    avg_msg += f"\n\nDetails: {avg_report}"
            self._ui_message("info", "Done", avg_msg)

        except Exception as e:
            print(f"[ERROR] {e}")
            self._ui_message("error", "Error", str(e))
        finally:
            self._processing = False

    # ——— lens correction ———

    def _lens_threaded(self):
        if self._processing:
            return
        try:
            cfg = self._collect_lens_config()
        except Exception as e:
            messagebox.showerror("Error", str(e), parent=self)
            return
        if cfg is None:
            return
        # Set the busy flag synchronously (before the thread starts) so a
        # rapid second click can't slip past the `if self._processing` guard
        # and launch a duplicate worker writing to the same outputs.
        self._processing = True
        self._cancel_requested = False
        threading.Thread(target=self._run_lens_correction,
                         args=(cfg,), daemon=True).start()

    def _run_lens_correction(self, cfg):
        """Apply lens distortion correction (cv2.undistort) to each image.

        Strategy
        --------
        - If the image size matches the calibration's `image_size`, K and D
          are used directly.
        - If the size differs but the aspect ratio matches, K is scaled
          proportionally (fx, fy, cx, cy by sx, sy).  This handles the
          common case of the same camera shooting at different resolutions.
        - If the aspect ratio differs, the image is skipped with a warning
          (D would no longer model the actual lens geometry).

        Output is written to ``<output>/<inputname>_lens_corrected/...``
        preserving sub-folder structure.  Originals are never modified.
        Rectification maps are cached per (W, H) so we don't recompute the
        maps for every image when sizes are homogeneous.
        """
        try:
            self._processing = True
            self._cancel_requested = False
            self._ui_set_progress_fraction(0)
            self._ui_set_eta("ETA: --")

            calib = cfg["calib_data"]
            K = np.asarray(calib["camera_matrix"], dtype=np.float64)
            D = np.asarray(calib["dist_coeff"], dtype=np.float64)
            img_size = calib.get("image_size")
            if img_size is not None and len(img_size) >= 2:
                # image_size is stored as (W, H) by lens_correction.py
                calib_W = int(img_size[0])
                calib_H = int(img_size[1])
            else:
                calib_H = calib_W = None
            alpha = 0.0 if cfg["crop_to_valid"] else 1.0

            images = self._gather_images_from(
                cfg["input_folder"], recursive=cfg["recursive"],
                exclude_bad=cfg["exclude_bad"], bad_list=cfg["bad_list"])
            if not images:
                self._ui_message("warning", "Warning", "No images remaining.")
                return

            mode_str = ("Crop to valid pixels (alpha=0)"
                        if cfg["crop_to_valid"] else
                        "Keep all pixels (alpha=1)")
            print(f"\nLens correction: {len(images)} images")
            print(f"  Calibration: {os.path.basename(cfg['calib_path'])}")
            if calib_W and calib_H:
                print(f"  Calibrated size: {calib_W} x {calib_H} px")
            print(f"  Mode: {mode_str}")

            # Up-front compatibility check against the first image
            sample = imread_safe(str(images[0]))
            if sample is not None and calib_W and calib_H:
                sH, sW = sample.shape[:2]
                if (sH, sW) != (calib_H, calib_W):
                    # Check for 90° rotation (same sensor, portrait↔landscape)
                    is_rotated = ((sW, sH) == (calib_H, calib_W) or
                                  (sH, sW) == (calib_W, calib_H))
                    if is_rotated:
                        print(f"  [INFO] First image size {sW}x{sH} is a 90° "
                              f"rotation of calibration {calib_W}x{calib_H}. "
                              f"Images will be rotated → undistorted → "
                              f"rotated back.")
                    else:
                        aspect_calib = calib_W / calib_H
                        aspect_sample = sW / sH
                        if abs(aspect_calib - aspect_sample) < 0.01:
                            print(f"  [INFO] First image size {sW}x{sH} differs "
                                  f"from calibration {calib_W}x{calib_H}, but "
                                  f"aspect matches — K will be scaled per image.")
                        else:
                            print(f"  [WARN] First image size {sW}x{sH} has a "
                                  f"different aspect ratio from calibration "
                                  f"{calib_W}x{calib_H}. Mismatched images will "
                                  f"be skipped.")

            input_root = Path(cfg["input_folder"])
            counts = {"corrected": 0, "skipped": 0, "failed": 0}
            # Names behind the counts.  The console deliberately prints only
            # the first few, so without this the rest were unrecoverable.
            not_corrected = []
            start_time = time.time()
            map_cache = {}  # (W, H) -> (mapx, mapy)

            for idx, p in enumerate(images):
                if self._cancel_requested:
                    self._ui_set_eta("Cancelled")
                    print("Lens correction cancelled.")
                    return
                img = imread_safe(str(p))
                if img is None:
                    counts["failed"] += 1
                    not_corrected.append((str(p), "unreadable"))
                    print(f"  [SKIP] {p.name}: cannot read")
                    self._update_progress(idx, len(images), start_time)
                    continue
                iH, iW = img.shape[:2]

                # Detect 90° rotation (portrait↔landscape from same sensor)
                rotated_90 = False
                if (calib_W and calib_H and
                    (iH, iW) != (calib_H, calib_W) and
                    (iW == calib_H and iH == calib_W)):
                    rotated_90 = True

                # If rotated, rotate image to match calibration orientation,
                # undistort, then rotate back.
                if rotated_90:
                    # Rotate 90° CW so image dims match (calib_H, calib_W)
                    img_work = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                    work_H, work_W = img_work.shape[:2]
                else:
                    img_work = img
                    work_H, work_W = iH, iW

                # Decide K for this (possibly rotated) image
                Ki = K
                if calib_W and calib_H and (work_H, work_W) != (calib_H, calib_W):
                    aspect_calib = calib_W / calib_H
                    aspect_img = work_W / work_H
                    if abs(aspect_calib - aspect_img) > 0.01:
                        counts["skipped"] += 1
                        not_corrected.append(
                            (str(p), f"aspect mismatch ({iW}x{iH} vs calib "
                                     f"{calib_W}x{calib_H})"))
                        if counts["skipped"] <= 5:
                            print(f"  [SKIP] {p.name}: aspect mismatch "
                                  f"({iW}x{iH} vs calib {calib_W}x{calib_H})")
                        self._update_progress(idx, len(images), start_time)
                        continue
                    sx = work_W / calib_W
                    sy = work_H / calib_H
                    Ki = K.copy()
                    Ki[0, 0] *= sx  # fx
                    Ki[1, 1] *= sy  # fy
                    Ki[0, 2] *= sx  # cx
                    Ki[1, 2] *= sy  # cy
                Di = D

                cache_key = (work_W, work_H)
                if cache_key not in map_cache:
                    new_K, _ = cv2.getOptimalNewCameraMatrix(
                        Ki, Di, (work_W, work_H), alpha, (work_W, work_H))
                    mapx, mapy = cv2.initUndistortRectifyMap(
                        Ki, Di, None, new_K, (work_W, work_H), cv2.CV_32FC1)
                    map_cache[cache_key] = (mapx, mapy)
                mapx, mapy = map_cache[cache_key]
                corrected = cv2.remap(img_work, mapx, mapy,
                                       interpolation=cv2.INTER_LINEAR)

                # Rotate back to original orientation if we rotated earlier
                if rotated_90:
                    corrected = cv2.rotate(corrected,
                                           cv2.ROTATE_90_COUNTERCLOCKWISE)

                out_path = self._build_preserved_output_path(
                    p, input_root, cfg["output_folder"], "lens_corrected")
                _save_preserving_tiff(out_path, corrected, _read_tiff_metadata(p), source_path=p)
                counts["corrected"] += 1

                if (idx + 1) % 20 == 0 or idx == 0:
                    print(f"  {idx + 1}/{len(images)} | {p.name}")
                self._update_progress(idx, len(images), start_time)

            elapsed = time.time() - start_time
            print(f"\nLens correction complete "
                  f"({format_eta(elapsed)} elapsed):")
            print(f"  corrected: {counts['corrected']}")
            if counts["skipped"] > 0:
                print(f"  skipped:   {counts['skipped']} "
                      f"(size / aspect mismatch)")
            if counts["failed"] > 0:
                print(f"  failed:    {counts['failed']} (unreadable)")

            out_root = self._build_output_root(
                cfg["output_folder"], input_root, "lens_corrected")
            report_path = None
            if not_corrected:
                try:
                    report_path = os.path.join(cfg["output_folder"],
                                               "lens_correction_skipped.txt")
                    with open(report_path, "w", encoding="utf-8") as fh:
                        fh.write("Lens correction - images not corrected\n")
                        fh.write(f"Run: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                        fh.write(f"Input folder: {cfg['input_folder']}\n")
                        fh.write(f"Corrected: {counts['corrected']}   "
                                 f"Skipped: {counts['skipped']}   "
                                 f"Failed: {counts['failed']}\n\n")
                        for name, reason in not_corrected:
                            fh.write(f"{name}\t{reason}\n")
                    print(f"  Skipped-image list saved to: {report_path}")
                except Exception as e:
                    report_path = None
                    print(f"  Could not write skipped-image list: {e}")
            self._ui_call(self._render_lens_results, counts, elapsed,
                          os.path.basename(cfg["calib_path"]))
            self._ui_set_eta("Done")
            done_msg = f"Lens-corrected images saved to:\n{out_root}"
            if not_corrected:
                done_msg += (f"\n\n{counts['corrected']} corrected, "
                             f"{counts['skipped']} skipped, "
                             f"{counts['failed']} unreadable.")
                if report_path:
                    done_msg += f"\n\nNot corrected: {report_path}"
            self._ui_message("info", "Done", done_msg)

        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback
            traceback.print_exc()
            self._ui_message("error", "Error", str(e))
        finally:
            self._processing = False


# ——— standalone ———
if __name__ == "__main__":
    root = ctk.CTk()
    root.withdraw()
    HarmoniseImagesWindow(master=root)
    root.mainloop()