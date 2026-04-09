import json
import os
import glob
import threading
import time
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageOps
from PIL.PngImagePlugin import PngInfo
import re
import cv2
import numpy as np
import tifffile
import customtkinter as ctk
import rasterio
from scipy.interpolate import interp1d
import concurrent.futures

from utils import (
    fit_geometry,
    resource_path,
    setup_console,
    restore_console,
    save_settings_json,
    load_settings_json,
    bring_child_to_front,
    format_eta,
    compute_eta,
    make_selector_payload,
)

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")

# %% helpers

# Robust timestamp parsing
# -------------------------------------------------------------------------------
_PATTERNS = [
    (re.compile(r'(?P<y>\d{4})[_-](?P<m>\d{2})[_-](?P<d>\d{2})[_-](?P<H>\d{2})[_-](?P<M>\d{2})[_-](?P<S>\d{2})[_-](?P<ms>\d{1,3})'),
     ("y", "m", "d", "H", "M", "S", "ms")),
    (re.compile(r'(?P<y>\d{4})[_-](?P<m>\d{2})[_-](?P<d>\d{2})[_-](?P<H>\d{2})[_-](?P<M>\d{2})[_-](?P<S>\d{2})'),
     ("y", "m", "d", "H", "M", "S")),
    (re.compile(r'(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})[T_](?P<H>\d{2})[:\-](?P<M>\d{2})[:\-](?P<S>\d{2})(?:[.\-](?P<ms>\d{1,3}))?Z?'),
     ("y", "m", "d", "H", "M", "S", "ms")),
    (re.compile(r'(?P<y>\d{4})(?P<m>\d{2})(?P<d>\d{2})[_-](?P<H>\d{2})(?P<M>\d{2})(?P<S>\d{2})(?:[.\-](?P<ms>\d{1,3}))?'),
     ("y", "m", "d", "H", "M", "S", "ms")),
]

IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff")
SUPPORTED_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
CANCELLED_ERROR = "__CANCELLED__"


def _from_exif_or_tiff(path):
    """Try EXIF/TIFF DateTime if filename parsing fails."""
    try:
        if path.lower().endswith((".jpg", ".jpeg", ".png")):
            img = Image.open(path)
            img = ImageOps.exif_transpose(img)
            exif = img.getexif()
            if exif:
                for tag in (36867, 306):
                    v = exif.get(tag)
                    if v:
                        try:
                            return datetime.strptime(str(v), "%Y:%m:%d %H:%M:%S"), f"EXIF:{tag}"
                        except Exception:
                            pass
        elif path.lower().endswith((".tif", ".tiff")):
            with tifffile.TiffFile(path) as tf:
                dt = tf.pages[0].tags.get('DateTime')
                if dt:
                    try:
                        return datetime.strptime(dt.value, "%Y:%m:%d %H:%M:%S"), "TIFF:DateTime"
                    except Exception:
                        pass
    except Exception:
        pass
    return None


def parse_dt_from_name(path):
    """Return (datetime, src) parsed from filename, EXIF/TIFF, or file mtime."""
    name = os.path.splitext(os.path.basename(path))[0]
    for rx, _fields in _PATTERNS:
        m = rx.search(name)
        if not m:
            continue
        g = m.groupdict()
        y = int(g["y"])
        mo = int(g["m"])
        d = int(g["d"])
        H = int(g["H"])
        Mi = int(g["M"])
        S = int(g["S"])
        ms = int(g.get("ms") or 0)
        if 0 < ms < 100:
            ms = int(f"{ms:0<3}")
        try:
            return datetime(y, mo, d, H, Mi, S, ms * 1000), f"FN:{m.group(0)}"
        except ValueError:
            continue

    md = _from_exif_or_tiff(path)
    if md:
        return md

    try:
        return datetime.fromtimestamp(os.path.getmtime(path)), "FS:mtime"
    except Exception:
        pass

    raise ValueError(f"Could not parse/derive datetime for: {os.path.basename(path)}")


def collect_dated_files(files):
    """Return (sorted_files, sorted_datetimes). Skips files we can't date."""
    parsed = []
    for f in files:
        try:
            dt, _src = parse_dt_from_name(f)
            parsed.append((dt, f))
        except Exception as e:
            print(f"[warn] Skipping file w/o timestamp: {os.path.basename(f)} ({e})")
    if not parsed:
        raise ValueError("No parsable timestamps in the input set.")
    parsed.sort(key=lambda t: t[0])
    files_sorted = [f for _, f in parsed]
    dts_sorted = [dt for dt, _ in parsed]
    return files_sorted, dts_sorted


def collect_images(folder):
    imgs = []
    for ext in IMAGE_EXTS:
        imgs.extend(glob.glob(os.path.join(folder, ext)))
    return imgs


def _read_rgb_image(path):
    img = tifffile.imread(path) if path.lower().endswith((".tif", ".tiff")) else np.array(Image.open(path).convert("RGB"))
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def _ensure_rgb(img):
    if img.ndim == 2:
        return np.stack([img] * 3, axis=-1)
    if img.shape[2] > 3:
        return img[:, :, :3]
    return img


def _extract_profile(img, x1, y1, x2, y2, width=1):
    dx, dy = x2 - x1, y2 - y1
    length = max(int(np.hypot(dx, dy)), 1)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    t = np.arange(length)
    cx, cy = x1 + t * ux, y1 + t * uy
    half_w = (width - 1) / 2.0
    offsets = np.linspace(-half_w, half_w, max(width, 1))
    h, w = img.shape[:2]
    is_rgb = img.ndim == 3
    profile = np.zeros((length, 3) if is_rgb else length, dtype=np.float64)
    count = np.zeros(length, dtype=np.float64)
    for off in offsets:
        sx = np.clip((cx + off * px).astype(int), 0, w - 1)
        sy = np.clip((cy + off * py).astype(int), 0, h - 1)
        if is_rgb:
            profile += img[sy, sx, :]
        else:
            profile += img[sy, sx]
        count += 1.0
    if is_rgb:
        profile /= count[:, None]
    else:
        profile /= count
    return profile


def _extract_polyline_profile(img, points, width=1):
    pts = [(float(x), float(y)) for x, y in points]
    if len(pts) < 2:
        raise ValueError("Freehand line requires at least 2 points.")
    pieces = []
    for i in range(len(pts) - 1):
        x1, y1 = pts[i]
        x2, y2 = pts[i + 1]
        seg = _extract_profile(img, x1, y1, x2, y2, width=width)
        if i > 0 and len(seg) > 0:
            seg = seg[1:]
        if len(seg) > 0:
            pieces.append(seg)
    if not pieces:
        raise ValueError("Freehand line selection is empty.")
    return np.vstack(pieces)


def _selector_mode(selector):
    if isinstance(selector, tuple):
        return "bbox"
    return str((selector or {}).get("mode", "bbox")).lower()


def _selector_bbox(selector):
    if isinstance(selector, tuple):
        return selector
    vals = (selector or {}).get("bbox_px") or (selector or {}).get("bbox")
    if not vals or len(vals) < 4:
        return None
    return tuple(int(round(v)) for v in vals[:4])


def _selector_points(selector):
    if isinstance(selector, tuple):
        return []
    pts = (selector or {}).get("points_px") or (selector or {}).get("points") or []
    return [(float(p[0]), float(p[1])) for p in pts if len(p) >= 2]


def _selector_line_width(selector):
    if isinstance(selector, tuple):
        return 1
    try:
        return max(1, int((selector or {}).get("line_width", 1)))
    except Exception:
        return 1


def _selector_length(selector):
    mode = _selector_mode(selector)
    if mode == "bbox":
        bbox = _selector_bbox(selector)
        if not bbox:
            return 0
        return int(bbox[2])
    pts = _selector_points(selector)
    if len(pts) < 2:
        return 0
    total = 0
    for i in range(len(pts) - 1):
        total += max(int(np.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])), 1)
    if mode == "freehand" and len(pts) > 2:
        total -= (len(pts) - 2)
    return max(total, 1)


def _extract_line_from_selector(img, selector):
    mode = _selector_mode(selector)
    if mode == "bbox":
        bbox = _selector_bbox(selector)
        if not bbox:
            raise ValueError("No bounding box selected.")
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            raise ValueError(f"Invalid bounding box {bbox}: width and height must be > 0")
        roi = img[y:y + h, x:x + w]
        if roi.size == 0:
            return np.full((w, 3), np.nan, dtype=float)
        roi = _ensure_rgb(roi)
        return np.round(np.mean(roi, axis=0)).astype(float)

    pts = _selector_points(selector)
    if len(pts) < 2:
        raise ValueError("Line selection requires at least 2 points.")
    width = _selector_line_width(selector)
    img = _ensure_rgb(img)
    if mode == "line":
        x1, y1 = pts[0]
        x2, y2 = pts[1]
        return _extract_profile(img, x1, y1, x2, y2, width=width).astype(float)
    if mode == "freehand":
        return _extract_polyline_profile(img, pts, width=width).astype(float)
    raise ValueError(f"Unsupported selector mode: {mode}")


def _selector_pnginfo(info, selector, resolution_x_m):
    mode = _selector_mode(selector)
    info.add_text("pixel_resolution", f"{resolution_x_m:.6f}")
    info.add_text("selector_mode", mode)
    if mode == "bbox":
        bbox = _selector_bbox(selector)
        if bbox:
            x, y, w, h = bbox
            info.add_text("bounding_box", f"{x},{y},{w},{h}")
    else:
        info.add_text("selector_points_px", json.dumps(_selector_points(selector)))
        info.add_text("selector_line_width", str(_selector_line_width(selector)))


def generate_with_fill(image_files, selector, resolution_x_m, output_path,
                       freq_hz=1.0, duration_s=600.0, progress_callback=None,
                       cancel_callback=None):
    expected_len = _selector_length(selector)
    if expected_len <= 0:
        raise ValueError("Invalid selector geometry.")

    files_sorted, dts_sorted = collect_dated_files(image_files)
    N = int(duration_s * freq_hz)
    dt_start = dts_sorted[0]
    tasks = [None] * N
    tasks_dt = [None] * N

    for dt, fn in zip(dts_sorted, files_sorted):
        offset = (dt - dt_start).total_seconds()
        raw_idx = offset * freq_hz
        idx = int(round(raw_idx))
        if 0 <= idx < N:
            if tasks[idx] is None:
                tasks[idx] = fn
                tasks_dt[idx] = dt
            else:
                ideal = dt_start + timedelta(seconds=idx / freq_hz)
                if abs((dt - ideal).total_seconds()) < abs((tasks_dt[idx] - ideal).total_seconds()):
                    tasks[idx] = fn
                    tasks_dt[idx] = dt

    lines = []
    for i, fn in enumerate(tasks, start=1):
        if cancel_callback and cancel_callback():
            raise RuntimeError(CANCELLED_ERROR)
        if progress_callback:
            progress_callback(i, N)
        if fn is None or not os.path.exists(fn):
            line = np.full((expected_len, 3), np.nan, dtype=float)
        else:
            img = _read_rgb_image(fn)
            try:
                line = _extract_line_from_selector(img, selector)
            except Exception:
                line = np.full((expected_len, 3), np.nan, dtype=float)
            if line.shape[0] != expected_len:
                if line.shape[0] <= 0:
                    line = np.full((expected_len, 3), np.nan, dtype=float)
                else:
                    idx_old = np.linspace(0, 1, line.shape[0])
                    idx_new = np.linspace(0, 1, expected_len)
                    resampled = np.zeros((expected_len, 3), dtype=float)
                    for c in range(3):
                        resampled[:, c] = np.interp(idx_new, idx_old, line[:, c])
                    line = resampled
        lines.append(line)

    ts = np.stack(lines, axis=0)
    nan_rows_before = np.all(np.isnan(ts), axis=(1, 2))

    tail_nan = 0
    for flag in nan_rows_before[::-1]:
        if flag:
            tail_nan += 1
        else:
            break

    extrapolate_tail = True
    if tail_nan > 5:
        print(f"Warning: {tail_nan} missing rows at tail; not extrapolating, leaving as NaN")
        extrapolate_tail = False

    idxs = np.arange(N)
    for j in range(expected_len):
        for c in range(3):
            col = ts[:, j, c]
            nanmask = np.isnan(col)
            if not nanmask.any():
                continue
            valid = ~nanmask
            vidx = idxs[valid]
            vvals = col[valid]
            if vidx.size == 0:
                continue
            first, last = vidx[0], vidx[-1]
            if first > 0:
                col[:first] = vvals[0]
            if last < N - 1 and extrapolate_tail:
                col[last + 1:] = vvals[-1]
            if last - first >= 1:
                f = interp1d(vidx, vvals, kind='cubic', bounds_error=False)
                mid = idxs[first:last + 1][nanmask[first:last + 1]]
                if mid.size > 0:
                    col[mid] = f(mid)
            ts[:, j, c] = col

    nan_rows_after = np.all(np.isnan(ts), axis=(1, 2))
    filled = int(((nan_rows_before) & (~nan_rows_after)).sum())
    remaining = int(nan_rows_after.sum())
    if filled > 0:
        print(f"Filled {filled} missing rows via interpolation/extrapolation.")
    if remaining > 0:
        print(f"Remaining {remaining} NaN rows (tail), left unfilled.")

    ts = ts[::-1, :, :]
    ts = np.nan_to_num(ts, nan=0)
    ts = np.clip(ts, 0, 255).astype(np.uint8)

    out_img = Image.fromarray(ts)
    if out_img.width != expected_len:
        out_img = out_img.resize((expected_len, ts.shape[0]), Image.NEAREST)

    info = PngInfo()
    _selector_pnginfo(info, selector, resolution_x_m)
    out_img.save(output_path, format="PNG", pnginfo=info)
    return output_path


def generate_no_fill(image_files, selector, resolution_x_m, output_path,
                     progress_callback=None, cancel_callback=None):
    files_sorted, _ = collect_dated_files(image_files)
    expected_len = _selector_length(selector)
    if expected_len <= 0:
        raise ValueError("Invalid selector geometry.")

    total = len(files_sorted)
    lines = []
    for i, fn in enumerate(files_sorted, start=1):
        if cancel_callback and cancel_callback():
            raise RuntimeError(CANCELLED_ERROR)
        if progress_callback:
            progress_callback(i, total)
        img = _read_rgb_image(fn)
        try:
            line = _extract_line_from_selector(img, selector)
        except Exception:
            continue
        if line.shape[0] != expected_len:
            idx_old = np.linspace(0, 1, line.shape[0])
            idx_new = np.linspace(0, 1, expected_len)
            resampled = np.zeros((expected_len, 3), dtype=float)
            for c in range(3):
                resampled[:, c] = np.interp(idx_new, idx_old, line[:, c])
            line = resampled
        lines.append(np.clip(np.round(line), 0, 255).astype(np.uint8))

    if not lines:
        raise ValueError("No valid ROI/profile slices—nothing to stack.")

    ts = np.stack(lines, axis=0)[::-1, :, :]
    out = Image.fromarray(ts)
    if out.width != expected_len:
        out = out.resize((expected_len, ts.shape[0]), Image.NEAREST)
    info = PngInfo()
    _selector_pnginfo(info, selector, resolution_x_m)
    out.save(output_path, format="PNG", pnginfo=info)
    return output_path


def _process_subfolder(sub_path: str, selector: dict, res_x: float,
                       freq: float, dur: float, fill_gaps: bool,
                       output_folder: str, cancel_callback=None) -> tuple:
    """
    Runs inside a ThreadPoolExecutor worker.
    Returns (sub_path, status, message) where status ∈
       {"processed", "skipped", "error", "no_imgs", "cancelled"}.
    """
    try:
        if cancel_callback and cancel_callback():
            return sub_path, "cancelled", "Cancelled"
        imgs = collect_images(sub_path)
        if not imgs:
            return sub_path, "no_imgs", "No supported images"

        try:
            files_sorted, dts_sorted = collect_dated_files(imgs)
            first_ts = dts_sorted[0].strftime("%Y_%m_%d_%H_%M")
            imgs = files_sorted
        except Exception:
            imgs.sort()
            first_ts = "_".join(os.path.basename(imgs[0]).split("_")[0:5])

        out_name = f"{first_ts}_raw_timestack.png"
        out_path = os.path.join(output_folder, out_name)

        if os.path.exists(out_path):
            return sub_path, "skipped", "Already processed"

        if fill_gaps:
            generate_with_fill(imgs, selector, res_x, out_path,
                               freq_hz=freq, duration_s=dur,
                               cancel_callback=cancel_callback)
        else:
            generate_no_fill(imgs, selector, res_x, out_path,
                             cancel_callback=cancel_callback)
        return sub_path, "processed", out_name
    except Exception as exc:
        if str(exc) == CANCELLED_ERROR:
            return sub_path, "cancelled", "Cancelled"
        return sub_path, "error", str(exc)


class ScrollZoomSelector(tk.Frame):
    """Scrollable/zoomable selector supporting bbox, straight line, and freehand."""

    def __init__(self, master, mode_var=None, line_width_var=None, **kwargs):
        super().__init__(master, **kwargs)
        master.title("Scrollable & Zoomable ROI / Line Selector")
        self.mode_var = mode_var or tk.StringVar(master=master, value="bbox")
        self.line_width_var = line_width_var or tk.StringVar(master=master, value="1")

        top_frame = tk.Frame(self)
        top_frame.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(top_frame, cursor="cross", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.v_scroll = tk.Scrollbar(top_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.h_scroll = tk.Scrollbar(self, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.configure(xscrollcommand=self.h_scroll.set, yscrollcommand=self.v_scroll.set)

        self.cv_image = None
        self.pil_image = None
        self.tk_image = None
        self.scale_factor = 1.0

        self.start_x_display = None
        self.start_y_display = None
        self.rect_id = None
        self.line_id = None
        self.freehand_id = None
        self.bbox = (0, 0, 0, 0)
        self.line_points = []
        self.freehand_points = []

        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        self.master.bind("<Return>", self.on_enter_key)

        button_frame = tk.Frame(self)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        tk.Button(button_frame, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Zoom In", command=self.zoom_in).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Zoom Out", command=self.zoom_out).pack(side=tk.LEFT, padx=5)
        tk.Label(button_frame, text="Mode:").pack(side=tk.LEFT, padx=(15, 3))
        tk.OptionMenu(button_frame, self.mode_var, "bbox", "line", "freehand").pack(side=tk.LEFT, padx=3)
        tk.Label(button_frame, text="Line width:").pack(side=tk.LEFT, padx=(12, 3))
        tk.Entry(button_frame, textvariable=self.line_width_var, width=4).pack(side=tk.LEFT, padx=3)
        tk.Label(button_frame, text="Drag to select; press Enter to confirm").pack(side=tk.LEFT, padx=10)

    def load_image(self, file_path=None):
        if file_path is None:
            file_path = filedialog.askopenfilename(
                title="Open Image",
                filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tif *.gif")],
            )
        if not file_path:
            return
        self.cv_image = cv2.imread(file_path)
        if self.cv_image is None:
            messagebox.showerror("Error", "Failed to load image.")
            return
        cv_rgb = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
        self.pil_image = Image.fromarray(cv_rgb)
        self.scale_factor = 1.0
        self.display_image()
        self.clear_selection()

    def clear_selection(self):
        self.bbox = (0, 0, 0, 0)
        self.line_points = []
        self.freehand_points = []
        for shape_id in (self.rect_id, self.line_id, self.freehand_id):
            if shape_id:
                self.canvas.delete(shape_id)
        self.rect_id = None
        self.line_id = None
        self.freehand_id = None

    def display_image(self):
        if self.pil_image is None:
            return
        w0, h0 = self.pil_image.size
        w = int(w0 * self.scale_factor)
        h = int(h0 * self.scale_factor)
        pil_resized = self.pil_image.resize((w, h), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(pil_resized)
        self.canvas.delete("all")
        self.canvas.config(scrollregion=(0, 0, w, h))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.rect_id = None
        self.line_id = None
        self.freehand_id = None

    def zoom_in(self):
        if self.pil_image is None:
            return
        self.scale_factor *= 1.25
        self.display_image()
        self.clear_selection()

    def zoom_out(self):
        if self.pil_image is None:
            return
        self.scale_factor = max(self.scale_factor * 0.8, 0.1)
        self.display_image()
        self.clear_selection()

    def on_button_press(self, event):
        mode = self.mode_var.get().lower()
        self.start_x_display = self.canvas.canvasx(event.x)
        self.start_y_display = self.canvas.canvasy(event.y)
        if mode == "bbox":
            if self.rect_id:
                self.canvas.delete(self.rect_id)
            self.rect_id = self.canvas.create_rectangle(
                self.start_x_display, self.start_y_display,
                self.start_x_display, self.start_y_display,
                outline="red", width=2,
            )
        elif mode == "line":
            if self.line_id:
                self.canvas.delete(self.line_id)
            self.line_id = self.canvas.create_line(
                self.start_x_display, self.start_y_display,
                self.start_x_display, self.start_y_display,
                fill="cyan", width=2,
            )
        elif mode == "freehand":
            self.freehand_points = [(self.start_x_display, self.start_y_display)]
            if self.freehand_id:
                self.canvas.delete(self.freehand_id)
            self.freehand_id = self.canvas.create_line(
                self.start_x_display, self.start_y_display,
                self.start_x_display, self.start_y_display,
                fill="lime", width=2, smooth=False,
            )

    def on_move_press(self, event):
        mode = self.mode_var.get().lower()
        cur_x = self.canvas.canvasx(event.x)
        cur_y = self.canvas.canvasy(event.y)
        if mode == "bbox" and self.rect_id:
            self.canvas.coords(self.rect_id, self.start_x_display, self.start_y_display, cur_x, cur_y)
        elif mode == "line" and self.line_id:
            self.canvas.coords(self.line_id, self.start_x_display, self.start_y_display, cur_x, cur_y)
        elif mode == "freehand" and self.freehand_id:
            self.freehand_points.append((cur_x, cur_y))
            flat = [coord for pt in self.freehand_points for coord in pt]
            self.canvas.coords(self.freehand_id, *flat)

    def on_button_release(self, event):
        mode = self.mode_var.get().lower()
        end_x = self.canvas.canvasx(event.x)
        end_y = self.canvas.canvasy(event.y)
        if mode == "bbox":
            x1_disp, x2_disp = sorted([self.start_x_display, end_x])
            y1_disp, y2_disp = sorted([self.start_y_display, end_y])
            x1 = int(x1_disp / self.scale_factor)
            y1 = int(y1_disp / self.scale_factor)
            w = int((x2_disp - x1_disp) / self.scale_factor)
            h = int((y2_disp - y1_disp) / self.scale_factor)
            self.bbox = (x1, y1, w, h)
        elif mode == "line":
            self.line_points = [
                (int(self.start_x_display / self.scale_factor), int(self.start_y_display / self.scale_factor)),
                (int(end_x / self.scale_factor), int(end_y / self.scale_factor)),
            ]
        elif mode == "freehand":
            self.freehand_points.append((end_x, end_y))
            self.freehand_points = [
                (int(x / self.scale_factor), int(y / self.scale_factor))
                for x, y in self.freehand_points
            ]

    def get_selector(self):
        mode = self.mode_var.get().lower()
        try:
            line_width = max(1, int(self.line_width_var.get()))
        except Exception:
            line_width = 1
        if mode == "bbox":
            return {"mode": "bbox", "bbox_px": list(self.bbox), "line_width": line_width}
        if mode == "line":
            return {"mode": "line", "points_px": [list(p) for p in self.line_points[:2]], "line_width": line_width}
        return {"mode": "freehand", "points_px": [list(p) for p in self.freehand_points], "line_width": line_width}

    def on_enter_key(self, _):
        print("Final selector:", self.get_selector())


class TimestackTool(ctk.CTkToplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("Time-stacking Tool")
        fit_geometry(self, 1200, 800, resizable=True)
        try:
            self.after(200, lambda: self.iconphoto(False, tk.PhotoImage(file=resource_path("launch_logo.png"))))
        except Exception:
            pass

        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.input_folder = tk.StringVar()
        self.output_folder = tk.StringVar()
        self.batch_folder = tk.StringVar()

        self.selector_state = None
        self.bbox = None  # backward-compatible alias for bbox mode
        self.selector_text = tk.StringVar()
        self.add_selector_text = tk.BooleanVar(value=False)
        self.selector_mode_var = tk.StringVar(value="bbox")
        self.line_width_var = tk.StringVar(value="1")

        self.default_resolution = 0.25
        self.identified_res_var = tk.StringVar(value="Not identified")
        self.add_resolution_manual = tk.BooleanVar(value=False)

        self.fill_gaps = tk.BooleanVar(value=True)
        self.freq_var = tk.DoubleVar(value=1.0)
        self.duration_var = tk.DoubleVar(value=600.0)

        self._cancel_requested = False
        self._single_running = False
        self._batch_running = False
        self._batch_executor = None
        self._single_start_time = None
        self._batch_start_time = None

        self._build_ui()
        self._console_redir = setup_console(
            self.console_text,
            "Here you may see console outputs\n--------------------------------\n"
        )

    def _on_close(self):
        self._request_cancel()
        restore_console(getattr(self, "_console_redir", None))
        self.destroy()

    def _request_cancel(self):
        self._cancel_requested = True
        try:
            if self._batch_executor is not None:
                self._batch_executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass

    def _ui_call(self, func, *args, **kwargs):
        try:
            self.after(0, lambda: func(*args, **kwargs))
        except Exception:
            pass

    def _ui_message(self, kind, title, message):
        fn = getattr(messagebox, f"show{kind}")
        self._ui_call(fn, title, message)

    def _ui_single_progress(self, current, total):
        total = max(int(total), 1)
        frac = max(0.0, min(1.0, float(current) / float(total)))
        self._ui_call(self.single_pb.set, frac)
        eta = compute_eta(self._single_start_time or time.time(), int(current), int(total)) if self._single_start_time else None
        self._ui_call(self.single_lbl.configure, text=f"{current} / {total} — ETA {format_eta(eta)}")

    def _apply_single_preview(self, image_path):
        pil = Image.open(image_path)
        preview = ctk.CTkImage(light_image=pil, size=(800, min(600, pil.height)))
        self.image_label.configure(image=preview)
        self.image_label.image = preview

    def _ui_batch_progress(self, fraction, label_text):
        self._ui_call(self.batch_pb.set, fraction)
        self._ui_call(self.batch_lbl.configure, text=label_text)

    def _build_ui(self):
        self.top_frame = ctk.CTkFrame(self, height=400, fg_color="black")
        self.top_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.image_label = ctk.CTkLabel(self.top_frame, text="")
        self.image_label.pack(expand=True)

        cf = ctk.CTkFrame(self)
        cf.pack(fill="both", expand=False, padx=10, pady=(0, 10))
        self.console_text = tk.Text(cf, wrap="word", height=10)
        self.console_text.pack(fill="both", expand=True, padx=5, pady=5)

        self.bottom_frame = ctk.CTkFrame(self, height=250)
        self.bottom_frame.pack(fill="x", padx=10, pady=(0, 10))

        ip = ctk.CTkFrame(self.bottom_frame)
        ip.pack(fill="x", padx=5, pady=5)
        ctk.CTkButton(ip, text="Browse Input Folder", command=self.browse_input_folder).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ctk.CTkLabel(ip, textvariable=self.input_folder).grid(row=0, column=1, padx=5, pady=5, sticky="w")
        ctk.CTkButton(ip, text="Select ROI / Line", command=self.select_bbox).grid(row=0, column=2, padx=5, pady=5)
        ctk.CTkCheckBox(ip, text="Add selector as text", variable=self.add_selector_text,
                        command=self.toggle_selector_entry).grid(row=0, column=3, padx=5, pady=5)
        self.selector_entry = ctk.CTkEntry(ip, textvariable=self.selector_text, width=360)
        self.selector_entry.grid(row=0, column=4, padx=5, pady=5)
        self.selector_entry.configure(state="disabled")

        rp = ctk.CTkFrame(self.bottom_frame)
        rp.pack(fill="x", padx=5, pady=5)
        ctk.CTkLabel(rp, text="Identified resolution:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ctk.CTkLabel(rp, textvariable=self.identified_res_var).grid(row=0, column=1, padx=5, pady=5, sticky="w")
        ctk.CTkCheckBox(rp, text="Add pixel resolution manually", variable=self.add_resolution_manual,
                        command=self.toggle_resolution_entry).grid(row=0, column=2, padx=5, pady=5)
        self.resolution_entry = ctk.CTkEntry(rp)
        self.resolution_entry.grid(row=0, column=3, padx=5, pady=5)
        self.resolution_entry.configure(state="disabled")
        ctk.CTkLabel(rp, text="m").grid(row=0, column=4, padx=5, pady=5, sticky="w")
        ctk.CTkCheckBox(rp, text="Fill gaps", variable=self.fill_gaps).grid(row=0, column=5, padx=15, pady=5)
        ctk.CTkLabel(rp, text="Freq (Hz):").grid(row=0, column=6, padx=5, pady=5, sticky="w")
        ctk.CTkEntry(rp, textvariable=self.freq_var, width=60).grid(row=0, column=7, padx=5, pady=5, sticky="w")
        ctk.CTkLabel(rp, text="Dur (s):").grid(row=0, column=8, padx=5, pady=5, sticky="w")
        ctk.CTkEntry(rp, textvariable=self.duration_var, width=60).grid(row=0, column=9, padx=5, pady=5, sticky="w")
        ctk.CTkLabel(rp, text="Line width:").grid(row=0, column=10, padx=(15, 5), pady=5, sticky="w")
        ctk.CTkEntry(rp, textvariable=self.line_width_var, width=50).grid(row=0, column=11, padx=5, pady=5, sticky="w")

        op = ctk.CTkFrame(self.bottom_frame)
        op.pack(fill="x", padx=5, pady=5)
        ctk.CTkButton(op, text="Browse Output Folder", command=self.select_output_folder, fg_color="#8C7738", hover_color="#A18A45").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ctk.CTkLabel(op, textvariable=self.output_folder).grid(row=0, column=1, padx=5, pady=5, sticky="w")
        ctk.CTkButton(op, text="Create Raw Timestack", command=self.create_timestack, fg_color="#0F52BA", hover_color="#2A6BD1").grid(row=0, column=2, padx=5, pady=5)
        self.single_pb = ctk.CTkProgressBar(op)
        self.single_pb.grid(row=0, column=3, padx=5, pady=5)
        self.single_pb.set(0)
        self.single_lbl = ctk.CTkLabel(op, text="")
        self.single_lbl.grid(row=0, column=4, padx=5, pady=5, sticky="w")
        ctk.CTkButton(op, text="Save Settings", command=self.save_settings, fg_color="#4F5D75",  hover_color="#61708A").grid(row=0, column=5, padx=(15, 5), pady=5)
        ctk.CTkButton(op, text="Load Settings", command=self.load_settings, fg_color="#4F5D75",  hover_color="#61708A").grid(row=0, column=6, padx=5, pady=5)

        bp = ctk.CTkFrame(self.bottom_frame)
        bp.pack(fill="x", padx=5, pady=5)
        ctk.CTkButton(bp, text="Select Batch Folder", command=self.browse_batch_folder).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ctk.CTkLabel(bp, textvariable=self.batch_folder).grid(row=0, column=1, padx=5, pady=5, sticky="w")
        ctk.CTkButton(bp, text="Batch Process", command=self.batch_process, fg_color="#0F52BA").grid(row=0, column=2, padx=5, pady=5)
        self.batch_pb = ctk.CTkProgressBar(bp)
        self.batch_pb.grid(row=0, column=3, padx=5, pady=5)
        self.batch_pb.set(0)
        self.batch_lbl = ctk.CTkLabel(bp, text="")
        self.batch_lbl.grid(row=0, column=4, padx=5, pady=5, sticky="w")
        ctk.CTkButton(bp, text="Reset", command=self.reset_to_initial, fg_color="#8B0000", hover_color="#A52A2A", text_color="white").grid(row=0, column=5, padx=(15, 5), pady=5)

    def toggle_selector_entry(self):
        self.selector_entry.configure(state="normal" if self.add_selector_text.get() else "disabled")

    def toggle_resolution_entry(self):
        self.resolution_entry.configure(state="normal" if self.add_resolution_manual.get() else "disabled")

    def browse_input_folder(self):
        f = filedialog.askdirectory(parent=self, title="Select Input Folder with Images")
        if f:
            self.input_folder.set(f)

    def select_output_folder(self):
        f = filedialog.askdirectory(parent=self, title="Select Output Folder for Raw Timestack")
        if f:
            self.output_folder.set(f)

    def browse_batch_folder(self):
        f = filedialog.askdirectory(parent=self, title="Select Main Batch Folder (sub-folders per batch)")
        if f:
            self.batch_folder.set(f)

    def _first_reference_image(self, folder=None):
        fld = folder or self.input_folder.get().strip()
        if not fld:
            return None
        imgs = collect_images(fld)
        if not imgs:
            return None
        tifs = [p for p in imgs if p.lower().endswith((".tif", ".tiff"))]
        return sorted(tifs)[0] if tifs else sorted(imgs)[0]

    def _world_points_for_selector(self, selector):
        ref = self._first_reference_image()
        if not ref or not ref.lower().endswith((".tif", ".tiff")):
            return None, None, None
        try:
            with rasterio.open(ref) as src:
                if src.crs is None:
                    return None, None, None
                crs = str(src.crs)
                mode = _selector_mode(selector)
                if mode == "bbox":
                    bbox = _selector_bbox(selector)
                    if not bbox:
                        return crs, None, None
                    x, y, w, h = bbox
                    p1 = src.transform * (x, y)
                    p2 = src.transform * (x + w, y + h)
                    bbox_world = [float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1])]
                    return crs, bbox_world, None
                points_world = []
                for x, y in _selector_points(selector):
                    wx, wy = src.transform * (x, y)
                    points_world.append([float(wx), float(wy)])
                return crs, None, points_world
        except Exception:
            return None, None, None

    def _selector_to_text(self, selector=None):
        selector = selector or self.selector_state
        if not selector:
            return ""
        mode = _selector_mode(selector)
        if mode == "bbox":
            bbox = _selector_bbox(selector)
            if not bbox:
                return ""
            return f"bbox:{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
        pts = _selector_points(selector)
        pts_txt = ";".join(f"{int(round(x))},{int(round(y))}" for x, y in pts)
        return f"{mode}:{pts_txt};width={_selector_line_width(selector)}"

    def _normalise_selector(self, selector):
        if not selector:
            return None
        mode = _selector_mode(selector)
        if mode == "bbox":
            bbox = _selector_bbox(selector)
            if not bbox or bbox[2] <= 0 or bbox[3] <= 0:
                return None
            self.bbox = bbox
            return {"mode": "bbox", "bbox_px": list(bbox), "line_width": _selector_line_width(selector)}
        pts = _selector_points(selector)
        if len(pts) < 2:
            return None
        self.bbox = None
        return {
            "mode": mode,
            "points_px": [[float(x), float(y)] for x, y in pts],
            "line_width": _selector_line_width(selector),
        }

    def select_bbox(self):
        fld = self.input_folder.get().strip()
        if not fld:
            messagebox.showerror("Error", "Please select an input folder first.", parent=self)
            return
        imgs = collect_images(fld)
        if not imgs:
            messagebox.showerror("Error", "No supported image files found in folder.", parent=self)
            return
        win = tk.Toplevel(self)
        sel = ScrollZoomSelector(win, mode_var=self.selector_mode_var, line_width_var=self.line_width_var)
        sel.pack(fill="both", expand=True)
        sel.load_image(sorted(imgs)[0])
        ctk.CTkButton(win, text="Confirm", command=lambda: self._save_bbox(sel, win)).pack(pady=10)
        bring_child_to_front(win, self, modal=False)

    def _save_bbox(self, sel, win):
        selector = self._normalise_selector(sel.get_selector())
        if not selector:
            messagebox.showwarning("Warning", "No valid selector selected.", parent=win)
            return
        self.selector_state = selector
        self.selector_text.set(self._selector_to_text(selector))
        fld = self.input_folder.get().strip()
        tifs = sorted(glob.glob(os.path.join(fld, "*.tif")) + glob.glob(os.path.join(fld, "*.tiff")))
        if tifs:
            try:
                with rasterio.open(tifs[0]) as src:
                    res = abs(src.transform.a)
            except Exception:
                res = self.default_resolution
        else:
            res = self.default_resolution
        self.identified_res_var.set(f"{res:.3f} m")
        win.destroy()

    def _parse_selector_text(self):
        txt = self.selector_text.get().strip()
        try:
            if not txt:
                return False
            if ":" not in txt:
                p = [int(x.strip()) for x in txt.split(",")]
                if len(p) != 4:
                    raise ValueError("Must be 4 values: x,y,w,h")
                selector = {"mode": "bbox", "bbox_px": p, "line_width": _selector_line_width(self.selector_state)}
            else:
                mode, rest = txt.split(":", 1)
                mode = mode.strip().lower()
                width = _selector_line_width(self.selector_state)
                if ";width=" in rest:
                    rest, width_text = rest.rsplit(";width=", 1)
                    width = max(1, int(width_text.strip()))
                    self.line_width_var.set(str(width))
                if mode == "bbox":
                    p = [int(x.strip()) for x in rest.split(",")]
                    if len(p) != 4:
                        raise ValueError("BBox must be x,y,w,h")
                    selector = {"mode": "bbox", "bbox_px": p, "line_width": width}
                else:
                    pts = []
                    for part in rest.split(";"):
                        if not part.strip():
                            continue
                        x_str, y_str = part.split(",", 1)
                        pts.append([float(x_str.strip()), float(y_str.strip())])
                    selector = {"mode": mode, "points_px": pts, "line_width": width}
            selector = self._normalise_selector(selector)
            if not selector:
                raise ValueError("Selector is incomplete.")
            self.selector_state = selector
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Invalid selector: {e}", parent=self)
            return False

    def _current_selector(self):
        if self.add_selector_text.get() and self.selector_text.get().strip():
            if not self._parse_selector_text():
                return None
        return self._normalise_selector(self.selector_state)

    def _current_resolution(self):
        if self.add_resolution_manual.get():
            try:
                return float(self.resolution_entry.get())
            except Exception:
                raise ValueError("Invalid manual resolution.")
        try:
            return float(self.identified_res_var.get().split()[0])
        except Exception:
            return self.default_resolution

    def _current_freq_duration(self):
        try:
            freq = float(self.freq_var.get())
            if freq <= 0:
                freq = 1.0
        except Exception:
            freq = 1.0
        try:
            dur = float(self.duration_var.get())
            if dur <= 0:
                dur = 600.0
        except Exception:
            dur = 600.0
        return freq, dur

    def save_settings(self):
        selector = self._current_selector() or self.selector_state
        crs, bbox_world, points_world = self._world_points_for_selector(selector) if selector else (None, None, None)
        payload = make_selector_payload(
            _selector_mode(selector or {}),
            bbox_px=_selector_bbox(selector),
            points_px=_selector_points(selector),
            crs=crs,
            bbox_world=bbox_world,
            points_world=points_world,
        ) if selector else {}
        data = {
            "module": "raw_timestacker",
            "settings_version": 1,
            "paths": {
                "input_folder": self.input_folder.get(),
                "output_folder": self.output_folder.get(),
                "batch_folder": self.batch_folder.get(),
            },
            "ui_state": {
                "add_selector_text": bool(self.add_selector_text.get()),
                "selector_text": self.selector_text.get(),
                "selector_mode": self.selector_mode_var.get(),
                "line_width": self.line_width_var.get(),
                "identified_resolution": self.identified_res_var.get(),
                "add_resolution_manual": bool(self.add_resolution_manual.get()),
                "manual_resolution": self.resolution_entry.get(),
                "fill_gaps": bool(self.fill_gaps.get()),
                "freq_hz": float(self.freq_var.get()),
                "duration_s": float(self.duration_var.get()),
            },
            "selectors": {"primary": payload},
        }
        initialdir = self.output_folder.get().strip() or self.input_folder.get().strip() or None
        path = save_settings_json(self, "Raw Timestacker", data, initialdir=initialdir)
        if path:
            print(f"Settings saved to: {path}")

    def load_settings(self):
        initialdir = self.output_folder.get().strip() or self.input_folder.get().strip() or None
        data, path = load_settings_json(self, "Raw Timestacker", initialdir=initialdir)
        if not data:
            return
        try:
            paths = data.get("paths", {})
            self.input_folder.set(paths.get("input_folder", ""))
            self.output_folder.set(paths.get("output_folder", ""))
            self.batch_folder.set(paths.get("batch_folder", ""))

            ui_state = data.get("ui_state", {})
            self.add_selector_text.set(bool(ui_state.get("add_selector_text", False)))
            self.toggle_selector_entry()
            self.selector_text.set(ui_state.get("selector_text", ""))
            self.selector_mode_var.set(ui_state.get("selector_mode", "bbox"))
            self.line_width_var.set(str(ui_state.get("line_width", "1")))
            self.identified_res_var.set(ui_state.get("identified_resolution", "Not identified"))
            self.add_resolution_manual.set(bool(ui_state.get("add_resolution_manual", False)))
            self.toggle_resolution_entry()
            self.resolution_entry.delete(0, tk.END)
            self.resolution_entry.insert(0, str(ui_state.get("manual_resolution", "")))
            self.fill_gaps.set(bool(ui_state.get("fill_gaps", True)))
            self.freq_var.set(float(ui_state.get("freq_hz", 1.0)))
            self.duration_var.set(float(ui_state.get("duration_s", 600.0)))

            selector = data.get("selectors", {}).get("primary")
            if selector:
                if selector.get("mode") == "bbox":
                    selector = {
                        "mode": "bbox",
                        "bbox_px": selector.get("bbox_px") or selector.get("bbox") or [],
                        "line_width": int(ui_state.get("line_width", 1) or 1),
                    }
                else:
                    selector = {
                        "mode": selector.get("mode", "line"),
                        "points_px": selector.get("points_px") or selector.get("points") or [],
                        "line_width": int(ui_state.get("line_width", 1) or 1),
                    }
                self.selector_state = self._normalise_selector(selector)
                self.selector_text.set(self._selector_to_text(self.selector_state))
            elif self.selector_text.get().strip():
                self._parse_selector_text()

            print(f"Settings loaded from: {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load settings:\n{e}", parent=self)

    def create_timestack(self):
        inf = self.input_folder.get().strip()
        outf = self.output_folder.get().strip()
        if not inf or not outf:
            messagebox.showerror("Error", "Select input & output folders.", parent=self)
            return
        if self._single_running or self._batch_running:
            messagebox.showwarning("Busy", "A process is already running.", parent=self)
            return

        selector = self._current_selector()
        if not selector:
            messagebox.showerror("Error", "Identify a valid ROI or line first.", parent=self)
            return

        imgs = collect_images(inf)
        if not imgs:
            messagebox.showerror("Error", "No images found in input folder.", parent=self)
            return

        try:
            res_x = self._current_resolution()
        except ValueError as e:
            messagebox.showerror("Error", str(e), parent=self)
            return
        freq, dur = self._current_freq_duration()

        try:
            files_sorted, dts_sorted = collect_dated_files(imgs)
            first_ts = dts_sorted[0].strftime("%Y_%m_%d_%H_%M")
            imgs = files_sorted
        except Exception:
            imgs.sort()
            first_ts = "_".join(os.path.basename(imgs[0]).split("_")[0:5])

        out_name = f"{first_ts}_raw_timestack.png"
        out_path = os.path.join(outf, out_name)
        fill_gaps = bool(self.fill_gaps.get())
        self._cancel_requested = False
        self._single_running = True
        self._single_start_time = time.time()
        self.single_pb.set(0)
        self.single_lbl.configure(text=f"0 / 1 — ETA {format_eta(None)}")

        def worker():
            def upd(c, t):
                self._ui_single_progress(c, t)
            try:
                if fill_gaps:
                    gen = generate_with_fill(
                        imgs, selector, res_x, out_path,
                        freq_hz=freq, duration_s=dur,
                        progress_callback=upd,
                        cancel_callback=lambda: self._cancel_requested,
                    )
                else:
                    gen = generate_no_fill(
                        imgs, selector, res_x, out_path,
                        progress_callback=upd,
                        cancel_callback=lambda: self._cancel_requested,
                    )
                self._ui_message("info", "Success", f"Timestack saved to:\n{gen}")
                self._ui_call(self._apply_single_preview, gen)
            except Exception as e:
                if str(e) == CANCELLED_ERROR:
                    self._ui_message("info", "Cancelled", "Raw timestack generation was cancelled.")
                else:
                    self._ui_message("error", "Error", str(e))
            finally:
                self._single_running = False
                self._single_start_time = None
                self._ui_call(self.single_pb.set, 0)
                self._ui_call(self.single_lbl.configure, text="")

        threading.Thread(target=worker, daemon=True).start()

    def batch_process(self):
        mbf = self.batch_folder.get().strip()
        outf = self.output_folder.get().strip()
        if not mbf or not outf:
            messagebox.showerror("Error", "Select batch & output folders.", parent=self)
            return
        if self._single_running or self._batch_running:
            messagebox.showwarning("Busy", "A process is already running.", parent=self)
            return

        selector = self._current_selector()
        if not selector:
            messagebox.showerror("Error", "Provide a valid ROI or line selection.", parent=self)
            return

        try:
            res_x = self._current_resolution()
        except ValueError as e:
            messagebox.showerror("Error", str(e), parent=self)
            return
        freq, dur = self._current_freq_duration()

        all_subs = [os.path.join(mbf, d) for d in os.listdir(mbf)
                    if os.path.isdir(os.path.join(mbf, d))]
        if not all_subs:
            messagebox.showerror("Error", "No sub-folders found in batch folder.", parent=self)
            return

        subs_to_do = []
        skipped = 0
        for sub in all_subs:
            imgs = collect_images(sub)
            if not imgs:
                continue
            try:
                files_sorted, dts_sorted = collect_dated_files(imgs)
                first_ts = dts_sorted[0].strftime("%Y_%m_%d_%H_%M")
                imgs = files_sorted
            except Exception:
                imgs.sort()
                first_ts = "_".join(os.path.basename(imgs[0]).split("_")[0:5])
            expected_name = f"{first_ts}_raw_timestack.png"
            if os.path.exists(os.path.join(outf, expected_name)):
                skipped += 1
            else:
                subs_to_do.append(sub)

        if not subs_to_do:
            messagebox.showinfo("Batch Done", f"Nothing to do: {skipped} sub-folders were already processed.", parent=self)
            return

        total = len(subs_to_do)
        self._cancel_requested = False
        self._batch_running = True
        self._batch_start_time = time.time()
        self.batch_pb.set(0)
        self.batch_lbl.configure(text=f"0 / {total} — ETA {format_eta(None)}")
        fill_gaps = bool(self.fill_gaps.get())
        print(f"Batch process has started – {total} new sub-folders (skipped {skipped} already done)")

        def update_ui(done_cnt: int):
            frac = done_cnt / total
            eta = compute_eta(self._batch_start_time or time.time(), done_cnt, total) if self._batch_start_time else None
            label = f"{done_cnt} / {total} — ETA {format_eta(eta)}"
            self._ui_batch_progress(frac, label)

        def controller():
            done = 0
            cancelled = False
            max_workers = min(4, os.cpu_count() or 1, total)
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers,
                                                           thread_name_prefix="batch") as pool:
                    self._batch_executor = pool
                    futures = [pool.submit(
                        _process_subfolder, sub, selector, res_x, freq, dur,
                        fill_gaps, outf, lambda: self._cancel_requested) for sub in subs_to_do]

                    for fut in concurrent.futures.as_completed(futures):
                        if self._cancel_requested:
                            cancelled = True
                            for pending in futures:
                                pending.cancel()
                            break
                        try:
                            sub_path, status, msg = fut.result()
                        except Exception as exc:
                            sub_path, status, msg = "", "error", str(exc)
                        if status == "error":
                            print(f"Error processing {sub_path}: {msg}")
                        elif status == "no_imgs":
                            print(f"{sub_path}: {msg}")
                        elif status == "skipped":
                            print(f"Skipped {sub_path} (already done).")
                        elif status == "processed":
                            print(f"Finished {os.path.basename(sub_path)}  ->  {msg}")
                        elif status == "cancelled":
                            cancelled = True
                            print(f"Cancelled {os.path.basename(sub_path)}")
                            break
                        done += 1
                        update_ui(done)
            finally:
                self._batch_executor = None
                elapsed = time.time() - (self._batch_start_time or time.time())
                elapsed_str = f"{elapsed/60:.1f} min" if elapsed >= 60 else f"{elapsed:.1f} s"
                if cancelled or self._cancel_requested:
                    print("Batch process cancelled.")
                    self._ui_message("info", "Cancelled", f"Batch process cancelled after {done} sub-folders.\nElapsed time: {elapsed_str}")
                else:
                    print(f"Batch process complete in {elapsed_str}")
                    self._ui_message(
                        "info",
                        "Batch Done",
                        f"Newly processed: {done}\n"
                        f"Previously done: {skipped}\n"
                        f"Total in folder: {len(all_subs)}\n"
                        f"Elapsed time: {elapsed_str}\n\n"
                        "Batch process complete"
                    )
                self._batch_running = False
                self._batch_start_time = None
                self._ui_batch_progress(0, "")

        threading.Thread(target=controller, daemon=True).start()

    def reset_to_initial(self):
        self._request_cancel()
        self.input_folder.set("")
        self.output_folder.set("")
        self.batch_folder.set("")
        self.selector_state = None
        self.bbox = None
        self.selector_text.set("")
        self.add_selector_text.set(False)
        self.toggle_selector_entry()
        self.selector_mode_var.set("bbox")
        self.line_width_var.set("1")
        self.identified_res_var.set("Not identified")
        self.add_resolution_manual.set(False)
        self.toggle_resolution_entry()
        self.resolution_entry.delete(0, tk.END)
        self.fill_gaps.set(True)
        self.freq_var.set(1.0)
        self.duration_var.set(600.0)
        self.single_pb.set(0)
        self.batch_pb.set(0)
        self.single_lbl.configure(text="")
        self.batch_lbl.configure(text="")
        self.image_label.configure(image=None, text="")
        self.image_label.image = None
        print("\n--- Session reset requested. Running processes will stop at the next safe checkpoint. ---\n")


if __name__ == "__main__":
    root = ctk.CTk()
    root.withdraw()
    TimestackTool(master=root)
    root.mainloop()
