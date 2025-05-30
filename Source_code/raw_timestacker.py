import os
import glob
import threading
import time
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from PIL.PngImagePlugin import PngInfo
import cv2
import numpy as np
import sys
import tifffile
import customtkinter as ctk
import rasterio
from scipy.interpolate import interp1d 
import concurrent.futures           
     

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")

# -------------------------------------------------------------------------------
# StdoutRedirector: Redirect console output to the built-in console widget
# -------------------------------------------------------------------------------
class StdoutRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        self.text_widget.insert(tk.END, message)
        self.text_widget.see(tk.END)

    def flush(self):
        pass

# -------------------------------------------------------------------------------
# ROI Selector (small helper window)
# -------------------------------------------------------------------------------
class ScrollZoomBBoxSelector(tk.Frame):
    """A scroll-zoom capable widget that lets the user draw a bounding box."""
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        master.title("Scrollable & Zoomable ROI Selector")

        # Canvas + Scrollbars
        top_frame = tk.Frame(self)
        top_frame.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(top_frame, cursor="cross", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.v_scroll = tk.Scrollbar(top_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.h_scroll = tk.Scrollbar(self, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.h_scroll.pack(side=tk.BOTTOM, fill=tk.X)

        self.canvas.configure(xscrollcommand=self.h_scroll.set, yscrollcommand=self.v_scroll.set)

        # Internal state
        self.cv_image = None
        self.pil_image = None
        self.tk_image = None
        self.scale_factor = 1.0

        self.start_x_display = None
        self.start_y_display = None
        self.rect_id = None
        self.bbox = (0, 0, 0, 0)

        # Mouse bindings
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        self.master.bind("<Return>", self.on_enter_key)

        # Bottom-bar
        button_frame = tk.Frame(self)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        tk.Button(button_frame, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Zoom In", command=self.zoom_in).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Zoom Out", command=self.zoom_out).pack(side=tk.LEFT, padx=5)
        tk.Label(button_frame, text="Drag to select ROI; press Enter to confirm").pack(side=tk.LEFT, padx=5)

    # (unchanged helper methods...)
    # ---------------------------------------------------------------------------
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
        self.bbox = (0, 0, 0, 0)
        if self.rect_id:
            self.canvas.delete(self.rect_id)
            self.rect_id = None

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

    def zoom_in(self):
        if self.pil_image is None:
            return
        self.scale_factor *= 1.25
        self.display_image()

    def zoom_out(self):
        if self.pil_image is None:
            return
        self.scale_factor = max(self.scale_factor * 0.8, 0.1)
        self.display_image()

    def on_button_press(self, event):
        self.start_x_display = self.canvas.canvasx(event.x)
        self.start_y_display = self.canvas.canvasy(event.y)
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(
            self.start_x_display,
            self.start_y_display,
            self.start_x_display,
            self.start_y_display,
            outline="red",
            width=2,
        )

    def on_move_press(self, event):
        if not self.rect_id:
            return
        cur_x = self.canvas.canvasx(event.x)
        cur_y = self.canvas.canvasy(event.y)
        self.canvas.coords(self.rect_id, self.start_x_display, self.start_y_display, cur_x, cur_y)

    def on_button_release(self, event):
        end_x = self.canvas.canvasx(event.x)
        end_y = self.canvas.canvasy(event.y)
        x1_disp, x2_disp = sorted([self.start_x_display, end_x])
        y1_disp, y2_disp = sorted([self.start_y_display, end_y])
        x1 = int(x1_disp / self.scale_factor)
        y1 = int(y1_disp / self.scale_factor)
        w = int((x2_disp - x1_disp) / self.scale_factor)
        h = int((y2_disp - y1_disp) / self.scale_factor)
        self.bbox = (x1, y1, w, h)

    def on_enter_key(self, _):
        print("Final bounding box:", self.bbox)

# -------------------------------------------------------------------------------
# Core timestack generators (unchanged)
# -------------------------------------------------------------------------------
def generate_with_fill(image_files, bbox, resolution_x_m, output_path,
                       freq_hz=1.0, duration_s=600.0, progress_callback=None):
    # ... (same as before)
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid bounding box {bbox}: width and height must be > 0")

    # parse and sort timestamps
    dt_files = []
    for f in image_files:
        name = os.path.splitext(os.path.basename(f))[0]
        parts = name.split("_")
        if len(parts) < 7:
            continue
        year, month, day, hour, minute, second, msec = parts[:7]
        try:
            dt = datetime(int(year), int(month), int(day),
                          int(hour), int(minute), int(second),
                          int(msec) * 1000)
            dt_files.append((dt, f))
        except ValueError:
            continue
    if not dt_files:
        raise ValueError("No valid timestamps found in image names.")
    dt_files.sort(key=lambda x: x[0])

    # prepare buckets
    N = int(duration_s * freq_hz)
    dt_start = dt_files[0][0]
    tasks = [None] * N
    tasks_dt = [None] * N

    # bucket-round images
    for dt, fn in dt_files:
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

    # build lines, using NaN for missing
    lines = []
    for i, fn in enumerate(tasks, start=1):
        if progress_callback:
            progress_callback(i, N)
        if fn is None or not os.path.exists(fn):
            line = np.full((w, 3), np.nan, dtype=float)
        else:
            img = tifffile.imread(fn) if fn.lower().endswith(".tif") else np.array(Image.open(fn).convert("RGB"))
            roi = img[y:y+h, x:x+w]
            if roi.size == 0:
                line = np.full((w, 3), np.nan, dtype=float)
            else:
                if roi.ndim == 2:
                    roi = np.stack([roi]*3, axis=-1)
                elif roi.shape[2] > 3:
                    roi = roi[:, :, :3]
                line = np.round(np.mean(roi, axis=0)).astype(float)
        lines.append(line)

    ts = np.stack(lines, axis=0)  # shape (N, w, 3) dtype=float with NaNs

    # (interpolation & saving unchanged...)
    nan_rows_before = np.all(np.isnan(ts), axis=(1, 2))

    # detect tail nan-run
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
    for j in range(w):
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
    if out_img.width != w:
        out_img = out_img.resize((w, ts.shape[0]), Image.NEAREST)

    info = PngInfo()
    info.add_text("pixel_resolution", f"{resolution_x_m:.6f}")
    info.add_text("bounding_box", f"{x},{y},{w},{h}")
    out_img.save(output_path, format="PNG", pnginfo=info)
    return output_path


def generate_no_fill(image_files, bbox, resolution_x_m, output_path, progress_callback=None):
    x, y, w, h = bbox

    def parse_dt(f):
        p = os.path.splitext(os.path.basename(f))[0].split("_")[:7]
        return datetime(int(p[0]), int(p[1]), int(p[2]),
                        int(p[3]), int(p[4]), int(p[5]),
                        int(p[6]) * 1000)

    files = sorted(image_files, key=parse_dt)
    total = len(files)
    lines = []
    for i, fn in enumerate(files, start=1):
        if progress_callback:
            progress_callback(i, total)
        img = tifffile.imread(fn) if fn.lower().endswith(".tif") else np.array(Image.open(fn).convert("RGB"))
        roi = img[y:y + h, x:x + w]
        if roi.size == 0:
            continue
        if roi.ndim == 2:
            roi = np.stack([roi] * 3, axis=-1)
        elif roi.shape[2] > 3:
            roi = roi[:, :, :3]
        line = np.round(np.mean(roi, axis=0)).astype(np.uint8)
        lines.append(line)

    if not lines:
        raise ValueError("No valid ROI slices—nothing to stack.")

    ts = np.stack(lines, axis=0)[::-1, :, :]
    out = Image.fromarray(ts)
    if out.width != w:
        out = out.resize((w, ts.shape[0]), Image.NEAREST)
    info = PngInfo()
    info.add_text("pixel_resolution", f"{resolution_x_m:.6f}")
    info.add_text("bounding_box", f"{x},{y},{w},{h}")
    out.save(output_path, format="PNG", pnginfo=info)
    return output_path

# -------------------------------------------------------------------------------
# Utility to get resource path
# -------------------------------------------------------------------------------
def resource_path(rel: str) -> str:
    try:
        base = sys._MEIPASS
    except Exception:
        base = os.path.dirname(__file__)
    return os.path.join(base, rel)

# -------------------------------------------------------------------------------
# Helper executed in worker processes  (NEW)
# -------------------------------------------------------------------------------
def _process_subfolder(sub_path: str, bbox: tuple, res_x: float,
                       freq: float, dur: float, fill_gaps: bool,
                       output_folder: str) -> tuple:
    """
    Runs inside a ThreadPoolExecutor worker.
    Returns (sub_path, status, message) where status ∈
       {"processed", "skipped", "error", "no_imgs"}.
    """
    try:
        imgs = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.tif"):
            imgs.extend(glob.glob(os.path.join(sub_path, ext)))
        if not imgs:
            return sub_path, "no_imgs", "No supported images"

        imgs.sort()
        first_ts  = "_".join(os.path.basename(imgs[0]).split("_")[0:5])
        out_name  = f"{first_ts}_raw_timestack.png"
        out_path  = os.path.join(output_folder, out_name)

        if os.path.exists(out_path):
            return sub_path, "skipped", "Already processed"

        if fill_gaps:
            generate_with_fill(imgs, bbox, res_x, out_path,
                               freq_hz=freq, duration_s=dur)
        else:
            generate_no_fill(imgs, bbox, res_x, out_path)
        return sub_path, "processed", out_name
    except Exception as exc:
        return sub_path, "error", str(exc)


# -------------------------------------------------------------------------------
# Main GUI class – TimestackTool
# -------------------------------------------------------------------------------
class TimestackTool(ctk.CTkToplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("Time-stacking Tool")
        self.geometry("1200x800")
        try:
            self.iconbitmap(resource_path("launch_logo.ico"))
        except Exception:
            pass

        self.input_folder = tk.StringVar()
        self.output_folder = tk.StringVar()
        self.batch_folder = tk.StringVar()

        self.bbox = None
        self.add_bbox_text = tk.BooleanVar(value=False)
        self.bbox_text = tk.StringVar()

        self.default_resolution = 0.25
        self.identified_res_var = tk.StringVar(value="Not identified")
        self.add_resolution_manual = tk.BooleanVar(value=False)

        self.fill_gaps = tk.BooleanVar(value=True)
        self.freq_var = tk.DoubleVar(value=1.0)
        self.duration_var = tk.DoubleVar(value=600.0)

        self._build_ui()

        self.stdout_redirector = StdoutRedirector(self.console_text)
        sys.stdout = self.stdout_redirector
        sys.stderr = self.stdout_redirector
        print("Here you may see console outputs\n--------------------------------\n")

    # ---------------------------------------------------------------------------
    # (all UI-building helpers unchanged)
    # ---------------------------------------------------------------------------
    def _build_ui(self):
        # Top: preview image
        self.top_frame = ctk.CTkFrame(self, height=400)
        self.top_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.image_label = ctk.CTkLabel(self.top_frame, text="")
        self.image_label.pack(expand=True)

        # Bottom: control panels
        self.bottom_frame = ctk.CTkFrame(self, height=250)
        self.bottom_frame.pack(fill="x", padx=10, pady=10)

        # Input / ROI panel
        ip = ctk.CTkFrame(self.bottom_frame)
        ip.pack(fill="x", padx=5, pady=5)
        ctk.CTkButton(ip, text="Browse Input Folder", command=self.browse_input_folder)\
            .grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ctk.CTkLabel(ip, textvariable=self.input_folder)\
            .grid(row=0, column=1, padx=5, pady=5, sticky="w")
        ctk.CTkButton(ip, text="Select BBox", command=self.select_bbox)\
            .grid(row=0, column=2, padx=5, pady=5)
        ctk.CTkCheckBox(ip, text="Add bbox as text", variable=self.add_bbox_text,
                        command=self.toggle_bbox_entry)\
            .grid(row=0, column=3, padx=5, pady=5)
        self.bbox_entry = ctk.CTkEntry(ip, textvariable=self.bbox_text)
        self.bbox_entry.grid(row=0, column=4, padx=5, pady=5)
        self.bbox_entry.configure(state="disabled")

        # Resolution / Fill / Freq / Duration panel
        rp = ctk.CTkFrame(self.bottom_frame)
        rp.pack(fill="x", padx=5, pady=5)
        ctk.CTkLabel(rp, text="Identified resolution:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ctk.CTkLabel(rp, textvariable=self.identified_res_var).grid(row=0, column=1, padx=5, pady=5, sticky="w")
        ctk.CTkCheckBox(rp, text="Add pixel resolution manually", variable=self.add_resolution_manual,
                        command=self.toggle_resolution_entry)\
            .grid(row=0, column=2, padx=5, pady=5)
        self.resolution_entry = ctk.CTkEntry(rp)
        self.resolution_entry.grid(row=0, column=3, padx=5, pady=5)
        self.resolution_entry.configure(state="disabled")
        ctk.CTkLabel(rp, text="m").grid(row=0, column=4, padx=5, pady=5, sticky="w")
        ctk.CTkCheckBox(rp, text="Fill gaps", variable=self.fill_gaps)\
            .grid(row=0, column=5, padx=15, pady=5)
        ctk.CTkLabel(rp, text="Freq (Hz):").grid(row=0, column=6, padx=5, pady=5, sticky="w")
        ctk.CTkEntry(rp, textvariable=self.freq_var, width=60).grid(row=0, column=7, padx=5, pady=5, sticky="w")
        ctk.CTkLabel(rp, text="Dur (s):").grid(row=0, column=8, padx=5, pady=5, sticky="w")
        ctk.CTkEntry(rp, textvariable=self.duration_var, width=60).grid(row=0, column=9, padx=5, pady=5, sticky="w")

        # Output & single-timestack panel
        op = ctk.CTkFrame(self.bottom_frame)
        op.pack(fill="x", padx=5, pady=5)
        ctk.CTkButton(op, text="Select Output Folder", command=self.select_output_folder)\
            .grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ctk.CTkLabel(op, textvariable=self.output_folder)\
            .grid(row=0, column=1, padx=5, pady=5, sticky="w")
        ctk.CTkButton(op, text="Create Raw Timestack", command=self.create_timestack)\
            .grid(row=0, column=2, padx=5, pady=5)
        self.single_pb = ctk.CTkProgressBar(op)
        self.single_pb.grid(row=0, column=3, padx=5, pady=5)
        self.single_pb.set(0)
        self.single_lbl = ctk.CTkLabel(op, text="")
        self.single_lbl.grid(row=0, column=4, padx=5, pady=5, sticky="w")

        # Batch panel
        bp = ctk.CTkFrame(self.bottom_frame)
        bp.pack(fill="x", padx=5, pady=5)
        ctk.CTkButton(bp, text="Select Batch Folder", command=self.browse_batch_folder)\
            .grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ctk.CTkLabel(bp, textvariable=self.batch_folder)\
            .grid(row=0, column=1, padx=5, pady=5, sticky="w")
        ctk.CTkButton(bp, text="Batch Process", command=self.batch_process)\
            .grid(row=0, column=2, padx=5, pady=5)
        self.batch_pb = ctk.CTkProgressBar(bp)
        self.batch_pb.grid(row=0, column=3, padx=5, pady=5)
        self.batch_pb.set(0)
        self.batch_lbl = ctk.CTkLabel(bp, text="")
        self.batch_lbl.grid(row=0, column=4, padx=5, pady=5, sticky="w")

        # Console panel
        cf = ctk.CTkFrame(self)
        cf.pack(fill="both", expand=False, padx=10, pady=10)
        self.console_text = tk.Text(cf, wrap="word", height=10)
        self.console_text.pack(fill="both", expand=True, padx=5, pady=5)

    # ---------------------------------------------------------------------------
    # Simple helpers (unchanged)
    # ---------------------------------------------------------------------------
    def toggle_bbox_entry(self):
        self.bbox_entry.configure(state="normal" if self.add_bbox_text.get() else "disabled")

    def toggle_resolution_entry(self):
        self.resolution_entry.configure(state="normal" if self.add_resolution_manual.get() else "disabled")

    def browse_input_folder(self):
        f = filedialog.askdirectory(title="Select Input Folder with Images")
        if f:
            self.input_folder.set(f)

    def select_output_folder(self):
        f = filedialog.askdirectory(title="Select Output Folder for Raw Timestack")
        if f:
            self.output_folder.set(f)

    def browse_batch_folder(self):
        f = filedialog.askdirectory(title="Select Main Batch Folder (sub-folders per batch)")
        if f:
            self.batch_folder.set(f)


    # Bounding-box selection
    def select_bbox(self):
        fld = self.input_folder.get().strip()
        if not fld:
            messagebox.showerror("Error", "Please select an input folder first.")
            return
        imgs = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.tif"):
            imgs.extend(glob.glob(os.path.join(fld, ext)))
        if not imgs:
            messagebox.showerror("Error", "No supported image files found in folder.")
            return
        win = tk.Toplevel(self)
        sel = ScrollZoomBBoxSelector(win)
        sel.pack(fill="both", expand=True)
        sel.load_image(imgs[0])
        ctk.CTkButton(win, text="Confirm", command=lambda: self._save_bbox(sel, win)).pack(pady=10)

    def _save_bbox(self, sel, win):
        if sel.bbox == (0, 0, 0, 0):
            messagebox.showwarning("Warning", "No bounding box selected.")
            return
        self.bbox = sel.bbox
        self.bbox_text.set(",".join(map(str, self.bbox)))
        fld = self.input_folder.get().strip()
        tifs = glob.glob(os.path.join(fld, "*.tif"))
        if tifs:
            try:
                with rasterio.open(tifs[0]) as src:
                    res = abs(src.transform.a)
            except:
                res = self.default_resolution
        else:
            res = self.default_resolution
        self.identified_res_var.set(f"{res:.3f} m")
        win.destroy()

    def _parse_bbox_text(self):
        try:
            p = [int(x.strip()) for x in self.bbox_text.get().split(",")]
            if len(p) != 4:
                raise ValueError("Must be 4 values: x,y,w,h")
            x, y, w, h = p
            if w <= 0 or h <= 0:
                raise ValueError("Width/height must be >0")
            self.bbox = (x, y, w, h)
            return True
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid bbox: {e}")
            return False

    def create_timestack(self):
        inf = self.input_folder.get().strip()
        outf = self.output_folder.get().strip()
        if not inf or not outf:
            messagebox.showerror("Error", "Select input & output folders.")
            return

        if self.add_bbox_text.get() and self.bbox_text.get().strip():
            if not self._parse_bbox_text():
                return
        if not self.bbox:
            messagebox.showerror("Error", "Identify a valid bounding box first.")
            return

        imgs = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.tif"):
            imgs.extend(glob.glob(os.path.join(inf, ext)))
        if not imgs:
            messagebox.showerror("Error", "No images found in input folder.")
            return

        # Resolution
        if self.add_resolution_manual.get():
            try:
                res_x = float(self.resolution_entry.get())
            except:
                messagebox.showerror("Error", "Invalid manual resolution.")
                return
        else:
            try:
                res_x = float(self.identified_res_var.get().split()[0])
            except:
                res_x = self.default_resolution

        # Freq / Duration
        try:
            freq = float(self.freq_var.get())
            if freq <= 0:
                freq = 1.0
        except:
            freq = 1.0
        try:
            dur = float(self.duration_var.get())
            if dur <= 0:
                dur = 600.0
        except:
            dur = 600.0

        imgs.sort()
        first_ts = "_".join(os.path.basename(imgs[0]).split("_")[0:5])
        out_name = f"{first_ts}_raw_timestack.png"
        out_path = os.path.join(outf, out_name)

        def worker():
            def upd(c, t):
                self.single_pb.set(c / t)
                self.single_lbl.configure(text=f"{c} / {t}")
            try:
                if self.fill_gaps.get():
                    gen = generate_with_fill(
                        imgs, self.bbox, res_x, out_path,
                        freq_hz=freq, duration_s=dur,
                        progress_callback=upd
                    )
                else:
                    gen = generate_no_fill(
                        imgs, self.bbox, res_x, out_path,
                        progress_callback=upd
                    )
                messagebox.showinfo("Success", f"Timestack saved to:\n{gen}")
                pil = Image.open(gen)
                preview = ctk.CTkImage(light_image=pil, size=(800, min(600, pil.height)))
                self.image_label.configure(image=preview)
                self.image_label.image = preview
            except Exception as e:
                messagebox.showerror("Error", str(e))
            finally:
                self.single_pb.set(0)
                self.single_lbl.configure(text="")

        threading.Thread(target=worker, daemon=True).start()
    # ---------------------------------------------------------------------------
    # Batch processing – logic for skipping & multiprocessing
    # ---------------------------------------------------------------------------
    def batch_process(self):
        mbf = self.batch_folder.get().strip()
        outf = self.output_folder.get().strip()
        if not mbf or not outf:
            messagebox.showerror("Error", "Select batch & output folders.")
            return

        if self.add_bbox_text.get() and self.bbox_text.get().strip():
            if not self._parse_bbox_text():
                return
        if not self.bbox:
            messagebox.showerror("Error", "Provide a valid bounding box.")
            return

        # Resolution
        if self.add_resolution_manual.get():
            try:
                res_x = float(self.resolution_entry.get())
            except Exception:
                messagebox.showerror("Error", "Invalid manual resolution.")
                return
        else:
            try:
                res_x = float(self.identified_res_var.get().split()[0])
            except Exception:
                res_x = self.default_resolution

        # Freq / Duration
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

        # Gather sub-folders
        all_subs = [os.path.join(mbf, d) for d in os.listdir(mbf)
                    if os.path.isdir(os.path.join(mbf, d))]
        if not all_subs:
            messagebox.showerror("Error", "No sub-folders found in batch folder.")
            return

        # ---------- Quick pass: skip already-processed sub-folders ----------
        subs_to_do = []
        skipped = 0
        for sub in all_subs:
            imgs = []
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.tif"):
                imgs.extend(glob.glob(os.path.join(sub, ext)))
            if not imgs:
                continue
            imgs.sort()
            first_ts = "_".join(os.path.basename(imgs[0]).split("_")[0:5])
            expected_name = f"{first_ts}_raw_timestack.png"
            if os.path.exists(os.path.join(outf, expected_name)):
                skipped += 1
            else:
                subs_to_do.append(sub)

        if not subs_to_do:
            messagebox.showinfo("Batch Done", f"Nothing to do: {skipped} sub-folders "
                                              f"were already processed.")
            return

        total = len(subs_to_do)
        self.batch_pb.set(0)
        self.batch_lbl.configure(text="ETA: --")
        start = time.time()
        print(f"Batch process has started – {total} new sub-folders "
              f"(skipped {skipped} already done)")

        def update_ui(done_cnt: int):
            frac = done_cnt / total
            self.batch_pb.set(frac)
        
            elapsed = time.time() - start
            if done_cnt:
                est_remaining = (total - done_cnt) / (done_cnt / elapsed)   # folders/s
            else:
                est_remaining = 0
            m, s = divmod(int(est_remaining), 60)
            self.batch_lbl.configure(text=f"{m}m {s}s" if m else f"{s}s")


        # ---------- Worker thread that controls the ProcessPool ----------
        def controller():
            done = 0
            max_workers = min(4, os.cpu_count() or 1, total)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers,
                                                       thread_name_prefix="batch") as pool:
                futures = [pool.submit(
                    _process_subfolder, sub, self.bbox, res_x, freq, dur,
                    self.fill_gaps.get(), outf) for sub in subs_to_do]

                for fut in concurrent.futures.as_completed(futures):
                    sub_path, status, msg = fut.result()
                    if status == "error":
                        print(f"Error processing {sub_path}: {msg}")
                    elif status == "no_imgs":
                        print(f"{sub_path}: {msg}")
                    elif status == "skipped":
                        print(f"Skipped {sub_path} (already done).")
                    elif status == "processed":
                        print(f"Finished {os.path.basename(sub_path)}  ->  {msg}")
                    done += 1
                    update_ui(done)

            elapsed = time.time() - start
            elapsed_str = f"{elapsed/60:.1f} min" if elapsed >= 60 else f"{elapsed:.1f} s"
            print(f"Batch process complete in {elapsed_str}")
            messagebox.showinfo(
                "Batch Done",
                f"Newly processed: {done}\n"
                f"Previously done: {skipped}\n"
                f"Total in folder: {len(all_subs)}\n"
                f"Elapsed time: {elapsed_str}\n\n"
                "Batch process complete"
            )
            self.batch_pb.set(0)
            self.batch_lbl.configure(text="")

        threading.Thread(target=controller, daemon=True).start()

# -------------------------------------------------------------------------------
# Run GUI
# -------------------------------------------------------------------------------
if __name__ == "__main__":

    root = ctk.CTk()
    root.withdraw()
    TimestackTool(master=root)
    root.mainloop()
