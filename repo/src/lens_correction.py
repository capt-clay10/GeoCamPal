"""
Intrinsic lens correction

Uses OpenCV checkerboard detection across multiple images to compute
camera intrinsic parameters (camera matrix + distortion coefficients).
Outputs a .pkl file compatible with the rest of the GeoCamPal pipeline.

Supports rectangular checkerboards where the square/cell width and
height may differ (e.g. 30 mm × 25 mm cells).

Expected workflow:
  1. Browse to a folder containing checkerboard calibration images.
  2. Enter the number of SQUARES (cols × rows) and cell width & height.
     The tool subtracts 1 internally to get inner corners for OpenCV.
  3. Click "Generate Lens Correction File".
  4. Review detected corners and undistorted preview in the top panel.
  5. .pkl is saved alongside a summary .txt report.
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
)

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")


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
            "Lens Correction Tool ready.\n"
            "Provide a folder of checkerboard images.\n"
            "Enter the number of SQUARES (not inner corners) — the\n"
            "tool automatically converts to inner corners (squares − 1).\n"
            "  e.g. 17 squares across × 19 down → 16 × 18 inner corners\n"
            "Cell width/height = physical size of each cell in mm.\n"
            "For square checkerboards, set both to the same value.\n"
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
        self._ui_call(fn, title, message)

    def _ui_progress(self, value, eta_text=None):
        self._ui_call(self._update_progress_ui, value, eta_text)

    def _apply_preview(self, vis_bgr, undist_bgr):
        self.axes[0].clear()
        self.axes[0].imshow(cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB))
        self.axes[0].set_title("Detected Corners (sample)")
        self.axes[0].axis("off")

        self.axes[1].clear()
        self.axes[1].imshow(cv2.cvtColor(undist_bgr, cv2.COLOR_BGR2RGB))
        self.axes[1].set_title("Undistorted Result")
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

                img = cv2.imread(img_path)
                if img is None:
                    print(f"[WARN] Cannot read: {os.path.basename(img_path)}")
                    continue

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if img_shape is None:
                    img_shape = gray.shape[::-1]

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

            print("Running calibration …")
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                obj_points, img_points, img_shape, None, None)

            print(f"\nCalibration RMS reprojection error: {ret:.4f} px")
            print(f"\nCamera matrix:\n{camera_matrix}")
            print(f"\nDistortion coefficients:\n{dist_coeffs.ravel()}")

            fx = camera_matrix[0, 0]
            fy = camera_matrix[1, 1]
            cx = camera_matrix[0, 2]
            cy = camera_matrix[1, 2]
            W, H = img_shape
            print(f"\nFocal length:  fx={fx:.2f} px,  fy={fy:.2f} px")
            print(f"Principal point:  cx={cx:.2f} px,  cy={cy:.2f} px")
            print(f"Image size: {W} × {H} px")

            if not is_square:
                print(f"\n[NOTE] Rectangular cells used ({cell_w_mm} × {cell_h_mm} mm). Object points have been scaled accordingly.")

            mean_errors = []
            for i in range(len(obj_points)):
                proj, _ = cv2.projectPoints(
                    obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
                err = cv2.norm(img_points[i], proj, cv2.NORM_L2) / len(proj)
                mean_errors.append(err)

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
            }
            self.calibration_data = cal_data
            pkl_path = os.path.join(output_folder, "lens_calibration.pkl")
            with open(pkl_path, "wb") as f:
                pickle.dump(cal_data, f)
            print(f"\nSaved: {pkl_path}")

            txt_path = os.path.join(output_folder, "calibration_report.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("=== Lens Calibration Report ===\n\n")
                f.write(f"Input folder: {input_folder}\n")
                f.write(f"Images found: {len(images)}\n")
                f.write(f"Images with corners: {detected_count}\n")
                f.write(f"Board: {sq_cols} × {sq_rows} squares\n")
                f.write(f"Inner corners: {n_cols} × {n_rows}\n")
                if is_square:
                    f.write(f"Cell size: {cell_w_mm} mm (square)\n\n")
                else:
                    f.write(f"Cell width: {cell_w_mm} mm\n")
                    f.write(f"Cell height: {cell_h_mm} mm\n\n")
                f.write(f"RMS reprojection error: {ret:.4f} px\n\n")
                f.write(f"Camera matrix:\n{camera_matrix}\n\n")
                f.write(f"Focal length: fx={fx:.2f} px, fy={fy:.2f} px\n")
                f.write(f"Principal point: cx={cx:.2f} px, cy={cy:.2f} px\n")
                f.write(f"Image size: {W} × {H} px\n\n")
                f.write(f"Distortion coefficients:\n{dist_coeffs.ravel()}\n\n")
                f.write("Per-image reprojection errors:\n")
                for i, err in enumerate(mean_errors):
                    f.write(f"  Image {i + 1}: {err:.4f} px\n")
            print(f"Saved: {txt_path}")

            if sample_img is not None:
                vis = cv2.drawChessboardCorners(sample_img.copy(), pattern_size, sample_corners, True)
                Himg, Wimg = sample_img.shape[:2]
                new_mtx, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (Wimg, Himg), 1, (Wimg, Himg))
                undist = cv2.undistort(sample_img, camera_matrix, dist_coeffs, None, new_mtx)
                self._ui_call(self._apply_preview, vis, undist)

            self._ui_progress(1.0, "ETA: Completed")
            self._ui_message(
                "info",
                "Done",
                f"Calibration complete!\nRMS error: {ret:.4f} px\n\nFiles saved to:\n{output_folder}")

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
