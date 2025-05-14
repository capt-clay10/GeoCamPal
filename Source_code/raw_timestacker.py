import os
import glob
import threading
import time
from datetime import datetime
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

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")

# -----------------------------------------------------------------------------
# StdoutRedirector: Redirect console output to the built‑in console widget
# -----------------------------------------------------------------------------
class StdoutRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        self.text_widget.insert(tk.END, message)
        self.text_widget.see(tk.END)  # auto‑scroll

    def flush(self):
        pass  # Nothing required for flush in this context

# -----------------------------------------------------------------------------
# ROI Selector (small helper window)
# -----------------------------------------------------------------------------
class ScrollZoomBBoxSelector(tk.Frame):
    """A scroll‑zoom capable widget that lets the user draw a bounding box."""

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        master.title("Scrollable & Zoomable ROI Selector")

        # Canvas + Scrollbars ---------------------------------------------------
        top_frame = tk.Frame(self)
        top_frame.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(top_frame, cursor="cross", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.v_scroll = tk.Scrollbar(top_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.h_scroll = tk.Scrollbar(self, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.h_scroll.pack(side=tk.BOTTOM, fill=tk.X)

        self.canvas.configure(xscrollcommand=self.h_scroll.set, yscrollcommand=self.v_scroll.set)

        # Internal state -------------------------------------------------------
        self.cv_image = None  # OpenCV BGR image
        self.pil_image = None  # PIL image (RGB)
        self.tk_image = None   # Tk image for canvas
        self.scale_factor = 1.0

        self.start_x_display = None
        self.start_y_display = None
        self.rect_id = None
        self.bbox = (0, 0, 0, 0)  # (x, y, w, h) – original coords

        # Mouse‑bindings --------------------------------------------------------
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        self.master.bind("<Return>", self.on_enter_key)

        # Bottom‑bar ------------------------------------------------------------
        button_frame = tk.Frame(self)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        tk.Button(button_frame, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Zoom In", command=self.zoom_in).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Zoom Out", command=self.zoom_out).pack(side=tk.LEFT, padx=5)
        tk.Label(button_frame, text="Drag to select ROI; press Enter to confirm").pack(side=tk.LEFT, padx=5)

    # ---------------- Image display helpers ----------------------------------
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

    # ---------------- Mouse callbacks ----------------------------------------
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
        end_x_display = self.canvas.canvasx(event.x)
        end_y_display = self.canvas.canvasy(event.y)
        x1_disp, x2_disp = sorted([self.start_x_display, end_x_display])
        y1_disp, y2_disp = sorted([self.start_y_display, end_y_display])
        x1_orig = int(x1_disp / self.scale_factor)
        y1_orig = int(y1_disp / self.scale_factor)
        w = int((x2_disp - x1_disp) / self.scale_factor)
        h = int((y2_disp - y1_disp) / self.scale_factor)
        self.bbox = (x1_orig, y1_orig, w, h)

    def on_enter_key(self, _):
        print("Final bounding box:", self.bbox)

# -----------------------------------------------------------------------------
# Core timestack generator 
# -----------------------------------------------------------------------------

def generate_calibrated_timestack(image_files, bbox, resolution_x_m, output_path, progress_callback=None):
    x, y, w, h = bbox

    # sanity‐check the box immediately
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid bounding box {bbox}: width and height must be > 0")

    timestack_lines = []
    image_files.sort(key=lambda f: datetime.strptime(
        "_".join(os.path.basename(f).split("_")[0:6]),
        "%Y_%m_%d_%H_%M_%S"
    ))
    total = len(image_files)

    for i, file in enumerate(image_files, 1):
        # load the image
        if file.lower().endswith(".tif"):
            img = tifffile.imread(file)
        else:
            img = np.array(Image.open(file).convert("RGB"))

        # extract ROI
        roi = img[y : y + h, x : x + w]

        # if it’s empty, warn + skip
        if roi.size == 0:
            print(f"Warning: Skipping '{os.path.basename(file)}' (empty ROI)")  
            if progress_callback:
                progress_callback(i, total)
            continue

        # normalize planes to 3 channels
        if roi.ndim == 2:
            roi = np.stack([roi] * 3, axis=-1)
        elif roi.shape[2] > 3:
            roi = roi[:, :, :3]

        # compute the mean‐line
        line_avg = np.round(np.mean(roi, axis=0)).astype(np.uint8)
        timestack_lines.append(line_avg)

        if progress_callback:
            progress_callback(i, total)

    if not timestack_lines:
        raise ValueError("No valid ROI slices—nothing to stack.")

    # stack + save (w,h>0 guaranteed)
    pseudo_ts = np.stack(timestack_lines, axis=0)
    out_img = Image.fromarray(pseudo_ts)
    if out_img.width != w:
        out_img = out_img.resize((w, pseudo_ts.shape[0]), Image.NEAREST)

    info = PngInfo()
    info.add_text("pixel_resolution", f"{resolution_x_m:.6f}")
    info.add_text("bounding_box", f"{x},{y},{w},{h}")
    out_img.save(output_path, format="PNG", pnginfo=info)

    return output_path



# -----------------------------------------------------------------------------
# Utility to get resource path
# -----------------------------------------------------------------------------

def resource_path(relative_path: str) -> str:
    try:
        base_path = sys._MEIPASS  # PyInstaller temp folder
    except Exception:
        base_path = os.path.dirname(__file__)
    return os.path.join(base_path, relative_path)

# -----------------------------------------------------------------------------
# Main GUI class – TimestackTool
# -----------------------------------------------------------------------------
class TimestackTool(ctk.CTkToplevel):
    """GUI for single and batch time‑stack creation."""

    def __init__(self, master=None):
        super().__init__(master)
        self.title("Time‑stacking Tool")
        self.geometry("1200x800")
        try:
            self.iconbitmap(resource_path("launch_logo.ico"))
        except Exception:
            pass

        # ---------------- State variables ------------------------------------
        self.input_folder = tk.StringVar()
        self.output_folder = tk.StringVar()
        self.batch_folder = tk.StringVar()

        self.bbox = None  # (x, y, w, h)
        self.add_bbox_text = tk.BooleanVar(value=False)
        self.bbox_text = tk.StringVar()

        self.default_resolution = 0.25  # m / pixel fallback
        self.identified_res_var = tk.StringVar(value="Not identified")
        self.add_resolution_manual = tk.BooleanVar(value=False)

        # ---------------- UI --------------------------------------------------
        self._build_ui()
        # Redirect stdout to internal console
        self.stdout_redirector = StdoutRedirector(self.console_text)
        sys.stdout = self.stdout_redirector
        sys.stderr = self.stdout_redirector
        print("Here you may see console outputs\n")

    # ---------------------------------------------------------------------
    # UI builder helper
    # ---------------------------------------------------------------------
    def _build_ui(self):
        # Top: preview image ----------------------------------------------
        self.top_frame = ctk.CTkFrame(self, height=400)
        self.top_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.image_label = ctk.CTkLabel(self.top_frame, text="")
        self.image_label.pack(expand=True)

        # Bottom: control panels ------------------------------------------
        self.bottom_frame = ctk.CTkFrame(self, height=250)
        self.bottom_frame.pack(fill="x", padx=10, pady=10)

        # ---- Input / ROI panel -----------------------------------------
        self.input_panel = ctk.CTkFrame(self.bottom_frame)
        self.input_panel.pack(fill="x", padx=5, pady=5)
        ctk.CTkButton(self.input_panel, text="Browse Input Folder", command=self.browse_input_folder).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ctk.CTkLabel(self.input_panel, textvariable=self.input_folder).grid(row=0, column=1, padx=5, pady=5, sticky="w")
        ctk.CTkButton(self.input_panel, text="Select BBox", command=self.select_bbox).grid(row=0, column=2, padx=5, pady=5)
        ctk.CTkCheckBox(self.input_panel, text="Add bbox as text", variable=self.add_bbox_text, command=self.toggle_bbox_entry).grid(row=0, column=3, padx=5, pady=5)
        self.bbox_entry = ctk.CTkEntry(self.input_panel, textvariable=self.bbox_text)
        self.bbox_entry.grid(row=0, column=4, padx=5, pady=5)
        self.bbox_entry.configure(state="disabled")

        # ---- Resolution panel -------------------------------------------
        self.res_panel = ctk.CTkFrame(self.bottom_frame)
        self.res_panel.pack(fill="x", padx=5, pady=5)
        ctk.CTkLabel(self.res_panel, text="Identified resolution:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ctk.CTkLabel(self.res_panel, textvariable=self.identified_res_var).grid(row=0, column=1, padx=5, pady=5, sticky="w")
        ctk.CTkCheckBox(self.res_panel, text="Add pixel resolution manually", variable=self.add_resolution_manual, command=self.toggle_resolution_entry).grid(row=0, column=2, padx=5, pady=5)
        self.resolution_entry = ctk.CTkEntry(self.res_panel)
        self.resolution_entry.grid(row=0, column=3, padx=5, pady=5)
        self.resolution_entry.configure(state="disabled")
        ctk.CTkLabel(self.res_panel, text="m").grid(row=0, column=4, padx=5, pady=5, sticky="w")

        # ---- Output & single‑timestack panel ---------------------------
        self.output_panel = ctk.CTkFrame(self.bottom_frame)
        self.output_panel.pack(fill="x", padx=5, pady=5)
        ctk.CTkButton(self.output_panel, text="Select Output Folder", command=self.select_output_folder).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ctk.CTkLabel(self.output_panel, textvariable=self.output_folder).grid(row=0, column=1, padx=5, pady=5, sticky="w")
        ctk.CTkButton(self.output_panel, text="Create Raw Timestack", command=self.create_timestack).grid(row=0, column=2, padx=5, pady=5)
        self.single_progress_bar = ctk.CTkProgressBar(self.output_panel)
        self.single_progress_bar.grid(row=0, column=3, padx=5, pady=5)
        self.single_progress_bar.set(0)
        self.single_progress_label = ctk.CTkLabel(self.output_panel, text="")
        self.single_progress_label.grid(row=0, column=4, padx=5, pady=5, sticky="w")

        # ---- Batch panel ------------------------------------------------
        self.batch_panel = ctk.CTkFrame(self.bottom_frame)
        self.batch_panel.pack(fill="x", padx=5, pady=5)
        ctk.CTkButton(self.batch_panel, text="Select Batch Folder", command=self.browse_batch_folder).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ctk.CTkLabel(self.batch_panel, textvariable=self.batch_folder).grid(row=0, column=1, padx=5, pady=5, sticky="w")
        ctk.CTkButton(self.batch_panel, text="Batch Process", command=self.batch_process).grid(row=0, column=2, padx=5, pady=5)
        self.batch_progress_bar = ctk.CTkProgressBar(self.batch_panel)
        self.batch_progress_bar.grid(row=0, column=3, padx=5, pady=5)
        self.batch_progress_bar.set(0)
        self.batch_progress_label = ctk.CTkLabel(self.batch_panel, text="")
        self.batch_progress_label.grid(row=0, column=4, padx=5, pady=5, sticky="w")

        # ---- Console panel ---------------------------------------------
        self.console_frame = ctk.CTkFrame(self)
        self.console_frame.pack(side="bottom", fill="both", expand=False, padx=10, pady=10)
        self.console_text = tk.Text(self.console_frame, wrap="word", height=10)
        self.console_text.pack(fill="both", expand=True, padx=5, pady=5)

    # ------------------------------------------------------------------
    # Simple UI helpers
    # ------------------------------------------------------------------
    def toggle_bbox_entry(self):
        self.bbox_entry.configure(state="normal" if self.add_bbox_text.get() else "disabled")

    def toggle_resolution_entry(self):
        self.resolution_entry.configure(state="normal" if self.add_resolution_manual.get() else "disabled")

    def browse_input_folder(self):
        folder = filedialog.askdirectory(title="Select Input Folder with Images")
        if folder:
            self.input_folder.set(folder)

    def select_output_folder(self):
        folder = filedialog.askdirectory(title="Select Output Folder for Raw Timestack")
        if folder:
            self.output_folder.set(folder)

    def browse_batch_folder(self):
        folder = filedialog.askdirectory(title="Select Main Batch Folder (sub‑folders per batch)")
        if folder:
            self.batch_folder.set(folder)

    # ------------------------------------------------------------------
    # Bounding‑box selection via helper window
    # ------------------------------------------------------------------
    def select_bbox(self):
        folder = self.input_folder.get().strip()
        if not folder:
            messagebox.showerror("Error", "Please select an input folder first.")
            return
        image_files = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.tif"):
            image_files.extend(glob.glob(os.path.join(folder, ext)))
        if not image_files:
            messagebox.showerror("Error", "No supported image files found in folder.")
            return
        selector_win = tk.Toplevel(self)
        selector = ScrollZoomBBoxSelector(selector_win)
        selector.pack(fill="both", expand=True)
        selector.load_image(image_files[0])
        ctk.CTkButton(selector_win, text="Confirm", command=lambda: self._save_bbox(selector, selector_win)).pack(pady=10)

    def _save_bbox(self, selector, win):
        if selector.bbox == (0, 0, 0, 0):
            messagebox.showwarning("Warning", "No bounding box selected.")
            return
        self.bbox = selector.bbox
        self.bbox_text.set(",".join(map(str, self.bbox)))
        # Try to auto‑detect resolution from first .tif in input folder
        folder = self.input_folder.get().strip()
        tif_files = glob.glob(os.path.join(folder, "*.tif"))
        if tif_files:
            try:
                with rasterio.open(tif_files[0]) as src:
                    res = abs(src.transform.a)
            except Exception:
                res = self.default_resolution
        else:
            res = self.default_resolution
        self.identified_res_var.set(f"{res:.3f} m")
        win.destroy()

    # ------------------------------------------------------------------
    # Single timestack creation
    # ------------------------------------------------------------------
    def create_timestack(self):
        folder = self.input_folder.get().strip()
        out_folder = self.output_folder.get().strip()
        if not folder:
            messagebox.showerror("Error", "Select an input folder first.")
            return
        if not out_folder:
            messagebox.showerror("Error", "Select an output folder first.")
            return

        # Parse bbox text if checkbox ticked --------------------------------
        if self.add_bbox_text.get() and self.bbox_text.get().strip():
            if not self._parse_bbox_text():
                return
        if not self.bbox or self.bbox == (0, 0, 0, 0):
            messagebox.showerror("Error", "Identify a valid bounding box first.")
            return

        # Collect images ------------------------------------------------------
        image_files = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.tif"):
            image_files.extend(glob.glob(os.path.join(folder, ext)))
        if not image_files:
            messagebox.showerror("Error", "No supported image files found in input folder.")
            return

        # Resolution ----------------------------------------------------------
        if self.add_resolution_manual.get():
            try:
                resolution_x_m = float(self.resolution_entry.get())
            except ValueError:
                messagebox.showerror("Error", "Invalid manual resolution.")
                return
        else:
            try:
                resolution_x_m = float(self.identified_res_var.get().split()[0])
            except Exception:
                resolution_x_m = self.default_resolution

        # Output filename -----------------------------------------------------
        image_files.sort(key=lambda f: os.path.basename(f))
        first_ts = "_".join(os.path.basename(image_files[0]).split("_")[0:5])
        out_name = f"{first_ts}_raw_timestack.png"
        out_path = os.path.join(out_folder, out_name)

        # Run in background ----------------------------------------------------
        def worker():
            def _update(cur, total):
                self.single_progress_bar.set(cur / total)
                self.single_progress_label.configure(text=f"{cur} / {total}")
            try:
                gen_path = generate_calibrated_timestack(image_files, self.bbox, resolution_x_m, out_path, _update)
                messagebox.showinfo("Success", f"Timestack saved to:\n{gen_path}")
                pil_img = Image.open(gen_path)
                preview = ctk.CTkImage(light_image=pil_img, size=(800, min(600, pil_img.height)))
                self.image_label.configure(image=preview)
                self.image_label.image = preview
            except Exception as exc:
                messagebox.showerror("Error", str(exc))
            finally:
                self.single_progress_bar.set(0)
                self.single_progress_label.configure(text="")

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------
    def batch_process(self):
        main_batch_folder = self.batch_folder.get().strip()
        out_folder = self.output_folder.get().strip()
        if not main_batch_folder:
            messagebox.showerror("Error", "Select a batch folder first.")
            return
        if not out_folder:
            messagebox.showerror("Error", "Select an output folder first.")
            return

        # Allow bbox via text
        if self.add_bbox_text.get() and self.bbox_text.get().strip():
            if not self._parse_bbox_text():
                return
        if not self.bbox or self.bbox == (0, 0, 0, 0):
            messagebox.showerror("Error", "Provide a valid bounding box (via selector or text).")
            return

        # Resolution
        if self.add_resolution_manual.get():
            try:
                resolution_x_m = float(self.resolution_entry.get())
            except ValueError:
                messagebox.showerror("Error", "Invalid manual resolution.")
                return
        else:
            try:
                resolution_x_m = float(self.identified_res_var.get().split()[0])
            except Exception:
                resolution_x_m = self.default_resolution

        # Collect sub-folders
        subfolders = [
            os.path.join(main_batch_folder, d)
            for d in os.listdir(main_batch_folder)
            if os.path.isdir(os.path.join(main_batch_folder, d))
        ]
        if not subfolders:
            messagebox.showerror("Error", "No sub-folders found in the selected batch folder.")
            return
        total = len(subfolders)

        # Reset batch progress bar & ETA label
        self.batch_progress_bar.set(0)
        self.batch_progress_label.configure(text="ETA: --")

        # Prepare a place to store our start time
        self.batch_start_time = None

        def process():
            # record when we actually start
            self.batch_start_time = time.time()
            processed = 0

            for sub in subfolders:
                # Gather images inside subfolder
                img_files = []
                for ext in ("*.jpg", "*.jpeg", "*.png", "*.tif"):
                    img_files.extend(glob.glob(os.path.join(sub, ext)))

                if img_files:
                    img_files.sort(key=lambda f: os.path.basename(f))
                    first_ts = "_".join(os.path.basename(img_files[0]).split("_")[0:5])
                    out_name = f"{first_ts}_raw_timestack.png"
                    out_path = os.path.join(out_folder, out_name)
                    try:
                        generate_calibrated_timestack(img_files, self.bbox, resolution_x_m, out_path)
                    except Exception as exc:
                        print(f"Error processing {sub}: {exc}")

                processed += 1
                _update_ui(processed, total)

            messagebox.showinfo("Batch Done", f"Processed {processed} sub-folders.")
            self.batch_progress_bar.set(0)
            self.batch_progress_label.configure(text="")

        def _update_ui(done, tot):
            # update progress bar
            frac = done / tot
            self.batch_progress_bar.set(frac)

            # compute ETA using the instance attribute
            elapsed = time.time() - self.batch_start_time
            if done > 0:
                rem = (elapsed / done) * (tot - done)
            else:
                rem = 0
            mins = int(rem) // 60
            secs = int(rem) % 60
            eta_str = f"{mins}m {secs}s" if mins else f"{secs}s"

            # show ETA next to the bar
            self.batch_progress_label.configure(text=f"ETA: {eta_str}")

        threading.Thread(target=process, daemon=True).start()


    # ------------------------------------------------------------------
    # Helper: parse bbox text 
    # ------------------------------------------------------------------
    def _parse_bbox_text(self):
        try:
            parts = [int(p.strip()) for p in self.bbox_text.get().split(",")]
            if len(parts) != 4:
                raise ValueError("Must be 4 values: x,y,w,h")
            x, y, w, h = parts
            if w <= 0 or h <= 0:
                raise ValueError("Width and height must both be > 0")
            self.bbox = (x, y, w, h)
            return True
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid bbox: {e}")
            return False


if __name__ == "__main__":
    root = ctk.CTk()
    root.withdraw()
    TimestackTool(master=root)
    root.mainloop()
