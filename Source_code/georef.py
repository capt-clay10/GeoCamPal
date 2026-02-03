import os
import sys
import time
import glob
import threading
import concurrent.futures                 
import numpy as np
import cv2
from PIL import Image, ImageTk
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
from osgeo import gdal, osr
osr.DontUseExceptions()

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")

# %% ------------------------------------------------------------------------------------------------------------ helpers ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def resource_path(relative_path: str) -> str:
    """Return absolute path to bundled resources (handles PyInstaller)."""
    try:                       # PyInstaller puts files in a temp folder
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.dirname(__file__)
    return os.path.join(base_path, relative_path)


class StdoutRedirector:
    """Redirect print/traceback output to a Tk Text widget."""
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        self.text_widget.insert(tk.END, message)
        self.text_widget.see(tk.END)

    def flush(self):
        pass


MAX_BATCH_WORKERS = max(1, min(4, os.cpu_count() or 1))

# Minimum valid GeoTIFF size threshold (in bytes) - files smaller than this are considered corrupted
MIN_VALID_GEOTIFF_SIZE = 10_000  # 10 KB - absolute minimum floor

# Maximum retries per individual file before giving up
MAX_RETRIES_PER_FILE = 10


def validate_geotiff(filepath: str, min_size: int = MIN_VALID_GEOTIFF_SIZE) -> tuple[bool, str]:
    """
    Validate a GeoTIFF file for corruption.
    
    Returns:
        tuple: (is_valid: bool, reason: str)
    """
    if not os.path.exists(filepath):
        return False, "File does not exist"
    
    file_size = os.path.getsize(filepath)
    if file_size < min_size:
        return False, f"File too small ({file_size:,} bytes < {min_size:,} bytes threshold)"
    
    # Try to open with GDAL for deeper validation
    try:
        ds = gdal.Open(filepath)
        if ds is None:
            return False, "GDAL could not open file"
        
        width = ds.RasterXSize
        height = ds.RasterYSize
        bands = ds.RasterCount
        
        # Check for obviously corrupt dimensions
        if width <= 1 or height <= 1:
            ds = None
            return False, f"Invalid dimensions: {width}x{height}"
        
        if bands < 1:
            ds = None
            return False, f"Invalid band count: {bands}"
        
        ds = None
        return True, "Valid"
    
    except Exception as e:
        return False, f"GDAL validation error: {str(e)}"


def calculate_adaptive_threshold(file_sizes: list[int], median_fraction: float = 0.5) -> int:
    """
    Calculate an adaptive minimum file size threshold based on the median file size.
    
    Files significantly smaller than the median are likely corrupted.
    
    Args:
        file_sizes: List of file sizes in bytes
        median_fraction: Fraction of median to use as threshold (default 0.5 = 50% of median)
    
    Returns:
        Adaptive threshold in bytes
    """
    if not file_sizes:
        return MIN_VALID_GEOTIFF_SIZE
    
    sorted_sizes = sorted(file_sizes)
    n = len(sorted_sizes)
    
    # Calculate median
    if n % 2 == 0:
        median_size = (sorted_sizes[n // 2 - 1] + sorted_sizes[n // 2]) / 2
    else:
        median_size = sorted_sizes[n // 2]
    
    # Threshold is 50% of median (files less than half the typical size are suspect)
    # But never less than the absolute minimum floor
    adaptive_threshold = max(MIN_VALID_GEOTIFF_SIZE, int(median_size * median_fraction))
    
    return adaptive_threshold


def sweep_and_validate_batch(output_folder: str, 
                              use_adaptive_threshold: bool = True) -> list[tuple[str, str]]:
    """
    Sweep through all TIF files in a folder and identify corrupted ones.
    
    Args:
        output_folder: Path to the output folder containing TIF files
        use_adaptive_threshold: If True, calculate threshold from median file size
    
    Returns:
        List of tuples: (filepath, reason) for each corrupted file
    """
    tif_files = glob.glob(os.path.join(output_folder, "*.tif"))
    
    if not tif_files:
        return []
    
    # Collect file sizes for adaptive thresholding
    file_sizes = []
    for f in tif_files:
        try:
            file_sizes.append(os.path.getsize(f))
        except:
            pass
    
    if use_adaptive_threshold and file_sizes:
        threshold = calculate_adaptive_threshold(file_sizes)
        median_size = sorted(file_sizes)[len(file_sizes) // 2]
        print(f"[Validation] Median file size: {median_size / 1024 / 1024:.2f} MB | "
              f"Threshold (50% of median): {threshold / 1024 / 1024:.2f} MB")
    else:
        threshold = MIN_VALID_GEOTIFF_SIZE
    
    corrupted_files = []
    for filepath in tif_files:
        is_valid, reason = validate_geotiff(filepath, min_size=threshold)
        if not is_valid:
            corrupted_files.append((filepath, reason))
    
    return corrupted_files  # â‰¤ 4 workers


# %% main window class 
class GeoReferenceModule(ctk.CTkToplevel):

    # ------------------------------------------------------------------------------------ init / UI scaffolding ------------------------------------------------------------------------------------------------------------------------
    def __init__(self, master=None, *args, **kwargs):
        super().__init__(master=master, *args, **kwargs)
        self.title("Georeferencing Tool")
        self.geometry("1200x800")

        try:
            self.iconbitmap(resource_path("launch_logo.ico"))
        except Exception as e:
            print("Warning: Could not load window icon:", e)

        # ---------------- state ----------------
        self.H: np.ndarray | None = None
        self.image_list = []
        self.input_folder = ""
        self.output_folder = ""
        self.batch_main_folder = ""
        self.scale_factor = 4
        self.user_epsg: int | None = None    # â† NEW
        self.executor: concurrent.futures.ThreadPoolExecutor | None = None  # â† NEW
        self._gdal_write_lock = threading.Lock()  # Lock for GDAL write operations

        # ---------------- UI -------------------
        self.initialize_components()

        # -------- console ------
        self.grid_rowconfigure(6, weight=0)
        console_frame = ctk.CTkFrame(self)
        console_frame.grid(row=6, column=0, sticky="nsew", padx=5, pady=5)
        console_text = tk.Text(console_frame, wrap="word", height=8)
        console_text.pack(fill="both", expand=True, padx=5, pady=5)
        sys.stdout = StdoutRedirector(console_text)
        sys.stderr = sys.stdout
        print("Here you may see console outputs\n--------------------------------\n")

        # graceful close
        self.protocol("WM_DELETE_WINDOW", self._on_close)      # â† NEW

    # ---------------- graceful shutdown  ----------------
    def _on_close(self):
        if self.executor:
            self.executor.shutdown(wait=False, cancel_futures=True)
        self.destroy()

    # ------------------------------------------------------------ ETA string ------------------------------------------------------------------------------------------------------------â”€
    @staticmethod
    def _eta_string(start_ts: float, frac_done: float) -> str:
        if frac_done <= 0:
            return "ETA: â€“"
        elapsed = time.time() - start_ts
        remaining = elapsed * (1 / frac_done - 1)
        if remaining >= 3600:
            return f"ETA: {remaining / 3600:.1f} h"
        if remaining >= 60:
            return f"ETA: {remaining / 60:.1f} min"
        return f"ETA: {int(remaining)} s"

    # ---------------------------------------------------------------
    def initialize_components(self):
        # grid rows: 0â€‘top images | 1â€‘file | 2â€‘AOI | 3â€‘crop | 4â€‘final(single)
        #            5â€‘batch(mainâ€‘folder) | 6â€‘console
        self.grid_columnconfigure(0, weight=1)
        for r in range(5):
            self.grid_rowconfigure(r, weight=1)

        # ---- TOP IMAGES  ----------------------------
        self.top_panel = ctk.CTkFrame(self)
        self.top_panel.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.orig_frame = ctk.CTkFrame(self.top_panel, width=400, height=400)
        self.orig_frame.pack_propagate(False)
        self.orig_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        self.orig_label = ctk.CTkLabel(self.orig_frame, text="Original Image")
        self.orig_label.pack(fill="both", expand=True)

        self.geo_frame = ctk.CTkFrame(self.top_panel, width=400, height=400)
        self.geo_frame.pack_propagate(False)
        self.geo_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        self.geo_label = ctk.CTkLabel(self.geo_frame, text="Initial Georeferenced Image")
        self.geo_label.pack(fill="both", expand=True)

        self.cropped_frame = ctk.CTkFrame(self.top_panel, width=400, height=400)
        self.cropped_frame.pack_propagate(False)
        self.cropped_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        self.cropped_label = ctk.CTkLabel(self.cropped_frame, text="Cropped Georeferenced Image")
        self.cropped_label.pack(fill="both", expand=True)

        # ---- 1  FILE CONTROLS -------------------------
        self.control_panel = ctk.CTkFrame(self)
        self.control_panel.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        self.file_frame = ctk.CTkFrame(self.control_panel)
        self.file_frame.pack(side="left", padx=10, pady=5)

        self.image_entry = ctk.CTkEntry(self.file_frame, width=300)
        self.image_entry.pack(side="left", padx=5)
        ctk.CTkButton(self.file_frame, text="Browse Image", command=self.load_image).pack(side="left", padx=5)

        self.homo_entry = ctk.CTkEntry(self.file_frame, width=300)
        self.homo_entry.pack(side="left", padx=5)
        ctk.CTkButton(self.file_frame, text="Load Homography", command=self.load_homography).pack(side="left", padx=5)

        ctk.CTkButton(self.control_panel, text="Initial Georeferencing",
                      command=self.perform_initial_georeferencing).pack(side="left", padx=10)

        # ---- 2  AOI  ----------------------------------
        self.aoi_panel = ctk.CTkFrame(self)
        self.aoi_panel.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)

        self.aoi_var = ctk.StringVar(value="bottom_left")
        for idx, (txt, val) in enumerate([
            ("Bottom Left", "bottom_left"), ("Bottom Right", "bottom_right"),
            ("Top Left", "top_left"), ("Top Right", "top_right"), ("Manual", "manual")]):
            ctk.CTkRadioButton(self.aoi_panel, text=txt, variable=self.aoi_var,
                               value=val).grid(row=0, column=idx, padx=5, pady=5, sticky="w")

        ctk.CTkButton(self.aoi_panel, text="Secondary Georeferencing",
                      command=self.perform_secondary_georeferencing).grid(row=0, column=5, padx=5, pady=5, sticky="w")

        self.manual_entry = ctk.CTkEntry(self.aoi_panel, width=300,
                                         placeholder_text="x1,y1,x2,y2,x3,y3,x4,y4")
        self.manual_entry.grid(row=0, column=6, padx=5, pady=5, sticky="w")
        self.manual_entry.grid_remove()
        self.aoi_var.trace_add("write", self.toggle_manual_entry)

        # ---- 3  CROP ---------------------------------
        self.crop_panel = ctk.CTkFrame(self)
        self.crop_panel.grid(row=3, column=0, sticky="nsew", padx=5, pady=5)

        crop_labels = ["South Crop (min_y)", "North Extend (max_y)",
                       "East Adjust (max_x)", "West Extend (min_x)"]
        self.crop_entries = []
        col = 0
        for lbl in crop_labels:
            ctk.CTkLabel(self.crop_panel, text=lbl).grid(row=0, column=col, padx=5)
            col += 1
            ent = ctk.CTkEntry(self.crop_panel, width=80)
            ent.grid(row=0, column=col, padx=5)
            self.crop_entries.append(ent)
            col += 1

        ctk.CTkLabel(self.crop_panel, text="Scale Factor").grid(row=0, column=col, padx=5)
        col += 1
        self.scale_entry = ctk.CTkEntry(self.crop_panel, width=80)
        self.scale_entry.grid(row=0, column=col, padx=5)
        col += 1
        ctk.CTkButton(self.crop_panel, text="Show Crop", command=self.show_crop).grid(row=0, column=col, padx=5)

        # ---- 4  SINGLEâ€‘FOLDER FINAL  ------
        self.final_panel = ctk.CTkFrame(self)
        self.final_panel.grid(row=4, column=0, sticky="nsew", padx=5, pady=5)

        self.input_folder_label = ctk.CTkLabel(self.final_panel, text="Input Folder: Not selected")
        self.input_folder_label.pack(side="left", padx=5)
        ctk.CTkButton(self.final_panel, text="Browse Input Folder", command=self.load_folder).pack(side="left", padx=5)

        self.output_folder_label = ctk.CTkLabel(self.final_panel, text="Output Folder: Not selected")
        self.output_folder_label.pack(side="left", padx=5)
        ctk.CTkButton(self.final_panel, text="Browse Output Folder", command=self.select_output_folder).pack(side="left", padx=5)

        self.epsg_entry = ctk.CTkEntry(self.final_panel, width=80, placeholder_text="EPSG Code")
        self.epsg_entry.pack(side="left", padx=5)

        ctk.CTkButton(self.final_panel, text="Final Georeferencing",
                      command=self.process_all_images).pack(side="left", padx=5)

        self.progress = ctk.CTkProgressBar(self.final_panel, width=220)  # â† NEW (was ttk)
        self.progress.set(0)
        self.progress.pack(side="left", padx=8)
        self.progress_eta = ctk.CTkLabel(self.final_panel, text="ETA: â€“")  # â† NEW
        self.progress_eta.pack(side="left", padx=5)

        # ---- 5  BATCHâ€‘MAINâ€‘FOLDER PANEL (single bar + ETA) -------
        self.batch_panel = ctk.CTkFrame(self)
        self.batch_panel.grid(row=5, column=0, sticky="nsew", padx=5, pady=5)

        self.batch_main_label = ctk.CTkLabel(self.batch_panel, text="Main Folder: Not selected")
        self.batch_main_label.pack(side="left", padx=5)
        ctk.CTkButton(self.batch_panel, text="Browse Main Folder",
                      command=self.select_batch_main_folder).pack(side="left", padx=5)
        ctk.CTkButton(self.batch_panel, text="Start Batch Process",
                      command=self.start_batch_process).pack(side="left", padx=5)

        self.batch_progress = ctk.CTkProgressBar(self.batch_panel, width=220)  # â† NEW (single bar)
        self.batch_progress.set(0)
        self.batch_progress.pack(side="left", padx=8)
        self.batch_eta_label = ctk.CTkLabel(self.batch_panel, text="ETA: â€“")  # â† NEW
        self.batch_eta_label.pack(side="left", padx=5)
        
        # ---- RESET BUTTON (red, bottom right) ----
        self.reset_button = ctk.CTkButton(
            self.batch_panel, 
            text="⟲ Reset", 
            command=self.reset_module,
            width=80,
            fg_color="#8B0000",      # Dark red
            hover_color="#B22222",   # Firebrick red on hover
            text_color="white"
        )
        self.reset_button.pack(side="right", padx=10)

    def toggle_manual_entry(self, *args):
        if self.aoi_var.get() == "manual":
            self.manual_entry.grid()
        else:
            self.manual_entry.grid_remove()

    def reset_module(self):
        """Reset the module to initial state - clear all images, settings, and entries."""
        # Confirm reset with user
        if not messagebox.askyesno("Confirm Reset", 
                                    "Are you sure you want to reset?\n\n"
                                    "This will clear all images, settings, and entries."):
            return
        
        # ---- Reset state variables ----
        self.H = None
        self.image_list = []
        self.input_folder = ""
        self.output_folder = ""
        self.batch_main_folder = ""
        self.user_epsg = None
        
        # ---- Clear image panels ----
        self.orig_label.configure(image=None, text="Original Image")
        self.orig_label.image = None
        self.geo_label.configure(image=None, text="Initial Georeferenced Image")
        self.geo_label.image = None
        self.cropped_label.configure(image=None, text="Cropped Georeferenced Image")
        self.cropped_label.image = None
        
        # ---- Clear entry fields ----
        self.image_entry.delete(0, "end")
        self.homo_entry.delete(0, "end")
        self.manual_entry.delete(0, "end")
        self.epsg_entry.delete(0, "end")
        self.scale_entry.delete(0, "end")
        
        # Clear crop entries
        for entry in self.crop_entries:
            entry.delete(0, "end")
        
        # ---- Reset labels ----
        self.input_folder_label.configure(text="Input Folder: Not selected")
        self.output_folder_label.configure(text="Output Folder: Not selected")
        self.batch_main_label.configure(text="Main Folder: Not selected")
        
        # ---- Reset progress bars and ETA ----
        self.progress.set(0)
        self.progress_eta.configure(text="ETA: —")
        self.batch_progress.set(0)
        self.batch_eta_label.configure(text="ETA: —")
        
        # ---- Reset AOI selection to default ----
        self.aoi_var.set("bottom_left")
        self.manual_entry.grid_remove()
        
        print("\n" + "="*50)
        print("Module has been reset to initial state")
        print("="*50 + "\n")

    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tif")])
        if path:
            self.image_entry.delete(0, "end")
            self.image_entry.insert(0, path)
            self.show_image(path, self.orig_label)

    def load_homography(self):
        path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        if path:
            try:
                self.H = np.loadtxt(path).reshape(3, 3)
                self.homo_entry.delete(0, "end")
                self.homo_entry.insert(0, path)
            except Exception as e:
                messagebox.showerror(
                    "Error", f"Invalid homography matrix: {str(e)}")

    def show_image(self, path, label):
        try:
            img = Image.open(path)
        except Exception as e:
            messagebox.showerror("Error", f"Unable to open image: {str(e)}")
            return
        # Using thumbnail() may downsize the image; adjust if full display is needed.
        img.thumbnail((500, 400))
        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
        label.configure(image=ctk_img)
        label.image = ctk_img

    # --- INITIAL GEOREFERENCING (using full image corners) ---
    def perform_initial_georeferencing(self):
        if not self.image_entry.get() or self.H is None:
            messagebox.showerror("Error", "Load an image and homography first")
            return
        try:
            img = cv2.imread(self.image_entry.get())
            if img is None:
                raise ValueError("Unable to read image")
            h, w = img.shape[:2]
            full_corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
                                    dtype=np.float32).reshape(-1, 1, 2)
            transformed = self.warp_image(img, full_corners)
            self.show_georeferenced(transformed)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # --- SECONDARY GEOREFERENCING (using AOI) ---
    def perform_secondary_georeferencing(self):
        if not self.image_entry.get() or self.H is None:
            messagebox.showerror("Error", "Load an image and homography first")
            return
        try:
            img = cv2.imread(self.image_entry.get())
            if img is None:
                raise ValueError("Unable to read image")
            h, w = img.shape[:2]
            original_corners = self.get_original_corners(h, w)
            transformed = self.warp_image_with_crop(img, original_corners)
            self.show_georeferenced(transformed)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # --- Get AOI Corners based on selection ---
    def get_original_corners(self, h, w):
        aoi_type = self.aoi_var.get()
        if aoi_type == "manual":
            return self.parse_manual_corners()
        elif aoi_type == "bottom_left":
            return np.array([[0, int(h * 0.5)], [w - 1, int(h * 0.5)],
                             [w - 1, h - 1], [0, h - 1]],
                            dtype=np.float32).reshape(-1, 1, 2)
        elif aoi_type == "bottom_right":
            return np.array([[int(w * 0.5), int(h * 0.5)], [w - 1, int(h * 0.5)],
                             [w - 1, h - 1], [int(w * 0.5), h - 1]],
                            dtype=np.float32).reshape(-1, 1, 2)
        elif aoi_type == "top_left":
            return np.array([[0, 0], [int(w * 0.5), 0],
                             [int(w * 0.5), int(h * 0.5)], [0, int(h * 0.5)]],
                            dtype=np.float32).reshape(-1, 1, 2)
        elif aoi_type == "top_right":
            return np.array([[int(w * 0.5), 0], [w - 1, 0],
                             [w - 1, int(h * 0.5)], [int(w * 0.5), int(h * 0.5)]],
                            dtype=np.float32).reshape(-1, 1, 2)
        else:
            return np.array([[0, 0], [w - 1, 0],
                             [w - 1, h - 1], [0, h - 1]],
                            dtype=np.float32).reshape(-1, 1, 2)

    def parse_manual_corners(self):
        try:
            vals = list(map(float, self.manual_entry.get().split(",")))
            if len(vals) != 8:
                raise ValueError("Must provide 8 comma-separated numbers")
            return np.array(vals, dtype=np.float32).reshape(4, 1, 2)
        except Exception as e:
            raise ValueError(f"Invalid manual corner coordinates: {e}")

    # --- Warp image using given corners (for initial georeferencing) ---
    def warp_image(self, img, original_corners):
        h, w = img.shape[:2]
        transformed = cv2.perspectiveTransform(original_corners, self.H)
        min_coords = np.floor(transformed.min(axis=(0, 1))).astype(int)
        max_coords = np.ceil(transformed.max(axis=(0, 1))).astype(int)
        # Use scale factor from the scale entry if provided, else default self.scale_factor
        try:
            scale = float(self.scale_entry.get())
        except:
            scale = self.scale_factor
        output_width = int((max_coords[0] - min_coords[0]) * scale)
        output_height = int((max_coords[1] - min_coords[1]) * scale)
        translation = np.array([
            [1, 0, -min_coords[0] * scale],
            [0, 1, -min_coords[1] * scale],
            [0, 0, 1]
        ], dtype=np.float32)
        H_scaled = self.H.copy()
        H_scaled[:2] *= scale
        H_final = translation @ H_scaled
        warped = cv2.warpPerspective(
            img, H_final, (output_width, output_height), flags=cv2.INTER_LANCZOS4)
        return cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)

    # --- Warp image with cropping adjustments (for secondary georeferencing and preview) ---
    def warp_image_with_crop(self, img, original_corners):
        h, w = img.shape[:2]
        transformed = cv2.perspectiveTransform(original_corners, self.H)
        min_coords = np.floor(transformed.min(axis=(0, 1))).astype(int)
        max_coords = np.ceil(transformed.max(axis=(0, 1))).astype(int)
        # Read crop adjustment factors (default 0)
        try:
            crop_south = float(self.crop_entries[0].get())
        except:
            crop_south = 0.0
        try:
            crop_north = float(self.crop_entries[1].get())
        except:
            crop_north = 0.0
        try:
            crop_east = float(self.crop_entries[2].get())
        except:
            crop_east = 0.0
        try:
            crop_west = float(self.crop_entries[3].get())
        except:
            crop_west = 0.0

        min_y = min_coords[1] + \
            int((max_coords[1] - min_coords[1]) * crop_south)
        max_y = max_coords[1] + \
            int((max_coords[1] - min_coords[1]) * crop_north)
        max_x = max_coords[0] - \
            int((max_coords[0] - min_coords[0]) * crop_east)
        min_x = min_coords[0] - \
            int((max_coords[0] - min_coords[0]) * crop_west)

        try:
            scale = float(self.scale_entry.get())
        except:
            scale = self.scale_factor
        output_width = int((max_x - min_x) * scale)
        output_height = int((max_y - min_y) * scale)
        translation = np.array([
            [1, 0, -min_x * scale],
            [0, 1, -min_y * scale],
            [0, 0, 1]
        ], dtype=np.float32)
        H_scaled = self.H.copy()
        H_scaled[:2] *= scale
        H_final = translation @ H_scaled
        warped = cv2.warpPerspective(
            img, H_final, (output_width, output_height), flags=cv2.INTER_LANCZOS4)
        return cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)

    # --- Show initial georeferenced image in the middle panel ---
    def show_georeferenced(self, image_array):
        img = Image.fromarray(image_array)
        # Adjust thumbnail if needed; note that this may not show full resolution.
        img.thumbnail((500, 400))
        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
        self.geo_label.configure(image=ctk_img)
        self.geo_label.image = ctk_img

    # --- Show cropped georeferenced image in the right panel (when "Show Crop" is clicked) ---
    def show_crop(self):
        if not self.image_entry.get() or self.H is None:
            messagebox.showerror("Error", "Load an image and homography first")
            return
        try:
            img = cv2.imread(self.image_entry.get())
            if img is None:
                raise ValueError("Unable to read image")
            h, w = img.shape[:2]
            original_corners = self.get_original_corners(h, w)
            cropped_img = self.warp_image_with_crop(img, original_corners)
            self.show_cropped_georeferenced(cropped_img)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def show_cropped_georeferenced(self, image_array):
        img = Image.fromarray(image_array)
        img.thumbnail((500, 400))
        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
        self.cropped_label.configure(image=ctk_img)
        self.cropped_label.image = ctk_img
    # --- Load folder for batch processing -------------------------
    def load_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.input_folder = folder
            self.image_list = sorted(glob.glob(os.path.join(folder, "*.bmp")) +
                                     glob.glob(os.path.join(folder, "*.jpg")) +
                                     glob.glob(os.path.join(folder, "*.jpeg")) +
                                     glob.glob(os.path.join(folder, "*.png")) +
                                     glob.glob(os.path.join(folder, "*.tif")))
            self.input_folder_label.configure(text=f"Input Folder: {folder}")
            messagebox.showinfo("Info", f"Loaded {len(self.image_list)} images")

    def select_output_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.output_folder = folder
            self.output_folder_label.configure(text=f"Output Folder: {folder}")

    # ------------------------------------------------------------------------ georeference_and_save_image ------------------------------------------------------------------------------------------------â”€
    def georeference_and_save_image(self, img_path: str, output_path: str) -> bool:
        """Same as before but uses self.user_epsg (validated once)."""
        img = cv2.imread(img_path)
        if img is None:
            print(f"Unable to read {img_path}")
            return False

        h, w = img.shape[:2]
        original_corners = self.get_original_corners(h, w)
        transformed = cv2.perspectiveTransform(original_corners, self.H)
        min_coords = np.floor(transformed.min(axis=(0, 1))).astype(int)
        max_coords = np.ceil(transformed.max(axis=(0, 1))).astype(int)

        # crop factors
        try:
            crop_south = float(self.crop_entries[0].get() or 0)
            crop_north = float(self.crop_entries[1].get() or 0)
            crop_east  = float(self.crop_entries[2].get() or 0)
            crop_west  = float(self.crop_entries[3].get() or 0)
        except ValueError:
            crop_south = crop_north = crop_east = crop_west = 0

        min_y = min_coords[1] + int((max_coords[1]-min_coords[1]) * crop_south)
        max_y = max_coords[1] + int((max_coords[1]-min_coords[1]) * crop_north)
        max_x = max_coords[0] - int((max_coords[0]-min_coords[0]) * crop_east)
        min_x = min_coords[0] - int((max_coords[0]-min_coords[0]) * crop_west)

        try:
            scale = float(self.scale_entry.get())
        except ValueError:
            scale = self.scale_factor

        output_width  = int((max_x - min_x) * scale)
        output_height = int((max_y - min_y) * scale)

        translation = np.array([[1, 0, -min_x * scale],
                                [0, 1, -min_y * scale],
                                [0, 0, 1                     ]], dtype=np.float32)

        H_scaled      = self.H.copy();  H_scaled[:2] *= scale
        H_final       = translation @ H_scaled
        warped        = cv2.warpPerspective(img, H_final,
                                            (output_width, output_height),
                                            flags=cv2.INTER_LANCZOS4)

        # auto-crop dark borders
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x_crop, y_crop, w_crop, h_crop = cv2.boundingRect(contours[0])
            warped = warped[y_crop:y_crop+h_crop, x_crop:x_crop+w_crop]
        else:
            x_crop = y_crop = 0

        # ---- write GeoTIFF (protected by lock for thread safety) ----------------
        gcps = []
        adj_corners = cv2.perspectiveTransform(original_corners, H_final).reshape(-1, 2)
        for (px, py), (utm_x, utm_y) in zip(adj_corners, transformed.reshape(-1, 2)):
            g = gdal.GCP()
            g.GCPX, g.GCPY = float(utm_x), float(utm_y)
            g.GCPPixel, g.GCPLine = float(px - x_crop), float(py - y_crop)
            gcps.append(g)

        srs = osr.SpatialReference();  srs.ImportFromEPSG(self.user_epsg)
        
        # Use lock to serialize GDAL file creation (prevents race conditions)
        with self._gdal_write_lock:
            driver = gdal.GetDriverByName("GTiff")
            dst_ds = driver.Create(output_path, warped.shape[1], warped.shape[0],
                                   4, gdal.GDT_Byte, options=["ALPHA=YES"])
            if dst_ds is None:
                print(f"GDAL failed to create file: {output_path}")
                return False
            
            dst_ds.SetGCPs(gcps, srs.ExportToWkt())
            gt = gdal.GCPsToGeoTransform(gcps)
            if gt: dst_ds.SetGeoTransform(gt)

            rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
            for i in range(3):
                dst_ds.GetRasterBand(i+1).WriteArray(rgb[..., i])
            alpha = np.where(np.all(rgb == [0,0,0], axis=-1), 0, 255).astype(np.uint8)
            dst_ds.GetRasterBand(4).WriteArray(alpha)
            dst_ds.FlushCache()
            dst_ds = None
        
        return True


    def process_all_images(self):
        if not self.image_list or self.H is None:
            messagebox.showerror("Error", "Load a folder and homography first")
            return
        try:
            self.user_epsg = int(self.epsg_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid EPSG code before running.")
            return

        def _worker():
            total, start_ts = len(self.image_list), time.time()
            # Track input->output mapping for validation
            img_to_input = {}
            
            for idx, path in enumerate(self.image_list, 1):
                try:
                    base = os.path.splitext(os.path.basename(path))[0]
                    out  = os.path.join(self.output_folder, f"{base}.tif")
                    self.georeference_and_save_image(path, out)
                    img_to_input[out] = path
                except Exception as e:
                    print(f"Error processing {path}: {e}")
                self.progress.set(idx/total)
                self.progress_eta.configure(text=self._eta_string(start_ts, idx/total))
                self.update_idletasks()
            
            # ========== VALIDATION AND REPROCESSING ==========
            print("\n[Validation] Starting validation sweep...")
            self.progress_eta.configure(text="Validating...")
            self.update_idletasks()
            
            # Track retry count per file
            file_retry_counts = {}  # output_path -> retry_count
            permanently_failed = []  # Files that failed after MAX_RETRIES_PER_FILE attempts
            
            validation_round = 0
            max_validation_rounds = 20  # Safety limit
            
            while validation_round < max_validation_rounds:
                validation_round += 1
                corrupted_files = sweep_and_validate_batch(self.output_folder, use_adaptive_threshold=True)
                
                # Filter out permanently failed files
                corrupted_files = [(p, r) for p, r in corrupted_files 
                                   if p not in [f[0] for f in permanently_failed]]
                
                if not corrupted_files:
                    print("[Validation] All files validated successfully!")
                    break
                
                print(f"\n[Validation] Round {validation_round}: Found {len(corrupted_files)} corrupted file(s)")
                
                files_to_retry = []
                for corrupted_path, reason in corrupted_files:
                    current_retries = file_retry_counts.get(corrupted_path, 0)
                    
                    if current_retries >= MAX_RETRIES_PER_FILE:
                        if corrupted_path not in [f[0] for f in permanently_failed]:
                            permanently_failed.append((corrupted_path, reason))
                            print(f"  [GAVE UP] {os.path.basename(corrupted_path)} - "
                                  f"failed after {MAX_RETRIES_PER_FILE} attempts")
                    else:
                        files_to_retry.append((corrupted_path, reason))
                        print(f"  - {os.path.basename(corrupted_path)}: {reason} "
                              f"(attempt {current_retries + 1}/{MAX_RETRIES_PER_FILE})")
                
                if not files_to_retry:
                    print(f"\n[Validation] No more files to retry.")
                    break
                
                # Reprocess files
                for corrupted_path, reason in files_to_retry:
                    input_path = img_to_input.get(corrupted_path)
                    if input_path and os.path.exists(input_path):
                        try:
                            print(f"[Reprocess] {os.path.basename(corrupted_path)}...")
                            if os.path.exists(corrupted_path):
                                os.remove(corrupted_path)
                            self.georeference_and_save_image(input_path, corrupted_path)
                            file_retry_counts[corrupted_path] = file_retry_counts.get(corrupted_path, 0) + 1
                        except Exception as e:
                            print(f"[Reprocess] Failed: {e}")
                            file_retry_counts[corrupted_path] = file_retry_counts.get(corrupted_path, 0) + 1
                    else:
                        permanently_failed.append((corrupted_path, "Input file not found"))
                
                self.update_idletasks()
            
            # Final summary
            if permanently_failed:
                print(f"\n" + "="*60)
                print(f"[WARNING] {len(permanently_failed)} FILE(S) COULD NOT BE PROCESSED:")
                print("="*60)
                for path, reason in permanently_failed:
                    print(f"  ✗ {os.path.basename(path)}: {reason}")
                print("="*60 + "\n")
                
                failed_names = "\n".join([os.path.basename(p) for p, _ in permanently_failed[:10]])
                if len(permanently_failed) > 10:
                    failed_names += f"\n... and {len(permanently_failed) - 10} more"
                messagebox.showwarning(
                    "Some Files Failed", 
                    f"{len(permanently_failed)} file(s) could not be processed after "
                    f"{MAX_RETRIES_PER_FILE} attempts each:\n\n{failed_names}"
                )
            
            # ========== END VALIDATION ==========
            
            self.progress_eta.configure(text="Done")
            if not permanently_failed:
                messagebox.showinfo("Complete", "Processing finished and validated!")

        threading.Thread(target=_worker, daemon=True).start()


    
    def select_batch_main_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.batch_main_folder = folder
            self.batch_main_label.configure(text=f"Main Folder: {folder}")
    
    def start_batch_process(self):
        if not self.batch_main_folder:
            messagebox.showerror("Error", "Please select a MAIN folder first.")
            return
        if not self.output_folder:
            messagebox.showerror("Error", "Please select an OUTPUT folder.")
            return
        if self.H is None:
            messagebox.showerror("Error", "Load a homography matrix first.")
            return
        try:
            self.user_epsg = int(self.epsg_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid EPSG code before running.")
            return

        print(f"Batch process has started (≤ {MAX_BATCH_WORKERS} workers)\n")

        def _batch_thread():
            all_subs = [f.path for f in os.scandir(self.batch_main_folder) if f.is_dir()]
            if not all_subs:
                messagebox.showerror("Error", "No sub-folders found.")
                return

            processed = {d for d in os.listdir(self.output_folder)
                           if os.path.isdir(os.path.join(self.output_folder, d))}
            todo = [s for s in sorted(all_subs) if os.path.basename(s) not in processed]

            if (skipped := len(all_subs) - len(todo)):
                print(f"[Batch] Skipping {skipped} sub-folder(s) already done.\n")
            if not todo:
                self.batch_progress.set(1.0); self.batch_eta_label.configure(text="Done")
                messagebox.showinfo("Batch Complete", "All sub-folders already processed.")
                return

            total, start_ts = len(todo), time.time()
            self.batch_progress.set(0)
            
            # Track input->output mapping for validation/reprocessing
            img_to_input = {}  # output_path -> input_path

            def _process_sub(sub):
                """Process a subfolder and return list of (input_path, output_path) tuples."""
                processed_files = []
                imgs = sorted(glob.glob(os.path.join(sub, "*.[jp][pn]*g")) +
                              glob.glob(os.path.join(sub, "*.tif")) +
                              glob.glob(os.path.join(sub, "*.bmp")))
                if not imgs:
                    print(f"[Batch] Skipping empty folder: {sub}")
                    return processed_files
                out_sub = os.path.join(self.output_folder, os.path.basename(sub))
                os.makedirs(out_sub, exist_ok=True)
                for img in imgs:
                    try:
                        out = os.path.join(out_sub,
                                           os.path.splitext(os.path.basename(img))[0] + ".tif")
                        self.georeference_and_save_image(img, out)
                        processed_files.append((img, out))
                    except Exception as e: 
                        print(f"[Batch] Error in {img}: {e}")
                return processed_files

            # pool inside the background thread
            self.executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=MAX_BATCH_WORKERS, thread_name_prefix="batch")
            with self.executor as executor:
                fut2sub = {executor.submit(_process_sub, s): s for s in todo}
                for idx, fut in enumerate(concurrent.futures.as_completed(fut2sub), 1):
                    if fut.exception():
                        print(f"[Batch] Exception in {fut2sub[fut]}: {fut.exception()}")
                    else:
                        # Collect processed file mappings
                        for inp, out in (fut.result() or []):
                            img_to_input[out] = inp
                    frac = idx / total
                    self.batch_progress.set(frac)
                    if idx:
                        remaining = (total - idx) / (idx / (time.time() - start_ts))
                    else:
                        remaining = 0
                    m, s = divmod(int(remaining), 60)
                    self.batch_eta_label.configure(text=f"ETA: {m}m {s}s" if m else f"ETA: {s}s")
                    self.update_idletasks()
            self.executor = None
            
            elapsed = time.time() - start_ts
            m, s = divmod(elapsed, 60)
            print(f"\nInitial batch process complete in {int(m)} min {s:.1f} s.")
            
            # ========== VALIDATION AND REPROCESSING SWEEP ==========
            print("\n[Validation] Starting post-processing validation sweep...")
            self.batch_eta_label.configure(text="Validating...")
            self.update_idletasks()
            
            # Track retry count per file
            file_retry_counts = {}  # output_path -> retry_count
            permanently_failed = []  # Files that failed after MAX_RETRIES_PER_FILE attempts
            
            validation_round = 0
            max_validation_rounds = 20  # Safety limit to prevent infinite loops
            
            while validation_round < max_validation_rounds:
                validation_round += 1
                
                # Sweep all output subfolders
                corrupted_files = []
                out_subs = [f.path for f in os.scandir(self.output_folder) if f.is_dir()]
                
                for out_sub in out_subs:
                    sub_corrupted = sweep_and_validate_batch(out_sub, use_adaptive_threshold=True)
                    corrupted_files.extend(sub_corrupted)
                
                # Filter out files that have already permanently failed
                corrupted_files = [(p, r) for p, r in corrupted_files 
                                   if p not in [f[0] for f in permanently_failed]]
                
                if not corrupted_files:
                    print(f"[Validation] All files validated successfully!")
                    break
                
                print(f"\n[Validation] Round {validation_round}: Found {len(corrupted_files)} corrupted file(s)")
                
                # Reprocess corrupted files (sequentially to avoid race conditions)
                files_to_retry = []
                for corrupted_path, reason in corrupted_files:
                    # Check retry count for this file
                    current_retries = file_retry_counts.get(corrupted_path, 0)
                    
                    if current_retries >= MAX_RETRIES_PER_FILE:
                        # This file has exceeded max retries - mark as permanently failed
                        if corrupted_path not in [f[0] for f in permanently_failed]:
                            permanently_failed.append((corrupted_path, reason))
                            print(f"  [GAVE UP] {os.path.basename(corrupted_path)} - "
                                  f"failed after {MAX_RETRIES_PER_FILE} attempts")
                    else:
                        files_to_retry.append((corrupted_path, reason))
                        print(f"  - {os.path.basename(corrupted_path)}: {reason} "
                              f"(attempt {current_retries + 1}/{MAX_RETRIES_PER_FILE})")
                
                if not files_to_retry:
                    # All remaining corrupted files have exceeded retry limit
                    print(f"\n[Validation] No more files to retry.")
                    break
                
                # Reprocess files that haven't exceeded retry limit
                reprocessed = 0
                for corrupted_path, reason in files_to_retry:
                    input_path = img_to_input.get(corrupted_path)
                    if input_path and os.path.exists(input_path):
                        try:
                            print(f"[Reprocess] {os.path.basename(corrupted_path)}...")
                            # Delete corrupted file first
                            if os.path.exists(corrupted_path):
                                os.remove(corrupted_path)
                            # Reprocess
                            self.georeference_and_save_image(input_path, corrupted_path)
                            reprocessed += 1
                            # Increment retry count
                            file_retry_counts[corrupted_path] = file_retry_counts.get(corrupted_path, 0) + 1
                        except Exception as e:
                            print(f"[Reprocess] Failed to reprocess {corrupted_path}: {e}")
                            file_retry_counts[corrupted_path] = file_retry_counts.get(corrupted_path, 0) + 1
                    else:
                        print(f"[Reprocess] Cannot find input for {corrupted_path}")
                        permanently_failed.append((corrupted_path, "Input file not found"))
                
                print(f"[Validation] Reprocessed {reprocessed} file(s)")
                self.update_idletasks()
            
            # Final summary
            if permanently_failed:
                print(f"\n" + "="*60)
                print(f"[WARNING] {len(permanently_failed)} FILE(S) COULD NOT BE PROCESSED:")
                print("="*60)
                for path, reason in permanently_failed:
                    print(f"  ✗ {os.path.basename(path)}")
                    print(f"    Reason: {reason}")
                    print(f"    Full path: {path}")
                print("="*60)
                print("These files may have issues with the source image or parameters.")
                print("Consider checking them manually.\n")
                
                # Show message box with failed files
                failed_names = "\n".join([os.path.basename(p) for p, _ in permanently_failed[:10]])
                if len(permanently_failed) > 10:
                    failed_names += f"\n... and {len(permanently_failed) - 10} more"
                messagebox.showwarning(
                    "Some Files Failed", 
                    f"{len(permanently_failed)} file(s) could not be processed after "
                    f"{MAX_RETRIES_PER_FILE} attempts each:\n\n{failed_names}\n\n"
                    "Check the console for details."
                )
            
            if validation_round >= max_validation_rounds:
                print(f"\n[WARNING] Reached maximum validation rounds ({max_validation_rounds}). Stopping.")
            
            # ========== END VALIDATION ==========
          
            self.batch_progress.set(1.0)
            self.batch_eta_label.configure(text="Done")
            
            if not permanently_failed:
                messagebox.showinfo("Batch Complete", "All files processed and validated successfully!")
            
            print("\nBatch process complete\n")

        threading.Thread(target=_batch_thread, daemon=True).start()

    # initialise_components and other helper methods â€¦ keep original code here
    # ---------------------------------------------------------------------


def main():
    root = ctk.CTk()
    root.withdraw()
    GeoReferenceModule(master=root).mainloop()


if __name__ == "__main__":
    main()