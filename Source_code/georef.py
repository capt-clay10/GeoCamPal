import numpy as np
from PIL import Image, ImageTk
import os
import cv2
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import glob
from osgeo import gdal, osr
osr.DontUseExceptions()
import sys
from tkinter import ttk
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")


def resource_path(relative_path: str) -> str:
    try:
        base_path = sys._MEIPASS  # If running in a PyInstaller .exe
    except Exception:
        base_path = os.path.dirname(__file__)  # Running directly from source
    return os.path.join(base_path, relative_path)


class StdoutRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        # Insert the message into the text widget and scroll to the end.
        self.text_widget.insert(tk.END, message)
        self.text_widget.see(tk.END)

    def flush(self):
        pass  # For compatibility with Python's IO system.


class GeoReferenceModule(ctk.CTkToplevel):
    def __init__(self, master=None, *args, **kwargs):
        super().__init__(master=master, *args, **kwargs)
        self.title("Advanced Georeferencing Tool")
        self.geometry("1200x800")

        try:
            self.iconbitmap(resource_path("launch_logo.ico"))
        except Exception as e:
            print("Warning: Could not load window icon:", e)

        self.H = None  # Homography matrix
        self.image_list = []
        self.current_index = 0
        self.input_folder = ""
        self.output_folder = ""
        self.scale_factor = 4  # Default scale factor; user can override
        self.current_zoom = 1.0
        self.initialize_components()

        # ---- Add CONSOLE PANEL at the bottom (new row 5) ----
        self.grid_rowconfigure(5, weight=0)
        self.console_frame = ctk.CTkFrame(self)
        self.console_frame.grid(row=5, column=0, sticky="nsew", padx=5, pady=5)
        self.console_text = tk.Text(self.console_frame, wrap="word", height=10)
        self.console_text.pack(fill="both", expand=True, padx=5, pady=5)

        # Redirect stdout and stderr to the console text widget.
        self.stdout_redirector = StdoutRedirector(self.console_text)
        sys.stdout = self.stdout_redirector
        sys.stderr = self.stdout_redirector
        print("Here you may see console outputs")

    def initialize_components(self):
        # Configure grid layout for five rows
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=2)  # Top panel (images)
        self.grid_rowconfigure(1, weight=1)  # File controls row
        self.grid_rowconfigure(2, weight=1)  # AOI selection row
        self.grid_rowconfigure(3, weight=1)  # Cropping adjustment row
        self.grid_rowconfigure(4, weight=1)  # Final processing row

        # ---- TOP PANEL: Three Panels Side by Side ----
        self.top_panel = ctk.CTkFrame(self)
        self.top_panel.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Original Image Panel (Left)
        self.orig_frame = ctk.CTkFrame(self.top_panel, width=400, height=400)
        self.orig_frame.pack_propagate(False)
        self.orig_frame.pack(side="left", fill="both",
                             expand=True, padx=5, pady=5)
        self.orig_label = ctk.CTkLabel(self.orig_frame, text="Original Image")
        self.orig_label.pack(fill="both", expand=True)

        # Initial Georeferenced Image Panel (Middle)
        self.geo_frame = ctk.CTkFrame(self.top_panel, width=400, height=400)
        self.geo_frame.pack_propagate(False)
        self.geo_frame.pack(side="left", fill="both",
                            expand=True, padx=5, pady=5)
        self.geo_label = ctk.CTkLabel(
            self.geo_frame, text="Initial Georeferenced Image")
        self.geo_label.pack(fill="both", expand=True)

        # Cropped Georeferenced Image Panel (Right)
        self.cropped_frame = ctk.CTkFrame(
            self.top_panel, width=400, height=400)
        self.cropped_frame.pack_propagate(False)
        self.cropped_frame.pack(side="left", fill="both",
                                expand=True, padx=5, pady=5)
        self.cropped_label = ctk.CTkLabel(
            self.cropped_frame, text="Cropped Georeferenced Image")
        self.cropped_label.pack(fill="both", expand=True)

        # ---- SECOND ROW: File Controls ----
        self.control_panel = ctk.CTkFrame(self)
        self.control_panel.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        self.file_frame = ctk.CTkFrame(self.control_panel)
        self.file_frame.pack(side="left", padx=10, pady=5)

        self.image_entry = ctk.CTkEntry(self.file_frame, width=300)
        self.image_entry.pack(side="left", padx=5)
        ctk.CTkButton(self.file_frame, text="Browse Image",
                      command=self.load_image).pack(side="left", padx=5)

        self.homo_entry = ctk.CTkEntry(self.file_frame, width=300)
        self.homo_entry.pack(side="left", padx=5)
        ctk.CTkButton(self.file_frame, text="Load Homography",
                      command=self.load_homography).pack(side="left", padx=5)

        ctk.CTkButton(self.control_panel, text="Initial Georeferencing",
                      command=self.perform_initial_georeferencing).pack(side="left", padx=10)

        # ---- THIRD ROW: AOI Selection and Secondary Georeferencing ----
        # All controls in one row: radio buttons, secondary georeferencing button, and manual AOI entry.
        self.aoi_panel = ctk.CTkFrame(self)
        self.aoi_panel.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)

        self.aoi_var = ctk.StringVar(value="bottom_left")
        aoi_options = [
            ("Bottom Left", "bottom_left"),
            ("Bottom Right", "bottom_right"),
            ("Top Left", "top_left"),
            ("Top Right", "top_right"),
            ("Manual", "manual")
        ]
        col = 0
        for text, value in aoi_options:
            rb = ctk.CTkRadioButton(
                self.aoi_panel, text=text, variable=self.aoi_var, value=value)
            rb.grid(row=0, column=col, padx=5, pady=5, sticky="w")
            col += 1

        # Secondary Georeferencing button (same row)
        ctk.CTkButton(self.aoi_panel, text="Secondary Georeferencing",
                      command=self.perform_secondary_georeferencing).grid(row=0, column=col, padx=5, pady=5, sticky="w")
        col += 1

        # Manual AOI entry (visible only if "Manual" is selected) in the same row
        self.manual_entry = ctk.CTkEntry(self.aoi_panel, width=300,
                                         placeholder_text="x1,y1,x2,y2,x3,y3,x4,y4")
        self.manual_entry.grid(row=0, column=col, padx=5, pady=5, sticky="w")
        self.manual_entry.grid_remove()  # Hide by default
        self.aoi_var.trace_add("write", self.toggle_manual_entry)

        # ---- FOURTH ROW: Cropping Adjustment & Scale Factor & Show Crop Button ----
        # Place cropping entries, a scale factor input, and the Show Crop button in one row.
        self.crop_panel = ctk.CTkFrame(self)
        self.crop_panel.grid(row=3, column=0, sticky="nsew", padx=5, pady=5)

        crop_labels = ["South Crop (min_y)", "North Extend (max_y)",
                       "East Adjust (max_x)", "West Extend (min_x)"]
        self.crop_entries = []
        col = 0
        for label in crop_labels:
            ctk.CTkLabel(self.crop_panel, text=label).grid(
                row=0, column=col, padx=5)
            col += 1
            entry = ctk.CTkEntry(self.crop_panel, width=80)
            entry.grid(row=0, column=col, padx=5)
            self.crop_entries.append(entry)
            col += 1

        # New: Scale Factor entry (overrides default if provided)
        ctk.CTkLabel(self.crop_panel, text="Scale Factor").grid(
            row=0, column=col, padx=5)
        col += 1
        self.scale_entry = ctk.CTkEntry(self.crop_panel, width=80)
        self.scale_entry.grid(row=0, column=col, padx=5)
        col += 1

        # Show Crop button aligned in the same row
        ctk.CTkButton(self.crop_panel, text="Show Crop",
                      command=self.show_crop).grid(row=0, column=col, padx=5)

        # ---- FIFTH ROW: Final Processing (Batch) ----
        self.final_panel = ctk.CTkFrame(self)
        self.final_panel.grid(row=4, column=0, sticky="nsew", padx=5, pady=5)

        # Browse Input Folder button and label
        self.input_folder_label = ctk.CTkLabel(
            self.final_panel, text="Input Folder: Not selected")
        self.input_folder_label.pack(side="left", padx=5)
        ctk.CTkButton(self.final_panel, text="Browse Input Folder",
                      command=self.load_folder).pack(side="left", padx=5)

        # Browse Output Folder button and label
        self.output_folder_label = ctk.CTkLabel(
            self.final_panel, text="Output Folder: Not selected")
        self.output_folder_label.pack(side="left", padx=5)
        ctk.CTkButton(self.final_panel, text="Browse Output Folder",
                      command=self.select_output_folder).pack(side="left", padx=5)

        # EPSG (CRS) entry for spatial reference
        self.epsg_entry = ctk.CTkEntry(
            self.final_panel, width=80, placeholder_text="EPSG Code")
        self.epsg_entry.pack(side="left", padx=5)

        ctk.CTkButton(self.final_panel, text="Final Georeferencing",
                      command=self.process_all_images).pack(side="left", padx=5)
        self.progress = ttk.Progressbar(
            self.final_panel, orient="horizontal", mode="determinate")
        self.progress.pack(side="left", fill="x", expand=True, padx=5)

    def toggle_manual_entry(self, *args):
        if self.aoi_var.get() == "manual":
            self.manual_entry.grid()
        else:
            self.manual_entry.grid_remove()

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

    # --- Load folder for batch processing (Input Folder) ---
    def load_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.input_folder = folder
            # Filter only image files (avoid non-image files like .zip)
            self.image_list = sorted(glob.glob(os.path.join(folder, "*.bmp")) +
                                     glob.glob(os.path.join(folder, "*.jpg")) +
                                     glob.glob(os.path.join(folder, "*.jpeg")) +
                                     glob.glob(os.path.join(folder, "*.png")) +
                                     glob.glob(os.path.join(folder, "*.tif")))
            self.input_folder_label.configure(text=f"Input Folder: {folder}")
            messagebox.showinfo(
                "Info", f"Loaded {len(self.image_list)} images")

    # --- Select Output Folder ---
    def select_output_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.output_folder = folder
            self.output_folder_label.configure(text=f"Output Folder: {folder}")

    # --- Batch processing: Final Georeferencing using original warp_and_autocrop_image logic ---
    def process_all_images(self):
        if not self.image_list or self.H is None:
            messagebox.showerror("Error", "Load a folder and homography first")
            return

        def process():
            total = len(self.image_list)
            for i, path in enumerate(self.image_list):
                try:
                    base_name = os.path.splitext(os.path.basename(path))[0]
                    output_path = os.path.join(
                        self.output_folder, f"{base_name}.tif")
                    self.georeference_and_save_image(path, output_path)
                    self.progress["value"] = (i + 1) / total * 100
                    self.update_idletasks()
                except Exception as e:
                    print(f"Error processing {path}: {str(e)}")
            messagebox.showinfo("Complete", "Processing finished!")

        threading.Thread(target=process, daemon=True).start()

    # --- Helper for batch processing that writes a 4-band GeoTIFF ---
    def georeference_and_save_image(self, img_path, output_path):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Unable to read {img_path}")
            return False
        h, w = img.shape[:2]

        # Use AOI selection for processing:
        original_corners = self.get_original_corners(h, w)
        transformed = cv2.perspectiveTransform(original_corners, self.H)
        min_coords = np.floor(transformed.min(axis=(0, 1))).astype(int)
        max_coords = np.ceil(transformed.max(axis=(0, 1))).astype(int)

        # Retrieve cropping adjustments from the cropping entries
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

        # --- AUTO-CROP: Adjust image to non-black pixels using contour detection ---
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x_crop, y_crop, w_crop, h_crop = cv2.boundingRect(contours[0])
            warped = warped[y_crop:y_crop+h_crop, x_crop:x_crop+w_crop]
        else:
            x_crop, y_crop = 0, 0

        # Compute GCPs (adjusting pixel values by auto-crop offsets)
        adjusted_corners = cv2.perspectiveTransform(
            original_corners, H_final).reshape(-1, 2)
        gcps = []
        for (px, py), (utm_x, utm_y) in zip(adjusted_corners, transformed.reshape(-1, 2)):
            gcp = gdal.GCP()
            gcp.GCPX = float(utm_x)
            gcp.GCPY = float(utm_y)
            # Adjust pixel coordinates for auto-crop offsets:
            gcp.GCPPixel = float(px - x_crop)
            gcp.GCPLine = float(py - y_crop)
            gcps.append(gcp)

        # Get user EPSG code from the final panel
        try:
            user_epsg = int(self.epsg_entry.get())
        except Exception as e:
            messagebox.showerror(
                "Error", "Invalid EPSG code provided. Please enter a valid EPSG number.")
            return False

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(user_epsg)

        driver = gdal.GetDriverByName("GTiff")
        dst_ds = driver.Create(
            output_path, warped.shape[1], warped.shape[0], 4, gdal.GDT_Byte, options=["ALPHA=YES"])
        dst_ds.SetGCPs(gcps, srs.ExportToWkt())
        geotransform = gdal.GCPsToGeoTransform(gcps)
        if geotransform is not None:
            dst_ds.SetGeoTransform(geotransform)
        else:
            print("Warning: Unable to compute geotransform from GCPs.")

        rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        for band_idx in range(3):
            out_band = dst_ds.GetRasterBand(band_idx + 1)
            out_band.WriteArray(rgb[:, :, band_idx])
        alpha = np.where(
            np.all(rgb == [0, 0, 0], axis=-1), 0, 255).astype(np.uint8)
        dst_ds.GetRasterBand(4).WriteArray(alpha)
        dst_ds.FlushCache()
        dst_ds = None
        return True


def main():
    root = ctk.CTk()
    root.withdraw()               # hide the dummy root
    win = GeoReferenceModule(master=root)
    win.mainloop()

if __name__ == "__main__":
    main()
