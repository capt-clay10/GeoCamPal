import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import welch
import customtkinter as ctk
import os
import datetime
import csv
import re
import sys
import json
from collections import defaultdict
from pathlib import Path

# Set appearance
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")

def resource_path(relative_path: str) -> str:
    """
    Get absolute path to resource, works for development and PyInstaller.
    """
    try:
        base_path = sys._MEIPASS  # type: ignore
    except Exception:
        base_path = os.path.dirname(__file__)
    return os.path.join(base_path, relative_path)


class StdoutRedirector:
    """Redirect stdout/stderr to a tk.Text widget."""
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        self.text_widget.insert(tk.END, message)
        self.text_widget.see(tk.END)

    def flush(self):
        pass


# =============================================================================
# Runup Extraction Functions - Supporting Multiple Formats
# =============================================================================

def extract_runup_from_mask(mask_path, resolution_x_m, time_interval_sec, flip_horizontal=False):
    """
    Extract runup from PNG mask image (legacy method - one point per row).
    
    Returns:
        time_array (s), distance_array (m), pixel_coords [(row, col), ...]
    Time convention: 0 s at the BOTTOM row, increasing upward.
    """
    mask_img = Image.open(mask_path).convert("L")
    mask_array = np.array(mask_img)
    H, W = mask_array.shape[:2]

    # Binary mask
    binary_mask = mask_array > 128

    rows, cols = np.where(binary_mask)
    if rows.size == 0:
        return np.array([]), np.array([]), []

    unique_rows = np.unique(rows)

    time_array = []
    distance_array = []
    pixel_coords = []

    for r in unique_rows:
        cols_in_row = cols[rows == r]
        if cols_in_row.size == 0:
            continue
        # Land on left means shoreline at min col; otherwise at max col
        runup_col = np.min(cols_in_row) if flip_horizontal else np.max(cols_in_row)

        # Bottom-origin time: bottom row (H-1) → t=0 ; top row (0) → t=max
        time_val = (H - 1 - r) * time_interval_sec
        distance_val = runup_col * resolution_x_m

        time_array.append(time_val)
        distance_array.append(distance_val)
        pixel_coords.append((r, runup_col))

    return np.array(time_array), np.array(distance_array), pixel_coords


def extract_runup_from_geojson(geojson_path, resolution_x_m, time_interval_sec, 
                                image_height, flip_horizontal=False):
    """
    Extract EXACT runup coordinates from GeoJSON LineString.
    Preserves the full hand-picked shape without losing any detail.
    
    Returns:
        time_array (s), distance_array (m), pixel_coords [(row, col), ...]
    """
    with open(geojson_path, 'r') as f:
        data = json.load(f)
    
    coords = None
    for feature in data.get('features', []):
        if feature['geometry']['type'] == 'LineString':
            coords = np.array(feature['geometry']['coordinates'])
            break
        elif feature['geometry']['type'] == 'Polygon':
            # Use first ring of polygon
            coords = np.array(feature['geometry']['coordinates'][0])
            break
    
    if coords is None or len(coords) == 0:
        return np.array([]), np.array([]), []
    
    # coords are [x, y] = [col, row]
    cols = coords[:, 0]
    rows = coords[:, 1]
    
    # Convert to physical units
    # Time: bottom row (H-1) = t=0, top row (0) = t=max
    time_array = (image_height - 1 - rows) * time_interval_sec
    
    # Distance: column * resolution, optionally flipped
    if flip_horizontal:
        distance_array = (cols.max() - cols) * resolution_x_m
    else:
        distance_array = cols * resolution_x_m
    
    # Pixel coordinates as (row, col) for compatibility
    pixel_coords = [(int(r), int(c)) for r, c in zip(rows, cols)]
    
    return time_array, distance_array, pixel_coords


def extract_runup_from_coco_json(json_path, resolution_x_m, time_interval_sec, 
                                  flip_horizontal=False):
    """
    Extract EXACT runup coordinates from COCO annotation JSON.
    The segmentation polygon contains the exact hand-picked coordinates.
    
    Returns:
        time_array (s), distance_array (m), pixel_coords [(row, col), ...], image_height
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Get image dimensions from JSON
    image_height = 600  # default
    image_width = 800
    if 'images' in data and data['images']:
        image_height = data['images'][0].get('height', image_height)
        image_width = data['images'][0].get('width', image_width)
    
    # Get segmentation polygon
    coords = None
    annotations = data.get('annotations', [])
    if annotations:
        seg = annotations[0].get('segmentation', [[]])[0]
        if seg:
            # Convert [x1,y1,x2,y2,...] to [[x1,y1], [x2,y2], ...]
            coords = np.array(seg).reshape(-1, 2)
    
    if coords is None or len(coords) == 0:
        return np.array([]), np.array([]), [], image_height, image_width
    
    cols = coords[:, 0]
    rows = coords[:, 1]
    
    # Convert to physical units
    time_array = (image_height - 1 - rows) * time_interval_sec
    
    if flip_horizontal:
        distance_array = (cols.max() - cols) * resolution_x_m
    else:
        distance_array = cols * resolution_x_m
    
    pixel_coords = [(int(r), int(c)) for r, c in zip(rows, cols)]
    
    return time_array, distance_array, pixel_coords, image_height, image_width


def detect_annotation_format(file_path):
    """
    Detect the format of an annotation file based on extension and content.
    
    Returns:
        str: 'mask', 'geojson', 'coco_json', or 'unknown'
    """
    ext = Path(file_path).suffix.lower()
    
    if ext == '.png':
        return 'mask'
    elif ext == '.geojson':
        return 'geojson'
    elif ext == '.json':
        # Check if it's COCO format
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            if 'annotations' in data and 'images' in data:
                return 'coco_json'
        except:
            pass
        return 'unknown'
    else:
        return 'unknown'


def create_mask_from_coordinates(pixel_coords, image_width, image_height):
    """
    Create a binary mask image from pixel coordinates for visualization.
    Draws the runup contour as a thick line, not a filled polygon.
    """
    # Create blank mask
    mask = Image.new('L', (image_width, image_height), 0)
    draw = ImageDraw.Draw(mask)
    
    # Draw as a line (not polygon) to avoid connecting first/last points
    if len(pixel_coords) > 1:
        # Convert (row, col) to (x, y) = (col, row) for PIL
        points = [(c, r) for r, c in pixel_coords]
        # Draw thick line for visibility
        draw.line(points, fill=255, width=3)
    
    return mask


class WaveRunUpCalculator(ctk.CTkToplevel):
    def __init__(self, master=None):
        super().__init__(master=master)
        self.title("Wave Run-Up Calculation")
        self.geometry("1200x800")
        try:
            self.iconbitmap(resource_path("launch_logo.ico"))
        except Exception:
            pass

        # Data holders
        self.raw_image = None
        self.raw_image_path = ""
        self.mask_image = None
        self.mask_image_path = ""
        self.annotation_format = None  # 'mask', 'geojson', 'coco_json'
        self.annotation_coords = None  # Store exact coordinates if available
        self.photo_raw = None
        self.runup_time = None
        self.runup_distance = None
        self.output_folder = ""
        self.batch_raw_folder = ""
        self.batch_mask_folder = ""
        self.batch_progress_bar = None

        # Top frame: three panels
        self.top_frame = ctk.CTkFrame(self)
        self.top_frame.pack(side="top", fill="both", expand=True)

        # Image panel - removed fixed size constraints to allow proper scaling
        self.image_panel = ctk.CTkFrame(self.top_frame)
        self.image_panel.pack(side="left", fill="both", expand=True, padx=5, pady=5)

        # Runup contour panel
        self.plot_panel = ctk.CTkFrame(self.top_frame)
        self.plot_panel.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=self.plot_panel)
        self.canvas_plot.get_tk_widget().pack(fill="both", expand=True)

        # Stats panel
        self.stats_panel = ctk.CTkFrame(self.top_frame)
        self.stats_panel.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        self.fig_stats, (self.ax_stats_psd, self.ax_stats_swash) = plt.subplots(2, 1, figsize=(5, 4))
        self.canvas_stats = FigureCanvasTkAgg(self.fig_stats, master=self.stats_panel)
        self.canvas_stats.get_tk_widget().pack(fill="both", expand=True)

        # Bottom frame
        self.bottom_panel = ctk.CTkFrame(self)
        self.bottom_panel.pack(side="bottom", fill="x", padx=5, pady=5)

        # Controls
        self.controls_panel = ctk.CTkFrame(self.bottom_panel)
        self.controls_panel.pack(side="top", fill="x", padx=5, pady=2)
        
        # Configure grid columns - make column 4 expand to push reset button to the right
        self.controls_panel.grid_columnconfigure(4, weight=1)
        
        self.btn_load_raw = ctk.CTkButton(self.controls_panel, text="Load Raw Time-Stack Image", command=self.load_raw_image)
        self.btn_load_raw.grid(row=0, column=0, padx=5, pady=5)
        
        # UPDATED: New button text and functionality
        self.btn_load_mask = ctk.CTkButton(
            self.controls_panel, 
            text="Load Runup (Mask/GeoJSON/COCO)", 
            command=self.load_annotation,
            width=220
        )
        self.btn_load_mask.grid(row=0, column=1, padx=5, pady=5)
        
        self.land_left = tk.BooleanVar()
        self.chk_land_left = ctk.CTkCheckBox(self.controls_panel, text="Land on left", variable=self.land_left)
        self.chk_land_left.grid(row=0, column=2, padx=5, pady=5)
        self.btn_calculate = ctk.CTkButton(self.controls_panel, text="Calculate Runup", command=self.calculate_runup)
        self.btn_calculate.grid(row=0, column=3, padx=5, pady=5)
        
        # Reset button - aligned to the right
        self.btn_reset = ctk.CTkButton(
            self.controls_panel, 
            text="Reset", 
            command=self.reset_all,
            width=80,
            fg_color="#8B0000",  # Dark red
            hover_color="#A52A2A"  # Lighter red on hover
        )
        self.btn_reset.grid(row=0, column=5, padx=5, pady=5, sticky="e")

        # Resolution
        self.resolution_panel = ctk.CTkFrame(self.bottom_panel)
        self.resolution_panel.pack(side="top", fill="x", padx=5, pady=2)
        self.pixel_res_label = ctk.CTkLabel(self.resolution_panel, text="Identified Pixel Resolution: N/A")
        self.pixel_res_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.manual_res_var = tk.BooleanVar()
        self.chk_manual_res = ctk.CTkCheckBox(self.resolution_panel, text="Manual Resolution", variable=self.manual_res_var)
        self.chk_manual_res.grid(row=0, column=1, padx=5, pady=5)
        self.manual_res_entry = ctk.CTkEntry(self.resolution_panel, width=80)
        self.manual_res_entry.grid(row=0, column=2, padx=5, pady=5)
        self.manual_res_label = ctk.CTkLabel(self.resolution_panel, text="m")
        self.manual_res_label.grid(row=0, column=3, padx=5, pady=5)
        
        # Annotation format indicator
        self.format_label = ctk.CTkLabel(self.resolution_panel, text="Annotation: None", text_color="gray")
        self.format_label.grid(row=0, column=4, padx=20, pady=5, sticky="w")

        # Export
        self.export_panel = ctk.CTkFrame(self.bottom_panel)
        self.export_panel.pack(side="top", fill="x", padx=5, pady=2)
        self.btn_select_out_folder = ctk.CTkButton(self.export_panel, text="Output Folder", command=self.select_output_folder)
        self.btn_select_out_folder.grid(row=0, column=0, padx=5, pady=5)
        self.out_folder_label = ctk.CTkLabel(self.export_panel, text="No folder selected")
        self.out_folder_label.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        self.btn_export_runup = ctk.CTkButton(self.export_panel, text="Export Runup", command=self.export_runup)
        self.btn_export_runup.grid(row=0, column=2, padx=5, pady=5)

        # Batch
        self.batch_panel = ctk.CTkFrame(self.bottom_panel)
        self.batch_panel.pack(side="top", fill="x", padx=5, pady=2)
        self.btn_select_batch_raw = ctk.CTkButton(self.batch_panel, text="Batch Raw Folder", command=self.select_batch_raw_folder)
        self.btn_select_batch_raw.grid(row=0, column=0, padx=5, pady=5)
        self.batch_raw_label = ctk.CTkLabel(self.batch_panel, text="No folder selected")
        self.batch_raw_label.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        self.btn_select_batch_mask = ctk.CTkButton(self.batch_panel, text="Batch Annotation Folder", command=self.select_batch_mask_folder)
        self.btn_select_batch_mask.grid(row=0, column=2, padx=5, pady=5)
        self.batch_mask_label = ctk.CTkLabel(self.batch_panel, text="No folder selected")
        self.batch_mask_label.grid(row=0, column=3, padx=5, pady=5, sticky="w")
        self.btn_batch_process = ctk.CTkButton(self.batch_panel, text="Batch Process", command=self.run_batch_process)
        self.btn_batch_process.grid(row=1, column=0, padx=5, pady=5)
        self.batch_progress_bar = ctk.CTkProgressBar(self.batch_panel, width=200)
        self.batch_progress_bar.grid(row=1, column=1, padx=5, pady=5)
        self.batch_progress_bar.set(0)
        self.batch_progress_label = ctk.CTkLabel(self.batch_panel, text="0 / 0 pairs processed")
        self.batch_progress_label.grid(row=1, column=2, padx=5, pady=5, sticky="w")

        # Console
        self.console_frame = ctk.CTkFrame(self)
        self.console_frame.pack(side="bottom", fill="both", expand=False, padx=10, pady=10)
        self.console_text = tk.Text(self.console_frame, wrap="word", height=10)
        self.console_text.pack(fill="both", expand=True, padx=5, pady=5)
        self.stdout_redirector = StdoutRedirector(self.console_text)
        sys.stdout = self.stdout_redirector
        sys.stderr = self.stdout_redirector
        print("Here you may see console outputs\n--------------------------------\n")

    def reset_all(self):
        """Reset all data and UI elements to initial state."""
        # Clear data holders
        self.raw_image = None
        self.raw_image_path = ""
        self.mask_image = None
        self.mask_image_path = ""
        self.annotation_format = None
        self.annotation_coords = None
        self.photo_raw = None
        self.runup_time = None
        self.runup_distance = None
        
        # Clear image panel
        for child in self.image_panel.winfo_children():
            child.destroy()
        
        # Clear plots
        self.ax.clear()
        self.ax.set_xlabel("Cross-shore distance (m)")
        self.ax.set_ylabel("Time (s)")
        self.ax.set_title("Runup Contour")
        self.ax.grid(True)
        self.canvas_plot.draw()
        
        self.ax_stats_psd.clear()
        self.ax_stats_psd.set_title('Power Spectral Density')
        self.ax_stats_swash.clear()
        self.ax_stats_swash.set_title('Detrended Swash Excursion')
        self.fig_stats.tight_layout()
        self.canvas_stats.draw()
        
        # Reset labels
        self.pixel_res_label.configure(text="Identified Pixel Resolution: N/A")
        self.format_label.configure(text="Annotation: None", text_color="gray")
        
        # Reset checkboxes and entries
        self.land_left.set(False)
        self.manual_res_var.set(False)
        self.manual_res_entry.delete(0, tk.END)
        
        # Clear console
        self.console_text.delete(1.0, tk.END)
        print("Session reset. Ready for new data.\n--------------------------------\n")

    def load_raw_image(self):
        file_path = filedialog.askopenfilename(title="Select Raw Time-Stack Image", filetypes=[("PNG Images", "*.png")])
        if file_path:
            self.raw_image_path = file_path
            self.raw_image = Image.open(file_path)
            self.photo_raw = ImageTk.PhotoImage(self.raw_image)
            for child in self.image_panel.winfo_children():
                child.destroy()
            self.raw_image_label = tk.Label(self.image_panel, image=self.photo_raw)
            self.raw_image_label.pack(fill="both", expand=True)

    def load_annotation(self):
        """
        Load runup annotation from mask PNG, GeoJSON, or COCO JSON file.
        Automatically detects format based on file extension.
        """
        file_path = filedialog.askopenfilename(
            title="Select Runup Annotation", 
            filetypes=[
                ("All Supported", "*.png *.geojson *.json"),
                ("PNG Mask", "*.png"),
                ("GeoJSON", "*.geojson"),
                ("COCO JSON", "*.json"),
            ]
        )
        if not file_path:
            return
            
        self.mask_image_path = file_path
        self.annotation_format = detect_annotation_format(file_path)
        
        print(f"Loading annotation: {os.path.basename(file_path)}")
        print(f"Detected format: {self.annotation_format}")
        
        # Update format indicator
        format_colors = {
            'mask': 'orange',
            'geojson': 'green', 
            'coco_json': 'green',
            'unknown': 'red'
        }
        format_names = {
            'mask': 'PNG Mask (1 pt/row)',
            'geojson': 'GeoJSON (exact)',
            'coco_json': 'COCO JSON (exact)',
            'unknown': 'Unknown'
        }
        self.format_label.configure(
            text=f"Annotation: {format_names.get(self.annotation_format, 'Unknown')}",
            text_color=format_colors.get(self.annotation_format, 'red')
        )
        
        # Handle different formats
        if self.annotation_format == 'mask':
            self._load_mask_annotation(file_path)
        elif self.annotation_format == 'geojson':
            self._load_geojson_annotation(file_path)
        elif self.annotation_format == 'coco_json':
            self._load_coco_annotation(file_path)
        else:
            messagebox.showerror("Error", f"Unknown annotation format: {file_path}")
            return
    
    def _load_mask_annotation(self, file_path):
        """Load PNG mask annotation."""
        self.mask_image = Image.open(file_path).convert("L")
        self.annotation_coords = None  # No exact coords for mask
        self._display_annotation_overlay()
        
    def _load_geojson_annotation(self, file_path):
        """Load GeoJSON annotation with exact coordinates."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            coords = None
            for feature in data.get('features', []):
                geom_type = feature['geometry']['type']
                if geom_type == 'LineString':
                    coords = feature['geometry']['coordinates']
                    break
                elif geom_type == 'Polygon':
                    coords = feature['geometry']['coordinates'][0]
                    break
            
            if coords is None:
                messagebox.showerror("Error", "No LineString or Polygon found in GeoJSON")
                return
            
            self.annotation_coords = np.array(coords)  # [x, y] = [col, row]
            print(f"Loaded {len(self.annotation_coords)} exact coordinate points")
            
            # Create mask for visualization if raw image is loaded
            if self.raw_image:
                w, h = self.raw_image.size
                pixel_coords = [(int(r), int(c)) for c, r in self.annotation_coords]
                self.mask_image = create_mask_from_coordinates(pixel_coords, w, h)
                self._display_annotation_overlay()
            else:
                print("Load raw image to see annotation overlay")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load GeoJSON: {e}")
    
    def _load_coco_annotation(self, file_path):
        """Load COCO JSON annotation with exact coordinates."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Get image dimensions
            image_width = 800
            image_height = 600
            if 'images' in data and data['images']:
                image_width = data['images'][0].get('width', image_width)
                image_height = data['images'][0].get('height', image_height)
            
            # Get segmentation
            annotations = data.get('annotations', [])
            if not annotations:
                messagebox.showerror("Error", "No annotations found in COCO JSON")
                return
            
            seg = annotations[0].get('segmentation', [[]])[0]
            if not seg:
                messagebox.showerror("Error", "No segmentation polygon found")
                return
            
            # Convert [x1,y1,x2,y2,...] to [[x1,y1], [x2,y2], ...]
            self.annotation_coords = np.array(seg).reshape(-1, 2)  # [x, y] = [col, row]
            print(f"Loaded {len(self.annotation_coords)} exact coordinate points")
            print(f"Image dimensions from JSON: {image_width} x {image_height}")
            
            # Create mask for visualization
            pixel_coords = [(int(r), int(c)) for c, r in self.annotation_coords]
            self.mask_image = create_mask_from_coordinates(pixel_coords, image_width, image_height)
            
            # If raw image not loaded, store dimensions for later
            if not self.raw_image:
                print("Load raw image to see annotation overlay")
            else:
                self._display_annotation_overlay()
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load COCO JSON: {e}")
    
    def _display_annotation_overlay(self):
        """Display raw image with annotation overlay."""
        if not self.raw_image or self.mask_image is None:
            return
            
        raw_copy = self.raw_image.copy().convert("RGBA")
        mask_resized = self.mask_image.resize(raw_copy.size)

        # Create red overlay for mask
        overlay = Image.new("RGBA", raw_copy.size, (255, 0, 0, 0))
        m = np.array(mask_resized) > 128
        ov = np.zeros((raw_copy.size[1], raw_copy.size[0], 4), dtype=np.uint8)
        ov[m] = np.array([255, 0, 0, 100], dtype=np.uint8)
        overlay = Image.fromarray(ov, mode="RGBA")
        combined = Image.alpha_composite(raw_copy, overlay)

        # Get resolution
        if self.manual_res_var.get():
            try:
                resolution_x_m = float(self.manual_res_entry.get())
            except:
                resolution_x_m = 0.25
                print("Warning: Invalid manual resolution; defaulting to 0.25 m/pixel.")
        else:
            try:
                resolution_x_m = float(self.raw_image.info.get("pixel_resolution", 0.25))
            except:
                resolution_x_m = 0.25
                print("Warning: Invalid metadata resolution; defaulting to 0.25 m/pixel.")
        self.pixel_res_label.configure(text=f"Identified Pixel Resolution: {resolution_x_m} m")

        try:
            time_interval_sec = float(self.raw_image.info.get("time_interval", 1))
        except:
            time_interval_sec = 1

        for child in self.image_panel.winfo_children(): 
            child.destroy()
        self.display_image_with_axes(combined, resolution_x_m, time_interval_sec)

    def display_image_with_axes(self, combined_image, resolution_x_m, time_interval_sec):
        width, height = combined_image.size
        
        # Use larger figure size that matches other panels
        fig_w, fig_h = 5, 4  # Same as the other panels
        
        # Flip image vertically so that:
        # - Row 0 (top of original image, t=max) goes to bottom of display
        # - Row H-1 (bottom of original image, t=0) goes to top of display
        # This aligns with origin='lower' where y=0 is at bottom
        flipped_image = combined_image.transpose(Image.FLIP_TOP_BOTTOM)
        
        # Bottom-origin axes: x = distance (m), y = time (s)
        # Time runs from 0 (bottom) to max (top)
        extent = [0, width * resolution_x_m, 0, height * time_interval_sec]
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        fig.subplots_adjust(left=0.15, right=0.95, top=0.92, bottom=0.12)
        ax.imshow(np.array(flipped_image), extent=extent, aspect='auto', origin='lower')
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Time (s)")
        ax.set_title("Raw Image with Annotation Overlay")
        ax.grid(True, alpha=0.3)

        self.canvas_img_fig = FigureCanvasTkAgg(fig, master=self.image_panel)
        self.canvas_img_fig.get_tk_widget().pack(fill="both", expand=True)
        self.canvas_img_fig.draw()

    def calculate_runup(self):
        if not self.raw_image_path or not self.mask_image_path:
            messagebox.showerror("Error", "Load both raw image and annotation first.")
            return

        # Get resolution and timing
        if self.manual_res_var.get():
            try:
                resolution_x_m = float(self.manual_res_entry.get())
            except:
                resolution_x_m = float(self.raw_image.info.get("pixel_resolution", 0.25))
        else:
            resolution_x_m = float(self.raw_image.info.get("pixel_resolution", 0.25))

        try:
            time_interval_sec = float(self.raw_image.info.get("time_interval", 1))
        except:
            time_interval_sec = 1

        flip_horizontal = self.land_left.get()
        
        # Get image height
        image_height = self.raw_image.height

        # Extract runup based on annotation format
        if self.annotation_format == 'geojson' and self.annotation_coords is not None:
            print("Extracting runup from GeoJSON (exact coordinates)...")
            t_arr, d_arr, pixel_coords = extract_runup_from_geojson(
                self.mask_image_path, resolution_x_m, time_interval_sec,
                image_height, flip_horizontal
            )
        elif self.annotation_format == 'coco_json' and self.annotation_coords is not None:
            print("Extracting runup from COCO JSON (exact coordinates)...")
            t_arr, d_arr, pixel_coords, _, _ = extract_runup_from_coco_json(
                self.mask_image_path, resolution_x_m, time_interval_sec,
                flip_horizontal
            )
        else:
            print("Extracting runup from mask (1 point per row)...")
            t_arr, d_arr, pixel_coords = extract_runup_from_mask(
                self.mask_image_path, resolution_x_m, time_interval_sec, flip_horizontal
            )
        
        if t_arr.size == 0:
            messagebox.showerror("Error", "Annotation contains no runup data.")
            return

        print(f"Extracted {len(t_arr)} runup points")

        # Sort by time ascending
        sort_idx = np.argsort(t_arr)
        t_sorted = t_arr[sort_idx]
        d_sorted = d_arr[sort_idx]

        # Store for export
        self.runup_time, self.runup_distance = t_sorted, d_sorted

        # Plot the extracted runup contour
        self.ax.clear()
        self.ax.plot(d_sorted, t_sorted, 'bo-', markersize=1, linewidth=0.5, label="Runup Contour")
        self.ax.set_xlabel("Cross-shore distance (m)")
        self.ax.set_ylabel("Time (s)")
        self.ax.set_title(f"Extracted Runup ({len(t_sorted)} points)")
        self.ax.grid(True)
        self.ax.legend()
        self.canvas_plot.draw()

        # --- Stats (single image) ---
        detr = d_sorted - np.mean(d_sorted)
        dt = np.diff(t_sorted)
        fs = 1.0 / np.mean(dt) if len(dt) > 0 else 1.0

        # PSD (positive freqs)
        fxx, pxx = welch(detr, fs=fs, nperseg=min(256, len(detr)))
        pos = fxx > 0
        fxx, pxx = fxx[pos], pxx[pos]
        ig_mask = fxx < 0.05
        E_ig = np.trapz(pxx[ig_mask], fxx[ig_mask])
        E_tot = np.trapz(pxx, fxx)
        ig_pct = 100 * E_ig / E_tot if E_tot > 0 else 0

        # PSD plot
        self.ax_stats_psd.clear()
        self.ax_stats_psd.plot(fxx, pxx, label='PSD')
        self.ax_stats_psd.fill_between(fxx, pxx, where=ig_mask, alpha=0.3,
                                       label=f'IG (<0.05Hz) {ig_pct:.1f}%')
        self.ax_stats_psd.set_xscale('log')
        self.ax_stats_psd.set_title('Power Spectral Density')
        self.ax_stats_psd.set_ylabel('PSD')
        self.ax_stats_psd.legend()

        # Detrended swash
        self.ax_stats_swash.clear()
        self.ax_stats_swash.plot(t_sorted, detr, linewidth=0.5, label="Detrended Swash")
        self.ax_stats_swash.set_title('Detrended Swash Excursion')
        self.ax_stats_swash.set_xlabel('Time (s)')
        self.ax_stats_swash.set_ylabel("d'(t) (m)")
        self.ax_stats_swash.legend()

        self.fig_stats.tight_layout()
        self.canvas_stats.draw()

    def select_output_folder(self):
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_folder = folder
            self.out_folder_label.configure(text=folder)

    def export_runup(self):
        if self.runup_time is None or self.runup_distance is None:
            messagebox.showerror("Error", "No runup data to export.")
            return
        if not self.output_folder:
            messagebox.showerror("Error", "Select an output folder.")
            return

        base_name = os.path.basename(self.raw_image_path)
        match = re.search(r"(\d{4})[-_](\d{2})[-_](\d{2})[-_](\d{2})[-_](\d{2})", base_name)
        if not match:
            messagebox.showerror("Error", "Could not extract date.")
            return
        year, month, day, hour, minute = map(int, match.groups())
        base_dt = datetime.datetime(year, month, day, hour, minute)

        out_name = os.path.splitext(base_name.replace("raw", "runup"))[0] + ".csv"
        out_path = os.path.join(self.output_folder, out_name)
        
        with open(out_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["time", "distance"])
            for sec, dist in zip(self.runup_time, self.runup_distance):
                ts = (base_dt + datetime.timedelta(seconds=float(sec))).strftime("%Y-%m-%d-%H-%M-%S")
                writer.writerow([ts, dist])
        messagebox.showinfo("Export Runup", f"Exported to:\n{out_path}")

    def select_batch_raw_folder(self):
        folder = filedialog.askdirectory(title="Select Folder with Raw TS Images")
        if folder:
            self.batch_raw_folder = folder
            self.batch_raw_label.configure(text=folder)

    def select_batch_mask_folder(self):
        folder = filedialog.askdirectory(title="Select Folder with Annotations (Masks/GeoJSON/JSON)")
        if folder:
            self.batch_mask_folder = folder
            self.batch_mask_label.configure(text=folder)

    def run_batch_process(self):
        # 1) Validate folders
        if not self.batch_raw_folder or not self.batch_mask_folder:
            messagebox.showerror("Error", "Please select both batch raw and batch annotation folders.")
            return
        print("Batch process has started")

        # 2) List all supported annotation files
        raw_files = [f for f in os.listdir(self.batch_raw_folder) if f.lower().endswith('.png')]
        annotation_files = [f for f in os.listdir(self.batch_mask_folder) 
                           if f.lower().endswith(('.png', '.geojson', '.json'))]

        # 3) Group by timestamp key
        date_pattern = r"(\d{4}[-_]\d{2}[-_]\d{2}[-_]\d{2}[-_]\d{2})"
        groups = defaultdict(lambda: {"raw": [], "annotation": []})
        for f in raw_files:
            m = re.search(date_pattern, f)
            if m:
                groups[m.group(1)]["raw"].append(f)
        for f in annotation_files:
            m = re.search(date_pattern, f)
            if m:
                groups[m.group(1)]["annotation"].append(f)

        # 4) Build valid pairs, preferring exact formats (geojson > json > png)
        valid_pairs = []
        for key, files in groups.items():
            if not files["raw"] or not files["annotation"]:
                continue
            
            # For each raw file, find best annotation
            for raw_f in files["raw"]:
                # Sort annotations by preference: geojson > json > png
                def annotation_priority(f):
                    if f.endswith('.geojson'):
                        return 0
                    elif f.endswith('.json'):
                        return 1
                    else:
                        return 2
                
                sorted_annotations = sorted(files["annotation"], key=annotation_priority)
                if sorted_annotations:
                    valid_pairs.append((raw_f, sorted_annotations[0]))
        
        total_pairs = len(valid_pairs)
        if total_pairs == 0:
            messagebox.showerror("Error", "No valid pairs found for batch processing.")
            return

        print(f"Found {total_pairs} raw/annotation pairs")

        # Default output if none
        if not self.output_folder:
            self.output_folder = self.batch_raw_folder

        # Reset progress UI
        self.batch_progress_bar.set(0)
        self.batch_progress_label.configure(text=f"0 / {total_pairs} pairs processed")
        self.update()

        all_runup_data = []
        processed = 0

        # 5) Loop through each pair
        for raw_name, annotation_name in valid_pairs:
            raw_path = os.path.join(self.batch_raw_folder, raw_name)
            annotation_path = os.path.join(self.batch_mask_folder, annotation_name)

            try:
                raw_img = Image.open(raw_path)
            except:
                continue

            # Resolution & timing
            if self.manual_res_var.get():
                try:
                    resolution_x_m = float(self.manual_res_entry.get())
                except:
                    resolution_x_m = float(raw_img.info.get("pixel_resolution", 0.25))
            else:
                resolution_x_m = float(raw_img.info.get("pixel_resolution", 0.25))

            try:
                time_interval_sec = float(raw_img.info.get("time_interval", 1))
            except:
                time_interval_sec = 1

            flip_horizontal = self.land_left.get()
            image_height = raw_img.height

            # Detect format and extract
            ann_format = detect_annotation_format(annotation_path)
            
            if ann_format == 'geojson':
                t_arr, d_arr, _ = extract_runup_from_geojson(
                    annotation_path, resolution_x_m, time_interval_sec,
                    image_height, flip_horizontal
                )
            elif ann_format == 'coco_json':
                t_arr, d_arr, _, _, _ = extract_runup_from_coco_json(
                    annotation_path, resolution_x_m, time_interval_sec,
                    flip_horizontal
                )
            else:
                t_arr, d_arr, _ = extract_runup_from_mask(
                    annotation_path, resolution_x_m, time_interval_sec, flip_horizontal
                )
            
            if t_arr.size == 0:
                continue

            # Sort by time asc
            sort_idx = np.argsort(t_arr)
            t_sorted = t_arr[sort_idx]
            d_sorted = d_arr[sort_idx]

            # Export CSV
            m = re.search(date_pattern, raw_name)
            if m:
                y, mo, da, hr, mi = map(int, re.split('[-_]', m.group(1)))
                base_dt = datetime.datetime(y, mo, da, hr, mi)
            else:
                base_dt = datetime.datetime.now()

            out_name = os.path.splitext(raw_name.replace("raw", "runup"))[0] + ".csv"
            out_path = os.path.join(self.output_folder, out_name)
            with open(out_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["time", "distance", "format"])
                for sec, dist in zip(t_sorted, d_sorted):
                    ts = (base_dt + datetime.timedelta(seconds=float(sec))).strftime("%Y-%m-%d-%H-%M-%S")
                    writer.writerow([ts, dist, ann_format])

            all_runup_data.append((d_sorted, t_sorted))
            processed += 1
            print(f"Processed: {raw_name} + {annotation_name} ({ann_format})")
            self.batch_progress_bar.set(processed / total_pairs)
            self.batch_progress_label.configure(text=f"{processed} / {total_pairs} pairs processed")
            self.update()

        # 6) Plot aggregated runup contours
        self.ax.clear()
        for d_arr, t_arr in all_runup_data:
            self.ax.plot(d_arr, t_arr, marker='o', markersize=1, linestyle='-')
        self.ax.set_xlabel("Distance (m)")
        self.ax.set_ylabel("Time (s)")
        self.ax.set_title("Aggregated Runup Contours")
        self.ax.grid(True)
        self.canvas_plot.draw()

        # 7) Compute IG% vs Incident% and swash envelope
        ig_list, inc_list = [], []
        self.ax_stats_psd.clear()
        self.ax_stats_swash.clear()

        for d_arr, t_arr in all_runup_data:
            detr = d_arr - np.mean(d_arr)
            dt = np.diff(t_arr)
            fs = 1.0 / np.mean(dt) if len(dt) > 0 else 1.0
            fxx, pxx = welch(detr, fs=fs, nperseg=min(256, len(detr)))
            pos = fxx > 0
            fxx, pxx = fxx[pos], pxx[pos]
            ig_mask = fxx < 0.05
            E_ig = np.trapz(pxx[ig_mask], fxx[ig_mask])
            E_tot = np.trapz(pxx, fxx)
            ig_pct = 100 * E_ig / E_tot if E_tot > 0 else 0
            ig_list.append(ig_pct)
            inc_list.append(100 - ig_pct)

            # Envelope plot
            self.ax_stats_swash.plot(t_arr, detr, linewidth=0.5, alpha=0.6)

        # Energy partitioning bar plot
        idx = np.arange(len(ig_list))
        self.ax_stats_psd.bar(idx, ig_list, label='IG%')
        self.ax_stats_psd.bar(idx, inc_list, bottom=ig_list, label='Incident%')
        self.ax_stats_psd.set_xticks(idx)
        self.ax_stats_psd.set_xticklabels([str(i + 1) for i in idx])
        self.ax_stats_psd.set_ylabel('%')
        self.ax_stats_psd.set_title('Energy Partitioning')
        self.ax_stats_psd.legend()

        self.ax_stats_swash.set_xlabel('Time (s)')
        self.ax_stats_swash.set_ylabel("d'(t) (m)")
        self.ax_stats_swash.set_title('Swash Excursions (Batch)')

        self.fig_stats.tight_layout()
        self.canvas_stats.draw()

        messagebox.showinfo("Batch Process", "Batch processing completed.")


def main():
    root = ctk.CTk()
    root.withdraw()
    win = WaveRunUpCalculator(master=root)
    win.mainloop()

if __name__ == "__main__":
    main()