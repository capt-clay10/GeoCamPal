import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import customtkinter as ctk
from collections import defaultdict
import os
import datetime
import csv
import re
import sys
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
        self.text_widget.insert(tk.END, message)
        self.text_widget.see(tk.END)

    def flush(self):
        pass


def extract_runup_from_mask(mask_path, resolution_x_m, time_interval_sec, flip_horizontal=False):
    """
    Extract runup contour coordinates from a binary mask image.
    
    Parameters:
      mask_path (str): File path to the mask image.
      resolution_x_m (float): Spatial resolution in meters per pixel.
      time_interval_sec (float): Time per row in seconds.
      flip_horizontal (bool): If True, selects the leftmost pixel per row; else the rightmost.
    
    Returns:
      np.array: Array of time values (s) as measured from the top of the image.
      np.array: Array of cross-shore distances (m).
      list: List of (row, col) tuples for the detected runup pixels.
    """
    mask_img = Image.open(mask_path).convert("L")
    mask_array = np.array(mask_img)
    threshold = 128
    binary_mask = mask_array > threshold

    rows, cols = np.where(binary_mask)
    unique_rows = np.unique(rows)

    time_array = []
    distance_array = []
    pixel_coords = []

    for r in unique_rows:
        cols_in_row = cols[rows == r]
        if len(cols_in_row) == 0:
            continue
        if flip_horizontal:
            runup_col = np.min(cols_in_row)
        else:
            runup_col = np.max(cols_in_row)
        # Each row corresponds to time_interval_sec seconds.
        time_val = r * time_interval_sec
        distance_val = runup_col * resolution_x_m  # conversion: pixel index × resolution
        time_array.append(time_val)
        distance_array.append(distance_val)
        pixel_coords.append((r, runup_col))
    
    return np.array(time_array), np.array(distance_array), pixel_coords

class WaveRunUpCalculator(ctk.CTkToplevel):
    def __init__(self, master=None):
        super().__init__(master=master)
        self.title("Wave Run-Up Calculation")
        self.geometry("1200x800")
        
        try:
            self.iconbitmap(resource_path("launch_logo.ico"))
        except Exception as e:
            print(f"Icon load failed: {e}")

        
        # Variables to store file paths and processed data.
        self.raw_image = None
        self.raw_image_path = ""
        self.mask_image = None
        self.mask_image_path = ""
        self.photo_raw = None
        self.canvas_img_fig = None   # Embedded Matplotlib figure for single processing
        
        # Store runup data for export.
        self.runup_time = None
        self.runup_distance = None
        self.output_folder = ""
        
        # For batch processing.
        self.batch_raw_folder = ""
        self.batch_mask_folder = ""
        self.batch_progress_bar = None
        
        # --- Top Frame containing two panels ---
        self.top_frame = ctk.CTkFrame(self)
        self.top_frame.pack(side="top", fill="both", expand=True)
        
        # Left Panel: Image display panel.
        self.image_panel = ctk.CTkFrame(self.top_frame, width=400, height=400)
        self.image_panel.pack_propagate(False)
        self.image_panel.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        
        # Right Panel: Plot panel for runup contour.
        self.plot_panel = ctk.CTkFrame(self.top_frame)
        self.plot_panel.pack(side="right", fill="both", expand=True, padx=5, pady=5)
        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=self.plot_panel)
        self.canvas_plot.get_tk_widget().pack(fill="both", expand=True)
        
        # --- Bottom Panel: Contains separate sub-panels ---
        self.bottom_panel = ctk.CTkFrame(self)
        self.bottom_panel.pack(side="bottom", fill="x", padx=5, pady=5)
        
        # Controls Panel: Load and process single images.
        self.controls_panel = ctk.CTkFrame(self.bottom_panel)
        self.controls_panel.pack(side="top", fill="x", padx=5, pady=2)
        
        self.btn_load_raw = ctk.CTkButton(
            self.controls_panel, text="Load Raw Time-Stack Image", command=self.load_raw_image
        )
        self.btn_load_raw.grid(row=0, column=0, padx=5, pady=5)
        
        self.btn_load_mask = ctk.CTkButton(
            self.controls_panel, text="Load Mask of Run-Up", command=self.load_mask
        )
        self.btn_load_mask.grid(row=0, column=1, padx=5, pady=5)
        
        self.land_left = tk.BooleanVar()
        self.chk_land_left = ctk.CTkCheckBox(
            self.controls_panel, text="Land on left", variable=self.land_left
        )
        self.chk_land_left.grid(row=0, column=2, padx=5, pady=5)
        
        self.btn_calculate = ctk.CTkButton(
            self.controls_panel, text="Calculate Runup", command=self.calculate_runup
        )
        self.btn_calculate.grid(row=0, column=3, padx=5, pady=5)
        
        # Resolution Panel: Display identified pixel resolution and manual resolution controls.
        self.resolution_panel = ctk.CTkFrame(self.bottom_panel)
        self.resolution_panel.pack(side="top", fill="x", padx=5, pady=2)
        
        self.pixel_res_label = ctk.CTkLabel(self.resolution_panel, text="Identified Pixel Resolution: N/A")
        self.pixel_res_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        self.manual_res_var = tk.BooleanVar()
        self.chk_manual_res = ctk.CTkCheckBox(
            self.resolution_panel, text="Manual Resolution", variable=self.manual_res_var
        )
        self.chk_manual_res.grid(row=0, column=1, padx=5, pady=5)
        
        self.manual_res_entry = ctk.CTkEntry(self.resolution_panel, width=80)
        self.manual_res_entry.grid(row=0, column=2, padx=5, pady=5)
        
        self.manual_res_label = ctk.CTkLabel(self.resolution_panel, text="m")
        self.manual_res_label.grid(row=0, column=3, padx=5, pady=5)
        
        # Export Panel: Select output folder and export single runup.
        self.export_panel = ctk.CTkFrame(self.bottom_panel)
        self.export_panel.pack(side="top", fill="x", padx=5, pady=2)
        
        self.btn_select_out_folder = ctk.CTkButton(
            self.export_panel, text="Output Folder", command=self.select_output_folder
        )
        self.btn_select_out_folder.grid(row=0, column=0, padx=5, pady=5)
        
        self.out_folder_label = ctk.CTkLabel(self.export_panel, text="No folder selected")
        self.out_folder_label.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        self.btn_export_runup = ctk.CTkButton(
            self.export_panel, text="Export Runup", command=self.export_runup
        )
        self.btn_export_runup.grid(row=0, column=2, padx=5, pady=5)
        
        # Batch Panel: For batch processing of multiple image pairs.
        self.batch_panel = ctk.CTkFrame(self.bottom_panel)
        self.batch_panel.pack(side="top", fill="x", padx=5, pady=2)
        
        # Row 0 of batch panel: Folder selection for raw images and masks.
        self.btn_select_batch_raw = ctk.CTkButton(
            self.batch_panel, text="Batch Raw Folder", command=self.select_batch_raw_folder
        )
        self.btn_select_batch_raw.grid(row=0, column=0, padx=5, pady=5)
        self.batch_raw_label = ctk.CTkLabel(self.batch_panel, text="No folder selected")
        self.batch_raw_label.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        self.btn_select_batch_mask = ctk.CTkButton(
            self.batch_panel, text="Batch Mask Folder", command=self.select_batch_mask_folder
        )
        self.btn_select_batch_mask.grid(row=0, column=2, padx=5, pady=5)
        self.batch_mask_label = ctk.CTkLabel(self.batch_panel, text="No folder selected")
        self.batch_mask_label.grid(row=0, column=3, padx=5, pady=5, sticky="w")
        
        # Row 1 of batch panel: Batch Process button and progress indicators.
        self.btn_batch_process = ctk.CTkButton(
            self.batch_panel, text="Batch Process", command=self.run_batch_process
        )
        self.btn_batch_process.grid(row=1, column=0, padx=5, pady=5)
        
        # Create a progress bar and progress text label.
        self.batch_progress_bar = ctk.CTkProgressBar(self.batch_panel, width=200)
        self.batch_progress_bar.grid(row=1, column=1, padx=5, pady=5)
        self.batch_progress_bar.set(0)
        self.batch_progress_label = ctk.CTkLabel(self.batch_panel, text="0 / 0 pairs processed")
        self.batch_progress_label.grid(row=1, column=2, padx=5, pady=5, sticky="w")
        
        # --- Console Panel at the bottom ---
        self.console_frame = ctk.CTkFrame(self)
        self.console_frame.pack(side="bottom", fill="both", expand=False, padx=10, pady=10)
        self.console_text = tk.Text(self.console_frame, wrap="word", height=10)
        self.console_text.pack(fill="both", expand=True, padx=5, pady=5)
        self.stdout_redirector = StdoutRedirector(self.console_text)
        sys.stdout = self.stdout_redirector
        sys.stderr = self.stdout_redirector
        print("Here you may see console outputs\n")
        
    def load_raw_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Raw Time-Stack Image", filetypes=[("PNG Images", "*.png")]
        )
        if file_path:
            self.raw_image_path = file_path
            self.raw_image = Image.open(file_path)
            self.photo_raw = ImageTk.PhotoImage(self.raw_image)
            # Clear previous widgets from the image panel.
            for child in self.image_panel.winfo_children():
                child.destroy()
            # Display the raw image.
            self.raw_image_label = tk.Label(self.image_panel, image=self.photo_raw)
            self.raw_image_label.pack(fill="both", expand=True)
    
    def load_mask(self):
        file_path = filedialog.askopenfilename(
            title="Select Run-Up Mask Image", filetypes=[("PNG Images", "*.png")]
        )
        if file_path:
            self.mask_image_path = file_path
            self.mask_image = Image.open(file_path).convert("L")
            if self.raw_image:
                # Create an RGBA copy of the raw image.
                raw_copy = self.raw_image.copy().convert("RGBA")
                # Resize mask to match raw image dimensions.
                mask_resized = self.mask_image.resize(raw_copy.size)
                # Create a transparent red overlay.
                overlay = Image.new("RGBA", raw_copy.size, (255, 0, 0, 0))
                overlay_data = overlay.load()
                mask_data = mask_resized.load()
                for i in range(raw_copy.width):
                    for j in range(raw_copy.height):
                        if mask_data[i, j] > 128:
                            overlay_data[i, j] = (255, 0, 0, 100)
                # Composite the overlay onto the raw image.
                combined = Image.alpha_composite(raw_copy, overlay)
                
                # Determine spatial resolution.
                if self.manual_res_var.get():
                    try:
                        resolution_x_m = float(self.manual_res_entry.get())
                    except Exception:
                        resolution_x_m = 0.25
                        print("Warning: Invalid manual resolution provided. Using default of 0.25 m/pixel.")
                else:
                    resolution_from_metadata = self.raw_image.info.get("pixel_resolution")
                    if resolution_from_metadata:
                        try:
                            resolution_x_m = float(resolution_from_metadata)
                        except ValueError:
                            resolution_x_m = 0.25
                            print("Warning: Non-numeric resolution in image metadata. Using default of 0.25 m/pixel.")
                    else:
                        resolution_x_m = 0.25
                        print("Warning: No resolution found in metadata and no manual resolution given; setting to default 0.25 m/pixel.")

                
                # Update the identified pixel resolution label.
                self.pixel_res_label.configure(text=f"Identified Pixel Resolution: {resolution_x_m} m")
                
                # Get time interval from metadata or use default.
                try:
                    time_interval_sec = float(self.raw_image.info.get("time_interval", 1))
                except Exception:
                    time_interval_sec = 1
                
                # Clear previous content from the image panel.
                for child in self.image_panel.winfo_children():
                    child.destroy()
                # Display the combined image with real-world axes.
                self.display_image_with_axes(combined, resolution_x_m, time_interval_sec)
    
    def display_image_with_axes(self, combined_image, resolution_x_m, time_interval_sec):
        """
        Displays the combined image (raw + mask overlay) using real-world units.
        The X-axis shows distance (m) and the Y-axis shows time (s) with time=0 at the bottom.
        """
        width, height = combined_image.size
        # Calculate figure size based on aspect ratio.
        max_size = 200  # max display size in pixels
        dpi = 150
        aspect_ratio = width / height
        if aspect_ratio > 1:
            fig_width = max_size / dpi
            fig_height = (max_size / aspect_ratio) / dpi
        else:
            fig_height = max_size / dpi
            fig_width = (max_size * aspect_ratio) / dpi
        
        # Define extents.
        extent = [0, width * resolution_x_m, 0, height * time_interval_sec]
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
        fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
        
        ax.imshow(np.array(combined_image), extent=extent, aspect='auto', origin='lower')
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Time (s)")
        ax.set_title("Raw Image with Mask Overlay")
        ax.grid(True)
        
        self.canvas_img_fig = FigureCanvasTkAgg(fig, master=self.image_panel)
        self.canvas_img_fig.get_tk_widget().pack(fill="both", expand=True)
        self.canvas_img_fig.draw()
    
    def calculate_runup(self):
        if not self.raw_image_path or not self.mask_image_path:
            messagebox.showerror("Error", "Please load both the raw image and mask image before calculating.")
            return
        
        # Determine spatial resolution.
        if self.manual_res_var.get():
            try:
                resolution_x_m = float(self.manual_res_entry.get())
            except Exception:
                resolution_x_m = 0.25
        else:
            try:
                resolution_x_m = float(self.raw_image.info.get("pixel_resolution", 0.25))
            except Exception:
                resolution_x_m = 0.25
        
        try:
            time_interval_sec = float(self.raw_image.info.get("time_interval", 1))
        except Exception:
            time_interval_sec = 1
        
        flip_horizontal = self.land_left.get()
        
        # Extract runup contour.
        time_array, distance_array, pixel_coords = extract_runup_from_mask(
            self.mask_image_path, resolution_x_m, time_interval_sec, flip_horizontal
        )
        
        # Adjust time values so that time=0 is at the bottom.
        total_time = self.raw_image.height * time_interval_sec
        time_array_adjusted = total_time - time_array
        
        # Store data for export.
        self.runup_time = time_array_adjusted
        self.runup_distance = distance_array
        
        # Plot the extracted runup contour.
        self.ax.clear()
        self.ax.plot(
            distance_array,
            time_array_adjusted,
            marker='o',
            markersize=2,
            linestyle='-',
            color='tab:blue',
            label="Runup Contour"
        )
        self.ax.set_xlabel("Cross-shore distance (m)")
        self.ax.set_ylabel("Time (s)")
        self.ax.set_title("Extracted Runup Contour")
        self.ax.legend()
        self.canvas_plot.draw()
    
    def select_output_folder(self):
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_folder = folder
            self.out_folder_label.configure(text=folder)
    
    def export_runup(self):
        # Check if runup data is available.
        if self.runup_time is None or self.runup_distance is None:
            messagebox.showerror("Error", "No runup data to export. Please calculate runup first.")
            return
        if not self.output_folder:
            messagebox.showerror("Error", "Please select an output folder.")
            return
        if not self.raw_image_path:
            messagebox.showerror("Error", "Raw image file not loaded.")
            return

        # Extract base datetime from raw image file name.
        base_name = os.path.basename(self.raw_image_path)
        match = re.search(r"(\d{4})[-_](\d{2})[-_](\d{2})[-_](\d{2})[-_](\d{2})", base_name)
        if not match:
            messagebox.showerror("Error", "Could not extract date from raw image file name.")
            return
        
        year, month, day, hour, minute = map(int, match.groups())
        base_datetime = datetime.datetime(year, month, day, hour, minute, 0)

        # Create CSV file name.
        out_file_name = base_name.replace("raw", "runup")
        out_file_name = os.path.splitext(out_file_name)[0] + ".csv"
        out_file_path = os.path.join(self.output_folder, out_file_name)
        
        try:
            with open(out_file_path, mode="w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["time", "distance"])
                for sec, dist in zip(self.runup_time, self.runup_distance):
                    new_time = base_datetime + datetime.timedelta(seconds=float(sec))
                    time_str = new_time.strftime("%Y-%m-%d-%H-%M-%S")
                    writer.writerow([time_str, dist])
            messagebox.showinfo("Export Runup", f"Runup data exported successfully to:\n{out_file_path}")
        except Exception as e:
            messagebox.showerror("Export Error", f"An error occurred while exporting:\n{e}")
    
    def select_batch_raw_folder(self):
        folder = filedialog.askdirectory(title="Select Folder with Raw TS Images")
        if folder:
            self.batch_raw_folder = folder
            self.batch_raw_label.configure(text=folder)
    
    def select_batch_mask_folder(self):
        folder = filedialog.askdirectory(title="Select Folder with Run-Up Masks")
        if folder:
            self.batch_mask_folder = folder
            self.batch_mask_label.configure(text=folder)
    
    def run_batch_process(self):
        if not self.batch_raw_folder or not self.batch_mask_folder:
            messagebox.showerror("Error", "Please select both batch raw and batch mask folders.")
            return

        # 1) list all PNGs in each folder
        raw_files  = [f for f in os.listdir(self.batch_raw_folder)  if f.lower().endswith('.png')]
        mask_files = [f for f in os.listdir(self.batch_mask_folder) if f.lower().endswith('.png')]

        # 2) group by the common YYYY_MM_DD_HH_MM key
        date_pattern = r"(\d{4}[-_]\d{2}[-_]\d{2}[-_]\d{2}[-_]\d{2})"
        groups = defaultdict(lambda: {"raw": [], "mask": []})

        for f in raw_files:
            m = re.search(date_pattern, f)
            if m:
                groups[m.group(1)]["raw"].append(f)
        for f in mask_files:
            m = re.search(date_pattern, f)
            if m:
                groups[m.group(1)]["mask"].append(f)

        # 3) build valid_pairs list of (raw, mask)
        valid_pairs = []
        for key, files in groups.items():
            for raw in files["raw"]:
                for mask in files["mask"]:
                    valid_pairs.append((raw, mask))

        total_pairs = len(valid_pairs)
        if total_pairs == 0:
            messagebox.showerror("Error", "No valid pairs found for batch processing.")
            return

        # default output into raw folder if not set
        if not self.output_folder:
            self.output_folder = self.batch_raw_folder

        # reset progress
        self.batch_progress_bar.set(0)
        self.batch_progress_label.configure(text=f"0 / {total_pairs} pairs processed")
        self.update()

        all_runup_data = []
        processed = 0

        for raw_name, mask_name in valid_pairs:
            raw_path  = os.path.join(self.batch_raw_folder, raw_name)
            mask_path = os.path.join(self.batch_mask_folder, mask_name)

            try:
                raw_img  = Image.open(raw_path)
                mask_img = Image.open(mask_path).convert("L")
            except:
                continue

            # resolution & timing
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

            # extract runup
            t_arr, d_arr, _ = extract_runup_from_mask(mask_path,
                                                      resolution_x_m,
                                                      time_interval_sec,
                                                      flip_horizontal)
            total_time = raw_img.height * time_interval_sec
            t_adj = total_time - t_arr

            # export CSV
            out_name = os.path.splitext(raw_name.replace("raw", "runup"))[0] + ".csv"
            out_path = os.path.join(self.output_folder, out_name)

            # find base datetime from raw_name
            m = re.search(date_pattern, raw_name)
            if m:
                year,mo,da,hr,mi = map(int, re.split('[-_]', m.group(1)))
                base_dt = datetime.datetime(year,mo,da,hr,mi)
            else:
                base_dt = datetime.datetime.now()

            with open(out_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["time","distance"])
                for sec,dist in zip(t_adj, d_arr):
                    ts = (base_dt + datetime.timedelta(seconds=float(sec))) \
                              .strftime("%Y-%m-%d-%H-%M-%S")
                    writer.writerow([ts, dist])

            all_runup_data.append((d_arr, t_adj))
            processed += 1
            self.batch_progress_bar.set(processed/total_pairs)
            self.batch_progress_label.configure(text=f"{processed} / {total_pairs} pairs processed")
            self.update()

        # plot aggregated
        self.ax.clear()
        for d_arr, t_arr in all_runup_data:
            self.ax.plot(
                d_arr,
                t_arr,
                marker='o',
                markersize=1,
                linestyle='-'
            )
        self.ax.set_xlabel("Distance (m)")
        self.ax.set_ylabel("Time (s)")
        self.ax.set_title("Aggregated Runup Contours")
        self.ax.grid(True)
        self.canvas_plot.draw()

        messagebox.showinfo("Batch Process", "Batch processing completed.")


def main():
    root = ctk.CTk()
    root.withdraw()
    # now spawn your tool as a child window
    win = WaveRunUpCalculator(master=root)
    win.mainloop()

if __name__ == "__main__":
    main()