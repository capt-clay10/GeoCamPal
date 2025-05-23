import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
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
from collections import defaultdict

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


def extract_runup_from_mask(mask_path, resolution_x_m, time_interval_sec, flip_horizontal=False):
    mask_img = Image.open(mask_path).convert("L")
    mask_array = np.array(mask_img)
    binary_mask = mask_array > 128
    rows, cols = np.where(binary_mask)
    unique_rows = np.unique(rows)

    time_array = []
    distance_array = []
    pixel_coords = []
    for r in unique_rows:
        cols_in_row = cols[rows == r]
        if len(cols_in_row) == 0:
            continue
        runup_col = np.min(cols_in_row) if flip_horizontal else np.max(cols_in_row)
        time_val = r * time_interval_sec
        distance_val = runup_col * resolution_x_m
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
        except Exception:
            pass

        # Data holders
        self.raw_image = None
        self.raw_image_path = ""
        self.mask_image = None
        self.mask_image_path = ""
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

        # Image panel
        self.image_panel = ctk.CTkFrame(self.top_frame, width=400, height=400)
        self.image_panel.pack_propagate(False)
        self.image_panel.pack(side="left", fill="both", expand=True, padx=5, pady=5)

        # Runup contour panel
        self.plot_panel = ctk.CTkFrame(self.top_frame)
        self.plot_panel.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        self.fig, self.ax = plt.subplots(figsize=(5,4))
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=self.plot_panel)
        self.canvas_plot.get_tk_widget().pack(fill="both", expand=True)

        # Stats panel
        self.stats_panel = ctk.CTkFrame(self.top_frame)
        self.stats_panel.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        self.fig_stats, (self.ax_stats_psd, self.ax_stats_swash) = plt.subplots(2,1, figsize=(5,4))
        self.canvas_stats = FigureCanvasTkAgg(self.fig_stats, master=self.stats_panel)
        self.canvas_stats.get_tk_widget().pack(fill="both", expand=True)

        # Bottom frame
        self.bottom_panel = ctk.CTkFrame(self)
        self.bottom_panel.pack(side="bottom", fill="x", padx=5, pady=5)

        # Controls
        self.controls_panel = ctk.CTkFrame(self.bottom_panel)
        self.controls_panel.pack(side="top", fill="x", padx=5, pady=2)
        self.btn_load_raw = ctk.CTkButton(self.controls_panel, text="Load Raw Time-Stack Image", command=self.load_raw_image)
        self.btn_load_raw.grid(row=0, column=0, padx=5, pady=5)
        self.btn_load_mask = ctk.CTkButton(self.controls_panel, text="Load Mask of Run-Up", command=self.load_mask)
        self.btn_load_mask.grid(row=0, column=1, padx=5, pady=5)
        self.land_left = tk.BooleanVar()
        self.chk_land_left = ctk.CTkCheckBox(self.controls_panel, text="Land on left", variable=self.land_left)
        self.chk_land_left.grid(row=0, column=2, padx=5, pady=5)
        self.btn_calculate = ctk.CTkButton(self.controls_panel, text="Calculate Runup", command=self.calculate_runup)
        self.btn_calculate.grid(row=0, column=3, padx=5, pady=5)

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
        self.btn_select_batch_mask = ctk.CTkButton(self.batch_panel, text="Batch Mask Folder", command=self.select_batch_mask_folder)
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

    def load_mask(self):
        file_path = filedialog.askopenfilename(title="Select Run-Up Mask Image", filetypes=[("PNG Images", "*.png")])
        if file_path:
            self.mask_image_path = file_path
            self.mask_image = Image.open(file_path).convert("L")
            if self.raw_image:
                raw_copy = self.raw_image.copy().convert("RGBA")
                mask_resized = self.mask_image.resize(raw_copy.size)
                overlay = Image.new("RGBA", raw_copy.size, (255,0,0,0))
                overlay_data = overlay.load()
                mask_data = mask_resized.load()
                for i in range(raw_copy.width):
                    for j in range(raw_copy.height):
                        if mask_data[i,j] > 128:
                            overlay_data[i,j] = (255,0,0,100)
                combined = Image.alpha_composite(raw_copy, overlay)
                if self.manual_res_var.get():
                    try:
                        resolution_x_m = float(self.manual_res_entry.get())
                    except:
                        resolution_x_m = 0.25
                        print("Warning: Invalid manual resolution; defaulting to 0.25 m/pixel.")
                else:
                    try:
                        resolution_x_m = float(self.raw_image.info.get("pixel_resolution",0.25))
                    except:
                        resolution_x_m = 0.25
                        print("Warning: Invalid metadata resolution; defaulting to 0.25 m/pixel.")
                self.pixel_res_label.configure(text=f"Identified Pixel Resolution: {resolution_x_m} m")
                try:
                    time_interval_sec = float(self.raw_image.info.get("time_interval",1))
                except:
                    time_interval_sec = 1
                for child in self.image_panel.winfo_children(): child.destroy()
                self.display_image_with_axes(combined, resolution_x_m, time_interval_sec)

    def display_image_with_axes(self, combined_image, resolution_x_m, time_interval_sec):
        width, height = combined_image.size
        max_size, dpi = 200, 150
        aspect_ratio = width/height
        if aspect_ratio>1:
            fig_w = max_size/dpi; fig_h=(max_size/aspect_ratio)/dpi
        else:
            fig_h = max_size/dpi; fig_w=(max_size*aspect_ratio)/dpi
        extent = [0, width*resolution_x_m, 0, height*time_interval_sec]
        fig, ax = plt.subplots(figsize=(fig_w,fig_h), dpi=dpi)
        fig.subplots_adjust(left=0.15,right=0.95,top=0.95,bottom=0.15)
        ax.imshow(np.array(combined_image),extent=extent,aspect='auto',origin='lower')
        ax.set_xlabel("Distance (m)"); ax.set_ylabel("Time (s)"); ax.set_title("Raw Image with Mask Overlay"); ax.grid(True)
        self.canvas_img_fig = FigureCanvasTkAgg(fig, master=self.image_panel)
        self.canvas_img_fig.get_tk_widget().pack(fill="both", expand=True)
        self.canvas_img_fig.draw()

    def calculate_runup(self):
        if not self.raw_image_path or not self.mask_image_path:
            messagebox.showerror("Error", "Load both raw image and mask first.")
            return

        # Get resolution and timing
        resolution_x_m = float(self.manual_res_entry.get()) if self.manual_res_var.get() else float(self.raw_image.info.get("pixel_resolution", 0.25))
        time_interval_sec = float(self.raw_image.info.get("time_interval", 1))
        flip_horizontal = self.land_left.get()

        # Extract runup contour
        t_arr, d_arr, _ = extract_runup_from_mask(
            self.mask_image_path, resolution_x_m, time_interval_sec, flip_horizontal
        )
        total_time = self.raw_image.height * time_interval_sec
        # Adjust time so zero at start and sort
        t_adj = total_time - t_arr
        # Sort by time ascending
        sort_idx = np.argsort(t_adj)
        t_sorted = t_adj[sort_idx]
        d_sorted = d_arr[sort_idx]

        # Store for export
        self.runup_time, self.runup_distance = t_sorted, d_sorted

        # Plot the extracted runup contour
        self.ax.clear()
        self.ax.plot(d_sorted, t_sorted, 'bo-', markersize=2, linewidth=1, label="Runup Contour")
        self.ax.set_xlabel("Cross-shore distance (m)")
        self.ax.set_ylabel("Time (s)")
        self.ax.set_title("Extracted Runup Contour")
        self.ax.legend()
        self.canvas_plot.draw()

        # --- Update stats for single image ---
        # Detrend swash
        detr = d_sorted - np.mean(d_sorted)
        # Compute sampling frequency (positive)
        dt = np.diff(t_sorted)
        fs = 1.0 / np.mean(dt) if len(dt) > 0 else 1.0

        # Compute PSD
        fxx, pxx = welch(detr, fs=fs, nperseg=min(256, len(detr)))
        # Keep only positive frequencies
        mask_pos = fxx > 0
        fxx = fxx[mask_pos]
        pxx = pxx[mask_pos]

        # IG band mask & percentage
        ig_mask = fxx < 0.05
        E_ig = np.trapz(pxx[ig_mask], fxx[ig_mask])
        E_tot = np.trapz(pxx, fxx)
        ig_pct = 100 * E_ig / E_tot if E_tot > 0 else 0

        # Plot PSD
        self.ax_stats_psd.clear()
        self.ax_stats_psd.plot(fxx, pxx, label='PSD')
        self.ax_stats_psd.fill_between(
            fxx, pxx, where=ig_mask, alpha=0.3,
            label=f'IG (<0.05Hz) {ig_pct:.1f}%'
        )
        self.ax_stats_psd.set_xscale('log')
        self.ax_stats_psd.set_title('Power Spectral Density')
        self.ax_stats_psd.set_ylabel('PSD')
        self.ax_stats_psd.legend()

        # Plot detrended swash excursion
        self.ax_stats_swash.clear()
        self.ax_stats_swash.plot(t_sorted, detr, label="Detrended Swash")
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
            messagebox.showerror("Error","No runup data to export.")
            return
        if not self.output_folder:
            messagebox.showerror("Error","Select an output folder.")
            return
        base_name = os.path.basename(self.raw_image_path)
        match = re.search(r"(\d{4})[-_](\d{2})[-_](\d{2})[-_](\d{2})[-_](\d{2})", base_name)
        if not match:
            messagebox.showerror("Error","Could not extract date.")
            return
        year, month, day, hour, minute = map(int, match.groups())
        base_dt = datetime.datetime(year, month, day, hour, minute)
        out_name = os.path.splitext(base_name.replace("raw","runup"))[0] + ".csv"
        out_path = os.path.join(self.output_folder, out_name)
        with open(out_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["time","distance"]
            )
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
        folder = filedialog.askdirectory(title="Select Folder with Run-Up Masks")
        if folder:
            self.batch_mask_folder = folder
            self.batch_mask_label.configure(text=folder)
            
            
            
            
    def run_batch_process(self):
        # 1) Validate folders
        if not self.batch_raw_folder or not self.batch_mask_folder:
            messagebox.showerror("Error", "Please select both batch raw and batch mask folders.")
            return
        print("Batch process has started")
        # 2) List all PNGs
        raw_files  = [f for f in os.listdir(self.batch_raw_folder)  if f.lower().endswith('.png')]
        mask_files = [f for f in os.listdir(self.batch_mask_folder) if f.lower().endswith('.png')]

        # 3) Group by timestamp key
        date_pattern = r"(\d{4}[-_]\d{2}[-_]\d{2}[-_]\d{2}[-_]\d{2})"
        groups = defaultdict(lambda: {"raw": [], "mask": []})
        for f in raw_files:
            m = re.search(date_pattern, f)
            if m: groups[m.group(1)]["raw"].append(f)
        for f in mask_files:
            m = re.search(date_pattern, f)
            if m: groups[m.group(1)]["mask"].append(f)

        # 4) Build valid pairs
        valid_pairs = [(r, m) 
                       for _, files in groups.items() 
                       for r in files["raw"] 
                       for m in files["mask"]]
        total_pairs = len(valid_pairs)
        if total_pairs == 0:
            messagebox.showerror("Error", "No valid pairs found for batch processing.")
            return

        # default output if none
        if not self.output_folder:
            self.output_folder = self.batch_raw_folder

        # reset progress UI
        self.batch_progress_bar.set(0)
        self.batch_progress_label.configure(text=f"0 / {total_pairs} pairs processed")
        self.update()

        all_runup_data = []
        processed = 0

        # 5) Loop through each pair
        for raw_name, mask_name in valid_pairs:
            raw_path  = os.path.join(self.batch_raw_folder, raw_name)
            mask_path = os.path.join(self.batch_mask_folder, mask_name)

            try:
                raw_img = Image.open(raw_path)
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
            t_arr, d_arr, _ = extract_runup_from_mask(
                mask_path, resolution_x_m, time_interval_sec, flip_horizontal
            )
            total_time = raw_img.height * time_interval_sec
            t_adj = total_time - t_arr

            # sort times & distances ascending
            sort_idx = np.argsort(t_adj)
            t_sorted = t_adj[sort_idx]
            d_sorted = d_arr[sort_idx]

            # export CSV
            m = re.search(date_pattern, raw_name)
            if m:
                y,mo,da,hr,mi = map(int, re.split('[-_]', m.group(1)))
                base_dt = datetime.datetime(y,mo,da,hr,mi)
            else:
                base_dt = datetime.datetime.now()

            out_name = os.path.splitext(raw_name.replace("raw", "runup"))[0] + ".csv"
            out_path = os.path.join(self.output_folder, out_name)
            with open(out_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["time","distance"])
                for sec, dist in zip(t_sorted, d_sorted):
                    ts = (base_dt + datetime.timedelta(seconds=float(sec)))\
                             .strftime("%Y-%m-%d-%H-%M-%S")
                    writer.writerow([ts, dist])

            all_runup_data.append((d_sorted, t_sorted))
            processed += 1
            self.batch_progress_bar.set(processed/total_pairs)
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
        for d_arr, t_arr in all_runup_data:
            detr = d_arr - np.mean(d_arr)
            dt   = np.diff(t_arr)
            fs   = 1.0 / np.mean(dt) if len(dt)>0 else 1.0
            fxx, pxx = welch(detr, fs=fs, nperseg=min(256, len(detr)))
            # drop zero/non-positive freqs
            pos = fxx > 0
            fxx, pxx = fxx[pos], pxx[pos]
            ig_mask = fxx < 0.05
            E_ig = np.trapz(pxx[ig_mask], fxx[ig_mask])
            E_tot = np.trapz(pxx, fxx)
            ig_pct = 100*E_ig/E_tot if E_tot>0 else 0
            ig_list.append(ig_pct)
            inc_list.append(100-ig_pct)

        # bar plot for energy partitioning
        self.ax_stats_psd.clear()
        idx = np.arange(len(ig_list))
        self.ax_stats_psd.bar(idx, ig_list, label='IG%')
        self.ax_stats_psd.bar(idx, inc_list, bottom=ig_list, label='Incident%')
        self.ax_stats_psd.set_xticks(idx)
        self.ax_stats_psd.set_xticklabels([str(i+1) for i in idx])
        self.ax_stats_psd.set_ylabel('%')
        self.ax_stats_psd.set_title('Energy Partitioning')
        self.ax_stats_psd.legend()

        # swash excursion envelope
        self.ax_stats_swash.clear()
        for d_arr, t_arr in all_runup_data:
            detr = d_arr - np.mean(d_arr)
            self.ax_stats_swash.plot(t_arr, detr, alpha=0.6)
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
