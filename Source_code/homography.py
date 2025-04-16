import numpy as np
import cv2
import pandas as pd
import os
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import random
import sys
import time
import threading


# --- StdoutRedirector class for redirecting console output into the GUI ---
class StdoutRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        # Insert the message into the text widget and scroll to the end.
        self.text_widget.insert(tk.END, message)
        self.text_widget.see(tk.END)

    def flush(self):
        pass  # For compatibility with Python's IO system.

def resource_path(relative_path: str) -> str:
    try:
        base_path = sys._MEIPASS  # If running in a PyInstaller .exe
    except Exception:
        base_path = os.path.dirname(__file__)  # Running directly from source
    return os.path.join(base_path, relative_path)


class CreateHomographyMatrixWindow(ctk.CTk):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("Create Homography Matrix")
        self.geometry("850x700")
        
        try:
            self.iconbitmap(resource_path("launch_logo.ico"))
        except Exception as e:
            print("Warning: Could not load window icon:", e)

        # Variables to store file and folder selections
        self.input_file = None
        self.output_folder = None
        # Variables to store computed data
        self.H_final = None
        self.pixel_points = None
        self.utm_points = None
        # Variable to store the best matrix from simulated annealing
        self.best_H_annealing = None
        self.best_cost = None

        # -------------------------------
        # Panel 1: Input GCP File Selection
        # -------------------------------
        self.panel1 = ctk.CTkFrame(self)
        self.panel1.pack(padx=10, pady=10, fill="x")
        
        self.label_input = ctk.CTkLabel(self.panel1, text="Input GCP File (.csv):")
        self.label_input.pack(side="left", padx=5, pady=5)
        
        self.btn_browse_input = ctk.CTkButton(self.panel1, text="Browse", command=self.browse_input)
        self.btn_browse_input.pack(side="left", padx=5, pady=5)
        
        self.input_file_label = ctk.CTkLabel(self.panel1, text="No file selected", fg_color="transparent")
        self.input_file_label.pack(side="left", padx=5, pady=5)
        
        # Note about required columns
        self.label_note = ctk.CTkLabel(
            self.panel1, 
            text="Note: File must include columns 'Pixel_X', 'Pixel_Y', 'Real_X', 'Real_Y'",
            fg_color="transparent"
        )
        self.label_note.pack(side="left", padx=10, pady=5)

        # -------------------------------
        # Panel 2: Output Name & Folder Selection
        # -------------------------------
        self.panel2 = ctk.CTkFrame(self)
        self.panel2.pack(padx=10, pady=10, fill="x")
        
        self.label_output_name = ctk.CTkLabel(self.panel2, text="Output File Name:")
        self.label_output_name.pack(side="left", padx=5, pady=5)
        
        self.entry_output_name = ctk.CTkEntry(self.panel2)
        self.entry_output_name.pack(side="left", padx=5, pady=5, fill="x", expand=True)
        
        self.btn_browse_output = ctk.CTkButton(self.panel2, text="Choose Output Folder", command=self.browse_output)
        self.btn_browse_output.pack(side="left", padx=5, pady=5)
        
        self.output_folder_label = ctk.CTkLabel(self.panel2, text="No folder selected", fg_color="transparent")
        self.output_folder_label.pack(side="left", padx=5, pady=5)

        # -------------------------------
        # Panel X: Exclude GCPs Input
        # -------------------------------
        self.panel_exclude = ctk.CTkFrame(self)
        self.panel_exclude.pack(padx=10, pady=10, fill="x")
        
        self.exclude_var = tk.BooleanVar(value=False)
        self.chk_exclude = ctk.CTkCheckBox(self.panel_exclude, text="Exclude GCPs", variable=self.exclude_var)
        self.chk_exclude.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        self.label_exclude = ctk.CTkLabel(self.panel_exclude, text="(List of integer indices, comma separated)")
        self.label_exclude.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        self.entry_exclude = ctk.CTkEntry(self.panel_exclude)
        self.entry_exclude.grid(row=0, column=2, padx=5, pady=5, sticky="we")
        self.panel_exclude.columnconfigure(2, weight=1)

        # -------------------------------
        # Panel 3: Compute Buttons (Matrix and Accuracy)
        # -------------------------------
        self.panel3 = ctk.CTkFrame(self)
        self.panel3.pack(padx=10, pady=10, fill="x")
        
        self.btn_compute = ctk.CTkButton(self.panel3, text="Compute Matrix", command=self.compute_matrix)
        self.btn_compute.grid(row=0, column=0, padx=5, pady=5)
        
        self.btn_compute_accuracy = ctk.CTkButton(self.panel3, text="Compute Accuracy", command=self.compute_accuracy)
        self.btn_compute_accuracy.grid(row=0, column=1, padx=5, pady=5)

        # -------------------------------
        # Panel 4: Advanced Options
        # -------------------------------
        # Panel with a checkbox for advanced mode. When checked, the advanced config appears.
        self.panel_adv = ctk.CTkFrame(self)
        self.panel_adv.pack(padx=10, pady=10, fill="x")

        self.advanced_var = tk.BooleanVar(value=False)
        self.chk_advanced = ctk.CTkCheckBox(self.panel_adv, text="Advanced", variable=self.advanced_var, command=self.toggle_advanced)
        self.chk_advanced.pack(side="left", padx=5, pady=5)

        self.lbl_adv_info = ctk.CTkLabel(self.panel_adv, text="", fg_color="transparent")
        self.lbl_adv_info.pack(side="left", padx=5, pady=5)

        # Frame that will hold advanced configuration options (initially hidden)
        self.advanced_frame = ctk.CTkFrame(self)
        # Do not pack it yet; it will be packed when advanced mode is enabled.

        # Advanced configuration inputs arranged in a grid
        self.adv_inputs = {}
        adv_options = [
            ("Number of GCPs", "90"),
            ("Max iterations", "300000"),
            ("Initial temperature", "100000.0"),
            ("Cooling rate", "0.99995"),
            ("Number of swaps", "1"),
            ("RANSAC threshold", "0.7")
        ]
        for idx, (label_text, default_val) in enumerate(adv_options):
            lbl = ctk.CTkLabel(self.advanced_frame, text=label_text + ":")
            lbl.grid(row=idx, column=0, padx=5, pady=3, sticky="e")
            entry = ctk.CTkEntry(self.advanced_frame)
            entry.insert(0, default_val)
            entry.grid(row=idx, column=1, padx=5, pady=3, sticky="we")
            self.advanced_frame.columnconfigure(1, weight=1)
            self.adv_inputs[label_text] = entry

        # Advanced buttons panel
        self.adv_buttons_frame = ctk.CTkFrame(self.advanced_frame)
        self.adv_buttons_frame.grid(row=len(adv_options), column=0, columnspan=2, pady=10)
        self.btn_sa_search = ctk.CTkButton(self.adv_buttons_frame, text="Simulated Annealing Search", command=self.run_simulated_annealing)
        self.btn_sa_search.grid(row=0, column=0, padx=5, pady=5)
        self.btn_accept_export = ctk.CTkButton(self.adv_buttons_frame, text="Accept and Export New Matrix", command=self.accept_and_export_new_matrix)
        self.btn_accept_export.grid(row=0, column=1, padx=5, pady=5)
        
        # -------------------------------
        # Panel 5: Console Output Panel
        # -------------------------------
        self.panel5 = ctk.CTkFrame(self)
        self.panel5.pack(padx=10, pady=10, fill="both", expand=True)
        
        self.text_console = tk.Text(self.panel5, wrap="word", height=10)
        self.text_console.pack(padx=5, pady=5, fill="both", expand=True)

        # Redirect stdout and stderr to the console text widget.
        self.stdout_redirector = StdoutRedirector(self.text_console)
        sys.stdout = self.stdout_redirector
        sys.stderr = self.stdout_redirector
        print("Here you may see console outputs\n")
        
    def log(self, message):
        """Append a message to the console output."""
        self.text_console.insert(tk.END, message + "\n")
        self.text_console.see(tk.END)

    def thread_log(self, message):
        """Schedule a log message to be added via the main thread."""
        self.after(0, lambda: self.log(message))

    def browse_input(self):
        """Open a file dialog to select the CSV file containing GCP data."""
        file_path = filedialog.askopenfilename(title="Select GCP CSV File", 
                                               filetypes=(("CSV Files", "*.csv"), ("All Files", "*.*")))
        if file_path:
            self.input_file = file_path
            self.input_file_label.configure(text=os.path.basename(file_path))
            self.log(f"Selected input file: {file_path}")

    def browse_output(self):
        """Open a dialog to select an output folder."""
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_folder = folder
            self.output_folder_label.configure(text=folder)
            self.log(f"Selected output folder: {folder}")

    def toggle_advanced(self):
        """Show or hide advanced configuration options."""
        if self.advanced_var.get():
            self.lbl_adv_info.configure(text="Using simulated annealing to identify best subset of GCPS")
            self.advanced_frame.pack(padx=10, pady=5, fill="x")
        else:
            self.lbl_adv_info.configure(text="")
            self.advanced_frame.forget()

    def normalize_points(self, points):
        """
        Normalize a set of 2D points so that the centroid is at (0,0)
        and the average distance from the origin is √2.
        """
        centroid = np.mean(points, axis=0)
        shifted = points - centroid
        dists = np.sqrt(np.sum(shifted ** 2, axis=1))
        mean_dist = np.mean(dists)
        scale = np.sqrt(2) / mean_dist
        T = np.array([
            [scale, 0, -scale * centroid[0]],
            [0, scale, -scale * centroid[1]],
            [0, 0, 1]
        ])
        ones = np.ones((points.shape[0], 1))
        points_h = np.hstack([points, ones])
        normalized_points_h = (T @ points_h.T).T
        return normalized_points_h, T

    def compute_matrix(self):
        """Perform the homography matrix computation using the selected GCP CSV file."""
        if not self.input_file:
            messagebox.showerror("Error", "Please select an input GCP file.")
            return
        if not self.output_folder:
            messagebox.showerror("Error", "Please select an output folder.")
            return
        output_name = self.entry_output_name.get().strip()
        if not output_name:
            messagebox.showerror("Error", "Please enter an output file name.")
            return

        self.log("Reading GCP file...")
        try:
            gcp_data = pd.read_csv(self.input_file)
        except Exception as e:
            self.log(f"Error reading CSV file: {e}")
            return

        # Ensure required columns exist
        required_columns = ['Pixel_X', 'Pixel_Y', 'Real_X', 'Real_Y']
        for col in required_columns:
            if col not in gcp_data.columns:
                self.log(f"Missing required column: {col}")
                messagebox.showerror("Error", f"Missing required column: {col}")
                return

        # Exclude GCPs if checkbox is ticked
        if self.exclude_var.get():
            exclude_str = self.entry_exclude.get()
            if exclude_str.strip():
                try:
                    exclude_list = [int(x.strip()) for x in exclude_str.split(',') if x.strip() != ""]
                    self.log(f"Excluding GCP indices: {exclude_list}")
                    gcp_data = gcp_data.drop(exclude_list, errors='ignore')
                except Exception as e:
                    self.log(f"Error parsing excluded GCPs: {e}")

        # Extract pixel and real-world points
        pixel_points = gcp_data[['Pixel_X', 'Pixel_Y']].values.astype(np.float32)
        utm_points = gcp_data[['Real_X', 'Real_Y']].values.astype(np.float32)
        self.pixel_points = pixel_points
        self.utm_points = utm_points

        if len(pixel_points) < 4 or len(utm_points) < 4:
            self.log("Not enough GCPs! Need at least 4.")
            messagebox.showerror("Error", "Not enough GCPs! Need at least 4.")
            return

        self.log("Normalizing points...")
        pixel_points_norm, T_pixel = self.normalize_points(pixel_points)
        utm_points_norm, T_utm = self.normalize_points(utm_points)

        pixel_points_norm_2d = pixel_points_norm[:, :2]
        utm_points_norm_2d = utm_points_norm[:, :2]

        self.log("Computing homography matrix using RANSAC...")
        H_norm, status = cv2.findHomography(pixel_points_norm_2d, utm_points_norm_2d, cv2.RANSAC, 0.5)
        if H_norm is None:
            self.log("Failed to compute homography matrix. Please check your input data.")
            messagebox.showerror("Error", "Failed to compute homography matrix. Please check your input data.")
            return

        H_final = np.linalg.inv(T_utm) @ H_norm @ T_pixel
        if abs(H_final[2, 2]) > 1e-6 and abs(H_final[2, 2] - 1) > 1e-3:
            H_final = H_final / H_final[2, 2]

        self.H_final = H_final

        output_path = os.path.join(self.output_folder, f"{output_name}.txt")
        try:
            np.savetxt(output_path, H_final)
        except Exception as e:
            self.log(f"Error saving homography matrix: {e}")
            messagebox.showerror("Error", f"Error saving homography matrix: {e}")
            return

        self.log("\nHomography Matrix Computed and Saved!")
        self.log(f"Output Path: {output_path}")
        self.log(f"Homography matrix:\n{H_final}")
        messagebox.showinfo("Success", "Homography matrix computed and saved successfully.")

    def compute_accuracy(self):
        """Compute and display the homography accuracy based on the GCPs."""
        if self.H_final is None or self.pixel_points is None or self.utm_points is None:
            messagebox.showerror("Error", "Please compute the homography matrix first!")
            return

        self.log("Computing accuracy using the computed homography matrix...")

        # Option 1: Using cv2.perspectiveTransform
        pixel_points_reshaped = self.pixel_points.reshape(-1, 1, 2)
        utm_estimated = cv2.perspectiveTransform(pixel_points_reshaped, self.H_final)
        utm_estimated = utm_estimated.reshape(-1, 2)

        # Option 2: Manual transformation
        ones = np.ones((self.pixel_points.shape[0], 1), dtype=np.float32)
        pixel_points_hom = np.hstack([self.pixel_points, ones])
        transformed = (self.H_final @ pixel_points_hom.T).T
        utm_estimated_manual = transformed[:, :2] / transformed[:, 2][:, np.newaxis]

        max_diff = np.max(np.abs(utm_estimated - utm_estimated_manual))
        accuracy_report = f"Max difference between methods: {max_diff:.3f}\n"
        errors = np.linalg.norm(utm_estimated - self.utm_points, axis=1)
        for idx, err in enumerate(errors, start=1):
            accuracy_report += f"GCP {idx}: Error = {err:.3f} meters\n"
        mean_error = np.mean(errors)
        accuracy_report += f"Mean error: {mean_error:.3f} meters\n"
        self.log(accuracy_report)

    # --- Advanced (Simulated Annealing) Methods ---
    
    def compute_homography_and_errors(self, pixel_points, utm_points, subset_indices, ransac_thresh=0.5):
        """
        Compute homography using a subset of points and return error metrics and H.
        """
        subset_px = pixel_points[subset_indices]
        subset_utm = utm_points[subset_indices]
        if len(subset_indices) < 4:
            return 9999.0, 9999.0, 9999.0, 9999.0, 0.0, 9999.0, None
        try:
            px_norm, T_px = self.normalize_points(subset_px)
            utm_norm, T_utm = self.normalize_points(subset_utm)
            px_norm_2d  = px_norm[:, :2]
            utm_norm_2d = utm_norm[:, :2]
            H_norm, _ = cv2.findHomography(px_norm_2d, utm_norm_2d, cv2.RANSAC, ransac_thresh)
            if H_norm is None:
                return 9999.0, 9999.0, 9999.0, 9999.0, 0.0, 9999.0, None
            H_final = np.linalg.inv(T_utm) @ H_norm @ T_px
            if abs(H_final[2, 2]) > 1e-6 and abs(H_final[2, 2] - 1) > 1e-3:
                H_final /= H_final[2, 2]
            ones = np.ones((subset_px.shape[0], 1), dtype=np.float32)
            subset_px_hom = np.hstack([subset_px, ones])
            utm_est = (H_final @ subset_px_hom.T).T
            utm_est = utm_est[:, :2] / utm_est[:, 2][:, np.newaxis]
            errors = np.linalg.norm(utm_est - subset_utm, axis=1)
            mean_err = np.mean(errors)
            median_err = np.median(errors)
            std_err = np.std(errors)
            max_err = np.max(errors)
            inlier_ratio = np.mean(errors < 10.0)
            count_above_5 = np.sum(errors > 5.0)
            return mean_err, median_err, std_err, max_err, inlier_ratio, count_above_5, H_final
        except Exception:
            return 9999.0, 9999.0, 9999.0, 9999.0, 0.0, 9999.0, None

    def compute_full_errors(self, H, pixel_points, utm_points):
        """
        Compute error metrics on all GCPs using homography H.
        """
        ones = np.ones((pixel_points.shape[0], 1), dtype=np.float32)
        pixel_points_hom = np.hstack([pixel_points, ones])
        utm_est = (H @ pixel_points_hom.T).T
        utm_est = utm_est[:, :2] / utm_est[:, 2][:, np.newaxis]
        errors = np.linalg.norm(utm_est - utm_points, axis=1)
        mean_err = np.mean(errors)
        median_err = np.median(errors)
        std_err = np.std(errors)
        max_err = np.max(errors)
        inlier_ratio = np.mean(errors < 10.0)
        count_above_5 = np.sum(errors > 5.0)
        return mean_err, median_err, std_err, max_err, inlier_ratio, count_above_5

    def run_simulated_annealing(self):
        """Start simulated annealing search in a separate thread."""
        threading.Thread(target=self.run_simulated_annealing_thread, daemon=True).start()

    def run_simulated_annealing_thread(self):
        """Execute simulated annealing search using advanced configuration parameters."""
        if self.pixel_points is None or self.utm_points is None:
            self.thread_log("Error: Please compute the homography matrix first (or load a valid GCP file).")
            return

        # Retrieve advanced configuration parameters
        try:
            n_subset = int(self.adv_inputs["Number of GCPs"].get())
            max_iterations = int(self.adv_inputs["Max iterations"].get())
            init_temp = float(self.adv_inputs["Initial temperature"].get())
            cooling_rate = float(self.adv_inputs["Cooling rate"].get())
            num_swaps = int(self.adv_inputs["Number of swaps"].get())
            ransac_thresh = float(self.adv_inputs["RANSAC threshold"].get())
        except Exception as e:
            self.thread_log(f"Invalid advanced configuration: {e}")
            return

        n_gcp = len(self.pixel_points)
        all_indices = np.arange(n_gcp)
        if n_subset > n_gcp:
            self.thread_log("Error: Number of GCPs for subset cannot exceed total GCP count.")
            return

        def cost_function(mean_err, median_err, std_err, max_err, inlier_ratio, count_above_5):
            penalty = 0.0
            if max_err > 10.0:
                penalty += 15000.0
            penalty += (1 - inlier_ratio) * 8000.0
            return mean_err + 2 * std_err + penalty

        current_subset = np.random.choice(all_indices, size=n_subset, replace=False)
        (sub_mean, sub_median, sub_std, sub_max, sub_inlier_ratio, sub_count_above_5, H_candidate) = \
            self.compute_homography_and_errors(self.pixel_points, self.utm_points, current_subset, ransac_thresh)
        full_metrics = self.compute_full_errors(H_candidate, self.pixel_points, self.utm_points)
        current_cost = cost_function(*full_metrics)

        best_subset = current_subset.copy()
        best_cost = current_cost
        best_H = H_candidate

        temperature = init_temp

        self.thread_log("Starting simulated annealing search...")
        start_time = time.time()
        for iteration in range(max_iterations):
            new_subset = best_subset.copy()
            for _ in range(num_swaps):
                swap_out = random.choice(new_subset)
                outside_indices = np.setdiff1d(all_indices, new_subset)
                swap_in = random.choice(outside_indices)
                idx_in_subset = np.where(new_subset == swap_out)[0][0]
                new_subset[idx_in_subset] = swap_in

            (sub_mean2, sub_median2, sub_std2, sub_max2, sub_inlier_ratio2, sub_count_above_5_2, H_candidate2) = \
                self.compute_homography_and_errors(self.pixel_points, self.utm_points, new_subset, ransac_thresh)
            full_metrics2 = self.compute_full_errors(H_candidate2, self.pixel_points, self.utm_points)
            new_cost = cost_function(*full_metrics2)
            cost_diff = new_cost - best_cost
            if cost_diff < 0:
                best_subset = new_subset.copy()
                best_cost = new_cost
                best_H = H_candidate2
            else:
                accept_prob = np.exp(-cost_diff / temperature) if temperature > 0 else 0.0
                if np.random.rand() < accept_prob:
                    best_subset = new_subset.copy()
                    best_cost = new_cost
                    best_H = H_candidate2

            temperature *= cooling_rate

            if (iteration + 1) % 50000 == 0:
                self.thread_log(f"Iteration {iteration+1}/{max_iterations} | Temp={temperature:.4f} | BestCost={best_cost:.3f}")

        elapsed = time.time() - start_time
        self.thread_log(f"Simulated annealing finished in {elapsed:.2f} seconds")
        self.thread_log(f"Best cost: {best_cost:.3f}")
        full_metrics_best = self.compute_full_errors(best_H, self.pixel_points, self.utm_points)
        self.thread_log(f"Final Performance on All GCPs (Simulated Annealing):\n"
                        f"Mean error: {full_metrics_best[0]:.2f} m, "
                        f"Median error: {full_metrics_best[1]:.2f} m, "
                        f"Std error: {full_metrics_best[2]:.2f} m, "
                        f"Max error: {full_metrics_best[3]:.2f} m, "
                        f"Inlier ratio: {full_metrics_best[4]*100:.1f}%, "
                        f"Errors >5m: {full_metrics_best[5]}")
        self.best_H_annealing = best_H

    def accept_and_export_new_matrix(self):
        """Export the best homography matrix found via simulated annealing."""
        if self.best_H_annealing is None:
            messagebox.showerror("Error", "No simulated annealing result found. Run the search first.")
            return
        output_name = self.entry_output_name.get().strip()
        if not output_name:
            messagebox.showerror("Error", "Please enter an output file name.")
            return
        output_path = os.path.join(self.output_folder, f"{output_name}_bestsubset.txt")
        try:
            np.savetxt(output_path, self.best_H_annealing)
        except Exception as e:
            self.log(f"Error exporting best homography matrix: {e}")
            messagebox.showerror("Error", f"Error exporting best homography matrix: {e}")
            return
        self.log(f"Best homography matrix exported to: {output_path}")
        messagebox.showinfo("Success", "Best homography matrix accepted and exported successfully.")


