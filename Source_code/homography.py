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

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")


# --- StdoutRedirector class for redirecting console output into the GUI ---
class StdoutRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        self.text_widget.insert(tk.END, message)
        self.text_widget.see(tk.END)

    def flush(self):
        pass


def resource_path(relative_path: str) -> str:
    """Return absolute path to resource, compatible with PyInstaller."""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.dirname(__file__)
    return os.path.join(base_path, relative_path)


class CreateHomographyMatrixWindow(ctk.CTkToplevel):
    def __init__(self, master=None, *args, **kwargs):
        super().__init__(master=master, *args, **kwargs)
        self.title("Create Homography Matrix")
        self.geometry("850x800")

        try:
            self.iconbitmap(resource_path("launch_logo.ico"))
        except Exception as e:
            print("Warning: Could not load window icon:", e)

        # ------------  state variables  ------------
        self.input_file = None
        self.output_folder = None
        self.H_final = None
        self.pixel_points = None
        self.utm_points = None
        self.gcp_ids = None
        self.best_H_annealing = None
        self.best_cost = None

        # -------------------------------  Panel 1  -------------------------------
        self.panel1 = ctk.CTkFrame(self)
        self.panel1.pack(padx=10, pady=10, fill="x")

        self.label_input = ctk.CTkLabel(self.panel1, text="Input GCP File (.csv):")
        self.label_input.pack(side="left", padx=5, pady=5)

        self.btn_browse_input = ctk.CTkButton(self.panel1, text="Browse", command=self.browse_input)
        self.btn_browse_input.pack(side="left", padx=5, pady=5)

        self.input_file_label = ctk.CTkLabel(self.panel1, text="No file selected", fg_color="transparent")
        self.input_file_label.pack(side="left", padx=5, pady=5)

        self.label_note = ctk.CTkLabel(
            self.panel1,
            text="Note: File must include columns GCP_ID, Pixel_X, Pixel_Y, Real_X, Real_Y",
            fg_color="white",
            text_color="black",
            corner_radius=0
        )
        self.label_note.pack(side="left", padx=10, pady=5)

        # -------------------------------  Panel 2  -------------------------------
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

        # -------------------------------  Panel Exclude  -------------------------------
        self.panel_exclude = ctk.CTkFrame(self)
        self.panel_exclude.pack(padx=10, pady=10, fill="x")

        self.exclude_var = tk.BooleanVar(value=False)
        self.chk_exclude = ctk.CTkCheckBox(self.panel_exclude, text="Exclude GCPs", variable=self.exclude_var)
        self.chk_exclude.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.label_exclude = ctk.CTkLabel(
            self.panel_exclude,
            text="(List of numeric parts, e.g. 6,7,12 — matches GCP_6, GCP_7, ...)",
        )
        self.label_exclude.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        self.entry_exclude = ctk.CTkEntry(self.panel_exclude)
        self.entry_exclude.grid(row=0, column=2, padx=5, pady=5, sticky="we")
        self.panel_exclude.columnconfigure(2, weight=1)

        # -------------------------------  Panel 3  -------------------------------
        self.panel3 = ctk.CTkFrame(self)
        self.panel3.pack(padx=10, pady=10, fill="x")

        self.btn_compute = ctk.CTkButton(self.panel3, text="Compute Matrix", command=self.compute_matrix)
        self.btn_compute.grid(row=0, column=0, padx=5, pady=5)

        self.btn_compute_accuracy = ctk.CTkButton(self.panel3, text="Compute Accuracy", command=self.compute_accuracy)
        self.btn_compute_accuracy.grid(row=0, column=1, padx=5, pady=5)

        # -------------------------------  Panel 4 (Advanced)  -------------------------------
        self.panel_adv = ctk.CTkFrame(self)
        self.panel_adv.pack(padx=10, pady=10, fill="x")

        self.advanced_var = tk.BooleanVar(value=False)
        self.chk_advanced = ctk.CTkCheckBox(
            self.panel_adv, text="Advanced", variable=self.advanced_var, command=self.toggle_advanced
        )
        self.chk_advanced.pack(side="left", padx=5, pady=5)

        self.lbl_adv_info = ctk.CTkLabel(self.panel_adv, text="", fg_color="transparent")
        self.lbl_adv_info.pack(side="left", padx=5, pady=5)

        self.advanced_frame = ctk.CTkFrame(self)  # hidden until checkbox ticked

        self.adv_inputs = {}
        adv_options = [
            ("Number of GCPs", "90"),
            ("Max iterations", "300000"),
            ("Initial temperature", "100000.0"),
            ("Cooling rate", "0.99995"),
            ("Number of swaps", "1"),
            ("RANSAC threshold", "0.7"),
        ]
        for idx, (label, default) in enumerate(adv_options):
            lbl = ctk.CTkLabel(self.advanced_frame, text=f"{label}:")
            lbl.grid(row=idx, column=0, padx=5, pady=3, sticky="e")
            ent = ctk.CTkEntry(self.advanced_frame)
            ent.insert(0, default)
            ent.grid(row=idx, column=1, padx=5, pady=3, sticky="we")
            self.advanced_frame.columnconfigure(1, weight=1)
            self.adv_inputs[label] = ent

        self.adv_buttons_frame = ctk.CTkFrame(self.advanced_frame)
        self.adv_buttons_frame.grid(row=len(adv_options), column=0, columnspan=2, pady=10)

        self.btn_sa_search = ctk.CTkButton(
            self.adv_buttons_frame, text="Simulated Annealing Search", command=self.run_simulated_annealing
        )
        self.btn_sa_search.grid(row=0, column=0, padx=5, pady=5)

        self.btn_accept_export = ctk.CTkButton(
            self.adv_buttons_frame, text="Accept and Export New Matrix", command=self.accept_and_export_new_matrix
        )
        self.btn_accept_export.grid(row=0, column=1, padx=5, pady=5)

        # -------------------------------  Panel 5 (Console)  -------------------------------
        self.panel5 = ctk.CTkFrame(self)
        self.panel5.pack(padx=10, pady=10, fill="both", expand=True)

        self.text_console = tk.Text(self.panel5, wrap="word", height=10)
        self.text_console.pack(padx=5, pady=5, fill="both", expand=True)

        sys.stdout = StdoutRedirector(self.text_console)
        sys.stderr = sys.stdout
        print("Here you may see console outputs\n")

    # --------------------------------------------------------------------------
    #                               Helper UI
    # --------------------------------------------------------------------------
    def log(self, msg: str):
        self.text_console.insert(tk.END, msg + "\n")
        self.text_console.see(tk.END)

    def thread_log(self, msg: str):
        self.after(0, lambda: self.log(msg))

    def browse_input(self):
        file_path = filedialog.askopenfilename(
            title="Select GCP CSV File",
            filetypes=(("CSV Files", "*.csv"), ("All Files", "*.*")),
        )
        if file_path:
            self.input_file = file_path
            self.input_file_label.configure(text=os.path.basename(file_path))
            self.log(f"Selected input file: {file_path}")

    def browse_output(self):
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_folder = folder
            self.output_folder_label.configure(text=folder)
            self.log(f"Selected output folder: {folder}")

    def toggle_advanced(self):
        if self.advanced_var.get():
            self.lbl_adv_info.configure(text="Using simulated annealing to identify best subset of GCPs")
            self.advanced_frame.pack(padx=10, pady=5, fill="x")
        else:
            self.lbl_adv_info.configure(text="")
            self.advanced_frame.forget()

    # --------------------------------------------------------------------------
    #                             Math helpers
    # --------------------------------------------------------------------------
    @staticmethod
    def normalize_points(points: np.ndarray):
        centroid = points.mean(axis=0)
        shifted = points - centroid
        dists = np.sqrt((shifted**2).sum(axis=1))
        scale = np.sqrt(2) / dists.mean()
        T = np.array([[scale, 0, -scale * centroid[0]], [0, scale, -scale * centroid[1]], [0, 0, 1]])
        ones = np.ones((points.shape[0], 1))
        pts_h = np.hstack([points, ones])
        norm_h = (T @ pts_h.T).T
        return norm_h, T

    # --------------------------------------------------------------------------
    #                              Core functions
    # --------------------------------------------------------------------------
    def compute_matrix(self):
        """Load CSV, optionally exclude by GCP_ID, compute homography, save matrix."""
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

        required = ["GCP_ID", "Pixel_X", "Pixel_Y", "Real_X", "Real_Y"]
        for col in required:
            if col not in gcp_data.columns:
                self.log(f"Missing required column: {col}")
                messagebox.showerror("Error", f"Missing required column: {col}")
                return

        # -------------- exclude by numeric parts ----------------
        if self.exclude_var.get():
            excl_str = self.entry_exclude.get().strip()
            if excl_str:
                try:
                    nums = [int(x.strip()) for x in excl_str.split(",") if x.strip()]
                    excl_ids = [f"GCP_{n}" for n in nums]
                    self.log(f"Excluding GCP IDs: {excl_ids}")
                    gcp_data = gcp_data[~gcp_data["GCP_ID"].isin(excl_ids)].reset_index(drop=True)
                except Exception as e:
                    self.log(f"Error parsing excluded GCPs: {e}")

        # -------------- store arrays ----------------
        self.gcp_ids = gcp_data["GCP_ID"].values
        self.pixel_points = gcp_data[["Pixel_X", "Pixel_Y"]].values.astype(np.float32)
        self.utm_points = gcp_data[["Real_X", "Real_Y"]].values.astype(np.float32)

        if len(self.pixel_points) < 4:
            self.log("Not enough GCPs! Need at least 4.")
            messagebox.showerror("Error", "Not enough GCPs! Need at least 4.")
            return

        # -------------- normalize & RANSAC ----------------
        self.log("Normalizing points…")
        p_norm, T_p = self.normalize_points(self.pixel_points)
        u_norm, T_u = self.normalize_points(self.utm_points)
        H_norm, _ = cv2.findHomography(p_norm[:, :2], u_norm[:, :2], cv2.RANSAC, 0.5)
        if H_norm is None:
            self.log("Failed to compute homography matrix.")
            messagebox.showerror("Error", "Failed to compute homography matrix. Check input data.")
            return

        H_final = np.linalg.inv(T_u) @ H_norm @ T_p
        if abs(H_final[2, 2]) > 1e-6 and abs(H_final[2, 2] - 1) > 1e-3:
            H_final /= H_final[2, 2]
        self.H_final = H_final

        # -------------- save ----------------
        out_path = os.path.join(self.output_folder, f"{output_name}.txt")
        try:
            np.savetxt(out_path, H_final)
        except Exception as e:
            self.log(f"Error saving homography matrix: {e}")
            messagebox.showerror("Error", f"Error saving homography matrix: {e}")
            return

        self.log("\nHomography Matrix Computed and Saved!")
        self.log(f"Output Path: {out_path}")
        self.log(f"Homography matrix:\n{H_final}")
        messagebox.showinfo("Success", "Homography matrix computed and saved successfully.")

    def compute_accuracy(self):
        """Apply H to each pixel point and report per-GCP error."""
        if self.H_final is None or self.pixel_points is None:
            messagebox.showerror("Error", "Please compute the homography matrix first!")
            return

        self.log("Computing accuracy…")
        pts = self.pixel_points.reshape(-1, 1, 2)
        utm_est = cv2.perspectiveTransform(pts, self.H_final).reshape(-1, 2)
        errors = np.linalg.norm(utm_est - self.utm_points, axis=1)

        ones = np.ones((self.pixel_points.shape[0], 1), np.float32)
        x_hom = np.hstack([self.pixel_points, ones])
        utm_est2 = (self.H_final @ x_hom.T).T
        utm_est2 = utm_est2[:, :2] / utm_est2[:, 2][:, None]
        max_diff = np.abs(utm_est - utm_est2).max()

        report = f"Max difference CV vs. manual: {max_diff:.3f}\n"
        for gid, err in zip(self.gcp_ids, errors):
            report += f"{gid}: Error = {err:.3f} m\n"
        report += f"Mean error: {errors.mean():.3f} m\n"
        self.log(report)

    # --------------------------------------------------------------------------
    #             Simulated-annealing helpers 
    # --------------------------------------------------------------------------
    def compute_homography_and_errors(
        self, pixel_points, utm_points, subset_indices, ransac_thresh=0.5
    ):
        """Return error stats + H for a subset of rows."""
        subset_px = pixel_points[subset_indices]
        subset_utm = utm_points[subset_indices]
        if len(subset_indices) < 4:
            return 9999, 9999, 9999, 9999, 0, 9999, None
        try:
            px_norm, T_px = self.normalize_points(subset_px)
            utm_norm, T_u = self.normalize_points(subset_utm)
            H_norm, _ = cv2.findHomography(px_norm[:, :2], utm_norm[:, :2], cv2.RANSAC, ransac_thresh)
            if H_norm is None:
                return 9999, 9999, 9999, 9999, 0, 9999, None
            H_fin = np.linalg.inv(T_u) @ H_norm @ T_px
            if abs(H_fin[2, 2]) > 1e-6 and abs(H_fin[2, 2] - 1) > 1e-3:
                H_fin /= H_fin[2, 2]
            ones = np.ones((subset_px.shape[0], 1), np.float32)
            est = (H_fin @ np.hstack([subset_px, ones]).T).T
            est = est[:, :2] / est[:, 2][:, None]
            errs = np.linalg.norm(est - subset_utm, axis=1)
            return (
                errs.mean(),
                np.median(errs),
                errs.std(),
                errs.max(),
                (errs < 10).mean(),
                (errs > 5).sum(),
                H_fin,
            )
        except Exception:
            return 9999, 9999, 9999, 9999, 0, 9999, None

    def compute_full_errors(self, H, pixel_points, utm_points):
        """Error metrics on *all* GCPs for candidate H."""
        ones = np.ones((pixel_points.shape[0], 1), np.float32)
        utm_est = (H @ np.hstack([pixel_points, ones]).T).T
        utm_est = utm_est[:, :2] / utm_est[:, 2][:, None]
        errs = np.linalg.norm(utm_est - utm_points, axis=1)
        return errs.mean(), np.median(errs), errs.std(), errs.max(), (errs < 10).mean(), (errs > 5).sum()

    # --------------------------------------------------------------------------
    #                        Simulated-annealing main loop
    # --------------------------------------------------------------------------
    def run_simulated_annealing(self):
        threading.Thread(target=self.run_simulated_annealing_thread, daemon=True).start()

    def run_simulated_annealing_thread(self):
        if self.pixel_points is None or self.utm_points is None:
            self.thread_log("Error: compute the homography matrix first.")
            return

        try:
            n_subset = int(self.adv_inputs["Number of GCPs"].get())
            max_iter = int(self.adv_inputs["Max iterations"].get())
            temp = float(self.adv_inputs["Initial temperature"].get())
            cooling = float(self.adv_inputs["Cooling rate"].get())
            swaps = int(self.adv_inputs["Number of swaps"].get())
            r_thresh = float(self.adv_inputs["RANSAC threshold"].get())
        except Exception as e:
            self.thread_log(f"Invalid advanced configuration: {e}")
            return

        n_total = len(self.pixel_points)
        if n_subset > n_total:
            self.thread_log("Error: subset size exceeds total GCPs.")
            return
        idx_all = np.arange(n_total)

        def cost(m, med, sd, mx, inl, big):
            pen = (mx > 10) * 15000 + (1 - inl) * 8000
            return m + 2 * sd + pen

        subset = np.random.choice(idx_all, n_subset, replace=False)
        _, _, _, _, _, _, H = self.compute_homography_and_errors(
            self.pixel_points, self.utm_points, subset, r_thresh
        )
        best_cost = cost(*self.compute_full_errors(H, self.pixel_points, self.utm_points))
        best_subset, best_H = subset.copy(), H

        self.thread_log("Starting simulated annealing …")
        t0 = time.time()
        for it in range(max_iter):
            new_subset = best_subset.copy()
            for _ in range(swaps):
                out = random.choice(new_subset)
                inp = random.choice(np.setdiff1d(idx_all, new_subset))
                new_subset[np.where(new_subset == out)[0][0]] = inp
            _, _, _, _, _, _, H2 = self.compute_homography_and_errors(
                self.pixel_points, self.utm_points, new_subset, r_thresh
            )
            new_cost = cost(*self.compute_full_errors(H2, self.pixel_points, self.utm_points))
            dc = new_cost - best_cost
            if dc < 0 or random.random() < np.exp(-dc / temp):
                best_subset, best_cost, best_H = new_subset.copy(), new_cost, H2
            temp *= cooling
            if (it + 1) % 50000 == 0:
                self.thread_log(
                    f"Iter {it+1}/{max_iter} | Temp={temp:.4f} | BestCost={best_cost:.2f}"
                )

        self.best_H_annealing = best_H
        elapsed = time.time() - t0
        m, med, sd, mx, inl, big = self.compute_full_errors(
            best_H, self.pixel_points, self.utm_points
        )
        self.thread_log(
            f"SA finished in {elapsed:.2f}s | cost={best_cost:.2f}\n"
            f"Mean={m:.2f} m, Median={med:.2f} m, σ={sd:.2f} m, Max={mx:.2f} m, "
            f"Inlier={inl*100:.1f} %, >5m={big}"
        )

    # --------------------------------------------------------------------------
    #                           Export SA result
    # --------------------------------------------------------------------------
    def accept_and_export_new_matrix(self):
        if self.best_H_annealing is None:
            messagebox.showerror("Error", "Run SA search first.")
            return
        out_name = self.entry_output_name.get().strip()
        if not out_name:
            messagebox.showerror("Error", "Please enter an output file name.")
            return
        path = os.path.join(self.output_folder, f"{out_name}_bestsubset.txt")
        try:
            np.savetxt(path, self.best_H_annealing)
        except Exception as e:
            self.log(f"Error exporting: {e}")
            messagebox.showerror("Error", f"Error exporting best matrix: {e}")
            return
        self.log(f"Best homography matrix exported to: {path}")
        messagebox.showinfo("Success", "Best homography matrix exported successfully.")


# --------------------------------------------------------------------------
#                                     Main
# --------------------------------------------------------------------------
def main():
    root = ctk.CTk()
    root.withdraw()
    win = CreateHomographyMatrixWindow(master=root)
    win.mainloop()


if __name__ == "__main__":
    main()
