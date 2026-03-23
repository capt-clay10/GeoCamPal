"""
Harmonise Images

Two sub‑tasks in one window:
  1. **Filter bad images** — robust blur, clipping‑based over‑exposure,
     under‑exposure, darkness, low‑information (entropy), rain/droplets
     on lens, and partial obstruction detection.
  2. **Harmonise histograms** — luminance‑based gain (with soft‑knee) or
     histogram matching, optional sky masking for reference computation,
     and post‑correction sanity check.

Outputs:
  • bad_images.txt            — list of identified bad image filenames
  • _harmonised/  subfolder   — corrected images with original structure
"""

# %% ————————————————————————————— imports —————————————————————————————
import sys
import os
import glob
import shutil
import threading
from pathlib import Path

import numpy as np
import cv2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox

try:
    from skimage import exposure as sk_exposure
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# %% ————————————————————————————— util helpers ————————————————————————
def resource_path(relative_path: str) -> str:
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.dirname(__file__)
    return os.path.join(base_path, relative_path)


class StdoutRedirector:
    def __init__(self, text_widget: tk.Text):
        self.text_widget = text_widget

    def write(self, message: str):
        self.text_widget.insert(tk.END, message)
        self.text_widget.see(tk.END)

    def flush(self):
        pass


# %% ————————————————————————————— image quality filters ———————————————

def is_blurry(img_bgr, lap_threshold, uniform_fraction=0.85):
    """
    Robust blur detector.
    Uses Laplacian variance, but guards against false positives on
    genuinely textureless scenes (calm water, overcast sky) by checking
    whether the image is mostly uniform *before* flagging it as blurry.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var < lap_threshold:
        # check if the scene is genuinely textureless (not blurry)
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
        low_gradient_frac = np.sum(np.abs(sobel) < 5) / sobel.size
        if low_gradient_frac > uniform_fraction:
            return False  # uniform scene, not actually blurry
        return True
    return False


def is_clipped_overexposed(img_bgr, clip_fraction=0.15):
    """
    Saturation‑based overexposure: checks for pixels clipped at 254–255
    across ALL three channels (no recoverable information), rather than
    just "bright" single‑channel pixels.  This avoids false positives
    on naturally bright scenes like white sand.
    """
    saturated = np.all(img_bgr >= 254, axis=2)
    return (saturated.sum() / saturated.size) > clip_fraction


def is_underexposed(img_bgr, dark_px_value, fraction):
    """Too many dark pixels (single‑channel check on grayscale)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    n_dark = np.sum(gray <= dark_px_value)
    return (n_dark / gray.size) > fraction


def is_too_dark(img_bgr, max_threshold):
    """Entire image maximum below threshold → near-black / nighttime."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return int(gray.max()) < max_threshold


def is_low_information(img_bgr, entropy_threshold=5.0):
    """
    Entropy‑based low‑information detector.  Catches fog, haze,
    washed‑out images, and lens condensation better than simple std.
    Typical good outdoor images have entropy 6–7.5; fog/haze drops
    below 5.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    return entropy < entropy_threshold


def has_lens_droplets(img_bgr, blob_fraction=0.08):
    """
    Rain / condensation droplet detector.
    Droplets create localised bright blobs surrounded by blur halos.
    We detect them as bright residuals after heavy Gaussian smoothing.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (31, 31), 0)
    diff = cv2.absdiff(gray, blurred)
    # droplets produce concentrated bright spots in the diff image
    bright_local = np.sum(diff > 40) / diff.size
    return bright_local > blob_fraction


def has_obstruction(img_bgr, grid_size=4, dark_mean_thresh=15,
                    block_fraction=0.35):
    """
    Partial obstruction detector (bird on housing, lens cap edge,
    foreign object).  Divides the image into a grid and checks whether
    a large contiguous region is near‑black.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    bh, bw = h // grid_size, w // grid_size
    dark_blocks = 0
    total_blocks = grid_size * grid_size
    for i in range(grid_size):
        for j in range(grid_size):
            block = gray[i * bh:(i + 1) * bh, j * bw:(j + 1) * bw]
            if block.mean() < dark_mean_thresh:
                dark_blocks += 1
    return (dark_blocks / total_blocks) > block_fraction


def filter_image(img_bgr,
                 blur_thresh,
                 clip_frac,
                 underexp_val, underexp_frac,
                 dark_thresh,
                 entropy_thresh,
                 droplet_frac,
                 obstruction_frac):
    """
    Run all filters.  Returns a list of reason strings; empty = passes.
    """
    reasons = []
    if is_blurry(img_bgr, blur_thresh):
        reasons.append("blurry")
    if is_clipped_overexposed(img_bgr, clip_frac):
        reasons.append("clipped_overexposed")
    if is_underexposed(img_bgr, underexp_val, underexp_frac):
        reasons.append("underexposed")
    if is_too_dark(img_bgr, dark_thresh):
        reasons.append("too_dark")
    if is_low_information(img_bgr, entropy_thresh):
        reasons.append("low_information")
    if has_lens_droplets(img_bgr, droplet_frac):
        reasons.append("lens_droplets")
    if has_obstruction(img_bgr, block_fraction=obstruction_frac):
        reasons.append("obstruction")
    return reasons


# %% ————————————————————————————— harmonisation helpers ————————————————

def mean_luminance(img_bgr, sky_mask_frac=0.0):
    """
    Mean L‑channel value (CIE LAB).
    If sky_mask_frac > 0, the top N% of the image is excluded so that
    bright skies don't skew the reference luminance.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L = lab[..., 0]
    if sky_mask_frac > 0:
        h = L.shape[0]
        cut = int(h * sky_mask_frac)
        if cut > 0:
            L = L[cut:, :]
    return float(L.mean())


def soft_gain_match(img_bgr, ref_mean_L, knee=200):
    """
    Luminance gain with soft‑knee rolloff above *knee* to avoid hard
    highlight clipping.  Pixels below the knee are scaled linearly;
    above the knee the gain tapers off smoothly toward 255.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0]
    current_mean = max(L.mean(), 1e-6)
    gain = ref_mean_L / current_mean

    result = L * gain
    # soft rolloff above knee
    above = result > knee
    if above.any():
        excess = result[above] - knee
        headroom = 255.0 - knee
        result[above] = knee + headroom * excess / (excess + headroom)

    lab[..., 0] = np.clip(result, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def hist_match(img_bgr, ref_bgr):
    """
    Full L‑channel histogram matching using scikit‑image.
    Falls back to soft_gain if skimage is not available.
    """
    if not HAS_SKIMAGE:
        return soft_gain_match(img_bgr, mean_luminance(ref_bgr))
    ref_lab = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB)
    tgt_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    matched_L = sk_exposure.match_histograms(
        tgt_lab[..., 0], ref_lab[..., 0], channel_axis=None)
    tgt_lab[..., 0] = np.clip(matched_L, 0, 255).astype(np.uint8)
    return cv2.cvtColor(tgt_lab, cv2.COLOR_LAB2BGR)


def correction_sanity_check(original_bgr, corrected_bgr, ref_mean_L,
                            sky_mask_frac=0.0):
    """
    Verify correction actually improved things.  If the corrected image's
    mean L is *further* from the reference than the original, something
    went wrong (can happen with histogram matching on unusual distributions).
    Returns (image, reverted_bool).
    """
    orig_L = mean_luminance(original_bgr, sky_mask_frac)
    corr_L = mean_luminance(corrected_bgr, sky_mask_frac)
    if abs(corr_L - ref_mean_L) > abs(orig_L - ref_mean_L):
        return original_bgr, True  # reverted
    return corrected_bgr, False


# %% ————————————————————————————— file helpers ————————————————————————

def collect_images(folder):
    """Collect image paths from a folder (non-recursive)."""
    return sorted(
        p for p in Path(folder).iterdir()
        if p.suffix.lower() in IMAGE_EXTS
    )


def collect_images_recursive(folder):
    """Collect image paths from a folder and all sub-folders."""
    results = []
    for root, _, files in os.walk(folder):
        for f in sorted(files):
            if Path(f).suffix.lower() in IMAGE_EXTS:
                results.append(Path(root) / f)
    return results


# %% ————————————————————————————— main GUI ————————————————————————————
class HarmoniseImagesWindow(ctk.CTkToplevel):
    """Filter bad images & harmonise histograms."""

    def __init__(self, master=None, **kw):
        super().__init__(master=master, **kw)
        self.title("Harmonise Images")
        self.geometry("1350x900")
        try:
            self.iconbitmap(resource_path("launch_logo.ico"))
        except Exception:
            pass

        # ——— state ———
        self.input_folder = None
        self.output_folder = None
        self.bad_list = []
        self.recursive_var = tk.BooleanVar(value=False)

        # ——— layout ———
        self.grid_rowconfigure(0, weight=3)
        self.grid_rowconfigure(1, weight=0)
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # ---- TOP: preview ----
        self.top_panel = ctk.CTkFrame(self)
        self.top_panel.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.fig, self.axes = plt.subplots(1, 3, figsize=(14, 4))
        self.axes[0].set_title("Luminance Distribution")
        self.axes[1].set_title("Filter Results")
        self.axes[1].axis("off")
        self.axes[2].set_title("Harmonisation Results")
        self.axes[2].axis("off")
        self.fig.tight_layout()
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=self.top_panel)
        self.canvas_plot.get_tk_widget().pack(fill="both", expand=True)

        # ---- BOTTOM: controls ----
        self.bottom_panel = ctk.CTkFrame(self)
        self.bottom_panel.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        # Row 1 — input folder
        row1 = ctk.CTkFrame(self.bottom_panel)
        row1.pack(fill="x", padx=5, pady=2)

        ctk.CTkButton(row1, text="Browse Image Folder",
                      command=self._browse_input).grid(
            row=0, column=0, padx=5, pady=5)
        self.input_label = ctk.CTkLabel(row1, text="No folder selected")
        self.input_label.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        ctk.CTkCheckBox(row1, text="Include sub-folders",
                        variable=self.recursive_var).grid(
            row=0, column=2, padx=10, pady=5)

        # Row 2 — filter parameters (two sub-rows for readability)
        row2 = ctk.CTkFrame(self.bottom_panel)
        row2.pack(fill="x", padx=5, pady=2)

        ctk.CTkLabel(row2, text="Filter Bad Images",
                     font=("Arial", 12, "bold")).grid(
            row=0, column=0, padx=5, pady=3, sticky="w", columnspan=2)

        # Each filter gets one checkbox + its parameter entries
        # Row A: 4 filters
        self.filter_entries = {}
        self.filter_enabled = {}

        filters_a = [
            ("Blur", "Blur threshold", "20"),
            ("Overexposure", "Clip overexp fraction", "0.15"),
            ("Underexposure", "Dark pixel value", "20",
                              "Under-exp fraction", "0.25"),
            ("Darkness", "Dark max threshold", "25"),
        ]
        col = 0
        for fdef in filters_a:
            name = fdef[0]
            var = tk.BooleanVar(value=True)
            ctk.CTkCheckBox(row2, text=name, variable=var,
                            width=20).grid(row=1, column=col, padx=2, pady=2)
            self.filter_enabled[name] = var
            col += 1
            # parameter pairs: (label, default), (label, default), ...
            params = fdef[1:]
            for i in range(0, len(params), 2):
                plbl, pdef = params[i], params[i + 1]
                e = ctk.CTkEntry(row2, width=50,
                                 placeholder_text=plbl)
                e.insert(0, pdef)
                e.grid(row=1, column=col, padx=1, pady=2)
                self.filter_entries[plbl] = e
                col += 1
        max_col = col  # remember widest row

        # Row B: 3 filters
        filters_b = [
            ("Low information", "Entropy threshold", "5.0"),
            ("Lens droplets", "Droplet fraction", "0.08"),
            ("Obstruction", "Obstruction fraction", "0.35"),
        ]
        col = 0
        for fdef in filters_b:
            name = fdef[0]
            var = tk.BooleanVar(value=True)
            ctk.CTkCheckBox(row2, text=name, variable=var,
                            width=20).grid(row=2, column=col, padx=2, pady=2)
            self.filter_enabled[name] = var
            col += 1
            params = fdef[1:]
            for i in range(0, len(params), 2):
                plbl, pdef = params[i], params[i + 1]
                e = ctk.CTkEntry(row2, width=50,
                                 placeholder_text=plbl)
                e.insert(0, pdef)
                e.grid(row=2, column=col, padx=1, pady=2)
                self.filter_entries[plbl] = e
                col += 1
        max_col = max(max_col, col)

        ctk.CTkButton(row2, text="Run Filter",
                      command=self._filter_threaded, fg_color="#0F52BA").grid(
            row=1, column=max_col, padx=10, pady=3, rowspan=2, sticky="ns")

        ctk.CTkLabel(
            row2, text="→ Flags blurry, overexposed, dark,\n"
                       "   foggy, rain-on-lens & blocked images",
            font=("Arial", 10), text_color="gray", justify="left",
        ).grid(row=1, column=max_col + 1, rowspan=2, padx=5, pady=3,
               sticky="w")

        # Row 3 — harmonise parameters
        row3 = ctk.CTkFrame(self.bottom_panel)
        row3.pack(fill="x", padx=5, pady=2)

        ctk.CTkLabel(row3, text="Harmonise Histograms",
                     font=("Arial", 12, "bold")).grid(
            row=0, column=0, padx=5, pady=3, sticky="w")

        ctk.CTkLabel(row3, text="L-tolerance (±)").grid(
            row=0, column=1, padx=3, pady=3)
        self.tol_entry = ctk.CTkEntry(row3, width=55)
        self.tol_entry.insert(0, "5")
        self.tol_entry.grid(row=0, column=2, padx=3, pady=3)

        ctk.CTkLabel(row3, text="Sky mask (%)").grid(
            row=0, column=3, padx=3, pady=3)
        self.sky_entry = ctk.CTkEntry(row3, width=55)
        self.sky_entry.insert(0, "0")
        self.sky_entry.grid(row=0, column=4, padx=3, pady=3)

        self.exclude_bad_var = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(row3, text="Exclude filtered bad images",
                        variable=self.exclude_bad_var).grid(
            row=0, column=5, padx=10, pady=3)

        ctk.CTkButton(row3, text="Run Harmonise",
                      command=self._harmonise_threaded, fg_color="#0F52BA").grid(
            row=0, column=6, padx=10, pady=3)

        ctk.CTkLabel(
            row3, text="→ Matches brightness across all images",
            font=("Arial", 10), text_color="gray",
        ).grid(row=0, column=7, padx=5, pady=3, sticky="w")

        # Row 4 — output & reset
        row4 = ctk.CTkFrame(self.bottom_panel)
        row4.pack(fill="x", padx=5, pady=2)

        ctk.CTkButton(row4, text="Browse Output Folder",
                      command=self._browse_output).grid(
            row=0, column=0, padx=5, pady=5)
        self.output_label = ctk.CTkLabel(row4, text="No output folder selected")
        self.output_label.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        self.progress_bar = ctk.CTkProgressBar(row4, width=200)
        self.progress_bar.grid(row=0, column=2, padx=10, pady=5)
        self.progress_bar.set(0)

        self.btn_reset = ctk.CTkButton(
            row4, text="Reset", command=self._reset,
            width=80, fg_color="#8B0000", hover_color="#A52A2A")
        self.btn_reset.grid(row=0, column=3, padx=5, pady=5, sticky="e")

        # ---- CONSOLE ----
        self.console_frame = ctk.CTkFrame(self)
        self.console_frame.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        self.console_text = tk.Text(self.console_frame, wrap="word", height=8)
        self.console_text.pack(fill="both", expand=True, padx=5, pady=5)
        self.stdout_redirector = StdoutRedirector(self.console_text)
        sys.stdout = self.stdout_redirector
        sys.stderr = self.stdout_redirector
        print("Harmonise Images — Tool Guide\n"
              "================================\n"
              "\n"
              "DISPLAY PANEL (top)\n"
              "  Left plot:   Luminance distribution of all images (histogram).\n"
              "  Centre plot: Number of images flagged by each filter.\n"
              "  Right plot:  Harmonisation action breakdown (copy / gain / histogram-match).\n"
              "\n"
              "FILTER BAD IMAGES  (uncheck a filter to disable it)\n"
              "  ☑ Blur            — Laplacian variance; images below threshold are blurry.\n"
              "                      Guards against false positives on calm water / clear sky.\n"
              "  ☑ Overexposure    — Fraction of pixels saturated at 254-255 in ALL channels.\n"
              "                      Detects blown highlights (not just bright sand).\n"
              "  ☑ Underexposure   — Two parameters: dark pixel value + fraction.\n"
              "                      If more than 'fraction' of pixels are below 'value' → bad.\n"
              "  ☑ Darkness        — If the brightest pixel in the image is below threshold\n"
              "                      → nighttime / lens cap on.\n"
              "  ☑ Low information — Histogram entropy; low values = fog, haze, condensation.\n"
              "                      Good outdoor images: 6-7.5; fog: below 5.\n"
              "  ☑ Lens droplets   — Detects rain or condensation droplets on the lens\n"
              "                      (bright blobs after Gaussian subtraction).\n"
              "  ☑ Obstruction     — Detects partial blockage (bird, lens cap, foreign object)\n"
              "                      by checking for large dark regions in a 4×4 grid.\n"
              "\n"
              "HARMONISE HISTOGRAMS\n"
              "  L-tolerance (±)     — Images within this luminance range of the median are\n"
              "                        kept unchanged. Outside this range:\n"
              "                          • Very dark (ΔL ≤ −2×tol) → histogram matching\n"
              "                          • Very bright (ΔL ≥ +2×tol) → soft-gain correction\n"
              "                          • Moderate deviation → soft-gain correction\n"
              "                        A sanity check reverts if correction makes things worse.\n"
              "  Sky mask (%)        — Ignore the top N% of the image when computing mean\n"
              "                        brightness. Useful when bright sky dominates and skews\n"
              "                        the reference luminance away from the foreground.\n"
              "================================\n")

    # ——— browse callbacks ———

    def _browse_input(self):
        d = filedialog.askdirectory(title="Select Image Folder")
        if d:
            self.input_folder = d
            self.input_label.configure(text=d)

    def _browse_output(self):
        d = filedialog.askdirectory(title="Select Output Folder")
        if d:
            self.output_folder = d
            self.output_label.configure(text=d)

    # ——— reset ———

    def _reset(self):
        self.input_folder = None
        self.output_folder = None
        self.bad_list = []
        self.input_label.configure(text="No folder selected")
        self.output_label.configure(text="No output folder selected")
        self.progress_bar.set(0)
        self.recursive_var.set(False)
        self.exclude_bad_var.set(True)
        for var in self.filter_enabled.values():
            var.set(True)

        for ax in self.axes:
            ax.clear()
            ax.axis("off")
        self.axes[0].set_title("Luminance Distribution")
        self.axes[1].set_title("Filter Results")
        self.axes[2].set_title("Harmonisation Results")
        self.fig.tight_layout()
        self.canvas_plot.draw()
        self.console_text.delete("1.0", tk.END)
        print("Session reset.\n--------------------------------")

    # ——— filtering ———

    def _filter_threaded(self):
        threading.Thread(target=self._run_filter, daemon=True).start()

    def _run_filter(self):
        try:
            self.progress_bar.set(0)

            if not self.input_folder:
                messagebox.showwarning("Warning",
                                       "Select an image folder first.")
                return
            if not self.output_folder:
                messagebox.showwarning("Warning",
                                       "Select an output folder first.")
                return

            blur_t = float(self.filter_entries["Blur threshold"].get())
            clip_f = float(self.filter_entries["Clip overexp fraction"].get())
            dark_v = int(self.filter_entries["Dark pixel value"].get())
            under_f = float(self.filter_entries["Under-exp fraction"].get())
            dark_max = int(self.filter_entries["Dark max threshold"].get())
            entropy_t = float(self.filter_entries["Entropy threshold"].get())
            droplet_f = float(self.filter_entries["Droplet fraction"].get())
            obstruct_f = float(
                self.filter_entries["Obstruction fraction"].get())

            # which filters are enabled
            en = {k: v.get() for k, v in self.filter_enabled.items()}

            if self.recursive_var.get():
                images = collect_images_recursive(self.input_folder)
            else:
                images = collect_images(self.input_folder)

            if not images:
                messagebox.showwarning("Warning", "No images found.")
                return

            active = [k for k, v in en.items() if v]
            print(f"\nFiltering {len(images)} images with "
                  f"{len(active)} active filter(s): "
                  f"{', '.join(active)} …")
            self.bad_list = []
            bad_reasons = {}
            reason_counts = {
                "blurry": 0, "clipped_overexposed": 0,
                "underexposed": 0, "too_dark": 0,
                "low_information": 0, "lens_droplets": 0,
                "obstruction": 0,
            }

            for idx, p in enumerate(images):
                img = cv2.imread(str(p))
                if img is None:
                    print(f"[WARN] Cannot read: {p.name}")
                    continue

                # run only enabled filters
                reasons = []
                if en.get("Blur") and is_blurry(img, blur_t):
                    reasons.append("blurry")
                if en.get("Overexposure") and is_clipped_overexposed(img, clip_f):
                    reasons.append("clipped_overexposed")
                if en.get("Underexposure") and is_underexposed(img, dark_v, under_f):
                    reasons.append("underexposed")
                if en.get("Darkness") and is_too_dark(img, dark_max):
                    reasons.append("too_dark")
                if en.get("Low information") and is_low_information(img, entropy_t):
                    reasons.append("low_information")
                if en.get("Lens droplets") and has_lens_droplets(img, droplet_f):
                    reasons.append("lens_droplets")
                if en.get("Obstruction") and has_obstruction(img, block_fraction=obstruct_f):
                    reasons.append("obstruction")

                if reasons:
                    self.bad_list.append(str(p))
                    bad_reasons[p.name] = reasons
                    for r in reasons:
                        reason_counts[r] = reason_counts.get(r, 0) + 1
                    print(f"  ✗ {p.name}: {', '.join(reasons)}")

                self.progress_bar.set((idx + 1) / len(images))

            n_bad = len(self.bad_list)
            n_good = len(images) - n_bad
            print(f"\nFilter complete: {n_bad} bad / {n_good} good "
                  f"out of {len(images)} images.")

            # save bad list
            txt_path = os.path.join(self.output_folder, "bad_images.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("filename\treason\n")
                for bp in self.bad_list:
                    name = os.path.basename(bp)
                    reasons = bad_reasons.get(name, [])
                    f.write(f"{name}\t{','.join(reasons)}\n")
            print(f"Bad image list saved: {txt_path}")

            # update plot
            self.axes[1].clear()
            labels = [k for k, v in reason_counts.items() if v > 0]
            values = [reason_counts[k] for k in labels]
            if not labels:
                labels = list(reason_counts.keys())
                values = [0] * len(labels)
            colors = ["#e74c3c", "#e67e22", "#3498db", "#2c3e50",
                      "#95a5a6", "#8e44ad", "#1abc9c"]
            self.axes[1].barh(labels, values,
                              color=colors[:len(labels)])
            self.axes[1].set_title(
                f"Filter Results ({n_bad} bad / {n_good} good)")
            self.axes[1].set_xlabel("Count")
            self.fig.tight_layout()
            self.canvas_plot.draw()

        except Exception as e:
            print(f"[ERROR] {e}")
            messagebox.showerror("Error", str(e))

    # ——— harmonisation ———

    def _harmonise_threaded(self):
        threading.Thread(target=self._run_harmonise, daemon=True).start()

    def _run_harmonise(self):
        try:
            self.progress_bar.set(0)

            if not self.input_folder:
                messagebox.showwarning("Warning",
                                       "Select an image folder first.")
                return
            if not self.output_folder:
                messagebox.showwarning("Warning",
                                       "Select an output folder first.")
                return

            tol = float(self.tol_entry.get())
            sky_pct = float(self.sky_entry.get())
            sky_frac = max(0.0, min(0.9, sky_pct / 100.0))
            exclude_bad = self.exclude_bad_var.get()
            bad_set = set(self.bad_list) if exclude_bad else set()

            if self.recursive_var.get():
                images = collect_images_recursive(self.input_folder)
            else:
                images = collect_images(self.input_folder)

            # filter out bad images if requested
            images = [p for p in images if str(p) not in bad_set]

            if not images:
                messagebox.showwarning("Warning",
                                       "No images remaining after filtering.")
                return

            sky_msg = (f", sky mask top {sky_pct:.0f}%"
                       if sky_frac > 0 else "")
            print(f"\nHarmonising {len(images)} images "
                  f"(tolerance ±{tol} L-units{sky_msg})…")

            # ---- pass 1: compute luminance stats ----
            means = []
            loaded = []
            for p in images:
                img = cv2.imread(str(p))
                if img is None:
                    means.append(np.nan)
                    loaded.append(None)
                    continue
                loaded.append(img)
                means.append(mean_luminance(img, sky_frac))

            means_arr = np.array(means)
            valid = ~np.isnan(means_arr)
            ref_mean = float(np.nanmedian(means_arr))

            # reference image: the one closest to median luminance
            # (not the median-index image, which could be anything)
            ref_idx = int(np.nanargmin(np.abs(means_arr - ref_mean)))
            ref_img = loaded[ref_idx]
            print(f"Reference luminance (median): {ref_mean:.1f}  "
                  f"(image: {images[ref_idx].name})")

            # plot luminance distribution
            self.axes[0].clear()
            self.axes[0].hist(means_arr[valid], bins=30, color="#3498db",
                              edgecolor="white", alpha=0.8)
            self.axes[0].axvline(ref_mean, color="red", linestyle="--",
                                 label=f"Ref={ref_mean:.1f}")
            self.axes[0].axvspan(ref_mean - tol, ref_mean + tol,
                                 alpha=0.15, color="green",
                                 label=f"±{tol} tolerance")
            self.axes[0].set_title("Luminance Distribution")
            self.axes[0].set_xlabel("Mean L-channel")
            self.axes[0].set_ylabel("Count")
            self.axes[0].legend(fontsize="small")

            # ---- pass 2: correct images ----
            input_root = Path(self.input_folder)
            counts = {"copy": 0, "histogram": 0, "soft-gain": 0,
                      "bright-gain": 0, "reverted": 0}

            for idx, (p, img, m) in enumerate(zip(images, loaded, means)):
                if img is None:
                    continue

                delta = m - ref_mean

                if abs(delta) <= tol:
                    mode, corrected = "copy", img
                elif delta <= -2 * tol:
                    mode = "histogram"
                    corrected = hist_match(img, ref_img)
                elif delta >= 2 * tol:
                    mode = "bright-gain"
                    corrected = soft_gain_match(img, ref_mean)
                else:
                    mode = "soft-gain"
                    corrected = soft_gain_match(img, ref_mean)

                # sanity check: did the correction actually help?
                if mode != "copy":
                    corrected, reverted = correction_sanity_check(
                        img, corrected, ref_mean, sky_frac)
                    if reverted:
                        counts["reverted"] += 1
                        print(f"  [REVERT] {p.name}: correction made "
                              f"luminance worse — kept original")

                counts[mode] = counts.get(mode, 0) + 1

                # preserve folder structure with _harmonised suffix
                rel = Path(p).relative_to(input_root)
                if len(rel.parts) > 1:
                    parent = rel.parent
                    out_parent = Path(self.output_folder) / (
                        str(parent) + "_harmonised")
                else:
                    out_parent = Path(self.output_folder) / (
                        input_root.name + "_harmonised")
                out_parent.mkdir(parents=True, exist_ok=True)
                out_path = out_parent / rel.name

                cv2.imwrite(str(out_path), corrected)

                if (idx + 1) % 20 == 0 or idx == 0:
                    print(f"  {idx + 1}/{len(images)} | "
                          f"{p.name} | ΔL={delta:+.1f} → {mode}")
                self.progress_bar.set((idx + 1) / len(images))

            print(f"\nHarmonisation complete:")
            for k, v in counts.items():
                if v > 0:
                    print(f"  {k}: {v}")

            # update harmonise plot
            self.axes[2].clear()
            plot_labels = [k for k in counts if counts[k] > 0]
            plot_values = [counts[k] for k in plot_labels]
            bar_colors = {
                "copy": "#2ecc71", "histogram": "#9b59b6",
                "soft-gain": "#e67e22", "bright-gain": "#e74c3c",
                "reverted": "#7f8c8d",
            }
            self.axes[2].bar(plot_labels,
                             plot_values,
                             color=[bar_colors.get(k, "#3498db")
                                    for k in plot_labels])
            self.axes[2].set_title("Harmonisation Results")
            self.axes[2].set_ylabel("Count")

            self.fig.tight_layout()
            self.canvas_plot.draw()

            messagebox.showinfo("Done",
                                f"Harmonised images saved to:\n"
                                f"{self.output_folder}")

        except Exception as e:
            print(f"[ERROR] {e}")
            messagebox.showerror("Error", str(e))


# ——— standalone ———
if __name__ == "__main__":
    root = ctk.CTk()
    root.withdraw()
    HarmoniseImagesWindow(master=root)
    root.mainloop()