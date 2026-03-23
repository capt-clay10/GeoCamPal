"""
Profile & Hovmöller Tool

Extract pixel profiles along a user‑defined line across a folder of images,
then visualise them as:
  1. **RGB Hovmöller** — actual pixel colours stacked in time (feature tracking)
  2. **Intensity Hovmöller** — grayscale heatmap for quantitative analysis
  3. **Profile overlay** — all individual profiles plotted on one axis

The user draws a profile line interactively (click start + end on a sample
image) or enters pixel coordinates manually.  Profile width (perpendicular
averaging) is configurable.

Outputs:
  • hovmuller_rgb.png / hovmuller_intensity.png
  • profile_overlay.png
  • profiles.txt  — tab‑separated, one column per image
"""

# %% ————————————————————————————— imports —————————————————————————————
import sys
import os
import re
import threading
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mdates
from matplotlib.colors import Normalize

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox

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


# %% ————————————————————————————— timestamp parsing ———————————————————
# (shared with exploration.py — duplicated here so module is standalone)

_COMMON_PATTERNS = [
    (r"(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})(?:_(\d{2}))?",
     "%Y_%m_%d_%H_%M_%S"),
    (r"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})",
     "%Y%m%d_%H%M%S"),
    (r"(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})",
     "%Y%m%dT%H%M%S"),
    (r"(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})",
     "%Y-%m-%d_%H-%M-%S"),
    (r"(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})",
     "%Y-%m-%dT%H:%M:%S"),
]


def parse_datetime_from_filename(filename, user_format=None):
    stem = Path(filename).stem
    if user_format:
        rx = user_format
        for code, pat in [("%Y", r"(\d{4})"), ("%m", r"(\d{2})"),
                          ("%d", r"(\d{2})"), ("%H", r"(\d{2})"),
                          ("%M", r"(\d{2})"), ("%S", r"(\d{2})")]:
            rx = rx.replace(code, pat)
        m = re.search(rx, stem)
        if m:
            try:
                return datetime.strptime(m.group(0), user_format)
            except ValueError:
                pass
    for pattern, fmt in _COMMON_PATTERNS:
        m = re.search(pattern, stem)
        if m:
            groups = m.groups()
            if len(groups) == 5 or (len(groups) == 6 and groups[5] is None):
                groups = groups[:5] + ("00",)
            date_str = "_".join(groups[:6])
            try:
                return datetime.strptime(date_str, "%Y_%m_%d_%H_%M_%S")
            except ValueError:
                continue
    return None


def collect_dated_images(folder, user_format=None, recursive=False):
    results = []
    if recursive:
        for root, _, files in os.walk(folder):
            for f in files:
                if Path(f).suffix.lower() in IMAGE_EXTS:
                    fp = Path(root) / f
                    dt = parse_datetime_from_filename(f, user_format)
                    if dt is not None:
                        results.append((fp, dt))
    else:
        for p in Path(folder).iterdir():
            if p.suffix.lower() in IMAGE_EXTS:
                dt = parse_datetime_from_filename(p.name, user_format)
                if dt is not None:
                    results.append((p, dt))
    results.sort(key=lambda x: x[1])
    return results


# %% ————————————————————————————— profile extraction ——————————————————

def extract_profile(img, x1, y1, x2, y2, width=1):
    """
    Extract pixel values along the line from (x1,y1) to (x2,y2),
    averaged over a perpendicular band of *width* pixels.

    Parameters
    ----------
    img    : numpy array (H, W) or (H, W, 3)
    x1, y1 : start point (pixel coordinates, x=col, y=row)
    x2, y2 : end point
    width  : number of pixels to average perpendicular to the profile

    Returns
    -------
    profile : 1D array of shape (n_points,) for grayscale
              or (n_points, 3) for RGB
    """
    dx = x2 - x1
    dy = y2 - y1
    length = max(int(np.hypot(dx, dy)), 1)

    # unit vectors: along profile and perpendicular
    ux, uy = dx / length, dy / length
    px, py = -uy, ux  # perpendicular

    # sample points along the profile
    t = np.arange(length)
    cx = x1 + t * ux  # centre x
    cy = y1 + t * uy  # centre y

    half_w = (width - 1) / 2.0
    offsets = np.linspace(-half_w, half_w, max(width, 1))

    h, w = img.shape[:2]
    is_rgb = img.ndim == 3

    if is_rgb:
        profile = np.zeros((length, 3), dtype=np.float64)
    else:
        profile = np.zeros(length, dtype=np.float64)

    count = np.zeros(length, dtype=np.float64)

    for off in offsets:
        sx = np.clip((cx + off * px).astype(int), 0, w - 1)
        sy = np.clip((cy + off * py).astype(int), 0, h - 1)
        if is_rgb:
            profile += img[sy, sx, :]
        else:
            profile += img[sy, sx]
        count += 1.0

    if is_rgb:
        profile /= count[:, None]
    else:
        profile /= count

    return profile


def compute_profile_length_m(x1, y1, x2, y2, resolution=None):
    """Compute profile length.  If resolution given, return in metres."""
    px_len = np.hypot(x2 - x1, y2 - y1)
    if resolution and resolution > 0:
        return px_len * resolution
    return px_len


# %% ————————————————————————————— main GUI ————————————————————————————
class ProfileHovmullerWindow(ctk.CTkToplevel):
    """Extract profiles and build Hovmöller diagrams from image folders."""

    def __init__(self, master=None, **kw):
        super().__init__(master=master, **kw)
        self.title("Profile & Hovmöller Tool")
        self.geometry("1400x950")
        try:
            self.iconbitmap(resource_path("launch_logo.ico"))
        except Exception:
            pass

        # ——— state ———
        self.image_folder = None
        self.output_folder = None
        self.sample_img = None
        self.sample_path = None
        self.click_points = []   # [(x, y), ...]
        self.profile_data = None  # dict after generation
        self.recursive_var = tk.BooleanVar(value=False)
        self._colorbars = []  # track colorbars so we can remove them

        # ——— layout ———
        self.grid_rowconfigure(0, weight=3)
        self.grid_rowconfigure(1, weight=0)
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # ---- TOP: display panel (3 plots) ----
        self.top_panel = ctk.CTkFrame(self)
        self.top_panel.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.fig, self.axes = plt.subplots(1, 3, figsize=(16, 5))
        self.axes[0].set_title("Sample Image — click to draw profile")
        self.axes[0].axis("off")
        self.axes[1].set_title("Hovmöller Diagram")
        self.axes[1].axis("off")
        self.axes[2].set_title("Profile Overlay")
        self.axes[2].axis("off")
        self.fig.tight_layout()
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=self.top_panel)
        self.canvas_plot.get_tk_widget().pack(fill="both", expand=True)

        # connect click events
        self.cid_click = self.fig.canvas.mpl_connect(
            "button_press_event", self._on_click)

        # ---- BOTTOM: controls ----
        self.bottom_panel = ctk.CTkFrame(self)
        self.bottom_panel.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        # Row 1 — folders & sample image
        row1 = ctk.CTkFrame(self.bottom_panel)
        row1.pack(fill="x", padx=5, pady=2)

        ctk.CTkButton(row1, text="Browse Image Folder",
                      command=self._browse_folder).grid(
            row=0, column=0, padx=5, pady=5)
        self.folder_label = ctk.CTkLabel(row1, text="No folder selected")
        self.folder_label.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        ctk.CTkCheckBox(row1, text="Include sub-folders",
                        variable=self.recursive_var).grid(
            row=0, column=2, padx=10, pady=5)

        ctk.CTkButton(row1, text="Load Sample Image",
                      command=self._load_sample).grid(
            row=0, column=3, padx=5, pady=5)
        self.sample_label = ctk.CTkLabel(row1, text="No sample loaded")
        self.sample_label.grid(row=0, column=4, padx=5, pady=5, sticky="w")

        # Row 2 — profile coordinates
        row2 = ctk.CTkFrame(self.bottom_panel)
        row2.pack(fill="x", padx=5, pady=2)

        ctk.CTkLabel(row2, text="Profile Start (x, y):").grid(
            row=0, column=0, padx=3, pady=3)
        self.x1_entry = ctk.CTkEntry(row2, width=60, placeholder_text="x1")
        self.x1_entry.grid(row=0, column=1, padx=2, pady=3)
        self.y1_entry = ctk.CTkEntry(row2, width=60, placeholder_text="y1")
        self.y1_entry.grid(row=0, column=2, padx=2, pady=3)

        ctk.CTkLabel(row2, text="End (x, y):").grid(
            row=0, column=3, padx=3, pady=3)
        self.x2_entry = ctk.CTkEntry(row2, width=60, placeholder_text="x2")
        self.x2_entry.grid(row=0, column=4, padx=2, pady=3)
        self.y2_entry = ctk.CTkEntry(row2, width=60, placeholder_text="y2")
        self.y2_entry.grid(row=0, column=5, padx=2, pady=3)

        ctk.CTkLabel(row2, text="Avg width (px):").grid(
            row=0, column=6, padx=3, pady=3)
        self.width_entry = ctk.CTkEntry(row2, width=50)
        self.width_entry.insert(0, "5")
        self.width_entry.grid(row=0, column=7, padx=2, pady=3)

        ctk.CTkLabel(row2, text="Mode:").grid(
            row=0, column=8, padx=3, pady=3)
        self.mode_var = ctk.StringVar(value="RGB")
        ctk.CTkOptionMenu(
            row2, variable=self.mode_var,
            values=["RGB", "Intensity"], width=100
        ).grid(row=0, column=9, padx=3, pady=3)

        ctk.CTkLabel(row2, text="→ Click two points on the sample image "
                     "or enter coordinates manually",
                     font=("Arial", 10), text_color="gray").grid(
            row=0, column=10, padx=8, pady=3, sticky="w")

        # Row 3 — datetime format, output, generate, reset
        row3 = ctk.CTkFrame(self.bottom_panel)
        row3.pack(fill="x", padx=5, pady=2)

        ctk.CTkLabel(row3, text="Filename datetime format:").grid(
            row=0, column=0, padx=3, pady=5)
        self.fmt_entry = ctk.CTkEntry(row3, width=150,
                                       placeholder_text="%Y_%m_%d_%H_%M")
        self.fmt_entry.grid(row=0, column=1, padx=3, pady=5)

        ctk.CTkButton(row3, text="Browse Output Folder",
                      command=self._browse_output).grid(
            row=0, column=2, padx=5, pady=5)
        self.output_label = ctk.CTkLabel(row3, text="No output folder")
        self.output_label.grid(row=0, column=3, padx=5, pady=5, sticky="w")

        ctk.CTkButton(row3, text="Generate Hovmöller",
                      command=self._generate_threaded, fg_color="#0F52BA").grid(
            row=0, column=4, padx=10, pady=5)

        self.progress_bar = ctk.CTkProgressBar(row3, width=180)
        self.progress_bar.grid(row=0, column=5, padx=5, pady=5)
        self.progress_bar.set(0)

        self.btn_reset = ctk.CTkButton(
            row3, text="Reset", command=self._reset,
            width=80, fg_color="#8B0000", hover_color="#A52A2A")
        self.btn_reset.grid(row=0, column=6, padx=5, pady=5, sticky="e")

        # ---- CONSOLE ----
        self.console_frame = ctk.CTkFrame(self)
        self.console_frame.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        self.console_text = tk.Text(self.console_frame, wrap="word", height=8)
        self.console_text.pack(fill="both", expand=True, padx=5, pady=5)
        self.stdout_redirector = StdoutRedirector(self.console_text)
        sys.stdout = self.stdout_redirector
        sys.stderr = self.stdout_redirector
        print("Profile & Hovmöller Tool\n"
              "========================\n"
              "\n"
              "WORKFLOW\n"
              "  1. Browse to an image folder and load a sample image.\n"
              "  2. Click two points on the sample to define a profile line,\n"
              "     or enter pixel coordinates manually.\n"
              "  3. Set the averaging width (perpendicular pixel band).\n"
              "  4. Choose mode: RGB (colour timestack) or Intensity (heatmap).\n"
              "  5. Click Generate Hovmöller.\n"
              "\n"
              "DISPLAY PANEL\n"
              "  Left:   Sample image with drawn profile line.\n"
              "  Centre: Hovmöller — profiles stacked in time.\n"
              "          RGB mode shows actual pixel colours (feature tracking).\n"
              "          Intensity mode shows a grayscale heatmap.\n"
              "  Right:  All individual profiles overlaid on one axis.\n"
              "\n"
              "OUTPUTS\n"
              "  hovmuller_rgb.png or hovmuller_intensity.png\n"
              "  profile_overlay.png\n"
              "  profiles.txt — tab-separated, one column per image\n"
              "================================")

    # ——— browse callbacks ———

    def _browse_folder(self):
        d = filedialog.askdirectory(title="Select Image Folder")
        if d:
            self.image_folder = d
            self.folder_label.configure(text=d)

    def _browse_output(self):
        d = filedialog.askdirectory(title="Select Output Folder")
        if d:
            self.output_folder = d
            self.output_label.configure(text=d)

    def _load_sample(self):
        p = filedialog.askopenfilename(
            title="Select Sample Image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")])
        if not p:
            return
        img = cv2.imread(p)
        if img is None:
            messagebox.showerror("Error", f"Cannot read: {p}")
            return
        self.sample_path = p
        self.sample_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.sample_label.configure(text=os.path.basename(p))
        self.click_points = []
        self._draw_sample()
        print(f"Loaded sample: {os.path.basename(p)} "
              f"({img.shape[1]}×{img.shape[0]} px)")

    def _draw_sample(self):
        """Show the sample image with any drawn profile line."""
        self.axes[0].clear()
        if self.sample_img is not None:
            self.axes[0].imshow(self.sample_img, aspect="equal")
        self.axes[0].set_title("Sample Image — click to draw profile")
        self.axes[0].axis("off")

        if len(self.click_points) >= 1:
            p1 = self.click_points[0]
            self.axes[0].plot(p1[0], p1[1], "r+", markersize=12,
                              markeredgewidth=2, zorder=10)
        if len(self.click_points) >= 2:
            p1, p2 = self.click_points[0], self.click_points[1]
            self.axes[0].plot([p1[0], p2[0]], [p1[1], p2[1]],
                              "r-", linewidth=1.5, zorder=9)
            self.axes[0].plot(p2[0], p2[1], "r+", markersize=12,
                              markeredgewidth=2, zorder=10)
            # show averaging width band
            try:
                w = int(self.width_entry.get())
            except ValueError:
                w = 1
            if w > 1:
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                length = max(np.hypot(dx, dy), 1)
                px, py = -dy / length, dx / length
                half_w = w / 2.0
                corners = [
                    (p1[0] + half_w * px, p1[1] + half_w * py),
                    (p2[0] + half_w * px, p2[1] + half_w * py),
                    (p2[0] - half_w * px, p2[1] - half_w * py),
                    (p1[0] - half_w * px, p1[1] - half_w * py),
                    (p1[0] + half_w * px, p1[1] + half_w * py),
                ]
                xs, ys = zip(*corners)
                self.axes[0].plot(xs, ys, "r--", linewidth=0.7,
                                  alpha=0.6, zorder=8)

        self.fig.tight_layout()
        self.canvas_plot.draw()

    # ——— interactive click ———

    def _on_click(self, event):
        if event.inaxes != self.axes[0]:
            return
        if self.sample_img is None:
            return

        x, y = int(round(event.xdata)), int(round(event.ydata))

        if len(self.click_points) >= 2:
            self.click_points = []  # restart

        self.click_points.append((x, y))

        # update coordinate entries
        if len(self.click_points) == 1:
            self.x1_entry.delete(0, tk.END)
            self.x1_entry.insert(0, str(x))
            self.y1_entry.delete(0, tk.END)
            self.y1_entry.insert(0, str(y))
            print(f"Start point: ({x}, {y})")
        elif len(self.click_points) == 2:
            self.x2_entry.delete(0, tk.END)
            self.x2_entry.insert(0, str(x))
            self.y2_entry.delete(0, tk.END)
            self.y2_entry.insert(0, str(y))
            dx = x - self.click_points[0][0]
            dy = y - self.click_points[0][1]
            px_len = np.hypot(dx, dy)
            print(f"End point: ({x}, {y})  |  "
                  f"Profile length: {px_len:.0f} px")

        self._draw_sample()

    # ——— reset ———

    def _reset(self):
        self.image_folder = None
        self.output_folder = None
        self.sample_img = None
        self.sample_path = None
        self.click_points = []
        self.profile_data = None
        self.folder_label.configure(text="No folder selected")
        self.output_label.configure(text="No output folder")
        self.sample_label.configure(text="No sample loaded")
        self.progress_bar.set(0)
        self.recursive_var.set(False)

        # remove colorbars
        for cb in self._colorbars:
            try:
                cb.remove()
            except Exception:
                pass
        self._colorbars = []

        for e in (self.x1_entry, self.y1_entry, self.x2_entry, self.y2_entry):
            e.delete(0, tk.END)
        self.width_entry.delete(0, tk.END)
        self.width_entry.insert(0, "5")
        self.mode_var.set("RGB")
        self.fmt_entry.delete(0, tk.END)

        for ax in self.axes:
            ax.clear()
            ax.axis("off")
        self.axes[0].set_title("Sample Image — click to draw profile")
        self.axes[1].set_title("Hovmöller Diagram")
        self.axes[2].set_title("Profile Overlay")
        self.fig.tight_layout()
        self.canvas_plot.draw()
        self.console_text.delete("1.0", tk.END)
        print("Session reset.\n================================")

    # ——— generation ———

    def _generate_threaded(self):
        threading.Thread(target=self._generate, daemon=True).start()

    def _generate(self):
        try:
            self.progress_bar.set(0)

            # ---- validate ----
            if not self.image_folder:
                messagebox.showwarning("Warning",
                                       "Select an image folder first.")
                return
            if not self.output_folder:
                messagebox.showwarning("Warning",
                                       "Select an output folder first.")
                return

            try:
                x1 = int(self.x1_entry.get())
                y1 = int(self.y1_entry.get())
                x2 = int(self.x2_entry.get())
                y2 = int(self.y2_entry.get())
            except (ValueError, TypeError):
                messagebox.showwarning(
                    "Warning",
                    "Define a profile line first (click on the image "
                    "or enter coordinates).")
                return

            avg_width = max(1, int(self.width_entry.get()))
            mode = self.mode_var.get()
            user_fmt = self.fmt_entry.get().strip() or None
            px_length = int(np.hypot(x2 - x1, y2 - y1))

            if px_length < 2:
                messagebox.showwarning("Warning",
                                       "Profile line is too short.")
                return

            print(f"\n=== Generating Hovmöller ===")
            print(f"Profile: ({x1},{y1}) → ({x2},{y2})  |  "
                  f"{px_length} px long  |  "
                  f"avg width: {avg_width} px  |  mode: {mode}")

            # ---- collect images ----
            image_list = collect_dated_images(
                self.image_folder, user_fmt, self.recursive_var.get())
            if not image_list:
                messagebox.showwarning(
                    "Warning",
                    "No images with parseable timestamps found.")
                return
            print(f"Found {len(image_list)} dated images.")

            # ---- extract profiles ----
            profiles_rgb = []
            profiles_gray = []
            timestamps = []
            filenames = []

            for idx, (fp, dt) in enumerate(image_list):
                img = cv2.imread(str(fp))
                if img is None:
                    print(f"[WARN] Cannot read: {fp.name}")
                    continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                prof_rgb = extract_profile(img_rgb, x1, y1, x2, y2, avg_width)
                prof_gray = extract_profile(gray, x1, y1, x2, y2, avg_width)

                profiles_rgb.append(prof_rgb)
                profiles_gray.append(prof_gray)
                timestamps.append(dt)
                filenames.append(fp.name)

                if (idx + 1) % 20 == 0 or idx == 0:
                    print(f"  Extracted {idx + 1}/{len(image_list)}")
                self.progress_bar.set((idx + 1) / len(image_list) * 0.7)

            if not profiles_rgb:
                messagebox.showwarning("Warning",
                                       "No profiles could be extracted.")
                return

            n_imgs = len(profiles_rgb)
            n_pts = profiles_rgb[0].shape[0]
            print(f"\nExtracted {n_imgs} profiles, {n_pts} points each.")

            # ---- build Hovmöller arrays ----
            # RGB: (n_imgs, n_pts, 3) uint8
            hov_rgb = np.stack(profiles_rgb, axis=0)
            hov_rgb = np.clip(hov_rgb, 0, 255).astype(np.uint8)

            # Intensity: (n_imgs, n_pts) float
            hov_gray = np.stack(profiles_gray, axis=0)

            self.profile_data = {
                "hov_rgb": hov_rgb,
                "hov_gray": hov_gray,
                "timestamps": timestamps,
                "filenames": filenames,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "width": avg_width, "mode": mode,
            }

            # ---- display ----
            # remove any previous colorbars
            for cb in self._colorbars:
                try:
                    cb.remove()
                except Exception:
                    pass
            self._colorbars = []

            # Hovmöller
            self.axes[1].clear()
            dist_axis = np.arange(n_pts)
            time_labels = [t.strftime("%Y-%m-%d\n%H:%M") for t in timestamps]

            if mode == "RGB":
                self.axes[1].imshow(hov_rgb, aspect="auto",
                                     interpolation="nearest",
                                     extent=[0, n_pts, n_imgs - 0.5, -0.5])
                self.axes[1].set_title("Hovmöller — RGB")
            else:
                im = self.axes[1].imshow(hov_gray, aspect="auto",
                                          cmap="gray",
                                          interpolation="nearest",
                                          extent=[0, n_pts, n_imgs - 0.5, -0.5])
                cb1 = self.fig.colorbar(im, ax=self.axes[1],
                                         fraction=0.03, pad=0.02)
                self._colorbars.append(cb1)
                self.axes[1].set_title("Hovmöller — Intensity")

            self.axes[1].set_xlabel("Distance along profile (px)")
            self.axes[1].set_ylabel("Image index")

            # add time labels on y-axis (subsample if many images)
            max_labels = 20
            if n_imgs > max_labels:
                step = n_imgs // max_labels
                tick_pos = list(range(0, n_imgs, step))
                tick_labels = [time_labels[i] for i in tick_pos]
            else:
                tick_pos = list(range(n_imgs))
                tick_labels = time_labels
            self.axes[1].set_yticks(tick_pos)
            self.axes[1].set_yticklabels(tick_labels, fontsize=6)

            # Profile overlay
            self.axes[2].clear()
            cmap = plt.cm.viridis
            for i, prof in enumerate(profiles_gray):
                color = cmap(i / max(n_imgs - 1, 1))
                self.axes[2].plot(dist_axis, prof, color=color,
                                  alpha=0.4, linewidth=0.5)
            self.axes[2].set_xlabel("Distance along profile (px)")
            self.axes[2].set_ylabel("Intensity")
            self.axes[2].set_title(f"Profile Overlay ({n_imgs} images)")
            self.axes[2].grid(True, alpha=0.3)

            # add colorbar-like time indicator
            sm = plt.cm.ScalarMappable(
                cmap=cmap, norm=Normalize(0, n_imgs - 1))
            sm.set_array([])
            cb2 = self.fig.colorbar(sm, ax=self.axes[2],
                                     fraction=0.03, pad=0.02)
            cb2.set_label("Image index", fontsize=8)
            self._colorbars.append(cb2)

            self.fig.tight_layout()
            self.canvas_plot.draw()
            self.progress_bar.set(0.85)

            # ---- save outputs ----
            # Hovmöller PNG
            hov_fig, hov_ax = plt.subplots(figsize=(12, max(6, n_imgs * 0.04)))
            if mode == "RGB":
                hov_ax.imshow(hov_rgb, aspect="auto",
                               interpolation="nearest",
                               extent=[0, n_pts, n_imgs - 0.5, -0.5])
                hov_ax.set_title("Hovmöller — RGB")
                hov_name = "hovmuller_rgb.png"
            else:
                hov_ax.imshow(hov_gray, aspect="auto", cmap="gray",
                               interpolation="nearest",
                               extent=[0, n_pts, n_imgs - 0.5, -0.5])
                hov_ax.set_title("Hovmöller — Intensity")
                hov_name = "hovmuller_intensity.png"
            hov_ax.set_xlabel("Distance along profile (px)")
            hov_ax.set_ylabel("Image")
            hov_ax.set_yticks(tick_pos)
            hov_ax.set_yticklabels(tick_labels, fontsize=6)
            hov_fig.tight_layout()
            hov_path = os.path.join(self.output_folder, hov_name)
            hov_fig.savefig(hov_path, dpi=200, bbox_inches="tight")
            plt.close(hov_fig)
            print(f"Saved: {hov_path}")

            # Overlay PNG
            ov_fig, ov_ax = plt.subplots(figsize=(10, 5))
            for i, prof in enumerate(profiles_gray):
                color = cmap(i / max(n_imgs - 1, 1))
                ov_ax.plot(dist_axis, prof, color=color,
                           alpha=0.4, linewidth=0.5)
            ov_ax.set_xlabel("Distance along profile (px)")
            ov_ax.set_ylabel("Intensity")
            ov_ax.set_title(f"Profile Overlay ({n_imgs} images)")
            ov_ax.grid(True, alpha=0.3)
            ov_fig.tight_layout()
            ov_path = os.path.join(self.output_folder, "profile_overlay.png")
            ov_fig.savefig(ov_path, dpi=200, bbox_inches="tight")
            plt.close(ov_fig)
            print(f"Saved: {ov_path}")

            # Text file: one column per image
            txt_path = os.path.join(self.output_folder, "profiles.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                # header
                f.write("distance_px\t" +
                        "\t".join(filenames) + "\n")
                for j in range(n_pts):
                    row = [str(j)]
                    for i in range(n_imgs):
                        row.append(f"{profiles_gray[i][j]:.2f}")
                    f.write("\t".join(row) + "\n")
            print(f"Saved: {txt_path}")

            self.progress_bar.set(1.0)
            messagebox.showinfo(
                "Done",
                f"Generated Hovmöller from {n_imgs} images.\n\n"
                f"Outputs saved to:\n{self.output_folder}")

        except Exception as e:
            print(f"[ERROR] {e}")
            messagebox.showerror("Error", str(e))


# ——— standalone ———
if __name__ == "__main__":
    root = ctk.CTk()
    root.withdraw()
    ProfileHovmullerWindow(master=root)
    root.mainloop()