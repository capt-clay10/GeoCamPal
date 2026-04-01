import os
import re
import sys
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
import pandas as pd
from PIL import Image, ImageTk
import utm

from utils import fit_geometry, resource_path, setup_console, restore_console

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")


# ── shared CSV utility ──
try:
    from csv_utils import read_gcp_id_csv, normalise_columns
except ImportError:
    # Fallback if csv_utils.py is not on the path yet
    def read_gcp_id_csv(path, verbose=True):
        df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
        df.columns = [c.strip() for c in df.columns]
        return df

    def normalise_columns(df, verbose=True):
        return df


# %% helpers

def convert_to_utm(lat, lon):
    """
    Convert latitude and longitude to UTM coordinates.
    Returns easting, northing, EPSG.
    """
    easting, northing, zone_number, _zone_letter = utm.from_latlon(lat, lon)
    epsg = 32600 + zone_number if lat >= 0 else 32700 + zone_number
    return easting, northing, epsg


_IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
_MIN_ZOOM = 0.05
_MAX_ZOOM = 64.0
_ZOOM_STEP = 1.4
_OVERVIEW_MAX_W = 280
_OVERVIEW_MAX_H = 220


def extract_gcp_number(name: str):
    """
    Extract GCP number robustly from filenames / IDs such as:
      GCP_1.bmp, GCP1, GCP-12_cam2, my_GCP_07_test
    Returns int or None.
    """
    stem = os.path.splitext(os.path.basename(str(name)))[0]

    m = re.search(r"(?i)gcp[_-]?(\d+)", stem)
    if m:
        return int(m.group(1))

    return None


def extract_camera_part(name: str) -> str:
    """Return a camera token such as cam1/camA if present, else '0'."""
    stem = os.path.splitext(os.path.basename(str(name)))[0]
    m = re.search(r"(?i)(cam[a-z0-9]+)", stem)
    if m:
        return m.group(1)
    return "0"


class PixelToGCPWindow(ctk.CTkToplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("Pixel to GCP Tool")
        fit_geometry(self, 1200, 800, resizable=True)

        try:
            self.iconbitmap(resource_path("launch_logo.ico"))
        except Exception as e:
            print("Warning: Could not load window icon:", e)

        # ——— close handler ———
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # -----------------------
        # Data / State Variables
        # -----------------------
        self.image_folder = None
        self.gcp_file = None
        self.output_folder = None
        self.bad_gcp_list = []
        self.gcp_df = None

        self.image_list = []
        self.current_index = 0
        self.selected_points = {}  # {filename: (x_full_int, y_full_int)}

        self.scale_factor = 1.0
        self.current_pil_img = None
        self.current_filename = None

        # Viewport-render state
        self._bg_image_id = None
        self._bg_photo_ref = None
        self._vp_render_bounds = None
        self._vp_render_zoom = None
        self._scroll_job = None
        self._selection_ids = []
        self._overview_scale = 1.0
        self._overview_size = (1, 1)

        self.convert_to_utm_var = tk.BooleanVar(value=True)

        # =============================
        # TOP SECTION: IMAGES + CONSOLE
        # =============================
        top_section = ctk.CTkFrame(self, fg_color="black")
        top_section.pack(side="top", fill="both", expand=True, padx=5, pady=5)

        # 1) Main Image Panel (left)
        self.main_image_frame = ctk.CTkFrame(top_section)
        self.main_image_frame.pack(side="left", fill="both", expand=True)

        self.scroll_x = tk.Scrollbar(self.main_image_frame, orient=tk.HORIZONTAL)
        self.scroll_x.pack(side="bottom", fill="x")
        self.scroll_y = tk.Scrollbar(self.main_image_frame, orient=tk.VERTICAL)
        self.scroll_y.pack(side="right", fill="y")

        self.main_canvas = tk.Canvas(
            self.main_image_frame,
            bg="black",
            highlightthickness=0,
            xscrollcommand=self._sync_xscroll,
            yscrollcommand=self._sync_yscroll,
        )
        self.main_canvas.pack(side="left", fill="both", expand=True)
        self.scroll_x.config(command=self._main_xview)
        self.scroll_y.config(command=self._main_yview)

        # Bind events for zooming, scrolling, clicking, and redraws
        self.main_canvas.bind("<Button-1>", self.on_main_click)
        self.main_canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.main_canvas.bind("<Shift-MouseWheel>", self._on_shift_mousewheel)
        self.main_canvas.bind("<Button-4>", self._on_mousewheel_linux_up)
        self.main_canvas.bind("<Button-5>", self._on_mousewheel_linux_down)
        self.main_canvas.bind("<Shift-Button-4>", self._on_shift_mousewheel_linux_up)
        self.main_canvas.bind("<Shift-Button-5>", self._on_shift_mousewheel_linux_down)
        self.main_canvas.bind("<Configure>", lambda _e: self._schedule_viewport_update())

        # 2) Overview Panel (center)
        self.overview_frame = ctk.CTkFrame(top_section, width=300)
        self.overview_frame.pack(side="left", fill="y")
        self.overview_canvas = tk.Canvas(
            self.overview_frame,
            width=_OVERVIEW_MAX_W,
            height=_OVERVIEW_MAX_H,
            bg="black",
            highlightthickness=1,
            highlightbackground="#777777",
        )
        self.overview_canvas.pack(fill="both", expand=True, padx=5, pady=5)

        instructions_panel = ctk.CTkFrame(self)
        instructions_panel.pack(side="top", fill="x", padx=5, pady=5)
        instructions_text = (
            "Instructions:\n"
            " • Left-click on the main image to select a pixel (saved in original-image pixel coords).\n"
            " • Press + to zoom in; press - to zoom out.\n"
            " • Use the mouse wheel to scroll vertically; hold Shift + mouse wheel to scroll horizontally.\n"
            " • Press Enter to advance to the next image.\n"
            " • The overview image shows the visible viewport in red."
        )
        instructions_label = ctk.CTkLabel(instructions_panel, text=instructions_text, justify="left")
        instructions_label.pack(side="left", padx=10, pady=5)

        # =============================
        # MIDDLE SECTION: CONSOLE
        # =============================
        console_frame = ctk.CTkFrame(self)
        console_frame.pack(side="top", fill="both", expand=False, padx=5, pady=5)
        self.console_text = tk.Text(console_frame, wrap="word", width=40, height=10)
        self.console_text.pack(fill="both", expand=True, padx=5, pady=5)
        self._console_redir = setup_console(self.console_text,
            "Here you may see console outputs\n--------------------------------\n")

        # =============================
        # BOTTOM SECTION: CONFIG PANEL
        # =============================
        config_panel = ctk.CTkFrame(self)
        config_panel.pack(side="bottom", fill="x", padx=5, pady=5)

        pnl_img_folder = ctk.CTkFrame(config_panel)
        pnl_img_folder.pack(side="top", fill="x", padx=5, pady=2)
        ctk.CTkButton(
            pnl_img_folder,
            text="Browse Image Folder",
            command=self.browse_image_folder,
        ).pack(side="left")
        self.label_image_folder = ctk.CTkLabel(pnl_img_folder, text="No folder selected")
        self.label_image_folder.pack(side="left", padx=5)

        pnl_gcp_file = ctk.CTkFrame(config_panel)
        pnl_gcp_file.pack(side="top", fill="x", padx=5, pady=2)
        ctk.CTkButton(
            pnl_gcp_file,
            text="Browse GCP File",
            command=self.browse_gcp_file,
        ).pack(side="left")
        self.label_gcp_file = ctk.CTkLabel(pnl_gcp_file, text="No file selected")
        self.label_gcp_file.pack(side="left", padx=5)

        self.label_gcp_note = ctk.CTkLabel(
            pnl_gcp_file,
            text="Accepts: latitude/lat, longitude/lon/lng, GCP_ID/id, elevation/elev (any case, any delimiter)",
            fg_color="white",
            text_color="black",
            corner_radius=0,
        )
        self.label_gcp_note.pack(side="left", padx=5)

        self.checkbox_utm = ctk.CTkCheckBox(
            pnl_gcp_file,
            text="Convert to UTM",
            variable=self.convert_to_utm_var,
        )
        self.checkbox_utm.pack(side="left", padx=5)

        pnl_bad_gcps = ctk.CTkFrame(config_panel)
        pnl_bad_gcps.pack(side="top", fill="x", padx=5, pady=2)
        ctk.CTkLabel(pnl_bad_gcps, text="Bad GCPs (comma sep):").pack(side="left")
        self.entry_bad_gcps = ctk.CTkEntry(pnl_bad_gcps)
        self.entry_bad_gcps.pack(side="left", fill="x", expand=True, padx=5)

        pnl_output_folder = ctk.CTkFrame(config_panel)
        pnl_output_folder.pack(side="top", fill="x", padx=5, pady=2)
        ctk.CTkButton(
            pnl_output_folder,
            text="Browse Output Folder",
            command=self.browse_output_folder,
            fg_color="#8C7738",
        ).pack(side="left")
        self.label_output_folder = ctk.CTkLabel(pnl_output_folder, text="No folder selected")
        self.label_output_folder.pack(side="left", padx=5)

        pnl_csv_name = ctk.CTkFrame(config_panel)
        pnl_csv_name.pack(side="top", fill="x", padx=5, pady=2)
        ctk.CTkLabel(pnl_csv_name, text="Output CSV Name:").pack(side="left")
        self.entry_output_filename = ctk.CTkEntry(pnl_csv_name)
        self.entry_output_filename.pack(side="left", fill="x", expand=True, padx=5)

        pnl_start = ctk.CTkFrame(config_panel)
        pnl_start.pack(side="top", fill="x", padx=5, pady=2)
        ctk.CTkButton(
            pnl_start,
            text="Start Process",
            command=self.start_process,
            fg_color="#0F52BA",
        ).pack(side="left")

        # BIND keys to the entire window (zoom + next image)
        self.bind("<Return>", self.next_image)
        self.bind("<plus>", self.zoom_in)
        self.bind("<minus>", self.zoom_out)
        self.bind("<KP_Add>", self.zoom_in)
        self.bind("<KP_Subtract>", self.zoom_out)

    # ——————————————————————————— close handler —————————————————————————
    def _on_close(self):
        """Clean up and close the window."""
        restore_console(self._console_redir)
        self.destroy()

    # LOGGING TO CONSOLE
    # ---------------
    def log(self, msg):
        self.console_text.insert(tk.END, msg + "\n")
        self.console_text.see(tk.END)

    # BROWSE / CONFIG
    # ---------------
    def browse_image_folder(self):
        folder = filedialog.askdirectory(parent= self,title="Select Image Folder")
        if folder:
            self.image_folder = folder
            self.label_image_folder.configure(text=folder)
            self.log(f"Image folder selected: {folder}")

    def browse_gcp_file(self):
        path = filedialog.askopenfilename(parent= self,
            title="Select GCP CSV",
            filetypes=[("CSV", "*.csv"), ("All Files", "*.*")],
        )
        if path:
            self.gcp_file = path
            self.label_gcp_file.configure(text=os.path.basename(path))
            self.log(f"GCP file selected: {path}")

    def browse_output_folder(self):
        folder = filedialog.askdirectory(parent= self,title="Select Output Folder")
        if folder:
            self.output_folder = folder
            self.label_output_folder.configure(text=folder)
            self.log(f"Output folder selected: {folder}")

    def start_process(self):
        # Parse bad GCPs
        bad_str = self.entry_bad_gcps.get().strip()
        if bad_str:
            try:
                self.bad_gcp_list = [int(x.strip()) for x in bad_str.split(",") if x.strip() != ""]
            except Exception as e:
                messagebox.showerror("Error", f"Invalid bad GCPs: {e}")
                return
        else:
            self.bad_gcp_list = []

        if not self.image_folder:
            messagebox.showerror("Error", "No image folder selected.")
            return

        # Build image_list from folder using robust GCP parsing
        all_files = os.listdir(self.image_folder)
        good_images = []
        for f in all_files:
            ext = os.path.splitext(f)[1].lower()
            if ext not in _IMAGE_EXTS:
                continue
            gcp_num = extract_gcp_number(f)
            if gcp_num is None:
                continue
            if gcp_num in self.bad_gcp_list:
                continue
            good_images.append(f)

        self.image_list = sorted(
            good_images,
            key=lambda x: (extract_gcp_number(x) or 10**9, x.lower()),
        )

        if not self.image_list:
            messagebox.showerror("Error", "No valid images found after filtering.")
            return

        self.current_index = 0
        self.log(f"Found {len(self.image_list)} images for processing.")

        # Load GCP file if provided — using csv_utils normalisation
        self.gcp_df = None
        if self.gcp_file:
            try:
                df = read_gcp_id_csv(self.gcp_file)
                df = normalise_columns(df, verbose=False)

                required_cols = ["latitude", "longitude", "GCP_ID"]
                missing = [c for c in required_cols if c not in df.columns]
                if missing:
                    messagebox.showerror(
                        "Error",
                        f"GCP file missing columns: {missing}\n\n"
                        f"Found columns: {list(df.columns)}\n\n"
                        f"Accepted names include: lat/latitude, lon/longitude/lng, GCP_ID/id/gcp_id",
                    )
                    return

                df["gcp_number"] = df["GCP_ID"].apply(extract_gcp_number)
                df = df.dropna(subset=["gcp_number"]).copy()
                df["gcp_number"] = df["gcp_number"].astype(int)
                df = df[~df["gcp_number"].isin(self.bad_gcp_list)].copy()

                if df.empty:
                    messagebox.showerror("Error", "No usable GCP rows remained after filtering.")
                    return

                if self.convert_to_utm_var.get():
                    df[["easting", "northing", "EPSG"]] = df.apply(
                        lambda row: pd.Series(convert_to_utm(row["latitude"], row["longitude"])),
                        axis=1,
                    )
                    epsg_val = int(df["EPSG"].iloc[0])
                    self.log(f"UTM conversion applied — detected EPSG:{epsg_val}")
                else:
                    # X = longitude (east-west), Y = latitude (north-south)
                    df["easting"] = df["longitude"]
                    df["northing"] = df["latitude"]
                    df["EPSG"] = 0

                self.gcp_df = df
                self.log(f"GCP file loaded: {os.path.basename(self.gcp_file)}")
            except Exception as e:
                self.log(f"Failed to load GCP file: {e}")
                self.gcp_df = None

        self.show_image(self.image_list[self.current_index])
        self.focus_set()

    # SHOW IMAGE
    # ---------------
    def show_image(self, filename):
        path = os.path.join(self.image_folder, filename)
        try:
            with Image.open(path) as pil_img:
                self.current_pil_img = pil_img.copy()
        except Exception as e:
            messagebox.showerror("Error", f"Cannot open image: {path}\n{e}")
            return

        self.current_filename = filename
        self.scale_factor = 1.0
        self._invalidate_viewport_cache()
        self.main_canvas.xview_moveto(0)
        self.main_canvas.yview_moveto(0)
        self._schedule_viewport_update(immediate=True)

        idx = self.current_index
        total = len(self.image_list)
        self.title(f"Pixel -> GCP: {filename} ({idx + 1}/{total})")

    # VIEWPORT RENDERING
    # ---------------
    def _invalidate_viewport_cache(self):
        self._vp_render_bounds = None
        self._vp_render_zoom = None

    def _get_viewport_bounds(self):
        try:
            x1 = int(self.main_canvas.canvasx(0))
            y1 = int(self.main_canvas.canvasy(0))
            x2 = int(self.main_canvas.canvasx(self.main_canvas.winfo_width()))
            y2 = int(self.main_canvas.canvasy(self.main_canvas.winfo_height()))
            return (x1, y1, x2, y2)
        except Exception:
            return None

    def _schedule_viewport_update(self, immediate=False):
        if immediate:
            if self._scroll_job is not None:
                try:
                    self.after_cancel(self._scroll_job)
                except Exception:
                    pass
                self._scroll_job = None
            self._do_viewport_update()
            return

        if self._scroll_job is not None:
            try:
                self.after_cancel(self._scroll_job)
            except Exception:
                pass
        self._scroll_job = self.after(25, self._do_viewport_update)

    def _do_viewport_update(self):
        self._scroll_job = None
        self._render_viewport()
        self._draw_selected_point()
        self._update_overview()

    def _render_viewport(self):
        """
        Render only the visible portion of the ORIGINAL image.

        Because the crop is taken directly from the source image and placed
        at the exact canvas coordinates for that crop, the mapping

            original_x = canvas_x / zoom
            original_y = canvas_y / zoom

        stays exact for clicks, even at very high zoom.
        """
        if self.current_pil_img is None:
            return

        src = self.current_pil_img
        full_w, full_h = src.size
        z = max(_MIN_ZOOM, min(_MAX_ZOOM, float(self.scale_factor)))

        full_zoom_w = max(1, int(round(full_w * z)))
        full_zoom_h = max(1, int(round(full_h * z)))
        self.main_canvas.config(scrollregion=(0, 0, full_zoom_w, full_zoom_h))

        vp = self._get_viewport_bounds()
        if vp is None:
            vp = (0, 0, full_zoom_w, full_zoom_h)
        vp_x1, vp_y1, vp_x2, vp_y2 = vp
        vp_w = max(1, vp_x2 - vp_x1)
        vp_h = max(1, vp_y2 - vp_y1)

        # Overscan: avoid a redraw on every tiny scroll step
        margin_w = int(vp_w * 0.5)
        margin_h = int(vp_h * 0.5)
        rx1 = max(0, vp_x1 - margin_w)
        ry1 = max(0, vp_y1 - margin_h)
        rx2 = min(full_zoom_w, vp_x2 + margin_w)
        ry2 = min(full_zoom_h, vp_y2 + margin_h)

        cached = self._vp_render_bounds
        if (
            cached is not None
            and self._vp_render_zoom == round(z, 3)
            and cached[0] <= vp_x1
            and cached[1] <= vp_y1
            and cached[2] >= vp_x2
            and cached[3] >= vp_y2
        ):
            return

        img_x1 = max(0, int(rx1 / z))
        img_y1 = max(0, int(ry1 / z))
        img_x2 = min(full_w, int(rx2 / z) + 1)
        img_y2 = min(full_h, int(ry2 / z) + 1)
        if img_x2 <= img_x1 or img_y2 <= img_y1:
            return

        canvas_x = int(round(img_x1 * z))
        canvas_y = int(round(img_y1 * z))
        out_w = max(1, int(round((img_x2 - img_x1) * z)))
        out_h = max(1, int(round((img_y2 - img_y1) * z)))

        crop = src.crop((img_x1, img_y1, img_x2, img_y2)).convert("RGB")
        if crop.size != (out_w, out_h):
            crop = crop.resize((out_w, out_h), Image.Resampling.LANCZOS)

        photo = ImageTk.PhotoImage(crop)
        if self._bg_image_id is None:
            self._bg_image_id = self.main_canvas.create_image(
                canvas_x,
                canvas_y,
                anchor=tk.NW,
                image=photo,
            )
        else:
            self.main_canvas.coords(self._bg_image_id, canvas_x, canvas_y)
            self.main_canvas.itemconfigure(self._bg_image_id, image=photo)

        self._bg_photo_ref = photo
        self._vp_render_bounds = (rx1, ry1, rx2, ry2)
        self._vp_render_zoom = round(z, 3)

    # SCROLLBAR / VIEW SYNC
    # ---------------
    def _main_xview(self, *args):
        self.main_canvas.xview(*args)
        self._schedule_viewport_update()

    def _main_yview(self, *args):
        self.main_canvas.yview(*args)
        self._schedule_viewport_update()

    def _sync_xscroll(self, first, last):
        self.scroll_x.set(first, last)
        self._update_overview()

    def _sync_yscroll(self, first, last):
        self.scroll_y.set(first, last)
        self._update_overview()

    # OVERVIEW
    # ---------------
    def _update_overview(self):
        self.overview_canvas.delete("all")
        if self.current_pil_img is None:
            return

        full_w, full_h = self.current_pil_img.size
        if full_w <= 0 or full_h <= 0:
            return

        scale = min(_OVERVIEW_MAX_W / full_w, _OVERVIEW_MAX_H / full_h, 1.0)
        ow = max(1, int(round(full_w * scale)))
        oh = max(1, int(round(full_h * scale)))

        ov_img = self.current_pil_img.convert("RGB").resize((ow, oh), Image.Resampling.LANCZOS)
        self.tk_overview_img = ImageTk.PhotoImage(ov_img)
        self.overview_canvas.config(width=ow, height=oh)
        self.overview_canvas.create_image(0, 0, anchor="nw", image=self.tk_overview_img)

        self._overview_scale = scale
        self._overview_size = (ow, oh)

        # Visible viewport rectangle, derived in ORIGINAL image coordinates
        z = max(_MIN_ZOOM, min(_MAX_ZOOM, float(self.scale_factor)))
        vp = self._get_viewport_bounds()
        if vp is None:
            return
        x1, y1, x2, y2 = vp

        img_x1 = max(0.0, min(full_w, x1 / z))
        img_y1 = max(0.0, min(full_h, y1 / z))
        img_x2 = max(0.0, min(full_w, x2 / z))
        img_y2 = max(0.0, min(full_h, y2 / z))

        self.overview_canvas.create_rectangle(
            img_x1 * scale,
            img_y1 * scale,
            img_x2 * scale,
            img_y2 * scale,
            outline="red",
            width=2,
        )

        # Selected point on overview
        if self.current_filename in self.selected_points:
            px, py = self.selected_points[self.current_filename]
            r = 3
            ox = px * scale
            oy = py * scale
            self.overview_canvas.create_oval(ox - r, oy - r, ox + r, oy + r, fill="yellow", outline="black")

    # CLICK & SCROLL
    # ---------------
    def on_main_click(self, event):
        if self.current_pil_img is None or not self.image_list:
            return

        z = max(_MIN_ZOOM, min(_MAX_ZOOM, float(self.scale_factor)))
        x_canvas = self.main_canvas.canvasx(event.x)
        y_canvas = self.main_canvas.canvasy(event.y)

        # Exact mapping from zoomed canvas back to original source pixels
        x_full = int(round(x_canvas / z))
        y_full = int(round(y_canvas / z))

        full_w, full_h = self.current_pil_img.size
        x_full = max(0, min(full_w - 1, x_full))
        y_full = max(0, min(full_h - 1, y_full))

        filename = self.image_list[self.current_index]
        self.selected_points[filename] = (x_full, y_full)
        self.log(f"Selected {filename}: (Pixel_X={x_full}, Pixel_Y={y_full})")
        self._draw_selected_point()
        self._update_overview()

    def _draw_selected_point(self):
        for item_id in self._selection_ids:
            try:
                self.main_canvas.delete(item_id)
            except Exception:
                pass
        self._selection_ids = []

        if self.current_filename not in self.selected_points:
            return

        px, py = self.selected_points[self.current_filename]
        z = max(_MIN_ZOOM, min(_MAX_ZOOM, float(self.scale_factor)))
        cx = px * z
        cy = py * z
        size = max(5, min(14, int(round(6 + z * 0.2))))

        self._selection_ids.append(
            self.main_canvas.create_line(cx - size, cy, cx + size, cy, fill="red", width=2)
        )
        self._selection_ids.append(
            self.main_canvas.create_line(cx, cy - size, cx, cy + size, fill="red", width=2)
        )
        self._selection_ids.append(
            self.main_canvas.create_oval(cx - size, cy - size, cx + size, cy + size, outline="yellow", width=2)
        )

    def _on_mousewheel(self, event):
        direction = -1 if event.delta > 0 else 1
        self.main_canvas.yview_scroll(direction, "units")
        self._schedule_viewport_update()

    def _on_shift_mousewheel(self, event):
        direction = -1 if event.delta > 0 else 1
        self.main_canvas.xview_scroll(direction, "units")
        self._schedule_viewport_update()

    def _on_mousewheel_linux_up(self, _event):
        self.main_canvas.yview_scroll(-1, "units")
        self._schedule_viewport_update()

    def _on_mousewheel_linux_down(self, _event):
        self.main_canvas.yview_scroll(1, "units")
        self._schedule_viewport_update()

    def _on_shift_mousewheel_linux_up(self, _event):
        self.main_canvas.xview_scroll(-1, "units")
        self._schedule_viewport_update()

    def _on_shift_mousewheel_linux_down(self, _event):
        self.main_canvas.xview_scroll(1, "units")
        self._schedule_viewport_update()

    # ZOOM
    # ---------------
    def _zoom_about_view_center(self, new_zoom):
        if self.current_pil_img is None:
            return

        old_zoom = max(_MIN_ZOOM, min(_MAX_ZOOM, float(self.scale_factor)))
        new_zoom = max(_MIN_ZOOM, min(_MAX_ZOOM, float(new_zoom)))
        if abs(new_zoom - old_zoom) < 1e-9:
            return

        full_w, full_h = self.current_pil_img.size
        can_w = max(1, self.main_canvas.winfo_width())
        can_h = max(1, self.main_canvas.winfo_height())

        center_canvas_x = self.main_canvas.canvasx(can_w / 2)
        center_canvas_y = self.main_canvas.canvasy(can_h / 2)
        center_img_x = center_canvas_x / old_zoom
        center_img_y = center_canvas_y / old_zoom

        self.scale_factor = new_zoom
        new_full_w = max(1, int(round(full_w * new_zoom)))
        new_full_h = max(1, int(round(full_h * new_zoom)))
        self.main_canvas.config(scrollregion=(0, 0, new_full_w, new_full_h))

        left = center_img_x * new_zoom - can_w / 2
        top = center_img_y * new_zoom - can_h / 2
        max_left = max(0, new_full_w - can_w)
        max_top = max(0, new_full_h - can_h)
        left = max(0, min(max_left, left))
        top = max(0, min(max_top, top))

        self.main_canvas.xview_moveto(0 if new_full_w <= 1 else left / new_full_w)
        self.main_canvas.yview_moveto(0 if new_full_h <= 1 else top / new_full_h)

        self._invalidate_viewport_cache()
        self._schedule_viewport_update(immediate=True)

    def zoom_in(self, event=None):
        self._zoom_about_view_center(self.scale_factor * _ZOOM_STEP)

    def zoom_out(self, event=None):
        self._zoom_about_view_center(self.scale_factor / _ZOOM_STEP)

    # NEXT IMAGE
    # ---------------
    def next_image(self, event=None):
        if not self.image_list:
            return

        self.current_index += 1
        if self.current_index >= len(self.image_list):
            self.log("All images processed. Saving CSV.")
            saved = self.save_output_csv()
            if saved:
                self.destroy()
        else:
            self.show_image(self.image_list[self.current_index])

    # SAVE OUTPUT
    # ---------------
    def save_output_csv(self):
        if not self.output_folder:
            messagebox.showerror("Error", "No output folder selected.")
            return False

        if self.gcp_df is None:
            messagebox.showerror("Error", "No valid GCP file is loaded.")
            return False

        filename = self.entry_output_filename.get().strip()
        if not filename:
            filename = "pixel_gcp_output"
        output_path = os.path.join(self.output_folder, f"{filename}.csv")

        rows = []
        for fname, (px_full, py_full) in self.selected_points.items():
            gcp_num = extract_gcp_number(fname)
            if gcp_num is None:
                continue

            camera_part = extract_camera_part(fname)
            match = self.gcp_df.loc[self.gcp_df["gcp_number"] == gcp_num]
            if match.empty:
                self.log(f"[warn] No GCP row found for image: {fname}")
                continue

            row = match.iloc[0]
            gcp_id = row["GCP_ID"]
            real_x = row["easting"]
            real_y = row["northing"]

            real_z = 0.0
            for z_col in ["Real_Z", "elevation", "elev", "height", "z", "alt"]:
                if z_col in row.index and pd.notna(row[z_col]):
                    try:
                        real_z = float(row[z_col])
                    except (ValueError, TypeError):
                        pass
                    break

            epsg_code = int(row["EPSG"]) if "EPSG" in row.index and pd.notna(row["EPSG"]) else 0

            rows.append(
                {
                    "Image_name": fname,
                    "Pixel_X": int(px_full),
                    "Pixel_Y": int(py_full),
                    "GCP_ID": gcp_id,
                    "camera": camera_part,
                    "Real_X": real_x,
                    "Real_Y": real_y,
                    "Real_Z": real_z,
                    "EPSG": epsg_code,
                }
            )

        if not rows:
            messagebox.showerror(
                "Error",
                "No rows could be written. Check that your image filenames and GCP_ID values share matching GCP numbers.",
            )
            return False

        df = pd.DataFrame(
            rows,
            columns=[
                "Image_name",
                "Pixel_X",
                "Pixel_Y",
                "GCP_ID",
                "camera",
                "Real_X",
                "Real_Y",
                "Real_Z",
                "EPSG",
            ],
        )

        try:
            df.to_csv(output_path, index=False)
            self.log(f"CSV saved to {output_path}")
            self.log(f"  {len(df)} GCPs, columns: {list(df.columns)}")
            if df["Real_Z"].eq(0).all():
                self.log(
                    "  Note: All Real_Z values are 0.0 — if you have elevation data, add an 'elevation' column to your GCP ID file."
                )
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save CSV: {e}")
            return False


def main():
    root = ctk.CTk()
    root.withdraw()
    win = PixelToGCPWindow(master=root)
    win.focus_force()
    root.mainloop()


if __name__ == "__main__":
    main()
