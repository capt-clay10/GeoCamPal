"""
hsv_mask_processing.py
──────────────────────
Image loading / display, mask calculation, edge detection,
ML-mask workflow, batch processing, and shortcut actions.

This module is a *mixin* — it is meant to be inherited by HSVMaskTool
together with the UI and editing mixins.
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
from PIL import Image
import cv2
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString
import rasterio
import warnings
from rasterio.errors import NotGeoreferencedWarning
from difflib import SequenceMatcher

from utils import restore_console

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


class HSVMaskProcessingMixin:
    """Mixin that supplies image I/O, mask calculation, edge detection,
    ML-mask workflow, batch processing, display helpers, and shortcuts."""

    # -------------- UTILITY METHODS --------------

    def distance(self, x1, y1, x2, y2):
        return ((x2 - x1)**2 + (y2 - y1)**2) ** 0.5

    def _clear_ctk_label(self, lbl):
        """Attach a 1×1 transparent CTkImage so CTk never points to a dead id."""
        if lbl and lbl.winfo_exists():
            # valid image id → no TclError
            lbl.configure(image=self._blank_img)
            lbl.image = self._blank_img           # keep reference

    @staticmethod
    def _fit_size(img_w, img_h, panel_w, panel_h):
        """Return (w, h) that fits img inside panel, preserving aspect ratio."""
        if img_w <= 0 or img_h <= 0 or panel_w <= 0 or panel_h <= 0:
            return max(1, panel_w), max(1, panel_h)
        scale = min(panel_w / img_w, panel_h / img_h, 1.0)
        return max(1, int(img_w * scale)), max(1, int(img_h * scale))

    # --- path display helpers for ML rows ---
    def _shorten_path(self, path, maxlen=40):
        """Return a middle-ellipsized path for display only."""
        if not path:
            return ""
        if len(path) <= maxlen:
            return path
        # keep start and end
        part = maxlen // 2 - 2
        return path[:part] + "…/" + path[-(part+2):]

    def _read_mask_image(self, path):
        """Read a mask as grayscale uint8 {0,255}. Returns None on failure."""
        if not path or not os.path.exists(path):
            return None

        m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if m is None:
            return None

        # If RGBA: use alpha ONLY if it actually encodes the mask (not uniformly opaque)
        if m.ndim == 3 and m.shape[2] == 4:
            b, g, r, a = cv2.split(m)

            # alpha is "useful" if it contains real transparency/background
            alpha_useful = (a.min() == 0) and (a.max() == 255) and (np.count_nonzero(a == 0) > 0)

            if alpha_useful:
                gray = a
            else:
                # alpha is just 255 everywhere → mask is in RGB
                gray = cv2.cvtColor(m[:, :, :3], cv2.COLOR_BGR2GRAY)

        elif m.ndim == 3:
            gray = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
        else:
            gray = m

        # Make strictly binary 0/255
        # If already binary-ish, keep it simple; else Otsu
        u = np.unique(gray)
        if len(u) <= 3 and set(u.tolist()).issubset({0, 1, 255}):
            mbin = (gray > 0).astype(np.uint8) * 255
        else:
            _, mbin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        return mbin


    def _resize_mask_to_cv(self, m):
        """Resize a binary mask to current cv_image size with nearest-neighbor."""
        if self.cv_image is None or m is None:
            return None

        m = self._ensure_binary_u8(m)

        # First, if we have the full image, snap mask to full-image size (keeps alignment consistent)
        if self.full_image is not None:
            Hf, Wf = self.full_image.shape[:2]
            if m.shape[:2] != (Hf, Wf):
                m = cv2.resize(m, (Wf, Hf), interpolation=cv2.INTER_NEAREST)

        # Then downscale to cv_image size
        h, w = self.cv_image.shape[:2]
        if m.shape[:2] != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)

        return m



    def _best_match_in_folder(self, img_base, folder, common_len_text):
        """
        Find the most appropriate mask file in 'folder' for an image basename 'img_base' (w/o extension).
        If common_len_text is provided and numeric, match by leading prefix of that many characters.
        Otherwise, pick by highest SequenceMatcher ratio against basenames in folder.
        """
        if not folder or not os.path.isdir(folder):
            return None
        mask_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        mask_files = [os.path.join(folder, f) for f in os.listdir(folder)
                      if f.lower().endswith(mask_exts)]

        if not mask_files:
            return None

        base_noext = img_base
        # Try "common name length" as integer prefix length
        if common_len_text.strip():
            try:
                n = int(common_len_text.strip())
                if n > 0:
                    prefix = base_noext[:n]
                    # exact prefix match preferred
                    candidates = [p for p in mask_files if os.path.basename(p).startswith(prefix)]
                    if candidates:
                        # if multiple, choose the one with closest overall similarity anyway
                        best = max(candidates,
                                   key=lambda p: SequenceMatcher(None, base_noext, os.path.splitext(os.path.basename(p))[0]).ratio())
                        return best
            except ValueError:
                # ignore and fall back to similarity
                pass

        # Fallback: highest similarity of basenames
        def sim(path):
            return SequenceMatcher(None, base_noext, os.path.splitext(os.path.basename(path))[0]).ratio()

        return max(mask_files, key=sim)

    def _ensure_binary_u8(self, img):
        """Return a strictly binary uint8 mask (0/255). Accepts BGR/GRAY/float/bool/0-1."""
        if img is None:
            return None

        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # bool -> 0/255
        if img.dtype == bool:
            return (img.astype(np.uint8) * 255)

        # float -> normalize then binarize
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # handle 0/1 masks safely
        if img.max() <= 1:
            img = (img > 0).astype(np.uint8) * 255
            return img

        # general case: any non-zero is foreground
        return (img > 0).astype(np.uint8) * 255

    # -------------- IMAGE DISPLAY METHODS --------------

    def update_image_display(self, event=None):
        if self.cv_image is None:
            return
        panel_w = self.top_left_frame.winfo_width()
        panel_h = self.top_left_frame.winfo_height()
        if panel_w < 1 or panel_h < 1:
            return

        disp = self.cv_image.copy()
        h0, w0 = disp.shape[:2]

        # bbox in CV coords
        if hasattr(self, 'bbox') and self.use_bbox.get():
            x0, y0, w, h = self.bbox
            x = int(x0 * self.scale); y = int(y0 * self.scale)
            w = int(w  * self.scale); h = int(h  * self.scale)
            cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Fit to panel preserving aspect ratio
        fit_w, fit_h = self._fit_size(w0, h0, panel_w, panel_h)
        resized = cv2.resize(disp, (fit_w, fit_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        ctk_img = ctk.CTkImage(light_image=pil, dark_image=pil, size=(fit_w, fit_h))
        self.image_label.configure(image=ctk_img)
        self.image_label.image = ctk_img


    def update_mask_display(self, event=None):
        if not self.mask_label.winfo_exists():
            return
        if self.current_mask is not None:
            panel_w = self.top_center_frame.winfo_width()
            panel_h = self.top_center_frame.winfo_height()
            if panel_w > 0 and panel_h > 0:
                h0, w0 = self.current_mask.shape[:2]
                fit_w, fit_h = self._fit_size(w0, h0, panel_w, panel_h)
                mask_resized = cv2.resize(self.current_mask, (fit_w, fit_h),
                                          interpolation=cv2.INTER_AREA)
                mask_rgb = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2RGB)
                pil_img = Image.fromarray(mask_rgb)
                ctk_img = ctk.CTkImage(
                    light_image=pil_img, dark_image=pil_img, size=(fit_w, fit_h))
                self.mask_label.configure(image=ctk_img)
                self.mask_label.image = ctk_img

    def update_edge_display(self, event=None):
        """Display all stored features in the right panel (scrollable canvas)."""
        if self.full_image is None:
            return

        # Force geometry update before getting dimensions
        try:
            self.top_right_frame.update_idletasks()
        except Exception:
            pass

        # Use the canvas frame for panel dimensions
        canvas = getattr(self, '_overlay_canvas', None)
        if canvas is None:
            # Fallback: no canvas setup (batch mode)
            return

        panel_w = canvas.winfo_width()
        panel_h = canvas.winfo_height()

        # If frame is still too small, schedule a retry
        if panel_w < 50 or panel_h < 50:
            self.after(100, self.update_edge_display)
            return

        overlay = self.full_image.copy()
        # Confirmed-feature colours (BGR, since OpenCV is BGR-native):
        #   polygon  → dark orange #cc5500  (BGR 0, 85, 204)
        #   polyline → teal       #0088aa  (BGR 170, 136, 0)
        # These match the active/reference palette in hsv_mask_editing
        # (orange family for polygons, cyan family for polylines), at the
        # dimmest stage to indicate "committed".
        _CONFIRMED_POLYGON_BGR  = (0, 85, 204)
        _CONFIRMED_POLYLINE_BGR = (170, 136, 0)
        for (feature_type, points) in self.features:
            if len(points) < 2:
                continue
            pts_np = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
            if feature_type == "polygon":
                cv2.polylines(overlay, [pts_np], isClosed=True,
                              color=_CONFIRMED_POLYGON_BGR, thickness=2)
            else:
                thickness = int(self.edge_thickness_slider.get()) if hasattr(
                    self, 'edge_thickness_slider') else 2
                cv2.polylines(overlay, [pts_np], isClosed=False,
                              color=_CONFIRMED_POLYLINE_BGR, thickness=thickness)

        h0, w0 = overlay.shape[:2]

        # Apply overlay zoom factor (default 1.0 = fit-to-panel)
        overlay_zoom = getattr(self, '_overlay_zoom', 1.0)
        fit_w, fit_h = self._fit_size(w0, h0, panel_w, panel_h)
        disp_w = max(1, int(fit_w * overlay_zoom))
        disp_h = max(1, int(fit_h * overlay_zoom))

        resized = cv2.resize(overlay, (disp_w, disp_h),
                             interpolation=cv2.INTER_AREA)
        overlay_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(overlay_rgb)

        # Draw on the scrollable canvas
        from PIL import ImageTk
        photo = ImageTk.PhotoImage(pil_img)
        self._overlay_photo_ref = photo  # prevent GC

        canvas.delete("all")
        # Centre image if smaller than canvas, otherwise top-left for scrolling
        cx = max(0, (panel_w - disp_w) // 2)
        cy = max(0, (panel_h - disp_h) // 2)
        canvas.create_image(cx, cy, anchor="nw", image=photo)
        canvas.configure(scrollregion=(0, 0,
                                       max(panel_w, disp_w + cx),
                                       max(panel_h, disp_h + cy)))

    # -------------- WINDOW MANAGEMENT --------------

    def checkbox_invert_mask_toggle(self):
        self.do_invert_mask.set(not self.do_invert_mask.get())
        # if mask is showing, recalculate
        if self.current_mask is not None and self.mode != 'batch':
            self.calculate_mask()

    def on_all_close(self):
        restore_console(getattr(self, "_console_redir", None))

        # Destroy image window if it still exists
        if hasattr(self, "image_display_window") and self.image_display_window.winfo_exists():
            self.image_display_window.destroy()
        self.destroy()

    def _restore_center_mask_panel(self):
        """Ensure the middle panel is the standard mask label (not the edit canvas)."""
        if hasattr(self, "top_center_frame") and self.top_center_frame.winfo_exists():
            # remove anything that may have been inserted by the editor
            for w in self.top_center_frame.winfo_children():
                try:
                    w.destroy()
                except Exception:
                    pass
            # recreate the standard mask label
            self.mask_label = ctk.CTkLabel(self.top_center_frame, text="", fg_color="black", anchor="center")
            self.mask_label.pack(fill="both", expand=True)

    # -------------- IMPORT/LOAD --------------

    def load_image(self):
        file_path = filedialog.askopenfilename(
            parent=self,
            title="Select Image",
            filetypes=[
                ("Image Files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("JPEG", "*.jpg *.jpeg"),
                ("PNG", "*.png"),
                ("Bitmap", "*.bmp"),
                ("TIFF", "*.tif *.tiff"),
                ("All Files", "*.*"),
            ],
        )
        if not file_path:
            return
    
        original_image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if original_image is None:
            messagebox.showerror("Error", f"Failed to load image: {file_path}", parent=self)
            return
    
        self.image_path = file_path
        self._current_input_path = file_path
        self._current_input_folder = os.path.dirname(file_path)
        self.filename_label.configure(text=os.path.basename(file_path))
        self.full_image = original_image.copy()
        self.compute_full_masks(original_image)
        if self.mode != "batch":
            self.process_loaded_image(original_image)

    def load_folder(self):
        folder = filedialog.askdirectory(parent=self)
        if not folder:
            return
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        files = [os.path.join(folder, f) for f in os.listdir(
            folder) if f.lower().endswith(exts)]
        if not files:
            messagebox.showerror(
                "Error", "No valid image files found in folder.", parent=self)
            return
        files.sort()
        self.image_files = files
        self._current_input_folder = folder
        self.current_index = 0
        if self.mode == "batch":
            self.filename_label.configure(
                text=f"Loaded {len(files)} images for batch")
        else:
            self.load_current_image()


    def _reset_image_state(self):
        # clear per-image drawing state
        self.features = []
        self.edge_points = []
        self.edited_edge_points = []
        self.initial_edge_points = []
        self.selected_vertex = None
        self.current_mask = None
        self.inner_bbox_mask = None

        # clear editor / preview caches
        self._bg_cache.clear()
        self._bg_current_zoom = None
        self._zoom_cache = {"zoom": None, "img": None}
        self.bg_image_id = None
        self._poly_id = None
        self._vertex_ids = []
        self._redraw_job = None
        self._preview_after_id = None
        self._pending_preview = False


    def load_current_image(self):
        if not self.image_files:
            return

        # 1) Clear old shapes
        self._restore_center_mask_panel()
        self._reset_image_state()

        self.features = []
        self.edge_points = []
        self.edited_edge_points = []

        try:
            self.top_center_frame.unbind("<Configure>")
        except Exception:
            pass
        self.top_center_frame.bind("<Configure>", self.update_mask_display)

        self._clear_ctk_label(self.edge_label)   # <-- use helper
        self._clear_ctk_label(self.mask_label)   # <-- use helper

        # 2) Check that the label widget still exists
        if hasattr(self, 'edge_label') and self.edge_label.winfo_exists():
            self.edge_label.configure(image=None, text="")
            self.edge_label.image = None

        if hasattr(self, 'mask_label') and self.mask_label.winfo_exists():
            self.mask_label.configure(image=None, text="")
            self.mask_label.image = None

        file_path = self.image_files[self.current_index]
        original_image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if original_image is None:
            messagebox.showerror("Error", f"Failed to load image {file_path}", parent=self)
            return

        # Update references
        self.image_path = file_path
        self.filename_label.configure(
            text=f"{os.path.basename(file_path)} ({self.current_index+1} / {len(self.image_files)})"
        )
        self.full_image = original_image.copy()
        self.compute_full_masks(original_image)

        if self.mode != "batch":
            self.process_loaded_image(original_image)

    def next_image(self):
        self._restore_center_mask_panel()
        if self.mode == "ml" and self.image_files:
            if self.current_index < len(self.image_files) - 1:
                self.current_index += 1
                self.load_current_image()
                self.update_image_display()
                self.update_mask_display()
            else:
                messagebox.showinfo("Info", "Already at the last image.", parent=self)

    def prev_image(self):
        self._restore_center_mask_panel()

        if self.mode == "ml" and self.image_files:
            if self.current_index > 0:
                self.current_index -= 1
                self._clear_ctk_label(self.mask_label)   # <-- changed
                self._clear_ctk_label(self.edge_label)   # <-- changed
                self.load_current_image()
                self.update_image_display()
                self.update_mask_display()
            else:
                messagebox.showinfo("Info", "Already at the first image.", parent=self)

    def compute_full_masks(self, original_image):
        if original_image.ndim == 3 and original_image.shape[2] == 4:
            b, g, r, a = cv2.split(original_image)
            full_alpha_mask = (a > 0).astype(np.uint8)
        else:
            full_alpha_mask = np.ones(
                (original_image.shape[0], original_image.shape[1]), dtype=np.uint8)
        kernel_inner_full = np.ones((28, 28), np.uint8)
        self.full_alpha_mask = full_alpha_mask
        self.full_alpha_mask_inner = cv2.erode(
            full_alpha_mask, kernel_inner_full, iterations=1)

    def process_loaded_image(self, original_image):
        if original_image.ndim == 3 and original_image.shape[2] == 4:
            b, g, r, a = cv2.split(original_image)
            alpha = (a > 0).astype(np.uint8)
            b[alpha == 0] = g[alpha == 0] = r[alpha == 0] = 0
            img_bgr = cv2.merge([b, g, r])
        else:
            if original_image.ndim == 2:
                img_bgr = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
            elif original_image.ndim == 3 and original_image.shape[2] >= 3:
                img_bgr = original_image[:, :, :3]
            else:
                img_bgr = original_image
            alpha = np.ones(
                (img_bgr.shape[0], img_bgr.shape[1]), dtype=np.uint8)

        h, w = img_bgr.shape[:2]
        max_dim = 600
        self.scale = min(max_dim / w, max_dim / h, 1.0)
        if self.scale < 1.0:
            new_size = (int(w*self.scale), int(h*self.scale))
            self.cv_image = cv2.resize(
                img_bgr, new_size, interpolation=cv2.INTER_AREA)
            self.alpha_mask = cv2.resize(
                alpha, new_size, interpolation=cv2.INTER_NEAREST)
        else:
            self.cv_image = img_bgr.copy()
            self.alpha_mask = alpha

        cv_rgb = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(cv_rgb)

        self.update_image_display()

    # -------------- MASK CALCULATION --------------

    def calculate_mask(self):
        if not isinstance(self.cv_image, np.ndarray):
            return
        if not self.use_bbox.get():
            bbox_mask = (self.alpha_mask * 255).astype(np.uint8)
        else:
            bbox_text = self.bbox_entry.get().strip("()")
            if not bbox_text:
                bbox_mask = (self.alpha_mask * 255).astype(np.uint8)
            else:
                try:
                    x, y, w, h = map(int, bbox_text.split(","))
                except Exception as e:
                    messagebox.showerror(
                        "Error", f"Invalid bounding box format: {e}", parent=self)
                    return
                x = int(x * self.scale)
                y = int(y * self.scale)
                w = int(w * self.scale)
                h = int(h * self.scale)
                bbox_mask = np.zeros(self.cv_image.shape[:2], dtype=np.uint8)
                bbox_mask[y:y+h, x:x+w] = 255
                bbox_mask = cv2.bitwise_and(
                    bbox_mask, (self.alpha_mask * 255).astype(np.uint8))

            if self.use_inner_mask.get():
                kernel_inner = np.ones((20, 20), np.uint8)
                self.inner_bbox_mask = cv2.erode(
                    bbox_mask, kernel_inner, iterations=1)
            else:
                self.inner_bbox_mask = None

        masked_bgr = cv2.bitwise_and(
            self.cv_image, self.cv_image, mask=bbox_mask)
        hsv = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2HSV)
        h_chan, s_chan, v_chan = cv2.split(hsv)
        if self.enable_enhancements.get():
            s_mult = self.s_multiplier_slider.get() / 100.0
            v_mult = self.v_multiplier_slider.get() / 100.0
            s_chan = np.clip(s_chan.astype(np.float32) *
                             s_mult, 0, 255).astype(np.uint8)
            v_chan = np.clip(v_chan.astype(np.float32) *
                             v_mult, 0, 255).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(32, 32))
            v_chan = clahe.apply(v_chan)

        enhanced_hsv = cv2.merge([h_chan, s_chan, v_chan])
        h_low = int(self.h_low_slider.get())
        s_low = int(self.s_low_slider.get())
        v_low = int(self.v_low_slider.get())
        h_high = int(self.h_high_slider.get())
        s_high = int(self.s_high_slider.get())
        v_high = int(self.v_high_slider.get())
        mask1 = cv2.inRange(enhanced_hsv, (h_low, s_low,
                            v_low), (h_high, s_high, v_high))

        if self.use_dual_hsv.get():
            h2_low = int(self.h2_low_slider.get())
            s2_low = int(self.s2_low_slider.get())
            v2_low = int(self.v2_low_slider.get())
            h2_high = int(self.h2_high_slider.get())
            s2_high = int(self.s2_high_slider.get())
            v2_high = int(self.v2_high_slider.get())
            mask2 = cv2.inRange(
                enhanced_hsv, (h2_low, s2_low, v2_low), (h2_high, s2_high, v2_high))
            mask_full = cv2.bitwise_or(mask1, mask2)
        else:
            mask_full = mask1

        kernel = np.ones((5, 5), np.uint8)
        mask_clean = cv2.morphologyEx(mask_full, cv2.MORPH_CLOSE, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel)

        # Re-apply alpha/bbox mask to ensure transparent regions are excluded
        mask_clean = cv2.bitwise_and(mask_clean, bbox_mask)

        # ── Apply AOI constraint from AOI / Profile Filter (if active) ──
        if getattr(self, 'use_aoi_filter', None) and self.use_aoi_filter.get() \
                and getattr(self, 'aoi_mask', None) is not None:
            aoi_cv = self.aoi_mask
            if aoi_cv.shape[:2] != mask_clean.shape[:2]:
                aoi_cv = cv2.resize(aoi_cv, (mask_clean.shape[1], mask_clean.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
            mask_clean = cv2.bitwise_and(mask_clean, aoi_cv)

        if self.do_invert_mask.get():
            mask_clean = cv2.bitwise_not(mask_clean)

        self.current_mask = mask_clean

        if self.mode != "batch":
            self.display_mask()

    def display_mask(self):
        if self.current_mask is None or self.mode == "batch":
            return
        to_show = self.current_mask
        h0, w0 = to_show.shape[:2]
        panel_w = self.top_center_frame.winfo_width()
        panel_h = self.top_center_frame.winfo_height()
        if panel_w < 1 or panel_h < 1:
            fit_w, fit_h = w0, h0
        else:
            fit_w, fit_h = self._fit_size(w0, h0, panel_w, panel_h)
        resized = cv2.resize(to_show, (fit_w, fit_h), interpolation=cv2.INTER_AREA)
        mask_rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        pil_img = Image.fromarray(mask_rgb)
        ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(
            fit_w, fit_h))
        self.mask_label.configure(image=ctk_img)
        self.mask_label.image = ctk_img

    # -------------- BOUNDARY / POLYGON EXTRACTION --------------

    def _get_clean_mask_for_extraction(self):
        if self.current_mask is None:
            return None
        clean_mask = self._ensure_binary_u8(self.current_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel)
        return clean_mask

    def _get_contour_size_limits(self):
        """Read min/max contour size from advanced settings. Returns (min, max) as floats."""
        min_size = 0
        max_size = 0
        if getattr(self, 'advanced_check_var', None) and self.advanced_check_var.get():
            try:
                val = self.min_contour_entry.get().strip()
                if val:
                    min_size = float(val)
            except Exception:
                pass
            try:
                val = self.max_contour_entry.get().strip()
                if val:
                    max_size = float(val)
            except Exception:
                pass
        return min_size, max_size

    def _get_exclusion_border_mask(self, shape_hw, thickness=3):
        h, w = shape_hw
        border_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(border_mask, (0, 0), (w - 1, h - 1), 255, thickness=max(1, int(thickness)))

        if getattr(self, 'use_aoi_filter', None) and self.use_aoi_filter.get() \
                and getattr(self, 'aoi_mask', None) is not None:
            aoi_mask = self.aoi_mask
            if aoi_mask.shape[:2] != (h, w):
                aoi_mask = cv2.resize(aoi_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            aoi_mask = self._ensure_binary_u8(aoi_mask)
            aoi_border = cv2.morphologyEx(
                aoi_mask, cv2.MORPH_GRADIENT,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            )
            border_mask = cv2.bitwise_or(border_mask, aoi_border)

        return border_mask

    def _contour_touch_ratio(self, contour, border_mask):
        if contour is None or len(contour) == 0:
            return 1.0
        pts = contour.reshape(-1, 2)
        h, w = border_mask.shape[:2]
        xs = np.clip(pts[:, 0], 0, w - 1)
        ys = np.clip(pts[:, 1], 0, h - 1)
        return float(np.mean(border_mask[ys, xs] > 0))

    def _select_primary_contour(self, contours, border_mask, metric="length",
                                prefer_interior=True, min_size=0, max_size=0):
        candidates = []
        for cnt in contours:
            if cnt is None or len(cnt) < 3:
                continue
            if metric == "area":
                score = float(cv2.contourArea(cnt))
            else:
                score = float(cv2.arcLength(cnt, True))
            if score <= 0:
                continue

            # Apply min/max contour size filter (advanced settings)
            if min_size > 0 and score < min_size:
                continue
            if max_size > 0 and score > max_size:
                continue

            touch_ratio = self._contour_touch_ratio(cnt, border_mask)
            candidates.append((cnt, score, touch_ratio))

        if not candidates:
            return None

        # Sort by score descending (largest first)
        candidates.sort(key=lambda item: item[1], reverse=True)
        overall_best = candidates[0]

        if prefer_interior:
            interior = [item for item in candidates if item[2] <= 0.05]
            if interior:
                interior.sort(key=lambda item: item[1], reverse=True)
                best_interior = interior[0]
                # Only prefer interior if it's at least 20% the size of the
                # overall largest — otherwise the interior contour is just
                # a small noise blob and the real feature touches the border.
                if best_interior[1] >= overall_best[1] * 0.20:
                    return best_interior[0]

        # Fall back to overall largest (lowest border-touch ratio as tiebreak)
        candidates.sort(key=lambda item: (-item[1], item[2]))
        return candidates[0][0]

    def _contour_to_full_coords(self, contour, approx_factor=0.0015):
        if contour is None or not isinstance(self.full_image, np.ndarray) or not isinstance(self.cv_image, np.ndarray):
            return []

        contour_len = cv2.arcLength(contour, True)
        epsilon = max(0.5, approx_factor * contour_len)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        scale_x = self.full_image.shape[1] / self.cv_image.shape[1]
        scale_y = self.full_image.shape[0] / self.cv_image.shape[0]
        pts_full = []
        for pt in approx.reshape(-1, 2):
            rx = int(pt[0] * scale_x)
            ry = int(pt[1] * scale_y)
            pts_full.append([rx, ry])
        return pts_full

    def _filter_points_to_inner_masks(self, points):
        valid_points = []
        if self.use_bbox.get() and self.use_inner_mask.get():
            try:
                x, y, w, h = map(int, self.bbox_entry.get().strip("()").split(","))
                margin = max(1, int(round(20 / max(float(self.scale), 1e-6))))
                x_min = x + margin
                y_min = y + margin
                x_max = x + max(1, w - margin)
                y_max = y + max(1, h - margin)
                for px, py in points:
                    if x_min <= px <= x_max and y_min <= py <= y_max:
                        valid_points.append([px, py])
                if valid_points:
                    return valid_points
            except Exception:
                pass

        if self.full_alpha_mask_inner is not None:
            for (x, y) in points:
                if 0 <= x < self.full_alpha_mask_inner.shape[1] and 0 <= y < self.full_alpha_mask_inner.shape[0]:
                    if self.full_alpha_mask_inner[y, x] > 0:
                        valid_points.append([x, y])
        else:
            valid_points = [list(pt) for pt in points]
        return valid_points

    def extract_boundary(self):
        clean_mask = self._get_clean_mask_for_extraction()
        if clean_mask is None or cv2.countNonZero(clean_mask) == 0:
            return

        contours, _ = cv2.findContours(clean_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return

        # Read advanced contour size filters
        min_size, max_size = self._get_contour_size_limits()

        border_mask = self._get_exclusion_border_mask(clean_mask.shape[:2], thickness=3)

        # Diagnostic: show contour stats
        scored = [(cv2.arcLength(c, True), self._contour_touch_ratio(c, border_mask))
                  for c in contours if c is not None and len(c) >= 3]
        scored.sort(key=lambda x: x[0], reverse=True)
        n_total = len(scored)
        n_interior = sum(1 for _, tr in scored if tr <= 0.05)
        if scored:
            print(f"[Boundary] {n_total} contours found "
                  f"({n_interior} interior, {n_total - n_interior} border-touching). "
                  f"Largest arc length: {scored[0][0]:.0f} px.")
        if min_size > 0 or max_size > 0:
            print(f"[Boundary] Advanced filter: min={min_size}, max={max_size}")

        largest = self._select_primary_contour(
            contours, border_mask, metric="length", prefer_interior=True,
            min_size=min_size, max_size=max_size)
        if largest is None:
            return

        rescaled_points = self._contour_to_full_coords(largest, approx_factor=0.0012)
        valid_points = self._filter_points_to_inner_masks(rescaled_points)
        if len(valid_points) < 2:
            return

        touch_ratio = self._contour_touch_ratio(largest, border_mask)
        if touch_ratio <= 0.05 and valid_points[0] != valid_points[-1]:
            valid_points = valid_points + [valid_points[0].copy()]

        self.edge_points = valid_points
        self.features = [("polyline", valid_points.copy())]

        if self.mode != "batch":
            self.update_edge_display()

    def _get_best_mask_for_polygon(self):
        """
        Return (is_full_res, mask) — the best available mask for polygon extraction.

        Prefers the full-resolution color_pick_mask (from multi-sample colour
        selection) so that polygon vertices are not degraded by the cv_image
        downscale.  Falls back to current_mask at cv_image resolution.
        """
        # Prefer full-res colour picker mask when available
        cpick = getattr(self, 'color_pick_mask', None)
        if cpick is not None and cpick.size > 0 and cv2.countNonZero(cpick) > 0:
            return True, cpick.copy()

        # Fall back to current_mask (cv_image resolution)
        if self.current_mask is not None and cv2.countNonZero(self.current_mask) > 0:
            return False, self.current_mask.copy()

        return False, None

    def extract_polygon(self):
        """
        Extract polygon(s) from the current mask.

        Uses the full-resolution color_pick_mask when available (from
        multi-sample colour selection), otherwise falls back to the
        downscaled current_mask.

        Workflow
        -------
        1. Pick the best available mask (full-res preferred).
        2. Morphological cleanup — close small gaps, remove noise.
        3. Fill interior holes by redrawing external contours as filled.
        4. Find all external contours, keep those above a minimum area.
        5. Simplify each with Douglas-Peucker and store as polygon features.
        """
        full_res, mask = self._get_best_mask_for_polygon()
        if mask is None:
            print("[Extract Polygon] No valid mask available.")
            return

        mask = self._ensure_binary_u8(mask)

        # --- morphological cleanup (more aggressive than boundary) ---
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k_open)

        h, w = mask.shape[:2]

        # --- fill interior holes ---
        # Draw every external contour as filled → holes disappear automatically
        ext_cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not ext_cnts:
            print("[Extract Polygon] No contours found after cleanup.")
            return
        filled = np.zeros_like(mask)
        cv2.drawContours(filled, ext_cnts, -1, 255, thickness=cv2.FILLED)

        # --- find contours on the hole-free mask ---
        contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            print("[Extract Polygon] No contours on filled mask.")
            return

        # --- filter by minimum area (0.1 % of image) and sort largest-first ---
        min_area = h * w * 0.001
        scored = [(cnt, cv2.contourArea(cnt)) for cnt in contours
                  if cv2.contourArea(cnt) >= min_area]
        if not scored:
            print("[Extract Polygon] No contours above minimum area.")
            return
        scored.sort(key=lambda x: x[1], reverse=True)

        # --- build polygon features ---
        self.features = []
        self.edge_points = []

        for cnt, area in scored:
            perimeter = cv2.arcLength(cnt, True)
            epsilon = max(1.0, 0.002 * perimeter)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) < 3:
                continue

            if full_res:
                # Contour is already in full-image coords
                pts = [[int(pt[0][0]), int(pt[0][1])] for pt in approx]
            else:
                # Scale from cv_image coords → full-image coords
                scale_x = self.full_image.shape[1] / self.cv_image.shape[1]
                scale_y = self.full_image.shape[0] / self.cv_image.shape[0]
                pts = [[int(pt[0][0] * scale_x), int(pt[0][1] * scale_y)]
                       for pt in approx]

            # NOTE: We intentionally skip _filter_points_to_inner_masks() here.
            # That filter erodes alpha boundaries by 28 px and rejects vertices
            # near the image edge — appropriate for polyline/boundary detection
            # but destructive for polygons whose contour legitimately follows the
            # panoramic border.  Border exclusion is already handled upstream in
            # the colour-picker working mask (alpha + AOI).

            # Remove duplicate closing point if present
            if len(pts) >= 2 and pts[0] == pts[-1]:
                pts = pts[:-1]
            if len(pts) < 3:
                continue

            self.features.append(("polygon", pts))
            if not self.edge_points:
                self.edge_points = pts  # keep first (largest) for back-compat

        if not self.features:
            print("[Extract Polygon] Could not form valid polygon(s).")
            return

        n_polys = len(self.features)
        n_verts = len(self.features[0][1])
        print(f"[Extract Polygon] Extracted {n_polys} polygon(s); "
              f"largest has {n_verts} vertices.")

        if self.mode != "batch":
            self.update_edge_display()

    def calculate_edge(self):
        """Backward-compatible alias for boundary extraction."""
        self.extract_boundary()

    # -------------- SKELETON / ML MASK EDGE EXTRACTION --------------

    def _centerline_polyline_from_skeleton(self, skel_bin):
        """
        Given a 1-pixel wide skeleton (uint8 {0,255} or {0,1}), return a single
        ordered polyline (list of [x,y] in cv_image coords) along the longest
        8-connected path. No contours used → no double line.
        """
        sk = (skel_bin > 0).astype(np.uint8)
        rows, cols = np.where(sk)
        if rows.size == 0:
            return []

        # map pixel -> node id
        H, W = sk.shape
        idx_map = -np.ones((H, W), dtype=np.int32)
        idx_map[rows, cols] = np.arange(rows.size, dtype=np.int32)

        # build adjacency (8-neighbors)
        offsets = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        adj = [[] for _ in range(rows.size)]
        deg = np.zeros(rows.size, dtype=np.int32)

        for i, (r, c) in enumerate(zip(rows, cols)):
            for dr, dc in offsets:
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W and sk[rr, cc]:
                    j = idx_map[rr, cc]
                    if j >= 0:
                        adj[i].append(int(j))
            deg[i] = len(adj[i])

        # pick start: an endpoint if available (degree 1), else any node
        endpoints = np.where(deg == 1)[0]
        start = int(endpoints[0]) if endpoints.size > 0 else 0

        # BFS to farthest node
        def bfs_far(src):
            from collections import deque
            q = deque([src])
            parent = {src: -1}
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if v not in parent:
                        parent[v] = u
                        q.append(v)
            far = u  # last popped is a farthest
            return far, parent

        a, _ = bfs_far(start)      # farthest from start
        b, pa = bfs_far(a)         # farthest from a (graph "diameter")
        # reconstruct path a→b using parents from the second BFS
        path = []
        cur = b
        while cur != -1:
            path.append(cur)
            cur = pa[cur]
        path.reverse()

        # convert node indices → (x,y) in cv_image coords (x=col, y=row)
        poly = [[int(cols[i]), int(rows[i])] for i in path]
        return poly if len(poly) >= 2 else []


    def _detect_edge_from_current_mask_robust(self):
        """
        Make a single editable polyline from current_mask by skeletonizing and
        taking the longest 8-connected path of skeleton pixels.
        Populates self.edge_points & self.features, updates right panel.
        """
        if self.current_mask is None or self.cv_image is None or self.full_image is None:
            return False

        # strict binary
        mask = self._ensure_binary_u8(self.current_mask)
        if cv2.countNonZero(mask) == 0:
            self.edge_points, self.features = [], []
            self.update_edge_display()
            return False

        # Skeletonize (prefer skimage; fallback to OpenCV thinning or distance ridge)
        try:
            from skimage.morphology import skeletonize
            skel = skeletonize((mask > 0)).astype(np.uint8) * 255
        except Exception:
            try:
                import cv2.ximgproc as xip
                skel = xip.thinning(mask, thinningType=xip.THINNING_ZHANGSUEN)
            except Exception:
                dt = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
                dt = cv2.normalize(dt, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                _, skel = cv2.threshold(dt, max(1, int(dt.max() * 0.6)), 255, cv2.THRESH_BINARY)

        # small close to connect tiny gaps but keep 1-px width
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        skel = cv2.morphologyEx(skel, cv2.MORPH_CLOSE, ker, iterations=1)

        # === centerline polyline in cv_image coords ===
        pts_cv = self._centerline_polyline_from_skeleton(skel)
        if len(pts_cv) < 2:
            self.edge_points, self.features = [], []
            self.update_edge_display()
            return False

        # filter in CV space (inner_bbox_mask is in CV space)
        if self.use_bbox.get() and self.use_inner_mask.get() and self.inner_bbox_mask is not None:
            kept_cv = []
            H, W = self.inner_bbox_mask.shape[:2]
            for x, y in pts_cv:
                if 0 <= x < W and 0 <= y < H and self.inner_bbox_mask[y, x] > 0:
                    kept_cv.append([x, y])
            if len(kept_cv) >= 2:
                pts_cv = kept_cv

        # scale to full_image coords (we render/export there)
        sx = self.full_image.shape[1] / self.cv_image.shape[1]
        sy = self.full_image.shape[0] / self.cv_image.shape[0]
        pts_full = [[int(x * sx), int(y * sy)] for x, y in pts_cv]

        # Store the results
        self.edge_points = pts_full
        self.features = [("polyline", pts_full)]

        # Force geometry update and delay display refresh to ensure frame is properly sized
        try:
            self.update_idletasks()
        except Exception:
            pass
        self.after(50, self.update_edge_display)

        return True


    def calculate_edge_with_ml_mask(self):
        """
        ML mask workflow:
          1) Load the ML mask
          2) Resize it to FULL image size (for extraction) and optionally clip by bbox
          3) Create a cv-sized copy only for display
          4) Extract shoreline from FULL-res mask (avoids thin-line loss)
        """
        if self.full_image is None:
            messagebox.showwarning("Image", "Please load an image first.", parent=self)
            return

        # Resolve mask path
        mask_path = None
        if self.mode == "individual":
            mask_path = self.ml_mask_file_path.get().strip()
            if not mask_path:
                messagebox.showwarning("ML mask", "Please load an associated mask.", parent=self)
                return
        elif self.mode == "ml":
            if not self.image_files:
                messagebox.showwarning("Folder", "Please load an image folder first.", parent=self)
                return
            folder = self.ml_mask_folder_path.get().strip()
            if not folder:
                messagebox.showwarning("Mask folder", "Please load the associated mask folder.", parent=self)
                return
            cur_img = self.image_files[self.current_index]
            base = os.path.splitext(os.path.basename(cur_img))[0]
            mask_path = self._best_match_in_folder(base, folder, self.common_name_len_var.get())
            if mask_path is None:
                messagebox.showerror("Mask match", f"No matching mask found for:\n{os.path.basename(cur_img)}", parent=self)
                return
        else:
            messagebox.showwarning("Mode", "Extract Boundary with Mask is not available in batch mode.", parent=self)
            return

        # Read mask (your _read_mask_image already returns binary-ish)
        m = self._read_mask_image(mask_path)
        if m is None:
            messagebox.showerror("Mask", f"Failed to read mask:\n{mask_path}", parent=self)
            return

        # --- FULL resolution mask for extraction ---
        Hf, Wf = self.full_image.shape[:2]
        m_full = self._ensure_binary_u8(m)
        if m_full.shape[:2] != (Hf, Wf):
            m_full = cv2.resize(m_full, (Wf, Hf), interpolation=cv2.INTER_NEAREST)

        # Optional bbox clip (bbox entry is in ORIGINAL coords -> use directly, no self.scale here)
        if self.use_bbox.get():
            try:
                x, y, w, h = map(int, self.bbox_entry.get().strip("()").split(","))
                x = max(0, x); y = max(0, y)
                w = max(1, w); h = max(1, h)
                bbox_full = np.zeros((Hf, Wf), dtype=np.uint8)
                bbox_full[y:y+h, x:x+w] = 255
                m_full = cv2.bitwise_and(m_full, bbox_full)
            except Exception:
                pass

        # If after clipping/binarizing nothing remains -> popup reason is clear
        if cv2.countNonZero(m_full) == 0:
            self.current_mask = np.zeros(self.cv_image.shape[:2], dtype=np.uint8) if self.cv_image is not None else None
            if self.current_mask is not None:
                self.display_mask()
            messagebox.showwarning("Edge", "Mask became empty after binarization/bbox clipping.", parent=self)
            return

        # --- CV resolution mask only for display ---
        if self.cv_image is not None:
            hc, wc = self.cv_image.shape[:2]
            m_cv = cv2.resize(m_full, (wc, hc), interpolation=cv2.INTER_NEAREST)
            self.current_mask = self._ensure_binary_u8(m_cv)
            self.display_mask()

        # --- Extract edge from FULL-res mask (avoid thin-line downscale failure) ---
        # Reuse your robust skeleton approach, but run it in full coords:
        try:
            from skimage.morphology import skeletonize
            skel = skeletonize((m_full > 0)).astype(np.uint8) * 255
        except Exception:
            try:
                import cv2.ximgproc as xip
                skel = xip.thinning(m_full, thinningType=xip.THINNING_ZHANGSUEN)
            except Exception:
                dt = cv2.distanceTransform((m_full > 0).astype(np.uint8), cv2.DIST_L2, 3)
                dt = cv2.normalize(dt, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                _, skel = cv2.threshold(dt, max(1, int(dt.max() * 0.6)), 255, cv2.THRESH_BINARY)

        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        skel = cv2.morphologyEx(skel, cv2.MORPH_CLOSE, ker, iterations=1)

        pts_full = self._centerline_polyline_from_skeleton(skel)
        if len(pts_full) < 2:
            messagebox.showwarning("Edge", "No boundary could be extracted from the mask.", parent=self)
            return
        
        if len(pts_full) >= 3:
            cnt = np.array(pts_full, dtype=np.int32).reshape((-1, 1, 2))
            epsilon = max(0.3, 0.001 * cv2.arcLength(cnt, False))
            approx = cv2.approxPolyDP(cnt, epsilon, False)
            pts_full = [[int(pt[0][0]), int(pt[0][1])] for pt in approx]

        # Store ONE polyline feature in FULL coords
        self.edge_points = pts_full
        self.features = [("polyline", pts_full.copy())]
        self.update_edge_display()


    def extract_polygon_with_ml_mask(self):
        """
        ML mask workflow — polygon extraction variant.
        Loads the ML predicted mask, resizes to the source image dimensions,
        and extracts polygon feature(s) instead of a boundary/polyline.

        The mask may differ in size from the source image; mapping is done
        by simple ratio scaling:
            x_src = round(x_mask * (src_w / mask_w))
            y_src = round(y_mask * (src_h / mask_h))
        """
        if self.full_image is None:
            messagebox.showwarning("Image", "Please load an image first.", parent=self)
            return

        # ── Resolve mask path (same logic as calculate_edge_with_ml_mask) ──
        mask_path = None
        if self.mode == "individual":
            mask_path = self.ml_mask_file_path.get().strip()
            if not mask_path:
                messagebox.showwarning("ML mask", "Please load an associated mask.", parent=self)
                return
        elif self.mode == "ml":
            if not self.image_files:
                messagebox.showwarning("Folder", "Please load an image folder first.", parent=self)
                return
            folder = self.ml_mask_folder_path.get().strip()
            if not folder:
                messagebox.showwarning("Mask folder", "Please load the associated mask folder.", parent=self)
                return
            cur_img = self.image_files[self.current_index]
            base = os.path.splitext(os.path.basename(cur_img))[0]
            mask_path = self._best_match_in_folder(base, folder, self.common_name_len_var.get())
            if mask_path is None:
                messagebox.showerror("Mask match", f"No matching mask found for:\n{os.path.basename(cur_img)}", parent=self)
                return
        else:
            messagebox.showwarning("Mode", "Extract Polygon with Mask is not available in batch mode.", parent=self)
            return

        # ── Read mask ──
        m = self._read_mask_image(mask_path)
        if m is None:
            messagebox.showerror("Mask", f"Failed to read mask:\n{mask_path}", parent=self)
            return

        # ── Resize mask to full-image dimensions ──
        Hf, Wf = self.full_image.shape[:2]
        m_full = self._ensure_binary_u8(m)
        if m_full.shape[:2] != (Hf, Wf):
            m_full = cv2.resize(m_full, (Wf, Hf), interpolation=cv2.INTER_NEAREST)
            print(f"[ML Polygon] Mask resized from {m.shape[:2]} → ({Hf}, {Wf})")

        # ── Optional bbox clip ──
        if self.use_bbox.get():
            try:
                x, y, w, h = map(int, self.bbox_entry.get().strip("()").split(","))
                x = max(0, x); y = max(0, y)
                w = max(1, w); h = max(1, h)
                bbox_full = np.zeros((Hf, Wf), dtype=np.uint8)
                bbox_full[y:y+h, x:x+w] = 255
                m_full = cv2.bitwise_and(m_full, bbox_full)
            except Exception:
                pass

        if cv2.countNonZero(m_full) == 0:
            self.current_mask = np.zeros(self.cv_image.shape[:2], dtype=np.uint8) if self.cv_image is not None else None
            if self.current_mask is not None:
                self.display_mask()
            messagebox.showwarning("Polygon", "Mask became empty after binarization/bbox clipping.", parent=self)
            return

        # ── Update display mask ──
        if self.cv_image is not None:
            hc, wc = self.cv_image.shape[:2]
            m_cv = cv2.resize(m_full, (wc, hc), interpolation=cv2.INTER_NEAREST)
            self.current_mask = self._ensure_binary_u8(m_cv)
            self.display_mask()

        # ── Morphological cleanup ──
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        m_full = cv2.morphologyEx(m_full, cv2.MORPH_CLOSE, k_close)
        m_full = cv2.morphologyEx(m_full, cv2.MORPH_OPEN,  k_open)

        # ── Fill interior holes ──
        ext_cnts, _ = cv2.findContours(m_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not ext_cnts:
            messagebox.showwarning("Polygon", "No contours found in mask after cleanup.", parent=self)
            return
        filled = np.zeros_like(m_full)
        cv2.drawContours(filled, ext_cnts, -1, 255, thickness=cv2.FILLED)

        # ── Find contours on hole-free mask ──
        contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            messagebox.showwarning("Polygon", "No contours on filled mask.", parent=self)
            return

        # ── Filter by minimum area (0.1% of image) and sort largest-first ──
        min_area = Hf * Wf * 0.001
        scored = [(cnt, cv2.contourArea(cnt)) for cnt in contours
                  if cv2.contourArea(cnt) >= min_area]
        if not scored:
            messagebox.showwarning("Polygon", "No contours above minimum area.", parent=self)
            return
        scored.sort(key=lambda x: x[1], reverse=True)

        # ── Build polygon features (already in full-image coords) ──
        self.features = []
        self.edge_points = []

        for cnt, area in scored:
            perimeter = cv2.arcLength(cnt, True)
            epsilon = max(0.3, 0.001 * perimeter)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) < 3:
                continue

            pts = [[int(pt[0][0]), int(pt[0][1])] for pt in approx]

            # Remove duplicate closing point if present
            if len(pts) >= 2 and pts[0] == pts[-1]:
                pts = pts[:-1]
            if len(pts) < 3:
                continue

            self.features.append(("polygon", pts))
            if not self.edge_points:
                self.edge_points = pts

        if not self.features:
            messagebox.showwarning("Polygon", "Could not form valid polygon(s) from mask.", parent=self)
            return

        n_polys = len(self.features)
        n_verts = len(self.features[0][1])
        print(f"[ML Polygon] Extracted {n_polys} polygon(s) from mask; "
              f"largest has {n_verts} vertices.")

        if self.mode != "batch":
            self.update_edge_display()


    # ────────────────────────────────────────────────────────────────────────────

    #  DROPDOWN 3: Colour Picker — detection methods
    # ────────────────────────────────────────────────────────────────────────────

    def _get_color_pick_patch_radius(self):
        value = getattr(self, "color_pick_patch_radius", None)
        try:
            if hasattr(value, "get"):
                value = value.get()
            radius = int(value)
        except Exception:
            radius = 7
        return max(1, min(radius, 50))

    def _get_color_pick_working_mask(self, shape_hw):
        h, w = shape_hw
        working_mask = None

        # Start with the image's alpha mask if available — excludes transparent /
        # black border regions (common in panoramic or warped images) from
        # colour classification so they don't contaminate the result.
        alpha = getattr(self, 'full_alpha_mask', None)
        if alpha is not None:
            if alpha.shape[:2] != (h, w):
                alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_NEAREST)
            working_mask = (alpha > 0).astype(np.uint8) * 255

        # Intersect with AOI filter if active
        if getattr(self, 'use_aoi_filter', None) and self.use_aoi_filter.get() \
                and getattr(self, 'aoi_mask', None) is not None:
            aoi_full = self.aoi_mask
            if aoi_full.shape[:2] != (h, w):
                aoi_full = cv2.resize(aoi_full, (w, h), interpolation=cv2.INTER_NEAREST)
            aoi_bin = (aoi_full > 0).astype(np.uint8) * 255
            if working_mask is not None:
                working_mask = cv2.bitwise_and(working_mask, aoi_bin)
            else:
                working_mask = aoi_bin

        return working_mask

    def _normalise_color_pick_points(self):
        pts = getattr(self, "color_pick_points", {})
        if not isinstance(pts, dict):
            pts = {}

        def _as_point_list(value):
            out = []
            if isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        try:
                            out.append((int(item[0]), int(item[1])))
                        except Exception:
                            pass
            return out

        remove_pts = _as_point_list(pts.get("remove", []))
        keep_pts = _as_point_list(pts.get("keep", []))

        legacy_a = pts.get("point_a")
        legacy_b = pts.get("point_b")
        if not keep_pts and isinstance(legacy_a, (list, tuple)) and len(legacy_a) >= 2:
            try:
                keep_pts = [(int(legacy_a[0]), int(legacy_a[1]))]
            except Exception:
                keep_pts = []
        if not remove_pts and isinstance(legacy_b, (list, tuple)) and len(legacy_b) >= 2:
            try:
                remove_pts = [(int(legacy_b[0]), int(legacy_b[1]))]
            except Exception:
                remove_pts = []

        self.color_pick_points = {"remove": remove_pts, "keep": keep_pts}
        return self.color_pick_points

    def _collect_color_pick_vectors(self, lab_img, points, patch_r, working_mask=None):
        h, w = lab_img.shape[:2]
        vectors = []
        for x, y in points:
            x = max(0, min(int(x), w - 1))
            y = max(0, min(int(y), h - 1))
            y1, y2 = max(0, y - patch_r), min(h, y + patch_r + 1)
            x1, x2 = max(0, x - patch_r), min(w, x + patch_r + 1)
            patch = lab_img[y1:y2, x1:x2]
            if patch.size == 0:
                continue
            flat_patch = patch.reshape(-1, lab_img.shape[2])
            if working_mask is not None:
                mask_patch = working_mask[y1:y2, x1:x2].reshape(-1) > 0
                if np.any(mask_patch):
                    flat_patch = flat_patch[mask_patch]
            if flat_patch.size == 0:
                continue
            vectors.append(flat_patch)
        if not vectors:
            return np.empty((0, lab_img.shape[2]), dtype=np.float32)
        return np.vstack(vectors).astype(np.float32)

    def _finalise_color_pick_mask(self, selected_mask, working_mask, method_name, quiet=False):
        selected_mask = self._ensure_binary_u8(selected_mask)

        kernel = np.ones((5, 5), np.uint8)
        selected_mask = cv2.morphologyEx(selected_mask, cv2.MORPH_CLOSE, kernel)
        selected_mask = cv2.morphologyEx(selected_mask, cv2.MORPH_OPEN, kernel)

        output_mode = getattr(self, "color_pick_output_mode", None)
        output_mode = output_mode.get() if hasattr(output_mode, "get") else "Remove selection"

        if output_mode in ("Keep selection only", "Keep selected class"):
            result_mask = selected_mask
        else:
            if working_mask is not None:
                result_mask = working_mask.copy()
                result_mask[selected_mask > 0] = 0
            else:
                result_mask = cv2.bitwise_not(selected_mask)

        self.color_pick_mask = result_mask
        if self.cv_image is not None:
            hc, wc = self.cv_image.shape[:2]
            self.current_mask = cv2.resize(result_mask, (wc, hc), interpolation=cv2.INTER_NEAREST)
            if self.mode != "batch":
                self.display_mask()
        else:
            self.current_mask = result_mask

        if not quiet:
            selected_count = int(cv2.countNonZero(selected_mask))
            final_count = int(cv2.countNonZero(result_mask))
            if working_mask is not None:
                total = int(cv2.countNonZero(working_mask))
            else:
                total = int(selected_mask.shape[0] * selected_mask.shape[1])
            print(
                f"[Color Picker] {method_name}: selected={selected_count}/{total} "
                f"({100 * selected_count / max(total, 1):.1f}%), "
                f"output={output_mode}, final_mask={final_count}px"
            )
        return result_mask

    def _color_pick_detect(self, quiet=False):
        """Run multi-sample colour selection using remove / keep sample points."""
        if self.full_image is None:
            if not quiet:
                messagebox.showwarning("Color Picker", "No image loaded.", parent=self)
            return False

        pts = self._normalise_color_pick_points()
        remove_pts = pts.get("remove", [])
        keep_pts = pts.get("keep", [])

        if not remove_pts:
            if not quiet:
                messagebox.showwarning("Color Picker", "Add at least one remove-sample first.", parent=self)
            return False

        method = self.color_pick_method.get()
        img = self.full_image
        h, w = img.shape[:2]
        working_mask = self._get_color_pick_working_mask((h, w))

        if method == "GrabCut":
            return self._color_pick_grabcut(img, remove_pts, keep_pts, working_mask, quiet=quiet)
        return self._color_pick_color_distance(img, remove_pts, keep_pts, working_mask, quiet=quiet)

    def _color_pick_color_distance(self, img, remove_pts, keep_pts, working_mask, quiet=False):
        """Classify pixels in Lab space using multiple remove / keep colour samples."""
        if img.ndim == 3:
            lab = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2Lab).astype(np.float32)
        else:
            lab = img.astype(np.float32)
            if lab.ndim == 2:
                lab = lab[:, :, np.newaxis]

        patch_r = self._get_color_pick_patch_radius()
        remove_vectors = self._collect_color_pick_vectors(lab, remove_pts, patch_r, working_mask)
        if remove_vectors.size == 0:
            if not quiet:
                messagebox.showwarning("Color Picker", "Could not extract valid remove-samples.", parent=self)
            return False

        keep_vectors = self._collect_color_pick_vectors(lab, keep_pts, patch_r, working_mask)

        flat = lab.reshape(-1, lab.shape[2])
        remove_mean = remove_vectors.mean(axis=0)
        dist_remove = np.linalg.norm(flat - remove_mean, axis=1)

        if keep_vectors.size > 0:
            keep_mean = keep_vectors.mean(axis=0)
            dist_keep = np.linalg.norm(flat - keep_mean, axis=1)
            selected = (dist_remove <= dist_keep).reshape(lab.shape[:2]).astype(np.uint8) * 255
        else:
            sample_dist = np.linalg.norm(remove_vectors - remove_mean, axis=1)
            if sample_dist.size == 0:
                threshold = 8.0
            else:
                p95 = float(np.percentile(sample_dist, 95))
                std = float(sample_dist.std())
                threshold = max(8.0, p95 + 0.75 * std)
            selected = (dist_remove.reshape(lab.shape[:2]) <= threshold).astype(np.uint8) * 255

        if working_mask is not None:
            selected = cv2.bitwise_and(selected, working_mask)

        self._finalise_color_pick_mask(selected, working_mask, "Color Distance", quiet=quiet)
        return True

    def _color_pick_grabcut(self, img, remove_pts, keep_pts, working_mask, quiet=False):
        """Use OpenCV GrabCut with multiple remove / keep sample seeds."""
        if not keep_pts:
            if not quiet:
                messagebox.showwarning(
                    "Color Picker",
                    "GrabCut needs at least one keep-sample as background.\n"
                    "Add keep-samples or use Color Distance.",
                    parent=self
                )
            return False

        h, w = img.shape[:2]
        if img.ndim == 2:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = img[:, :, :3].copy()

        patch_r = max(self._get_color_pick_patch_radius(), 5)
        gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)

        if working_mask is not None:
            gc_mask[working_mask == 0] = cv2.GC_BGD

        for x, y in keep_pts:
            x = max(0, min(int(x), w - 1))
            y = max(0, min(int(y), h - 1))
            gc_mask[max(0, y - patch_r):min(h, y + patch_r + 1),
                    max(0, x - patch_r):min(w, x + patch_r + 1)] = cv2.GC_BGD

        for x, y in remove_pts:
            x = max(0, min(int(x), w - 1))
            y = max(0, min(int(y), h - 1))
            gc_mask[max(0, y - patch_r):min(h, y + patch_r + 1),
                    max(0, x - patch_r):min(w, x + patch_r + 1)] = cv2.GC_FGD

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        try:
            cv2.grabCut(img_bgr, gc_mask, None, bgd_model, fgd_model,
                        5, cv2.GC_INIT_WITH_MASK)
        except cv2.error as e:
            if not quiet:
                messagebox.showerror("GrabCut Error", f"GrabCut failed: {e}", parent=self)
            return False

        selected = np.where(
            (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)

        if working_mask is not None:
            selected = cv2.bitwise_and(selected, working_mask)

        self._finalise_color_pick_mask(selected, working_mask, "GrabCut", quiet=quiet)
        return True

    # -------------- BATCH PROCESS --------------

    def batch_process(self):
        import warnings
        from rasterio.errors import NotGeoreferencedWarning
        warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
        warnings.filterwarnings(
                                "ignore",
                                category=UserWarning,
                                message=".*crs.*was not provided.*"
                            )

        if not self.image_files:
            messagebox.showwarning("Warning", "No images loaded for batch.", parent=self)
            return
        if not getattr(self, '_batch_settings_loaded', False) and self.mode == "batch":
            messagebox.showwarning(
                "Settings Required",
                "Batch mode requires a settings file.\n\n"
                "Use Single Image or Folder Processing to configure your\n"
                "detection pipeline, then Save Settings and load it here.",
                parent=self)
            return
        export_path = self.export_path_entry.get().strip()
        if not export_path:
            messagebox.showerror("Error", "Please select an export folder.", parent=self)
            return

        # Log active pipeline steps
        active_steps = []
        if getattr(self, 'use_aoi_filter', None) and self.use_aoi_filter.get():
            active_steps.append("AOI Filter")
        if getattr(self, 'use_hsv_masking', None) and self.use_hsv_masking.get():
            active_steps.append("HSV Masking")
        if getattr(self, 'use_color_picker', None) and self.use_color_picker.get():
            active_steps.append("Colour Picker")
        if self.use_ml_pred_mask.get():
            active_steps.append("ML Mask")
        print(f"Batch process started — active pipeline: {', '.join(active_steps) or 'HSV only'}")
        geojson_folder = os.path.join(export_path, "geojson")
        overlay_folder = os.path.join(export_path, "shoreline overlay")
        os.makedirs(geojson_folder, exist_ok=True)
        os.makedirs(overlay_folder, exist_ok=True)

        processed_count = 0

        for file_path in self.image_files:
            original = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if original is None:
                continue

            self.full_image = original.copy()
            self.compute_full_masks(original)
            self.cv_image, self.alpha_mask, self.scale = self.prepare_cv_image_for_batch(original)

            # Build AOI mask per-image if AOI filter is active
            if getattr(self, 'use_aoi_filter', None) and self.use_aoi_filter.get():
                try:
                    aoi_min = int(self.aoi_min_entry.get())
                    aoi_max = int(self.aoi_max_entry.get())
                    gray_full = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY) if original.ndim == 3 else original
                    self.aoi_mask = cv2.inRange(gray_full, aoi_min, aoi_max)
                except Exception:
                    self.aoi_mask = None

            # If user opted to use ML masks, try loading; otherwise use HSV pipeline
            if self.use_ml_pred_mask.get() and self.ml_mask_folder_path.get():
                # derive mask for current file
                base = os.path.splitext(os.path.basename(file_path))[0]
                best_mask = self._best_match_in_folder(base, self.ml_mask_folder_path.get(),
                                                       self.common_name_len_var.get())
                if best_mask is not None:
                    m = self._read_mask_image(best_mask)
                    if m is not None:
                        m = cv2.resize(m, (self.cv_image.shape[1], self.cv_image.shape[0]),
                                       interpolation=cv2.INTER_NEAREST)
                        self.current_mask = m
                    else:
                        self.calculate_mask()
                else:
                    self.calculate_mask()
            else:
                self.calculate_mask()

            if getattr(self, 'use_color_picker', None) and self.use_color_picker.get():
                ok = self._color_pick_detect(quiet=True)
                if not ok:
                    print(f"[batch_process] Colour selection skipped for {os.path.basename(file_path)}")

            self.extract_boundary()

            base = os.path.splitext(os.path.basename(file_path))[0]
            out_geo = os.path.join(geojson_folder, base + ".geojson")
            out_ovl = os.path.join(overlay_folder, base + "_overlay.png")

            # Try to read georef
            try:
                with rasterio.open(file_path) as src:
                    transform = src.transform
                    crs       = src.crs
            except:
                transform = None
                crs       = None

            if crs is not None:
                world_coords = [(transform * (x, y))[0:2] for x, y in self.edge_points]
                geom = LineString(world_coords)
            else:
                # pixel coords
                geom = LineString(self.edge_points)
                print(f"[batch_process] No georeference for {file_path}, writing pixel‐coordinate GeoJSON to console.")

            gdf = gpd.GeoDataFrame(geometry=[geom], crs=crs)
            gdf.to_file(out_geo, driver="GeoJSON")

            # Write overlay PNG (use confirmed-polyline teal to match the
            # on-screen overlay colour scheme).
            overlay = self.full_image.copy()
            if self.edge_points:
                pts = np.array(self.edge_points, dtype=np.int32).reshape((-1, 1, 2))
                thickness = int(self.edge_thickness_slider.get()) if hasattr(self, 'edge_thickness_slider') else 2
                cv2.polylines(overlay, [pts], False, (170, 136, 0), thickness)
            cv2.imwrite(out_ovl, overlay)

            processed_count += 1

        print(f"[batch_process] Processed {processed_count} images. GeoJSONs in {geojson_folder}, overlays in {overlay_folder}")
        messagebox.showinfo(
            "Batch Process",
            f"Processed {processed_count} images.\n"
            f"GeoJSON => {geojson_folder}\n"
            f"Overlay => {overlay_folder}",
            parent=self
        )

    def prepare_cv_image_for_batch(self, original_image):
        if original_image.ndim == 3 and original_image.shape[2] == 4:
            b, g, r, a = cv2.split(original_image)
            alpha = (a > 0).astype(np.uint8)
            b[alpha == 0] = g[alpha == 0] = r[alpha == 0] = 0
            img_bgr = cv2.merge([b, g, r])
        else:
            if original_image.ndim == 2:
                img_bgr = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
            elif original_image.ndim == 3 and original_image.shape[2] >= 3:
                img_bgr = original_image[:, :, :3]
            else:
                img_bgr = original_image
            alpha = np.ones(
                (img_bgr.shape[0], img_bgr.shape[1]), dtype=np.uint8)

        h, w = img_bgr.shape[:2]
        max_dim = 1200
        scale = min(max_dim / w, max_dim / h, 1.0)
        if scale < 1.0:
            new_size = (int(w*scale), int(h*scale))
            small_bgr = cv2.resize(
                img_bgr, new_size, interpolation=cv2.INTER_AREA)
            small_alpha = cv2.resize(
                alpha, new_size, interpolation=cv2.INTER_NEAREST)
        else:
            small_bgr = img_bgr
            small_alpha = alpha

        return small_bgr, small_alpha, scale

    # -------------- SHORTCUT ACTIONS --------------

    def f5_action(self):
        if self.mode == "batch":
            return
        if not isinstance(self.full_image, np.ndarray):
            messagebox.showwarning("Warning", "No image loaded!", parent=self)
            return
        self.do_invert_mask.set(False)     # ← ensure checkbox is unchecked

        # If ML predicted mask is enabled and available, use it
        if self.use_ml_pred_mask.get():
            if self.mode == "individual" and self.ml_mask_file_path.get():
                self.calculate_edge_with_ml_mask()
                return
            if self.mode == "ml" and self.ml_mask_folder_path.get():
                self.calculate_edge_with_ml_mask()
                return

        # otherwise, do normal HSV flow
        self.process_loaded_image(self.full_image)
        self.calculate_mask()
        self.extract_boundary()

    def f6_action(self):
        if self.mode == "batch":
            return
        if not isinstance(self.full_image, np.ndarray):
            messagebox.showwarning("Warning", "No image loaded!", parent=self)
            return
        self.do_invert_mask.set(True)      # ← ensure checkbox is checked

        # If ML predicted mask is enabled and available, use it then invert mask
        if self.use_ml_pred_mask.get():
            used_ml = False
            if self.mode == "individual" and self.ml_mask_file_path.get():
                self.calculate_edge_with_ml_mask()
                used_ml = True
            elif self.mode == "ml" and self.ml_mask_folder_path.get():
                self.calculate_edge_with_ml_mask()
                used_ml = True
            if used_ml and self.current_mask is not None:
                self.current_mask = cv2.bitwise_not(self.current_mask)
                self.display_mask()
                self.extract_boundary()
                return

        # otherwise, do normal HSV flow with invert
        self.process_loaded_image(self.full_image)
        self.calculate_mask()
        self.extract_boundary()

    def space_action(self):
        if self.mode == "ml":
            self.export_as_test_data()
            if self.image_files:
                self.image_files.pop(self.current_index)
                if not self.image_files:
                    self.cv_image = None
                    self.full_image = None
                    self.filename_label.configure(text="No file loaded")
                    messagebox.showinfo("Info", "All images processed.", parent=self)
                else:
                    if self.current_index >= len(self.image_files):
                        self.current_index = len(self.image_files) - 1
                    self.load_current_image()