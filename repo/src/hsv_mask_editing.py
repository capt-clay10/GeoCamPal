"""
hsv_mask_editing.py  —  GeoCamPal Feature Editing Mixin
========================================================

Purpose
-------
Mixin class that adds interactive feature editing, canvas rendering,
zoom/pan, non-destructive image adjustments, and all export methods
to the Feature Identifier tool.  It is designed to be inherited
alongside HSVMaskUIMixin and HSVMaskProcessingMixin by the
FeatureIdentifier class in feature_identifier.py.

This module must not be instantiated on its own.  It expects the
full FeatureIdentifier state (cv_image, full_image, features, etc.)
to be present via the MRO at runtime.

Edit session model
------------------
"Edit Detected Feature" (cut_detected_feature) opens the shape editor
with every confirmed feature of the image:

    edited_edge_points     — the active shape: red vertices, editable,
                             its label shown in the Label box.
    _edit_session_features — every other shape, drawn as a labelled
                             reference shape.  New shapes go here too.
                             Click a reference shape (or use Prev/Next)
                             to make it the active shape.

Nothing is committed back to self.features until Confirm Feature (or
Enter) is pressed; Cancel (or Esc) restores the features as they were
when the editor opened.  Delete Shape removes the active shape, Delete
All removes every shape on the image (the next shape then follows the
Feature ID again).  Shapes whose vertices are not touched keep their
exact full-image coordinates.  While the editor is open the
image cannot be exported, and changing image asks before discarding
unconfirmed edits.

Labels
------
Each shape can carry its own label (feature_model.Feature).  Type the
label in the editor's Label box, then draw: a shape keeps the label it
had when its first vertex was placed, and the next shape starts with the
same label.  A label typed while a shape is being drawn is kept for the
next shape (the Label box and the shape counter say so).  To rename a
shape, select it (click it, or Prev/Next) and type.  Double-click ends a
line; clicking near the first vertex or double-clicking closes a polygon.
When a shape is finished with no label while the Feature ID is empty,
the editor asks for one.  A shape without a label of its own takes the
Feature ID of the configuration window, so an image with one label
behaves exactly as in earlier releases.  Category IDs come from a label
catalogue (feature_model.LabelCatalog) saved with the settings; the
Feature ID's category is the Category ID entry of the configuration
window.

Large-image proxy
-----------------
Images wider or taller than 2000 px are downscaled to a working proxy
before the edit canvas opens.  Vertex coordinates are scaled
proportionally.  On confirm, coordinates are scaled back to the
original full-image resolution so exports are always at full precision.

Canvas rendering
----------------
The edit canvas uses a viewport-only rendering strategy: only the
currently visible region of the image (plus a small margin) is
rendered at the current zoom level.  This keeps memory usage and
redraw time bounded regardless of image size or zoom factor.

Slider adjustments (saturation, exposure, highlights) apply to the
proxy image and are debounced to ~30 fps during drag, with a
higher-quality pass on mouse release.

Undo / Redo
-----------
Every vertex add, delete or move, freehand stroke, shape created,
closed, selected or deleted, label change and Reset pushes a snapshot
of the whole session (active shape with its type and label, and every
other shape) to edit_history (capped at 50 entries).  Any new action
clears the redo stack.  Keyboard shortcuts: U = undo, R = redo.

Keyboard shortcuts (edit canvas)
---------------------------------
    d   — delete-vertex mode
    m   — add/move-vertex mode
    f   — freehand drawing
    e   — create new polyline
    p   — create new polygon
    c   — continue existing line
    U   — undo
    R   — redo
    Enter — confirm feature
    Esc   — cancel editing

Export formats
--------------
Every dataset export goes into one top folder, Dataset/, inside the
chosen export folder (or straight into the export folder when that is
itself a folder named "Dataset"):

    Dataset/
      training dataset/            — one set per label, named <image>_<label>
        images/          — copy of the source image (with label suffix)
        masks/           — binary PNG mask (255 = feature, 0 = background)
        overlays/        — source image with the features drawn on top
        geojson/         — GeoJSON LineString or Polygon; CRS from source
                           TIFF when available, pixel coords otherwise
        coco/            — COCO JSON with segmentation polygon, bbox,
                           image dimensions, category and annotation IDs
        class_points/    — CSV of colour-picker sample pixels with RGB,
                           HSV, intensity, class label, and detection
                           settings per point

      training_dataset_multiclass/ — only for an image with 2+ labels
        images/          — the source image, copied once
        masks/           — <image>_labelmap.png, pixel value = category ID
        overlays/        — all shapes with their labels
        geojson/         — one feature per shape with feature_id,
                           category_id, feature_type, image_id and
                           annotation_id properties
        coco/            — <image>_coco.json, each shape with its own
                           category; categories = every label exported
                           to this folder
        categories.json  — label <-> category ID for the whole folder,
                           and per image which labels were used

      test dataset/                — plain copy of the source image

Dependencies
------------
    numpy, opencv-python (cv2), Pillow, shapely, geopandas, rasterio,
    customtkinter
"""

import os
import json
import shutil
import time
import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk
from PIL import Image, ImageTk, ImageEnhance
import cv2
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, Polygon
import rasterio
import warnings
from rasterio.errors import NotGeoreferencedWarning
from feature_model import Feature, feature_label, with_label, slug, LabelCatalog
from utils import imwrite_safe, resource_path, __version__ as GEOCAMPAL_VERSION

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


class _LabelDialog(ctk.CTkToplevel):
    """Small customtkinter text prompt (OK / Cancel) with the app icon."""

    def __init__(self, parent, icon_path, title, message, initial=""):
        super().__init__(parent)
        self.title(title)
        self.result = None
        self.resizable(False, False)
        try:
            self.transient(parent)
        except Exception:
            pass
        ctk.CTkLabel(self, text=message, wraplength=440, justify="left"
                     ).pack(padx=16, pady=(14, 8), anchor="w")
        self._entry = ctk.CTkEntry(self, width=440)
        self._entry.pack(padx=16, pady=(0, 12))
        if initial:
            self._entry.insert(0, initial)
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(padx=16, pady=(0, 14))
        ctk.CTkButton(row, text="OK", width=110, command=self._ok
                      ).pack(side="left", padx=6)
        ctk.CTkButton(row, text="Cancel", width=110, command=self._cancel,
                      fg_color="gray40", hover_color="gray30"
                      ).pack(side="left", padx=6)
        self.bind("<Return>", lambda e: self._ok())
        self.bind("<Escape>", lambda e: self._cancel())
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self._icon_path = icon_path
        # CTkToplevel puts its own icon on the window shortly after
        # creation, so ours goes on a little later
        self.after(260, self._set_icon)
        self.after(60, self._place)

    def _set_icon(self):
        try:
            self._icon_img = tk.PhotoImage(file=self._icon_path)
            self.iconphoto(False, self._icon_img)
        except Exception:
            try:
                self.iconbitmap(str(self._icon_path).replace(".png", ".ico"))
            except Exception:
                pass

    def _place(self):
        try:
            self.update_idletasks()
            p = self.master
            x = p.winfo_rootx() + (p.winfo_width() - self.winfo_width()) // 2
            y = p.winfo_rooty() + (p.winfo_height() - self.winfo_height()) // 2
            self.geometry(f"+{max(0, x)}+{max(0, y)}")
        except Exception:
            pass
        try:
            self.lift()
            self.focus_force()
            self._entry.focus_set()
            self.grab_set()
        except Exception:
            pass

    def _ok(self):
        self.result = self._entry.get()
        self._close()

    def _cancel(self):
        self.result = None
        self._close()

    def _close(self):
        try:
            self.grab_release()
        except Exception:
            pass
        self.destroy()


def ask_label_dialog(parent, icon_path, title, message, initial=""):
    """Modal text prompt; returns the text, or None when cancelled."""
    dlg = _LabelDialog(parent, icon_path, title, message, initial)
    parent.wait_window(dlg)
    return dlg.result


class HSVMaskEditingMixin:
    """Mixin that supplies feature editing, canvas interaction,
    zoom/pan, preview adjust, and export methods."""

    # -------------- FEATURE EDITING --------------

    def start_freehand(self):
        if not self.edited_edge_points:
            # A stroke on an empty canvas is a new line, whatever came before
            self.is_polygon_mode = False
        self.freehand_mode = True
        # Keep the freehand line visually clean: hide vertex markers
        # for the current shape until the user explicitly switches back
        # to a vertex-editing mode.
        self._hide_vertices_for_current_shape = True
        self._record_history()

        # unbind the old click-based handlers
        self.edit_canvas.unbind("<ButtonPress-1>")
        self.edit_canvas.unbind("<B1-Motion>")
        self.edit_canvas.unbind("<ButtonRelease-1>")
        self.edit_canvas.unbind("<Double-Button-1>")

        # now bind free-hand drawing
        self.edit_canvas.bind("<B1-Motion>",       self.on_freehand_draw)
        self.edit_canvas.bind("<ButtonRelease-1>", self.finish_freehand)

    def on_freehand_draw(self, event):
        # convert canvas coords back into image coords
        x = self.edit_canvas.canvasx(event.x) / self.zoom_scale
        y = self.edit_canvas.canvasy(event.y) / self.zoom_scale
        self.edited_edge_points.append([x, y])
        self.redraw_canvas()

    def finish_freehand(self, event):
        self.freehand_mode = False
        # A freehand stroke ends in vertex-editing mode, not click-to-add
        self.creation_mode = False
        # unbind only the free-hand events
        self.edit_canvas.unbind("<B1-Motion>")
        self.edit_canvas.unbind("<ButtonRelease-1>")

        # record the end of this stroke for undo/redo
        self._record_history()

        # re-bind the normal vertex-edit handlers
        self._bind_vertex_handlers()

        # and redraw to show the final segment
        self._update_poly_nav_label()
        self.redraw_canvas()


    def _bind_edit_shortcuts(self):
        """
        Single-letter editor shortcuts, bound on the edit canvas only.

        They used to be bound on the configuration window as well, which
        made typing a label such as "runup" in an entry fire Undo/Redo and
        removed the window's own R = Edit Detected Feature shortcut.
        """
        if not hasattr(self, "edit_canvas"):
            return

        # modes/tools
        self.edit_canvas.bind("d", self._btn_mode_delete)
        self.edit_canvas.bind("m", self._btn_mode_add)
        self.edit_canvas.bind("f", self._btn_freehand)
        self.edit_canvas.bind("e", self._btn_create_edge)
        self.edit_canvas.bind("p", self._btn_create_polygon)
        self.edit_canvas.bind("c", self._btn_continue_line)

        # Keep the canvas focused so single-letter keys work, except while
        # the user is typing in an entry (e.g. the Label box).  Clicks on
        # the canvas take focus in the click handlers themselves.
        self.edit_canvas.bind("<Motion>", self._canvas_autofocus, add="+")
        self.edit_canvas.bind("<Enter>", self._canvas_take_focus, add="+")
        self.edit_canvas.focus_set()

        # Undo / Redo on U / R (both cases)
        for seq in ("u", "U"):
            self.edit_canvas.bind(seq, self._btn_undo)
        for seq in ("r", "R"):
            self.edit_canvas.bind(seq, self._btn_redo)

    def _canvas_autofocus(self, _event=None):
        """Give the canvas keyboard focus unless an entry is being typed in."""
        try:
            w = self.focus_get()
        except Exception:
            w = None
        try:
            if w is not None and w.winfo_class() in (
                    "Entry", "TEntry", "Text", "Spinbox", "TCombobox"):
                return
            self.edit_canvas.focus_set()
        except Exception:
            pass



    def _canvas_take_focus(self, _event=None):
        """
        Pointer enters the canvas: take keyboard focus from the editor's
        own Label / Category ID boxes (leaving them applies what was typed),
        so shortcut keys reach the canvas.  Entries in the configuration
        window keep their focus.
        """
        try:
            w = self.focus_get()
        except Exception:
            w = None
        try:
            if (w is not None
                    and w.winfo_class() in ("Entry", "TEntry", "Text", "Spinbox", "TCombobox")
                    and str(w.winfo_toplevel()) != str(self.image_display_window)):
                return
            self.edit_canvas.focus_set()
        except Exception:
            pass


    def _unbind_edit_shortcuts(self):
        if hasattr(self, "edit_canvas"):
            for seq in ("d", "m", "f", "e", "p", "c",
                        "<Double-Button-1>", "<Double-1>", "<Motion>", "<Enter>",
                        "u", "U", "r", "R"):
                try:
                    self.edit_canvas.unbind(seq)
                except Exception:
                    pass

    def _refocus_canvas(self):
        try:
            self.edit_canvas.focus_set()
        except Exception:
            pass

    def _btn_undo(self, *_):
        self.undo_last_action()
        self._refocus_canvas()

    def _btn_redo(self, *_):
        self.redo_last_action()
        self._refocus_canvas()

    def _btn_freehand(self, *_, **__):
        self.start_freehand()
        self._refocus_canvas()

    def _btn_create_edge(self, *_, **__):
        self.create_new_edge()
        self._refocus_canvas()

    def _btn_create_polygon(self, *_, **__):
        self.create_new_polygon()
        self._refocus_canvas()

    def _btn_continue_line(self, *_, **__):
        self.continue_line()
        self._refocus_canvas()

    def _btn_mode_delete(self, *_, **__):
        self.set_vertex_mode("delete")
        self._refocus_canvas()

    def _btn_mode_add(self, *_, **__):
        self.set_vertex_mode("add")
        self._refocus_canvas()


    def _btn_delete_shape(self, *_, **__):
        self.delete_active_shape()
        self._refocus_canvas()

    def _btn_delete_all(self, *_, **__):
        self.delete_all_shapes()
        self._refocus_canvas()

    def _btn_cancel(self, *_, **__):
        self.cancel_edit_session()

    def _btn_apply_label_all(self, *_, **__):
        self._apply_label_to_all()
        self._refocus_canvas()


    # ---------- editor toolbar layout (right panel) ----------

    def _layout_toolbar(self, width=None):
        """Grid the toolbar buttons in 2-5 columns depending on the width
        of the right panel, and wrap the text to that width.  Every row of
        buttons fills the whole width (a row with fewer buttons gets wider
        buttons), so there are no gaps."""
        groups = getattr(self, "_toolbar_groups", None)
        if not groups:
            return
        if width is None:
            try:
                width = self.control_frame.winfo_width()
            except Exception:
                return
        width = int(width)
        for lbl, margin in ((getattr(self, "_toolbar_info_label", None), 24),
                            (getattr(self, "_poly_nav_label", None), 24)):
            try:
                if lbl is not None and lbl.winfo_exists():
                    lbl.configure(wraplength=max(200, width - margin))
            except Exception:
                pass
        cols = max(2, min(5, width // 125))
        if cols == getattr(self, "_toolbar_cols", None):
            return
        self._toolbar_cols = cols
        SUB = 60                      # divisible by 1..5: rows of any length fill evenly
        for frame, btns in groups:
            try:
                for c in range(SUB):
                    frame.grid_columnconfigure(c, weight=1, uniform="tb")
                n = len(btns)
                for i, btn in enumerate(btns):
                    r, j = divmod(i, cols)
                    in_row = min(cols, n - r * cols)
                    span = SUB // in_row
                    btn.grid(row=r, column=j * span, columnspan=span,
                             sticky="ew", padx=2, pady=1)
            except Exception:
                pass
        self.after(30, self._fit_toolbar_height)

    def _fit_toolbar_height(self):
        """Make the toolbar exactly as tall as its contents so the preview
        above gets the rest of the panel; on a short screen the toolbar
        scrolls and the preview keeps about a third of the panel."""
        cf = getattr(self, "control_frame", None)
        if cf is None or not getattr(self, "_edit_session_open", False):
            return
        try:
            if not cf.winfo_exists():
                return
            cf.update_idletasks()             # sizes after the latest re-flow
            needed = cf.winfo_reqheight() + 4
            panel = self.top_right_frame.winfo_height()
            limit = int(panel * 0.68) if panel > 100 else needed
            target = max(160, min(needed, limit))
            scale = 1.0
            try:
                scale = float(cf._get_widget_scaling())
            except Exception:
                pass
            # CTk applies its widget scaling to the height it is given
            cf.configure(height=max(1, int(round(target / scale))))
        except Exception:
            pass

    def _on_right_panel_resize(self, _event=None):
        if getattr(self, "_edit_session_open", False):
            self.after(30, self._fit_toolbar_height)


    def _install_slider_handlers(self):
        # Live (throttled) while dragging
        for s in (self.slider_sat, self.slider_exp, self.slider_hil):
            s.configure(command=self._on_adjust_change_live)
            # single "commit" pass on mouse-up
            s.bind("<ButtonRelease-1>", self._on_adjust_commit)

    def _on_adjust_change_live(self, _=None):
        """Debounce to ~30fps while dragging."""
        self._pending_preview = True
        if self._preview_after_id is None:
            self._preview_after_id = self.after(33, self._flush_preview)

    def _flush_preview(self):
        self._preview_after_id = None
        if not self._pending_preview:
            return
        self._pending_preview = False

        # Apply adjustments to the small proxy (fast, max ~2000 px)
        self._apply_adjustments_to_proxy(high_quality=False)
        # Invalidate viewport cache so _render_viewport picks up the adjusted proxy
        self._invalidate_viewport_cache()
        self._render_viewport(high_quality=False)
        self._refresh_overlays()

        if self._pending_preview:
            self._preview_after_id = self.after(33, self._flush_preview)

    def _on_adjust_commit(self, _evt=None):
        """Mouse released — do one nicer pass (still at current zoom)."""
        self._apply_adjustments_to_proxy(high_quality=True)
        self._invalidate_viewport_cache()
        self._render_viewport(high_quality=True)
        self._refresh_overlays()

    def _invalidate_zoom_cache(self):
        self._zoom_cache["zoom"] = None
        self._zoom_cache["img"] = None
        self._invalidate_viewport_cache()

    def _ensure_scaled_base_for_zoom(self, high_quality=False):
        """
        Return a PIL RGB image already resized to the current zoom.
        Cached so we don't resize per slider tick.
        Uses _edit_proxy if available (large images).
        """
        if self.full_image is None:
            return None
        # Use proxy for slider previews — much smaller memory footprint
        src = getattr(self, '_edit_proxy', None)
        if src is None:
            src = self.full_image
        z = max(0.1, min(5.0, float(getattr(self, "zoom_scale", 1.0))))
        if self._zoom_cache["zoom"] == z and self._zoom_cache["img"] is not None:
            return self._zoom_cache["img"]

        src_rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        h, w = src_rgb.shape[:2]
        sw, sh = max(1, int(w * z)), max(1, int(h * z))
        resample = Image.LANCZOS if high_quality else Image.BILINEAR
        pil = Image.fromarray(src_rgb).resize((sw, sh), resample)
        self._zoom_cache["zoom"] = z
        self._zoom_cache["img"] = pil
        return pil


    def cut_detected_feature(self):
        """
        Open the shape editor with ALL confirmed features of this image.

        The first feature becomes the active shape (red vertices; its label
        is shown in the Label box).  Every other feature is kept in
        _edit_session_features and drawn as a labelled reference shape.
        New shapes are added to the session as well.  Nothing is committed
        to self.features until Confirm Feature is pressed; Cancel (or Esc)
        puts the features back as they were when the editor opened.
        """
        if getattr(self, "_edit_session_open", False):
            print("[Edit] The editor is already open. Press Confirm Feature "
                  "or Cancel in the image window first.")
            try:
                self.image_display_window.deiconify()
                self.image_display_window.lift()
                self.edit_canvas.focus_set()
            except Exception:
                pass
            return
        if not isinstance(getattr(self, "full_image", None), np.ndarray):
            messagebox.showwarning("Edit Detected Feature",
                                   "Load an image first.", parent=self)
            return

        self.vertex_mode = "delete"
        self.freehand_mode = False
        self.creation_mode = False
        self._hide_vertices_for_current_shape = False

        # Committed state, restored by Cancel or by leaving the image
        self._features_before_edit = list(self.features)
        self._edge_points_before_edit = list(self.edge_points or [])

        # Category IDs already used for multi-label exports to this folder
        self._merge_export_folder_categories()

        # ── Move ALL features into the edit session ──
        self._edit_session_features = []
        points_to_edit = []
        self._editing_feature_type = "polyline"
        active_label = ""

        active_feat = None
        for feat in self.features:
            ftype, pts = feat
            if len(pts) < 2:
                continue
            if not points_to_edit:
                # First valid feature → active edit target
                self._editing_feature_type = ftype
                points_to_edit = [list(p) for p in pts]
                active_label = feature_label(feat)
                active_feat = feat
            else:
                # Remaining features → session references (visible on canvas)
                self._edit_session_features.append(feat)

        # Fall back to edge_points if no features exist
        if not points_to_edit and self.edge_points and len(self.edge_points) >= 2:
            points_to_edit = [list(p) for p in self.edge_points]

        # Everything lives in the edit session now.  edge_points is cleared
        # as well, so nothing can export the pre-edit geometry meanwhile.
        self.features = []
        self.edge_points = []

        if not points_to_edit:
            print("No valid detected feature to edit. Starting empty.")

        # Stored form: "" means the shape follows the Feature ID entry,
        # even if that entry is changed while the editor is open.
        self._active_label = active_label
        self._pending_label = None
        self.initial_edge_points = [list(p) for p in points_to_edit]
        self.edited_edge_points  = [list(p) for p in points_to_edit]
        # Remember where the active shape came from, so a shape that is not
        # edited keeps its exact full-image coordinates (see _stash_active_shape)
        self._active_source = None
        self._active_source_feat = active_feat
        self.edit_history = []
        self.redo_history = []
        self.selected_vertex = None
        self.is_polygon_mode = (self._editing_feature_type == 'polygon')

        # Clear the center frame and create editing canvas
        for widget in self.top_center_frame.winfo_children():
            widget.destroy()

        # Right panel: clear since everything is now in the edit session
        self.update_edge_display()

        self.zoom_scale = 1.0
        self.pan_x = self.pan_y = 0

        # --- Reset stale canvas item IDs from any previous edit session ---
        self._poly_id = None
        self._vertex_ids = []
        self.bg_image_id = None
        self._bg_cache.clear()
        self._bg_current_zoom = None
        self._zoom_cache = {"zoom": None, "img": None}
        self._redraw_job = None
        self._preview_after_id = None
        self._pending_preview = False
        self.edit_original_pil = None

        # --- Create a downscaled proxy for editing large images ---
        # This prevents creating massive PIL images when zooming
        MAX_EDIT_DIM = 2000
        h_full, w_full = self.full_image.shape[:2]
        max_dim = max(h_full, w_full)
        if max_dim > MAX_EDIT_DIM:
            self._edit_proxy_scale = MAX_EDIT_DIM / max_dim
            new_w = max(1, int(w_full * self._edit_proxy_scale))
            new_h = max(1, int(h_full * self._edit_proxy_scale))
            self._edit_proxy = cv2.resize(self.full_image, (new_w, new_h),
                                          interpolation=cv2.INTER_AREA)
            print(f"[Edit] Large image ({w_full}×{h_full}) → proxy ({new_w}×{new_h})")
            # Scale vertex coords from full-image space to proxy space
            s = self._edit_proxy_scale
            self.edited_edge_points = [[x * s, y * s] for x, y in self.edited_edge_points]
            self.initial_edge_points = [[x * s, y * s] for x, y in self.initial_edge_points]
        else:
            self._edit_proxy = self.full_image
            self._edit_proxy_scale = 1.0
        if self._active_source_feat is not None:
            self._active_source = (self._active_source_feat,
                                   [list(p) for p in self.edited_edge_points])

        self.edit_canvas_container = tk.Frame(self.top_center_frame)
        self.edit_canvas_container.pack(fill="both", expand=True)

        self.edit_canvas = tk.Canvas(self.edit_canvas_container)
        self.edit_canvas.grid(row=0, column=0, sticky="nsew")

        v_scroll = tk.Scrollbar(
            self.edit_canvas_container, orient="vertical", command=self.edit_canvas.yview
        )
        v_scroll.grid(row=0, column=1, sticky="ns")
        h_scroll = tk.Scrollbar(
            self.edit_canvas_container, orient="horizontal", command=self.edit_canvas.xview
        )
        h_scroll.grid(row=1, column=0, sticky="ew")

        self.edit_canvas.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        self.edit_canvas_container.grid_rowconfigure(0, weight=1)
        self.edit_canvas_container.grid_columnconfigure(0, weight=1)

        # --- Viewport rendering state for this edit session ---
        self._adjusted_proxy = None
        self._vp_render_bounds = None
        self._vp_render_zoom = None
        self._scroll_job = None
        self._bg_photo_ref = None

        # Intercept scroll commands so viewport re-renders on pan/scroll
        def _yscroll_cb(first, last):
            v_scroll.set(first, last)
            self._on_scroll_update()

        def _xscroll_cb(first, last):
            h_scroll.set(first, last)
            self._on_scroll_update()

        self.edit_canvas.configure(yscrollcommand=_yscroll_cb,
                                   xscrollcommand=_xscroll_cb)

        # Bindings for editing
        self.edit_canvas.bind("<Configure>", self.redraw_canvas)
        self.edit_canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.edit_canvas.bind("<Button-4>", self.on_mousewheel)
        self.edit_canvas.bind("<Button-5>", self.on_mousewheel)
        self.edit_canvas.bind("<ButtonPress-1>", self.on_canvas_press)
        self.edit_canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.edit_canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.edit_canvas.bind("<Double-Button-1>", self.on_canvas_double_click)
        self.edit_canvas.bind("<BackSpace>", self.delete_selected_vertex)

        # Make sure single-letter keys work right away
        try:
            self.edit_canvas.focus_set()
        except Exception:
            pass

        # ── Expand the display window while in edit mode ──
        try:
            disp_win = self.image_display_window
            screen_w = disp_win.winfo_screenwidth()
            screen_h = disp_win.winfo_screenheight()
            # use up to 95% of screen when editing
            new_w = min(int(screen_w * 0.95), max(1400, disp_win.winfo_width()))
            new_h = min(int(screen_h * 0.92), max(900, disp_win.winfo_height()))
            x = max(0, (screen_w - new_w) // 2)
            y = max(0, (screen_h - new_h) // 2)
            disp_win.geometry(f"{new_w}x{new_h}+{x}+{y}")
        except Exception:
            pass

        # ── Control panel: in the RIGHT panel, below the overlay preview, so
        #    the middle panel is the editing canvas only.  Scrollable for
        #    small screens. ──
        right = self.top_right_frame
        self.control_frame = ctk.CTkScrollableFrame(right, height=420, orientation="vertical")
        zoom_bar = getattr(self, "_overlay_zoom_bar", None)
        try:
            if zoom_bar is not None and zoom_bar.winfo_exists():
                # packed before the zoom bar: preview, zoom bar, then toolbar
                self.control_frame.pack(side="bottom", fill="x", pady=(12, 0), before=zoom_bar)
            else:
                self.control_frame.pack(side="bottom", fill="x", pady=(12, 0))
        except Exception:
            self.control_frame.pack(side="bottom", fill="x", pady=(12, 0))

        BTN_H = 28
        BTN_FONT = ("Arial", 11)
        RED = dict(fg_color="#8B0000", hover_color="#A52A2A", text_color="white")
        DARK_RED = dict(fg_color="#5C0000", hover_color="#A52A2A", text_color="white")
        WHITE = dict(fg_color="white", text_color="black")

        # Button groups; _layout_toolbar puts them in as many columns as
        # the panel width allows (2 to 5), so nothing is cut off.
        self._toolbar_groups = []
        self._toolbar_cols = None

        def group(specs):
            frame = ctk.CTkFrame(self.control_frame)
            frame.pack(side="top", fill="x", pady=2)
            btns = [ctk.CTkButton(frame, text=text, width=40, height=BTN_H,
                                  font=BTN_FONT, command=cmd, **colours)
                    for text, cmd, colours in specs]
            self._toolbar_groups.append((frame, btns))
            return btns

        # Shape creation
        g1 = group([("Create Line", self._btn_create_edge, {}),
                    ("Create Polygon", self._btn_create_polygon, {}),
                    ("Add/Move Vertex", self._btn_mode_add, {}),
                    ("Continue Line", self._btn_continue_line, {}),
                    ("Freehand", self._btn_freehand, {})])
        self.btn_add_mode = g1[2]
        # View / history
        g2 = group([("Zoom In", lambda: self.adjust_zoom(1.2), {}),
                    ("Zoom Out", lambda: self.adjust_zoom(0.8), {}),
                    ("Undo", self._btn_undo, {}),
                    ("Redo", self._btn_redo, {}),
                    ("Delete Vertex", self._btn_mode_delete, {})])
        self.btn_delete_mode = g2[4]
        # Move between shapes
        group([("◀ Prev shape", self._edit_prev_polygon, WHITE),
               ("Next shape ▶", self._edit_next_polygon, WHITE)])

        # Label / Category ID of the active shape
        self._build_label_row(self.control_frame, btn_h=BTN_H, font=BTN_FONT)

        # Confirm Feature, Cancel, then the shape counter
        row5 = ctk.CTkFrame(self.control_frame)
        row5.pack(side="top", fill="x", pady=2)
        self.confirm_button = ctk.CTkButton(
            row5, text="Confirm Feature", width=40, height=BTN_H,
            font=BTN_FONT, command=self.confirm_feature_cuts,
            fg_color="#0F52BA", hover_color="#2A6BD1", text_color="white")
        self.confirm_button.pack(side="left", expand=True, fill="x", padx=2, pady=1)
        ctk.CTkButton(row5, text="Cancel", width=40, height=BTN_H,
                      font=BTN_FONT, command=self._btn_cancel,
                      fg_color="gray40", hover_color="gray30", text_color="white"
                      ).pack(side="left", expand=True, fill="x", padx=2, pady=1)
        self._poly_nav_label = ctk.CTkLabel(
            self.control_frame, text="", font=("Arial", 10), anchor="w",
            justify="left", wraplength=330)
        self._poly_nav_label.pack(side="top", fill="x", padx=8, pady=(0, 2))
        self._update_poly_nav_label()

        # Saturation / Exposure / Highlights sliders
        self._build_adjust_row(self.control_frame)
        self._install_slider_handlers()

        # Help text
        info_label = ctk.CTkLabel(
            self.control_frame,
            text="Type the label, then draw: a shape keeps the label it had when "
                 "you started it, and the next shape starts with the same label. "
                 "Double-click ends a line; a click near the first vertex (or a "
                 "double-click) closes a polygon. To rename a shape, click it (or "
                 "use Prev shape / Next shape) and type. Scroll = zoom; in vertex "
                 "modes double-click = delete/add vertex. Keys: d = delete vertex, "
                 "m = move, f = freehand, e = line, p = polygon, c = continue, "
                 "U = undo, R = redo, Enter = confirm, Esc = cancel.",
            font=("Arial", 9), wraplength=330, justify="left", anchor="w"
        )
        info_label.pack(side="top", fill="x", padx=6, pady=(2, 4))
        self._toolbar_info_label = info_label

        # Destructive actions last: Reset, Delete Shape = the active shape
        # (red vertices) only, Delete All = every shape on this image
        # (asks first; Undo brings shapes back)
        group([("Reset", self.reset_to_initial, RED),
               ("Delete Shape", self._btn_delete_shape, RED),
               ("Delete All", self._btn_delete_all, DARK_RED)])
        self._toolbar_groups[-1][0].pack_configure(pady=(8, 6))

        # Re-flow buttons and text when the panel is resized
        outer = getattr(self.control_frame, "_parent_canvas", None)
        if outer is not None:
            try:
                outer.bind("<Configure>", lambda e: self._layout_toolbar(e.width), add="+")
            except Exception:
                pass
        if not getattr(self, "_right_panel_resize_bound", False):
            try:
                self.top_right_frame.bind("<Configure>", self._on_right_panel_resize, add="+")
                self._right_panel_resize_bound = True
            except Exception:
                pass
        try:
            self._layout_toolbar(max(300, self.top_right_frame.winfo_width() - 20))
        except Exception:
            pass
        self.after_idle(self._fit_toolbar_height)
        # the preview is smaller now: redraw it for its new size
        self.after(120, self.update_edge_display)

        # enable edit-mode shortcuts (binds d/m/f/e/p/c + U/R and keeps canvas focused)
        self._bind_edit_shortcuts()

        # Enter = Confirm Feature, Esc = Cancel.  Bound once, on the image
        # window: the canvas has that window in its bindtags, so binding the
        # canvas as well made one key press confirm twice.
        try:
            self.image_display_window.bind("<Return>", self._on_display_return)
            self.image_display_window.bind("<Escape>", self._on_display_escape)
        except Exception:
            pass

        # Undo history and the Reset target start here
        self._reset_history()
        self._edit_initial_state = self.edit_history[0]
        self._edit_session_open = True
        self._update_default_entries_state()
        self._update_panel_titles()     # "Shape editor" / "Editing tools"

        # Initial draw
        base = self._ensure_scaled_base_for_zoom(high_quality=False)
        if base is not None:
            self._set_edit_preview(base)   # show something right away
        self.redraw_canvas()


    def create_new_edge(self):
        """
        Start creating a new polyline.  Non-destructive: the current
        active shape (if any) stays in the session with its label, and the
        new line starts with the same label.
        """
        self._stash_active_shape()
        self.edited_edge_points = []
        self._active_source = None
        self.selected_vertex = None
        self._hide_vertices_for_current_shape = False
        self.freehand_mode = False
        self.creation_mode = True
        self.is_polygon_mode = False
        self._bind_creation_handlers(polygon=False)
        self._record_history()
        self._update_poly_nav_label()
        self.redraw_canvas()

    def create_new_polygon(self):
        """
        Start creating a new polygon.  Non-destructive (see create_new_edge).
        Close it by clicking near the first vertex or by double-clicking.
        """
        self._stash_active_shape()
        self.edited_edge_points = []
        self._active_source = None
        self.selected_vertex = None
        self._hide_vertices_for_current_shape = False
        self.freehand_mode = False
        self.creation_mode = True
        self.is_polygon_mode = True
        self._bind_creation_handlers(polygon=True)
        self._record_history()
        self._update_poly_nav_label()
        self.redraw_canvas()

    def _close_polygon_double_click(self, event):
        """
        Close the polygon being created on double-click.

        Tk fires <Button-1> for the first click of a double-click, so that
        click has already added the last vertex.
        """
        if not self.is_polygon_mode or not getattr(self, 'creation_mode', False):
            return

        if len(self.edited_edge_points) < 3:
            print("[Edit] Need at least 3 vertices to close a polygon.")
            return

        self._close_active_polygon()
        print("Polygon closed (double-click) and added to session. "
              "You can create more or press Confirm.")

    def _close_active_polygon(self):
        """Close the polygon being drawn and keep it in the session."""
        pts = self.edited_edge_points
        if pts[-1] != pts[0]:
            pts.append(list(pts[0]))
        self.is_polygon_mode = True
        self._stash_active_shape()
        self.edited_edge_points = []
        self.selected_vertex = None
        self._record_history()
        self._update_poly_nav_label()
        self.redraw_canvas()

    def continue_line(self):
        """
        Continue adding points to the current polyline from its last vertex.
        Unlike Create Line, this does NOT save existing points to session or
        clear them.  New clicks are appended to the end of the existing
        edited_edge_points, seamlessly extending the line.

        If there are no existing points this behaves like Create Line.
        """
        if getattr(self, "creation_mode", False):
            # Already in creation mode — do nothing special
            print("[Edit] Already in creation/continue mode.")
            return

        self._hide_vertices_for_current_shape = False
        self.freehand_mode = False
        self.creation_mode = True
        # Keep is_polygon_mode as-is (if user was editing a polygon, continue
        # as polygon); with nothing to continue, it is a new line.
        if not self.edited_edge_points:
            self.is_polygon_mode = False
        self._bind_creation_handlers(polygon=False)
        self.redraw_canvas()

        n = len(self.edited_edge_points)
        if n > 0:
            print(f"[Edit] Continue line from point {n} (last vertex).")
        else:
            print("[Edit] Continue line — starting fresh (no existing points).")


    def confirm_feature_cuts(self, event=None):
        """
        Confirm Feature (button or Enter): commit the active shape and every
        other session shape, with their labels, to self.features and close
        the editor.
        """
        if not getattr(self, "_edit_session_open", False):
            return

        if not self._default_label():
            n_unl = sum(1 for f in self._edit_session_features if not feature_label(f))
            if (len(self.edited_edge_points) >= 2
                    and not self._stored_label(getattr(self, "_active_label", ""))):
                n_unl += 1
            if n_unl and not messagebox.askyesno(
                    "Shapes without a label",
                    f"{n_unl} shape(s) have no label and the Feature ID in the "
                    "configuration window is empty, so they cannot be exported "
                    "yet.\n\nConfirm anyway? (Click a shape in the editor to "
                    "label it, or type a Feature ID before exporting.)",
                    parent=self.image_display_window):
                self._refocus_canvas()
                return

        # The active shape goes first, as before
        self._stash_active_shape(position=0, ask_label=False)
        session = self._edit_session_features

        if not session:
            if not messagebox.askyesno(
                    "No shapes left",
                    "There are no shapes left on this image.\n\n"
                    "Confirm anyway? The image will then have no feature "
                    "to export.",
                    parent=self.image_display_window):
                self._refocus_canvas()
                return

        # Commit all session features into the main features list
        self.features = list(session)
        # Labels were used in the editor: shapes that follow the Feature ID
        # get it as their own label, so the image's labels are fixed here
        # and the Feature ID / Category ID entries are not used any more.
        d = self._default_label()
        if (d and slug(d) and any(feature_label(f) for f in self.features)
                and not all(feature_label(f) for f in self.features)):
            self._sync_default_into_catalog()
            self.features = [f if feature_label(f) else with_label(f, d)
                             for f in self.features]
        self.edge_points = ([list(p) for p in self.features[0][1]]
                            if self.features else [])

        self._end_edit_session()

        n = len(self.features)
        if n:
            parts = [f"{name or '(no label)'} x{k}"
                     for name, k in self._label_counts(self.features)]
            print(f"[Confirm] Committed {n} feature(s): {', '.join(parts)}.")
        else:
            print("[Confirm] Committed 0 features: this image has no "
                  "feature to export.")

    def cancel_edit_session(self, prompt=True):
        """Cancel (button or Esc): leave the editor without keeping changes."""
        if not getattr(self, "_edit_session_open", False):
            return
        if prompt and self._edit_session_changed():
            if not messagebox.askyesno(
                    "Cancel editing",
                    "Discard the changes made in the editor?\n\n"
                    "The shapes go back to how they were before you opened it.",
                    parent=self.image_display_window):
                self._refocus_canvas()
                return
        self.features = list(getattr(self, "_features_before_edit", []))
        self.edge_points = list(getattr(self, "_edge_points_before_edit", []))
        self._end_edit_session()
        print("[Edit] Editing cancelled. The shapes are as they were before.")

    def _abort_edit_session(self, prompt=True, refresh=True):
        """
        Close an open editor before the image changes (Next/Prev, Load,
        Reset Session).  Unconfirmed edits are discarded, after asking when
        *prompt* is set and something was changed.

        Returns False if the user chose to stay in the editor.
        """
        if not getattr(self, "_edit_session_open", False):
            return True
        if prompt and self._edit_session_changed():
            if not messagebox.askyesno(
                    "Unconfirmed edits",
                    "You are still editing shapes on this image and have not "
                    "pressed Confirm Feature.\n\n"
                    "Discard these edits and continue?",
                    parent=self):
                try:
                    self.image_display_window.lift()
                except Exception:
                    pass
                return False
        self.features = list(getattr(self, "_features_before_edit", []))
        self.edge_points = list(getattr(self, "_edge_points_before_edit", []))
        self._end_edit_session(refresh=refresh)
        return True

    def _edit_session_changed(self):
        init = getattr(self, "_edit_initial_state", None)
        if init is None:
            return False
        return self._snapshot_key(self._snapshot()) != self._snapshot_key(init)

    def _on_display_return(self, _event=None):
        if getattr(self, "_edit_session_open", False):
            self.confirm_feature_cuts()
        return "break"

    def _on_display_escape(self, _event=None):
        if getattr(self, "_edit_session_open", False):
            self.cancel_edit_session()
        return "break"

    def _end_edit_session(self, refresh=True):
        """
        Close the shape editor and restore the normal view.  Does not touch
        self.features: Confirm, Cancel and the navigation hooks set them
        before calling this.
        """
        self._edit_session_open = False
        self._update_default_entries_state()

        # Pending redraw / scroll / drag / preview callbacks
        canvas = getattr(self, "edit_canvas", None)
        for attr in ("_redraw_job", "_scroll_job", "_drag_job"):
            job = getattr(self, attr, None)
            if job and canvas is not None:
                try:
                    canvas.after_cancel(job)
                except Exception:
                    pass
            setattr(self, attr, None)
        pid = getattr(self, "_preview_after_id", None)
        if pid:
            try:
                self.after_cancel(pid)
            except Exception:
                pass
        self._preview_after_id = None
        self._pending_preview = False

        if getattr(self, "_adjust_row", None) is not None:
            try:
                self._adjust_row.destroy()
            except Exception:
                pass
            self._adjust_row = None
        self.edit_original_pil = None

        # disable edit-mode shortcuts and the Enter / Esc bindings
        self._unbind_edit_shortcuts()
        for seq in ("<Return>", "<Escape>"):
            try:
                self.image_display_window.unbind(seq)
            except Exception:
                pass

        # --- Reset editor canvas state so re-editing works ---
        self._poly_id = None
        self._vertex_ids = []
        self.bg_image_id = None
        self._bg_cache.clear()
        self._bg_current_zoom = None
        self._zoom_cache = {"zoom": None, "img": None}
        self._adjusted_proxy = None
        self._vp_render_bounds = None
        self._vp_render_zoom = None
        self._bg_photo_ref = None

        # customtkinter 5.2.2: a CTkComboBox's dropdown menu keeps its DPI
        # scaling callback registered after destroy(); the next DPI change
        # would then raise inside customtkinter and leave the image window
        # dimmed.  Remove the callback before the widgets go.
        combo = getattr(self, "edit_label_combo", None)
        menu = getattr(combo, "_dropdown_menu", None) if combo is not None else None
        if menu is not None:
            try:
                from customtkinter.windows.widgets.scaling import ScalingTracker
                ScalingTracker.remove_widget(menu._set_scaling, menu)
            except Exception:
                pass

        # Destroy the editing UI elements.  The toolbar lives in the right
        # panel; CTkScrollableFrame.destroy() leaves its outer container,
        # so remove that explicitly.
        cf = getattr(self, "control_frame", None)
        outer = getattr(cf, "_parent_frame", None) if cf is not None else None
        for attr in ("edit_canvas", "control_frame", "edit_canvas_container"):
            w = getattr(self, attr, None)
            if w is not None:
                try:
                    w.destroy()
                except Exception:
                    pass
        if outer is not None:
            try:
                outer.destroy()
            except Exception:
                pass
        self._toolbar_groups = []
        self._toolbar_info_label = None
        # Drop references so hasattr() guards treat the editor as closed
        for attr in ("edit_canvas", "edit_label_combo", "edit_cat_entry"):
            if hasattr(self, attr):
                try:
                    delattr(self, attr)
                except Exception:
                    pass

        self.creation_mode = False
        self.freehand_mode = False
        self._hide_vertices_for_current_shape = False
        self._edit_session_features = []
        self.edited_edge_points = []
        self.selected_vertex = None
        self._edit_initial_state = None
        self.edit_history = []
        self.redo_history = []

        if refresh:
            # Restore the normal view in the center (mask display).
            self._restore_center_mask_panel()
            try:
                self.top_center_frame.bind("<Configure>", self.update_mask_display)
            except Exception:
                pass
            # mask or AOI preview, whichever was there before the editor
            # opened (an AOI preview used to be lost here)
            self._refresh_center_panel()
            self.update_edge_display()
            # the preview gets its full height back once the toolbar is gone
            try:
                self.after(80, self.update_edge_display)
            except Exception:
                pass
            self._refresh_label_summary()


    # ---------- POLYGON NAVIGATION (within edit session) ----------

    def _update_poly_nav_label(self):
        """Update the shape counter label in the editor toolbar."""
        lbl = getattr(self, '_poly_nav_label', None)
        if lbl is None:
            return
        try:
            if not lbl.winfo_exists():
                return
        except Exception:
            return
        session = getattr(self, '_edit_session_features', [])
        n_active = len(self.edited_edge_points) if self.edited_edge_points else 0
        n_others = len(session)
        default = self._default_label()
        if n_active >= 1:
            ftype = "polygon" if self.is_polygon_mode else "polyline"
            name = self._active_display() or "(no label)"
            if self._drawing_in_progress():
                text = f"Drawing {ftype} ({n_active} pts) '{name}'"
                p = getattr(self, "_pending_label", None)
                if p is not None and slug(p or default) != slug(self._active_label or default):
                    text += f" | next: '{p or default or '(no label)'}'"
            else:
                text = f"Active: {ftype} ({n_active} pts) '{name}'"
        else:
            text = (f"No active shape (click one to select it) | "
                    f"next: '{self._box_display() or '(no label)'}'")
        if n_others:
            text += f" | {n_others} other(s)"
        lbl.configure(text=text)

    def _swap_active_polygon(self, direction):
        """
        Make the next (+1) or previous (-1) session shape the active one.
        The shapes form a ring, so Prev and Next undo each other.
        """
        session = getattr(self, '_edit_session_features', [])
        if not session:
            print("[Edit] No other shapes to switch to.")
            return

        if direction >= 0:
            self._stash_active_shape()              # active → end of ring
            new_feat = session.pop(0)
        else:
            self._stash_active_shape(position=0)    # active → start of ring
            new_feat = session.pop(-1)

        self._load_active_feature(new_feat)
        self._record_history()
        self._after_active_change()

        print(f"[Edit] Switched to {new_feat[0]} '{self._active_display() or '(no label)'}' "
              f"({len(self.edited_edge_points)} vertices). "
              f"{len(session)} other feature(s) in session.")

    def _stash_active_shape(self, position=None, ask_label=True):
        """
        Move the active shape (if it has 2+ points) into the session as a
        labelled Feature in full-image coordinates.  Returns the Feature or
        None.  The active shape is left empty.

        A shape whose vertices were not touched keeps the exact coordinates
        it was loaded with, so opening the editor only to label or to add
        other shapes never alters existing geometry (the proxy round trip
        would otherwise add floating-point noise).
        """
        # A Category ID typed for this shape but not confirmed with Enter
        self._on_category_committed()
        pts = self.edited_edge_points or []
        if len(pts) < 2:
            self._active_source = None
            self._consume_pending()
            return None
        ftype = "polygon" if self.is_polygon_mode else "polyline"
        if ftype == "polygon" and len({(round(float(x), 6), round(float(y), 6))
                                       for x, y in pts}) < 3:
            print("[Edit] A polygon needs at least 3 different vertices; "
                  "kept as a line instead.")
            ftype = "polyline"
        source = getattr(self, "_active_source", None)
        if source is not None and source[1] == pts:
            full = [list(p) for p in source[0][1]]      # untouched: exact
        else:
            proxy_s = getattr(self, '_edit_proxy_scale', 1.0)
            if proxy_s != 1.0 and proxy_s > 0:
                inv = 1.0 / proxy_s
                full = [[x * inv, y * inv] for x, y in pts]
            else:
                full = [list(pt) for pt in pts]
        feat = Feature(ftype, full, self._stored_label(getattr(self, "_active_label", "")))
        if ask_label and not feat.label and not self._default_label():
            feat = self._ask_label_for(feat)
        if feat.label:
            self._category_id_for(feat.label)      # register a new label
        session = self._edit_session_features
        if position is None:
            session.append(feat)
        else:
            session.insert(position, feat)
        self.edited_edge_points = []
        self.selected_vertex = None
        self._active_source = None
        self._consume_pending(after_label=feat.label)
        return feat

    def _load_active_feature(self, feat):
        """Make *feat* (full-image coords) the active shape."""
        ftype, pts = feat
        self._leave_creation_mode()
        self.is_polygon_mode = (ftype == "polygon")
        self._editing_feature_type = ftype
        s = getattr(self, '_edit_proxy_scale', 1.0)
        if s != 1.0 and s > 0:
            self.edited_edge_points = [[x * s, y * s] for x, y in pts]
        else:
            self.edited_edge_points = [list(pt) for pt in pts]
        self._active_source = (feat, [list(p) for p in self.edited_edge_points])
        self._active_label = feature_label(feat)
        self._pending_label = None
        self.selected_vertex = None
        self._hide_vertices_for_current_shape = False

    def _activate_session_feature(self, idx):
        """Make session shape *idx* the active one; the old active shape
        takes its place in the session."""
        session = self._edit_session_features
        if not (0 <= idx < len(session)):
            return
        feat = session.pop(idx)
        self._stash_active_shape(position=idx)
        self._load_active_feature(feat)
        self._record_history()
        self._after_active_change()
        print(f"[Edit] Selected {feat[0]} '{self._active_display() or '(no label)'}' "
              f"({len(feat[1])} vertices).")

    def _after_active_change(self):
        self._refresh_label_widgets()
        self._update_poly_nav_label()
        self.redraw_canvas()

    @staticmethod
    def _min_dist_to_polyline(p, pts):
        """Smallest distance from point p to the segments of pts (N x 2)."""
        a = pts[:-1]
        b = pts[1:]
        ab = b - a
        ab2 = (ab ** 2).sum(axis=1)
        ab2[ab2 == 0] = 1e-12
        t = np.clip(((p - a) * ab).sum(axis=1) / ab2, 0.0, 1.0)
        proj = a + t[:, None] * ab
        return float(np.sqrt(((p - proj) ** 2).sum(axis=1)).min())

    def _pick_tolerance(self):
        """About 8 screen pixels, in proxy-image units."""
        return 8.0 / max(float(getattr(self, "zoom_scale", 1.0)), 0.1)

    def _near_active_shape(self, x, y):
        pts = self.edited_edge_points or []
        if len(pts) < 2:
            return False
        arr = np.asarray(pts, dtype=float)
        if self.is_polygon_mode and len(arr) >= 3:
            arr = np.vstack([arr, arr[:1]])
        return self._min_dist_to_polyline(np.array([x, y], float), arr) <= self._pick_tolerance()

    def _hit_test_session_shape(self, x, y):
        """Index of the session shape within pick tolerance of (x, y) in
        proxy coords, nearest first, or None."""
        session = getattr(self, '_edit_session_features', [])
        if not session:
            return None
        s = getattr(self, '_edit_proxy_scale', 1.0) or 1.0
        p = np.array([x, y], float)
        best, best_d = None, self._pick_tolerance()
        for i, feat in enumerate(session):
            ftype, pts = feat
            if len(pts) < 2:
                continue
            arr = np.asarray(pts, dtype=float) * s
            if ftype == "polygon" and len(arr) >= 3:
                arr = np.vstack([arr, arr[:1]])
            d = self._min_dist_to_polyline(p, arr)
            if d <= best_d:
                best, best_d = i, d
        return best

    def _bind_vertex_handlers(self):
        """Mouse bindings for vertex editing (select / move / add / delete)."""
        c = self.edit_canvas
        c.bind("<ButtonPress-1>",   self.on_canvas_press)
        c.bind("<B1-Motion>",       self.on_canvas_drag)
        c.bind("<ButtonRelease-1>", self.on_canvas_release)
        c.bind("<Double-Button-1>", self.on_canvas_double_click)

    def _bind_creation_handlers(self, polygon=False):
        """Mouse bindings for click-to-add shape creation."""
        c = self.edit_canvas
        # <Button-1> is the same event as <ButtonPress-1>
        c.bind("<Button-1>",        self.on_canvas_single_click)
        c.bind("<B1-Motion>",       self.on_canvas_drag)
        c.bind("<ButtonRelease-1>", self.on_canvas_release)
        if polygon:
            c.bind("<Double-Button-1>", self._close_polygon_double_click)
        else:
            c.bind("<Double-Button-1>", self._finish_line_double_click)

    def _leave_creation_mode(self):
        """Back to vertex editing if creating a shape or drawing freehand."""
        if getattr(self, "creation_mode", False) or getattr(self, "freehand_mode", False):
            self.creation_mode = False
            self.freehand_mode = False
            if hasattr(self, "edit_canvas"):
                self._bind_vertex_handlers()

    # ---------- LABELS (per-shape Feature ID / Category ID) ----------

    def _get_label_catalog(self):
        cat = getattr(self, "label_catalog", None)
        if cat is None:
            cat = self.label_catalog = LabelCatalog()
        return cat

    def _default_label(self):
        """Feature ID typed in the configuration window (the default label)."""
        e = getattr(self, "feature_id_entry", None)
        try:
            return e.get().strip() if e is not None else ""
        except Exception:
            return ""

    def _default_category_id(self):
        """Category ID entry of the configuration window (same parsing as export)."""
        e = getattr(self, "category_id_entry", None)
        try:
            return int(e.get().strip())
        except Exception:
            return 1

    def _reserved_category_ids(self):
        return {self._default_category_id()} if self._default_label() else set()

    def _stored_label(self, text):
        """
        Label stored on a shape.  Empty when it matches the Feature ID (or
        has no letters or digits), so such shapes keep following the
        Feature ID entry exactly as before per-shape labels existed.
        """
        t = (text or "").strip()
        d = self._default_label()
        if not t or not slug(t) or (d and slug(t) == slug(d)):
            return ""
        return t

    def _active_display(self):
        """Label shown for the active shape: its own, else the Feature ID."""
        return getattr(self, "_active_label", "") or self._default_label()

    @staticmethod
    def _warn_unusable_label(text):
        if text and not slug(text):
            print(f"[Labels] '{text}' has no letters or digits, so it cannot "
                  f"name export files; the shape keeps the Feature ID instead.")
            return True
        return False

    def _display_label(self, feat):
        """The label a feature is exported under."""
        return feature_label(feat) or self._default_label()

    # ---------- Label box state ----------

    def _drawing_in_progress(self):
        """True while a shape is being drawn click by click (or continued)."""
        return (bool(getattr(self, "creation_mode", False))
                and len(self.edited_edge_points or []) >= 1)

    def _box_label(self):
        """Stored-form label the Label box stands for: the label typed for
        the next shape while one is being drawn, else the active shape's."""
        p = getattr(self, "_pending_label", None)
        return p if p is not None else (getattr(self, "_active_label", "") or "")

    def _box_display(self):
        return self._box_label() or self._default_label()

    def _committed_box_label(self):
        hist = getattr(self, "edit_history", None)
        if hist:
            s = hist[-1]
            return s["pending"] if s.get("pending") is not None else s.get("label", "")
        return self._box_label()

    def _set_box_label(self, stored):
        """Apply a label from the Label box: to the next shape while drawing,
        otherwise to the active (selected) shape."""
        if self._drawing_in_progress():
            self._pending_label = stored
        else:
            self._active_label = stored
            self._pending_label = None

    def _consume_pending(self, after_label=None):
        """A shape was finished or removed: the label typed meanwhile (or the
        label the finished shape was given) is the label for the next shape."""
        p = getattr(self, "_pending_label", None)
        if p is not None:
            self._active_label = p
            self._pending_label = None
        elif after_label:
            self._active_label = after_label

    def _ask_label_for(self, feat):
        """A shape was finished with no label while the Feature ID is empty:
        ask for one (leave empty to decide later)."""
        try:
            icon = self.resource_path("launch_logo.png")
        except Exception:
            icon = resource_path("launch_logo.png")
        try:
            text = ask_label_dialog(
                self.image_display_window, icon, "Label this shape",
                "This shape has no label and the Feature ID in the configuration "
                "window is empty, so it cannot be exported yet.\n\n"
                "Label for this shape (leave empty to decide later):")
        except Exception:
            text = None
        self._refocus_canvas()
        if text and self._stored_label(text):
            return with_label(feat, self._stored_label(text))
        if text:
            self._warn_unusable_label(text)
        return feat

    def _on_default_label_changed(self, _event=None):
        """Feature ID / Category ID entry edited: refresh what is shown."""
        self._refresh_label_summary()
        if getattr(self, "_edit_session_open", False):
            self._refresh_label_widgets()
            self._update_poly_nav_label()
            self._draw_reference_features()
            self._draw_poly_in_place()


    def _set_default_entries_state(self, state, names=("feature_id_entry", "category_id_entry")):
        """Enable/disable entries of the configuration window."""
        for name in names:
            e = getattr(self, name, None)
            if e is None:
                continue
            try:
                e.configure(state=state)
            except Exception:
                pass

    def _labels_from_editor(self):
        """True when every confirmed shape of this image has a label of its
        own (set in the shape editor), so the Feature ID and Category ID
        entries have no effect on this image."""
        feats = list(getattr(self, "features", []) or [])
        return bool(feats) and all(feature_label(f) for f in feats)

    def _update_default_entries_state(self):
        """
        Feature ID and Category ID are greyed out while the shape editor is
        open and while the image's labels come from the editor, so a stray
        edit cannot relabel shapes.  They are available again for an image
        with unlabelled shapes, a new detection or another image.  Image ID
        stays editable: the editor has no Image ID and it is still written
        into every file of the image.
        """
        lock = (getattr(self, "_edit_session_open", False)
                or self._labels_from_editor())
        self._set_default_entries_state("disabled" if lock else "normal")


    def _category_id_for(self, label, register=True):
        """
        Category ID for *label*: the Category ID entry for the Feature ID,
        otherwise the catalogue's ID (a new label gets the next free ID
        when *register* is set; otherwise None).
        """
        text = (label or "").strip()
        d = self._default_label()
        if not text or (d and slug(text) == slug(d)):
            return self._default_category_id()
        cat = self._get_label_catalog()
        cid = cat.id_for(text)
        if cid is None and register:
            cid = cat.ensure(text, reserved=self._reserved_category_ids())
        return cid

    def _sync_default_into_catalog(self):
        """Record Feature ID -> Category ID in the catalogue (never refuses)."""
        d = self._default_label()
        if not d or not slug(d):
            return
        try:
            self._get_label_catalog().set_id(d, self._default_category_id())
        except ValueError as e:
            print(f"[Labels] Feature ID '{d}': {e} "
                  f"Change the Category ID in the configuration window.")

    def _label_catalog_for_settings(self):
        """Catalogue entries (Feature ID included) for the settings file."""
        cat = self._get_label_catalog().copy()
        d = self._default_label()
        if d and slug(d):
            try:
                cat.set_id(d, self._default_category_id())
            except ValueError:
                pass
        return cat.to_list()

    def _export_categories_path(self):
        e = getattr(self, "export_path_entry", None)
        try:
            p = e.get().strip() if e is not None else ""
        except Exception:
            p = ""
        if not p:
            return None
        return os.path.join(self._dataset_root(p), "training_dataset_multiclass",
                            "categories.json")

    DATASET_FOLDER = "Dataset"

    @classmethod
    def _dataset_root(cls, export_path):
        """
        Top folder of every dataset export: <export folder>/Dataset, or the
        export folder itself when it already is a folder named "Dataset"
        (so choosing that folder again does not nest Dataset/Dataset).
        """
        name = os.path.basename(os.path.normpath(export_path or ""))
        if name.lower() == cls.DATASET_FOLDER.lower():
            return export_path
        return os.path.join(export_path, cls.DATASET_FOLDER)


    @staticmethod
    def _read_export_categories(path):
        """Category list of a categories.json ([] when absent); raises if unreadable."""
        if not path or not os.path.isfile(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cats = data.get("categories", []) if isinstance(data, dict) else []
        return [c for c in cats if isinstance(c, dict)]

    @staticmethod
    def _read_export_doc(path):
        """The whole categories.json document ({} when absent); raises if unreadable."""
        if not path or not os.path.isfile(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}


    def _merge_export_folder_categories(self):
        """Adopt label IDs from earlier multi-label exports to this folder."""
        path = self._export_categories_path()
        try:
            cats = self._read_export_categories(path)
        except Exception as e:
            print(f"[Labels] Could not read {path}: {type(e).__name__}: {e}")
            return
        for c in self._get_label_catalog().merge(cats):
            print(f"[Labels] {path}: {c}")

    def _label_values(self):
        """Labels offered in the editor's Label box."""
        names, seen = [], set()
        d = self._default_label()
        for n in ([d] if d else []) + self._get_label_catalog().names():
            k = slug(n)
            if n and k not in seen:
                seen.add(k)
                names.append(n)
        return names

    def _label_counts(self, features):
        """[(label, number of features)] in order of first appearance."""
        counts, names = {}, {}
        for f in features:
            name = self._display_label(f)
            k = slug(name) if name else ""
            if k not in counts:
                counts[k] = 0
                names[k] = name
            counts[k] += 1
        return [(names[k], counts[k]) for k in counts]

    def _refresh_label_summary(self):
        """'Labels on this image' line in the configuration window."""
        # Batch mode has no editor, and its worker thread must never touch
        # Tk widgets (update_edge_display can be reached from that thread).
        import threading
        in_batch = getattr(self, "mode", "") == "batch"
        if in_batch or threading.current_thread() is not threading.main_thread():
            return
        self._update_default_entries_state()
        lbl = getattr(self, "label_summary_label", None)
        if lbl is None:
            return
        try:
            if not lbl.winfo_exists():
                return
        except Exception:
            return
        feats = list(getattr(self, "features", []) or [])
        if not any(feature_label(f) for f in feats):
            lbl.configure(text="")
            return
        parts = []
        for name, n in self._label_counts(feats):
            cid = self._category_id_for(name, register=False)
            parts.append(f"{name or '(no label)'} x{n} "
                         f"(cat {cid if cid is not None else 'new'})")
        text = "Labels on this image: " + ", ".join(parts)
        if all(feature_label(f) for f in feats):
            text += ("  |  set in the editor: Feature ID and Category ID are "
                     "locked for this image (Edit Detected Feature to change)")
        lbl.configure(text=text)

    def _build_label_row(self, parent, btn_h=28, font=("Arial", 11)):
        """Label / Category ID rows for the active shape (editor toolbar)."""
        row = ctk.CTkFrame(parent)
        row.pack(side="top", fill="x", pady=(4, 1))
        ctk.CTkLabel(row, text="Label:", font=font).pack(side="left", padx=(6, 2))
        self.edit_label_combo = ctk.CTkComboBox(
            row, width=120, height=btn_h, font=font, dropdown_font=font,
            values=self._label_values(), command=self._on_label_picked)
        self.edit_label_combo.pack(side="left", expand=True, fill="x", padx=(2, 6))
        self.edit_label_combo.set(self._box_display())
        self.edit_label_combo.bind("<KeyRelease>", self._on_label_typed, add="+")
        self.edit_label_combo.bind("<Return>", self._on_label_committed, add="+")
        self.edit_label_combo.bind("<FocusOut>", self._on_label_committed, add="+")
        self.edit_label_combo.bind("<Escape>", self._on_label_escape, add="+")

        row2 = ctk.CTkFrame(parent)
        row2.pack(side="top", fill="x", pady=(1, 2))
        ctk.CTkLabel(row2, text="Category ID:", font=font).pack(side="left", padx=(6, 2))
        self.edit_cat_entry = ctk.CTkEntry(row2, width=50, height=btn_h, font=font,
                                           placeholder_text="auto")
        self.edit_cat_entry.pack(side="left", padx=2)
        self.edit_cat_entry.bind("<Return>", self._on_category_committed, add="+")
        self.edit_cat_entry.bind("<FocusOut>", self._on_category_committed, add="+")
        self.edit_cat_entry.bind("<Escape>", self._on_category_escape, add="+")
        ctk.CTkButton(row2, text="Apply label to all", width=40, height=btn_h,
                      font=font, command=self._btn_apply_label_all
                      ).pack(side="left", expand=True, fill="x", padx=(8, 6))
        self._show_category_for(self._box_label())

    def _refresh_label_widgets(self, keep_text=False):
        combo = getattr(self, "edit_label_combo", None)
        if combo is None:
            return
        try:
            combo.configure(values=self._label_values())
            if not keep_text:
                combo.set(self._box_display())
        except Exception:
            pass
        self._show_category_for(self._box_label())

    def _show_category_for(self, label):
        e = getattr(self, "edit_cat_entry", None)
        if e is None:
            return
        try:
            cid = self._category_id_for(label, register=False)
            e.delete(0, "end")
            if cid is not None:
                e.insert(0, str(cid))
        except Exception:
            pass

    @staticmethod
    def _is_return(event):
        return event is not None and getattr(event, "keysym", "") in ("Return", "KP_Enter")

    def _committed_label(self):
        """Label of the active shape in the latest undo snapshot."""
        hist = getattr(self, "edit_history", None)
        if hist:
            return hist[-1].get("label", "")
        return getattr(self, "_active_label", "")

    def _on_label_typed(self, _event=None):
        """Typing in the Label box: renames the selected shape, or sets the
        label for the next shape while one is being drawn."""
        if not getattr(self, "_edit_session_open", False):
            return
        try:
            text = self.edit_label_combo.get().strip()
        except Exception:
            return
        self._set_box_label(self._stored_label(text))
        self._show_category_for(self._box_label())
        self._draw_poly_in_place()
        self._update_poly_nav_label()

    def _on_label_committed(self, event=None):
        """Enter / leaving the Label box / picking from the list."""
        if not getattr(self, "_edit_session_open", False):
            return
        try:
            text = self.edit_label_combo.get().strip()
        except Exception:
            return
        unusable = self._warn_unusable_label(text)
        stored = self._stored_label(text)
        self._set_box_label(stored)
        if stored:
            self._category_id_for(stored)          # registers a new label
        self._refresh_label_widgets(keep_text=not unusable)
        if self._box_label() != self._committed_box_label():
            self._record_history()
        self._update_poly_nav_label()
        self._draw_poly_in_place()
        if self._is_return(event):
            self._refocus_canvas()
            return "break"

    def _on_label_picked(self, _value=None):
        self._on_label_committed()
        self._refocus_canvas()

    def _on_label_escape(self, _event=None):
        """Esc in the Label box: undo the typing, keep the editor open."""
        if not getattr(self, "_edit_session_open", False):
            return
        hist = self.edit_history[-1] if getattr(self, "edit_history", None) else None
        if hist is not None:
            self._pending_label = hist.get("pending")
            self._active_label = hist.get("label", "")
        self._refresh_label_widgets()
        self._update_poly_nav_label()
        self._draw_poly_in_place()
        self._refocus_canvas()
        return "break"

    def _on_category_committed(self, event=None):
        """Enter / leaving the Category ID box: give the active label an ID."""
        if not getattr(self, "_edit_session_open", False):
            return
        brk = "break" if self._is_return(event) else None
        try:
            raw = self.edit_cat_entry.get().strip()
        except Exception:
            return brk
        label = self._box_label()
        current = self._category_id_for(label, register=False)
        if raw == "" or (current is not None and raw == str(current)):
            self._show_category_for(label)
            if brk:
                self._refocus_canvas()
            return brk
        try:
            cid = int(raw)
            if cid < 1:
                raise ValueError
        except ValueError:
            print("[Labels] Category ID must be a whole number of 1 or more.")
            self._show_category_for(label)
            return brk
        if cid > 65535:
            print("[Labels] Category ID must be 65535 or less (label-map PNG limit).")
            self._show_category_for(label)
            return brk

        d = self._default_label()
        cat = self._get_label_catalog()
        owner = cat.owner_of(cid)
        if not label or (d and slug(label) == slug(d)):
            # The Feature ID's category lives in the configuration window
            if owner is not None and (not d or slug(owner) != slug(d)):
                print(f"[Labels] Category ID {cid} is already used by '{owner}'.")
                self._show_category_for(label)
                return brk
            try:
                self.category_id_entry.delete(0, "end")
                self.category_id_entry.insert(0, str(cid))
            except Exception:
                pass
            if d:
                try:
                    cat.set_id(d, cid)
                except ValueError:
                    pass
            print(f"[Labels] '{d or '(no label)'}' now has category ID {cid} "
                  f"(Category ID in the configuration window).")
        else:
            if d and cid == self._default_category_id():
                print(f"[Labels] Category ID {cid} is used by the Feature ID "
                      f"'{d}' in the configuration window.")
                self._show_category_for(label)
                return brk
            try:
                cat.set_id(label, cid)
            except ValueError as e:
                print(f"[Labels] {e}")
                self._show_category_for(label)
                return brk
            print(f"[Labels] '{cat.name_for(label)}' now has category ID {cid}.")
        self._show_category_for(label)
        self._update_poly_nav_label()
        if brk:
            self._refocus_canvas()
        return brk

    def _on_category_escape(self, _event=None):
        self._show_category_for(getattr(self, "_active_label", ""))
        self._refocus_canvas()
        return "break"

    def _apply_label_to_all(self):
        """Give every shape on the image the label in the Label box."""
        self._on_category_committed()       # a Category ID typed without Enter
        try:
            text = self.edit_label_combo.get().strip()
        except Exception:
            text = self._box_display()
        self._warn_unusable_label(text)
        stored = self._stored_label(text)
        self._active_label = stored
        self._pending_label = None
        if stored:
            self._category_id_for(stored)
        self._edit_session_features = [with_label(f, stored)
                                       for f in self._edit_session_features]
        self._record_history()
        self._after_active_change()
        n = len(self._edit_session_features) + (1 if len(self.edited_edge_points) >= 2 else 0)
        print(f"[Edit] Label '{self._active_display() or '(no label)'}' "
              f"applied to {n} shape(s).")

    def _edit_next_polygon(self):
        self._swap_active_polygon(+1)

    def _edit_prev_polygon(self):
        self._swap_active_polygon(-1)


    # ---------- PREVIEW ADJUST ROW ----------

    def _build_adjust_row(self, parent):
        """Create blue sliders (0..100) under the edit toolbar."""
        # Keep a pristine copy for non-destructive preview
        # Make sure self.edit_original_pil is set once when the edit window opens
        if not hasattr(self, "edit_original_pil") or self.edit_original_pil is None:
            # Fallback: duplicate whatever PIL image you're showing in the edit pane
            base = self._get_current_edit_pil()
            if base is None:
                return  # no image to preview; bail out gracefully
            self.edit_original_pil = base  # already a copy-safe PIL instance


        row = ctk.CTkFrame(parent)
        row.pack(fill="x", padx=8, pady=(4, 2))

        # labels + sliders
        lbl_sat = ctk.CTkLabel(row, text="Saturation", width=80, anchor="w")
        lbl_exp = ctk.CTkLabel(row, text="Exposure",   width=80, anchor="w")
        lbl_hil = ctk.CTkLabel(row, text="Highlights", width=80, anchor="w")

        # Blue color for CustomTkinter sliders
        blue = "#1f6aa5"

        self.slider_sat = ctk.CTkSlider(row, from_=0, to=100,
                                        command=lambda v: self._on_adjust_change(),
                                        progress_color=blue, button_color=blue, fg_color="#0b2740")
        self.slider_exp = ctk.CTkSlider(row, from_=0, to=100,
                                        command=lambda v: self._on_adjust_change(),
                                        progress_color=blue, button_color=blue, fg_color="#0b2740")
        self.slider_hil = ctk.CTkSlider(row, from_=0, to=100,
                                        command=lambda v: self._on_adjust_change(),
                                        progress_color=blue, button_color=blue, fg_color="#0b2740")

        # Set neutral defaults:
        # saturation: 50 = factor 1.0
        # exposure:   50 = factor 1.0
        # highlights: 50 = neutral curve
        for s in (self.slider_sat, self.slider_exp, self.slider_hil):
            s.set(50)

        # one label + slider per row (the toolbar sits in the side panel)
        for r, (lbl, sld) in enumerate(((lbl_sat, self.slider_sat),
                                        (lbl_exp, self.slider_exp),
                                        (lbl_hil, self.slider_hil))):
            lbl.grid(row=r, column=0, padx=(4, 8), pady=3, sticky="w")
            sld.grid(row=r, column=1, padx=(0, 8), pady=3, sticky="ew")

        # allow the sliders to expand
        row.grid_columnconfigure(1, weight=1)

        self._adjust_row = row  # keep handle so we can destroy on confirm


    def _apply_edit_ops_fast(self, pil_img, high_quality=False):
        """
        Cheap live adjustments on a PIL (zoom-sized) RGB image.
        Does: exposure (gain), saturation (via mean), highlights (curve on brights).
        """
        if pil_img is None:
            return None

        arr = np.asarray(pil_img).astype(np.float32) / 255.0  # HxWx3

        # read sliders once
        sat_v = float(self.slider_sat.get())
        exp_v = float(self.slider_exp.get())
        hil_v = float(self.slider_hil.get())

        def midspan(v, span=2.0):
            # 0..100 -> [1/span..span], neutral at 50
            if v >= 50:  return 1.0 + (span - 1.0) * ((v - 50.0) / 50.0)
            else:        return 1.0 - (1.0 - 1.0/span) * ((50.0 - v) / 50.0)

        satf = midspan(sat_v, 2.0)
        expf = midspan(exp_v, 2.0)
        lift = (hil_v - 50.0) / 50.0  # -1..+1

        # exposure (gain)
        if abs(expf - 1.0) > 1e-3:
            arr *= expf

        # saturation: push channels away from per-pixel mean
        if abs(satf - 1.0) > 1e-3:
            mean = arr.mean(axis=2, keepdims=True)
            arr = mean + (arr - mean) * satf

        # highlights: gamma on bright areas only
        if abs(lift) > 1e-3:
            luma = 0.2126 * arr[...,0] + 0.7152 * arr[...,1] + 0.0722 * arr[...,2]
            t = 0.6  # threshold to start affecting
            w = np.clip((luma - t) / (1.0 - t + 1e-6), 0.0, 1.0)[..., None]
            gamma = np.interp(lift, [-1, 1], [1.6, 0.7])  # compress..lift
            curved = np.power(np.clip(arr, 0, 1), gamma)
            arr = arr * (1.0 - w) + curved * w

        arr = np.clip(arr, 0, 1)
        out = (arr * 255.0).astype(np.uint8)
        return Image.fromarray(out)

    def _on_adjust_change(self):
        """Apply non-destructive preview from sliders to the edit pane."""
        if not hasattr(self, "edit_original_pil") or self.edit_original_pil is None:
            return

        sat_v = float(self.slider_sat.get())  # 0..100
        exp_v = float(self.slider_exp.get())  # 0..100
        hil_v = float(self.slider_hil.get())  # 0..100

        # Map sliders -> factors:
        # 50 == neutral (1.0). Below 50 reduces, above 50 increases.
        def _factor_from_mid(v, span=2.0):
            # v in [0,100] -> factor in [1/span .. span], symmetric around 50
            # span=2 => factor range 0.5..2.0
            if v >= 50:
                return 1.0 + (span - 1.0) * ((v - 50.0) / 50.0)
            else:
                return 1.0 - (1.0 - 1.0/span) * ((50.0 - v) / 50.0)

        sat_factor = _factor_from_mid(sat_v, span=2.0)  # saturation factor
        exp_factor = _factor_from_mid(exp_v, span=2.0)  # brightness/exposure factor
        hil_strength = (hil_v - 50.0) / 50.0            # -1..+1

        # Start from the pristine original each time (non-destructive)
        out = self.edit_original_pil

        # Exposure preview (brightness)
        if abs(exp_factor - 1.0) > 1e-3:
            out = ImageEnhance.Brightness(out).enhance(exp_factor)

        # Saturation preview
        if abs(sat_factor - 1.0) > 1e-3:
            out = ImageEnhance.Color(out).enhance(sat_factor)

        # Highlights: gentle lift (> mid) or roll-off
        if abs(hil_strength) > 1e-3:
            out = self._apply_highlights_preview(out, hil_strength)

        # Push to the edit display widget (label/canvas)
        self._set_edit_preview(out)


    def _apply_highlights_preview(self, pil_img, strength):
        """
        strength in [-1, 1]:
          >0 : lift highlights, <0 : compress highlights
        This is a light-weight tone curve that only affects bright values.
        """
        arr = np.asarray(pil_img).astype(np.float32)  # H x W x C
        if arr.ndim == 2:  # gray
            arr = np.stack([arr, arr, arr], axis=-1)

        # work in 0..1
        arr /= 255.0

        # Luma proxy to find bright regions
        luma = 0.2126 * arr[...,0] + 0.7152 * arr[...,1] + 0.0722 * arr[...,2]

        # mask for highlights (above ~0.6), soft transition
        t = 0.6
        w = np.clip((luma - t) / (1.0 - t + 1e-6), 0.0, 1.0)  # 0..1 weight in highlights

        # build a simple curve: y = x^(gamma)  (gamma<1 lifts; >1 darkens)
        # strength +1.0 -> gamma ~0.7 lift; -1.0 -> gamma ~1.6 compress
        gamma = np.interp(strength, [-1, 1], [1.6, 0.7])

        # Apply curve only to highlight region, blend by weight w
        curved = np.power(arr, gamma)
        w = w[..., None]  # broadcast to channels
        arr = arr * (1.0 - w) + curved * w

        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)


    def _set_edit_preview(self, pil_img):
        if not hasattr(self, "edit_canvas") or self.edit_canvas is None:
            return
        if pil_img is None:
            return

        # pil_img is already zoom-sized from _ensure_scaled_base_for_zoom
        self.zoomed_image = ImageTk.PhotoImage(pil_img)

        if getattr(self, "bg_image_id", None):
            self.edit_canvas.itemconfigure(self.bg_image_id, image=self.zoomed_image)
        else:
            self.bg_image_id = self.edit_canvas.create_image(0, 0, anchor=tk.NW, image=self.zoomed_image)

        # keep scrollregion in sync (cheap)
        self.edit_canvas.config(scrollregion=(0, 0, self.zoomed_image.width(), self.zoomed_image.height()))

        # refresh overlays (lines/vertices)
        self._refresh_overlays()



    def _get_current_edit_pil(self):
        """
        Return a PIL.Image for the current base image shown in the edit view.
        Uses _edit_proxy if available (for large images).
        """
        src = getattr(self, '_edit_proxy', None)
        if src is None:
            src = getattr(self, "full_image", None)
        if src is None:
            return None
        return Image.fromarray(cv2.cvtColor(src, cv2.COLOR_BGR2RGB))



    # -------------- CANVAS INTERACTION (EDITING) --------------

    def _apply_preview_to_bg(self):
        """
        Lightweight preview updater that respects the current zoom.
        Now delegates to viewport renderer (which uses _adjusted_proxy
        if sliders are active).
        """
        try:
            self._invalidate_viewport_cache()
            self._render_viewport()
        except Exception:
            # Fallback: at least keep overlays fresh if something goes wrong
            if hasattr(self, "_refresh_overlays"):
                self._refresh_overlays()



    def _zoom_key(self):
        # quantize zoom so small mousewheel steps reuse cache
        return round(float(getattr(self, "zoom_scale", 1.0)), 1)

    def _ensure_bg_cached(self, high_quality=False):
        """Backward-compatible entry point — delegates to viewport renderer."""
        self._render_viewport(high_quality=high_quality)

    # ─── Viewport-based rendering (tile-free zoom optimisation) ───

    def _get_viewport_bounds(self):
        """Return visible viewport bounds in canvas coordinates."""
        try:
            x1 = int(self.edit_canvas.canvasx(0))
            y1 = int(self.edit_canvas.canvasy(0))
            x2 = int(self.edit_canvas.canvasx(self.edit_canvas.winfo_width()))
            y2 = int(self.edit_canvas.canvasy(self.edit_canvas.winfo_height()))
            return (x1, y1, x2, y2)
        except Exception:
            return None

    def _render_viewport(self, high_quality=False):
        """
        Render only the visible portion of the image (plus overscan margin).

        Instead of resizing the *entire* proxy to zoom scale (which at 5×
        on a 2000 px proxy means a 10 000 × 10 000 image), this method
        crops the source to the visible viewport in image coords, then
        scales only that small crop.  Result: memory and CPU stay constant
        regardless of zoom level.

        Coordinate accuracy
        ───────────────────
        Annotations are stored in image coords and mapped to canvas coords
        via ``_scaled(x, y) = (x * zoom, y * zoom)``.  The viewport crop
        is placed at the exact canvas position matching its image origin,
        so overlay alignment is pixel-perfect.
        """
        if not hasattr(self, "edit_canvas") or self.full_image is None:
            return

        # Choose source: adjusted proxy → raw proxy → full image
        src = getattr(self, '_adjusted_proxy', None)
        if src is None:
            src = getattr(self, '_edit_proxy', None)
        if src is None:
            src = self.full_image

        h_src, w_src = src.shape[:2]
        z = max(0.1, min(5.0, float(self.zoom_scale)))

        # Full zoomed dimensions (scrollregion — so scrollbars work)
        full_w = max(1, int(w_src * z))
        full_h = max(1, int(h_src * z))
        self.edit_canvas.config(scrollregion=(0, 0, full_w, full_h))

        # Determine visible viewport
        vp = self._get_viewport_bounds()
        if vp is None:
            vp = (0, 0, full_w, full_h)
        vp_x1, vp_y1, vp_x2, vp_y2 = vp
        vp_w = max(1, vp_x2 - vp_x1)
        vp_h = max(1, vp_y2 - vp_y1)

        # Overscan margin (50 % each side) — avoids re-render on small scrolls
        margin_w = int(vp_w * 0.5)
        margin_h = int(vp_h * 0.5)
        rx1 = max(0, vp_x1 - margin_w)
        ry1 = max(0, vp_y1 - margin_h)
        rx2 = min(full_w, vp_x2 + margin_w)
        ry2 = min(full_h, vp_y2 + margin_h)

        # Skip re-render when the cached image still covers the viewport
        cached = getattr(self, '_vp_render_bounds', None)
        cached_z = getattr(self, '_vp_render_zoom', None)
        if (cached is not None and cached_z == round(z, 3)
                and cached[0] <= vp_x1 and cached[1] <= vp_y1
                and cached[2] >= vp_x2 and cached[3] >= vp_y2):
            return

        # Map render region → image coords
        img_x1 = max(0, int(rx1 / z))
        img_y1 = max(0, int(ry1 / z))
        img_x2 = min(w_src, int(rx2 / z) + 1)
        img_y2 = min(h_src, int(ry2 / z) + 1)
        if img_x2 <= img_x1 or img_y2 <= img_y1:
            return

        # Exact canvas placement from integer image coords
        canvas_x = int(img_x1 * z)
        canvas_y = int(img_y1 * z)
        out_w = max(1, int((img_x2 - img_x1) * z))
        out_h = max(1, int((img_y2 - img_y1) * z))

        # Crop → RGB → resize (always viewport-sized, never enormous)
        crop = src[img_y1:img_y2, img_x1:img_x2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        resample = Image.LANCZOS if high_quality else Image.BILINEAR
        pil = Image.fromarray(crop_rgb).resize((out_w, out_h), resample)

        photo = ImageTk.PhotoImage(pil)

        if getattr(self, "bg_image_id", None):
            self.edit_canvas.coords(self.bg_image_id, canvas_x, canvas_y)
            self.edit_canvas.itemconfigure(self.bg_image_id, image=photo)
        else:
            self.bg_image_id = self.edit_canvas.create_image(
                canvas_x, canvas_y, anchor=tk.NW, image=photo)

        # Prevent GC  &  cache bounds
        self._bg_photo_ref = photo
        self._vp_render_bounds = (rx1, ry1, rx2, ry2)
        self._vp_render_zoom = round(z, 3)
        self._bg_current_zoom = z

    def _invalidate_viewport_cache(self):
        """Force a full viewport re-render on the next draw."""
        self._vp_render_bounds = None
        self._vp_render_zoom = None

    def _on_scroll_update(self, *_):
        """Debounced handler for canvas scroll / pan events."""
        job = getattr(self, '_scroll_job', None)
        if job:
            try:
                self.edit_canvas.after_cancel(job)
            except Exception:
                pass
        self._scroll_job = self.edit_canvas.after(30, self._do_scroll_update)

    def _do_scroll_update(self):
        """Re-render viewport after a scroll or pan."""
        self._scroll_job = None
        if not hasattr(self, "edit_canvas"):
            return
        self._render_viewport()
        self._draw_reference_features()
        self._draw_poly_in_place()
        self._draw_vertices_in_place()

    def _apply_adjustments_to_proxy(self, high_quality=False):
        """
        Apply saturation / exposure / highlights to the small proxy
        image (max ~2000 × 2000).  Result stored as ``_adjusted_proxy``
        (BGR numpy).  This is orders of magnitude faster than applying
        adjustments to the full-zoom-sized image.
        """
        src = getattr(self, '_edit_proxy', None)
        if src is None:
            src = self.full_image
        if src is None:
            self._adjusted_proxy = None
            return

        sat_v = float(self.slider_sat.get())
        exp_v = float(self.slider_exp.get())
        hil_v = float(self.slider_hil.get())

        # Neutral → clear adjusted proxy (use raw)
        if abs(sat_v - 50) < 0.5 and abs(exp_v - 50) < 0.5 and abs(hil_v - 50) < 0.5:
            self._adjusted_proxy = None
            return

        arr = src.astype(np.float32) / 255.0          # BGR HxWx3
        arr = arr[:, :, ::-1].copy()                   # → RGB

        def midspan(v, span=2.0):
            if v >= 50:
                return 1.0 + (span - 1.0) * ((v - 50.0) / 50.0)
            return 1.0 - (1.0 - 1.0 / span) * ((50.0 - v) / 50.0)

        expf = midspan(exp_v, 2.0)
        satf = midspan(sat_v, 2.0)
        lift = (hil_v - 50.0) / 50.0

        if abs(expf - 1.0) > 1e-3:
            arr *= expf
        if abs(satf - 1.0) > 1e-3:
            mean = arr.mean(axis=2, keepdims=True)
            arr = mean + (arr - mean) * satf
        if abs(lift) > 1e-3:
            luma = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
            t = 0.6
            w = np.clip((luma - t) / (1.0 - t + 1e-6), 0.0, 1.0)[..., None]
            gamma = np.interp(lift, [-1, 1], [1.6, 0.7])
            curved = np.power(np.clip(arr, 0, 1), gamma)
            arr = arr * (1.0 - w) + curved * w

        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        self._adjusted_proxy = arr[:, :, ::-1].copy()  # → BGR


    def redraw_canvas(self, event=None):
        # Throttle rapid resize with after_idle
        if self._redraw_job:
            self.edit_canvas.after_cancel(self._redraw_job)
        def _do():
            if not hasattr(self, "edit_canvas"): return
            # Invalidate viewport cache on explicit redraw (zoom change, etc.)
            self._invalidate_viewport_cache()
            self._render_viewport()
            # (re)draw overlays in place
            self._draw_reference_features()
            self._draw_poly_in_place()
            self._draw_vertices_in_place()
        self._redraw_job = self.edit_canvas.after_idle(_do)

    def _scaled(self, x, y):
        z = float(getattr(self, "zoom_scale", 1.0))
        return x * z, y * z

    # Colours on the edit canvas.  The active shape (the one being drawn or
    # selected) is bright and colour-coded by type; every other shape of the
    # session is grey so it is clearly not the one being edited.  Confirmed
    # (committed) colours live in hsv_mask_processing.update_edge_display.
    _COLOR_ACTIVE_LINE    = "#00e5ff"    # active polyline (bright cyan)
    _COLOR_ACTIVE_POLYGON = "#ffaa00"    # active polygon  (gold)
    _COLOR_REF_LINE       = "#c8c8c8"    # inactive polyline (light grey)
    _COLOR_REF_POLYGON    = "#9a9a9a"    # inactive polygon  (mid grey)

    def _draw_reference_features(self):
        """
        Draw all session shapes (everything except the active one) on the
        edit canvas, each with its label, in grey (polylines lighter than
        polygons).
        """
        if not hasattr(self, "edit_canvas"):
            return
        # Delete any previous reference lines and labels
        self.edit_canvas.delete("__ref_feature__")

        session = getattr(self, '_edit_session_features', [])
        if not session:
            return

        proxy_s = getattr(self, '_edit_proxy_scale', 1.0)
        for feat in session:
            ftype, pts = feat
            if len(pts) < 2:
                continue
            # Convert full-image coords → proxy coords → zoom-scaled canvas coords
            flat = []
            for x, y in pts:
                px, py = x * proxy_s, y * proxy_s
                sx, sy = self._scaled(px, py)
                flat.extend([sx, sy])
            is_closed = (ftype == "polygon")
            if is_closed and len(pts) >= 3:
                # Close the loop visually
                px, py = pts[0][0] * proxy_s, pts[0][1] * proxy_s
                sx, sy = self._scaled(px, py)
                flat.extend([sx, sy])
            colour = self._COLOR_REF_POLYGON if is_closed else self._COLOR_REF_LINE
            self.edit_canvas.create_line(
                *flat, fill=colour, width=2, tags="__ref_feature__"
            )
            self._canvas_text(flat[0], flat[1],
                              self._display_label(feat) or "(no label)",
                              colour, "__ref_feature__")

    def _canvas_text(self, x, y, text, colour, tag):
        """Label text with a dark shadow, just above a shape's first vertex."""
        for dx, dy, fill in ((1, 1, "black"), (0, 0, colour)):
            self.edit_canvas.create_text(
                x + 6 + dx, y - 6 + dy, text=text, anchor="sw", fill=fill,
                font=("Arial", 10, "bold"), tags=tag)

    def _draw_poly_in_place(self):
        if not hasattr(self, "edit_canvas"): return
        pts = getattr(self, "edited_edge_points", []) or []
        self.edit_canvas.delete("__poly_label__")
        if len(pts) < 2:
            # Remove the line item.  (coords(item, ()) only *reads* the
            # coordinates in Tk, so it left the old line on screen.)
            if getattr(self, "_poly_id", None) is not None:
                try:
                    self.edit_canvas.delete(self._poly_id)
                except Exception:
                    pass
                self._poly_id = None
            return

        flat = []
        for x, y in pts:
            sx, sy = self._scaled(x, y)
            flat.extend([sx, sy])

        polygon = getattr(self, "is_polygon_mode", False)
        # A finished polygon is drawn closed; one being drawn stays open
        if (polygon and not getattr(self, "creation_mode", False)
                and len(pts) >= 3 and list(pts[0]) != list(pts[-1])):
            flat.extend(flat[:2])

        # Choose colour by type so polygons (orange) are visually distinct
        # from polylines (cyan).  Refresh fill on each redraw so toggling
        # mode (e/p/c shortcuts) updates the colour live.
        colour = (self._COLOR_ACTIVE_POLYGON if polygon
                  else self._COLOR_ACTIVE_LINE)

        if getattr(self, "_poly_id", None) is None:
            self._poly_id = self.edit_canvas.create_line(
                *flat, fill=colour, width=2, tags="__poly__"
            )
        else:
            self.edit_canvas.coords(self._poly_id, *flat)
            try:
                self.edit_canvas.itemconfigure(self._poly_id, fill=colour)
            except Exception:
                pass

        self._canvas_text(flat[0], flat[1],
                          self._active_display() or "(no label)",
                          colour, "__poly_label__")

    def _draw_vertices_in_place(self):
        if not hasattr(self, "edit_canvas"): return
        pts = getattr(self, "edited_edge_points", []) or []

        r = 5
        if not hasattr(self, "_vertex_ids"):
            self._vertex_ids = []

        hide_vertices = bool(
            getattr(self, "freehand_mode", False)
            or getattr(self, "_hide_vertices_for_current_shape", False)
        )

        # add missing
        while len(self._vertex_ids) < len(pts):
            vid = self.edit_canvas.create_oval(0, 0, 0, 0,
                                               fill="red", outline="black",
                                               tags="__vertex__")
            self._vertex_ids.append(vid)
        # remove extras
        while len(self._vertex_ids) > len(pts):
            vid = self._vertex_ids.pop()
            try: self.edit_canvas.delete(vid)
            except: pass

        if hide_vertices:
            for vid in self._vertex_ids:
                try:
                    self.edit_canvas.itemconfigure(vid, state="hidden")
                except Exception:
                    pass
            return

        # position all
        for i, (x, y) in enumerate(pts):
            sx, sy = self._scaled(x, y)
            vid = self._vertex_ids[i]
            self.edit_canvas.coords(vid, sx - r, sy - r, sx + r, sy + r)
            try:
                self.edit_canvas.itemconfigure(vid, state="normal")
            except Exception:
                pass

    def _refresh_overlays(self):
        if getattr(self, "_skip_overlay", False):
            return
        self._draw_poly_in_place()
        self._draw_vertices_in_place()


    def draw_edge_on_canvas(self, *args, **kwargs):
        # Backward-compatible wrapper
        self._refresh_overlays()

    def _snapshot(self):
        """Whole edit session: active shape, its type and label, and every
        other shape.  Lists are copied; shapes are never changed in place."""
        return {
            "active": list(self.edited_edge_points or []),
            "poly": bool(getattr(self, "is_polygon_mode", False)),
            "label": getattr(self, "_active_label", "") or "",
            "session": list(getattr(self, "_edit_session_features", []) or []),
            "creating": bool(getattr(self, "creation_mode", False)),
            "source": getattr(self, "_active_source", None),
            "pending": getattr(self, "_pending_label", None),
        }

    def _snapshot_key(self, snap):
        """Comparable form of a snapshot (used to detect unsaved changes)."""
        default = self._default_label()

        def norm(pts):
            return tuple((round(float(p[0]), 4), round(float(p[1]), 4)) for p in pts)

        active = snap["active"]
        a = ((norm(active), bool(snap["poly"]), slug(snap["label"] or default))
             if len(active) >= 2 else None)
        sess = tuple((f[0], norm(f[1]), slug(feature_label(f) or default))
                     for f in snap["session"])
        return (a, sess)

    def _apply_snapshot(self, snap):
        self.edited_edge_points = list(snap["active"])
        self.is_polygon_mode = bool(snap["poly"])
        self._active_label = snap["label"]
        self._edit_session_features = list(snap["session"])
        self._active_source = snap.get("source")
        self._pending_label = snap.get("pending")
        self.selected_vertex = None
        # Restore click-to-add or vertex-edit mode together with the shape,
        # so a click after Undo/Redo does what the restored state expects
        if snap.get("creating") and hasattr(self, "edit_canvas"):
            self.freehand_mode = False
            self.creation_mode = True
            self._bind_creation_handlers(polygon=self.is_polygon_mode)
        else:
            self._leave_creation_mode()
        self._after_active_change()

    def _reset_history(self):
        self.edit_history = [self._snapshot()]
        self.redo_history = []

    def _record_history(self):
        """Push a snapshot of the whole session and clear redo (new branch)."""
        self.edit_history.append(self._snapshot())
        # optional cap to keep memory sane
        if len(self.edit_history) > 50:
            self.edit_history.pop(0)
        # any new action invalidates redo chain
        self.redo_history.clear()


    def on_canvas_single_click(self, event):
        """
        Click-to-add shape creation.  With no shape in progress, a click on
        another shape selects it instead (so it can be renamed or edited).
        In polygon mode a click near the first vertex closes the polygon.
        """
        try:
            self.edit_canvas.focus_set()
        except Exception:
            pass
        x_canvas = self.edit_canvas.canvasx(event.x)
        y_canvas = self.edit_canvas.canvasy(event.y)
        x_img = x_canvas / self.zoom_scale
        y_img = y_canvas / self.zoom_scale

        if not self.edited_edge_points:
            idx = self._hit_test_session_shape(x_img, y_img)
            if idx is not None:
                self._activate_session_feature(idx)
                print("[Edit] Shape selected: type its label, edit its vertices, "
                      "or press Create Line / Create Polygon for a new shape.")
                return

        if self.is_polygon_mode and len(self.edited_edge_points) > 2:
            # If close to the first vertex => close the polygon
            first_pt = self.edited_edge_points[0]
            if self.distance(x_img, y_img, first_pt[0], first_pt[1]) < 10:
                self._close_active_polygon()
                print("Polygon closed and added to session. You can create "
                      "more or press Confirm.")
                return

        # For polylines, extend at whichever endpoint (first or last) is
        # nearer to the click, so the user can grow the line in either
        # direction (e.g. extending a shoreline both north and south).
        if (not self.is_polygon_mode) and len(self.edited_edge_points) >= 2:
            first_pt = self.edited_edge_points[0]
            last_pt  = self.edited_edge_points[-1]
            d_first = self.distance(x_img, y_img, first_pt[0], first_pt[1])
            d_last  = self.distance(x_img, y_img, last_pt[0],  last_pt[1])
            if d_first < d_last:
                self.edited_edge_points.insert(0, [x_img, y_img])
            else:
                self.edited_edge_points.append([x_img, y_img])
        else:
            self.edited_edge_points.append([x_img, y_img])

        self._record_history()
        self._update_poly_nav_label()
        self.redraw_canvas()

    def _finish_line_double_click(self, event):
        """Double-click ends the line being drawn (the first click of the
        double-click already placed its last vertex)."""
        if self.is_polygon_mode or not getattr(self, "creation_mode", False):
            return
        if len(self.edited_edge_points) < 2:
            return
        self._finish_active_shape()
        print("[Edit] Line finished. Type the next label and draw the next "
              "shape, or click a shape to select it.")

    def _finish_active_shape(self):
        """Finish the shape being drawn: it joins the session with its
        label; the next click starts a new shape."""
        self._stash_active_shape()
        self._record_history()
        self._after_active_change()

    def on_canvas_press(self, event):
        try:
            self.edit_canvas.focus_set()
        except Exception:
            pass
        x_canvas = self.edit_canvas.canvasx(event.x)
        y_canvas = self.edit_canvas.canvasy(event.y)
        x_img = x_canvas / self.zoom_scale
        y_img = y_canvas / self.zoom_scale

        min_dist = 10
        self.selected_vertex = None
        self._just_selected_shape = False
        for idx, (vx, vy) in enumerate(self.edited_edge_points):
            if self.distance(x_img, y_img, vx, vy) < min_dist:
                self.selected_vertex = idx
                self.edit_canvas.itemconfig(f"vertex_{idx}", fill="yellow")
                return

        # Not on a vertex: clicking another shape makes it the active one
        if self._near_active_shape(x_img, y_img):
            return
        idx = self._hit_test_session_shape(x_img, y_img)
        if idx is not None:
            self._activate_session_feature(idx)
            self._just_selected_shape = True

    def on_canvas_drag(self, event):
        if self.selected_vertex is None:
            return
        if self.selected_vertex >= len(self._vertex_ids):
            return
        x_canvas = self.edit_canvas.canvasx(event.x)
        y_canvas = self.edit_canvas.canvasy(event.y)
        z = float(getattr(self, "zoom_scale", 1.0))
        self.edited_edge_points[self.selected_vertex] = [x_canvas / z, y_canvas / z]

        # Throttle coordinate pushes to the canvas (~60 fps)
        if self._drag_job:
            self.edit_canvas.after_cancel(self._drag_job)
        def _do():
            # move just this vertex oval
            r = 5
            sx, sy = x_canvas, y_canvas
            vid = self._vertex_ids[self.selected_vertex]
            self.edit_canvas.coords(vid, sx - r, sy - r, sx + r, sy + r)
            # update polyline coords for all points (fast in Tk)
            self._draw_poly_in_place()
        self._drag_job = self.edit_canvas.after(16, _do)

    def on_canvas_release(self, event):
        if self.selected_vertex is not None:
            self._record_history()
        self.selected_vertex = None


    def on_canvas_double_click(self, event):
        # The first click of this double-click selected another shape:
        # do not also delete or add a vertex on it.
        if getattr(self, "_just_selected_shape", False):
            self._just_selected_shape = False
            return
        x_canvas = self.edit_canvas.canvasx(event.x)
        y_canvas = self.edit_canvas.canvasy(event.y)
        x_img = x_canvas / self.zoom_scale
        y_img = y_canvas / self.zoom_scale

        threshold = 10
        if self.vertex_mode == "delete":
            for idx, (vx, vy) in enumerate(self.edited_edge_points):
                dist = self.distance(x_img, y_img, vx, vy)
                if dist < threshold:
                    del self.edited_edge_points[idx]
                    self._record_history()
                    self.redraw_canvas()
                    return
        elif self.vertex_mode == "add":
            # Insert near the nearest segment
            best_distance = float("inf")
            best_index = None
            num_points = len(self.edited_edge_points)
            if num_points < 1:
                self.edited_edge_points.append([x_img, y_img])
            else:
                for i in range(num_points - 1):
                    A = self.edited_edge_points[i]
                    B = self.edited_edge_points[i + 1]
                    dist, _ = self.distance_to_segment((x_img, y_img), A, B)
                    if dist < best_distance:
                        best_distance = dist
                        best_index = i + 1
                if best_index is None:
                    self.edited_edge_points.append([x_img, y_img])
                else:
                    self.edited_edge_points.insert(best_index, [x_img, y_img])
            self._record_history()
            self.redraw_canvas()

    def distance_to_segment(self, P, A, B):
        A = np.array(A, dtype=float)
        B = np.array(B, dtype=float)
        P = np.array(P, dtype=float)
        AB = B - A
        ab_squared = np.dot(AB, AB)
        if ab_squared == 0:
            return np.linalg.norm(P - A), A
        t = np.clip(np.dot(P - A, AB) / ab_squared, 0, 1)
        projection = A + t * AB
        distance = np.linalg.norm(P - projection)
        return distance, projection

    def delete_selected_vertex(self, event):
        if self.selected_vertex is not None:
            del self.edited_edge_points[self.selected_vertex]
            self.selected_vertex = None
            self._record_history()
            self.redraw_canvas()

    def set_vertex_mode(self, mode):
        self.vertex_mode = mode
        self._hide_vertices_for_current_shape = False
        # Leave click-to-add or freehand mode and restore the edit bindings
        self._leave_creation_mode()
        self.redraw_canvas()


    def delete_all_vertices(self):
        """Backward-compatible name for Delete Shape (active shape only)."""
        self.delete_active_shape()

    def delete_active_shape(self):
        """Delete Shape: remove the active shape (red vertices) only."""
        if not self.edited_edge_points:
            print("[Edit] No active shape to delete. Click a shape or use "
                  "Prev/Next to select one.")
            return
        self.edited_edge_points = []
        self._active_source = None
        self.selected_vertex = None
        if not getattr(self, "creation_mode", False):
            self.is_polygon_mode = False   # a freehand / continued line next is a line
        self._consume_pending()
        self._record_history()
        self._refresh_label_widgets()
        self._update_poly_nav_label()
        self.redraw_canvas()

    def delete_all_shapes(self):
        """Delete All: remove every shape on this image (undoable)."""
        n = (1 if self.edited_edge_points else 0) + len(self._edit_session_features)
        if n == 0:
            print("[Edit] Nothing to delete.")
            return
        if n >= 2 and not messagebox.askyesno(
                "Delete all shapes",
                f"Delete all {n} shapes on this image?\n\n"
                "Undo (U) brings them back.",
                parent=self.image_display_window):
            self._refocus_canvas()
            return
        self.edited_edge_points = []
        self._edit_session_features = []
        self._active_source = None
        self.selected_vertex = None
        if not getattr(self, "creation_mode", False):
            self.is_polygon_mode = False   # a freehand / continued line next is a line
        # A clean start: the next shape follows the Feature ID again
        self._active_label = ""
        self._pending_label = None
        self._record_history()
        self._refresh_label_widgets()
        self._update_poly_nav_label()
        self.redraw_canvas()
        print(f"[Edit] Deleted {n} shape(s). Draw a new shape, or press "
              f"Confirm Feature to leave this image without features.")

    def undo_last_action(self, event=None):
        """Step back one action (vertex edits, shapes added, closed,
        selected or deleted, label changes, Reset)."""
        if len(self.edit_history) > 1:
            current = self.edit_history.pop()              # the state we're leaving
            self.redo_history.append(current)              # enable redo
            self._apply_snapshot(self.edit_history[-1])
        else:
            print("[Edit] Nothing to undo.")

    def redo_last_action(self, event=None):
        """Re-apply the next state from redo."""
        if self.redo_history:
            nxt = self.redo_history.pop()
            self.edit_history.append(nxt)
            self._apply_snapshot(nxt)
        else:
            print("[Edit] Nothing to redo.")


    def reset_to_initial(self):
        """Reset: put every shape back the way it was when the editor
        opened (undoable)."""
        init = getattr(self, "_edit_initial_state", None)
        if init is None:
            return
        if not self._edit_session_changed():
            print("[Edit] Nothing to reset.")
            return
        if not messagebox.askyesno(
                "Reset",
                "Put all shapes back the way they were when the editor "
                "opened?\n\nUndo (U) brings your edits back.",
                parent=self.image_display_window):
            self._refocus_canvas()
            return
        self._leave_creation_mode()
        self._apply_snapshot(init)
        self._record_history()
        print("[Edit] Shapes reset to the state when the editor opened.")

    # -------------- ZOOM & PAN --------------

    def on_mousewheel(self, event):
        x_img = (event.x - self.pan_x) / self.zoom_scale
        y_img = (event.y - self.pan_y) / self.zoom_scale

        if event.num == 4 or event.delta > 0:
            self.zoom_scale *= 1.1
        else:
            self.zoom_scale *= 0.9

        self.zoom_scale = max(0.1, min(self.zoom_scale, 5.0))
        self.pan_x = event.x - x_img * self.zoom_scale
        self.pan_y = event.y - y_img * self.zoom_scale
        self.redraw_canvas()
        self._invalidate_zoom_cache()

    def adjust_zoom(self, factor):
        self.zoom_scale *= factor
        self.zoom_scale = max(0.1, min(self.zoom_scale, 5.0))
        self.redraw_canvas()
        self._invalidate_zoom_cache()

    def reset_view(self):
        self.zoom_scale = 1.0
        self.pan_x = self.pan_y = 0
        self.redraw_canvas()
        self._invalidate_zoom_cache()

    # -------------- EXPORT METHODS --------------

    def _sanitize_feature_id(self, s: str) -> str:
        """Make a filesystem-safe suffix from the feature ID."""
        if not s:
            return ""
        s = s.strip().lower().replace(" ", "_")
        # keep alnum, underscore, dash
        return "".join(ch for ch in s if ch.isalnum() or ch in ("_", "-"))


    def _export_class_points(self, export_path):
        """
        Export colour-picker sample points to a CSV file under
        <export_path>/training dataset/class_points/.

        Each row contains:
            class_name, class_label, x, y, R, G, B, H, S, V,
            intensity, image_name, image_width, image_height,
            patch_radius, detection_method, output_mode

        Returns the output file path, or None if there are no points.
        """
        pts = getattr(self, "color_pick_points", {})
        remove_pts = pts.get("remove", [])
        keep_pts = pts.get("keep", [])
        if not remove_pts and not keep_pts:
            return None
        if self.full_image is None or self.image_path is None:
            return None

        class_a_name = getattr(self, 'color_pick_class_a_name', None)
        class_a_name = class_a_name.get().strip() if class_a_name else "class_A"
        class_b_name = getattr(self, 'color_pick_class_b_name', None)
        class_b_name = class_b_name.get().strip() if class_b_name else "class_B"
        if not class_a_name:
            class_a_name = "class_A"
        if not class_b_name:
            class_b_name = "class_B"

        img = self.full_image
        h_img, w_img = img.shape[:2]
        is_color = img.ndim == 3

        # Prepare HSV version for HSV column data
        if is_color:
            hsv_img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2HSV)
        else:
            hsv_img = None

        base_name = os.path.basename(self.image_path)
        stem = os.path.splitext(base_name)[0]

        # Read detection settings
        method = getattr(self, 'color_pick_method', None)
        method = method.get() if method else "unknown"
        output_mode = getattr(self, 'color_pick_output_mode', None)
        output_mode = output_mode.get() if output_mode else "unknown"
        patch_r = getattr(self, 'color_pick_patch_radius', None)
        patch_r = patch_r.get() if patch_r else "7"

        # Build output folder
        base_folder = os.path.join(self._dataset_root(export_path),
                                   "training dataset", "class_points")
        os.makedirs(base_folder, exist_ok=True)
        out_path = os.path.join(base_folder, f"{stem}_class_points.csv")

        header = (
            "class_name,class_label,x,y,"
            "R,G,B,H,S,V,intensity,"
            "image_name,image_width,image_height,"
            "patch_radius,detection_method,output_mode\n"
        )

        def _row(class_name, class_label, x, y):
            x = max(0, min(int(x), w_img - 1))
            y = max(0, min(int(y), h_img - 1))
            if is_color:
                bgr = img[y, x]
                r, g, b = int(bgr[2]), int(bgr[1]), int(bgr[0])
                hv = hsv_img[y, x]
                h_val, s_val, v_val = int(hv[0]), int(hv[1]), int(hv[2])
                intensity = round(0.2989 * r + 0.5870 * g + 0.1140 * b, 2)
            else:
                val = int(img[y, x])
                r = g = b = val
                h_val = s_val = 0
                v_val = val
                intensity = float(val)
            return (
                f"{class_name},{class_label},{x},{y},"
                f"{r},{g},{b},{h_val},{s_val},{v_val},{intensity},"
                f"{base_name},{w_img},{h_img},"
                f"{patch_r},{method},{output_mode}\n"
            )

        with open(out_path, "w") as f:
            f.write(header)
            for x, y in remove_pts:
                f.write(_row(class_a_name, "remove", x, y))
            for x, y in keep_pts:
                f.write(_row(class_b_name, "keep", x, y))

        n_total = len(remove_pts) + len(keep_pts)
        print(f"[export_class_points] Wrote {n_total} sample points "
              f"({len(remove_pts)} {class_a_name}, {len(keep_pts)} {class_b_name}) → {out_path}")
        return out_path


    @staticmethod
    def _overlay_colour(bgr, img):
        """
        Drawing colour for *img*.  On a 4-channel (BGRA) image the colour
        gets full opacity: OpenCV pads a 3-value colour with alpha 0, which
        made drawn features invisible in QGIS and web browsers.
        """
        if getattr(img, "ndim", 0) == 3 and img.shape[2] == 4:
            if np.issubdtype(img.dtype, np.integer):
                opaque = int(np.iinfo(img.dtype).max)     # 255, or 65535 for 16-bit
            else:
                opaque = 1.0
            return (int(bgr[0]), int(bgr[1]), int(bgr[2]), opaque)
        return tuple(bgr)

    _WRITE_FAILED_HINT = (
        "The export folder path may be too long for Windows (over 260 "
        "characters), the disk may be full, or the file may be open in "
        "another program. Choose a shorter export folder and export again.")

    def export_training_data(self):
        """
        Export the confirmed features as training data.

        Features are grouped by label: a feature's own label, or the
        Feature ID for features without one.  Each label gets the five
        files every earlier release wrote, named <image>_<label>, so an
        image with one label exports exactly as before.  An image with two
        or more labels also gets a combined multi-class set in
        Dataset/training_dataset_multiclass/ (see _write_multiclass_bundle).
        """
        import warnings
        from rasterio.errors import NotGeoreferencedWarning
        warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
        warnings.filterwarnings(
            "ignore", category=UserWarning, message=".*crs.*was not provided.*"
        )

        # 0) Never export while the editor holds the shapes
        if getattr(self, "_edit_session_open", False):
            messagebox.showwarning(
                "Finish editing first",
                "The shape editor is still open.\n\n"
                "Press Confirm Feature (or Cancel) in the image window, then "
                "export again. Nothing was exported.",
                parent=self)
            try:
                self.image_display_window.lift()
            except Exception:
                pass
            return

        # 1) Read user input
        feature_name_raw = self.feature_id_entry.get().strip()
        needs_default = (not self.features) or any(
            not feature_label(f) for f in self.features)
        if needs_default and not feature_name_raw:
            if any(feature_label(f) for f in self.features):
                msg = ("Some shapes have no label of their own and the Feature "
                       "ID is empty.\n\nType a Feature ID (it is used for "
                       "shapes without a label), or give every shape a label "
                       "in the editor.")
            else:
                msg = "Feature ID is missing. Provide a label/name for your dataset."
            messagebox.showerror("Error", msg, parent=self)
            return

        try:
            image_id = int(self.image_id_entry.get().strip())
        except ValueError:
            image_id = 1
        try:
            category_id = int(self.category_id_entry.get().strip())
        except ValueError:
            category_id = 1

        # 2) Ensure we have an image and features
        if not isinstance(self.full_image, np.ndarray):
            messagebox.showerror("Error", "No full image is loaded or invalid image data.", parent=self)
            return
        if not self.features and self.edge_points:
            self.features = [("polyline", self.edge_points.copy())]
        if not self.features:
            messagebox.showerror("Error", "No features to export. Please confirm a shape or polygon first.", parent=self)
            return

        # 3) Prepare output folders
        export_path = self.export_path_entry.get().strip()
        if not export_path:
            messagebox.showerror("Error", "Please specify an export folder.", parent=self)
            return

        groups = self._resolve_export_groups(feature_name_raw, category_id, export_path)
        if not groups:
            return

        base_folder     = os.path.join(self._dataset_root(export_path), "training dataset")
        images_folder   = os.path.join(base_folder, "images")
        masks_folder    = os.path.join(base_folder, "masks")
        overlays_folder = os.path.join(base_folder, "overlays")
        geojson_folder  = os.path.join(base_folder, "geojson")
        coco_folder     = os.path.join(base_folder, "coco")
        os.makedirs(images_folder, exist_ok=True)
        os.makedirs(masks_folder,  exist_ok=True)
        os.makedirs(overlays_folder, exist_ok=True)
        os.makedirs(geojson_folder, exist_ok=True)
        os.makedirs(coco_folder, exist_ok=True)

        base_name = os.path.basename(self.image_path)           # e.g. name.tif
        stem, src_ext = os.path.splitext(base_name)             # stem, .tif

        height, width = self.full_image.shape[:2]

        # 5) Read georeference if any
        #
        # Failing to READ the image is not the same as an image that simply
        # has no georeference: a PNG/JPG opens fine and reports crs=None,
        # while a locked or missing file raises.  Sharing one silent fallback
        # between the two writes a GeoJSON that looks perfectly valid but
        # holds pixel coordinates, which Create DEM cannot use.
        georef_error = None
        try:
            with rasterio.open(self.image_path) as src:
                transform = src.transform
                crs = src.crs
        except Exception as e:
            transform = None
            crs = None
            georef_error = e
        if georef_error is not None:
            print(f"[export_training_data] Could not read georeference from "
                  f"{base_name}: {type(georef_error).__name__}: {georef_error}")
            messagebox.showwarning(
                "Georeference unreadable",
                f"Could not read georeference information from:\n{base_name}\n\n"
                f"{type(georef_error).__name__}: {georef_error}\n\n"
                "The GeoJSON will be written with PIXEL coordinates, which "
                "tools such as Create DEM cannot use.\n\n"
                "Close the image in any other program (QGIS, etc.) and export "
                "again.",
                parent=self)
        elif crs is None:
            print(f"[export_training_data] No georeference found on {base_name}; using pixel coords in GeoJSON.")

        thickness = int(self.edge_thickness_slider.get()) if hasattr(self, 'edge_thickness_slider') else 2

        ctx = {
            "image_id": image_id, "stem": stem, "src_ext": src_ext,
            "width": width, "height": height,
            "transform": transform, "crs": crs, "thickness": thickness,
            "folders": {"images": images_folder, "masks": masks_folder,
                        "overlays": overlays_folder, "geojson": geojson_folder,
                        "coco": coco_folder},
        }

        # ── One label: exactly the export of earlier releases ──
        if len(groups) == 1:
            g = groups[0]
            res = self._write_training_set(g["features"], g["name"], g["category_id"], ctx)

            # final pop-up + console summary
            class_pts_path = self._export_class_points(export_path)
            print(f"[export_training_data] Export complete for {base_name}:")
            print(f"    image copy → {res['image']}")
            print(f"    mask       → {res['mask']}")
            print(f"    overlay    → {res['overlay']}")
            print(f"    geojson    → {res['geojson']}")
            print(f"    coco       → {res['coco']}")
            if class_pts_path:
                print(f"    class pts  → {class_pts_path}")
            if res["failed"]:
                self._report_write_failures(res["failed"])
                return
            cls_msg = f"\n- Class pts ⇒ {class_pts_path}" if class_pts_path else ""
            messagebox.showinfo(
                "Export Training Data",
                f"Export complete:\n"
                f"- Image   ⇒ {res['image']}\n"
                f"- Mask    ⇒ {res['mask']}\n"
                f"- Overlay ⇒ {res['overlay']}\n"
                f"- GeoJSON ⇒ {res['geojson']}\n"
                f"- COCO    ⇒ {res['coco']}"
                f"{cls_msg}",
                parent=self,
            )
            return

        # ── Two or more labels: one set per label + the multi-class set ──
        results = []
        for g in groups:
            results.append((g, self._write_training_set(
                g["features"], g["name"], g["category_id"], ctx)))
        mc = self._write_multiclass_bundle(groups, ctx, export_path)
        class_pts_path = self._export_class_points(export_path)

        print(f"[export_training_data] Export complete for {base_name}: "
              f"{len(groups)} labels")
        for g, res in results:
            print(f"    {g['name']} (category {g['category_id']}, "
                  f"{res['n_shapes']} shape(s)):")
            for key in ("image", "mask", "overlay", "geojson", "coco"):
                print(f"        {key:<8}→ {res[key]}")
        print(f"    multiclass → {mc['folder']}")
        for key in ("image", "labelmap", "overlay", "geojson", "coco", "categories"):
            print(f"        {key:<10}→ {mc[key]}")
        if class_pts_path:
            print(f"    class pts  → {class_pts_path}")

        failed = [p for _, res in results for p in res["failed"]] + mc["failed"]
        if failed:
            self._report_write_failures(failed)
            return
        lines = [f"Export complete for {base_name}: {len(groups)} labels.", ""]
        for g, res in results:
            lines.append(f"- {g['name']} (category {g['category_id']}): "
                         f"{res['n_shapes']} shape(s)")
        lines += ["",
                  f"Per-label files ⇒ {base_folder}",
                  f"Multi-class files ⇒ {mc['folder']}"]
        if class_pts_path:
            lines.append(f"Class pts ⇒ {class_pts_path}")
        messagebox.showinfo("Export Training Data", "\n".join(lines), parent=self)

    def _report_write_failures(self, failed):
        for p in failed:
            print(f"[export] NOT WRITTEN: {p}")
        shown = "\n".join(f"- {p}" for p in failed[:8])
        more = f"\n... and {len(failed) - 8} more" if len(failed) > 8 else ""
        messagebox.showwarning(
            "Export incomplete",
            f"These files could not be written:\n{shown}{more}\n\n"
            f"{self._WRITE_FAILED_HINT}",
            parent=self)

    def _resolve_export_groups(self, default_raw, default_cid, export_path):
        """
        Group self.features by label for export.

        A feature's label is its own label, or *default_raw* (the Feature
        ID) for features without one.  Labels with the same file-name form
        (slug) are one group, so two groups can never write the same files.
        Returns a list of dicts {"name", "slug", "category_id", "features",
        "positions"} in order of first appearance, or None after showing an
        error.
        """
        groups, index = [], {}
        for pos, feat in enumerate(self.features):
            ftype, pts = feat
            own = feature_label(feat)
            name = own or default_raw
            if ftype == "polygon" and len(pts) >= 2 and len(
                    {(float(p[0]), float(p[1])) for p in pts}) < 3:
                print("[export_training_data] A polygon with fewer than 3 "
                      "different vertices is exported as a line.")
                feat = Feature("polyline", pts, own)
            key = slug(name)
            if key not in index:
                index[key] = len(groups)
                groups.append({"name": name, "slug": key,
                               "features": [], "positions": []})
            groups[index[key]]["features"].append(feat)
            groups[index[key]]["positions"].append(pos)

        cat = self._get_label_catalog()
        # IDs already used by multi-label exports to this folder, so a
        # label keeps its ID across images whatever the export order
        self._merge_export_folder_categories()
        cat_path = os.path.join(self._dataset_root(export_path),
                                "training_dataset_multiclass", "categories.json")
        try:
            file_cat = LabelCatalog(self._read_export_categories(cat_path))
            file_cat_error = None
        except Exception as e:
            file_cat, file_cat_error = LabelCatalog(), e
        for g in groups:
            if default_raw and g["slug"] == slug(default_raw):
                # The Feature ID and its Category ID, exactly as before
                g["name"] = default_raw
                g["category_id"] = default_cid
            else:
                if g["name"] in cat:
                    g["name"] = cat.name_for(g["name"])
                g["category_id"] = self._category_id_for(g["name"])
        self._sync_default_into_catalog()
        if len(groups) == 1:
            g = groups[0]
            fid = file_cat.id_for(g["name"])
            owner = file_cat.owner_of(g["category_id"])
            if fid is not None and fid != g["category_id"]:
                print(f"[Labels] Note: '{g['name']}' is category {fid} in "
                      f"{cat_path}, but this export uses {g['category_id']}.")
            elif owner is not None and slug(owner) != g["slug"]:
                print(f"[Labels] Note: category {g['category_id']} is '{owner}' in "
                      f"{cat_path}, but this export uses it for '{g['name']}'.")
            return groups

        # ── Several labels: category IDs must be unique and consistent ──
        problems, by_id = [], {}
        for g in groups:
            # one spelling per label across all files of the folder
            if file_cat.id_for(g["name"]) is not None:
                g["name"] = file_cat.name_for(g["name"])
            cid = g["category_id"]
            if not g["slug"]:
                problems.append(f"'{g['name']}' has no letters or digits, so it "
                                f"cannot name files in a multi-label export.")
            if cid < 1:
                problems.append(f"'{g['name']}' has category ID {cid}; use 1 or larger.")
            if cid > 65535:
                problems.append(f"'{g['name']}' has category ID {cid}; the label "
                                f"map cannot hold IDs above 65535.")
            other = by_id.get(cid)
            if other is not None:
                problems.append(f"'{other['name']}' and '{g['name']}' both have "
                                f"category ID {cid}.")
            else:
                by_id[cid] = g
        if file_cat_error is not None:
            problems.append(f"{cat_path} could not be read "
                            f"({type(file_cat_error).__name__}: {file_cat_error}).")
        for g in groups:
            fid = file_cat.id_for(g["name"])
            if fid is not None and fid != g["category_id"]:
                problems.append(f"'{g['name']}' is category {fid} in earlier "
                                f"exports to this folder, but {g['category_id']} now.")
            owner = file_cat.owner_of(g["category_id"])
            if owner is not None and slug(owner) != g["slug"]:
                problems.append(f"Category {g['category_id']} is '{owner}' in "
                                f"earlier exports to this folder, but "
                                f"'{g['name']}' now.")
        if problems:
            messagebox.showerror(
                "Export Training Data",
                "Labels and category IDs do not match, so nothing was "
                "exported:\n\n" + "\n".join(f"- {p}" for p in problems) +
                "\n\nChange the Category ID in the shape editor (or in the "
                "configuration window for the Feature ID) and export again.\n\n"
                f"Earlier multi-label exports to this folder are listed in:\n{cat_path}",
                parent=self)
            return None
        return groups

    def _write_training_set(self, features, feature_name_raw, category_id, ctx):
        """
        Write one label's training set (image copy, binary mask, overlay,
        GeoJSON, COCO JSON) named <image>_<label>.  This is the export of
        earlier releases, unchanged except that overlay colours keep full
        opacity on RGBA images and files that could not be written are
        reported instead of passing silently.
        """
        folders = ctx["folders"]
        images_folder   = folders["images"]
        masks_folder    = folders["masks"]
        overlays_folder = folders["overlays"]
        geojson_folder  = folders["geojson"]
        coco_folder     = folders["coco"]
        image_id = ctx["image_id"]
        stem, src_ext = ctx["stem"], ctx["src_ext"]
        width, height = ctx["width"], ctx["height"]
        transform, crs = ctx["transform"], ctx["crs"]
        thickness = ctx["thickness"]
        failed = []

        feature_name = self._sanitize_feature_id(feature_name_raw)
        suffix = f"_{feature_name}" if feature_name else ""
        out_stem = f"{stem}{suffix}"                            # e.g. name_shoreline

        # 4) Copy original image but with feature suffix in filename
        image_copy_name = f"{out_stem}{src_ext}"                # name_shoreline.tif
        image_copy_path = os.path.join(images_folder, image_copy_name)
        try:
            shutil.copy2(self.image_path, image_copy_path)
        except Exception as e:
            # Still write the annotations, but report the missing copy
            failed.append(f"{image_copy_path} ({type(e).__name__}: {e})")

        # 6) Build mask, overlay, COCO annotations, and GeoJSON shapes
        mask = np.zeros((height, width), dtype=np.uint8)
        overlay = self.full_image.copy()
        # Confirmed BGR colours: polygon → dark orange #cc5500, polyline → teal #0088aa
        poly_bgr = self._overlay_colour((0, 85, 204), overlay)
        line_bgr = self._overlay_colour((170, 136, 0), overlay)

        coco_annotations = []
        annotation_id = 1
        shape_list = []

        for feature_type, pts in features:
            if len(pts) < 2:
                continue

            pts_np = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            is_poly = (feature_type == "polygon")
            if is_poly:
                closed = pts[:] + ([pts[0]] if pts[0] != pts[-1] else [])
                cv2.fillPoly(mask, [np.array(closed, np.int32).reshape((-1, 1, 2))], 255)
                cv2.polylines(overlay, [pts_np], True, poly_bgr, 2)
            else:
                cv2.polylines(mask, [pts_np], False, 255, thickness)
                cv2.polylines(overlay, [pts_np], False, line_bgr, thickness)

            seg_pts = pts[:] + ([pts[0]] if is_poly else [])
            seg = [coord for p in seg_pts for coord in p]
            xs, ys = zip(*seg_pts)
            bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
            area = bbox[2] * bbox[3]
            coco_annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": [seg],
                "area": area,
                "bbox": bbox,
                "iscrowd": 0
            })
            annotation_id += 1

            if transform is not None and crs is not None:
                world_pts = [(transform * (x, y))[0:2] for x, y in pts]
            else:
                world_pts = pts
            shape_list.append(Polygon(world_pts) if is_poly else LineString(world_pts))

        # 7) Save mask & overlay with feature suffix
        mask_path    = os.path.join(masks_folder,    f"{out_stem}_mask.png")
        overlay_path = os.path.join(overlays_folder, f"{out_stem}_overlay.png")
        if not imwrite_safe(mask_path, mask):
            failed.append(mask_path)
        if not imwrite_safe(overlay_path, overlay):
            failed.append(overlay_path)

        # 8) Write GeoJSON (always) with feature suffix
        geojson_path = os.path.join(geojson_folder, f"{out_stem}.geojson")
        gdf = gpd.GeoDataFrame(geometry=shape_list, crs=crs)
        gdf.to_file(geojson_path, driver="GeoJSON")

        # 9) Write COCO JSON; reference the **renamed** image file
        coco_dict = {
            "images": [{
                "id": image_id,
                "file_name": image_copy_name,  # name_shoreline.tif
                "width": width,
                "height": height
            }],
            "annotations": coco_annotations,
            "categories": [{
                "id": category_id,
                "name": feature_name_raw  # keep the original label text here
            }]
        }
        coco_path = os.path.join(coco_folder, f"{out_stem}.json")
        with open(coco_path, "w") as f:
            json.dump(coco_dict, f, indent=2)

        return {"image": image_copy_path, "mask": mask_path,
                "overlay": overlay_path, "geojson": geojson_path,
                "coco": coco_path, "failed": failed,
                "n_shapes": annotation_id - 1}

    def _write_multiclass_bundle(self, groups, ctx, export_path):
        """
        Combined set for an image with two or more labels, in its own
        folder next to "training dataset" inside Dataset/:

            Dataset/training_dataset_multiclass/
                images/<image><ext>           the source image (copied once)
                masks/<image>_labelmap.png    pixel value = category ID,
                                              0 = background
                overlays/<image>_overlay.png  all shapes with their labels
                geojson/<image>.geojson       one feature per shape with
                                              feature_id, category_id,
                                              feature_type, image_id,
                                              annotation_id
                coco/<image>_coco.json        every shape with its own
                                              category_id; categories =
                                              every label exported here
                categories.json               label <-> category ID for
                                              the whole folder, and per
                                              image which labels were used

        It has its own folder so the per-label folders, which Create DEM
        and Wave Run-up read, never hold a file that mixes labels.

        Label map: uint8, or uint16 when a category ID exceeds 255.  All
        polygons are filled first, then all polylines are drawn on top at
        the Line Thickness; within each pass, ascending category ID, so a
        higher ID wins where shapes overlap.
        """
        mc = os.path.join(self._dataset_root(export_path), "training_dataset_multiclass")
        sub = {k: os.path.join(mc, k)
               for k in ("images", "masks", "overlays", "geojson", "coco")}
        for d in sub.values():
            os.makedirs(d, exist_ok=True)
        stem, src_ext = ctx["stem"], ctx["src_ext"]
        width, height = ctx["width"], ctx["height"]
        transform, crs = ctx["transform"], ctx["crs"]
        thickness, image_id = ctx["thickness"], ctx["image_id"]
        failed = []

        # Shapes in confirmed order, each with its label group
        items = sorted(((pos, f, g) for g in groups
                        for pos, f in zip(g["positions"], g["features"])),
                       key=lambda t: t[0])
        items = [(f, g) for _, f, g in items if len(f[1]) >= 2]

        # Source image, copied once
        img_name = f"{stem}{src_ext}"
        img_path = os.path.join(sub["images"], img_name)
        try:
            shutil.copy2(self.image_path, img_path)
        except Exception as e:
            failed.append(f"{img_path} ({type(e).__name__}: {e})")

        # Label map
        max_id = max(g["category_id"] for g in groups)
        dtype = np.uint16 if max_id > 255 else np.uint8
        if dtype is np.uint16:
            print(f"[export_training_data] Category ID {max_id} > 255: "
                  f"the label map is a 16-bit PNG.")
        labelmap = np.zeros((height, width), dtype=dtype)
        by_id = sorted(groups, key=lambda g: g["category_id"])
        for kind in ("polygon", "polyline"):
            for g in by_id:
                for f, fg in items:
                    if fg is not g or f[0] != kind:
                        continue
                    arr = np.array(f[1], dtype=np.int32).reshape((-1, 1, 2))
                    if kind == "polygon":
                        cv2.fillPoly(labelmap, [arr], int(g["category_id"]))
                    else:
                        cv2.polylines(labelmap, [arr], False,
                                      int(g["category_id"]), thickness)
        labelmap_path = os.path.join(sub["masks"], f"{stem}_labelmap.png")
        if not imwrite_safe(labelmap_path, labelmap):
            failed.append(labelmap_path)

        # Overlay with label text
        overlay = self.full_image.copy()
        poly_bgr = self._overlay_colour((0, 85, 204), overlay)
        line_bgr = self._overlay_colour((170, 136, 0), overlay)
        black = self._overlay_colour((0, 0, 0), overlay)
        for f, g in items:
            arr = np.array(f[1], dtype=np.int32).reshape((-1, 1, 2))
            if f[0] == "polygon":
                cv2.polylines(overlay, [arr], True, poly_bgr, 2)
            else:
                cv2.polylines(overlay, [arr], False, line_bgr, thickness)
        fs = max(0.6, width / 1600.0)
        tt = max(1, int(round(fs * 2)))
        for f, g in items:
            x0, y0 = int(f[1][0][0]), int(f[1][0][1])
            org = (x0 + 6, max(int(24 * fs), y0 - 6))
            colour = poly_bgr if f[0] == "polygon" else line_bgr
            cv2.putText(overlay, g["name"], org, cv2.FONT_HERSHEY_SIMPLEX,
                        fs, black, tt + 2, cv2.LINE_AA)
            cv2.putText(overlay, g["name"], org, cv2.FONT_HERSHEY_SIMPLEX,
                        fs, colour, tt, cv2.LINE_AA)
        overlay_path = os.path.join(sub["overlays"], f"{stem}_overlay.png")
        if not imwrite_safe(overlay_path, overlay):
            failed.append(overlay_path)

        # GeoJSON (one feature per shape, with attributes) and COCO
        props = {"feature_id": [], "category_id": [], "feature_type": [],
                 "image_id": [], "annotation_id": []}
        shapes, annotations = [], []
        for ann_id, (f, g) in enumerate(items, start=1):
            ftype, pts = f
            pts = [list(p) for p in pts]
            is_poly = (ftype == "polygon")
            if transform is not None and crs is not None:
                world_pts = [(transform * (x, y))[0:2] for x, y in pts]
            else:
                world_pts = pts
            shapes.append(Polygon(world_pts) if is_poly else LineString(world_pts))
            props["feature_id"].append(g["name"])
            props["category_id"].append(int(g["category_id"]))
            props["feature_type"].append(ftype)
            props["image_id"].append(int(image_id))
            props["annotation_id"].append(ann_id)

            seg_pts = pts + ([pts[0]] if is_poly else [])
            seg = [coord for p in seg_pts for coord in p]
            xs, ys = zip(*seg_pts)
            bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": int(g["category_id"]),
                "segmentation": [seg],
                "area": bbox[2] * bbox[3],
                "bbox": bbox,
                "iscrowd": 0,
                "geometry_type": ftype,
            })
        geojson_path = os.path.join(sub["geojson"], f"{stem}.geojson")
        gdf = gpd.GeoDataFrame(props, geometry=shapes, crs=crs)
        gdf.to_file(geojson_path, driver="GeoJSON")

        # Folder-wide label list (merged with earlier exports)
        cat_path = os.path.join(mc, "categories.json")
        try:
            file_cat = LabelCatalog(self._read_export_categories(cat_path))
        except Exception:
            file_cat = LabelCatalog()
        for g in groups:
            try:
                file_cat.set_id(g["name"], g["category_id"])
            except ValueError as e:
                print(f"[export_training_data] categories.json: {e}")
        cat_rows = file_cat.to_list()

        coco_dict = {
            "images": [{"id": image_id, "file_name": img_name,
                        "width": width, "height": height}],
            "annotations": annotations,
            "categories": [{"id": r["category_id"], "name": r["feature_id"],
                            "supercategory": "feature"} for r in cat_rows],
        }
        coco_path = os.path.join(sub["coco"], f"{stem}_coco.json")
        with open(coco_path, "w", encoding="utf-8") as f:
            json.dump(coco_dict, f, indent=2, ensure_ascii=False)

        # Per-image record: which labels this image used (replaced when the
        # same image is exported again)
        try:
            prev_doc = self._read_export_doc(cat_path)
        except Exception:
            prev_doc = {}
        images = [e for e in prev_doc.get("images", [])
                  if isinstance(e, dict) and e.get("stem") != stem]
        images.append({
            "stem": stem,
            "image": img_name,
            "image_id": image_id,
            "labels": {g["name"]: int(g["category_id"]) for g in groups},
            "shapes": len(items),
            "exported": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        images.sort(key=lambda e: str(e.get("stem", "")))

        cat_doc = {
            "geocampal_version": GEOCAMPAL_VERSION,
            "description": ("Labels and category IDs of the multi-label exports "
                            "in this folder, and per image which labels it "
                            "used. GeoCamPal keeps it up to date and refuses an "
                            "export whose IDs disagree with it."),
            "categories": [{"id": r["category_id"], "name": r["feature_id"],
                            "slug": slug(r["feature_id"])} for r in cat_rows],
            "images": images,
            "labelmap": {
                "files": "masks/<image>_labelmap.png",
                "background": 0,
                "pixel_value": "category id",
                "dtype": "uint8, or uint16 when a category id exceeds 255",
                "draw_order": ("polygons filled first, then polylines on top; "
                               "within each, ascending category id"),
            },
        }
        with open(cat_path, "w", encoding="utf-8") as f:
            json.dump(cat_doc, f, indent=2, ensure_ascii=False)

        return {"folder": mc, "image": img_path, "labelmap": labelmap_path,
                "overlay": overlay_path, "geojson": geojson_path,
                "coco": coco_path, "categories": cat_path, "failed": failed}


    def export_mask_as_training_data(self):
        if self.current_mask is None:
            messagebox.showerror("Error", "No mask available for exporting.", parent=self)
            return
        if not isinstance(self.full_image, np.ndarray):
            messagebox.showerror("Error", "No full image available.", parent=self)
            return
        export_path = self.export_path_entry.get().strip()
        if not export_path:
            messagebox.showerror("Error", "Please select an export folder.", parent=self)
            return

        base_folder = os.path.join(self._dataset_root(export_path), "training dataset")
        masks_folder = os.path.join(base_folder, "masks")
        overlays_folder = os.path.join(base_folder, "overlays")
        os.makedirs(masks_folder, exist_ok=True)
        os.makedirs(overlays_folder, exist_ok=True)

        # The mask is computed on the <= 600 px working copy; a training
        # mask must match the image it labels, so scale it to full size.
        h_full, w_full = self.full_image.shape[:2]
        mask = self.current_mask
        if mask.shape[:2] != (h_full, w_full):
            mask = cv2.resize(mask, (w_full, h_full),
                              interpolation=cv2.INTER_NEAREST)

        base_name = os.path.basename(self.image_path)
        mask_filename = os.path.splitext(base_name)[0] + "_mask.png"
        mask_path = os.path.join(masks_folder, mask_filename)
        failed = []
        if not imwrite_safe(mask_path, mask):
            failed.append(mask_path)

        # Create overlay
        if self.full_image.ndim == 2:
            overlay = cv2.cvtColor(self.full_image, cv2.COLOR_GRAY2BGR)
        elif self.full_image.shape[2] == 4:
            overlay = cv2.cvtColor(self.full_image, cv2.COLOR_BGRA2BGR)
        else:
            overlay = self.full_image.copy()

        mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(overlay, 0.7, mask_color, 0.3, 0)

        overlay_filename = os.path.splitext(base_name)[0] + "_overlay.png"
        overlay_path = os.path.join(overlays_folder, overlay_filename)
        if not imwrite_safe(overlay_path, overlay):
            failed.append(overlay_path)

        self._export_class_points(export_path)

        if failed:
            self._report_write_failures(failed)
            return
        messagebox.showinfo("Export Mask Training Data",
                            f"Mask training data exported.\n{base_folder}", parent=self)

    def export_as_test_data(self):
        if not isinstance(self.full_image, np.ndarray):
            messagebox.showerror("Error", "No full image available.", parent=self)
            return
        export_path = self.export_path_entry.get().strip()
        if not export_path:
            messagebox.showerror(
                "Error", "Please specify a path to save files.", parent=self)
            return
        test_folder = os.path.join(self._dataset_root(export_path), "test dataset")
        os.makedirs(test_folder, exist_ok=True)
        base_name = os.path.basename(self.image_path)
        dest_path = os.path.join(test_folder, base_name)
        shutil.copy2(self.image_path, dest_path)
        messagebox.showinfo("Export Test Data",
                            f"Test image exported to:\n{dest_path}", parent=self)

    def export_as_overlay(self):
        if getattr(self, "_edit_session_open", False):
            messagebox.showwarning(
                "Finish editing first",
                "The shape editor is still open.\n\n"
                "Press Confirm Feature (or Cancel) in the image window, then "
                "export again.",
                parent=self)
            return
        if not isinstance(self.full_image, np.ndarray):
            messagebox.showerror("Error", "No full image available.", parent=self)
            return
        features = [f for f in (self.features or []) if len(f[1]) >= 2]
        if not features and self.edge_points and len(self.edge_points) >= 2:
            features = [("polyline", self.edge_points)]
        if not features:
            messagebox.showerror("Error", "No valid edge to export.", parent=self)
            return
        export_path = self.export_path_entry.get().strip()
        if not export_path:
            messagebox.showerror(
                "Error", "Please specify a path to save files.", parent=self)
            return
        base_name = os.path.basename(self.image_path)
        overlay = self.full_image.copy()
        thickness = int(self.edge_thickness_slider.get())
        # Confirmed colours: polygon dark orange #cc5500, polyline teal #0088aa
        poly_bgr = self._overlay_colour((0, 85, 204), overlay)
        line_bgr = self._overlay_colour((170, 136, 0), overlay)
        for ftype, fpts in features:
            pts = np.array(fpts, dtype=np.int32).reshape((-1, 1, 2))
            if ftype == "polygon":
                cv2.polylines(overlay, [pts], isClosed=True,
                              color=poly_bgr, thickness=2)
            else:
                cv2.polylines(overlay, [pts], isClosed=False,
                              color=line_bgr, thickness=thickness)
        out_overlay = os.path.join(
            export_path, os.path.splitext(base_name)[0] + "_overlay.png")
        if not imwrite_safe(out_overlay, overlay):
            self._report_write_failures([out_overlay])
            return
        messagebox.showinfo(
            "Export Overlay", f"Overlay saved to:\n{out_overlay}", parent=self)

