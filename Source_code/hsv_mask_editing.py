"""
hsv_mask_editing.py
───────────────────
Feature editing (cut/create/confirm), canvas interaction (drag, click,
double-click, freehand), zoom & pan, non-destructive preview adjustments,
and all export methods.

This module is a *mixin* — it is meant to be inherited by HSVMaskTool
together with the UI and processing mixins.
"""

import os
import json
import shutil
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

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


class HSVMaskEditingMixin:
    """Mixin that supplies feature editing, canvas interaction,
    zoom/pan, preview adjust, and export methods."""

    # -------------- FEATURE EDITING --------------

    def start_freehand(self):
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
        # unbind only the free-hand events
        self.edit_canvas.unbind("<B1-Motion>")
        self.edit_canvas.unbind("<ButtonRelease-1>")

        # record the end of this stroke for undo/redo
        self._record_history()

        # re-bind the normal vertex-edit handlers
        self.edit_canvas.bind("<ButtonPress-1>",    self.on_canvas_press)
        self.edit_canvas.bind("<B1-Motion>",        self.on_canvas_drag)
        self.edit_canvas.bind("<ButtonRelease-1>",  self.on_canvas_release)
        self.edit_canvas.bind("<Double-Button-1>",  self.on_canvas_double_click)

        # and redraw to show the final segment
        self.redraw_canvas()


    def _bind_edit_shortcuts(self):
        if not hasattr(self, "edit_canvas"):
            return

        # modes/tools
        self.edit_canvas.bind("d", self._btn_mode_delete)
        self.edit_canvas.bind("m", self._btn_mode_add)
        self.edit_canvas.bind("f", self._btn_freehand)
        self.edit_canvas.bind("e", self._btn_create_edge)
        self.edit_canvas.bind("p", self._btn_create_polygon)
        self.edit_canvas.bind("c", self._btn_continue_line)

        # make sure the canvas keeps focus so single-letter keys work
        self.edit_canvas.bind("<Button-1>", lambda e: self.edit_canvas.focus_set(), add="+")
        self.edit_canvas.bind("<Motion>",   lambda e: self.edit_canvas.focus_set(), add="+")
        self.edit_canvas.focus_set()

        # Undo / Redo on U / R (both cases)
        for seq in ("u", "U"):
            self.edit_canvas.bind(seq, self._btn_undo)
            self.bind(seq, self._btn_undo)   # backup on toplevel
        for seq in ("r", "R"):
            self.edit_canvas.bind(seq, self._btn_redo)
            self.bind(seq, self._btn_redo)



    def _unbind_edit_shortcuts(self):
        if hasattr(self, "edit_canvas"):
            for seq in ("d", "m", "f", "e", "p", "c",
                        "<Double-Button-1>", "<Double-1>",
                        "u", "U", "r", "R"):
                try:
                    self.edit_canvas.unbind(seq)
                except Exception:
                    pass
        # also remove toplevel backups
        for seq in ("u", "U", "r", "R"):
            try:
                self.unbind(seq)
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
        Opens the edit window with ALL detected features visible.

        The first (largest) feature is loaded into edited_edge_points for
        vertex editing.  All other features are stored in
        _edit_session_features and rendered as green reference lines on
        the canvas.  The user can create additional polygons — those also
        go into _edit_session_features.  Nothing is committed to
        self.features or the right panel until Confirm Feature is pressed.
        """
        self.vertex_mode = "delete"
        self.freehand_mode = False

        # ── Move ALL features into the edit session ──
        self._edit_session_features = []
        points_to_edit = []
        self._editing_feature_type = "polyline"

        for ftype, pts in self.features:
            if len(pts) < 2:
                continue
            if not points_to_edit:
                # First valid feature → active edit target
                self._editing_feature_type = ftype
                points_to_edit = pts.copy()
            else:
                # Remaining features → session references (visible on canvas)
                self._edit_session_features.append((ftype, pts))

        # Fall back to edge_points if no features exist
        if not points_to_edit and self.edge_points and len(self.edge_points) >= 2:
            points_to_edit = self.edge_points.copy()

        # Clear main features list — everything lives in the edit session now
        self.features = []

        if not points_to_edit:
            print("No valid detected feature to edit. Starting empty.")
            self.edge_points = []

        self.initial_edge_points = points_to_edit.copy()
        self.edited_edge_points  = points_to_edit.copy()
        self.edit_history = [self.edited_edge_points.copy()]
        self.selected_vertex = None
        self.is_polygon_mode = (self._editing_feature_type == 'polygon')
        self.freehand_mode = False

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

        # ── Control buttons – use scrollable frame for small-screen safety ──
        self.control_frame = ctk.CTkScrollableFrame(
            self.top_center_frame, height=180, orientation="vertical"
        )
        self.control_frame.pack(side="bottom", fill="x", pady=2)

        # Adaptive button width helper
        BTN_W = 105     # compact width so all buttons fit on 1200-px screens
        BTN_H = 28
        BTN_FONT = ("Arial", 11)

        # ── ROW 1: Create Line, Create Polygon, Add/Move Vertex, Continue Line, Freehand ──
        row1 = ctk.CTkFrame(self.control_frame)
        row1.pack(side="top", fill="x", pady=2)

        ctk.CTkButton(row1, text="Create Line", width=BTN_W, height=BTN_H,
                       font=BTN_FONT, command=self._btn_create_edge
                       ).pack(side="left", padx=2, pady=1)
        ctk.CTkButton(row1, text="Create Polygon", width=BTN_W, height=BTN_H,
                       font=BTN_FONT, command=self._btn_create_polygon
                       ).pack(side="left", padx=2, pady=1)
        self.btn_add_mode = ctk.CTkButton(
            row1, text="Add/Move Vertex", width=BTN_W, height=BTN_H,
            font=BTN_FONT, command=self._btn_mode_add)
        self.btn_add_mode.pack(side="left", padx=2, pady=1)
        ctk.CTkButton(row1, text="Continue Line", width=BTN_W, height=BTN_H,
                       font=BTN_FONT, command=self._btn_continue_line
                       ).pack(side="left", padx=2, pady=1)
        ctk.CTkButton(row1, text="Freehand", width=BTN_W, height=BTN_H,
                       font=BTN_FONT, command=self._btn_freehand
                       ).pack(side="left", padx=2, pady=1)

        # ── ROW 2: Zoom In, Zoom Out, Undo, Redo, Delete Vertex, Reset, Delete All ──
        row2 = ctk.CTkFrame(self.control_frame)
        row2.pack(side="top", fill="x", pady=2)

        ctk.CTkButton(row2, text="Zoom In", width=BTN_W, height=BTN_H,
                       font=BTN_FONT, command=lambda: self.adjust_zoom(1.2)
                       ).pack(side="left", padx=2, pady=1)
        ctk.CTkButton(row2, text="Zoom Out", width=BTN_W, height=BTN_H,
                       font=BTN_FONT, command=lambda: self.adjust_zoom(0.8)
                       ).pack(side="left", padx=2, pady=1)
        ctk.CTkButton(row2, text="Undo", width=BTN_W, height=BTN_H,
                       font=BTN_FONT, command=self._btn_undo
                       ).pack(side="left", padx=2, pady=1)
        ctk.CTkButton(row2, text="Redo", width=BTN_W, height=BTN_H,
                       font=BTN_FONT, command=self._btn_redo
                       ).pack(side="left", padx=2, pady=1)
        self.btn_delete_mode = ctk.CTkButton(
            row2, text="Delete Vertex", width=BTN_W, height=BTN_H,
            font=BTN_FONT, command=self._btn_mode_delete)
        self.btn_delete_mode.pack(side="left", padx=2, pady=1)
        ctk.CTkButton(row2, text="Reset", width=BTN_W, height=BTN_H,
                       font=BTN_FONT, command=self.reset_to_initial,
                       fg_color="#8B0000", hover_color="#A52A2A", text_color="white"
                       ).pack(side="left", padx=2, pady=1)
        ctk.CTkButton(row2, text="Delete All", width=BTN_W, height=BTN_H,
                       font=BTN_FONT, command=self.delete_all_vertices,
                       fg_color="#8B0000", hover_color="#A52A2A", text_color="white"
                       ).pack(side="left", padx=2, pady=1)

        # ── ROW 3: Confirm Feature, ◀ Prev Polygon, feature list label, Next Polygon ▶ ──
        row3 = ctk.CTkFrame(self.control_frame)
        row3.pack(side="top", fill="x", pady=2)

        self.confirm_button = ctk.CTkButton(
            row3, text="Confirm Feature", width=BTN_W + 10, height=BTN_H,
            font=BTN_FONT, command=self.confirm_feature_cuts,
            fg_color="#0F52BA",hover_color="#2A6BD1", text_color="white"
        )
        self.confirm_button.pack(side="left", padx=4, pady=1)

        ctk.CTkButton(row3, text="◀ Prev", width=80, height=BTN_H,
                       font=BTN_FONT, command=self._edit_prev_polygon,
                       fg_color="white", text_color='black'
                       ).pack(side="left", padx=2, pady=1)
        self._poly_nav_label = ctk.CTkLabel(
            row3, text="", font=("Arial", 10), width=220)
        self._poly_nav_label.pack(side="left", padx=4)
        ctk.CTkButton(row3, text="Next ▶", width=80, height=BTN_H,
                       font=BTN_FONT, command=self._edit_next_polygon,
                       fg_color="white", text_color='black'
                       ).pack(side="left", padx=2, pady=1)
        self._update_poly_nav_label()

        # ── ROW 4: Saturation / Exposure / Highlights sliders ──
        self._build_adjust_row(self.control_frame)
        self._install_slider_handlers()

        # ── Info label ──
        info_label = ctk.CTkLabel(
            self.control_frame,
            text="Scroll=zoom | Dbl-click=delete/add | "
                 "Keys: d=del, m=move, f=freehand, e=line, p=polygon, c=continue, U=undo, R=redo, Enter=confirm",
            font=("Arial", 9)
        )
        info_label.pack(side="top", pady=1)

        # enable edit-mode shortcuts (binds d/m/f/e/p/c + U/R and keeps canvas focused)
        self._bind_edit_shortcuts()

        # Enter key → confirm feature (works on canvas and parent window)
        self.edit_canvas.bind("<Return>", lambda e: self.confirm_feature_cuts())
        try:
            self.image_display_window.bind("<Return>", lambda e: self.confirm_feature_cuts())
        except Exception:
            pass

        # Initial draw
        base = self._ensure_scaled_base_for_zoom(high_quality=False)
        if base is not None:
            self._set_edit_preview(base)   # show something right away
        self.redraw_canvas()


    def create_new_edge(self):
        """
        Start creating a new polyline.  Non-destructive: if there are
        existing edited points they are saved to the session features
        first, so only Reset / Delete All are destructive.
        """
        # Save current points to session if they form a valid feature
        if len(self.edited_edge_points) >= 2:
            ftype = "polygon" if self.is_polygon_mode else "polyline"
            proxy_s = getattr(self, '_edit_proxy_scale', 1.0)
            if proxy_s != 1.0 and proxy_s > 0:
                inv = 1.0 / proxy_s
                saved_pts = [[x * inv, y * inv] for x, y in self.edited_edge_points]
            else:
                saved_pts = [list(pt) for pt in self.edited_edge_points]
            self._edit_session_features.append((ftype, saved_pts))
            self._update_poly_nav_label()

        self.edited_edge_points = []
        self.edit_history = []
        self._hide_vertices_for_current_shape = False
        self._record_history()
        self.redraw_canvas()
        self.creation_mode = True
        self.is_polygon_mode = False
        self.edit_canvas.unbind("<Double-Button-1>")
        self.edit_canvas.bind("<Button-1>", self.on_canvas_single_click)

    def create_new_polygon(self):
        """
        Start creating a new polygon.  Non-destructive: if there are
        existing edited points they are saved to the session features first.
        Double-click to close the polygon (in addition to clicking near
        the first vertex).
        """
        # Save current points to session if they form a valid feature
        if len(self.edited_edge_points) >= 2:
            ftype = "polygon" if self.is_polygon_mode else "polyline"
            proxy_s = getattr(self, '_edit_proxy_scale', 1.0)
            if proxy_s != 1.0 and proxy_s > 0:
                inv = 1.0 / proxy_s
                saved_pts = [[x * inv, y * inv] for x, y in self.edited_edge_points]
            else:
                saved_pts = [list(pt) for pt in self.edited_edge_points]
            self._edit_session_features.append((ftype, saved_pts))
            self._update_poly_nav_label()

        self.edited_edge_points = []
        self.edit_history = []
        self._hide_vertices_for_current_shape = False
        self._record_history()
        self.redraw_canvas()
        self.creation_mode = True
        self.is_polygon_mode = True
        self.edit_canvas.bind("<Button-1>", self.on_canvas_single_click)
        self.edit_canvas.bind("<Double-Button-1>", self._close_polygon_double_click)

    def _close_polygon_double_click(self, event):
        """
        Close the polygon being created on double-click.

        Tk fires <Button-1> before <Double-Button-1>, so the single-click
        handler has already appended a point.  We remove that extra point
        before closing, so the polygon ends at the last intentional vertex.
        """
        if not self.is_polygon_mode or not getattr(self, 'creation_mode', False):
            return

        # Remove the spurious point added by the preceding <Button-1>
        # if len(self.edited_edge_points) > 3:
        #     self.edited_edge_points.pop()

        if len(self.edited_edge_points) < 3:
            print("[Edit] Need at least 3 vertices to close a polygon.")
            return

        # Build closed polygon
        closed_points = self.edited_edge_points[:]
        if closed_points[-1] != closed_points[0]:
            closed_points.append(closed_points[0])

        # Scale from proxy coords back to full-image coords
        proxy_s = getattr(self, '_edit_proxy_scale', 1.0)
        if proxy_s != 1.0 and proxy_s > 0:
            inv = 1.0 / proxy_s
            closed_points = [[x * inv, y * inv] for x, y in closed_points]

        self._edit_session_features.append(("polygon", closed_points))
        print("Polygon closed (double-click) and added to session. "
              "You can create more or press Confirm.")

        self.edited_edge_points = []
        self.edit_history = []
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
        # If we were in add/move/delete mode, switch to creation mode
        # but keep existing points.
        if getattr(self, "creation_mode", False):
            # Already in creation mode — do nothing special
            print("[Edit] Already in creation/continue mode.")
            return

        self._record_history()
        self._hide_vertices_for_current_shape = False
        self.creation_mode = True
        # Keep is_polygon_mode as-is (if user was editing a polygon, continue as polygon)

        self.edit_canvas.unbind("<Double-Button-1>")
        self.edit_canvas.bind("<Button-1>", self.on_canvas_single_click)

        n = len(self.edited_edge_points)
        if n > 0:
            print(f"[Edit] Continue line from point {n} (last vertex).")
        else:
            print("[Edit] Continue line — starting fresh (no existing points).")


    def confirm_feature_cuts(self):
        """
        Called when the user clicks the "Confirm Feature" button.
        Merges all edit session features (including the active edit target
        and any polygons created during the session) back into
        self.features, then restores the normal view.
        """
        session = getattr(self, '_edit_session_features', [])

        # Add the active edited points (if any) as a feature
        if len(self.edited_edge_points) >= 2:
            if hasattr(self, 'creation_mode') and self.creation_mode:
                self.edit_canvas.unbind("<Button-1>")
                self.creation_mode = False

            feature_type = "polygon" if self.is_polygon_mode else "polyline"
            new_points = self.edited_edge_points.copy()

            # Scale coordinates back from proxy space to full-image space
            proxy_s = getattr(self, '_edit_proxy_scale', 1.0)
            if proxy_s != 1.0 and proxy_s > 0:
                inv = 1.0 / proxy_s
                new_points = [[x * inv, y * inv] for x, y in new_points]

            session.insert(0, (feature_type, new_points))
            self.edge_points = new_points

        if not session:
            messagebox.showwarning("Warning", "No features to confirm.")
            return

        # Commit all session features into the main features list
        self.features = list(session)
        self._edit_session_features = []

        if not self.edge_points and self.features:
            self.edge_points = self.features[0][1]

        # put original back & remove the adjust row (non-destructive preview reset)
        try:
            if getattr(self, "edit_original_pil", None) is not None:
                self._set_edit_preview(self.edit_original_pil)
        except Exception:
            pass

        if hasattr(self, "_adjust_row") and self._adjust_row is not None:
            try:
                self._adjust_row.destroy()
            except Exception:
                pass
            self._adjust_row = None

        self.edit_original_pil = None

        # disable edit-mode shortcuts
        self._unbind_edit_shortcuts()

        # Remove edit-mode Enter binding from the display window
        try:
            self.image_display_window.unbind("<Return>")
        except Exception:
            pass

        # --- Reset editor canvas state so re-editing works ---
        self._poly_id = None
        self._vertex_ids = []
        self.bg_image_id = None
        self._bg_cache.clear()
        self._bg_current_zoom = None
        self._zoom_cache = {"zoom": None, "img": None}
        self._redraw_job = None
        self._preview_after_id = None
        self._pending_preview = False

        # Clear viewport rendering state
        self._adjusted_proxy = None
        self._vp_render_bounds = None
        self._vp_render_zoom = None
        self._scroll_job = None
        self._bg_photo_ref = None

        # Destroy the editing UI elements
        if hasattr(self, 'edit_canvas'):
            self.edit_canvas.destroy()
        if hasattr(self, 'control_frame'):
            self.control_frame.destroy()
        if hasattr(self, 'edit_canvas_container'):
            self.edit_canvas_container.destroy()

        # Restore the normal view in the center (mask display).
        self._restore_center_mask_panel()
        self.top_center_frame.bind("<Configure>", self.update_mask_display)

        # Actively show the mask and overlay — don't wait for a resize event
        if self.current_mask is not None:
            self.display_mask()
        self.update_edge_display()

        n = len(self.features)
        print(f"[Confirm] Committed {n} feature(s) to the main view.")


    # ---------- POLYGON NAVIGATION (within edit session) ----------

    def _update_poly_nav_label(self):
        """Update the polygon counter label in the editor toolbar."""
        if not hasattr(self, '_poly_nav_label'):
            return
        session = getattr(self, '_edit_session_features', [])
        n_active_pts = len(self.edited_edge_points) if self.edited_edge_points else 0
        n_others = len(session)
        total = n_others + (1 if n_active_pts >= 2 else 0)
        ftype = "polygon" if self.is_polygon_mode else "polyline"
        if total <= 0:
            self._poly_nav_label.configure(text="No features")
        elif n_others == 0:
            self._poly_nav_label.configure(
                text=f"Active: {ftype} ({n_active_pts} pts) — no others")
        else:
            self._poly_nav_label.configure(
                text=f"Active: {ftype} ({n_active_pts} pts) | {n_others} other(s)")

    def _swap_active_polygon(self, direction):
        """
        Swap the currently active polygon (edited_edge_points) with
        another polygon in _edit_session_features.

        direction: +1 for next, -1 for previous.
        """
        session = getattr(self, '_edit_session_features', [])
        if not session:
            print("[Edit] No other polygons to switch to.")
            return

        proxy_s = getattr(self, '_edit_proxy_scale', 1.0)

        # 1) Save current active polygon back into session (in full-image coords)
        if len(self.edited_edge_points) >= 2:
            ftype = "polygon" if self.is_polygon_mode else "polyline"
            if proxy_s != 1.0 and proxy_s > 0:
                inv = 1.0 / proxy_s
                saved_pts = [[x * inv, y * inv] for x, y in self.edited_edge_points]
            else:
                saved_pts = [list(pt) for pt in self.edited_edge_points]
            session.append((ftype, saved_pts))

        # 2) Pop the next/prev polygon from session
        idx = 0 if direction >= 0 else -1
        new_ftype, new_pts = session.pop(idx)

        # 3) Load it as the active polygon (convert to proxy coords)
        self.is_polygon_mode = (new_ftype == "polygon")
        self._editing_feature_type = new_ftype
        if proxy_s != 1.0 and proxy_s > 0:
            self.edited_edge_points = [[x * proxy_s, y * proxy_s] for x, y in new_pts]
        else:
            self.edited_edge_points = [list(pt) for pt in new_pts]

        self.initial_edge_points = self.edited_edge_points.copy()
        self.edit_history = [self.edited_edge_points.copy()]
        self.selected_vertex = None

        # 4) Reset canvas overlay IDs so they get recreated fresh
        if getattr(self, '_poly_id', None) is not None:
            try:
                self.edit_canvas.delete(self._poly_id)
            except Exception:
                pass
            self._poly_id = None
        for vid in getattr(self, '_vertex_ids', []):
            try:
                self.edit_canvas.delete(vid)
            except Exception:
                pass
        self._vertex_ids = []

        self._update_poly_nav_label()
        self.redraw_canvas()

        n_pts = len(self.edited_edge_points)
        n_remaining = len(session)
        print(f"[Edit] Switched to {new_ftype} ({n_pts} vertices). "
              f"{n_remaining} other feature(s) in session.")

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

        # simple layout: 3 columns of label+slider
        lbl_sat.grid(row=0, column=0, padx=(4, 8), pady=6, sticky="w")
        self.slider_sat.grid(row=0, column=1, padx=(0, 20), pady=6, sticky="ew")

        lbl_exp.grid(row=0, column=2, padx=(4, 8), pady=6, sticky="w")
        self.slider_exp.grid(row=0, column=3, padx=(0, 20), pady=6, sticky="ew")

        lbl_hil.grid(row=0, column=4, padx=(4, 8), pady=6, sticky="w")
        self.slider_hil.grid(row=0, column=5, padx=(0, 4),  pady=6, sticky="ew")

        # allow sliders to expand
        row.grid_columnconfigure(1, weight=1)
        row.grid_columnconfigure(3, weight=1)
        row.grid_columnconfigure(5, weight=1)

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

    def _draw_reference_features(self):
        """Draw all session features (non-active) on the edit canvas as green lines."""
        if not hasattr(self, "edit_canvas"):
            return
        # Delete any previous reference lines
        self.edit_canvas.delete("__ref_feature__")

        session = getattr(self, '_edit_session_features', [])
        if not session:
            return

        proxy_s = getattr(self, '_edit_proxy_scale', 1.0)
        for ftype, pts in session:
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
            self.edit_canvas.create_line(
                *flat, fill="#00cc00", width=2, tags="__ref_feature__"
            )

    def _draw_poly_in_place(self):
        if not hasattr(self, "edit_canvas"): return
        pts = getattr(self, "edited_edge_points", []) or []
        if len(pts) < 2:
            if getattr(self, "_poly_id", None):
                self.edit_canvas.coords(self._poly_id, ())
            return

        flat = []
        for x, y in pts:
            sx, sy = self._scaled(x, y)
            flat.extend([sx, sy])

        if getattr(self, "_poly_id", None) is None:
            self._poly_id = self.edit_canvas.create_line(
                *flat, fill="cyan", width=2, tags="__poly__"
            )
        else:
            self.edit_canvas.coords(self._poly_id, *flat)

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

    def _record_history(self):
        """Push a snapshot and clear redo (new branch)."""
        self.edit_history.append(self.edited_edge_points.copy())
        # optional cap to keep memory sane
        if len(self.edit_history) > 50:
            self.edit_history.pop(0)
        # any new action invalidates redo chain
        self.redo_history.clear()


    def on_canvas_single_click(self, event):
        """
        For new shape creation. If is_polygon_mode is True,
        we auto-close the polygon if the user clicks near the first vertex.
        Otherwise we keep adding points.
        """
        x_canvas = self.edit_canvas.canvasx(event.x)
        y_canvas = self.edit_canvas.canvasy(event.y)
        x_img = x_canvas / self.zoom_scale
        y_img = y_canvas / self.zoom_scale

        if self.is_polygon_mode and len(self.edited_edge_points) > 2:
            # If close to the first vertex => close the polygon
            first_pt = self.edited_edge_points[0]
            if self.distance(x_img, y_img, first_pt[0], first_pt[1]) < 10:
                # Close it
                closed_points = self.edited_edge_points[:]
                # Ensure polygon is closed
                if closed_points[-1] != closed_points[0]:
                    closed_points.append(closed_points[0])

                # Scale from proxy coords back to full-image coords
                proxy_s = getattr(self, '_edit_proxy_scale', 1.0)
                if proxy_s != 1.0 and proxy_s > 0:
                    inv = 1.0 / proxy_s
                    closed_points = [[x * inv, y * inv] for x, y in closed_points]

                self._edit_session_features.append(("polygon", closed_points))
                print(
                    "Polygon closed and added to session. You can create more or press Confirm.")
                self.edited_edge_points = []
                self.edit_history = []
                self.selected_vertex = None
                self._record_history()
                self._update_poly_nav_label()
                self.redraw_canvas()
                return

        self.edited_edge_points.append([x_img, y_img])
        self._record_history()
        self.redraw_canvas()

    def on_canvas_press(self, event):
        x_canvas = self.edit_canvas.canvasx(event.x)
        y_canvas = self.edit_canvas.canvasy(event.y)
        x_img = x_canvas / self.zoom_scale
        y_img = y_canvas / self.zoom_scale

        min_dist = 10
        self.selected_vertex = None
        for idx, (vx, vy) in enumerate(self.edited_edge_points):
            if self.distance(x_img, y_img, vx, vy) < min_dist:
                self.selected_vertex = idx
                self.edit_canvas.itemconfig(f"vertex_{idx}", fill="yellow")
                break

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

        # If we were in creation mode (single-click adds points),
        # turn it off and restore the normal edit bindings.
        if getattr(self, "creation_mode", False):
            try:
                self.edit_canvas.unbind("<Button-1>")
            except Exception:
                pass
            self.creation_mode = False
            # re-bind the standard edit handlers
            self.edit_canvas.bind("<ButtonPress-1>",    self.on_canvas_press)
            self.edit_canvas.bind("<B1-Motion>",        self.on_canvas_drag)
            self.edit_canvas.bind("<ButtonRelease-1>",  self.on_canvas_release)
            self.edit_canvas.bind("<Double-Button-1>",  self.on_canvas_double_click)


    def delete_all_vertices(self):
        self.edited_edge_points = []
        self._record_history()
        self.redraw_canvas()

    def undo_last_action(self, event=None):
        """Ctrl+Z: move one step back and push current into redo."""
        if len(self.edit_history) > 1:
            current = self.edit_history.pop()              # the state we're leaving
            self.redo_history.append(current)              # enable redo
            self.edited_edge_points = self.edit_history[-1].copy()
            self.redraw_canvas()

    def redo_last_action(self, event=None):
        """Ctrl+R: re-apply the next state from redo."""
        if self.redo_history:
            nxt = self.redo_history.pop()
            self.edit_history.append(nxt)
            self.edited_edge_points = nxt.copy()
            self.redraw_canvas()


    def reset_to_initial(self):
        self.edited_edge_points = self.initial_edge_points.copy()
        self.edit_history = [self.edited_edge_points.copy()]
        self.draw_edge_on_canvas()

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
        base_folder = os.path.join(export_path, "training dataset", "class_points")
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


    def export_training_data(self):
        """
        Exports confirmed features into masks, overlays, GeoJSON, and COCO JSON.
        Uses Feature ID as a filename suffix for all outputs.
        """
        import warnings
        from rasterio.errors import NotGeoreferencedWarning
        warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
        warnings.filterwarnings(
            "ignore", category=UserWarning, message=".*crs.*was not provided.*"
        )

        # 1) Read user input
        feature_name_raw = self.feature_id_entry.get().strip()
        if not feature_name_raw:
            messagebox.showerror("Error", "Feature ID is missing. Provide a label/name for your dataset.")
            return
        feature_name = self._sanitize_feature_id(feature_name_raw)

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
            messagebox.showerror("Error", "No full image is loaded or invalid image data.")
            return
        if not self.features and self.edge_points:
            self.features = [("polyline", self.edge_points.copy())]
        if not self.features:
            messagebox.showerror("Error", "No features to export. Please confirm a shape or polygon first.")
            return

        # 3) Prepare output folders
        export_path = self.export_path_entry.get().strip()
        if not export_path:
            messagebox.showerror("Error", "Please specify an export folder.")
            return
        base_folder     = os.path.join(export_path, "training dataset")
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
        suffix = f"_{feature_name}" if feature_name else ""
        out_stem = f"{stem}{suffix}"                            # e.g. name_shoreline

        height, width = self.full_image.shape[:2]

        # 4) Copy original image but with feature suffix in filename
        image_copy_name = f"{out_stem}{src_ext}"                # name_shoreline.tif
        image_copy_path = os.path.join(images_folder, image_copy_name)
        try:
            shutil.copy2(self.image_path, image_copy_path)
        except Exception:
            # If copy fails, still proceed with annotations for robustness
            pass

        # 5) Read georeference if any
        try:
            with rasterio.open(self.image_path) as src:
                transform = src.transform
                crs = src.crs
        except Exception:
            transform = None
            crs = None
        if crs is None:
            print(f"[export_training_data] No georeference found on {base_name}; using pixel coords in GeoJSON.")

        # 6) Build mask, overlay, COCO annotations, and GeoJSON shapes
        mask = np.zeros((height, width), dtype=np.uint8)
        overlay = self.full_image.copy()
        thickness = int(self.edge_thickness_slider.get()) if hasattr(self, 'edge_thickness_slider') else 2

        coco_annotations = []
        annotation_id = 1
        shape_list = []

        for feature_type, pts in self.features:
            if len(pts) < 2:
                continue

            pts_np = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            is_poly = (feature_type == "polygon")
            if is_poly:
                closed = pts[:] + ([pts[0]] if pts[0] != pts[-1] else [])
                cv2.fillPoly(mask, [np.array(closed, np.int32).reshape((-1, 1, 2))], 255)
                cv2.polylines(overlay, [pts_np], True, (0, 255, 0), 2)
            else:
                cv2.polylines(mask, [pts_np], False, 255, thickness)
                cv2.polylines(overlay, [pts_np], False, (0, 255, 0), thickness)

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
        cv2.imwrite(mask_path, mask)
        cv2.imwrite(overlay_path, overlay)

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

        # final pop-up + console summary
        class_pts_path = self._export_class_points(export_path)
        print(f"[export_training_data] Export complete for {base_name}:")
        print(f"    image copy → {image_copy_path}")
        print(f"    mask       → {mask_path}")
        print(f"    overlay    → {overlay_path}")
        print(f"    geojson    → {geojson_path}")
        print(f"    coco       → {coco_path}")
        if class_pts_path:
            print(f"    class pts  → {class_pts_path}")
        cls_msg = f"\n- Class pts ⇒ {class_pts_path}" if class_pts_path else ""
        messagebox.showinfo(
            "Export Training Data",
            f"Export complete:\n"
            f"- Image   ⇒ {image_copy_path}\n"
            f"- Mask    ⇒ {mask_path}\n"
            f"- Overlay ⇒ {overlay_path}\n"
            f"- GeoJSON ⇒ {geojson_path}\n"
            f"- COCO    ⇒ {coco_path}"
            f"{cls_msg}"
        )


    def export_mask_as_training_data(self):
        if self.current_mask is None:
            messagebox.showerror("Error", "No mask available for exporting.")
            return
        export_path = self.export_path_entry.get().strip()
        if not export_path:
            messagebox.showerror("Error", "Please select an export folder.")
            return

        base_folder = os.path.join(export_path, "training dataset")
        masks_folder = os.path.join(base_folder, "masks")
        overlays_folder = os.path.join(base_folder, "overlays")
        os.makedirs(masks_folder, exist_ok=True)
        os.makedirs(overlays_folder, exist_ok=True)

        base_name = os.path.basename(self.image_path)
        mask_filename = os.path.splitext(base_name)[0] + "_mask.png"
        mask_path = os.path.join(masks_folder, mask_filename)
        cv2.imwrite(mask_path, self.current_mask)

        # Create overlay
        if self.full_image.shape[2] == 4:
            overlay = cv2.cvtColor(self.full_image, cv2.COLOR_BGRA2BGR)
        else:
            overlay = self.full_image.copy()

        mask_color = cv2.cvtColor(self.current_mask, cv2.COLOR_GRAY2BGR)
        mask_color_resized = cv2.resize(
            mask_color, (overlay.shape[1], overlay.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
        overlay = cv2.addWeighted(overlay, 0.7, mask_color_resized, 0.3, 0)

        overlay_filename = os.path.splitext(base_name)[0] + "_overlay.png"
        overlay_path = os.path.join(overlays_folder, overlay_filename)
        cv2.imwrite(overlay_path, overlay)

        self._export_class_points(export_path)

        messagebox.showinfo("Export Mask Training Data",
                            f"Mask training data exported.\n{base_folder}")

    def export_as_test_data(self):
        if not isinstance(self.full_image, np.ndarray):
            messagebox.showerror("Error", "No full image available.")
            return
        export_path = self.export_path_entry.get().strip()
        if not export_path:
            messagebox.showerror(
                "Error", "Please specify a path to save files.")
            return
        test_folder = os.path.join(export_path, "test dataset")
        os.makedirs(test_folder, exist_ok=True)
        base_name = os.path.basename(self.image_path)
        dest_path = os.path.join(test_folder, base_name)
        shutil.copy2(self.image_path, dest_path)
        messagebox.showinfo("Export Test Data",
                            f"Test image exported to:\n{dest_path}")

    def export_as_overlay(self):
        if not isinstance(self.full_image, np.ndarray):
            messagebox.showerror("Error", "No full image available.")
            return
        if not self.edge_points or len(self.edge_points) < 2:
            messagebox.showerror("Error", "No valid edge to export.")
            return
        export_path = self.export_path_entry.get().strip()
        if not export_path:
            messagebox.showerror(
                "Error", "Please specify a path to save files.")
            return
        base_name = os.path.basename(self.image_path)
        overlay = self.full_image.copy()
        thickness = int(self.edge_thickness_slider.get())
        pts = np.array(self.edge_points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(overlay, [pts], isClosed=False,
                      color=(0, 255, 0), thickness=thickness)
        out_overlay = os.path.join(
            export_path, os.path.splitext(base_name)[0] + "_overlay.png")
        cv2.imwrite(out_overlay, overlay)
        messagebox.showinfo(
            "Export Overlay", f"Overlay saved to:\n{out_overlay}")