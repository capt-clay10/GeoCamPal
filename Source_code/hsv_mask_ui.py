"""
hsv_mask_ui.py
──────────────
UI construction, control panel, settings I/O, toggle helpers,
and standalone helper classes (BBoxSelectorWindow, StdoutRedirector).

This module is a *mixin* — it is meant to be inherited by HSVMaskTool
together with the processing and editing mixins.
"""

import os
import json
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
import numpy as np

# %% window resizer 

def fit_geometry(window, design_w, design_h, resizable=True, margin=0.90):
    """
    Scale a window to fit the current screen while preserving
    the aspect ratio of the original design size.
    Centers the result on screen.  Never upscales beyond the design size.

    Parameters
    ----------
    window      : Tk / CTk / CTkToplevel instance
    design_w/h  : the "intended" pixel size (the old hardcoded values)
    resizable   : whether the user can drag-resize afterward
    margin      : fraction of screen to occupy at most (0.90 = 90 %)
    """
    screen_w = window.winfo_screenwidth()
    screen_h = window.winfo_screenheight()

    max_w = int(screen_w * margin)
    max_h = int(screen_h * margin)

    scale = min(max_w / design_w, max_h / design_h, 1.0)

    final_w = int(design_w * scale)
    final_h = int(design_h * scale)

    x = (screen_w - final_w) // 2
    y = max(0, (screen_h - final_h) // 2)

    window.geometry(f"{final_w}x{final_h}+{x}+{y}")
    window.resizable(resizable, resizable)


# %% Standalone helper classes
# ──────────────────────────────────────────────

class BBoxSelectorWindow(tk.Toplevel):
    def __init__(self, master, pil_image, callback, **kwargs):
        """
        A small window for selecting a bounding box.
        pil_image: a PIL.Image instance (the original image).
        callback: a function that accepts a bbox tuple (x, y, w, h) in original coords.
        """
        super().__init__(master, **kwargs)
        self.title("Select Bounding Box")
        self.callback = callback
        self.original_pil_image = pil_image
        self.zoom_scale = 1.0

        top_frame = tk.Frame(self)
        top_frame.pack(fill="both", expand=True)

        container = tk.Frame(top_frame)
        container.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(container, cursor="cross")
        self.canvas.grid(row=0, column=0, sticky="nsew")

        v_scroll = tk.Scrollbar(
            container, orient="vertical", command=self.canvas.yview)
        v_scroll.grid(row=0, column=1, sticky="ns")
        h_scroll = tk.Scrollbar(
            container, orient="horizontal", command=self.canvas.xview)
        h_scroll.grid(row=1, column=0, sticky="ew")
        self.canvas.configure(yscrollcommand=v_scroll.set,
                              xscrollcommand=h_scroll.set)

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        bottom_frame = tk.Frame(self)
        bottom_frame.pack(fill="x", pady=5)

        btn_zoom_in = tk.Button(
            bottom_frame, text="Zoom In", command=self.zoom_in)
        btn_zoom_in.pack(side="left", padx=5)
        btn_zoom_out = tk.Button(
            bottom_frame, text="Zoom Out", command=self.zoom_out)
        btn_zoom_out.pack(side="left", padx=5)
        btn_confirm = tk.Button(
            bottom_frame, text="Confirm BBox", command=self.confirm_bbox)
        btn_confirm.pack(side="right", padx=5)

        self.display_image()

        # Rectangle selection
        self.rect_id = None
        self.start_x = None
        self.start_y = None
        self.bbox = None

        # Bind
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

    def display_image(self):
        orig_w, orig_h = self.original_pil_image.size
        new_w = int(orig_w * self.zoom_scale)
        new_h = int(orig_h * self.zoom_scale)
        self.resized_image = self.original_pil_image.resize(
            (new_w, new_h), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(self.resized_image)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.canvas.config(scrollregion=(0, 0, new_w, new_h))

    def zoom_in(self):
        self.zoom_scale *= 1.25
        self.display_image()
        if self.rect_id:
            self.canvas.delete(self.rect_id)
            self.rect_id = None

    def zoom_out(self):
        self.zoom_scale *= 0.8
        if self.zoom_scale < 0.1:
            self.zoom_scale = 0.1
        self.display_image()
        if self.rect_id:
            self.canvas.delete(self.rect_id)
            self.rect_id = None

    def on_button_press(self, event):
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y,
            outline="red", width=2
        )

    def on_move_press(self, event):
        cur_x = self.canvas.canvasx(event.x)
        cur_y = self.canvas.canvasy(event.y)
        if self.rect_id:
            self.canvas.coords(self.rect_id, self.start_x,
                               self.start_y, cur_x, cur_y)

    def on_button_release(self, event):
        end_x = self.canvas.canvasx(event.x)
        end_y = self.canvas.canvasy(event.y)
        x1, x2 = sorted([self.start_x, end_x])
        y1, y2 = sorted([self.start_y, end_y])
        self.bbox = (x1, y1, x2 - x1, y2 - y1)

    def confirm_bbox(self):
        if not self.bbox:
            messagebox.showwarning(
                "BBox Selection", "No bounding box was selected.")
            return
        x, y, w, h = self.bbox
        x_orig = int(x / self.zoom_scale)
        y_orig = int(y / self.zoom_scale)
        w_orig = int(w / self.zoom_scale)
        h_orig = int(h / self.zoom_scale)
        final_bbox = (x_orig, y_orig, w_orig, h_orig)
        self.callback(final_bbox)
        self.destroy()


class StdoutRedirector:
    """
    A class to redirect stdout to a given text widget.
    """

    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        self.text_widget.insert(tk.END, message)
        self.text_widget.see(tk.END)  # auto-scroll

    def flush(self):
        pass  # no-op for Python's IO flush requirements


# %% UI Mixin
# ──────────────────────────────────────────────

class HSVMaskUIMixin:
    """Mixin that supplies all UI‑construction, control‑panel, toggle, and settings methods."""

    # -------------- MAIN CONTROL PANEL --------------

    def setup_controls(self, parent):
        """
        Build out the controls (import, HSV sliders, edge detection, export).
        """
        # ── Row 1: Import Frame ─────────────────────────────────────────────────────
        import_frame = ctk.CTkFrame(parent)
        import_frame.pack(side="top", fill="x", pady=5)

        if self.mode in ("ml", "batch"):
            load_btn = ctk.CTkButton(import_frame, text="Load Folder", command=self.load_folder)
        else:
            load_btn = ctk.CTkButton(import_frame, text="Load Image", command=self.load_image)
        load_btn.pack(side="left", padx=5)

        bbox_control_frame = ctk.CTkFrame(import_frame)
        bbox_control_frame.pack(side="left", padx=5)

        self.bbox_frame = ctk.CTkFrame(import_frame)
        self.bbox_frame.pack(side="left", padx=5)

        self.bbox_check = ctk.CTkCheckBox(
            self.bbox_frame,
            text="Use Bounding Box?",
            variable=self.use_bbox,
            command=self.toggle_bbox_options
        )
        self.bbox_check.pack(side="left", padx=5)

        self.bbox_entry = ctk.CTkEntry(self.bbox_frame, width=160)
        self.bbox_entry.insert(0, "(460, 302, 2761, 1782)")

        self.inner_mask_check = ctk.CTkCheckBox(
            bbox_control_frame, text="Use Inner Mask", variable=self.use_inner_mask
        )
        self.inner_mask_check.pack(side="left", padx=5)
        self.inner_mask_check.configure(state="disabled")

        # ── DROPDOWN 1: AOI / Profile-based Filter ─────────────────────────────────
        self.aoi_check_frame = ctk.CTkFrame(parent)
        self.aoi_check_frame.pack(side="top", fill="x", pady=(5, 0))
        self.aoi_check = ctk.CTkCheckBox(
            self.aoi_check_frame, text="AOI / Profile Filter",
            variable=self.use_aoi_filter, command=self._toggle_aoi_controls)
        self.aoi_check.pack(side="left", padx=5)
        ctk.CTkLabel(self.aoi_check_frame,
                     text="(narrow search region using intensity profile)",
                     font=("Arial", 9), text_color="gray").pack(side="left", padx=5)

        # AOI controls (hidden by default)
        self.aoi_opts_frame = ctk.CTkFrame(parent)
        aoi_coord_row = ctk.CTkFrame(self.aoi_opts_frame)
        aoi_coord_row.pack(fill="x", padx=5, pady=2)
        ctk.CTkLabel(aoi_coord_row, text="Profile Start (x,y):").pack(side="left", padx=2)
        self.aoi_x1_entry = ctk.CTkEntry(aoi_coord_row, width=50, placeholder_text="x1")
        self.aoi_x1_entry.pack(side="left", padx=2)
        self.aoi_y1_entry = ctk.CTkEntry(aoi_coord_row, width=50, placeholder_text="y1")
        self.aoi_y1_entry.pack(side="left", padx=2)
        ctk.CTkLabel(aoi_coord_row, text="End (x,y):").pack(side="left", padx=2)
        self.aoi_x2_entry = ctk.CTkEntry(aoi_coord_row, width=50, placeholder_text="x2")
        self.aoi_x2_entry.pack(side="left", padx=2)
        self.aoi_y2_entry = ctk.CTkEntry(aoi_coord_row, width=50, placeholder_text="y2")
        self.aoi_y2_entry.pack(side="left", padx=2)
        ctk.CTkLabel(aoi_coord_row, text="Width:").pack(side="left", padx=2)
        self.aoi_width_entry = ctk.CTkEntry(aoi_coord_row, width=40)
        self.aoi_width_entry.insert(0, "5")
        self.aoi_width_entry.pack(side="left", padx=2)
        ctk.CTkButton(aoi_coord_row, text="Draw on Image", width=110,
                      command=self._aoi_draw_on_image).pack(side="left", padx=5)
        ctk.CTkButton(aoi_coord_row, text="Draw AOI Polygon", width=130,
                      command=self._aoi_draw_polygon).pack(side="left", padx=5)

        # Polygon coordinates row (saveable text)
        aoi_poly_row = ctk.CTkFrame(self.aoi_opts_frame)
        aoi_poly_row.pack(fill="x", padx=5, pady=2)
        ctk.CTkLabel(aoi_poly_row, text="Polygon AOI (x,y; ...):").pack(side="left", padx=2)
        self.aoi_polygon_entry = ctk.CTkEntry(aoi_poly_row, width=400,
                                               placeholder_text="e.g. 100,50; 300,50; 300,200; 100,200")
        self.aoi_polygon_entry.pack(side="left", padx=2, fill="x", expand=True)

        aoi_method_row = ctk.CTkFrame(self.aoi_opts_frame)
        aoi_method_row.pack(fill="x", padx=5, pady=2)
        ctk.CTkLabel(aoi_method_row, text="Method:").pack(side="left", padx=2)
        ctk.CTkOptionMenu(aoi_method_row, variable=self.aoi_method,
                          values=["Threshold", "Variance", "Otsu"],
                          width=100).pack(side="left", padx=2)
        ctk.CTkLabel(aoi_method_row, text="Min:").pack(side="left", padx=2)
        self.aoi_min_entry = ctk.CTkEntry(aoi_method_row, width=50)
        self.aoi_min_entry.insert(0, "0")
        self.aoi_min_entry.pack(side="left", padx=2)
        ctk.CTkLabel(aoi_method_row, text="Max:").pack(side="left", padx=2)
        self.aoi_max_entry = ctk.CTkEntry(aoi_method_row, width=50)
        self.aoi_max_entry.insert(0, "255")
        self.aoi_max_entry.pack(side="left", padx=2)
        ctk.CTkButton(aoi_method_row, text="Preview Profile", width=110,
                      command=self._aoi_preview_profile).pack(side="left", padx=5)
        ctk.CTkButton(aoi_method_row, text="Apply AOI", width=90,
                      command=self._aoi_apply_filter, fg_color="#0F52BA").pack(side="left", padx=5)
        ctk.CTkButton(aoi_method_row, text="Clear AOI", width=80,
                      command=self._aoi_clear,
                      fg_color="#8B0000", hover_color="#A52A2A").pack(side="left", padx=3)
        self.aoi_opts_frame.pack_forget()

        # ── Row 1.5: ML predicted mask (directly under import row) ─────────────────
        # Checkbox row (always visible)
        self.ml_row = ctk.CTkFrame(parent)
        self.ml_row.pack(side="top", fill="x", pady=(2, 0))
        self.ml_check = ctk.CTkCheckBox(
            self.ml_row,
            text="Use ML predicted mask",
            variable=self.use_ml_pred_mask,
            command=self.toggle_ml_mask_options
        )
        self.ml_check.pack(side="left", padx=5)

        # Options row (appears only when checkbox is ticked) — created now, packed later
        self.ml_opts_row = ctk.CTkFrame(parent)
        self.ml_opts_row.pack_forget()

        if self.mode == "individual":
            # Use a small grid so the calc button doesn't get squished by a long path
            self.ml_opts_row.grid_columnconfigure(1, weight=1)

            self.btn_browse_ml_mask = ctk.CTkButton(
                self.ml_opts_row, text="Load associated mask", command=self.browse_ml_mask_file
            )
            self.btn_browse_ml_mask.grid(row=0, column=0, padx=5, pady=3, sticky="w")

            # Shortened (display) text; full path is kept in self.ml_mask_file_path
            self.lbl_ml_mask_path = ctk.CTkLabel(
                self.ml_opts_row, textvariable=self.ml_mask_file_disp, width=320, anchor="w"
            )
            self.lbl_ml_mask_path.grid(row=0, column=1, padx=5, pady=3, sticky="we")


        elif self.mode == "ml":
            self.ml_opts_row.grid_columnconfigure(1, weight=1)

            self.btn_browse_ml_mask_folder = ctk.CTkButton(
                self.ml_opts_row, text="Load associated mask folder", command=self.browse_ml_mask_folder
            )
            self.btn_browse_ml_mask_folder.grid(row=0, column=0, padx=5, pady=3, sticky="w")

            self.lbl_ml_mask_folder = ctk.CTkLabel(
                self.ml_opts_row, textvariable=self.ml_mask_folder_disp, width=320, anchor="w"
            )
            self.lbl_ml_mask_folder.grid(row=0, column=1, padx=5, pady=3, sticky="we")

            ctk.CTkLabel(self.ml_opts_row, text="Common file name length").grid(
                row=0, column=2, padx=(10, 2), pady=3, sticky="e"
            )
            self.entry_common_len = ctk.CTkEntry(
                self.ml_opts_row, width=80, textvariable=self.common_name_len_var, placeholder_text=""
            )
            self.entry_common_len.grid(row=0, column=3, padx=2, pady=3, sticky="w")


        # ── DROPDOWN 2: HSV Masking (checkbox + collapsible) ─────────────────────
        self.hsv_label_frame = ctk.CTkFrame(parent)
        self.hsv_label_frame.pack(side="top", fill="x", pady=(5, 0))
        self.hsv_check = ctk.CTkCheckBox(
            self.hsv_label_frame, text="HSV Colour Masking",
            variable=self.use_hsv_masking,
            command=self._toggle_hsv_controls)
        self.hsv_check.pack(side="left", padx=5)
        ctk.CTkLabel(self.hsv_label_frame,
                     text="(constrained to the AOI when AOI / Profile Filter is active)",
                     font=("Arial", 9), text_color="gray").pack(side="left", padx=5)

        # Container for all HSV controls (collapsible)
        self.hsv_controls_container = ctk.CTkFrame(parent)
        self.hsv_controls_container.pack(side="top", fill="x")

        # ── Row 2: Enhance Frame ──
        self.enhance_frame = ctk.CTkFrame(self.hsv_controls_container)
        self.enhance_frame.pack(side="top", fill="x", pady=5)

        self.enable_enhancements = tk.BooleanVar(master=self, value=True)
        enhance_chk = ctk.CTkCheckBox(
            self.enhance_frame, text="Enhance?", variable=self.enable_enhancements
        )
        enhance_chk.pack(side="left", padx=5)

        ctk.CTkLabel(self.enhance_frame, text="S Mult").pack(side="left", padx=2)
        self.s_multiplier_slider = ctk.CTkSlider(self.enhance_frame, from_=100, to=500)
        self.s_multiplier_slider.set(100)
        self.s_multiplier_slider.pack(side="left", padx=2)

        ctk.CTkLabel(self.enhance_frame, text="V Mult").pack(side="left", padx=2)
        self.v_multiplier_slider = ctk.CTkSlider(self.enhance_frame, from_=100, to=500)
        self.v_multiplier_slider.set(100)
        self.v_multiplier_slider.pack(side="left", padx=2)

        self.use_dual_hsv = tk.BooleanVar(master=self, value=False)
        dual_chk = ctk.CTkCheckBox(
            self.enhance_frame,
            text="Use Dual HSV Range",
            variable=self.use_dual_hsv,
            command=self.toggle_dual_sliders
        )
        dual_chk.pack(side="left", padx=10)

        # ── Row 3: HSV Sliders ──────────────────────────────────────────────────────
        hsv_frame = ctk.CTkFrame(self.hsv_controls_container)
        hsv_frame.pack(side="top", fill="x", pady=5)

        # Lower row
        lower_container = ctk.CTkFrame(hsv_frame)
        lower_container.pack(side="top", fill="x", pady=2)

        first_lower_frame = ctk.CTkFrame(lower_container)
        first_lower_frame.pack(side="left", fill="x", expand=True)
        ctk.CTkLabel(first_lower_frame, text="Lower HSV:").pack(side="left", padx=5)
        self.h_low_slider, self.h_low_var, self.h_low_lbl = self.make_slider_with_label(
            first_lower_frame, "H", 0, 255, 0
        )
        self.s_low_slider, self.s_low_var, self.s_low_lbl = self.make_slider_with_label(
            first_lower_frame, "S", 0, 255, 0
        )
        self.v_low_slider, self.v_low_var, self.v_low_lbl = self.make_slider_with_label(
            first_lower_frame, "V", 0, 255, 0
        )

        self.dual_lower_frame = ctk.CTkFrame(lower_container)
        self.dual_lower_frame.pack(side="left", fill="x", expand=True)
        ctk.CTkLabel(self.dual_lower_frame, text="Lower HSV (Dual):").pack(side="left", padx=5)
        self.h2_low_slider, self.h2_low_var, self.h2_low_lbl = self.make_slider_with_label(
            self.dual_lower_frame, "H2", 0, 255, 0
        )
        self.s2_low_slider, self.s2_low_var, self.s2_low_lbl = self.make_slider_with_label(
            self.dual_lower_frame, "S2", 0, 255, 0
        )
        self.v2_low_slider, self.v2_low_var, self.v2_low_lbl = self.make_slider_with_label(
            self.dual_lower_frame, "V2", 0, 255, 0
        )

        # Upper row
        upper_container = ctk.CTkFrame(hsv_frame)
        upper_container.pack(side="top", fill="x", pady=2)

        first_upper_frame = ctk.CTkFrame(upper_container)
        first_upper_frame.pack(side="left", fill="x", expand=True)
        ctk.CTkLabel(first_upper_frame, text="Upper HSV:").pack(side="left", padx=5)
        self.h_high_slider, self.h_high_var, self.h_high_lbl = self.make_slider_with_label(
            first_upper_frame, "H", 0, 255, 255
        )
        self.s_high_slider, self.s_high_var, self.s_high_lbl = self.make_slider_with_label(
            first_upper_frame, "S", 0, 255, 255
        )
        self.v_high_slider, self.v_high_var, self.v_high_lbl = self.make_slider_with_label(
            first_upper_frame, "V", 0, 255, 255
        )

        self.dual_upper_frame = ctk.CTkFrame(upper_container)
        self.dual_upper_frame.pack(side="left", fill="x", expand=True)
        ctk.CTkLabel(self.dual_upper_frame, text="Upper HSV (Dual):").pack(side="left", padx=5)
        self.h2_high_slider, self.h2_high_var, self.h2_high_lbl = self.make_slider_with_label(
            self.dual_upper_frame, "H2", 0, 255, 255
        )
        self.s2_high_slider, self.s2_high_var, self.s2_high_lbl = self.make_slider_with_label(
            self.dual_upper_frame, "S2", 0, 255, 255
        )
        self.v2_high_slider, self.v2_high_var, self.v2_high_lbl = self.make_slider_with_label(
            self.dual_upper_frame, "V2", 0, 255, 255
        )

        hsv_action_row = ctk.CTkFrame(self.hsv_controls_container)
        hsv_action_row.pack(side="top", fill="x", pady=(2, 5))
        if self.mode != "batch":
            ctk.CTkButton(hsv_action_row, text="Calculate Mask", command=self.calculate_mask, fg_color="#0F52BA").pack(side="left", padx=5)
        self.hsv_invert_check = ctk.CTkCheckBox(
            hsv_action_row, text="Invert Mask?", variable=self.do_invert_mask
        )
        self.hsv_invert_check.pack(side="left", padx=10)

        # ── DROPDOWN 3: Colour Picker ──────────────────────────────────────────────
        self.cpick_check_frame = ctk.CTkFrame(parent)
        self.cpick_check_frame.pack(side="top", fill="x", pady=(5, 0))
        self.cpick_check = ctk.CTkCheckBox(
            self.cpick_check_frame, text="Multi-sample Colour Selection",
            variable=self.use_color_picker,
            command=self._toggle_color_picker_controls)
        self.cpick_check.pack(side="left", padx=5)
        ctk.CTkLabel(self.cpick_check_frame,
                     text="(sample a class to remove or keep within the image / AOI)",
                     font=("Arial", 9), text_color="gray").pack(side="left", padx=5)

        self.cpick_opts_frame = ctk.CTkFrame(parent)

        cpick_row1 = ctk.CTkFrame(self.cpick_opts_frame)
        cpick_row1.pack(fill="x", padx=5, pady=2)
        ctk.CTkLabel(cpick_row1, text="Method:").pack(side="left", padx=2)
        ctk.CTkOptionMenu(cpick_row1, variable=self.color_pick_method,
                          values=["Color Distance", "GrabCut"],
                          width=130).pack(side="left", padx=2)
        ctk.CTkLabel(cpick_row1, text="Output:").pack(side="left", padx=(10, 2))
        ctk.CTkOptionMenu(cpick_row1, variable=self.color_pick_output_mode,
                          values=["Remove selection", "Keep selection only"],
                          width=170).pack(side="left", padx=2)
        ctk.CTkLabel(cpick_row1, text="Patch radius:").pack(side="left", padx=(10, 2))
        self.cpick_patch_radius_entry = ctk.CTkEntry(cpick_row1, width=55,
                                                     textvariable=self.color_pick_patch_radius)
        self.cpick_patch_radius_entry.pack(side="left", padx=2)
        ctk.CTkLabel(cpick_row1, text="px", font=("Arial", 9)).pack(side="left", padx=(0, 8))

        # ── Row 2: sample buttons ──
        cpick_row2 = ctk.CTkFrame(self.cpick_opts_frame)
        cpick_row2.pack(fill="x", padx=5, pady=2)
        ctk.CTkButton(cpick_row2, text="Add Remove Samples", width=150,
                      command=lambda: self._start_color_pick("remove")).pack(side="left", padx=5)
        self.cpick_remove_label = ctk.CTkLabel(
            cpick_row2, text="Remove: 0 pts", font=("Arial", 9), anchor="w", width=260
        )
        self.cpick_remove_label.pack(side="left", padx=3)
        ctk.CTkButton(cpick_row2, text="Add Keep Samples", width=150,
                      command=lambda: self._start_color_pick("keep")).pack(side="left", padx=5)
        self.cpick_keep_label = ctk.CTkLabel(
            cpick_row2, text="Keep: 0 pts", font=("Arial", 9), anchor="w", width=260
        )
        self.cpick_keep_label.pack(side="left", padx=3)

        # ── Row 3: class name entries ──
        cpick_row_names = ctk.CTkFrame(self.cpick_opts_frame)
        cpick_row_names.pack(fill="x", padx=5, pady=2)
        ctk.CTkLabel(cpick_row_names, text="Class A name (Remove):",
                     font=("Arial", 10)).pack(side="left", padx=(5, 2))
        self.cpick_class_a_entry = ctk.CTkEntry(
            cpick_row_names, width=120, textvariable=self.color_pick_class_a_name)
        self.cpick_class_a_entry.pack(side="left", padx=(0, 15))
        ctk.CTkLabel(cpick_row_names, text="Class B name (Keep):",
                     font=("Arial", 10)).pack(side="left", padx=(5, 2))
        self.cpick_class_b_entry = ctk.CTkEntry(
            cpick_row_names, width=120, textvariable=self.color_pick_class_b_name)
        self.cpick_class_b_entry.pack(side="left", padx=(0, 5))
        ctk.CTkLabel(cpick_row_names,
                     text="(names used when exporting class sample points)",
                     font=("Arial", 9), text_color="gray").pack(side="left", padx=5)

        # ── Row 4: tip text ──
        cpick_row3 = ctk.CTkFrame(self.cpick_opts_frame)
        cpick_row3.pack(fill="x", padx=5, pady=(0, 2))
        ctk.CTkLabel(
            cpick_row3,
            text="Tip: add 2–5 points on the class you want to remove. Keep-samples are optional but improve separation.",
            font=("Arial", 9),
            text_color="gray"
        ).pack(side="left", padx=5)

        # ── Row 4: Detect Class + Clear (below sample rows for natural flow) ──
        cpick_row4 = ctk.CTkFrame(self.cpick_opts_frame)
        cpick_row4.pack(fill="x", padx=5, pady=(2, 4))
        ctk.CTkButton(cpick_row4, text="Detect Class", width=140,
                      command=self._color_pick_detect,
                      fg_color="#1a6b3c").pack(side="left", padx=5)
        ctk.CTkButton(cpick_row4, text="Clear All Samples", width=120,
                      command=self._color_pick_clear,
                      fg_color="#8B0000", hover_color="#A52A2A").pack(side="left", padx=3)

        self.cpick_opts_frame.pack_forget()
        # ── Extraction label ───────────────────────────────────────────────────────
        self.step4_label_frame = ctk.CTkFrame(parent)
        self.step4_label_frame.pack(side="top", fill="x", pady=(5, 0))
        ctk.CTkLabel(self.step4_label_frame, text="Boundary / Polygon Extraction & Manual Editing",
                     font=("Arial", 11)).pack(side="left", padx=7)

        # ── Row 4: Edge Controls ───────────────────────────────────────────────────
        edge_container = ctk.CTkFrame(parent)
        edge_container.pack(side="top", fill="x", pady=5)

        calc_frame = ctk.CTkFrame(edge_container)
        calc_frame.pack(side="top", fill="x", pady=2)

        if self.mode == "ml":
            btn_prev = ctk.CTkButton(calc_frame, text="Previous", command=self.prev_image)
            btn_prev.pack(side="left", padx=5)
            btn_next = ctk.CTkButton(calc_frame, text="Next", command=self.next_image)
            btn_next.pack(side="left", padx=5)

        if self.mode in ("ml", "individual"):
            btn_boundary = ctk.CTkButton(calc_frame, text="Extract Boundary", command=self.extract_boundary_universal, fg_color="#0F52BA")
            btn_boundary.pack(side="left", padx=5)
            btn_polygon = ctk.CTkButton(calc_frame, text="Extract Polygon", command=self.extract_polygon_universal, fg_color="#0F52BA")
            btn_polygon.pack(side="left", padx=5)
            btn_cut_feature = ctk.CTkButton(
                calc_frame, text="Edit Detected Feature", command=self.cut_detected_feature, fg_color="#0F52BA"
            )
            btn_cut_feature.pack(side="left", padx=5)

            thickness_frame = ctk.CTkFrame(calc_frame)
            thickness_frame.pack(side="left", padx=3)
            ctk.CTkLabel(thickness_frame, text="Line Thickness (pixels)").pack(side="top")
            self.thickness_value_label = ctk.CTkLabel(thickness_frame, text="2")
            self.thickness_value_label.pack(side="top")
            self.edge_thickness_slider = ctk.CTkSlider(
                thickness_frame,
                from_=1,
                to=50,
                command=lambda val: self.thickness_value_label.configure(text=f"{int(float(val))}")
            )
            self.edge_thickness_slider.set(2)
            self.edge_thickness_slider.pack(side="top")

        adv_frame = ctk.CTkFrame(edge_container)
        adv_frame.pack(side="top", fill="x", pady=2)
        self.advanced_check = ctk.CTkCheckBox(
            adv_frame,
            text="Advanced Settings",
            variable=self.advanced_check_var,
            command=self.toggle_advanced_settings
        )
        self.advanced_check.pack(side="left", padx=5)

        self.min_contour_label = ctk.CTkLabel(adv_frame, text="Min contour size")
        self.min_contour_entry = ctk.CTkEntry(adv_frame, width=60, placeholder_text="Min")
        self.max_contour_label = ctk.CTkLabel(adv_frame, text="Max contour size")
        self.max_contour_entry = ctk.CTkEntry(adv_frame, width=60, placeholder_text="Max")

        self.min_contour_label.pack_forget()
        self.min_contour_entry.pack_forget()
        self.max_contour_label.pack_forget()
        self.max_contour_entry.pack_forget()

        # ── Row 5: Export Options ──────────────────────────────────────────────────
        export_frame = ctk.CTkFrame(parent)
        export_frame.pack(side="top", fill="x", pady=5)

        ctk.CTkLabel(export_frame, text="Export Folder:").pack(side="left", padx=5)
        self.export_path_entry = ctk.CTkEntry(export_frame, width=200)
        self.export_path_entry.pack(side="left", padx=5)
        btn_browse = ctk.CTkButton(export_frame, text="Browse output folder", command=self.browse_export_folder, fg_color="#8C7738")
        btn_browse.pack(side="left", padx=5)

        # ── Row 6: Feature frame ───────────────────────────────────────────────────
        featureid_frame = ctk.CTkFrame(parent)
        featureid_frame.pack(side="top", fill="x", pady=5)

        ctk.CTkLabel(featureid_frame, text="Feature ID:").pack(side="left", padx=5)
        self.feature_id_entry = ctk.CTkEntry(featureid_frame, width=100)
        self.feature_id_entry.pack(side="left", padx=5)

        ctk.CTkLabel(featureid_frame, text="Image ID:").pack(side="left", padx=5)
        self.image_id_entry = ctk.CTkEntry(featureid_frame, width=50)
        self.image_id_entry.insert(0, "1")
        self.image_id_entry.pack(side="left", padx=5)

        ctk.CTkLabel(featureid_frame, text="Category ID:").pack(side="left", padx=5)
        self.category_id_entry = ctk.CTkEntry(featureid_frame, width=50)
        self.category_id_entry.insert(0, "1")
        self.category_id_entry.pack(side="left", padx=5)

        # ── Row 7: Export buttons or Batch button ──────────────────────────────────
        export_buttons_frame = ctk.CTkFrame(parent)
        export_buttons_frame.pack(side="top", fill="x", pady=5)

        if self.mode in ("individual", "ml"):
            btn_export_edge = ctk.CTkButton(
                export_buttons_frame, text="Export feature as training data", command=self.export_training_data, fg_color="#6693F5"
            )
            btn_export_edge.pack(side="left", padx=5)
            btn_export_mask = ctk.CTkButton(
                export_buttons_frame, text="Export mask as training data", command=self.export_mask_as_training_data, fg_color="#6693F5"
            )
            btn_export_mask.pack(side="left", padx=5)
            btn_export_test = ctk.CTkButton(
                export_buttons_frame, text="Export as Test Data", command=self.export_as_test_data, fg_color="#6693F5"
            )
            btn_export_test.pack(side="left", padx=5)
            if self.mode == "individual":
                btn_export_overlay = ctk.CTkButton(
                    export_buttons_frame, text="Export as Overlay", command=self.export_as_overlay, fg_color="#6693F5"
                )
                btn_export_overlay.pack(side="left", padx=5)
        else:
            # batch mode
            btn_batch_process = ctk.CTkButton(export_frame, text="Batch Process", command=self.batch_process)
            btn_batch_process.pack(side="left", padx=5)

        btn_save_settings = ctk.CTkButton(export_frame, text="Save Settings", command=self.save_settings)
        btn_save_settings.pack(side="left", padx=5)
        btn_load_settings = ctk.CTkButton(export_frame, text="Load Settings", command=self.load_settings)
        btn_load_settings.pack(side="left", padx=5)

        # Reset button (all modes)
        btn_reset = ctk.CTkButton(
            export_frame, text="Reset Session", command=self.reset_session,
            fg_color="#8B0000", hover_color="#A52A2A", width=110)
        btn_reset.pack(side="left", padx=10)

        # In batch mode, add a prominent Load Settings button at the top
        if self.mode == "batch":
            batch_top_frame = ctk.CTkFrame(parent)
            # Pack this BEFORE all other frames by re-packing at the start
            batch_top_frame.pack(side="top", fill="x", pady=5, before=export_frame)
            ctk.CTkButton(
                batch_top_frame, text="⚙ Load Settings File",
                command=self.load_settings,
                fg_color="#1a6b3c", hover_color="#258c50",
                font=("Arial", 13, "bold"), height=35, width=200
            ).pack(side="left", padx=10)

        # ── Row 8: Shortcuts ───────────────────────────────────────────────────────
        if self.mode in ("individual", "ml"):
            shortcut_frame = ctk.CTkFrame(parent)
            shortcut_frame.pack(side="top", fill="x", pady=5)
            ctk.CTkLabel(shortcut_frame, text="Shortcuts:", font=("Arial", 10, "bold")).pack(side="left", padx=5)
            shortcuts = [
                ("Left/Right", "Prev/Next"),
                ("Plus", "Calculate HSV mask"),
                ("Minus", "Invert HSV mask"),
                ("F5", "Extract boundary"),
                ("F6", "Extract boundary w/ inverted mask"),
                ("Space", "Export Test"),
                ("Return", "Export Training data"),
            ]
            for i, (key, action) in enumerate(shortcuts):
                ctk.CTkLabel(shortcut_frame, text=key, font=("Arial", 10, "bold")).pack(side="left")
                ctk.CTkLabel(shortcut_frame, text=f" = {action}", font=("Arial", 10)).pack(side="left")
                if i < len(shortcuts) - 1:
                    ctk.CTkLabel(shortcut_frame, text=" || ", font=("Arial", 10)).pack(side="left")

        # Initial visibility/placement for dual sliders and ML rows
        self.toggle_dual_sliders()
        self.toggle_ml_mask_options()  # ensures ML rows stay directly under the import row


    # -------------- ML MASK TOGGLE & ACTIONS --------------

    def toggle_ml_mask_options(self):
        """Show/hide the ML mask rows directly under the import row."""
        self.ml_row.pack_forget()
        self.ml_row.pack(side="top", fill="x", pady=(2, 0), before=self.aoi_check_frame)
    
        self.ml_opts_row.pack_forget()
        if self.use_ml_pred_mask.get():
            self.ml_opts_row.pack(side="top", fill="x", pady=(2, 5), before=self.aoi_check_frame)


    def extract_boundary_universal(self):
        """Use the ML-mask workflow when enabled, otherwise use the normal boundary extraction."""
        if getattr(self, "use_ml_pred_mask", None) and self.use_ml_pred_mask.get():
            if self.mode == "individual" and self.ml_mask_file_path.get().strip():
                self.calculate_edge_with_ml_mask()
                return
            if self.mode == "ml" and self.ml_mask_folder_path.get().strip():
                self.calculate_edge_with_ml_mask()
                return
        self.extract_boundary()

    def extract_polygon_universal(self):
        """Use the ML-mask workflow when enabled, otherwise use the normal polygon extraction."""
        if getattr(self, "use_ml_pred_mask", None) and self.use_ml_pred_mask.get():
            if self.mode == "individual" and self.ml_mask_file_path.get().strip():
                self.extract_polygon_with_ml_mask()
                return
            if self.mode == "ml" and self.ml_mask_folder_path.get().strip():
                self.extract_polygon_with_ml_mask()
                return
        self.extract_polygon()


    def browse_ml_mask_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("Mask/Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff")])
        if path:
            self.ml_mask_file_path.set(path)
            self.ml_mask_file_disp.set(self._shorten_path(path))  # show short path

    def browse_ml_mask_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.ml_mask_folder_path.set(folder)
            self.ml_mask_folder_disp.set(self._shorten_path(folder))  # show short path

    # -------------- BBOX TOGGLE --------------

    def toggle_bbox_options(self):
        if self.use_bbox.get():
            self.bbox_entry.pack(side="left", padx=5)
            if not hasattr(self, 'select_bbox_button'):
                self.select_bbox_button = ctk.CTkButton(
                    self.bbox_frame, text="Select BBox on image", command=self.open_bbox_selector)
            self.select_bbox_button.pack(side="left", padx=5)
            self.inner_mask_check.configure(state="normal")
        else:
            self.bbox_entry.pack_forget()
            if hasattr(self, 'select_bbox_button'):
                self.select_bbox_button.pack_forget()
            self.inner_mask_check.configure(state="disabled")

    def open_bbox_selector(self):
        if self.full_image is None:
            messagebox.showwarning(
                "BBox Selector", "Please load an image first.")
            return
        if self.full_image.ndim == 3:
            pil_img = Image.fromarray(cv2.cvtColor(
                self.full_image, cv2.COLOR_BGR2RGB))
        else:
            pil_img = Image.fromarray(self.full_image)
        BBoxSelectorWindow(self, pil_img, self.set_bbox_from_selector)

    def set_bbox_from_selector(self, bbox):
        # 1) store the box
        self.bbox = bbox

        # 2) show it in the text entry
        self.bbox_entry.delete(0, tk.END)
        self.bbox_entry.insert(
            0, f"({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]})")

    # -------------- DUAL SLIDERS --------------

    def toggle_dual_sliders(self):
        if not hasattr(self, 'use_dual_hsv'):
            return
        if self.use_dual_hsv.get():
            self.dual_lower_frame.pack(side="left", fill="x", expand=True)
            self.dual_upper_frame.pack(side="left", fill="x", expand=True)
        else:
            self.dual_lower_frame.pack_forget()
            self.dual_upper_frame.pack_forget()

    # -------------- ADVANCED SETTINGS --------------

    def toggle_advanced_settings(self):
        if self.advanced_check_var.get():
            self.min_contour_label.pack(side="left", padx=2)
            self.min_contour_entry.pack(side="left", padx=2)
            self.max_contour_label.pack(side="left", padx=2)
            self.max_contour_entry.pack(side="left", padx=2)
        else:
            self.min_contour_label.pack_forget()
            self.min_contour_entry.pack_forget()
            self.max_contour_label.pack_forget()
            self.max_contour_entry.pack_forget()


    #  DROPDOWN 1: AOI / Profile-based Filter — toggle & actions
    # ────────────────────────────────────────────────────────────────────────────

    def _toggle_aoi_controls(self):
        if self.use_aoi_filter.get():
            try:
                self.aoi_opts_frame.pack(side="top", fill="x", pady=(2, 5),
                                         before=self.hsv_label_frame)
            except Exception:
                self.aoi_opts_frame.pack(side="top", fill="x", pady=(2, 5))
        else:
            self.aoi_opts_frame.pack_forget()

    def _toggle_hsv_controls(self):
        """Show/hide the HSV masking controls."""
        if self.use_hsv_masking.get():
            try:
                self.hsv_controls_container.pack(side="top", fill="x",
                                                  before=self.cpick_check_frame)
            except Exception:
                self.hsv_controls_container.pack(side="top", fill="x")
        else:
            self.hsv_controls_container.pack_forget()

    def _aoi_draw_on_image(self):
        """Open a click-to-draw popup to define the profile line."""
        if self.full_image is None:
            messagebox.showwarning("AOI", "Please load an image first.")
            return
        if self.full_image.ndim == 3:
            pil_img = Image.fromarray(cv2.cvtColor(self.full_image, cv2.COLOR_BGR2RGB))
        else:
            pil_img = Image.fromarray(self.full_image)

        self.aoi_click_points = []
        win = tk.Toplevel(self)
        win.title("Draw Profile Line — click start then end")
        win.geometry("800x600")
        canvas = tk.Canvas(win, cursor="cross")
        canvas.pack(fill="both", expand=True)

        max_w, max_h = 780, 560
        iw, ih = pil_img.size
        scale = min(max_w / iw, max_h / ih, 1.0)
        disp_w, disp_h = max(1, int(iw * scale)), max(1, int(ih * scale))
        disp_img = pil_img.resize((disp_w, disp_h), Image.BILINEAR)
        tk_img = ImageTk.PhotoImage(disp_img)
        canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
        canvas.image = tk_img
        line_id = [None]

        def on_click(event):
            import numpy as np
            x_orig = int(event.x / scale)
            y_orig = int(event.y / scale)
            self.aoi_click_points.append((x_orig, y_orig))
            r = 4
            canvas.create_oval(event.x - r, event.y - r,
                               event.x + r, event.y + r, fill="red", outline="red")
            if len(self.aoi_click_points) == 1:
                self.aoi_x1_entry.delete(0, tk.END)
                self.aoi_x1_entry.insert(0, str(x_orig))
                self.aoi_y1_entry.delete(0, tk.END)
                self.aoi_y1_entry.insert(0, str(y_orig))
            elif len(self.aoi_click_points) >= 2:
                p1, p2 = self.aoi_click_points[0], self.aoi_click_points[1]
                self.aoi_x2_entry.delete(0, tk.END)
                self.aoi_x2_entry.insert(0, str(p2[0]))
                self.aoi_y2_entry.delete(0, tk.END)
                self.aoi_y2_entry.insert(0, str(p2[1]))
                if line_id[0]:
                    canvas.delete(line_id[0])
                line_id[0] = canvas.create_line(
                    p1[0] * scale, p1[1] * scale,
                    p2[0] * scale, p2[1] * scale, fill="red", width=2)
                print(f"[AOI] Profile: ({p1[0]},{p1[1]}) → ({p2[0]},{p2[1]})  "
                      f"length={np.hypot(p2[0]-p1[0], p2[1]-p1[1]):.0f} px")
                win.after(600, win.destroy)

        canvas.bind("<Button-1>", on_click)

    def _aoi_preview_profile(self):
        """Show 1D intensity profile with threshold markers, zoom toolbar, and hover cursor."""
        import numpy as np
        if self.full_image is None:
            messagebox.showwarning("AOI", "No image loaded.")
            return
        try:
            x1 = int(self.aoi_x1_entry.get())
            y1 = int(self.aoi_y1_entry.get())
            x2 = int(self.aoi_x2_entry.get())
            y2 = int(self.aoi_y2_entry.get())
        except (ValueError, TypeError):
            messagebox.showwarning("AOI", "Define profile coordinates first.")
            return

        avg_w = max(1, int(self.aoi_width_entry.get() or "5"))
        gray = cv2.cvtColor(self.full_image, cv2.COLOR_BGR2GRAY) if self.full_image.ndim == 3 else self.full_image

        from feature_identifier import extract_profile
        profile = extract_profile(gray, x1, y1, x2, y2, width=avg_w)

        method = self.aoi_method.get()

        # Read current user entries — these are NEVER overwritten by preview
        user_min = self.aoi_min_entry.get().strip()
        user_max = self.aoi_max_entry.get().strip()
        has_user_values = (user_min not in ("", "0") or user_max not in ("", "255"))

        # Compute auto-suggestions based on method (for display in plot only)
        auto_min, auto_max = None, None
        if method == "Otsu":
            try:
                from skimage.filters import threshold_otsu
                thr = threshold_otsu(profile.astype(np.uint8))
                auto_min, auto_max = 0, int(thr)
            except Exception:
                auto_min, auto_max = 0, 128
        elif method == "Variance":
            win_size = max(5, len(profile) // 20)
            local_var = np.array([np.var(profile[max(0, i-win_size):i+win_size])
                                  for i in range(len(profile))])
            thr = np.percentile(local_var, 70)
            above = np.where(local_var > thr)[0]
            if len(above) > 0:
                auto_min, auto_max = int(profile[above].min()), int(profile[above].max())
            else:
                auto_min, auto_max = 0, 255

        # Only auto-fill entries if user hasn't set custom values
        if auto_min is not None and not has_user_values:
            self.aoi_min_entry.delete(0, tk.END)
            self.aoi_min_entry.insert(0, str(auto_min))
            self.aoi_max_entry.delete(0, tk.END)
            self.aoi_max_entry.insert(0, str(auto_max))

        # Final values for the plot: always use whatever is in the entries right now
        plot_min = int(self.aoi_min_entry.get() or "0")
        plot_max = int(self.aoi_max_entry.get() or "255")

        # Build popup with embedded figure — use Figure() directly to avoid plt ghost window
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

        popup = tk.Toplevel(self)
        popup.title("Profile Preview — AOI Filter")
        popup.geometry("750x450")

        fig = Figure(figsize=(8, 3.5), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(profile, color="steelblue", linewidth=0.8, label="Intensity")
        ax.axhline(plot_min, color="red", linestyle="--", linewidth=1, label=f"Min: {plot_min}")
        ax.axhline(plot_max, color="green", linestyle="--", linewidth=1, label=f"Max: {plot_max}")
        ax.fill_between(range(len(profile)), plot_min, plot_max,
                        alpha=0.15, color="orange", label="Your AOI band")

        # Show auto-suggestion as thin dotted lines if they differ from user values
        if auto_min is not None and (auto_min != plot_min or auto_max != plot_max):
            ax.axhline(auto_min, color="red", linestyle=":", linewidth=0.7, alpha=0.5,
                       label=f"Auto min: {auto_min}")
            ax.axhline(auto_max, color="green", linestyle=":", linewidth=0.7, alpha=0.5,
                       label=f"Auto max: {auto_max}")

        ax.set_xlabel("Distance along profile (px)")
        ax.set_ylabel("Intensity")
        ax.set_title(f"Profile ({method}) — adjust Min/Max entries and click Apply AOI")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        canvas_widget = FigureCanvasTkAgg(fig, master=popup)
        canvas_widget.get_tk_widget().pack(fill="both", expand=True)

        # Zoom/pan toolbar
        toolbar_frame = tk.Frame(popup)
        toolbar_frame.pack(fill="x")
        toolbar = NavigationToolbar2Tk(canvas_widget, toolbar_frame)
        toolbar.update()

        # Hover cursor — show pixel index and intensity on mouseover
        annot = ax.annotate("", xy=(0, 0), xytext=(15, 15),
                            textcoords="offset points",
                            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.9),
                            fontsize=8, visible=False)

        def on_hover(event):
            if event.inaxes != ax:
                annot.set_visible(False)
                canvas_widget.draw_idle()
                return
            xi = int(round(event.xdata))
            if 0 <= xi < len(profile):
                val = profile[xi]
                annot.xy = (xi, val)
                annot.set_text(f"px: {xi}\nintensity: {val:.1f}")
                annot.set_visible(True)
                canvas_widget.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", on_hover)
        canvas_widget.draw()

        auto_info = ""
        if auto_min is not None and (auto_min != plot_min or auto_max != plot_max):
            auto_info = f"  (auto-suggestion: [{auto_min}, {auto_max}])"
        print(f"[AOI] Profile preview: {len(profile)} points, "
              f"your range [{plot_min}, {plot_max}] using {method}{auto_info}")

    def _aoi_apply_filter(self):
        """Build the AOI mask from the profile threshold range or polygon."""
        import numpy as np
        if self.full_image is None:
            messagebox.showwarning("AOI", "No image loaded.")
            return

        # Check if we have a polygon AOI (drawn or from text entry)
        aoi_poly = getattr(self, '_aoi_polygon_pts', None)
        if (aoi_poly is None or len(aoi_poly) < 3) and hasattr(self, 'aoi_polygon_entry'):
            # Try parsing from text entry: "x1,y1; x2,y2; ..."
            poly_text = self.aoi_polygon_entry.get().strip()
            if poly_text:
                try:
                    aoi_poly = []
                    for pair in poly_text.split(";"):
                        x, y = pair.strip().split(",")
                        aoi_poly.append((int(x.strip()), int(y.strip())))
                    self._aoi_polygon_pts = aoi_poly
                except Exception:
                    aoi_poly = None

        if aoi_poly is not None and len(aoi_poly) >= 3:
            h, w = self.full_image.shape[:2]
            self.aoi_mask = np.zeros((h, w), dtype=np.uint8)
            pts = np.array(aoi_poly, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(self.aoi_mask, [pts], 255)
            print(f"[AOI] Polygon mask applied: {len(aoi_poly)} vertices")
        else:
            # Threshold-based AOI
            try:
                min_val = int(self.aoi_min_entry.get())
                max_val = int(self.aoi_max_entry.get())
            except (ValueError, TypeError):
                messagebox.showwarning("AOI", "Set Min and Max values first (use Preview Profile).")
                return

            gray = cv2.cvtColor(self.full_image, cv2.COLOR_BGR2GRAY) if self.full_image.ndim == 3 else self.full_image.copy()
            self.aoi_mask = cv2.inRange(gray, min_val, max_val)
            print(f"[AOI] Threshold mask applied: range [{min_val}, {max_val}]")

        if self.use_bbox.get():
            try:
                bx, by, bw, bh = map(int, self.bbox_entry.get().strip("()").split(","))
                bbox_m = np.zeros_like(self.aoi_mask)
                bbox_m[by:by+bh, bx:bx+bw] = 255
                self.aoi_mask = cv2.bitwise_and(self.aoi_mask, bbox_m)
            except Exception:
                pass

        nz = cv2.countNonZero(self.aoi_mask)
        total = self.aoi_mask.shape[0] * self.aoi_mask.shape[1]
        print(f"[AOI] {nz}/{total} pixels selected ({100*nz/total:.1f}%)")

        # Show AOI preview: original RGB with outside-AOI dimmed
        self._display_aoi_preview()

    def _display_aoi_preview(self):
        """Display the AOI as a dimmed overlay of the original image in the mask panel."""
        import numpy as np
        if self.full_image is None or self.aoi_mask is None:
            return
        if not hasattr(self, 'mask_label') or not self.mask_label.winfo_exists():
            return

        if self.full_image.ndim == 3:
            img_rgb = cv2.cvtColor(self.full_image, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = cv2.cvtColor(self.full_image, cv2.COLOR_GRAY2RGB)

        aoi = self.aoi_mask
        if aoi.shape[:2] != img_rgb.shape[:2]:
            aoi = cv2.resize(aoi, (img_rgb.shape[1], img_rgb.shape[0]),
                             interpolation=cv2.INTER_NEAREST)

        # Dim outside, keep inside bright, draw green boundary
        dimmed = (img_rgb.astype(np.float32) * 0.25).astype(np.uint8)
        display = np.where(aoi[:, :, np.newaxis] > 0, img_rgb, dimmed)
        contours, _ = cv2.findContours(aoi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(display, contours, -1, (0, 255, 0), 2)

        # Resize to mask panel size
        try:
            pw = self.top_center_frame.winfo_width()
            ph = self.top_center_frame.winfo_height()
            if pw > 50 and ph > 50:
                display = cv2.resize(display, (pw, ph), interpolation=cv2.INTER_AREA)
        except Exception:
            pass

        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(display)
        import customtkinter as ctk
        ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img,
                               size=(pil_img.width, pil_img.height))
        self.mask_label.configure(image=ctk_img)
        self.mask_label.image = ctk_img

    def _aoi_draw_polygon(self):
        """Open a popup where the user clicks vertices to define a polygon AOI."""
        import numpy as np
        if self.full_image is None:
            messagebox.showwarning("AOI", "Please load an image first.")
            return

        if self.full_image.ndim == 3:
            pil_img = Image.fromarray(cv2.cvtColor(self.full_image, cv2.COLOR_BGR2RGB))
        else:
            pil_img = Image.fromarray(self.full_image)

        self._aoi_polygon_pts = []

        win = tk.Toplevel(self)
        win.title("Draw AOI Polygon — click vertices, double-click to close")
        win.geometry("900x650")

        canvas = tk.Canvas(win, cursor="cross")
        canvas.pack(fill="both", expand=True)

        # Info bar
        info_frame = tk.Frame(win)
        info_frame.pack(fill="x")
        info_label = tk.Label(info_frame, text="Click to add vertices. Double-click to close polygon.")
        info_label.pack(side="left", padx=10)
        coord_label = tk.Label(info_frame, text="Vertices: 0")
        coord_label.pack(side="right", padx=10)

        max_w, max_h = 880, 590
        iw, ih = pil_img.size
        scale = min(max_w / iw, max_h / ih, 1.0)
        disp_w, disp_h = max(1, int(iw * scale)), max(1, int(ih * scale))
        disp_img = pil_img.resize((disp_w, disp_h), Image.BILINEAR)
        tk_img = ImageTk.PhotoImage(disp_img)
        canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
        canvas.image = tk_img

        line_ids = []

        def on_click(event):
            x_orig = int(event.x / scale)
            y_orig = int(event.y / scale)
            self._aoi_polygon_pts.append((x_orig, y_orig))
            r = 3
            canvas.create_oval(event.x - r, event.y - r,
                               event.x + r, event.y + r, fill="lime", outline="lime")
            pts = self._aoi_polygon_pts
            if len(pts) >= 2:
                p1, p2 = pts[-2], pts[-1]
                lid = canvas.create_line(
                    p1[0] * scale, p1[1] * scale,
                    p2[0] * scale, p2[1] * scale,
                    fill="lime", width=2)
                line_ids.append(lid)
            coord_label.config(text=f"Vertices: {len(pts)}")

        def on_double_click(event):
            pts = self._aoi_polygon_pts
            if len(pts) >= 3:
                # Close polygon visually
                p1, p2 = pts[-1], pts[0]
                canvas.create_line(
                    p1[0] * scale, p1[1] * scale,
                    p2[0] * scale, p2[1] * scale,
                    fill="lime", width=2, dash=(4, 2))
                # Store polygon coords as text for saving
                coord_str = "; ".join(f"{x},{y}" for x, y in pts)
                if hasattr(self, 'aoi_polygon_entry'):
                    self.aoi_polygon_entry.delete(0, tk.END)
                    self.aoi_polygon_entry.insert(0, coord_str)
                print(f"[AOI] Polygon closed: {len(pts)} vertices")
                win.after(500, win.destroy)

        canvas.bind("<Button-1>", on_click)
        canvas.bind("<Double-Button-1>", on_double_click)

    def _aoi_clear(self):
        self.aoi_mask = None
        self._aoi_polygon_pts = None
        if hasattr(self, 'aoi_polygon_entry'):
            self.aoi_polygon_entry.delete(0, tk.END)
        # Restore normal mask display if possible
        if hasattr(self, 'current_mask') and self.current_mask is not None:
            self.display_mask()
        print("[AOI] Filter cleared.")


    #  DROPDOWN 3: Colour Picker — toggle & actions
    # ────────────────────────────────────────────────────────────────────────────

    def _toggle_color_picker_controls(self):
        if self.use_color_picker.get():
            try:
                self.cpick_opts_frame.pack(side="top", fill="x", pady=(2, 5),
                                           before=self.step4_label_frame)
            except Exception:
                self.cpick_opts_frame.pack(side="top", fill="x", pady=(2, 5))
            self._update_color_pick_labels()
        else:
            self.cpick_opts_frame.pack_forget()

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

    def _format_color_pick_group(self, label):
        pts = self._normalise_color_pick_points().get(label, [])
        prefix = "Remove" if label == "remove" else "Keep"
        if not pts:
            return f"{prefix}: 0 pts"
        preview = ", ".join(f"({x},{y})" for x, y in pts[:3])
        if len(pts) > 3:
            preview += ", …"
        return f"{prefix}: {len(pts)} pts  {preview}"

    def _update_color_pick_labels(self):
        self._normalise_color_pick_points()
        if hasattr(self, "cpick_remove_label"):
            self.cpick_remove_label.configure(text=self._format_color_pick_group("remove"))
        if hasattr(self, "cpick_keep_label"):
            self.cpick_keep_label.configure(text=self._format_color_pick_group("keep"))

    def _start_color_pick(self, label):
        """Open a click popup to add one or more colour sample points.
        If AOI is active, dims pixels outside the AOI and rejects clicks there."""
        import numpy as np

        if self.full_image is None:
            messagebox.showwarning("Color Picker", "Load an image first.")
            return

        self._normalise_color_pick_points()

        aoi_active = (getattr(self, 'use_aoi_filter', None)
                      and self.use_aoi_filter.get()
                      and getattr(self, 'aoi_mask', None) is not None)

        if self.full_image.ndim == 3:
            img_rgb = cv2.cvtColor(self.full_image, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = cv2.cvtColor(self.full_image, cv2.COLOR_GRAY2RGB)

        if aoi_active:
            aoi = self.aoi_mask
            if aoi.shape[:2] != img_rgb.shape[:2]:
                aoi = cv2.resize(aoi, (img_rgb.shape[1], img_rgb.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
            dimmed = (img_rgb.astype(np.float32) * 0.3).astype(np.uint8)
            display_rgb = np.where(aoi[:, :, np.newaxis] > 0, img_rgb, dimmed)
            contours, _ = cv2.findContours(aoi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(display_rgb, contours, -1, (0, 255, 0), 2)
        else:
            display_rgb = img_rgb

        pil_img = Image.fromarray(display_rgb)

        is_remove = (label == "remove")
        display_name = "Remove Samples" if is_remove else "Keep Samples"
        marker_colour = "orange" if is_remove else "cyan"
        title_suffix = " (pick within highlighted AOI)" if aoi_active else ""

        win = tk.Toplevel(self)
        win.title(f"Click to add {display_name}{title_suffix}")
        win.geometry("820x680")

        # Top bar with info + undo button
        top_bar = tk.Frame(win)
        top_bar.pack(fill="x", padx=8, pady=(8, 2))
        info_label = tk.Label(
            top_bar,
            text="Left-click to add samples. Right-click or Escape to finish.",
            anchor="w"
        )
        info_label.pack(side="left", fill="x", expand=True)

        # Track canvas marker IDs so undo can remove them visually
        marker_ids = []

        def undo_last_in_window():
            pts = self.color_pick_points.get(label, [])
            if not pts:
                print(f"[Color Picker] No {display_name} to undo.")
                return
            removed_pt = pts.pop()
            # Remove the visual marker from canvas
            if marker_ids:
                old_id = marker_ids.pop()
                try:
                    canvas.delete(old_id)
                except Exception:
                    pass
            self._update_color_pick_labels()
            coord_label.configure(text=self._format_color_pick_group(label))
            print(f"[Color Picker] Undid last {display_name[:-1]}: "
                  f"({removed_pt[0]}, {removed_pt[1]}). Remaining: {len(pts)} pts.")

        undo_btn = tk.Button(top_bar, text="Undo Last", bg="#8B0000", fg="white",
                             command=undo_last_in_window)
        undo_btn.pack(side="right", padx=5)

        coord_label = tk.Label(win, text=self._format_color_pick_group(label), anchor="w")
        coord_label.pack(fill="x", padx=8, pady=(0, 6))

        canvas = tk.Canvas(win, cursor="cross")
        canvas.pack(fill="both", expand=True)

        max_w, max_h = 790, 560
        iw, ih = pil_img.size
        scale = min(max_w / iw, max_h / ih, 1.0)
        disp_w, disp_h = max(1, int(iw * scale)), max(1, int(ih * scale))
        disp_img = pil_img.resize((disp_w, disp_h), Image.BILINEAR)
        tk_img = ImageTk.PhotoImage(disp_img)
        canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
        canvas.image = tk_img

        def draw_marker(x_orig, y_orig, colour, radius=4, track=False):
            x_disp = x_orig * scale
            y_disp = y_orig * scale
            oval_id = canvas.create_oval(
                x_disp - radius, y_disp - radius,
                x_disp + radius, y_disp + radius,
                fill=colour, outline="black", width=1
            )
            if track:
                marker_ids.append(oval_id)
            return oval_id

        # show existing samples (not tracked — they were added in previous sessions)
        for x0, y0 in self.color_pick_points.get("remove", []):
            draw_marker(x0, y0, "orange", track=(label == "remove"))
        for x0, y0 in self.color_pick_points.get("keep", []):
            draw_marker(x0, y0, "cyan", track=(label == "keep"))

        def finish(_event=None):
            self._update_color_pick_labels()
            win.destroy()

        def on_click(event):
            x_orig = int(event.x / scale)
            y_orig = int(event.y / scale)
            h, w = self.full_image.shape[:2]
            x_orig = max(0, min(x_orig, w - 1))
            y_orig = max(0, min(y_orig, h - 1))

            if aoi_active:
                aoi_check = self.aoi_mask
                if aoi_check.shape[:2] != (h, w):
                    aoi_check = cv2.resize(aoi_check, (w, h),
                                           interpolation=cv2.INTER_NEAREST)
                if aoi_check[y_orig, x_orig] == 0:
                    print(f"[Color Picker] Click at ({x_orig}, {y_orig}) is outside the AOI — ignored.")
                    oval = canvas.create_oval(event.x - 4, event.y - 4,
                                              event.x + 4, event.y + 4,
                                              fill="red", outline="red")
                    canvas.after(400, lambda: canvas.delete(oval))
                    return

            self.color_pick_points.setdefault(label, [])
            self.color_pick_points[label].append((x_orig, y_orig))

            if self.full_image.ndim == 3:
                bgr = self.full_image[y_orig, x_orig]
                col_str = f"RGB=({bgr[2]},{bgr[1]},{bgr[0]})"
            else:
                col_str = f"Value={self.full_image[y_orig, x_orig]}"

            draw_marker(x_orig, y_orig, marker_colour, track=True)
            self._update_color_pick_labels()
            coord_label.configure(text=self._format_color_pick_group(label))
            print(f"[Color Picker] {display_name}: ({x_orig}, {y_orig}) {col_str}")

        canvas.bind("<Button-1>", on_click)
        canvas.bind("<Button-3>", finish)
        win.bind("<Return>", finish)
        win.bind("<Escape>", finish)

    def _color_pick_clear(self):
        self.color_pick_points = {"remove": [], "keep": []}
        self.color_pick_mask = None
        self._update_color_pick_labels()
        print("[Color Picker] Cleared all samples.")

    def _color_pick_undo_last(self, label):
        """Remove the last sample point from the given group ('remove' or 'keep')."""
        self._normalise_color_pick_points()
        pts = self.color_pick_points.get(label, [])
        if not pts:
            display_name = "Remove" if label == "remove" else "Keep"
            print(f"[Color Picker] No {display_name} samples to undo.")
            return
        removed_pt = pts.pop()
        self._update_color_pick_labels()
        display_name = "Remove" if label == "remove" else "Keep"
        print(f"[Color Picker] Undid last {display_name} sample: ({removed_pt[0]}, {removed_pt[1]}). "
              f"Remaining: {len(pts)} pts.")

    # -------------- SETTINGS --------------

    def make_slider_with_label(self, parent, text, minv, maxv, init):
        frame = ctk.CTkFrame(parent)
        frame.pack(side="left", padx=2)
        lbl_title = ctk.CTkLabel(frame, text=text)
        lbl_title.pack(side="top")
        value_var = tk.IntVar(master=self, value=init)
        lbl_value = ctk.CTkLabel(frame, textvariable=value_var)
        lbl_value.pack(side="top")
        sld = ctk.CTkSlider(frame, from_=minv, to=maxv, width=120, number_of_steps=(maxv - minv),
                            command=lambda val: value_var.set(int(float(val))))
        sld.set(init)
        sld.pack(side="top")
        return sld, value_var, lbl_value

    def browse_export_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.export_path_entry.delete(0, tk.END)
            self.export_path_entry.insert(0, folder)

    def save_settings(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        if not file_path:
            return
        data = {
            "bbox_text": self.bbox_entry.get(),
            "enable_enhancements": bool(self.enable_enhancements.get()),
            "s_multiplier": int(self.s_multiplier_slider.get()),
            "v_multiplier": int(self.v_multiplier_slider.get()),
            "h_low": int(self.h_low_slider.get()),
            "s_low": int(self.s_low_slider.get()),
            "v_low": int(self.v_low_slider.get()),
            "h_high": int(self.h_high_slider.get()),
            "s_high": int(self.s_high_slider.get()),
            "v_high": int(self.v_high_slider.get()),
            "use_dual_hsv": bool(self.use_dual_hsv.get()),
            "h2_low": int(self.h2_low_slider.get()),
            "s2_low": int(self.s2_low_slider.get()),
            "v2_low": int(self.v2_low_slider.get()),
            "h2_high": int(self.h2_high_slider.get()),
            "s2_high": int(self.s2_high_slider.get()),
            "v2_high": int(self.v2_high_slider.get()),
            "edge_thickness": int(self.edge_thickness_slider.get()),
            "export_path": self.export_path_entry.get(),
            "do_invert_mask": bool(self.do_invert_mask.get()),
            "advanced_settings_enabled": self.advanced_check_var.get(),
            "min_contour_size": self.min_contour_entry.get(),
            "max_contour_size": self.max_contour_entry.get(),
            # New ML mask settings
            "use_ml_pred_mask": bool(self.use_ml_pred_mask.get()),
            "ml_mask_file_path": self.ml_mask_file_path.get(),
            "ml_mask_folder_path": self.ml_mask_folder_path.get(),
            "common_name_length": self.common_name_len_var.get(),
            # AOI Filter settings
            "use_aoi_filter": bool(self.use_aoi_filter.get()),
            "aoi_method": self.aoi_method.get(),
            "aoi_x1": self.aoi_x1_entry.get() if hasattr(self, 'aoi_x1_entry') else "",
            "aoi_y1": self.aoi_y1_entry.get() if hasattr(self, 'aoi_y1_entry') else "",
            "aoi_x2": self.aoi_x2_entry.get() if hasattr(self, 'aoi_x2_entry') else "",
            "aoi_y2": self.aoi_y2_entry.get() if hasattr(self, 'aoi_y2_entry') else "",
            "aoi_width": self.aoi_width_entry.get() if hasattr(self, 'aoi_width_entry') else "5",
            "aoi_min": self.aoi_min_entry.get() if hasattr(self, 'aoi_min_entry') else "0",
            "aoi_max": self.aoi_max_entry.get() if hasattr(self, 'aoi_max_entry') else "255",
            "aoi_polygon": self.aoi_polygon_entry.get() if hasattr(self, 'aoi_polygon_entry') else "",
            # HSV masking toggle
            "use_hsv_masking": bool(self.use_hsv_masking.get()),
            # Color picker settings
            "use_color_picker": bool(self.use_color_picker.get()),
            "color_pick_method": self.color_pick_method.get(),
            "color_pick_output_mode": self.color_pick_output_mode.get(),
            "color_pick_patch_radius": self.color_pick_patch_radius.get(),
            "color_pick_remove_points": [list(pt) for pt in self._normalise_color_pick_points().get("remove", [])],
            "color_pick_keep_points": [list(pt) for pt in self._normalise_color_pick_points().get("keep", [])],
            "color_pick_class_a_name": self.color_pick_class_a_name.get(),
            "color_pick_class_b_name": self.color_pick_class_b_name.get(),
        }
        try:
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)
            messagebox.showinfo(
                "Save Settings", f"Settings saved to {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {e}")

    def load_settings(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        if not file_path:
            return
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load settings: {e}")
            return

        self.bbox_entry.delete(0, "end")
        self.bbox_entry.insert(0, data.get("bbox_text", ""))
        self.enable_enhancements.set(data.get("enable_enhancements", True))
        self.s_multiplier_slider.set(data.get("s_multiplier", 100))
        self.v_multiplier_slider.set(data.get("v_multiplier", 100))

        low_h = data.get("h_low", 0)
        self.h_low_slider.set(low_h)
        # note: value vars update via slider callbacks
        low_s = data.get("s_low", 0)
        self.s_low_slider.set(low_s)
        low_v = data.get("v_low", 0)
        self.v_low_slider.set(low_v)

        high_h = data.get("h_high", 255)
        self.h_high_slider.set(high_h)
        high_s = data.get("s_high", 255)
        self.s_high_slider.set(high_s)
        high_v = data.get("v_high", 255)
        self.v_high_slider.set(high_v)

        self.use_dual_hsv.set(data.get("use_dual_hsv", False))

        low_h2 = data.get("h2_low", 0)
        self.h2_low_slider.set(low_h2)
        low_s2 = data.get("s2_low", 0)
        self.s2_low_slider.set(low_s2)
        low_v2 = data.get("v2_low", 0)
        self.v2_low_slider.set(low_v2)
        high_h2 = data.get("h2_high", 255)
        self.h2_high_slider.set(high_h2)
        high_s2 = data.get("s2_high", 255)
        self.s2_high_slider.set(high_s2)
        high_v2 = data.get("v2_high", 255)
        self.v2_high_slider.set(high_v2)

        edge_thickness = data.get("edge_thickness", 2)
        self.edge_thickness_slider.set(edge_thickness)
        self.thickness_value_label.configure(text=str(edge_thickness))

        self.export_path_entry.delete(0, "end")
        self.export_path_entry.insert(0, data.get("export_path", ""))

        self.do_invert_mask.set(data.get("do_invert_mask", False))
        self.toggle_dual_sliders()

        self.advanced_check_var.set(
            data.get("advanced_settings_enabled", False))
        self.min_contour_entry.delete(0, "end")
        self.min_contour_entry.insert(0, data.get("min_contour_size", ""))
        self.max_contour_entry.delete(0, "end")
        self.max_contour_entry.insert(0, data.get("max_contour_size", ""))
        self.toggle_advanced_settings()

        # Load ML mask settings
        self.use_ml_pred_mask.set(data.get("use_ml_pred_mask", False))
        self.ml_mask_file_path.set(data.get("ml_mask_file_path", ""))
        self.ml_mask_folder_path.set(data.get("ml_mask_folder_path", ""))
        self.common_name_len_var.set(data.get("common_name_length", ""))

        # refresh label display text
        self.ml_mask_file_disp.set(self._shorten_path(self.ml_mask_file_path.get()))
        self.ml_mask_folder_disp.set(self._shorten_path(self.ml_mask_folder_path.get()))


        # reflect rows visibility
        self.toggle_ml_mask_options()

        # Load AOI filter settings
        self.use_aoi_filter.set(data.get("use_aoi_filter", False))
        self.aoi_method.set(data.get("aoi_method", "Threshold"))
        if hasattr(self, 'aoi_x1_entry'):
            for entry, key, default in [
                (self.aoi_x1_entry, "aoi_x1", ""),
                (self.aoi_y1_entry, "aoi_y1", ""),
                (self.aoi_x2_entry, "aoi_x2", ""),
                (self.aoi_y2_entry, "aoi_y2", ""),
                (self.aoi_width_entry, "aoi_width", "5"),
                (self.aoi_min_entry, "aoi_min", "0"),
                (self.aoi_max_entry, "aoi_max", "255"),
            ]:
                entry.delete(0, "end")
                val = data.get(key, default)
                if val:
                    entry.insert(0, str(val))
        # Restore polygon AOI coords
        if hasattr(self, 'aoi_polygon_entry'):
            self.aoi_polygon_entry.delete(0, "end")
            poly_text = data.get("aoi_polygon", "")
            if poly_text:
                self.aoi_polygon_entry.insert(0, poly_text)
                # Rebuild internal polygon points from text
                try:
                    self._aoi_polygon_pts = []
                    for pair in poly_text.split(";"):
                        x, y = pair.strip().split(",")
                        self._aoi_polygon_pts.append((int(x.strip()), int(y.strip())))
                except Exception:
                    self._aoi_polygon_pts = None
            else:
                self._aoi_polygon_pts = None
        self._toggle_aoi_controls()

        # Load HSV masking toggle
        self.use_hsv_masking.set(data.get("use_hsv_masking", True))
        if hasattr(self, '_toggle_hsv_controls'):
            self._toggle_hsv_controls()

        # Load color picker settings
        self.use_color_picker.set(data.get("use_color_picker", False))
        self.color_pick_method.set(data.get("color_pick_method", "Color Distance"))
        # Backward-compat: map old terminology to new
        _output_mode_raw = data.get("color_pick_output_mode", "Remove selection")
        _output_mode_map = {
            "Remove selected class": "Remove selection",
            "Keep selected class": "Keep selection only",
        }
        self.color_pick_output_mode.set(_output_mode_map.get(_output_mode_raw, _output_mode_raw))
        self.color_pick_patch_radius.set(str(data.get("color_pick_patch_radius", "7")))

        remove_pts = data.get("color_pick_remove_points", [])
        keep_pts = data.get("color_pick_keep_points", [])
        if not remove_pts and not keep_pts:
            legacy_a = data.get("color_pick_point_a", data.get("point_a"))
            legacy_b = data.get("color_pick_point_b", data.get("point_b"))
            self.color_pick_points = {
                "remove": [tuple(legacy_b)] if isinstance(legacy_b, (list, tuple)) and len(legacy_b) >= 2 else [],
                "keep": [tuple(legacy_a)] if isinstance(legacy_a, (list, tuple)) and len(legacy_a) >= 2 else [],
            }
            if legacy_a is not None or legacy_b is not None:
                self.color_pick_output_mode.set("Keep selection only")
        else:
            self.color_pick_points = {
                "remove": [tuple(pt) for pt in remove_pts if isinstance(pt, (list, tuple)) and len(pt) >= 2],
                "keep": [tuple(pt) for pt in keep_pts if isinstance(pt, (list, tuple)) and len(pt) >= 2],
            }
        self._toggle_color_picker_controls()
        self._update_color_pick_labels()

        # Load class names
        self.color_pick_class_a_name.set(data.get("color_pick_class_a_name", "water"))
        self.color_pick_class_b_name.set(data.get("color_pick_class_b_name", "sand"))

        # Mark settings as loaded (important for batch mode)
        self._batch_settings_loaded = True
        if hasattr(self, 'settings_summary_label') and self.mode == "batch":
            self._update_batch_settings_summary(data)

        messagebox.showinfo(
            "Load Settings", f"Settings loaded from {file_path}")

    def _update_batch_settings_summary(self, data):
        """Update the batch mode summary label showing which pipeline steps are active."""
        lines = ["Settings loaded — active pipeline:"]
        if data.get("use_aoi_filter", False):
            poly = data.get("aoi_polygon", "")
            if poly:
                n_verts = len(poly.split(";"))
                lines.append(f"  ✓ AOI / Profile Filter: polygon with {n_verts} vertices")
            else:
                lines.append(f"  ✓ AOI / Profile Filter: method={data.get('aoi_method','?')}, "
                             f"range=[{data.get('aoi_min','?')}, {data.get('aoi_max','?')}]")
        else:
            lines.append("  ✗ AOI / Profile Filter: disabled")
        if data.get("use_hsv_masking", True):
            lines.append(f"  ✓ HSV Masking: H=[{data.get('h_low',0)}–{data.get('h_high',255)}] "
                         f"S=[{data.get('s_low',0)}–{data.get('s_high',255)}] "
                         f"V=[{data.get('v_low',0)}–{data.get('v_high',255)}]"
                         f"  invert={data.get('do_invert_mask', False)}")
        else:
            lines.append("  ✗ HSV Masking: disabled")
        if data.get("use_color_picker", False):
            n_remove = len(data.get("color_pick_remove_points", []))
            n_keep = len(data.get("color_pick_keep_points", []))
            lines.append(
                f"  ✓ Multi-sample Colour Selection: method={data.get('color_pick_method','?')}, "
                f"output={data.get('color_pick_output_mode', '?')}, "
                f"remove={n_remove}, keep={n_keep}"
            )
        else:
            lines.append("  ✗ Multi-sample Colour Selection: disabled")
        if data.get("use_ml_pred_mask", False):
            lines.append(f"  ✓ ML Mask: folder={data.get('ml_mask_folder_path','?')}")
        lines.append("  ✓ Boundary extraction: automatic — no manual editing in batch")
        try:
            self.settings_summary_label.configure(text="\n".join(lines))
        except Exception:
            pass