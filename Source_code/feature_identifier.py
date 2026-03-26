"""
feature_identifier.py
─────────────────────
Orchestrator module for the Feature Identifier tool (formerly HSV Mask Tool).

The class is composed from three mixin modules:

  • hsv_mask_ui.py          – UI construction, control panel, settings, toggles
  • hsv_mask_processing.py  – Image I/O, mask calculation, edge detection, batch
  • hsv_mask_editing.py     – Feature editing, canvas interaction, zoom/pan, export

Detection workflow:
  AOI / Profile-based region filtering
  HSV colour masking
  Multi-sample colour class selection
  Boundary / polygon extraction and manual editing
"""

import os
import sys
import tkinter as tk
from tkinter import scrolledtext
import customtkinter as ctk
from PIL import Image
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

# Import profile extraction from the profile tool (used for AOI filtering)
try:
    from profile_tool import extract_profile
except ImportError:
    # Fallback: inline minimal version if profile_tool is not available
    def extract_profile(img, x1, y1, x2, y2, width=1):
        dx, dy = x2 - x1, y2 - y1
        length = max(int(np.hypot(dx, dy)), 1)
        ux, uy = dx / length, dy / length
        px, py = -uy, ux
        t = np.arange(length)
        cx, cy = x1 + t * ux, y1 + t * uy
        half_w = (width - 1) / 2.0
        offsets = np.linspace(-half_w, half_w, max(width, 1))
        h, w = img.shape[:2]
        is_rgb = img.ndim == 3
        profile = np.zeros((length, 3) if is_rgb else length, dtype=np.float64)
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

# Theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")

# %% Import mixin classes and helpers
from hsv_mask_ui import BBoxSelectorWindow, StdoutRedirector, HSVMaskUIMixin
from hsv_mask_processing import HSVMaskProcessingMixin
from hsv_mask_editing import HSVMaskEditingMixin


class FeatureIdentifier(HSVMaskEditingMixin, HSVMaskProcessingMixin, HSVMaskUIMixin, ctk.CTkToplevel):
    """
    Feature Identifier tool — modular detection workflow.

    AOI / Profile filter        (narrow search region)
    HSV colour masking          (within AOI if active)
    Colour picker detection     (multi-sample remove / keep class selection)
    Boundary / polygon tools    (polyline / polygon / freehand editing)

    3 modes: individual, ml, batch

    Behaviour is composed from three mixin classes:
      HSVMaskUIMixin          – UI layout & settings
      HSVMaskProcessingMixin  – image processing & batch
      HSVMaskEditingMixin     – feature editing, canvas, export
    """

    @staticmethod
    def resource_path(relative_path: str) -> str:
        try:
            base_path = sys._MEIPASS  # If running in a PyInstaller .exe
        except Exception:
            # Running directly from source
            base_path = os.path.dirname(__file__)
        return os.path.join(base_path, relative_path)

    def __init__(self, master=None, mode="individual", *args, **kwargs):

        super().__init__(master=master, *args, **kwargs)

        self.mode = mode
        ctk.set_widget_scaling(1)
        try:
            self.iconbitmap(self.resource_path("launch_logo.ico"))
        except Exception:
            pass

        # --- State variables ---
        self.do_invert_mask = tk.BooleanVar(master=self, value=False)
        self.use_bbox = tk.BooleanVar(master=self, value=False)
        self.use_inner_mask = tk.BooleanVar(master=self, value=False)
        self.enable_enhancements = tk.BooleanVar(master=self, value=True)
        self.advanced_check_var = tk.BooleanVar(master=self, value=False)
        self.use_dual_hsv = tk.BooleanVar(master=self, value=False)

        # >>> New: ML predicted mask controls <<<
        self.use_ml_pred_mask = tk.BooleanVar(master=self, value=False)
        self.ml_mask_file_path = tk.StringVar(master=self, value="")     # for individual
        self.ml_mask_folder_path = tk.StringVar(master=self, value="")   # for ml/folder
        self.common_name_len_var = tk.StringVar(master=self, value="")   # for ml/folder
        # display-only (shortened) text for the UI labels
        self.ml_mask_file_disp = tk.StringVar(master=self, value="")
        self.ml_mask_folder_disp = tk.StringVar(master=self, value="")

        # >>> Dropdown 1: AOI / Profile filter <<<
        self.use_aoi_filter = tk.BooleanVar(master=self, value=False)
        self.aoi_mask = None           # binary mask (0/255) limiting detection region
        self.aoi_method = tk.StringVar(master=self, value="Threshold")
        self.aoi_click_points = []     # [(x,y), (x,y)] for profile line

        # >>> Dropdown 2: HSV masking (enabled by default) <<<
        self.use_hsv_masking = tk.BooleanVar(master=self, value=True)

        # >>> Dropdown 3: Colour picker <<<
        self.use_color_picker = tk.BooleanVar(master=self, value=False)
        self.color_pick_points = {"remove": [], "keep": []}   # sample points in full-image coords
        self.color_pick_method = tk.StringVar(master=self, value="Color Distance")
        self.color_pick_output_mode = tk.StringVar(master=self, value="Remove selection")
        self.color_pick_patch_radius = tk.StringVar(master=self, value="7")
        self.color_pick_mask = None    # result mask from colour picker
        # Class names for sample groups (used in exports)
        self.color_pick_class_a_name = tk.StringVar(master=self, value="water")
        self.color_pick_class_b_name = tk.StringVar(master=self, value="sand")


        if self.mode in ("individual", "ml"):
            self.title("Feature Identifier- Configuration")
            # self.geometry("1100x850")
            # self.resizable(True, True)
            # self.minsize(900, 600)
            fit_geometry(self, 1100, 900, resizable=True)
            self.minsize(900, 600)

            self.filename_label = ctk.CTkLabel(
                self, text="No file loaded", font=("Arial", 14))
            self.filename_label.pack(side="top", fill="x", pady=5)

            # Scrollable frame for the main controls (fixes squishing when all dropdowns open)
            self.bottom_frame = ctk.CTkScrollableFrame(self, height=400)
            self.bottom_frame.pack(side="top", fill="both", expand=True)
            self.setup_controls(self.bottom_frame)
            self._editing_feature_idx = None  # index of the feature being edited (or None)


            # Keybindings for configuration window.
            self.bind("<Left>", lambda e: self.prev_image())
            self.bind("<Right>", lambda e: self.next_image())
            self.bind("<plus>", lambda e: self.calculate_mask())
            self.bind("<KP_Add>", lambda e: self.calculate_mask())
            self.bind("<minus>", lambda e: self.checkbox_invert_mask_toggle())
            self.bind("<KP_Subtract>",
                      lambda e: self.checkbox_invert_mask_toggle())
            self.bind("<F5>", lambda e: self.f5_action())
            self.bind("<F6>", lambda e: self.f6_action())
            self.bind("<space>", lambda e: self.space_action())
            self.bind("<Return>", lambda e: self.export_training_data())

            # --------------------------
            # Console output area
            # --------------------------
            self.console_frame = ctk.CTkFrame(self)
            self.console_frame.pack(
                side="bottom", fill="both", expand=False, padx=5, pady=2)
            self.console_text = scrolledtext.ScrolledText(
                self.console_frame, wrap="word", height=10)
            self.console_text.pack(side="left", fill="both", expand=True)
            self.stdout_redirector = StdoutRedirector(self.console_text)
            self.original_stdout = sys.stdout
            sys.stdout = self.stdout_redirector
            self.original_stderr = sys.stderr
            sys.stderr = self.stdout_redirector
            print("Here you may see console outputs\n--------------------------------\n")

            # image display window
            self.image_display_window = ctk.CTkToplevel(self)
            self.image_display_window.title(
                "Feature identifier - Image display")
            self.image_display_window.geometry("1200x800")

            try:
                self.image_display_window.iconbitmap(
                    self.resource_path("launch_logo.ico"))
            except:
                pass

            self.top_frame = ctk.CTkFrame(self.image_display_window)
            self.top_frame.pack(fill="both", expand=True)

            self.top_frame.grid_columnconfigure(0, weight=1, minsize=400, uniform="panels")
            self.top_frame.grid_columnconfigure(1, weight=2, minsize=400, uniform="panels")
            self.top_frame.grid_columnconfigure(2, weight=1, minsize=400, uniform="panels")


            self.top_frame.grid_rowconfigure(0, weight=1)

            self.top_left_frame = ctk.CTkFrame(self.top_frame)
            self.top_left_frame.pack_propagate(False)
            self.top_left_frame.grid_propagate(False)
            self.top_left_frame.configure(width=400, height=400)

            self.top_center_frame = ctk.CTkFrame(
                self.top_frame, fg_color="white")
            self.top_right_frame = ctk.CTkFrame(
                self.top_frame, fg_color="white")

            self.top_left_frame.grid(row=0, column=0, sticky="nsew")
            self.top_center_frame.grid(row=0, column=1, sticky="nsew")
            self.top_right_frame.grid(row=0, column=2, sticky="nsew")

            self.image_label = ctk.CTkLabel(
                self.top_left_frame, text="", anchor="center")
            self.image_label.pack(fill="both", expand=True)

            self.mask_label = ctk.CTkLabel(
                self.top_center_frame, text="", fg_color="white", anchor="center")
            self.mask_label.pack(fill="both", expand=True)

            self.edge_label = ctk.CTkLabel(
                self.top_right_frame, text="", fg_color="white", anchor="center")
            self.edge_label.pack(fill="both", expand=True)

            self.top_center_frame.grid_propagate(False)
            self.top_right_frame.grid_propagate(False)

            self.image_display_window.protocol(
                "WM_DELETE_WINDOW", self.on_all_close)
            self.protocol("WM_DELETE_WINDOW", self.on_all_close)

            self._blank_img = ctk.CTkImage(Image.new("RGBA", (1, 1), (0, 0, 0, 0)),
                                           size=(1, 1))

            # Prevent the launcher (master) from popping up over these windows
            self.after(200, self._suppress_master_raise)



        else:
            # BATCH MODE: requires a settings file from Single/Folder processing
            self.title("Feature Identifier — Batch Process")
            #self.geometry("1200x600")
            #self.resizable(False, False)
            fit_geometry(self, 1200, 600, resizeable = True)

            self.do_invert_mask = tk.BooleanVar(master=self, value=False)
            self.use_bbox = tk.BooleanVar(master=self, value=False)
            self.use_inner_mask = tk.BooleanVar(master=self, value=False)
            self.enable_enhancements = tk.BooleanVar(master=self, value=True)
            self.advanced_check_var = tk.BooleanVar(master=self, value=False)
            self.use_dual_hsv = tk.BooleanVar(master=self, value=False)
            self._batch_settings_loaded = False

            main_frame = ctk.CTkFrame(self)
            main_frame.pack(fill="both", expand=True)

            self.filename_label = ctk.CTkLabel(
                main_frame, text="No file loaded", font=("Arial", 14))
            self.filename_label.pack(side="top", fill="x", pady=5)

            # Instruction label
            ctk.CTkLabel(
                main_frame,
                text="Batch mode requires a settings file created with Single Image or Folder Processing.\n"
                     "1) Load Settings File  →  2) Load Image Folder  →  3) Run Batch Process",
                font=("Arial", 11), justify="left", text_color="gray"
            ).pack(side="top", fill="x", padx=10, pady=5)

            self.top_frame = ctk.CTkFrame(main_frame)
            self.top_frame.pack(side="top", fill="both", expand=True)

            # Settings summary label — shows what pipeline is configured
            self.settings_summary_label = ctk.CTkLabel(
                self.top_frame,
                text="No settings loaded yet.",
                font=("Arial", 10), justify="left", anchor="w",
                wraplength=1100)
            self.settings_summary_label.pack(pady=10, padx=10, anchor="w")

            self.progress_label = ctk.CTkLabel(
                self.top_frame, text="Ready for batch processing.")
            self.progress_label.pack(pady=5)

            self.batch_progress_bar = ctk.CTkProgressBar(
                self.top_frame, width=500)
            self.batch_progress_bar.set(0)
            self.batch_progress_bar.pack(pady=10)

            # Pack the console frame inside the main_frame:
            self.console_frame = ctk.CTkFrame(main_frame)
            self.console_frame.pack(
                side="bottom", fill="x", expand=False, padx=5, pady=0)

            self.console_text = scrolledtext.ScrolledText(
                self.console_frame, wrap="word", height=6)
            self.console_text.pack(side="left", fill="both", expand=True)

            self.stdout_redirector = StdoutRedirector(self.console_text)
            self.original_stdout = sys.stdout
            sys.stdout = self.stdout_redirector
            self.original_stderr = sys.stderr
            sys.stderr = self.stdout_redirector
            print("Here you may see console outputs\n")

            self.bottom_frame = ctk.CTkFrame(main_frame)
            self.bottom_frame.pack(side="bottom", fill="x", expand=False)

            # Setup controls for batch
            self.setup_controls(self.bottom_frame)

        # Shared variables
        self.cv_image = None
        self.full_image = None
        self.scale = 1.0
        self.full_alpha_mask = None
        self.full_alpha_mask_inner = None
        self.alpha_mask = None
        self.current_mask = None
        self.inner_bbox_mask = None

        self.image_path = None
        self.image_files = []
        self.current_index = 0

        # Single "edge points" from a direct detection:
        self.edge_points = []
        self.edited_edge_points = []
        self.initial_edge_points = []
        self.edit_history = []
        self.selected_vertex = None
        self.redo_history = []

        # More general shape storage:
        # list of (feature_type, [(x,y), (x,y), ...]) for polylines or polygons
        self.features = []
        self._edit_session_features = []  # temporary store during edit sessions
        self.is_polygon_mode = False
        self.creation_mode = False

        # For zoom and pan in the "cut/edit" mode
        self.zoom_scale = getattr(self, "zoom_scale", 1.0)  # default zoom if not set elsewhere
        self.bg_image_id = None
        self.edit_original_pil = None
        # self.zoom_scale = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.zoomed_image = None
        self._restore_center_mask_panel()
        
        # --- performance caches / throttles ---
        self._bg_cache = {}          # { zoom_key: ImageTk.PhotoImage }
        self._bg_current_zoom = None
        self._bg_item_id = None      # canvas item id for bg image
        self._poly_id = None         # single polyline item id
        self._vertex_ids = []        # per-vertex oval ids
        self._redraw_job = None      # after() token for throttled redraw
        self._preview_job = None     # after() token for debounced preview
        self._drag_job = None        # after() token for throttled drag updates
        # --- live preview throttling / caches ---
        self._preview_after_id = None
        self._pending_preview = False
        self._zoom_cache = {"zoom": None, "img": None}  # PIL RGB image at current zoom

        # --- Edit-mode proxy for large images (Issue 4 fix) ---
        self._edit_proxy = None      # downscaled numpy BGR for editing
        self._edit_proxy_scale = 1.0 # ratio: proxy / full

    def _suppress_master_raise(self):
        """Prevent the launcher window from popping up over the tool windows."""
        try:
            if self.master and self.master.winfo_exists():
                self.master.lower()
            self.lift()
            self.focus_force()
            if hasattr(self, 'image_display_window') and self.image_display_window.winfo_exists():
                self.image_display_window.lift()
        except Exception:
            pass

    def reset_session(self):
        """Reset all state to initial — clears image, masks, features, AOI, color picker."""
        # Clear images
        self.cv_image = None
        self.full_image = None
        self._edit_proxy = None
        self._edit_proxy_scale = 1.0
        self.scale = 1.0
        self.full_alpha_mask = None
        self.full_alpha_mask_inner = None
        self.alpha_mask = None
        self.current_mask = None
        self.inner_bbox_mask = None
        self.image_path = None
        self.image_files = []
        self.current_index = 0

        # Clear features/editing
        self.edge_points = []
        self.edited_edge_points = []
        self.initial_edge_points = []
        self.edit_history = []
        self.selected_vertex = None
        self.redo_history = []
        self.features = []
        self._edit_session_features = []
        self.is_polygon_mode = False
        self.creation_mode = False

        # Clear AOI
        self.aoi_mask = None
        self.aoi_click_points = []

        # Clear color picker
        self.color_pick_points = {"remove": [], "keep": []}
        self.color_pick_mask = None
        if hasattr(self, "_update_color_pick_labels"):
            self._update_color_pick_labels()

        # Clear zoom/pan caches
        self.zoom_scale = 1.0
        self.pan_x = self.pan_y = 0
        self.bg_image_id = None
        self.edit_original_pil = None
        self.zoomed_image = None
        self._bg_cache.clear()
        self._bg_current_zoom = None
        self._poly_id = None
        self._vertex_ids = []
        self._zoom_cache = {"zoom": None, "img": None}

        # Reset UI labels
        if hasattr(self, 'filename_label') and self.filename_label.winfo_exists():
            self.filename_label.configure(text="No file loaded")
        if hasattr(self, 'cpick_a_label'):
            self.cpick_a_label.configure(text="A: —")
            self.cpick_b_label.configure(text="B: —")

        # Restore mask panel and clear displays
        self._restore_center_mask_panel()
        if hasattr(self, 'image_label') and self.image_label.winfo_exists():
            self._clear_ctk_label(self.image_label)
        if hasattr(self, 'edge_label') and self.edge_label.winfo_exists():
            self._clear_ctk_label(self.edge_label)

        # Clear console
        if hasattr(self, 'console_text') and self.console_text.winfo_exists():
            self.console_text.delete("1.0", "end")

        print("Session reset.\n================================")


# -------------- ENTRY POINT --------------

def main():
    root = ctk.CTk()
    root.withdraw()
    mode = sys.argv[1] if len(sys.argv) > 1 else 'individual'
    win = FeatureIdentifier(master=root, mode=mode)
    root.mainloop()

# Backward-compatible alias so existing imports still work
HSVMaskTool = FeatureIdentifier


if __name__ == '__main__':
    main()
