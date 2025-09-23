import os
import json
import shutil
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import customtkinter as ctk
from PIL import Image, ImageTk, ImageEnhance
import cv2
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, Polygon
import rasterio
import warnings
from rasterio.errors import NotGeoreferencedWarning
from difflib import SequenceMatcher  # <-- added for filename similarity
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
# Theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")


# %% helper class
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

# %% Helper functions


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


# %% Main class

class HSVMaskTool(ctk.CTkToplevel):
    """
    A top-level window for the HSV Mask Tool. 3 modes:
      - individual
      - ml
      - batch
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


        if self.mode in ("individual", "ml"):
            # super().__init__(*args, **kwargs)
            self.title("Feature Identifier- Configuration")
            # Increase height to accommodate console
            self.geometry("1100x750")
            self.resizable(False, False)

            self.filename_label = ctk.CTkLabel(
                self, text="No file loaded", font=("Arial", 14))
            self.filename_label.pack(side="top", fill="x", pady=5)

            # Bottom frame for the main controls
            self.bottom_frame = ctk.CTkFrame(self)
            self.bottom_frame.pack(side="top", fill="x", expand=False)
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



        else:
            # BATCH MODE:
            # super().__init__(*args, **kwargs)
            self.title("Feature identifier- batch process")
            self.geometry("1200x600")
            self.resizable(False, False)

            self.do_invert_mask = tk.BooleanVar(master=self, value=False)
            self.use_bbox = tk.BooleanVar(master=self, value=False)
            self.use_inner_mask = tk.BooleanVar(master=self, value=False)
            self.enable_enhancements = tk.BooleanVar(master=self, value=True)
            self.advanced_check_var = tk.BooleanVar(master=self, value=False)
            self.use_dual_hsv = tk.BooleanVar(master=self, value=False)

            main_frame = ctk.CTkFrame(self)
            main_frame.pack(fill="both", expand=True)

            self.filename_label = ctk.CTkLabel(
                main_frame, text="No file loaded", font=("Arial", 14))
            self.filename_label.pack(side="top", fill="x", pady=5)

            self.top_frame = ctk.CTkFrame(main_frame)
            self.top_frame.pack(side="top", fill="both", expand=True)

            self.progress_label = ctk.CTkLabel(
                self.top_frame, text="Ready for batch processing.")
            self.progress_label.pack(pady=10)

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



    def checkbox_invert_mask_toggle(self):
        self.do_invert_mask.set(not self.do_invert_mask.get())
        # if mask is showing, recalculate
        if self.current_mask is not None and self.mode != 'batch':
            self.calculate_mask()

    def on_all_close(self):
        # Restore stdout
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

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
            self.mask_label = ctk.CTkLabel(self.top_center_frame, text="", fg_color="white", anchor="center")
            self.mask_label.pack(fill="both", expand=True)


    # -------------- UTILITY METHODS --------------

    def distance(self, x1, y1, x2, y2):
        return ((x2 - x1)**2 + (y2 - y1)**2) ** 0.5

    def _clear_ctk_label(self, lbl):
        """Attach a 1×1 transparent CTkImage so CTk never points to a dead id."""
        if lbl and lbl.winfo_exists():
            # valid image id → no TclError
            lbl.configure(image=self._blank_img)
            lbl.image = self._blank_img           # keep reference

    # >>> New helpers for external mask workflow <<<

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
        if m.ndim == 3:
            # if RGB/RGBA, convert to grayscale
            if m.shape[2] == 4:
                b, g, r, a = cv2.split(m)
                m = a  # prefer alpha if provided
            else:
                m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
        # binarize
        _, mbin = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return mbin

    def _resize_mask_to_cv(self, m):
        """Resize a binary mask to current cv_image size with nearest-neighbor."""
        if self.cv_image is None or m is None:
            return None
        m = self._ensure_binary_u8(m)
        h, w = self.cv_image.shape[:2]
        if m.shape != (h, w):
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

    # -------------- IMAGE DISPLAY METHODS --------------
    def update_image_display(self, event=None):
        if self.cv_image is None:
            return
        width  = self.top_left_frame.winfo_width()
        height = self.top_left_frame.winfo_height()
        if width < 1 or height < 1:
            return
    
        disp = self.cv_image.copy()
        h0, w0 = disp.shape[:2]
    
        # bbox in CV coords
        if hasattr(self, 'bbox') and self.use_bbox.get():
            x0, y0, w, h = self.bbox
            x = int(x0 * self.scale); y = int(y0 * self.scale)
            w = int(w  * self.scale); h = int(h  * self.scale)
            cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
        # now resize once to panel size
        resized = cv2.resize(disp, (width, height), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        ctk_img = ctk.CTkImage(light_image=pil, dark_image=pil, size=(width, height))
        self.image_label.configure(image=ctk_img)
        self.image_label.image = ctk_img


    def update_mask_display(self, event=None):
        if not self.mask_label.winfo_exists():
            return
        if self.current_mask is not None:
            width = self.top_center_frame.winfo_width()
            height = self.top_center_frame.winfo_height()
            if width > 0 and height > 0:
                mask_rgb = cv2.cvtColor(self.current_mask, cv2.COLOR_GRAY2RGB)
                pil_img = Image.fromarray(mask_rgb)
                ctk_img = ctk.CTkImage(
                    light_image=pil_img, dark_image=pil_img, size=(width, height))
                self.mask_label.configure(image=ctk_img)
                self.mask_label.image = ctk_img

    def update_edge_display(self, event=None):
        """Display all stored features in the right panel."""
        if self.full_image is None:
            return
        width = self.top_right_frame.winfo_width()
        height = self.top_right_frame.winfo_height()
        if width < 1 or height < 1:
            return

        overlay = self.full_image.copy()
        for (feature_type, points) in self.features:
            if len(points) < 2:
                continue
            pts_np = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
            if feature_type == "polygon":
                # Outline in green, thickness=2
                cv2.polylines(overlay, [pts_np], isClosed=True, color=(
                    0, 255, 0), thickness=2)
            else:
                # polyline
                thickness = int(self.edge_thickness_slider.get()) if hasattr(
                    self, 'edge_thickness_slider') else 2
                cv2.polylines(overlay, [pts_np], isClosed=False, color=(
                    0, 255, 0), thickness=thickness)

        resized = cv2.resize(overlay, (width, height),
                             interpolation=cv2.INTER_AREA)
        overlay_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(overlay_rgb)
        ctk_img = ctk.CTkImage(light_image=pil_img,
                               dark_image=pil_img, size=(width, height))
        self.edge_label.configure(image=ctk_img)
        self.edge_label.image = ctk_img

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
    
            self.btn_calc_edge_with_ml = ctk.CTkButton(
                self.ml_opts_row, text="Calculate Edge with Mask", command=self.calculate_edge_with_ml_mask
            )
            self.btn_calc_edge_with_ml.grid(row=0, column=2, padx=8, pady=3, sticky="e")
    
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
    
            self.btn_calc_edge_with_ml = ctk.CTkButton(
                self.ml_opts_row, text="Calculate Edge with Mask", command=self.calculate_edge_with_ml_mask
            )
            self.btn_calc_edge_with_ml.grid(row=0, column=4, padx=8, pady=3, sticky="e")
    
        # ── Row 2: Enhance Frame (we keep a handle to pack ML options before this row) ──
        self.enhance_frame = ctk.CTkFrame(parent)
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
        hsv_frame = ctk.CTkFrame(parent)
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
    
        # ── Row 4: Edge Controls ───────────────────────────────────────────────────
        edge_container = ctk.CTkFrame(parent)
        edge_container.pack(side="top", fill="x", pady=5)
    
        calc_frame = ctk.CTkFrame(edge_container)
        calc_frame.pack(side="top", fill="x", pady=2)
    
        if self.mode == "batch":
            invert_chk = ctk.CTkCheckBox(calc_frame, text="Invert Mask?", variable=self.do_invert_mask)
            invert_chk.pack(side="left", padx=10)
    
        if self.mode != "batch":
            btn_calc = ctk.CTkButton(calc_frame, text="Calculate Mask", command=self.calculate_mask)
            btn_calc.pack(side="left", padx=5)
            invert_chk = ctk.CTkCheckBox(calc_frame, text="Invert Mask?", variable=self.do_invert_mask)
            invert_chk.pack(side="left", padx=10)
    
        if self.mode == "ml":
            btn_prev = ctk.CTkButton(calc_frame, text="Previous", command=self.prev_image)
            btn_prev.pack(side="left", padx=5)
            btn_next = ctk.CTkButton(calc_frame, text="Next", command=self.next_image)
            btn_next.pack(side="left", padx=5)
    
        if self.mode in ("ml", "individual"):
            btn_edge = ctk.CTkButton(calc_frame, text="Calculate Edge", command=self.calculate_edge)
            btn_edge.pack(side="left", padx=5)
            btn_cut_feature = ctk.CTkButton(
                calc_frame, text="Edit Detected Feature", command=self.cut_detected_feature
            )
            btn_cut_feature.pack(side="left", padx=5)
    
            thickness_frame = ctk.CTkFrame(calc_frame)
            thickness_frame.pack(side="left", padx=3)
            ctk.CTkLabel(thickness_frame, text="Edge Thickness (pixels)").pack(side="top")
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
        btn_browse = ctk.CTkButton(export_frame, text="Browse", command=self.browse_export_folder)
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
                export_buttons_frame, text="Export feature as training data", command=self.export_training_data
            )
            btn_export_edge.pack(side="left", padx=5)
            btn_export_mask = ctk.CTkButton(
                export_buttons_frame, text="Export mask as training data", command=self.export_mask_as_training_data
            )
            btn_export_mask.pack(side="left", padx=5)
            btn_export_test = ctk.CTkButton(
                export_buttons_frame, text="Export as Test Data", command=self.export_as_test_data
            )
            btn_export_test.pack(side="left", padx=5)
            if self.mode == "individual":
                btn_export_overlay = ctk.CTkButton(
                    export_buttons_frame, text="Export as Overlay", command=self.export_as_overlay
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
    
        # ── Row 8: Shortcuts ───────────────────────────────────────────────────────
        if self.mode in ("individual", "ml"):
            shortcut_frame = ctk.CTkFrame(parent)
            shortcut_frame.pack(side="top", fill="x", pady=5)
            ctk.CTkLabel(shortcut_frame, text="Shortcuts:", font=("Arial", 10, "bold")).pack(side="left", padx=5)
            shortcuts = [
                ("Left/Right", "Prev/Next"),
                ("Plus", "Calculate mask"),
                ("Minus", "Invert"),
                ("F5", "Calculate Edge w/ mask"),
                ("F6", "Calculate Edge w/ inverted mask"),
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
        """Show/hide the ML mask options row immediately below the checkbox row (and above Enhance row)."""
        # Ensure checkbox row stays right under import row:
        try:
            # Put the checkbox row right before the enhance frame if it's not already
            self.ml_row.pack_forget()
            self.ml_row.pack(side="top", fill="x", pady=(2, 0), before=self.enhance_frame)
        except Exception:
            # fallback to regular pack if 'before' not available
            self.ml_row.pack(side="top", fill="x", pady=(2, 0))

        if self.use_ml_pred_mask.get():
            self.ml_opts_row.pack_forget()
            try:
                self.ml_opts_row.pack(side="top", fill="x", pady=(2, 5), before=self.enhance_frame)
            except Exception:
                self.ml_opts_row.pack(side="top", fill="x", pady=(2, 5))
        else:
            self.ml_opts_row.pack_forget()


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
    
    def _ensure_binary_u8(self, img):
        """Return a strictly binary uint8 mask (0/255). Accepts BGR/GRAY/float/bool."""
        if img is None:
            return None
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = (img > 0).astype(np.uint8) * 255 if img.dtype == bool else img
        # Normalize dynamic range then hard threshold
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, bin8 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        return bin8

    def _centerline_polyline_from_skeleton(self, skel_bin):
        """
        Given a 1-pixel wide skeleton (uint8 {0,255} or {0,1}), return a single
        ordered polyline (list of [x,y] in cv_image coords) along the longest
        8-connected path. No contours used → no double line.
        """
        import numpy as np
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
        import numpy as np
        import cv2
    
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
    
        # scale to full_image coords (we render/export there)
        sx = self.full_image.shape[1] / self.cv_image.shape[1]
        sy = self.full_image.shape[0] / self.cv_image.shape[0]
        pts_full = [[int(x * sx), int(y * sy)] for x, y in pts_cv]
    
        # optional inner-bbox filtering (non-destructive)
        if self.use_bbox.get() and self.use_inner_mask.get() and self.inner_bbox_mask is not None:
            kept = []
            H, W = self.inner_bbox_mask.shape[:2]
            for x, y in pts_full:
                if 0 <= x < W and 0 <= y < H and self.inner_bbox_mask[y, x] > 0:
                    kept.append([x, y])
            if len(kept) >= 2:
                pts_full = kept
    
        # store exactly ONE polyline → editor shows one line
        self.edge_points = pts_full
        self.features = [("polyline", pts_full.copy())]
        self.update_edge_display()
        return True




    def calculate_edge_with_ml_mask(self):
        """
        Load/choose the ML mask for the current image, set it as current_mask,
        and extract an edge that populates self.features (so the editor works).
        """
        if self.cv_image is None:
            # Attempt to get an image ready in ml mode
            if self.mode == "ml" and self.image_files:
                self.load_current_image()
            if self.cv_image is None:
                messagebox.showwarning("Image", "Please load an image first.")
                return
    
        # Resolve a mask path based on mode
        mask_path = None
        if self.mode == "individual":
            mask_path = self.ml_mask_file_path.get().strip()
            if not mask_path:
                messagebox.showwarning("ML mask", "Please load an associated mask.")
                return
        elif self.mode == "ml":
            if not self.image_files:
                messagebox.showwarning("Folder", "Please load an image folder first.")
                return
            folder = self.ml_mask_folder_path.get().strip()
            if not folder:
                messagebox.showwarning("Mask folder", "Please load the associated mask folder.")
                return
            # pick best match for current image
            cur_img = self.image_files[self.current_index]
            base = os.path.splitext(os.path.basename(cur_img))[0]
            mask_path = self._best_match_in_folder(base, folder, self.common_name_len_var.get())
            if mask_path is None:
                messagebox.showerror("Mask match", f"No matching mask found for:\n{os.path.basename(cur_img)}")
                return
        else:
            messagebox.showwarning("Mode", "Calculate Edge with Mask is not available in batch mode.")
            return
    
        # Read mask and prepare
        m = self._read_mask_image(mask_path)
        if m is None:
            messagebox.showerror("Mask", f"Failed to read mask:\n{mask_path}")
            return
    
        m = self._resize_mask_to_cv(m)
        if m is None:
            messagebox.showerror("Mask", "Could not resize mask to the working image size.")
            return
    
        # Optional: intersect with bbox if user enabled
        if self.use_bbox.get():
            try:
                x, y, w, h = map(int, self.bbox_entry.get().strip("()").split(","))
                x = int(x * self.scale); y = int(y * self.scale)
                w = int(w * self.scale); h = int(h * self.scale)
                bbox_mask = np.zeros(self.cv_image.shape[:2], dtype=np.uint8)
                bbox_mask[y:y+h, x:x+w] = 255
                m = cv2.bitwise_and(m, bbox_mask)
                if self.use_inner_mask.get():
                    kernel_inner = np.ones((20, 20), np.uint8)
                    self.inner_bbox_mask = cv2.erode(bbox_mask, kernel_inner, iterations=1)
            except Exception:
                pass  # ignore malformed bbox
    
        # Store & show
        self.current_mask = self._ensure_binary_u8(m)
        self.display_mask()
    
        # Now do the robust edge extraction
        ok = self._detect_edge_from_current_mask_robust()
        if not ok:
            messagebox.showwarning("Edge", "No contour/edge could be extracted from the mask.")


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

    # -------------- IMPORT/LOAD --------------

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff")]
        )
        if not file_path:
            return
        original_image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if original_image is None:
            messagebox.showerror("Error", f"Failed to load image: {file_path}")
            return
        self.image_path = file_path
        self.filename_label.configure(text=os.path.basename(file_path))
        self.full_image = original_image.copy()
        self.compute_full_masks(original_image)
        if self.mode != "batch":
            self.process_loaded_image(original_image)

    def load_folder(self):
        folder = filedialog.askdirectory()
        if not folder:
            return
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        files = [os.path.join(folder, f) for f in os.listdir(
            folder) if f.lower().endswith(exts)]
        if not files:
            messagebox.showerror(
                "Error", "No valid image files found in folder.")
            return
        files.sort()
        self.image_files = files
        self.current_index = 0
        if self.mode == "batch":
            self.filename_label.configure(
                text=f"Loaded {len(files)} images for batch")
        else:
            self.load_current_image()

    def load_current_image(self):
        if not self.image_files:
            return

        # 1) Clear old shapes
        self._restore_center_mask_panel()

        self.features = []
        self.edge_points = []
        self.edited_edge_points = []

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
            messagebox.showerror("Error", f"Failed to load image {file_path}")
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
                messagebox.showinfo("Info", "Already at the last image.")

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
                messagebox.showinfo("Info", "Already at the first image.")

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
                        "Error", f"Invalid bounding box format: {e}")
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

        if self.do_invert_mask.get():
            mask_clean = cv2.bitwise_not(mask_clean)

        self.current_mask = mask_clean

        if self.mode != "batch":
            self.display_mask()

    def display_mask(self):
        if self.current_mask is None or self.mode == "batch":
            return
        to_show = self.current_mask
        mask_rgb = cv2.cvtColor(to_show, cv2.COLOR_GRAY2RGB)
        pil_img = Image.fromarray(mask_rgb)
        ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(
            pil_img.width, pil_img.height))
        self.mask_label.configure(image=ctk_img)
        self.mask_label.image = ctk_img

    # -------------- EDGE DETECTION --------------

    def calculate_edge(self):
        if self.current_mask is None:
            return
        clean_mask = self.current_mask.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(
            clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return

        if self.advanced_check_var.get():
            min_contour = self.min_contour_entry.get().strip()
            max_contour = self.max_contour_entry.get().strip()
            min_size = 0
            max_size = float('inf')
            try:
                if min_contour:
                    min_size = int(min_contour)
                if max_contour:
                    max_size = int(max_contour)
            except ValueError:
                messagebox.showerror(
                    "Error", "Invalid min or max contour size. Please enter integers.")
                return
            if max_size < min_size:
                messagebox.showerror(
                    "Error", "Maximum contour size must be >= minimum.")
                return
            filtered_contours = []
            for cnt in contours:
                length = cv2.arcLength(cnt, True)
                if min_size <= length <= max_size:
                    filtered_contours.append(cnt)
            if not filtered_contours:
                messagebox.showwarning(
                    "No Contours", "No contours found in size range.")
                return
            largest = max(filtered_contours,
                          key=lambda c: cv2.arcLength(c, True))
        else:
            largest = max(contours, key=lambda c: cv2.arcLength(c, True))

        epsilon = 0.0001 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)
        if not isinstance(self.full_image, np.ndarray):
            return

        scale_x = self.full_image.shape[1] / self.cv_image.shape[1]
        scale_y = self.full_image.shape[0] / self.cv_image.shape[0]
        rescaled_points = []
        for pt in approx:
            rx = int(pt[0][0] * scale_x)
            ry = int(pt[0][1] * scale_y)
            rescaled_points.append([rx, ry])

        valid_points = []
        if self.use_bbox.get() and self.use_inner_mask.get() and self.inner_bbox_mask is not None:
            for (x, y) in rescaled_points:
                if 0 <= x < self.inner_bbox_mask.shape[1] and 0 <= y < self.inner_bbox_mask.shape[0]:
                    if self.inner_bbox_mask[y, x] > 0:
                        valid_points.append([x, y])
        elif self.full_alpha_mask_inner is not None:
            for (x, y) in rescaled_points:
                if 0 <= x < self.full_alpha_mask_inner.shape[1] and 0 <= y < self.full_alpha_mask_inner.shape[0]:
                    if self.full_alpha_mask_inner[y, x] > 0:
                        valid_points.append([x, y])
        else:
            valid_points = rescaled_points

        self.edge_points = valid_points
        

        # Also reflect this as a feature so Edit Detected Feature & thickness apply
        if valid_points:
            self.features = [("polyline", valid_points.copy())]
        else:
            self.features = []

        if self.mode != "batch":
            overlay = self.full_image.copy()
            thickness = int(self.edge_thickness_slider.get())
            if len(valid_points) > 1:
                pts = np.array(
                    valid_points, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(overlay, [pts], isClosed=False, color=(
                    0, 255, 0), thickness=thickness)
            disp_overlay = cv2.resize(
                overlay, (self.cv_image.shape[1], self.cv_image.shape[0]), interpolation=cv2.INTER_AREA)
            overlay_rgb = cv2.cvtColor(disp_overlay, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(overlay_rgb)
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(
                pil_img.width, pil_img.height))
            self.edge_label.configure(image=ctk_img)
            self.edge_label.image = ctk_img

    # -------------- FEATURE EDITING --------------

    def start_freehand(self):
        self.freehand_mode = True
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
            for seq in ("d", "m", "f", "e", "p",
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
    
        # fast path: operate on cached zoom-sized base, cheap math
        base = self._ensure_scaled_base_for_zoom(high_quality=False)
        preview = self._apply_edit_ops_fast(base, high_quality=False)
        self._set_edit_preview(preview)
    
        if self._pending_preview:
            self._preview_after_id = self.after(33, self._flush_preview)
    
    def _on_adjust_commit(self, _evt=None):
        """Mouse released — do one nicer pass (still at current zoom)."""
        base = self._ensure_scaled_base_for_zoom(high_quality=True)
        preview = self._apply_edit_ops_fast(base, high_quality=True)
        self._set_edit_preview(preview)
        
    def _invalidate_zoom_cache(self):
        self._zoom_cache["zoom"] = None
        self._zoom_cache["img"] = None
    
    def _ensure_scaled_base_for_zoom(self, high_quality=False):
        """
        Return a PIL RGB image already resized to the current zoom.
        Cached so we don't resize per slider tick.
        """
        if self.full_image is None:
            return None
        z = max(0.1, min(10.0, float(getattr(self, "zoom_scale", 1.0))))
        if self._zoom_cache["zoom"] == z and self._zoom_cache["img"] is not None:
            return self._zoom_cache["img"]
    
        # BGR numpy -> PIL RGB exactly once
        src_rgb = cv2.cvtColor(self.full_image, cv2.COLOR_BGR2RGB)
        h, w = src_rgb.shape[:2]
        sw, sh = max(1, int(w * z)), max(1, int(h * z))
        resample = Image.LANCZOS if high_quality else Image.BILINEAR
        pil = Image.fromarray(src_rgb).resize((sw, sh), resample)
        self._zoom_cache["zoom"] = z
        self._zoom_cache["img"] = pil
        return pil


    def cut_detected_feature(self):
        """
        Allows editing of the last-detected shape (self.edge_points).
        The user can add new polylines or polygons or draw freehand.
        """
        self.vertex_mode = "delete"
        self.freehand_mode = False
    
        # Decide which feature we’re editing:
        # 1) Prefer an existing polyline in self.features (last one)
        # 2) Otherwise fall back to self.edge_points
        points_to_edit = []
        self._editing_feature_idx = None
    
        # Find last polyline feature
        for idx in range(len(self.features) - 1, -1, -1):
            ftype, pts = self.features[idx]
            if ftype == "polyline" and len(pts) >= 2:
                self._editing_feature_idx = idx
                points_to_edit = pts.copy()
                break
    
        # If no polyline feature found but we have edge_points, use them
        if self._editing_feature_idx is None and self.edge_points and len(self.edge_points) >= 2:
            points_to_edit = self.edge_points.copy()
    
        # If we are editing an existing feature, temporarily remove it from the list
        # so the right panel won’t render it during editing (prevents double lines).
        if self._editing_feature_idx is not None:
            self._removed_feature = self.features.pop(self._editing_feature_idx)
        else:
            self._removed_feature = None
    
        # If nothing to edit, start empty
        if not points_to_edit:
            print("No valid detected feature to edit. Starting empty if you want to create new shapes.")
            self.edge_points = []
    
        self.initial_edge_points = points_to_edit.copy()
        self.edited_edge_points  = points_to_edit.copy()
        self.edit_history = [self.edited_edge_points.copy()]
        self.selected_vertex = None
        self.is_polygon_mode = False
        self.freehand_mode = False
    
        # Clear the center frame and create editing canvas
        for widget in self.top_center_frame.winfo_children():
            widget.destroy()
    
        # Since we removed the feature from the list, refresh the right pane to avoid showing it
        self.update_edge_display()
    
        self.zoom_scale = 1.0
        self.pan_x = self.pan_y = 0
    
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
    
        # Control buttons
        self.control_frame = ctk.CTkFrame(self.top_center_frame)
        self.control_frame.pack(side="bottom", fill="x", pady=5)
    
        mode_frame = ctk.CTkFrame(self.control_frame)
        mode_frame.pack(side="top", pady=2)
    
        # Delete / Move vertex mode
        self.btn_delete_mode = ctk.CTkButton(
            mode_frame, text="Delete Vertex", command=self._btn_mode_delete
        )
        self.btn_delete_mode.pack(side="left", padx=3)
        self.btn_add_mode = ctk.CTkButton(
            mode_frame, text="Add/Move Vertex", command=self._btn_mode_add
        )
        self.btn_add_mode.pack(side="left", padx=3)
    
        # Freehand drawing
        btn_freehand = ctk.CTkButton(mode_frame, text="Freehand", command=self._btn_freehand)
        btn_freehand.pack(side="left", padx=3)
    
        btn_delete_all = ctk.CTkButton(
            mode_frame, text="Delete All",
            command=self.delete_all_vertices, fg_color="red", text_color="white"
        )
        btn_delete_all.pack(side="left", padx=3)
    
        # Create new edge / polygon
        btn_create_edge = ctk.CTkButton(
            mode_frame, text="Create New Edge", command=self._btn_create_edge
        )
        btn_create_edge.pack(side="left", padx=3)
        btn_create_polygon = ctk.CTkButton(
            mode_frame, text="Create Polygon", command=self._btn_create_polygon
        )
        btn_create_polygon.pack(side="left", padx=3)
    
        self.reset_btn = ctk.CTkButton(mode_frame, text="Reset", command=self.reset_to_initial,fg_color="red", text_color="white")
        self.reset_btn.pack(side="left", padx=3)
    
        # Undo / Reset / Confirm (+ Redo)
        btn_frame = ctk.CTkFrame(self.control_frame)
        btn_frame.pack(side="top", pady=5)
    
        self.undo_btn = ctk.CTkButton(btn_frame, text="Undo", command=self._btn_undo)
        self.undo_btn.pack(side="left", padx=5)
    
    
        redo_btn = ctk.CTkButton(btn_frame, text="Redo", command=self._btn_redo)
        redo_btn.pack(side="left", padx=5)
        
        self.confirm_button = ctk.CTkButton(
            btn_frame, text="Confirm Feature",
            command=self.confirm_feature_cuts, fg_color="white", text_color="black"
        )
        self.confirm_button.pack(side="left", padx=5)
    
        # Zoom controls
        ctk.CTkButton(btn_frame, text="Zoom In",
                      command=lambda: self.adjust_zoom(1.2)).pack(side="left", padx=2)
        ctk.CTkButton(btn_frame, text="Zoom Out",
                      command=lambda: self.adjust_zoom(0.8)).pack(side="left", padx=2)
        ctk.CTkButton(btn_frame, text="Reset View",
                      command=self.reset_view).pack(side="left", padx=2)
    
        # Build the non-destructive preview adjust row (Saturation / Exposure / Highlights)
        self._build_adjust_row(self.control_frame)
        self._install_slider_handlers()

    
        info_label = ctk.CTkLabel(
            self.control_frame,
            text="Scroll to zoom | Double-click (delete/add) depending on mode | "
                 "Keys: d=delete, m=move, f=freehand, e=new edge, p=new polygon, U=undo, R=redo",
            font=("Arial", 10)
        )
        info_label.pack(side="top", pady=2)
    
        # enable edit-mode shortcuts (binds d/m/f/e/p + U/R and keeps canvas focused)
        self._bind_edit_shortcuts()
    
        # Initial draw
        self.redraw_canvas()


    def create_new_edge(self):
        self.edited_edge_points = []
        self.edit_history = []
        self._record_history()
        self.redraw_canvas()
        self.creation_mode = True
        self.is_polygon_mode = False
        self.edit_canvas.unbind("<Double-Button-1>")
        self.edit_canvas.bind("<Button-1>", self.on_canvas_single_click)

    def create_new_polygon(self):
        self.edited_edge_points = []
        self.edit_history = []
        self._record_history()
        self.redraw_canvas()
        self.creation_mode = True
        self.is_polygon_mode = True
        self.edit_canvas.unbind("<Double-Button-1>")
        self.edit_canvas.bind("<Button-1>", self.on_canvas_single_click)


    def confirm_feature_cuts(self):
        """
        Called when the user clicks the "Confirm Feature" button.
        If there are enough points in the current edited shape, the shape is stored
        as a feature (either polygon or polyline). If there are no points because an
        auto-closing action already stored a polygon, then the UI simply returns to normal.
        """
    
        if len(self.edited_edge_points) < 2:
            if not self.features:
                messagebox.showwarning("Warning", "Not enough points to form a feature.")
                return
            else:
                pass
        else:
            if hasattr(self, 'creation_mode') and self.creation_mode:
                self.edit_canvas.unbind("<Button-1>")
                self.creation_mode = False
    
            feature_type = "polygon" if self.is_polygon_mode else "polyline"
            new_points = self.edited_edge_points.copy()
            self.features.append((feature_type, new_points))
            self.edge_points = new_points

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

        # Destroy the editing UI elements (your existing code)
        if hasattr(self, 'edit_canvas'):
            self.edit_canvas.destroy()
        if hasattr(self, 'control_frame'):
            self.control_frame.destroy()
        if hasattr(self, 'edit_canvas_container'):
            self.edit_canvas_container.destroy()
    
        # Restore the normal view in the center (mask display).
        self.mask_label = ctk.CTkLabel(self.top_center_frame, text="")
        self.mask_label.pack(fill="both", expand=True)
        self.top_center_frame.bind("<Configure>", self.update_mask_display)
        self.update_edge_display()



    # ---------- PREVIEW ADJUST ROW (NEW) ----------
    
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
        We derive it directly from self.full_image (BGR numpy array).
        """
        img = getattr(self, "full_image", None)
        if img is None:
            return None
        # full_image is BGR -> convert to RGB and PIL
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))



    # -------------- CANVAS INTERACTION (EDITING) --------------
    
    def _apply_preview_to_bg(self):
        """
        Lightweight preview updater that respects the current zoom.
        """
        try:
            base = self._ensure_scaled_base_for_zoom(high_quality=False)
            if base is None:
                return
            preview = self._apply_edit_ops_fast(base, high_quality=False)
            self._set_edit_preview(preview)  # writes to bg_image_id
        except Exception:
            # Fallback: at least keep overlays fresh if something goes wrong
            if hasattr(self, "_refresh_overlays"):
                self._refresh_overlays()



    def _zoom_key(self):
        # quantize zoom so small mousewheel steps reuse cache
        return round(float(getattr(self, "zoom_scale", 1.0)), 1)
    
    def _ensure_bg_cached(self, high_quality=False):
        if self.full_image is None or not hasattr(self, "edit_canvas"):
            return
        key = self._zoom_key()
        cache_key = (key, high_quality)
        if self._bg_cache.get(cache_key) is None:
            img_rgb = cv2.cvtColor(self.full_image, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(img_rgb)
            h, w = self.full_image.shape[:2]
            sw, sh = max(1, int(w * key)), max(1, int(h * key))
            resample = Image.LANCZOS if high_quality else Image.BILINEAR
            scaled = pil.resize((sw, sh), resample)
            self._bg_cache[cache_key] = ImageTk.PhotoImage(scaled)
    
        photo = self._bg_cache[cache_key]
        if getattr(self, "bg_image_id", None):
            self.edit_canvas.itemconfigure(self.bg_image_id, image=photo)
        else:
            self.bg_image_id = self.edit_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.edit_canvas.config(scrollregion=(0, 0, photo.width(), photo.height()))
        self._bg_current_zoom = key

    
    def redraw_canvas(self, event=None):
        # Throttle rapid resize with after_idle
        if self._redraw_job:
            self.edit_canvas.after_cancel(self._redraw_job)
        def _do():
            if not hasattr(self, "edit_canvas"): return
            self._ensure_bg_cached()
            # (re)draw overlays in place
            self._draw_poly_in_place()
            self._draw_vertices_in_place()
            # Re-apply preview (debounced separately) if sliders moved
            if hasattr(self, "slider_sat"):
                # Don't recompute PIL now; preview uses lightweight item image swap:
                self._apply_preview_to_bg()  # see section 4
        self._redraw_job = self.edit_canvas.after_idle(_do)

    def _scaled(self, x, y):
        z = float(getattr(self, "zoom_scale", 1.0))
        return x * z, y * z
    
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
    
        # position all
        for i, (x, y) in enumerate(pts):
            sx, sy = self._scaled(x, y)
            vid = self._vertex_ids[i]
            self.edit_canvas.coords(vid, sx - r, sy - r, sx + r, sy + r)
            
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
                self.features.append(("polygon", closed_points))
                print(
                    "Polygon closed and stored. You can make another polygon if you wish.")
                self.edited_edge_points = []
                self.edit_history = []
                self.selected_vertex = None
                # self.edited_edge_points.append([x_img, y_img])
                self._record_history()
                self.redraw_canvas()
                self.update_edge_display()
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
        import numpy as np
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
            current = self.edit_history.pop()              # the state we’re leaving
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

        self.zoom_scale = max(0.1, min(self.zoom_scale, 10.0))
        self.pan_x = event.x - x_img * self.zoom_scale
        self.pan_y = event.y - y_img * self.zoom_scale
        self.redraw_canvas()
        self._invalidate_zoom_cache()
    
    def adjust_zoom(self, factor):
        self.zoom_scale *= factor
        self.zoom_scale = max(0.1, min(self.zoom_scale, 10.0))
        self.redraw_canvas()
        self._invalidate_zoom_cache()
    
    def reset_view(self):
        self.zoom_scale = 1.0
        self.pan_x = self.pan_y = 0
        self.redraw_canvas()
        self._invalidate_zoom_cache()


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
            messagebox.showwarning("Warning", "No images loaded for batch.")
            return
        export_path = self.export_path_entry.get().strip()
        if not export_path:
            messagebox.showerror("Error", "Please select an export folder.")
            return
        print("Batch process has started")
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

            # If user opted to use ML masks in batch-like processing (mode 'batch' not used here,
            # but safe to reuse behavior if they enabled the checkbox while in ml/individual),
            # default to HSV if not using ML external mask.
            if self.use_ml_pred_mask.get() and self.mode == "ml" and self.ml_mask_folder_path.get():
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

            self.calculate_edge()
    
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
    
            # Write overlay PNG
            overlay = self.full_image.copy()
            if self.edge_points:
                pts = np.array(self.edge_points, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(overlay, [pts], False, (0, 255, 0), int(self.edge_thickness_slider.get()))
            cv2.imwrite(out_ovl, overlay)
    
            processed_count += 1
    
        print(f"[batch_process] Processed {processed_count} images. GeoJSONs in {geojson_folder}, overlays in {overlay_folder}")
        messagebox.showinfo(
            "Batch Process",
            f"Processed {processed_count} images.\n"
            f"GeoJSON => {geojson_folder}\n"
            f"Overlay => {overlay_folder}"
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

    # -------------- EXPORT METHODS --------------
    
    def _sanitize_feature_id(self, s: str) -> str:
        """Make a filesystem-safe suffix from the feature ID."""
        if not s:
            return ""
        s = s.strip().lower().replace(" ", "_")
        # keep alnum, underscore, dash
        return "".join(ch for ch in s if ch.isalnum() or ch in ("_", "-"))

    

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
        print(f"[export_training_data] Export complete for {base_name}:")
        print(f"    image copy → {image_copy_path}")
        print(f"    mask       → {mask_path}")
        print(f"    overlay    → {overlay_path}")
        print(f"    geojson    → {geojson_path}")
        print(f"    coco       → {coco_path}")
        messagebox.showinfo(
            "Export Training Data",
            f"Export complete:\n"
            f"- Image   ⇒ {image_copy_path}\n"
            f"- Mask    ⇒ {mask_path}\n"
            f"- Overlay ⇒ {overlay_path}\n"
            f"- GeoJSON ⇒ {geojson_path}\n"
            f"- COCO    ⇒ {coco_path}"
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

    # -------------- SHORTCUT ACTIONS --------------

    def f5_action(self):
        if self.mode == "batch":
            return
        if not isinstance(self.full_image, np.ndarray):
            messagebox.showwarning("Warning", "No image loaded!")
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
        self.calculate_edge()

    def f6_action(self):
        if self.mode == "batch":
            return
        if not isinstance(self.full_image, np.ndarray):
            messagebox.showwarning("Warning", "No image loaded!")
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
                self.calculate_edge()
                return

        # otherwise, do normal HSV flow with invert
        self.process_loaded_image(self.full_image)
        self.calculate_mask()
        self.calculate_edge()

    def space_action(self):
        if self.mode == "ml":
            self.export_as_test_data()
            if self.image_files:
                self.image_files.pop(self.current_index)
                if not self.image_files:
                    self.cv_image = None
                    self.full_image = None
                    self.filename_label.configure(text="No file loaded")
                    messagebox.showinfo("Info", "All images processed.")
                else:
                    if self.current_index >= len(self.image_files):
                        self.current_index = len(self.image_files) - 1
                    self.load_current_image()

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

        messagebox.showinfo(
            "Load Settings", f"Settings loaded from {file_path}")

# -------------- ENTRY POINT --------------

def main():
    root = ctk.CTk()
    root.withdraw()
    mode = sys.argv[1] if len(sys.argv) > 1 else 'individual'
    win = HSVMaskTool(master=root, mode=mode)
    root.mainloop()


if __name__ == '__main__':
    main()
