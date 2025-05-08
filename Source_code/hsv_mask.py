import os
import json
import shutil
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, Polygon
import rasterio
import warnings
from rasterio.errors import NotGeoreferencedWarning
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

        if self.mode in ("individual", "ml"):
            # super().__init__(*args, **kwargs)
            self.title("Feature Identifier- Configuration")
            # Increase height to accommodate console
            self.geometry("1100x650")
            self.resizable(False, False)

            self.filename_label = ctk.CTkLabel(
                self, text="No file loaded", font=("Arial", 14))
            self.filename_label.pack(side="top", fill="x", pady=5)

            # Bottom frame for the main controls
            self.bottom_frame = ctk.CTkFrame(self)
            self.bottom_frame.pack(side="top", fill="x", expand=False)
            self.setup_controls(self.bottom_frame)

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
            print("Here you may see console outputs\n")

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

            # for col in range(3):
            #     self.top_frame.grid_columnconfigure(col, weight=1, minsize=400)
            # left column stays 400px, center/right share the rest
            self.top_frame.grid_columnconfigure(0, weight=1, minsize=400)
            self.top_frame.grid_columnconfigure(1, weight=2, minsize=400)
            self.top_frame.grid_columnconfigure(2, weight=1, minsize=400)

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

            # try:
            #     self.iconbitmap(resource_path("launch_logo.ico"))
            # except Exception as e:
            #     print("Warning: Could not load window icon:", e)

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

        # More general shape storage:
        # list of (feature_type, [(x,y), (x,y), ...]) for polylines or polygons
        self.features = []
        self.is_polygon_mode = False
        self.creation_mode = False

        # For zoom and pan in the "cut/edit" mode
        self.zoom_scale = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.zoomed_image = None

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

    # -------------- UTILITY METHODS --------------

    def distance(self, x1, y1, x2, y2):
        return ((x2 - x1)**2 + (y2 - y1)**2) ** 0.5

    def _clear_ctk_label(self, lbl):
        """Attach a 1×1 transparent CTkImage so CTk never points to a dead id."""
        if lbl and lbl.winfo_exists():
            # valid image id → no TclError
            lbl.configure(image=self._blank_img)
            lbl.image = self._blank_img           # keep reference

    # -------------- IMAGE DISPLAY METHODS --------------

    def update_image_display(self, event=None):
        if self.cv_image is None:
            return
        # panel size
        width = self.top_left_frame.winfo_width()
        height = self.top_left_frame.winfo_height()
        if width < 1 or height < 1:
            return
        disp = self.cv_image.copy()
        # bbox
        if hasattr(self, 'bbox') and self.use_bbox.get():
            x, y, w, h = self.bbox
        else:
            h0, w0 = disp.shape[:2]
            x, y, w, h = 0, 0, w0, h0
        # compute how the cv_image → panel resize scales it
        sx = width / self.cv_image.shape[1]
        sy = height / self.cv_image.shape[0]
        # then
        x1 = int(x * sx)
        y1 = int(y * sy)
        x2 = int((x + w) * sx)
        y2 = int((y + h) * sy)

        if self.use_bbox.get():
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # resize
        resized = cv2.resize(disp, (width, height),
                             interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        ctk_img = ctk.CTkImage(
            light_image=pil, dark_image=pil, size=(width, height))
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
        # Row 1: Import Frame
        import_frame = ctk.CTkFrame(parent)
        import_frame.pack(side="top", fill="x", pady=5)

        if self.mode in ("ml", "batch"):
            load_btn = ctk.CTkButton(
                import_frame, text="Load Folder", command=self.load_folder)
        else:
            load_btn = ctk.CTkButton(
                import_frame, text="Load Image", command=self.load_image)
        load_btn.pack(side="left", padx=5)

        bbox_control_frame = ctk.CTkFrame(import_frame)
        bbox_control_frame.pack(side="left", padx=5)

        # self.use_bbox = tk.BooleanVar(master=self, value=False)
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

        self.use_inner_mask = tk.BooleanVar(master=self, value=False)
        self.inner_mask_check = ctk.CTkCheckBox(
            bbox_control_frame, text="Use Inner Mask", variable=self.use_inner_mask)
        self.inner_mask_check.pack(side="left", padx=5)
        self.inner_mask_check.configure(state="disabled")

        # Row 2: Enhance Frame
        enhance_frame = ctk.CTkFrame(parent)
        enhance_frame.pack(side="top", fill="x", pady=5)

        self.enable_enhancements = tk.BooleanVar(master=self, value=True)
        enhance_chk = ctk.CTkCheckBox(
            enhance_frame, text="Enhance?", variable=self.enable_enhancements)
        enhance_chk.pack(side="left", padx=5)

        ctk.CTkLabel(enhance_frame, text="S Mult").pack(side="left", padx=2)
        self.s_multiplier_slider = ctk.CTkSlider(
            enhance_frame, from_=100, to=500)
        self.s_multiplier_slider.set(100)
        self.s_multiplier_slider.pack(side="left", padx=2)

        ctk.CTkLabel(enhance_frame, text="V Mult").pack(side="left", padx=2)
        self.v_multiplier_slider = ctk.CTkSlider(
            enhance_frame, from_=100, to=500)
        self.v_multiplier_slider.set(100)
        self.v_multiplier_slider.pack(side="left", padx=2)

        self.use_dual_hsv = tk.BooleanVar(master=self, value=False)
        dual_chk = ctk.CTkCheckBox(enhance_frame, text="Use Dual HSV Range", variable=self.use_dual_hsv,
                                   command=self.toggle_dual_sliders)
        dual_chk.pack(side="left", padx=10)

        # Row 3: HSV Sliders
        hsv_frame = ctk.CTkFrame(parent)
        hsv_frame.pack(side="top", fill="x", pady=5)

        # Lower row
        lower_container = ctk.CTkFrame(hsv_frame)
        lower_container.pack(side="top", fill="x", pady=2)

        first_lower_frame = ctk.CTkFrame(lower_container)
        first_lower_frame.pack(side="left", fill="x", expand=True)
        ctk.CTkLabel(first_lower_frame, text="Lower HSV:").pack(
            side="left", padx=5)
        self.h_low_slider, self.h_low_var, self.h_low_lbl = self.make_slider_with_label(
            first_lower_frame, "H", 0, 255, 0)
        self.s_low_slider, self.s_low_var, self.s_low_lbl = self.make_slider_with_label(
            first_lower_frame, "S", 0, 255, 0)
        self.v_low_slider, self.v_low_var, self.v_low_lbl = self.make_slider_with_label(
            first_lower_frame, "V", 0, 255, 0)

        self.dual_lower_frame = ctk.CTkFrame(lower_container)
        self.dual_lower_frame.pack(side="left", fill="x", expand=True)
        ctk.CTkLabel(self.dual_lower_frame, text="Lower HSV (Dual):").pack(
            side="left", padx=5)
        self.h2_low_slider, self.h2_low_var, self.h2_low_lbl = self.make_slider_with_label(
            self.dual_lower_frame, "H2", 0, 255, 0)
        self.s2_low_slider, self.s2_low_var, self.s2_low_lbl = self.make_slider_with_label(
            self.dual_lower_frame, "S2", 0, 255, 0)
        self.v2_low_slider, self.v2_low_var, self.v2_low_lbl = self.make_slider_with_label(
            self.dual_lower_frame, "V2", 0, 255, 0)

        # Upper row
        upper_container = ctk.CTkFrame(hsv_frame)
        upper_container.pack(side="top", fill="x", pady=2)

        first_upper_frame = ctk.CTkFrame(upper_container)
        first_upper_frame.pack(side="left", fill="x", expand=True)
        ctk.CTkLabel(first_upper_frame, text="Upper HSV:").pack(
            side="left", padx=5)
        self.h_high_slider, self.h_high_var, self.h_high_lbl = self.make_slider_with_label(
            first_upper_frame, "H", 0, 255, 255)
        self.s_high_slider, self.s_high_var, self.s_high_lbl = self.make_slider_with_label(
            first_upper_frame, "S", 0, 255, 255)
        self.v_high_slider, self.v_high_var, self.v_high_lbl = self.make_slider_with_label(
            first_upper_frame, "V", 0, 255, 255)

        self.dual_upper_frame = ctk.CTkFrame(upper_container)
        self.dual_upper_frame.pack(side="left", fill="x", expand=True)
        ctk.CTkLabel(self.dual_upper_frame, text="Upper HSV (Dual):").pack(
            side="left", padx=5)
        self.h2_high_slider, self.h2_high_var, self.h2_high_lbl = self.make_slider_with_label(
            self.dual_upper_frame, "H2", 0, 255, 255)
        self.s2_high_slider, self.s2_high_var, self.s2_high_lbl = self.make_slider_with_label(
            self.dual_upper_frame, "S2", 0, 255, 255)
        self.v2_high_slider, self.v2_high_var, self.v2_high_lbl = self.make_slider_with_label(
            self.dual_upper_frame, "V2", 0, 255, 255)

        # Row 4: Edge Controls
        edge_container = ctk.CTkFrame(parent)
        edge_container.pack(side="top", fill="x", pady=5)

        calc_frame = ctk.CTkFrame(edge_container)
        calc_frame.pack(side="top", fill="x", pady=2)

        if self.mode == "batch":
            invert_chk = ctk.CTkCheckBox(
                calc_frame, text="Invert Mask?", variable=self.do_invert_mask)
            invert_chk.pack(side="left", padx=10)

        if self.mode != "batch":
            btn_calc = ctk.CTkButton(
                calc_frame, text="Calculate Mask", command=self.calculate_mask)
            btn_calc.pack(side="left", padx=5)
            invert_chk = ctk.CTkCheckBox(
                calc_frame, text="Invert Mask?", variable=self.do_invert_mask)
            invert_chk.pack(side="left", padx=10)

        if self.mode == "ml":
            btn_prev = ctk.CTkButton(
                calc_frame, text="Previous", command=self.prev_image)
            btn_prev.pack(side="left", padx=5)
            btn_next = ctk.CTkButton(
                calc_frame, text="Next", command=self.next_image)
            btn_next.pack(side="left", padx=5)

        if self.mode in ("ml", "individual"):
            btn_edge = ctk.CTkButton(
                calc_frame, text="Calculate Edge", command=self.calculate_edge)
            btn_edge.pack(side="left", padx=5)
            # Renamed "Cut Detected Edge" -> "Edit Detected Feature"
            btn_cut_feature = ctk.CTkButton(
                calc_frame, text="Edit Detected Feature", command=self.cut_detected_feature)
            btn_cut_feature.pack(side="left", padx=5)

            thickness_frame = ctk.CTkFrame(calc_frame)
            thickness_frame.pack(side="left", padx=3)
            ctk.CTkLabel(thickness_frame,
                         text="Edge Thickness (pixels)").pack(side="top")
            self.thickness_value_label = ctk.CTkLabel(
                thickness_frame, text="2")
            self.thickness_value_label.pack(side="top")
            self.edge_thickness_slider = ctk.CTkSlider(
                thickness_frame, from_=1, to=50,
                command=lambda val: self.thickness_value_label.configure(
                    text=f"{int(float(val))}")
            )
            self.edge_thickness_slider.set(2)
            self.edge_thickness_slider.pack(side="top")

        adv_frame = ctk.CTkFrame(edge_container)
        adv_frame.pack(side="top", fill="x", pady=2)
        self.advanced_check = ctk.CTkCheckBox(adv_frame, text="Advanced Settings", variable=self.advanced_check_var,
                                              command=self.toggle_advanced_settings)
        self.advanced_check.pack(side="left", padx=5)

        self.min_contour_label = ctk.CTkLabel(
            adv_frame, text="Min contour size")
        self.min_contour_entry = ctk.CTkEntry(
            adv_frame, width=60, placeholder_text="Min")
        self.max_contour_label = ctk.CTkLabel(
            adv_frame, text="Max contour size")
        self.max_contour_entry = ctk.CTkEntry(
            adv_frame, width=60, placeholder_text="Max")

        self.min_contour_label.pack_forget()
        self.min_contour_entry.pack_forget()
        self.max_contour_label.pack_forget()
        self.max_contour_entry.pack_forget()

        # Row 5: Export Options
        export_frame = ctk.CTkFrame(parent)
        export_frame.pack(side="top", fill="x", pady=5)

        ctk.CTkLabel(export_frame, text="Export Folder:").pack(
            side="left", padx=5)
        self.export_path_entry = ctk.CTkEntry(export_frame, width=200)
        self.export_path_entry.pack(side="left", padx=5)
        btn_browse = ctk.CTkButton(
            export_frame, text="Browse", command=self.browse_export_folder)
        btn_browse.pack(side="left", padx=5)

        # Row 6: Feature frame
        featureid_frame = ctk.CTkFrame(parent)
        featureid_frame.pack(side="top", fill="x", pady=5)

        ctk.CTkLabel(featureid_frame, text="Feature ID:").pack(
            side="left", padx=5)
        self.feature_id_entry = ctk.CTkEntry(featureid_frame, width=100)
        self.feature_id_entry.pack(side="left", padx=5)

        ctk.CTkLabel(featureid_frame, text="Image ID:").pack(
            side="left", padx=5)
        self.image_id_entry = ctk.CTkEntry(featureid_frame, width=50)
        self.image_id_entry.insert(0, "1")
        self.image_id_entry.pack(side="left", padx=5)

        ctk.CTkLabel(featureid_frame, text="Category ID:").pack(
            side="left", padx=5)
        self.category_id_entry = ctk.CTkEntry(featureid_frame, width=50)
        self.category_id_entry.insert(0, "1")
        self.category_id_entry.pack(side="left", padx=5)

        export_buttons_frame = ctk.CTkFrame(parent)
        export_buttons_frame.pack(side="top", fill="x", pady=5)

        if self.mode in ("individual", "ml"):
            btn_export_edge = ctk.CTkButton(
                export_buttons_frame, text="Export feature as training data", command=self.export_training_data)
            btn_export_edge.pack(side="left", padx=5)
            btn_export_mask = ctk.CTkButton(
                export_buttons_frame, text="Export mask as training data", command=self.export_mask_as_training_data)
            btn_export_mask.pack(side="left", padx=5)
            btn_export_test = ctk.CTkButton(
                export_buttons_frame, text="Export as Test Data", command=self.export_as_test_data)
            btn_export_test.pack(side="left", padx=5)
            if self.mode == "individual":
                btn_export_overlay = ctk.CTkButton(
                    export_buttons_frame, text="Export as Overlay", command=self.export_as_overlay)
                btn_export_overlay.pack(side="left", padx=5)
        else:
            btn_batch_process = ctk.CTkButton(
                export_frame, text="Batch Process", command=self.batch_process)
            btn_batch_process.pack(side="left", padx=5)

        btn_save_settings = ctk.CTkButton(
            export_frame, text="Save Settings", command=self.save_settings)
        btn_save_settings.pack(side="left", padx=5)
        btn_load_settings = ctk.CTkButton(
            export_frame, text="Load Settings", command=self.load_settings)
        btn_load_settings.pack(side="left", padx=5)

        # Row 6: Shortcuts
        if self.mode in ("individual", "ml"):
            shortcut_frame = ctk.CTkFrame(parent)
            shortcut_frame.pack(side="top", fill="x", pady=5)
            ctk.CTkLabel(shortcut_frame, text="Shortcuts:", font=(
                "Arial", 10, "bold")).pack(side="left", padx=5)
            shortcuts = [
                ("Left/Right", "Prev/Next"),
                ("Plus", "Calculate mask"),
                ("Minus", "Invert"),
                ("F5", "Calculate Edge w/ mask"),
                ("F6", "Calculate Edge w/ inverted mask"),
                ("Space", "Export Test"),
                ("Return", "Export Training data")
            ]
            for i, (key, action) in enumerate(shortcuts):
                ctk.CTkLabel(shortcut_frame, text=key, font=(
                    "Arial", 10, "bold")).pack(side="left")
                ctk.CTkLabel(shortcut_frame, text=f" = {
                             action}", font=("Arial", 10)).pack(side="left")
                if i < len(shortcuts) - 1:
                    ctk.CTkLabel(shortcut_frame, text=" || ",
                                 font=("Arial", 10)).pack(side="left")

        self.toggle_dual_sliders()

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
        self.features = []
        self.edge_points = []
        self.edited_edge_points = []

        self._clear_ctk_label(self.edge_label)   # <‑‑ use helper
        self._clear_ctk_label(self.mask_label)   # <‑‑ use helper

        # 2) Check that the label widget still exists
        if hasattr(self, 'edge_label') and self.edge_label.winfo_exists():
            # Instead of image=None, use an empty string for safety
            self.edge_label.configure(image=None, text="")
            self.edge_label.image = None

        # Also reset mask_label if needed
        if hasattr(self, 'mask_label') and self.mask_label.winfo_exists():
            self.mask_label.configure(image=None, text="")
            self.mask_label.image = None

        # Proceed to load the new file
        file_path = self.image_files[self.current_index]
        original_image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if original_image is None:
            messagebox.showerror("Error", f"Failed to load image {file_path}")
            return

        # Update references
        self.image_path = file_path
        self.filename_label.configure(
            text=f"{os.path.basename(file_path)} ({
                self.current_index+1} / {len(self.image_files)})"
        )
        self.full_image = original_image.copy()
        self.compute_full_masks(original_image)

        if self.mode != "batch":
            self.process_loaded_image(original_image)

    def next_image(self):
        if self.mode == "ml" and self.image_files:
            if self.current_index < len(self.image_files) - 1:
                self.current_index += 1
                # Remove any polygons or edges from memory if you want, e.g.:
                # self.edge_points.clear()  # if you want a fresh start
                # But do NOT destroy or re-create the label! Just call:
                self.load_current_image()
                self.update_image_display()
                self.update_mask_display()
            else:
                messagebox.showinfo("Info", "Already at the last image.")

    def prev_image(self):
        if self.mode == "ml" and self.image_files:
            if self.current_index > 0:
                self.current_index -= 1
                self._clear_ctk_label(self.mask_label)   # <‑‑ changed
                self._clear_ctk_label(self.edge_label)   # <‑‑ changed
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
        # ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(pil_img.width, pil_img.height))
        # self.image_label.configure(image=ctk_img)
        # self.image_label.image = ctk_img

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

        # Also store it as a feature if desired:
        # But we only do that after the user finishes editing. So just keep edge_points for now.

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
        # record the state so we can undo/redo
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
        # just append, so it tacks onto whatever was there before
        self.edited_edge_points.append([x, y])
        # redraw the line immediately
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


    def cut_detected_feature(self):
        """
        Allows editing of the last-detected shape (self.edge_points).
        The user can add new polylines or polygons or draw freehand.
        """
        
        self.vertex_mode = "delete"
        self.freehand_mode = False
        
        if not self.edge_points or len(self.edge_points) < 2:
            print(
                "No valid detected feature to edit. Starting empty if you want to create new shapes.")
            self.edge_points = []

        self.initial_edge_points = self.edge_points.copy()
        self.edited_edge_points = self.edge_points.copy()
        self.edit_history = [self.edited_edge_points.copy()]
        self.selected_vertex = None
        self.is_polygon_mode = False
        self.freehand_mode = False

        # Clear the center frame and create editing canvas
        for widget in self.top_center_frame.winfo_children():
            widget.destroy()

        self.zoom_scale = 1.0
        self.pan_x = self.pan_y = 0

        self.edit_canvas_container = tk.Frame(self.top_center_frame)
        self.edit_canvas_container.pack(fill="both", expand=True)

        self.edit_canvas = tk.Canvas(self.edit_canvas_container)
        self.edit_canvas.grid(row=0, column=0, sticky="nsew")

        v_scroll = tk.Scrollbar(
            self.edit_canvas_container, orient="vertical", command=self.edit_canvas.yview)
        v_scroll.grid(row=0, column=1, sticky="ns")
        h_scroll = tk.Scrollbar(
            self.edit_canvas_container, orient="horizontal", command=self.edit_canvas.xview)
        h_scroll.grid(row=1, column=0, sticky="ew")

        self.edit_canvas.configure(
            yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
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

        # Control buttons
        self.control_frame = ctk.CTkFrame(self.top_center_frame)
        self.control_frame.pack(side="bottom", fill="x", pady=5)

        mode_frame = ctk.CTkFrame(self.control_frame)
        mode_frame.pack(side="top", pady=2)

        # Delete / Add vertex buttons
        self.btn_delete_mode = ctk.CTkButton(mode_frame, text="Delete Vertex Mode",
                                             command=lambda: self.set_vertex_mode("delete"))
        self.btn_delete_mode.pack(side="left", padx=5)
        self.btn_add_mode = ctk.CTkButton(mode_frame, text="Add Vertex Mode",
                                          command=lambda: self.set_vertex_mode("add"))
        self.btn_add_mode.pack(side="left", padx=5)

        # Freehand drawing button
        btn_freehand = ctk.CTkButton(
            mode_frame, text="Freehand", command=self.start_freehand)
        btn_freehand.pack(side="left", padx=5)

        btn_delete_all = ctk.CTkButton(mode_frame, text="Delete All Vertices",
                                       command=self.delete_all_vertices, fg_color="red", text_color="white")
        btn_delete_all.pack(side="left", padx=5)

        # Create new edge / polygon
        btn_create_edge = ctk.CTkButton(
            mode_frame, text="Create New Edge", command=self.create_new_edge)
        btn_create_edge.pack(side="left", padx=5)
        btn_create_polygon = ctk.CTkButton(
            mode_frame, text="Create Polygon", command=self.create_new_polygon)
        btn_create_polygon.pack(side="left", padx=5)

        # Undo / Reset / Confirm
        btn_frame = ctk.CTkFrame(self.control_frame)
        btn_frame.pack(side="top", pady=5)

        self.undo_btn = ctk.CTkButton(
            btn_frame, text="Undo", command=self.undo_last_action)
        self.undo_btn.pack(side="left", padx=5)
        self.reset_btn = ctk.CTkButton(
            btn_frame, text="Reset", command=self.reset_to_initial)
        self.reset_btn.pack(side="left", padx=5)
        self.confirm_button = ctk.CTkButton(btn_frame, text="Confirm Feature",
                                            command=self.confirm_feature_cuts,
                                            fg_color="white", text_color="black")
        self.confirm_button.pack(side="left", padx=5)

        # Zoom controls
        ctk.CTkButton(btn_frame, text="Zoom In", command=lambda: self.adjust_zoom(
            1.2)).pack(side="left", padx=2)
        ctk.CTkButton(btn_frame, text="Zoom Out", command=lambda: self.adjust_zoom(
            0.8)).pack(side="left", padx=2)
        ctk.CTkButton(btn_frame, text="Reset View",
                      command=self.reset_view).pack(side="left", padx=2)

        info_label = ctk.CTkLabel(
            self.control_frame,
            text="Scroll to zoom | Double-click (delete/add) depending on mode | Press Confirm when done",
            font=("Arial", 10)
        )
        info_label.pack(side="top", pady=2)

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
        # If there are fewer than 2 points in the current edit session...
        if len(self.edited_edge_points) < 2:
            # ...but if we already have at least one feature stored, then assume the polygon was auto-closed.
            if not self.features:
                messagebox.showwarning(
                    "Warning", "Not enough points to form a feature.")
                return
            else:
                # Nothing to add; simply clear the editing UI.
                pass
        else:
            # Unbind creation events if we are in creation mode.
            if hasattr(self, 'creation_mode') and self.creation_mode:
                self.edit_canvas.unbind("<Button-1>")
                self.creation_mode = False

            # Determine feature type based on editing mode.
            feature_type = "polygon" if self.is_polygon_mode else "polyline"
            new_points = self.edited_edge_points.copy()
            self.features.append((feature_type, new_points))
            # Also update self.edge_points (legacy reference).
            self.edge_points = new_points

        # Destroy the editing UI elements.
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

    # -------------- CANVAS INTERACTION (EDITING) --------------

    def redraw_canvas(self, event=None):
        if not hasattr(self, 'edit_canvas') or self.full_image is None:
            return
        self.edit_canvas.delete("all")
        img_height, img_width = self.full_image.shape[:2]
        scaled_width = int(img_width * self.zoom_scale)
        scaled_height = int(img_height * self.zoom_scale)

        img_rgb = cv2.cvtColor(self.full_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        resized_img = pil_img.resize(
            (scaled_width, scaled_height), Image.LANCZOS)
        self.zoomed_image = ImageTk.PhotoImage(resized_img)

        self.edit_canvas.create_image(
            0, 0, anchor=tk.NW, image=self.zoomed_image)
        self.edit_canvas.config(scrollregion=(
            0, 0, scaled_width, scaled_height))
        self.draw_edge_on_canvas(0, 0)

    def draw_edge_on_canvas(self, x_offset=0, y_offset=0):
        for obj in getattr(self, 'line_objects', []):
            self.edit_canvas.delete(obj)
        for obj in getattr(self, 'vertex_objects', []):
            self.edit_canvas.delete(obj)

        self.line_objects = []
        self.vertex_objects = []

        if len(self.edited_edge_points) > 1:
            for i in range(len(self.edited_edge_points) - 1):
                x1, y1 = self.edited_edge_points[i]
                x2, y2 = self.edited_edge_points[i + 1]
                tx1 = x_offset + x1 * self.zoom_scale
                ty1 = y_offset + y1 * self.zoom_scale
                tx2 = x_offset + x2 * self.zoom_scale
                ty2 = y_offset + y2 * self.zoom_scale
                line_id = self.edit_canvas.create_line(tx1, ty1, tx2, ty2,
                                                       fill="cyan", width=2, tags="edge_line")
                self.line_objects.append(line_id)

        for idx, (x, y) in enumerate(self.edited_edge_points):
            tx = x_offset + x * self.zoom_scale
            ty = y_offset + y * self.zoom_scale
            r = 5
            vertex_id = self.edit_canvas.create_oval(tx - r, ty - r, tx + r, ty + r,
                                                     fill="red", outline="black",
                                                     tags=("vertex", f"vertex_{idx}"))
            self.vertex_objects.append(vertex_id)

    def _record_history(self):
        # keep only the last, say, 50 steps so it doesn’t explode
        self.edit_history.append(self.edited_edge_points.copy())
        if len(self.edit_history) > 10:
            self.edit_history.pop(0)

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
        x_canvas = self.edit_canvas.canvasx(event.x)
        y_canvas = self.edit_canvas.canvasy(event.y)
        x_img = x_canvas / self.zoom_scale
        y_img = y_canvas / self.zoom_scale

        self.edited_edge_points[self.selected_vertex] = [x_img, y_img]
        self.redraw_canvas()

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

    def delete_all_vertices(self):
        self.edited_edge_points = []
        self._record_history()
        self.redraw_canvas()

    def undo_last_action(self):
        if len(self.edit_history) > 1:
            self.edit_history.pop()
            self.edited_edge_points = self.edit_history[-1].copy()
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

    def adjust_zoom(self, factor):
        self.zoom_scale *= factor
        self.zoom_scale = max(0.1, min(self.zoom_scale, 10.0))
        self.redraw_canvas()

    def reset_view(self):
        self.zoom_scale = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.redraw_canvas()

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
    
            # Always write a GeoJSON; if crs available, transform to real world,
            # otherwise use pixel coords
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

    def export_training_data(self):
        """
        Exports confirmed features into masks, overlays, GeoJSON, and COCO JSON.
        If the image has no georeferencing, GeoJSON will use pixel coordinates
        and a message is printed to the console.
        """
        import warnings
        from rasterio.errors import NotGeoreferencedWarning
        warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
        warnings.filterwarnings(
                            "ignore",
                            category=UserWarning,
                            message=".*crs.*was not provided.*"
                        )
    
        # 1) Read user input
        feature_name = self.feature_id_entry.get().strip()
        if not feature_name:
            messagebox.showerror("Error", "Feature ID is missing. Provide a label/name for your dataset.")
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
            messagebox.showerror("Error", "No full image is loaded or invalid image data.")
            return
        if not self.features and self.edge_points:
            self.features = [("polyline", self.edge_points.copy())]
        if not self.features:
            messagebox.showerror("Error", "No features to export. Please confirm a shape or polygon first.")
            return
    
        # 3) Prepare output folders
        export_path    = self.export_path_entry.get().strip()
        if not export_path:
            messagebox.showerror("Error", "Please specify an export folder.")
            return
        base_folder    = os.path.join(export_path, "training dataset")
        images_folder  = os.path.join(base_folder, "images")
        masks_folder   = os.path.join(base_folder, "masks")
        overlays_folder= os.path.join(base_folder, "overlays")
        geojson_folder = os.path.join(base_folder, "geojson")
        coco_folder    = os.path.join(base_folder, "coco")
        os.makedirs(images_folder,  exist_ok=True)
        os.makedirs(masks_folder,   exist_ok=True)
        os.makedirs(overlays_folder,exist_ok=True)
        os.makedirs(geojson_folder, exist_ok=True)
        os.makedirs(coco_folder,    exist_ok=True)
    
        base_name = os.path.basename(self.image_path)
        height, width = self.full_image.shape[:2]
    
        # copy original
        shutil.copy2(self.image_path, os.path.join(images_folder, base_name))
    
        # 4) Read georeference if any
        try:
            with rasterio.open(self.image_path) as src:
                transform = src.transform
                crs       = src.crs
        except Exception:
            transform = None
            crs       = None
    
        if crs is None:
            # no pop-up, just console output
            print(f"[export_training_data] No georeference found on {base_name}; using pixel coords in GeoJSON.")
    
        # 5) Build mask, overlay, COCO annotations, and GeoJSON shapes
        mask    = np.zeros((height, width), dtype=np.uint8)
        overlay = self.full_image.copy()
        thickness = int(self.edge_thickness_slider.get()) if hasattr(self, 'edge_thickness_slider') else 2
    
        coco_annotations = []
        annotation_id = 1
        shape_list = []
    
        for feature_type, pts in self.features:
            if len(pts) < 2:
                continue
    
            # draw on mask & overlay
            pts_np = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            is_poly = (feature_type == "polygon")
            if is_poly:
                closed = pts[:] + ([pts[0]] if pts[0] != pts[-1] else [])
                cv2.fillPoly(mask, [np.array(closed, np.int32).reshape((-1,1,2))], 255)
                cv2.polylines(overlay, [pts_np], True, (0,255,0), 2)
            else:
                cv2.polylines(mask, [pts_np], False, 255, thickness)
                cv2.polylines(overlay, [pts_np], False, (0,255,0), thickness)
    
            # COCO segmentation
            seg = [coord for p in (pts[:] + ([pts[0]] if is_poly else [])) for coord in p]
            xs, ys = zip(*(pts[:] + ([pts[0]] if is_poly else [])))
            bbox = [min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys)]
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
    
            # GeoJSON geometry
            if transform is not None and crs is not None:
                world_pts = [(transform * (x, y))[0:2] for x, y in pts]
            else:
                world_pts = pts  # pixel coords
            shape_list.append(Polygon(world_pts) if is_poly else LineString(world_pts))
    
        # 6) Save mask & overlay
        mask_path    = os.path.join(masks_folder,   os.path.splitext(base_name)[0] + "_mask.png")
        overlay_path = os.path.join(overlays_folder,os.path.splitext(base_name)[0] + "_overlay.png")
        cv2.imwrite(mask_path, mask)
        cv2.imwrite(overlay_path, overlay)
    
        # 7) Write GeoJSON (always)
        geojson_path = os.path.join(geojson_folder, os.path.splitext(base_name)[0] + ".geojson")
        gdf = gpd.GeoDataFrame(geometry=shape_list, crs=crs)
        gdf.to_file(geojson_path, driver="GeoJSON")
    
        # 8) Write COCO JSON
        coco_dict = {
            "images": [{
                "id": image_id,
                "file_name": base_name,
                "width": width,
                "height": height
            }],
            "annotations": coco_annotations,
            "categories": [{
                "id": category_id,
                "name": feature_name
            }]
        }
        coco_path = os.path.join(coco_folder, os.path.splitext(base_name)[0] + ".json")
        with open(coco_path, "w") as f:
            json.dump(coco_dict, f, indent=2)
    
        # final pop-up + console summary
        print(f"[export_training_data] Export complete for {base_name}:")
        print(f"    mask → {mask_path}")
        print(f"    overlay → {overlay_path}")
        print(f"    geojson → {geojson_path}")
        print(f"    coco → {coco_path}")
        messagebox.showinfo(
            "Export Training Data",
            f"Export complete:\n"
            f"- Mask ⇒ {mask_path}\n"
            f"- Overlay ⇒ {overlay_path}\n"
            f"- GeoJSON ⇒ {geojson_path}\n"
            f"- COCO JSON ⇒ {coco_path}"
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

    def f5_action(self):
        if self.mode == "batch":
            return
        if not isinstance(self.full_image, np.ndarray):
            messagebox.showwarning("Warning", "No image loaded!")
            return
        self.do_invert_mask.set(False)     # ← ensure checkbox is unchecked
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
        self.h_low_var.set(low_h)
        low_s = data.get("s_low", 0)
        self.s_low_slider.set(low_s)
        self.s_low_var.set(low_s)
        low_v = data.get("v_low", 0)
        self.v_low_slider.set(low_v)
        self.v_low_var.set(low_v)

        high_h = data.get("h_high", 255)
        self.h_high_slider.set(high_h)
        self.h_high_var.set(high_h)
        high_s = data.get("s_high", 255)
        self.s_high_slider.set(high_s)
        self.s_high_var.set(high_s)
        high_v = data.get("v_high", 255)
        self.v_high_slider.set(high_v)
        self.v_high_var.set(high_v)

        self.use_dual_hsv.set(data.get("use_dual_hsv", False))

        low_h2 = data.get("h2_low", 0)
        self.h2_low_slider.set(low_h2)
        self.h2_low_var.set(low_h2)
        low_s2 = data.get("s2_low", 0)
        self.s2_low_slider.set(low_s2)
        self.s2_low_var.set(low_s2)
        low_v2 = data.get("v2_low", 0)
        self.v2_low_slider.set(low_v2)
        self.v2_low_var.set(low_v2)
        high_h2 = data.get("h2_high", 255)
        self.h2_high_slider.set(high_h2)
        self.h2_high_var.set(high_h2)
        high_s2 = data.get("s2_high", 255)
        self.s2_high_slider.set(high_s2)
        self.s2_high_var.set(high_s2)
        high_v2 = data.get("v2_high", 255)
        self.v2_high_slider.set(high_v2)
        self.v2_high_var.set(high_v2)

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

        messagebox.showinfo(
            "Load Settings", f"Settings loaded from {file_path}")


def main():
    root = ctk.CTk()
    root.withdraw()
    mode = sys.argv[1] if len(sys.argv) > 1 else 'individual'
    win = HSVMaskTool(master=root, mode=mode)
    root.mainloop()


if __name__ == '__main__':
    main()
