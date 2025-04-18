import os
import glob
import threading
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from PIL.PngImagePlugin import PngInfo
import cv2
import numpy as np
import sys
import tifffile
import customtkinter as ctk
import rasterio
import customtkinter as ctk
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")

# --- StdoutRedirector class for redirecting console output into the GUI ---
class StdoutRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        self.text_widget.insert(tk.END, message)
        self.text_widget.see(tk.END)  # auto-scroll

    def flush(self):
        pass  # For compatibility.

# =================== ROI Selector (ScrollZoomBBoxSelector) ===================
class ScrollZoomBBoxSelector(tk.Frame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        master.title("Scrollable & Zoomable ROI Selector")

        # ================ Top Frame: Canvas & Scrollbars ================
        top_frame = tk.Frame(self)
        top_frame.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(top_frame, cursor="cross", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.v_scroll = tk.Scrollbar(top_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.h_scroll = tk.Scrollbar(self, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.h_scroll.pack(side=tk.BOTTOM, fill=tk.X)

        self.canvas.configure(xscrollcommand=self.h_scroll.set, yscrollcommand=self.v_scroll.set)

        # Original image variables
        self.cv_image = None     # OpenCV (BGR)
        self.pil_image = None    # PIL Image (RGB)
        self.tk_image = None     # Tkinter image
        self.scale_factor = 1.0

        # ROI selection state
        self.start_x_display = None
        self.start_y_display = None
        self.rect_id = None
        self.bbox = (0, 0, 0, 0)  # (x, y, w, h)

        # Bind mouse events for ROI selection
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        self.master.bind("<Return>", self.on_enter_key)

        # ================ Bottom Frame: Buttons ================
        button_frame = tk.Frame(self)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        self.load_button = tk.Button(button_frame, text="Load Image", command=self.load_image)
        self.load_button.pack(side=tk.LEFT, padx=5)

        self.zoom_in_button = tk.Button(button_frame, text="Zoom In", command=self.zoom_in)
        self.zoom_in_button.pack(side=tk.LEFT, padx=5)

        self.zoom_out_button = tk.Button(button_frame, text="Zoom Out", command=self.zoom_out)
        self.zoom_out_button.pack(side=tk.LEFT, padx=5)

        self.info_label = tk.Label(button_frame, text="Drag to select ROI; click 'Confirm' when done.")
        self.info_label.pack(side=tk.LEFT, padx=5)

    def load_image(self, file_path=None):
        if file_path is None:
            file_path = filedialog.askopenfilename(
                title="Open Image",
                filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tif *.gif")]
            )
        if file_path:
            self.cv_image = cv2.imread(file_path)
            if self.cv_image is None:
                print("Failed to load image.")
                return
            cv_image_rgb = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
            self.pil_image = Image.fromarray(cv_image_rgb)
            self.scale_factor = 1.0
            self.display_image()
            self.bbox = (0, 0, 0, 0)
            if self.rect_id:
                self.canvas.delete(self.rect_id)
                self.rect_id = None

    def display_image(self):
        if not self.pil_image:
            return
        orig_width, orig_height = self.pil_image.size
        display_width = int(orig_width * self.scale_factor)
        display_height = int(orig_height * self.scale_factor)
        pil_resized = self.pil_image.resize((display_width, display_height), resample=Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(pil_resized)
        self.canvas.delete("all")
        self.canvas.config(scrollregion=(0, 0, display_width, display_height))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

    def zoom_in(self):
        if not self.pil_image:
            return
        self.scale_factor *= 1.25
        self.display_image()

    def zoom_out(self):
        if not self.pil_image:
            return
        self.scale_factor *= 0.8
        if self.scale_factor < 0.1:
            self.scale_factor = 0.1
        self.display_image()

    def on_button_press(self, event):
        self.start_x_display = self.canvas.canvasx(event.x)
        self.start_y_display = self.canvas.canvasy(event.y)
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(
            self.start_x_display, self.start_y_display,
            self.start_x_display, self.start_y_display,
            outline="red", width=2
        )

    def on_move_press(self, event):
        if self.rect_id:
            cur_x = self.canvas.canvasx(event.x)
            cur_y = self.canvas.canvasy(event.y)
            self.canvas.coords(self.rect_id, self.start_x_display, self.start_y_display, cur_x, cur_y)

    def on_button_release(self, event):
        end_x_display = self.canvas.canvasx(event.x)
        end_y_display = self.canvas.canvasy(event.y)
        x1_disp, x2_disp = sorted([self.start_x_display, end_x_display])
        y1_disp, y2_disp = sorted([self.start_y_display, end_y_display])
        x1_orig = x1_disp / self.scale_factor
        y1_orig = y1_disp / self.scale_factor
        x2_orig = x2_disp / self.scale_factor
        y2_orig = y2_disp / self.scale_factor
        w_orig = x2_orig - x1_orig
        h_orig = y2_orig - y1_orig
        self.bbox = (int(x1_orig), int(y1_orig), int(w_orig), int(h_orig))

    def on_enter_key(self, event):
        print("Final bounding box in original image coordinates:", self.bbox)

    def confirm_bbox(self):
        print("Final bounding box in original image coordinates:", self.bbox)
        self.master.focus_set()
        self.master.destroy()

# =================== Timestack Generation Function ===================
def generate_calibrated_timestack(image_files, bbox, resolution_x_m, output_path, progress_callback=None):
    """
    Generate a calibrated time-stack image.
    Crops the ROI (bbox) from each image, averages the pixels across ROI height, and stacks them.
    The output image width equals the ROI width.
    """
    x, y, w, h = bbox
    timestack_lines = []
    time_seconds = []

    # Sort images by timestamp; filename assumed to start with YYYY_MM_DD_HH_MM_SS
    image_files.sort(key=lambda f: datetime.strptime("_".join(os.path.basename(f).split("_")[0:6]),
                                                       "%Y_%m_%d_%H_%M_%S"))
    start_time = datetime.strptime("_".join(os.path.basename(image_files[0]).split("_")[0:6]),
                                   "%Y_%m_%d_%H_%M_%S")
    total = len(image_files)

    for i, file in enumerate(image_files):
        timestamp_str = "_".join(os.path.basename(file).split("_")[0:6])
        timestamp = datetime.strptime(timestamp_str, "%Y_%m_%d_%H_%M_%S")
        elapsed_sec = int((timestamp - start_time).total_seconds())
        time_seconds.append(elapsed_sec)

        if file.lower().endswith(".tif"):
            img = tifffile.imread(file)
        else:
            img = np.array(Image.open(file).convert("RGB"))

        roi = img[y:y+h, x:x+w]
        if roi.ndim == 2:
            roi = np.stack([roi] * 3, axis=-1)
        elif roi.shape[2] > 3:
            roi = roi[:, :, :3]
        line_avg = np.mean(roi, axis=0)
        line_avg = np.round(line_avg).astype(np.uint8)
        timestack_lines.append(line_avg)

        if progress_callback:
            progress_callback(i+1, total)

    pseudo_timestack = np.stack(timestack_lines, axis=0)
    # Output image width equals ROI width
    desired_width = w
    pil_image = Image.fromarray(pseudo_timestack)
    
    if pil_image.width != desired_width:
        resized_image = pil_image.resize((desired_width, pseudo_timestack.shape[0]), resample=Image.NEAREST)
    else:
        resized_image = pil_image
        
    pnginfo = PngInfo()
    pnginfo.add_text("pixel_resolution", f"{resolution_x_m:.6f}")
    pnginfo.add_text("bounding_box", f"{x},{y},{w},{h}")

    resized_image.save(
        output_path,
        format="PNG",
        pnginfo=pnginfo,
    )

    return output_path

def resource_path(relative_path: str) -> str:
    try:
        base_path = sys._MEIPASS  # If running in a PyInstaller .exe
    except Exception:
        base_path = os.path.dirname(__file__)  # Running from source
    return os.path.join(base_path, relative_path)

# =================== TimestackTool GUI ===================
class TimestackTool(ctk.CTkToplevel):
    def __init__(self,master=None):
        super().__init__(master)
        self.title("Time-stacking Tool")
        self.geometry("1200x800")
        
        try:
            self.iconbitmap(resource_path("launch_logo.ico"))
        except Exception as e:
            print("Warning: Could not load window icon:", e)
            
        # Variables for input, output, bbox and resolution
        self.input_folder = tk.StringVar()
        self.output_folder = tk.StringVar()
        self.bbox = None  # (x, y, w, h)
        self.add_bbox_text = tk.BooleanVar(value=False)
        self.bbox_text = tk.StringVar()
        self.default_resolution = 0.25  # fallback resolution in m/pixel
        
        # Resolution row variables
        self.identified_res_var = tk.StringVar(value="Not identified")
        self.add_resolution_manual = tk.BooleanVar(value=False)
        
        # Variable for Batch Processing
        self.batch_folder = tk.StringVar()
        
        # ------------------ Top Panel: Image Display ------------------
        self.top_frame = ctk.CTkFrame(self, height=400)
        self.top_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.image_label = ctk.CTkLabel(self.top_frame, text="")
        self.image_label.pack(padx=10, pady=10)
        
        # ------------------ Bottom Panel: Controls ------------------
        self.bottom_frame = ctk.CTkFrame(self, height=250)
        self.bottom_frame.pack(fill="x", padx=10, pady=10)
        
        # --- Create Separate Panels for Each Configuration Group ---
        # Input & ROI Panel
        self.input_panel = ctk.CTkFrame(self.bottom_frame)
        self.input_panel.pack(fill="x", padx=5, pady=5)
        self.browse_input_button = ctk.CTkButton(self.input_panel, text="Browse Input Folder", command=self.browse_input_folder)
        self.browse_input_button.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.input_folder_label = ctk.CTkLabel(self.input_panel, textvariable=self.input_folder)
        self.input_folder_label.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        self.bbox_button = ctk.CTkButton(self.input_panel, text="Select BBox", command=self.select_bbox)
        self.bbox_button.grid(row=0, column=2, padx=5, pady=5)
        self.bbox_checkbox = ctk.CTkCheckBox(self.input_panel, text="Add bbox as text", variable=self.add_bbox_text, command=self.toggle_bbox_entry)
        self.bbox_checkbox.grid(row=0, column=3, padx=5, pady=5)
        self.bbox_entry = ctk.CTkEntry(self.input_panel, textvariable=self.bbox_text)
        self.bbox_entry.grid(row=0, column=4, padx=5, pady=5)
        self.bbox_entry.configure(state="disabled")
        
        # Resolution Panel
        self.res_panel = ctk.CTkFrame(self.bottom_frame)
        self.res_panel.pack(fill="x", padx=5, pady=5)
        self.identified_res_label = ctk.CTkLabel(self.res_panel, text="Identified resolution:")
        self.identified_res_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.identified_res_value = ctk.CTkLabel(self.res_panel, textvariable=self.identified_res_var)
        self.identified_res_value.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        self.add_res_cb = ctk.CTkCheckBox(self.res_panel, text="Add pixel resolution manually", variable=self.add_resolution_manual, command=self.toggle_resolution_entry)
        self.add_res_cb.grid(row=0, column=2, padx=5, pady=5)
        self.resolution_entry = ctk.CTkEntry(self.res_panel)
        self.resolution_entry.grid(row=0, column=3, padx=5, pady=5)
        self.resolution_entry.configure(state="disabled")
        self.res_unit_label = ctk.CTkLabel(self.res_panel, text="m")
        self.res_unit_label.grid(row=0, column=4, padx=5, pady=5, sticky="w")
        
        # Output Panel (Raw Timestack creation & progress)
        self.output_panel = ctk.CTkFrame(self.bottom_frame)
        self.output_panel.pack(fill="x", padx=5, pady=5)
        self.select_output_button = ctk.CTkButton(self.output_panel, text="Select Output Folder", command=self.select_output_folder)
        self.select_output_button.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.output_folder_label = ctk.CTkLabel(self.output_panel, textvariable=self.output_folder)
        self.output_folder_label.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        self.create_timestack_button = ctk.CTkButton(self.output_panel, text="Create Raw Timestack", command=self.create_timestack)
        self.create_timestack_button.grid(row=0, column=2, padx=5, pady=5)
        self.progress_bar = ctk.CTkProgressBar(self.output_panel)
        self.progress_bar.grid(row=0, column=3, padx=5, pady=5)
        self.progress_bar.set(0)
        self.progress_text_label = ctk.CTkLabel(self.output_panel, text="")
        self.progress_text_label.grid(row=0, column=4, padx=5, pady=5, sticky="w")
        
        # Batch Process Panel
        self.batch_panel = ctk.CTkFrame(self.bottom_frame)
        self.batch_panel.pack(fill="x", padx=5, pady=5)
        self.select_batch_folder_button = ctk.CTkButton(self.batch_panel, text="Select Batch Folder", command=self.browse_batch_folder)
        self.select_batch_folder_button.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.batch_folder_label = ctk.CTkLabel(self.batch_panel, textvariable=self.batch_folder)
        self.batch_folder_label.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        self.batch_process_button = ctk.CTkButton(self.batch_panel, text="Batch Process", command=self.batch_process)
        self.batch_process_button.grid(row=0, column=2, padx=5, pady=5)
        
        # --- Console Panel at the bottom ---
        self.console_frame = ctk.CTkFrame(self)
        self.console_frame.pack(side="bottom", fill="both", expand=False, padx=10, pady=10)
        self.console_text = tk.Text(self.console_frame, wrap="word", height=10)
        self.console_text.pack(fill="both", expand=True, padx=5, pady=5)
        # Redirect stdout and stderr to our console widget
        self.stdout_redirector = StdoutRedirector(self.console_text)
        sys.stdout = self.stdout_redirector
        sys.stderr = self.stdout_redirector
        print("Here you may see console outputs\n")
        
    def toggle_bbox_entry(self):
        if self.add_bbox_text.get():
            self.bbox_entry.configure(state="normal")
        else:
            self.bbox_entry.configure(state="disabled")
            
    def toggle_resolution_entry(self):
        if self.add_resolution_manual.get():
            self.resolution_entry.configure(state="normal")
        else:
            self.resolution_entry.configure(state="disabled")
        
    def browse_input_folder(self):
        folder_selected = filedialog.askdirectory(title="Select Input Folder with Images")
        if folder_selected:
            self.input_folder.set(folder_selected)
        
    def select_output_folder(self):
        folder_selected = filedialog.askdirectory(title="Select Output Folder for Raw Timestack")
        if folder_selected:
            self.output_folder.set(folder_selected)
            
    def select_bbox(self):
        folder = self.input_folder.get()
        if not folder:
            messagebox.showerror("Error", "Please select an input folder first.")
            return
        # Get one image file (first supported image)
        supported_exts = ("*.jpg", "*.jpeg", "*.png", "*.tif")
        image_files = []
        for ext in supported_exts:
            image_files.extend(glob.glob(os.path.join(folder, ext)))
        if not image_files:
            messagebox.showerror("Error", "No supported image files found in folder.")
            return
        image_file = image_files[0]
        selector_window = tk.Toplevel(self)
        selector_window.title("Select Bounding Box")
        selector = ScrollZoomBBoxSelector(selector_window)
        selector.pack(fill="both", expand=True)
        selector.load_image(image_file)
        
        confirm_btn = ctk.CTkButton(selector_window, text="Confirm BBox",
                                    command=lambda: self.save_bbox(selector, selector_window))
        confirm_btn.pack(pady=10)
        
    def save_bbox(self, selector, window):
        bbox = selector.bbox
        if bbox == (0, 0, 0, 0):
            messagebox.showwarning("Warning", "No bounding box selected.")
        else:
            self.bbox = bbox
            self.bbox_text.set(",".join(map(str, self.bbox)))
            messagebox.showinfo("BBox Selected", f"Bounding box: {self.bbox}")
            folder = self.input_folder.get()
            tif_files = glob.glob(os.path.join(folder, "*.tif"))
            if tif_files:
                try:
                    with rasterio.open(tif_files[0]) as src:
                        res = abs(src.transform.a)
                except Exception as e:
                    res = self.default_resolution
            else:
                res = self.default_resolution
            self.identified_res_var.set(f"{res:.3f} m")
        window.destroy()
        
    def create_timestack(self):
        folder = self.input_folder.get()
        out_folder = self.output_folder.get()
        if not folder:
            messagebox.showerror("Error", "Please select an input folder first.")
            return
        if not out_folder:
            messagebox.showerror("Error", "Please select an output folder for raw timestack.")
            return
        if self.add_bbox_text.get() and self.bbox_text.get().strip():
            try:
                parts = self.bbox_text.get().split(",")
                if len(parts) != 4:
                    raise ValueError
                self.bbox = tuple(int(p.strip()) for p in parts)
            except Exception as e:
                messagebox.showerror("Error", "Invalid bbox format. Please enter as x,y,w,h")
                return
        if not self.bbox or self.bbox == (0, 0, 0, 0):
            messagebox.showerror("Error", "Please identify a valid bounding box first.")
            return
        
        supported_exts = ("*.jpg", "*.jpeg", "*.png", "*.tif")
        image_files = []
        for ext in supported_exts:
            image_files.extend(glob.glob(os.path.join(folder, ext)))
        if not image_files:
            messagebox.showerror("Error", "No supported image files found in input folder.")
            return
        
        x, y, w, h = self.bbox
        desired_width = w
        
        if self.add_resolution_manual.get():
            try:
                resolution_x_m = float(self.resolution_entry.get())
            except Exception as e:
                messagebox.showerror("Error", "Invalid manual resolution. Please enter a numeric value.")
                return
        else:
            try:
                resolution_x_m = float(self.identified_res_var.get().split()[0])
            except Exception as e:
                resolution_x_m = self.default_resolution
        
        image_files.sort(key=lambda f: datetime.strptime("_".join(os.path.basename(f).split("_")[0:6]), "%Y_%m_%d_%H_%M_%S"))
        first_file = os.path.basename(image_files[0])
        parts = first_file.split("_")
        output_timestamp = "_".join(parts[0:5])
        output_name = output_timestamp + "_raw_timestack.png"
        output_path = os.path.join(out_folder, output_name)
        
        def run_timestack():
            try:
                # Update progress with both current count and total number of images
                def update_progress(current, total):
                    fraction = current / total
                    self.progress_bar.set(fraction)
                    self.progress_text_label.configure(text=f"{current} / {total}")
                generated_path = generate_calibrated_timestack(image_files, self.bbox, resolution_x_m, output_path, progress_callback=update_progress)
                messagebox.showinfo("Success", f"Calibrated timestack image saved to: {generated_path}")
                pil_img = Image.open(generated_path)
                ts_image = ctk.CTkImage(light_image=pil_img, size=(800, 600))
                self.image_label.configure(image=ts_image)
                self.image_label.image = ts_image
            except Exception as e:
                messagebox.showerror("Error", str(e))
            finally:
                self.progress_bar.set(0)
                self.progress_text_label.configure(text="")
        threading.Thread(target=run_timestack).start()
        
    def browse_batch_folder(self):
        folder_selected = filedialog.askdirectory(title="Select Main Batch Folder")
        if folder_selected:
            self.batch_folder.set(folder_selected)
    
    def batch_process(self):
        main_batch_folder = self.batch_folder.get()
        out_folder = self.output_folder.get()
        if not main_batch_folder:
            messagebox.showerror("Error", "Please select a batch folder first.")
            return
        if not out_folder:
            messagebox.showerror("Error", "Please select an output folder first.")
            return
        if not self.bbox or self.bbox == (0, 0, 0, 0):
            messagebox.showerror("Error", "Please select a valid bounding box first.")
            return
        
        if self.add_resolution_manual.get():
            try:
                resolution_x_m = float(self.resolution_entry.get())
            except:
                messagebox.showerror("Error", "Invalid manual resolution. Please enter a numeric value.")
                return
        else:
            try:
                resolution_x_m = float(self.identified_res_var.get().split()[0])
            except:
                resolution_x_m = self.default_resolution
        
        subfolders = [os.path.join(main_batch_folder, name) for name in os.listdir(main_batch_folder)
                      if os.path.isdir(os.path.join(main_batch_folder, name))]
        if not subfolders:
            messagebox.showerror("Error", "No subfolders found in the selected batch folder.")
            return
        
        total_folders = len(subfolders)
        
        def process_batch():
            processed = 0
            self.progress_text_label.configure(text=f"{processed} / {total_folders}")
            for idx, subfolder in enumerate(subfolders, start=1):
                supported_exts = ("*.jpg", "*.jpeg", "*.png", "*.tif")
                image_files = []
                for ext in supported_exts:
                    image_files.extend(glob.glob(os.path.join(subfolder, ext)))
                if not image_files:
                    processed += 1
                    self.progress_bar.set(processed / total_folders)
                    self.progress_text_label.configure(text=f"{processed} / {total_folders}")
                    continue

                try:
                    image_files.sort(key=lambda f: datetime.strptime("_".join(os.path.basename(f).split("_")[0:6]),
                                                                       "%Y_%m_%d_%H_%M_%S"))
                except Exception as e:
                    processed += 1
                    self.progress_bar.set(processed / total_folders)
                    self.progress_text_label.configure(text=f"{processed} / {total_folders}")
                    continue

                first_file = os.path.basename(image_files[0])
                parts = first_file.split("_")
                output_timestamp = "_".join(parts[0:5])
                output_name = f"{idx:03d}_{output_timestamp}_raw_timestack.png"
                output_path = os.path.join(out_folder, output_name)
                try:
                    generate_calibrated_timestack(image_files, self.bbox, resolution_x_m, output_path)
                except Exception as e:
                    print(f"Error processing folder {subfolder}: {e}")
                processed += 1
                self.progress_bar.set(processed / total_folders)
                self.progress_text_label.configure(text=f"{processed} / {total_folders}")
            messagebox.showinfo("Batch Process Completed", f"Processed {processed} subfolders.")
            self.progress_bar.set(0)
            self.progress_text_label.configure(text="")
        
        threading.Thread(target=process_batch).start()

def main():
    # single hidden root for all CTkToplevels
    root = ctk.CTk()
    root.withdraw()
    win = TimestackTool(master=root)
    win.mainloop()


if __name__ == "__main__":
    main()
