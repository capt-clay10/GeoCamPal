import os
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import pandas as pd
from PIL import Image, ImageTk
import sys
import utm
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")

def convert_to_utm(lat, lon):
    """
    Convert latitude and longitude to UTM coordinates.
    utm.from_latlon returns a tuple: (easting, northing, zone number, zone letter).
    We only extract the easting and northing.
    """
    easting, northing, zone_number, zone_letter = utm.from_latlon(lat, lon)
    return easting, northing

def resource_path(relative_path: str) -> str:
    try:
        base_path = sys._MEIPASS  # If running in a PyInstaller .exe
    except Exception:
        base_path = os.path.dirname(__file__)  # Running directly from source
    return os.path.join(base_path, relative_path)


class StdoutRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        # Insert the message into the text widget and scroll to the end.
        self.text_widget.insert(tk.END, message)
        self.text_widget.see(tk.END)

    def flush(self):
        pass  # This is needed for Python's IO requirements.


class PixelToGCPWindow(ctk.CTkToplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("Pixel to GCP Tool")
        self.geometry("1200x800")
        
        try:
            self.iconbitmap(resource_path("launch_logo.ico"))
        except Exception as e:
            print("Warning: Could not load window icon:", e)
    
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
        self.selected_points = {}  # {filename: (x_full, y_full)}
    
        self.scale_factor = 1.0    # Zoom level for main image
        self.overview_scale = 0.2  # Overview image scale
        self.current_pil_img = None
        
        # New variable for UTM conversion decision.
        self.convert_to_utm_var = tk.BooleanVar(value=True)
    
        # =============================
        # TOP SECTION: IMAGES + CONSOLE
        # =============================
        top_section = ctk.CTkFrame(self)
        top_section.pack(side="top", fill="both", expand=True, padx=5, pady=5)
    
        # 1) Main Image Panel (left)
        self.main_image_frame = ctk.CTkFrame(top_section)
        self.main_image_frame.pack(side="left", fill="both", expand=True)
    
        # Create canvas + scrollbars in main_image_frame
        self.scroll_x = tk.Scrollbar(self.main_image_frame, orient=tk.HORIZONTAL)
        self.scroll_x.pack(side="bottom", fill="x")
        self.scroll_y = tk.Scrollbar(self.main_image_frame, orient=tk.VERTICAL)
        self.scroll_y.pack(side="right", fill="y")
        self.main_canvas = tk.Canvas(
            self.main_image_frame,
            xscrollcommand=self.scroll_x.set,
            yscrollcommand=self.scroll_y.set,
            bg="gray"
        )
        self.main_canvas.pack(side="left", fill="both", expand=True)
        self.scroll_x.config(command=self.main_canvas.xview)
        self.scroll_y.config(command=self.main_canvas.yview)
    
        # Bind events for zooming, scrolling, and clicking
        self.main_canvas.bind("<Button-1>", self.on_main_click)
        self.main_canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.main_canvas.bind("<Shift-MouseWheel>", self._on_shift_mousewheel)
        # Bind scrollbar motion events to update the overview dynamically
        self.scroll_x.bind("<B1-Motion>", lambda event: self._update_overview())
        self.scroll_x.bind("<ButtonRelease-1>", lambda event: self._update_overview())
        self.scroll_y.bind("<B1-Motion>", lambda event: self._update_overview())
        self.scroll_y.bind("<ButtonRelease-1>", lambda event: self._update_overview())
    
        # 2) Overview Panel (center)
        self.overview_frame = ctk.CTkFrame(top_section, width=300)
        self.overview_frame.pack(side="left", fill="y")
        self.overview_canvas = tk.Canvas(self.overview_frame, width=300, height=200, bg="white")
        self.overview_canvas.pack(fill="both", expand=True)
    
        # 3) Console Panel (right)
        console_frame = ctk.CTkFrame(top_section, width=300)
        console_frame.pack(side="left", fill="y")
        self.console_text = tk.Text(console_frame, wrap="word", width=40)
        self.console_text.pack(fill="both", expand=True, padx=5, pady=5)
        self.stdout_redirector = StdoutRedirector(self.console_text)
        sys.stdout = self.stdout_redirector
        sys.stderr = self.stdout_redirector
        print("Here you may see console outputs\n")
    
        # =============================
        # MIDDLE SECTION: INSTRUCTIONS
        # =============================
        instructions_panel = ctk.CTkFrame(self)
        instructions_panel.pack(side="top", fill="x", padx=5, pady=5)
        instructions_text = (
            "Instructions:\n"
            " • Left-click on the main image to select a pixel (saved in full-resolution coords).\n"
            " • Press + to zoom in; press - to zoom out.\n"
            " • Use the mouse wheel to scroll vertically; hold Shift + mouse wheel to scroll horizontally.\n"
            " • Press Enter to advance to the next image.\n"
            " • The small overview image in the center shows your location in red."
        )
        instructions_label = ctk.CTkLabel(instructions_panel, text=instructions_text, justify="left")
        instructions_label.pack(side="left", padx=10, pady=5)
    
        # =============================
        # BOTTOM SECTION: CONFIG PANEL (Each control in its own sub-panel)
        # =============================
        config_panel = ctk.CTkFrame(self)
        config_panel.pack(side="bottom", fill="x", padx=5, pady=5)
    
        # Sub-panel for Image Folder
        pnl_img_folder = ctk.CTkFrame(config_panel)
        pnl_img_folder.pack(side="top", fill="x", padx=5, pady=2)
        ctk.CTkButton(pnl_img_folder, text="Browse Image Folder", command=self.browse_image_folder).pack(side="left")
        self.label_image_folder = ctk.CTkLabel(pnl_img_folder, text="No folder selected")
        self.label_image_folder.pack(side="left", padx=5)
    
        # Sub-panel for GCP File and UTM conversion choice
        pnl_gcp_file = ctk.CTkFrame(config_panel)
        pnl_gcp_file.pack(side="top", fill="x", padx=5, pady=2)
        ctk.CTkButton(pnl_gcp_file, text="Browse GCP File", command=self.browse_gcp_file).pack(side="left")
        self.label_gcp_file = ctk.CTkLabel(pnl_gcp_file, text="No file selected")
        self.label_gcp_file.pack(side="left", padx=5)
        
        self.label_gcp_note = ctk.CTkLabel(
            pnl_gcp_file,
            text="Required columns: latitude, longitude, GCP_ID",
            fg_color="white",       # white background
            text_color="black",     # black text
            corner_radius=0         # optional: makes the background solid rectangle
        )
        self.label_gcp_note.pack(side="left", padx=5)

        
        
        self.checkbox_utm = ctk.CTkCheckBox(pnl_gcp_file, text="Convert to UTM", variable=self.convert_to_utm_var)
        self.checkbox_utm.pack(side="left", padx=5)
    
        # Sub-panel for Bad GCPs
        pnl_bad_gcps = ctk.CTkFrame(config_panel)
        pnl_bad_gcps.pack(side="top", fill="x", padx=5, pady=2)
        ctk.CTkLabel(pnl_bad_gcps, text="Bad GCPs (comma sep):").pack(side="left")
        self.entry_bad_gcps = ctk.CTkEntry(pnl_bad_gcps)
        self.entry_bad_gcps.pack(side="left", fill="x", expand=True, padx=5)
    
        # Sub-panel for Output Folder
        pnl_output_folder = ctk.CTkFrame(config_panel)
        pnl_output_folder.pack(side="top", fill="x", padx=5, pady=2)
        ctk.CTkButton(pnl_output_folder, text="Output Folder", command=self.browse_output_folder).pack(side="left")
        self.label_output_folder = ctk.CTkLabel(pnl_output_folder, text="No folder selected")
        self.label_output_folder.pack(side="left", padx=5)
    
        # Sub-panel for Output CSV Name
        pnl_csv_name = ctk.CTkFrame(config_panel)
        pnl_csv_name.pack(side="top", fill="x", padx=5, pady=2)
        ctk.CTkLabel(pnl_csv_name, text="Output CSV Name:").pack(side="left")
        self.entry_output_filename = ctk.CTkEntry(pnl_csv_name)
        self.entry_output_filename.pack(side="left", fill="x", expand=True, padx=5)
    
        # Sub-panel for the Start Button
        pnl_start = ctk.CTkFrame(config_panel)
        pnl_start.pack(side="top", fill="x", padx=5, pady=2)
        ctk.CTkButton(pnl_start, text="Start Process", command=self.start_process).pack(side="left")
    
        # BIND keys to the entire window (zoom + next image)
        self.bind("<Return>", self.next_image)
        self.bind("<plus>", self.zoom_in)
        self.bind("<minus>", self.zoom_out)
        self.bind("<KP_Add>", self.zoom_in)
        self.bind("<KP_Subtract>", self.zoom_out)
    
    # ---------------
    # LOGGING TO CONSOLE
    # ---------------
    def log(self, msg):
        self.console_text.insert(tk.END, msg + "\n")
        self.console_text.see(tk.END)
    
    # ---------------
    # BROWSE / CONFIG
    # ---------------
    def browse_image_folder(self):
        folder = filedialog.askdirectory(title="Select Image Folder")
        if folder:
            self.image_folder = folder
            self.label_image_folder.configure(text=folder)
            self.log(f"Image folder selected: {folder}")
    
    def browse_gcp_file(self):
        path = filedialog.askopenfilename(title="Select GCP CSV", filetypes=[("CSV", "*.csv"), ("All Files", "*.*")])
        if path:
            self.gcp_file = path
            self.label_gcp_file.configure(text=os.path.basename(path))
            self.log(f"GCP file selected: {path}")
    
    def browse_output_folder(self):
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_folder = folder
            self.label_output_folder.configure(text=folder)
            self.log(f"Output folder selected: {folder}")
    
    def start_process(self):
        # Parse bad GCPs
        bad_str = self.entry_bad_gcps.get().strip()
        if bad_str:
            try:
                self.bad_gcp_list = [int(x.strip()) for x in bad_str.split(',') if x.strip() != ""]
            except Exception as e:
                messagebox.showerror("Error", f"Invalid bad GCPs: {e}")
                return
        else:
            self.bad_gcp_list = []
    
        if not self.image_folder:
            messagebox.showerror("Error", "No image folder selected.")
            return
    
        # Build image_list from folder
        all_files = os.listdir(self.image_folder)
        good_images = []
        for f in all_files:
            if f.lower().endswith((".bmp", ".jpg", ".jpeg", ".png")):
                try:
                    # e.g. "GCP_12_cam1.bmp"
                    parts = f.split("_")
                    gcp_num = int(parts[1])
                    if gcp_num not in self.bad_gcp_list:
                        good_images.append(f)
                except:
                    continue
        self.image_list = sorted(good_images, key=lambda x: int(x.split("_")[1]))
        if not self.image_list:
            messagebox.showerror("Error", "No valid images found after filtering.")
            return
        self.current_index = 0
        self.log(f"Found {len(self.image_list)} images for processing.")
    
        # Load GCP file if provided
        if self.gcp_file:
            try:
                df = pd.read_csv(self.gcp_file)
                # Check that required columns exist
                required_cols = ['latitude', 'longitude', 'GCP_ID']
                if not all(col in df.columns for col in required_cols):
                    messagebox.showerror("Error", f"GCP file must contain columns: {required_cols}")
                    return
                df['gcp_number'] = df['GCP_ID'].apply(lambda x: int(x.split('_')[-1]))
                df = df[~df['gcp_number'].isin(self.bad_gcp_list)]
                if self.convert_to_utm_var.get():
                    # Perform UTM conversion using the helper function.
                    df[['easting', 'northing']] = df.apply(
                        lambda row: pd.Series(convert_to_utm(row['latitude'], row['longitude'])),
                        axis=1
                    )
                else:
                    # If not converting, use latitude and longitude directly.
                    df['easting'] = df['latitude']
                    df['northing'] = df['longitude']
                self.gcp_df = df
                self.log(f"GCP file loaded: {os.path.basename(self.gcp_file)}")
            except Exception as e:
                self.log(f"Failed to load GCP file: {e}")
                self.gcp_df = None
    
        self.show_image(self.image_list[self.current_index])
        self.focus_set()
    
    # ---------------
    # SHOW IMAGE
    # ---------------
    def show_image(self, filename):
        path = os.path.join(self.image_folder, filename)
        try:
            pil_img = Image.open(path)
        except Exception as e:
            messagebox.showerror("Error", f"Cannot open image: {path}\n{e}")
            return
        self.current_pil_img = pil_img
        self.scale_factor = 1.0
        self._update_main_canvas()
        self._update_overview()
        idx = self.current_index
        total = len(self.image_list)
        self.title(f"Pixel -> GCP: {filename} ({idx+1}/{total})")
    
    def _update_main_canvas(self):
        if not self.current_pil_img:
            return
        w = int(self.current_pil_img.width * self.scale_factor)
        h = int(self.current_pil_img.height * self.scale_factor)
        disp_img = self.current_pil_img.resize((w, h), Image.Resampling.LANCZOS)
        self.tk_main_img = ImageTk.PhotoImage(disp_img)
        self.main_canvas.delete("all")
        self.main_canvas.config(scrollregion=(0, 0, w, h))
        self.main_canvas.create_image(0, 0, anchor='nw', image=self.tk_main_img)
    
    def _update_overview(self):
        self.overview_canvas.delete("all")
        if not self.current_pil_img:
            return
        ow = int(self.current_pil_img.width * self.overview_scale)
        oh = int(self.current_pil_img.height * self.overview_scale)
        ov_img = self.current_pil_img.resize((ow, oh), Image.Resampling.LANCZOS)
        self.tk_overview_img = ImageTk.PhotoImage(ov_img)
        self.overview_canvas.config(width=ow, height=oh)
        self.overview_canvas.create_image(0, 0, anchor='nw', image=self.tk_overview_img)
        xview0, xview1 = self.main_canvas.xview()  # fractions of total width
        yview0, yview1 = self.main_canvas.yview()  # fractions of total height
        w_disp = int(self.current_pil_img.width * self.scale_factor)
        h_disp = int(self.current_pil_img.height * self.scale_factor)
        left = w_disp * xview0
        right = w_disp * xview1
        top = h_disp * yview0
        bottom = h_disp * yview1
        scale_ratio = self.overview_scale / self.scale_factor
        ov_left = left * scale_ratio
        ov_right = right * scale_ratio
        ov_top = top * scale_ratio
        ov_bottom = bottom * scale_ratio
        self.overview_canvas.create_rectangle(
            ov_left, ov_top, ov_right, ov_bottom,
            outline="red", width=2
        )
    
    # ---------------
    # CLICK & SCROLL
    # ---------------
    def on_main_click(self, event):
        x_canvas = self.main_canvas.canvasx(event.x)
        y_canvas = self.main_canvas.canvasy(event.y)
        x_full = x_canvas / self.scale_factor
        y_full = y_canvas / self.scale_factor
        filename = self.image_list[self.current_index]
        self.selected_points[filename] = (x_full, y_full)
        self.log(f"Selected {filename}: (full={x_full:.2f}, {y_full:.2f})")
    
    def _on_mousewheel(self, event):
        direction = -1 if event.delta > 0 else 1
        self.main_canvas.yview_scroll(direction, "units")
        self._update_overview()
    
    def _on_shift_mousewheel(self, event):
        direction = -1 if event.delta > 0 else 1
        self.main_canvas.xview_scroll(direction, "units")
        self._update_overview()
    
    # ---------------
    # ZOOM
    # ---------------
    def zoom_in(self, event=None):
        old_xview = self.main_canvas.xview()
        old_yview = self.main_canvas.yview()
        self.scale_factor *= 1.5
        self._update_main_canvas()
        self.main_canvas.xview_moveto(old_xview[0])
        self.main_canvas.yview_moveto(old_yview[0])
        self._update_overview()
    
    def zoom_out(self, event=None):
        old_xview = self.main_canvas.xview()
        old_yview = self.main_canvas.yview()
        self.scale_factor = max(self.scale_factor / 1.5, 0.1)
        self._update_main_canvas()
        self.main_canvas.xview_moveto(old_xview[0])
        self.main_canvas.yview_moveto(old_yview[0])
        self._update_overview()
    
    # ---------------
    # NEXT IMAGE
    # ---------------
    def next_image(self, event=None):
        self.current_index += 1
        if self.current_index >= len(self.image_list):
            self.log("All images processed. Saving CSV and closing.")
            self.save_output_csv()
            self.destroy()
        else:
            self.show_image(self.image_list[self.current_index])
    
    # ---------------
    # SAVE OUTPUT
    # ---------------
    def save_output_csv(self):
        if not self.output_folder:
            messagebox.showerror("Error", "No output folder selected.")
            return
        filename = self.entry_output_filename.get().strip()
        if not filename:
            filename = "pixel_gcp_output"
        output_path = os.path.join(self.output_folder, f"{filename}.csv")
        rows = []
        for fname, (px_full, py_full) in self.selected_points.items():
            parts = fname.split("_")
            try:
                gcp_num = int(parts[1])
            except:
                continue
            camera_part = parts[2].split('.')[0] if len(parts) > 2 else "0"
            if self.gcp_df is None:
                continue
            match = self.gcp_df.loc[self.gcp_df['gcp_number'] == gcp_num]
            if match.empty:
                continue
            row = match.iloc[0]
            gcp_id = row['GCP_ID']
            real_x = row['easting']
            real_y = row['northing']
            real_z = row['elevation'] if 'elevation' in row and pd.notna(row['elevation']) else 0
            rows.append({
                "Image_name": fname,
                "Pixel_X": px_full,
                "Pixel_Y": py_full,
                "gcp_id": gcp_id,
                "camera": camera_part,
                "Real_X": real_x,
                "Real_Y": real_y,
                "Real_Z": real_z
            })
        df = pd.DataFrame(rows, columns=[
            "Image_name", "Pixel_X", "Pixel_Y", "gcp_id", "camera", "Real_X", "Real_Y", "Real_Z"
        ])
        try:
            df.to_csv(output_path, index=False)
            self.log(f"CSV saved to {output_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save CSV: {e}")


def main():
    root = ctk.CTk()
    root.withdraw()
    win = PixelToGCPWindow(master=root)
    win.focus_force()
    root.mainloop()

if __name__ == "__main__":
    main()
