import sys
import os
import customtkinter as ctk
from PIL import Image

# %% Import the refactored submodule classes:
from pixel_to_gcp import PixelToGCPWindow
from hsv_mask import HSVMaskTool
from dem_generator import CreateDemWindow
from georef import GeoReferenceModule
from homography import CreateHomographyMatrixWindow
from raw_timestacker import TimestackTool
from wave_runup import WaveRunUpCalculator

# 1) Resource helper for PyInstaller or direct script
def resource_path(relative_path: str) -> str:
    try:
        base_path = sys._MEIPASS 
    except Exception:
        base_path = os.path.dirname(__file__)  
    return os.path.join(base_path, relative_path)

# 2) Splash screen
def show_splash(duration_ms=1000):
    splash = ctk.CTk()
    splash.iconbitmap(resource_path("launch_logo.ico"))
    splash.title("Loading...")
    splash.geometry("400x280")
    splash.resizable(False, False)

    # Center on screen
    sw, sh = splash.winfo_screenwidth(), splash.winfo_screenheight()
    splash.geometry(f"400x280+{(sw-400)//2}+{(sh-280)//2}")

    try:
        img_path = resource_path("splash_image.png")
        pil_img = Image.open(img_path).resize((400, 280), Image.Resampling.LANCZOS)
        ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(400, 280))
        ctk.CTkLabel(splash, image=ctk_img, text="").pack()
    except Exception:
        ctk.CTkLabel(splash, text="Loading...", font=("Arial", 18, "bold")).pack(expand=True)

    splash.after(duration_ms, splash.destroy)
    splash.mainloop()

# 3) Set CustomTkinter appearance globally
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")

# 4) Launcher window
def launcher_window():
    def open_pixel_to_gcp():
        PixelToGCPWindow(master=dialog)

    def open_hsv_individual():
        HSVMaskTool(master=dialog, mode="individual")

    def open_hsv_ml():
        HSVMaskTool(master=dialog, mode="ml")

    def open_hsv_batch():
        HSVMaskTool(master=dialog, mode="batch")

    def open_dem():
        CreateDemWindow(master=dialog)

    def open_georef():
        GeoReferenceModule(master=dialog)

    def open_homography():
        CreateHomographyMatrixWindow(master=dialog)

    def open_timestack():
        TimestackTool(master=dialog)

    def open_wave_run():
        WaveRunUpCalculator(master=dialog)

    def on_close():
        dialog.destroy()
        sys.exit(0)

    dialog = ctk.CTk()
    dialog.title("GeoCamPal")
    dialog.geometry("600x600")
    dialog.resizable(False, False)
    dialog.protocol("WM_DELETE_WINDOW", on_close)
    dialog.iconbitmap(resource_path("launch_logo.ico"))

    frame = ctk.CTkFrame(dialog)
    frame.pack(fill="both", expand=True, padx=20, pady=20)

    # Logo
    try:
        logo = Image.open(resource_path("launch_logo.png"))
        logo_img = ctk.CTkImage(light_image=logo, dark_image=logo, size=(100, 70))
        ctk.CTkLabel(frame, image=logo_img, text="").pack(pady=(0,20))
    except Exception:
        pass

    ctk.CTkLabel(frame, text="Select a tool to open", font=("Arial", 18, "bold")).pack(pady=(0,15))

    # (C) Georeferencing
    ctk.CTkLabel(frame, text="Georeferencing", font=("Arial", 14, "bold")).pack(anchor="w")
    geo_frame = ctk.CTkFrame(frame, fg_color="transparent")
    geo_frame.pack(fill="x", pady=5)
    ctk.CTkButton(geo_frame, text="Pixel to GCP", command=open_pixel_to_gcp).pack(side="left", padx=5)
    ctk.CTkButton(geo_frame, text="Homography", command=open_homography).pack(side="left", padx=5)
    ctk.CTkButton(geo_frame, text="Georef Images", command=open_georef).pack(side="left", padx=5)

    # (D) HSV Tool
    ctk.CTkLabel(frame, text="Feature Identifier tool", font=("Arial", 14, "bold")).pack(anchor="w", pady=(10,0))
    hsv_frame = ctk.CTkFrame(frame, fg_color="transparent")
    hsv_frame.pack(fill="x", pady=5)
    ctk.CTkButton(hsv_frame, text="Individual Analysis", command=open_hsv_individual).pack(side="left", padx=5)
    ctk.CTkButton(hsv_frame, text="Machine Learning",  command=open_hsv_ml).pack(side="left", padx=5)
    ctk.CTkButton(hsv_frame, text="Batch Process",       command=open_hsv_batch).pack(side="left", padx=5)

    # (E) DEM Generator
    ctk.CTkLabel(frame, text="DEM Generator", font=("Arial", 14, "bold")).pack(anchor="w", pady=(10,0))
    ctk.CTkButton(frame, text="Create DEM", command=open_dem).pack(fill="x", pady=5)

    # (F) Time-stacking
    ctk.CTkLabel(frame, text="Time-stacking", font=("Arial", 14, "bold")).pack(anchor="w", pady=(10,0))
    ts_frame = ctk.CTkFrame(frame, fg_color="transparent")
    ts_frame.pack(fill="x", pady=5)
    ctk.CTkButton(ts_frame, text="Raw Timestack Image", command=open_timestack).pack(side="left", padx=5)
    ctk.CTkButton(ts_frame, text="Wave Run Up",          command=open_wave_run).pack(side="left", padx=5)

    # Footer
    footer = ctk.CTkLabel(frame,
        text="Application created by Clayton Soares\nUniversity of Kiel",
        font=("Arial", 10)
    )
    footer.pack(side="bottom", anchor="w", pady=(10,0))

    dialog.mainloop()

# 5) MAIN
if __name__ == "__main__":
    show_splash(duration_ms=1000)
    launcher_window()
