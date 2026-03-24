import sys
import os
import customtkinter as ctk
from PIL import Image

from pixel_to_gcp import PixelToGCPWindow
from feature_identifier import FeatureIdentifier
from dem_generator import CreateDemWindow
from georef import GeoReferenceModule
from homography import CreateHomographyMatrixWindow
from raw_timestacker import TimestackTool
from wave_runup import WaveRunUpCalculator

from fov_generator import FOVGeneratorWindow
from lens_correction import LensCorrectionWindow
from harmonise_images import HarmoniseImagesWindow

from exploration import TimeSeriesExplorerWindow
from profile_tool import ProfileHovmullerWindow
from colour_explorer import ColorSpaceExplorerWindow

# 1) Resource helper for PyInstaller or direct script
def resource_path(relative_path: str) -> str:
    try:
        base_path = sys._MEIPASS  # PyInstaller temp folder
    except Exception:
        base_path = os.path.dirname(__file__)  # Running from source
    return os.path.join(base_path, relative_path)

# 2) Optional splash screen
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
        ctk.CTkLabel(splash, text="Loading...", font=("Serif", 18, "bold")).pack(expand=True)

    splash.after(duration_ms, splash.destroy)
    splash.mainloop()

# 3) Set CustomTkinter appearance globally
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")

# 4) Launcher window
def launcher_window():
    # --- Pre-prep tools ---
    def open_fov():
        FOVGeneratorWindow(master=dialog)

    def open_lens():
        LensCorrectionWindow(master=dialog)

    def open_harmonise():
        HarmoniseImagesWindow(master=dialog)

    # --- Georeferencing ---
    def open_pixel_to_gcp():
        PixelToGCPWindow(master=dialog)

    def open_feature_individual():
        FeatureIdentifier(master=dialog, mode="individual")

    def open_feature_ml():
        FeatureIdentifier(master=dialog, mode="ml")

    def open_feature_batch():
        FeatureIdentifier(master=dialog, mode="batch")

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

    # --- Data exploration ---
    def open_timeseries():
        TimeSeriesExplorerWindow(master=dialog)

    def open_color_explorer():
        ColorSpaceExplorerWindow(master=dialog)

    def open_profile():
        ProfileHovmullerWindow(master=dialog)

    def on_close():
        dialog.destroy()
        sys.exit(0)

    dialog = ctk.CTk()
    dialog.title("GeoCamPal")
    dialog.geometry("600x700")
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

    ctk.CTkLabel(frame, text="Select a tool", font=("Serif", 18, "bold")) \
        .pack(pady=(0,15))

    # (A) Pre-prep Tools
    ctk.CTkLabel(frame, text="Pre-prep Tools", font=("Serif", 14, "bold")) \
        .pack(anchor="w")
    prep_frame = ctk.CTkFrame(frame, fg_color="transparent")
    prep_frame.pack(fill="x", pady=5)
    ctk.CTkButton(prep_frame, text="FOV Generator",    command=open_fov).pack(side="left", padx=5)
    ctk.CTkButton(prep_frame, text="Lens Correction",  command=open_lens).pack(side="left", padx=5)
    ctk.CTkButton(prep_frame, text="Harmonise Images", command=open_harmonise).pack(side="left", padx=5)

    # (B) Georeferencing
    ctk.CTkLabel(frame, text="Georeferencing", font=("Serif", 14, "bold")) \
        .pack(anchor="w", pady=(10,0))
    geo_frame = ctk.CTkFrame(frame, fg_color="transparent")
    geo_frame.pack(fill="x", pady=5)
    ctk.CTkButton(geo_frame, text="Pixel to GCP", command=open_pixel_to_gcp).pack(side="left", padx=5)
    ctk.CTkButton(geo_frame, text="Homography", command=open_homography).pack(side="left", padx=5)
    ctk.CTkButton(geo_frame, text="Georef Images", command=open_georef).pack(side="left", padx=5)

    # (C) Feature Identifier Tool
    ctk.CTkLabel(frame, text="Feature Identifier Tool", font=("Serif", 14, "bold")) \
        .pack(anchor="w", pady=(10,0))
    fi_frame = ctk.CTkFrame(frame, fg_color="transparent")
    fi_frame.pack(fill="x", pady=5)
    ctk.CTkButton(fi_frame, text="Single Image",       command=open_feature_individual).pack(side="left", padx=5)
    ctk.CTkButton(fi_frame, text="Folder Processing",  command=open_feature_ml).pack(side="left", padx=5)
    ctk.CTkButton(fi_frame, text="Batch Process",      command=open_feature_batch).pack(side="left", padx=5)

    # (F) Data Exploration
    ctk.CTkLabel(frame, text="Data Exploration", font=("Serif", 14, "bold")) \
        .pack(anchor="w", pady=(10,0))
    exp_frame = ctk.CTkFrame(frame, fg_color="transparent")
    exp_frame.pack(fill="x", pady=5)
    ctk.CTkButton(exp_frame, text="Time Series Analysis", command=open_timeseries).pack(side="left", padx=5)
    ctk.CTkButton(exp_frame, text="Color Space Explorer", command=open_color_explorer).pack(side="left", padx=5)


    # (D) DEM Generator
    ctk.CTkLabel(frame, text="DEM Generator", font=("Serif", 14, "bold")) \
        .pack(anchor="w", pady=(10,0))
    dem_frame = ctk.CTkFrame(frame, fg_color="transparent")
    dem_frame.pack(fill="x", pady=5)
    ctk.CTkButton(dem_frame, text="Create DEM", command=open_dem) \
        .pack(side="left", padx=5)

    # (E) Time-stacking
    ctk.CTkLabel(frame, text="Time-stacking", font=("Serif", 14, "bold")) \
        .pack(anchor="w", pady=(10,0))
    ts_frame = ctk.CTkFrame(frame, fg_color="transparent")
    ts_frame.pack(fill="x", pady=5)
    ctk.CTkButton(ts_frame, text="Profile & Hovmöller",  command=open_profile).pack(side="left", padx=5)
    ctk.CTkButton(ts_frame, text="Raw Timestack Image", command=open_timestack).pack(side="left", padx=5)
    ctk.CTkButton(ts_frame, text="Wave Run Up",          command=open_wave_run).pack(side="left", padx=5)



    # ——— Footer ———
    footer_text = (
        "creator: Clayton Soares\n"
        "contact: clayton.soares@ifg.uni-kiel.de\n"
        "Institute: Institute of Geosciences, CAU, Kiel\n"
        "source code: https://github.com/capt-clay10/GeoCamPal.git"
    )
    footer = ctk.CTkLabel(
        frame,
        text=footer_text,
        font=("Times New Roman", 10),
        justify="left",
        anchor="w"
    )
    footer.pack(side="bottom", anchor="w", fill="x", pady=(10,0))

    dialog.mainloop()

# 5) MAIN
if __name__ == "__main__":
    show_splash(duration_ms=1000)
    launcher_window()