import sys
import subprocess  # <-- Added for launching submodules

if sys.stdout is None:
    sys.stdout = sys.__stdout__
if sys.stderr is None:
    sys.stderr = sys.__stderr__
import os
import tkinter as tk
import customtkinter as ctk
from PIL import Image

# 1) Resource helper for PyInstaller or direct script
def resource_path(relative_path: str) -> str:
    try:
        base_path = sys._MEIPASS  # If running in a PyInstaller .exe
    except Exception:
        base_path = os.path.dirname(__file__)  # Running directly from source
    return os.path.join(base_path, relative_path)

# 2) Optional splash screen
def show_splash(duration_ms=1000):
    splash = ctk.CTk()
    splash.title("Loading...")

    # Set the splash size
    splash.geometry("400x280")
    splash.resizable(False, False)

    # Center on screen
    screen_w = splash.winfo_screenwidth()
    screen_h = splash.winfo_screenheight()
    x_coord = int((screen_w - 450) / 2)
    y_coord = int((screen_h - 280) / 2)
    splash.geometry(f"400x280+{x_coord}+{y_coord}")

    # Try loading a splash image
    try:
        img_path = resource_path("splash_image.png")
        pil_img = Image.open(img_path).resize((450, 280), Image.Resampling.LANCZOS)
        ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(450, 280))
        ctk.CTkLabel(splash, image=ctk_img, text="").pack()
    except Exception as e:
        print("Warning: Could not load splash_image.png:", e)
        ctk.CTkLabel(splash, text="Loading...", font=("Arial", 18, "bold")).pack(expand=True)

    # Close after duration_ms
    splash.after(duration_ms, splash.destroy)
    splash.mainloop()

# 3) Set CustomTkinter appearance
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")

# 4) The main launcher window function
def launcher_window():
    """
    Creates the main launcher window for GeoCamPal. 
    It remains open after launching submodules.
    """
    # Each pick_* function launches the appropriate submodule via subprocess
    def pick_pixel_to_gcp():
        subprocess.Popen([sys.executable, resource_path("pixel_to_gcp.py")])
    
    def pick_hsv_individual():
        subprocess.Popen([sys.executable, resource_path("hsv_mask.py"), "individual"])
    
    def pick_hsv_ml():
        subprocess.Popen([sys.executable, resource_path("hsv_mask.py"), "ml"])
    
    def pick_hsv_batch():
        subprocess.Popen([sys.executable, resource_path("hsv_mask.py"), "batch"])
    
    def pick_dem():
        subprocess.Popen([sys.executable, resource_path("dem_generator.py")])
    
    def pick_georef():
        subprocess.Popen([sys.executable, resource_path("georef.py")])
    
    def pick_homography():
        subprocess.Popen([sys.executable, resource_path("homography.py")])
    
    def pick_timestack():
        subprocess.Popen([sys.executable, resource_path("raw_timestacker.py"), "raw"])
    
    def pick_wave_run():
        subprocess.Popen([sys.executable, resource_path("wave_runup.py"), "wave"])
    
    def on_close():
        """Close everything and exit immediately."""
        dialog.destroy()
        sys.exit(0)
    
    # Build the launcher GUI
    dialog = ctk.CTk()
    dialog.title("GeoCamPal Launcher")
    dialog.geometry("600x600")
    dialog.resizable(False, False)
    dialog.protocol("WM_DELETE_WINDOW", on_close)
    
    main_frame = ctk.CTkFrame(dialog)
    main_frame.pack(padx=20, pady=20, fill="both", expand=True)
    dialog.iconbitmap(resource_path("launch_logo.ico"))
    
    # (A) Logo
    try:
        logo_path = resource_path("launch_logo.png")
        pil_img = Image.open(logo_path)
        logo_image = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(100, 70))
        logo_label = ctk.CTkLabel(main_frame, image=logo_image, text="")
        logo_label.pack(pady=(0, 20))
    except Exception as e:
        print("Warning: Could not load launch_logo.png:", e)
    
    # (B) Header
    header_label = ctk.CTkLabel(main_frame, text="Select a tool to open", font=("Arial", 18, "bold"))
    header_label.pack(pady=(0, 15))
    
    # (C) Georeferencing
    geo_label = ctk.CTkLabel(main_frame, text="Georeferencing", font=("Arial", 14, "bold"))
    geo_label.pack(anchor="w")
    
    geo_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
    geo_frame.pack(pady=5, fill="x")
    ctk.CTkButton(geo_frame, text="Pixel to GCP", command=pick_pixel_to_gcp).pack(side="left", padx=5)
    ctk.CTkButton(geo_frame, text="Homography", command=pick_homography).pack(side="left", padx=5)
    ctk.CTkButton(geo_frame, text="Georef Images", command=pick_georef).pack(side="left", padx=5)
    
    # (D) HSV Tool
    hsv_label = ctk.CTkLabel(main_frame, text="Feature Identifier tool", font=("Arial", 14, "bold"))
    hsv_label.pack(anchor="w", pady=(10, 0))
    
    hsv_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
    hsv_frame.pack(pady=5, fill="x")
    ctk.CTkButton(hsv_frame, text="Individual Analysis", command=pick_hsv_individual).pack(side="left", padx=5)
    ctk.CTkButton(hsv_frame, text="Machine Learning", command=pick_hsv_ml).pack(side="left", padx=5)
    ctk.CTkButton(hsv_frame, text="Batch Process", command=pick_hsv_batch).pack(side="left", padx=5)
    
    # (E) DEM Generator
    dem_label = ctk.CTkLabel(main_frame, text="DEM Generator", font=("Arial", 14, "bold"))
    dem_label.pack(anchor="w", pady=(10, 0))
    ctk.CTkButton(main_frame, text="Create DEM", command=pick_dem).pack(pady=5, fill="x")
    
    # (F) Time-stacking
    ts_label = ctk.CTkLabel(main_frame, text="Time-stacking", font=("Arial", 14, "bold"))
    ts_label.pack(anchor="w", pady=(10, 0))
    
    ts_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
    ts_frame.pack(pady=5, fill="x")
    ctk.CTkButton(ts_frame, text="Raw Timestack Image", command=pick_timestack).pack(side="left", padx=5)
    ctk.CTkButton(ts_frame, text="Wave Run Up", command=pick_wave_run).pack(side="left", padx=5)
    
    # (G) Footer
    footer_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
    footer_frame.pack(side="bottom", fill="x", pady=10)
    footer_label = ctk.CTkLabel(footer_frame,
                                text="Application created by Clayton Soares\nUniversity of Kiel",
                                font=("Arial", 10))
    footer_label.pack(side="left", anchor="sw")
    
    dialog.mainloop()

# Function to run submodules based on command-line args
def run_submodule():
    if len(sys.argv) < 2:
        return
    module = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else None

    if module == "pixel_to_gcp":
        from pixel_to_gcp import PixelToGCPWindow
        app = PixelToGCPWindow()
    elif module == "hsv":
        from hsv_mask import HSVMaskTool
        app = HSVMaskTool(mode=mode)
    elif module == "dem":
        from dem_generator import CreateDemWindow
        app = CreateDemWindow()
    elif module == "georef":
        from georef import GeoReferenceModule
        app = GeoReferenceModule()
    elif module == "homography":
        from homography import CreateHomographyMatrixWindow
        app = CreateHomographyMatrixWindow()
    elif module == "raw_timestacker":
        from raw_timestacker import TimestackTool
        app = TimestackTool()
    elif module == "wave_runup":
        from wave_runup import WaveRunUpCalculator
        app = WaveRunUpCalculator()
    else:
        print(f"Unknown submodule: {module}")
        return

    app.mainloop()

# --------------------------------------------------------------------
# 5) MAIN
# --------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run the submodule if arguments are provided
        run_submodule()
    else:
        # Show the launcher if no arguments are given
        show_splash(duration_ms=1000)
        launcher_window()
