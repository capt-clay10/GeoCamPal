import sys

# Windows DPI awareness — must be set before any GUI imports
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # system-aware
    except Exception:
        pass

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

from utils import fit_geometry, resource_path


def set_window_icon(window):
    """
    Set the window icon in a cross-platform way.
    - Windows: uses iconbitmap() with a .ico file.
    - Linux / macOS: uses iconphoto() with a .png file because
      Tk on these platforms does not support .ico via iconbitmap().
    Both paths are wrapped in try/except so a missing asset is silently ignored.
    """
    try:
        if sys.platform == "win32":
            window.iconbitmap(resource_path("launch_logo.ico"))
        else:
            from PIL import ImageTk
            png_path = resource_path("launch_logo.png")
            img = ImageTk.PhotoImage(file=png_path)
            window.iconphoto(True, img)
            # Keep a reference so the image is not garbage-collected
            window._icon_image = img
    except Exception:
        pass  # Missing icon asset — not fatal


# 2) Optional splash screen
def show_splash(duration_ms=1000):
    splash = ctk.CTk()
    set_window_icon(splash)
    splash.title("Loading...")
    fit_geometry(splash, 400, 280, resizable=False)

    # update() (not just update_idletasks()) forces a full render pass so that
    # winfo_width/height() return the real pixel dimensions rather than 1.
    splash.update()
    sw = splash.winfo_width()
    sh = splash.winfo_height()

    try:
        img_path = resource_path("splash_image.png")
        pil_img = Image.open(img_path).resize((sw, sh), Image.Resampling.LANCZOS)
        ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(sw, sh))
        ctk.CTkLabel(splash, image=ctk_img, text="").pack()
    except Exception:
        ctk.CTkLabel(splash, text="Loading...", font=("Serif", 18, "bold")).pack(expand=True)

    # Render the image before starting the timer
    splash.update()

    splash.after(duration_ms, splash.destroy)

    # Drive the event loop manually instead of calling mainloop().
    # mainloop() on a CTk root can tear down the Tk interpreter on destroy(),
    # which prevents the launcher window from opening afterwards.
    try:
        while splash.winfo_exists():
            splash.update()
    except Exception:
        pass  # Window was destroyed — expected exit path

# 3) Set CustomTkinter appearance globally
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")

# 4) Launcher window
def launcher_window():
    # ── Single-tool guard ──────────────────────────────────────────
    # Only one tool window may be open at a time.  This avoids the
    # global sys.stdout collision where the last-opened window
    # captures all print output from every module.
    _active_tool = [None]  # mutable container so closures can update it

    def _open_tool(factory, **kwargs):
        """Open a tool window, enforcing one-at-a-time."""
        current = _active_tool[0]
        if current is not None:
            try:
                if current.winfo_exists():
                    from tkinter import messagebox
                    messagebox.showinfo(
                        "Tool already open",
                        "Please close the current tool before opening another.")
                    current.lift()
                    return
            except Exception:
                pass  # widget was destroyed — safe to proceed
        win = factory(master=dialog, **kwargs)
        _active_tool[0] = win

    # --- Pre-prep tools ---
    def open_fov():
        _open_tool(FOVGeneratorWindow)

    def open_lens():
        _open_tool(LensCorrectionWindow)

    def open_harmonise():
        _open_tool(HarmoniseImagesWindow)

    # --- Georeferencing ---
    def open_pixel_to_gcp():
        _open_tool(PixelToGCPWindow)

    def open_feature_individual():
        _open_tool(FeatureIdentifier, mode="individual")

    def open_feature_ml():
        _open_tool(FeatureIdentifier, mode="ml")

    def open_feature_batch():
        _open_tool(FeatureIdentifier, mode="batch")

    def open_dem():
        _open_tool(CreateDemWindow)

    def open_georef():
        _open_tool(GeoReferenceModule)

    def open_homography():
        _open_tool(CreateHomographyMatrixWindow)

    def open_timestack():
        _open_tool(TimestackTool)

    def open_wave_run():
        _open_tool(WaveRunUpCalculator)

    # --- Data exploration ---
    def open_timeseries():
        _open_tool(TimeSeriesExplorerWindow)

    def open_color_explorer():
        _open_tool(ColorSpaceExplorerWindow)

    def open_profile():
        _open_tool(ProfileHovmullerWindow)

    def on_close():
        dialog.destroy()
        sys.exit(0)

    dialog = ctk.CTk()
    dialog.title("GeoCamPal")
    fit_geometry(dialog, 600, 700, resizable=False)
    dialog.protocol("WM_DELETE_WINDOW", on_close)
    set_window_icon(dialog)

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
        "source code (v1.0.0) : https://github.com/capt-clay10/GeoCamPal.git"
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