"""
utils.py — GeoCamPal shared utilities
──────────────────────────────────────
Consolidates code that was duplicated across 14+ modules:

  • fit_geometry()         — screen-aware window sizing
  • resource_path()        — PyInstaller / source compatible path lookup
  • ThreadSafeRedirector   — thread-safe sys.stdout replacement
  • setup_console()        — one-call console setup for any tool window

Cross-platform notes
--------------------
  Windows : DPI awareness is handled in main.py (ctypes.windll).
            fit_geometry reads the DPI-aware screen size automatically.
  macOS   : Retina displays may report virtual pixels via
            winfo_screenwidth/height.  The 0.90 margin absorbs this.
            Tkinter on macOS uses NSView — all after() dispatches run
            on the main thread, which is required.
  Linux   : X11/Wayland — winfo_screenwidth/height return the primary
            monitor size.  No special handling needed.
"""

import json
import os
import queue
import sys
import threading
import time
import contextlib
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────
#  Version  —  single source of truth for the whole application
# ─────────────────────────────────────────────────────────────────────
#  Import this everywhere the version is shown or recorded (launcher
#  footer, calibration reports, saved-settings provenance) so the
#  number can never drift between the GUI, the code, and the paper.
__version__ = "1.0.4"


# ─────────────────────────────────────────────────────────────────────
#  resource_path  —  works with PyInstaller bundles and source
# ─────────────────────────────────────────────────────────────────────


def resource_path(relative_path: str) -> str:
    """
    Return the absolute path to a bundled resource.

    When running from a PyInstaller .exe, files are extracted to a
    temporary folder referenced by sys._MEIPASS.  When running from
    source, we use the directory containing *this* file (utils.py),
    which must sit alongside the other GeoCamPal modules and assets.
    """
    try:
        # PyInstaller stores the temp extraction path here
        base_path = sys._MEIPASS  # type: ignore[attr-defined]
    except AttributeError:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)


# ─────────────────────────────────────────────────────────────────────
#  fit_geometry  —  scale window to screen, preserve aspect ratio
# ─────────────────────────────────────────────────────────────────────


def fit_geometry(window, design_w, design_h, resizable=True, margin=0.90):
    """
    Give *window* its design size (design_w × design_h), or what the
    screen has room for if that is less, and centre it.

    The design size is the size on a display at 100 %.  On a display
    scaled above 100 % a customtkinter window is larger by the display
    scaling, as its widgets are: 1200 × 800 is 2100 × 1400 screen pixels
    at 175 %.

    On Windows the whole window, title bar and border included, is kept
    inside the work area (the screen without the taskbar) and centred in
    it.  Width and height are each limited on their own: a window that is
    too high for the screen keeps its width.  The layouts stretch, so a
    window reduced in one direction only shows more of its controls than
    one reduced in both.

    Where the work area cannot be read (Linux, macOS) the window is
    reduced with its aspect ratio kept to at most *margin* of the screen
    and centred on the screen.

    Size and place are worked out in screen pixels, which is what Tk and
    Windows report, and handed over with geometry_px().  customtkinter's
    own geometry() multiplies a size by the display scaling: a window
    fitted to the screen in pixels and then enlarged by that factor is
    placed for a size it does not have, and reaches beyond the screen.

    Parameters
    ----------
    window     : Tk / CTk / CTkToplevel instance
    design_w   : intended width at 100 %  (the "old hardcoded value")
    design_h   : intended height at 100 %
    resizable  : allow the user to drag-resize afterwards
    margin     : fraction of the screen to occupy at most where the work
                 area is not known (0.90 = 90 %)
    """
    screen_w = window.winfo_screenwidth()
    screen_h = window.winfo_screenheight()

    # Design size in screen pixels (factor 1.0 at 100 % and for plain Tk).
    f = window_scaling(window)
    want_w = design_w * f
    want_h = design_h * f

    area = work_area(window)
    frame = window_frame(window)

    if area is not None and frame is not None:
        left, top, right, bottom = area
        side, title, border = frame

        final_w = int(min(want_w, (right - left) - 2 * border))
        final_h = int(min(want_h, (bottom - top) - title - border))

        # geometry() places the window by its outer corner, which lies
        # `side` left of the inside; `border` of that is seen.
        x = left + ((right - left) - final_w - 2 * border) // 2 - side + border
        y = top + ((bottom - top) - final_h - title - border) // 2
    else:
        max_w = int(screen_w * margin)
        max_h = int(screen_h * margin)

        scale = min(max_w / want_w, max_h / want_h, 1.0)

        final_w = int(want_w * scale)
        final_h = int(want_h * scale)

        x = (screen_w - final_w) // 2
        y = max(0, (screen_h - final_h) // 2)

    geometry_px(window, final_w, final_h, x, y)
    window.resizable(resizable, resizable)


def work_area(window=None):
    """
    The part of the primary screen that the taskbar leaves free, as
    (left, top, right, bottom) in screen pixels.

    Returns None where it cannot be read (not Windows).
    """
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        rect = wintypes.RECT()
        # SPI_GETWORKAREA
        if not ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0):
            return None
        if rect.right - rect.left < 200 or rect.bottom - rect.top < 200:
            return None
        return rect.left, rect.top, rect.right, rect.bottom
    except Exception:
        return None


def window_frame(window=None):
    """
    What Windows adds around the inside of a window that has a title bar,
    as (side, title, border) in screen pixels:

    side   : border at the left, the right and the bottom.  Most of it is
             not seen; it is where the mouse grips the window to resize it.
    title  : title bar, from the top of the window to its inside.
    border : the part of `side` that is seen.

    Returns None where it cannot be read (not Windows).
    """
    if sys.platform != "win32":
        return None
    try:
        import ctypes

        user32 = ctypes.windll.user32
        padded = user32.GetSystemMetrics(92)             # SM_CXPADDEDBORDER
        side = user32.GetSystemMetrics(32) + padded      # SM_CXSIZEFRAME
        title = (user32.GetSystemMetrics(4)              # SM_CYCAPTION
                 + user32.GetSystemMetrics(33) + padded)  # SM_CYSIZEFRAME
        border = max(1, round(user32.GetDpiForSystem() / 96))
        return side, title, border
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────
#  ThreadSafeRedirector  —  drop-in sys.stdout replacement
# ─────────────────────────────────────────────────────────────────────


class ThreadSafeRedirector:
    """
    A file-like object that can replace sys.stdout / sys.stderr and
    safely route output to a ``tk.Text`` widget from **any** thread.

    How it works
    ------------
    1.  ``write()`` pushes messages into a ``queue.Queue`` (thread-safe).
    2.  A 50 ms ``after()`` poll on the main thread drains the queue
        and inserts text into the widget.
    3.  If the widget is destroyed, polling stops gracefully — no
        TclError crashes.

    Why not just ``widget.insert()`` directly?
    -------------------------------------------
    Tkinter is single-threaded on all platforms.  Calling any widget
    method from a background thread is undefined behaviour:
      • Windows : often seems fine, then crashes on heavy load
      • Linux   : random segfaults if X11 handles are touched
      • macOS   : silent corruption; AppKit objects aren't re-entrant

    The queue+after pattern is the canonical cross-platform solution.
    """

    _POLL_MS = 50  # drain interval — 50 ms balances responsiveness vs. CPU usage

    def __init__(self, text_widget: tk.Text):
        self._widget = text_widget
        self._queue = queue.Queue()
        self._alive = True
        self._schedule_poll()

    # ── file-like interface (so print() works) ──

    def write(self, message: str) -> int:
        """Enqueue a message. Safe to call from any thread."""
        if message:
            self._queue.put(message)
        return len(message) if message else 0

    def flush(self) -> None:
        """No-op — required by the file-like protocol."""
        pass

    @property
    def encoding(self) -> str:
        """Some libraries (e.g. tqdm) check this attribute."""
        return "utf-8"

    def isatty(self) -> bool:
        """Some libraries check if stdout is a real terminal."""
        return False

    # ── internal polling ──

    def _schedule_poll(self):
        """Schedule the next drain. Runs on the main thread."""
        if not self._alive:
            return
        try:
            self._widget.after(self._POLL_MS, self._poll)
        except (tk.TclError, RuntimeError):
            # widget has been destroyed — stop polling
            self._alive = False

    def _poll(self):
        """Drain the queue and insert into the widget. Main thread only."""
        if not self._alive:
            return
        try:
            batch = []
            while True:
                try:
                    batch.append(self._queue.get_nowait())
                except queue.Empty:
                    break

            if batch:
                self._widget.insert(tk.END, "".join(batch))
                self._widget.see(tk.END)

        except tk.TclError:
            self._alive = False
            return

        self._schedule_poll()

    def stop(self):
        """Explicitly stop polling (call on window close)."""
        self._alive = False


# ─────────────────────────────────────────────────────────────────────
#  setup_console / restore_console  —  one-call stdout management
# ─────────────────────────────────────────────────────────────────────

# We store the original streams here so every module restores
# to the same place, even if multiple modules are opened in sequence.
_original_stdout = sys.stdout
_original_stderr = sys.stderr


def setup_console(text_widget: tk.Text, greeting: str = "") -> ThreadSafeRedirector:
    """
    Wire up a ``tk.Text`` widget as the console for a tool window.

    Returns the ``ThreadSafeRedirector`` instance so the caller can
    call ``redirector.stop()`` on window close.

    Usage inside a CTkToplevel.__init__::

        ct = tk.Text(frame, wrap="word", height=9)
        ct.pack(fill="both", expand=True)
        self._console_redir = setup_console(ct, "GeoCamPal — My Tool")

    And in your close handler::

        def _on_close(self):
            restore_console(self._console_redir)
            self.destroy()
    """
    try:
        text_widget.configure(
            bg="white",
            fg="black",
            insertbackground="black",
            height=10,
        )
    except Exception:
        pass

    redir = ThreadSafeRedirector(text_widget)
    sys.stdout = redir
    sys.stderr = redir

    if greeting:
        redir.write(greeting + "\n")

    return redir


def restore_console(redirector=None):
    """
    Restore sys.stdout / sys.stderr to the originals saved at import
    time, and stop the redirector's polling loop.

    Safe to call multiple times or with None.
    """
    if redirector is not None:
        redirector.stop()
    sys.stdout = _original_stdout
    sys.stderr = _original_stderr


# ─────────────────────────────────────────────────────────────────────
#  Settings JSON helpers  —  shared save/load pattern
# ─────────────────────────────────────────────────────────────────────


def _clean_module_name(module_name: str) -> str:
    text = (module_name or "settings").strip().lower()
    cleaned = []
    prev_sep = False
    for ch in text:
        if ch.isalnum():
            cleaned.append(ch)
            prev_sep = False
        else:
            if not prev_sep:
                cleaned.append("_")
            prev_sep = True
    name = "".join(cleaned).strip("_")
    return name or "settings"


def suggest_settings_filename(module_name: str) -> str:
    """Return a stable default JSON filename for a tool's settings."""
    return f"{_clean_module_name(module_name)}_settings.json"


def save_settings_json(
    parent,
    module_name: str,
    data: Dict[str, Any],
    initialdir: Optional[str] = None,
    suggested_name: Optional[str] = None,
) -> Optional[str]:
    """
    Ask the user where to save a settings JSON file and write it.

    Returns the saved path, or None if the dialog was cancelled.
    """
    initialfile = suggested_name or suggest_settings_filename(module_name)
    kwargs = {
        "parent": parent,
        "title": f"Save {module_name} settings",
        "defaultextension": ".json",
        "initialfile": initialfile,
        "filetypes": [("JSON files", "*.json"), ("All files", "*.*")],
    }
    if initialdir and os.path.isdir(initialdir):
        kwargs["initialdir"] = initialdir

    path = filedialog.asksaveasfilename(**kwargs)
    if not path:
        return None

    # Record which release wrote the file (provenance for reproducibility).
    # The key is additive: earlier releases ignore unknown keys on load.
    payload = {"geocampal_version": __version__}
    payload.update({k: v for k, v in data.items() if k != "geocampal_version"})
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return path


def _looks_like_path_key(key: str) -> bool:
    """
    Return True if *key* suggests its value is a file or directory path.

    Matched patterns (case-insensitive):
      • contains ``path``, ``folder``
      • ends with ``_dir``, ``_file``
    """
    k = key.lower()
    if "path" in k or "folder" in k:
        return True
    if k.endswith("_dir") or k.endswith("_file"):
        return True
    return False


def validate_settings_paths(
    data: Dict[str, Any],
    parent=None,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Walk a loaded settings dict and clear file/directory path values
    that no longer exist on disk.

    Detection heuristic
    -------------------
    A JSON value is treated as a path if **all** of the following hold:

      1. It is a non-empty string.
      2. ``os.path.isabs(value)`` is True (avoids false positives on
         format strings, column names, etc.).
      3. Either the key name looks like a path key (see
         ``_looks_like_path_key``), **or** it sits directly inside a
         dictionary whose key is ``"paths"`` (the convention used by
         every GeoCamPal module).

    Values that fail the existence check are replaced with ``""``
    (empty string).  Because every module already guards path usage
    with ``if value:`` or ``value or None``, this is sufficient to
    prevent stale paths from appearing in the GUI while keeping all
    other settings intact.

    Returns ``(data, missing)`` where *missing* is a list of the
    path strings that were cleared.
    """
    missing: List[str] = []

    def _walk(obj, in_paths_dict: bool = False):
        if isinstance(obj, dict):
            for key, value in obj.items():
                is_paths_key = (key.lower() == "paths")
                if isinstance(value, str) and value:
                    should_check = in_paths_dict or _looks_like_path_key(key)
                    if should_check and os.path.isabs(value) and not os.path.exists(value):
                        missing.append(value)
                        obj[key] = ""
                elif isinstance(value, (dict, list)):
                    _walk(value, in_paths_dict=is_paths_key or in_paths_dict)
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, (dict, list)):
                    _walk(item, in_paths_dict=in_paths_dict)

    _walk(data)

    if missing:
        for p in missing:
            print(f"[WARNING] Settings path not found: {p}")
        summary = "\n".join(f"  \u2022 {p}" for p in missing)
        msg = (
            "The following files/directories from the saved settings "
            "could not be found:\n\n"
            f"{summary}\n\n"
            "Those entries have been cleared.\n"
            "All other settings were loaded normally."
        )
        try:
            messagebox.showwarning(
                "Settings \u2014 Missing Files",
                msg,
                parent=parent if parent is not None else None,
            )
        except Exception:
            pass  # headless / testing — console message is enough

    return data, missing


def load_settings_json(
    parent,
    module_name: str,
    initialdir: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Ask the user for a settings JSON file and load it.

    After loading, any file/directory paths stored in the JSON that no
    longer exist on disk are automatically cleared so the GUI does not
    show stale entries.  A single warning is shown listing every path
    that was not found.

    Returns ``(data, path)`` or ``(None, None)`` if cancelled.
    """
    kwargs = {
        "parent": parent,
        "title": f"Load {module_name} settings",
        "filetypes": [("JSON files", "*.json"), ("All files", "*.*")],
    }
    if initialdir and os.path.isdir(initialdir):
        kwargs["initialdir"] = initialdir

    path = filedialog.askopenfilename(**kwargs)
    if not path:
        return None, None

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    written_by = data.get("geocampal_version")
    if written_by:
        print(f"[Settings] {os.path.basename(path)} written by GeoCamPal {written_by}")
    else:
        print(f"[Settings] {os.path.basename(path)} carries no version stamp "
              f"(written before v1.0.3)")

    # Validate file/directory paths — clears missing ones in-place
    validate_settings_paths(data, parent=parent)

    return data, path


# ─────────────────────────────────────────────────────────────────────
#  Child window helpers  —  bring selector / popup windows to front
# ─────────────────────────────────────────────────────────────────────


def bring_child_to_front(child, parent=None, modal: bool = False) -> None:
    """
    Best-effort helper to raise a child window above its parent.

    Useful for AOI selectors and other temporary tool windows that may
    otherwise open behind the main configuration window on some systems.
    """
    try:
        if parent is not None and parent.winfo_exists():
            child.transient(parent)
    except Exception:
        pass

    try:
        child.lift()
    except Exception:
        pass

    try:
        child.focus_force()
    except Exception:
        pass

    try:
        child.attributes("-topmost", True)
        child.after(150, lambda: child.attributes("-topmost", False))
    except Exception:
        pass

    if modal:
        def _grab():
            try:
                if child.winfo_exists():
                    child.grab_set()
            except Exception:
                pass
        try:
            child.grab_set()
        except Exception:
            # A CTkToplevel is withdrawn for a few milliseconds while its
            # title bar is restyled, and a grab on an unmapped window fails.
            # Retry once the window is viewable.
            try:
                child.after(200, _grab)
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────
#  Window chrome  —  logo icon and DPI-safe sizing for child windows
# ─────────────────────────────────────────────────────────────────────

# Background for tk.Canvas widgets embedded in dark customtkinter windows.
# "gray14" is customtkinter's dark-mode window colour, so the canvas blends
# into the window instead of showing as a light grey rectangle.
DARK_CANVAS_BG = "#242424"


def apply_window_icon(window, delay_ms: int = 250) -> None:
    """
    Give any child window (CTkToplevel, tk.Toplevel, CTk) the GeoCamPal logo.

    The .png via iconphoto() works on every platform.  On Windows the .ico
    via iconbitmap() is layered on top for a crisper title bar; calling
    iconbitmap() straight away also tells CTkToplevel that an icon has been
    chosen, so it does not install its own default icon 200 ms later.

    The icon is set once immediately and once more after ``delay_ms``
    because Tk can drop an icon set before the window is mapped, and
    CTkToplevel reconfigures its title bar shortly after construction.
    A missing asset is not fatal.
    """
    def _set():
        try:
            if not window.winfo_exists():
                return
        except Exception:
            return
        try:
            img = tk.PhotoImage(master=window,
                                file=resource_path("launch_logo.png"))
            window.iconphoto(False, img)
            window._icon_image = img  # keep a reference alive
        except Exception:
            pass
        if sys.platform == "win32":
            try:
                window.iconbitmap(resource_path("launch_logo.ico"))
            except Exception:
                pass

    _set()
    try:
        window.after(delay_ms, _set)
    except Exception:
        pass


def window_scaling(window) -> float:
    """
    Factor by which a customtkinter window multiplies geometry(), minsize()
    and maxsize() requests (DPI scaling x user window scaling).

    Plain Tk windows return 1.0, as does any window whose factor cannot be
    read.
    """
    try:
        from customtkinter import ScalingTracker
        return float(ScalingTracker.get_window_scaling(window)) or 1.0
    except Exception:
        return 1.0


def widget_scaling(widget) -> float:
    """
    Factor by which customtkinter enlarges the widgets of *widget*'s window:
    their sizes, their fonts and the pictures given as CTkImage (DPI scaling
    x user widget scaling).  1.0 on a display at 100 %, 1.75 at 175 %.

    A size measured in screen pixels (winfo_width() and the like) is divided
    by this factor before it is given to a customtkinter widget or CTkImage.

    Returns 1.0 for a widget whose factor cannot be read.
    """
    try:
        from customtkinter import ScalingTracker
        return float(ScalingTracker.get_widget_scaling(widget)) or 1.0
    except Exception:
        return 1.0


def geometry_px(window, width, height, x=None, y=None) -> None:
    """
    Size (and optionally place) *window* in physical screen pixels.

    CTkToplevel.geometry() multiplies the requested size by the DPI scaling
    factor, so a popup converted from tk.Toplevel would otherwise grow on
    high-DPI screens.  Dividing by that factor first keeps the size the
    design intended.  Positions are never scaled by customtkinter.
    """
    f = window_scaling(window)
    w = max(1, int(round(width / f)))
    h = max(1, int(round(height / f)))
    if x is None or y is None:
        window.geometry(f"{w}x{h}")
    else:
        window.geometry(f"{w}x{h}+{int(x)}+{int(y)}")


def figure_canvas(figure, master):
    """
    Return a matplotlib FigureCanvasTkAgg for *figure* inside *master* that
    keeps asking for the size of the figure as it was created.

    On a display scaled above 100 % matplotlib's Tk canvas, when it is first
    shown, takes the room it has just been given, multiplies it by the
    display scaling and asks for that.  A plot that fills its panel then
    asks for 1.75 times the panel at 175 %, and what is placed after it
    (console, controls, toolbar) is pushed out of the window.

    Here the canvas asks for the figure's own size instead (figsize x dpi,
    the dpi then including the display scaling), which is what it asks for
    at 100 %.  At 100 % matplotlib never changes the canvas for the display
    scaling, and this function changes nothing.
    """
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    own_w, own_h = figure.get_size_inches()
    canvas = FigureCanvasTkAgg(figure, master=master)
    widget = canvas.get_tk_widget()
    seen = [getattr(canvas, "device_pixel_ratio", 1)]

    def _keep_own_size(event=None):
        ratio = getattr(canvas, "device_pixel_ratio", 1)
        if ratio != seen[0]:
            seen[0] = ratio
            widget.configure(width=int(own_w * figure.dpi),
                             height=int(own_h * figure.dpi))

    # Added after matplotlib's own <Map> handler, which applies the display
    # scaling; both run before the window is laid out again.
    widget.bind("<Map>", _keep_own_size, add="+")
    return canvas


# ─────────────────────────────────────────────────────────────────────
#  ETA helpers  —  shared progress/ETA formatting
# ─────────────────────────────────────────────────────────────────────


def format_eta(seconds: Optional[float]) -> str:
    """Format seconds into a short human-readable ETA string."""
    if seconds is None:
        return "estimating..."
    try:
        seconds = float(seconds)
    except Exception:
        return "estimating..."
    if seconds < 0 or not (seconds < float("inf")):
        return "estimating..."

    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)

    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def compute_eta(start_time: float, done: int, total: int) -> Optional[float]:
    """
    Estimate remaining seconds from average time per completed item.

    Returns None until at least one item has completed.
    """
    if total <= 0 or done <= 0:
        return None
    elapsed = max(0.0, time.time() - start_time)
    rate = elapsed / float(done)
    remaining = max(0, total - done)
    return remaining * rate


# ─────────────────────────────────────────────────────────────────────
#  Selector geometry helpers  —  pixel-first, world-if-available
# ─────────────────────────────────────────────────────────────────────


def _serialise_points(points: Optional[Iterable[Iterable[Any]]]) -> List[List[float]]:
    serialised: List[List[float]] = []
    if points is None:
        return serialised
    for pt in points:
        vals = list(pt)
        if len(vals) >= 2:
            serialised.append([float(vals[0]), float(vals[1])])
    return serialised


def make_selector_payload(
    mode: str,
    bbox_px: Optional[Iterable[Any]] = None,
    points_px: Optional[Iterable[Iterable[Any]]] = None,
    crs: Optional[str] = None,
    bbox_world: Optional[Iterable[Any]] = None,
    points_world: Optional[Iterable[Iterable[Any]]] = None,
) -> Dict[str, Any]:
    """
    Build a compact JSON-safe selector payload.

    Pixel geometry is always primary. World geometry is attached only
    when it is explicitly provided.
    """
    payload: Dict[str, Any] = {"mode": mode}

    if bbox_px is not None:
        vals = list(bbox_px)
        if len(vals) >= 4:
            payload["bbox_px"] = [
                float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])
            ]

    px_points = _serialise_points(points_px)
    if px_points:
        payload["points_px"] = px_points

    if crs:
        payload["crs"] = str(crs)

    if bbox_world is not None:
        vals = list(bbox_world)
        if len(vals) >= 4:
            payload["bbox_world"] = [
                float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])
            ]

    world_points = _serialise_points(points_world)
    if world_points:
        payload["points_world"] = world_points

    return payload


# Path labels -----------------------------------------------------------------
# Chosen folders and files are shown as their last two path components, so a
# label stays short but still distinguishes .../site_A/images from
# .../site_B/images.  Colour says which side of the job it is, following the
# convention already used for buttons: a dark shade fills the button, a light
# tint of the same hue is legible as text on the dark background.
PATH_INPUT_COLOR = "#81C784"    # light green  (input / source)
PATH_OUTPUT_COLOR = "#CBB77C"   # light tan    (output / destination)
PATH_UNSET_COLOR = "gray"


def short_path(path: Any, depth: int = 2) -> str:
    """Return the last `depth` components of `path` (e.g. 'site_A/images').

    Falls back to the whole path when it is shorter than `depth`, and keeps
    the drive or UNC share on those short paths so 'D:\\' stays meaningful.
    A trailing separator is ignored rather than producing an empty name.
    """
    p = os.path.normpath(str(path))
    drive, tail = os.path.splitdrive(p)
    parts = [q for q in tail.split(os.sep) if q]
    if not parts:
        return p
    if len(parts) <= depth:
        return (drive + os.sep + os.sep.join(parts)) if drive else \
               (os.sep + os.sep.join(parts) if tail.startswith(os.sep)
                else os.sep.join(parts))
    return os.sep.join(parts[-depth:])


def show_path(label, path: Any, kind: str = "input",
              empty: str = "No folder selected", depth: int = 2) -> None:
    """Show a chosen path on a CTkLabel, or a placeholder when unset.

    `kind` is "input" (green) or "output" (tan); anything else is treated as
    an input.  Unset paths show `empty` in gray.
    """
    if path:
        colour = PATH_OUTPUT_COLOR if kind == "output" else PATH_INPUT_COLOR
        label.configure(text=short_path(path, depth), text_color=colour)
    else:
        label.configure(text=empty, text_color=PATH_UNSET_COLOR)


def describe_saved_path(path: Optional[str]) -> Dict[str, Any]:
    """Return a JSON-safe saved-path record with an existence flag."""
    text = path or ""
    return {
        "path": text,
        "exists": bool(text and os.path.exists(text)),
    }

# ─────────────────────────────────────────────────────────────────────
#  CV.imread helper  —  sometimes cannot read non-ascii
# ─────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────
#  Atomic file output  —  write-to-temp, then os.replace
# ─────────────────────────────────────────────────────────────────────
#  Long batch loops run in daemon threads that are killed abruptly when
#  the tool window closes.  Writing directly to the final path can leave
#  a half-written, corrupt GeoTIFF/CSV on disk.  This context manager
#  writes to a sibling '.tmp' file and only renames it into place once
#  the write completes.  os.replace is atomic on the same filesystem, so
#  a reader never sees a partial file, and an interrupted write leaves
#  the previous good file (if any) untouched.


@contextlib.contextmanager
def atomic_output_path(final_path):
    """
    Yield a temporary path to write to; on clean exit, atomically move it
    to *final_path*.  On any exception, remove the temp file and re-raise.

    Usage::

        with atomic_output_path(out_tif) as tmp:
            with rasterio.open(tmp, "w", ...) as dst:
                dst.write(arr, 1)
        # out_tif now exists atomically
    """
    final_path = os.fspath(final_path)
    tmp_path = final_path + ".tmp"
    try:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    except OSError:
        pass
    try:
        yield tmp_path
        os.replace(tmp_path, final_path)  # atomic on the same filesystem
    except BaseException:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        raise


# ─────────────────────────────────────────────────────────────────────
#  Lens-calibration I/O  —  pickle (primary) + JSON sidecar
# ─────────────────────────────────────────────────────────────────────
#  Lens calibration files are commonly shared between collaborators.
#  The historical format is a Python pickle (.pkl), which (a) requires
#  pickle.load on a possibly-untrusted file and (b) is brittle across
#  NumPy versions.  These helpers keep the .pkl as the primary output
#  for full backward compatibility while ALSO writing a plain-text JSON
#  sidecar that any tool (or a human) can read safely.
#
#  load_lens_calibration() accepts EITHER format and always returns a
#  dict whose 'camera_matrix' and 'dist_coeff' are NumPy arrays, so all
#  existing downstream code keeps working unchanged.

_LENS_ARRAY_KEYS = ("camera_matrix", "dist_coeff")
_LENS_TUPLE_KEYS = ("image_size", "pattern_size", "board_squares")


def _json_safe_calibration(cal_data: Dict[str, Any]) -> Dict[str, Any]:
    """Return a JSON-serialisable copy of a lens-calibration dict."""
    out: Dict[str, Any] = {}
    for k, v in cal_data.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, tuple):
            out[k] = list(v)
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = v.item()
        else:
            out[k] = v
    out["_meta"] = {
        "geocampal_version": __version__,
        "produced_by": "GeoCamPal lens_correction",
    }
    return out


def save_lens_calibration(cal_data: Dict[str, Any], pkl_path: str) -> Tuple[str, Optional[str]]:
    """
    Write a lens-calibration dict as a pickle (primary) plus a JSON
    sidecar alongside it.

    The pickle is written first and exactly as before, so nothing about
    the existing workflow changes.  The JSON is best-effort: if it fails
    to write, the calibration is still considered saved (the .pkl exists)
    and a warning is printed.

    Returns ``(pkl_path, json_path_or_None)``.
    """
    with open(pkl_path, "wb") as f:
        import pickle
        pickle.dump(cal_data, f)

    json_path: Optional[str] = os.path.splitext(pkl_path)[0] + ".json"
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(_json_safe_calibration(cal_data), f, indent=2)
    except Exception as e:  # noqa: BLE001 — JSON is a convenience, never fatal
        print(f"[lens] Could not write JSON sidecar ({e}); .pkl saved normally.")
        json_path = None

    return pkl_path, json_path


def load_lens_calibration(path: str) -> Dict[str, Any]:
    """
    Load a lens calibration from a ``.json`` or ``.pkl`` file.

    Regardless of source, the returned dict has ``camera_matrix`` and
    ``dist_coeff`` as float64 NumPy arrays, so existing callers that do
    ``d["camera_matrix"].astype(...)`` keep working unchanged.

    Raises the underlying I/O / parse error on failure (callers already
    wrap this in try/except).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data.pop("_meta", None)
        for k in _LENS_ARRAY_KEYS:
            if k in data and data[k] is not None:
                data[k] = np.asarray(data[k], dtype=np.float64)
        for k in _LENS_TUPLE_KEYS:
            if k in data and isinstance(data[k], list):
                data[k] = tuple(data[k])
        return data

    # Default / .pkl path — unchanged behaviour.
    with open(path, "rb") as f:
        import pickle
        return pickle.load(f)


def imread_safe(path, flags=cv2.IMREAD_COLOR, *, verbose=True):
    """
    Drop-in replacement for cv2.imread that:
      * accepts str or pathlib.Path,
      * works with non-ASCII / Unicode paths (the classic cv2.imread failure
        mode on Windows), and
      * prints a clear reason when an image can't be read, instead of
        failing silently.

    Returns the image array on success, or None on failure — identical to
    cv2.imread — so existing 'if img is None:' checks keep working unchanged.
    """
    path = os.fspath(path)  # accepts Path or str

    # 1) Missing file — the most common real cause of "can't find images".
    if not os.path.exists(path):
        if verbose:
            print(f"[imread] File not found: {path}")
        return None

    # 2) Empty file (failed/partial copy or download).
    try:
        if os.path.getsize(path) == 0:
            if verbose:
                print(f"[imread] File is 0 bytes (empty): {path}")
            return None
    except OSError:
        pass

    # 3) Normal fast path.
    img = cv2.imread(path, flags)
    if img is not None:
        return img

    # 4) Fallback: read raw bytes via numpy, decode in memory.
    #    This is the standard fix for Unicode/non-ASCII paths that
    #    cv2.imread cannot open directly.
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, flags)
    except Exception as e:
        if verbose:
            print(f"[imread] Error decoding {path}: {e}")
        return None

    if img is None and verbose:
        print(f"[imread] File exists but could not be decoded "
              f"(unsupported format or corrupt): {path}")
    return img


def imwrite_safe(path, img, params=None, *, verbose=True):
    """
    Drop-in replacement for cv2.imwrite that does not fail silently.

    cv2.imwrite returns False, without raising, when it cannot write the
    file: for example for non-ASCII paths on Windows or paths it treats as
    too long.  This writes with cv2.imwrite first (identical output), then
    falls back to encoding in memory and writing the bytes with numpy.

    Returns True when the file was written, False otherwise (with the
    reason printed), so callers can report the failure.
    """
    path = os.fspath(path)
    params = list(params) if params else []
    try:
        # On Windows cv2.imwrite "succeeds" on a non-ASCII path but writes a
        # file whose name is the UTF-8 bytes read as ANSI; take the numpy
        # path for those straight away.
        if path.isascii() and cv2.imwrite(path, img, params):
            return True
    except cv2.error as e:
        if verbose:
            print(f"[imwrite] cv2.imwrite failed for {path}: {e}")
    ext = os.path.splitext(path)[1] or ".png"
    try:
        ok, buf = cv2.imencode(ext, img, params)
        if not ok:
            if verbose:
                print(f"[imwrite] Could not encode the image as {ext}: {path}")
            return False
        buf.tofile(path)
        return True
    except Exception as e:
        if verbose:
            print(f"[imwrite] Could not write {path}: {type(e).__name__}: {e}")
        return False

# ─────────────────────────────────────────────────────────────────────
#  Backward-compatible alias
# ─────────────────────────────────────────────────────────────────────
#  Some modules import ``StdoutRedirector`` by name. This alias lets
#  them keep working during the transition while you migrate them
#  one at a time to ``setup_console()``.

StdoutRedirector = ThreadSafeRedirector