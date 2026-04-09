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
import tkinter as tk
from tkinter import filedialog
from typing import Any, Dict, Iterable, List, Optional, Tuple


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
    Scale *window* so that it fits the current screen while preserving
    the aspect ratio of the original (design_w × design_h) layout.

    • Centres the window on screen.
    • Never upscales beyond the design size.
    • The *margin* parameter (default 90 %) leaves a gutter so the
      window doesn't touch the taskbar / dock.

    Parameters
    ----------
    window     : Tk / CTk / CTkToplevel instance
    design_w   : intended width in pixels  (the "old hardcoded value")
    design_h   : intended height in pixels
    resizable  : allow the user to drag-resize afterwards
    margin     : fraction of screen to occupy at most (0.90 = 90 %)

    Cross-platform
    --------------
    On macOS with Retina displays, winfo_screenwidth/height may return
    logical (1×) pixels.  The 0.90 margin absorbs any mismatch so the
    window never spills off-screen.
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

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return path


def load_settings_json(
    parent,
    module_name: str,
    initialdir: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Ask the user for a settings JSON file and load it.

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
        try:
            child.grab_set()
        except Exception:
            pass


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


def describe_saved_path(path: Optional[str]) -> Dict[str, Any]:
    """Return a JSON-safe saved-path record with an existence flag."""
    text = path or ""
    return {
        "path": text,
        "exists": bool(text and os.path.exists(text)),
    }


# ─────────────────────────────────────────────────────────────────────
#  Backward-compatible alias
# ─────────────────────────────────────────────────────────────────────
#  Some modules import ``StdoutRedirector`` by name. This alias lets
#  them keep working during the transition while you migrate them
#  one at a time to ``setup_console()``.

StdoutRedirector = ThreadSafeRedirector
