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

import sys
import os
import queue
import threading
import tkinter as tk


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
        base_path = sys._MEIPASS          # type: ignore[attr-defined]
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

    _POLL_MS = 50     # drain interval — 50 ms balances responsiveness
                      # vs. CPU usage; safe on slow machines too

    def __init__(self, text_widget: tk.Text):
        self._widget = text_widget
        self._queue: queue.Queue = queue.Queue()
        self._alive = True
        self._schedule_poll()

    # ── file-like interface (so print() works) ──

    def write(self, message: str) -> int:
        """Enqueue a message.  Safe to call from any thread."""
        if message:                      # ignore empty strings
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
        """Schedule the next drain.  Runs on the main thread."""
        if not self._alive:
            return
        try:
            self._widget.after(self._POLL_MS, self._poll)
        except (tk.TclError, RuntimeError):
            # widget has been destroyed — stop polling
            self._alive = False

    def _poll(self):
        """Drain the queue and insert into the widget.  Main thread only."""
        if not self._alive:
            return
        try:
            # batch-drain: grab everything available right now
            batch = []
            while True:
                try:
                    batch.append(self._queue.get_nowait())
                except queue.Empty:
                    break

            if batch:
                # Single insert is faster than per-message inserts
                self._widget.insert(tk.END, "".join(batch))
                self._widget.see(tk.END)

        except tk.TclError:
            # widget was destroyed between the after() and now
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
        # Use redir.write() directly so it goes through the queue
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
#  Backward-compatible alias
# ─────────────────────────────────────────────────────────────────────
#  Some modules import ``StdoutRedirector`` by name.  This alias lets
#  them keep working during the transition while you migrate them
#  one at a time to ``setup_console()``.

StdoutRedirector = ThreadSafeRedirector
