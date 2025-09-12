import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional, Callable, Tuple
import threading, time


try:
    import cv2
    from PIL import Image, ImageTk
except Exception:
    cv2 = None
    Image = None
    ImageTk = None

import mss


import numpy as np

class VideoPlayer(ttk.Frame):
    """Video/image preview with ROI drawing and on_frame processing hook."""
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.canvas = tk.Canvas(self, bg="#0b0f14", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self._cap: Optional["cv2.VideoCapture"] = None
        self._frame = None
        self._stop = True
        self._loop_thread: Optional[threading.Thread] = None

        self.on_frame: Optional[Callable[[np.ndarray], np.ndarray]] = None

        self._drawing = False
        self._roi_start: Optional[Tuple[int,int]] = None
        self._roi_rect_id: Optional[int] = None
        self._roi: Optional[Tuple[int,int,int,int]] = None

        self.canvas.bind("<Configure>", lambda e: self._render_frame())

                # ADD in __init__
        self._source_mode = "none"   # "none" | "file" | "camera" | "screen"
        self._screen_ctx = None      # mss.mss() object
        self._screen_monitor = None  # dict with 'left','top','width','height'
        self._target_fps = 30
        self._lock = threading.RLock()   # <-- ADD
        self._black_on_close = True   # preference: show black when closed
        self._show_black = False      # state flag
        self._last_source = {"mode": "none"}   # remembers args to restart (file/camera/screen)
        self._infer_lock = threading.RLock()

    # ROI drawing
    def enable_roi_draw(self):
        self.canvas.bind("<Button-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)

    def disable_roi_draw(self):
        self.canvas.unbind("<Button-1>")
        self.canvas.unbind("<B1-Motion>")
        self.canvas.unbind("<ButtonRelease-1>")

    def _on_mouse_down(self, event):
        self._drawing = True
        self._roi_start = (event.x, event.y)
        if self._roi_rect_id:
            self.canvas.delete(self._roi_rect_id)
            self._roi_rect_id = None

    def _on_mouse_drag(self, event):
        if not self._drawing: return
        x0, y0 = self._roi_start
        x1, y1 = event.x, event.y
        if self._roi_rect_id:
            self.canvas.coords(self._roi_rect_id, x0, y0, x1, y1)
        else:
            self._roi_rect_id = self.canvas.create_rectangle(
                x0, y0, x1, y1, outline="#4ade80", width=2, dash=(4,2)
            )

    def _on_mouse_up(self, event):
        if not self._drawing: return
        x0, y0 = self._roi_start
        x1, y1 = event.x, event.y
        self._roi = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        self._drawing = False

    def get_roi(self): return self._roi

    def _start_loop(self):
        if self._loop_thread and self._loop_thread.is_alive():
            return
        self._stop = False
        self._loop_thread = threading.Thread(target=self._loop, daemon=True)
        self._loop_thread.start()
        
    def clear_roi(self):
        self._roi = None
        if self._roi_rect_id:
            self.canvas.delete(self._roi_rect_id)
            self._roi_rect_id = None

    # Video controls
    def open(self, path: str):
        self._show_black = False
        self._last_source = {"mode": "file", "path": path}   # <-- add
        if cv2 is None:
            messagebox.showwarning("Dependency missing", "Install opencv-python and pillow for video preview")
            return
        self.close()
        self._cap = cv2.VideoCapture(path)
        self._source_mode = "file"   # <-- ADD this line
        self._start_loop()

    def show_image(self, path: str):
        if cv2 is None:
            messagebox.showwarning("Dependency missing", "Install pillow for image preview")
            return
        self.close()
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Open image", "Failed to load image")
            return
        self._frame = img
        self._render_frame()

    def open_camera(self, index: int = 0):
        self._show_black = False
        self._last_source = {"mode": "camera", "index": int(index)}  # <-- add
        if cv2 is None:
            messagebox.showwarning("Dependency missing", "Install opencv-python for camera")
            return
        self.close()
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            messagebox.showerror("Camera", f"Cannot open camera index {index}")
            self._cap = None
            return
        self._source_mode = "camera"
        self._start_loop()

    def open_screen(self, monitor: int = 1, fps: int = 20, region: tuple[int,int,int,int] | None = None):
        """
        monitor: which monitor (1 = primary). If region is given, it overrides monitor bounds.
        region: (left, top, width, height)
        """
        self._show_black = False
        self._last_source = {
            "mode": "screen", "monitor": int(monitor), "fps": int(fps), "region": region
        }
        if mss is None:
            messagebox.showwarning("Dependency missing", "Install mss for screen capture")
            return
        self.close()
        self._target_fps = max(5, min(60, int(fps)))
        self._screen_ctx = mss.mss()
        mons = self._screen_ctx.monitors
        if not mons or monitor >= len(mons):
            monitor = 1
        mon = mons[monitor]
        if region is not None:
            left, top, width, height = region
            mon = {"left": int(left), "top": int(top), "width": int(width), "height": int(height)}
        self._screen_monitor = mon
        self._source_mode = "screen"
        self._start_loop()

    def pause(self): self._stop = True

    def restart(self):
        """
        Restart the current source.
        - file: re-open from the beginning
        - camera/screen: re-open with the same parameters
        """
        mode = self._last_source.get("mode", "none")
        if mode == "none":
            return

        # Stop current loop & release cleanly
        self.close()
        # ✅ Ensure no inference is running
        if hasattr(self, "_infer_lock"):
            with self._infer_lock:
                pass  # wait until any active on_frame finishes 
        self._show_black = False

        try:
            if mode == "file":
                path = self._last_source.get("path")
                if path:
                    self.open(path)
            elif mode == "camera":
                idx = self._last_source.get("index", 0)
                self.open_camera(index=idx)
            elif mode == "screen":
                mon = self._last_source.get("monitor", 0)
                fps = self._last_source.get("fps", 20)
                region = self._last_source.get("region", None)
                self.open_screen(monitor=mon, fps=fps, region=region)
        except Exception as e:
            # If anything fails, fall back to black screen
            print("restart error:", e)
            self._show_black = True
            self._frame = None
            self._render_frame()


    def resume(self):
        self._show_black = False
        if self._cap is not None and self._stop:
            self._stop = False
            if not (self._loop_thread and self._loop_thread.is_alive()):
                self._loop_thread = threading.Thread(target=self._loop, daemon=True)
                self._loop_thread.start()

    def close(self):
        # Stop loop
        self._stop = True

        # Join thread if running
        t = self._loop_thread
        if t and t.is_alive():
            try:
                t.join(timeout=0.5)
            except Exception:
                pass
        self._loop_thread = None

        # wait until on_frame finishes
        if hasattr(self, "_infer_lock"):
            with self._infer_lock:
                pass
        # Release resources
        if self._cap is not None:
            try: self._cap.release()
            except Exception: pass
            self._cap = None

        if self._screen_ctx is not None:
            try: self._screen_ctx.close()
            except Exception: pass
            self._screen_ctx = None

        self._screen_monitor = None
        self._source_mode = "none"
        self._frame = None

        # Switch to black placeholder and repaint
        self._show_black = True
        try:
            self._render_frame()
        except Exception:
            # Fallback to clearing if render during teardown fails
            self.canvas.after(0, lambda: self.canvas.delete("all"))



    def _loop(self):
        while not self._stop:
            self._show_black = False
            # If widget destroyed, exit gracefully
            try:
                if not self.winfo_exists():
                    break
            except Exception:
                break

            frame = None
            try:
                if self._source_mode in ("file", "camera"):
                    cap = self._cap
                    if cap is None:
                        break
                    ok, f = cap.read()
                    if not ok:
                        # For files: end of stream -> stop loop
                        # For cameras: break as well; user can reopen
                        break
                    frame = f

                elif self._source_mode == "screen":
                    if self._screen_ctx is None or self._screen_monitor is None:
                        break
                    shot = self._screen_ctx.grab(self._screen_monitor)  # BGRA
                    frame = np.array(shot)[:, :, :3].copy()  # -> BGR (drop alpha), NO channel reversal

                else:
                    break

                self._frame = frame

                # Apply processing hook
                if self.on_frame is not None:
                    try:
                        with self._infer_lock:           # <-- lock
                            self._frame = self.on_frame(self._frame)
                    except Exception as e:
                        print("on_frame error:", e)

                # Render (guard against Tk shutdown)
                try:
                    self._render_frame()
                except Exception as e:
                    print("render error:", e)
                    break

                # pacing
                if self._source_mode == "screen":
                    time.sleep(1 / float(self._target_fps))
                else:
                    time.sleep(1 / 30)

            except Exception as e:
                # Any unexpected error: log and exit loop cleanly
                print("loop error:", e)
                break

    def _render_frame(self):
        if self._frame is None or Image is None:
            self.canvas.delete("all")
            self.canvas.create_text(
                self.canvas.winfo_width()//2,
                self.canvas.winfo_height()//2,
                text="Video/Image Preview",
                fill="#8aa0b6",
                font=("Helvetica", 14, "bold"),
            )
            return
        frame = self._frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()

        # Show solid black only if requested AND no frame is available
        if self._show_black and self._frame is None:
            self.canvas.delete("all")
            self.canvas.create_rectangle(0, 0, max(1, cw), max(1, ch), fill="#000000", outline="")
            return

        if self._frame is None or Image is None:
            self.canvas.delete("all")
            self.canvas.create_text(
                cw // 2, ch // 2,
                text="Video/Image Preview",
                fill="#8aa0b6",
                font=("Helvetica", 14, "bold"),
            )
            return

        scale = min(cw / w, ch / h) if w and h and cw and ch else 1
        nw, nh = max(1, int(w*scale)), max(1, int(h*scale))
        frame_resized = cv2.resize(frame, (nw, nh))
        img = Image.fromarray(frame_resized)
        self._tk_img = ImageTk.PhotoImage(image=img)
        self.canvas.delete("all")
        self.canvas.create_image(cw//2, ch//2, image=self._tk_img, anchor=tk.CENTER)
        if self._roi is not None:
            x0,y0,x1,y1 = self._roi
            self._roi_rect_id = self.canvas.create_rectangle(
                x0,y0,x1,y1, outline="#4ade80", width=2, dash=(4,2)
            )
