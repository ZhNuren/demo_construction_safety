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

    def clear_roi(self):
        self._roi = None
        if self._roi_rect_id:
            self.canvas.delete(self._roi_rect_id)
            self._roi_rect_id = None

    # Video controls
    def open(self, path: str):
        if cv2 is None:
            messagebox.showwarning("Dependency missing",
                                   "Install opencv-python and pillow for video preview")
            return
        self.close()
        self._cap = cv2.VideoCapture(path)
        self._stop = False
        self._loop_thread = threading.Thread(target=self._loop, daemon=True)
        self._loop_thread.start()

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

    def pause(self): self._stop = True

    def resume(self):
        if self._cap is not None and self._stop:
            self._stop = False
            if not (self._loop_thread and self._loop_thread.is_alive()):
                self._loop_thread = threading.Thread(target=self._loop, daemon=True)
                self._loop_thread.start()

    def close(self):
        self._stop = True
        if self._cap is not None:
            try: self._cap.release()
            except Exception: pass
            self._cap = None

    def _loop(self):
        while not self._stop and self._cap is not None:
            ok, frame = self._cap.read()
            if not ok: break
            self._frame = frame
            if self.on_frame is not None:
                try:
                    self._frame = self.on_frame(self._frame)
                except Exception as e:
                    print("on_frame error:", e)
            self._render_frame()
            time.sleep(1/30)

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
