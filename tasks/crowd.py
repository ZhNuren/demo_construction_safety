# tasks/crowd.py
from __future__ import annotations
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import cv2
from typing import Optional

from .base import TaskPage
from models.crowd import CrowdCounter


class CrowdPage(TaskPage):
    def __init__(self, master, **kwargs):
        self._counter: Optional[CrowdCounter] = None
        self._enabled = False
        self._use_roi = False   # flag: count inside ROI vs whole frame
        self._alert_count_thr = 10
        self._alert_time_thr = 3.0  # seconds
        self._over_count_frames = 0
        self._fps = 30  # assume ~30 fps
        self._roi_scale = 2.0
        self._preprocess = tk.BooleanVar(value=True)
        super().__init__(master, task_key="Crowd Counting", task_title="Crowd Density", **kwargs)

    def _build_controls(self):
        super()._build_controls()

        nb = ttk.Notebook(self.toolbar)
        nb.pack(side=tk.LEFT, padx=6)

        # --- Run tab ---
        tab_run = ttk.Frame(nb)
        nb.add(tab_run, text="Run")

        frm_r1 = ttk.Frame(tab_run, padding=(6, 6))
        frm_r1.pack(fill=tk.X)

        # Row 0 → start/stop + mode
        ttk.Button(frm_r1, text="▶ Start", width=8, command=self._start).grid(row=0, column=0, padx=4, pady=2)
        ttk.Button(frm_r1, text="■ Stop", width=8, command=self._stop).grid(row=0, column=1, padx=4, pady=2)
        self.mode_btn = ttk.Button(frm_r1, text="Mode: Full Frame", width=16, command=self._toggle_mode)
        self.mode_btn.grid(row=0, column=2, padx=10, pady=2)

        # Row 1 → thresholds
        ttk.Label(frm_r1, text="Count ≥").grid(row=1, column=0, sticky="e", padx=(4, 2), pady=2)
        self.count_thr_var = tk.IntVar(value=self._alert_count_thr)
        ttk.Spinbox(frm_r1, from_=1, to=500, width=6,
                    textvariable=self.count_thr_var).grid(row=1, column=1, sticky="w", padx=(0, 8), pady=2)

        ttk.Label(frm_r1, text="Time ≥ s").grid(row=1, column=2, sticky="e", padx=(4, 2), pady=2)
        self.time_thr_var = tk.DoubleVar(value=self._alert_time_thr)
        ttk.Spinbox(frm_r1, from_=0.5, to=60, increment=0.5, width=6,
                    textvariable=self.time_thr_var).grid(row=1, column=3, sticky="w", padx=(0, 8), pady=2)

        # --- ROI tab ---
        tab_roi = ttk.Frame(nb)
        nb.add(tab_roi, text="ROI")

        frm_roi = ttk.Frame(tab_roi, padding=(6, 6))
        frm_roi.pack(fill=tk.X)

        ttk.Label(frm_roi, text="Region of Interest").grid(row=0, column=0, columnspan=2, pady=(0, 6))

        ttk.Button(frm_roi, text="Draw ROI", width=12, command=self.player.enable_roi_draw).grid(row=1, column=0, padx=6, pady=2)
        ttk.Button(frm_roi, text="Clear ROI", width=12, command=self.player.clear_roi).grid(row=1, column=1, padx=6, pady=2)

        ttk.Label(frm_roi, text="Scale ×").grid(row=2, column=0, padx=(6, 2), pady=4)
        self.scale_var = tk.DoubleVar(value=self._roi_scale)
        ttk.Spinbox(frm_roi, from_=1.0, to=4.0, increment=0.5, width=6,
                    textvariable=self.scale_var, command=self._update_scale).grid(row=2, column=1, pady=4)

        ttk.Checkbutton(frm_roi, text="Preprocess ROI", variable=self._preprocess).grid(row=3, column=0, columnspan=2, pady=4)

    def _update_scale(self):
        try:
            self._roi_scale = float(self.scale_var.get())
        except Exception:
            self._roi_scale = 2.0

    def _toggle_mode(self):
        self._use_roi = not self._use_roi
        mode = "ROI" if self._use_roi else "Full"
        self.mode_btn.config(text=f"Mode: {mode}")
        self.notify(f"Crowd counting mode → {mode}")

    def _ensure_counter(self):
        if self._counter is None:
            try:
                self._counter = CrowdCounter(det_model="yolo11x.pt")
            except Exception as e:
                messagebox.showerror("Crowd Counter init failed", str(e))
                return None
        return self._counter

    def _start(self):
        if self._ensure_counter() is None:
            return
        self._enabled = True
        self.player.on_frame = self._process
        self.notify("Crowd counting started")

    def _stop(self):
        self._enabled = False
        self.player.on_frame = None
        self.notify("Stopped")

    def _process(self, frame: np.ndarray) -> np.ndarray:
        if not self._enabled:
            return frame
        counter = self._ensure_counter()
        if counter is None:
            return frame

        roi = self.player.get_roi()
        boxes = []
        num_people = 0

        if self._use_roi and roi is not None:
            x0, y0, x1, y1 = roi
            roi_frame = frame[y0:y1, x0:x1]

            # --- optional preprocessing ---
            if self._preprocess.get():
                roi_frame = cv2.GaussianBlur(roi_frame, (3, 3), 0)
                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                roi_frame = cv2.filter2D(roi_frame, -1, kernel)
                roi_frame = cv2.convertScaleAbs(roi_frame, alpha=1.3, beta=15)

            # --- scale / zoom ---
            scale_factor = self._roi_scale
            roi_h, roi_w = roi_frame.shape[:2]
            zoomed = cv2.resize(
                roi_frame,
                (int(roi_w * scale_factor), int(roi_h * scale_factor)),
                interpolation=cv2.INTER_CUBIC
            )

            res = counter.detector.predict(zoomed, verbose=False, conf=0.4, device="mps", classes=[0])[0]
            boxes = res.boxes.xyxy.cpu().numpy().astype(int).tolist()
            num_people = len(boxes)

            # map boxes back
            for (bx1, by1, bx2, by2) in boxes:
                bx1 = int(bx1 / scale_factor) + x0
                bx2 = int(bx2 / scale_factor) + x0
                by1 = int(by1 / scale_factor) + y0
                by2 = int(by2 / scale_factor) + y0
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 255), 2)

            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 200, 200), 2)
        else:
            res = counter.detector.predict(frame, verbose=False, conf=0.4, device="mps", classes=[0])[0]
            boxes = res.boxes.xyxy.cpu().numpy().astype(int).tolist()
            num_people = len(boxes)
            for (bx1, by1, bx2, by2) in boxes:
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)

        # thresholds
        count_thr = int(self.count_thr_var.get())
        time_thr_s = float(self.time_thr_var.get())
        frame_thr = int(time_thr_s * self._fps)

        if num_people >= count_thr:
            self._over_count_frames += 1
        else:
            self._over_count_frames = 0

        if self._over_count_frames >= frame_thr:
            cv2.putText(frame, "ALERT: CROWD DETECTED!", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        else:
            cv2.putText(frame, f"People: {num_people}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        return frame
