
from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox

from .base import TaskPage
from models.objects import ObjectPresenceDetector


class ObjectsPage(TaskPage):
    """
    Tkinter page for object presence detection.
    - Start/Stop controls
    - Full-frame vs ROI mode
    - Live time-threshold changes
    - Single, readable ALERT banner drawn once (UI side)
    """

    def __init__(self, master, **kwargs):
        self._detector: Optional[ObjectPresenceDetector] = None
        self._enabled = False
        self._use_roi = False   # detect inside ROI vs whole frame
        self._alert_time_thr = 5.0  # seconds
        self._processing = False     # re-entrancy guard for _process
        self._last_infer_ts = 0.0    # simple FPS throttle
        self._infer_min_gap = 0.0   # ~7 FPS; adjust as needed
        self._last_alert_state = False  # remember last alert for skipped frames

        super().__init__(master, task_key="Object Detection", task_title="Box/Plastic Bag Detection", **kwargs)

    # ---------- UI ----------

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

        # Row 1 → time threshold
        ttk.Label(frm_r1, text="Time ≥ s").grid(row=1, column=0, sticky="e", padx=(4, 2), pady=2)
        self.time_thr_var = tk.DoubleVar(value=self._alert_time_thr)
        ttk.Spinbox(frm_r1, from_=1, to=60, increment=1, width=6,
                    textvariable=self.time_thr_var).grid(row=1, column=1, sticky="w", padx=(0, 8), pady=2)

        # 🔗 Live-update detector when the spinbox changes
        def _on_time_thr_change(*_):
            if self._detector is not None:
                self._detector.set_persistence(self.time_thr_var.get(), reset=True)
                self._last_alert_state = False
                self.notify(f"Time threshold → {self.time_thr_var.get():.1f}s (timers reset)")
        self.time_thr_var.trace_add("write", _on_time_thr_change)

        # --- ROI tab ---
        tab_roi = ttk.Frame(nb)
        nb.add(tab_roi, text="ROI")

        frm_roi = ttk.Frame(tab_roi, padding=(6, 6))
        frm_roi.pack(fill=tk.X)

        ttk.Label(frm_roi, text="Region of Interest").grid(row=0, column=0, columnspan=2, pady=(0, 6))

        ttk.Button(
            frm_roi,
            text="Draw ROI",
            width=12,
            command=lambda: (self.player.enable_roi_draw(), self._enable_roi())
        ).grid(row=1, column=0, padx=6, pady=2)

        ttk.Button(
            frm_roi,
            text="Clear ROI",
            width=12,
            command=lambda: (self.player.clear_roi(), self._disable_roi())
        ).grid(row=1, column=1, padx=6, pady=2)

    # ---------- Helpers ----------

    def _draw_alert_banner(self, frame: np.ndarray, text: str = "ALERT: OBJECT STAYED TOO LONG!") -> None:
        """Readable, auto-scaled banner drawn once on the full frame."""
        h, w = frame.shape[:2]
        scale = max(0.9, min(3.0, h / 720 * 1.2))
        thickness = max(2, int(2 * scale))
        (tw, th_txt), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)

        # Center near top
        x = (w - tw) // 2
        y = int(60 * scale)

        # Semi-opaque red rectangle behind text
        pad = int(12 * scale)
        x1, y1 = max(0, x - pad), max(0, y - th_txt - pad // 2)
        x2, y2 = min(w - 1, x + tw + pad), min(h - 1, y + baseline + pad // 2)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

        # Outline + fill
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    def _enable_roi(self):
        self._use_roi = True
        if self._detector:
            self._detector.reset()
        self._last_alert_state = False
        self.notify("ROI enabled")

    def _disable_roi(self):
        self._use_roi = False
        if self._detector:
            self._detector.reset()
        self._last_alert_state = False
        self.notify("ROI disabled")

    def _toggle_mode(self):
        self._use_roi = not self._use_roi
        if self._detector:
            self._detector.reset()
        self._last_alert_state = False
        mode = "ROI" if self._use_roi else "Full Frame"
        self.mode_btn.config(text=f"Mode: {mode}")
        self.notify(f"Object detection mode → {mode}")

    def _ensure_detector(self) -> Optional[ObjectPresenceDetector]:
        if self._detector is None:
            try:
                self._detector = ObjectPresenceDetector(
                    det_model="best.pt",
                    conf=0.4,
                    device="cpu",
                    persistence_sec=float(self.time_thr_var.get()),
                    dist_thr=100.0,
                    show_timers=True,
                )
            except Exception as e:
                messagebox.showerror("Object Detector init failed", str(e))
                return None
        return self._detector

    # ---------- Lifecycle ----------

    def _start(self):
        det = self._ensure_detector()
        if det is None:
            return
        # Ensure latest threshold + clean timers
        det.set_persistence(self.time_thr_var.get(), reset=True)
        self._last_alert_state = False
        self._enabled = True
        self.player.on_frame = self._process
        self.notify("Object detection started")

    def _stop(self):
        self._enabled = False
        self.player.on_frame = None
        if self._detector:
            self._detector.reset()
        self._last_alert_state = False
        self.notify("Stopped")

    # ---------- Frame callback ----------

    def _process(self, frame: np.ndarray) -> np.ndarray:
        # Keep UI responsive: avoid re-entrancy
        if self._processing:
            # Even when skipping, draw current ROI outline for user feedback
            roi = self.player.get_roi()
            if roi is not None:
                x0, y0, x1, y1 = roi
                color = (0, 0, 255) if self._last_alert_state else (0, 255, 0)
                cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
            if self._last_alert_state:
                self._draw_alert_banner(frame)
            return frame

        self._processing = True
        try:
            detector = self._ensure_detector()
            if detector is None:
                return frame

            roi = self.player.get_roi()
            alert = self._last_alert_state  # default to last known when skipping

            if self._enabled:
                now_ts = time.time()
                # Optional throttle to keep UI snappy (especially on CPU)
                if now_ts - self._last_infer_ts < self._infer_min_gap:
                    # Draw ROI border and previous alert state, then return
                    if roi is not None:
                        x0, y0, x1, y1 = roi
                        color = (0, 0, 255) if self._last_alert_state else (0, 255, 0)
                        cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
                    if self._last_alert_state:
                        self._draw_alert_banner(frame)
                    return frame

                self._last_infer_ts = now_ts

                if self._use_roi and roi is not None:
                    x0, y0, x1, y1 = roi
                    roi_frame = frame[y0:y1, x0:x1].copy()
                    roi_frame, alert = detector.process(roi_frame)
                    frame[y0:y1, x0:x1] = roi_frame
                else:
                    frame, alert = detector.process(frame)

                # Persist alert state for skipped frames
                self._last_alert_state = bool(alert)

            # Draw ROI border last and in the correct color
            if roi is not None:
                x0, y0, x1, y1 = roi
                color = (0, 0, 255) if self._last_alert_state else (0, 255, 0)
                cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)

            # Draw ONE global banner here
            if self._last_alert_state:
                self._draw_alert_banner(frame, "ALERT: OBJECT STAYED TOO LONG!")

            return frame
        finally:
            self._processing = False


