# tasks/violence.py
from __future__ import annotations
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import cv2
from typing import Optional

from .base import TaskPage
from models.violence import ViolenceDetector


class ViolencePage(TaskPage):
    """
    Tkinter page that runs violence detection.
    Alerts immediately when any relevant detection appears (no persistence window).
    """

    def __init__(self, master, **kwargs):
        self._detector: Optional[ViolenceDetector] = None
        self._enabled = False

        # UI-configurable bits (optional)
        self._conf_default = 0.4
        self._class_whitelist_default: list[str] | None = None  # e.g., ['fight', 'violence']

        super().__init__(master, task_key="Violence Detection", task_title="Violence / Fight Detection", **kwargs)

    def _build_controls(self):
        super()._build_controls()

        nb = ttk.Notebook(self.toolbar)
        nb.pack(side=tk.LEFT, padx=6)

        # --- Run tab
        tab_run = ttk.Frame(nb)
        nb.add(tab_run, text="Run")

        frm = ttk.Frame(tab_run, padding=(6, 6))
        frm.pack(fill=tk.X)

        ttk.Button(frm, text="▶ Start", width=8, command=self._start).grid(row=0, column=0, padx=4, pady=2)
        ttk.Button(frm, text="■ Stop", width=8, command=self._stop).grid(row=0, column=1, padx=4, pady=2)

        ttk.Label(frm, text="Conf ≥").grid(row=1, column=0, sticky="e", padx=(4, 2))
        self.conf_var = tk.DoubleVar(value=self._conf_default)
        ttk.Spinbox(frm, from_=0.1, to=0.99, increment=0.05, width=6,
                    textvariable=self.conf_var).grid(row=1, column=1, sticky="w", padx=(0, 8))

        # --- ROI tab (optional; uses player's ROI if available)
        tab_roi = ttk.Frame(nb)
        nb.add(tab_roi, text="ROI")
        frm_roi = ttk.Frame(tab_roi, padding=(6, 6))
        frm_roi.pack(fill=tk.X)
        ttk.Button(frm_roi, text="Draw ROI", width=12,
                   command=lambda: (self.player.enable_roi_draw(), self.notify("ROI enabled"))).grid(row=0, column=0, padx=6, pady=2)
        ttk.Button(frm_roi, text="Clear ROI", width=12,
                   command=lambda: (self.player.clear_roi(), self.notify("ROI cleared"))).grid(row=0, column=1, padx=6, pady=2)

    def _ensure_detector(self):
        if self._detector is None:
            try:
                self._detector = ViolenceDetector(
                    det_model="violence.pt",
                    conf=float(self.conf_var.get()),
                    # Set to a list like ['violence', 'fight'] if your model has multiple labels and you only want some
                    target_classes=self._class_whitelist_default,
                    device="cpu",
                    draw=True,
                )
                self.notify("Violence model loaded: violence.pt")
            except Exception as e:
                messagebox.showerror("Violence Detector init failed", str(e))
                return None
        return self._detector

    def _start(self):
        # Recreate detector to apply any changed UI params
        self._detector = None
        if self._ensure_detector() is None:
            return
        self._enabled = True
        self.player.on_frame = self._process
        self.notify("Violence detection started")

    def _stop(self):
        self._enabled = False
        self.player.on_frame = None
        self.notify("Stopped")

    def _process(self, frame: np.ndarray) -> np.ndarray:
        detector = self._ensure_detector()
        if detector is None:
            return frame

        alert = False
        if self._enabled:
            roi = self.player.get_roi()
            if roi is not None:
                x0, y0, x1, y1 = roi
                roi_frame = frame[y0:y1, x0:x1].copy()
                roi_frame, alert = detector.process(roi_frame)
                frame[y0:y1, x0:x1] = roi_frame

                # Draw ROI overlay last
                color = (0, 0, 255) if alert else (0, 255, 0)
                cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
            else:
                frame, alert = detector.process(frame)

        # Global banner (the model already draws labels/boxes)
        if alert:
            cv2.putText(frame, "ALERT: VIOLENCE DETECTED!", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        return frame
