from typing import Optional
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np

from .base import TaskPage
from models.detector import YOLODetector
from tracking.simple_tracker import SimpleTracker

class ObjectTrackingPage(TaskPage):
    def __init__(self, master, **kwargs):
        super().__init__(master, task_key="Object Tracking", task_title="Tracking with trails", **kwargs)
        self._tracker_enabled = False
        self._tracker = SimpleTracker(max_lost=20, iou_thr=0.35, trail=40)
        self._detector: Optional[YOLODetector] = None

    def _build_controls(self):
        super()._build_controls()
        ttk.Separator(self.toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=12)
        section = ttk.Frame(self.toolbar)
        section.pack(side=tk.LEFT)
        ttk.Button(section, text="Start tracking", command=self._start_tracking).grid(row=0, column=0)
        ttk.Button(section, text="Stop", command=self._stop_tracking).grid(row=0, column=1, padx=6)
        ttk.Button(section, text="Clear trails", command=self._clear_trails).grid(row=0, column=2)
        ttk.Label(section, text="Classes:").grid(row=1, column=0, pady=(6,0))
        self.class_mode = tk.StringVar(value="all")
        ttk.Combobox(section, state="readonly", width=12,
                     values=["all", "person-only", "animal-ish"],
                     textvariable=self.class_mode).grid(row=1, column=1, columnspan=2, sticky="w", pady=(6,0))

    # ---- actions ----
    def _ensure_detector(self):
        if self._detector is None:
            try:
                self._detector = YOLODetector("yolov8n.pt", conf=0.35, imgsz=640)
            except Exception as e:
                messagebox.showerror("Detector init failed", str(e))
                return None
        return self._detector

    def _start_tracking(self):
        if self._ensure_detector() is None: return
        self._tracker_enabled = True
        self.player.on_frame = self._process_tracking_frame
        self.notify("Tracking started")

    def _stop_tracking(self):
        self._tracker_enabled = False
        self.player.on_frame = None
        self.notify("Tracking stopped")

    def _clear_trails(self):
        for tr in self._tracker.tracks.values():
            tr['trail'].clear()
        self.notify("Trails cleared")

    def _cls_filter(self):
        mode = self.class_mode.get()
        if mode == "person-only":
            return [0]  # COCO person
        elif mode == "animal-ish":
            # COCO animal-ish class ids
            return [14,15,16,17,18,19,20,21,22,23]  # bird->zebra
        return None

    # ---- per-frame processing ----
    def _process_tracking_frame(self, frame: np.ndarray) -> np.ndarray:
        if not self._tracker_enabled: return frame
        det = self._ensure_detector()
        if det is None: return frame

        xyxy, scores, clss = det.detect_xyxy(frame, classes=self._cls_filter())
        tracks = self._tracker.update([tuple(map(int,b)) for b in xyxy], scores)

        for tid, tr in tracks.items():
            x1,y1,x2,y2 = tr['bbox']
            cv2.rectangle(frame, (x1,y1), (x2,y2), (50,205,50), 2)
            cv2.putText(frame, f"ID {tid}", (x1, max(20,y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            pts = list(tr['trail'])
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i-1], pts[i], (50,205,50), 2)
        return frame
