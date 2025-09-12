from typing import Optional
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np

from .base import TaskPage
from models.detector import YOLODetector
from tracking.simple_tracker import SimpleTracker


COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
    "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
    "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
    "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

class ObjectTrackingPage(TaskPage):
    def __init__(self, master, **kwargs):
        super().__init__(master, task_key="Object Tracking", task_title="Tracking with trails", **kwargs)
        self._tracker_enabled = False
        self._tracker = SimpleTracker(max_lost=20, iou_thr=0.35, trail=120)
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
        ttk.Label(section, text="Trail").grid(row=2, column=0, pady=(6,0))
        self.trail_len = tk.IntVar(value=120)
        ttk.Spinbox(section, from_=10, to=1000, width=6, textvariable=self.trail_len).grid(row=2, column=1, sticky="w", pady=(6,0))
        ttk.Button(section, text="Apply", command=self._apply_trail_len).grid(row=2, column=2, padx=(6,0), pady=(6,0))

    def _ensure_detector(self):
        if self._detector is None:
            try:
                device = "mps"
                try:
                    import torch
                    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                        device = "cpu"
                except Exception:
                    device = "cpu"

                self._detector = YOLODetector("yolo11x.pt", conf=0.35, imgsz=640, device=device)
                if device == "mps":
                    self.notify("YOLO on MPS (Apple GPU)")
                else:
                    self.notify("YOLO on CPU (MPS not available)")
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
    
    def _apply_trail_len(self):
        n = int(self.trail_len.get())
        n = max(5, min(2000, n))
        # Update tracker default
        self._tracker.trail = n
        # Rebuild each track's deque with new maxlen
        from collections import deque
        for tr in self._tracker.tracks.values():
            tr['trail'] = deque(tr['trail'], maxlen=n)
        self.notify(f"Trail length set to {n}")

    def _cls_filter(self):
        mode = self.class_mode.get()
        if mode == "person-only":
            return [0]
        elif mode == "animal-ish":
            return [14,15,16,17,18,19,20,21,22,23]
        return None

    def _process_tracking_frame(self, frame: np.ndarray) -> np.ndarray:
        if not self._tracker_enabled:
            return frame
        det = self._ensure_detector()
        if det is None:
            return frame

        xyxy, scores, clss = det.detect_xyxy(frame, classes=self._cls_filter())

        # Convert to int bboxes and include class info
        dets = [(tuple(map(int, b)), float(s), int(c)) for b, s, c in zip(xyxy, scores, clss)]
        tracks = self._tracker.update([d[0] for d in dets], [d[1] for d in dets])

        # Attach class IDs to tracks (simple: match order of detections)
        for (bbox, _, cls), (tid, tr) in zip(dets, tracks.items()):
            tr['cls'] = cls

        # Define some colors for classes
        def class_color(cls_id: int) -> tuple[int,int,int]:
            # Person (0) = green, animals = orange, others = blue
            if cls_id == 0:
                return (50, 205, 50)  # lime green
            elif 14 <= cls_id <= 23:
                return (0, 140, 255)  # orange
            else:
                return (255, 128, 0)  # cyan

        for tid, tr in tracks.items():
            x1, y1, x2, y2 = tr['bbox']
            cls_id = tr.get('cls', -1)
            color = class_color(cls_id)

            # Get human-readable class name
            cls_name = COCO_CLASSES[cls_id] if 0 <= cls_id < len(COCO_CLASSES) else f"cls{cls_id}"

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label: ID + class name
            label = f"ID {tid} {cls_name}"
            cv2.putText(frame, label, (x1, max(20, y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Draw trail
            pts = list(tr['trail'])
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i-1], pts[i], color, 2)

        return frame
