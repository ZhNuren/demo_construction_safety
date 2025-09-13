# tasks/ppe.py
from __future__ import annotations
from typing import Optional, Dict, Set, Tuple, List

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import cv2

from .base import TaskPage
from tracking.simple_tracker import SimpleTracker
from ultralytics import YOLO


class PPEPage(TaskPage):
    """
    Person + PPE detection:
      - Detects persons (YOLO, class 0)
      - Tracks persons with SimpleTracker to stabilize IDs across frames
      - For each person track, crops with padding and runs PPE YOLO
      - Maps PPE boxes from crop back to the original frame
      - If any required PPE is missing for >= hold time, shows an alert and highlights that person
    """

    # Edit these to match your PPE checkpoint class order
    PPE_CLASSES = ["Hardhat", "Sensor", "Bag"]
    # default colors for PPE classes (BGR)
    PPE_COLORS = {
        "Hardhat": (0, 255, 255),  # yellow
        "Sensor":  (255, 0, 255),  # magenta
        "Bag":     (0, 165, 255),  # orange
    }

    def __init__(self, master, **kwargs):
        # ---- runtime state used by _build_controls must be created BEFORE super().__init__ ----
        self._enabled = False
        self._fps = 30  # used for hold time → frames

        # person & ppe detection knobs
        self._person_conf = tk.DoubleVar(value=0.35)
        self._ppe_conf = tk.DoubleVar(value=0.25)

        # device (auto-select mps if available, else cpu)
        self._device = "mps"
        try:
            import torch
            if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                self._device = "cpu"
        except Exception:
            self._device = "cpu"

        # required PPE toggles (default: require hardhat only)
        self._req_hardhat = tk.BooleanVar(value=True)
        self._req_sensor  = tk.BooleanVar(value=False)
        self._req_bag     = tk.BooleanVar(value=False)

        # alert hold
        self._hold_seconds = tk.DoubleVar(value=1.0)

        # ROI option
        self._use_roi = tk.BooleanVar(value=False)

        # trackers / models
        self._tracker = SimpleTracker(max_lost=15, iou_thr=0.35, trail=100)
        self._person_model: Optional[YOLO] = None
        self._ppe_model: Optional[YOLO] = None

        # per-track miss counters for hold logic
        self._miss_frames_by_tid: Dict[int, int] = {}

        super().__init__(master, task_key="PPE", task_title="PPE Compliance", **kwargs)

    # ---------------- UI ----------------
    def _build_controls(self):
        super()._build_controls()

        nb = ttk.Notebook(self.toolbar)
        nb.pack(side=tk.LEFT, padx=8)

        # --- Run tab ---
        tab_run = ttk.Frame(nb)
        nb.add(tab_run, text="Run")
        r = ttk.Frame(tab_run, padding=(6, 6))
        r.pack(fill=tk.X)

        ttk.Button(r, text="▶ Start", command=self._start, width=10).grid(row=0, column=0, padx=4, pady=2)
        ttk.Button(r, text="■ Stop",  command=self._stop,  width=10).grid(row=0, column=1, padx=4, pady=2)

        # --- Settings tab ---
        tab_set = ttk.Frame(nb)
        nb.add(tab_set, text="Settings")
        s = ttk.Frame(tab_set, padding=(6, 6))
        s.pack(fill=tk.X)

        ttk.Label(s, text="Person conf").grid(row=0, column=0, sticky="e")
        ttk.Spinbox(s, from_=0.05, to=0.9, increment=0.05, width=6, textvariable=self._person_conf)\
            .grid(row=0, column=1, sticky="w", padx=(4, 12))

        ttk.Label(s, text="PPE conf").grid(row=0, column=2, sticky="e")
        ttk.Spinbox(s, from_=0.05, to=0.9, increment=0.05, width=6, textvariable=self._ppe_conf)\
            .grid(row=0, column=3, sticky="w", padx=(4, 12))

        ttk.Label(s, text="Alert hold (s)").grid(row=1, column=0, sticky="e", pady=(6, 0))
        ttk.Spinbox(s, from_=0.0, to=10.0, increment=0.5, width=6, textvariable=self._hold_seconds)\
            .grid(row=1, column=1, sticky="w", padx=(4, 12), pady=(6, 0))

        ttk.Label(s, text="Require:").grid(row=1, column=2, sticky="e", pady=(6, 0))
        req = ttk.Frame(s)
        req.grid(row=1, column=3, sticky="w", pady=(6, 0))
        ttk.Checkbutton(req, text="Hardhat", variable=self._req_hardhat).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(req, text="Sensor",  variable=self._req_sensor).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(req, text="Bag",     variable=self._req_bag).pack(side=tk.LEFT, padx=2)

        # --- ROI tab ---
        tab_roi = ttk.Frame(nb)
        nb.add(tab_roi, text="ROI")
        rr = ttk.Frame(tab_roi, padding=(6, 6))
        rr.pack(fill=tk.X)
        ttk.Checkbutton(rr, text="Only check inside ROI", variable=self._use_roi).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Button(rr, text="Draw ROI",  width=12, command=self.player.enable_roi_draw).grid(row=1, column=0, padx=6, pady=2)
        ttk.Button(rr, text="Clear ROI", width=12, command=self.player.clear_roi).grid(row=1, column=1, padx=6, pady=2)

    # ---------------- models ----------------
    def _ensure_models(self):
        ok = True
        if self._person_model is None:
            try:
                # You can change to your preferred person model here (e.g., "yolo11x.pt")
                self._person_model = YOLO("yolo11x.pt")
            except Exception as e:
                ok = False
                messagebox.showerror("PPE", f"Failed to load person model: {e}")
        if self._ppe_model is None:
            try:
                # Your PPE checkpoint (3 classes: Hardhat, Sensor, Bag)
                self._ppe_model = YOLO("best_3_ppe.pt")
            except Exception as e:
                ok = False
                messagebox.showerror("PPE", f"Failed to load PPE model: {e}")
        return ok

    # ---------------- lifecycle ----------------
    def _start(self):
        if not self._ensure_models():
            return
        # reset counters
        self._miss_frames_by_tid.clear()

        self._enabled = True
        self.player.on_frame = self._process
        dev = "MPS" if self._device == "mps" else "CPU"
        self.notify(f"PPE started ({dev})")

    def _stop(self):
        self._enabled = False
        self.player.on_frame = None
        self.notify("PPE stopped")

    # ---------------- helpers ----------------
    @staticmethod
    def _clip_box(x1, y1, x2, y2, w, h) -> Tuple[int, int, int, int]:
        x1 = max(0, min(int(x1), w))
        y1 = max(0, min(int(y1), h))
        x2 = max(0, min(int(x2), w))
        y2 = max(0, min(int(y2), h))
        return x1, y1, x2, y2

    @staticmethod
    def _expand_and_crop(frame: np.ndarray, box: Tuple[int, int, int, int], gain=1.02, pad=20):
        """Expand a bbox slightly and crop. Returns crop and top-left offset."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, box)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        bw = (x2 - x1) * gain + pad
        bh = (y2 - y1) * gain + pad
        nx1 = int(cx - bw / 2)
        ny1 = int(cy - bh / 2)
        nx2 = int(cx + bw / 2)
        ny2 = int(cy + bh / 2)
        nx1, ny1, nx2, ny2 = PPEPage._clip_box(nx1, ny1, nx2, ny2, w, h)
        crop = frame[ny1:ny2, nx1:nx2].copy()
        return crop, (nx1, ny1)

    def _required_set(self) -> Set[int]:
        req = set()
        if self._req_hardhat.get():
            req.add(0)
        if self._req_sensor.get():
            req.add(1)
        if self._req_bag.get():
            req.add(2)
        return req

    def _inside_roi(self, bbox: Tuple[int, int, int, int]) -> bool:
        roi = self.player.get_roi()
        if not self._use_roi.get() or roi is None:
            return True
        x0, y0, x1, y1 = roi
        bx1, by1, bx2, by2 = bbox
        cx, cy = (bx1 + bx2) // 2, (by1 + by2) // 2
        return (x0 <= cx <= x1) and (y0 <= cy <= y1)

    # ---------------- main loop ----------------
    def _process(self, frame: np.ndarray) -> np.ndarray:
        if not self._enabled:
            return frame
        if self._person_model is None or self._ppe_model is None:
            return frame

        H, W = frame.shape[:2]
        person_conf = float(self._person_conf.get())
        ppe_conf = float(self._ppe_conf.get())

        # 1) Detect persons
        pres = self._person_model.predict(
            frame, verbose=False, conf=person_conf, device=self._device, classes=[0], iou=0.45, max_det=300
        )[0]
        boxes = pres.boxes.xyxy.cpu().numpy().astype(int) if len(pres) else np.empty((0, 4), dtype=int)
        scores = pres.boxes.conf.cpu().numpy().astype(float) if len(pres) else np.empty((0,), dtype=float)

        # 2) Track persons (stabilize IDs)
        tracks = self._tracker.update([tuple(map(int, b)) for b in boxes], scores)

        # 3) For each track, run PPE on expanded crop
        required = self._required_set()
        hold_frames = max(1, int(float(self._hold_seconds.get()) * self._fps))

        any_violation = False

        for tid, tr in tracks.items():
            x1, y1, x2, y2 = tr["bbox"]
            if not self._inside_roi((x1, y1, x2, y2)):
                # draw dimmed person box and skip PPE check
                cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 2)
                cv2.putText(frame, f"ID {tid}", (x1, max(20, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (192, 192, 192), 2)
                continue

            crop, (ox, oy) = self._expand_and_crop(frame, (x1, y1, x2, y2), gain=1.05, pad=25)
            if crop.size == 0:
                continue

            # Run PPE model on crop
            ppe_res = self._ppe_model.predict(
                crop, verbose=False, conf=ppe_conf, device=self._device, iou=0.45, max_det=200
            )[0]
            found: Set[int] = set()

            head_threshold = int(crop.shape[0] * 0.3)  # only accept hardhat detections near top
            for b in (ppe_res.boxes.xyxy.cpu().numpy().astype(int) if len(ppe_res) else []):
                px1, py1, px2, py2 = b
                cls_id = int(ppe_res.boxes.cls.cpu().numpy()[0]) if len(ppe_res.boxes.cls) == 1 else None
                # If model returns multiple classes, iterate with zip instead
            # Proper iteration (handles N detections)
            if len(ppe_res):
                bxs = ppe_res.boxes.xyxy.cpu().numpy().astype(int)
                cls_arr = ppe_res.boxes.cls.cpu().numpy().astype(int)
                conf_arr = ppe_res.boxes.conf.cpu().numpy().astype(float)
                for (px1, py1, px2, py2), c_id, cf in zip(bxs, cls_arr, conf_arr):
                    # accept hardhat only if near head region
                    if c_id == 0 and py1 > head_threshold:
                        continue
                    found.add(c_id)

                    # map PPE box to full frame
                    gx1, gy1 = ox + int(px1), oy + int(py1)
                    gx2, gy2 = ox + int(px2), oy + int(py2)
                    name = self.PPE_CLASSES[c_id] if 0 <= c_id < len(self.PPE_CLASSES) else f"ppe{c_id}"
                    color = self.PPE_COLORS.get(name, (255, 255, 0))
                    cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), color, 2)
                    cv2.putText(frame, f"{name}:{cf:.2f}", (gx1, gy2 + 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # compute missing set vs required
            missing = sorted(list(required - found))
            missed = len(missing) > 0

            # update hold counter
            cur = self._miss_frames_by_tid.get(tid, 0)
            cur = cur + 1 if missed else 0
            self._miss_frames_by_tid[tid] = cur

            # draw person box
            color = (0, 200, 0) if not missed else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID {tid}"
            if missed:
                miss_names = [self.PPE_CLASSES[i] if i < len(self.PPE_CLASSES) else f"ppe{i}" for i in missing]
                label += "  MISSING: " + ",".join(miss_names)
            cv2.putText(frame, label, (x1, max(20, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # global alert if any track violates long enough
            if missed and cur >= hold_frames:
                any_violation = True

            # draw trail (optional, from SimpleTracker)
            pts = list(tr["trail"])
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i - 1], pts[i], color, 2)

        # show alert banner
        if any_violation:
            cv2.putText(frame, "ALERT: PPE VIOLATION DETECTED!", (40, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        # If ROI is drawn, outline it
        roi = self.player.get_roi()
        if self._use_roi.get() and roi is not None:
            x0, y0, x1, y1 = roi
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 200, 200), 2)

        return frame
