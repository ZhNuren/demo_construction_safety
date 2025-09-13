# tasks/smoke_fire.py
from __future__ import annotations
from typing import Optional, Iterable, Set, Dict

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import cv2

from .base import TaskPage
from ultralytics import YOLO


class SmokeFirePage(TaskPage):
    """
    Custom YOLO smoke/fire detection.
      - Load your custom weights (browse or type path).
      - Optional ROI (only alert on detections inside ROI).
      - Alert only if >= N detections AND persistent for 'hold' seconds.
      - Area filter to ignore tiny boxes.
      - Draws individual boxes + a merged cluster box.
    """

    def __init__(self, master, **kwargs):
        # runtime state used by _build_controls must exist before super().__init__
        self._enabled = False
        self._model: Optional[YOLO] = None
        self._device = self._pick_device()

        # model params
        self._weights = tk.StringVar(value="smoke_fire2.pt")  # put your weights file here
        self._conf = tk.DoubleVar(value=0.35)
        self._iou = tk.DoubleVar(value=0.45)

        # alert logic
        self._count_thr = tk.IntVar(value=1)          # at least N detections
        self._hold_seconds = tk.DoubleVar(value=1.0)  # must persist this many seconds
        self._min_area_pct = tk.DoubleVar(value=0.002)  # ignore tiny boxes (fraction of frame area)

        # ROI control
        self._use_roi = tk.BooleanVar(value=False)

        # optional manual class filter (IDs or names, comma-separated). If blank → auto infer.
        self._class_filter_text = tk.StringVar(value="")  # e.g. "fire,smoke" or "0,1"

        # alert accumulator
        self._fps = 30
        self._alert_frames = 0

        super().__init__(master, task_key="Smoke/Fire", task_title="Smoke / Fire Detection", **kwargs)

    # ---------- UI ----------
    def _build_controls(self):
        super()._build_controls()

        nb = ttk.Notebook(self.toolbar)
        nb.pack(side=tk.LEFT, padx=8)

        # Run tab
        tab_run = ttk.Frame(nb)
        nb.add(tab_run, text="Run")
        r = ttk.Frame(tab_run, padding=(6, 6)); r.pack(fill=tk.X)

        ttk.Button(r, text="▶ Start", command=self._start, width=10).grid(row=0, column=0, padx=4, pady=2)
        ttk.Button(r, text="■ Stop",  command=self._stop,  width=10).grid(row=0, column=1, padx=4, pady=2)

        # Settings tab
        tab_set = ttk.Frame(nb)
        nb.add(tab_set, text="Settings")
        s = ttk.Frame(tab_set, padding=(6, 6)); s.pack(fill=tk.X)

        ttk.Label(s, text="Weights").grid(row=0, column=0, sticky="e")
        ttk.Entry(s, textvariable=self._weights, width=28).grid(row=0, column=1, sticky="w", padx=(4, 6))
        ttk.Button(s, text="Browse…", command=self._browse_weights).grid(row=0, column=2, padx=2)
        ttk.Button(s, text="Reload", command=self._reload_model).grid(row=0, column=3, padx=6)

        ttk.Label(s, text="Conf").grid(row=1, column=0, sticky="e", pady=(6,0))
        ttk.Spinbox(s, from_=0.05, to=0.95, increment=0.05, width=6, textvariable=self._conf)\
            .grid(row=1, column=1, sticky="w", padx=(4, 12), pady=(6,0))

        ttk.Label(s, text="IoU").grid(row=1, column=2, sticky="e", pady=(6,0))
        ttk.Spinbox(s, from_=0.1, to=0.9, increment=0.05, width=6, textvariable=self._iou)\
            .grid(row=1, column=3, sticky="w", padx=(4, 12), pady=(6,0))

        ttk.Label(s, text="Min area %").grid(row=2, column=0, sticky="e", pady=(6,0))
        ttk.Spinbox(s, from_=0.0005, to=0.05, increment=0.0005, width=8, textvariable=self._min_area_pct)\
            .grid(row=2, column=1, sticky="w", padx=(4, 12), pady=(6,0))

        ttk.Label(s, text="Class filter (IDs or names)").grid(row=2, column=2, sticky="e", pady=(6,0))
        ttk.Entry(s, textvariable=self._class_filter_text, width=10).grid(row=2, column=3, sticky="w", padx=(4, 12), pady=(6,0))

        ttk.Label(s, text="Count ≥").grid(row=3, column=0, sticky="e", pady=(6,0))
        ttk.Spinbox(s, from_=1, to=50, width=6, textvariable=self._count_thr)\
            .grid(row=3, column=1, sticky="w", padx=(4, 12), pady=(6,0))

        ttk.Label(s, text="Hold (s)").grid(row=3, column=2, sticky="e", pady=(6,0))
        ttk.Spinbox(s, from_=0.0, to=10.0, increment=0.5, width=6, textvariable=self._hold_seconds)\
            .grid(row=3, column=3, sticky="w", padx=(4, 12), pady=(6,0))

        # ROI tab
        tab_roi = ttk.Frame(nb)
        nb.add(tab_roi, text="ROI")
        rr = ttk.Frame(tab_roi, padding=(6,6)); rr.pack(fill=tk.X)
        ttk.Checkbutton(rr, text="Only detect inside ROI", variable=self._use_roi).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Button(rr, text="Draw ROI",  width=12, command=self.player.enable_roi_draw).grid(row=1, column=0, padx=6, pady=2)
        ttk.Button(rr, text="Clear ROI", width=12, command=self.player.clear_roi).grid(row=1, column=1, padx=6, pady=2)

    # ---------- lifecycle ----------
    def _start(self):
        if not self._ensure_model():
            return
        self._enabled = True
        self._alert_frames = 0
        self.player.on_frame = self._process
        dev = "MPS" if self._device == "mps" else "CPU"
        self.notify(f"Smoke/Fire started ({dev})")

    def _stop(self):
        self._enabled = False
        self.player.on_frame = None
        self.notify("Smoke/Fire stopped")

    # ---------- model mgmt ----------
    def _pick_device(self) -> str:
        try:
            import torch
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    def _ensure_model(self) -> bool:
        if self._model is None:
            try:
                self._model = YOLO(self._weights.get())
            except Exception as e:
                messagebox.showerror("Smoke/Fire", f"Failed to load YOLO weights:\n{e}")
                return False
        return True

    def _reload_model(self):
        self._model = None
        if self._ensure_model():
            self.notify("Weights loaded")

    def _browse_weights(self):
        p = filedialog.askopenfilename(
            title="Select YOLO weights",
            filetypes=[("YOLO weights", "*.pt *.onnx *.engine *.tflite"), ("All files", "*.*")]
        )
        if p:
            self._weights.set(p)
            self._reload_model()

    # ---------- helpers ----------
    def _names_dict(self) -> Dict[int, str]:
        if self._model is None:
            return {}
        names = self._model.names
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        # list
        return {i: str(n) for i, n in enumerate(names)}

    def _auto_target_ids(self) -> Set[int]:
        """
        Try to find classes with names like 'fire', 'smoke', 'flame'.
        If nothing found, return all model classes.
        """
        nd = self._names_dict()
        hits = {i for i, n in nd.items() if any(t in n.lower() for t in ("fire", "smoke", "flame"))}
        return hits if hits else set(nd.keys())

    def _parse_class_filter(self) -> Set[int]:
        txt = (self._class_filter_text.get() or "").strip()
        if not txt:
            return self._auto_target_ids()
        nd = self._names_dict()
        inv = {v.lower(): i for i, v in nd.items()}
        ids: Set[int] = set()
        for tok in txt.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if tok.isdigit():
                ids.add(int(tok))
            else:
                i = inv.get(tok.lower())
                if i is not None:
                    ids.add(i)
        return ids if ids else self._auto_target_ids()

    def _inside_roi(self, box) -> bool:
        if not self._use_roi.get():
            return True
        roi = self.player.get_roi()
        if roi is None:
            return True
        x0, y0, x1, y1 = roi
        bx1, by1, bx2, by2 = box
        cx, cy = (bx1 + bx2) // 2, (by1 + by2) // 2
        return (x0 <= cx <= x1) and (y0 <= cy <= y1)

    # ---------- main loop ----------
    def _process(self, frame: np.ndarray) -> np.ndarray:
        if not self._enabled or self._model is None:
            return frame

        H, W = frame.shape[:2]
        area = float(H * W)
        min_area = max(1.0, float(self._min_area_pct.get()) * area)

        # infer target ids
        target_ids = list(self._parse_class_filter())

        res = self._model.predict(
            frame,
            verbose=False,
            conf=float(self._conf.get()),
            iou=float(self._iou.get()),
            device=self._device,
            classes=target_ids if target_ids else None,
            max_det=300
        )[0]

        nd = self._names_dict()

        # gather accepted detections
        accepted = []
        if len(res):
            bxs = res.boxes.xyxy.cpu().numpy().astype(int)
            cls_arr = res.boxes.cls.cpu().numpy().astype(int)
            conf_arr = res.boxes.conf.cpu().numpy().astype(float)

            for (x1, y1, x2, y2), c_id, cf in zip(bxs, cls_arr, conf_arr):
                if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                    continue
                if not self._inside_roi((x1, y1, x2, y2)):
                    continue
                if ((x2 - x1) * (y2 - y1)) < min_area:
                    continue
                name = nd.get(c_id, f"cls{c_id}")
                accepted.append(((x1, y1, x2, y2), name, cf))

        # draw per-box + cluster
        for (x1, y1, x2, y2), name, cf in accepted:
            color = (0, 0, 255) if "fire" in name.lower() or "flame" in name.lower() else (128, 128, 128)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{name}:{cf:.2f}", (x1, max(18, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if accepted:
            xs1 = [b[0][0] for b in accepted]
            ys1 = [b[0][1] for b in accepted]
            xs2 = [b[0][2] for b in accepted]
            ys2 = [b[0][3] for b in accepted]
            gx1, gy1, gx2, gy2 = min(xs1), min(ys1), max(xs2), max(ys2)
            cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (0, 0, 200), 2)

        # alert gating
        cnt = len(accepted)
        need = int(self._count_thr.get())
        hold_frames = max(1, int(float(self._hold_seconds.get()) * self._fps))

        if cnt >= need:
            self._alert_frames += 1
        else:
            self._alert_frames = max(0, self._alert_frames - 1)  # a bit of hysteresis

        if self._alert_frames >= hold_frames:
            cv2.putText(frame, "ALERT: SMOKE / FIRE DETECTED!", (40, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        # outline ROI if used
        if self._use_roi.get():
            roi = self.player.get_roi()
            if roi is not None:
                x0, y0, x1, y1 = roi
                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 200, 200), 2)

        # status line
        cv2.putText(frame, f"Detections: {cnt}", (18, H - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        return frame
