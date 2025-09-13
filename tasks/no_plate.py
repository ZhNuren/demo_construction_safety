# tasks/no_plate.py
from __future__ import annotations
from typing import Optional, List, Tuple, Dict

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import cv2

from .base import TaskPage
from ultralytics import YOLO
from models.lpr import LicensePlateRecognizer


# COCO-like vehicle class ids for YOLOv8/11 default models
# person=0, bicycle=1, car=2, motorcycle=3, airplane=4, bus=5, train=6, truck=7
VEHICLE_MODES = {
    "cars":             [2],
    "cars+motorcycles": [2, 3],
    "cars+trucks+bus":  [2, 5, 7],
    "all vehicles":     [2, 3, 5, 7],
}


def iou_xyxy(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    b_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = a_area + b_area - inter
    return inter / max(1e-6, union)


def center_inside(box: Tuple[int,int,int,int], roi: Tuple[int,int,int,int]) -> bool:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    rx1, ry1, rx2, ry2 = roi
    return (rx1 <= cx <= rx2) and (ry1 <= cy <= ry2)


def box_area(b: Tuple[int,int,int,int]) -> int:
    return max(0, (b[2] - b[0])) * max(0, (b[3] - b[1]))


def inter_area(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> int:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    return iw * ih


class NoPlatePage(TaskPage):
    """
    Detect vehicles. For each vehicle, look for overlapping license plate.
    Alert if:
      - no plate overlaps the vehicle bbox, OR
      - a plate overlaps but OCR text is empty/unknown.

    Uses:
      - YOLO (vehicle detector)
      - LicensePlateRecognizer (plate detector + TrOCR)
    """

    def __init__(self, master, **kwargs):
        # runtime state (used by _build_controls)
        self._enabled = False
        self._fps = 30
        self._alert_frames = 0

        # models
        self._veh_model: Optional[YOLO] = None
        self._veh_device = self._pick_device()
        self._veh_weights = tk.StringVar(value="yolo11x.pt")
        self._veh_conf = tk.DoubleVar(value=0.35)
        self._veh_iou = tk.DoubleVar(value=0.45)
        self._veh_mode = tk.StringVar(value="cars+trucks+bus")

        self._lpr: Optional[LicensePlateRecognizer] = None
        self._lpr_weights = tk.StringVar(value="best_lpr.pt")  # your plate detector weights
        self._lpr_ocr_interval = tk.IntVar(value=3)

        # association / alert logic
        self._assoc_iou = tk.DoubleVar(value=0.0001)              # IOU to consider a plate belongs to a vehicle
        self._min_missing = tk.IntVar(value=1)                  # how many missing plates to alarm
        self._hold_seconds = tk.DoubleVar(value=3.0)            # persistence
        self._min_plate_area_pct = tk.DoubleVar(value=0.0)   # ignore tiny plate bboxes

        # Foreground overlap ignore (NEW)
        self._fg_ignore_overlap = tk.DoubleVar(value=0.0)      # ignore vehicles overlapping FG car by ≥ this ratio

        # ROI
        self._use_roi = tk.BooleanVar(value=False)

        super().__init__(master, task_key="No-Plate Vehicles", task_title="No-Plate Vehicle Detection", **kwargs)

    # ---------------- UI ----------------
    def _build_controls(self):
        super()._build_controls()

        nb = ttk.Notebook(self.toolbar)
        nb.pack(side=tk.LEFT, padx=8)

        # Run
        tab_run = ttk.Frame(nb); nb.add(tab_run, text="Run")
        rr = ttk.Frame(tab_run, padding=(6,6)); rr.pack(fill=tk.X)
        ttk.Button(rr, text="▶ Start", width=10, command=self._start).grid(row=0, column=0, padx=4, pady=2)
        ttk.Button(rr, text="■ Stop",  width=10, command=self._stop).grid(row=0, column=1, padx=4, pady=2)

        ttk.Label(rr, text="Alert if missing ≥").grid(row=1, column=0, sticky="e")
        ttk.Spinbox(rr, from_=1, to=50, width=6, textvariable=self._min_missing).grid(row=1, column=1, sticky="w", padx=(4,10))
        ttk.Label(rr, text="Hold (s)").grid(row=1, column=2, sticky="e")
        ttk.Spinbox(rr, from_=0.0, to=10.0, increment=0.5, width=6, textvariable=self._hold_seconds).grid(row=1, column=3, sticky="w", padx=(4,10))

        # Vehicle settings
        tab_v = ttk.Frame(nb); nb.add(tab_v, text="Vehicles")
        vs = ttk.Frame(tab_v, padding=(6,6)); vs.pack(fill=tk.X)

        ttk.Label(vs, text="Weights").grid(row=0, column=0, sticky="e")
        ttk.Entry(vs, textvariable=self._veh_weights, width=10).grid(row=0, column=1, sticky="w", padx=(4,6))
        ttk.Button(vs, text="Browse…", command=self._browse_veh).grid(row=0, column=2, padx=2)
        ttk.Button(vs, text="Reload", command=self._reload_veh).grid(row=0, column=3, padx=6)

        ttk.Label(vs, text="Classes").grid(row=1, column=0, sticky="e", pady=(6,0))
        ttk.Combobox(vs, state="readonly", width=18,
                     values=list(VEHICLE_MODES.keys()), textvariable=self._veh_mode).grid(row=1, column=1, sticky="w", padx=(4,10), pady=(6,0))

        ttk.Label(vs, text="Conf").grid(row=1, column=2, sticky="e", pady=(6,0))
        ttk.Spinbox(vs, from_=0.05, to=0.95, increment=0.05, width=6, textvariable=self._veh_conf).grid(row=1, column=3, sticky="w", padx=(4,10), pady=(6,0))

        ttk.Label(vs, text="IoU").grid(row=1, column=4, sticky="e", pady=(6,0))
        ttk.Spinbox(vs, from_=0.10, to=0.90, increment=0.05, width=6, textvariable=self._veh_iou).grid(row=1, column=5, sticky="w", padx=(4,10), pady=(6,0))

        # NEW: Foreground overlap ignore
        ttk.Label(vs, text="Ignore if FG overlap ≥").grid(row=2, column=0, sticky="e", pady=(6,0))
        ttk.Spinbox(vs, from_=0.00, to=0.90, increment=0.01, width=6,
                    textvariable=self._fg_ignore_overlap).grid(row=2, column=1, sticky="w", padx=(4,10), pady=(6,0))

        # Plate/OCR settings
        tab_p = ttk.Frame(nb); nb.add(tab_p, text="Plates/OCR")
        ps = ttk.Frame(tab_p, padding=(6,6)); ps.pack(fill=tk.X)

        ttk.Label(ps, text="Plate weights").grid(row=0, column=0, sticky="e")
        ttk.Entry(ps, textvariable=self._lpr_weights, width=10).grid(row=0, column=1, sticky="w", padx=(4,6))
        ttk.Button(ps, text="Browse…", command=self._browse_lpr).grid(row=0, column=2, padx=2)
        ttk.Button(ps, text="Reload", command=self._reload_lpr).grid(row=0, column=3, padx=6)

        ttk.Label(ps, text="OCR interval (frames)").grid(row=1, column=0, sticky="e", pady=(6,0))
        ttk.Spinbox(ps, from_=1, to=10, width=6, textvariable=self._lpr_ocr_interval).grid(row=1, column=1, sticky="w", padx=(4,10), pady=(6,0))

        ttk.Label(ps, text="Assoc IoU ≥").grid(row=1, column=2, sticky="e", pady=(6,0))
        ttk.Spinbox(ps, from_=0.0, to=0.9, increment=0.01, width=6, textvariable=self._assoc_iou).grid(row=1, column=3, sticky="w", padx=(4,10), pady=(6,0))

        ttk.Label(ps, text="Min plate area %").grid(row=1, column=4, sticky="e", pady=(6,0))
        ttk.Spinbox(ps, from_=0.0001, to=0.01, increment=0.0001, width=6, textvariable=self._min_plate_area_pct).grid(row=1, column=5, sticky="w", padx=(4,10), pady=(6,0))

        # ROI tab
        tab_roi = ttk.Frame(nb); nb.add(tab_roi, text="ROI")
        rs = ttk.Frame(tab_roi, padding=(6,6)); rs.pack(fill=tk.X)
        ttk.Checkbutton(rs, text="Only evaluate vehicles inside ROI", variable=self._use_roi).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Button(rs, text="Draw ROI",  width=12, command=self.player.enable_roi_draw).grid(row=1, column=0, padx=6, pady=2)
        ttk.Button(rs, text="Clear ROI", width=12, command=self.player.clear_roi).grid(row=1, column=1, padx=6, pady=2)

    # ---------------- lifecycle ----------------
    def _start(self):
        if not self._ensure_models():
            return
        self._enabled = True
        self._alert_frames = 0
        self.player.on_frame = self._process
        dev = "MPS" if self._veh_device == "mps" else "CPU"
        self.notify(f"No-Plate detector started ({dev})")

    def _stop(self):
        self._enabled = False
        self.player.on_frame = None
        self.notify("No-Plate detector stopped")

    # ---------------- model mgmt ----------------
    def _pick_device(self) -> str:
        try:
            import torch
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    def _ensure_models(self) -> bool:
        if self._veh_model is None:
            try:
                self._veh_model = YOLO(self._veh_weights.get())
            except Exception as e:
                messagebox.showerror("Vehicles model", f"Failed to load YOLO weights:\n{e}")
                return False
        if self._lpr is None:
            try:
                self._lpr = LicensePlateRecognizer(
                    det_model=self._lpr_weights.get(),
                    conf=0.40,
                    yolo_device=self._veh_device,
                    trocr_device="mps" if self._veh_device == "mps" else "cpu",
                    ocr_interval=int(self._lpr_ocr_interval.get()),
                    ocr_backend = 'easyocr',
                )
            except Exception as e:
                messagebox.showerror("LPR", f"Failed to initialize LPR:\n{e}")
                return False
        return True

    def _reload_veh(self):
        self._veh_model = None
        self._ensure_models()
        self.notify("Vehicle weights loaded")

    def _reload_lpr(self):
        self._lpr = None
        self._ensure_models()
        self.notify("Plate/OCR reloaded")

    def _browse_veh(self):
        p = filedialog.askopenfilename(title="Select vehicle YOLO weights",
                                       filetypes=[("YOLO weights","*.pt *.onnx *.engine *.tflite"),("All files","*.*")])
        if p:
            self._veh_weights.set(p)
            self._reload_veh()

    def _browse_lpr(self):
        p = filedialog.askopenfilename(title="Select plate YOLO weights",
                                       filetypes=[("YOLO weights","*.pt *.onnx *.engine *.tflite"),("All files","*.*")])
        if p:
            self._lpr_weights.set(p)
            self._reload_lpr()

    # ---------------- helpers ----------------
    def _vehicle_class_ids(self) -> List[int]:
        return VEHICLE_MODES.get(self._veh_mode.get(), [2,5,7])

    # ---------------- main ----------------
    def _process(self, frame: np.ndarray) -> np.ndarray:
        if not self._enabled or self._veh_model is None or self._lpr is None:
            return frame

        H, W = frame.shape[:2]
        roi = self.player.get_roi() if self._use_roi.get() else None

        # 1) VEHICLE DETECTION
        vres = self._veh_model.predict(
            frame, verbose=False,
            conf=float(self._veh_conf.get()),
            iou=float(self._veh_iou.get()),
            device=self._veh_device,
            classes=self._vehicle_class_ids(),
            max_det=200
        )[0]

        veh_boxes: List[Tuple[int,int,int,int]] = []
        veh_cls:   List[int] = []
        veh_conf:  List[float] = []

        if len(vres):
            b = vres.boxes.xyxy.cpu().numpy().astype(int)
            c = vres.boxes.cls.cpu().numpy().astype(int)
            s = vres.boxes.conf.cpu().numpy().astype(float)
            for (x1,y1,x2,y2), ci, sc in zip(b, c, s):
                if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                    continue
                if roi is not None and not center_inside((x1,y1,x2,y2), roi):
                    continue
                veh_boxes.append((x1,y1,x2,y2))
                veh_cls.append(ci)
                veh_conf.append(sc)

        # ---- Foreground car & overlap filtering (NEW) ----
        ignored_boxes: List[Tuple[int,int,int,int]] = []
        fg_thr = float(self._fg_ignore_overlap.get())

        if veh_boxes:
            # Prefer the largest 'car' (COCO class 2); otherwise largest vehicle
            car_idxs = [i for i, ci in enumerate(veh_cls) if ci == 2]
            candidates = car_idxs if car_idxs else list(range(len(veh_boxes)))
            fg_idx = max(candidates, key=lambda i: box_area(veh_boxes[i]))

            kept_idx = []
            fg_box = veh_boxes[fg_idx]
            for i, vb in enumerate(veh_boxes):
                if i == fg_idx:
                    kept_idx.append(i)
                    continue
                ov = inter_area(vb, fg_box)
                ratio = ov / max(1, box_area(vb))   # overlap wrt the other vehicle's area
                if ratio >= fg_thr:
                    ignored_boxes.append(vb)
                else:
                    kept_idx.append(i)

            # keep only not-ignored vehicles (incl. FG)
            veh_boxes = [veh_boxes[i] for i in kept_idx]
            veh_cls   = [veh_cls[i]   for i in kept_idx]
            veh_conf  = [veh_conf[i]  for i in kept_idx]

        # 2) PLATE DETECTION + OCR
        plate_results = self._lpr.detect_and_identify(frame)
        # Filter out tiny plate boxes
        min_plate_area = float(self._min_plate_area_pct.get()) * (H * W)
        plates: List[Tuple[Tuple[int,int,int,int], str]] = []
        for r in plate_results:
            (px1, py1, px2, py2) = r["bbox"]
            area = max(0, (px2 - px1)) * max(0, (py2 - py1))
            if area < min_plate_area:
                continue
            text = (r.get("plate_raw") or "").strip()
            plates.append(((int(px1),int(py1),int(px2),int(py2)), text))

        # 3) ASSOCIATE plates -> vehicles
        assoc_thr = float(self._assoc_iou.get())
        missing_indices: List[int] = []
        ok_indices: List[int] = []
        best_plate_for_veh: Dict[int, Tuple[Tuple[int,int,int,int], str, float]] = {}

        for i, vbox in enumerate(veh_boxes):
            best_iou, best = 0.0, None
            for pbox, ptxt in plates:
                iou = iou_xyxy(vbox, pbox)
                if iou > best_iou:
                    best_iou, best = iou, (pbox, ptxt, iou)
            if best is None or best_iou < assoc_thr:
                # no overlapping plate
                missing_indices.append(i)
            else:
                # has plate, but is OCR empty?
                _, ptxt, _ = best
                if ptxt == "" or ptxt == "???":
                    missing_indices.append(i)
                    best_plate_for_veh[i] = best  # still draw plate box for debug
                else:
                    ok_indices.append(i)
                    best_plate_for_veh[i] = best

        # 4) DRAW
        # (Optional) draw ignored vehicles so you can see what's skipped
        for (ix1, iy1, ix2, iy2) in ignored_boxes:
            cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), (150,150,150), 1)
            cv2.putText(frame, "IGN", (ix1, max(18, iy1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,180,180), 1)

        # vehicles OK (green)
        for i in ok_indices:
            x1,y1,x2,y2 = veh_boxes[i]
            cv2.rectangle(frame, (x1,y1), (x2,y2), (60,220,60), 2)
            label = "PLATE OK"
            # show matched plate text
            p = best_plate_for_veh.get(i)
            if p:
                _, ptxt, _ = p
                if ptxt:
                    label = f"{label}"
            cv2.putText(frame, label, (x1, max(18, y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

        # vehicles missing (red) + cluster box
        cluster_boxes = []
        for i in missing_indices:
            x1,y1,x2,y2 = veh_boxes[i]
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
            reason = "NO PLATE"
            p = best_plate_for_veh.get(i)
            if p and (p[1] == "" or p[1] == "???"):
                reason = "PLATE OCR EMPTY"
                # draw plate box to help debugging
                px1,py1,px2,py2 = p[0]
                cv2.rectangle(frame, (px1,py1), (px2,py2), (0,0,180), 1)
            cv2.putText(frame, reason, (x1, max(18, y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
            cluster_boxes.append((x1,y1,x2,y2))

        # merged cluster for all missing vehicles
        if cluster_boxes:
            xs1 = [b[0] for b in cluster_boxes]; ys1 = [b[1] for b in cluster_boxes]
            xs2 = [b[2] for b in cluster_boxes]; ys2 = [b[3] for b in cluster_boxes]
            gx1, gy1, gx2, gy2 = min(xs1), min(ys1), max(xs2), max(ys2)
            cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (0,0,200), 2)

        # 5) ALERT gating
        missing_cnt = len(missing_indices)
        need = int(self._min_missing.get())
        hold_frames = max(1, int(float(self._hold_seconds.get()) * self._fps))

        if missing_cnt >= need:
            self._alert_frames += 1
        else:
            self._alert_frames = max(0, self._alert_frames - 1)

        if self._alert_frames >= hold_frames:
            cv2.putText(frame, "ALERT: VEHICLE WITHOUT LICENSE PLATE!", (40, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)

        # draw ROI if used
        if self._use_roi.get():
            r = self.player.get_roi()
            if r is not None:
                x0,y0,x1,y1 = r
                cv2.rectangle(frame, (x0,y0), (x1,y1), (0,200,200), 2)

        # status line
        cv2.putText(frame, f"Vehicles: {len(veh_boxes)}  Missing: {missing_cnt}  Ignored: {len(ignored_boxes)}",
                    (18, H - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        return frame
