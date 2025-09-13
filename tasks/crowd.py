# tasks/crowd.py
from __future__ import annotations
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import cv2
from typing import Optional, List, Tuple

from .base import TaskPage
from models.crowd import CrowdCounter


class CrowdPage(TaskPage):
    def __init__(self, master, **kwargs):
        self._counter: Optional[CrowdCounter] = None
        self._enabled = False
        self._use_roi = False   # count inside ROI vs whole frame
        self._alert_count_thr = 10
        self._alert_time_thr = 3.0  # seconds
        self._over_count_frames = 0
        self._fps = 30  # assume ~30 fps
        self._roi_scale = 2.0
        self._preprocess = tk.BooleanVar(value=True)
        self._prox_px_default = 80  # proximity threshold in pixels (center-to-center)

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

        # Row 2 → proximity px for clustering
        ttk.Label(frm_r1, text="Proximity (px)").grid(row=2, column=0, sticky="e", padx=(4, 2), pady=2)
        self.prox_px_var = tk.IntVar(value=self._prox_px_default)
        ttk.Spinbox(frm_r1, from_=10, to=500, increment=5, width=6,
                    textvariable=self.prox_px_var).grid(row=2, column=1, sticky="w", padx=(0, 8), pady=2)

        # --- ROI tab ---
        tab_roi = ttk.Frame(nb)
        nb.add(tab_roi, text="ROI")

        frm_roi = ttk.Frame(tab_roi, padding=(6, 6))
        frm_roi.pack(fill=tk.X)

        ttk.Label(frm_roi, text="Region of Interest").grid(row=0, column=0, columnspan=2, pady=(0, 6))

        ttk.Button(frm_roi, text="Draw ROI", width=12,
                   command=self.player.enable_roi_draw).grid(row=1, column=0, padx=6, pady=2)
        ttk.Button(frm_roi, text="Clear ROI", width=12,
                   command=self.player.clear_roi).grid(row=1, column=1, padx=6, pady=2)

        ttk.Label(frm_roi, text="Scale ×").grid(row=2, column=0, padx=(6, 2), pady=4)
        self.scale_var = tk.DoubleVar(value=self._roi_scale)
        ttk.Spinbox(frm_roi, from_=1.0, to=4.0, increment=0.5, width=6,
                    textvariable=self.scale_var, command=self._update_scale).grid(row=2, column=1, pady=4)

        ttk.Checkbutton(frm_roi, text="Preprocess ROI", variable=self._preprocess)\
            .grid(row=3, column=0, columnspan=2, pady=4)

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

    # ---------- clustering helpers ----------
    @staticmethod
    def _box_centers(boxes: List[Tuple[int,int,int,int]]) -> np.ndarray:
        if not boxes:
            return np.zeros((0, 2), dtype=np.float32)
        centers = []
        for (x1, y1, x2, y2) in boxes:
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            centers.append((cx, cy))
        return np.array(centers, dtype=np.float32)

    @staticmethod
    def _cluster_by_distance(centers: np.ndarray, prox_px: float) -> List[List[int]]:
        """
        Simple BFS clustering: group points whose pairwise distance <= prox_px
        (single-link). Returns list of clusters as lists of indices.
        """
        n = len(centers)
        if n == 0:
            return []
        visited = np.zeros(n, dtype=bool)
        clusters: List[List[int]] = []
        prox2 = float(prox_px) * float(prox_px)

        for i in range(n):
            if visited[i]:
                continue
            # BFS
            q = [i]
            visited[i] = True
            cluster = [i]
            while q:
                u = q.pop()
                # compare u to all not-visited points
                diff = centers[~visited] - centers[u]
                dist2 = (diff[:, 0] ** 2 + diff[:, 1] ** 2)
                within = np.where(dist2 <= prox2)[0]
                # map back to absolute indices
                cand_idx = np.flatnonzero(~visited)
                to_add = cand_idx[within]
                for j in to_add:
                    visited[j] = True
                    q.append(j)
                    cluster.append(j)
            clusters.append(cluster)
        return clusters

    @staticmethod
    def _merge_cluster_box(indices: List[int], boxes: List[Tuple[int,int,int,int]]) -> Tuple[int,int,int,int]:
        xs1, ys1, xs2, ys2 = [], [], [], []
        for k in indices:
            x1, y1, x2, y2 = boxes[k]
            xs1.append(x1); ys1.append(y1); xs2.append(x2); ys2.append(y2)
        return (int(min(xs1)), int(min(ys1)), int(max(xs2)), int(max(ys2)))

    # ---------- main per-frame ----------
    def _process(self, frame: np.ndarray) -> np.ndarray:
        if not self._enabled:
            return frame
        counter = self._ensure_counter()
        if counter is None:
            return frame

        roi = self.player.get_roi()
        boxes: List[Tuple[int,int,int,int]] = []
        num_people = 0

        if self._use_roi and roi is not None:
            x0, y0, x1, y1 = roi
            roi_frame = frame[y0:y1, x0:x1]

            # optional preprocessing to help dense scenes
            if self._preprocess.get():
                roi_frame = cv2.GaussianBlur(roi_frame, (3, 3), 0)
                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                roi_frame = cv2.filter2D(roi_frame, -1, kernel)
                roi_frame = cv2.convertScaleAbs(roi_frame, alpha=1.3, beta=15)

            # scale / zoom
            scale = float(self._roi_scale)
            rh, rw = roi_frame.shape[:2]
            zoomed = cv2.resize(roi_frame, (int(rw * scale), int(rh * scale)), interpolation=cv2.INTER_CUBIC)

            res = counter.detector.predict(zoomed, verbose=False, conf=0.4, device="mps", classes=[0])[0]
            det_boxes = res.boxes.xyxy.cpu().numpy().astype(int).tolist()

            # map boxes back to full-frame coordinates
            for (bx1, by1, bx2, by2) in det_boxes:
                bx1 = int(bx1 / scale) + x0
                bx2 = int(bx2 / scale) + x0
                by1 = int(by1 / scale) + y0
                by2 = int(by2 / scale) + y0
                boxes.append((bx1, by1, bx2, by2))
                # draw person box (red in ROI mode)
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 255), 2)

            num_people = len(boxes)
            # draw ROI outline
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 200, 200), 2)

        else:
            # full-frame detection
            res = counter.detector.predict(frame, verbose=False, conf=0.4, device="mps", classes=[0])[0]
            boxes = res.boxes.xyxy.cpu().numpy().astype(int).tolist()
            num_people = len(boxes)
            # draw person boxes (green)
            for (bx1, by1, bx2, by2) in boxes:
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)

        # ----- cluster logic: crowd = cluster size >= Count ≥ -----
        prox_px = int(self.prox_px_var.get())
        centers = self._box_centers(boxes)
        clusters = self._cluster_by_distance(centers, prox_px)

        # draw merged boxes for clusters meeting the threshold
        count_thr = int(self.count_thr_var.get())
        qualifying = [c for c in clusters if len(c) >= count_thr]

        for c in qualifying:
            gx1, gy1, gx2, gy2 = self._merge_cluster_box(c, boxes)
            cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (0, 0, 255), 3)
            cv2.putText(frame, f"CROWD: {len(c)}", (gx1, max(20, gy1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # persistence: if any cluster meets the size thr, increment; else reset
        time_thr_s = float(self.time_thr_var.get())
        frame_thr = max(1, int(time_thr_s * self._fps))
        if any(len(c) >= count_thr for c in clusters):
            self._over_count_frames += 1
        else:
            self._over_count_frames = 0

        # alert vs info text
        if self._over_count_frames >= frame_thr:
            cv2.putText(frame, "ALERT: CROWD DETECTED!", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        else:
            cv2.putText(frame, f"People: {num_people}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        return frame
