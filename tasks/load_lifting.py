from __future__ import annotations

from typing import Optional, Dict, Deque, List, Tuple
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
import numpy as np
from collections import deque

from .base import TaskPage
from models.load_lifting import CraneDetector
from tracking.simple_tracker import SimpleTracker


class CraneTrackingPage(TaskPage):
    """
    Crane model tracking with:
      • filtering to classes: 'crane' and 'hanging head'
      • Lifting mode  -> detect Δy movement on 'hanging head'
      • Rotation mode -> detect Δx movement on 'hanging head'
      • On-video banner alerts and per-box tags
      • Apply buttons for Trail and Motion params
    """
    TARGET_CLASS_NAMES = {"Crane", "Hanging head"}  # names from your model (case-insensitive matching is applied)

    def __init__(self, master, **kwargs):
        # --- runtime state ---
        self._tracker_enabled = False
        self._tracker = SimpleTracker(max_lost=20, iou_thr=0.35, trail=120)
        self._detector: Optional[CraneDetector] = None
        self._allowed_ids: set[int] = set()

        # --- UI state ---
        self.trail_len = tk.IntVar(value=120)

        self.motion_thr = tk.IntVar(value=20)       # px threshold for Δx/Δy
        self.motion_window = tk.IntVar(value=5)     # compare vs N frames ago
        self.mode = tk.StringVar(value="Lifting")   # "Lifting" or "Rotation"

        # Applied params (take effect after pressing Apply)
        self._applied_thr = int(self.motion_thr.get())
        self._applied_window = int(self.motion_window.get())

        # HUD banner state
        self._hud_alert_text = ""
        self._hud_alert_frames = 0

        # Per-track motion state:
        # tid -> {"xbuf": deque, "ybuf": deque, "moved": bool, "cool": int, "count": int}
        self._motion_state: Dict[int, Dict[str, object]] = {}

        super().__init__(master, task_key="Crane Tracking", task_title="Crane Tracking", **kwargs)

    # ---------- UI ----------
    def _build_controls(self):
        super()._build_controls()
        ttk.Separator(self.toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=12)

        section = ttk.Frame(self.toolbar)
        section.pack(side=tk.LEFT)

        ttk.Button(section, text="Start", command=self._start_tracking).grid(row=0, column=0)
        ttk.Button(section, text="Stop", command=self._stop_tracking).grid(row=0, column=1, padx=6)
        ttk.Button(section, text="Clear trails", command=self._clear_trails).grid(row=0, column=2)

        # Mode
        ttk.Label(section, text="Mode").grid(row=1, column=0, pady=(6, 0))
        ttk.Combobox(section, state="readonly", width=10,
                     values=["Lifting", "Rotation"],
                     textvariable=self.mode).grid(row=1, column=1, sticky="w", pady=(6, 0))

        # Trail + Apply
        ttk.Label(section, text="Trail").grid(row=2, column=0, pady=(6, 0))
        ttk.Spinbox(section, from_=10, to=2000, width=6, textvariable=self.trail_len)\
            .grid(row=2, column=1, sticky="w", pady=(6, 0))
        ttk.Button(section, text="Apply", command=self._apply_trail_len)\
            .grid(row=2, column=2, padx=(6, 0), pady=(6, 0))

        # Motion params (Px thr, Window) + Apply
        ttk.Label(section, text="Px thr").grid(row=3, column=0, pady=(6, 0))
        ttk.Spinbox(section, from_=1, to=400, width=6, textvariable=self.motion_thr)\
            .grid(row=3, column=1, sticky="w", pady=(6, 0))

        ttk.Label(section, text="Window").grid(row=4, column=0, pady=(6, 0))
        ttk.Spinbox(section, from_=2, to=60, width=6, textvariable=self.motion_window)\
            .grid(row=4, column=1, sticky="w", pady=(6, 0))
        ttk.Button(section, text="Apply", command=self._apply_params)\
            .grid(row=4, column=2, padx=(6, 0), pady=(6, 0))

    # ---------- Helpers ----------
    @staticmethod
    def _norm_name(s: str) -> str:
        return " ".join(s.lower().replace("_", " ").replace("-", " ").split())

    def _resolve_allowed_ids(self, class_map: dict[int, str]) -> set[int]:
        wanted = {self._norm_name(n) for n in self.TARGET_CLASS_NAMES}
        return {cid for cid, name in class_map.items() if self._norm_name(name) in wanted}

    def _iou(self, a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        iw = max(0, min(ax2, bx2) - max(ax1, bx1))
        ih = max(0, min(ay2, by2) - max(ay1, by1))
        inter = iw * ih
        if inter <= 0:
            return 0.0
        a_area = max(0, (ax2 - ax1)) * max(0, (ay2 - ay1))
        b_area = max(0, (bx2 - bx1)) * max(0, (by2 - by1))
        union = a_area + b_area - inter + 1e-6
        return float(inter / union)

    def _attach_labels_by_iou(
        self,
        det_bboxes: List[Tuple[int, int, int, int]],
        det_labels: List[str],
        tracks: Dict[int, dict],
        min_iou: float = 0.1,
    ) -> Dict[int, str]:
        """Assign best-matching detection label to each track using IoU."""
        out: Dict[int, str] = {}
        for tid, tr in tracks.items():
            tb = tr['bbox']
            best_iou, best_lab = 0.0, None
            for db, lb in zip(det_bboxes, det_labels):
                i = self._iou(tb, db)
                if i > best_iou:
                    best_iou, best_lab = i, lb
            if best_iou >= min_iou and best_lab is not None:
                out[tid] = str(best_lab)
            else:
                out[tid] = str(tr.get('label', ""))  # keep prior label if no decent match
        return out

    # ---------- Detector ----------
    def _ensure_detector(self):
        if self._detector is None:
            try:
                self._detector = CraneDetector(model_name="crane.pt", conf=0.35, imgsz=640, device="mps")
                self._allowed_ids = self._resolve_allowed_ids(self._detector.get_class_map())
                if not self._allowed_ids:
                    self.notify("Warning: target classes not found; will filter by names per frame.")
                else:
                    found = [self._detector.get_class_map()[i] for i in sorted(self._allowed_ids)]
                    self.notify("Filtering classes: " + ", ".join(found))
            except Exception as e:
                messagebox.showerror("Detector init failed", str(e))
                return None
        return self._detector

    # ---------- Controls ----------
    def _start_tracking(self):
        if self._ensure_detector() is None:
            return
        self._tracker_enabled = True
        self.player.on_frame = self._process_tracking_frame
        self.notify(f"Crane tracking started in {self.mode.get()} mode")

    def _stop_tracking(self):
        self._tracker_enabled = False
        self.player.on_frame = None
        self._hud_alert_text = ""
        self._hud_alert_frames = 0
        self._motion_state.clear()
        self.notify("Crane tracking stopped")

    def _clear_trails(self):
        for tr in self._tracker.tracks.values():
            tr['trail'].clear()
        self._motion_state.clear()
        self._hud_alert_text = ""
        self._hud_alert_frames = 0
        self.notify("Trails & motion state cleared")

    # ---------- Apply handlers ----------
    def _apply_trail_len(self):
        n = max(5, min(2000, int(self.trail_len.get())))
        self._tracker.trail = n
        from collections import deque as _dq
        for tr in self._tracker.tracks.values():
            tr['trail'] = _dq(tr['trail'], maxlen=n)
        self.notify(f"Trail length set to {n}")

    def _apply_params(self):
        """Commit Px thr and Window, rebuild buffers if window changed."""
        thr = max(1, int(self.motion_thr.get()))
        win = max(2, int(self.motion_window.get()))
        changed = []

        if thr != self._applied_thr:
            self._applied_thr = thr
            changed.append(f"thr={thr}")

        if win != self._applied_window:
            self._applied_window = win
            for st in self._motion_state.values():
                st["xbuf"] = deque(st.get("xbuf", []), maxlen=win)
                st["ybuf"] = deque(st.get("ybuf", []), maxlen=win)
            changed.append(f"window={win}")

        self.notify("Applied " + (", ".join(changed) if changed else "no changes"))

    # ---------- HUD ----------
    def _on_motion_alert(self, tid: int):
        fps = getattr(getattr(self, "player", None), "fps", 24) or 24
        if self.mode.get() == "Lifting":
            self._hud_alert_text = f"LIFT: Hanging head ID {tid}"
        else:
            self._hud_alert_text = f"ROTATE: Hanging head ID {tid}"
        self._hud_alert_frames = int(fps * 2.5)

    def _draw_video_alert(self, frame):
        if self._hud_alert_frames <= 0 or not self._hud_alert_text:
            return frame
        h, w = frame.shape[:2]
        banner_h = max(36, int(0.08 * h))
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, dst=frame)

        text = self._hud_alert_text
        scale = max(0.8, min(2.2, w / 900.0))
        thick = max(2, int(scale * 2))
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        x = max(8, (w - tw) // 2)
        y = max(th + 8, (banner_h + th) // 2)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (0, 0, 0), thick + 2, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (255, 255, 255), thick, cv2.LINE_AA)

        self._hud_alert_frames -= 1
        if self._hud_alert_frames <= 0:
            self._hud_alert_text = ""
        return frame

    # ---------- Motion detection (uses *applied* params) ----------
    def _update_motion_state(self, tid: int, cx: int, cy: int, is_head: bool):
        """
        Detect motion depending on mode:
          - Lifting: use Δy
          - Rotation: use Δx
        Returns (is_new_event, count)
        """
        if not is_head:
            st = self._motion_state.get(tid)
            return False, (st or {}).get("count", 0)

        st = self._motion_state.get(tid)
        win = int(self._applied_window)
        thr = int(self._applied_thr)

        if st is None:
            st = {
                "xbuf": deque(maxlen=win),
                "ybuf": deque(maxlen=win),
                "moved": False,
                "cool": 0,
                "count": 0,
            }
            self._motion_state[tid] = st
        else:
            # Rebuild buffers if window size changed since last Apply
            if st["xbuf"].maxlen != win:
                st["xbuf"] = deque(st["xbuf"], maxlen=win)
                st["ybuf"] = deque(st["ybuf"], maxlen=win)

        xbuf: Deque[int] = st["xbuf"]  # type: ignore
        ybuf: Deque[int] = st["ybuf"]  # type: ignore
        moved: bool = bool(st["moved"])  # type: ignore
        cool: int = int(st["cool"])      # type: ignore
        count: int = int(st["count"])    # type: ignore

        xbuf.append(cx)
        ybuf.append(cy)

        if cool > 0:
            cool -= 1

        if len(xbuf) >= win and len(ybuf) >= win:
            dx = xbuf[-1] - xbuf[-win]
            dy = ybuf[-1] - ybuf[-win]

            if self.mode.get() == "Lifting":
                condition = abs(dy) >= thr
            else:  # Rotation
                condition = abs(dx) >= thr

            new_evt = condition and (not moved) and (cool == 0)
            if new_evt:
                count += 1
                moved = True
                cool = 10
                st.update({
                    "xbuf": xbuf, "ybuf": ybuf,
                    "moved": moved, "cool": cool, "count": count
                })
                return True, count

            # Hysteresis to re-arm after motion settles
            recent_dx = abs(xbuf[-1] - xbuf[-2]) if len(xbuf) >= 2 else 0
            recent_dy = abs(ybuf[-1] - ybuf[-2]) if len(ybuf) >= 2 else 0
            if moved and max(recent_dx, recent_dy) <= max(1, thr // 10):
                moved = False

        st.update({
            "xbuf": xbuf, "ybuf": ybuf,
            "moved": moved, "cool": cool, "count": count
        })
        return False, count

    # ---------- Frame processing ----------
    def _process_tracking_frame(self, frame):
        if not self._tracker_enabled:
            return frame

        det = self._ensure_detector()
        if det is None:
            return frame

        xyxy, scores, clss, labels = det.detect_xyxy(frame)

        # Filter to crane + hanging head; if nothing matches, keep all so crane still shows
        if self._allowed_ids:
            idxs = [i for i, c in enumerate(clss) if c in self._allowed_ids]
        else:
            tgt = {self._norm_name(n) for n in self.TARGET_CLASS_NAMES}
            idxs = [i for i, lb in enumerate(labels) if self._norm_name(lb) in tgt]
        if not idxs and xyxy:
            idxs = list(range(len(xyxy)))  # fallback: keep everything

        det_bboxes = [tuple(map(int, xyxy[i])) for i in idxs]
        det_scores = [float(scores[i]) for i in idxs]
        det_labels = [str(labels[i]) for i in idxs]

        if not det_bboxes:
            return self._draw_video_alert(frame)

        # Update tracker
        tracks = self._tracker.update(det_bboxes, det_scores)

        # Attach labels to tracks via IoU
        tid2label = self._attach_labels_by_iou(det_bboxes, det_labels, tracks, min_iou=0.1)

        # Update states and trigger alerts
        for tid, tr in tracks.items():
            tr['label'] = tid2label.get(tid, tr.get('label', ''))
            x1, y1, x2, y2 = tr['bbox']
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            tr['center'] = (cx, cy)

            name_norm = self._norm_name(tr['label'])
            is_head = ("head" in name_norm) or (name_norm == "hanging head")

            is_new_evt, count = self._update_motion_state(tid, cx, cy, is_head)
            tr['is_motion_now'] = is_new_evt
            tr['motion_count'] = count

            if is_head and is_new_evt:
                self._on_motion_alert(tid)

        # --- Drawing ---
        crane_color = (0, 180, 255)   # orange
        head_color  = (50, 205, 50)   # green
        text_color  = (255, 255, 255)

        for tid, tr in tracks.items():
            x1, y1, x2, y2 = tr['bbox']
            name_norm = self._norm_name(tr.get('label', ''))
            is_head = ("head" in name_norm) or (name_norm == "hanging head")
            color = head_color if is_head else crane_color

            # box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # label
            label_text = f"ID {tid} {tr.get('label','')}".strip()
            box_h = max(1, y2 - y1)
            scale = float(np.clip(box_h / 80.0, 0.9, 2.8))
            thick = max(2, int(scale * 2))
            (tw, th), base = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
            tl = (x1, max(0, y1 - th - base - 6))
            br = (x1 + tw + 6, y1)
            cv2.rectangle(frame, tl, br, color, -1)
            cv2.putText(frame, label_text, (x1 + 3, br[1] - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, text_color, thick, cv2.LINE_AA)

            # per-object tag
            if is_head and tr.get('is_motion_now', False):
                tag = "LIFT!" if self.mode.get() == "Lifting" else "ROTATE!"
                (lw, lh), lbase = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, scale * 1.2, thick + 1)
                tl2 = (x1, max(0, tl[1] - lh - lbase - 4))
                br2 = (x1 + lw + 8, tl[1] - 2)
                cv2.rectangle(frame, tl2, br2, (0, 0, 255), -1)
                cv2.putText(frame, tag, (tl2[0] + 4, br2[1] - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, scale * 1.2, (255, 255, 255), thick + 1, cv2.LINE_AA)

            # trail
            pts = list(tr['trail'])
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i - 1], pts[i], color, 2)

        return self._draw_video_alert(frame)

