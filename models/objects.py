import time
import math
from typing import Dict, Tuple, Optional

import cv2
import numpy as np
from ultralytics import YOLO


class ObjectPresenceDetector:
    """
    Tracks whether at least one detected object has persisted longer than a time threshold.
    - Nearest-center ID assignment across frames.
    - Draws per-box rectangles:
        * green while elapsed < threshold (optional small timer)
        * red when elapsed ≥ threshold, with 'stay: Ns' label that updates every second
    - DOES NOT draw any global ALERT banner; the UI layer should handle that.
    """

    def __init__(
        self,
        det_model: str = "best.pt",
        conf: float = 0.4,
        device: str = "cpu",
        persistence_sec: float = 5.0,
        dist_thr: float = 100.0,
        show_timers: bool = True,  # show pre-threshold small timer
    ):
        """
        det_model: YOLO weights file (e.g. best.pt)
        conf: confidence threshold
        device: "cpu", "cuda", "mps"
        persistence_sec: how many seconds before alert
        dist_thr: center-to-center euclidean distance threshold for matching
        """
        self.detector = YOLO(det_model)
        self.conf = conf
        self.device = device
        self.persistence_sec = float(persistence_sec)
        self.dist_thr = float(dist_thr)
        self.show_timers = show_timers

        self.tracked: Dict[int, Dict[str, Tuple[int, int] | float]] = {}  # id -> {"center":(x,y), "start_time":float}
        self.next_id: int = 0

    # ----- Live controls -----

    def reset(self) -> None:
        """Clear IDs and timers."""
        self.tracked.clear()
        self.next_id = 0

    def set_persistence(self, seconds: float, reset: bool = True) -> None:
        """Update time threshold on the fly."""
        try:
            self.persistence_sec = float(seconds)
        except Exception:
            pass
        if reset:
            self.reset()

    # ----- Internal utils -----

    def _assign_id(self, cx: int, cy: int) -> Optional[int]:
        """Assign an ID by nearest-center distance (euclidean)."""
        for obj_id, data in self.tracked.items():
            px, py = data["center"]  # type: ignore[index]
            if math.hypot(cx - px, cy - py) < self.dist_thr:
                return obj_id
        return None

    def _stay_seconds_label(self, elapsed: float) -> int:
        """
        Convert elapsed time to an integer 'stay seconds' that starts at the threshold value.
        Example: threshold=3.0 -> at elapsed=3.2 returns 3; at 4.1 returns 4, etc.
        Works for non-integer thresholds by starting at ceil(threshold).
        """
        base = math.ceil(self.persistence_sec)
        extra = int(max(0.0, elapsed - self.persistence_sec))
        return base + extra

    def _draw_text_with_bg(
        self,
        frame: np.ndarray,
        text: str,
        org: Tuple[int, int],
        scale: float,
        fg: Tuple[int, int, int],
        thickness: int,
        bg: Tuple[int, int, int] = (0, 0, 0),
        alpha: float = 0.45,
        pad_factor: float = 0.6,
    ) -> None:
        """
        Draw text with a semi-transparent background box for readability.
        org is the baseline-left text origin like cv2.putText.
        """
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        x, y = org
        pad = int(pad_factor * (th + base))
        x1, y1 = max(0, x - pad), max(0, y - th - pad // 2)
        x2, y2 = min(frame.shape[1] - 1, x + tw + pad), min(frame.shape[0] - 1, y + base + pad // 2)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), bg, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        # Outline + fill for crisp text
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, fg, thickness, cv2.LINE_AA)

    # ----- Main -----

    def process(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, bool]:
        """
        Run detection on the frame, update timers, draw boxes, and return (frame, alert).
        'alert' is True if any object persisted ≥ self.persistence_sec.
        """
        results = self.detector.predict(
            frame_bgr,
            conf=self.conf,
            device=self.device,
            verbose=False
        )

        if not results:
            self.tracked = {}
            return frame_bgr, False

        res = results[0]
        if res.boxes is None or len(res.boxes) == 0:
            self.tracked = {}
            return frame_bgr, False

        boxes = res.boxes.xyxy.to("cpu").numpy().astype(int).tolist()

        new_tracked: Dict[int, Dict[str, Tuple[int, int] | float]] = {}
        any_alert = False
        now = time.time()

        for x1, y1, x2, y2 in boxes:
            # Clamp to frame
            h, w = frame_bgr.shape[:2]
            x1 = max(0, min(w - 1, x1))
            x2 = max(0, min(w - 1, x2))
            y1 = max(0, min(h - 1, y1))
            y2 = max(0, min(h - 1, y2))

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            obj_id = self._assign_id(cx, cy)
            if obj_id is None:
                obj_id = self.next_id
                self.next_id += 1
                start_time = now
            else:
                start_time = float(self.tracked[obj_id]["start_time"])  # type: ignore[index]

            elapsed = now - start_time
            new_tracked[obj_id] = {"center": (cx, cy), "start_time": start_time}

            # Visuals
            box_h = max(1, y2 - y1)
            scale = max(0.6, min(1.8, box_h / 180.0))
            thickness = max(1, int(2 * scale))

            if elapsed >= self.persistence_sec:

                color = (0, 0, 255)
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)

                stay_s = self._stay_seconds_label(elapsed)
                text = f"stay: {stay_s}s"
                tx, ty = x1, max(12, y1 - 8)

                # No semi-transparent background — just text.
                # Keep a subtle outline so it stays readable; remove the first line if you want *no* outline.
                cv2.putText(frame_bgr, text, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)  # outline
                cv2.putText(frame_bgr, text, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 255), thickness, cv2.LINE_AA)    # red text
                any_alert = True

            else:
                # GREEN box; optional small running timer pre-threshold
                color = (0, 255, 0)
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
                if self.show_timers:
                    text = f"{elapsed:.1f}s"
                    tx, ty = x1, max(12, y1 - 6)
                    cv2.putText(
                        frame_bgr, text, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, max(0.5, scale * 0.9), color, max(1, thickness - 1), cv2.LINE_AA
                    )

        self.tracked = new_tracked
        return frame_bgr, any_alert

