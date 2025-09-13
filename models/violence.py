# models/violence.py
import cv2
from ultralytics import YOLO


class ViolenceDetector:
    """
    Wraps a YOLO model trained for violence/fight detection.
    Triggers alert immediately if any relevant detection appears (no time threshold).
    """

    def __init__(self, det_model: str = "violence.pt", conf: float = 0.4, device: str = "cpu",
                 target_classes: list[str] | None = None, draw: bool = True):
        """
        det_model: path to YOLO weights (e.g., 'violence.pt')
        conf: confidence threshold for predictions
        device: 'cpu' | 'cuda' | 'mps'
        target_classes: optional whitelist of class names to keep (e.g., ['fight', 'violence'])
                        If None -> treat any detection from the model as violence.
        draw: draw boxes/labels on frames
        """
        self.model = YOLO(det_model)
        self.conf = conf
        self.device = device
        self.target_classes = target_classes
        self.draw = draw

    def _keep_detection(self, class_name: str) -> bool:
        if self.target_classes is None:
            return True
        return class_name in self.target_classes

    def process(self, frame_bgr):
        """
        Runs the model and returns (annotated_frame, alert_bool).
        Alert is True iff at least one kept detection exists OR a classification top-1 is in target list.
        """
        results = self.model.predict(
            frame_bgr,
            conf=self.conf,
            device=self.device,
            verbose=False
        )
        if not results:
            return frame_bgr, False

        res = results[0]
        names = res.names if hasattr(res, "names") else {}

        alert = False

        # --- Detection path (boxes)
        if hasattr(res, "boxes") and res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy().astype(int).tolist()
            clss = res.boxes.cls.cpu().numpy().astype(int).tolist()
            confs = res.boxes.conf.cpu().numpy().tolist()

            for (x1, y1, x2, y2), cls_id, score in zip(xyxy, clss, confs):
                class_name = names.get(cls_id, str(cls_id))
                if not self._keep_detection(class_name):
                    continue

                alert = True
                if self.draw:
                    label = f"{class_name} {score:.2f}"
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame_bgr, label, (x1, max(15, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # --- Classification path (if your weights are classification-only)
        elif hasattr(res, "probs") and res.probs is not None:
            # Top-1 prediction
            top_idx = int(res.probs.top1)
            top_prob = float(res.probs.top1conf)
            class_name = names.get(top_idx, str(top_idx))
            print(class_name)

            # If target_classes is provided, require membership; else any class triggers (common for violence-only heads)
            if self._keep_detection(class_name) and top_prob >= self.conf:
                alert = True
                if self.draw:
                    cv2.putText(frame_bgr, f"{class_name} {top_prob:.2f}", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)


        return frame_bgr, alert
