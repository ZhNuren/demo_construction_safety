from typing import Optional, List, Tuple
import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

class YOLODetector:
    """Thin wrapper around Ultralytics YOLOv8n for object detection."""
    def __init__(self, model_name: str = "yolov8n.pt", conf: float = 0.35, imgsz: int = 640):
        if YOLO is None:
            raise ImportError("Ultralytics not installed. Run: pip install ultralytics")
        self.model = YOLO(model_name)
        self.conf = conf
        self.imgsz = imgsz

    def detect_xyxy(self, frame: np.ndarray, classes: Optional[List[int]] = None
                    ) -> Tuple[list, list, list]:
        """Returns (boxes_xyxy[int], scores[float], classes[int])"""
        res = self.model.predict(frame, verbose=False, conf=self.conf, imgsz=self.imgsz)[0]
        boxes = res.boxes
        xyxy = boxes.xyxy.cpu().numpy().astype(int).tolist()
        confs = boxes.conf.cpu().numpy().tolist()
        clss = boxes.cls.cpu().numpy().astype(int).tolist()
        if classes is not None:
            filtered = [(b,s,c) for b,s,c in zip(xyxy, confs, clss) if c in classes]
            if filtered:
                xyxy, confs, clss = map(list, zip(*filtered))
            else:
                xyxy, confs, clss = [], [], []
        return xyxy, confs, clss
