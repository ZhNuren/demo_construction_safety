from typing import Optional, List, Tuple
import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

class YOLODetector:
    def __init__(self, model_name: str = "yolov8n.pt", conf: float = 0.45, imgsz: int = 512, device: str = "mps"):
        if YOLO is None:
            raise ImportError("Ultralytics not installed. Run: pip install ultralytics")
        self.model = YOLO(model_name)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device  # "mps" on Apple Silicon, "cpu" fallback


    def detect_xyxy(self, frame, classes: Optional[List[int]] = None) -> Tuple[list, list, list]:
        res = self.model.predict(
            frame,
            verbose=False,
            conf=self.conf,     # set self.conf to 0.45 below
            imgsz=self.imgsz,   # consider 512 if 640 is slow
            max_det=100,         # cap total boxes per image
            device = self.device
        )[0]
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
