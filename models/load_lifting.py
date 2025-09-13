# models/tracking.py
from typing import Tuple, List, Dict
import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


class CraneDetector:
    """
    YOLO wrapper for a custom crane model.
    - Exposes class map via self.names (dict: id -> name)
    - detect_xyxy returns xyxy, confs, clss, labels (decoded names)
    """
    def __init__(
        self,
        model_name: str = "crane.pt",
        conf: float = 0.35,
        imgsz: int = 640,
        device: str = "mps",   # falls back to CPU if MPS not available
        max_det: int = 100,
    ):
        if YOLO is None:
            raise ImportError("Ultralytics not installed. Run: pip install ultralytics")

        # Device fallback
        try:
            import torch
            if device == "mps":
                if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                    device = "cpu"
        except Exception:
            device = "cpu"

        self.model = YOLO(model_name)
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self.device = device
        self.max_det = int(max_det)

        # Load class names map (robust across Ultralytics versions)
        nm = getattr(self.model, "names", None)
        if nm is None:
            try:
                nm = self.model.model.names  # older API fallback
            except Exception:
                nm = None
        if nm is None:
            nm = {}
        if isinstance(nm, list):
            nm = {i: v for i, v in enumerate(nm)}
        self.names: Dict[int, str] = dict(nm)

    def get_class_map(self) -> Dict[int, str]:
        """Return {class_id: class_name} mapping."""
        return dict(self.names)

    def _id_to_name(self, cls_id: int) -> str:
        return self.names.get(int(cls_id), f"cls{int(cls_id)}")

    def detect_xyxy(
        self, frame: np.ndarray
    ) -> Tuple[List[List[int]], List[float], List[int], List[str]]:
        """
        Returns: (xyxy_list, conf_list, cls_id_list, label_list)
        - xyxy_list: [[x1,y1,x2,y2], ...] (ints)
        - conf_list: [float, ...]
        - cls_id_list: [int, ...]
        - label_list: [str, ...] (decoded class names)
        """
        res = self.model.predict(
            frame,
            verbose=False,
            conf=self.conf,
            imgsz=self.imgsz,
            max_det=self.max_det,
            device=self.device,
        )[0]

        boxes = getattr(res, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return [], [], [], []

        xyxy = boxes.xyxy.detach().cpu().numpy().astype(int).tolist()
        confs = boxes.conf.detach().cpu().numpy().astype(float).tolist()
        clss = boxes.cls.detach().cpu().numpy().astype(int).tolist()

        # Prefer names from result if present; otherwise use self.names
        names_map = getattr(res, "names", None)
        if names_map is None:
            names_map = self.names
        if isinstance(names_map, list):
            names_map = {i: v for i, v in enumerate(names_map)}

        labels = [names_map.get(c, f"cls{c}") for c in clss]
        return xyxy, confs, clss, labels
