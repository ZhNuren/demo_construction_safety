# models/crowd.py
import cv2
from ultralytics import YOLO

class CrowdCounter:
    def __init__(self, det_model="yolo11n.pt", conf=0.4, device="mps"):
        """
        det_model: path or name of YOLOv11 model (e.g. yolo11n.pt, yolo11s.pt).
        """
        self.detector = YOLO(det_model)
        self.conf = conf
        self.device = device

    def count_people(self, frame_bgr):
        # Run YOLO detection with person class only (class 0 = person in COCO)
        results = self.detector.predict(
            frame_bgr, verbose=False,
            conf=self.conf, device=self.device, classes=[0]
        )
        if not results:
            return 0, frame_bgr

        res = results[0]
        num_people = len(res.boxes)

        # Draw bounding boxes
        for box in res.boxes.xyxy.cpu().numpy().astype(int).tolist():
            x1, y1, x2, y2 = box
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(frame_bgr, f"People: {num_people}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return num_people, frame_bgr
