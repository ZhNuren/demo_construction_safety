# import cv2
# from ultralytics import YOLO


# class ObjectPresenceDetector:
#     def __init__(self, det_model="best.pt", conf=0.4, device="cpu", persistence_sec=5.0, fps=30, dist_thr=50):
#         """
#         det_model: path to your YOLO weights file (e.g. best.pt)
#         conf: confidence threshold
#         device: "cpu", "cuda", "mps"
#         persistence_sec: how many seconds before alert is triggered
#         fps: assumed video FPS
#         dist_thr: distance threshold for tracking IDs
#         """
#         self.detector = YOLO(det_model)
#         self.conf = conf
#         self.device = device
#         self.frame_thr = int(persistence_sec * fps)
#         self.tracked = {}   # object_id -> {"count": consecutive_frames, "center": (x,y)}
#         self.next_id = 0
#         self.frame_idx = 0
#         self.dist_thr = dist_thr

#     def _assign_id(self, box, prev_boxes):
#         """Assign ID by checking center distance from previous detections"""
#         cx = (box[0] + box[2]) // 2
#         cy = (box[1] + box[3]) // 2
#         for obj_id, (px, py) in prev_boxes.items():
#             if abs(cx - px) < self.dist_thr and abs(cy - py) < self.dist_thr:
#                 return obj_id, cx, cy
#         return None, cx, cy

#     def process(self, frame_bgr):
#         self.frame_idx += 1

#         # Run YOLO detection
#         results = self.detector.predict(
#             frame_bgr,
#             conf=self.conf,
#             device=self.device,
#             verbose=False
#         )

#         if not results:
#             return frame_bgr, False

#         res = results[0]
#         boxes = res.boxes.xyxy.cpu().numpy().astype(int).tolist()

#         # previous centers for ID matching
#         prev_centers = {obj_id: data["center"] for obj_id, data in self.tracked.items()}

#         # update tracking
#         new_tracked = {}
#         alert = False
#         for box in boxes:
#             obj_id, cx, cy = self._assign_id(box, prev_centers)
#             if obj_id is None:
#                 obj_id = self.next_id
#                 self.next_id += 1

#             count = self.tracked.get(obj_id, {}).get("count", 0) + 1
#             new_tracked[obj_id] = {"count": count, "center": (cx, cy)}

#             # draw box
#             x1, y1, x2, y2 = box
#             color = (0, 0, 255) if count >= self.frame_thr else (0, 255, 0)
#             cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
#             cv2.putText(frame_bgr, f"ID:{obj_id} {count/self.frame_thr:.1f}s",
#                         (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#             if count >= self.frame_thr:
#                 alert = True

#         self.tracked = new_tracked

#         # global alert
#         if alert:
#             cv2.putText(frame_bgr, "ALERT: Object present > threshold!",
#                         (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

#         return frame_bgr, alert


# import cv2
# from ultralytics import YOLO


# class ObjectPresenceDetector:
#     def __init__(self, det_model="best.pt", conf=0.4, device="cpu", persistence_sec=5.0, fps=30, dist_thr=50):
#         """
#         det_model: YOLO weights file (e.g. best.pt)
#         conf: confidence threshold
#         device: "cpu", "cuda", "mps"
#         persistence_sec: how many seconds before alert
#         fps: assumed FPS of video
#         dist_thr: distance threshold for tracking IDs
#         """
#         self.detector = YOLO(det_model)
#         self.conf = conf
#         self.device = device
#         self.frame_thr = int(persistence_sec * fps)
#         self.tracked = {}   # object_id -> {"count": consecutive_frames, "center": (x,y)}
#         self.next_id = 0
#         self.dist_thr = dist_thr
#         self.frame_idx = 0

#     def _assign_id(self, box, prev_boxes):
#         """Assign an ID by checking center distance from previous detections"""
#         cx = (box[0] + box[2]) // 2
#         cy = (box[1] + box[3]) // 2
#         for obj_id, (px, py) in prev_boxes.items():
#             if abs(cx - px) < self.dist_thr and abs(cy - py) < self.dist_thr:
#                 return obj_id, cx, cy
#         return None, cx, cy

#     def process(self, frame_bgr):
#         self.frame_idx += 1

#         # Run YOLO detection
#         results = self.detector.predict(
#             frame_bgr,
#             conf=self.conf,
#             device=self.device,
#             verbose=False
#         )

#         if not results:
#             return frame_bgr, False

#         res = results[0]
#         boxes = res.boxes.xyxy.cpu().numpy().astype(int).tolist()
#         labels = res.boxes.cls.cpu().numpy().astype(int).tolist()
#         scores = res.boxes.conf.cpu().numpy().tolist()

#         # previous centers for ID matching
#         prev_centers = {obj_id: data["center"] for obj_id, data in self.tracked.items()}

#         # update tracking
#         new_tracked = {}
#         alert = False
#         for (box, cls_id, score) in zip(boxes, labels, scores):
#             obj_id, cx, cy = self._assign_id(box, prev_centers)
#             if obj_id is None:
#                 obj_id = self.next_id
#                 self.next_id += 1

#             count = self.tracked.get(obj_id, {}).get("count", 0) + 1
#             new_tracked[obj_id] = {"count": count, "center": (cx, cy)}

#             # draw box
#             x1, y1, x2, y2 = box
#             label = f"{self.detector.names[cls_id]} {score:.2f}"
#             color = (0, 0, 255) if count >= self.frame_thr else (0, 255, 0)
#             cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
#             cv2.putText(frame_bgr, f"ID:{obj_id} {count/self.frame_thr:.1f}s {label}",
#                         (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#             if count >= self.frame_thr:
#                 alert = True

#         self.tracked = new_tracked

#         # global alert message
#         if alert:
#             cv2.putText(frame_bgr, "ALERT: Object present > threshold!",
#                         (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

#         return frame_bgr, alert


import cv2
import time
from ultralytics import YOLO


class ObjectPresenceDetector:
    def __init__(self, det_model="best.pt", conf=0.4, device="cpu",
                 persistence_sec=5.0, dist_thr=100):
        """
        det_model: YOLO weights file (e.g. best.pt)
        conf: confidence threshold
        device: "cpu", "cuda", "mps"
        persistence_sec: how many seconds before alert
        dist_thr: distance threshold for matching detections
        """
        self.detector = YOLO(det_model)
        self.conf = conf
        self.device = device
        self.persistence_sec = persistence_sec
        self.dist_thr = dist_thr

        self.tracked = {}   # object_id -> {"center": (x,y), "start_time": float}
        self.next_id = 0

        print("✅ Loaded model:", det_model)
        print("✅ Classes available:", self.detector.names)

    def _assign_id(self, cx, cy):
        """Assign an ID by checking distance from previous centers"""
        for obj_id, data in self.tracked.items():
            px, py = data["center"]
            if abs(cx - px) < self.dist_thr and abs(cy - py) < self.dist_thr:
                return obj_id
        return None

    def process(self, frame_bgr):
        results = self.detector.predict(
            frame_bgr,
            conf=self.conf,
            device=self.device,
            verbose=False
        )
        if not results:
            return frame_bgr, False

        res = results[0]
        boxes = res.boxes.xyxy.cpu().numpy().astype(int).tolist()

        new_tracked = {}
        alert = False
        now = time.time()

        for box in boxes:
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            obj_id = self._assign_id(cx, cy)
            if obj_id is None:
                obj_id = self.next_id
                self.next_id += 1
                start_time = now
            else:
                start_time = self.tracked[obj_id]["start_time"]

            elapsed = now - start_time
            new_tracked[obj_id] = {"center": (cx, cy), "start_time": start_time}

            # draw box
            color = (0, 0, 255) if elapsed >= self.persistence_sec else (0, 255, 0)
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_bgr, f"{elapsed:.1f}s",
                        (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if elapsed >= self.persistence_sec:
                alert = True

        self.tracked = new_tracked

        if alert:
            cv2.putText(frame_bgr, "ALERT: Object present > threshold!",
                        (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 255), 3)

        return frame_bgr, alert
