# # models/lpr.py
# import os
# import json
# from typing import List, Dict

# import cv2
# import numpy as np
# from PIL import Image

# import torch
# from ultralytics import YOLO
# from transformers import TrOCRProcessor, VisionEncoderDecoderModel


# class LicensePlateRecognizer:
#     """
#     YOLOv8 for license-plate detection + TrOCR (small) for text recognition.
#     Optimizations:
#       1. Use smaller model (`trocr-small-printed`).
#       2. Greedy decoding (fast).
#       3. Throttle OCR (every N frames).
#       4. Batch OCR crops per frame.
#     """

#     def __init__(
#         self,
#         db_path: str = "plates_db.json",
#         det_model: str = "yolov8n.pt",
#         conf: float = 0.40,
#         yolo_device: str = "mps",
#         trocr_device: str = "cpu",
#         ocr_interval: int = 2,   # run OCR every N frames
#     ):
#         # --- Detector (YOLOv8) ---
#         self.detector = YOLO(det_model)
#         self.conf = float(conf)
#         self.yolo_device = yolo_device

#         # --- OCR (TrOCR small) ---
#         self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed")
#         self.ocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-printed")

#         self.trocr_device = trocr_device if trocr_device in ("cpu", "cuda") else "cpu"
#         self.ocr_model.to(self.trocr_device)

#         # DB
#         self.db_path = db_path
#         self._load_db()

#         # OCR optimizations
#         self.ocr_interval = ocr_interval
#         self._frame_i = 0
#         self._last_results = []  # cache last OCR results

#         try:
#             cv2.setNumThreads(1)
#         except Exception:
#             pass

#     # -------------------- DB --------------------
#     def _load_db(self):
#         if os.path.exists(self.db_path):
#             with open(self.db_path, "r") as f:
#                 self.db: Dict[str, Dict[str, bool]] = json.load(f)
#         else:
#             self.db = {}

#     def _save_db(self):
#         with open(self.db_path, "w") as f:
#             json.dump(self.db, f, indent=2)

#     def enroll_plate(self, plate: str, not_allowed: bool = False):
#         plate = (plate or "").upper().strip()
#         if plate:
#             self.db[plate] = {"not_allowed": bool(not_allowed)}
#             self._save_db()

#     def delete_plate(self, plate: str):
#         plate = (plate or "").upper().strip()
#         if plate in self.db:
#             del self.db[plate]
#             self._save_db()

#     def list_plates(self) -> List[Dict[str, str]]:
#         return [{"plate": p, "not_allowed": rec["not_allowed"]} for p, rec in self.db.items()]

#     # -------------------- OCR --------------------
#     def _preprocess_crop_for_ocr(self, crop_bgr: np.ndarray) -> Image.Image:
#         h, w = crop_bgr.shape[:2]
#         if w < 120:
#             scale = 120.0 / max(1, w)
#             crop_bgr = cv2.resize(crop_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
#         return Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))

#     def _ocr_batch(self, crops: List[np.ndarray]) -> List[str]:
#         if not crops:
#             return []
#         try:
#             imgs = [self._preprocess_crop_for_ocr(c) for c in crops]
#             pixel_values = self.processor(images=imgs, return_tensors="pt").pixel_values.to(self.trocr_device)

#             with torch.no_grad():
#                 generated_ids = self.ocr_model.generate(
#                     pixel_values,
#                     max_length=16,
#                     num_beams=1,   # greedy decoding
#                 )

#             texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
#             results = []
#             for text in texts:
#                 text = (text or "").upper().strip()
#                 filtered = "".join(ch for ch in text if ("A" <= ch <= "Z") or ("0" <= ch <= "9"))
#                 results.append(filtered if filtered else (text if text else "???"))
#             return results
#         except Exception as e:
#             print("TrOCR batch error:", e)
#             return ["???"] * len(crops)

#     # -------------------- Detection + OCR --------------------
#     def detect_and_identify(self, frame_bgr: np.ndarray) -> List[Dict]:
#         results: List[Dict] = []
#         if frame_bgr is None or frame_bgr.size == 0:
#             return results

#         # YOLO detection
#         try:
#             yres = self.detector.predict(
#                 frame_bgr,
#                 verbose=False,
#                 conf=self.conf,
#                 device=self.yolo_device,
#                 max_det=100,
#             )[0]
#         except Exception as e:
#             print("YOLO detection error:", e)
#             return results

#         boxes = yres.boxes.xyxy.cpu().numpy().astype(int).tolist()
#         crops = []
#         valid_boxes = []

#         h, w = frame_bgr.shape[:2]
#         for box in boxes:
#             x1, y1, x2, y2 = box
#             x1, y2 = max(0, x1), min(h, y2)
#             x2, y1 = min(w, x2), max(0, y1)
#             if x2 <= x1 or y2 <= y1:
#                 continue
#             crop = frame_bgr[y1:y2, x1:x2]
#             if crop.size == 0:
#                 continue
#             valid_boxes.append((x1, y1, x2, y2))
#             crops.append(crop)

#         # Increment frame counter
#         self._frame_i += 1
#         do_ocr = (self._frame_i % self.ocr_interval == 0)

#         plates = []
#         if do_ocr and crops:
#             plates = self._ocr_batch(crops)
#             self._last_results = plates
#         else:
#             # reuse last OCR results if available
#             if self._last_results and len(self._last_results) == len(crops):
#                 plates = self._last_results
#             else:
#                 plates = ["???"] * len(crops)

#         for (x1, y1, x2, y2), plate in zip(valid_boxes, plates):
#             status = "unknown"
#             rec = self.db.get(plate)
#             if rec is not None:
#                 status = "not_allowed" if rec.get("not_allowed") else "allowed"
#             if rec is None:
#                 plate = ""
#             results.append(
#                 {
#                     "bbox": (int(x1), int(y1), int(x2), int(y2)),
#                     "plate": plate,
#                     "status": status,
#                 }
#             )

#         return results
# models/lpr.py
import os
import json
from typing import List, Dict

import cv2
import numpy as np
from PIL import Image

import torch
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


class LicensePlateRecognizer:
    def __init__(
        self,
        db_path: str = "plates_db.json",
        det_model: str = "yolov8n.pt",
        conf: float = 0.4,
        yolo_device: str = "mps",
        trocr_device: str = "mps",
        ocr_interval: int = 3,
    ):
        # YOLO
        self.detector = YOLO(det_model)
        self.conf = float(conf)
        self.yolo_device = yolo_device


        # OCR (TrOCR small)
        self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
        self.ocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
        self.trocr_device = trocr_device if trocr_device in ("cpu", "cuda", "mps") else "cpu"
        self.ocr_model.to(self.trocr_device)

        # DB
        self.db_path = db_path
        self._load_db()

        # OCR optimization
        self.ocr_interval = ocr_interval
        self._frame_i = 0
        self._last_results = []

        try:
            cv2.setNumThreads(1)
        except Exception:
            pass

    # -------------------- DB --------------------
    def _load_db(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, "r") as f:
                self.db: Dict[str, Dict[str, bool]] = json.load(f)
        else:
            self.db = {}

    def _save_db(self):
        with open(self.db_path, "w") as f:
            json.dump(self.db, f, indent=2)

    def enroll_plate(self, plate: str, not_allowed: bool = False):
        plate = (plate or "").upper().strip()
        if plate:
            self.db[plate] = {"not_allowed": bool(not_allowed)}
            self._save_db()

    def delete_plate(self, plate: str):
        plate = (plate or "").upper().strip()
        if plate in self.db:
            del self.db[plate]
            self._save_db()

    def list_plates(self) -> List[Dict[str, str]]:
        return [{"plate": p, "not_allowed": rec["not_allowed"]} for p, rec in self.db.items()]

    # -------------------- OCR --------------------
    def _preprocess_crop_for_ocr(self, crop_bgr: np.ndarray) -> Image.Image:
        return Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))

    def _ocr_batch(self, crops: List[np.ndarray]) -> List[str]:
        if not crops:
            return []
        try:
            imgs = [self._preprocess_crop_for_ocr(c) for c in crops]
            pixel_values = self.processor(images=imgs, return_tensors="pt").pixel_values.to(self.trocr_device)

            with torch.no_grad():
                generated_ids = self.ocr_model.generate(
                    pixel_values,
                    max_length=8,
                    num_beams=3,
                )

            texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            results = []
            for text in texts:
                text = (text or "").upper().strip()
                filtered = "".join(ch for ch in text if ("A" <= ch <= "Z") or ("0" <= ch <= "9"))
                results.append(filtered if filtered else (text if text else "???"))
            return results
        except Exception as e:
            print("TrOCR batch error:", e)
            return ["???"] * len(crops)

    # -------------------- Detection + OCR --------------------
    def detect_and_identify(self, frame_bgr: np.ndarray) -> List[Dict]:
        results: List[Dict] = []
        if frame_bgr is None or frame_bgr.size == 0:
            return results

        try:
            yres = self.detector.predict(
                frame_bgr, verbose=False, conf=self.conf, device=self.yolo_device, max_det=100
            )[0]
        except Exception as e:
            print("YOLO detection error:", e)
            return results

        boxes = yres.boxes.xyxy.cpu().numpy().astype(int).tolist()
        crops, valid_boxes = [], []

        h, w = frame_bgr.shape[:2]
        for x1, y1, x2, y2 in boxes:
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            valid_boxes.append((x1, y1, x2, y2))
            crops.append(crop)

        self._frame_i += 1
        do_ocr = (self._frame_i % self.ocr_interval == 0)

        if do_ocr and crops:
            plates = self._ocr_batch(crops)
            self._last_results = plates
        else:
            if self._last_results and len(self._last_results) == len(crops):
                plates = self._last_results
            else:
                plates = ["???"] * len(crops)

        for (x1, y1, x2, y2), plate in zip(valid_boxes, plates):
            status = "unknown"
            rec = self.db.get(plate)
            if rec is not None:
                status = "not_allowed" if rec.get("not_allowed") else "allowed"
            if rec is None:
                plate = ""
            results.append(
                {"bbox": (int(x1), int(y1), int(x2), int(y2)), "plate": plate, "status": status}
            )

        return results

