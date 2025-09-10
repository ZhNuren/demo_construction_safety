from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import numpy as np
import cv2
import json
import os

try:
    from insightface.app import FaceAnalysis
except Exception as e:
    FaceAnalysis = None

class FaceIdentifier:
    """Face enrollment + identification using InsightFace (ArcFace embeddings)."""
    def __init__(self, db_path: str = "faces_db.json", det_size=(640,640), provider: str = "CPUExecutionProvider"):
        if FaceAnalysis is None:
            raise ImportError("insightface not installed. Run: pip install insightface onnxruntime")
        self.app = FaceAnalysis(providers=[provider])
        self.app.prepare(ctx_id=0, det_size=det_size)
        self.db_path = db_path
        self.database: List[Dict] = []
        self.load()

    def load(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, "r") as f:
                raw = json.load(f)
            self.database = []
            for r in raw:
                emb = np.array(r["embedding"], dtype=np.float32)
                n = np.linalg.norm(emb) + 1e-9
                emb = emb / n
                self.database.append({
                    "name": r["name"],
                    "not_allowed": bool(r.get("not_allowed", False)),
                    "embedding": emb.tolist()
                })
        else:
            self.database = []

    def save(self):
        serial = []
        for r in self.database:
            serial.append({
                "name": r["name"],
                "not_allowed": bool(r.get("not_allowed", False)),
                "embedding": list(map(float, r["embedding"])),
            })
        with open(self.db_path, "w") as f:
            json.dump(serial, f, indent=2)

    def enroll_from_image(self, image_bgr, name: str, not_allowed: bool=False) -> bool:
        faces = self.app.get(image_bgr)
        if not faces:
            return False
        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        emb = face.normed_embedding  # L2-normalized
        self.database.append({
            "name": name,
            "not_allowed": bool(not_allowed),
            "embedding": emb.astype(np.float32).tolist()
        })
        self.save()
        return True

    def list_identities(self) -> List[Dict]:
        return [{"name": r["name"], "not_allowed": bool(r.get("not_allowed", False))} for r in self.database]

    def delete_identity(self, name: str):
        self.database = [r for r in self.database if r["name"] != name]
        self.save()

    def identify_in_frame(self, frame_bgr, sim_threshold: float=0.35):
        faces = self.app.get(frame_bgr)
        results = []
        if not faces:
            return results
        if self.database:
            db_embs = np.stack([np.array(r["embedding"], dtype=np.float32) for r in self.database], axis=0)
        else:
            db_embs = None
        for f in faces:
            bbox = tuple(int(v) for v in f.bbox.astype(int))
            emb = f.normed_embedding.astype(np.float32)
            name, score, not_allowed = "Unknown", 0.0, False
            if db_embs is not None and len(db_embs) > 0:
                sims = (db_embs @ emb).astype(np.float32)
                idx = int(np.argmax(sims))
                score = float(sims[idx])
                if score >= sim_threshold:
                    rec = self.database[idx]
                    name = rec["name"]
                    not_allowed = bool(rec.get("not_allowed", False))
            results.append({"bbox": bbox, "name": name, "score": score, "not_allowed": not_allowed})
        return results
