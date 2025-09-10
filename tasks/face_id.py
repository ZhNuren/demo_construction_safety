from __future__ import annotations
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from typing import Optional

from .base import TaskPage
from models.faceid import FaceIdentifier

class FaceIDPage(TaskPage):
    def __init__(self, master, **kwargs):
        super().__init__(master, task_key="Face ID", task_title="Face Identification", **kwargs)
        self._identifier: Optional[FaceIdentifier] = None
        self._recognition_enabled = False

    def _build_controls(self):
        super()._build_controls()
        ttk.Separator(self.toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=12)

        enroll = ttk.Frame(self.toolbar)
        enroll.pack(side=tk.LEFT)
        ttk.Label(enroll, text="Name:").grid(row=0, column=0, sticky="w")
        self.name_var = tk.StringVar()
        ttk.Entry(enroll, textvariable=self.name_var, width=16).grid(row=0, column=1, padx=(4,8))
        self.na_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(enroll, text="Not allowed", variable=self.na_var).grid(row=0, column=2, padx=(0,8))
        ttk.Button(enroll, text="Upload & Enroll", command=self._upload_and_enroll).grid(row=0, column=3)

        ttk.Separator(self.toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=12)
        ctrl = ttk.Frame(self.toolbar)
        ctrl.pack(side=tk.LEFT)
        ttk.Button(ctrl, text="Start recognition", command=self._start).grid(row=0, column=0)
        ttk.Button(ctrl, text="Stop", command=self._stop).grid(row=0, column=1, padx=6)
        ttk.Label(ctrl, text="Threshold").grid(row=0, column=2, padx=(8,4))
        self.thr = tk.DoubleVar(value=0.35)
        ttk.Spinbox(ctrl, from_=0.1, to=0.9, increment=0.01, width=6, textvariable=self.thr).grid(row=0, column=3)

    def _ensure_identifier(self):
        if self._identifier is None:
            try:
                self._identifier = FaceIdentifier(db_path="faces_db.json")
            except Exception as e:
                messagebox.showerror("Face ID init failed", str(e))
                return None
        return self._identifier

    def _upload_and_enroll(self):
        ident = self._ensure_identifier()
        if ident is None:
            return
        if not self.name_var.get().strip():
            messagebox.showwarning("Missing name", "Please enter a name for this person.")
            return
        path = filedialog.askopenfilename(
            title="Select face image",
            filetypes=[("Images","*.jpg *.jpeg *.png *.bmp"), ("All files","*.*")]
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Open image", "Failed to load image")
            return
        ok = ident.enroll_from_image(img, self.name_var.get().strip(), self.na_var.get())
        if not ok:
            messagebox.showwarning("No face detected", "Couldn't detect a face in that image.")
        else:
            self.notify(f"Enrolled: {self.name_var.get().strip()} ({'NOT ALLOWED' if self.na_var.get() else 'allowed'})")

    def _start(self):
        if self._ensure_identifier() is None:
            return
        self._recognition_enabled = True
        self.player.on_frame = self._process_frame
        self.notify("Recognition started")

    def _stop(self):
        self._recognition_enabled = False
        self.player.on_frame = None
        self.notify("Recognition stopped")

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        if not self._recognition_enabled:
            return frame
        ident = self._ensure_identifier()
        if ident is None:
            return frame
        results = ident.identify_in_frame(frame, sim_threshold=float(self.thr.get()))
        for r in results:
            x1,y1,x2,y2 = r["bbox"]
            name = r["name"]
            score = r["score"]
            not_allowed = r["not_allowed"]
            color = (40,200,80) if name != "Unknown" and not not_allowed else (30,30,230)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            label = f"{name if name!='Unknown' else 'Unknown'}"
            if name != "Unknown":
                label += f"  sim={score:.2f}"
            if not_allowed and name != "Unknown":
                label += "  NOT ALLOWED"
            cv2.putText(frame, label, (x1, max(20,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        return frame
