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
        self._identifier: Optional[FaceIdentifier] = None  # init before super().__init__
        self._recognition_enabled = False
        super().__init__(master, task_key="Face ID", task_title="Face Identification", **kwargs)

    def _build_controls(self):
        # Keep the common media toolbar + ROI (from base)
        super()._build_controls()

        # Replace long, wide toolbar with a compƒact notebook
        nb = ttk.Notebook(self.toolbar)
        nb.pack(side=tk.LEFT, padx=8)

        # --- Enroll tab ---
        tab_enroll = ttk.Frame(nb)
        nb.add(tab_enroll, text="Enroll")

        frm_e = ttk.Frame(tab_enroll, padding=(6,6))
        frm_e.pack(fill=tk.X)

        ttk.Label(frm_e, text="Name:").grid(row=0, column=0, sticky="w")
        self.name_var = tk.StringVar()
        ttk.Entry(frm_e, textvariable=self.name_var, width=16).grid(row=0, column=1, padx=(4,8))
        self.na_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm_e, text="Not allowed", variable=self.na_var).grid(row=0, column=2, padx=(0,8))
        ttk.Button(frm_e, text="Upload & Enroll", command=self._upload_and_enroll).grid(row=0, column=3, padx=(4,0))

        # --- Run tab (recognition + sources) ---
        tab_run = ttk.Frame(nb)
        nb.add(tab_run, text="Run")

        frm_r1 = ttk.Frame(tab_run, padding=(6,6))   # row 1: run buttons + threshold
        frm_r1.pack(fill=tk.X)
        ttk.Button(frm_r1, text="Start", command=self._start).grid(row=0, column=0)
        ttk.Button(frm_r1, text="Stop", command=self._stop).grid(row=0, column=1, padx=(6,0))

        ttk.Label(frm_r1, text="Thr").grid(row=0, column=2, padx=(10,4))
        self.thr = tk.DoubleVar(value=0.35)
        ttk.Spinbox(frm_r1, from_=0.1, to=0.9, increment=0.01, width=6,
                    textvariable=self.thr).grid(row=0, column=3)

            # --- Database tab ---
        tab_db = ttk.Frame(nb)
        nb.add(tab_db, text="Database")

        frm_db = ttk.Frame(tab_db, padding=(6,6))
        frm_db.pack(fill=tk.X)

        ttk.Label(frm_db, text="Allowed").grid(row=0, column=0)
        self.allowed_list = tk.Listbox(frm_db, height=5, width=16, exportselection=False)
        self.allowed_list.grid(row=1, column=0, padx=4)
        ttk.Button(frm_db, text="Delete", command=lambda: self._delete_selected(self.allowed_list)).grid(row=2, column=0, pady=(2,0))

        ttk.Label(frm_db, text="Not Allowed").grid(row=0, column=1)
        self.not_allowed_list = tk.Listbox(frm_db, height=5, width=16, exportselection=False)
        self.not_allowed_list.grid(row=1, column=1, padx=4)
        ttk.Button(frm_db, text="Delete", command=lambda: self._delete_selected(self.not_allowed_list)).grid(row=2, column=1, pady=(2,0))

        # Refresh lists when UI is built
        self._refresh_lists()

        # row 2: camera + screen options (stacked below)
        frm_r2 = ttk.Frame(tab_run, padding=(6,0))
        frm_r2.pack(fill=tk.X, pady=(4,0))

        # Camera
        ttk.Label(frm_r2, text="Cam").grid(row=0, column=0, sticky="w")
        self.cam_idx = tk.IntVar(value=0)
        ttk.Spinbox(frm_r2, from_=0, to=10, width=4, textvariable=self.cam_idx).grid(row=0, column=1, padx=(4,8))
        ttk.Button(frm_r2, text="Open Camera", command=self._open_camera).grid(row=0, column=2)

        # Screen
        ttk.Label(frm_r2, text="FPS").grid(row=0, column=3, padx=(10,4))
        self.scr_fps = tk.IntVar(value=20)
        ttk.Spinbox(frm_r2, from_=5, to=60, width=4, textvariable=self.scr_fps).grid(row=0, column=4, padx=(0,8))
        ttk.Button(frm_r2, text="Share Screen", command=self._open_screen).grid(row=0, column=5)


    def _refresh_lists(self):
        ident = self._ensure_identifier()
        if ident is None:
            return
        self.allowed_list.delete(0, tk.END)
        self.not_allowed_list.delete(0, tk.END)
        for rec in ident.list_identities():
            if rec["not_allowed"]:
                self.not_allowed_list.insert(tk.END, rec["name"])
            else:
                self.allowed_list.insert(tk.END, rec["name"])

    def _delete_selected(self, listbox):
        ident = self._ensure_identifier()
        if ident is None:
            return
        sel = listbox.curselection()
        if not sel:
            return
        name = listbox.get(sel[0])
        ident.delete_identity(name)
        self._refresh_lists()
        self.notify(f"Deleted {name}")

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
        
        self._refresh_lists()


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
    
    def _open_camera(self):
        idx = int(self.cam_idx.get())
        self.player.open_camera(index=idx)
        self.notify(f"Camera opened (index {idx})")

    def _open_screen(self):
        fps = int(self.scr_fps.get())
        # Full primary monitor. For a region: pass region=(left, top, width, height)
        self.player.open_screen(monitor=1, fps=fps, region=None)
        self.notify(f"Screen sharing started ({fps} FPS)")

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
