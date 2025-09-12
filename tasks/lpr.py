import tkinter as tk
from tkinter import ttk, messagebox
import cv2, numpy as np
from typing import Optional

from .base import TaskPage
from models.lpr import LicensePlateRecognizer

class LPRPage(TaskPage):
    def __init__(self, master, **kwargs):
        super().__init__(master, task_key="LPR", task_title="License Plate Recognition", **kwargs)
        self._lpr = LicensePlateRecognizer(det_model="best_lpr.pt")
        self._enabled = False
        self._refresh_lists()


    def _build_controls(self):
        super()._build_controls()

        nb = ttk.Notebook(self.toolbar)
        nb.pack(side=tk.LEFT, padx=8)

        # --- Manage tab (add/list plates) ---
        tab_manage = ttk.Frame(nb)
        nb.add(tab_manage, text="Manage")

        frm_m = ttk.Frame(tab_manage, padding=(6,6))
        frm_m.pack(fill=tk.X)

        ttk.Label(frm_m, text="Plate:").grid(row=0, column=0, sticky="w")
        self.plate_var = tk.StringVar()
        ttk.Entry(frm_m, textvariable=self.plate_var, width=12).grid(row=0, column=1, padx=(4,8))
        ttk.Button(frm_m, text="Add Allowed", command=lambda: self._add_plate(False)).grid(row=0, column=2, padx=2)
        ttk.Button(frm_m, text="Add Not Allowed", command=lambda: self._add_plate(True)).grid(row=0, column=3, padx=2)

        # Lists
        lists = ttk.Frame(frm_m, padding=(0,6))
        lists.grid(row=1, column=0, columnspan=4, pady=(8,0))

        ttk.Label(lists, text="Allowed").grid(row=0, column=0)
        self.allowed_list = tk.Listbox(lists, height=4, width=12, exportselection=False)
        self.allowed_list.grid(row=1, column=0, padx=4)
        ttk.Button(lists, text="Delete", command=lambda: self._delete_selected(self.allowed_list)).grid(row=2, column=0, pady=(2,0))

        ttk.Label(lists, text="Not Allowed").grid(row=0, column=1)
        self.not_allowed_list = tk.Listbox(lists, height=4, width=12, exportselection=False)
        self.not_allowed_list.grid(row=1, column=1, padx=4)
        ttk.Button(lists, text="Delete", command=lambda: self._delete_selected(self.not_allowed_list)).grid(row=2, column=1, pady=(2,0))

        # --- Run tab (start/stop recognition) ---
        tab_run = ttk.Frame(nb)
        nb.add(tab_run, text="Run")

        frm_r = ttk.Frame(tab_run, padding=(6,6))
        frm_r.pack(fill=tk.X)
        ttk.Button(frm_r, text="Start", command=self._start).grid(row=0, column=0, padx=2)
        ttk.Button(frm_r, text="Stop", command=self._stop).grid(row=0, column=1, padx=2)

    
    def _ensure_lpr(self):
        if self._lpr is None:
            try:
                self._lpr = LicensePlateRecognizer(det_model="lpr.pt")  # supply path to model
            except Exception as e:
                messagebox.showerror("LPR init failed", str(e))
                return None
        return self._lpr

    def _refresh_lists(self):
        lpr = self._ensure_lpr()
        if lpr is None:
            return
        self.allowed_list.delete(0, tk.END)
        self.not_allowed_list.delete(0, tk.END)
        for plate, rec in sorted(lpr.db.items()):
            if rec["not_allowed"]:
                self.not_allowed_list.insert(tk.END, plate)
            else:
                self.allowed_list.insert(tk.END, plate)

    def _add_plate(self, not_allowed: bool):
        lpr = self._ensure_lpr()
        if lpr is None: return
        plate = self.plate_var.get().strip().upper()
        if not plate:
            messagebox.showwarning("Missing", "Enter a plate string.")
            return
        lpr.enroll_plate(plate, not_allowed)
        status = "NOT ALLOWED" if not_allowed else "ALLOWED"
        self.notify(f"Plate {plate} added ({status})")
        self._refresh_lists()

    def _delete_selected(self, listbox: tk.Listbox):
        lpr = self._ensure_lpr()
        if lpr is None: return
        sel = listbox.curselection()
        if not sel:
            return
        plate = listbox.get(sel[0])
        lpr.delete_plate(plate)
        self.notify(f"Deleted plate {plate}")
        self._refresh_lists()

    def _start(self):
        if self._ensure_lpr() is None: return
        self._enabled = True
        self.player.on_frame = self._process
        self.notify("Plate recognition started")

    def _stop(self):
        self._enabled = False
        self._refresh_lists()
        self.player.on_frame = None
        self.notify("Stopped")

    def _process(self, frame: np.ndarray) -> np.ndarray:
        if not self._enabled:
            return frame
        lpr = self._ensure_lpr()
        if lpr is None: return frame

        results = lpr.detect_and_identify(frame)
        for r in results:
            x1, y1, x2, y2 = r["bbox"]
            plate, status = r["plate"], r["status"]

            if status == "allowed":
                color = (0,200,0)   # green
                label = f"{plate}  ALLOWED"
            elif status == "not_allowed":
                color = (0,0,255)   # red
                label = f"{plate}  NOT ALLOWED"
            else:
                color = (200,200,0) # yellow for unknown
                label = f"{plate}  UNKNOWN"

            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, label, (x1, max(20,y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        return frame
