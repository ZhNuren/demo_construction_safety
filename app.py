import tkinter as tk
from tkinter import ttk, messagebox
from ui.style import configure_style, APP_TITLE, APP_MIN_W, APP_MIN_H
from tasks.base import TaskPage
from tasks.object_tracking import ObjectTrackingPage
from tasks.face_id import FaceIDPage
from tasks.lpr import LPRPage
from tasks.crowd import CrowdPage
from tasks.ppe import PPEPage
from tasks.smoke_fire import SmokeFirePage
from tasks.no_plate import NoPlatePage

TASKS = [
    ("LPR", "License Plate Recognition"),
    ("Face ID", "Face Identification"),
    ("Object Detection", "General Object Detection"),
    ("Object Tracking", "Tracking with trails"),
    ("Behavior/Intrusion", "Behavior Analytics + Intrusion"),
    ("Crowd Counting", "Crowd Density"),
    ("PPE", "Personal Protective Equipment"),
    ("Load Lifting", "Unsafe lifting detection"),
    ("Tricycle Plates", "No-plate vehicle detection"),
    ("Fire/Smoke", "Fire & Smoke Detection"),
]

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.minsize(APP_MIN_W, APP_MIN_H)
        configure_style(self)

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        self.sidebar = self._build_sidebar(self)
        self.sidebar.grid(row=0, column=0, sticky="nsw")

        self.content = ttk.Frame(self)
        self.content.grid(row=0, column=1, sticky="nsew")
        self.content.grid_rowconfigure(1, weight=1)
        self.content.grid_columnconfigure(0, weight=1)

        self._build_header(self.content).grid(row=0, column=0, sticky="ew")

        self.tabs = ttk.Notebook(self.content)
        self.tabs.grid(row=1, column=0, sticky="nsew")

        self.tabs.bind("<<NotebookTabChanged>>", self._on_tab_change)

        self.pages = {}
        self._build_footer(self.content).grid(row=2, column=0, sticky="ew")

    def _build_sidebar(self, parent):
        frame = ttk.Frame(parent, padding=(12,16))
        ttk.Label(frame, text="Tasks", style="SidebarTitle.TLabel").pack(anchor="w", pady=(0,8))
        for key, title in TASKS:
            ttk.Button(
                frame, text=f"{key}", style="Sidebar.TButton",
                command=lambda k=key, t=title: self.open_task(k, t)
            ).pack(fill="x", pady=4)
        ttk.Separator(frame).pack(fill="x", pady=12)
        ttk.Button(frame, text="Settings", style="Sidebar.TButton", command=self._open_settings).pack(fill="x")
        return frame

    def _build_header(self, parent):
        frame = ttk.Frame(parent, padding=(16,12))
        ttk.Label(frame, text=APP_TITLE, style="H1.TLabel").pack(side="left")
        ttk.Label(frame, text="Modular demo — plug in your models", style="Subtle.TLabel").pack(side="left", padx=(12,0))
        return frame

    def _build_footer(self, parent):
        frame = ttk.Frame(parent)
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(frame, textvariable=self.status_var, anchor="w", style="Status.TLabel").pack(side="left", fill="x", expand=True, padx=12, pady=6)
        ttk.Button(frame, text="About", command=self._about).pack(side="right", padx=12)
        return frame

    def open_task(self, key: str, title: str):
        if key not in self.pages:
            if key == "Object Tracking":
                page = ObjectTrackingPage(self.tabs)
            elif key == "Face ID":
                page = FaceIDPage(self.tabs)
            elif key == "LPR":
                page = LPRPage(self.tabs)
            elif key == "Crowd Counting":
                page = CrowdPage(self.tabs)
            elif key == "PPE":
                page = PPEPage(self.tabs)
            elif key == "Fire/Smoke":
                page = SmokeFirePage(self.tabs)
            elif key == "Tricycle Plates":
                page = NoPlatePage(self.tabs)
            else:
                page = TaskPage(self.tabs, key, title)
            self.pages[key] = page
            self.tabs.add(page, text=key)
        idx = list(self.pages.keys()).index(key)
        self.tabs.select(idx)
        self.status_var.set(f"Opened: {title}")

    def _open_settings(self):
        messagebox.showinfo("Settings", "Global settings dialog can go here.")

    def _about(self):
        messagebox.showinfo("About", "AI Safety Vision Demo (modular)\nUse Face ID and Object Tracking tabs.")

    def _on_tab_change(self, event):
        # Keep only the selected tab active; close sources in others.
        current_tab_id = self.tabs.select()
        for page in self.pages.values():
            if str(page) != current_tab_id:
                # This tab is not selected → ensure streams are closed
                if hasattr(page, "cleanup"):
                    page.cleanup()
def main():
    app = App()
    app.mainloop()

if __name__ == "__main__":
    main()
