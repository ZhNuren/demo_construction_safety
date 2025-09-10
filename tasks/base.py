import tkinter as tk
from tkinter import ttk
from ui.video_player import VideoPlayer
from ui.toolbars import MediaToolbar, ROIControls

class TaskPage(ttk.Frame):
    def __init__(self, master, task_key: str, task_title: str, **kwargs):
        super().__init__(master, padding=0, **kwargs)
        self.task_key = task_key
        self.task_title = task_title

        self.toolbar = ttk.Frame(self, padding=(10,10))
        self.toolbar.pack(fill=tk.X)

        self.player = VideoPlayer(self)
        self.player.pack(fill=tk.BOTH, expand=True)

        self.status = ttk.Label(self, text=f"Ready — {task_title}", anchor=tk.W)
        self.status.pack(fill=tk.X)

        self._build_controls()
        self.after(200, self._poll_roi)

    def _poll_roi(self):
        self.after(200, self._poll_roi)

    def _build_controls(self):
        MediaToolbar(self.toolbar, self.player, allow_image=True).pack(side=tk.LEFT)
        ttk.Separator(self.toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=12)
        if self.task_key in {"Behavior/Intrusion", "Object Tracking", "Object Detection", "Crowd Counting", "PPE", "Load Lifting", "Fire/Smoke"}:
            ROIControls(self.toolbar, self.player).pack(side=tk.LEFT)

    def notify(self, msg: str):
        self.status.configure(text=f"{self.task_title}: {msg}")
    
    def cleanup(self):
        """Called when the tab is hidden or app closes."""
        try:
            self.player.close()
        except Exception:
            pass
