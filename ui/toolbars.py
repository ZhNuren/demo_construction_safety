import tkinter as tk
from tkinter import ttk, filedialog
from .video_player import VideoPlayer

class MediaToolbar(ttk.Frame):
    def __init__(self, master, player: VideoPlayer, *, allow_image=True, **kwargs):
        super().__init__(master, padding=(8,8), **kwargs)
        self.player = player

        # Put all buttons in one row frame with small padding
        row = ttk.Frame(self)
        row.pack(side=tk.LEFT)

        # Use shorter labels
        ttk.Button(row, text="Vid", width=4, command=self._open_video).pack(side=tk.LEFT, padx=2)
        if allow_image:
            ttk.Button(row, text="Img", width=4, command=self._open_image).pack(side=tk.LEFT, padx=2)

        ttk.Separator(row, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=4)

        ttk.Button(row, text="▶", width=3, command=self.player.resume).pack(side=tk.LEFT, padx=2)
        ttk.Button(row, text="⟳", width=3, command=self.player.restart).pack(side=tk.LEFT, padx=2)
        ttk.Button(row, text="⏸", width=3, command=self.player.pause).pack(side=tk.LEFT, padx=2)
        ttk.Button(row, text="✖", width=3, command=self.player.close).pack(side=tk.LEFT, padx=2)

    def _open_video(self):
        path = filedialog.askopenfilename(
            title="Select video",
            filetypes=[("Video files","*.mp4 *.avi *.mov *.mkv"), ("All files","*.*")]
        )
        if path: self.player.open(path)

    def _open_image(self):
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Images","*.jpg *.jpeg *.png *.bmp"), ("All files","*.*")]
        )
        if path: self.player.show_image(path)


class ROIControls(ttk.Frame):
    def __init__(self, master, player: VideoPlayer, **kwargs):
        super().__init__(master, padding=(8,8), **kwargs)
        self.player = player
        ttk.Label(self, text="Zone / ROI:").pack(side=tk.LEFT)
        ttk.Button(self, text="Draw", command=self.player.enable_roi_draw).pack(side=tk.LEFT, padx=(8,0))
        ttk.Button(self, text="Clear", command=self._clear).pack(side=tk.LEFT, padx=(6,0))

    def _clear(self):
        self.player.clear_roi()
