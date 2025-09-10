import tkinter as tk
from tkinter import ttk

APP_TITLE = "AI Safety Vision — Demo Console"
APP_MIN_W, APP_MIN_H = 1200, 760

def configure_style(root: tk.Tk):
    style = ttk.Style()
    try:
        root.call("tk", "scaling", 1.25)
    except Exception:
        pass
    style.configure("H1.TLabel", font=("Helvetica", 18, "bold"))
    style.configure("Subtle.TLabel", foreground="#6b7280")
    style.configure("Status.TLabel", foreground="#94a3b8")
    style.configure("SidebarTitle.TLabel", font=("Helvetica", 12, "bold"))
    style.configure("Sidebar.TButton", padding=(10, 8))
