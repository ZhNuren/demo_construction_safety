# AI Safety Vision — Tkinter Demo (Modular)

A modular Tkinter desktop app scaffold for computer-vision safety demos:
- Separate **tabs** per task (LPR, FaceID, Object Detection, Tracking, etc.)
- Reusable **VideoPlayer** with ROI drawing and an `on_frame` hook
- **Object Tracking** powered by Ultralytics YOLOv8n detections + a tiny IoU tracker

## Quickstart

```bash
pip install -r requirements.txt
python app.py
```

Open the **Object Tracking** tab, load a video, and click **Start tracking**.

## Structure

```
ai_safety_vision_demo/
├─ app.py                      # Main entry
├─ requirements.txt            # Minimal deps
├─ ui/
│  ├─ video_player.py          # Canvas + video loop + ROI drawing
│  ├─ toolbars.py              # Media + ROI toolbars
│  └─ style.py                 # Ttk styles and layout helpers
├─ tasks/
│  ├─ base.py                  # TaskPage base class
│  └─ object_tracking.py       # YOLOv8n + SimpleTracker integration
├─ models/
│  └─ detector.py              # YOLO wrapper (lazy init)
└─ tracking/
   └─ simple_tracker.py        # IoU-based multi-object tracker with trails
```

## Notes

- YOLO model downloads on first run. If you want another model (ByteTrack/DeepSORT),
  replace the tracker while keeping the `on_frame` hook.
- To filter classes (e.g., only person/animal), see `tasks/object_tracking.py`.
