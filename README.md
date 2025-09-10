# AI Safety Vision — Tkinter Demo (Modular)

A modular Tkinter desktop app scaffold for computer-vision safety demos:
- Separate **tabs** per task (LPR, FaceID, Object Detection, Tracking, etc.)
- Reusable **VideoPlayer** with ROI drawing and an `on_frame` hook
- **Object Tracking** powered by Ultralytics YOLOv8n + a tiny IoU tracker
- **Face ID** enrollment (image + label + "Not allowed") and live recognition via InsightFace

## Quickstart

```bash
pip install -r requirements.txt
python app.py
```

Open **Face ID** tab to enroll and recognize.  
Open **Object Tracking** to see multi-object tracking with trails.

## Structure

```
ai_safety_vision_demo/
├─ app.py
├─ requirements.txt
├─ README.md
├─ ui/
│  ├─ video_player.py
│  ├─ toolbars.py
│  └─ style.py
├─ tasks/
│  ├─ base.py
│  ├─ object_tracking.py
│  └─ face_id.py
├─ models/
│  ├─ detector.py           # YOLO
│  └─ faceid.py             # InsightFace enroll + identify
└─ tracking/
   └─ simple_tracker.py
```

## Notes

- YOLO + InsightFace download models on first run.
- Face DB is a simple `faces_db.json` next to `app.py`. Delete it to reset.
- You can swap the IoU tracker with ByteTrack/DeepSORT later without changing the UI hooks.
