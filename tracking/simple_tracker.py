from collections import deque

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter + 1e-6
    return inter / union

class SimpleTracker:
    def __init__(self, max_lost=30, iou_thr=0.3, trail=30):
        self.max_lost = max_lost
        self.iou_thr = iou_thr
        self.trail = trail
        self.next_id = 1
        self.tracks: dict[int, dict] = {}

    def update(self, detections, scores):
        unmatched_tracks = set(self.tracks.keys())
        matches = []
        for det in detections:
            best_iou, best_id = 0, None
            for tid in list(unmatched_tracks):
                i = iou_xyxy(self.tracks[tid]['bbox'], det)
                if i > best_iou:
                    best_iou, best_id = i, tid
            if best_id is not None and best_iou >= self.iou_thr:
                matches.append((best_id, det))
                unmatched_tracks.remove(best_id)
            else:
                tid = self.next_id
                self.next_id += 1
                cx = (det[0]+det[2])//2
                cy = (det[1]+det[3])//2
                self.tracks[tid] = {
                    'bbox': det,
                    'lost': 0,
                    'trail': deque([(cx, cy)], maxlen=self.trail)
                }
        for tid, det in matches:
            cx = (det[0]+det[2])//2
            cy = (det[1]+det[3])//2
            tr = self.tracks[tid]
            tr['bbox'] = det
            tr['lost'] = 0
            tr['trail'].append((cx, cy))
        for tid in list(unmatched_tracks):
            self.tracks[tid]['lost'] += 1
            if self.tracks[tid]['lost'] > self.max_lost:
                del self.tracks[tid]
        return self.tracks
