from __future__ import annotations

from .. import config


def iou(box_a, box_b) -> float:
    """Intersection-over-union of two xyxy boxes. 0.0 when they don't overlap."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


class IoUTracker:
    """Gives every detected object a track_id that survives across frames.

    Why this and not Ultralytics' own model.track(): its trackers import
    `lap` at module import time, and when that package is missing they call
    check_requirements(), which pip-installs at runtime. This app ships as a
    self-contained folder that is explicitly meant to need no install and no
    network (see RUN_AI_Detection_Lab.command), and `lap` is not in the
    bundled venv - so model.track() would break on a student's Mac the first
    time a detection happened offline. Greedy IoU matching needs no
    dependency at all, and at this app's scale (a handful of objects per
    frame, one field) it is entirely adequate.

    Matching is scoped per (model_name, class_name): two different models,
    or two different classes, never share a track, so IDs can't collide
    across the models loaded from MODEL_DIR.
    """

    def __init__(self):
        self._tracks: list[dict] = []
        self._frame_no = 0
        self._next_id = 1

    def assign(self, detections: list[dict]) -> list[dict]:
        """Stamps a "track_id" into each detection dict, in place."""
        self._frame_no += 1

        # Score every plausible (detection, track) pairing first, then take
        # them best-first, so a detection that overlaps two tracks goes to
        # the one it actually matches better instead of whichever happened
        # to be created first.
        pairs = []
        for det_idx, det in enumerate(detections):
            for track_idx, track in enumerate(self._tracks):
                if track["model_name"] != det["model_name"]:
                    continue
                if track["class_name"] != det["class_name"]:
                    continue
                score = iou(det["xyxy"], track["box"])
                if score >= config.TRACK_IOU_MATCH_THRES:
                    pairs.append((score, det_idx, track_idx))
        pairs.sort(key=lambda p: p[0], reverse=True)

        matched_dets: set[int] = set()
        matched_tracks: set[int] = set()
        for _score, det_idx, track_idx in pairs:
            if det_idx in matched_dets or track_idx in matched_tracks:
                continue
            matched_dets.add(det_idx)
            matched_tracks.add(track_idx)
            track = self._tracks[track_idx]
            track["box"] = list(detections[det_idx]["xyxy"])
            track["last_frame"] = self._frame_no
            detections[det_idx]["track_id"] = track["track_id"]

        for det_idx, det in enumerate(detections):
            if det_idx in matched_dets:
                continue
            det["track_id"] = self._next_id
            self._tracks.append(
                {
                    "track_id": self._next_id,
                    "model_name": det["model_name"],
                    "class_name": det["class_name"],
                    "box": list(det["xyxy"]),
                    "last_frame": self._frame_no,
                }
            )
            self._next_id += 1

        # Forget tracks that haven't been seen for a while. IDs are never
        # reused, so an object that comes back after the gap is reported as
        # a genuinely new sighting rather than silently inheriting the old
        # object's identity.
        max_age = config.TRACK_MAX_AGE_FRAMES
        self._tracks = [t for t in self._tracks if self._frame_no - t["last_frame"] <= max_age]
        return detections

    @property
    def active_track_count(self) -> int:
        return len(self._tracks)
