from __future__ import annotations
from datetime import datetime
from pathlib import Path
import json

from .. import config
from ..image_io import imwrite_any


class CaptureManager:
    def __init__(self):
        self.mission_dir = config.CAPTURE_DIR / datetime.now().strftime("mission_%Y%m%d_%H%M%S")
        self.mission_dir.mkdir(parents=True, exist_ok=True)
        self.index = 0
        self.records = self.mission_dir / "detections.json"
        self.decisions = self.mission_dir / "decisions.json"

    def log_decision(self, record: dict) -> Path:
        """Append an operator CONFIRM/REJECT decision to decisions.json.
        This is the human ground-truth log — separate from the raw AI
        detections.json, since Stability/Confidence != Accuracy."""
        row = dict(record)
        row["timestamp"] = datetime.now().isoformat(timespec="seconds")
        rows = []
        if self.decisions.exists():
            try:
                rows = json.loads(self.decisions.read_text(encoding="utf-8"))
            except Exception:
                rows = []
        rows.append(row)
        self.decisions.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        return self.decisions

    def save(self, frame, reason: str, count: int, detections: list[dict], survey: dict | None = None) -> Path:
        self.index += 1
        safe_reason = "".join(c if c.isalnum() or c in "-_" else "_" for c in reason)
        path = self.mission_dir / f"{self.index:03d}_{safe_reason}_count{count}.jpg"
        imwrite_any(path, frame)

        record = {
            "id": self.index,
            "image": path.name,
            "reason": reason,
            "count_in_frame": count,
            "detections": detections,
            "survey_position": survey or {},
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        rows = []
        if self.records.exists():
            try:
                rows = json.loads(self.records.read_text(encoding="utf-8"))
            except Exception:
                rows = []
        rows.append(record)
        self.records.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        return path