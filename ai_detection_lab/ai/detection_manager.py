from __future__ import annotations
import math
import time
import threading
from collections import deque

from .yolo_detector import YOLODetector
from .tracker import IoUTracker
from .. import config
from ..capture.manager import CaptureManager


# ---------------------------------------------------------------------------
# Design principle (see project discussion): Confidence, Stability, Position
# Consistency, and Accuracy are FOUR SEPARATE things. A model that is
# consistently wrong will show high Confidence + high Stability, but that is
# NOT the same as being correct. This module therefore never labels anything
# "ACCURATE" — only SEARCHING / CANDIDATE / STABLE, and CONFIRMED / REJECTED
# once a human operator has actually looked at it via confirm_target() /
# reject_target(). Ground truth only exists once a human provides it.
# ---------------------------------------------------------------------------

STATUS_SEARCHING = "SEARCHING"   # nothing in frame right now
STATUS_CANDIDATE = "CANDIDATE"   # seen, but not yet consistent/confident enough
STATUS_STABLE = "STABLE"         # consistent across recent frames + position steady
STATUS_CONFIRMED = "CONFIRMED"   # operator confirmed it's real
STATUS_REJECTED = "REJECTED"     # operator rejected it (false positive)


class DetectionManager:
    def __init__(self, on_update, on_status):
        self.detector = YOLODetector()
        self.tracker = IoUTracker()
        self.capture = CaptureManager()
        self.on_update = on_update
        self.on_status = on_status
        self.lock = threading.Lock()

        self.current_count = 0
        self.maximum_count = 0
        self.total_frames = 0
        self.detection_frames = 0
        self.capture_count = 0
        self.unique_target_count = 0
        self.last_capture_time = 0.0
        self.unique_targets: list[dict] = []
        self.survey = {"waypoint": 0, "total_waypoints": 0, "x_cm": 0, "y_cm": 0, "heading": "EAST"}

        # Async AI worker:
        # keep only the newest camera frame; never build a backlog.
        self._frame_lock = threading.Lock()
        self._latest_frame = None
        self._frame_event = threading.Event()
        self._stop_event = threading.Event()
        self._worker = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="AI-LAB-DETECTION",
        )
        self._worker.start()

    def submit_frame(self, frame):
        """Submit only the newest frame for AI processing."""
        if self._stop_event.is_set() or frame is None:
            return
        with self._frame_lock:
            self._latest_frame = frame
        self._frame_event.set()

    def stop(self):
        self._stop_event.set()
        self._frame_event.set()
        if self._worker.is_alive():
            self._worker.join(timeout=2.0)

    def _worker_loop(self):
        while not self._stop_event.is_set():
            self._frame_event.wait(timeout=0.25)
            self._frame_event.clear()

            if self._stop_event.is_set():
                break

            with self._frame_lock:
                frame = self._latest_frame
                self._latest_frame = None

            if frame is None:
                continue

            try:
                self.process(frame)
            except Exception as exc:
                self.on_status(f"⚠️ Detection error: {exc}")

    def set_survey_position(self, waypoint: int, total_waypoints: int, x_cm: int, y_cm: int, heading: str):
        with self.lock:
            self.survey = {
                "waypoint": int(waypoint),
                "total_waypoints": int(total_waypoints),
                "x_cm": int(x_cm),
                "y_cm": int(y_cm),
                "heading": str(heading),
            }

    # ------------------------------------------------------------------
    # Main per-frame pipeline
    # ------------------------------------------------------------------
    def process(self, frame):
        results = self.detector.predict(frame)
        count = self.detector.count(results)
        detections = self.tracker.assign(self.detector.detections(results))
        annotated = self.detector.annotate(results, frame=frame)

        with self.lock:
            self.total_frames += 1
            self.current_count = count
            if count > 0:
                self.detection_frames += 1
            if count > self.maximum_count:
                self.maximum_count = count

            matched_ids: set[int] = set()
            new_targets: list[dict] = []
            now = time.monotonic()

            # 1) Match every detection in THIS frame to an existing target
            #    (or create a new CANDIDATE), and record a "hit".
            det_target_map: dict[int, dict] = {}  # id(det) -> target, for step 4 below
            for det in detections:
                target = self._find_existing_target(det)
                if target is None:
                    target = self._create_target(det)
                    new_targets.append(target)
                self._record_hit(target, det)
                matched_ids.add(target["id"])
                det_target_map[id(det)] = target

            # 2) Every OTHER known target that did not appear this frame is a
            #    "miss" for stability purposes (it doesn't disappear, it just
            #    stops accumulating hits — so a real object flying past still
            #    reads as stable while it's actually in view).
            for target in self.unique_targets:
                if target["id"] not in matched_ids:
                    self._record_miss(target)

            # 3) Recompute status for every target from its updated history.
            promoted_to_stable: list[dict] = []
            for target in self.unique_targets:
                was_decided = target["status"] in (STATUS_CONFIRMED, STATUS_REJECTED)
                was_stable = target["status"] == STATUS_STABLE
                self._update_status(target)
                if (
                    not was_decided
                    and not was_stable
                    and target["status"] == STATUS_STABLE
                ):
                    promoted_to_stable.append(target)

            self.unique_target_count = len(self.unique_targets)

            # 4) Auto-capture: once on first sighting (evidence), once more
            #    when a target first becomes STABLE (worth a human look).
            for target in new_targets:
                self._maybe_capture(target, annotated, count, detections, now, reason="first_detection")
            for target in promoted_to_stable:
                self._maybe_capture(target, annotated, count, detections, now, reason="stable_detection")

            # 5) Decide what the "current target" panel should show: the
            #    target behind the highest-confidence detection THIS frame.
            primary_target = None
            if detections:
                best_det = max(detections, key=lambda d: float(d.get("confidence", 0.0)))
                primary_target = det_target_map.get(id(best_det))

            if primary_target:
                stability_info = {
                    "percentage": round(primary_target["stability"] * 100.0, 1),
                    "detected_frames": primary_target["frames_seen"],
                    "window": primary_target["frames_window"],
                }
                status = primary_target["status"]
                primary_target_id = primary_target["id"]
            else:
                stability_info = {}
                status = STATUS_SEARCHING
                primary_target_id = None

            snapshot = {
                "current_count": self.current_count,
                "maximum_count": self.maximum_count,
                "unique_target_count": self.unique_target_count,
                "total_frames": self.total_frames,
                "detection_frames": self.detection_frames,
                "capture_count": self.capture_count,
                "detections": detections,
                "targets": [self._public_target(t) for t in self.unique_targets],
                "survey": dict(self.survey),
                "models": self.detector.model_status(),
                # Fields the "CURRENT TARGET" panel reads directly.
                "stability": stability_info,
                "status": status,
                "target_id": primary_target_id,
            }

        self.on_update(annotated, snapshot)
        return annotated, snapshot

    # ------------------------------------------------------------------
    # Target lifecycle helpers
    # ------------------------------------------------------------------
    def _create_target(self, det: dict) -> dict:
        target = {
            "id": len(self.unique_targets) + 1,
            "track_id": det.get("track_id"),
            "model_name": det["model_name"],
            "class_name": det["class_name"],
            "confidence": det["confidence"],
            "x_cm": self.survey["x_cm"],
            "y_cm": self.survey["y_cm"],
            "waypoint": self.survey["waypoint"],
            "heading": self.survey["heading"],
            "status": STATUS_CANDIDATE,
            "operator_decision": None,
            "reject_reason": None,
            # internal rolling history — not exposed directly to the UI
            "_hits": deque(maxlen=config.STABILITY_WINDOW_FRAMES),
            "_confidences": deque(maxlen=config.STABILITY_WINDOW_FRAMES),
            "_centers": deque(maxlen=config.STABILITY_WINDOW_FRAMES),
        }
        self.unique_targets.append(target)
        return target

    @staticmethod
    def _bbox_center(det: dict) -> tuple[float, float]:
        x1, y1, x2, y2 = det["xyxy"]
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def _record_hit(self, target: dict, det: dict):
        target["_hits"].append(True)
        target["_confidences"].append(float(det["confidence"]))
        target["_centers"].append(self._bbox_center(det))
        target["confidence"] = float(det["confidence"])  # latest single-frame confidence

    def _record_miss(self, target: dict):
        target["_hits"].append(False)

    def _update_status(self, target: dict):
        hits = target["_hits"]
        window = len(hits)
        seen = sum(1 for h in hits if h)
        stability = (seen / window) if window else 0.0

        confs = target["_confidences"]
        confidence_avg = (sum(confs) / len(confs)) if confs else target["confidence"]

        position_consistent = self._is_position_consistent(target["_centers"])

        target["stability"] = stability
        target["frames_seen"] = seen
        target["frames_window"] = window
        target["confidence_avg"] = confidence_avg
        target["position_consistent"] = position_consistent

        if target["operator_decision"] == "CONFIRMED":
            target["status"] = STATUS_CONFIRMED
            return
        if target["operator_decision"] == "REJECTED":
            target["status"] = STATUS_REJECTED
            return

        if (
            window >= config.STABILITY_MIN_FRAMES
            and stability >= config.STABILITY_MIN_RATIO
            and confidence_avg >= config.YOLO_CONF_THRES
            and position_consistent
        ):
            target["status"] = STATUS_STABLE
        else:
            target["status"] = STATUS_CANDIDATE

    @staticmethod
    def _is_position_consistent(centers) -> bool:
        pts = list(centers)
        if len(pts) < 2:
            return True  # not enough history yet to call it inconsistent
        max_jump = 0.0
        for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
            max_jump = max(max_jump, math.hypot(x2 - x1, y2 - y1))
        return max_jump <= config.POSITION_JUMP_MAX_PX

    def _public_target(self, target: dict) -> dict:
        """Sanitized copy for the UI/snapshot — no internal deques leak out."""
        return {
            "id": target["id"],
            "track_id": target.get("track_id"),
            "model_name": target["model_name"],
            "class_name": target["class_name"],
            "confidence": round(target["confidence"], 4),
            "confidence_avg": round(target.get("confidence_avg", target["confidence"]), 4),
            "stability": round(target.get("stability", 0.0), 4),
            "frames_seen": target.get("frames_seen", 0),
            "frames_window": target.get("frames_window", 0),
            "position_consistent": target.get("position_consistent", True),
            "x_cm": target["x_cm"],
            "y_cm": target["y_cm"],
            "waypoint": target["waypoint"],
            "heading": target["heading"],
            "status": target["status"],
            "operator_decision": target["operator_decision"],
            "reject_reason": target["reject_reason"],
        }

    def _find_existing_target(self, det: dict):
        """Which known target is this detection? Answered by object identity
        (the tracker's track_id) rather than by where the DRONE happens to
        be standing.

        The old rule - same class within UNIQUE_DETECTION_RADIUS_CM of the
        drone's own position - merged every same-class object seen from one
        spot into a single target, so three objects in one camera frame
        counted as one. Track IDs keep them apart. The radius rule is kept
        only as a fallback for detections the tracker didn't stamp.
        """
        track_id = det.get("track_id")
        if track_id is not None:
            for target in self.unique_targets:
                if target.get("track_id") == track_id and target["model_name"] == det["model_name"]:
                    return target
            return None

        name = det["class_name"]
        x = self.survey["x_cm"]
        y = self.survey["y_cm"]
        radius = float(config.UNIQUE_DETECTION_RADIUS_CM)
        for target in self.unique_targets:
            if target["class_name"] != name:
                continue
            if math.hypot(target["x_cm"] - x, target["y_cm"] - y) <= radius:
                return target
        return None

    def _get_target(self, target_id: int):
        for t in self.unique_targets:
            if t["id"] == target_id:
                return t
        return None

    def _maybe_capture(self, target, annotated, count, detections, now, reason: str):
        if not config.AUTO_CAPTURE:
            return
        if reason == "first_detection" and not config.CAPTURE_ON_FIRST_DETECTION:
            return
        if reason == "stable_detection" and not config.CAPTURE_ON_STABLE:
            return
        if now - self.last_capture_time < config.CAPTURE_COOLDOWN_SEC:
            return

        path = self.capture.save(
            annotated,
            reason,
            count,
            detections,
            survey={**self.survey, "target_id": target["id"], "status": target["status"]},
        )
        self.last_capture_time = now
        self.capture_count += 1

        icon = "🟢" if target["status"] == STATUS_STABLE else "🟡"
        self.on_status(
            f"{icon} TARGET #{target['id']} {target['status']} | "
            f"WP {target['waypoint']} | ({target['x_cm']},{target['y_cm']}) cm | "
            f"model={target['model_name']} | conf_avg={target.get('confidence_avg', target['confidence']):.0%} | "
            f"stability={target.get('stability', 0.0):.0%} | 📸 {path.name}"
        )
        if target["status"] == STATUS_STABLE:
            self.on_status(f"   ⚠ TARGET #{target['id']} AI RESULT REQUIRES HUMAN REVIEW")

    # ------------------------------------------------------------------
    # Operator confirmation API — this is the ONLY place "ground truth"
    # enters the system. Nothing else in this file may claim accuracy.
    # ------------------------------------------------------------------
    def confirm_target(self, target_id: int) -> bool:
        with self.lock:
            target = self._get_target(target_id)
            if target is None:
                return False
            target["operator_decision"] = "CONFIRMED"
            target["reject_reason"] = None
            target["status"] = STATUS_CONFIRMED
            record = {
                "target_id": target_id,
                "decision": "CONFIRMED",
                "reason": None,
                "class_name": target["class_name"],
                "model_name": target["model_name"],
                "confidence_avg": target.get("confidence_avg", target["confidence"]),
                "stability": target.get("stability", 0.0),
                "position": {"x_cm": target["x_cm"], "y_cm": target["y_cm"]},
            }
            self.on_status(f"🔵 TARGET #{target_id} CONFIRMED by operator")
        self.capture.log_decision(record)
        return True

    def reject_target(self, target_id: int, reason: str = "Wrong Object") -> bool:
        with self.lock:
            target = self._get_target(target_id)
            if target is None:
                return False
            target["operator_decision"] = "REJECTED"
            target["reject_reason"] = reason
            target["status"] = STATUS_REJECTED
            record = {
                "target_id": target_id,
                "decision": "REJECTED",
                "reason": reason,
                "class_name": target["class_name"],
                "model_name": target["model_name"],
                "confidence_avg": target.get("confidence_avg", target["confidence"]),
                "stability": target.get("stability", 0.0),
                "position": {"x_cm": target["x_cm"], "y_cm": target["y_cm"]},
            }
            self.on_status(f"❌ TARGET #{target_id} REJECTED by operator | reason={reason}")
        self.capture.log_decision(record)
        return True