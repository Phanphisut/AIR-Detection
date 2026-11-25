from __future__ import annotations
import threading
import time

from .planner import generate_waypoint_objects
from .. import config
from ..drone.controller import HulaDroneController


# Same cyclic compass order the planner uses for its own turn-diff math.
_COMPASS_ORDER = ["EAST", "SOUTH", "WEST", "NORTH"]


class AutonomousGridMission(threading.Thread):
    """Lawnmower survey flown with a FIXED heading the whole time.

    The drone/camera never rotates during the mission. Row traversal and
    row-to-row spacing are both flown as body-relative strafes:

        Takeoff -> Survey Altitude (once) -> Hold ->
        Strafe(F/B/L/R) -> Detect -> Strafe(F/B/L/R) -> Detect -> ... -> Land

    Each planner waypoint's compass heading is translated into a
    forward/back/left/right move relative to the heading the drone launched
    with — no single_fly_turnleft/turnright/turn_to call is ever issued.
    """

    def __init__(self, drone: HulaDroneController, on_status, on_progress, on_finished, on_detect=None):
        super().__init__(daemon=True, name="AI-LAB-GRID-FLIGHT")
        self.drone = drone
        self.on_status = on_status
        self.on_progress = on_progress
        self.on_finished = on_finished
        # Called after each successful move, before the next one. No-op by default.
        self.on_detect = on_detect or (lambda wp: None)
        self.stop_event = threading.Event()
        # Fixed at launch — everything is measured relative to this, never
        # to the drone's live self.drone.heading (which no longer changes).
        self._launch_heading = config.INITIAL_HEADING

    def request_stop(self):
        self.stop_event.set()
        self.drone.emergency_stop_requested = True

    def _relative_move(self, desired_heading: str, distance: int) -> bool:
        """Translate a planner compass heading into a body-relative strafe.
        The drone body (and camera) is never rotated to face `desired_heading`."""
        ci = _COMPASS_ORDER.index(self._launch_heading)
        di = _COMPASS_ORDER.index(desired_heading)
        diff = (di - ci) % 4
        if diff == 0:
            return self.drone.move_forward(distance)
        if diff == 1:
            return self.drone.move_right(distance)
        if diff == 2:
            return self.drone.move_back(distance)
        return self.drone.move_left(distance)  # diff == 3

    def run(self):
        success = False
        reason = "unknown"
        try:
            if not config.AUTONOMOUS_FLIGHT_ENABLED or not config.TAKEOFF_ENABLED:
                reason = "autonomous flight is disabled in config"
                self.on_status(f"🔒 GRID BLOCKED: {reason}")
                return
            if not self.drone.connected:
                reason = "drone is not connected"
                self.on_status("❌ GRID BLOCKED: drone is not connected")
                return
            if not self.drone.camera_down_ok:
                reason = "camera-down preflight check failed"
                self.on_status("❌ GRID BLOCKED: camera must be DOWN 90°")
                return

            points = generate_waypoint_objects()
            if not points:
                reason = "no survey waypoints generated"
                self.on_status(f"❌ GRID BLOCKED: {reason}")
                return

            self.drone.autonomous_running = True
            self.drone.emergency_stop_requested = False
            self.on_status(f"🗺️ AUTONOMOUS GRID: {len(points)} survey positions")
            self.on_status(
                f"📐 FIELD {config.FIELD_WIDTH_CM}×{config.FIELD_HEIGHT_CM} cm | "
                f"SPACING {config.GRID_SPACING_CM} cm"
            )
            self.on_status(f"🧭 Flight mode: FIXED HEADING ({self._launch_heading}) — strafe only, no rotation")

            # Takeoff -> climb to survey altitude once -> hold (guarded
            # inside controller.takeoff() against repeat climbs).
            if not self.drone.takeoff():
                reason = "takeoff failed"
                return

            first = points[0]
            self.on_progress(first.index, len(points), first.x_cm, first.y_cm, first.heading, first.turn)
            self.on_status(f"🔍 DETECT @ WP {first.index}")
            self.on_detect(first)
            time.sleep(config.GRID_WAYPOINT_PAUSE_SEC)

            previous = first
            for wp in points[1:]:
                if self.stop_event.is_set() or self.drone.emergency_stop_requested:
                    reason = "emergency stop requested"
                    self.on_status("🛑 GRID STOPPED")
                    return

                px, py = previous.x_cm, previous.y_cm
                distance = abs(wp.x_cm - px) + abs(wp.y_cm - py)
                self.on_status(
                    f"📍 WP {wp.index}/{len(points)} | POS=({wp.x_cm},{wp.y_cm}) cm | "
                    f"COMPASS={wp.heading} (body-relative) | MOVE {distance} cm"
                )

                if distance and not self._relative_move(wp.heading, distance):
                    reason = "lateral/forward movement failed"
                    return

                if self.stop_event.is_set() or self.drone.emergency_stop_requested:
                    reason = "emergency stop requested"
                    self.on_status("🛑 GRID STOPPED")
                    return

                # Detect right after arriving at each waypoint.
                self.on_status(f"🔍 DETECT @ WP {wp.index}")
                self.on_detect(wp)

                self.on_progress(wp.index, len(points), wp.x_cm, wp.y_cm, wp.heading, wp.turn)
                previous = wp
                time.sleep(config.GRID_WAYPOINT_PAUSE_SEC)

            success = True
            reason = "completed"
            self.on_status("🏁 SURVEY PLAN COMPLETE — all waypoints scanned")

        except Exception as exc:
            reason = str(exc)
            self.on_status(f"❌ GRID ERROR: {exc}")
        finally:
            self.drone.autonomous_running = False
            if success and config.AUTO_LAND_ON_COMPLETE and not self.drone.emergency_stop_requested:
                self.drone.touchdown()
            self.on_finished(success, reason)