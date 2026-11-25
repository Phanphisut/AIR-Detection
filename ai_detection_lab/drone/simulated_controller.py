from __future__ import annotations
import threading
import time

import cv2

from .. import config
from . import alerts, limits

_COMPASS_ORDER = ["EAST", "SOUTH", "WEST", "NORTH"]


class _SimulatedFrame:
    __slots__ = ("image",)

    def __init__(self, image):
        self.image = image


class _SimulatedVideoStream:
    """Stands in for pyhulax's video_stream under Simulation Mode.

    Opens the laptop's own webcam (same device/backend as the existing
    Webcam Test feature) so YOLO detection, capture, and the post-flight
    report all get exercised against a REAL image feed instead of a blank
    one - only the drone's flight itself is simulated here, not the vision
    pipeline. CameraWorker only ever touches `.latest_frame.image`,
    `.start()` and `.stop()`, so this only needs to satisfy that surface.
    """

    def __init__(self, camera_index: int):
        self._camera_index = camera_index
        self._cap = None
        self._thread = None
        self._stop_event = threading.Event()
        self.latest_frame = None

    def start(self, blocking: bool = False):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        self._cap = cv2.VideoCapture(self._camera_index, config.webcam_capture_backend())
        while not self._stop_event.is_set():
            if not self._cap.isOpened():
                time.sleep(0.2)
                continue
            ok, frame = self._cap.read()
            if ok and frame is not None:
                self.latest_frame = _SimulatedFrame(frame)
            time.sleep(0.01)

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class _SimulatedAPI:
    """Just enough of pyhulax.DroneAPI's surface for main_window.py's
    direct `self.drone.api.get_battery()` battery poll to keep working."""

    def __init__(self, controller: "SimulatedDroneController"):
        self._controller = controller

    def get_battery(self) -> int:
        return self._controller.get_battery_pct()


class SimulatedDroneController:
    """Drop-in stand-in for HulaDroneController with no real drone/Wi-Fi.

    Every call succeeds after a short realistic delay and just updates
    in-memory state, so Manual Flight, the Autonomous Grid mission and
    the post-flight report can all be exercised
    without physical hardware. main_window.py's "🧪 Simulation Mode"
    toggle swaps this in for HulaDroneController before anything connects.

    Implements the exact same public surface HulaDroneController exposes
    (attributes and methods) since every caller — CameraWorker,
    AutonomousGridMission, ManualFlightDialog — takes
    whichever controller instance it's handed and calls that surface
    without knowing which implementation it is.
    """

    def __init__(self):
        self.api = _SimulatedAPI(self)
        self.connected = False
        self.camera_down_requested = False
        self.camera_down_ok = False
        self.stream_started = False
        self.video_stream = None
        self.flying = False
        self.autonomous_running = False
        self.emergency_stop_requested = False
        self.heading = config.INITIAL_HEADING
        self.at_survey_altitude = False
        self.manual_flight_used = False
        # Battery is derived from accumulated simulated flight time rather
        # than a plain countdown, so repeated get_battery() polls (every 5s
        # from main_window's battery watchdog) agree with each other and
        # with whether the drone is currently flying.
        self._accumulated_flight_sec = 0.0
        self._flight_started_at = None
        self.last_error = ""

    # Same checks + messages as the real HulaDroneController, so Simulation
    # Mode fails exactly where a real drone would (it used to accept 90.0
    # degrees etc. that the real firmware rejects).
    def _fail(self, message: str) -> bool:
        self.last_error = str(message)
        print(f"❌ [SIM] COMMAND: {self.last_error}")
        return False

    def _can_fly_command(self) -> bool:
        if self.emergency_stop_requested:
            return self._fail("Emergency stop is on — take off again first")
        if not self.connected:
            return self._fail("The drone is not connected")
        if not self.flying:
            return self._fail("The drone isn't flying — take off first")
        self.last_error = ""
        return True

    def telemetry_age_sec(self):
        return 0.0 if self.connected else None

    reconnecting = False

    def reconnect(self) -> bool:
        return self.connected

    def disconnect(self) -> None:
        self.stop_stream()
        self.connected = False

    def battery_pct(self):
        return self.get_battery_pct()

    def get_battery_pct(self) -> int:
        elapsed = self._accumulated_flight_sec
        if self.flying and self._flight_started_at is not None:
            elapsed += time.monotonic() - self._flight_started_at
        drain_sec_per_pct = max(0.1, config.SIMULATION_BATTERY_DRAIN_SEC_PER_PCT)
        return max(0, 100 - int(elapsed / drain_sec_per_pct))

    def connect(self, retries: int = 5, retry_delay: float = 1.0) -> bool:
        self.camera_down_ok = False
        self.stream_started = False
        self.manual_flight_used = False
        time.sleep(0.3)
        self.connected = True
        print("✅ [SIM] Drone connected (simulated — no real drone)")
        return True

    def set_camera_down(self) -> bool:
        if not config.CAMERA_AUTO_DOWN:
            self.camera_down_requested = False
            self.camera_down_ok = True
            return True
        if not self.connected:
            print("❌ [SIM] CAMERA ANGLE ERROR: drone is not connected")
            return False
        self.camera_down_requested = True
        self.camera_down_ok = True
        print(f"📷 [SIM] CAMERA DOWN → {config.CAMERA_DOWN_ANGLE}°")
        return True

    def start_stream(self) -> bool:
        if not self.connected:
            return False
        try:
            self.video_stream = _SimulatedVideoStream(config.SIMULATION_CAMERA_INDEX)
            self.video_stream.start(blocking=False)
            self.stream_started = True
            print("🎥 [SIM] VIDEO STREAM ACTIVE (laptop webcam standing in for the drone camera)")
            return True
        except Exception as exc:
            self.stream_started = False
            self.video_stream = None
            print(f"❌ [SIM] VIDEO START ERROR: {exc}")
            return False

    def stop_stream(self):
        if self.video_stream is not None:
            self.video_stream.stop()
        self.stream_started = False
        self.video_stream = None

    def takeoff(self, require_camera_down: bool = True) -> bool:
        self.last_error = ""
        if not config.AUTONOMOUS_FLIGHT_ENABLED or not config.TAKEOFF_ENABLED:
            print("🔒 [SIM] TAKEOFF BLOCKED: autonomous flight is disabled")
            self.last_error = "Flying is turned off in config.py (TAKEOFF_ENABLED / AUTONOMOUS_FLIGHT_ENABLED)"
            return False
        if not self.connected or (require_camera_down and not self.camera_down_ok):
            print("🔒 [SIM] TAKEOFF BLOCKED: preflight failed")
            self.last_error = "The drone is not connected" if not self.connected else "The camera isn't pointing down 90° yet"
            return False
        if self.flying:
            return True
        low = alerts.takeoff_block_reason(self.get_battery_pct())
        if low:
            print(f"🔒 [SIM] TAKEOFF BLOCKED: {low}")
            return self._fail(low)
        time.sleep(0.5)
        self.flying = True
        self.emergency_stop_requested = False
        self.heading = config.INITIAL_HEADING
        self.at_survey_altitude = True
        self._flight_started_at = time.monotonic()
        print(f"🛫 [SIM] TAKEOFF OK — survey altitude {config.SURVEY_ALTITUDE_CM} cm")
        return True

    def move_forward(self, distance_cm: int) -> bool:
        return self._move("FORWARD", distance_cm)

    def move_left(self, distance_cm: int) -> bool:
        return self._move("LEFT", distance_cm)

    def move_right(self, distance_cm: int) -> bool:
        return self._move("RIGHT", distance_cm)

    def move_back(self, distance_cm: int) -> bool:
        return self._move("BACK", distance_cm)

    def _move(self, label: str, distance_cm: int) -> bool:
        if self.emergency_stop_requested or not self.flying:
            return False
        time.sleep(min(1.0, max(0.05, distance_cm / 200)))
        print(f"   ↳ [SIM] {label:<8} +{distance_cm} cm")
        return True

    def manual_move(self, direction, distance_cm: int) -> bool:
        if not self._can_fly_command():
            return False
        try:
            dist = limits.move_cm(distance_cm)
        except limits.CommandValueError as exc:
            return self._fail(str(exc))
        self.manual_flight_used = True
        time.sleep(0.1)
        print(f"   ↳ [SIM MANUAL] {getattr(direction, 'name', direction)} {dist}cm")
        return True

    def manual_rotate(self, degrees: float) -> bool:
        if not self._can_fly_command():
            return False
        try:
            size = limits.rotate_deg(degrees)
        except limits.CommandValueError as exc:
            return self._fail(str(exc))
        self.manual_flight_used = True
        time.sleep(0.1)
        print(f"   ↳ [SIM MANUAL] ROTATE {size if float(degrees) >= 0 else -size}°")
        return True

    def turn_right_90(self) -> bool:
        return self._turn("RIGHT")

    def turn_left_90(self) -> bool:
        return self._turn("LEFT")

    def _turn(self, direction: str) -> bool:
        if self.emergency_stop_requested or not self.flying:
            return False
        time.sleep(0.3)
        self.heading = _next_heading(self.heading, direction)
        print(f"   ↳ [SIM] TURN {direction:<5} 90°")
        return True

    def turn_to(self, desired_heading: str) -> bool:
        if desired_heading == self.heading:
            return True
        ci, di = _COMPASS_ORDER.index(self.heading), _COMPASS_ORDER.index(desired_heading)
        diff = (di - ci) % 4
        if diff == 1:
            return self.turn_right_90()
        if diff == 3:
            return self.turn_left_90()
        if diff == 2:
            return self.turn_right_90() and self.turn_right_90()
        return True

    def touchdown(self) -> bool:
        if not self.connected or not self.flying:
            return True
        time.sleep(0.5)
        if self._flight_started_at is not None:
            self._accumulated_flight_sec += time.monotonic() - self._flight_started_at
            self._flight_started_at = None
        self.flying = False
        self.at_survey_altitude = False
        print("🛬 [SIM] TOUCHDOWN")
        return True

    def flip(self, direction: str) -> bool:
        if not self._can_fly_command():
            return False
        if str(direction).lower() not in ("forward", "back", "left", "right"):
            return self._fail('flip must be "forward", "back", "left" or "right"')
        self.manual_flight_used = True
        time.sleep(0.3)
        print(f"   ↳ [SIM] FLIP {str(direction).upper()}")
        return True

    def set_led(self, r: int, g: int, b: int) -> bool:
        if not self.connected:
            return self._fail("The drone is not connected")
        self.led = tuple(max(0, min(255, int(round(float(v))))) for v in (r, g, b))
        print(f"   ↳ [SIM] LED {self.led}")
        return True

    def set_camera_pitch(self, degrees_down: int) -> bool:
        if not self.connected:
            return self._fail("The drone is not connected")
        self.camera_pitch = max(0, min(90, int(round(float(degrees_down)))))
        self.camera_down_ok = self.camera_pitch == int(config.CAMERA_DOWN_ANGLE)
        print(f"   ↳ [SIM] CAMERA PITCH {self.camera_pitch}° down")
        return True

    def emergency_stop(self):
        self.emergency_stop_requested = True
        self.autonomous_running = False
        print("🛑 [SIM] EMERGENCY STOP REQUESTED")
        self.touchdown()
        self.stop_stream()


def _next_heading(current: str, direction: str) -> str:
    i = _COMPASS_ORDER.index(current)
    return _COMPASS_ORDER[(i + (1 if direction == "RIGHT" else -1)) % 4]
