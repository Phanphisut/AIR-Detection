from __future__ import annotations
import os
import threading
import time
import pyhulax
import pyhulax.fylo.config as _pyhulax_config
import pyhulax.fylo.controlserver as _pyhulax_server
from pyhulax.core.types import CameraPitchMode, CommandResult, Direction, FlipDirection, LEDMode, VelocityLevel
from pyhulax.system.datacenter import DataCenter as _DataCenter
from pyhulax.video import VideoStream

from .. import config
from . import alerts, discovery, limits

# drone_status in REPORT_STATS (seen on a real F09-lite, 2026-10-07).
# 0 flickers with 2 while the drone sits ready on the floor.
FLIGHT_STATES = {
    2: ("ground", "ready", "ready"),
    3: ("takeoff", "taking off", "taking off"),
    1: ("flying", "flying", "flying"),
    4: ("landing", "landing", "landing"),
}
# How long a status-2 reading still counts as "ready" (the 2 -> 0 -> 2 flicker
# lasts 1-2 s on the real drone).
READY_WINDOW_SEC = 3.0


class DroneNotReadyAbort(RuntimeError):
    """pyhulax wanted to os._exit(0) the whole app because the drone was not ready."""


class _NoExitOs:
    """Stand-in for the `os` module inside pyhulax.fylo.controlserver.

    single_fly_takeoff() calls os._exit(0) - killing the app - if the drone's
    latest heartbeat is not status 2. Our own check runs first, but the real
    drone's status flickers 2 -> 0 every second or two, so the two reads can
    disagree. This turns that exit into an exception the controller catches."""

    def __getattr__(self, name):
        return getattr(os, name)

    @staticmethod
    def _exit(code=0):
        raise DroneNotReadyAbort("The drone isn't ready to take off (its state changed while sending)")


_pyhulax_server.os = _NoExitOs()


# v9.0.0: when did the drone last send anything? pyhulax only keeps the latest
# value of each message (get_battery() answers the old % forever after the
# drone drops out), so stamp every write to its DataCenter.
_last_telemetry_at = 0.0
_orig_set_data = _DataCenter.set_data


def _stamped_set_data(self, device, datatype, value, id=0):
    global _last_telemetry_at
    if device == "Plane":
        _last_telemetry_at = time.monotonic()
    return _orig_set_data(self, device, datatype, value, id)


if not getattr(_DataCenter.set_data, "_aidl_stamped", False):
    _stamped_set_data._aidl_stamped = True
    _DataCenter.set_data = _stamped_set_data


class HulaDroneController:
    """AI-Lab-only Hula controller; the game has a completely separate API instance.

    Talks to the drone through ``pyhulax`` (a clean-room, cross-platform
    MAVLink client for the same HG-Fly F09-lite/Hula hardware) instead of the
    Windows-only ``pyhula`` SDK, so this runs on macOS/Linux too.
    """

    def __init__(self):
        self.api = self._new_api()
        self._lock = threading.RLock()
        self.connected = False
        self.camera_down_requested = False
        self.camera_down_ok = False
        self.stream_started = False
        self.video_stream = None
        self.flying = False
        self.autonomous_running = False
        self.emergency_stop_requested = False
        self.heading = config.INITIAL_HEADING
        # Set True as soon as takeoff() reaches survey altitude (pyhulax's
        # takeoff() climbs straight to a target height in one blocking call,
        # so there is no separate mid-flight climb step left to guard here).
        self.at_survey_altitude = False
        # Set once any Manual Flight (keyboard, Developer Mode) command runs.
        # Manual rotate() turns by arbitrary angles, which desyncs the
        # EAST/SOUTH/WEST/NORTH heading tracking that turn_to() and the
        # autonomous grid mission rely on (they only ever expect
        # 90 degree turns) — and manual moves leave the drone at an
        # untracked position relative to the grid's origin. Reset only by
        # connect() (a fresh SDK session), so autonomous features refuse to
        # start until the operator reconnects after manual flying.
        self.manual_flight_used = False
        # Human-readable reason for the most recent failed command, shown in
        # the Flight Plan console / Keyboard Control log next to the ❌ (so a
        # student sees WHY, not just that it failed).
        self.last_error = ""
        # Drone id used for flight commands and the video port (9000 + id*2).
        self.drone_id = None
        self.drone_ip = config.DRONE_IP
        self.last_found_drones = []
        self._last_ready_at = 0.0
        self._connected_at = 0.0
        # True while reconnect() runs in the background after the link dropped.
        self.reconnecting = False
        self._reconnect_lock = threading.Lock()

    def _fail(self, message: str) -> bool:
        self.last_error = str(message)
        print(f"❌ AI LAB COMMAND: {self.last_error}")
        return False

    @staticmethod
    def _explain(exc) -> str:
        name = type(exc).__name__
        if name == "LowBattery":
            return f"Battery too low — charge it first: {exc}"
        if name == "NotReady":
            return f"Drone not ready — connection lost?: {exc}"
        if isinstance(exc, limits.CommandValueError):
            return str(exc)
        return f"{name}: {exc}"

    @staticmethod
    def _result_text(result) -> str:
        name = getattr(result, "name", str(result))
        if name == "TIMEOUT":
            return "No reply from the drone (timeout)"
        if name in ("FAILED_241", "FAILED_242"):
            return ("The drone refused the command — height limit, obstacle or low battery "
                    f"(drone rejected the command, code {name})")
        if name == "BUSY":
            return "The drone is still busy with the last command"
        return f"result {name}"

    def _heartbeat(self):
        """Latest REPORT_STATS (207) from the drone, or None."""
        try:
            return self.api._server._datacenter.get_data("Plane", "heartbeat")
        except Exception:
            return None

    def _note_ready(self, hb) -> None:
        if hb is not None and getattr(hb, "drone_status", None) == 2:
            self._last_ready_at = time.monotonic()

    def is_ready_for_takeoff(self) -> bool:
        """Status 2 seen within READY_WINDOW_SEC - not just the latest packet."""
        self._note_ready(self._heartbeat())
        return time.monotonic() - self._last_ready_at < READY_WINDOW_SEC

    def _wait_status_2(self, timeout: float) -> bool:
        """Wait until the latest heartbeat itself says 2 (pyhulax re-reads it)."""
        end = time.monotonic() + timeout
        while True:
            hb = self._heartbeat()
            self._note_ready(hb)
            if hb is not None and getattr(hb, "drone_status", None) == 2:
                return True
            if time.monotonic() >= end:
                return False
            time.sleep(0.05)

    def _takeoff_state_ok(self) -> bool:
        """pyhulax's single_fly_takeoff() calls os._exit(0) - killing the
        WHOLE APP - when the drone's heartbeat is missing or its status is
        not 2 (ready on the ground). Check the same thing first and refuse
        with a message instead.

        The real drone's status flickers 2 -> 0 -> 2 every 1-2 s while it sits
        ready, so wait up to READY_WINDOW_SEC for a fresh 2 instead of failing
        on whichever value happened to arrive last."""
        if self._heartbeat() is None:
            return self._fail("No signal from the drone yet — wait 2-3 seconds and try again")
        if self._wait_status_2(READY_WINDOW_SEC):
            return True
        status = getattr(self._heartbeat(), "drone_status", "?")
        return self._fail(
            f"Drone not ready to take off (state {status}) — put it on flat ground, wait for a steady light, try again "
            "(drone not ready to take off)"
        )

    def telemetry_age_sec(self):
        """Seconds since the drone last sent anything; None when not connected."""
        if not self.connected:
            return None
        return time.monotonic() - max(_last_telemetry_at, self._connected_at)

    def battery_pct(self):
        """Last battery % the drone reported, or None if it never did."""
        try:
            return int(self.api.get_battery())
        except Exception:
            return None

    def flight_state(self):
        """(key, Thai text, English text) from the drone's own status, or None."""
        hb = self._heartbeat()
        if hb is None:
            return None
        status = getattr(hb, "drone_status", None)
        self._note_ready(hb)
        if status == 0 and self.is_ready_for_takeoff():
            status = 2
        return FLIGHT_STATES.get(status, ("unknown", f"state {status}", f"status {status}"))

    def temperature_c(self):
        hb = self._heartbeat()
        temp = getattr(hb, "temperature", None) if hb is not None else None
        return int(temp) if isinstance(temp, (int, float)) else None

    def sync_flying_from_drone(self) -> None:
        """If the drone landed by itself (low battery, landed from elsewhere),
        stop believing it is still flying. Only when no command is running."""
        # The raw status, not flight_state(): that one also shows the 0
        # flicker as "ready", which must never clear the flag mid-flight.
        status = getattr(self._heartbeat(), "drone_status", None)
        if not self.connected or status != 2 or not self._lock.acquire(blocking=False):
            return
        try:
            if self.flying:
                print("ℹ️ AI LAB: drone reports it is on the ground — flying flag cleared")
                self.flying = False
                self.at_survey_altitude = False
        finally:
            self._lock.release()

    def _sync_drone_id(self, wait: float = 3.0) -> int:
        """pyhulax only learns the drone id from PLANE_STATUS (231). The real
        F09-lite (id 5) did not send 231 in a 1-minute test, so pyhulax would
        fill flight commands with None and the video would listen on the
        id-1 port 9002 instead of 9010. Fall back to REPORT_STATS.drone_id and
        give pyhulax that id too, so commands and video agree."""
        end = time.monotonic() + wait
        drone_id = None
        while True:
            drone_id = _pyhulax_config.drone_id
            if drone_id is not None:
                source = "PLANE_STATUS"
                break
            hb_id = getattr(self._heartbeat(), "drone_id", None)
            if time.monotonic() >= end:
                drone_id, source = (hb_id, "REPORT_STATS") if hb_id is not None else (1, "default")
                break
            time.sleep(0.1)
        _pyhulax_config.drone_id = drone_id
        self.drone_id = drone_id
        print(f"🆔 AI LAB: drone id {drone_id} (from {source}) — video port {9000 + drone_id * 2}")
        return drone_id

    @staticmethod
    def _new_api() -> pyhulax.DroneAPI:
        # This app keeps its own capture/decision logs (see capture/manager.py)
        # - pyhulax's own JSONL file/command logging would just be a second,
        # redundant log stream cluttering the project folder.
        return pyhulax.DroneAPI(enable_file_logging=False, enable_command_logging=False)

    def _close_session(self) -> None:
        """Stop the pyhulax session's threads and sockets and forget its data.

        pyhulax never does this by itself: a new session started next to the
        old one (and the old TCP socket stays dead after a drop), and
        DataCenter is a singleton that keeps the last heartbeat and battery
        forever - so a new connect() "succeeded" on stale data and the values
        never updated again until the app was restarted."""
        server = getattr(self.api, "_server", None)
        tasks = getattr(server, "_taskcontroller", None)
        if tasks is not None:
            try:
                tasks.stop_all_task()
            except Exception as exc:
                print(f"⚠️ AI LAB: could not stop the old drone session: {exc}")
            server._taskcontroller = None
            server._connect_status = 0
        try:
            self.api._connected = False
        except Exception:
            pass
        _DataCenter().empty_datacenter()

    def disconnect(self) -> None:
        """End the session so the next connect() starts a fresh one."""
        self.stop_stream()
        self._close_session()
        self.connected = False

    def reconnect(self) -> bool:
        """One fresh session to the same drone, while the link is lost (run
        by the app in the background every few seconds). `connected` stays
        True meanwhile so the "link lost" alert stays up until the drone
        answers again. Never flies the drone."""
        if not self.connected or not self._reconnect_lock.acquire(blocking=False):
            return False
        self.reconnecting = True
        try:
            # The drone may have rebooted: its camera angle is unknown again.
            self.camera_down_ok = False
            ok = self._connect_to(self.drone_ip, 1, 0.5)
            if ok:
                print("🔁 AI LAB: drone link restored")
            return ok
        finally:
            self.reconnecting = False
            self._reconnect_lock.release()

    def connect(self, retries: int = 5, retry_delay: float = 1.0) -> bool:
        if self.connected:
            age = self.telemetry_age_sec()
            if age is None or age <= float(getattr(config, "DRONE_LINK_LOST_SEC", 5.0)):
                return True
            # Marked connected but the drone has been silent: the old session
            # is dead, so start a new one instead of pretending.
            print(f"📡 AI LAB: no data from the drone for {age:.0f}s — starting a new session")
            self.disconnect()
        # A fresh connection attempt means any previous camera-down /
        # stream state belonged to the old session and can't be trusted
        # anymore (e.g. after a drop + reconnect the physical camera may
        # no longer actually be angled down).
        self.camera_down_ok = False
        self.stream_started = False
        self.manual_flight_used = False
        # pyhulax keeps the drone id in a module global that it only fills when
        # it is None - clear it so a different drone is not given the old id.
        _pyhulax_config.drone_id = None
        self.drone_id = None
        # macOS Wi-Fi association with the drone's AP can still be settling
        # right as this runs, so attempt 1 sometimes hits ENETUNREACH before
        # the interface actually has a route to config.DRONE_IP. A short
        # pre-wait avoids burning the first retry (and the stale-socket
        # ADDRINUSE from pyhulax's reconnect right after it) on that.
        time.sleep(1.5)
        if self._connect_to(config.DRONE_IP, retries, retry_delay):
            return True

        # Nothing at the configured / default address (192.168.100.1). The
        # drone may be on a router instead of its own Wi-Fi: listen for its
        # announcements (UDP 8668, every 8-15 s) and try the free ones.
        self.last_found_drones = []
        try:
            found = discovery.discover(16.0)
        except OSError as exc:
            print(f"⚠️ AI LAB: drone search failed: {exc}")
            found = []
        self.last_found_drones = found
        for d in found:
            print(f"🔎 AI LAB: found {d.describe(discovery.my_bind_client(d.ip))}")
        for d in found:
            if d.ip == (config.DRONE_IP or "192.168.100.1"):
                continue
            if d.availability(discovery.my_bind_client(d.ip)) == "taken":
                continue
            print(f"📡 AI LAB: trying discovered drone at {d.ip}")
            if self._connect_to(d.ip, 2, retry_delay):
                return True
        if any(d.availability(discovery.my_bind_client(d.ip)) == "taken" for d in found):
            self.last_error = "Found the drone, but another computer is connected to it — disconnect it there first"
        print("❌ AI LAB: Could not connect to drone")
        return False

    def _connect_to(self, ip, retries: int, retry_delay: float) -> bool:
        """robust_connect() to one address; ip None = pyhulax default (192.168.100.1)."""
        for attempt in range(1, retries + 1):
            try:
                print(f"📡 AI LAB: Connecting to drone {ip or '(default)'} ({attempt}/{retries})...")
                # Each robust_connect() starts a new pyhulax session: stop the
                # previous one first, and clear its old heartbeat so the check
                # inside robust_connect() only passes on a live drone.
                self._close_session()
                if self.api.robust_connect(ip, verbose=False):
                    self.connected = True
                    self._connected_at = time.monotonic()
                    self.drone_ip = ip
                    print("✅ AI LAB: Drone connected")
                    self._sync_drone_id()
                    # Firmware-level backup for our own app-side battery
                    # polling in main_window.py (5s interval, auto-lands at
                    # config.BATTERY_CRITICAL_PCT) - if the app itself ever
                    # hangs or crashes mid-flight, the drone still protects
                    # its own battery. Never fatal if unsupported.
                    try:
                        self.api.enable_battery_failsafe()
                    except Exception as exc:
                        print(f"⚠️ AI LAB: could not enable firmware battery failsafe: {exc}")
                    return True
            except Exception as exc:
                print(f"⚠️ AI LAB connection error: {exc}")
            time.sleep(retry_delay)

        # All retries on this SDK session failed. Some drone SDK sessions
        # leave internal socket/session state stuck after a failed connect()
        # — every future attempt on the *same* DroneAPI instance then keeps
        # failing even once the drone/WiFi is fine again, until the whole
        # app is restarted. Throw the instance away and build a fresh one so
        # the *next* connect() call (e.g. the operator pressing Start again)
        # gets a clean session.
        self._close_session()
        try:
            self.api = self._new_api()
        except Exception as exc:
            print(f"⚠️ AI LAB: could not reset SDK session: {exc}")
        return False

    def set_camera_down(self) -> bool:
        if not config.CAMERA_AUTO_DOWN:
            self.camera_down_requested = False
            self.camera_down_ok = True
            return True
        if not self.connected:
            print("❌ AI LAB CAMERA ANGLE ERROR: drone is not connected")
            return False
        if self.camera_down_ok:
            # Already set for this connected session (connect() resets
            # this flag on every fresh reconnect) — skip re-issuing the
            # same angle command every time Start Camera is pressed.
            # Repeating it back-to-back while the previous command is
            # still settling can return errors or leave the gimbal in an
            # inconsistent state.
            print(f"📷 AI LAB CAMERA already DOWN {config.CAMERA_DOWN_ANGLE}° — skipping repeat command")
            return True
        try:
            self.camera_down_requested = True
            mode = (
                CameraPitchMode.DOWN_ABSOLUTE
                if config.CAMERA_DOWN_DIRECTION == 1
                else CameraPitchMode.UP_ABSOLUTE
            )
            self.api.set_camera_angle(mode, config.CAMERA_DOWN_ANGLE)
            self.camera_down_ok = True
            print(f"📷 AI LAB CAMERA DOWN → {config.CAMERA_DOWN_ANGLE}°")
            return True
        except Exception as exc:
            self.camera_down_ok = False
            print(f"❌ AI LAB CAMERA ANGLE ERROR: {exc}")
            return False

    def start_stream(self) -> bool:
        if not self.connected:
            return False
        try:
            self.api.set_video_stream(True)
            time.sleep(1.0)
            # NOTE: pyhulax's flip_video() does not actually send a flip
            # command to the drone - it calls single_fly_flip_rtp(), which
            # starts an internal debug RTP receiver + cv2.namedWindow preview
            # thread on the SAME UDP port our own video_stream listens on.
            # That second listener steals video packets (causing "Invalid
            # data found when processing input" here) and its cv2.namedWindow
            # call crashes on macOS anyway (GUI calls must run on the main
            # thread). Do not call it.
            # Do NOT call api.create_video_stream() - internally it resolves
            # the RTP port via `self.get_drone_id() or 1`, and 0 is falsy in
            # Python, so any drone whose real ID is 0 silently gets treated
            # as ID 1 and the app listens on the wrong UDP port
            # (9000 + id*2) with no error. Build the VideoStream ourselves
            # with an explicit `is None` check so ID 0 is preserved.
            drone_id = self.drone_id
            if drone_id is None:
                drone_id = self._sync_drone_id(wait=0)
            self.video_stream = VideoStream(
                drone_ip=self.drone_ip,
                drone_id=drone_id,
                config=self.api.config,
            )
            self.video_stream.start(blocking=False)
            self.stream_started = True
            print("🎥 AI LAB RTP STREAM ACTIVE")
            return True
        except Exception as exc:
            self.stream_started = False
            self.video_stream = None
            print(f"❌ AI LAB RTP START ERROR: {exc}")
            return False

    def stop_stream(self):
        try:
            if self.video_stream is not None:
                self.video_stream.stop()
            if self.stream_started:
                self.api.set_video_stream(False)
        except Exception as exc:
            print(f"⚠️ AI LAB RTP stop warning: {exc}")
        finally:
            self.stream_started = False
            self.video_stream = None

    def takeoff(self, require_camera_down: bool = True) -> bool:
        """require_camera_down=False is used by the Missions (Flight Plan,
        Keyboard Control, Follow Target), which choose their own camera
        angle - only the Autonomous Grid needs the 90° down view."""
        self.last_error = ""
        if not config.AUTONOMOUS_FLIGHT_ENABLED or not config.TAKEOFF_ENABLED:
            print("🔒 TAKEOFF BLOCKED: autonomous flight is disabled")
            self.last_error = "Flying is turned off in config.py (TAKEOFF_ENABLED / AUTONOMOUS_FLIGHT_ENABLED)"
            return False
        if not self.connected or (require_camera_down and not self.camera_down_ok):
            print("🔒 TAKEOFF BLOCKED: preflight failed")
            self.last_error = "The drone is not connected" if not self.connected else "The camera isn't pointing down 90° yet"
            return False
        if self.flying:
            # Never re-issue takeoff/climb commands on an already-airborne drone.
            print("⚠️ TAKEOFF SKIPPED: already flying")
            return True
        low = alerts.takeoff_block_reason(self.battery_pct())
        if low:
            print(f"🔒 TAKEOFF BLOCKED: {low}")
            return self._fail(low)
        if not self._takeoff_state_ok():
            return False
        try:
            with self._lock:
                # Climbs straight to survey altitude in one blocking call.
                try:
                    result = self.api.takeoff(height_cm=config.SURVEY_ALTITUDE_CM, blocking=True)
                except DroneNotReadyAbort:
                    # status flickered between our check and pyhulax's: one retry
                    if not self._wait_status_2(READY_WINDOW_SEC):
                        raise
                    result = self.api.takeoff(height_cm=config.SURVEY_ALTITUDE_CM, blocking=True)
                if result != CommandResult.SUCCESS:
                    raise RuntimeError(self._result_text(result))
                self.flying = True
                self.emergency_stop_requested = False
                self.heading = config.INITIAL_HEADING
                self.at_survey_altitude = True
            time.sleep(config.TAKEOFF_SETTLE_SEC)

            # Explicit HOLD before any lateral (forward) movement begins.
            print(f"⏸️ HOLD at survey altitude {config.SURVEY_ALTITUDE_CM} cm")
            time.sleep(config.SURVEY_ALTITUDE_SETTLE_SEC)

            print(f"🛫 TAKEOFF OK — survey altitude target {config.SURVEY_ALTITUDE_CM} cm")
            return True
        except Exception as exc:
            self.flying = False
            self.at_survey_altitude = False
            return self._fail(f"Take-off failed: {self._explain(exc)}")

    def move_forward(self, distance_cm: int) -> bool:
        return self._move_axis(Direction.FORWARD, distance_cm, "FORWARD")

    def move_left(self, distance_cm: int) -> bool:
        return self._move_axis(Direction.LEFT, distance_cm, "LEFT")

    def move_right(self, distance_cm: int) -> bool:
        return self._move_axis(Direction.RIGHT, distance_cm, "RIGHT")

    def move_back(self, distance_cm: int) -> bool:
        return self._move_axis(Direction.BACK, distance_cm, "BACK")

    def _move_axis(self, direction: Direction, distance_cm: int, label: str) -> bool:
        remaining = max(0, int(distance_cm))
        while remaining > 0:
            if self.emergency_stop_requested or not self.flying:
                return False
            step = min(config.GRID_MOVE_STEP_CM, remaining)
            try:
                with self._lock:
                    result = self.api.move(direction, step, blocking=True)
                if result != CommandResult.SUCCESS:
                    raise RuntimeError(f"move returned {result.name}")
                print(f"   ↳ {label:<8} +{step} cm")
                remaining -= step
                time.sleep(config.GRID_MOVE_INTERVAL_SEC)
            except Exception as exc:
                print(f"❌ MOVE {label} ERROR: {exc}")
                return False
        return True

    def turn_right_90(self) -> bool:
        return self._turn("RIGHT", 90, -90)

    def turn_left_90(self) -> bool:
        return self._turn("LEFT", 90, 90)

    def _turn(self, direction: str, degrees: int, signed_angle: int) -> bool:
        if self.emergency_stop_requested or not self.flying:
            return False
        try:
            with self._lock:
                # pyhulax's rotate(): positive angle = CCW (left), negative = CW (right).
                self.api.rotate(signed_angle, blocking=True)
            print(f"   ↳ TURN {direction:<5} {degrees}°")
            time.sleep(config.TURN_SETTLE_SEC)
            self.heading = _next_heading(self.heading, direction)
            return True
        except Exception as exc:
            print(f"❌ TURN {direction} ERROR: {exc}")
            return False

    def turn_to(self, desired_heading: str) -> bool:
        if desired_heading == self.heading:
            return True
        order = ["EAST", "SOUTH", "WEST", "NORTH"]
        ci, di = order.index(self.heading), order.index(desired_heading)
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
        try:
            with self._lock:
                self.api.land(blocking=True)
                self.flying = False
                self.at_survey_altitude = False
            time.sleep(2.0)
            print("🛬 TOUCHDOWN")
            return True
        except Exception as exc:
            return self._fail(f"Landing failed: {self._explain(exc)}")

    def manual_move(self, direction: Direction, distance_cm: int) -> bool:
        """One keyboard-tap worth of movement for Manual Flight (Developer
        Mode). Unlike _move_axis() (used by the autonomous grid), this is a
        single step, not a loop broken into GRID_MOVE_STEP_CM chunks — the
        operator supplies the next step by pressing the key again."""
        if not self._can_fly_command():
            return False
        try:
            dist = limits.move_cm(distance_cm)
        except limits.CommandValueError as exc:
            return self._fail(str(exc))
        self.manual_flight_used = True
        try:
            with self._lock:
                result = self.api.move(
                    direction, dist,
                    blocking=True, speed=VelocityLevel.MEDIUM,
                )
            if result != CommandResult.SUCCESS:
                raise RuntimeError(self._result_text(result))
            return True
        except Exception as exc:
            return self._fail(f"Move failed: {self._explain(exc)}")

    def _can_fly_command(self) -> bool:
        if self.emergency_stop_requested:
            return self._fail("Emergency stop is on — take off again first")
        if not self.connected:
            return self._fail("The drone is not connected")
        if not self.flying:
            return self._fail("The drone isn't flying — take off first")
        self.last_error = ""
        return True

    def manual_rotate(self, degrees: float) -> bool:
        """Arbitrary-angle yaw for Manual Flight. Positive = left (CCW),
        negative = right (CW) - see api.rotate()."""
        if not self._can_fly_command():
            return False
        try:
            # The firmware only accepts a WHOLE number of degrees (int):
            # pyhulax rejects 15.0 for a left turn ("parameter error" ->
            # timeout) and crashes MAVLink packing for a right turn.
            size = limits.rotate_deg(degrees)
        except limits.CommandValueError as exc:
            return self._fail(str(exc))
        signed = size if float(degrees) >= 0 else -size
        self.manual_flight_used = True
        try:
            with self._lock:
                result = self.api.rotate(signed, blocking=True)
            if result != CommandResult.SUCCESS:
                raise RuntimeError(self._result_text(result))
            return True
        except Exception as exc:
            return self._fail(f"Turn failed: {self._explain(exc)}")

    # --- Mission helpers (Flight Plan / Follow Target) -----------------
    def flip(self, direction: str) -> bool:
        """Somersault: direction is "forward", "back", "left" or "right"."""
        if not self._can_fly_command():
            return False
        try:
            fd = FlipDirection[str(direction).upper()]
        except KeyError:
            return self._fail('flip must be "forward", "back", "left" or "right"')
        self.manual_flight_used = True
        try:
            with self._lock:
                result = self.api.flip(fd, blocking=True)
            if result != CommandResult.SUCCESS:
                raise RuntimeError(self._result_text(result))
            return True
        except Exception as exc:
            return self._fail(f"Flip failed: {self._explain(exc)}")

    def set_led(self, r: int, g: int, b: int) -> bool:
        if not self.connected:
            return self._fail("The drone is not connected")
        try:
            clamp = lambda v: max(0, min(255, int(round(float(v)))))
            with self._lock:
                # pyhulax SFLamplight packs b | r<<8 | g<<16, the drone reads
                # r | g<<8 | b<<16 (verified on a real F09-lite 2026-10-07:
                # red showed as green). Hand the channels over pre-rotated.
                self.api.set_led(clamp(g), clamp(b), clamp(r), LEDMode.CONSTANT)
            return True
        except Exception as exc:
            return self._fail(f"LED change failed: {self._explain(exc)}")

    def set_camera_pitch(self, degrees_down: int) -> bool:
        """Tilt the camera: 0 = straight ahead, 90 = straight down."""
        if not self.connected:
            return self._fail("The drone is not connected")
        try:
            angle = max(0, min(90, int(round(float(degrees_down)))))
        except (TypeError, ValueError):
            return self._fail(f"Camera angle must be a number 0-90: {degrees_down!r}")
        try:
            with self._lock:
                self.api.set_camera_angle(CameraPitchMode.DOWN_ABSOLUTE, angle)
            # Only an exact CAMERA_DOWN_ANGLE counts as "ready for the grid".
            self.camera_down_ok = angle == int(config.CAMERA_DOWN_ANGLE)
            return True
        except Exception as exc:
            return self._fail(f"Camera change failed: {self._explain(exc)}")

    def emergency_stop(self):
        self.emergency_stop_requested = True
        self.autonomous_running = False
        print("🛑 EMERGENCY STOP REQUESTED")
        self.touchdown()
        self.stop_stream()


def _next_heading(current: str, direction: str) -> str:
    order = ["EAST", "SOUTH", "WEST", "NORTH"]
    i = order.index(current)
    return order[(i + (1 if direction == "RIGHT" else -1)) % 4]
