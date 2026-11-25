"""🧊 3D simulation + ✔ Check Code for the 🐍 Flight Plan mission.

Three pieces, all without touching a real drone:

    SimTrack       thread-safe pose + flown path, read by the 3D view
                   (ui/flight_sim_view.py) on its own timer.
    VirtualDrone   a drone that only exists in the 3D view. It has the same
                   surface FlightPlanRunner uses on HulaDroneController
                   (takeoff / touchdown / manual_move / manual_rotate / flip /
                   set_led / set_camera_pitch) and the same checks
                   (drone.limits, "take off first", ...), so a Flight Plan
                   fails in the simulation exactly where the real drone would.
    check_plan()   ✔ Check Code: syntax + unknown commands / bad numbers in
                   EVERY branch (AST), then a dry run on an instant
                   VirtualDrone that reports the first runtime error with its
                   line and the planned path for the 3D view.

MirroredDrone wraps the real controller during a real run so the 3D view
follows the commands the real drone actually completed.

Coordinates: x = right of the start point, y = ahead of the start point
(the direction the drone faced at takeoff), z = height; all in cm.
yaw is in degrees, counter-clockwise (= turn_left) positive.
"""
from __future__ import annotations

import ast
import difflib
import math
import threading
import time
from dataclasses import dataclass, field, replace

from .. import config
from ..drone import limits
from .flight_plan import AICommands, DroneCommands, FlightPlanRunner, check_syntax

# Timing at speed ×1 - close to a Hula drone at VelocityLevel.MEDIUM.
MOVE_CM_PER_SEC = 50.0
TURN_DEG_PER_SEC = 90.0
TAKEOFF_SEC = 2.5
LAND_CM_PER_SEC = 50.0
FLIP_SEC = 1.2
FRAME_SEC = 1.0 / 30.0
MIN_HEIGHT_CM = 20         # a real drone stops itself about here above the floor
CEILING_WARN_CM = 250      # typical classroom ceiling
DRY_RUN_LINE_BUDGET = 200_000
DRY_RUN_TIMEOUT_SEC = 5.0

_DIRS = {
    # name: (right, forward, up) in the drone's own frame
    "FORWARD": (0, 1, 0), "BACK": (0, -1, 0),
    "LEFT": (-1, 0, 0), "RIGHT": (1, 0, 0),
    "UP": (0, 0, 1), "DOWN": (0, 0, -1),
}


@dataclass
class Pose:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0
    flip_pitch: float = 0.0   # 0..360 while flipping forward/back
    flip_roll: float = 0.0    # 0..360 while flipping left/right
    led: tuple = (103, 221, 195)
    camera: int = 0
    flying: bool = False


@dataclass
class TrackSnapshot:
    pose: Pose
    path: list
    photos: list
    plan: list
    line: int | None
    label: str
    version: int
    mode: str
    world: list = field(default_factory=list)
    banner: str = ""


class SimTrack:
    """Pose + history shared between a flight (worker thread) and the 3D view."""

    def __init__(self):
        self._lock = threading.Lock()
        self.plan: list = []          # planned path from ✔ Check Code
        self.mode = "sim"             # "sim" / "mirror" (a real drone run)
        self.world: list = []         # 🎯 mission objects (missions/challenges.py)
        self.banner = ""              # big result text in the 3D view
        self.reset()

    def reset(self, keep_plan=True, start=None):
        """start: a Pose to begin at (default: on the ground at the origin)."""
        with self._lock:
            self.pose = start if start is not None else Pose()
            self.path = [(self.pose.x, self.pose.y, self.pose.z)]
            self.photos = []
            self.line = None
            self.label = ""
            if not keep_plan:
                self.plan = []
            self.version = getattr(self, "version", 0) + 1

    def set_world(self, objects, banner=None):
        with self._lock:
            self.world = list(objects or [])
            if banner is not None:
                self.banner = banner
            self.version += 1

    def set_banner(self, text):
        with self._lock:
            self.banner = text or ""
            self.version += 1

    def set_plan(self, points):
        with self._lock:
            self.plan = list(points or [])
            self.version += 1

    def set_pose(self, pose: Pose):
        with self._lock:
            self.pose = pose
            last = self.path[-1]
            if math.dist(last, (pose.x, pose.y, pose.z)) >= 1.0:
                self.path.append((pose.x, pose.y, pose.z))
            self.version += 1

    def set_status(self, line=None, label=None):
        with self._lock:
            if line is not None:
                self.line = line
            if label is not None:
                self.label = label
            self.version += 1

    def add_photo(self):
        with self._lock:
            p = self.pose
            self.photos.append((p.x, p.y, p.z))
            self.version += 1

    def snapshot(self) -> TrackSnapshot:
        with self._lock:
            return TrackSnapshot(replace(self.pose), list(self.path), list(self.photos), list(self.plan),
                                 self.line, self.label, self.version, self.mode, list(self.world), self.banner)


class _SimAPI:
    def get_battery(self):
        return 100


class VirtualDrone:
    """A drone that flies only in the 3D view (see module docstring)."""

    simulated = True

    def __init__(self, track: SimTrack | None = None, speed: float = 1.0, instant: bool = False):
        self.track = track or SimTrack()
        self.speed = speed
        self.instant = instant
        self.stop_event = threading.Event()
        self._rush = threading.Event()  # MirroredDrone: jump to the end
        self.api = _SimAPI()
        self.connected = True
        self.flying = False
        self.emergency_stop_requested = False
        self.camera_down_ok = False
        self.last_error = ""
        self.pose = Pose()
        self.points = [(0.0, 0.0, 0.0)]
        self.line = None
        self.warnings = []        # [(line, message)]
        self.sim_seconds = 0.0    # flight time at speed ×1
        self.distance_cm = 0.0
        self.max_height_cm = 0.0
        self.max_range_cm = 0.0
        self.wait_seconds = 0.0
        # 🎯 what missions/challenges.py grades: where things happened
        self.photo_events = []    # [(x, y, z, camera_deg)]
        self.led_events = []      # [((r, g, b), (x, y, z), flying)]
        self.flip_events = []     # [(direction, z, (r, g, b))]

    # -- helpers -----------------------------------------------------------
    def _fail(self, message) -> bool:
        self.last_error = str(message)
        return False

    def _warn(self, message):
        item = (self.line, message)
        if item not in self.warnings:
            self.warnings.append(item)

    def _can_fly_command(self) -> bool:
        if self.emergency_stop_requested:
            return self._fail("Emergency stop is on — take off again first")
        if not self.flying:
            return self._fail("The drone isn't flying — take off first")
        self.last_error = ""
        return True

    def _publish(self, pose):
        self.pose = pose
        self.track.set_pose(replace(pose))

    def _animate(self, target: Pose, seconds: float, abortable=True, travel=True) -> bool:
        """Move self.pose to `target` over `seconds` (at speed ×1). Returns
        False if Stop was pressed half-way (the drone stays where it got to)."""
        start = self.pose
        self.sim_seconds += seconds
        if travel:  # takeoff / landing are not counted as "distance flown"
            self.distance_cm += math.dist((start.x, start.y, start.z), (target.x, target.y, target.z))
        if not self.instant and seconds > 0:
            done = 0.0
            last = time.monotonic()
            while done < 1.0:
                if abortable and self.stop_event.is_set():
                    return False
                if self._rush.is_set():
                    break
                time.sleep(FRAME_SEC)
                now = time.monotonic()
                done = min(1.0, done + (now - last) * max(0.05, float(self.speed)) / seconds)
                last = now
                self._publish(_lerp(start, target, _ease(done)))
        final = replace(target, flip_pitch=0.0, flip_roll=0.0)
        self._publish(final)
        self.points.append((final.x, final.y, final.z))
        self.max_height_cm = max(self.max_height_cm, final.z)
        self.max_range_cm = max(self.max_range_cm, math.hypot(final.x, final.y))
        return True

    # -- the controller surface FlightPlanRunner uses ----------------------
    def takeoff(self, require_camera_down: bool = False) -> bool:
        self.last_error = ""
        if not config.AUTONOMOUS_FLIGHT_ENABLED or not config.TAKEOFF_ENABLED:
            return self._fail("Flying is turned off in config.py (TAKEOFF_ENABLED / AUTONOMOUS_FLIGHT_ENABLED)")
        if self.flying:
            return True
        self.emergency_stop_requested = False
        self.flying = True
        target = replace(self.pose, z=float(config.SURVEY_ALTITUDE_CM), flying=True)
        return self._animate(target, TAKEOFF_SEC, travel=False)

    def touchdown(self) -> bool:
        if not self.flying:
            return True
        # Landing is never cut short: Stop / closing must leave it on the floor.
        ok = self._animate(replace(self.pose, z=0.0, flying=False),
                           max(0.6, self.pose.z / LAND_CM_PER_SEC), abortable=False, travel=False)
        self.flying = False
        return ok

    def manual_move(self, direction, distance_cm) -> bool:
        if not self._can_fly_command():
            return False
        try:
            dist = limits.move_cm(distance_cm)
        except limits.CommandValueError as exc:
            return self._fail(str(exc))
        name = getattr(direction, "name", str(direction)).upper()
        if name not in _DIRS:
            return self._fail(f"unknown direction {direction!r}")
        r, f, u = _DIRS[name]
        yaw = math.radians(self.pose.yaw)
        # forward = (-sin yaw, cos yaw), right = (cos yaw, sin yaw)
        dx = dist * (r * math.cos(yaw) - f * math.sin(yaw))
        dy = dist * (r * math.sin(yaw) + f * math.cos(yaw))
        z = self.pose.z + dist * u
        if z < MIN_HEIGHT_CM:
            self._warn(f"down({dist}) is too low — the real drone stops at about {MIN_HEIGHT_CM} cm "
                       f"(too low, the drone stops near the floor)")
            z = float(MIN_HEIGHT_CM)
        elif z > CEILING_WARN_CM and self.pose.z <= CEILING_WARN_CM:
            self._warn(f"Height {z:.0f} cm — mind the ceiling (over {CEILING_WARN_CM} cm)")
        target = replace(self.pose, x=self.pose.x + dx, y=self.pose.y + dy, z=z)
        return self._animate(target, dist / MOVE_CM_PER_SEC)

    def manual_rotate(self, degrees) -> bool:
        if not self._can_fly_command():
            return False
        try:
            size = limits.rotate_deg(degrees)
        except limits.CommandValueError as exc:
            return self._fail(str(exc))
        signed = size if float(degrees) >= 0 else -size
        return self._animate(replace(self.pose, yaw=self.pose.yaw + signed), size / TURN_DEG_PER_SEC)

    def flip(self, direction: str) -> bool:
        if not self._can_fly_command():
            return False
        direction = str(direction).lower()
        if direction not in ("forward", "back", "left", "right"):
            return self._fail('flip must be "forward", "back", "left" or "right"')
        self.flip_events.append((direction, self.pose.z, self.pose.led))
        target = replace(self.pose,
                         flip_pitch={"forward": 360.0, "back": -360.0}.get(direction, 0.0),
                         flip_roll={"right": 360.0, "left": -360.0}.get(direction, 0.0))
        return self._animate(target, FLIP_SEC)

    def set_led(self, r, g, b) -> bool:
        led = tuple(max(0, min(255, int(round(float(v))))) for v in (r, g, b))
        self.led_events.append((led, (self.pose.x, self.pose.y, self.pose.z), self.flying))
        self._publish(replace(self.pose, led=led))
        return True

    def set_camera_pitch(self, degrees_down) -> bool:
        angle = max(0, min(90, int(round(float(degrees_down)))))
        self.camera_down_ok = angle == int(config.CAMERA_DOWN_ANGLE)
        self._publish(replace(self.pose, camera=angle))
        return True

    def emergency_stop(self):
        self.emergency_stop_requested = True
        self.stop_event.set()
        self._publish(replace(self.pose, z=0.0, flying=False, flip_pitch=0.0, flip_roll=0.0))
        self.flying = False

    # -- hooks DroneCommands looks for (sim only) -------------------------
    def sim_wait(self, seconds) -> bool:
        self.sim_seconds += seconds
        if self.flying:
            self.wait_seconds += seconds
        if self.instant:
            return True
        end = 0.0
        while end < seconds:
            if self.stop_event.wait(FRAME_SEC * 3):
                return False
            end += FRAME_SEC * 3 * max(0.05, float(self.speed))
        return True

    def sim_photo(self):
        p = self.pose
        self.photo_events.append((p.x, p.y, p.z, p.camera))
        self.track.add_photo()


class MirroredDrone:
    """The real controller for a real Flight Plan run. Every command goes to
    the real drone; a shadow VirtualDrone animates the same command in the
    3D view WHILE the real one flies it, then snaps to the end once the real
    command has finished (or goes back if it failed)."""

    simulated = False

    def __init__(self, real, track: SimTrack):
        self._real = real
        self._shadow = VirtualDrone(track)

    def __getattr__(self, name):
        # flying / connected / emergency_stop_requested / api / last_error ...
        return getattr(self._real, name)

    def _mirror(self, real_call, shadow_call):
        shadow = self._shadow
        before = (replace(shadow.pose), shadow.flying)
        shadow._rush.clear()
        anim = threading.Thread(target=shadow_call, daemon=True, name="AI-LAB-SIM-MIRROR")
        anim.start()
        try:
            ok = bool(real_call())
        finally:
            shadow._rush.set()
            anim.join(timeout=2.0)
        if not ok:
            shadow.flying = before[1]
            shadow._publish(before[0])
        return ok

    def takeoff(self, *args, **kwargs):
        return self._mirror(lambda: self._real.takeoff(*args, **kwargs), self._shadow.takeoff)

    def touchdown(self):
        return self._mirror(self._real.touchdown, self._shadow.touchdown)

    def manual_move(self, direction, distance_cm):
        return self._mirror(lambda: self._real.manual_move(direction, distance_cm),
                            lambda: self._shadow.manual_move(direction, distance_cm))

    def manual_rotate(self, degrees):
        return self._mirror(lambda: self._real.manual_rotate(degrees),
                            lambda: self._shadow.manual_rotate(degrees))

    def flip(self, direction):
        return self._mirror(lambda: self._real.flip(direction), lambda: self._shadow.flip(direction))

    def set_led(self, r, g, b):
        ok = self._real.set_led(r, g, b)
        if ok:
            self._shadow.set_led(r, g, b)
        return ok

    def set_camera_pitch(self, degrees_down):
        ok = self._real.set_camera_pitch(degrees_down)
        if ok:
            self._shadow.set_camera_pitch(degrees_down)
        return ok

    def land_in_view(self):
        """The real drone landed outside the Flight Plan (e.g. EMERGENCY)."""
        self._shadow.flying = False
        self._shadow._publish(replace(self._shadow.pose, z=0.0, flying=False))


def _ease(t: float) -> float:
    return t * t * (3 - 2 * t)


def _lerp(a: Pose, b: Pose, t: float) -> Pose:
    mix = lambda p, q: p + (q - p) * t
    return replace(b, x=mix(a.x, b.x), y=mix(a.y, b.y), z=mix(a.z, b.z), yaw=mix(a.yaw, b.yaw),
                   flip_pitch=mix(a.flip_pitch, b.flip_pitch), flip_roll=mix(a.flip_roll, b.flip_roll),
                   flying=a.flying or b.flying)


def fake_detections(names):
    """ai.* in the simulation: pretend the AI sees these class names."""
    return [{"class_name": n, "confidence": 0.99, "xyxy": [0, 0, 0, 0]} for n in names]


# =====================================================================
# ✔ Check Code
# =====================================================================
@dataclass
class Issue:
    line: int | None
    level: str        # "error" / "warning" / "info"
    message: str


@dataclass
class CheckReport:
    issues: list = field(default_factory=list)
    commands: int = 0
    distance_cm: float = 0.0
    max_height_cm: float = 0.0
    max_range_cm: float = 0.0
    seconds: float = 0.0
    path: list = field(default_factory=list)
    output: list = field(default_factory=list)   # what print() showed in the dry run
    drone: object = None       # the dry run's VirtualDrone (🎯 mission grading)
    auto_landed: bool = False  # the code ended in the air

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def errors(self):
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self):
        return [i for i in self.issues if i.level == "warning"]

    def add(self, line, level, message):
        if not any(i.line == line and i.message == message for i in self.issues):
            self.issues.append(Issue(line, level, message))


DRONE_METHODS = sorted(n for n in dir(DroneCommands) if not n.startswith("_"))
AI_METHODS = sorted(n for n in dir(AICommands) if not n.startswith("_"))
_MOVES = {"forward", "back", "left", "right", "up", "down",
          "single_fly_forward", "single_fly_back", "single_fly_left", "single_fly_right",
          "single_fly_up", "single_fly_down"}
_TURNS = {"turn_left", "turn_right", "single_fly_turnleft", "single_fly_turnright"}


def _number_arg(call: ast.Call):
    if not call.args:
        return None
    node = call.args[0]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
        node = ast.Constant(-node.operand.value) if isinstance(node.operand.value, (int, float)) else node
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    return None


def _lint(tree: ast.AST, report: CheckReport):
    """Things that are wrong even in branches the dry run never reaches."""
    move_cap = int(getattr(config, "FLIGHT_PLAN_MAX_MOVE_CM", 200))
    turn_cap = int(min(getattr(config, "FLIGHT_PLAN_MAX_ROTATE_DEG", 360), limits.ROTATE_MAX_DEG))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            owner, name = node.value.id, node.attr
            known = DRONE_METHODS if owner in ("drone", "api") else AI_METHODS if owner == "ai" else None
            if known is not None and name not in known:
                close = difflib.get_close_matches(name, known, n=1, cutoff=0.6)
                hint = f" — did you mean {owner}.{close[0]}()?" if close else ""
                report.add(node.lineno, "error", f"Unknown command {owner}.{name}{hint}")
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id in ("drone", "api")):
            continue
        name = node.func.attr
        value = _number_arg(node)
        if name in _MOVES:
            if not node.args:
                report.add(node.lineno, "error", f"{name}() needs a distance in cm, like {name}(50)")
            elif value is not None:
                if value < 0:
                    report.add(node.lineno, "error", f"{name}({value}) distance must be positive")
                elif 0 < value < limits.MOVE_MIN_CM:
                    report.add(node.lineno, "error",
                               f"{name}({value}) is too small — minimum {limits.MOVE_MIN_CM} cm")
                elif value > move_cap:
                    report.add(node.lineno, "warning",
                               f"{name}({value}) will be capped at {move_cap} cm per command")
        elif name in _TURNS and value is not None and abs(value) > turn_cap:
            report.add(node.lineno, "warning", f"{name}({value}) will be capped at {turn_cap}°")
        elif name == "flip":
            report.add(node.lineno, "warning", "flip — open space only, at least 1 m high")
        elif name == "camera" and value is not None and not 0 <= value <= 90:
            report.add(node.lineno, "warning", f"camera({value}) is outside 0-90° — it will be set to {max(0, min(90, value))}°")


def check_plan(code: str, ai_names=()) -> CheckReport:
    """✔ Check Code: syntax, every command name/number, then a dry run on an
    instant VirtualDrone. Never touches the real drone."""
    report = CheckReport()
    err = check_syntax(code)
    if err is not None:
        line, msg = err
        report.add(line, "error", f"Code mistake (SyntaxError: {msg})")
        return report
    tree = ast.parse(code)
    _lint(tree, report)
    calls = {n.func.attr for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and isinstance(n.func.value, ast.Name) and n.func.value.id in ("drone", "api")}
    if not calls & {"takeoff", "single_fly_takeoff"} and calls & (_MOVES | _TURNS | {"flip"}):
        report.add(None, "error", "No drone.takeoff() — take off before moving")
    if report.errors:
        return report  # the dry run would only repeat the first of them

    track = SimTrack()
    vdrone = VirtualDrone(track, instant=True)
    finished = threading.Event()

    def on_line(n):
        vdrone.line = n

    def on_log(text):
        report.output.append(text)

    runner = FlightPlanRunner(
        code, vdrone, lambda: fake_detections(ai_names),
        on_log=on_log, on_line=on_line, on_finished=lambda *_: finished.set(),
        line_budget=DRY_RUN_LINE_BUDGET,
    )
    runner.start()
    if not finished.wait(DRY_RUN_TIMEOUT_SEC):
        runner.stop()
        report.add(vdrone.line, "error",
                   f"The code ran over {DRY_RUN_TIMEOUT_SEC:g} seconds without finishing "
                   "(the code did not finish; check loops / time.sleep)")
        return report
    for text in report.output:
        if text.startswith("❌"):
            msg = text.split(":", 1)[1].strip() if text.startswith("❌ Line") else text[1:].strip()
            report.add(runner.error_line, "error", msg)
        elif text.startswith("[") and "] ⚠️" in text:
            head, msg = text.split("] ⚠️", 1)
            line = int(head[1:]) if head[1:].isdigit() else None
            if not any(i.line == line and i.level == "warning" for i in report.issues):
                report.add(line, "warning", msg.strip())  # (not already said by _lint)
    for line, msg in vdrone.warnings:
        report.add(line, "warning", msg)
    if runner.auto_landed and not report.errors:
        report.add(None, "warning", "The code ends without drone.land() — the app will land automatically")
    if runner.commands == 0 and not report.errors:
        report.add(None, "info", "No flight command ran")
    if any(isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "ai"
           for n in ast.walk(tree)):
        seen = ", ".join(ai_names) if ai_names else "nothing"
        report.add(None, "info", f"ai.* when checking/simulating: pretend the AI sees → {seen}")
    report.commands = runner.commands
    report.distance_cm = vdrone.distance_cm
    report.max_height_cm = vdrone.max_height_cm
    report.max_range_cm = vdrone.max_range_cm
    report.seconds = vdrone.sim_seconds
    report.path = list(vdrone.points)
    report.drone = vdrone
    report.auto_landed = runner.auto_landed
    return report
