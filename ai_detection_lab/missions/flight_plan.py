"""🐍 Flight Plan mission - students write Python that flies the drone.

The student's code runs INSIDE the app (a worker thread), using the app's
own drone connection and the live AI Detection results. It must not open its
own connection (a Hula drone accepts one controller at a time, and the app
is already receiving its video) - which is why the student gets two ready
objects instead of importing pyhula/pyhulax:

    drone   - flight commands   (drone.forward(100), drone.land(), ...)
    ai      - detection results (ai.sees("cone"), ai.count("person"), ...)

Safety:
  * Stop / Emergency Stop interrupt the code between ANY two lines (a line
    tracer raises MissionStopped), even inside `while True:` loops.
  * Each command is clamped (config.FLIGHT_PLAN_MAX_MOVE_CM /
    FLIGHT_PLAN_MAX_ROTATE_DEG) and a run is limited to
    config.FLIGHT_PLAN_MAX_COMMANDS flight commands.
  * If the code ends (normally or with an error) while still flying, the
    drone lands automatically.
  * Syntax errors are reported with their line number BEFORE takeoff.
"""
from __future__ import annotations

import sys
import threading
import time
import traceback

from .. import config
from ..drone import limits

FILENAME = "<flight_plan>"

EXAMPLE_CODE = '# 🐍 Flight Plan — write code to fly the drone, then press ▶ Run\n# Distances are in centimeters (cm), angles in degrees\n# Press ⏹ Stop or EMERGENCY at any time — the drone stops right away\n\nprint("🔋 Battery:", drone.battery(), "%")\n\ndrone.takeoff()          # take off (about 1 meter high)\n\ndrone.up(30)             # up 30 cm\ndrone.down(30)           # down 30 cm\n\ndrone.forward(50)        # forward 50 cm\ndrone.back(50)           # back 50 cm\n\ndrone.left(50)           # left 50 cm\ndrone.right(50)          # right 50 cm\n\ndrone.turn_left(90)      # turn left 90 degrees\ndrone.turn_right(90)     # turn right 90 degrees\n\ndrone.led(0, 255, 0)     # green light (R, G, B)\n\n# --- let the AI help decide ---------------------------------\ndrone.camera(90)         # point the camera at the floor (0 = straight ahead, 90 = down)\ndrone.wait(2)            # give the AI 2 seconds to look\nif ai.sees("person"):    # a class name from your model, like "cone"\n    print("👀 Found", ai.count("person"))\n    drone.led(255, 0, 0)\n    drone.photo()        # save a photo (see the Post Flight Report)\nelse:\n    print("Target not found")\n\n# --- flips (open space only) -----------------------\n# drone.flip("forward")   # "forward" / "back" / "left" / "right"\n\ndrone.land()             # land\n'


class MissionStopped(Exception):
    """Raised inside the student's code when Stop / Emergency is pressed."""


class FlightCommandError(Exception):
    """A drone command failed (e.g. not flying, drone disconnected)."""


class _Context:
    def __init__(self, drone, on_log, stop_event):
        self.drone = drone
        self.on_log = on_log
        self.stop_event = stop_event
        self.commands = 0
        self.took_off = False

    def check_stop(self):
        if self.stop_event.is_set():
            raise MissionStopped("stopped by user")
        if self.took_off and getattr(self.drone, "emergency_stop_requested", False):
            # Emergency Stop pressed elsewhere in the app during this run.
            raise MissionStopped("emergency stop")

    def caller_line(self) -> int | None:
        f = sys._getframe(2)
        while f is not None:
            if f.f_code.co_filename == FILENAME:
                return f.f_lineno
            f = f.f_back
        return None

    def count_command(self):
        self.commands += 1
        limit = int(getattr(config, "FLIGHT_PLAN_MAX_COMMANDS", 200))
        if self.commands > limit:
            raise FlightCommandError(
                f"Too many commands ("
                f"more than {limit} flight commands in one run — stops endless loops)"
            )

    def log(self, text):
        line = self.caller_line()
        self.on_log(f"[{line}] {text}" if line else text)


def _number(value, what):
    try:
        return float(value)
    except (TypeError, ValueError):
        raise FlightCommandError(f"{what} must be a number like 50, got {value!r}")


def _clamp_cm(value, log=None) -> int:
    """0 = do nothing; otherwise a whole number of cm the drone accepts
    (limits.MOVE_MIN_CM..) capped at config.FLIGHT_PLAN_MAX_MOVE_CM."""
    v = int(round(_number(value, "distance")))
    if v < 0:
        raise FlightCommandError("Distance must be positive")
    if v == 0:
        return 0
    cap = int(getattr(config, "FLIGHT_PLAN_MAX_MOVE_CM", 200))
    if v > cap:
        if log:
            log(f"⚠️ {v} cm is too far — limited to {cap} cm per command (safety limit)")
        v = cap
    if v < limits.MOVE_MIN_CM:
        raise FlightCommandError(
            f"The smallest move is {limits.MOVE_MIN_CM} cm (the drone can't move less), but the code says {v} cm"
        )
    return v


def _turn_deg(value, log=None) -> int:
    v = int(round(abs(_number(value, "angle"))))
    cap = int(min(getattr(config, "FLIGHT_PLAN_MAX_ROTATE_DEG", 360), limits.ROTATE_MAX_DEG))
    if v > cap:
        if log:
            log(f"⚠️ {v}° is too much — limited to {cap}° per command")
        v = cap
    return v


class DroneCommands:
    """The `drone` object students use. Every method blocks until the drone
    has finished the movement, so the code reads top-to-bottom like a plan."""

    def __init__(self, ctx: _Context, save_photo=None):
        self._ctx = ctx
        self._save_photo = save_photo

    # -- internal ------------------------------------------------------
    def _run(self, label, fn, *args):
        ctx = self._ctx
        ctx.check_stop()
        ctx.count_command()
        ctx.log(f"✈️ {label}")
        ok = fn(*args)
        ctx.check_stop()
        if not ok:
            if ctx.stop_event.is_set() or (ctx.took_off and getattr(ctx.drone, "emergency_stop_requested", False)):
                raise MissionStopped("emergency stop")
            if not ctx.drone.flying and label.split()[0] not in ("takeoff", "led", "camera", "land"):
                raise FlightCommandError(f"'{label}' failed — the drone is not flying (forgot drone.takeoff()?)")
            reason = str(getattr(ctx.drone, "last_error", "") or "").strip()
            raise FlightCommandError(f"'{label}' failed" + (f" — {reason}" if reason else ""))
        return True

    def _move(self, name, direction_name, cm):
        from pyhulax.core.types import Direction
        dist = _clamp_cm(cm, self._ctx.log)
        if dist == 0:
            return True
        return self._run(f"{name} {dist} cm", self._ctx.drone.manual_move,
                         Direction[direction_name], dist)

    # -- flight --------------------------------------------------------
    def takeoff(self):
        """ขึ้นบิน / take off (hovers at about config.SURVEY_ALTITUDE_CM)."""
        if self._ctx.drone.flying:
            self._ctx.log("ℹ️ takeoff: already flying")
            return True
        ok = self._run("takeoff", lambda: self._ctx.drone.takeoff(require_camera_down=False))
        self._ctx.took_off = True
        return ok

    def land(self):
        """ลงจอด / land."""
        return self._run("land", self._ctx.drone.touchdown)

    def forward(self, cm):
        return self._move("forward", "FORWARD", cm)

    def back(self, cm):
        return self._move("back", "BACK", cm)

    def left(self, cm):
        return self._move("left", "LEFT", cm)

    def right(self, cm):
        return self._move("right", "RIGHT", cm)

    def up(self, cm):
        return self._move("up", "UP", cm)

    def down(self, cm):
        return self._move("down", "DOWN", cm)

    def turn_left(self, degrees=90):
        d = _turn_deg(degrees, self._ctx.log)
        if d == 0:
            return True
        return self._run(f"turn_left {d}°", self._ctx.drone.manual_rotate, d)

    def turn_right(self, degrees=90):
        d = _turn_deg(degrees, self._ctx.log)
        if d == 0:
            return True
        return self._run(f"turn_right {d}°", self._ctx.drone.manual_rotate, -d)

    def flip(self, direction="forward"):
        """ตีลังกา: "forward" / "back" / "left" / "right"."""
        direction = str(direction).lower()
        if direction not in ("forward", "back", "left", "right"):
            raise FlightCommandError('flip must be "forward", "back", "left" or "right"')
        return self._run(f"flip {direction}", self._ctx.drone.flip, direction)

    def led(self, r, g, b):
        """เปลี่ยนสีไฟ LED (0-255)."""
        r, g, b = (int(round(_number(v, "color"))) for v in (r, g, b))
        return self._run(f"led ({r}, {g}, {b})", self._ctx.drone.set_led, r, g, b)

    def camera(self, degrees_down):
        """มุมกล้อง: 0 = มองตรงไปข้างหน้า, 90 = มองลงพื้น."""
        deg = max(0, min(90, int(round(_number(degrees_down, "camera angle")))))
        return self._run(f"camera {deg}°", self._ctx.drone.set_camera_pitch, deg)

    def wait(self, seconds):
        """รอ (ลอยนิ่ง) กี่วินาที."""
        seconds = max(0.0, _number(seconds, "time"))
        self._ctx.check_stop()
        self._ctx.log(f"⏳ wait {seconds:g} s")
        sim_wait = getattr(self._ctx.drone, "sim_wait", None)
        if sim_wait is not None:
            # 🧊 3D simulation: hover on the simulation's own clock (speed ×N).
            if not sim_wait(seconds):
                raise MissionStopped("stopped by user")
            return True
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if self._ctx.stop_event.wait(min(0.1, max(0.0, end - time.monotonic()))):
                raise MissionStopped("stopped by user")
        return True

    hover = wait

    def photo(self):
        """บันทึกภาพจากกล้องตอนนี้เป็นหลักฐาน (ดูใน Post Flight Report)."""
        self._ctx.check_stop()
        sim_photo = getattr(self._ctx.drone, "sim_photo", None)
        if sim_photo is not None:
            sim_photo()
            self._ctx.log("📸 [SIM] photo — simulation, no real picture")
            return None
        if self._save_photo is None:
            self._ctx.log("⚠️ photo: camera not running")
            return None
        path = self._save_photo()
        self._ctx.log(f"📸 photo saved: {path.name}" if path else "⚠️ photo: no camera frame yet")
        return path

    def battery(self):
        try:
            return int(self._ctx.drone.api.get_battery())
        except Exception:
            return -1

    def is_flying(self):
        return bool(self._ctx.drone.flying)

    # -- names from the classic pyhula examples (FlightPlan.py) --------
    def single_fly_takeoff(self, *_):
        return self.takeoff()

    def single_fly_touchdown(self, *_):
        return self.land()

    def single_fly_forward(self, distance, *_):
        return self.forward(distance)

    def single_fly_back(self, distance, *_):
        return self.back(distance)

    def single_fly_left(self, distance, *_):
        return self.left(distance)

    def single_fly_right(self, distance, *_):
        return self.right(distance)

    def single_fly_up(self, distance, *_):
        return self.up(distance)

    def single_fly_down(self, distance, *_):
        return self.down(distance)

    def single_fly_turnleft(self, angle, *_):
        return self.turn_left(angle)

    def single_fly_turnright(self, angle, *_):
        return self.turn_right(angle)

    def single_fly_somersault(self, direction, *_):
        return self.flip({0: "forward", 1: "back", 2: "left", 3: "right"}.get(int(direction), "forward"))

    def get_battery(self):
        return self.battery()


class AICommands:
    """The `ai` object: read the LIVE results of the app's AI Detection."""

    def __init__(self, ctx: _Context, get_detections):
        self._ctx = ctx
        self._get = get_detections

    def detections(self):
        """รายการที่ AI เห็นตอนนี้: [{"class_name", "confidence", "xyxy"}, ...]"""
        self._ctx.check_stop()
        try:
            return [dict(d) for d in (self._get() or [])]
        except Exception:
            return []

    def count(self, name=None, min_confidence=0.0):
        """จำนวนวัตถุที่ AI เห็นตอนนี้ (ระบุชื่อคลาสหรือไม่ก็ได้)."""
        return sum(
            1 for d in self.detections()
            if (name is None or str(d.get("class_name", "")).lower() == str(name).lower())
            and float(d.get("confidence", 0.0)) >= float(min_confidence)
        )

    def sees(self, name=None, min_confidence=0.0):
        """True ถ้า AI เห็นวัตถุนี้อยู่ตอนนี้."""
        return self.count(name, min_confidence) > 0

    def names(self):
        """ชื่อคลาสที่ AI เห็นอยู่ตอนนี้."""
        return sorted({str(d.get("class_name", "")) for d in self.detections()})


# v6.0.0: a Flight Plan runs inside the app, so it gets only what flying
# needs. Without this a student script could `import ai_detection_lab`,
# `import os` or open() files and poke around the app's internals.
ALLOWED_IMPORTS = frozenset({"math", "random", "time"})
_SAFE_BUILTIN_NAMES = (
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter", "float", "format",
    "int", "isinstance", "len", "list", "map", "max", "min", "pow", "range", "reversed",
    "round", "set", "sorted", "str", "sum", "tuple", "zip", "True", "False", "None",
    "Exception", "ValueError", "TypeError", "ZeroDivisionError", "IndexError", "KeyError",
    "StopIteration", "ArithmeticError", "RuntimeError",
)


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split(".")[0] in ALLOWED_IMPORTS:
        return __import__(name, globals, locals, fromlist, level)
    raise ImportError(
        f"'{name}' is not allowed "
        f"(Flight Plan can only import {', '.join(sorted(ALLOWED_IMPORTS))})"
    )


def safe_builtins() -> dict:
    import builtins
    safe = {n: getattr(builtins, n) for n in _SAFE_BUILTIN_NAMES if hasattr(builtins, n)}
    safe["__import__"] = _safe_import
    return safe


def check_sandbox(code: str):
    """(line, message) for the first thing a Flight Plan may not use, else None:
    dunder names (`__class__`, `__subclasses__` ... the usual way out of a
    restricted namespace) and imports outside ALLOWED_IMPORTS."""
    import ast
    try:
        tree = ast.parse(code, FILENAME)
    except SyntaxError:
        return None  # check_syntax reports it
    for node in ast.walk(tree):
        line = getattr(node, "lineno", 0)
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return line, f"'{node.attr}' can't be used in a Flight Plan"
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            return line, f"'{node.id}' can't be used in a Flight Plan"
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_IMPORTS:
                    return line, f"import {alias.name} is not allowed — only {', '.join(sorted(ALLOWED_IMPORTS))}"
        if isinstance(node, ast.ImportFrom):
            if node.level or (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                return line, f"from {node.module} import is not allowed — only {', '.join(sorted(ALLOWED_IMPORTS))}"
    return None


def check_syntax(code: str):
    """Returns None if OK, otherwise (line, message)."""
    try:
        compile(code, FILENAME, "exec")
    except SyntaxError as exc:
        return exc.lineno or 0, f"{exc.msg}"
    return check_sandbox(code)


class FlightPlanRunner(threading.Thread):
    """Runs one Flight Plan. Callbacks are called from this worker thread
    (wire them to Qt signals): on_log(str), on_line(int), on_finished(ok, msg)."""

    def __init__(self, code, drone, get_detections, on_log, on_line, on_finished, save_photo=None,
                 line_budget=None):
        super().__init__(daemon=True, name="AI-LAB-FLIGHT-PLAN")
        self.code = code
        self.drone = drone
        self.on_log = on_log
        self.on_line = on_line
        self.on_finished = on_finished
        self._stop_event = threading.Event()
        self._ctx = _Context(drone, on_log, self._stop_event)
        self._drone_cmds = DroneCommands(self._ctx, save_photo)
        self._ai = AICommands(self._ctx, get_detections)
        self._last_line = None
        # Check Code's dry run: give up after this many executed lines
        # (a `while True:` with no drone command would otherwise never end).
        self._line_budget = line_budget
        self._lines_run = 0
        self.error_line = None    # line of the error that ended the run
        self.auto_landed = False  # code ended while still flying

    @property
    def stop_event(self) -> threading.Event:
        return self._stop_event

    @property
    def commands(self) -> int:
        return self._ctx.commands

    def stop(self):
        self._stop_event.set()

    def _tracer(self, frame, event, arg):
        if frame.f_code.co_filename != FILENAME:
            return None  # don't trace library code
        if event == "line":
            if self._stop_event.is_set():
                raise MissionStopped("stopped by user")
            if self._line_budget is not None:
                self._lines_run += 1
                if self._lines_run > self._line_budget:
                    self._last_line = frame.f_lineno
                    raise FlightCommandError(
                        "The code loops too long (maybe an endless loop)"
                    )
            if frame.f_lineno != self._last_line:
                self._last_line = frame.f_lineno
                try:
                    self.on_line(frame.f_lineno)
                except Exception:
                    pass
        return self._tracer

    def run(self):
        ok, message = False, ""
        err = check_syntax(self.code)
        if err is not None:
            line, msg = err
            self.error_line = line
            self.on_log(f"❌ Line {line}: code mistake (SyntaxError: {msg}) — the drone did not fly")
            self.on_finished(False, f"SyntaxError line {line}")
            return
        if not getattr(self.drone, "connected", False):
            self.on_log("❌ The drone is not connected — connect it first")
            self.on_finished(False, "not connected")
            return

        def _print(*args, sep=" ", end="\n", **_):
            self.on_log(sep.join(str(a) for a in args))

        namespace = {
            "__name__": "__flight_plan__",
            "__builtins__": safe_builtins(),
            "drone": self._drone_cmds,
            "ai": self._ai,
            "api": self._drone_cmds,  # so pyhula-style `api.single_fly_forward(100)` also works
            "wait": self._drone_cmds.wait,
            "sleep": self._drone_cmds.wait,
            "print": _print,
        }
        self.on_log("▶ Running the Flight Plan")
        sys.settrace(self._tracer)
        try:
            exec(compile(self.code, FILENAME, "exec"), namespace)
            ok, message = True, "done"
            self.on_log(f"✅ Flight Plan finished ({self._ctx.commands} commands)")
        except MissionStopped as exc:
            message = str(exc)
            self.on_log(f"⏹ Stopped ({exc})")
        except FlightCommandError as exc:
            message = str(exc)
            self.error_line = self._last_line
            self.on_log(f"❌ Line {self._last_line}: {exc}")
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            line = self._last_line
            tb = traceback.extract_tb(exc.__traceback__)
            for fs in reversed(tb):
                if fs.filename == FILENAME:
                    line = fs.lineno
                    break
            self.error_line = line
            self.on_log(f"❌ Line {line}: {type(exc).__name__}: {exc}")
        finally:
            sys.settrace(None)
            # Never leave the drone hovering with nothing controlling it.
            if getattr(self.drone, "flying", False) and not getattr(self.drone, "emergency_stop_requested", False):
                self.auto_landed = True
                self.on_log("🛬 The code ended while the drone was flying — landing automatically")
                try:
                    self.drone.touchdown()
                except Exception as exc:
                    self.on_log(f"⚠️ landing error: {exc}")
            self.on_finished(ok, message)
