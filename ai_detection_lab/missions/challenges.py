"""🎯 Mission Guide for the 🐍 Flight Plan: game-like levels for students.

Each Mission has a story, objects placed in the 3D world (rings, landing pads,
buildings, photo spots, checkpoints), starter code, hints, a solution, and
objectives that are graded on a dry run of the student's code (the same
instant VirtualDrone ✔ Check Code uses - never the real drone):

    ★      every required objective
    ★★     + bonus 1
    ★★★    + bonus 2

Missions that use ai.sees() are graded in more than one "scenario" (what the
pretend AI sees), so both sides of the `if` must be right.

Coordinates are the simulation's: x = right of the start pad, y = ahead of
it, z = height, all cm; takeoff hovers at config.SURVEY_ALTITUDE_CM (100).
"""
from __future__ import annotations

import ast
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from .flight_sim import check_plan

HOME = (0.0, 0.0)
DRONE_RADIUS_CM = 10.0


# =====================================================================
# World objects (drawn by ui/flight_sim_view.py)
# =====================================================================
@dataclass
class Ring:
    """Fly THROUGH it. `facing` = the yaw of the direction to pass (0 = +y,
    90 = -x, -90 = +x)."""
    x: float
    y: float
    z: float
    r: float = 32.0
    facing: float = 0.0
    kind: str = "ring"

    def normal(self):
        a = math.radians(self.facing)
        return (-math.sin(a), math.cos(a), 0.0)


@dataclass
class Pad:
    """Land on it."""
    x: float
    y: float
    r: float = 30.0
    label: str = "B"
    kind: str = "pad"


@dataclass
class Box:
    """A building: don't touch it. (x, y) is the centre of its footprint."""
    x: float
    y: float
    w: float
    d: float
    h: float
    kind: str = "box"


@dataclass
class Spot:
    """Take a drone.photo() above it."""
    x: float
    y: float
    r: float = 30.0
    label: str = "📸"
    kind: str = "spot"


@dataclass
class Checkpoint:
    """Fly through this point in the air."""
    x: float
    y: float
    z: float
    r: float = 25.0
    label: str = ""
    kind: str = "checkpoint"


@dataclass
class Area:
    """A flat rectangle on the floor: the survey field (Survey Route view).
    (x, y) is its lower-left corner; `north` marks the +y edge as north."""
    x: float
    y: float
    w: float
    d: float
    label: str = ""
    north: bool = False
    kind: str = "area"


# =====================================================================
# Geometry on the flown path (list of (x, y, z) points)
# =====================================================================
def _seg_point_dist(a, b, p):
    ab = [b[i] - a[i] for i in range(3)]
    ap = [p[i] - a[i] for i in range(3)]
    n = sum(v * v for v in ab)
    t = 0.0 if n == 0 else max(0.0, min(1.0, sum(ab[i] * ap[i] for i in range(3)) / n))
    q = [a[i] + ab[i] * t for i in range(3)]
    return math.dist(q, p)


def first_reach(path, obj):
    """Index of the first path segment that passes the object, else None."""
    if obj.kind == "checkpoint":
        target = (obj.x, obj.y, obj.z)
        if path and math.dist(path[0], target) <= obj.r:
            return 0
        for i, (a, b) in enumerate(zip(path, path[1:])):
            if _seg_point_dist(a, b, target) <= obj.r:
                return i
    elif obj.kind == "ring":
        n, c = obj.normal(), (obj.x, obj.y, obj.z)
        for i, (a, b) in enumerate(zip(path, path[1:])):
            da = sum((a[k] - c[k]) * n[k] for k in range(3))
            db = sum((b[k] - c[k]) * n[k] for k in range(3))
            if da < 0 <= db or db < 0 <= da:
                t = da / (da - db) if da != db else 0.0
                q = [a[k] + (b[k] - a[k]) * t for k in range(3)]
                if math.dist(q, c) <= obj.r - DRONE_RADIUS_CM * 0.5:
                    return i
    return None


def hits_box(path, box, margin=DRONE_RADIUS_CM):
    """True if any path segment (ignoring the drone sitting on the floor)
    goes through the building's volume grown by the drone's size."""
    lo = (box.x - box.w / 2 - margin, box.y - box.d / 2 - margin, -1.0)
    hi = (box.x + box.w / 2 + margin, box.y + box.d / 2 + margin, box.h + margin)
    for a, b in zip(path, path[1:]):
        t0, t1 = 0.0, 1.0
        for k in range(3):
            d = b[k] - a[k]
            if abs(d) < 1e-9:
                if a[k] < lo[k] or a[k] > hi[k]:
                    break
                continue
            u0, u1 = (lo[k] - a[k]) / d, (hi[k] - a[k]) / d
            if u0 > u1:
                u0, u1 = u1, u0
            t0, t1 = max(t0, u0), min(t1, u1)
            if t0 > t1:
                break
        else:
            if t1 - t0 > 1e-6 or (t0 == t1 and 0 <= t0 <= 1):
                return True
    return False


def world_progress(world, path, photos):
    """Indices of world objects already done on this (live) path - lit green in 3D."""
    done = set()
    for i, obj in enumerate(world):
        if obj.kind in ("ring", "checkpoint"):
            if first_reach(path, obj) is not None:
                done.add(i)
        elif obj.kind == "spot":
            if any(math.hypot(px - obj.x, py - obj.y) <= obj.r for px, py, *_ in photos):
                done.add(i)
        elif obj.kind == "box":
            if hits_box(path, obj):
                done.add(i)  # "done" for a building = it was HIT (drawn red)
    return done


COLORS = {
    "red": (255, 0, 0), "orange": (255, 128, 0), "yellow": (255, 230, 0), "green": (0, 255, 0),
    "cyan": (0, 255, 255), "blue": (0, 0, 255), "purple": (160, 0, 255), "pink": (255, 0, 160),
    "white": (255, 255, 255), "off": (0, 0, 0),
}
COLOR_TH = {"red": "red", "orange": "orange", "yellow": "yellow", "green": "green", "cyan": "sky blue",
            "blue": "blue", "purple": "purple", "pink": "pink", "white": "white", "off": "off"}


def color_name(rgb):
    return min(COLORS, key=lambda n: sum((rgb[i] - COLORS[n][i]) ** 2 for i in range(3)))


# =====================================================================
# One graded run
# =====================================================================
@dataclass
class Flight:
    code: str
    report: object
    tree: ast.AST | None

    @property
    def ok(self):
        return self.report.ok and self.report.drone is not None

    @property
    def drone(self):
        return self.report.drone

    @property
    def path(self):
        return self.drone.points if self.ok else []

    @property
    def landed_at(self):
        """(x, y) where the code itself landed, else None (crashed / auto-land)."""
        if not self.ok or self.report.auto_landed or self.drone.flying:
            return None
        return self.drone.pose.x, self.drone.pose.y

    def calls(self, *names):
        if self.tree is None:
            return False
        return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                   and n.func.attr in names for n in ast.walk(self.tree))

    def has(self, *node_types):
        return self.tree is not None and any(isinstance(n, node_types) for n in ast.walk(self.tree))

    def code_lines(self):
        return sum(1 for line in self.code.splitlines() if line.strip() and not line.strip().startswith("#"))


# Objective: (text, check(flights) -> (ok, detail)). `flights` maps a
# scenario name ("main", or what the pretend AI sees) to a Flight.
@dataclass
class Objective:
    text: str
    check: object

    def run(self, flights):
        try:
            return self.check(flights)
        except Exception as exc:  # a broken run must read as "not yet", never crash the UI
            return False, f"Can't check: {exc}"


def _main(flights):
    return flights["main"]


def land_on(x, y, r=30.0, name="goal"):
    def check(flights):
        f = _main(flights)
        at = f.landed_at
        if at is None:
            return False, "drone.land() was not called to land"
        d = math.hypot(at[0] - x, at[1] - y)
        return d <= r, f"Landed {d:.0f} cm from the {name}" + ("" if d <= r else f" (must be ≤ {r:.0f})")
    return check


def land_home():
    return land_on(*HOME, name="pad H")


def objects_in_order(indices, world, label):
    def check(flights):
        path = _main(flights).path
        start, got = 0, 0
        for i in indices:
            # search from the segment where the previous one was passed
            k = first_reach(path[start:], world[i])
            if k is None:
                break
            start += k
            got += 1
        return got == len(indices), f"Passed {label} {got}/{len(indices)}"
    return check


def reach_height(cm):
    def check(flights):
        top = max((p[2] for p in _main(flights).path), default=0)
        return top >= cm - 1, f"Highest {top:.0f} cm (must be ≥ {cm})"
    return check


def max_height(cm):
    def check(flights):
        top = max((p[2] for p in _main(flights).path), default=0)
        return top <= cm + 1, f"Highest {top:.0f} cm (must be ≤ {cm})"
    return check


def no_crash(boxes, world):
    def check(flights):
        path = _main(flights).path
        hit = [i for i in boxes if hits_box(path, world[i])]
        return bool(path) and not hit, "Didn't hit any building" if not hit else f"Hit {len(hit)} buildings 💥"
    return check


def commands_at_most(n):
    def check(flights):
        c = _main(flights).report.commands
        return c <= n, f"Used {c} commands (goal ≤ {n})"
    return check


def lines_at_most(n):
    def check(flights):
        c = _main(flights).code_lines()
        return c <= n, f"Code is {c} lines (goal ≤ {n}, comments not counted)"
    return check


def seconds_at_most(n):
    def check(flights):
        s = _main(flights).report.seconds
        return s <= n, f"Took {s:.1f} s (goal ≤ {n} s)"
    return check


def uses_loop():
    def check(flights):
        ok = _main(flights).has(ast.For, ast.While)
        return ok, "Used a for/while loop 🔁" if ok else "No for/while loop yet"
    return check


def uses_call(name, text):
    def check(flights):
        ok = _main(flights).calls(name)
        return ok, text if ok else f"{name}() not used yet"
    return check


def hover_seconds(n):
    def check(flights):
        s = _main(flights).drone.wait_seconds if _main(flights).ok else 0
        return s >= n, f"Hovered {s:g} s (must be ≥ {n})"
    return check


def led_sequence(names):
    def check(flights):
        f = _main(flights)
        seen = [color_name(rgb) for rgb, _, flying in (f.drone.led_events if f.ok else []) if flying]
        i = 0
        for c in seen:
            if i < len(names) and c == names[i]:
                i += 1
        want = " → ".join(COLOR_TH[n] for n in names)
        return i == len(names), f"Lights while flying {i}/{len(names)} colors ({want})"
    return check


def led_spread(n, apart=40.0):
    def check(flights):
        f = _main(flights)
        spots = []
        for _, pos, flying in (f.drone.led_events if f.ok else []):
            if flying and all(math.dist(pos, s) >= apart for s in spots):
                spots.append(pos)
        return len(spots) >= n, f"Changed lights at {len(spots)} places (goal {n})"
    return check


def led_last(name):
    def check(flights):
        f = _main(flights)
        flying = [rgb for rgb, _, fl in (f.drone.led_events if f.ok else [])]
        ok = bool(flying) and color_name(flying[-1]) == name
        return ok, f"Last light is {COLOR_TH[name]}" if ok else f"Last light is not {COLOR_TH[name]} yet"
    return check


def photos_at(spot_indices, world, camera_down=False):
    def check(flights):
        f = _main(flights)
        shots = f.drone.photo_events if f.ok else []
        got = 0
        for i in spot_indices:
            s = world[i]
            if any(math.hypot(x - s.x, y - s.y) <= s.r and (not camera_down or cam >= 80)
                   for x, y, z, cam in shots):
                got += 1
        what = "Photo spots 📸 (camera down 90°)" if camera_down else "Photo spots 📸"
        return got == len(spot_indices), f"{what} {got}/{len(spot_indices)}"
    return check


def flips(n, min_z, same_color=None):
    def check(flights):
        f = _main(flights)
        ev = f.drone.flip_events if f.ok else []
        good = [e for e in ev if e[1] >= min_z - 1 and (same_color is None or color_name(e[2]) == same_color)]
        kinds = {e[0] for e in good}
        extra = f" with {COLOR_TH[same_color]} light" if same_color else ""
        return len(kinds) >= n, f"Flips in different directions{extra} at ≥ {min_z} cm: {len(kinds)}/{n}"
    return check


def faces_start():
    def check(flights):
        f = _main(flights)
        yaw = f.drone.pose.yaw % 360 if f.ok else 1
        ok = min(yaw, 360 - yaw) < 1
        return ok, "Facing the starting way at the end 🧭" if ok else f"Facing {(-yaw) % 360:.0f}° at the end (must be 0°)"
    return check


def uses_else():
    def check(flights):
        tree = _main(flights).tree
        ok = tree is not None and any(isinstance(n, ast.If) and n.orelse for n in ast.walk(tree))
        return ok, "Has else ✓" if ok else "No else yet"
    return check


def uses_def():
    def check(flights):
        ok = _main(flights).has(ast.FunctionDef)
        return ok, "Has a def function ✓" if ok else "No def yet"
    return check


def scenario(name, title, *checks):
    """All `checks` must pass in scenario `name` (what the pretend AI sees)."""
    def check(flights):
        sub = {"main": flights[name]}
        for c in checks:
            ok, detail = c(sub)
            if not ok:
                return False, f"{title}: {detail}"
        return True, f"{title}: correct"
    return check


def no_photo():
    def check(flights):
        f = _main(flights)
        n = len(f.drone.photo_events) if f.ok else 0
        return f.ok and n == 0, "No photo taken" if n == 0 else f"Took {n} photos even though nobody was found"
    return check


# =====================================================================
# Missions
# =====================================================================
@dataclass
class Mission:
    id: str
    emoji: str
    title: str
    level: int                 # 1 = ง่าย, 2 = กลาง, 3 = ยาก
    teaches: str
    story: str
    goal: str
    world: list
    starter: str
    hints: list
    solution: str
    objectives: list           # required -> ★
    bonus: list                # 2 items -> ★★, ★★★
    scenarios: dict = field(default_factory=lambda: {"main": ()})

    @property
    def name(self):
        return f"{self.emoji} {self.title}"


@dataclass
class Grade:
    stars: int
    required: list             # [(text, ok, detail)]
    bonus: list
    errors: list               # Check Code errors (the code did not run)

    @property
    def passed(self):
        return self.stars > 0


def grade(mission: Mission, code: str) -> Grade:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        tree = None
    flights = {}
    errors = []
    for name, ai in mission.scenarios.items():
        report = check_plan(code, list(ai))
        flights[name] = Flight(code, report, tree)
        for issue in report.errors:
            where = f"Line {issue.line}: " if issue.line else ""
            if f"{where}{issue.message}" not in errors:
                errors.append(f"{where}{issue.message}")
    if errors:
        req = [(o.text, False, "The code doesn't pass ✔ Check code yet") for o in mission.objectives]
        return Grade(0, req, [(o.text, False, "") for o in mission.bonus], errors)
    req = [(o.text, *o.run(flights)) for o in mission.objectives]
    bon = [(o.text, *o.run(flights)) for o in mission.bonus]
    stars = 0
    if all(ok for _, ok, _ in req):
        stars = 1 + sum(1 for _, ok, _ in bon if ok)
    return Grade(stars, req, bon, [])


def _star_points(n=5, side=100.0, turn=144.0):
    """Corners of the star drawn by forward(side) + turn_right(turn), n times."""
    x = y = yaw = 0.0
    pts = []
    for _ in range(n):
        x, y = x - side * math.sin(math.radians(yaw)), y + side * math.cos(math.radians(yaw))
        pts.append((x, y))
        yaw -= turn
    return pts


def _build():
    M = []

    # 1 ----------------------------------------------------------------
    w1 = [Checkpoint(0, 0, 150, 20, "1")]
    M.append(Mission(
        "first_flight", "🛫", "First flight", 1, "Sequence",
        "Welcome, trainee pilot! The control tower has cleared your first flight. Fly up and touch the golden ball above the pad, hover and enjoy the view, then land gently.",
        "Fly up to touch the golden ball at 150 cm, then land on pad H",
        w1,
        "# 🛫 First flight\ndrone.takeoff()      # take off (hovers at ~100 cm)\n\n# TODO: fly up another 50 cm\n\n# TODO: hover for 2 seconds\n\ndrone.land()\n",
        ["Fly up with drone.up(distance in cm)",
         "takeoff already hovers at 100 cm, so go up 150 - 100 = 50 cm more",
         "Hover with drone.wait(seconds)"],
        "drone.takeoff()\ndrone.up(50)\ndrone.wait(2)\ndrone.land()\n",
        [Objective("Touch the golden ball (150 cm high)", objects_in_order([0], w1, "ball")),
         Objective("Land on pad H", land_home())],
        [Objective("Hover and enjoy the view ≥ 2 seconds", hover_seconds(2)),
         Objective("Use 4 commands or fewer", commands_at_most(4))],
    ))

    # 2 ----------------------------------------------------------------
    M.append(Mission(
        "delivery", "📦", "Express delivery", 1, "Distance (cm)",
        "The pizza shop called! A customer is waiting on pad B, 2 meters ahead. Deliver before the pizza gets cold, and turn on a green light to say \"it's here!\"",
        "Fly to pad B, 200 cm ahead, and land there",
        [Pad(0, 200, 30, "B")],
        "# 📦 Express delivery — pad B is 200 cm ahead\ndrone.takeoff()\n\n# TODO: fly forward to pad B\n\ndrone.land()\n",
        ["Go forward with drone.forward(distance in cm)",
         "1 meter = 100 cm → 2 meters = 200 cm",
         "Green light is drone.led(0, 255, 0). Put it before land()"],
        "drone.takeoff()\ndrone.forward(200)\ndrone.led(0, 255, 0)\ndrone.land()\n",
        [Objective("Land on pad B", land_on(0, 200, 30, "pad B"))],
        [Objective("Green light on delivery 💚", led_last("green")),
         Objective("Use 4 commands or fewer", commands_at_most(4))],
    ))

    # 3 ----------------------------------------------------------------
    w3 = [Checkpoint(0, 150, 100, 25, "angle"), Pad(150, 150, 30, "C")]
    M.append(Mission(
        "corner", "↪️", "Turn the corner", 1, "Turning",
        "Pad C is hidden around the corner of a building. Fly to the corner, turn right, then land right on it — pro pilots always point the drone where it's going.",
        "Pass the corner flag (150 cm ahead), then land on pad C (150 cm right of the corner)",
        w3,
        "# ↪️ Turn the corner\ndrone.takeoff()\ndrone.forward(150)   # reach the corner\n\n# TODO: turn right, then fly to pad C\n\ndrone.land()\n",
        ["Turn right with drone.turn_right(degrees) — a corner is 90 degrees",
         "After a turn, the drone's \"forward\" changes too",
         "turn_right(90) then forward(150)"],
        "drone.takeoff()\ndrone.forward(150)\ndrone.turn_right(90)\ndrone.forward(150)\ndrone.land()\n",
        [Objective("Pass the corner flag", objects_in_order([0], w3, "corner")),
         Objective("Land on pad C", land_on(150, 150, 30, "pad C"))],
        [Objective("Really turn with turn_right()", uses_call("turn_right", "Turned with turn_right() ✓")),
         Objective("Use 5 commands or fewer", commands_at_most(5))],
    ))

    # 4 ----------------------------------------------------------------
    w4 = [Checkpoint(0, 150, 100, 25, "1"), Checkpoint(150, 150, 100, 25, "2"),
          Checkpoint(150, 0, 100, 25, "3")]
    M.append(Mission(
        "square", "🟦", "Square patrol", 2, "for loop (repeat)",
        "The night guard must fly once around the landing area, past all 3 lamp posts, and back to base. Notice the commands repeat in groups… is there a shorter way to write it? 🤔",
        "Pass lamp posts 1 → 2 → 3 (a 150 × 150 cm square), then land back on pad H",
        w4,
        "# 🟦 Square patrol\ndrone.takeoff()\n\ndrone.forward(150)\ndrone.turn_right(90)\n# TODO: finish all 4 sides\n\ndrone.land()\n",
        ["A square has 4 sides. Each side = forward(150) then turn_right(90)",
         "To repeat 4 times use  for i in range(4):  and indent the commands inside by 4 spaces",
         "for i in range(4):\n    drone.forward(150)\n    drone.turn_right(90)"],
        "drone.takeoff()\nfor i in range(4):\n    drone.forward(150)\n    drone.turn_right(90)\ndrone.land()\n",
        [Objective("Pass lamp posts 1 → 2 → 3", objects_in_order([0, 1, 2], w4, "lamp posts")),
         Objective("Land back on pad H", land_home())],
        [Objective("Use a for loop", uses_loop()),
         Objective("5 lines of code or fewer", lines_at_most(5))],
    ))

    # 5 ----------------------------------------------------------------
    w5 = [Ring(0, 100, 100, 32, 0), Ring(0, 200, 150, 32, 0), Ring(-100, 250, 100, 32, 90),
          Pad(-150, 250, 30, "🏁")]
    M.append(Mission(
        "rings", "💍", "Through the fire rings", 2, "3D: up-down + turns",
        "The school drone race has started! 3 fire rings float at different heights. Fly through them in order and reach the finish 🏁 as fast as you can.",
        "Fly through rings 1 → 2 → 3, then land at the finish 🏁",
        w5,
        "# 💍 Through the fire rings\n# Ring 1: 100 cm ahead, 100 cm high\n# Ring 2: 200 cm ahead, 150 cm high\n# Ring 3: on the left after a turn (x = -100, y = 250), 100 cm high\ndrone.takeoff()\n\n# TODO\n\ndrone.land()\n",
        ["Ring 2 is 50 cm higher than ring 1 — up(50) after ring 1 but before ring 2",
         "forward(150), then up(50), then forward(100) → you're at y = 250, 150 high",
         "Then turn_left(90), down(50), forward(150) goes through ring 3 to the finish"],
        "drone.takeoff()\ndrone.forward(150)\ndrone.up(50)\ndrone.forward(100)\n"
        "drone.turn_left(90)\ndrone.down(50)\ndrone.forward(150)\ndrone.land()\n",
        [Objective("Through rings 1 → 2 → 3", objects_in_order([0, 1, 2], w5, "rings")),
         Objective("Land at the finish 🏁", land_on(-150, 250, 30, "finish"))],
        [Objective("Use 8 commands or fewer", commands_at_most(8)),
         Objective("Finish within 20 seconds ⏱", seconds_at_most(20))],
    ))

    # 6 ----------------------------------------------------------------
    w6 = [Box(0, 100, 80, 60, 260), Box(100, 200, 60, 60, 70), Box(-110, 160, 60, 80, 260),
          Pad(0, 260, 30, "🏥")]
    M.append(Mission(
        "city", "🏢", "Dodge the buildings", 2, "Route planning",
        "Rush medicine to the hospital 🏥 — but a tall building is in the way! You can fly over low buildings, but must go around tall ones. Don't touch anything.",
        "Land at the hospital 🏥 (260 cm ahead) without hitting a building",
        w6,
        "# 🏢 Dodge the buildings — a tall one is right in front!\ndrone.takeoff()\n\n# TODO: go around the tall building and land at 🏥\n\ndrone.land()\n",
        ["Press ✔ Check code and look at the yellow path in 3D to see if it hits a building",
         "The tall building in the middle is 80 cm wide — moving 100 cm right clears it",
         "The low building on the right is only 70 cm tall, so flying at 100 cm clears it: right(100), forward(200), forward(60), left(100)"],
        "drone.takeoff()\ndrone.right(100)\ndrone.forward(200)\ndrone.forward(60)\ndrone.left(100)\ndrone.land()\n",
        [Objective("Don't hit a building", no_crash([0, 1, 2], w6)),
         Objective("Land at the hospital 🏥", land_on(0, 260, 30, "🏥"))],
        [Objective("Fly low, 120 cm or less (saves battery)", max_height(120)),
         Objective("Use 6 commands or fewer", commands_at_most(6))],
    ))

    # 7 ----------------------------------------------------------------
    M.append(Mission(
        "rainbow", "🌈", "Rainbow light show", 2, "lists + loops",
        "There's a festival tonight and our drone opens the show! Fly forward step by step and change the light red → yellow → green → blue so the crowd says wow ✨",
        "While flying, change the light red → yellow → green → blue, then land back on pad H",
        [Checkpoint(0, 50, 100, 18, "🔴"), Checkpoint(0, 100, 100, 18, "🟡"),
         Checkpoint(0, 150, 100, 18, "🟢"), Checkpoint(0, 200, 100, 18, "🔵")],
        "# 🌈 Rainbow light show\ndrone.takeoff()\n\ndrone.forward(50)\ndrone.led(255, 0, 0)     # red\n# TODO: yellow, green, blue, one step each\n\n# back to base\ndrone.back(200)\ndrone.land()\n",
        ["Colors (R, G, B): yellow (255, 230, 0), green (0, 255, 0), blue (0, 0, 255)",
         "Keep the colors in a list: colors = [(255,0,0), (255,230,0), (0,255,0), (0,0,255)]",
         "for r, g, b in colors:\n    drone.forward(50)\n    drone.led(r, g, b)"],
        "drone.takeoff()\ncolors = [(255, 0, 0), (255, 230, 0), (0, 255, 0), (0, 0, 255)]\n"
        "for r, g, b in colors:\n    drone.forward(50)\n    drone.led(r, g, b)\ndrone.back(200)\ndrone.land()\n",
        [Objective("Lights red → yellow → green → blue", led_sequence(["red", "yellow", "green", "blue"])),
         Objective("Land back on pad H", land_home())],
        [Objective("Change lights at 4 different places", led_spread(4)),
         Objective("Use a for loop", uses_loop())],
    ))

    # 8 ----------------------------------------------------------------
    w8 = [Spot(100, 0, 30, "A"), Spot(100, 150, 30, "B"), Spot(-50, 150, 30, "C")]
    M.append(Mission(
        "detective", "📸", "Photo detective", 2, "functions (def)",
        "There are 3 mysterious footprints in the field! Take a photo of each one from above. Real detectives always point the camera down before a photo 🕵️",
        "Take photos (drone.photo()) above spots A, B, C, then land back on pad H",
        w8,
        "# 📸 Photo detective\n# A: right 100 cm   B: right 100 + forward 150   C: left 50 + forward 150\ndrone.takeoff()\ndrone.camera(90)       # point the camera at the floor\n\ndrone.right(100)\ndrone.photo()          # spot A\n\n# TODO: go to spots B and C\n\n# TODO: back to base\ndrone.land()\n",
        ["Careful! The shortest move is 5 cm and it can't be negative. Use left/right/forward/back for direction",
         "A to B: forward(150)  B to C: left(150)  C back to base: back(150) + right(50)",
         "Make a helper function:  def snap():\n    drone.wait(1)\n    drone.photo()"],
        "drone.takeoff()\ndrone.camera(90)\n\ndef snap():\n    drone.wait(1)\n    drone.photo()\n\n"
        "drone.right(100)\nsnap()\ndrone.forward(150)\nsnap()\ndrone.left(150)\nsnap()\n"
        "drone.back(150)\ndrone.right(50)\ndrone.land()\n",
        [Objective("Photos of spots A, B, C", photos_at([0, 1, 2], w8)),
         Objective("Land back on pad H", land_home())],
        [Objective("Camera down 90° before every photo", photos_at([0, 1, 2], w8, camera_down=True)),
         Objective("Make your own function (def)", uses_def())],
    ))

    # 9 ----------------------------------------------------------------
    star = _star_points()
    w9 = [Checkpoint(x, y, 100, 22, str(i + 1)) for i, (x, y) in enumerate(star)]
    M.append(Mission(
        "star", "⭐", "Draw a star in the sky", 3, "outside angles + loops",
        "An artist wants the drone to draw a 5-point star in the air for a long-exposure photo! The secret is the turning angle… it's definitely not 90 degrees 📐",
        "Fly past all 5 star points in order (100 cm per side), then land on pad H",
        w9,
        "# ⭐ Draw a star in the sky — 100 cm per side\ndrone.takeoff()\n\n# TODO: fly a 5-point star\n# the star's turning angle = ?\n\ndrone.land()\n",
        ["A 5-point star is: fly straight, turn right, repeat 5 times",
         "The whole star turns 720° (2 full turns) ÷ 5 points = ? degrees each time",
         "for i in range(5):\n    drone.forward(100)\n    drone.turn_right(144)"],
        "drone.takeoff()\nfor i in range(5):\n    drone.forward(100)\n    drone.turn_right(144)\ndrone.land()\n",
        [Objective("Pass star points 1 → 5", objects_in_order(list(range(5)), w9, "star points")),
         Objective("Land on pad H", land_home())],
        [Objective("Use a for loop", uses_loop()),
         Objective("5 lines of code or fewer", lines_at_most(5))],
    ))

    # 10 ---------------------------------------------------------------
    w10 = [Checkpoint(0, 100, 125, 22, "1"), Checkpoint(-100, 100, 150, 22, "2"),
           Checkpoint(-100, 0, 175, 22, "3"), Checkpoint(0, 0, 200, 22, "4")]
    M.append(Mission(
        "spiral", "🌀", "Spiral stairs to the sky", 3, "loops with several commands",
        "4 energy balls form a spiral staircase, each 25 cm higher than the last. Fly counterclockwise to collect them all, then come back down to base.",
        "Collect balls 1 → 4 (circle left, 25 cm higher each time), then land on pad H",
        w10,
        "# 🌀 Spiral stairs to the sky\ndrone.takeoff()\n\n# first round:\ndrone.forward(100)\ndrone.up(25)\ndrone.turn_left(90)\n# TODO: 3 more rounds (or use a loop!)\n\ndrone.land()\n",
        ["The balls sit at the corners of a 100 × 100 cm square, each corner 25 cm higher",
         "1 round = forward(100), up(25), turn_left(90)",
         "for i in range(4):\n    drone.forward(100)\n    drone.up(25)\n    drone.turn_left(90)"],
        "drone.takeoff()\nfor i in range(4):\n    drone.forward(100)\n    drone.up(25)\n    drone.turn_left(90)\ndrone.land()\n",
        [Objective("Collect balls 1 → 4", objects_in_order([0, 1, 2, 3], w10, "ball")),
         Objective("Land on pad H", land_home())],
        [Objective("Use a for loop", uses_loop()),
         Objective("Face the starting way at the end", faces_start())],
    ))

    # 11 ---------------------------------------------------------------
    w11 = [Spot(0, 150, 35, "?")]
    M.append(Mission(
        "rescue", "🤖", "AI rescue team", 3, "if / else + AI",
        'Someone may be trapped at spot ? in the middle of the field! Fly there and point the camera down for the AI. If the AI sees a person: red light 🚨 and a photo for the rescue team. If not: green light ✅, then back to base.\nThis mission checks 2 cases: the AI sees a "person", and the AI sees nobody.',
        'Go to spot ? → if ai.sees("person"): red light + photo, else: green light → land back on pad H',
        w11,
        '# 🤖 AI rescue team\ndrone.takeoff()\ndrone.forward(150)\ndrone.camera(90)\ndrone.wait(1)          # let the AI look\n\n# TODO: if the AI sees "person" → red light + photo\n#       else → green light\n\ndrone.back(150)\ndrone.land()\n',
        ['Use  if ai.sees("person"):  and indent the commands inside',
         "For the not-found case use  else:",
         "if ai.sees(\"person\"):\n    drone.led(255, 0, 0)\n    drone.photo()\nelse:\n    drone.led(0, 255, 0)"],
        "drone.takeoff()\ndrone.forward(150)\ndrone.camera(90)\ndrone.wait(1)\n"
        "if ai.sees(\"person\"):\n    drone.led(255, 0, 0)\n    drone.photo()\nelse:\n    drone.led(0, 255, 0)\n"
        "drone.back(150)\ndrone.land()\n",
        [Objective("AI sees a person → red light + photo at spot ?",
                   scenario("person", "person found", led_last("red"), photos_at([0], w11), land_home())),
         Objective("AI sees nobody → green light, no photo",
                   scenario("nobody", "nobody found", led_last("green"), no_photo(), land_home()))],
        [Objective("Camera down 90° before the photo", scenario("person", "camera", photos_at([0], w11, camera_down=True))),
         Objective("Use else", uses_else())],
        scenarios={"main": ("person",), "person": ("person",), "nobody": ()},
    ))

    # 12 ---------------------------------------------------------------
    M.append(Mission(
        "acrobat", "🤸", "Sky acrobat", 3, "safety + order",
        "Closing show of sports day! Climb high enough (≥ 150 cm), then flip forward and back with a purple light for style 💜 — flipping too low is dangerous, open space only.",
        "Flip in at least 2 directions at ≥ 150 cm, then land on pad H",
        [Checkpoint(0, 0, 150, 20, "150")],
        "# 🤸 Sky acrobat (flips in open space only)\ndrone.takeoff()\n\n# TODO: climb high enough, turn on a purple light, then flip in 2 directions\n\ndrone.land()\n",
        ["drone.flip(\"forward\") / \"back\" / \"left\" / \"right\"",
         "takeoff = 100 cm, so up(50) before flipping",
         "Purple light: drone.led(160, 0, 255)"],
        "drone.takeoff()\ndrone.up(50)\ndrone.led(160, 0, 255)\ndrone.flip(\"forward\")\ndrone.flip(\"back\")\ndrone.land()\n",
        [Objective("Flip in 2 directions at ≥ 150 cm", flips(2, 150)),
         Objective("Land on pad H", land_home())],
        [Objective("Flip with a purple light 💜", flips(2, 150, "purple")),
         Objective("Use 6 commands or fewer", commands_at_most(6))],
    ))

    # 13 ---------------------------------------------------------------
    w13 = [Checkpoint(80, 120, 140, 22, "📍")]
    M.append(Mission(
        "lost", "🏠", "Bring the drone home", 3, "thinking backwards",
        "The drone is lost! Someone wrote code that took it to spot 📍 but forgot to bring it back. Don't change the code that's there — add code at the end to bring it back to pad H, facing the way it started.",
        "Pass 📍 with the existing code, then come back and land on pad H",
        w13,
        "# 🏠 Bring the drone home — don't change these 5 lines\ndrone.takeoff()\ndrone.forward(120)\ndrone.turn_right(90)\ndrone.forward(80)\ndrone.up(40)\n# ---- the drone is at 📍 now, facing right ----\n\n# TODO: bring it home\n\ndrone.land()\n",
        ["The drone now faces +x (right of the start) — press ✔ Check code and rotate the 3D view",
         "back(80) returns to the first line, then turn_left(90) to turn back",
         "back(80) → turn_left(90) → back(120) → land()"],
        "drone.takeoff()\ndrone.forward(120)\ndrone.turn_right(90)\ndrone.forward(80)\ndrone.up(40)\n"
        "drone.back(80)\ndrone.turn_left(90)\ndrone.back(120)\ndrone.land()\n",
        [Objective("Pass 📍 with the existing code", objects_in_order([0], w13, "📍")),
         Objective("Land back on pad H", land_home())],
        [Objective("Face the starting way at the end", faces_start()),
         Objective("Use 9 commands or fewer", commands_at_most(9))],
    ))
    return M


MISSIONS = _build()
MISSIONS_BY_ID = {m.id: m for m in MISSIONS}
MAX_STARS = 3 * len(MISSIONS)

RANKS = [  # (stars needed, title)
    (0, "🐣 Trainee pilot"),
    (6, "🧑‍✈️ Co-pilot"),
    (15, "🛩️ Captain"),
    (25, "🚀 Astronaut"),
    (MAX_STARS, "🏆 Sky ace"),
]
LEVELS = {1: "Easy", 2: "Medium", 3: "Hard"}


def rank(total_stars):
    return [t for need, t in RANKS if total_stars >= need][-1]


class Progress:
    """Best stars per mission, saved next to the flight plans."""

    def __init__(self, path: Path):
        self.path = Path(path)
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.stars = {k: int(v) for k, v in data.get("stars", {}).items() if k in MISSIONS_BY_ID}
        except Exception:
            self.stars = {}

    @property
    def total(self):
        return sum(self.stars.values())

    def record(self, mission_id, stars) -> bool:
        """Returns True if this is a new best."""
        if stars <= self.stars.get(mission_id, 0):
            return False
        self.stars[mission_id] = stars
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"stars": self.stars}, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
        return True
