"""🧊 3D view of a Flight Plan: the drone, the path it flew and the path
✔ Check Code planned. Pure QPainter (a small perspective projection), so it
needs no OpenGL / Qt3D and draws the same in the packaged app and offscreen.

It polls a missions.flight_sim.SimTrack on its own timer - the flight runs in
a worker thread and never calls into Qt.

Mouse: drag = orbit, wheel = zoom, double-click = reset the camera.
"""
from __future__ import annotations

import math

from PySide6 import QtCore, QtGui, QtWidgets
from .pixel_theme import draw_pixel_rect

from ..i18n import tr
from ..missions.challenges import world_progress
from ..missions.flight_sim import SimTrack

BG_TOP = QtGui.QColor("#101a26")
BG_BOTTOM = QtGui.QColor("#070b10")
GRID_MINOR = QtGui.QColor(42, 60, 78)
GRID_MAJOR = QtGui.QColor(62, 88, 112)
ACCENT = QtGui.QColor("#67ddc3")
PLAN = QtGui.QColor("#f2c778")
TEXT = QtGui.QColor("#dce8f5")
MUTED = QtGui.QColor("#8ea0b2")
GOAL = QtGui.QColor("#ffb454")      # 🎯 mission objects still to do
DONE = QtGui.QColor("#5ef0a0")      # ... and done
HIT = QtGui.QColor("#ff6b7d")       # a building that was hit
HEAD = QtGui.QColor("#ff5a5f")      # the drone's head (nose cone): where it faces

DEFAULT_AZIMUTH = -60.0
DEFAULT_ELEVATION = 32.0
DEFAULT_DISTANCE = 430.0
DRONE_SCALE = 2.4  # draw the ~20 cm drone bigger than life so it reads at room scale
NEAR = 8.0


class _Camera:
    def __init__(self, target, azimuth, elevation, distance, width, height):
        a, e = math.radians(azimuth), math.radians(elevation)
        tx, ty, tz = target
        self.pos = (tx + distance * math.cos(e) * math.cos(a),
                    ty + distance * math.cos(e) * math.sin(a),
                    tz + distance * math.sin(e))
        f = _norm((tx - self.pos[0], ty - self.pos[1], tz - self.pos[2]))
        r = _norm(_cross(f, (0.0, 0.0, 1.0)))
        self.f, self.r, self.u = f, r, _cross(r, f)
        self.cx, self.cy = width / 2.0, height / 2.0
        self.focal = (height / 2.0) / math.tan(math.radians(30))

    def to_cam(self, p):
        v = (p[0] - self.pos[0], p[1] - self.pos[1], p[2] - self.pos[2])
        return _dot(v, self.r), _dot(v, self.u), _dot(v, self.f)

    def screen(self, c):
        x, y, z = c
        return QtCore.QPointF(self.cx + self.focal * x / z, self.cy - self.focal * y / z)

    def point(self, p):
        c = self.to_cam(p)
        return None if c[2] < NEAR else self.screen(c)

    def segment(self, a, b):
        """Both ends on screen, clipped at the near plane (None if behind)."""
        ca, cb = self.to_cam(a), self.to_cam(b)
        if ca[2] < NEAR and cb[2] < NEAR:
            return None
        if ca[2] < NEAR or cb[2] < NEAR:
            t = (NEAR - ca[2]) / (cb[2] - ca[2])
            cut = tuple(ca[i] + (cb[i] - ca[i]) * t for i in range(3))
            ca, cb = (cut, cb) if ca[2] < NEAR else (ca, cut)
        return self.screen(ca), self.screen(cb)

    def depth(self, p):
        return self.to_cam(p)[2]


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _norm(v):
    n = math.sqrt(_dot(v, v)) or 1.0
    return (v[0] / n, v[1] / n, v[2] / n)


def _body_transform(pose, scale=DRONE_SCALE):
    """Local drone point (right, forward, up) in cm -> world point."""
    yaw = math.radians(pose.yaw)
    pitch = math.radians(pose.flip_pitch)
    roll = math.radians(pose.flip_roll)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)

    def apply(p):
        x, y, z = (v * scale for v in p)
        # flip forward/back: rotate about the right axis (nose goes down first)
        y, z = y * cp + z * sp, -y * sp + z * cp
        # flip left/right: rotate about the forward axis
        x, z = x * cr - z * sr, x * sr + z * cr
        # yaw (CCW positive): right=(cos, sin), forward=(-sin, cos)
        wx = x * cy - y * sy
        wy = x * sy + y * cy
        return (pose.x + wx, pose.y + wy, max(0.0, pose.z) + z)
    return apply


class FlightSim3DView(QtWidgets.QWidget):
    def __init__(self, track: SimTrack | None = None, parent=None):
        super().__init__(parent)
        self.track = track or SimTrack()
        self.setMinimumSize(320, 220)
        self.setMouseTracking(False)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.follow = True
        self.show_takeoff_arrow = True   # the "forward at takeoff" hint at the start pad
        self.show_start_pad = True       # the "H" pad at the origin (Flight Plan takeoff point)
        self.show_position = True        # the X / Y line in the HUD (in this view's own axes)
        self.drone_scale = DRONE_SCALE
        self.reset_camera()
        self._drag = None
        self._rotor = 0.0
        self._version = -1
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(33)

    # -- camera ------------------------------------------------------------
    def reset_camera(self):
        self.azimuth = DEFAULT_AZIMUTH
        self.elevation = DEFAULT_ELEVATION
        self.distance = DEFAULT_DISTANCE
        self.target = [0.0, 60.0, 50.0]
        self.update()

    def frame_world(self, world):
        """Point the camera at the whole mission area (start pad included)."""
        self.reset_camera()
        if not world:
            return
        xs = [0.0] + [o.x for o in world]
        ys = [0.0] + [o.y for o in world]
        span = max(max(xs) - min(xs), max(ys) - min(ys), 150.0)
        self.target = [(max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2, 60.0]
        self.distance = max(DEFAULT_DISTANCE, span * 1.7)
        self.update()

    def set_follow(self, on: bool):
        self.follow = bool(on)
        self.update()

    def mousePressEvent(self, event):
        self._drag = event.position()

    def mouseMoveEvent(self, event):
        if self._drag is None:
            return
        d = event.position() - self._drag
        self._drag = event.position()
        self.azimuth -= d.x() * 0.4
        self.elevation = max(5.0, min(88.0, self.elevation + d.y() * 0.3))
        self.update()

    def mouseReleaseEvent(self, _event):
        self._drag = None

    def mouseDoubleClickEvent(self, _event):
        self.reset_camera()

    def wheelEvent(self, event):
        steps = event.angleDelta().y() / 120.0
        self.distance = max(120.0, min(4000.0, self.distance * (0.88 ** steps)))
        self.update()

    def _tick(self):
        if not self.isVisible():
            return
        snap = self.track.snapshot()
        moving = snap.pose.flying
        if moving:
            self._rotor = (self._rotor + 47.0) % 360.0
        if self.follow and snap.pose.flying:
            goal = (snap.pose.x, snap.pose.y, max(40.0, snap.pose.z * 0.7))
            for i in range(3):
                self.target[i] += (goal[i] - self.target[i]) * 0.12
            moving = moving or math.dist(goal, self.target) > 1.0
        if moving or snap.version != self._version:
            self._version = snap.version
            self.update()

    # -- drawing -----------------------------------------------------------
    def paintEvent(self, _event):
        snap = self.track.snapshot()
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        w, h = self.width(), self.height()
        bg = QtGui.QLinearGradient(0, 0, 0, h)
        bg.setColorAt(0, BG_TOP)
        bg.setColorAt(1, BG_BOTTOM)
        p.fillRect(self.rect(), bg)
        cam = _Camera(self.target, self.azimuth, self.elevation, self.distance, w, h)

        self._draw_ground(p, cam, snap)
        self._draw_world(p, cam, snap)
        self._draw_path(p, cam, snap.plan, PLAN, dashed=True)
        self._draw_path(p, cam, snap.path, ACCENT, dashed=False)
        for ph in snap.photos:
            s = cam.point(ph)
            if s is not None:
                p.setPen(QtGui.QPen(QtGui.QColor("#ffffff"), 1.5))
                p.setBrush(QtGui.QColor("#ff8fa3"))
                p.drawEllipse(s, 5, 5)
        self._draw_drone(p, cam, snap.pose)
        self._draw_hud(p, snap)
        if snap.banner:
            self._draw_banner(p, snap.banner)
        p.end()

    def _extent(self, snap):
        pts = snap.path + snap.plan + [(snap.pose.x, snap.pose.y, 0)] + [(o.x, o.y, 0) for o in snap.world]
        r = max(max(abs(x), abs(y)) for x, y, _ in pts) if pts else 0
        return max(200, int(math.ceil((r + 100) / 100.0)) * 100)

    def _draw_ground(self, p, cam, snap):
        ext = self._extent(snap)
        for v in range(-ext, ext + 1, 50):
            major = v % 100 == 0
            p.setPen(QtGui.QPen(GRID_MAJOR if major else GRID_MINOR, 1.2 if major else 0.8))
            for a, b in (((v, -ext, 0), (v, ext, 0)), ((-ext, v, 0), (ext, v, 0))):
                seg = cam.segment(a, b)
                if seg:
                    p.drawLine(*seg)
        # start pad: an "H" in a circle where the drone took off
        if not self.show_start_pad:
            return
        ring = [cam.point((22 * math.cos(t / 12 * math.tau), 22 * math.sin(t / 12 * math.tau), 0)) for t in range(13)]
        if None not in ring:
            p.setPen(QtGui.QPen(ACCENT, 2))
            p.setBrush(QtCore.Qt.NoBrush)
            p.drawPolyline(QtGui.QPolygonF(ring))
        for a, b in (((-9, -10, 0), (-9, 10, 0)), ((9, -10, 0), (9, 10, 0)), ((-9, 0, 0), (9, 0, 0))):
            seg = cam.segment(a, b)
            if seg:
                p.drawLine(*seg)
        # which way "forward" was at takeoff
        if not self.show_takeoff_arrow:
            return
        seg = cam.segment((0, 26, 0), (0, 70, 0))
        heads = [cam.segment((0, 70, 0), (sx * 10, 58, 0)) for sx in (-1, 1)]
        base = cam.point((30, -30, 0))
        if seg and all(heads) and base is not None:
            p.setPen(QtGui.QPen(QtGui.QColor(103, 221, 195, 160), 2))
            for line in [seg] + heads:
                p.drawLine(*line)
            small = p.font()
            small.setPointSizeF(8)
            p.setFont(small)
            p.setPen(MUTED)
            p.drawText(base, "↑ " + tr("forward at takeoff"))

    # -- 🎯 mission objects ---------------------------------------------------
    def _draw_world(self, p, cam, snap):
        if not snap.world:
            return
        # the flown path; before any flight, the path ✔ Check Code / 🏁 Grade planned
        # (not in the Survey Route view: there a waypoint is green only once flown)
        planned_counts = not snap.mode.startswith("route")
        done = world_progress(snap.world, snap.path if len(snap.path) > 2 or not planned_counts else snap.plan,
                              snap.photos)
        small = QtGui.QFont(p.font())
        small.setPointSizeF(9)
        small.setBold(True)
        p.setFont(small)
        # far objects first so near ones paint over them
        order = sorted(range(len(snap.world)), key=lambda i: -cam.depth((snap.world[i].x, snap.world[i].y,
                                                                          getattr(snap.world[i], "z", 0))))
        for obj in snap.world:
            if obj.kind == "area":
                self._draw_area(p, cam, obj)
        for i in order:
            obj = snap.world[i]
            hit = i in done
            if obj.kind == "area":
                continue
            if obj.kind == "box":
                self._draw_box(p, cam, obj, hit)
            elif obj.kind in ("pad", "spot"):
                landed = (obj.kind == "pad" and not snap.pose.flying
                          and math.hypot(snap.pose.x - obj.x, snap.pose.y - obj.y) <= obj.r and len(snap.path) > 2)
                ok = hit or landed
                col = DONE if ok else (GOAL if obj.kind == "pad" else QtGui.QColor("#ff8fa3"))
                ring = [cam.point((obj.x + obj.r * math.cos(t / 24 * math.tau),
                                   obj.y + obj.r * math.sin(t / 24 * math.tau), 0)) for t in range(24)]
                if None not in ring:
                    fill = QtGui.QColor(col)
                    fill.setAlpha(70 if obj.kind == "pad" else 40)
                    p.setBrush(fill)
                    p.setPen(QtGui.QPen(col, 2, QtCore.Qt.SolidLine if obj.kind == "pad" else QtCore.Qt.DashLine))
                    p.drawPolygon(QtGui.QPolygonF(ring))
                self._label(p, cam, (obj.x, obj.y, 0), obj.label, col)
            elif obj.kind == "checkpoint":
                col = DONE if hit else GOAL
                seg = cam.segment((obj.x, obj.y, 0), (obj.x, obj.y, obj.z))
                if seg:
                    p.setPen(QtGui.QPen(QtGui.QColor(col.red(), col.green(), col.blue(), 70), 1, QtCore.Qt.DotLine))
                    p.drawLine(*seg)
                c = cam.point((obj.x, obj.y, obj.z))
                edge = cam.point((obj.x + cam.r[0] * obj.r, obj.y + cam.r[1] * obj.r, obj.z + cam.r[2] * obj.r))
                if c is not None and edge is not None:
                    rad = max(4.0, math.hypot(edge.x() - c.x(), edge.y() - c.y()))
                    g = QtGui.QRadialGradient(c - QtCore.QPointF(rad * .3, rad * .3), rad * 1.3)
                    g.setColorAt(0, QtGui.QColor("#ffffff"))
                    g.setColorAt(0.35, col)
                    g.setColorAt(1, col.darker(220))
                    p.setPen(QtCore.Qt.NoPen)
                    p.setBrush(g)
                    p.drawEllipse(c, rad, rad)
                    if obj.label:
                        p.setPen(QtGui.QColor("#10161f"))
                        p.drawText(QtCore.QRectF(c.x() - rad, c.y() - rad, 2 * rad, 2 * rad), QtCore.Qt.AlignCenter, obj.label)
            elif obj.kind == "ring":
                col = DONE if hit else GOAL
                a = math.radians(obj.facing)
                t = (math.cos(a), math.sin(a), 0.0)
                pole = cam.segment((obj.x, obj.y, 0), (obj.x, obj.y, obj.z - obj.r))
                if pole:
                    p.setPen(QtGui.QPen(QtGui.QColor("#4d6274"), 3))
                    p.drawLine(*pole)
                pts = [cam.point((obj.x + obj.r * math.cos(k / 28 * math.tau) * t[0],
                                  obj.y + obj.r * math.cos(k / 28 * math.tau) * t[1],
                                  obj.z + obj.r * math.sin(k / 28 * math.tau))) for k in range(29)]
                if None not in pts:
                    glow = QtGui.QColor(col)
                    glow.setAlpha(60)
                    p.setBrush(QtCore.Qt.NoBrush)
                    p.setPen(QtGui.QPen(glow, 9))
                    p.drawPolyline(QtGui.QPolygonF(pts))
                    p.setPen(QtGui.QPen(col, 3.5))
                    p.drawPolyline(QtGui.QPolygonF(pts))
                number = sum(1 for o in snap.world[:i + 1] if o.kind == "ring")
                self._label(p, cam, (obj.x, obj.y, obj.z + obj.r + 8), str(number), col)

    def _draw_area(self, p, cam, area):
        corners = [(area.x, area.y, 0), (area.x + area.w, area.y, 0),
                   (area.x + area.w, area.y + area.d, 0), (area.x, area.y + area.d, 0)]
        scr = [cam.point(c) for c in corners]
        if None not in scr:
            p.setPen(QtCore.Qt.NoPen)
            p.setBrush(QtGui.QColor(103, 221, 195, 22))
            p.drawPolygon(QtGui.QPolygonF(scr))
        p.setPen(QtGui.QPen(QtGui.QColor("#9fb3c4"), 2.2))
        for a, b in zip(corners, corners[1:] + corners[:1]):
            seg = cam.segment(a, b)
            if seg:
                p.drawLine(*seg)
        if area.north:
            mid_x = area.x + area.w / 2
            top = area.y + area.d
            arrow = [cam.segment((mid_x, top + 12, 0), (mid_x, top + 60, 0)),
                     cam.segment((mid_x, top + 60, 0), (mid_x - 12, top + 44, 0)),
                     cam.segment((mid_x, top + 60, 0), (mid_x + 12, top + 44, 0))]
            p.setPen(QtGui.QPen(QtGui.QColor("#f2c778"), 2.4))
            for seg in arrow:
                if seg:
                    p.drawLine(*seg)
            self._label(p, cam, (mid_x, top + 78, 0), "N", QtGui.QColor("#f2c778"))
        if area.label:
            self._label(p, cam, (area.x + area.w, area.y - 18, 0), area.label, MUTED)

    def _label(self, p, cam, pos, text, col):
        s = cam.point(pos)
        if s is None or not text:
            return
        fm = QtGui.QFontMetrics(p.font())
        w = fm.horizontalAdvance(text) + 10
        rect = QtCore.QRectF(s.x() - w / 2, s.y() - fm.height() / 2 - 2, w, fm.height() + 4)
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QColor(10, 16, 24, 190))
        draw_pixel_rect(p, rect, 5)
        p.setPen(col)
        p.drawText(rect, QtCore.Qt.AlignCenter, text)

    def _draw_box(self, p, cam, box, hit):
        hx, hy = box.w / 2, box.d / 2
        corner = lambda i, j, k: (box.x + i * hx, box.y + j * hy, box.h if k else 0.0)
        faces = [
            ([(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)], (0, 0, 1)),
            ([(-1, 1, 0), (-1, 1, 1), (1, 1, 1), (1, 1, 0)], (0, 1, 0)),
            ([(-1, -1, 0), (1, -1, 0), (1, -1, 1), (-1, -1, 1)], (0, -1, 0)),
            ([(1, -1, 0), (1, 1, 0), (1, 1, 1), (1, -1, 1)], (1, 0, 0)),
            ([(-1, -1, 0), (-1, -1, 1), (-1, 1, 1), (-1, 1, 0)], (-1, 0, 0)),
        ]
        base = HIT if hit else QtGui.QColor("#6f8aa6")
        shown = []
        for idx, n in faces:
            pts = [corner(*c) for c in idx]
            mid = tuple(sum(q[k] for q in pts) / 4 for k in range(3))
            view = (cam.pos[0] - mid[0], cam.pos[1] - mid[1], cam.pos[2] - mid[2])
            if _dot(n, view) <= 0:
                continue
            light = 0.5 + 0.5 * max(0.0, _dot(n, _norm((0.4, -0.3, 0.86))))
            shown.append((cam.depth(mid), pts, light))
        for _, pts, light in sorted(shown, key=lambda it: -it[0]):
            scr = [cam.point(q) for q in pts]
            if None in scr:
                continue
            col = QtGui.QColor(int(base.red() * light), int(base.green() * light), int(base.blue() * light), 165)
            p.setBrush(col)
            p.setPen(QtGui.QPen(base.lighter(130), 1.2))
            p.drawPolygon(QtGui.QPolygonF(scr))
        # windows: a few lit dots on the tall ones make them read as buildings
        if box.h > 120 and not hit:
            p.setPen(QtCore.Qt.NoPen)
            p.setBrush(QtGui.QColor(255, 220, 140, 120))
            for zz in range(40, int(box.h) - 20, 40):
                for xx in (-hx * 0.5, 0, hx * 0.5):
                    s = cam.point((box.x + xx, box.y - hy - 0.5, zz))
                    if s is not None and cam.depth((box.x, box.y - hy, zz)) < cam.depth((box.x, box.y + hy, zz)):
                        p.drawRect(QtCore.QRectF(s.x() - 2, s.y() - 3, 4, 6))

    def _draw_banner(self, p, text):
        font = QtGui.QFont(p.font())
        font.setPointSizeF(15)
        font.setBold(True)
        p.setFont(font)
        fm = QtGui.QFontMetrics(font)
        lines = text.split("\n")
        w = max(fm.horizontalAdvance(t) for t in lines) + 40
        h = fm.height() * len(lines) + 24
        rect = QtCore.QRectF((self.width() - w) / 2, self.height() - h - 30, w, h)
        p.setPen(QtGui.QPen(QtGui.QColor("#67ddc3"), 1.5))
        p.setBrush(QtGui.QColor(8, 24, 26, 225))
        draw_pixel_rect(p, rect, 12)
        p.setPen(QtGui.QColor("#f4fbff"))
        p.drawText(rect, QtCore.Qt.AlignCenter, text)

    def _draw_path(self, p, cam, points, color, dashed):
        if len(points) < 2:
            return
        pen = QtGui.QPen(color, 1.6 if dashed else 2.6)
        if dashed:
            pen.setStyle(QtCore.Qt.DashLine)
            color = QtGui.QColor(color)
            color.setAlpha(170)
            pen.setColor(color)
        p.setPen(pen)
        for a, b in zip(points, points[1:]):
            seg = cam.segment(a, b)
            if seg:
                p.drawLine(*seg)

    def _draw_drone(self, p, cam, pose):
        # shadow + height line
        rad = 20 * self.drone_scale
        shadow = [cam.point((pose.x + rad * math.cos(t / 10 * math.tau), pose.y + rad * math.sin(t / 10 * math.tau), 0))
                  for t in range(10)]
        if None not in shadow:
            p.setPen(QtCore.Qt.NoPen)
            p.setBrush(QtGui.QColor(0, 0, 0, 110))
            p.drawPolygon(QtGui.QPolygonF(shadow))
        if pose.z > 2:
            seg = cam.segment((pose.x, pose.y, 0), (pose.x, pose.y, pose.z))
            if seg:
                p.setPen(QtGui.QPen(QtGui.QColor(142, 160, 178, 140), 1, QtCore.Qt.DotLine))
                p.drawLine(*seg)

        to_world = _body_transform(pose, self.drone_scale)
        led = QtGui.QColor(*pose.led)
        items = []  # (depth, kind, data)

        # body: a box 14 wide x 20 long x 6 high (cm), a bit larger than life to stay readable
        bx, by, bz = 7.0, 10.0, 3.0
        corners = {(i, j, k): to_world((i * bx, j * by, 3 + k * bz)) for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)}
        faces = [
            ([(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)], (0, 0, 1)),     # top
            ([(-1, -1, -1), (-1, 1, -1), (1, 1, -1), (1, -1, -1)], (0, 0, -1)),  # bottom
            ([(-1, 1, -1), (-1, 1, 1), (1, 1, 1), (1, 1, -1)], (0, 1, 0)),      # nose
            ([(-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)], (0, -1, 0)),  # tail
            ([(1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1)], (1, 0, 0)),      # right
            ([(-1, -1, -1), (-1, -1, 1), (-1, 1, 1), (-1, 1, -1)], (-1, 0, 0)),  # left
        ]
        center = to_world((0, 0, 3))
        for idx, normal in faces:
            pts = [corners[i] for i in idx]
            n_end = to_world((normal[0], normal[1], 3 + normal[2]))
            n = _norm((n_end[0] - center[0], n_end[1] - center[1], n_end[2] - center[2]))
            mid = tuple(sum(c[i] for c in pts) / 4 for i in range(3))
            view = _norm((cam.pos[0] - mid[0], cam.pos[1] - mid[1], cam.pos[2] - mid[2]))
            if _dot(n, view) <= 0:
                continue  # back face
            light = 0.45 + 0.55 * max(0.0, _dot(n, _norm((0.4, -0.3, 0.86))))
            base = QtGui.QColor("#e8eef5") if normal == (0, 0, 1) else QtGui.QColor("#9fb0c2")
            if normal == (0, 1, 0):
                base = HEAD  # the nose, mostly hidden by the head
            col = QtGui.QColor(int(base.red() * light), int(base.green() * light), int(base.blue() * light))
            items.append((cam.depth(mid), "poly", (pts, col)))
            if normal == (0, 0, 1):
                # forward arrow on the top face, pointing at the head (shows the LED colour)
                arrow = [to_world((0, 8, 6.05)), to_world((4.5, 0.5, 6.05)), to_world((1.6, 0.5, 6.05)),
                         to_world((1.6, -7, 6.05)), to_world((-1.6, -7, 6.05)), to_world((-1.6, 0.5, 6.05)),
                         to_world((-4.5, 0.5, 6.05))]
                items.append((cam.depth(mid) - 0.05, "poly", (arrow, QtGui.QColor(int(led.red() * light), int(led.green() * light), int(led.blue() * light)))))

        # head: a pointed nose cone in front of the body, with a camera at the tip,
        # so it is obvious which way the drone faces
        apex = to_world((0, 19, 3))
        head_mid = to_world((0, 13, 3))
        for a, b in (((-1, 1, 1), (1, 1, 1)), ((1, 1, 1), (1, 1, -1)),
                     ((1, 1, -1), (-1, 1, -1)), ((-1, 1, -1), (-1, 1, 1))):
            tri = [corners[a], corners[b], apex]
            n = _norm(_cross(tuple(tri[1][i] - tri[0][i] for i in range(3)), tuple(tri[2][i] - tri[0][i] for i in range(3))))
            mid = tuple(sum(c[i] for c in tri) / 3 for i in range(3))
            if _dot(n, tuple(mid[i] - head_mid[i] for i in range(3))) < 0:
                n = (-n[0], -n[1], -n[2])
            view = _norm((cam.pos[0] - mid[0], cam.pos[1] - mid[1], cam.pos[2] - mid[2]))
            if _dot(n, view) <= 0:
                continue
            light = 0.45 + 0.55 * max(0.0, _dot(n, _norm((0.4, -0.3, 0.86))))
            col = QtGui.QColor(int(HEAD.red() * light), int(HEAD.green() * light), int(HEAD.blue() * light))
            items.append((cam.depth(mid), "poly", (tri, col)))
        items.append((cam.depth(apex) - 0.1, "lens", (to_world((0, 18.2, 3)), 2.2 * self.drone_scale)))

        # arms + rotors
        arm = 15.0
        for sx, sy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
            hub = (sx * arm, sy * arm, 4.0)
            a, b = to_world((sx * 5, sy * 7, 3)), to_world(hub)
            items.append((cam.depth(b), "line", (a, b, QtGui.QColor("#5f7387"), 3.0)))
            ring = [to_world((hub[0] + 9 * math.cos(t / 14 * math.tau), hub[1] + 9 * math.sin(t / 14 * math.tau), 5.5))
                    for t in range(14)]
            front = sy > 0
            items.append((cam.depth(to_world(hub)) - 0.1, "rotor", (ring, front)))
            blade = math.radians(self._rotor * (1 if sx * sy > 0 else -1))
            ends = [to_world((hub[0] + 8 * math.cos(blade + k * math.pi), hub[1] + 8 * math.sin(blade + k * math.pi), 5.6))
                    for k in (0, 1)]
            items.append((cam.depth(to_world(hub)) - 0.2, "line", (ends[0], ends[1], QtGui.QColor("#dfe7ef"), 1.6)))

        for _, kind, data in sorted(items, key=lambda it: -it[0]):
            if kind == "poly":
                pts, col = data
                scr = [cam.point(q) for q in pts]
                if None not in scr:
                    p.setPen(QtGui.QPen(col.darker(140), 1))
                    p.setBrush(col)
                    p.drawPolygon(QtGui.QPolygonF(scr))
            elif kind == "line":
                a, b, col, width = data
                seg = cam.segment(a, b)
                if seg:
                    p.setPen(QtGui.QPen(col, width, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
                    p.drawLine(*seg)
            elif kind == "lens":
                pos, size = data
                scr = cam.point(pos)
                if scr is not None:
                    r = max(2.0, cam.focal * size / max(NEAR, cam.depth(pos)))
                    p.setPen(QtGui.QPen(QtGui.QColor("#0b1118"), 1))
                    p.setBrush(QtGui.QColor("#1d2a38"))
                    p.drawEllipse(scr, r, r)
                    p.setPen(QtCore.Qt.NoPen)
                    p.setBrush(QtGui.QColor(160, 220, 255, 200))
                    p.drawEllipse(scr + QtCore.QPointF(-r * 0.3, -r * 0.3), r * 0.3, r * 0.3)
            else:
                ring, front = data
                scr = [cam.point(q) for q in ring]
                if None not in scr:
                    p.setPen(QtGui.QPen(QtGui.QColor("#f2a0ae") if front else QtGui.QColor("#93a7bb"), 1.2))
                    p.setBrush(QtGui.QColor(200, 220, 240, 45))
                    p.drawPolygon(QtGui.QPolygonF(scr))

        if pose.flying:
            glow = cam.point(to_world((0, 0, 8)))
            if glow is not None:
                g = QtGui.QRadialGradient(glow, 40)
                c = QtGui.QColor(led)
                c.setAlpha(90)
                g.setColorAt(0, c)
                c.setAlpha(0)
                g.setColorAt(1, c)
                p.setPen(QtCore.Qt.NoPen)
                p.setBrush(g)
                p.drawEllipse(glow, 40, 40)

    def _draw_hud(self, p, snap):
        pose = snap.pose
        badge = {"sim": tr("🧊 SIMULATION — not the real drone"),
                 "route": tr("🗺️ SURVEY ROUTE — preview, not flying"),
                 "route_live": tr("📡 LIVE — real drone position")}.get(snap.mode, tr("📡 MIRROR — real drone commands"))
        font = p.font()
        font.setPointSizeF(9.5)
        font.setBold(True)
        p.setFont(font)
        fm = QtGui.QFontMetrics(font)
        rect = QtCore.QRectF(10, 10, fm.horizontalAdvance(badge) + 18, fm.height() + 8)
        calm = snap.mode in ("sim", "route")
        p.setPen(QtGui.QPen(QtGui.QColor("#54462d") if calm else QtGui.QColor("#295647"), 1))
        p.setBrush(QtGui.QColor("#362d1e") if calm else QtGui.QColor("#15392f"))
        draw_pixel_rect(p, rect, 8)
        p.setPen(QtGui.QColor("#f2c778") if calm else QtGui.QColor("#7ee9cf"))
        p.drawText(rect, QtCore.Qt.AlignCenter, badge)

        font.setBold(False)
        font.setPointSizeF(9)
        p.setFont(font)
        heading = (-pose.yaw) % 360
        lines = [
            (f"X {pose.x / 100:+.2f} m   Y {pose.y / 100:+.2f} m   " if self.show_position else "")
            + tr("height") + f" {pose.z / 100:.2f} m",
            tr("heading") + f" {heading:.0f}°   " + tr("camera") + f" {pose.camera}°   "
            + (tr("flying") if pose.flying else tr("on the ground")),
        ]
        if snap.line:
            lines.append(tr("line") + f" {snap.line}" + (f"  •  {snap.label}" if snap.label else ""))
        elif snap.label:
            lines.append(snap.label)  # e.g. the Survey Route's "WP 3/9 • EAST"
        p.setPen(TEXT)
        y = rect.bottom() + 18
        for text in lines:
            p.drawText(QtCore.QPointF(12, y), text)
            y += 16

        p.setPen(MUTED)
        hint = tr("drag = rotate  •  wheel = zoom  •  double-click = reset")
        p.drawText(QtCore.QPointF(12, self.height() - 10), hint)
        lx = self.width() - 12
        plan_text = tr("planned route") if snap.mode.startswith("route") else tr("planned (Check Code)")
        for text, col, dashed in ((plan_text, PLAN, True), (tr("flown"), ACCENT, False)):
            tw = QtGui.QFontMetrics(font).horizontalAdvance(text)
            pen = QtGui.QPen(col, 2, QtCore.Qt.DashLine if dashed else QtCore.Qt.SolidLine)
            p.setPen(pen)
            p.drawLine(QtCore.QPointF(lx - tw - 30, 22), QtCore.QPointF(lx - tw - 8, 22))
            p.setPen(TEXT)
            p.drawText(QtCore.QPointF(lx - tw, 26), text)
            lx -= tw + 46
