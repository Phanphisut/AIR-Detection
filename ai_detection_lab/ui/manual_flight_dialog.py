from __future__ import annotations
import math
import threading
from PySide6 import QtCore, QtGui, QtWidgets

from pyhulax.core.types import Direction
from .. import config
from ..i18n import tr


class ManualFlightDialog(QtWidgets.QDialog):
    """Developer Mode only: fly the drone by keyboard instead of the
    autonomous grid mission. No waypoints, no geofence - the
    operator is the only safety check, which is why main_window.py only
    ever lets this open once "🔧 Developer Mode" has been switched on.

    Every drone call here (connect/takeoff/land/emergency_stop and the two
    manual_* methods below) already goes through HulaDroneController's own
    RLock and preflight guards, so this dialog just maps keys to those
    calls - it does not talk to pyhulax directly.

    The top-down radar view is dead reckoning, not telemetry: neither
    HulaDroneController nor pyhulax's manual RC path reports back a live
    X/Y position, so this dialog tracks its own running (x, y, yaw) purely
    from the step size of each command it just sent and draws THAT. It is
    a "here's roughly where your commands should have put it" aid, not a
    GPS fix - real drift (wind, uneven strafes) is not reflected.
    """

    status_signal = QtCore.Signal(str)
    state_signal = QtCore.Signal()

    _LEGEND = (
        f"W/S/A/D {tr('or')} ← →  :  {tr('Forward')} / {tr('Back')} / {tr('Left')} / {tr('Right')}\n"
        f"↑ / ↓  {tr('or')} R / F  :  {tr('Up')} / {tr('Down')}\n"
        f"Q / E  :  {tr('Rotate Left')} / {tr('Rotate Right')}\n"
        f"T  :  {tr('Takeoff')}          L  :  {tr('Land')}\n"
        f"SPACE {tr('or')} ESC  :  🛑 {tr('EMERGENCY STOP')}"
    )

    _MOVE_KEYS = {
        QtCore.Qt.Key_W: (Direction.FORWARD, tr("FORWARD")),
        QtCore.Qt.Key_S: (Direction.BACK, tr("BACK")),
        QtCore.Qt.Key_A: (Direction.LEFT, tr("LEFT")),
        QtCore.Qt.Key_Left: (Direction.LEFT, tr("LEFT")),
        QtCore.Qt.Key_D: (Direction.RIGHT, tr("RIGHT")),
        QtCore.Qt.Key_Right: (Direction.RIGHT, tr("RIGHT")),
    }
    _ALTITUDE_KEYS = {
        QtCore.Qt.Key_R: (Direction.UP, tr("UP")),
        QtCore.Qt.Key_Up: (Direction.UP, tr("UP")),
        QtCore.Qt.Key_F: (Direction.DOWN, tr("DOWN")),
        QtCore.Qt.Key_Down: (Direction.DOWN, tr("DOWN")),
    }

    def __init__(self, drone, parent=None):
        super().__init__(parent)
        self.drone = drone
        self._busy = False

        # Dead-reckoning pose for the radar view only - see class
        # docstring. World frame: +X = east/right, +Y = north/up (screen),
        # yaw measured clockwise from north, matching a normal map/compass.
        self._x_cm = 0.0
        self._y_cm = 0.0
        self._yaw_deg = 0.0
        self._altitude_cm = 0
        self._trail = [(0.0, 0.0)]

        self.setWindowTitle(tr("⌨️ Keyboard Control"))
        self.setModal(False)
        self.resize(520, 720)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

        from .panel_theme import DIALOG_STYLE, prepare, heading
        self.setStyleSheet(DIALOG_STYLE)
        self.resize(640, 850)
        root = QtWidgets.QVBoxLayout(self)
        prepare(root)
        heading(root, tr("Manual flight"), tr("Developer controls · keyboard and position estimate"), "drone")

        warn = QtWidgets.QLabel(tr(
            "⚠️ Keyboard flight — no waypoints, no geofence, nothing stops the "
            "drone but you. AI Detection keeps running on the main window. "
            "Click inside this window first so it has keyboard focus."
        ))
        warn.setWordWrap(True)
        warn.setStyleSheet("color:#ffd166; font-weight:700;")
        root.addWidget(warn)

        self.state_label = QtWidgets.QLabel()
        self.state_label.setStyleSheet("font-weight:800; padding:4px 0;")
        root.addWidget(self.state_label)

        legend = QtWidgets.QLabel(self._LEGEND)
        legend.setStyleSheet("font-family: Menlo, monospace; font-size:13px;")
        root.addWidget(legend)

        self.radar = _DroneRadarWidget(self)
        self.radar.setMinimumHeight(260)
        root.addWidget(self.radar, stretch=1)

        btn_row = QtWidgets.QHBoxLayout()
        self.takeoff_btn = QtWidgets.QPushButton(tr("🛫 Takeoff (T)"))
        self.takeoff_btn.clicked.connect(self._takeoff)
        # A focused QPushButton eats Space/Enter itself (clicks the button)
        # instead of passing the key up to this dialog's keyPressEvent -
        # and clicking a button is exactly how it gets focus in the first
        # place. Without this, Space stops being "EMERGENCY STOP" the
        # moment a student clicks Takeoff or Land with the mouse; only Esc
        # would still work. NoFocus leaves clicking these buttons working
        # exactly as before, it just stops them from ever holding keyboard
        # focus, so every key always reaches keyPressEvent below.
        self.takeoff_btn.setFocusPolicy(QtCore.Qt.NoFocus)
        btn_row.addWidget(self.takeoff_btn)
        self.land_btn = QtWidgets.QPushButton(tr("🛬 Land (L)"))
        self.land_btn.clicked.connect(self._land)
        self.land_btn.setFocusPolicy(QtCore.Qt.NoFocus)
        btn_row.addWidget(self.land_btn)
        root.addLayout(btn_row)

        self.estop_btn = QtWidgets.QPushButton(tr("🛑 EMERGENCY STOP (Space / Esc)"))
        self.estop_btn.setStyleSheet(
            "background:#c0392b; color:white; font-weight:800; padding:10px;"
        )
        self.estop_btn.clicked.connect(self._emergency_stop)
        self.estop_btn.setFocusPolicy(QtCore.Qt.NoFocus)
        root.addWidget(self.estop_btn)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(200)
        self.log.setMaximumHeight(140)
        root.addWidget(self.log)

        self.status_signal.connect(self.log.appendPlainText)
        self.state_signal.connect(self._refresh_state)
        self._refresh_state()

    def showEvent(self, event):
        super().showEvent(event)
        self.setFocus()
        self._refresh_state()

    def _refresh_state(self):
        conn = tr("🟢 Connected") if self.drone.connected else tr("⚪ Not connected")
        fly = tr("🚁 FLYING") if self.drone.flying else tr("on ground")
        self.state_label.setText(
            f"{conn}  —  {fly}  —  ALT {self._altitude_cm} cm"
        )
        self.radar.set_pose(self._x_cm, self._y_cm, self._yaw_deg, self._trail, self.drone.flying)

    # --- keyboard -----------------------------------------------------

    def keyPressEvent(self, event):
        # Auto-repeat would otherwise queue a new blocking move()/rotate()
        # for every ~30ms OS repeat tick while a key is held - ignore those
        # and require a fresh physical keydown per step, "tap to move" feel.
        if event.isAutoRepeat():
            return
        key = event.key()

        if key in (QtCore.Qt.Key_Space, QtCore.Qt.Key_Escape):
            self._emergency_stop()
            return
        if key == QtCore.Qt.Key_T:
            self._takeoff()
            return
        if key == QtCore.Qt.Key_L:
            self._land()
            return
        if key in self._MOVE_KEYS:
            direction, label = self._MOVE_KEYS[key]
            self._dispatch_move(direction, label, config.MANUAL_FLIGHT_MOVE_STEP_CM)
            return
        if key in self._ALTITUDE_KEYS:
            direction, label = self._ALTITUDE_KEYS[key]
            self._dispatch_move(direction, label, config.MANUAL_FLIGHT_ALTITUDE_STEP_CM)
            return
        if key == QtCore.Qt.Key_Q:
            self._dispatch_rotate(config.MANUAL_FLIGHT_ROTATE_DEG, tr("LEFT"))
            return
        if key == QtCore.Qt.Key_E:
            self._dispatch_rotate(-config.MANUAL_FLIGHT_ROTATE_DEG, tr("RIGHT"))
            return
        super().keyPressEvent(event)

    # --- dispatch (each command runs off the GUI thread since move()/
    # rotate() block for as long as the physical step takes; _busy drops
    # any keypress that arrives while the previous command is still in
    # flight instead of piling up a queue of them) -----------------------

    def _dispatch_move(self, direction, label, step_cm):
        def run():
            ok = self.drone.manual_move(direction, step_cm)
            if ok:
                self._apply_move(direction, step_cm)
            self.status_signal.emit(f"{'✅' if ok else '❌'} {label} {step_cm}cm")
            self.state_signal.emit()
            self._busy = False

        self._run_exclusive(run)

    def _dispatch_rotate(self, degrees, label):
        def run():
            ok = self.drone.manual_rotate(degrees)
            if ok:
                # Turning left (CCW, positive `degrees`) REDUCES the
                # clockwise-from-north yaw this radar uses; turning right
                # (CW, negative `degrees`) increases it - so subtracting
                # `degrees` handles both signs correctly in one line.
                self._yaw_deg = (self._yaw_deg - degrees) % 360
            self.status_signal.emit(f"{'✅' if ok else '❌'} {tr('ROTATE')} {label} {abs(degrees)}°")
            self.state_signal.emit()
            self._busy = False

        self._run_exclusive(run)

    def _apply_move(self, direction, step_cm):
        if direction == Direction.UP:
            self._altitude_cm += step_cm
            return
        if direction == Direction.DOWN:
            self._altitude_cm = max(0, self._altitude_cm - step_cm)
            return
        yaw_rad = math.radians(self._yaw_deg)
        forward = (math.sin(yaw_rad), math.cos(yaw_rad))
        right = (math.cos(yaw_rad), -math.sin(yaw_rad))
        sign, vec = {
            Direction.FORWARD: (1, forward),
            Direction.BACK: (-1, forward),
            Direction.RIGHT: (1, right),
            Direction.LEFT: (-1, right),
        }[direction]
        self._x_cm += sign * vec[0] * step_cm
        self._y_cm += sign * vec[1] * step_cm
        self._trail.append((self._x_cm, self._y_cm))
        del self._trail[:-200]

    def _run_exclusive(self, run):
        if self._busy:
            return
        self._busy = True
        threading.Thread(target=run, daemon=True).start()

    # --- takeoff / land / emergency stop --------------------------------

    def _takeoff(self):
        def run():
            if not self.drone.connected:
                self.status_signal.emit(tr("📡 Connecting..."))
                if not self.drone.connect():
                    self.status_signal.emit(tr("❌ Connect failed"))
                    self.state_signal.emit()
                    return
            if not self.drone.camera_down_ok:
                self.drone.set_camera_down()
            ok = self.drone.takeoff()
            if ok:
                self._altitude_cm = config.SURVEY_ALTITUDE_CM
            self.status_signal.emit(tr("🛫 Takeoff OK") if ok else tr("❌ Takeoff failed"))
            self.state_signal.emit()

        threading.Thread(target=run, daemon=True).start()

    def _land(self):
        def run():
            ok = self.drone.touchdown()
            if ok:
                self._altitude_cm = 0
            self.status_signal.emit(tr("🛬 Landed") if ok else tr("❌ Land error"))
            self.state_signal.emit()

        threading.Thread(target=run, daemon=True).start()

    def _emergency_stop(self):
        def run():
            self.drone.emergency_stop()
            self._altitude_cm = 0
            self.status_signal.emit(tr("🛑 Emergency stop executed"))
            self.state_signal.emit()

        threading.Thread(target=run, daemon=True).start()

    def closeEvent(self, event):
        # No other keyboard control exists once this window is gone -
        # leaving the drone airborne with nothing left to fly it would be
        # worse than landing without being asked.
        if self.drone.flying:
            self.status_signal.emit(tr("🛬 Closing — landing first..."))
            self.drone.touchdown()
            self._altitude_cm = 0
            self._refresh_state()
        super().closeEvent(event)


class _DroneRadarWidget(QtWidgets.QWidget):
    """Top-down dead-reckoning view of the dialog's own (x, y, yaw) pose -
    see ManualFlightDialog's docstring for what that is and isn't. Visual
    style (dark ground, teal route, red current-position marker) matches
    _GridPreviewWidget elsewhere in this app so both flight views read as
    one family."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background:#071018;border:1px solid #263442;")
        self._x_cm = 0.0
        self._y_cm = 0.0
        self._yaw_deg = 0.0
        self._trail = [(0.0, 0.0)]
        self._flying = False

    def set_pose(self, x_cm, y_cm, yaw_deg, trail, flying: bool):
        self._x_cm = x_cm
        self._y_cm = y_cm
        self._yaw_deg = yaw_deg
        self._trail = list(trail)
        self._flying = flying
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QtGui.QColor("#071018"))

        margin = 30
        area = self.rect().adjusted(margin, margin, -margin, -margin)
        if area.width() <= 0 or area.height() <= 0:
            return

        xs = [p[0] for p in self._trail] + [0.0]
        ys = [p[1] for p in self._trail] + [0.0]
        # Autoscale to whatever the drone has actually covered, always
        # including the launch point, with a minimum span so a stationary
        # drone right after connecting doesn't render at infinite zoom.
        half_span = max(100.0, max(max(xs) - min(xs), max(ys) - min(ys)) / 2 + 40)
        cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2

        def pos(x, y):
            px = area.center().x() + (x - cx) / half_span * (area.width() / 2)
            # Screen Y grows downward; world +Y (north) must go up.
            py = area.center().y() - (y - cy) / half_span * (area.height() / 2)
            return QtCore.QPointF(px, py)

        painter.setPen(QtGui.QPen(QtGui.QColor("#7f8c9a"), 2))
        painter.drawRect(area)

        # Grid lines every 50cm.
        painter.setPen(QtGui.QPen(QtGui.QColor("#22313d"), 1))
        step = 50.0
        start_gx = math.floor((cx - half_span) / step) * step
        gx = start_gx
        while gx <= cx + half_span:
            a, b = pos(gx, cy - half_span), pos(gx, cy + half_span)
            painter.drawLine(a, b)
            gx += step
        start_gy = math.floor((cy - half_span) / step) * step
        gy = start_gy
        while gy <= cy + half_span:
            a, b = pos(cx - half_span, gy), pos(cx + half_span, gy)
            painter.drawLine(a, b)
            gy += step

        # Launch point.
        launch_pt = pos(0.0, 0.0)
        painter.setBrush(QtGui.QColor("#ffd166"))
        painter.setPen(QtGui.QPen(QtGui.QColor("#ffd166"), 2))
        painter.drawEllipse(launch_pt, 6, 6)
        painter.setPen(QtGui.QColor("#d8dee9"))
        painter.drawText(launch_pt + QtCore.QPointF(9, 4), tr("LAUNCH"))

        # Flown trail.
        if len(self._trail) > 1:
            painter.setPen(QtGui.QPen(QtGui.QColor("#55d6be"), 2))
            for a, b in zip(self._trail, self._trail[1:]):
                painter.drawLine(pos(*a), pos(*b))

        # Drone marker: a triangle pointing along its current yaw.
        drone_pt = pos(self._x_cm, self._y_cm)
        yaw_rad = math.radians(self._yaw_deg)
        size = 11.0
        tip = QtCore.QPointF(drone_pt.x() + size * math.sin(yaw_rad), drone_pt.y() - size * math.cos(yaw_rad))
        back_l = QtCore.QPointF(
            drone_pt.x() + size * math.sin(yaw_rad + 2.5), drone_pt.y() - size * math.cos(yaw_rad + 2.5)
        )
        back_r = QtCore.QPointF(
            drone_pt.x() + size * math.sin(yaw_rad - 2.5), drone_pt.y() - size * math.cos(yaw_rad - 2.5)
        )
        color = QtGui.QColor("#ff5f6d") if self._flying else QtGui.QColor("#9aa7b2")
        painter.setBrush(color)
        painter.setPen(QtGui.QPen(QtGui.QColor("#fff0ef"), 1.5))
        painter.drawPolygon(QtGui.QPolygonF([tip, back_l, back_r]))

        # Compass + legend.
        font = painter.font()
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QtGui.QColor("#d8dee9"))
        painter.drawText(12, 22, f"{tr('N')} ↑")
        painter.drawText(12, self.height() - 12, f"{int(self._x_cm)}, {int(self._y_cm)} cm  •  {tr('yaw')} {int(self._yaw_deg)}°")
        status = tr("🚁 FLYING (dead-reckoning, not GPS)") if self._flying else tr("on ground")
        painter.drawText(self.width() - 8 - QtGui.QFontMetrics(font).horizontalAdvance(status), 22, status)
