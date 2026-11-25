"""Mission windows: 🐍 Flight Plan (Python) and ⌨️ Keyboard Control.

Both are non-modal windows that sit next to the main window. Each one has
its own "🔌 Connect" step (it calls the main window's normal Start AI
Camera path, so the drone connection, video and AI Detection are the same
ones the rest of the app uses) - a student never has to go back to the main
window to get started. They never talk to pyhulax directly - only through
the HulaDroneController / SimulatedDroneController the app already has.
"""
from __future__ import annotations

import re
import threading
import time
from pathlib import Path

import cv2
from PySide6 import QtCore, QtGui, QtWidgets
from pyhulax.core.types import Direction

from ..text_overlay import put_text

from .. import config
from ..i18n import tr
from ..missions.flight_plan import EXAMPLE_CODE, FlightPlanRunner
from ..missions.flight_sim import MirroredDrone, SimTrack, VirtualDrone, check_plan, fake_detections
from .code_editor import CodeTextEdit, DIALOG_STYLESHEET, PythonHighlighter
from ..missions.challenges import grade as grade_mission_code
from .flight_sim_view import FlightSim3DView
from .mission_guide import MissionGuidePanel

FLIGHT_PLAN_DIR = config.DATA_ROOT / "flight_plans"
DEFAULT_PLAN = FLIGHT_PLAN_DIR / "my_flight_plan.py"

BLANK_CODE = '# 🐍 My Flight Plan\n# The drone follows the commands "one line at a time, top to bottom"\n# Press a command button on the right to insert it at the cursor, then change the numbers\n\ndrone.takeoff()\n\n# ⬇️ Put your commands here\n\n\ndrone.land()\n'

# (button label, code to insert). Multi-line snippets keep their own indent.
# LED_PICKER instead opens a colour picker and inserts drone.led(R, G, B).
LED_PICKER = None
LED_CALL = re.compile(r"drone\.led\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)")
SIM_SPEEDS = [("×0.5", 0.5), ("×1", 1.0), ("×2", 2.0), ("×4", 4.0)]

COMMAND_BUTTONS = [
    ("🛫 Take off", "drone.takeoff()"),
    ("🛬 Land", "drone.land()"),
    ("⬆️ Forward 50 cm", "drone.forward(50)"),
    ("⬇️ Back 50 cm", "drone.back(50)"),
    ("⬅️ Left 50 cm", "drone.left(50)"),
    ("➡️ Right 50 cm", "drone.right(50)"),
    ("🔼 Up 30 cm", "drone.up(30)"),
    ("🔽 Down 30 cm", "drone.down(30)"),
    ("↩️ Turn left 90°", "drone.turn_left(90)"),
    ("↪️ Turn right 90°", "drone.turn_right(90)"),
    ("⏳ Wait 2 seconds", "drone.wait(2)"),
    ("🎨 Pick LED color…", LED_PICKER),
    ("📷 Camera down", "drone.camera(90)"),
    ("📷 Camera forward", "drone.camera(0)"),
    ("📸 Take a photo", "drone.photo()"),
    ("🔁 Repeat 4 times (square)", "for i in range(4):\n    drone.forward(50)\n    drone.turn_right(90)"),
    ("🤖 If the AI sees...", 'if ai.sees("person"):\n    print("Found", ai.count("person"))\n    drone.photo()\nelse:\n    print("Not found")'),
    ("💬 Show a message", 'print("Hello")'),
]


def _state_style(ok: bool) -> str:
    return "color:#8ce8c0; font-weight:800;" if ok else "color:#ffd166; font-weight:800;"


class _MissionWindow(QtWidgets.QDialog):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowFlag(QtCore.Qt.Window, True)
        self.setModal(False)
        self.setStyleSheet(DIALOG_STYLESHEET)

    def is_busy(self) -> bool:
        return False


# =====================================================================
# Live drone video with the AI boxes (shared by both mission windows)
# =====================================================================
class LiveAIView(QtWidgets.QWidget):
    """The drone camera as the main window's AI pipeline sees it: every box
    drawn here comes from the SAME detections the main window uses (the .pt
    models chosen in 🧠 Choose Models). It also says plainly when there is
    no video, instead of a frozen last picture."""

    STALE_SEC = 3.0

    def __init__(self, min_size=(480, 320), parent=None):
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        from .mission_widgets import VideoSurface
        self.video = VideoSurface(tr("Camera offline"), tr("Connect the drone to receive live video."))
        self.video.setAlignment(QtCore.Qt.AlignCenter)
        self.video.setWordWrap(True)
        self.video.setMinimumSize(*min_size)
        self.video.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.video.setStyleSheet(
            "background:#05080c; border:1px solid #1a2530; border-radius:8px; color:#8ea0b2; font-size:12pt;"
        )
        lay.addWidget(self.video, 1)
        self.det_label = QtWidgets.QLabel(tr("🤖 AI sees: —"))
        self.det_label.setWordWrap(True)
        self.det_label.setStyleSheet("font-size:12pt; font-weight:800; color:#8ce8c0; padding:4px;")
        lay.addWidget(self.det_label)
        self._has_frame = False
        self.set_message(tr("📷 Press ① Connect — the drone camera with the AI boxes appears here"))

    def set_message(self, text):
        if not self._has_frame and self.video.text() == text:
            return  # already showing it - avoid needless repaints
        self._has_frame = False
        self.video.clear()
        self.video.setText(text)

    def show_frame(self, frame, detections):
        if frame is None:
            return
        img = frame.copy()
        names = []
        for d in detections or []:
            xyxy = d.get("xyxy")
            if not xyxy or len(xyxy) != 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            label = f"{d.get('class_name', '?')} {float(d.get('confidence', 0)):.0%}"
            names.append(label)
            cv2.rectangle(img, (x1, y1), (x2, y2), (85, 214, 190), 3)
            put_text(img, label, (x1, max(22, y1 - 8)), 0.7, (223, 252, 247), 2)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888).copy()
        self.video.setPixmap(QtGui.QPixmap.fromImage(qimg).scaled(
            self.video.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation))
        self._has_frame = True
        self.det_label.setText(
            "🤖 " + tr("AI sees") + f" ({len(names)}): " + (", ".join(names) if names else "—")
        )

    def update_state(self, connected: bool, video_age: float):
        """Called by the window's timer: replace a frozen picture with a
        clear message when the video is not (or no longer) arriving."""
        if not connected:
            self.set_message(tr("📷 Press ① Connect — the drone camera with the AI boxes appears here"))
            self.det_label.setText(tr("🤖 AI sees: —"))
        elif video_age == float("inf"):
            self.set_message(tr("⏳ Connected — waiting for the drone video (a few seconds)..."))
        elif video_age > self.STALE_SEC:
            self.set_message(tr("⚠️ No video for a few seconds — check the drone's Wi-Fi. "
                                "If it doesn't come back, press ① Connect again."))
            self.det_label.setText(tr("🤖 AI sees: —"))


def _video_ok(age) -> bool:
    return age is not None and age <= LiveAIView.STALE_SEC


# =====================================================================
# 🐍 Flight Plan (Python)
# =====================================================================
class FlightPlanDialog(_MissionWindow):
    log_signal = QtCore.Signal(str)
    line_signal = QtCore.Signal(int)
    finished_signal = QtCore.Signal(bool, str)

    def __init__(self, drone, get_detections, save_photo, connect_drone=None, is_ready=None,
                 video_age=None, parent=None):
        super().__init__(tr("🐍 Flight Plan (Python)"), parent)
        self.drone = drone
        self._video_age = video_age
        self._is_ready = is_ready
        self._get_detections = get_detections
        self._save_photo = save_photo
        self._connect_drone = connect_drone
        self._runner = None
        self.last_report = None  # CheckReport from the last ✔ Check Code
        self._sim_drone = None   # VirtualDrone (🧊 Simulate) or MirroredDrone (real run)
        self.path = DEFAULT_PLAN
        self.resize(1320, 820)

        root = QtWidgets.QVBoxLayout(self)
        from .panel_theme import prepare, heading
        prepare(root)
        heading(root, tr("Flight plan"), tr("Write a mission, check it, watch it in 3D, then fly it with live AI feedback."), "code")

        # --- step bar: ① write  ② check  ③ simulate  ④ connect + run ----------
        steps = QtWidgets.QHBoxLayout()
        self.connect_btn = QtWidgets.QPushButton(tr("① 🔌 Connect Drone"))
        self.connect_btn.setObjectName("primaryButton")
        self.check_btn = QtWidgets.QPushButton(tr("✔ Check Code"))
        self.check_btn.setToolTip(tr("Find mistakes and draw the planned path in 3D — the drone does not move."))
        self.sim_btn = QtWidgets.QPushButton(tr("🧊 Simulate 3D"))
        self.sim_btn.setToolTip(tr("Fly the code in the 3D simulation only — the real drone does not move."))
        self.run_btn = QtWidgets.QPushButton(tr("③ ▶ Run on drone"))
        self.run_btn.setObjectName("primaryButton")
        self.run_btn.setToolTip(tr("Check the code, then fly it on the real drone (the 3D view follows it)."))
        self.stop_btn = QtWidgets.QPushButton(tr("⏹ Stop"))
        self.stop_btn.setObjectName("dangerButton")
        self.stop_btn.setEnabled(False)
        write_lbl = QtWidgets.QLabel(tr("② Write code below  →"))
        write_lbl.setStyleSheet("font-weight:800; padding:0 8px;")
        steps.addWidget(self.connect_btn)
        steps.addWidget(write_lbl)
        steps.addWidget(self.check_btn)
        steps.addWidget(self.sim_btn)
        steps.addWidget(self.run_btn)
        steps.addWidget(self.stop_btn)
        steps.addStretch()
        self.conn_label = QtWidgets.QLabel()
        steps.addWidget(self.conn_label)
        root.addLayout(steps)

        files = QtWidgets.QHBoxLayout()
        self.new_btn = QtWidgets.QPushButton(tr("📄 New"))
        self.example_btn = QtWidgets.QPushButton(tr("📚 Example"))
        self.open_btn = QtWidgets.QPushButton(tr("📂 Open"))
        self.save_btn = QtWidgets.QPushButton(tr("💾 Save"))
        for b in (self.new_btn, self.example_btn, self.open_btn, self.save_btn):
            files.addWidget(b)
        files.addStretch()
        self.file_label = QtWidgets.QLabel()
        self.file_label.setObjectName("muted")
        files.addWidget(self.file_label)
        root.addLayout(files)

        # --- editor + command palette --------------------------------------
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.editor = CodeTextEdit()
        self._highlighter = PythonHighlighter(self.editor.document())
        split.addWidget(self.editor)

        palette = QtWidgets.QWidget()
        pal_layout = QtWidgets.QVBoxLayout(palette)
        pal_layout.setContentsMargins(6, 0, 0, 0)
        pal_title = QtWidgets.QLabel(tr("Click a command to insert it at the cursor:"))
        pal_title.setWordWrap(True)
        pal_title.setStyleSheet("font-weight:800;")
        pal_layout.addWidget(pal_title)
        grid = QtWidgets.QGridLayout()
        for i, (label, code) in enumerate(COMMAND_BUTTONS):
            b = QtWidgets.QPushButton(label)
            b.setStyleSheet("text-align:left; padding:6px 8px;")
            if code is LED_PICKER:
                b.setToolTip("drone.led(R, G, B)")
                b.clicked.connect(self.pick_led_color)
                self.led_btn = b
            else:
                b.setToolTip(code)
                b.clicked.connect(lambda _=False, c=code: self.insert_code(c))
            grid.addWidget(b, i // 2, i % 2)
        pal_layout.addLayout(grid)
        hint = QtWidgets.QLabel(tr(
            "Distances are in cm, angles in degrees. Edit the numbers after inserting. "
            "Lines starting with # are notes and are not run."
        ))
        hint.setWordWrap(True)
        hint.setObjectName("muted")
        pal_layout.addWidget(hint)
        pal_layout.addStretch()
        pal_scroll = QtWidgets.QScrollArea()
        pal_scroll.setWidgetResizable(True)
        pal_scroll.setWidget(palette)
        # Right column: 🧊 3D simulation / the drone camera with the AI boxes
        # (what ai.sees(...) is looking at) on top, the command buttons below.
        right = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.track = SimTrack()
        self.view_tabs = QtWidgets.QTabWidget()
        self.view_tabs.addTab(self._build_sim_tab(), tr("🧊 3D Simulation"))
        self.view = LiveAIView(min_size=(360, 240))
        self.view_tabs.addTab(self.view, tr("📷 Drone camera + AI"))
        right.addWidget(self.view_tabs)
        right.addWidget(pal_scroll)
        right.setSizes([470, 300])
        split.addWidget(right)
        # Left column: 🎯 Mission Guide (levels with goals + stars).
        self.guide = MissionGuidePanel(DEFAULT_PLAN.parent / "mission_progress.json")
        self.guide.setMinimumWidth(260)
        guide_box = QtWidgets.QWidget()
        gl = QtWidgets.QVBoxLayout(guide_box)
        gl.setContentsMargins(0, 0, 6, 0)
        guide_title = QtWidgets.QLabel(tr("🎯 Missions"))
        guide_title.setStyleSheet("font-size:14px; font-weight:700; color:#7be1cd; padding:4px 6px 0;")
        gl.addWidget(guide_title)
        gl.addWidget(self.guide, 1)
        split.insertWidget(0, guide_box)
        split.setSizes([330, 430, 560])
        root.addWidget(split, 3)

        self.console = QtWidgets.QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumBlockCount(2000)
        self.console.setObjectName("terminal")
        self.console.setMaximumHeight(130)
        root.addWidget(self.console, 1)

        self.connect_btn.clicked.connect(self.connect_drone)
        self.check_btn.clicked.connect(self.check_code)
        self.sim_btn.clicked.connect(self.simulate_plan)
        self.run_btn.clicked.connect(self.run_plan)
        self.stop_btn.clicked.connect(self.stop_plan)
        self.new_btn.clicked.connect(self._new_plan)
        self.example_btn.clicked.connect(self._load_example)
        self.open_btn.clicked.connect(self._open_file)
        self.save_btn.clicked.connect(self.save)
        self.guide.mission_changed.connect(self._on_mission_changed)
        self.guide.starter_requested.connect(self._load_starter)
        self.guide.grade_requested.connect(self.grade_mission)
        self._banner_timer = QtCore.QTimer(self)
        self._banner_timer.setSingleShot(True)
        self._banner_timer.timeout.connect(lambda: self.track.set_banner(""))
        self.log_signal.connect(self._append_log)
        self.line_signal.connect(self._show_running_line)
        self.finished_signal.connect(self._on_finished)

        self._conn_timer = QtCore.QTimer(self)
        self._conn_timer.timeout.connect(self._refresh_connection)
        self._conn_timer.start(700)

        self._load_initial()
        self._refresh_connection()

    def _build_sim_tab(self):
        box = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(box)
        lay.setContentsMargins(0, 6, 0, 0)
        lay.setSpacing(6)
        self.sim_view = FlightSim3DView(self.track)
        self.sim_view.setMinimumSize(360, 240)
        lay.addWidget(self.sim_view, 1)
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel(tr("Speed")))
        self.speed_combo = QtWidgets.QComboBox()
        for label, value in SIM_SPEEDS:
            self.speed_combo.addItem(label, value)
        self.speed_combo.setCurrentIndex(1)
        self.speed_combo.currentIndexChanged.connect(self._speed_changed)
        bar.addWidget(self.speed_combo)
        self.follow_chk = QtWidgets.QCheckBox(tr("Follow drone"))
        self.follow_chk.setChecked(True)
        self.follow_chk.toggled.connect(self.sim_view.set_follow)
        bar.addWidget(self.follow_chk)
        reset = QtWidgets.QPushButton(tr("Reset view"))
        reset.clicked.connect(self.sim_view.reset_camera)
        bar.addWidget(reset)
        bar.addSpacing(8)
        bar.addWidget(QtWidgets.QLabel(tr("🤖 Pretend AI sees:")))
        self.sim_ai_edit = QtWidgets.QLineEdit()
        self.sim_ai_edit.setPlaceholderText(tr("e.g. person, cone"))
        self.sim_ai_edit.setToolTip(tr("Only for Check Code / Simulate 3D: what ai.sees() should find, so you can test both sides of an if."))
        bar.addWidget(self.sim_ai_edit, 1)
        lay.addLayout(bar)
        return box

    # -- 🎯 missions -----------------------------------------------------------
    def _mission_file(self, mission):
        return DEFAULT_PLAN if mission is None else DEFAULT_PLAN.parent / f"mission_{mission.id}.py"

    def _on_mission_changed(self, mission):
        """Each mission keeps its own code file; free flight is my_flight_plan.py."""
        if self._runner is not None:
            return
        self.save()
        self.path = self._mission_file(mission)
        try:
            if mission is not None and not self.path.exists():
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.write_text(mission.starter, encoding="utf-8")
            self.editor.setPlainText(self.path.read_text(encoding="utf-8") if self.path.exists() else BLANK_CODE)
        except Exception as exc:
            self._append_log(f"⚠️ {exc}")
        self._update_file_label()
        self.editor.setExtraSelections([])
        self.track.mode = "sim"
        self.track.reset()
        self.track.set_plan([])
        self.track.set_world(mission.world if mission else [], banner="")
        self.sim_ai_edit.setText(", ".join(mission.scenarios.get("main", ())) if mission else "")
        self.sim_view.frame_world([] if mission is None else mission.world)
        self.view_tabs.setCurrentIndex(0)
        self.console.clear()
        if mission is not None:
            self._append_log(f"🎯 {mission.name} — {mission.goal}")
            self._append_log("   " + tr("Edit the code, then press 🏁 Grade mission or 🧊 Simulate 3D."))

    def _load_starter(self):
        mission = self.guide.current()
        if mission is None or self._runner is not None:
            return
        if QtWidgets.QMessageBox.question(
            self, tr("Starter code"), tr("Replace your code with the mission's starter code?"),
        ) == QtWidgets.QMessageBox.Yes:
            self.editor.setPlainText(mission.starter)
            self.save()

    def grade_mission(self):
        """🏁 Grade: Check Code (draws the path) + the mission's goals."""
        mission = self.guide.current()
        if mission is None or self._runner is not None:
            return None
        self.check_code()
        result = grade_mission_code(mission, self.editor.toPlainText())
        self.guide.show_grade(result)
        for text, ok, detail in result.required + result.bonus:
            if ok is not None:
                self._append_log(f"{'✅' if ok else '❌'} {text} — {detail}")
        if result.errors:
            self._show_banner("🛠 " + tr("Fix the code first"))
        elif result.passed:
            self._append_log(f"🏆 {mission.name}: {'★' * result.stars}{'☆' * (3 - result.stars)}")
            self._show_banner(f"{'★' * result.stars}{'☆' * (3 - result.stars)}\n" + tr("Mission complete!"))
        else:
            self._show_banner("🚧 " + tr("Not yet — see the goals"))
        return result

    def _show_banner(self, text):
        self.track.set_banner(text)
        self._banner_timer.start(5000)

    def _sim_ai_names(self):
        return [n.strip() for n in self.sim_ai_edit.text().replace(";", ",").split(",") if n.strip()]

    def _speed_changed(self):
        if isinstance(self._sim_drone, VirtualDrone):
            self._sim_drone.speed = float(self.speed_combo.currentData())

    # -- connection --------------------------------------------------------
    def _ready(self) -> bool:
        if self._is_ready is not None:
            return bool(self._is_ready())
        return bool(getattr(self.drone, "connected", False))

    def show_frame(self, frame, detections):
        if frame is not None and self.isVisible():
            self.view.show_frame(frame, detections)

    def _age(self) -> float:
        if self._video_age is None:
            return 0.0 if self._ready() else float("inf")
        try:
            return float(self._video_age())
        except Exception:
            return float("inf")

    def _refresh_connection(self):
        connected = self._ready()
        self.view.update_state(connected, self._age())
        self.conn_label.setText(
            "🟢 " + tr("Drone connected") if connected else "⚪ " + tr("Drone not connected")
        )
        self.conn_label.setStyleSheet(_state_style(connected))
        self.connect_btn.setEnabled(not connected and self._runner is None)
        self.connect_btn.setText(tr("① ✅ Connected") if connected else tr("① 🔌 Connect Drone"))
        if (self._runner is None and isinstance(self._sim_drone, MirroredDrone)
                and not getattr(self.drone, "flying", False) and self.track.snapshot().pose.flying):
            self._sim_drone.land_in_view()  # e.g. EMERGENCY STOP on the main window

    def connect_drone(self) -> bool:
        if self._ready():
            return True
        if self._connect_drone is None:
            self._append_log("⚠️ " + tr("Press ▶ Start AI Camera on the main window to connect."))
            return False
        self._append_log("📡 " + tr("Connecting to the drone and starting the AI camera..."))
        QtWidgets.QApplication.processEvents()
        ok = bool(self._connect_drone())
        self._append_log("✅ " + tr("Connected") if ok else "❌ " + tr(
            "Could not connect — check the drone is on and this computer is on the drone's Wi-Fi."
        ))
        self._refresh_connection()
        return ok

    # -- file handling -------------------------------------------------------
    def _load_initial(self):
        try:
            FLIGHT_PLAN_DIR.mkdir(parents=True, exist_ok=True)
            if not DEFAULT_PLAN.exists():
                DEFAULT_PLAN.write_text(BLANK_CODE, encoding="utf-8")
            self.editor.setPlainText(DEFAULT_PLAN.read_text(encoding="utf-8"))
        except Exception as exc:
            self.editor.setPlainText(BLANK_CODE)
            self._append_log(f"⚠️ {exc}")
        self._update_file_label()
        self._put_cursor_at_placeholder()

    def _put_cursor_at_placeholder(self):
        doc = self.editor.document()
        block = doc.find("# ⬇️")
        if not block.isNull():
            cur = QtGui.QTextCursor(block)
            cur.movePosition(QtGui.QTextCursor.NextBlock)
            self.editor.setTextCursor(cur)

    def _update_file_label(self):
        self.file_label.setText(f"📄 {self.path.name}")
        self.file_label.setToolTip(str(self.path))

    def save(self) -> bool:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(self.editor.toPlainText(), encoding="utf-8")
            self._append_log(f"💾 {tr('Saved')}: {self.path.name}")
            return True
        except Exception as exc:
            self._append_log(f"❌ {tr('Could not save')}: {exc}")
            return False

    def _open_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, tr("Open Flight Plan"), str(FLIGHT_PLAN_DIR), "Python (*.py)"
        )
        if not path:
            return
        try:
            self.editor.setPlainText(Path(path).read_text(encoding="utf-8"))
            self.path = Path(path)
            self._update_file_label()
        except Exception as exc:
            self._append_log(f"❌ {exc}")

    def _new_plan(self):
        if QtWidgets.QMessageBox.question(
            self, tr("New"), tr("Start a new, empty flight plan? (the current code is replaced)"),
        ) == QtWidgets.QMessageBox.Yes:
            self.editor.setPlainText(BLANK_CODE)
            self._put_cursor_at_placeholder()

    def _load_example(self):
        if QtWidgets.QMessageBox.question(
            self, tr("Example"), tr("Replace the code in the editor with the full example?"),
        ) == QtWidgets.QMessageBox.Yes:
            self.editor.setPlainText(EXAMPLE_CODE)

    def insert_code(self, code: str):
        """Insert a command on its own line at the cursor, matching the
        indentation of the line the cursor is on (so it works inside
        `if` / `for` blocks too)."""
        if self.editor.isReadOnly():
            return
        cur = self.editor.textCursor()
        line = cur.block().text()
        indent = line[: len(line) - len(line.lstrip())]
        lines = [indent + part for part in code.split("\n")]
        text = "\n".join(lines)
        if len(lines) > 1:
            # A block (loop / if): leave the cursor on a new line AFTER it,
            # so the next command doesn't silently land inside the block.
            text += "\n" + indent
        if line.strip():
            cur.movePosition(QtGui.QTextCursor.EndOfBlock)
            cur.insertText("\n" + text)
        else:
            cur.movePosition(QtGui.QTextCursor.StartOfBlock)
            cur.movePosition(QtGui.QTextCursor.EndOfBlock, QtGui.QTextCursor.KeepAnchor)
            cur.insertText(text)
        self.editor.setTextCursor(cur)
        self.editor.setFocus()

    def pick_led_color(self):
        """🎨 Choose the LED colour in a colour picker instead of typing
        R, G, B. If the cursor's line already has drone.led(...), that
        colour is preselected and the line is updated in place."""
        if self.editor.isReadOnly():
            return
        cur = self.editor.textCursor()
        match = LED_CALL.search(cur.block().text())
        initial = QtGui.QColor(*(min(255, int(v)) for v in match.groups())) if match else QtGui.QColor(0, 255, 0)
        color = QtWidgets.QColorDialog.getColor(initial, self, tr("💡 Choose the LED colour"))
        if not color.isValid():
            return
        code = f"drone.led({color.red()}, {color.green()}, {color.blue()})"
        if match:
            start = cur.block().position()
            cur.setPosition(start + match.start())
            cur.setPosition(start + match.end(), QtGui.QTextCursor.KeepAnchor)
            cur.insertText(code)
            self.editor.setTextCursor(cur)
            self.editor.setFocus()
        else:
            self.insert_code(code)

    # -- run -----------------------------------------------------------------
    def check_code(self) -> bool:
        """✔ Check Code: report every mistake it can find and draw the
        planned path in the 3D view. Never moves the real drone."""
        if self._runner is not None:
            return False
        self.console.clear()
        self._append_log("✔ " + tr("Checking the code (the drone does not move)..."))
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            report = check_plan(self.editor.toPlainText(), self._sim_ai_names())
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.last_report = report
        icons = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}
        for issue in report.issues:
            where = f"{tr('Line')} {issue.line}: " if issue.line else ""
            self._append_log(f"{icons[issue.level]} {where}{issue.message}")
        if report.ok:
            self.track.set_plan(report.path)
            self._append_log(
                "✅ " + tr("Code OK") + f" — {report.commands} " + tr("commands") + ", "
                + tr("distance") + f" {report.distance_cm / 100:.1f} m, "
                + tr("max height") + f" {report.max_height_cm / 100:.1f} m, "
                + tr("farthest from start") + f" {report.max_range_cm / 100:.1f} m, "
                + tr("about") + f" {report.seconds:.0f} s"
            )
            self.editor.setExtraSelections([])
        else:
            self.track.set_plan([])
            self._append_log("🛠 " + tr("fix it, then press Check Code again"))
            first = next((i.line for i in report.errors if i.line), None)
            if first:
                self._show_running_line(first, color="#5a1f24")
        self.view_tabs.setCurrentIndex(0)
        return report.ok

    def simulate_plan(self):
        """🧊 Fly the code on a VirtualDrone in the 3D view only."""
        if self._runner is not None:
            return
        if not self.check_code():
            return
        self.save()
        self.track.mode = "sim"
        self.track.reset()
        self.track.set_banner("")
        vdrone = VirtualDrone(self.track, speed=float(self.speed_combo.currentData()))
        names = self._sim_ai_names()
        self._append_log("🧊 " + tr("3D simulation — the real drone does not move"))
        self._start_runner(vdrone, lambda: fake_detections(names), save_photo=None)

    def _start_runner(self, drone, get_detections, save_photo):
        self._sim_drone = drone
        self.editor.setReadOnly(True)
        for b in (self.run_btn, self.sim_btn, self.check_btn):
            b.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._runner = FlightPlanRunner(
            self.editor.toPlainText(), drone, get_detections,
            on_log=self.log_signal.emit,
            on_line=self._on_runner_line,
            on_finished=self.finished_signal.emit,
            save_photo=save_photo,
        )
        if isinstance(drone, VirtualDrone):
            drone.stop_event = self._runner.stop_event  # ⏹ Stop also halts the 3D animation
        self._runner.start()

    def _on_runner_line(self, line):
        """Worker thread: highlight the line in the editor + 3D HUD."""
        self.track.set_status(line=line, label=self._line_text(line))
        self.line_signal.emit(line)

    def _line_text(self, line):
        lines = self.editor.toPlainText().split("\n")
        return lines[line - 1].strip()[:48] if 0 < line <= len(lines) else ""

    def is_busy(self) -> bool:
        return self._runner is not None

    def run_plan(self):
        if self._runner is not None:
            return
        if not self.check_code():
            return
        self.save()
        if not self.connect_drone():
            return
        if "ai." in self.editor.toPlainText() and not _video_ok(self._age()):
            self._append_log("⚠️ " + tr("No drone video yet — ai.sees() / ai.count() only see something once the video (and its AI boxes) appears on the right."))
        self.track.mode = "mirror"
        self.track.reset()
        self._append_log("📡 " + tr("Flying the REAL drone — the 3D view follows each finished command"))
        self._start_runner(MirroredDrone(self.drone, self.track), self._get_detections, self._save_photo)

    def stop_plan(self):
        if self._runner is not None:
            self._append_log("⏹ " + tr("Stopping..."))
            self._runner.stop()

    @QtCore.Slot(str)
    def _append_log(self, text):
        self.console.appendPlainText(text)

    @QtCore.Slot(int)
    def _show_running_line(self, line, color="#1d3b2f"):
        block = self.editor.document().findBlockByLineNumber(max(0, line - 1))
        if not block.isValid():
            return
        sel = QtWidgets.QTextEdit.ExtraSelection()
        sel.format.setBackground(QtGui.QColor(color))
        sel.format.setProperty(QtGui.QTextFormat.FullWidthSelection, True)
        sel.cursor = QtGui.QTextCursor(block)
        self.editor.setExtraSelections([sel])
        self.editor.setTextCursor(QtGui.QTextCursor(block))
        self.editor.ensureCursorVisible()

    @QtCore.Slot(bool, str)
    def _on_finished(self, ok, _msg):
        was_sim = isinstance(self._sim_drone, VirtualDrone)
        self._runner = None
        self.editor.setReadOnly(False)
        for b in (self.run_btn, self.sim_btn, self.check_btn):
            b.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.track.set_status(label="")
        self._refresh_connection()
        if ok and was_sim and self.guide.current() is not None:
            QtCore.QTimer.singleShot(400, self.grade_mission)  # 🎯 grade right after the 3D flight

    def closeEvent(self, event):
        if self._runner is not None:
            self._runner.stop()  # the runner lands the drone itself
        else:
            self.save()
        super().closeEvent(event)


# =====================================================================
# ⌨️ Keyboard Control
# =====================================================================
_KEY_ON = ("QPushButton { background:#1b4738; border:2px solid #8ce8c0; color:#ffffff; "
           "border-radius:10px; font-size:11pt; font-weight:800; padding:4px; }")
_KEY_OFF = ("QPushButton { background:#17222d; border:1px solid #2b3b4a; color:#dfe7ef; "
            "border-radius:10px; font-size:11pt; font-weight:800; padding:4px; }"
            "QPushButton:hover { background:#1d2c39; border-color:#466078; }"
            "QPushButton:disabled { background:#10171f; color:#4a5866; border-color:#1a2530; }")


class KeyboardControlDialog(_MissionWindow):
    """One window, three steps: ① Connect (drone + AI camera) → ② Take off →
    ③ fly with the keyboard, watching the drone's camera with the AI boxes
    from the trained models."""

    status_signal = QtCore.Signal(str)
    _cmd_done = QtCore.Signal()

    # key -> (kind, direction/sign, button id)
    _KEYS = {
        QtCore.Qt.Key_W: ("move", Direction.FORWARD, "W"),
        QtCore.Qt.Key_S: ("move", Direction.BACK, "S"),
        QtCore.Qt.Key_A: ("move", Direction.LEFT, "A"),
        QtCore.Qt.Key_D: ("move", Direction.RIGHT, "D"),
        QtCore.Qt.Key_Up: ("alt", Direction.UP, "UP"),
        QtCore.Qt.Key_Down: ("alt", Direction.DOWN, "DOWN"),
        QtCore.Qt.Key_Left: ("rot", 1, "LEFT"),
        QtCore.Qt.Key_Right: ("rot", -1, "RIGHT"),
        QtCore.Qt.Key_Q: ("rot", 1, "LEFT"),    # old layout, still works
        QtCore.Qt.Key_E: ("rot", -1, "RIGHT"),
    }
    _WHAT = {
        "W": "⬆️ forward", "S": "⬇️ back", "A": "⬅️ left", "D": "➡️ right",
        "UP": "🔼 up", "DOWN": "🔽 down", "LEFT": "↩️ turn left", "RIGHT": "↪️ turn right",
    }
    _SHORT = {  # second line on the on-screen key caps
        "W": "forward", "S": "back", "A": "left", "D": "right",
        "UP": "up", "DOWN": "down", "LEFT": "turn left", "RIGHT": "turn right",
    }

    def __init__(self, drone, connect_drone=None, is_ready=None, video_age=None, parent=None):
        super().__init__(tr("⌨️ Keyboard Control"), parent)
        self.drone = drone
        self._is_ready = is_ready
        self._connect_drone = connect_drone
        self._video_age = video_age
        self._busy = False
        self._running_btn = None
        self._pending_key = None
        self._battery_next = 0.0
        self.resize(1320, 800)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

        from .panel_theme import prepare, heading
        outer = QtWidgets.QVBoxLayout(self)
        prepare(outer)
        heading(outer, tr("Flight controls"), tr("Live camera, keyboard navigation and emergency controls."), "keyboard")
        root = QtWidgets.QHBoxLayout()
        root.setSpacing(20)
        outer.addLayout(root, 1)

        # ---------------- left: steps + live AI video ---------------------
        left = QtWidgets.QVBoxLayout()
        chips = QtWidgets.QHBoxLayout()
        self._chips = []
        for text in (tr("① Connect"), tr("② Take off"), tr("③ Fly")):
            c = QtWidgets.QLabel(text)
            c.setAlignment(QtCore.Qt.AlignCenter)
            chips.addWidget(c, 1)
            self._chips.append(c)
        left.addLayout(chips)
        self.hint_label = QtWidgets.QLabel()
        self.hint_label.setWordWrap(True)
        self.hint_label.setStyleSheet("font-size:13pt; font-weight:800; color:#ffd166; padding:4px 2px;")
        left.addWidget(self.hint_label)
        self.view = LiveAIView(min_size=(400, 280))
        left.addWidget(self.view, 1)
        info = QtWidgets.QHBoxLayout()
        self.battery_label = QtWidgets.QLabel("🔋 --")
        self.video_label = QtWidgets.QLabel()
        self.focus_label = QtWidgets.QLabel()
        for w in (self.battery_label, self.video_label, self.focus_label):
            w.setStyleSheet("font-weight:800; padding:2px 8px;")
            info.addWidget(w)
        info.addStretch()
        left.addLayout(info)
        root.addLayout(left, 3)

        # ---------------- right: one big action + key pad -----------------
        controls_card = QtWidgets.QFrame()
        controls_card.setObjectName("card")
        right = QtWidgets.QVBoxLayout(controls_card)
        right.setContentsMargins(18, 18, 18, 18)
        right.setSpacing(12)
        self.action_btn = QtWidgets.QPushButton()
        self.action_btn.setObjectName("primaryButton")
        self.action_btn.setMinimumHeight(58)
        self.action_btn.setStyleSheet("font-size:14pt;")
        right.addWidget(self.action_btn)

        pad_box = QtWidgets.QGroupBox(tr("🎮 Fly: press a key (or click it)"))
        pad_box.setStyleSheet("QGroupBox { font-weight:800; font-size:11pt; }")
        pads = QtWidgets.QHBoxLayout(pad_box)
        self._key_buttons = {}

        def cluster(title, keys):
            box = QtWidgets.QVBoxLayout()
            t = QtWidgets.QLabel(title)
            t.setAlignment(QtCore.Qt.AlignCenter)
            t.setStyleSheet("color:#8ea0b2; font-weight:800;")
            box.addWidget(t)
            grid = QtWidgets.QGridLayout()
            grid.setSpacing(6)
            for cap, bid, qtkey, r, c in keys:
                b = QtWidgets.QPushButton(f"{cap}\n{tr(self._SHORT[bid])}")
                b.setMinimumSize(68, 64)
                b.setStyleSheet(_KEY_OFF)
                b.clicked.connect(lambda _=False, k=qtkey: self._handle_key(k))
                grid.addWidget(b, r, c)
                self._key_buttons[bid] = b
            box.addLayout(grid)
            return box

        pads.addLayout(cluster(tr("MOVE  (left hand)"), [
            ("W", "W", QtCore.Qt.Key_W, 0, 1),
            ("A", "A", QtCore.Qt.Key_A, 1, 0),
            ("S", "S", QtCore.Qt.Key_S, 1, 1),
            ("D", "D", QtCore.Qt.Key_D, 1, 2),
        ]))
        pads.addSpacing(18)
        pads.addLayout(cluster(tr("HEIGHT / TURN  (right hand)"), [
            ("↑", "UP", QtCore.Qt.Key_Up, 0, 1),
            ("←", "LEFT", QtCore.Qt.Key_Left, 1, 0),
            ("↓", "DOWN", QtCore.Qt.Key_Down, 1, 1),
            ("→", "RIGHT", QtCore.Qt.Key_Right, 1, 2),
        ]))
        right.addWidget(pad_box)

        steps = QtWidgets.QGridLayout()
        steps.addWidget(QtWidgets.QLabel(tr("Move per press:")), 0, 0)
        self.move_step = QtWidgets.QComboBox()
        for v in (20, 30, 50, 100):
            self.move_step.addItem(f"{v} cm", v)
        self.move_step.setCurrentIndex(max(0, self.move_step.findData(int(config.MANUAL_FLIGHT_MOVE_STEP_CM))))
        steps.addWidget(self.move_step, 0, 1)
        steps.addWidget(QtWidgets.QLabel(tr("Up/down per press:")), 1, 0)
        self.alt_step = QtWidgets.QComboBox()
        for v in (20, 30, 50):
            self.alt_step.addItem(f"{v} cm", v)
        self.alt_step.setCurrentIndex(max(0, self.alt_step.findData(int(config.MANUAL_FLIGHT_ALTITUDE_STEP_CM))))
        steps.addWidget(self.alt_step, 1, 1)
        steps.addWidget(QtWidgets.QLabel(tr("Turn per press:")), 2, 0)
        self.turn_step = QtWidgets.QComboBox()
        for v in (15, 30, 45, 90):
            self.turn_step.addItem(f"{v}°", v)
        self.turn_step.setCurrentIndex(max(0, self.turn_step.findData(int(config.MANUAL_FLIGHT_ROTATE_DEG))))
        steps.addWidget(self.turn_step, 2, 1)
        right.addLayout(steps)

        cam_row = QtWidgets.QHBoxLayout()
        cam_row.addWidget(QtWidgets.QLabel(tr("📷 Camera:")))
        self._cam_buttons = []
        for text, deg in ((tr("look ahead 0°"), 0), ("45°", 45), (tr("look down 90°"), 90)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(lambda _=False, d=deg: self._set_camera(d))
            cam_row.addWidget(b)
            self._cam_buttons.append(b)
        right.addLayout(cam_row)

        keys_help = QtWidgets.QLabel(tr(
            "T = take off   •   L = land   •   Space / Esc = EMERGENCY STOP\n"
            "Each press moves ONE step, then the drone hovers and waits."
        ))
        keys_help.setObjectName("muted")
        keys_help.setWordWrap(True)
        right.addWidget(keys_help)

        self.estop_btn = QtWidgets.QPushButton(tr("🛑 EMERGENCY STOP  (Space / Esc)"))
        self.estop_btn.setObjectName("dangerButton")
        self.estop_btn.setMinimumHeight(50)
        right.addWidget(self.estop_btn)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(300)
        right.addWidget(self.log, 1)
        root.addWidget(controls_card, 2)

        # Nothing inside may take keyboard focus: every key must reach
        # keyPressEvent (Space on a focused button would "click" it, arrows
        # would change a combo box instead of flying).
        for w in self.findChildren(QtWidgets.QWidget):
            w.setFocusPolicy(QtCore.Qt.NoFocus)

        self.action_btn.clicked.connect(self._main_action)
        self.estop_btn.clicked.connect(self._emergency)
        self.status_signal.connect(self.log.appendPlainText)
        self._cmd_done.connect(self._on_cmd_done)

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._refresh_state)
        self._timer.start(300)
        self._refresh_state()

    # -- state -----------------------------------------------------------
    def is_busy(self) -> bool:
        return bool(getattr(self.drone, "flying", False))

    def _ready(self) -> bool:
        if self._is_ready is not None:
            return bool(self._is_ready())
        return bool(getattr(self.drone, "connected", False))

    def _age(self) -> float:
        if self._video_age is None:
            return 0.0 if self._ready() else float("inf")
        try:
            return float(self._video_age())
        except Exception:
            return float("inf")

    def _step(self) -> int:
        connected = self._ready()
        if not connected:
            return 0
        return 2 if getattr(self.drone, "flying", False) else 1

    def _refresh_state(self):
        connected = self._ready()
        flying = bool(getattr(self.drone, "flying", False))
        step = self._step()
        for i, chip in enumerate(self._chips):
            if i < step:
                style, prefix = "background:#12301f; color:#8ce8c0; border:1px solid #2c6b4c;", "✅ "
            elif i == step:
                style, prefix = "background:#1b4738; color:#ffffff; border:2px solid #8ce8c0;", "👉 "
            else:
                style, prefix = "background:#10171f; color:#596978; border:1px solid #1a2530;", ""
            chip.setStyleSheet(style + " border-radius:8px; padding:8px; font-size:12pt; font-weight:800;")
            base = chip.text().replace("✅ ", "").replace("👉 ", "")
            chip.setText(prefix + base)

        if step == 0:
            self.hint_label.setText(tr("Step ①: press the green button to connect the drone and start the AI camera"))
            self.action_btn.setText(tr("① 🔌 Connect drone + AI camera"))
        elif step == 1:
            self.hint_label.setText(tr("Step ②: put the drone on a flat floor, then press Take off (or T)"))
            self.action_btn.setText(tr("② 🛫 Take off  (T)"))
        else:
            self.hint_label.setText(tr("Step ③: fly with W A S D and the arrow keys — the AI boxes show what it sees"))
            self.action_btn.setText(tr("🛬 Land  (L)"))
        self.action_btn.setEnabled(not (self._busy and step == 1))

        for b in self._key_buttons.values():
            b.setEnabled(flying)
        for b in self._cam_buttons:
            b.setEnabled(connected)

        age = self._age()
        self.view.update_state(connected, age)
        if not connected:
            self.video_label.setText("🎥 " + tr("video: off"))
            self.video_label.setStyleSheet("font-weight:800; padding:2px 8px; color:#8ea0b2;")
        elif _video_ok(age):
            self.video_label.setText("🎥 " + tr("video + AI: live"))
            self.video_label.setStyleSheet("font-weight:800; padding:2px 8px; color:#8ce8c0;")
        else:
            self.video_label.setText("🎥 " + tr("video: waiting..."))
            self.video_label.setStyleSheet("font-weight:800; padding:2px 8px; color:#ffd166;")

        if self.isActiveWindow():
            self.focus_label.setText("⌨️ " + tr("keyboard ready"))
            self.focus_label.setStyleSheet("font-weight:800; padding:2px 8px; color:#8ce8c0;")
        else:
            self.focus_label.setText("⚠️ " + tr("click this window first — keys go to the active window"))
            self.focus_label.setStyleSheet("font-weight:800; padding:2px 8px; color:#ffd166;")

        now = time.monotonic()
        if connected and now >= self._battery_next:
            self._battery_next = now + 3.0
            try:
                pct = int(self.drone.api.get_battery())
                self.battery_label.setText(f"🔋 {pct}%")
            except Exception:
                self.battery_label.setText("🔋 --")
        elif not connected:
            self.battery_label.setText("🔋 --")

    def showEvent(self, event):
        super().showEvent(event)
        self.activateWindow()
        self.setFocus()
        self._refresh_state()

    # -- live video from the main window's camera + AI pipeline -----------
    def show_frame(self, frame, detections):
        if frame is None or not self.isVisible():
            return
        self.view.show_frame(frame, detections)

    # -- actions -----------------------------------------------------------
    def _main_action(self):
        step = self._step()
        if step == 0:
            self.connect_drone()
        elif step == 1:
            self._takeoff()
        else:
            self._land()

    def connect_drone(self) -> bool:
        if self._ready():
            return True
        if self._connect_drone is None:
            self.status_signal.emit("⚠️ " + tr("Press ▶ Start AI Camera on the main window to connect."))
            return False
        self.status_signal.emit("📡 " + tr("Connecting to the drone and starting the AI camera..."))
        self.view.set_message(tr("📡 Connecting..."))
        QtWidgets.QApplication.processEvents()
        ok = bool(self._connect_drone())
        self.status_signal.emit("✅ " + tr("Connected — the AI camera is on") if ok else "❌ " + tr(
            "Could not connect — check the drone is on and this computer is on the drone's Wi-Fi."
        ))
        self._refresh_state()
        self.activateWindow()
        self.setFocus()
        return ok

    def _reason(self) -> str:
        r = str(getattr(self.drone, "last_error", "") or "").strip()
        return f" — {r}" if r else ""

    def _bg(self, fn, button_id=None):
        """Run one drone command off the GUI thread. The drone does one
        command at a time: a key pressed meanwhile is remembered (the last
        one only) and runs right after - never silently dropped."""
        self._busy = True
        self._running_btn = button_id
        if button_id in self._key_buttons:
            self._key_buttons[button_id].setStyleSheet(_KEY_ON)

        def run():
            try:
                fn()
            except Exception as exc:
                self.status_signal.emit(f"❌ {exc}")
            finally:
                self._cmd_done.emit()
        threading.Thread(target=run, daemon=True, name="AI-LAB-KEYBOARD-CMD").start()

    @QtCore.Slot()
    def _on_cmd_done(self):
        if self._running_btn in self._key_buttons:
            self._key_buttons[self._running_btn].setStyleSheet(_KEY_OFF)
        self._running_btn = None
        self._busy = False
        pending, self._pending_key = self._pending_key, None
        if pending is not None and getattr(self.drone, "flying", False):
            self._handle_key(pending)
        self._refresh_state()

    def _takeoff(self):
        if not self._ready():
            self.status_signal.emit("⚪ " + tr("Step ① first: press Connect"))
            return
        if self._busy:
            return

        def run():
            self.status_signal.emit("🛫 " + tr("Taking off..."))
            ok = self.drone.takeoff(require_camera_down=False)
            self.status_signal.emit(
                "✅ " + tr("Airborne — fly with the keys") if ok else "❌ " + tr("Takeoff failed") + self._reason()
            )
        self._bg(run)

    def _land(self):
        self._pending_key = None

        def run():
            self.status_signal.emit("🛬 " + tr("Landing..."))
            ok = self.drone.touchdown()
            self.status_signal.emit("✅ " + tr("Landed") if ok else "❌ " + tr("Landing failed") + self._reason())
        # Landing must never wait behind a queued move.
        threading.Thread(target=run, daemon=True, name="AI-LAB-KEYBOARD-LAND").start()

    def _emergency(self):
        self._pending_key = None
        threading.Thread(target=self.drone.emergency_stop, daemon=True).start()
        self.status_signal.emit("🛑 " + tr("EMERGENCY STOP — the drone lands and the video stops. Press ① Connect to start again."))

    def _set_camera(self, deg):
        if self._busy:
            self.status_signal.emit("⏳ " + tr("Wait — the drone is still doing the last command"))
            return
        self._bg(lambda: self.status_signal.emit(
            f"📷 {tr('Camera')} {deg}°" if self.drone.set_camera_pitch(deg)
            else "❌ " + tr("Camera angle failed") + self._reason()
        ))

    def _handle_key(self, key):
        if key not in self._KEYS:
            return
        if not getattr(self.drone, "flying", False):
            self.status_signal.emit("⚪ " + tr("Take off first (T)"))
            return
        kind, value, bid = self._KEYS[key]
        if self._busy:
            self._pending_key = key
            self.status_signal.emit(f"⏳ {tr(self._WHAT[bid])} — " + tr("queued, runs when the current move finishes"))
            return
        what = tr(self._WHAT[bid])
        if kind == "rot":
            deg = int(self.turn_step.currentData()) * value
            self._bg(lambda: self.status_signal.emit(
                f"✅ {what} {abs(deg)}°" if self.drone.manual_rotate(deg)
                else f"❌ {what} {abs(deg)}°{self._reason()}"), bid)
            return
        step = int((self.alt_step if kind == "alt" else self.move_step).currentData())
        self._bg(lambda: self.status_signal.emit(
            f"✅ {what} {step} cm" if self.drone.manual_move(value, step)
            else f"❌ {what} {step} cm{self._reason()}"), bid)

    def keyPressEvent(self, event):
        if event.isAutoRepeat():
            return
        key = event.key()
        if key in (QtCore.Qt.Key_Space, QtCore.Qt.Key_Escape):
            self._emergency()   # handled here, so Esc never closes the window
        elif key == QtCore.Qt.Key_T:
            self._takeoff()
        elif key == QtCore.Qt.Key_L:
            self._land()
        elif key in self._KEYS:
            self._handle_key(key)
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        # Nothing else is flying the drone once this window is gone.
        self._pending_key = None
        if getattr(self.drone, "flying", False):
            self.status_signal.emit("🛬 " + tr("Closing — landing first..."))
            self.drone.touchdown()
        super().closeEvent(event)
