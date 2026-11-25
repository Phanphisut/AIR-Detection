from __future__ import annotations
import sys
import threading
import time
import cv2
from PySide6 import QtCore, QtGui, QtWidgets

from ..drone.controller import HulaDroneController
from ..drone.camera import CameraWorker
from ..drone import alerts as drone_alerts
from ..text_overlay import put_text
# DetectionManager and YOLODetector are deliberately NOT imported here.
# Importing either pulls in ultralytics -> torch, which measured at ~46
# seconds of the ~75 second cold start, all of it spent before the window
# can appear. Nothing on this screen needs the AI stack until the operator
# actually starts a camera, so both are imported at their use sites
# (_start_webcam_test's thread and start_camera) instead. Keep it that way:
# a module-level import here would put the whole delay back in front of the
# first paint.
from ..grid.planner import generate_zigzag_waypoints, generate_waypoint_objects
from ..grid.autonomous import AutonomousGridMission
from .. import config
from ..i18n import tr
from ..image_io import imwrite_any
from .post_flight_report import PostFlightReportDialog
from .code_editor import CodeEditorDialog
from .label_train_dialog import LabelTrainDialog
from .manual_flight_dialog import ManualFlightDialog
from .mission_dialogs import FlightPlanDialog, KeyboardControlDialog
from ..drone.simulated_controller import SimulatedDroneController


# Maps DetectionManager status strings to what the operator sees.
# Deliberately never says "ACCURATE" — Stability/Confidence != Accuracy.
# Only a human CONFIRM/REJECT decision represents ground truth.
STATUS_DISPLAY = {
    "SEARCHING": tr("⚪ SEARCHING"),
    "CANDIDATE": tr("🟡 CANDIDATE"),
    "STABLE": tr("🟢 STABLE DETECTION"),
    "CONFIRMED": tr("🔵 HUMAN CONFIRMED"),
    "REJECTED": tr("❌ REJECTED BY OPERATOR"),
}

# Which camera device the webcam test opens. 0 is the built-in webcam on
# almost every laptop, so it isn't exposed as a setting - one less thing
# to configure before a student can just click Start and test the AI.
DEFAULT_WEBCAM_INDEX = 0


class WebcamAIThread(threading.Thread):
    """Opens the laptop's own front-facing webcam so students can check the
    camera itself works BEFORE ever touching the drone. Runs YOLO detection
    on top of that feed when a trained model is available, but the camera
    preview does not require one - a missing/untrained model degrades to a
    plain live preview with a one-time warning, it never blocks the test.

    Lives on the main Mission Control dashboard (a single Start/Stop toggle
    button sits right next to the drone controls) rather than inside the
    Code Editor popup, so it is reachable with one click from the first
    screen the app shows.
    """

    def __init__(self, camera_index, on_frame, on_status, on_error, on_finished):
        super().__init__(daemon=True)
        self.camera_index = camera_index
        self.on_frame = on_frame
        self.on_status = on_status
        self.on_error = on_error
        self.on_finished = on_finished
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        # Track whether the camera ever actually opened so the "stopped"
        # status line only appears when there was really something running
        # to stop (avoids a confusing "stopped" message right after a
        # "could not open camera" error).
        started_ok = False
        cap = None

        # A missing/untrained YOLO model must NOT stop the student from
        # testing that the camera itself works - that is the whole point
        # of this feature ("test the camera before using the drone"). Fall
        # back to a raw preview and just warn once instead of refusing to
        # even open the camera.
        detector = None
        try:
            from ..ai.yolo_detector import YOLODetector

            detector = YOLODetector()
        except Exception as exc:
            self.on_status(
                f"⚠️ No YOLO model loaded ({exc}) — showing the raw webcam "
                "feed without AI detection. Put a trained .pt file in the "
                "models folder to see detection overlays."
            )

        try:
            cap = cv2.VideoCapture(self.camera_index, config.webcam_capture_backend())
            if not cap.isOpened():
                self.on_error(
                    f"❌ Webcam test: could not open camera index {self.camera_index}. "
                    "Try a different camera index."
                )
                self.on_finished()
                return

            started_ok = True
            mode = "AI detection ON" if detector is not None else "raw preview, no AI model"
            self.on_status(f"🖥️ Webcam test running (camera index {self.camera_index}, {mode})")

            while not self._stop_event.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    time.sleep(0.05)
                    continue
                if detector is not None:
                    try:
                        results = detector.predict(frame)
                        detections = detector.detections(results)
                        annotated = detector.annotate(results, frame)
                    except Exception as exc:
                        self.on_error(f"❌ Webcam test: inference error: {exc}")
                        annotated = frame
                        detections = []
                else:
                    annotated = frame
                    detections = []
                self.on_frame(annotated, detections)
                time.sleep(0.01)
        except Exception as exc:
            self.on_error(f"❌ Webcam test error: {exc}")
        finally:
            if cap is not None:
                cap.release()
            if started_ok:
                self.on_status("🖥️ Webcam test stopped")
            self.on_finished()


class MainWindow(QtWidgets.QMainWindow):
    frame_signal = QtCore.Signal(object, object)
    status_signal = QtCore.Signal(str)
    # AutonomousGridMission runs in a Python worker thread. Its completion
    # callback must be marshalled back to the Qt GUI thread before touching
    # widgets or opening the post-flight report.
    grid_finished_signal = QtCore.Signal(bool, str)
    # Carries raw camera frames from CameraWorker's thread into the GUI
    # thread. Qt widgets (self.video.setPixmap, etc.) may only ever be
    # touched from the GUI thread — a Queued Connection via this signal
    # is what makes that safe. Do NOT wire CameraWorker's on_frame to a
    # method that touches widgets directly; that was the root cause of
    # intermittent crashes after the app had been running for a while.
    live_frame_signal = QtCore.Signal(object)
    # Carries annotated frames from WebcamAIThread's worker thread into the
    # GUI thread, same Queued-Connection safety rule as live_frame_signal.
    webcam_frame_signal = QtCore.Signal(object, object)
    # Lets WebcamAIThread tell the GUI "I'm done" exactly once, whether it
    # stopped normally, failed to open the camera, or hit an error -
    # keeps the Start/Stop button state resetting reliably either way.
    webcam_thread_stopped_signal = QtCore.Signal()
    # AutonomousGridMission calls on_progress(...) from its own worker
    # thread (see grid/autonomous.py AutonomousGridMission.run()), same as
    # on_finished above. _grid_progress() must not touch the Grid Preview
    # dialog's widgets directly from that thread - this signal carries
    # (waypoint_index, total_waypoints, x_cm, y_cm, heading, turn) across
    # to the GUI thread, same Queued-Connection safety rule as the other
    # cross-thread signals on this class.
    grid_progress_signal = QtCore.Signal(int, int, int, int, str, str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle(tr("AI Drone Detection Lab"))
        self.resize(1440, 900)
        self.setMinimumSize(1100, 700)
        self.drone = HulaDroneController()
        self.detector = None
        self.camera = None
        self.grid_mission = None
        self.last_snapshot = {}
        self._latest_detections = []
        self._current_target_id = None
        self._code_editor_dialog = None
        # "📸 Record Training Photos" state - raw drone frames saved to
        # config.DRONE_PHOTOS_DIR for training (see _toggle_photo_recording).
        self._photo_record_dir = None
        self._photo_record_count = 0
        self._photo_record_last = 0.0
        # Missions (🐍 Flight Plan / ⌨️ Keyboard Control)
        self._flight_plan_dialog = None
        self._keyboard_dialog = None
        self._last_raw_frame = None
        self._last_frame_time = 0.0
        self._label_train_dialog = None
        self._webcam_thread = None
        # The Grid Preview dialog is reused (like the two dialogs above)
        # rather than rebuilt every click, so a live flight can keep
        # updating the SAME open window - see preview_grid() /
        # _on_grid_progress_ui() / _on_grid_preview_closed().
        self._grid_preview_dialog = None
        self._grid_preview_widget = None
        self._grid_details_widget = None
        self._grid_live_label = None
        self._manual_flight_dialog = None
        self._build_ui()
        self.frame_signal.connect(self._on_frame)
        self.status_signal.connect(self._on_status)
        self.live_frame_signal.connect(self._display_live_frame)
        self.grid_finished_signal.connect(self._grid_finished_ui)
        self.webcam_frame_signal.connect(self._display_webcam_frame)
        self.webcam_thread_stopped_signal.connect(self._on_webcam_thread_stopped)
        self.grid_progress_signal.connect(self._on_grid_progress_ui)
        # v9.0.0: drone problem alerts (connection lost / battery low / too low
        # to fly). Runs on its own timer - the battery check above only ran
        # while camera frames were arriving.
        self._alert_monitor = drone_alerts.AlertMonitor()
        self._alert_boxes = {}
        self._alert_timer = QtCore.QTimer(self)
        self._last_reconnect_try = 0.0
        self._alert_timer.timeout.connect(self._check_drone_alerts)
        self._alert_timer.start(1000)
        if sys.platform == "win32":
            QtCore.QTimer.singleShot(1500, self._report_gpupack_result)

    def _build_ui(self):
        # =========================================================
        # 4-ZONE LAB DASHBOARD
        # LEFT | CENTER | RIGHT
        # ----------------------
        #          BOTTOM
        # =========================================================
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background: #0b1118;
                color: #e8eef5;
                font-family: "Segoe UI";
                font-size: 10pt;
            }
            QLabel {
                color: #dfe7ef;
            }
            QFrame#headerCard, QFrame#card {
                background: #111923;
                border: 1px solid #263544;
                border-radius: 10px;
            }
            QLabel#appTitle {
                font-size: 21px;
                font-weight: 800;
                color: #f4f8fb;
            }
            QLabel#appSubtitle {
                font-size: 9pt;
                color: #8ea0b2;
            }
            QLabel#sectionTitle {
                font-size: 9pt;
                font-weight: 800;
                color: #9fb0c0;
            }
            QLabel#valueBig {
                font-size: 22px;
                font-weight: 800;
                color: #ffffff;
            }
            QLabel#valueAccent {
                font-size: 16px;
                font-weight: 700;
                color: #8ce8c0;
            }
            QLabel#muted {
                color: #8999aa;
            }
            QLabel#dangerPill {
                background: #2a171a;
                border: 1px solid #5b2d33;
                border-radius: 8px;
                padding: 5px 10px;
                color: #ff9ca5;
                font-weight: 700;
            }
            QLabel#videoPlaceholder {
                background: #070c11;
                border: 1px solid #263544;
                border-radius: 8px;
                color: #738596;
                font-size: 12pt;
            }
            QPlainTextEdit#modelList,
            QPlainTextEdit#terminal {
                background: #080d12;
                color: #a9bac9;
                border: 1px solid #263544;
                border-radius: 7px;
                padding: 6px;
                font-family: "Cascadia Mono", "Consolas";
                font-size: 8.5pt;
            }
            QPlainTextEdit#terminal {
                background: #070b10;
                color: #9fb3c4;
            }
            QPushButton {
                background: #17222d;
                border: 1px solid #2b3b4a;
                border-radius: 7px;
                padding: 7px 9px;
                min-height: 18px;
                font-weight: 700;
            }
            QPushButton:hover {
                background: #1d2c39;
                border-color: #466078;
            }
            QPushButton:pressed {
                background: #223746;
            }
            QPushButton:disabled {
                background: #10171f;
                color: #596978;
                border-color: #1a2530;
            }
            QPushButton#primaryButton {
                background: #1b4738;
                border-color: #34785e;
                color: #dfffee;
            }
            QPushButton#primaryButton:hover {
                background: #235842;
            }
            QPushButton#dangerButton {
                background: #4a1b20;
                border-color: #7a3038;
                color: #ffdfe2;
            }
            QPushButton#dangerButton:hover {
                background: #612229;
            }
            /* An ID selector (#primaryButton/#dangerButton) outranks the
               plain-type ":disabled" rule above in Qt's stylesheet
               specificity, so without an explicit override here a
               disabled auto_btn/start_btn/webcam_toggle_btn/confirm_btn
               kept rendering in full "active" color - confirmed via a
               real screenshot (auto_btn stayed bright green while
               genuinely disabled). Same fix as code_editor.py's shared
               DIALOG_STYLESHEET (#primaryButton:disabled). */
            QPushButton#primaryButton:disabled {
                background: #10171f;
                color: #596978;
                border-color: #1a2530;
            }
            QPushButton#dangerButton:disabled {
                background: #10171f;
                color: #596978;
                border-color: #1a2530;
            }
            QSplitter::handle {
                background: #182430;
            }
            QScrollArea {
                border: none;
                background: transparent;
            }
        """)

        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        root_layout = QtWidgets.QVBoxLayout(root)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(8)

        # =========================================================
        # HEADER
        # =========================================================
        header = QtWidgets.QFrame()
        header.setObjectName("headerCard")
        # Kept as self._header_frame so the fullscreen-video toggle can
        # hide/show it as one unit (see _enter_video_fullscreen below).
        self._header_frame = header
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(14, 8, 14, 8)

        title_box = QtWidgets.QVBoxLayout()
        title_box.setSpacing(0)

        title = QtWidgets.QLabel(tr("AI DRONE DETECTION LAB"))
        title.setObjectName("appTitle")
        subtitle = QtWidgets.QLabel(
            tr("Autonomous AI Survey  •  Hula Drone Operations")
        )
        subtitle.setObjectName("appSubtitle")

        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header_layout.addLayout(title_box)
        header_layout.addStretch()

        self.code_editor_btn = QtWidgets.QPushButton(tr("🧑‍💻 Code Editor"))
        self.code_editor_btn.clicked.connect(self._open_code_editor)
        header_layout.addWidget(self.code_editor_btn)

        self.label_train_btn = QtWidgets.QPushButton(tr("🏷️ Label & Train AI"))
        self.label_train_btn.setToolTip(tr(
            "Turn a video into a working AI model: extract frames, draw "
            "bounding boxes, train a YOLOv8 model, and save it straight "
            "into this project's models folder."
        ))
        self.label_train_btn.clicked.connect(self._open_label_train_dialog)
        header_layout.addWidget(self.label_train_btn)

        # Developer Mode gate: Manual Flight (keyboard) skips every
        # waypoint/geofence safety check the Grid mission
        # has, so it stays hidden behind this toggle rather than being a
        # button any student can press by accident.
        self.dev_mode_btn = QtWidgets.QPushButton(tr("🔧 Developer Mode"))
        self.dev_mode_btn.setCheckable(True)
        self.dev_mode_btn.setToolTip(tr(
            "Unlocks 🕹️ Manual Flight (raw keyboard control - no waypoints, "
            "no geofence). Off by default on purpose."
        ))
        self.dev_mode_btn.toggled.connect(self._toggle_developer_mode)
        header_layout.addWidget(self.dev_mode_btn)

        self.manual_flight_btn = QtWidgets.QPushButton(tr("🕹️ Manual Flight"))
        self.manual_flight_btn.setToolTip(tr(
            "Fly the drone directly by keyboard (WASD/arrows, R/F, Q/E). "
            "Requires Developer Mode."
        ))
        self.manual_flight_btn.clicked.connect(self._open_manual_flight_dialog)
        self.manual_flight_btn.setVisible(False)
        header_layout.addWidget(self.manual_flight_btn)

        # Simulation Mode: swaps self.drone for SimulatedDroneController (no
        # real drone/Wi-Fi needed - laptop webcam stands in for the drone
        # camera) so every flow above (manual flight, autonomous grid)
        # can be tried out without hardware. Only
        # switchable before anything has connected - see
        # _toggle_simulation_mode().
        self.sim_mode_btn = QtWidgets.QPushButton(tr("🧪 Simulation Mode"))
        self.sim_mode_btn.setCheckable(True)
        self.sim_mode_btn.setToolTip(tr(
            "No real drone needed - flight commands are simulated and the "
            "laptop webcam stands in for the drone camera. Switch this "
            "BEFORE connecting/flying."
        ))
        self.sim_mode_btn.toggled.connect(self._toggle_simulation_mode)
        header_layout.addWidget(self.sim_mode_btn)

        self.camera_label = QtWidgets.QLabel(tr("● CAMERA OFF"))
        self.camera_label.setObjectName("dangerPill")
        self.camera_label.setAlignment(QtCore.Qt.AlignCenter)
        header_layout.addWidget(self.camera_label)

        root_layout.addWidget(header)

        # =========================================================
        # MAIN 3-COLUMN AREA
        # LEFT = NAV / DETECTION
        # CENTER = CAMERA
        # RIGHT = MODELS / TELEMETRY / CONTROLS
        # =========================================================
        main_split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_split.setChildrenCollapsible(False)
        root_layout.addWidget(main_split, 1)

        # ---------------------------------------------------------
        # LEFT PANEL
        # ---------------------------------------------------------
        left_scroll = QtWidgets.QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarAlwaysOff
        )
        left_scroll.setMinimumWidth(240)
        left_scroll.setMaximumWidth(300)
        # Hidden (not removed) while the live camera is expanded to
        # fullscreen - see _enter_video_fullscreen.
        self._left_panel = left_scroll

        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)
        left_scroll.setWidget(left_widget)

        # Detection summary
        detection_card = QtWidgets.QFrame()
        detection_card.setObjectName("card")
        detection_layout = QtWidgets.QGridLayout(detection_card)
        detection_layout.setContentsMargins(10, 10, 10, 10)
        detection_layout.setHorizontalSpacing(8)
        detection_layout.setVerticalSpacing(2)

        section = QtWidgets.QLabel(tr("DETECTION"))
        section.setObjectName("sectionTitle")
        detection_layout.addWidget(section, 0, 0, 1, 2)

        self.current_label = QtWidgets.QLabel("0")
        self.current_label.setObjectName("valueBig")
        self.max_label = QtWidgets.QLabel("0")
        self.max_label.setObjectName("valueBig")
        self.targets_label = QtWidgets.QLabel("0")
        self.targets_label.setObjectName("valueAccent")

        detection_layout.addWidget(QtWidgets.QLabel(tr("Current")), 1, 0)
        detection_layout.addWidget(self.current_label, 2, 0)
        detection_layout.addWidget(QtWidgets.QLabel(tr("Maximum")), 1, 1)
        detection_layout.addWidget(self.max_label, 2, 1)
        detection_layout.addWidget(QtWidgets.QLabel(tr("Unique Targets")), 3, 0)
        detection_layout.addWidget(self.targets_label, 3, 1)
        detection_layout.setColumnStretch(0, 1)
        detection_layout.setColumnStretch(1, 1)

        left_layout.addWidget(detection_card)

        # Current target
        target_card = QtWidgets.QFrame()
        target_card.setObjectName("card")
        target_layout = QtWidgets.QVBoxLayout(target_card)
        target_layout.setContentsMargins(10, 10, 10, 10)
        target_layout.setSpacing(4)

        section = QtWidgets.QLabel(tr("CURRENT TARGET"))
        section.setObjectName("sectionTitle")
        target_layout.addWidget(section)

        self.target_label = QtWidgets.QLabel(tr("NO TARGET"))
        self.target_label.setObjectName("valueAccent")
        self.target_label.setWordWrap(True)
        target_layout.addWidget(self.target_label)

        self.target_conf_label = QtWidgets.QLabel(tr("Confidence: --"))
        self.target_stability_label = QtWidgets.QLabel(tr("Stability: --"))
        self.target_frames_label = QtWidgets.QLabel(tr("Frames: --"))
        self.target_status_label = QtWidgets.QLabel(f"{tr('STATUS')}: {tr('SEARCHING')}")
        self.target_conf_label.setObjectName("muted")
        self.target_stability_label.setObjectName("muted")
        self.target_frames_label.setObjectName("muted")
        self.target_status_label.setObjectName("muted")
        self.target_conf_label.setWordWrap(True)
        self.target_stability_label.setWordWrap(True)
        self.target_frames_label.setWordWrap(True)
        self.target_status_label.setWordWrap(True)

        target_layout.addWidget(self.target_conf_label)
        target_layout.addWidget(self.target_stability_label)
        target_layout.addWidget(self.target_frames_label)
        target_layout.addWidget(self.target_status_label)

        # Operator review (CONFIRM/REJECT) — hidden by request, since this
        # setup relies on automatic detection rather than a manual
        # human-in-the-loop review step. The widgets and their underlying
        # logic (detector.confirm_target/reject_target, enable/disable
        # wiring in _update_target_display) are left fully intact and just
        # not shown, so this is a one-line change to bring back if a
        # future setup wants the manual review workflow again.
        review_row = QtWidgets.QHBoxLayout()
        self.confirm_btn = QtWidgets.QPushButton(tr("✔ CONFIRM"))
        self.confirm_btn.setObjectName("primaryButton")
        self.reject_btn = QtWidgets.QPushButton(tr("✘ REJECT"))
        self.reject_btn.setObjectName("dangerButton")
        self.confirm_btn.setEnabled(False)
        self.reject_btn.setEnabled(False)
        review_row.addWidget(self.confirm_btn)
        review_row.addWidget(self.reject_btn)
        target_layout.addLayout(review_row)

        self.reject_reason_combo = QtWidgets.QComboBox()
        self.reject_reason_combo.addItems([
            tr("Wrong Object"), tr("Shadow / Lighting"), tr("Background Clutter"),
            tr("Duplicate Target"), tr("Other"),
        ])
        target_layout.addWidget(self.reject_reason_combo)

        self.confirm_btn.clicked.connect(self._confirm_current_target)
        self.reject_btn.clicked.connect(self._reject_current_target)

        self.confirm_btn.setVisible(False)
        self.reject_btn.setVisible(False)
        self.reject_reason_combo.setVisible(False)

        left_layout.addWidget(target_card)

        # Models
        models_card = QtWidgets.QFrame()
        models_card.setObjectName("card")
        models_layout = QtWidgets.QVBoxLayout(models_card)
        models_layout.setContentsMargins(10, 10, 10, 10)
        models_layout.setSpacing(5)

        self.model_label = QtWidgets.QLabel(tr("AI MODELS  •  scanning..."))
        self.model_label.setObjectName("sectionTitle")
        self.model_label.setWordWrap(True)
        models_layout.addWidget(self.model_label)

        self.model_list = QtWidgets.QPlainTextEdit()
        self.model_list.setObjectName("modelList")
        self.model_list.setReadOnly(True)
        self.model_list.setMinimumHeight(70)
        self.model_list.setMaximumHeight(110)
        models_layout.addWidget(self.model_list)
        self.choose_models_btn = QtWidgets.QPushButton(tr("🧠  Choose Models"))
        self.choose_models_btn.setToolTip(tr(
            "Pick which .pt models the AI uses (all of them, or only the ones you tick). "
            "Works while the camera is running."
        ))
        self.choose_models_btn.clicked.connect(self._open_model_select)
        models_layout.addWidget(self.choose_models_btn)

        left_layout.addWidget(models_card)
        left_layout.addStretch()

        # ---------------------------------------------------------
        # CENTER PANEL - LIVE CAMERA
        # ---------------------------------------------------------
        center = QtWidgets.QFrame()
        center_layout = QtWidgets.QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(8)

        video_card = QtWidgets.QFrame()
        video_card.setObjectName("card")
        video_layout = QtWidgets.QVBoxLayout(video_card)
        video_layout.setContentsMargins(9, 9, 9, 9)
        video_layout.setSpacing(7)

        video_header = QtWidgets.QHBoxLayout()
        video_title = QtWidgets.QLabel(tr("LIVE CAMERA / AI OVERLAY"))
        video_title.setObjectName("sectionTitle")
        self.video_info_label = QtWidgets.QLabel(tr("WAITING FOR VIDEO"))
        self.video_info_label.setObjectName("muted")

        # Toggles the live camera to real OS fullscreen (hides every other
        # panel). This only changes what is drawn on screen - the camera
        # and AI threads (CameraWorker / DetectionManager) run completely
        # independently of this widget and keep working exactly the same
        # whether the video is small or fullscreen. See
        # _enter_video_fullscreen / _exit_video_fullscreen.
        self.video_fullscreen_btn = QtWidgets.QPushButton(tr("⛶ Expand"))
        self.video_fullscreen_btn.setToolTip(tr(
            "Expand the live camera to fill the whole screen.\n"
            "Object detection keeps running exactly the same - this only "
            "changes what's shown, not the AI pipeline."
        ))

        video_header.addWidget(video_title)
        video_header.addStretch()
        video_header.addWidget(self.video_info_label)
        video_header.addSpacing(10)
        video_header.addWidget(self.video_fullscreen_btn)
        video_layout.addLayout(video_header)

        self.video = QtWidgets.QLabel(tr("Waiting for Hula camera..."))
        self.video.setObjectName("videoPlaceholder")
        self.video.setAlignment(QtCore.Qt.AlignCenter)
        self.video.setMinimumSize(480, 270)
        self.video.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding
        )
        video_layout.addWidget(self.video, 1)

        center_layout.addWidget(video_card, 1)

        # ---------------------------------------------------------
        # RIGHT PANEL
        # ---------------------------------------------------------
        right_scroll = QtWidgets.QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarAlwaysOff
        )
        right_scroll.setMinimumWidth(260)
        right_scroll.setMaximumWidth(330)
        # Hidden (not removed) while the live camera is expanded to
        # fullscreen - see _enter_video_fullscreen. Mission controls
        # (Stop/Emergency Stop) live here, which is why fullscreen mode
        # provides its own floating Emergency Stop button instead of
        # leaving the operator with no way to land the drone.
        self._right_panel = right_scroll

        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)
        right_scroll.setWidget(right_widget)

        # Mission / telemetry
        telemetry_card = QtWidgets.QFrame()
        telemetry_card.setObjectName("card")
        telemetry_layout = QtWidgets.QGridLayout(telemetry_card)
        telemetry_layout.setContentsMargins(10, 10, 10, 10)
        telemetry_layout.setHorizontalSpacing(8)
        telemetry_layout.setVerticalSpacing(4)

        section = QtWidgets.QLabel(tr("MISSION / TELEMETRY"))
        section.setObjectName("sectionTitle")
        telemetry_layout.addWidget(section, 0, 0, 1, 2)

        self.position_label = QtWidgets.QLabel(
            tr("WP -- / --  •  X=0  Y=0 cm  •  EAST")
        )
        self.position_label.setObjectName("valueAccent")
        self.position_label.setWordWrap(True)

        self.battery_label = QtWidgets.QLabel("🔋 --%")
        self.grid_label = QtWidgets.QLabel(
            f"{tr('Field')} {config.FIELD_WIDTH_CM} × {config.FIELD_HEIGHT_CM} cm\n"
            f"{tr('Grid spacing')} {config.GRID_SPACING_CM} cm"
        )
        self.frames_label = QtWidgets.QLabel(f"{tr('Frames analyzed')}: 0")
        self.captures_label = QtWidgets.QLabel(f"{tr('Images captured')}: 0")
        self.frames_label.setObjectName("muted")
        self.captures_label.setObjectName("muted")
        self.grid_label.setWordWrap(True)
        self.frames_label.setWordWrap(True)
        self.captures_label.setWordWrap(True)

        telemetry_layout.addWidget(self.position_label, 1, 0, 1, 2)
        telemetry_layout.addWidget(self.battery_label, 2, 0)
        telemetry_layout.addWidget(self.grid_label, 2, 1)
        telemetry_layout.addWidget(self.frames_label, 3, 0)
        telemetry_layout.addWidget(self.captures_label, 3, 1)
        telemetry_layout.setColumnStretch(0, 1)
        telemetry_layout.setColumnStretch(1, 1)

        right_layout.addWidget(telemetry_card)

        # Controls
        controls_card = QtWidgets.QFrame()
        controls_card.setObjectName("card")
        controls_layout = QtWidgets.QVBoxLayout(controls_card)
        controls_layout.setContentsMargins(10, 10, 10, 10)
        controls_layout.setSpacing(6)

        # ---------------------------------------------------------
        # AI TEST MODE - laptop webcam instead of the drone camera.
        # Lets a student test the YOLO model(s) with zero drone hardware.
        # ---------------------------------------------------------
        webcam_section = QtWidgets.QLabel(tr("🎓 AI TEST — WEBCAM (NO DRONE NEEDED)"))
        webcam_section.setObjectName("sectionTitle")
        controls_layout.addWidget(webcam_section)

        # Single on/off toggle button rather than separate Start/Stop
        # buttons - one click turns the webcam on, the same button (now
        # relabelled) turns it back off.
        self.webcam_toggle_btn = QtWidgets.QPushButton(tr("🖥️  START WEBCAM TEST"))
        self.webcam_toggle_btn.setObjectName("primaryButton")
        controls_layout.addWidget(self.webcam_toggle_btn)

        self.webcam_status_label = QtWidgets.QLabel(tr("⚪ Webcam: OFF"))
        self.webcam_status_label.setObjectName("muted")
        self.webcam_status_label.setWordWrap(True)
        controls_layout.addWidget(self.webcam_status_label)

        webcam_sep = QtWidgets.QFrame()
        webcam_sep.setFrameShape(QtWidgets.QFrame.HLine)
        controls_layout.addWidget(webcam_sep)

        section = QtWidgets.QLabel(tr("DRONE MISSION CONTROLS"))
        section.setObjectName("sectionTitle")
        controls_layout.addWidget(section)

        def group_label(text):
            lbl = QtWidgets.QLabel(text)
            lbl.setStyleSheet(
                "color:#5d7185; font-size:8.5pt; font-weight:700; "
                "letter-spacing:1px; padding-top:2px;"
            )
            return lbl

        self.preflight_btn = QtWidgets.QPushButton(tr("📷  Camera Down 90°"))
        self.start_btn = QtWidgets.QPushButton(tr("▶  Start AI Camera"))
        self.stop_btn = QtWidgets.QPushButton(tr("■  Stop"))
        self.stop_btn.setEnabled(False)
        self.record_photos_btn = QtWidgets.QPushButton(tr("📸  Record Training Photos"))
        self.record_photos_btn.setToolTip(tr(
            "Saves the drone camera's real view (no AI boxes drawn) once per "
            "second into AI Data & Training/drone_photos/. Import that folder in "
            "Label & Train AI - a model trained on the drone's own top-down view "
            "detects far more reliably in flight than one trained on phone photos."
        ))
        self.record_photos_btn.setEnabled(False)
        self.grid_btn = QtWidgets.QPushButton(tr("🗺️  Preview Grid"))
        self.auto_btn = QtWidgets.QPushButton(tr("🚁  START AUTONOMOUS GRID"))
        self.auto_btn.setEnabled(False)
        self.report_btn = QtWidgets.QPushButton(tr("📊  Post Flight Report"))
        self.emergency_btn = QtWidgets.QPushButton(tr("⚠  EMERGENCY STOP / LAND"))

        self.start_btn.setObjectName("primaryButton")
        self.auto_btn.setObjectName("primaryButton")
        self.emergency_btn.setObjectName("dangerButton")

        # The main "go" button for a mission gets extra height - the
        # single most important, most-pressed button on this panel should
        # be the easiest one to hit. Qt style sheets' "min-height" always
        # wins over a plain setMinimumHeight() call once any stylesheet is
        # in play (true here - see the QPushButton rule in setStyleSheet()
        # above), so the taller size has to be set in CSS, not C++.
        self.auto_btn.setStyleSheet("font-size: 10.5pt; min-height: 30px; padding: 12px 9px;")

        # --- 1. Prepare: setup actions taken before flying. This right-hand
        # column is only ~300px wide (see main_split.setSizes below) - not
        # enough room for two buttons side-by-side without clipping their
        # labels, so grouping here means a shared header + tight spacing,
        # not pairing buttons into rows.
        controls_layout.addWidget(group_label(tr("1 · PREPARE")))
        controls_layout.addWidget(self.preflight_btn)
        controls_layout.addWidget(self.grid_btn)

        # --- 2. Fly: the actual mission-start button stands alone,
        # full-width and taller, as the clear main action of this panel.
        controls_layout.addWidget(group_label(tr("2 · FLY")))
        controls_layout.addWidget(self.start_btn)
        controls_layout.addWidget(self.stop_btn)
        controls_layout.addWidget(self.record_photos_btn)
        controls_layout.addWidget(self.auto_btn)

        # --- 2B. Missions: other ways to fly, all on top of the same live
        # AI Detection (detections, captures and the post-flight report
        # keep working). One mission at a time; Autonomous Grid above is
        # unchanged.
        controls_layout.addWidget(group_label(tr("🎯 MISSIONS")))
        self.flight_plan_btn = QtWidgets.QPushButton(tr("🐍  Flight Plan (Python)"))
        self.flight_plan_btn.setToolTip(tr("Write Python code that flies the drone and reacts to what the AI sees."))
        self.keyboard_btn = QtWidgets.QPushButton(tr("⌨️  Keyboard Control"))
        self.keyboard_btn.setToolTip(tr("Connect, take off and fly with the keyboard while the AI detects objects from your trained model."))
        for b in (self.flight_plan_btn, self.keyboard_btn):
            controls_layout.addWidget(b)
        # Mission windows can be opened any time (write/edit code, look at
        # settings) - only actually FLYING needs a connected drone. Keep the
        # buttons' enabled state current even when no camera frames arrive.
        self._mission_btn_timer = QtCore.QTimer(self)
        self._mission_btn_timer.timeout.connect(self._refresh_mission_buttons)
        self._mission_btn_timer.timeout.connect(self._camera_watchdog)
        self._mission_btn_timer.start(500)

        # --- 3. After flight
        controls_layout.addWidget(group_label("3 · AFTER FLIGHT"))
        controls_layout.addWidget(self.report_btn)

        # Emergency stop is deliberately set apart with its own separator
        # and extra top margin, rather than sitting flush under Report -
        # a control this critical should never be one misplaced click away
        # from a routine button.
        emergency_sep = QtWidgets.QFrame()
        emergency_sep.setFrameShape(QtWidgets.QFrame.HLine)
        controls_layout.addSpacing(4)
        controls_layout.addWidget(emergency_sep)
        controls_layout.addSpacing(4)
        controls_layout.addWidget(self.emergency_btn)

        right_layout.addWidget(controls_card)
        right_layout.addStretch()

        main_split.addWidget(left_scroll)
        main_split.addWidget(center)
        main_split.addWidget(right_scroll)

        # Keep three zones visually balanced:
        # left 270 | center grows | right 300
        main_split.setSizes([270, 870, 300])
        main_split.setStretchFactor(0, 0)
        main_split.setStretchFactor(1, 1)
        main_split.setStretchFactor(2, 0)

        # =========================================================
        # BOTTOM: SYSTEM LOG
        # =========================================================
        terminal_card = QtWidgets.QFrame()
        terminal_card.setObjectName("card")
        # Hidden (not removed) while the live camera is expanded to
        # fullscreen - see _enter_video_fullscreen.
        self._log_card = terminal_card
        terminal_layout = QtWidgets.QVBoxLayout(terminal_card)
        terminal_layout.setContentsMargins(9, 7, 9, 7)
        terminal_layout.setSpacing(4)

        terminal_header = QtWidgets.QHBoxLayout()
        terminal_title = QtWidgets.QLabel(tr("SYSTEM LOG"))
        terminal_title.setObjectName("sectionTitle")
        terminal_hint = QtWidgets.QLabel(
            tr("Hula SDK  •  AI  •  Navigation")
        )
        terminal_hint.setObjectName("muted")

        terminal_header.addWidget(terminal_title)
        terminal_header.addStretch()
        terminal_header.addWidget(terminal_hint)
        terminal_layout.addLayout(terminal_header)

        self.terminal = QtWidgets.QPlainTextEdit()
        self.terminal.setObjectName("terminal")
        self.terminal.setReadOnly(True)
        self.terminal.setMaximumHeight(105)
        terminal_layout.addWidget(self.terminal)

        root_layout.addWidget(terminal_card)

        # =========================================================
        # FOOTER: sponsor / partner logos
        # =========================================================
        # A slim strip at the very bottom of the window rather than in the
        # header - the source file (logo.png at the project root) is a
        # very wide banner (1920x283, about 6.8:1) with its own white
        # background baked in, which read as an oversized white block
        # competing with the title when it was tried up top. Down here,
        # small and out of the way, it stays pure attribution rather than
        # UI. Guarded so a missing/corrupt logo.png just means no footer
        # row - never a startup crash.
        # Wrapped in a QWidget (rather than added as a bare QHBoxLayout) so
        # it is one widget the fullscreen-video toggle can hide/show, same
        # as the header/left/right panels and the system log below.
        self._footer_widget = None
        logo_path = config.PROJECT_ROOT / "logo.png"
        if logo_path.exists():
            logo_pixmap = QtGui.QPixmap(str(logo_path))
            if not logo_pixmap.isNull():
                self._footer_widget = QtWidgets.QWidget()
                footer_row = QtWidgets.QHBoxLayout(self._footer_widget)
                footer_row.setContentsMargins(0, 2, 0, 0)
                logo_label = QtWidgets.QLabel()
                logo_label.setPixmap(
                    logo_pixmap.scaledToHeight(26, QtCore.Qt.SmoothTransformation)
                )
                footer_row.addStretch()
                footer_row.addWidget(logo_label)
                footer_row.addStretch()
                root_layout.addWidget(self._footer_widget)

        # =========================================================
        # FULLSCREEN OVERLAY (Emergency Stop + Exit)
        # =========================================================
        # Only shown while the live camera view is expanded to fill the
        # whole screen (header/left/right panels/system log/footer all
        # hidden then) - without this, an operator in fullscreen would
        # have no way to land the drone or get back to the normal layout.
        # Floats on top of `root`, absolute-positioned in the top-right
        # corner; repositioned on every resize while active (resizeEvent
        # below) since it isn't managed by root_layout.
        self._is_video_fullscreen = False
        self.fullscreen_overlay = QtWidgets.QWidget(root)
        self.fullscreen_overlay.setStyleSheet(
            "background: rgba(10,16,22,225); border: 1px solid #33495c; "
            "border-radius: 10px;"
        )
        overlay_layout = QtWidgets.QHBoxLayout(self.fullscreen_overlay)
        overlay_layout.setContentsMargins(10, 8, 10, 8)
        overlay_layout.setSpacing(8)

        self.fullscreen_emergency_btn = QtWidgets.QPushButton(tr("⚠ EMERGENCY STOP / LAND"))
        self.fullscreen_emergency_btn.setObjectName("dangerButton")

        self.fullscreen_exit_btn = QtWidgets.QPushButton(tr("✕ Exit Fullscreen (Esc)"))

        overlay_layout.addWidget(self.fullscreen_emergency_btn)
        overlay_layout.addWidget(self.fullscreen_exit_btn)
        self.fullscreen_overlay.hide()

        # =========================================================
        # SIGNALS - SAME LOGIC AS THE ORIGINAL MAIN WINDOW
        # =========================================================
        self.start_btn.clicked.connect(self.start_camera)
        self.stop_btn.clicked.connect(self.stop_camera)
        self.preflight_btn.clicked.connect(self.set_camera_down)
        self.grid_btn.clicked.connect(self.preview_grid)
        self.auto_btn.clicked.connect(self.start_autonomous_grid)
        self.report_btn.clicked.connect(self._show_post_flight_report)
        self.record_photos_btn.clicked.connect(self._toggle_photo_recording)
        self.flight_plan_btn.clicked.connect(self._open_flight_plan_mission)
        self.keyboard_btn.clicked.connect(self._open_keyboard_mission)
        self.emergency_btn.clicked.connect(self.emergency_stop)
        self.webcam_toggle_btn.clicked.connect(self._toggle_webcam_test)
        self.video_fullscreen_btn.clicked.connect(self._toggle_video_fullscreen)
        # Same connect pattern as self.emergency_btn above (clicked's bool
        # "checked" arg lands in emergency_stop's blocking=True default
        # param position either way - this matches existing behavior).
        self.fullscreen_emergency_btn.clicked.connect(self.emergency_stop)
        self.fullscreen_exit_btn.clicked.connect(self._exit_video_fullscreen)

    def _toggle_video_fullscreen(self):
        if self._is_video_fullscreen:
            self._exit_video_fullscreen()
        else:
            self._enter_video_fullscreen()

    def _enter_video_fullscreen(self):
        """Hide every panel except the live camera and blow the whole
        window up to real OS fullscreen. This is purely a display change:
        CameraWorker and DetectionManager keep running on their own
        threads exactly as before (see live_frame_signal /
        _display_live_frame) - they have no idea whether self.video is on
        screen at all, so hiding/showing panels here cannot affect
        detection in any way."""
        if self._is_video_fullscreen:
            return
        self._is_video_fullscreen = True
        self._header_frame.hide()
        self._left_panel.hide()
        self._right_panel.hide()
        self._log_card.hide()
        if self._footer_widget is not None:
            self._footer_widget.hide()
        self.video_fullscreen_btn.setText(tr("🗗 Restore"))
        self.showFullScreen()
        self.fullscreen_overlay.show()
        self.fullscreen_overlay.raise_()
        self._reposition_fullscreen_overlay()

    def _exit_video_fullscreen(self):
        if not self._is_video_fullscreen:
            return
        self._is_video_fullscreen = False
        self._header_frame.show()
        self._left_panel.show()
        self._right_panel.show()
        self._log_card.show()
        if self._footer_widget is not None:
            self._footer_widget.show()
        self.video_fullscreen_btn.setText(tr("⛶ Expand"))
        self.fullscreen_overlay.hide()
        self.showNormal()

    def _reposition_fullscreen_overlay(self):
        """Keeps the floating Emergency Stop / Exit Fullscreen controls
        pinned to the top-right corner of the window as it resizes (the
        overlay is absolute-positioned on `root`, not managed by
        root_layout, so it needs manual repositioning)."""
        self.fullscreen_overlay.adjustSize()
        parent = self.fullscreen_overlay.parentWidget()
        if parent is None:
            return
        margin = 18
        x = max(margin, parent.width() - self.fullscreen_overlay.width() - margin)
        self.fullscreen_overlay.move(x, margin)

    def _open_code_editor(self):
        if self._code_editor_dialog is None:
            self._code_editor_dialog = CodeEditorDialog(self)
        self._code_editor_dialog.show()
        self._code_editor_dialog.raise_()
        self._code_editor_dialog.activateWindow()

    def _open_label_train_dialog(self):
        if self._label_train_dialog is None:
            self._label_train_dialog = LabelTrainDialog(self)
        self._label_train_dialog.show()
        self._label_train_dialog.raise_()
        self._label_train_dialog.activateWindow()

    def _toggle_simulation_mode(self, enabled: bool):
        if self.drone.connected or self.camera is not None:
            # Revert the click - swapping self.drone out from under an
            # already-connected session (or a running CameraWorker/grid
            # mission still holding a reference to the old instance) would
            # leave those threads pointed at a stale controller.
            self.sim_mode_btn.blockSignals(True)
            self.sim_mode_btn.setChecked(not enabled)
            self.sim_mode_btn.blockSignals(False)
            self._on_status(tr(
                "❌ Can't switch Simulation Mode after connecting — restart "
                "the app first"
            ))
            return
        self.drone = SimulatedDroneController() if enabled else HulaDroneController()
        # Any previously-built mission window holds a reference to the OLD
        # drone instance - point them all at the new one (they can already
        # be open, e.g. a Flight Plan written before switching modes).
        for dlg in (self._manual_flight_dialog, self._flight_plan_dialog, self._keyboard_dialog):
            if dlg is not None:
                dlg.drone = self.drone
        self.sim_mode_btn.setText(tr("🧪 Simulation: ON (no real drone)") if enabled else tr("🧪 Simulation Mode"))
        self._on_status(
            tr("🧪 Simulation Mode ON — flight is simulated, webcam stands in for the drone camera")
            if enabled else tr("🧪 Simulation Mode OFF — using the real drone")
        )

    def _toggle_developer_mode(self, enabled: bool):
        self.manual_flight_btn.setVisible(enabled)
        self._on_status(
            tr("🔧 Developer Mode ON — Manual Flight unlocked") if enabled
            else tr("🔧 Developer Mode OFF")
        )

    def _open_manual_flight_dialog(self):
        if self._manual_flight_dialog is None:
            self._manual_flight_dialog = ManualFlightDialog(self.drone, self)
        self._manual_flight_dialog.show()
        self._manual_flight_dialog.raise_()
        self._manual_flight_dialog.activateWindow()

    def set_camera_down(self):
        if not self.drone.connected:
            ok = self.drone.connect()
            if not ok:
                self._on_status(tr("❌ Drone connection failed"))
                return False
        ok = self.drone.set_camera_down()
        self._on_status(tr("✓ Camera DOWN 90° ready") if ok else tr("❌ Camera DOWN 90° failed"))
        return ok

    def preview_grid(self):
        # Reused rather than rebuilt on every click (same pattern as the
        # Code Editor / Label & Train dialogs) - this is what lets it stay
        # open and keep updating while a mission actually flies, instead
        # of being a one-shot "look, then close" popup.
        if self._grid_preview_dialog is not None:
            self._grid_preview_dialog.show()
            self._grid_preview_dialog.raise_()
            self._grid_preview_dialog.activateWindow()
            return

        points = generate_waypoint_objects()
        self._on_status(f"🗺️ Grid generated: {len(points)} waypoints (flight commands NOT sent)")

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(tr("🗺️ Autonomous Survey Grid Preview"))
        # Non-modal: previously this was exec()'d, which blocks the main
        # window while open - fine for a pre-flight-only look, but it
        # would make it impossible to also press Start Autonomous Grid and
        # watch this window track the flight live, which is the whole
        # point of the position marker below.
        dialog.setModal(False)
        dialog.resize(1050, 760)
        dialog.setMinimumSize(850, 620)

        from .panel_theme import DIALOG_STYLE, prepare, heading
        dialog.setStyleSheet(DIALOG_STYLE)
        root = QtWidgets.QVBoxLayout(dialog)
        prepare(root)
        heading(root, tr("Survey route"), tr("Review the route and follow mission progress."), "route")
        title = QtWidgets.QLabel(
            f"AUTONOMOUS SURVEY PLAN  |  {config.FIELD_WIDTH_CM} × {config.FIELD_HEIGHT_CM} cm  |  "
            f"Spacing {config.GRID_SPACING_CM} cm  |  {len(points)} waypoints"
        )
        title.setWordWrap(True)
        title.setStyleSheet("font-size:13px;font-weight:600;padding:8px;color:#a1b7c9;")
        root.addWidget(title)

        # Live flight status - starts as an idle/preview note, gets
        # rewritten with the current waypoint/position/heading by
        # _on_grid_progress_ui() while an autonomous mission is running,
        # and with a finished/stopped note by _grid_finished_ui().
        live_label = QtWidgets.QLabel(tr("⚪ PREVIEW ONLY — showing the planned route (not flying)"))
        live_label.setStyleSheet(
            "color:#8ea0b2; font-size:11px; font-weight:800; padding:0 8px 6px 8px;"
        )
        root.addWidget(live_label)
        self._grid_live_label = live_label

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        root.addWidget(splitter, 1)

        # 🧊 3D route (default tab) + the original 2D top-down map, both
        # following the live drone position (ui/route_3d.py).
        from .route_3d import SurveyRouteViews
        canvas = SurveyRouteViews(points, _GridPreviewWidget(points, self))
        splitter.addWidget(canvas)
        self._grid_preview_widget = canvas

        details = QtWidgets.QPlainTextEdit()
        details.setReadOnly(True)
        details.setMinimumWidth(310)
        lines = [
            tr("FLIGHT COMMANDS: NOT SENT"),
            "",
            tr("Route: SERPENTINE / LAWN-MOWER"),
            f"{tr('Field')}: {config.FIELD_WIDTH_CM} × {config.FIELD_HEIGHT_CM} cm",
            f"{tr('Grid spacing')}: {config.GRID_SPACING_CM} cm",
            f"{tr('Survey altitude')}: {config.SURVEY_ALTITUDE_CM} cm",
            f"{tr('Waypoints')}: {len(points)}",
            "",
            tr("WAYPOINT PLAN"),
        ]
        for wp in points:
            turn = f" | {wp.turn}" if wp.turn != "NONE" else ""
            lines.append(
                f"WP {wp.index:03d}  X={wp.x_cm:>3} Y={wp.y_cm:>3}  "
                f"{wp.heading:<5}{turn}"
            )
        details.setPlainText("\n".join(lines))
        splitter.addWidget(details)
        splitter.setSizes([700, 350])
        self._grid_details_widget = details

        close_btn = QtWidgets.QPushButton(tr("Close Preview"))
        close_btn.clicked.connect(dialog.close)
        root.addWidget(close_btn)

        dialog.finished.connect(self._on_grid_preview_closed)
        self._grid_preview_dialog = dialog
        dialog.show()

    def _on_grid_preview_closed(self, *_args):
        """Runs when the Grid Preview dialog is closed (Close Preview
        button, window X, or Qt destroying it on app exit). Drop every
        reference to its widgets so _on_grid_progress_ui() correctly
        no-ops instead of touching deleted Qt objects, and so the next
        Preview Grid click builds a fresh dialog rather than reusing a
        dead one."""
        self._grid_preview_dialog = None
        self._grid_preview_widget = None
        self._grid_details_widget = None
        self._grid_live_label = None

    def _highlight_current_waypoint_line(self, wp_index):
        """Best-effort: bold/tint the WP NNN line in the waypoint-list
        panel and scroll it into view, so the text list also points at
        where the drone currently is - not just the map. Silently does
        nothing if the dialog/text isn't there or the line can't be found
        (e.g. a future waypoint-numbering change); this is a convenience
        highlight, never something the live position display depends on."""
        details = self._grid_details_widget
        if details is None:
            return
        cursor = details.document().find(f"WP {wp_index:03d}")
        if cursor.isNull():
            return
        cursor.select(QtGui.QTextCursor.LineUnderCursor)
        # QPlainTextEdit.setExtraSelections() reuses QTextEdit's
        # ExtraSelection structure - QPlainTextEdit has no ExtraSelection
        # class of its own in PySide6 (confirmed by a real AttributeError
        # here during QA against actual waypoint data before this fix).
        selection = QtWidgets.QTextEdit.ExtraSelection()
        selection.cursor = cursor
        fmt = QtGui.QTextCharFormat()
        fmt.setBackground(QtGui.QColor("#1b4738"))
        fmt.setForeground(QtGui.QColor("#dfffee"))
        fmt.setFontWeight(QtGui.QFont.Bold)
        selection.format = fmt
        details.setExtraSelections([selection])
        details.setTextCursor(cursor)
        details.ensureCursorVisible()

    def _show_post_flight_report(self, modal: bool = True):
        # Confirmed against detection_manager.py: DetectionManager.capture
        # is a CaptureManager instance exposing `.records` and `.mission_dir`.
        # modal=False shows the report without blocking the GUI thread.
        if not self.detector:
            self._on_status(tr("❌ Start the AI camera first — no capture data yet"))
            return
        dialog = PostFlightReportDialog(self.detector.capture, self)
        if modal:
            dialog.exec()
        else:
            dialog.setModal(False)
            dialog.show()

    def start_camera(self):
        if self._webcam_thread is not None:
            self._on_status(tr("❌ Stop the webcam AI test before starting the drone camera"))
            return
        # Re-arm the battery-critical / disconnect alerts for this new
        # flight session so they can fire again (they latch "shown" once
        # per session to avoid spamming the same popup every 5s poll).
        self._battery_alert_shown = False
        self._battery_poll_fail_count = 0
        try:
            # Connect first; CameraWorker repeats this check safely.
            if not self.drone.connect():
                self._on_status(tr("❌ Drone connection failed"))
                return
            # First touch of the AI stack (ultralytics -> torch). It costs
            # tens of seconds the first time, which is why it happens here,
            # on an explicit Start, rather than during app startup.
            self._on_status(tr("🧠 Loading AI engine (first start takes a moment)..."))
            from ..ai.detection_manager import DetectionManager

            self.detector = DetectionManager(
                on_update=lambda frame, data: self.frame_signal.emit(frame, data),
                on_status=lambda msg: self.status_signal.emit(msg),
            )
            models = self.detector.detector.model_status()
            self.model_label.setText(f"{tr('Models loaded')}: {len(models)}")
            self.model_list.setPlainText("\n".join(f"✅ {m['name']}" for m in models) or tr("❌ No .pt models loaded"))
            self._on_status(f"🤖 AI LAB MODELS: {len(models)} loaded")
            for model in models:
                self._on_status(f"   • {model['name']}")
            from ..ai import devices
            yolo = self.detector.detector
            self._on_status(f"🖥️ AI detection on {devices.describe(yolo.resolve_device())}")
            if yolo.device_note:
                self._on_status(yolo.device_note)
        except Exception as exc:
            self._on_status(f"❌ AI model startup failed: {exc}")
            return

        self.camera = CameraWorker(
            self.drone,
            # Cross-thread: never call a widget-touching method directly
            # from the camera thread. Emit the signal instead — Qt queues
            # the delivery onto the GUI thread automatically.
            on_frame=lambda f: self.live_frame_signal.emit(f),
            on_ai_frame=self.detector.submit_frame,
            on_status=lambda msg: self.status_signal.emit(msg),
        )
        self.camera.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.record_photos_btn.setEnabled(True)
        self.camera_label.setText(tr("● CAMERA STARTING"))

    # ------------------------------------------------------------------
    # Missions
    # ------------------------------------------------------------------
    def _active_mission(self):
        """Name of the mission window that is open (or still flying), else None."""
        for name, dlg in (
            ("flight_plan", self._flight_plan_dialog),
            ("keyboard", self._keyboard_dialog),
            ("dev_manual", self._manual_flight_dialog),
        ):
            if dlg is not None and (dlg.isVisible() or (hasattr(dlg, "is_busy") and dlg.is_busy())):
                return name
        return None

    def _refresh_mission_buttons(self):
        grid_running = bool(self.grid_mission and self.grid_mission.is_alive())
        active = self._active_mission()
        for name, btn in (
            ("flight_plan", self.flight_plan_btn),
            ("keyboard", self.keyboard_btn),
        ):
            btn.setEnabled(not grid_running and active in (None, name))
        if active is not None:
            self.auto_btn.setEnabled(False)

    def _mission_precheck(self) -> bool:
        if self.grid_mission and self.grid_mission.is_alive():
            self._on_status(tr("❌ Autonomous grid is running — wait for it to finish"))
            return False
        return True

    def _show_mission(self, dlg):
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()
        self._refresh_mission_buttons()

    def _open_flight_plan_mission(self):
        if not self._mission_precheck():
            return
        if self._flight_plan_dialog is None:
            self._flight_plan_dialog = FlightPlanDialog(
                self.drone,
                get_detections=lambda: list(self._latest_detections),
                save_photo=self._save_mission_photo,
                connect_drone=self._connect_for_mission,
                is_ready=self._mission_ready,
                video_age=self.video_age_sec,
                parent=self,
            )
            self._flight_plan_dialog.finished_signal.connect(lambda *_: self._refresh_mission_buttons())
        self._show_mission(self._flight_plan_dialog)

    def _open_keyboard_mission(self):
        if not self._mission_precheck():
            return
        if self._keyboard_dialog is None:
            self._keyboard_dialog = KeyboardControlDialog(
                self.drone, connect_drone=self._connect_for_mission,
                is_ready=self._mission_ready, video_age=self.video_age_sec, parent=self,
            )
        self._show_mission(self._keyboard_dialog)

    def _open_model_select(self):
        # Model selection imports the AI stack. Defer until requested so
        # the v5.1 dashboard can paint without waiting for PyTorch.
        from .model_select_dialog import ModelSelectDialog

        dlg = ModelSelectDialog(self)
        dlg.saved.connect(self._on_models_chosen)
        dlg.exec()

    def _on_models_chosen(self, mode, selected):
        if mode == "all":
            text = tr("🧠 Models: ALL .pt files in the models folder")
        elif mode == "selected":
            text = tr("🧠 Models chosen") + ": " + ", ".join(selected)
        else:
            text = tr("🧠 Models: automatic (every model you trained)")
        self._on_status(text)
        if self.detector is None and self._webcam_thread is None:
            self.model_label.setText(tr("AI MODELS  •  used at next start"))
            self.model_list.setPlainText(
                "\n".join(f"☑ {n}" for n in selected) if mode == "selected" else text
            )
        else:
            self._on_status(tr("   ↳ applied to the running AI — takes effect on the next frame"))

    def _camera_alive(self) -> bool:
        return bool(self.camera is not None and self.camera.is_alive())

    def video_age_sec(self) -> float:
        """Seconds since the last drone camera frame (inf = none yet)."""
        if not self._camera_alive() or not self._last_frame_time:
            return float("inf")
        return time.monotonic() - self._last_frame_time

    def _mission_ready(self) -> bool:
        # The camera THREAD must be running - not just "was started once".
        # It ends on its own after an Emergency Stop (which stops the video
        # stream), a 20 s no-video timeout or an RTP error; the old check
        # (self.camera is not None) then kept saying "Connected" while no
        # video - and therefore no AI boxes - ever arrived again.
        return bool(self.drone.connected and self._camera_alive())

    def _release_dead_camera(self):
        """The camera thread has ended by itself: release it (and its AI
        worker) so the next Connect / Start AI Camera starts a fresh one.
        Never lands or stops the drone - it only tidies up the video side."""
        cam = self.camera
        if cam is None or cam.is_alive():
            return
        self.camera = None
        self._last_frame_time = 0.0
        if self.detector:
            try:
                self.detector.stop()
            except Exception:
                pass
            self.detector = None
        try:
            self.drone.stop_stream()
        except Exception:
            pass
        if self._photo_record_dir is not None:
            self._toggle_photo_recording()
        self.record_photos_btn.setEnabled(False)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.camera_label.setText(tr("● CAMERA OFF"))
        self.video_info_label.setText(tr("WAITING FOR VIDEO"))
        self.video.setText(tr("Waiting for Hula camera..."))
        self._on_status(tr("📷 The drone video stopped — press Connect (or ▶ Start AI Camera) to start it again"))

    def _camera_watchdog(self):
        if self.camera is not None and not self.camera.is_alive():
            self._release_dead_camera()

    def _connect_for_mission(self) -> bool:
        """The ① Connect button inside a mission window: the same path as
        ▶ Start AI Camera (drone connection + live video + AI Detection with
        the models in the models folder)."""
        self._camera_watchdog()
        if self.camera is None:
            if self._webcam_thread is not None:
                # start_camera() refuses while the webcam test still runs -
                # stop it and WAIT for it (it used to be only asked to stop,
                # so Connect silently did nothing).
                wt = self._webcam_thread
                wt.stop()
                wt.join(timeout=3.0)
                self._on_webcam_thread_stopped()
            self.start_camera()
        return self._mission_ready()

    def _save_mission_photo(self):
        """drone.photo() in a Flight Plan: saves the current camera frame as
        evidence (same captures folder / Post Flight Report as the AI's own
        automatic captures). Called from the Flight Plan worker thread."""
        frame = self._last_raw_frame
        detector = self.detector
        if frame is None or detector is None:
            return None
        detections = list(self._latest_detections)
        with detector.lock:
            return detector.capture.save(frame.copy(), "flight_plan_photo", len(detections), detections)

    def _stop_missions(self):
        dlg = self._flight_plan_dialog
        if dlg is not None and dlg._runner is not None:
            dlg._runner.stop()

    def _toggle_photo_recording(self):
        """Start/stop saving raw camera frames for training. Frames are the
        camera's own image (saved BEFORE any AI boxes are drawn), one every
        config.TRAINING_PHOTO_INTERVAL_SEC, into a new folder per recording
        under config.DRONE_PHOTOS_DIR."""
        if self._photo_record_dir is None:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            folder = config.DRONE_PHOTOS_DIR / f"flight_{stamp}"
            try:
                folder.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                self._on_status(f"❌ {tr('Could not create photo folder')}: {exc}")
                return
            self._photo_record_dir = folder
            self._photo_record_count = 0
            self._photo_record_last = 0.0
            self.record_photos_btn.setText(tr("⏹  Stop Recording Photos") + " (0)")
            self._on_status(f"📸 {tr('Recording training photos to')}: {folder}")
        else:
            folder, count = self._photo_record_dir, self._photo_record_count
            self._photo_record_dir = None
            self.record_photos_btn.setText(tr("📸  Record Training Photos"))
            self._on_status(
                f"📸 {tr('Saved')} {count} {tr('photo(s) in')} {folder} — "
                + tr("import this folder in Label & Train AI (📁 Import a Folder of Photos).")
            )

    def _maybe_record_training_photo(self, frame):
        if self._photo_record_dir is None:
            return
        now = time.monotonic()
        interval = float(getattr(config, "TRAINING_PHOTO_INTERVAL_SEC", 1.0) or 1.0)
        if now - self._photo_record_last < interval:
            return
        self._photo_record_last = now
        path = self._photo_record_dir / f"drone_{self._photo_record_count:05d}.jpg"
        if imwrite_any(path, frame):
            self._photo_record_count += 1
            self.record_photos_btn.setText(
                tr("⏹  Stop Recording Photos") + f" ({self._photo_record_count})"
            )
        else:
            self._on_status(f"⚠️ {tr('Could not save photo')}: {path}")

    def _display_live_frame(self, frame):
        """Render the newest camera frame immediately; never run YOLO here."""
        if frame is None:
            return

        # Raw frame (before any overlay is drawn below) - see
        # _toggle_photo_recording() and _save_mission_photo().
        self._last_raw_frame = frame
        self._last_frame_time = time.monotonic()
        for dlg in (self._keyboard_dialog, self._flight_plan_dialog):
            if dlg is not None and dlg.isVisible():
                dlg.show_frame(frame, self._latest_detections)
        self._maybe_record_training_photo(frame)

        display = frame.copy()

        # Overlay the latest AI results onto the current live frame.
        # Drawing rectangles/text is intentionally lightweight compared with
        # YOLO inference, so the camera path stays responsive.
        for det in getattr(self, "_latest_detections", []):
            xyxy = det.get("xyxy")
            if not xyxy or len(xyxy) != 4:
                continue

            x1, y1, x2, y2 = [int(v) for v in xyxy]
            label = (
                f"{det.get('model_name', 'model')} | "
                f"{det.get('class_name', '?')} "
                f"{float(det.get('confidence', 0.0)):.0%}"
            )
            cv2.rectangle(display, (x1, y1), (x2, y2), (85, 214, 190), 2)
            put_text(display, label, (x1, max(20, y1 - 8)), 0.55, (223, 252, 247), 2)

        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QtGui.QImage(
            rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888
        ).copy()

        # Fast scaling keeps the live path lighter than smooth interpolation.
        pix = QtGui.QPixmap.fromImage(qimg)
        self.video.setPixmap(
            pix.scaled(
                self.video.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.FastTransformation,
            )
        )

    def stop_camera(self):
        if (self.grid_mission and self.grid_mission.is_alive()) or (
            self._active_mission() is not None and self.drone.flying
        ):
            self.emergency_stop()
            return
        if self.camera:
            self.camera.stop()
            # Must exceed HulaDroneController.stop_stream()'s own 5s join
            # timeout on the PyAV decode thread (see controller.py) — if we
            # give up before that inner join does, the decode thread (and
            # the UDP socket bound to its RTP port) can keep running in the
            # background after this call returns. The *next* drone's camera
            # session then reuses that same RTP port and never receives a
            # packet, hanging forever waiting for its first frame.
            self.camera.join(timeout=7.0)
            if self.camera.is_alive():
                self._on_status(
                    tr(
                        "⚠️ Previous camera session did not fully stop — "
                        "wait a moment before connecting the next drone "
                        "(its video may otherwise hang waiting for frames)"
                    )
                )
            self.camera = None

        if self.detector:
            self.detector.stop()
            self.detector = None

        self.drone.stop_stream()
        if self._photo_record_dir is not None:
            self._toggle_photo_recording()  # finish + report the recording
        self.record_photos_btn.setEnabled(False)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._refresh_mission_buttons()
        self.auto_btn.setEnabled(False)
        self.camera_label.setText(tr("● CAMERA OFF"))
        self.video_info_label.setText(tr("WAITING FOR VIDEO"))
        self._on_status(tr("🔒 AI Lab camera stopped"))

    def _toggle_webcam_test(self):
        """Single on/off button: starts the webcam test if it's off, stops
        it if it's on. This is the "test the camera before the drone"
        control - it must work with or without a trained YOLO model."""
        if self._webcam_thread is None:
            self._start_webcam_test()
        else:
            self._stop_webcam_test()

    def _set_webcam_button_running(self, running: bool) -> None:
        """Re-styles the toggle button in place (Qt requires an
        unpolish/polish cycle for an objectName-based stylesheet change to
        actually repaint)."""
        if running:
            self.webcam_toggle_btn.setText(tr("⏹  STOP WEBCAM TEST"))
            self.webcam_toggle_btn.setObjectName("dangerButton")
        else:
            self.webcam_toggle_btn.setText(tr("🖥️  START WEBCAM TEST"))
            self.webcam_toggle_btn.setObjectName("primaryButton")
        self.webcam_toggle_btn.style().unpolish(self.webcam_toggle_btn)
        self.webcam_toggle_btn.style().polish(self.webcam_toggle_btn)

    def _start_webcam_test(self):
        """Start testing the laptop's own webcam - no drone needed at all.
        Opens the camera and shows the live feed even if no YOLO model is
        loaded yet (AI overlays only appear once a .pt model exists), so a
        student can always confirm "does my camera work" first. Lives on
        the main dashboard so it's one click away from the first screen
        the app shows."""
        if self._webcam_thread is not None:
            return

        if self.camera is not None:
            answer = QtWidgets.QMessageBox.question(
                self,
                tr("Stop drone camera?"),
                tr(
                    "The drone AI camera must stop before testing with the webcam.\n"
                    "Stop it now and continue?"
                ),
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if answer != QtWidgets.QMessageBox.Yes:
                return
            self.stop_camera()

        self._set_webcam_button_running(True)
        self.webcam_status_label.setText(tr("🟢 Webcam: starting..."))
        self.video.setText(tr("Starting webcam..."))
        self.video_info_label.setText(tr("🖥️ WEBCAM TEST"))
        self._on_status(tr("🖥️ Starting webcam test..."))

        self._webcam_thread = WebcamAIThread(
            camera_index=DEFAULT_WEBCAM_INDEX,
            # Cross-thread: never touch widgets directly from the worker
            # thread - always emit a signal and let Qt queue delivery onto
            # the GUI thread, same rule CameraWorker follows above.
            on_frame=lambda frame, dets: self.webcam_frame_signal.emit(frame, dets),
            on_status=lambda msg: self.status_signal.emit(msg),
            on_error=lambda msg: self.status_signal.emit(msg),
            on_finished=lambda: self.webcam_thread_stopped_signal.emit(),
        )
        self._webcam_thread.start()

    def _stop_webcam_test(self):
        if self._webcam_thread is None:
            self._on_webcam_thread_stopped()
            return
        self._webcam_thread.stop()
        self.webcam_toggle_btn.setEnabled(False)
        self.webcam_status_label.setText(tr("🟡 Webcam: stopping..."))
        self._on_status(tr("🖥️ Stopping webcam test..."))

    @QtCore.Slot()
    def _on_webcam_thread_stopped(self):
        """Runs on the GUI thread once WebcamAIThread has fully exited,
        whatever the reason (normal stop, no camera found, model missing,
        or an inference error) - resets the toggle button exactly once."""
        self._webcam_thread = None
        self.webcam_toggle_btn.setEnabled(True)
        self._set_webcam_button_running(False)
        self.webcam_status_label.setText(tr("⚪ Webcam: OFF"))
        self.video_info_label.setText(tr("WAITING FOR VIDEO"))
        self.video.setText(tr("Waiting for Hula camera..."))

    @QtCore.Slot(object, object)
    def _display_webcam_frame(self, frame, detections):
        """Render the newest webcam AI-test frame (already annotated by
        YOLODetector.annotate when a model is loaded). Runs on the GUI
        thread only."""
        if frame is None:
            return
        # First frame that actually arrives is the real confirmation the
        # camera opened - flip the status line from "starting..." to a
        # clear "live" state so a student isn't left staring at "starting"
        # forever if they don't notice the video panel updating.
        if self._webcam_thread is not None:
            live_text = (
                f"{tr('🟢 Webcam: LIVE')} ({len(detections)} detection"
                f"{'s' if len(detections) != 1 else ''})"
                if detections else tr("🟢 Webcam: LIVE")
            )
            if self.webcam_status_label.text() != live_text:
                self.webcam_status_label.setText(live_text)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QtGui.QImage(
            rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888
        ).copy()
        pix = QtGui.QPixmap.fromImage(qimg)
        self.video.setPixmap(
            pix.scaled(
                self.video.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.FastTransformation,
            )
        )

    def start_autonomous_grid(self):
        if not config.AUTONOMOUS_FLIGHT_ENABLED or not config.TAKEOFF_ENABLED:
            self._on_status(tr("🔒 Autonomous flight is disabled in config.py"))
            return
        if not self.drone.connected:
            self._on_status(tr("❌ Connect the drone first"))
            return
        if self._active_mission() is not None:
            self._on_status(tr("❌ Close the open mission window before starting the autonomous grid"))
            return
        if self.drone.manual_flight_used:
            if self.drone.flying:
                self._on_status(tr("❌ Land the drone first (it was flown by a mission)"))
                return
            # A mission moved/turned the drone, so the grid's start
            # position and heading are unknown - ask the operator to put it
            # back, instead of refusing until the app is restarted.
            reply = QtWidgets.QMessageBox.question(
                self, tr("Put the drone back at the start"),
                tr("The drone was flown by a mission, so the grid no longer knows where it is.")
                + "\n\n" + tr("Place the drone at the grid's START corner, facing")
                + f" {config.INITIAL_HEADING}, " + tr("then press Yes."),
            )
            if reply != QtWidgets.QMessageBox.Yes:
                return
            self.drone.manual_flight_used = False
            self.drone.heading = config.INITIAL_HEADING
        if not self.drone.camera_down_ok:
            # Missions may have tilted the camera - point it down again.
            self.drone.set_camera_down()
        if not self.drone.camera_down_ok:
            self._on_status(tr("❌ Camera must be DOWN 90° before flight"))
            return
        if not self.camera:
            self._on_status(tr("❌ Start AI camera before autonomous flight"))
            return
        if self.grid_mission and self.grid_mission.is_alive():
            return

        points = generate_zigzag_waypoints()
        msg = (
            f"{tr('Start autonomous grid?')}\n\n"
            f"{tr('Field')}: {config.FIELD_WIDTH_CM} × {config.FIELD_HEIGHT_CM} cm\n"
            f"{tr('Spacing')}: {config.GRID_SPACING_CM} cm\n"
            f"{tr('Waypoints')}: {len(points)}\n"
            f"{tr('Survey altitude')}: {config.SURVEY_ALTITUDE_CM} cm\n\n"
            + tr("Make sure the 400×400 cm field is clear and the drone has enough battery.")
        )
        answer = QtWidgets.QMessageBox.warning(
            self, tr("Confirm Autonomous Flight"), msg,
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )
        if answer != QtWidgets.QMessageBox.Yes:
            return

        self.auto_btn.setEnabled(False)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_signal.emit(tr("🚁 AUTONOMOUS GRID STARTING..."))

        # If a student has the Grid Preview open (maybe still showing the
        # last flight's final position), clear it to a "taking off" state
        # right away rather than leaving a stale marker up until the first
        # real on_progress callback arrives a few seconds from now.
        if self._grid_preview_widget is not None:
            self._grid_preview_widget.clear_current_waypoint()
        if self._grid_live_label is not None:
            self._grid_live_label.setText(tr("🟡 TAKING OFF…"))
            self._grid_live_label.setStyleSheet(
                "color:#ffd166; font-size:11px; font-weight:800; padding:0 8px 6px 8px;"
            )
        if self._grid_details_widget is not None:
            self._grid_details_widget.setExtraSelections([])
        self.grid_mission = AutonomousGridMission(
            self.drone,
            on_status=lambda msg: self.status_signal.emit(msg),
            on_progress=lambda i, total, x, y, heading, turn: self._grid_progress(i, total, x, y, heading, turn),
            on_finished=self._grid_finished,
        )
        self.grid_mission.start()

    def _grid_progress(self, i, total, x, y, heading, turn):
        if self.detector:
            self.detector.set_survey_position(i, total, x, y, heading)
        self.status_signal.emit(f"📍 WP {i}/{total} | POS=({x},{y}) cm | HEADING={heading}" + (f" | {turn}" if turn != "NONE" else ""))
        # Marshal onto the GUI thread (this callback runs on
        # AutonomousGridMission's own worker thread) so the Grid Preview
        # dialog, if a student has it open, can move its position marker.
        self.grid_progress_signal.emit(i, total, x, y, heading, turn)

    @QtCore.Slot(int, int, int, int, str, str)
    def _on_grid_progress_ui(self, i, total, x, y, heading, turn):
        """Runs on the GUI thread. Moves the live position marker in the
        Grid Preview dialog to the waypoint the drone just reached, if
        that dialog happens to be open - a no-op otherwise (this never
        opens the dialog itself; Preview Grid does that)."""
        if self._grid_preview_widget is not None:
            self._grid_preview_widget.set_current_waypoint(i, x, y, heading)
        if self._grid_live_label is not None:
            self._grid_live_label.setText(
                f"🟢 LIVE — WP {i}/{total}  •  X={x} Y={y} cm  •  HEADING={heading}"
            )
            self._grid_live_label.setStyleSheet(
                "color:#8ce8c0; font-size:11px; font-weight:800; padding:0 8px 6px 8px;"
            )
        self._highlight_current_waypoint_line(i)

    def _grid_finished(self, success, reason):
        """Worker-thread callback: hand completion to the Qt GUI thread."""
        self.grid_finished_signal.emit(bool(success), str(reason))

    @QtCore.Slot(bool, str)
    def _grid_finished_ui(self, success, reason):
        """Runs on the MainWindow/Qt GUI thread."""
        self.grid_mission = None
        self.auto_btn.setEnabled(bool(self.camera and self.drone.camera_down_ok))
        self.start_btn.setEnabled(self.camera is None)
        self.stop_btn.setEnabled(self.camera is not None)
        self.status_signal.emit(
            tr("✅ Autonomous grid finished") if success else f"🛑 {tr('Autonomous grid stopped')}: {reason}"
        )
        # Leave the position marker exactly where it last was (the final
        # waypoint reached) rather than clearing it - that's genuinely
        # useful info ("this is as far as it got"), just relabel the live
        # status line so it reads as finished, not still-in-progress.
        # The 3D view lands its drone there.
        if self._grid_preview_widget is not None and hasattr(self._grid_preview_widget, "set_finished"):
            self._grid_preview_widget.set_finished(success)
        if self._grid_live_label is not None:
            if success:
                self._grid_live_label.setText(tr("✅ MISSION COMPLETE — showing final position reached"))
                self._grid_live_label.setStyleSheet(
                    "color:#8ce8c0; font-size:11px; font-weight:800; padding:0 8px 6px 8px;"
                )
            else:
                self._grid_live_label.setText(f"🛑 {tr('MISSION STOPPED')} — {reason} ({tr('showing last position reached')})")
                self._grid_live_label.setStyleSheet(
                    "color:#ff9ca5; font-size:11px; font-weight:800; padding:0 8px 6px 8px;"
                )

        # Flight plan is done (whether it completed normally or was
        # stopped early) — surface the post-flight report right away
        # instead of waiting for the operator to click the button.
        if self.detector:
            self._show_post_flight_report()

    def emergency_stop(self, blocking: bool = True):
        """Land immediately and kill the stream.

        `blocking=True` (default, used by the Emergency button and
        closeEvent) runs on the calling thread — fine there, since the
        caller is already expecting a short pause. `blocking=False` (used
        by the battery-critical / disconnect handlers below, which fire
        on the GUI thread via frame_signal) instead runs the actual SDK
        landing calls on a background thread so the ~2+ second
        touchdown() sleep never freezes the UI mid-flight.
        """
        self._on_status(tr("🛑 EMERGENCY STOP REQUESTED"))
        if self.grid_mission:
            self.grid_mission.request_stop()
        self._stop_missions()
        self.auto_btn.setEnabled(False)

        def _land():
            try:
                # If the link is already gone (e.g. battery-critical mid-flight
                # or a dropped connection) the underlying SDK calls inside
                # emergency_stop() can raise. Never let that propagate and
                # take the whole app down — this is a shutdown path, it
                # must not fail.
                self.drone.emergency_stop()
            except Exception as exc:
                self.status_signal.emit(f"⚠️ Emergency stop warning: {exc}")
            self.status_signal.emit(tr("🛬 Drone landing / stream shutdown requested"))

        if blocking:
            _land()
        else:
            threading.Thread(
                target=_land, daemon=True, name="AI-LAB-EMERGENCY-LAND"
            ).start()

    def _handle_battery_critical(self, pct: int):
        """Battery dropped to/below the critical threshold. Land safely,
        stop any running mission, and surface the post-flight report —
        the app stays open the whole time; nothing here exits/closes it."""
        self._on_status(f"🔋 {tr('BATTERY CRITICAL')} ({pct}%) — {tr('auto-landing and ending flight')}")
        self.emergency_stop(blocking=False)
        # Small delay lets the landing status message land in the terminal
        # before the modal report dialog grabs focus.
        QtCore.QTimer.singleShot(500, self._show_post_flight_report)

    def _handle_drone_disconnected(self):
        """Repeated failures to read telemetry are treated as a dropped
        link. Best-effort safe landing (may itself fail since the link is
        down — that's fine, emergency_stop() already swallows it), then
        keep the app running and open the report so nothing is lost."""
        self._on_status(tr("📡 DRONE CONNECTION LOST — attempting safe shutdown"))
        self.camera_label.setText(tr("● DRONE DISCONNECTED"))
        self.emergency_stop(blocking=False)
        self.drone.connected = False
        QtCore.QTimer.singleShot(500, self._show_post_flight_report)

    @QtCore.Slot(object, object)
    def _on_frame(self, frame, data):
        if frame is None:
            return

        # AI callback arrives independently from the camera callback.
        # Cache only the newest result; the live camera remains responsible
        # for continuous rendering.
        self.last_snapshot = data
        self._latest_detections = list(data.get("detections", []))
        models = data.get("models", [])
        self.model_label.setText(f"{tr('Models loaded')}: {len(models)}")
        self.model_list.setPlainText("\n".join(f"✅ {m['name']}" for m in models) or tr("❌ No .pt models loaded"))
        self.current_label.setText(f"{tr('Current Detection')}: {data['current_count']}")
        self.max_label.setText(f"{tr('MAXIMUM DETECTED')}: {data['maximum_count']}")
        self.targets_label.setText(f"💣 {tr('Unique Targets Found')}: {data.get('unique_target_count', 0)}")

        detections = data.get("detections", [])
        self._current_target_id = data.get("target_id")

        if detections:
            best = max(
                detections,
                key=lambda d: float(d.get("confidence", 0.0))
            )
            self.target_label.setText(
                f"{best.get('class_name', '--')}  •  "
                f"{best.get('model_name', '--')}"
            )
            self.target_conf_label.setText(
                f"{tr('Confidence')}: {float(best.get('confidence', 0.0)):.0%}"
            )

            stability = data.get("stability", {})
            if stability:
                pct = stability.get("percentage", 0.0)
                self.target_stability_label.setText(
                    f"{tr('Stability')}: {float(pct):.0f}%"
                )
                self.target_frames_label.setText(
                    f"{tr('Frames')}: {stability.get('detected_frames', '--')} / "
                    f"{stability.get('window', '--')}"
                )
            else:
                self.target_stability_label.setText(tr("Stability: --"))
                self.target_frames_label.setText(tr("Frames: --"))

            status = data.get("status", "CANDIDATE")
            status_text = STATUS_DISPLAY.get(status, status)
            # The "requires human review" note used to point at the
            # CONFIRM/REJECT buttons above, which are now hidden (see the
            # target_layout setup above) - dropped so this label doesn't
            # ask for an action there's no visible way to take.
            self.target_status_label.setText(f"{tr('STATUS')}: {status_text}")

            # Only STABLE targets are worth an operator decision; once
            # already CONFIRMED/REJECTED, don't re-offer the buttons.
            reviewable = status in ("STABLE", "CANDIDATE") and self._current_target_id is not None
            self.confirm_btn.setEnabled(reviewable)
            self.reject_btn.setEnabled(reviewable)
        else:
            self.target_label.setText(tr("NO TARGET"))
            self.target_conf_label.setText(tr("Confidence: --"))
            self.target_stability_label.setText(tr("Stability: --"))
            self.target_frames_label.setText(tr("Frames: --"))
            self.target_status_label.setText(f"{tr('STATUS')}: {STATUS_DISPLAY['SEARCHING']}")
            self.confirm_btn.setEnabled(False)
            self.reject_btn.setEnabled(False)

        s = data.get("survey", {})
        self.position_label.setText(f"📍 WP {s.get('waypoint', 0)}/{s.get('total_waypoints', 0)} | X={s.get('x_cm', 0)} Y={s.get('y_cm', 0)} cm | {s.get('heading', 'EAST')}")
        self.frames_label.setText(f"{tr('Frames analyzed')}: {data['total_frames']}")
        self.captures_label.setText(f"{tr('Images captured')}: {data['capture_count']}")
        self.camera_label.setText(tr("● CAMERA LIVE"))
        self.video_info_label.setText(tr("LIVE • AI DETECTION ACTIVE"))
        if not hasattr(self, "_last_battery_poll"):
            self._last_battery_poll = 0.0
        if not hasattr(self, "_battery_poll_fail_count"):
            self._battery_poll_fail_count = 0
        if not hasattr(self, "_battery_alert_shown"):
            self._battery_alert_shown = False
        now = time.monotonic()
        if now - self._last_battery_poll >= 5.0:
            self._last_battery_poll = now
            try:
                pct = int(self.drone.api.get_battery())
                self.battery_label.setText(self._battery_text(pct))
                self._battery_poll_fail_count = 0
                sync = getattr(self.drone, "sync_flying_from_drone", None)
                if sync is not None:   # the simulated drone has no real status
                    sync()

                critical_pct = getattr(config, "BATTERY_CRITICAL_PCT", 10)
                if pct <= critical_pct and not self._battery_alert_shown:
                    self._battery_alert_shown = True
                    self._handle_battery_critical(pct)
            except Exception:
                self.battery_label.setText("🔋 --%")
                # A fresh session (reconnect) has no telemetry yet: not a failure.
                if not getattr(self.drone, "reconnecting", False):
                    self._battery_poll_fail_count += 1
                # A single failed read can just be RF noise. Several IN A
                # ROW (default: 3, i.e. ~15s of silence at this 5s poll
                # interval) is treated as a genuinely dropped connection.
                disconnect_after = getattr(config, "BATTERY_POLL_FAIL_DISCONNECT_COUNT", 3)
                if self._battery_poll_fail_count >= disconnect_after and not self._battery_alert_shown:
                    self._battery_alert_shown = True
                    self._handle_drone_disconnected()
        if self.drone.camera_down_ok and not self.drone.autonomous_running:
            self.auto_btn.setEnabled(True)
        self._refresh_mission_buttons()

    def _report_gpupack_result(self):
        """After `--apply-gpupack` reopened the app: say how it went."""
        from ..ai import gpupack
        res = gpupack.last_result()
        if res is None:
            return
        if res.get("ok"):
            self._on_status(f"⚡ ติดตั้ง GPU Pack แล้ว ({res.get('files')} ไฟล์) — เลือก cuda ในหน้า Train ได้เลย")
        else:
            self._on_status(f"❌ ติดตั้ง GPU Pack ไม่สำเร็จ: {res.get('error')}")

    def _check_drone_alerts(self):
        drone = self.drone
        connected = bool(getattr(drone, "connected", False))
        battery = drone.battery_pct() if connected and hasattr(drone, "battery_pct") else None
        age = drone.telemetry_age_sec() if hasattr(drone, "telemetry_age_sec") else None
        flying = bool(getattr(drone, "flying", False))
        for alert in self._alert_monitor.check(connected, battery, age, flying):
            self._show_drone_alert(alert)
            if alert.kind == "link_lost" and flying:
                # Best effort: the command may not reach a drone that is out
                # of range, but it will if the link comes back.
                self.emergency_stop(blocking=False)
        tone = None
        if connected and battery is not None and not self._alert_monitor.link_lost:
            self.battery_label.setText(self._battery_text(battery))
            tone = ("danger" if battery <= drone_alerts.takeoff_min_pct() else
                    "warn" if battery <= drone_alerts.warn_pct() else "live")
        elif connected and self._alert_monitor.link_lost:
            self.battery_label.setText("🔋 --%")      # the last value is stale
            tone = "danger"
        if tone is not None:
            self.battery_label.setProperty("tone", tone)
            self.battery_label.style().unpolish(self.battery_label)
            self.battery_label.style().polish(self.battery_label)
        if connected and self._alert_monitor.link_lost:
            self._try_reconnect()

    def _try_reconnect(self):
        """While the link is lost, start a fresh drone session every few
        seconds in the background, so values come back by themselves when
        the drone does (it used to need an app restart)."""
        reconnect = getattr(self.drone, "reconnect", None)
        now = time.monotonic()
        if reconnect is None or getattr(self.drone, "reconnecting", False) or now - self._last_reconnect_try < 3.0:
            return
        self._last_reconnect_try = now
        threading.Thread(target=reconnect, daemon=True, name="AI-LAB-DRONE-RECONNECT").start()

    def _show_drone_alert(self, alert):
        icon = {"danger": "🚨", "warn": "⚠️", "info": "✅"}.get(alert.severity, "⚠️")
        self._on_status(f"{icon} {alert.title} — {alert.message.splitlines()[0]}")
        if alert.sound:
            drone_alerts.play_alarm()
        if alert.kind == "link_back":
            old = self._alert_boxes.pop("link_lost", None)
            if old is not None:
                old.close()
            return
        old = self._alert_boxes.pop(alert.kind, None)
        if old is not None:
            old.close()
        box = QtWidgets.QMessageBox(
            QtWidgets.QMessageBox.Critical if alert.severity == "danger" else QtWidgets.QMessageBox.Warning,
            alert.title, f"{icon} {alert.title}\n\n{alert.message}", QtWidgets.QMessageBox.Ok, self)
        box.setObjectName(f"droneAlert_{alert.kind}")
        box.setWindowModality(QtCore.Qt.NonModal)   # never blocks flying / Emergency Stop
        box.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        box.finished.connect(lambda *_a, k=alert.kind, b=box: self._alert_boxes.pop(k, None) if self._alert_boxes.get(k) is b else None)
        self._alert_boxes[alert.kind] = box
        box.show()

    def _battery_text(self, pct: int) -> str:
        """Battery + the drone's own flight state + temperature (REPORT_STATS)."""
        parts = [f"🔋 {pct}%"]
        temp = self.drone.temperature_c() if hasattr(self.drone, "temperature_c") else None
        if temp is not None:
            warn = temp >= getattr(config, "DRONE_TEMP_WARN_C", 60)
            parts.append(f"{'⚠️' if warn else '🌡️'} {temp}°C")
        state = self.drone.flight_state() if hasattr(self.drone, "flight_state") else None
        if state is not None:
            parts.append(state[1] if getattr(config, "LANGUAGE", "th") == "th" else state[2])
        return "  ".join(parts)

    def _confirm_current_target(self):
        if not self.detector or self._current_target_id is None:
            return
        self.detector.confirm_target(self._current_target_id)
        self.confirm_btn.setEnabled(False)
        self.reject_btn.setEnabled(False)

    def _reject_current_target(self):
        if not self.detector or self._current_target_id is None:
            return
        reason = self.reject_reason_combo.currentText()
        self.detector.reject_target(self._current_target_id, reason)
        self.confirm_btn.setEnabled(False)
        self.reject_btn.setEnabled(False)

    @QtCore.Slot(str)
    def _on_status(self, msg):
        self.terminal.appendPlainText(msg)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Keep the floating fullscreen controls pinned to the corner as
        # the window resizes (only matters while fullscreen is active).
        if getattr(self, "_is_video_fullscreen", False):
            self._reposition_fullscreen_overlay()

    def keyPressEvent(self, event):
        # Esc is the expected way out of fullscreen video - it must never
        # get swallowed by whatever widget currently has focus.
        if event.key() == QtCore.Qt.Key_Escape and getattr(self, "_is_video_fullscreen", False):
            self._exit_video_fullscreen()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        if self.grid_mission and self.grid_mission.is_alive():
            self.emergency_stop()
        for dlg in (self._flight_plan_dialog, self._keyboard_dialog, self._manual_flight_dialog):
            if dlg is not None and dlg.isVisible():
                dlg.close()
        self.stop_camera()
        if self._webcam_thread is not None:
            self._webcam_thread.stop()
            self._webcam_thread.join(timeout=2.0)
            self._webcam_thread = None
        if self._label_train_dialog is not None:
            # Best-effort: ask any running frame-extraction/training thread
            # to wind down. Training in particular can take a while and is
            # a daemon thread, so this does not block app exit on it.
            if self._label_train_dialog._extract_thread is not None:
                self._label_train_dialog._extract_thread.stop()
            if self._label_train_dialog._train_thread is not None:
                self._label_train_dialog._train_thread.stop()
        event.accept()


class _GridPreviewWidget(QtWidgets.QWidget):
    """Read-only visual preview of the planned survey route. No flight commands are sent."""

    def __init__(self, waypoints, parent=None):
        super().__init__(parent)
        self.waypoints = list(waypoints)
        self.setMinimumSize(500, 500)
        self.setStyleSheet("background:#071018;border:1px solid #263442;")
        # Live drone position, set from MainWindow._on_grid_progress_ui()
        # while an AutonomousGridMission is actually flying. None means
        # "no live flight" - the widget then renders exactly as the
        # original static pre-flight preview always did.
        self.current_wp_index = None
        self.current_x = None
        self.current_y = None
        self.current_heading = None

    def set_current_waypoint(self, index, x, y, heading):
        self.current_wp_index = index
        self.current_x = x
        self.current_y = y
        self.current_heading = heading
        self.update()

    def clear_current_waypoint(self):
        self.current_wp_index = None
        self.current_x = None
        self.current_y = None
        self.current_heading = None
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QtGui.QColor("#071018"))

        if not self.waypoints:
            painter.setPen(QtGui.QColor("#d8dee9"))
            painter.drawText(self.rect(), QtCore.Qt.AlignCenter, "No waypoints")
            return

        margin = 55
        area = self.rect().adjusted(margin, margin, -margin, -margin)
        max_x = max(1, config.FIELD_WIDTH_CM)
        max_y = max(1, config.FIELD_HEIGHT_CM)

        def pos(x, y):
            # x_cm increases EAST -> right on screen (unchanged).
            # y_cm increases SOUTH (see planner.py: dy > 0 -> "SOUTH"),
            # so it must move DOWN on screen to match the "N ↑" label
            # and normal map orientation (north-up, south-down).
            px = area.left() + (x / max_x) * area.width()
            py = area.top() + (y / max_y) * area.height()
            return QtCore.QPointF(px, py)

        # Field boundary.
        painter.setPen(QtGui.QPen(QtGui.QColor("#7f8c9a"), 2))
        painter.drawRect(area)

        # Grid lines.
        painter.setPen(QtGui.QPen(QtGui.QColor("#22313d"), 1))
        xs = sorted(set(w.x_cm for w in self.waypoints))
        ys = sorted(set(w.y_cm for w in self.waypoints))
        for x in xs:
            a, b = pos(x, 0), pos(x, max_y)
            painter.drawLine(a, b)
        for y in ys:
            a, b = pos(0, y), pos(max_x, y)
            painter.drawLine(a, b)

        # Route. While a mission is live (current_wp_index is set), the
        # already-flown portion is drawn brighter/solid and the remaining
        # portion dimmed, so it reads at a glance as a progress bar over
        # the field, not just a static plan. With no live flight this is
        # unchanged from the original single-color planned route.
        route_pen = QtGui.QPen(QtGui.QColor("#55d6be"), 3)
        flown_pen = QtGui.QPen(QtGui.QColor("#e7fff9"), 4)
        upcoming_pen = QtGui.QPen(QtGui.QColor("#22383a"), 3)
        for a, b in zip(self.waypoints, self.waypoints[1:]):
            if self.current_wp_index is None:
                painter.setPen(route_pen)
            elif b.index <= self.current_wp_index:
                painter.setPen(flown_pen)
            else:
                painter.setPen(upcoming_pen)
            painter.drawLine(pos(a.x_cm, a.y_cm), pos(b.x_cm, b.y_cm))

        # Waypoint dots + index. The current waypoint (if a mission is
        # live) is drawn larger and in the "current position" color so it
        # stands out from the rest of the plan.
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        for wp in self.waypoints:
            pt = pos(wp.x_cm, wp.y_cm)
            is_current = self.current_wp_index is not None and wp.index == self.current_wp_index
            if is_current:
                painter.setBrush(QtGui.QColor("#ff5f6d"))
                painter.setPen(QtGui.QPen(QtGui.QColor("#fff0ef"), 2))
                painter.drawEllipse(pt, 7, 7)
            else:
                painter.setBrush(QtGui.QColor("#55d6be"))
                painter.setPen(QtGui.QPen(QtGui.QColor("#dffcf7"), 1))
                painter.drawEllipse(pt, 5, 5)
            painter.setPen(QtGui.QColor("#d8dee9"))
            painter.drawText(pt + QtCore.QPointF(7, -7), str(wp.index))

        # Live drone marker: a pulsing-style ring around the current
        # waypoint dot plus a "🚁 WP N" label, using the drone's actual
        # reported X/Y (not just the matching waypoint's coordinates) so
        # it stays correct even if a future flight mode reports positions
        # that don't land exactly on a planned dot.
        if self.current_wp_index is not None and self.current_x is not None:
            cur_pt = pos(self.current_x, self.current_y)
            painter.setBrush(QtGui.QColor(0, 0, 0, 0))
            painter.setPen(QtGui.QPen(QtGui.QColor("#ff5f6d"), 3))
            painter.drawEllipse(cur_pt, 14, 14)

            drone_font = painter.font()
            drone_font.setPointSize(10)
            drone_font.setBold(True)
            painter.setFont(drone_font)
            painter.setPen(QtGui.QColor("#ff9199"))
            label_text = (
                f"🚁 WP {self.current_wp_index}"
                + (f" • {self.current_heading}" if self.current_heading else "")
            )
            # Flip the label to the left of the marker when there isn't
            # room to the right (e.g. any waypoint in the rightmost
            # column) - drawText() doesn't clip/wrap on its own, it just
            # runs text off the edge of the widget, which was confirmed
            # visually (label reading "WP" with everything after it cut
            # off) before this fix.
            label_w = QtGui.QFontMetrics(drone_font).horizontalAdvance(label_text)
            if cur_pt.x() + 17 + label_w > self.width() - 6:
                text_pt = cur_pt + QtCore.QPointF(-17 - label_w, 5)
            else:
                text_pt = cur_pt + QtCore.QPointF(17, 5)
            painter.drawText(text_pt, label_text)
            painter.setFont(font)

        # Start marker and heading arrows at turns.
        start = pos(self.waypoints[0].x_cm, self.waypoints[0].y_cm)
        painter.setBrush(QtGui.QColor("#ffd166"))
        painter.setPen(QtGui.QPen(QtGui.QColor("#ffd166"), 2))
        painter.drawEllipse(start, 8, 8)
        painter.drawText(start + QtCore.QPointF(12, 18), "START")

        font = painter.font()
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QtGui.QColor("#d8dee9"))
        painter.drawText(12, 22, "N ↑")
        painter.drawText(12, 40, "0,0 cm")  # origin is now top-left, matching the flipped Y axis above
        painter.drawText(self.width() - 115, self.height() - 15, f"{config.FIELD_WIDTH_CM},{config.FIELD_HEIGHT_CM} cm")
        legend = "GREEN = PLANNED ROUTE  |  YELLOW = START"
        if self.current_wp_index is not None:
            legend += "  |  RED = CURRENT POSITION"
        painter.drawText(12, self.height() - 40, legend)


def run():
    app = QtWidgets.QApplication(sys.argv)
    from . import pixel_theme

    pixel_theme.install(app, config.PROJECT_ROOT)

    # Single-instance guard: without this, double-clicking the desktop
    # icon again (or launching from both the Start Menu and desktop, or
    # just double-clicking impatiently before the first window has
    # finished its ~75s cold start - see the ultralytics/torch import
    # comment below) spins up a second, fully independent copy of the
    # whole app: two MainWindows, each free to open its own connection to
    # the same physical drone and step on the other's commands. A
    # QSharedMemory segment under a fixed key is the standard
    # cross-process "is another copy of me already running" check -
    # attaching to a key that already exists means some other process
    # created it first. The stray attach+detach before create() clears a
    # segment orphaned by a previous instance that crashed without
    # cleaning up (a known QSharedMemory quirk on Unix; harmless as a
    # no-op on Windows, where the OS already reclaims it on process
    # death) so a real crash doesn't permanently lock later launches out.
    single_instance_guard = QtCore.QSharedMemory("AI_Detection_Lab-SingleInstanceGuard")
    if single_instance_guard.attach():
        single_instance_guard.detach()
    if not single_instance_guard.create(1):
        QtWidgets.QMessageBox.warning(
            None, tr("Already Running"),
            tr("AI Detection Lab is already open - check your taskbar for the existing window."),
        )
        return 0

    # A Windows install set to a Thai display language can have "native
    # digits" turned on in Regional Settings, which makes Qt's default
    # locale format numbers with Thai numerals (๐-๙). Pin the app-wide
    # default locale so every numeric widget (spin boxes, etc.) always
    # shows plain 0-9, regardless of the OS's regional/number settings.
    QtCore.QLocale.setDefault(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))

    # App-wide icon (title bar, taskbar, Alt-Tab) - set once here rather than
    # per-window so every dialog inherits it automatically. Guarded so a
    # missing/corrupt app_icon.png just means the Qt default icon, never a
    # startup crash (same convention as the logo.png footer below).
    icon_path = config.PROJECT_ROOT / "app_icon.png"
    if icon_path.exists():
        app.setWindowIcon(QtGui.QIcon(str(icon_path)))

    def _log_uncaught_exceptions(exc_type, exc_value, exc_tb):
        # Belt-and-suspenders: nothing in this app (a low-battery SDK call,
        # a dropped-connection race, etc.) should be able to silently kill
        # the process mid-flight. Log it to the console instead of letting
        # Python's default excepthook tear down the interpreter.
        import traceback
        traceback.print_exception(exc_type, exc_value, exc_tb)

    sys.excepthook = _log_uncaught_exceptions

    from .mission_control import MissionControlWindow

    window = MissionControlWindow()
    window.show()
    return app.exec()
