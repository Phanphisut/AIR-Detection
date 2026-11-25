"""v6.0.0 dashboard. Reuses the proven v5.0 controller/signal wiring.

Only presentation is replaced: original widgets are moved into the new shell
so their handlers, enabled states, worker signals and flight guards survive.
"""
from __future__ import annotations

import time
from PySide6 import QtCore, QtGui, QtWidgets

from .. import config
from .main_window import MainWindow
from .mission_theme import STYLE
from .mission_widgets import ElidedLabel, VideoSurface, icon

VERSION = "9.0.0"


def text(en, th):
    return th if config.LANGUAGE == "th" else en


def label(value, name="muted"):
    widget = QtWidgets.QLabel(value)
    widget.setObjectName(name)
    return widget


def column(parent, margin=16, spacing=10):
    layout = QtWidgets.QVBoxLayout(parent)
    layout.setContentsMargins(margin, margin, margin, margin)
    layout.setSpacing(spacing)
    return layout


def panel(name="panel"):
    widget = QtWidgets.QFrame()
    widget.setObjectName(name)
    return widget


class MissionControlWindow(MainWindow):
    def _build_ui(self):
        super()._build_ui()
        previous = self.takeCentralWidget()
        self.setWindowTitle(f"AI Detection Lab — Mission Control v{VERSION}")
        self.setMinimumSize(1120, 760)
        self.resize(1440, 920)
        self._webcam_frames = 0
        self._webcam_peak = 0
        self._webcam_last_frame = 0.0
        self._display_source = None

        root = QtWidgets.QWidget()
        root.setObjectName("workspace")
        self.setCentralWidget(root)
        shell = QtWidgets.QHBoxLayout(root)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setSpacing(0)

        sidebar = self._build_sidebar()
        self._left_panel = sidebar
        shell.addWidget(sidebar)

        body = QtWidgets.QWidget()
        body.setObjectName("body")
        body_layout = column(body, 0, 0)
        shell.addWidget(body, 1)
        self._header_frame = self._build_header()
        body_layout.addWidget(self._header_frame)

        content = QtWidgets.QWidget()
        self._content_layout = column(content, 22, 16)
        body_layout.addWidget(content, 1)
        self._page_heading = self._build_heading()
        self._content_layout.addWidget(self._page_heading)
        self._metrics = self._build_metrics()
        self._content_layout.addWidget(self._metrics)

        work = QtWidgets.QHBoxLayout()
        work.setSpacing(16)
        self._content_layout.addLayout(work, 1)
        center = QtWidgets.QWidget()
        center_layout = column(center, 0, 14)
        self._center_layout = center_layout
        self._inspector_dialog = None
        work.addWidget(center, 1)
        center_layout.addWidget(self._build_video(), 1)
        self._inspector = self._build_inspector()
        center_layout.addWidget(self._inspector)
        self._right_panel = self._build_controls()
        work.addWidget(self._right_panel)

        self._log_card = self._build_log()
        self._content_layout.addWidget(self._log_card)

        # Preserve hidden operator-review controls without restoring the old
        # automatic workflow's intentionally removed confirm/reject step.
        for widget in (self.confirm_btn, self.reject_btn, self.reject_reason_combo, self.grid_label):
            widget.setParent(root)
            widget.hide()
        self.fullscreen_overlay.setParent(root)
        self.fullscreen_overlay.hide()
        self.fullscreen_overlay.setStyleSheet("background:#172535; border:1px solid #476079; border-radius:10px;")
        self._style_button(self.fullscreen_emergency_btn, text("Stop / Land", "หยุด / ลงจอด"), "shield", "dangerButton")
        self._style_button(self.fullscreen_exit_btn, text("Exit fullscreen · Esc", "ออกจากเต็มจอ · Esc"), "expand")
        previous.deleteLater()

        # Re-polish after reparenting legacy widgets and assigning their new
        # names; otherwise Qt keeps cached colors/metrics from the old shell.
        self.setStyleSheet(STYLE)

        self._mission_btn_timer.timeout.connect(self._sync_dashboard)
        self._scan_models()
        self._sync_dashboard()
        self._on_status(text("Workspace ready. Test a webcam or prepare a drone mission.",
                             "พร้อมใช้งาน · เริ่มทดสอบเว็บแคม หรือเตรียมภารกิจโดรนได้เลย"))
        self._escape_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Escape"), self)
        self._escape_shortcut.activated.connect(self._exit_video_fullscreen)

    def _style_button(self, button, title, symbol=None, kind=""):
        button.setText(title)
        button.setStyleSheet("")
        button.setObjectName(kind)
        button.setMinimumWidth(0)
        button.setIcon(icon(symbol, "#153e34" if kind == "primaryButton" else "#a1b9cc") if symbol else QtGui.QIcon())
        button.setIconSize(QtCore.QSize(17, 17))
        button.setCursor(QtCore.Qt.PointingHandCursor)
        button.setAccessibleName(title)
        return button

    def _build_sidebar(self):
        side = panel("sidebar")
        side.setFixedWidth(205)
        layout = column(side, 15, 6)
        layout.addSpacing(12)
        mark = QtWidgets.QHBoxLayout()
        logo = label("")
        logo.setPixmap(icon("drone", "#77e4d0").pixmap(34, 34))
        mark.addWidget(logo)
        names = QtWidgets.QVBoxLayout()
        names.setSpacing(0)
        names.addWidget(label("DETECTION", "brand"))
        names.addWidget(label("LAB / MISSION CONTROL", "wordmark"))
        mark.addLayout(names)
        layout.addLayout(mark)
        layout.addSpacing(30)
        layout.addWidget(label("WORKSPACE", "eyebrow"))
        layout.addSpacing(4)

        home = QtWidgets.QPushButton()
        self._style_button(home, text("Mission control", "ศูนย์ควบคุม"), "dashboard", "nav")
        home.setCheckable(True)
        home.setChecked(True)
        home.clicked.connect(lambda: home.setChecked(True))
        layout.addWidget(home)
        for button, title, symbol in (
            (self.grid_btn, text("Survey route", "เส้นทางสำรวจ"), "route"),
            (self.report_btn, text("Mission report", "รายงานภารกิจ"), "report"),
        ):
            layout.addWidget(self._style_button(button, title, symbol, "nav"))
        layout.addSpacing(24)
        layout.addWidget(label("AI STUDIO", "eyebrow"))
        layout.addSpacing(4)
        for button, title, symbol in (
            (self.label_train_btn, text("Label & train", "ติดป้ายและฝึก AI"), "train"),
            (self.choose_models_btn, text("Model library", "คลังโมเดล"), "model"),
            (self.code_editor_btn, text("Code & settings", "โค้ดและตั้งค่า"), "code"),
        ):
            layout.addWidget(self._style_button(button, title, symbol, "nav"))
        layout.addStretch()

        tip = panel()
        tip_layout = column(tip, 12, 6)
        tip_title = label(text("Try your webcam", "เริ่มได้โดยไม่ใช้โดรน"), "sectionTitle")
        tip_title.setWordWrap(True)
        tip_layout.addWidget(tip_title)
        hint = label(text("Test your AI model before the first flight.", "ลองตรวจจับด้วยเว็บแคม\nก่อนเริ่มภารกิจจริง"))
        hint.setWordWrap(True)
        tip_layout.addWidget(hint)
        layout.addWidget(tip)
        layout.addSpacing(12)
        layout.addWidget(self._style_button(self.dev_mode_btn, text("Developer mode", "โหมดนักพัฒนา"), "code", "nav"))
        layout.addWidget(self._style_button(self.manual_flight_btn, text("Manual flight", "บินด้วยมือ"), "keyboard", "nav"))
        layout.addSpacing(8)
        layout.addWidget(label(f"VERSION {VERSION}  /  UX EDITION", "eyebrow"))
        if self._footer_widget is not None:
            # Preserve partner attribution, sized to the quieter sidebar.
            for item in self._footer_widget.findChildren(QtWidgets.QLabel):
                pix = item.pixmap()
                if pix and not pix.isNull():
                    item.setPixmap(pix.scaledToWidth(168, QtCore.Qt.SmoothTransformation))
            layout.addWidget(self._footer_widget)
        return side

    def _build_header(self):
        header = panel("topbar")
        header.setFixedHeight(60)
        row = QtWidgets.QHBoxLayout(header)
        row.setContentsMargins(22, 0, 22, 0)
        row.addWidget(label("WORKSPACE", "eyebrow"))
        row.addWidget(label("  /  "))
        row.addWidget(label(text("Autonomous survey", "สำรวจอัตโนมัติ")))
        row.addStretch()
        self.connection_badge = label("", "badge")
        self.connection_badge.setFixedHeight(30)
        row.addWidget(self.connection_badge)
        self.battery_label.setObjectName("badge")
        self.battery_label.setFixedHeight(30)
        self.battery_label.setText(text("Battery  —", "แบตเตอรี่  —"))
        self.battery_label.setToolTip(text("Read from the drone every second while connected. Warns at low battery.", "อ่านค่าจากโดรนทุกวินาทีขณะเชื่อมต่อ · เตือนเมื่อแบตต่ำ"))
        row.addWidget(self.battery_label)
        return header

    def _build_heading(self):
        widget = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)
        title = QtWidgets.QVBoxLayout()
        title.setSpacing(3)
        title.addWidget(label(text("Mission control", "ศูนย์ควบคุมภารกิจ"), "pageTitle"))
        title.addWidget(label(text("Your flight, your AI, one clear view.", "เตรียมบิน ตรวจจับ และติดตามภารกิจในหน้าจอเดียว")))
        row.addLayout(title)
        row.addStretch()
        row.addWidget(self._style_button(self.sim_mode_btn, text("Simulation · Off", "โหมดจำลอง · ปิด"), "drone", "quiet"))
        return widget

    def _build_metrics(self):
        widget = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(12)
        self.capture_value = label("0", "valueBig")
        entries = (
            (self.current_label, text("Current detections", "ตรวจจับขณะนี้"), text("Objects in this frame", "วัตถุในเฟรมปัจจุบัน"), False),
            (self.max_label, text("Peak detections", "ตรวจจับสูงสุด"), text("Highest count this session", "จำนวนสูงสุดในรอบนี้"), False),
            (self.targets_label, text("Unique targets", "เป้าหมายไม่ซ้ำ"), text("Survey-position deduplication", "แยกซ้ำตามตำแหน่งสำรวจ"), True),
            (self.capture_value, text("Saved captures", "ภาพที่บันทึก"), text("Evidence from this mission", "หลักฐานจากภารกิจนี้"), False),
        )
        for value, title, hint, accent in entries:
            card = panel("accentMetric" if accent else "metric")
            box = column(card, 14, 2)
            value.setObjectName("valueAccent" if accent else "valueBig")
            value.setText("0")
            box.addWidget(label(title))
            box.addWidget(value)
            note = ElidedLabel(hint)
            note.setObjectName("muted")
            note.setToolTip(hint)
            box.addWidget(note)
            row.addWidget(card, 1)
        return widget

    def _build_video(self):
        card = panel("videoCard")
        layout = column(card, 12, 10)
        header = QtWidgets.QHBoxLayout()
        header.addWidget(label(text("Live view", "ภาพสด"), "sectionTitle"))
        header.addStretch()
        self.inspector_btn = QtWidgets.QPushButton(text("AI details", "ข้อมูล AI"))
        self.inspector_btn.setObjectName("quiet")
        self.inspector_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.inspector_btn.clicked.connect(self._show_inspector)
        header.addWidget(self.inspector_btn)
        self.inspector_btn.hide()
        self.camera_label.setObjectName("badge")
        header.addWidget(self.camera_label)
        self._style_button(self.video_fullscreen_btn, "", "expand", "quiet")
        self.video_fullscreen_btn.setAccessibleName(text("Expand camera", "ขยายภาพกล้อง"))
        self.video_fullscreen_btn.setFixedWidth(38)
        header.addWidget(self.video_fullscreen_btn)
        layout.addLayout(header)

        self.video = VideoSurface(text("Ready when you are", "พร้อมเมื่อคุณพร้อม"),
                                  text("Start the drone camera or test a model with your webcam.",
                                       "เริ่มกล้องโดรน หรือกดทดสอบเว็บแคมด้านล่าง"))
        layout.addWidget(self.video, 1)
        video_meta = QtWidgets.QHBoxLayout()
        self.video_info_label.setObjectName("muted")
        video_meta.addWidget(self.video_info_label, 1)
        self.frames_label.setObjectName("muted")
        self.frames_label.setWordWrap(False)
        video_meta.addWidget(self.frames_label)
        layout.addLayout(video_meta)

        self._camera_actions = QtWidgets.QWidget()
        actions = QtWidgets.QHBoxLayout(self._camera_actions)
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setSpacing(8)
        self._style_button(self.webcam_toggle_btn, text("Test webcam", "ทดสอบเว็บแคม"), "camera")
        self.webcam_toggle_btn.setToolTip(text("Test the laptop camera and AI; no drone connection needed.", "ทดสอบกล้องและ AI บนคอมพิวเตอร์ โดยไม่ต้องเชื่อมต่อโดรน"))
        actions.addWidget(self.webcam_toggle_btn)
        self._style_button(self.record_photos_btn, text("Capture for training", "บันทึกภาพฝึก AI"), "photo", "quiet")
        actions.addWidget(self.record_photos_btn)
        actions.addStretch()
        # The webcam's detailed state remains visible even before a first frame.
        self.webcam_status_label.setObjectName("muted")
        self.webcam_status_label.setWordWrap(False)
        layout.addWidget(self._camera_actions)
        self.position_label.setObjectName("muted")
        self.position_label.setWordWrap(True)
        self.position_label.setText(text("Survey position  —  No mission started", "ตำแหน่งสำรวจ  —  ยังไม่เริ่มภารกิจ"))
        layout.addWidget(self.position_label)
        return card

    def _build_inspector(self):
        widget = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(12)
        target = panel()
        box = column(target, 14, 4)
        box.addWidget(label(text("Target insights", "รายละเอียดเป้าหมาย"), "sectionTitle"))
        self.target_label.setObjectName("targetName")
        self.target_label.setText(text("No target yet", "ยังไม่พบเป้าหมาย"))
        self.target_label.setWordWrap(True)
        box.addWidget(self.target_label)
        stats = QtWidgets.QHBoxLayout()
        for widget_label in (self.target_conf_label, self.target_stability_label):
            widget_label.setObjectName("muted")
            stats.addWidget(widget_label)
        box.addLayout(stats)
        for widget_label in (self.target_status_label, self.target_frames_label):
            widget_label.setObjectName("muted")
            widget_label.setWordWrap(True)
            box.addWidget(widget_label)
        box.addStretch()
        row.addWidget(target, 1)

        models = panel()
        box = column(models, 14, 4)
        heading = QtWidgets.QHBoxLayout()
        heading.addWidget(label(text("AI models", "โมเดล AI"), "sectionTitle"))
        heading.addStretch()
        manage = QtWidgets.QPushButton(text("Manage", "จัดการ"))
        manage.setObjectName("quiet")
        manage.setCursor(QtCore.Qt.PointingHandCursor)
        manage.clicked.connect(self._open_model_select)
        heading.addWidget(manage)
        box.addLayout(heading)
        self.model_label.setObjectName("muted")
        self.model_label.setWordWrap(True)
        box.addWidget(self.model_label)
        self.model_list.setMinimumHeight(40)
        self.model_list.setMaximumHeight(55)
        box.addWidget(self.model_list)
        self.model_note = label(text("Files on disk; loaded when AI starts.", "ไฟล์ในเครื่อง · โหลดเมื่อเริ่มใช้งาน AI"))
        self.model_note.setWordWrap(True)
        box.addWidget(self.model_note)
        box.addStretch()
        row.addWidget(models, 1)
        return widget

    def _build_controls(self):
        card = panel()
        card.setFixedWidth(284)
        layout = column(card, 14, 10)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(label(text("Flight checklist", "เตรียมภารกิจ"), "sectionTitle"))
        row.addStretch()
        self.checklist_count = label("0 / 3", "wordmark")
        row.addWidget(self.checklist_count)
        layout.addLayout(row)
        self.mission_hint = label("")
        self.mission_hint.setWordWrap(True)
        layout.addWidget(self.mission_hint)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        inside = QtWidgets.QWidget()
        controls = column(inside, 0, 10)
        scroll.setWidget(inside)
        layout.addWidget(scroll, 1)
        self._steps = []
        for number, title, hint, button in (
            (1, text("Connect the camera", "เชื่อมต่อกล้อง"), text("Join the drone Wi-Fi first.", "เชื่อมต่อ Wi-Fi ของโดรนก่อน"), self.start_btn),
            (2, text("Point the camera down", "ตั้งกล้องมองลง"), text("Downward view at 90°.", "ปรับมุมกล้องลง 90°"), self.preflight_btn),
            (3, text("Review the survey route", "ตรวจเส้นทางสำรวจ"), f"{config.FIELD_WIDTH_CM} × {config.FIELD_HEIGHT_CM} cm  ·  {config.GRID_SPACING_CM} cm / grid", None),
        ):
            step = QtWidgets.QHBoxLayout()
            step.setSpacing(9)
            number_label = label(str(number), "step")
            number_label.setFixedSize(26, 26)
            number_label.setAlignment(QtCore.Qt.AlignCenter)
            self._steps.append(number_label)
            step.addWidget(number_label)
            step.addWidget(label(title, "sectionTitle"))
            step.addStretch()
            controls.addLayout(step)
            help_label = label(hint)
            help_label.setWordWrap(True)
            controls.addWidget(help_label)
            if button is self.start_btn:
                actions = QtWidgets.QHBoxLayout()
                actions.addWidget(self._style_button(button, text("Start camera", "เริ่มกล้อง AI"), "camera"), 1)
                actions.addWidget(self._style_button(self.stop_btn, text("Stop", "หยุด"), "stop", "quiet"))
                controls.addLayout(actions)
            elif button is not None:
                controls.addWidget(self._style_button(button, text("Camera down · 90°", "ปรับกล้องลง 90°"), "camera"))
            else:
                preview = QtWidgets.QPushButton()
                self._style_button(preview, text("Preview route", "ดูเส้นทางบิน"), "route")
                preview.clicked.connect(self.preview_grid)
                controls.addWidget(preview)
            controls.addSpacing(4)
        self._route_reviewed = False
        controls.addWidget(self._style_button(self.auto_btn, text("Start survey mission", "เริ่มภารกิจสำรวจ"), "play", "primaryButton"))
        self.auto_btn.setMinimumHeight(43)
        self.mission_progress = QtWidgets.QProgressBar()
        self.mission_progress.setRange(0, 100)
        self.mission_progress.setValue(0)
        self.mission_progress.setTextVisible(False)
        controls.addWidget(self.mission_progress)
        self.progress_label = label(text("No active mission", "ยังไม่มีภารกิจที่กำลังบิน"))
        self.progress_label.setWordWrap(True)
        controls.addWidget(self.progress_label)

        more = QtWidgets.QPushButton(text("Other flight modes  +", "รูปแบบการบินอื่น  +"))
        more.setObjectName("quiet")
        more.setCheckable(True)
        more.setCursor(QtCore.Qt.PointingHandCursor)
        controls.addWidget(more)
        self._other_modes = QtWidgets.QWidget()
        modes = column(self._other_modes, 0, 6)
        modes.addWidget(self._style_button(self.flight_plan_btn, text("Python flight plan", "แผนการบิน Python"), "code"))
        modes.addWidget(self._style_button(self.keyboard_btn, text("Keyboard control", "บินด้วยคีย์บอร์ด"), "keyboard"))
        controls.addWidget(self._other_modes)
        self._other_modes.hide()
        more.toggled.connect(self._other_modes.setVisible)
        more.toggled.connect(lambda checked: more.setText(text("Other flight modes", "รูปแบบการบินอื่น") + ("  −" if checked else "  +")))
        controls.addStretch()

        # Always outside the scrolling area, including at minimum window size.
        self._style_button(self.emergency_btn, text("Emergency stop / Land", "หยุดฉุกเฉิน / ลงจอด"), "shield", "dangerButton")
        self.emergency_btn.setMinimumHeight(39)
        layout.addWidget(self.emergency_btn)
        layout.addWidget(label(text("Stops the mission and lands the drone.", "หยุดภารกิจและสั่งโดรนลงจอด")))
        return card

    def _build_log(self):
        card = panel("logCard")
        layout = column(card, 10, 4)
        row = QtWidgets.QHBoxLayout()
        self.log_toggle = QtWidgets.QPushButton(text("Activity log  +", "บันทึกระบบ  +"))
        self.log_toggle.setObjectName("logToggle")
        self.log_toggle.setCheckable(True)
        self.log_toggle.setCursor(QtCore.Qt.PointingHandCursor)
        self.log_toggle.toggled.connect(self._toggle_log)
        row.addWidget(self.log_toggle)
        self.latest_activity = ElidedLabel()
        self.latest_activity.setObjectName("muted")
        row.addWidget(self.latest_activity, 1)
        layout.addLayout(row)
        self.terminal.setMaximumBlockCount(2000)
        self.terminal.setFixedHeight(110)
        layout.addWidget(self.terminal)
        self.terminal.hide()
        self.captures_label.setObjectName("muted")
        # This label is updated by the base controller; render its value in
        # the metric card while keeping the controller's widget alive.
        self.captures_label.setParent(card)
        self.captures_label.hide()
        self.webcam_status_label.setParent(card)
        self.webcam_status_label.hide()
        return card

    def _toggle_log(self, opened):
        self.terminal.setVisible(opened)
        self.log_toggle.setText(text("Activity log", "บันทึกระบบ") + ("  −" if opened else "  +"))
        self._update_compact_layout()

    def _compact_layout(self):
        return self.height() < 850 or self.width() < 1240 or (self.terminal.isVisible() and self.height() < 1020)

    def _update_compact_layout(self):
        if not hasattr(self, "_inspector") or getattr(self, "_is_video_fullscreen", False):
            return
        compact = self._compact_layout()
        self.inspector_btn.setVisible(compact)
        if self._inspector_dialog is None:
            self._inspector.setVisible(not compact)

    def _show_inspector(self):
        if self._inspector_dialog is not None:
            self._inspector_dialog.raise_()
            self._inspector_dialog.activateWindow()
            return
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(text("Target insights & AI models", "ข้อมูลเป้าหมายและโมเดล AI"))
        dialog.resize(700, 260)
        layout = column(dialog, 16, 10)
        layout.addWidget(self._inspector)
        self._inspector.show()
        self._inspector_dialog = dialog
        dialog.finished.connect(self._restore_inspector)
        dialog.show()

    def _restore_inspector(self, *_):
        dialog = self._inspector_dialog
        if dialog is None:
            return
        self._center_layout.addWidget(self._inspector)
        self._inspector_dialog = None
        self._update_compact_layout()
        dialog.deleteLater()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_compact_layout()

    def _on_status(self, msg):
        super()._on_status(msg)
        if hasattr(self, "latest_activity"):
            self.latest_activity.setText(msg)

    def _scan_models(self):
        models = sorted(config.MODEL_DIR.glob("*.pt")) if config.MODEL_DIR.exists() else []
        self.model_label.setText(text(f"{len(models)} available on disk", f"มี {len(models)} โมเดลในเครื่อง"))
        self.model_list.setPlainText("\n".join(p.name for p in models) or text("No models yet. Press Train AI at the top to begin.", "ยังไม่มีโมเดล · เริ่มได้ที่ปุ่ม สอน AI ด้านบน"))
        self.model_list.setToolTip(str(config.MODEL_DIR))

    @staticmethod
    def _property(widget, name, value):
        if widget.property(name) != value:
            widget.setProperty(name, value)
            widget.style().unpolish(widget)
            widget.style().polish(widget)
            widget.update()

    def _sync_dashboard(self):
        connected = self.drone.connected
        simulation = self.sim_mode_btn.isChecked()
        self.connection_badge.setText(text("● Simulation", "● โหมดจำลอง") if simulation else
                                      text("● Drone connected", "● เชื่อมต่อโดรนแล้ว") if connected else
                                      text("○ Drone offline", "○ ยังไม่เชื่อมต่อโดรน"))
        self._property(self.connection_badge, "tone", "warn" if simulation else "live" if connected else "idle")
        self.sim_mode_btn.setText(text("Simulation · On", "โหมดจำลอง · เปิด") if simulation else text("Simulation · Off", "โหมดจำลอง · ปิด"))
        self.sim_mode_btn.setEnabled(not connected and self.camera is None)

        webcam = self._webcam_thread is not None
        camera_live = self._camera_alive() and self.video_age_sec() < 3
        webcam_live = webcam and time.monotonic() - self._webcam_last_frame < 3
        self.camera_label.setText(text("● Webcam live", "● เว็บแคมสด") if webcam_live else
                                  text("● Camera live", "● กล้องสด") if camera_live else
                                  text("◌ Waiting for video", "◌ รอภาพกล้อง") if (webcam or self.camera) else
                                  text("○ Offline", "○ ยังไม่มีกล้อง"))
        self._property(self.camera_label, "tone", "live" if camera_live or webcam_live else "warn" if webcam or self.camera else "idle")
        if webcam:
            self.video_info_label.setText(self.webcam_status_label.text())
        elif camera_live:
            self.video_info_label.setText(text("DRONE CAMERA  /  AI OVERLAY", "กล้องโดรน  /  ภาพซ้อน AI"))
        elif self.camera:
            self.video_info_label.setText(text("Waiting for a fresh camera frame…", "กำลังรอภาพใหม่จากกล้อง…"))
        else:
            self.video_info_label.setText(text("CAMERA PREVIEW  /  NO SIGNAL", "ภาพจากกล้อง  /  ยังไม่มีสัญญาณ"))
        if not connected:
            self.battery_label.setText(text("Battery  —", "แบตเตอรี่  —"))

        completed = [camera_live, bool(self.drone.camera_down_ok), self._route_reviewed]
        for i, (step, done) in enumerate(zip(self._steps, completed), 1):
            step.setText("✓" if done else str(i))
            self._property(step, "complete", done)
        self.checklist_count.setText(f"{sum(completed)} / 3")
        grid_running = bool(self.grid_mission and self.grid_mission.is_alive())
        ready = bool(camera_live and self.drone.connected and self.drone.camera_down_ok and
                     not grid_running and self._active_mission() is None and
                     not self.drone.emergency_stop_requested and
                     config.AUTONOMOUS_FLIGHT_ENABLED and config.TAKEOFF_ENABLED)
        self.auto_btn.setEnabled(ready)
        self.auto_btn.setToolTip(text("Start camera, point it down, then review the route before flying.", "เริ่มกล้อง ตั้งมุมกล้องลง และตรวจเส้นทางก่อนบิน"))
        self.report_btn.setEnabled(self.detector is not None)
        self.report_btn.setToolTip(text("Available after an AI camera session has started.", "เปิดได้หลังเริ่มกล้อง AI และมีข้อมูลภารกิจ"))
        self.preflight_btn.setEnabled(not grid_running and not webcam)
        self.mission_hint.setText(text("Survey in progress", "กำลังบินสำรวจ") if grid_running else
                                  text("Camera ready. Review the route before takeoff.", "กล้องพร้อม · ตรวจเส้นทางก่อนขึ้นบิน") if ready else
                                  text("Complete the steps before takeoff.", "เตรียมความพร้อมตามขั้นตอนก่อนขึ้นบิน"))

    def preview_grid(self):
        super().preview_grid()
        self._route_reviewed = self._grid_preview_dialog is not None
        self._sync_dashboard()

    def _on_frame(self, frame, data):
        super()._on_frame(frame, data)
        if frame is None:
            return
        self._display_source = "drone"
        self.current_label.setText(str(data.get("current_count", 0)))
        self.max_label.setText(str(data.get("maximum_count", 0)))
        self.targets_label.setText(str(data.get("unique_target_count", 0)))
        self.capture_value.setText(str(data.get("capture_count", 0)))
        self.model_note.setText(text("Models loaded by the current AI session.", "โมเดลที่โหลดในรอบการตรวจจับนี้"))
        self._sync_dashboard()

    def _display_webcam_frame(self, frame, detections):
        super()._display_webcam_frame(frame, detections)
        if frame is None:
            return
        self._display_source = "webcam"
        self._webcam_last_frame = time.monotonic()
        self._webcam_frames += 1
        self._webcam_peak = max(self._webcam_peak, len(detections))
        self.current_label.setText(str(len(detections)))
        self.max_label.setText(str(self._webcam_peak))
        self.targets_label.setText("—")
        self.capture_value.setText("—")
        self.frames_label.setText(text(f"Frames: {self._webcam_frames:,}", f"เฟรม: {self._webcam_frames:,}"))
        self.position_label.setText(text("Webcam test · Survey tracking and evidence capture are inactive.", "ทดสอบเว็บแคม · ไม่มีการติดตามตำแหน่งหรือบันทึกหลักฐานภารกิจ"))
        best = max(detections, key=lambda d: float(d.get("confidence", 0)), default={})
        self.target_label.setText(str(best.get("class_name", text("No target yet", "ยังไม่พบเป้าหมาย"))))
        self.target_conf_label.setText(text("Confidence: ", "ความเชื่อมั่น: ") + (f"{float(best.get('confidence', 0)):.0%}" if best else "—"))
        self.target_stability_label.setText(text("Stability: —", "ความเสถียร: —"))
        self.target_status_label.setText(text("Webcam test · Not a survey mission", "ทดสอบเว็บแคม · ไม่ใช่ภารกิจสำรวจ"))
        self.target_frames_label.setText(text("No survey target tracking", "ไม่มีการติดตามเป้าหมายแบบสำรวจ"))

    def _start_webcam_test(self):
        if self._webcam_thread is None and self.camera is None:
            self._webcam_frames = self._webcam_peak = 0
            self._webcam_last_frame = 0.0
        super()._start_webcam_test()
        self._sync_dashboard()

    def _set_webcam_button_running(self, running):
        self._style_button(self.webcam_toggle_btn,
                           text("Stop webcam", "หยุดเว็บแคม") if running else text("Test webcam", "ทดสอบเว็บแคม"),
                           "stop" if running else "camera", "dangerButton" if running else "")
        self.webcam_toggle_btn.style().unpolish(self.webcam_toggle_btn)
        self.webcam_toggle_btn.style().polish(self.webcam_toggle_btn)

    def _on_webcam_thread_stopped(self):
        super()._on_webcam_thread_stopped()
        self.video.setText(self.video.empty_title)
        self._sync_dashboard()

    def stop_camera(self):
        super().stop_camera()
        if self.camera is None and self._webcam_thread is None:
            self.video.setText(self.video.empty_title)
        if hasattr(self, "connection_badge"):
            self._sync_dashboard()

    def _toggle_photo_recording(self):
        super()._toggle_photo_recording()
        self._style_button(self.record_photos_btn,
                           text("Stop recording", "หยุดบันทึกภาพ") if self._photo_record_dir else text("Capture for training", "บันทึกภาพฝึก AI"),
                           "stop" if self._photo_record_dir else "photo", "quiet")

    def _on_grid_progress_ui(self, i, total, x, y, heading, turn):
        super()._on_grid_progress_ui(i, total, x, y, heading, turn)
        self.mission_progress.setValue(round(100 * i / total) if total else 0)
        self.progress_label.setText(text(f"Waypoint {i} / {total}", f"จุดสำรวจ {i} / {total}"))
        self.position_label.setText(f"WP {i}/{total}  ·  X {x}  /  Y {y} cm  ·  {heading}")

    def start_autonomous_grid(self):
        previous = self.grid_mission
        super().start_autonomous_grid()
        if self.grid_mission is not previous:
            self.mission_progress.setValue(0)
            self.progress_label.setText(text("Taking off…", "กำลังขึ้นบิน…"))
        self._sync_dashboard()

    def _grid_finished_ui(self, success, reason):
        super()._grid_finished_ui(success, reason)
        if success:
            self.mission_progress.setValue(100)
        self.progress_label.setText(text("Mission complete", "ภารกิจเสร็จสิ้น") if success else text("Mission stopped: ", "หยุดภารกิจ: ") + str(reason))

    def _enter_video_fullscreen(self):
        if self._is_video_fullscreen:
            return
        self._saved_window_state = self.windowState()
        self._extra_hidden = (self._page_heading, self._metrics, self._inspector, self._camera_actions)
        self._saved_visibility = [(w, w.isVisible()) for w in self._extra_hidden]
        for widget in self._extra_hidden:
            widget.hide()
        super()._enter_video_fullscreen()
        self.video_fullscreen_btn.setText("")
        self._content_layout.setContentsMargins(8, 8, 8, 8)

    def _exit_video_fullscreen(self):
        if not self._is_video_fullscreen:
            return
        super()._exit_video_fullscreen()
        self.video_fullscreen_btn.setText("")
        for widget, was_visible in self._saved_visibility:
            widget.setVisible(was_visible)
        self._content_layout.setContentsMargins(22, 22, 22, 22)
        if self._saved_window_state & QtCore.Qt.WindowMaximized:
            self.showMaximized()
        self._update_compact_layout()
