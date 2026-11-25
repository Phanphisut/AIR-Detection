"""
app_window.py
=============
หน้าต่างหลักของ AIR Detection: แถบด้านบนสลับ 2 หน้า

  - ลำเดียว + AI : บินโดรน 1 ลำพร้อม AI ตรวจจับ 4 ขั้น (fly/)
                   งานจริงใช้ MissionControlWindow เดิมของ AI Detection Lab v9 ที่ไม่แสดงบนจอ
  - สอน AI       : สอน AI จากคลิป 4 ขั้น (teach/) โปรเจกต์ -> อัปโหลดคลิป -> วาดกรอบทุกรูป -> สอนและทดสอบ

แต่ละหน้าสร้างตอนกดเข้าครั้งแรก (หน้าโดรนโหลด torch/YOLO นาน จึงไม่บังคับให้รอตอนเปิดแอป)
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

APP_NAME = "AIR Detection"
APP_VERSION = "1.0.0"
# ชื่อเดียวกับ AI Detection Lab v9 และ Swarm Detection Lab: เปิดซ้อนกันไม่ได้ (แย่งโดรนและกล้องกัน)
SINGLE_INSTANCE_KEY = "AI_Detection_Lab-SingleInstanceGuard"

QSS = """
QWidget#appRoot { background: #0b1118; }
QFrame#modeBar { background: #0f1720; border-bottom: 1px solid #263544; }
QLabel#brand { color: #f4f8fb; font-family: "Segoe UI"; font-size: 15px; font-weight: 800; }
QLabel#brandSub { color: #77e4d0; font-family: "Segoe UI"; font-size: 8pt; font-weight: 700; letter-spacing: 1px; }
QPushButton#modeBtn {
    background: #111923; color: #c9d5e0; border: 1px solid #263544; border-radius: 8px;
    padding: 7px 16px; font-family: "Segoe UI"; font-size: 10pt; font-weight: 600;
}
QPushButton#modeBtn:hover { border-color: #3b5568; color: #f4f8fb; }
QPushButton#modeBtn:checked { background: #123a37; color: #8ff0dc; border-color: #2f8f80; }
QWidget#pageHost { background: #0b1118; }
QLabel#loading { color: #8ea0b2; font-family: "Segoe UI"; font-size: 12pt; }
QLabel#errorTitle { color: #f4f8fb; font-family: "Segoe UI"; font-size: 20px; font-weight: 800; }
QPlainTextEdit#errorText { background: #140d10; color: #f0a8a8; border: 1px solid #5a2a2a; border-radius: 8px;
    font-family: Consolas; font-size: 9pt; }
"""


def build_fly_page():
    from ai_detection_lab.fly.page import FlyPage

    return FlyPage()


def build_teach_page():
    from ai_detection_lab.teach.page import TeachPage

    return TeachPage()


def fly_busy(page):
    """หน้าโดรนกำลังใช้โดรน/กล้อง/บินสำรวจอยู่ไหม (หน้าสอน AI ถามก่อนเริ่มสอน เพราะสอนใช้ CPU เต็ม)"""
    engine = page.engine
    grid = engine.grid_mission
    return bool(engine.drone.connected) or engine.camera is not None or bool(grid and grid.is_alive())


class LazyPage(QtWidgets.QWidget):
    """แสดง "กำลังโหลด" ก่อน แล้วค่อยสร้างหน้าจริงตอนเปิดครั้งแรก"""
    ready = QtCore.Signal(object)

    def __init__(self, factory, loading_text, parent=None):
        super().__init__(parent)
        self.setObjectName("pageHost")
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self._factory = factory
        self.page = None
        self._started = False
        self._stack = QtWidgets.QStackedLayout(self)
        loading = QtWidgets.QLabel(loading_text)
        loading.setObjectName("loading")
        loading.setAlignment(QtCore.Qt.AlignCenter)
        self._stack.addWidget(loading)

    def ensure_loaded(self):
        if self._started:
            return
        self._started = True
        QtCore.QTimer.singleShot(60, self._load)  # ให้ข้อความ "กำลังโหลด" ขึ้นจอก่อน

    def _load(self):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            page = self._factory()
        except Exception:
            self._show_error(traceback.format_exc())
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self._stack.addWidget(page)
        self._stack.setCurrentWidget(page)
        self.page = page
        self.ready.emit(page)

    def _show_error(self, text):
        print(text, file=sys.stderr)
        box = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(box)
        lay.setContentsMargins(40, 40, 40, 40)
        title = QtWidgets.QLabel("This page couldn't open")
        title.setObjectName("errorTitle")
        lay.addWidget(title)
        err = QtWidgets.QPlainTextEdit(text)
        err.setObjectName("errorText")
        err.setReadOnly(True)
        lay.addWidget(err, 1)
        self._stack.addWidget(box)
        self._stack.setCurrentWidget(box)

    def close_page(self):
        """ปิดหน้าที่โหลดไว้ (หน้าโดรนจะลงจอด/ตัดการเชื่อมต่อเอง) — False = ผู้ใช้ยกเลิก"""
        return self.page is None or self.page.close()


class AppWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1400, 900)
        self.setMinimumSize(1100, 740)
        self.setStyleSheet(QSS)
        self.pages = {}
        self.buttons = {}
        self.current = None

        root = QtWidgets.QWidget()
        root.setObjectName("appRoot")
        self.setCentralWidget(root)
        col = QtWidgets.QVBoxLayout(root)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)
        col.addWidget(self._build_bar())
        self.stack = QtWidgets.QStackedWidget()
        col.addWidget(self.stack, 1)

        self._add("fly", LazyPage(build_fly_page, "กำลังโหลด AI (YOLO / PyTorch)… ครั้งแรกอาจใช้เวลาสักครู่"))
        self._add("teach", LazyPage(build_teach_page, "กำลังเปิดหน้า Train AI…"))
        self.pages["fly"].ready.connect(lambda page: page.teach_requested.connect(lambda: self.switch("teach")))
        self.pages["teach"].ready.connect(self._on_teach_ready)
        # เปิดมาที่หน้า Fly & Detect (ผู้ใช้เลือก) — หน้านี้โหลด AI ราว 15 วินาที ระหว่างนั้นขึ้นข้อความกำลังโหลด
        self.switch("fly")

    def _build_bar(self):
        bar = QtWidgets.QFrame()
        bar.setObjectName("modeBar")
        bar.setFixedHeight(58)
        row = QtWidgets.QHBoxLayout(bar)
        row.setContentsMargins(18, 0, 18, 0)
        row.setSpacing(8)
        names = QtWidgets.QVBoxLayout()
        names.setSpacing(0)
        brand = QtWidgets.QLabel(APP_NAME)
        brand.setObjectName("brand")
        sub = QtWidgets.QLabel("TEACH  ·  FLY  ·  DETECT")
        sub.setObjectName("brandSub")
        names.addStretch(1)
        names.addWidget(brand)
        names.addWidget(sub)
        names.addStretch(1)
        row.addLayout(names)
        row.addSpacing(28)
        group = QtWidgets.QButtonGroup(self)
        group.setExclusive(True)
        for key, text in (("fly", "Fly && Detect"), ("teach", "Train AI")):
            btn = QtWidgets.QPushButton(text)
            btn.setObjectName("modeBtn")
            btn.setCheckable(True)
            btn.setCursor(QtCore.Qt.PointingHandCursor)
            btn.clicked.connect(lambda _=False, k=key: self.switch(k))
            group.addButton(btn)
            row.addWidget(btn)
            self.buttons[key] = btn
        row.addStretch(1)
        return bar

    def _add(self, key, page):
        self.pages[key] = page
        self.stack.addWidget(page)

    def switch(self, key):
        self.current = key
        self.buttons[key].setChecked(True)
        page = self.pages[key]
        self.stack.setCurrentWidget(page)
        page.ensure_loaded()

    def _on_teach_ready(self, page):
        fly = self.pages["fly"]
        page.busy_probe = lambda: fly.page is not None and fly_busy(fly.page)
        page.use_model_requested.connect(lambda: self.switch("fly"))

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        for key in ("fly", "teach"):
            if not self.pages[key].close_page():
                event.ignore()
                return
        event.accept()


def fit_to_screen(window):
    """จอเล็ก -> เต็มจอ, จอใหญ่ -> 90% ของพื้นที่จอ วางกลางจอ"""
    screen = window.screen() or QtWidgets.QApplication.primaryScreen()
    area = screen.availableGeometry()
    if area.width() < 1500 or area.height() < 950:
        window.showMaximized()
        return
    w, h = int(area.width() * 0.9), int(area.height() * 0.9)
    window.resize(w, h)
    window.move(area.x() + (area.width() - w) // 2, area.y() + (area.height() - h) // 2)
    window.show()


def run():
    """เปิดแอป (entry_point.py เรียกหลังตั้ง stdout/stderr และ freeze_support แล้ว)"""
    for stream in (sys.stdout, sys.stderr):  # Windows ภาษาไทยใช้ codepage แคบ: อีโมจิในข้อความแล้วพัง
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
    from ai_detection_lab import config

    QtCore.QLocale.setDefault(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setApplicationName(APP_NAME)
    from ai_detection_lab.ui import pixel_theme

    pixel_theme.install(app, config.PROJECT_ROOT)  # หน้าจอแบบ Pixel Art (ปิดด้วย AIR_PIXEL_UI=0)
    icon = Path(config.PROJECT_ROOT) / "app_icon.png"
    if icon.is_file():
        app.setWindowIcon(QtGui.QIcon(str(icon)))

    guard = QtCore.QSharedMemory(SINGLE_INSTANCE_KEY)
    if guard.attach():  # ค้างจากครั้งก่อนที่แอปถูกปิดกะทันหัน (Windows ปล่อยให้เองเมื่อไม่มีใครถือ)
        guard.detach()
    if not guard.create(1):
        QtWidgets.QMessageBox.information(
            None, APP_NAME,
            "มี AIR Detection, AI Detection Lab หรือ Swarm Detection Lab เปิดอยู่แล้ว\nกรุณาปิดโปรแกรมนั้นก่อน (สองโปรแกรมใช้โดรนและกล้องพร้อมกันไม่ได้)")
        return 0

    def log_uncaught(exc_type, exc_value, exc_tb):
        # ข้อผิดพลาดที่หลุดมา (เช่นโดรนหลุดกลางคำสั่ง) ต้องไม่ทำให้โปรแกรมปิดกลางอากาศ
        traceback.print_exception(exc_type, exc_value, exc_tb)

    sys.excepthook = log_uncaught
    config.ensure_default_models_seeded()
    window = AppWindow()
    fit_to_screen(window)
    code = app.exec()
    guard.detach()
    return code
