"""
page.py
=======
หน้า "ลำเดียว + AI" แบบเข้าใจง่าย 4 ขั้น: เชื่อมต่อ -> เลือก AI -> เลือกวิธีบิน -> บิน

หน้านี้เป็นแค่หน้าตา งานจริงทั้งหมด (เชื่อมต่อโดรน, กล้อง, AI, บินสำรวจ, แจ้งเตือนแบต/หลุด, ลงจอดฉุกเฉิน,
รายงานหลังบิน, หน้าเขียนโค้ดที่มีจำลอง 3D) ใช้ MissionControlWindow เดิมของ AI Detection Lab ที่ทดสอบกับโดรนจริงแล้ว
โดยสร้างไว้แบบไม่แสดง (self.engine) แล้วสั่งงาน/ฟังสัญญาณจากมัน จึงไม่ต้องเขียนส่วนควบคุมโดรนใหม่
"""

from __future__ import annotations

import shutil
import time

from PySide6 import QtCore, QtGui, QtWidgets

from ai_detection_lab import config
from ai_detection_lab.ai.yolo_detector import load_model_selection, save_model_selection
from ..teach import project as tp
from ..teach.page import QSS, _button, _card, _label
from ..teach.widgets import FrameView, bgr_to_pixmap

STOCK_MODEL = "yolov8n.pt"
SKIP_MODELS = {"yolov8n-cls.pt"}      # โมเดลแยกรูป ตัวตรวจจับไม่ใช้
THUMB = (120, 68)

METHODS = {
    "grid": ("Survey flight", "โดรนบินซิกแซกทั่วสนาม ดูเส้นทางก่อนบินได้"),
    "keyboard": ("Keyboard control", "ขึ้น ลง เดินหน้า หมุน ด้วยปุ่มบนคีย์บอร์ด"),
    "code": ("Code flight", "เขียนคำสั่งเอง ลองในจำลอง 3D ก่อนบินจริงได้"),
}

FLY_QSS = QSS + """
QFrame#tile { background: #16212c; border: 1px solid #2d3f50; border-radius: 12px; }
QFrame#tile[on="true"] { background: #123a37; border: 2px solid #2f8f80; }
QLabel#tileTitle { color: #f4f8fb; font-size: 13pt; font-weight: 700; }
QLabel#status { color: #8ea0b2; font-size: 10pt; }
QLabel#chipOk { background: #13301f; color: #7fe0a0; border: 1px solid #2c6b45; border-radius: 8px; padding: 4px 10px; }
QLabel#chipWarn { background: #2b2410; color: #f2c46b; border: 1px solid #6b5520; border-radius: 8px; padding: 4px 10px; }
QLabel#chipOff { background: #14212c; color: #8ea0b2; border: 1px solid #263544; border-radius: 8px; padding: 4px 10px; }
QLabel#found { background: #13301f; color: #7fe0a0; border-radius: 10px; padding: 12px 14px; font-size: 13pt; font-weight: 700; }
QPushButton#danger { background: #a3262c; color: #ffffff; border: none; border-radius: 10px; padding: 9px 18px;
    font-size: 11pt; font-weight: 700; }
QPushButton#danger:hover { background: #c22e35; }
"""


def build_engine():
    """MissionControlWindow เดิม ไม่แสดงบนจอ ใช้เป็นเครื่องยนต์ของหน้านี้"""
    from ai_detection_lab.ui.mission_control import MissionControlWindow

    config.ensure_default_models_seeded()
    return MissionControlWindow()


def model_files():
    model_dir = config.MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    # AI ทั่วไปต้องมีเสมอ: ถ้าโฟลเดอร์ว่าง ตัวตรวจจับเปิดไม่ได้ และกล้องโดรนก็จะไม่เริ่ม
    stock = model_dir / STOCK_MODEL
    # รันจากซอร์ส: ไฟล์อยู่ข้างโปรแกรม / ตัว .exe: อยู่ในโฟลเดอร์ models ของแอป
    source = next((p for p in (config.PROJECT_ROOT / STOCK_MODEL, config.PROJECT_ROOT / "models" / STOCK_MODEL)
                   if p.exists()), None)
    if not stock.exists() and source is not None:
        shutil.copy2(source, stock)
    return sorted((p for p in model_dir.glob("*.pt") if p.name not in SKIP_MODELS),
                  key=lambda p: (p.name == STOCK_MODEL, p.name.lower()))


def model_choices():
    """AI ที่เลือกได้ (ใช้ทั้งหน้าลำเดียวและหน้าฝูง): [(ไฟล์, ชื่อที่โชว์, คำอธิบาย, เลือกอยู่ไหม)]"""
    projects = {p.model: p for p in tp.list_projects() if p.model}
    files = model_files()
    selection = load_model_selection(config.MODEL_DIR)
    names = [p.name for p in files]
    if selection["mode"] == "selected":
        chosen = set(selection["selected"]) & set(names)
    else:  # ค่าเดิม "อัตโนมัติ": โมเดลที่สอนเอง ถ้าไม่มีใช้ AI ทั่วไป
        chosen = {n for n in names if n != STOCK_MODEL} or {STOCK_MODEL}
    out = []
    for path in files:
        project = projects.get(path.name)
        if path.name == STOCK_MODEL:
            title, sub = "General AI", "รู้จัก 80 อย่าง เช่น คน รถ จักรยาน"
        elif project is not None:
            title = project.name
            score = (project.last_result or {}).get("score")
            sub = "สอนเองในหน้า Train AI" + (f" · ทดสอบได้ {score}%" if score is not None else "")
        else:
            title, sub = path.stem, "โมเดลที่นำมาใส่เอง"
        out.append((path.name, title, sub, path.name in chosen))
    return out


class ConnectWorker(QtCore.QThread):
    """ต่อโดรนเบื้องหลัง: ระบบเดิมลองซ้ำและค้นหาโดรนผ่านเราเตอร์ได้นานเกือบนาที ถ้าทำบนหน้าจอ แอปจะค้างทั้งหมด"""
    done = QtCore.Signal(bool)

    def __init__(self, drone):
        super().__init__()
        self.drone = drone

    def run(self):
        try:
            ok = bool(self.drone.connect())
        except Exception as exc:  # pyhulax ยก error ได้หลายแบบเมื่อไม่เจอโดรน
            print(f"connect failed: {exc}")
            ok = False
        if ok:
            try:  # โหลด AI ไว้ก่อนในเบื้องหลัง ขั้นถัดไปบนหน้าจอจะไม่ค้างนาน
                import ultralytics  # noqa: F401
            except Exception:
                pass
        self.done.emit(ok)


class Tile(QtWidgets.QFrame):
    """การ์ดกดเลือกได้ (แบบเลือกอันเดียว หรือติ๊กได้หลายอัน)"""
    clicked = QtCore.Signal()

    def __init__(self, title, sub, parent=None, sprite=None):
        super().__init__(parent)
        self.setObjectName("tile")
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setMinimumHeight(96)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(16, 12, 16, 12)
        self.sprite = None
        if sprite:  # ตัวละครพิกเซลขยับได้ (fly/sprites.py)
            from .sprites import PixelSprite

            self.sprite = PixelSprite(sprite)
            lay.addWidget(self.sprite)
        lay.addWidget(_label(title, "tileTitle"))
        sub_label = _label(sub, "muted", wrap=True)
        lay.addWidget(sub_label)
        lay.addStretch(1)
        self.on = False

    def set_on(self, flag):
        self.on = flag
        if self.sprite is not None:
            self.sprite.set_lively(flag)
        self.setProperty("on", "true" if flag else "false")
        self.style().unpolish(self)
        self.style().polish(self)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.clicked.emit()


class FlyPage(QtWidgets.QWidget):
    teach_requested = QtCore.Signal()

    def __init__(self, engine=None, parent=None):
        super().__init__(parent)
        self.setObjectName("teachRoot")
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.setStyleSheet(FLY_QSS)
        self.engine = engine or build_engine()
        self.mode = None            # "drone" / "sim" / "webcam" หลังเชื่อมต่อ
        self.method = "grid"
        self._found = 0
        self._grid_step = None
        self._current_step = 0
        self._connect_worker = None
        self._build()
        e = self.engine
        e.live_frame_signal.connect(self._on_drone_frame)
        e.webcam_frame_signal.connect(self._on_webcam_frame)
        e.frame_signal.connect(self._on_ai_result)
        e.status_signal.connect(self._on_status)
        e.grid_progress_signal.connect(self._on_grid_progress)
        e.grid_finished_signal.connect(self._on_grid_finished)
        e.webcam_thread_stopped_signal.connect(self._refresh_all)
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(1000)
        self._go(0)

    # ------------------------------------------------------------
    # โครงหน้า
    # ------------------------------------------------------------
    def _build(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        bar = QtWidgets.QFrame()
        bar.setObjectName("stepBar")
        row = QtWidgets.QHBoxLayout(bar)
        row.setContentsMargins(16, 8, 16, 8)
        self.step_buttons = []
        group = QtWidgets.QButtonGroup(self)
        for i, text in enumerate(("1  Connect", "2  Choose AI", "3  Flight mode", "4  Fly")):
            btn = _button(text, "stepBtn", lambda i=i: self._go(i))
            btn.setCheckable(True)
            group.addButton(btn)
            self.step_buttons.append(btn)
            row.addWidget(btn)
        row.addStretch(1)
        self.conn_chip = _label("", "chipOff")
        row.addWidget(self.conn_chip)
        self.battery_chip = _label("", "chipOk")
        self.battery_chip.hide()  # โผล่เมื่ออ่านค่าแบตได้แล้วเท่านั้น
        row.addWidget(self.battery_chip)
        # ปุ่มลงจอดอยู่ที่แถบบนเสมอเมื่อต่อโดรนแล้ว ไม่ว่าอยู่ขั้นไหน
        self.land_btn = _button("Land now", "danger", self._land_now)
        row.addWidget(self.land_btn)
        root.addWidget(bar)
        self.stack = QtWidgets.QStackedWidget()
        root.addWidget(self.stack, 1)
        for page in (self._build_connect(), self._build_ai(), self._build_method(), self._build_fly()):
            self.stack.addWidget(page)
        self.status_line = _label("", "status")
        self.status_line.setContentsMargins(18, 4, 18, 8)
        root.addWidget(self.status_line)

    def _go(self, step):
        if not self.step_buttons[step].isEnabled():
            step = 0
        self._current_step = step
        self.step_buttons[step].setChecked(True)
        self.stack.setCurrentIndex(step)
        if step == 1:
            self._refresh_models()
        self._refresh_all()

    def _connected(self):
        return bool(self.engine.drone.connected) and self.mode in ("drone", "sim")

    def _video_running(self):
        return self.engine._mission_ready() or self.engine._webcam_thread is not None

    def _refresh_all(self):
        connected = self._connected()
        webcam = self.engine._webcam_thread is not None
        self.step_buttons[2].setEnabled(connected)
        self.step_buttons[3].setEnabled(connected or webcam)
        if connected:
            ready = self.engine._mission_ready()
            name = "Simulator" if self.mode == "sim" else "Drone"
            self.conn_chip.setText(f"{name} connected" + ("" if ready else " · waiting for camera"))
            self.conn_chip.setObjectName("chipOk" if ready else "chipWarn")
        elif webcam:
            self.conn_chip.setText("Webcam")
            self.conn_chip.setObjectName("chipOk")
        else:
            self.conn_chip.setText("Not connected")
            self.conn_chip.setObjectName("chipOff")
        self.conn_chip.style().unpolish(self.conn_chip)
        self.conn_chip.style().polish(self.conn_chip)
        self.land_btn.setVisible(connected)
        self._refresh_connect()
        self._refresh_fly()

    def _tick(self):
        drone = self.engine.drone
        pct = None
        if self._connected() and hasattr(drone, "battery_pct"):
            try:
                pct = drone.battery_pct()
            except Exception:  # ลิงก์หลุดกลางทาง: ระบบแจ้งเตือนเดิมจัดการเอง
                pct = None
        self.battery_chip.setVisible(pct is not None)
        if pct is not None:
            self.battery_chip.setText(f"Battery {pct}%")
            from ai_detection_lab.drone import alerts

            self.battery_chip.setObjectName("chipWarn" if pct <= alerts.warn_pct() else "chipOk")
            self.battery_chip.style().unpolish(self.battery_chip)
            self.battery_chip.style().polish(self.battery_chip)
        self._refresh_all()
        if self._current_step == 3:
            self._refresh_captures()

    @QtCore.Slot(str)
    def _on_status(self, message):
        self.status_line.setText(message.splitlines()[0][:160] if message else "")

    # ------------------------------------------------------------
    # ขั้นที่ 1: เชื่อมต่อ
    # ------------------------------------------------------------
    def _build_connect(self):
        page = QtWidgets.QWidget()
        col = QtWidgets.QVBoxLayout(page)
        col.setContentsMargins(40, 28, 40, 24)
        col.setSpacing(12)
        col.addWidget(_label("Connect the drone", "title"))
        col.addWidget(_label("ต่อ Wi-Fi ของโดรนก่อน แล้วเลือกว่าจะใช้อะไร", "sub"))
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(14)
        self.connect_tiles = {}
        for key, title, sub in (
            ("drone", "Real drone", "ต่อ Wi-Fi ของโดรนให้เรียบร้อยก่อนกด"),
            ("sim", "Simulator", "ลองได้โดยไม่ต้องบินจริง ใช้เว็บแคมแทนกล้องโดรน"),
            ("webcam", "Webcam only", "ลอง AI โดยไม่บิน ไม่ต้องมีโดรน"),
        ):
            tile = Tile(title, sub, sprite=key)
            tile.clicked.connect(lambda k=key: self._connect(k))
            self.connect_tiles[key] = tile
            row.addWidget(tile, 1)
        col.addLayout(row)
        self.connect_state = _label("", "sub", wrap=True)
        col.addWidget(self.connect_state)
        buttons = QtWidgets.QHBoxLayout()
        self.disconnect_btn = _button("Disconnect", "plain", self._disconnect)
        buttons.addWidget(self.disconnect_btn)
        buttons.addStretch(1)
        self.connect_next = _button("Next: choose AI", "primary", lambda: self._go(1))
        buttons.addWidget(self.connect_next)
        col.addLayout(buttons)
        col.addStretch(1)
        return page

    def _refresh_connect(self):
        connected = self._connected()
        webcam = self.engine._webcam_thread is not None
        active = self.mode if (connected or webcam) else None
        for key, tile in self.connect_tiles.items():
            tile.set_on(key == active)
        if self._connect_worker is not None:
            seconds = int(time.monotonic() - self._connect_started)
            what = "Simulator" if self.mode == "sim" else "โดรน"
            text = f"กำลังเชื่อมต่อ{what}… {seconds} วินาที (ถ้าไม่เจอที่ Wi-Fi ของโดรน จะค้นหาผ่านเราเตอร์ต่อ อาจถึง 1 นาที)"
            for key, tile in self.connect_tiles.items():
                tile.set_on(key == self.mode)
            self.connect_state.setText(text)
            self.disconnect_btn.setVisible(False)
            return
        if connected and self.engine._mission_ready():
            text = "เชื่อมต่อแล้ว ภาพกล้องมาแล้ว กดต่อไปได้เลย"
        elif connected:
            text = "เชื่อมต่อแล้ว กำลังรอภาพจากกล้อง (ไม่กี่วินาที)"
        elif webcam:
            text = "เปิดเว็บแคมแล้ว เลือก AI แล้วไปขั้น Fly เพื่อดูภาพ"
        else:
            text = "กดการ์ดด้านบนเพื่อเชื่อมต่อ"
        self.connect_state.setText(text)
        self.disconnect_btn.setVisible(connected or webcam)

    def _connect(self, key):
        engine = self.engine
        if self._connect_worker is not None:
            return
        if self._connected() or engine._webcam_thread is not None:
            if key == self.mode:
                return
            QtWidgets.QMessageBox.information(self, "Already connected", "กด Disconnect ก่อน แล้วค่อยเลือกแบบอื่น")
            return
        self.mode = key
        if key == "webcam":
            engine._start_webcam_test()
            self._refresh_all()
            return
        want_sim = key == "sim"
        if engine.sim_mode_btn.isChecked() != want_sim:
            engine.sim_mode_btn.setChecked(want_sim)   # สลับตัวควบคุมได้เฉพาะตอนยังไม่เชื่อมต่อ
            if engine.sim_mode_btn.isChecked() != want_sim:
                QtWidgets.QMessageBox.warning(self, "Can't switch", "ปิดแล้วเปิดแอปใหม่ก่อน แล้วค่อยเลือกแบบนี้")
                self.mode = None
                return
        self._connect_started = time.monotonic()
        self._connect_worker = ConnectWorker(engine.drone)
        self._connect_worker.done.connect(self._on_connected)
        self._connect_worker.start()
        self._refresh_all()

    @QtCore.Slot(bool)
    def _on_connected(self, ok):
        worker, self._connect_worker = self._connect_worker, None
        worker.wait()
        engine = self.engine
        if not ok:
            self.mode = None
            reason = getattr(engine.drone, "last_error", "") or ""
            QtWidgets.QMessageBox.warning(
                self, "Can't connect",
                "เชื่อมต่อโดรนไม่ได้\n\n• เปิดโดรนแล้วหรือยัง\n• คอมต่อ Wi-Fi ของโดรนอยู่ไหม\n• ปิดแอปอื่นที่ใช้โดรนอยู่"
                + (f"\n\n{reason}" if "another computer" in reason else ""))
            self._refresh_all()
            return
        # โดรนต่อแล้ว ขั้นนี้แค่เปิดกล้องและ AI (ระบบเดิมแตะหน้าจอของมันเอง จึงต้องทำบน GUI thread)
        self.connect_state.setText("เชื่อมต่อแล้ว กำลังเปิดกล้องและ AI…")
        QtWidgets.QApplication.processEvents()
        engine._connect_for_mission()
        self._refresh_all()

    def _disconnect(self):
        engine = self.engine
        if engine._webcam_thread is not None:
            engine._stop_webcam_test()
        if engine.drone.connected or engine.camera is not None:
            if getattr(engine.drone, "flying", False):
                QtWidgets.QMessageBox.information(self, "Still flying", "กด Land now ก่อน แล้วค่อยตัดการเชื่อมต่อ")
                return
            engine.stop_camera()
            try:
                engine.drone.disconnect()
            except Exception as exc:  # หลุดอยู่แล้วก็ถือว่าตัดสำเร็จ
                self._on_status(f"ตัดการเชื่อมต่อ: {exc}")
        self.mode = None
        self._go(0)

    # ------------------------------------------------------------
    # ขั้นที่ 2: เลือก AI (เลือกได้หลายตัว)
    # ------------------------------------------------------------
    def _build_ai(self):
        page = QtWidgets.QWidget()
        col = QtWidgets.QVBoxLayout(page)
        col.setContentsMargins(40, 28, 40, 24)
        col.setSpacing(12)
        col.addWidget(_label("What should the drone find?", "title"))
        col.addWidget(_label("กดเลือก AI ได้มากกว่า 1 ตัว ยิ่งเลือกเยอะ เครื่องยิ่งทำงานช้าลง", "sub"))
        self.model_grid = QtWidgets.QGridLayout()
        self.model_grid.setSpacing(12)
        col.addLayout(self.model_grid)
        self.ai_hint = _label("", "warn")
        col.addWidget(self.ai_hint)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(_button("ยังไม่มี AI ที่ต้องการ? ไปหน้า Train AI", "link", self.teach_requested.emit))
        row.addStretch(1)
        self.ai_next = _button("Next", "primary", self._ai_next)
        row.addWidget(self.ai_next)
        col.addLayout(row)
        col.addStretch(1)
        return page

    def _refresh_models(self):
        while self.model_grid.count():
            item = self.model_grid.takeAt(0)
            if item.widget():
                item.widget().hide()
                item.widget().deleteLater()
        self.model_tiles = {}
        for i, (name, title, sub, on) in enumerate(model_choices()):
            tile = Tile(title, sub)
            tile.set_on(on)
            tile.clicked.connect(lambda n=name: self._toggle_model(n))
            self.model_tiles[name] = tile
            self.model_grid.addWidget(tile, i // 3, i % 3)
        self._save_models()

    def _toggle_model(self, name):
        tile = self.model_tiles[name]
        tile.set_on(not tile.on)
        self._save_models()

    def _save_models(self):
        chosen = [n for n, t in self.model_tiles.items() if t.on]
        self.ai_hint.setText("" if chosen else "เลือก AI อย่างน้อย 1 ตัว")
        self.ai_next.setEnabled(bool(chosen))
        if chosen:
            # ตัวตรวจจับอ่านไฟล์นี้ทุกภาพ จึงเปลี่ยนได้ทันทีแม้ AI กำลังทำงานอยู่
            save_model_selection(config.MODEL_DIR, "selected", chosen)

    def _ai_next(self):
        self._go(3 if self.mode == "webcam" else 2)

    # ------------------------------------------------------------
    # ขั้นที่ 3: เลือกวิธีบิน
    # ------------------------------------------------------------
    def _build_method(self):
        page = QtWidgets.QWidget()
        col = QtWidgets.QVBoxLayout(page)
        col.setContentsMargins(40, 28, 40, 24)
        col.setSpacing(12)
        col.addWidget(_label("Choose a flight mode", "title"))
        col.addWidget(_label("AI จะคอยหาของและถ่ายรูปเก็บไว้ให้เองทุกแบบ", "sub"))
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(14)
        self.method_tiles = {}
        for key, (title, sub) in METHODS.items():
            tile = Tile(title, sub)
            tile.clicked.connect(lambda k=key: self._select_method(k))
            self.method_tiles[key] = tile
            row.addWidget(tile, 1)
        col.addLayout(row)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(_button("Next: fly", "primary", lambda: self._go(3)))
        col.addLayout(buttons)
        col.addStretch(1)
        self._select_method(self.method)
        return page

    def _select_method(self, key):
        self.method = key
        for k, tile in self.method_tiles.items():
            tile.set_on(k == key)

    # ------------------------------------------------------------
    # ขั้นที่ 4: บิน
    # ------------------------------------------------------------
    def _build_fly(self):
        page = QtWidgets.QWidget()
        outer = QtWidgets.QHBoxLayout(page)
        outer.setContentsMargins(24, 16, 24, 12)
        outer.setSpacing(16)
        left = QtWidgets.QVBoxLayout()
        self.video = FrameView()
        self.video.placeholder = "รอภาพจากกล้อง…"
        left.addWidget(self.video, 1)
        actions = QtWidgets.QHBoxLayout()
        self.start_btn = _button("", "primary", self._start_flying)
        actions.addWidget(self.start_btn)
        self.preview_btn = _button("Preview route", "plain", self.engine.preview_grid)
        actions.addWidget(self.preview_btn)
        actions.addStretch(1)
        self.report_btn = _button("Flight report", "plain", lambda: self.engine._show_post_flight_report(modal=False))
        actions.addWidget(self.report_btn)
        left.addLayout(actions)
        outer.addLayout(left, 3)

        side = QtWidgets.QVBoxLayout()
        side.setSpacing(12)
        self.found_label = _label("ยังไม่เจอ", "found")
        self.found_label.setWordWrap(True)
        side.addWidget(self.found_label)
        progress_card, pl = _card()
        pl.addWidget(_label("Route", "muted"))
        self.progress = QtWidgets.QProgressBar()
        self.progress.setTextVisible(False)
        pl.addWidget(self.progress)
        self.progress_label = _label("", "body")
        pl.addWidget(self.progress_label)
        self.progress_card = progress_card
        side.addWidget(progress_card)
        shots_card, sl = _card()
        sl.addWidget(_label("Photos saved by the AI", "muted"))
        self.shots_row = QtWidgets.QGridLayout()
        self.shots_row.setSpacing(6)
        sl.addLayout(self.shots_row)
        self.shots_empty = _label("ยังไม่มี", "body")
        sl.addWidget(self.shots_empty)
        side.addWidget(shots_card)
        side.addStretch(1)
        outer.addLayout(side, 1)
        self._shot_files = []
        return page

    def _refresh_fly(self):
        if not hasattr(self, "start_btn"):
            return
        flying_grid = bool(self.engine.grid_mission and self.engine.grid_mission.is_alive())
        webcam = self.mode == "webcam"
        self.start_btn.setVisible(not webcam)
        self.preview_btn.setVisible(not webcam and self.method == "grid")
        self.progress_card.setVisible(not webcam and self.method == "grid")
        self.start_btn.setText({
            "grid": "Surveying…" if flying_grid else "Start survey",
            "keyboard": "Open keyboard control",
            "code": "Open code editor (3D sim)",
        }[self.method])
        self.start_btn.setEnabled(self._connected() and not flying_grid and self.engine._mission_ready())
        self.report_btn.setEnabled(self.engine.detector is not None)

    def _start_flying(self):
        engine = self.engine
        if self.method == "grid":
            engine.start_autonomous_grid()
        elif self.method == "keyboard":
            engine._open_keyboard_mission()
        else:
            engine._open_flight_plan_mission()
        self._refresh_all()

    def _land_now(self):
        self.engine.emergency_stop(blocking=False)
        self._on_status("สั่งลงจอดแล้ว")

    @QtCore.Slot(object)
    def _on_drone_frame(self, frame):
        if frame is None or self.mode == "webcam":
            return
        self.video.set_image(frame)
        h, w = frame.shape[:2]
        boxes = []
        for det in self.engine._latest_detections:
            x1, y1, x2, y2 = det.get("xyxy", (0, 0, 0, 0))
            boxes.append(dict(box=[x1 / w, y1 / h, x2 / w, y2 / h], color="#6ff0d8",
                              label=f"{det.get('class_name', '?')} {float(det.get('confidence', 0)):.0%}"))
        self.video.set_boxes(boxes)

    @QtCore.Slot(object, object)
    def _on_webcam_frame(self, frame, _detections):
        if frame is not None:
            self.video.set_image(frame)  # ตัวตรวจจับวาดกรอบบนภาพเว็บแคมมาแล้ว
            self.video.set_boxes([])

    @QtCore.Slot(object, object)
    def _on_ai_result(self, _frame, data):
        found = int(data.get("unique_target_count", 0) or 0)
        now = data.get("detections", [])
        if now:
            names = sorted({d.get("class_name", "?") for d in now})
            self.found_label.setText(f"เห็น {', '.join(names)} อยู่ตอนนี้\nเจอแล้วทั้งหมด {found} ครั้ง")
        elif found:
            self.found_label.setText(f"เจอแล้วทั้งหมด {found} ครั้ง")

    @QtCore.Slot(int, int, int, int, str, str)
    def _on_grid_progress(self, index, total, *_rest):
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(index + 1)
        self.progress_label.setText(f"จุดที่ {index + 1} จาก {total}")

    @QtCore.Slot(bool, str)
    def _on_grid_finished(self, success, reason):
        self.progress_label.setText("บินครบแล้ว" if success else f"หยุดแล้ว: {reason}")
        self._refresh_all()

    def _refresh_captures(self):
        detector = self.engine.detector
        folder = getattr(getattr(detector, "capture", None), "mission_dir", None)
        files = sorted(folder.glob("*.jpg"))[-6:] if folder is not None and folder.exists() else []
        if files == self._shot_files:
            return
        self._shot_files = files
        while self.shots_row.count():
            item = self.shots_row.takeAt(0)
            if item.widget():
                item.widget().hide()
                item.widget().deleteLater()
        from ai_detection_lab.image_io import imread_any

        for i, path in enumerate(reversed(files)):
            image = imread_any(path)
            if image is None:
                continue
            thumb = QtWidgets.QLabel()
            thumb.setPixmap(bgr_to_pixmap(image).scaled(*THUMB, QtCore.Qt.KeepAspectRatio,
                                                        QtCore.Qt.SmoothTransformation))
            thumb.setToolTip(path.name)
            self.shots_row.addWidget(thumb, i // 2, i % 2)
        self.shots_empty.setVisible(not files)

    # ------------------------------------------------------------
    def closeEvent(self, event):
        if self._connect_worker is not None:
            self._connect_worker.wait(60000)  # ปล่อย thread ระหว่างทำงานไม่ได้ (Qt จะปิดโปรแกรม)
        self._timer.stop()
        # ให้ระบบเดิมลงจอด/ปิดกล้อง/หยุดเว็บแคมตามแบบของมัน
        if self.engine.close():
            event.accept()
        else:
            event.ignore()
