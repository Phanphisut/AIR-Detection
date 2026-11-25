"""ทดสอบหน้า "ลำเดียว + AI" แบบใหม่ ด้วยเครื่องยนต์จริง (ไม่แสดงหน้าจอ ไม่แตะข้อมูลจริง)

ทดสอบ: สถานะเริ่มต้น, เลือก AI หลายตัว, ต่อโดรนจริงไม่ได้ต้องบอกเหตุผล, โดรนจำลอง (ถ้ามีเว็บแคมจะลองบินสำรวจ),
ลงจอด, ตัดการเชื่อมต่อ, ชื่อไทยบนกรอบ
"""
import faulthandler
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

faulthandler.enable()
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")
ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "vendor"):
    sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
from PySide6 import QtCore, QtWidgets  # noqa: E402

app = QtWidgets.QApplication(sys.argv)
from ai_detection_lab import config  # noqa: E402
from ai_detection_lab.teach import project as tp  # noqa: E402

work = Path(tempfile.mkdtemp(prefix="fly_"))
config.MODEL_DIR = work / "models"
config.CAPTURE_DIR = work / "captures"
tp.TRAINING_ROOT = work / "sessions"
tp.TRAINING_ROOT.mkdir()
shots = Path(sys.argv[1]) if len(sys.argv) > 1 else None


class Dialogs(QtCore.QObject):
    def __init__(self):
        super().__init__()
        self.seen = []
        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.tick)
        timer.start(50)

    def tick(self):
        for w in app.topLevelWidgets():
            if isinstance(w, QtWidgets.QMessageBox) and w.isVisible():
                self.seen.append(w.text())
                (w.button(QtWidgets.QMessageBox.Yes) or w.button(QtWidgets.QMessageBox.Ok) or w.defaultButton()).click()
            elif isinstance(w, QtWidgets.QDialog) and w.isVisible():
                self.seen.append(type(w).__name__)  # เช่น รายงานหลังบินที่เด้งขึ้นเองตอนบินจบ
                w.reject()


dialogs = Dialogs()


def pump(seconds=0.1):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.01)


def shot(name):
    if shots:
        page.grab().save(str(shots / f"fly_{name}.png"))


# โปรเจกต์ที่สอนแล้ว 1 อัน: ต้องโชว์ชื่อของ ไม่ใช่ชื่อไฟล์
taught = tp.TeachProject.create("รถสีส้ม")
taught.install_model(ROOT / "yolov8n.pt", dict(conf_thres=0.5), 320)
taught.last_result = dict(score=83, frames=6, wrong=0)
taught.save()

t0 = time.time()
from ai_detection_lab.fly.page import FlyPage  # noqa: E402

page = FlyPage()
page.resize(1400, 860)
page.show()
pump(0.3)
print(f"page built in {time.time()-t0:.1f}s")
assert page._current_step == 0
assert not page.step_buttons[2].isEnabled() and not page.step_buttons[3].isEnabled()
assert not page.land_btn.isVisible() and page.conn_chip.text() == "Not connected"
assert not page.engine.isVisible()
shot("1_connect")

# ขั้นที่ 2: เลือก AI หลายตัว (ทำได้ก่อนเชื่อมต่อ)
page._go(1)
pump(0.2)
assert (config.MODEL_DIR / "yolov8n.pt").exists(), "general AI is always available"
titles = {name: tile.findChild(QtWidgets.QLabel, "tileTitle").text() for name, tile in page.model_tiles.items()}
print("models:", titles)
assert titles[taught.model] == "รถสีส้ม" and titles["yolov8n.pt"] == "General AI"
assert page.model_tiles[taught.model].on and not page.model_tiles["yolov8n.pt"].on, "default: own models"
page.model_tiles["yolov8n.pt"].clicked.emit()
sel = json.loads((config.MODEL_DIR / "active_models.json").read_text(encoding="utf-8"))
assert sel["mode"] == "selected" and set(sel["selected"]) == {"yolov8n.pt", taught.model}, sel
shot("2_ai")
for tile in page.model_tiles.values():
    if tile.on:
        tile.clicked.emit()
assert not page.ai_next.isEnabled() and page.ai_hint.text()
page.model_tiles[taught.model].clicked.emit()
assert page.ai_next.isEnabled()
assert json.loads((config.MODEL_DIR / "active_models.json").read_text(encoding="utf-8"))["selected"] == [taught.model]

# ต่อโดรนจริงไม่ได้ (ไม่มีโดรนในห้องทดสอบ): ต้องบอกเหตุผลเป็นภาษาไทย และกลับสู่สถานะเดิม
page._go(0)
dialogs.seen.clear()
t0 = time.time()
page.connect_tiles["drone"].clicked.emit()
assert time.time() - t0 < 1.0, "the screen must not freeze while searching for the drone"
pump(3.0)
assert "กำลังเชื่อมต่อโดรน" in page.connect_state.text(), page.connect_state.text()
page.connect_tiles["sim"].clicked.emit()  # กดซ้ำระหว่างค้นหา ต้องไม่เริ่มอีกรอบ
assert page.mode == "drone"
while page._connect_worker is not None and time.time() - t0 < 120:
    pump(0.2)
pump(0.5)
print(f"real drone attempt: {time.time()-t0:.0f}s, message: {dialogs.seen[:1]}")
assert dialogs.seen and "Wi-Fi" in dialogs.seen[0] and page.mode is None and not page._connected()

# โดรนจำลอง
page.connect_tiles["sim"].clicked.emit()
t0 = time.time()
while page._connect_worker is not None and time.time() - t0 < 60:
    pump(0.2)
pump(1.0)
assert page._connected() and page.mode == "sim", page.conn_chip.text()
assert page.land_btn.isVisible() and page.step_buttons[2].isEnabled()
t0 = time.time()
while not page.engine._mission_ready() and time.time() - t0 < 25:
    pump(0.2)
camera = page.engine._mission_ready()
print(f"sim connected; camera (laptop webcam) ready: {camera}")
shot("1_connected")
page._go(2)
page._select_method("keyboard")
page._go(3)
pump(0.3)
assert page.start_btn.text() == "Open keyboard control" and not page.preview_btn.isVisible()
page._go(2)
page._select_method("grid")
page._go(3)
pump(0.3)
assert page.start_btn.text() == "Start survey" and page.preview_btn.isVisible()

# ชื่อไทยบนกรอบ
frame = np.zeros((360, 640, 3), np.uint8)
page.engine._latest_detections = [dict(xyxy=[64, 36, 320, 180], class_name="รถสีส้ม", confidence=0.91)]
page.mode = "sim"
page._on_drone_frame(frame)
assert page.video._boxes[0]["label"] == "รถสีส้ม 91%" and abs(page.video._boxes[0]["box"][2] - 0.5) < 1e-6

if camera:
    page._start_flying()
    t0 = time.time()
    while page.progress.value() < 2 and time.time() - t0 < 60:
        pump(0.3)
    print("grid progress:", page.progress_label.text())
    shot("4_flying")
    page.land_btn.click()
    t0 = time.time()
    while page.engine.grid_mission and time.time() - t0 < 30:
        pump(0.3)
    assert page.engine.grid_mission is None, "landing stops the survey"
    print("landed:", page.progress_label.text(), "| popups:", [s for s in dialogs.seen if "Dialog" in s])
else:
    print("no webcam here: survey flight in simulation not tested")
    shot("4_fly")

# ตัดการเชื่อมต่อ: ระหว่างที่ยังบินอยู่ต้องไม่ยอม รอลงจอดเสร็จแล้วค่อยตัด
if page.engine.drone.flying:
    dialogs.seen.clear()
    page._disconnect()
    pump(0.3)
    assert page._connected() and any("Land now ก่อน" in t for t in dialogs.seen), "no disconnect while flying"
t0 = time.time()
while page.engine.drone.flying and time.time() - t0 < 30:
    pump(0.3)
print(f"flying after landing request: {page.engine.drone.flying} ({time.time()-t0:.0f}s)")
page._disconnect()
pump(0.5)
assert page.mode is None and not page._connected() and page._current_step == 0
assert not page.land_btn.isVisible()
print("disconnected")

page.close()
pump(0.5)
shutil.rmtree(work, ignore_errors=True)
print("ALL OK")
