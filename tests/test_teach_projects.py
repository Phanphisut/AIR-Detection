"""ทดสอบการสลับและลบโปรเจกต์ในหน้า "สอน AI" ด้วยการกดปุ่มจริง (ไม่แสดงหน้าจอ ไม่แตะข้อมูลจริง)"""
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
for p in (ROOT, ROOT / "vendor", Path(__file__).resolve().parent):
    sys.path.insert(0, str(p))

from PySide6 import QtCore, QtWidgets  # noqa: E402

app = QtWidgets.QApplication(sys.argv)

import make_sample_clip  # noqa: E402
from ai_detection_lab import config  # noqa: E402
from ai_detection_lab.teach import project as tp  # noqa: E402
from ai_detection_lab.teach.clips import extract_frames  # noqa: E402
from ai_detection_lab.teach.page import TeachPage  # noqa: E402

work = Path(tempfile.mkdtemp(prefix="teach_projects_"))
tp.TRAINING_ROOT = work / "sessions"
tp.TRAINING_ROOT.mkdir()
config.MODEL_DIR = work / "models"


class Dialogs(QtCore.QObject):
    """ตอบกล่องข้อความอัตโนมัติด้วยคำตอบที่ตั้งไว้"""

    def __init__(self):
        super().__init__()
        self.answer = QtWidgets.QMessageBox.Yes
        self.seen = []
        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.tick)
        timer.start(50)

    def tick(self):
        for w in app.topLevelWidgets():
            if isinstance(w, QtWidgets.QMessageBox) and w.isVisible():
                self.seen.append(w.text())
                (w.button(self.answer) or w.button(QtWidgets.QMessageBox.Ok)).click()


dialogs = Dialogs()


def pump(seconds=0.1):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.01)


def rows():
    return [b for b in page.findChildren(QtWidgets.QPushButton, "project") if b.isVisible()]


def delete_buttons():
    return [b for b in page.projects_card.findChildren(QtWidgets.QPushButton, "plain") if b.text() == "Delete" and b.isVisible()]


# โปรเจกต์ที่พร้อมทดสอบ มีคลิป กรอบ และโมเดลจริง
clip = work / "c.mp4"
truth = make_sample_clip.make_clip(clip, seconds=14, w=480, h=270)
old = tp.TeachProject.create("รถสีส้ม")
cid, dest = old.copy_clip_in(clip)
old.add_clip(cid, "c.mp4", dest, extract_frames(dest, old.frames_dir, 0, cid))
for f in old.frames:
    old.set_box(f, truth[f["src"]])
old.install_model(ROOT / "yolov8n.pt", dict(conf_thres=0.5), 320)
old_model = config.MODEL_DIR / old.model
assert old_model.exists()

page = TeachPage()
page.resize(1400, 860)
page.show()
pump(0.3)
assert len(rows()) == 1 and page.stack.currentIndex() == 0 and page.project is None
assert not page.change_btn.isVisible()

# เผลอกดเข้าโปรเจกต์เก่า: เข้าหน้าทดสอบทันที แล้วกดเปลี่ยนโปรเจกต์ระหว่างที่ AI ยังทดสอบอยู่
rows()[0].click()
pump(0.1)
assert page.project is not None and page.stack.currentIndex() == 3 and page._test_worker is not None
assert page.change_btn.isVisible()
page.change_btn.click()
pump(0.1)
assert page.project is None and page.stack.currentIndex() == 0, "back to the project list"
assert not page.change_btn.isVisible() and not page.step_buttons[1].isEnabled()
assert page._test_worker is None
print("left while test was running; retired workers:", len(page._retired_workers))

# เข้าโปรเจกต์ใหม่ทันที: ผลที่ค้างจากโปรเจกต์เก่าต้องไม่โผล่มา
page.name_input.setText("ลูกบอล")
page._create_project()
new = page.project
assert new.name == "ลูกบอล" and page.stack.currentIndex() == 1
t0 = time.time()
while page._retired_workers and time.time() - t0 < 120:
    pump(0.1)
assert not page._retired_workers, "retired worker finishes"
assert page._results == [] and page.project is new and page.stack.currentIndex() == 1
print(f"old test finished after {time.time()-t0:.1f}s without touching the new project")

# ปุ่มขั้นที่ 1 ก็กลับหน้าเลือกโปรเจกต์ได้เหมือนกัน
page.step_buttons[0].click()
pump(0.1)
assert page.project is None and len(rows()) == 2

# ออกไม่ได้ระหว่างสอน ต้องบอกเหตุผลและอยู่หน้าเดิม
rows()[0].click()
pump(0.1)
opened, step = page.project, page._current_step
page._train_thread = object()
dialogs.seen.clear()
page.change_btn.click()
pump(0.3)
assert page.project is opened and dialogs.seen and page.step_buttons[step].isChecked(), "blocked during training"
page.step_buttons[0].click()
pump(0.3)
assert page.project is opened and page._current_step == step and page.step_buttons[step].isChecked()
page._train_thread = None
page.change_btn.click()
pump(0.1)

# ตอบ "ไม่" ต้องไม่ลบอะไรเลย
buttons = delete_buttons()
assert len(buttons) == 2
by_name = {p.name: p for p in tp.list_projects()}
target = by_name["รถสีส้ม"]
index = [p.name for p in tp.list_projects()].index("รถสีส้ม")
dialogs.answer = QtWidgets.QMessageBox.No
dialogs.seen.clear()
buttons[index].click()
pump(0.3)
assert dialogs.seen and "กู้คืนไม่ได้" in dialogs.seen[0] and "โมเดล" in dialogs.seen[0]
assert target.path.exists() and old_model.exists() and len(rows()) == 2

# ตอบ "ใช่": โฟลเดอร์ โมเดล และค่าในไฟล์ registry หายไป โปรเจกต์อื่นอยู่ครบ
dialogs.answer = QtWidgets.QMessageBox.Yes
delete_buttons()[index].click()
pump(0.3)
assert not target.path.exists() and not old_model.exists()
registry = json.loads((config.MODEL_DIR / "model_confidence.json").read_text(encoding="utf-8"))
assert old.model not in registry
assert len(rows()) == 1 and new.path.exists()

# ลบอันสุดท้าย: การ์ดรายการซ่อนไป ใช้งานต่อได้
delete_buttons()[0].click()
pump(0.3)
assert not new.path.exists() and not page.projects_card.isVisible() and tp.list_projects() == []
page.name_input.setText("กรวย")
page._create_project()
assert page.project.name == "กรวย"

page.close()
shutil.rmtree(work, ignore_errors=True)
print("ALL OK")
