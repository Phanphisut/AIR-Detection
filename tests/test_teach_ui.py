"""ทดสอบหน้า "สอน AI" ตั้งแต่อัปโหลดคลิปถึงทดสอบโมเดล ด้วยการกดปุ่มจริง (ไม่แสดงหน้าจอ) รวมการสอนจริงบนเครื่อง

python tests/test_teach_ui.py [โฟลเดอร์เก็บภาพหน้าจอ] [quick|normal|detailed]
"""
import faulthandler
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
shots = Path(sys.argv[1]) if len(sys.argv) > 1 else None
preset = sys.argv[2] if len(sys.argv) > 2 else "quick"

from PySide6 import QtCore, QtTest, QtWidgets  # noqa: E402

app = QtWidgets.QApplication(sys.argv)
app.setStyle("Fusion")

import make_sample_clip  # noqa: E402
from ai_detection_lab import config  # noqa: E402
from ai_detection_lab.teach import project as tp  # noqa: E402
from ai_detection_lab.teach.page import TeachPage  # noqa: E402

work = Path(tempfile.mkdtemp(prefix="teach_ui_"))
tp.TRAINING_ROOT = work / "sessions"
tp.TRAINING_ROOT.mkdir()
config.MODEL_DIR = work / "models"  # ไม่แตะคลังโมเดลจริงของผู้ใช้


class AutoAnswer(QtCore.QObject):
    """ตอบกล่องข้อความอัตโนมัติ (ตอบ Yes) และบันทึกไว้ตรวจ"""

    def __init__(self):
        super().__init__()
        self.seen = []
        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.tick)
        timer.start(100)

    def tick(self):
        for w in app.topLevelWidgets():
            if isinstance(w, QtWidgets.QMessageBox) and w.isVisible():
                self.seen.append(w.text())
                btn = w.button(QtWidgets.QMessageBox.Yes) or w.button(QtWidgets.QMessageBox.Ok)
                (btn or w.defaultButton()).click()


answers = AutoAnswer()


def pump(seconds=0.05):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.01)


def wait(cond, timeout, what):
    t = time.time()
    while not cond():
        pump(0.05)
        if time.time() - t > timeout:
            raise TimeoutError(what)
    return time.time() - t


def shot(page, name):
    if shots:
        page.grab().save(str(shots / f"teach_{name}.png"))


truth_clip = work / "คลิปลูกบอล.mp4"          # ชื่อไทย: ต้องใช้ได้
truth = make_sample_clip.make_clip(truth_clip, seconds=40, w=960, h=540)
other_clip = work / "other.mp4"
make_sample_clip.make_clip(other_clip, seconds=6, w=640, h=360, seed=7)

page = TeachPage()
page.resize(1400, 860)
page.show()
pump(0.3)
shot(page, "1_name")

# ขั้น 1: ชื่อว่างไม่ผ่าน
page._create_project()
assert page.project is None and page.name_error.text()
page.name_input.setText("ลูกบอล")
page._create_project()
assert page.project is not None and page.stack.currentIndex() == 1
project = page.project

# ขั้น 2: ไฟล์ที่ไม่ใช่วิดีโอถูกปฏิเสธ แล้วอัปโหลดคลิปจริง
bad = work / "note.txt"
bad.write_text("x")
page._add_clips([str(bad)])
pump(0.3)
assert not project.clips and answers.seen, "non-video should be rejected"
page._add_clips([str(truth_clip)])
waited = wait(lambda: project.clips and page._extract is None, 120, "extract")
pump(0.1)
print(f"extract: {len(project.frames)} frames in {waited:.1f}s; warnings={project.clips[0]['warnings']}")
assert project.clips[0]["name"] == truth_clip.name
assert page.upload_next.isEnabled()
shot(page, "2_upload")

# ขั้น 3: วาดกรอบตามตำแหน่งจริง จนไม่เหลือรูปที่ต้องทำ
page._go(2)
pump(0.2)
assert page.stack.currentIndex() == 2 and page._frame is not None
seen_before = len(answers.seen)
page._finish_draw()  # ยังไม่มีกรอบ ต้องไม่ผ่าน
pump(0.3)
assert page.stack.currentIndex() == 2 and len(answers.seen) > seen_before
draws = nones = 0
t0 = time.time()
while project.todo_count():
    frame = page._frame
    if frame["state"] != "unset":
        page._goto_todo()
        pump(0.02)
        frame = page._frame
    box = truth[frame["src"]]
    if box[2] < 0.97 and box[0] < 0.9:  # เห็นของชัด -> วาดกรอบ
        page.draw_view.box_drawn.emit(list(box))
        draws += 1
        assert page._frame is frame, "drawing must stay on this frame so the box can be adjusted"
        if page.next_btn.isEnabled():
            page.next_btn.click()
    else:                               # ของออกนอกจอ -> ไม่มีของ
        page._mark_none()
        nones += 1
    assert frame["state"] in ("drawn", "none")
    pump(0.02)
print(f"draw: {draws} boxes drawn, {nones} no-object clicks for {len(project.frames)} frames "
      f"in {time.time()-t0:.1f}s")
assert draws + nones == len(project.frames), "every frame is decided by hand"
assert draws == project.labeled_count()

# ลากด้วยเมาส์จริง: วาดใหม่, ปรับขนาดที่มุม, ย้าย, และลากจนเล็กเกินต้องกลับค่าเดิม
target = next(f for f in project.frames if f["state"] == "drawn")
page._show_frame(target)
pump(0.1)
view = page.draw_view


def to_px(x, y):
    r = view._image_rect()
    return QtCore.QPoint(round(r.x() + x * r.width()), round(r.y() + y * r.height()))


def drag(a, b):
    QtTest.QTest.mousePress(view, QtCore.Qt.LeftButton, pos=to_px(*a))
    QtTest.QTest.mouseMove(view, to_px((a[0] + b[0]) / 2, (a[1] + b[1]) / 2))
    QtTest.QTest.mouseMove(view, to_px(*b))
    QtTest.QTest.mouseRelease(view, QtCore.Qt.LeftButton, pos=to_px(*b))
    pump(0.05)


def near(box, want):
    assert all(abs(p - q) < 0.01 for p, q in zip(box, want)), (box, want)


drag((0.1, 0.1), (0.3, 0.5))
near(target["boxes"][0], [0.1, 0.1, 0.3, 0.5])
assert page._frame is target, "stays on this frame after drawing"
drag((0.3, 0.5), (0.6, 0.7))            # มุมขวาล่าง: ขยาย รถเปลี่ยนจากแนวตั้งเป็นแนวนอนก็ปรับได้
near(target["boxes"][0], [0.1, 0.1, 0.6, 0.7])
drag((0.1, 0.4), (0.2, 0.4))            # ขอบซ้าย
near(target["boxes"][0], [0.2, 0.1, 0.6, 0.7])
drag((0.4, 0.4), (0.5, 0.3))            # กลางกรอบ: ย้าย ขนาดเท่าเดิม
near(target["boxes"][0], [0.3, 0.0, 0.7, 0.6])
drag((0.5, 0.3), (0.9, 0.9))            # ย้ายชนขอบ: ต้องไม่หลุดจากรูป
near(target["boxes"][0], [0.6, 0.4, 1.0, 1.0])
before = list(target["boxes"][0])
drag((0.6, 0.7), (1.0, 0.7))            # บีบจนแทบไม่เหลือ: กลับค่าเดิม
near(target["boxes"][0], before)
assert page._frame is target
print("mouse: draw, resize, move and undersize guard ok")
saved = tp.TeachProject(project.path)
near(next(f for f in saved.frames if f["file"] == target["file"])["boxes"][0], before)
project.set_box(target, truth[target["src"]])
project.save()

page._go(1)
pump(0.2)
thumbs = page.findChildren(QtWidgets.QLabel, "thumb")
assert any(t.pixmap() and not t.pixmap().isNull() for t in thumbs), "clip row shows a frame from the clip"
shot(page, "2_upload_done")
page._go(2)
pump(0.2)
page._show_frame(project.frames[len(project.frames) // 3])
pump(0.2)
shot(page, "3_done")

# ขั้น 4: สอนจริงบน CPU
page._finish_draw()
pump(0.2)
assert page.stack.currentIndex() == 3 and page.teach_stack.currentIndex() == 0
page._select_preset(preset)
shot(page, "4_setup")
t0 = time.time()
page._start_training()
pump(0.5)
assert page._train_thread is not None
shot_taken = False
while page._train_thread is not None:
    pump(0.5)
    if not shot_taken and page.preview_label.text() and page.train_bar.value() > 2:
        shot(page, "4_training")
        shot_taken = True
    if time.time() - t0 > 900:
        raise TimeoutError("training")
elapsed = time.time() - t0
print(f"train[{preset}]: {elapsed:.0f}s, epochs={page.train_bar.maximum()}, "
      f"last preview={page.preview_label.text()!r}")
assert project.model and (config.MODEL_DIR / project.model).is_file(), "model installed"
assert project.model.startswith("teach_") and project.model.endswith("_ลูกบอล.pt"), project.model
print("model file:", project.model)
print("registry:", (config.MODEL_DIR / "model_confidence.json").read_text(encoding="utf-8")[:200].replace("\n", " "))

# ทดสอบกับช่วงท้ายคลิป
wait(lambda: page._test_worker is None, 300, "test")
pump(0.3)
print("result:", page.score_label.text(), "|", page.score_sub.text(), "|", project.last_result)
assert project.last_result and project.last_result["frames"] >= 1
assert project.last_result["score"] >= 80, "the model finds the ball in the held-out part of this easy clip"
shot(page, "4_result")

# ทดสอบกับคลิปอื่น แล้วบอก AI ว่ากรอบผิด -> รูปถูกเก็บเข้าชุดสอน
page._start_external_test(str(other_clip))
pump(0.2)
wait(lambda: page._test_worker is None, 300, "external test")
pump(0.3)
print("external:", page.score_label.text(), f"results={len(page._results)}")
hit = next((i for i, r in enumerate(page._results) if r["dets"]), None)
if hit is not None:
    page._show_result(hit)
    pump(0.1)
    before = len(project.frames)
    page.test_view.box_clicked.emit(0)
    pump(0.5)
    assert len(project.frames) == before + 1 and project.frames[-1]["correction"], "external correction saved"
    plan = project.split_plan()
    assert project.frames[-1] in plan["train"]
    print("correction saved; train images now", len(plan["train"]))
    shot(page, "4_corrected")
else:
    print("external clip: no detections to correct")

# เปิดโปรเจกต์ใหม่จากไฟล์ได้ และสถานะครบ
again = tp.TeachProject(project.path)
assert again.model == project.model and again.labeled_count() == project.labeled_count()
page.close()
shutil.rmtree(work, ignore_errors=True)
print("ALL OK")
