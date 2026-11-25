"""
workers.py
==========
งานหนักของหน้า "สอน AI" ที่ต้องทำนอก GUI thread: ตัดคลิป, ทดสอบโมเดล และตัวเลือกความเร็วการสอน
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import cv2
from PySide6 import QtCore

from ai_detection_lab import config
from ai_detection_lab.image_io import imread_any

from .clips import extract_frames
from .project import box_iou

FOUND_IOU = 0.3             # กรอบ AI ทับกรอบจริงเท่านี้ถือว่า "เจอ"
EXTERNAL_TEST_FPS = 5       # ทดสอบกับคลิปอื่น: ดูกี่รูปต่อวินาที
PREVIEW_EVERY_SECONDS = 40.0  # พรีวิวระหว่างสอน: วัดจริงบน CPU ถ้าถี่กว่านี้กินเวลาสอนมาก

# batch คงที่ + นับจำนวนก้าวแทน epoch: จำนวนรอบจึงคำนวณจากจำนวนรูปในโปรเจกต์
# warm-up ของ ultralytics ยาวอย่างน้อย 100 ก้าว (ดู TrainAIThread.run) แบบเร็วมีก้าวน้อยจึงปิด warm-up ไม่งั้นแทบไม่ได้เรียน
# วัดบน CPU 4 thread กับ 40 รูป: ค่าเดิม (mosaic, ไม่ freeze, 200 ก้าว) 504 วินาที -> ค่า quick นี้ 161 วินาที
# บนคลิปทดสอบหาเจอเท่าเดิม แต่ไม่มี mosaic/freeze อาจแม่นน้อยลงกับคลิปจริงที่ยากกว่า แบบละเอียดจึงยังใช้ค่าเต็ม
FAST = dict(warmup_epochs=0, mosaic=0.0, val=False)
# จำนวนรอบคงที่ตามที่ผู้ใช้กำหนด (2026-11-10): เร็ว 20 / ปกติ 40 / ละเอียด 60 รอบ ไม่ขึ้นกับจำนวนรูป
PRESETS = {
    "quick": dict(title="Quick", imgsz=320, batch=4, epochs=20, overrides=dict(FAST, freeze=10)),
    "normal": dict(title="Normal", imgsz=416, batch=6, epochs=40, overrides=dict(FAST)),
    "detailed": dict(title="Detailed", imgsz=640, batch=8, epochs=60, overrides={}),
}


def epochs_for(preset, train_images=None):
    return int(preset["epochs"])


class ExtractWorker(QtCore.QThread):
    progress = QtCore.Signal(int, int)
    finished_clip = QtCore.Signal(object, object, object)  # clip_id, dest, result
    failed = QtCore.Signal(str)

    def __init__(self, project, source):
        super().__init__()
        self.project = project
        self.source = Path(source)
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        dest = None
        try:
            clip_id, dest = self.project.copy_clip_in(self.source)
            result = extract_frames(dest, self.project.frames_dir, self.project.next_index(), clip_id,
                                    on_progress=self.progress.emit, should_stop=lambda: self._stop)
            self.finished_clip.emit(clip_id, dest, result)
        except InterruptedError:
            if dest is not None:
                dest.unlink(missing_ok=True)
        except (ValueError, OSError) as exc:
            if dest is not None:
                dest.unlink(missing_ok=True)
            self.failed.emit(str(exc))


def _detect(model, image, conf, imgsz=None):
    """รูป BGR -> รายการ dict(box=[x1,y1,x2,y2 แบบ 0-1], conf)"""
    from ai_detection_lab.ai import devices

    device, _ = devices.resolve("auto")
    result = model.predict(image, conf=conf, imgsz=imgsz or config.YOLO_IMGSZ, verbose=False, device=device)[0]
    h, w = image.shape[:2]
    out = []
    if result.boxes is not None:
        for box, score in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
            out.append(dict(box=[box[0] / w, box[1] / h, box[2] / w, box[3] / h], conf=float(score)))
    return out


class TestWorker(QtCore.QThread):
    """ทดสอบโมเดลทีละรูป ส่งผลกลับมาเล่นเหมือนวิดีโอ

    source = รายการรูปในชุดทดสอบของโปรเจกต์ หรือ path คลิปอื่น (ไม่มีเฉลย จึงนับคะแนนไม่ได้)
    ใช้ความมั่นใจและขนาดภาพชุดเดียวกับตอนบินจริง คะแนนจึงตรงกับที่จะเจอจริง
    """
    item = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, model_path, conf, imgsz, frames=None, frames_dir=None, video=None):
        super().__init__()
        self.model_path, self.conf, self.imgsz = str(model_path), conf, imgsz
        self.frames, self.frames_dir, self.video = frames, frames_dir, video
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            from ultralytics import YOLO

            model = YOLO(self.model_path)
            if self.video is not None:
                self._run_video(model)
            else:
                self._run_frames(model)
        except Exception as exc:  # ไม่ให้โปรแกรมปิด: โมเดลเสีย/ไฟล์หาย แจ้งเด็กแทน
            self.failed.emit(f"ทดสอบไม่สำเร็จ: {exc}")

    def _run_frames(self, model):
        for frame in self.frames:
            if self._stop:
                return
            image = imread_any(Path(self.frames_dir) / frame["file"])
            if image is None:
                continue
            dets = _detect(model, image, self.conf, self.imgsz)
            truth = frame["boxes"]
            found = any(box_iou(d["box"], t) >= FOUND_IOU for d in dets for t in truth) if truth else None
            wrong = sum(1 for d in dets if not any(box_iou(d["box"], t) >= FOUND_IOU for t in truth))
            self.item.emit(dict(image=image, dets=dets, truth=truth, found=found, wrong=wrong, frame=frame))

    def _run_video(self, model):
        cap = cv2.VideoCapture(str(self.video))
        if not cap.isOpened():
            self.failed.emit("เปิดคลิปนี้ไม่ได้ ลองแปลงเป็นไฟล์ .mp4")
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, round(fps / EXTERNAL_TEST_FPS))
        index, limit = 0, int(60 * fps)
        while index < limit and not self._stop:
            if index % step != 0:
                if not cap.grab():
                    break
                index += 1
                continue
            ok, image = cap.read()
            if not ok:
                break
            dets = _detect(model, image, self.conf, self.imgsz)
            self.item.emit(dict(image=image, dets=dets, truth=None, found=bool(dets) if dets else None,
                                wrong=0, frame=None))
            index += 1
        cap.release()


def copy_external_clip(project, source):
    """คลิปทดสอบคัดลอกมาที่ชื่ออังกฤษก่อน: VideoCapture ไม่ถูกกับ path ภาษาไทยบน Windows"""
    source = Path(source)
    dest = project.path / "clips" / f"_external_test{source.suffix.lower()}"
    shutil.copy2(source, dest)
    return dest


def make_preview_callback(sample_image, truth, emit, imgsz):
    """callback ของ ultralytics: หลังจบบางรอบ ลองให้ AI ทายรูปตัวอย่าง ส่งผลไปโชว์ (เด็กจะเห็นว่ามันเก่งขึ้นจริง)"""
    state = dict(last=0.0)

    def on_model_save(trainer):
        now = time.monotonic()
        if now - state["last"] < PREVIEW_EVERY_SECONDS:
            return
        state["last"] = now
        try:
            from ultralytics import YOLO

            model = YOLO(str(trainer.last))
            dets = _detect(model, sample_image, 0.25, imgsz)
            best = max(dets, key=lambda d: d["conf"], default=None)
            iou = box_iou(best["box"], truth) if best else 0.0
            emit(int(trainer.epoch) + 1, best, iou)
        except Exception as exc:  # พรีวิวเป็นของแถม ห้ามทำให้การสอนล้ม
            print(f"preview skipped: {exc}")

    return on_model_save
