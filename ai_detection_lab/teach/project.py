"""
project.py
==========
โปรเจกต์ "สอน AI" 1 โปรเจกต์ = สอนของ 1 อย่าง เก็บที่ TRAINING_ROOT/<id>/ ในรูปแบบเดียวกับ Label & Train
(frames/*.jpg + .txt + .noobject + classes.txt) ส่วนสถานะของแต่ละรูปอยู่ใน teach.json

สถานะของรูป (เด็กตัดสินเองทุกรูป ไม่มีการเดากรอบให้):
  unset       ยังไม่ได้ทำ
  drawn       เด็กวาดกรอบแล้ว
  none        ไม่มีของในรูปนี้ (ใช้เป็นตัวอย่าง "ไม่ใช่")
  correction  รูปจากคลิปอื่นที่เด็กบอกว่า AI ทักผิด

การแบ่งชุดใช้ "ช่วงเวลา" ไม่สุ่มทีละรูป เพราะรูปจากวิดีโอที่ติดกันแทบเหมือนกัน ถ้าสุ่มคะแนนจะสูงเกินจริง
"""

from __future__ import annotations

import datetime
import json
import math
import re
import shutil
from pathlib import Path

from ai_detection_lab import config
from ai_detection_lab.ui.label_train_dialog import (
    MAX_BACKGROUND_RATIO,
    MODEL_CONF_REGISTRY_FILENAME,
    TRAINING_ROOT,
    label_path_for,
    next_frame_index,
    no_object_path_for,
    write_classes,
    write_yolo_labels,
)

from .clips import MAX_CLIPS

META_FILE = "teach.json"
MIN_LABELED = 10            # มีกรอบน้อยกว่านี้ยังไม่เริ่มสอน
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
SPLIT_GAP_SECONDS = 0.6     # เว้นรูปรอยต่อระหว่างชุด ไม่ให้รูปเกือบเหมือนกันอยู่คนละชุด
DEFAULT_CONF = 0.4
READY_STATES = ("drawn", "none")


def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def list_projects():
    """โปรเจกต์สอน AI ทั้งหมด ล่าสุดก่อน"""
    if not TRAINING_ROOT.is_dir():
        return []
    found = []
    for folder in TRAINING_ROOT.iterdir():
        if (folder / META_FILE).is_file():
            try:
                found.append(TeachProject(folder))
            except (OSError, ValueError, KeyError):
                continue  # ไฟล์เสีย ข้ามไป ไม่ให้ทั้งหน้าเปิดไม่ได้
    return sorted(found, key=lambda p: p.path.stat().st_mtime, reverse=True)


def _load_registry():
    path = config.MODEL_DIR / MODEL_CONF_REGISTRY_FILENAME
    try:
        registry = json.loads(path.read_text(encoding="utf-8"))
        return path, registry if isinstance(registry, dict) else {}
    except (OSError, ValueError):
        return path, {}


class TeachProject:
    def __init__(self, path):
        self.path = Path(path)
        data = json.loads((self.path / META_FILE).read_text(encoding="utf-8"))
        self.name = data["name"]
        self.clips = data["clips"]
        self.frames = data["frames"]
        self._upgrade_old_frames()
        self.model = data.get("model")  # ชื่อไฟล์โมเดลล่าสุดใน config.MODEL_DIR
        self.version = data.get("version", 0)
        self.last_result = data.get("last_result")

    def _upgrade_old_frames(self):
        # โปรเจกต์จากรุ่นที่มีตัวติดตามกรอบ: เก็บเฉพาะกรอบที่เด็กวาดเอง กรอบที่ไหลตามมาเองเชื่อไม่ได้
        # (กรอบไม่เปลี่ยนรูปตามของ เช่น รถหมุนจากแนวตั้งเป็นแนวนอน) ถ้าปล่อยไว้จะสอน AI ด้วยกรอบผิด
        for frame in self.frames:
            state = frame["state"]
            if state == "anchor":
                frame["state"] = "drawn"
            elif state in ("tracked", "lost"):
                frame["state"] = "unset"
                frame["boxes"] = []

    @classmethod
    def create(cls, name):
        name = name.strip()
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # ชื่อโฟลเดอร์ใช้ตัวอังกฤษเท่านั้น: ไลบรารีบางตัวบน Windows ไม่ถูกกับ path ภาษาไทย
        slug = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")[:30]
        base = f"teach_{slug}_{stamp}" if slug else f"teach_{stamp}"
        folder, n = TRAINING_ROOT / base, 1
        while folder.exists():  # สร้าง 2 โปรเจกต์ในวินาทีเดียวกัน (กดรัว) ต้องไม่เขียนทับกัน
            n += 1
            folder = TRAINING_ROOT / f"{base}_{n}"
        (folder / "frames").mkdir(parents=True)
        (folder / "clips").mkdir(exist_ok=True)
        write_classes(folder / "classes.txt", [name])
        (folder / META_FILE).write_text(
            json.dumps(dict(name=name, clips=[], frames=[]), ensure_ascii=False), encoding="utf-8")
        return cls(folder)

    def save(self):
        data = dict(name=self.name, clips=self.clips, frames=self.frames, model=self.model,
                    version=self.version, last_result=self.last_result)
        tmp = self.path / (META_FILE + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.path / META_FILE)  # ไฟดับกลางคันก็ไม่ทำให้ไฟล์เดิมเสีย

    # ------------------------------------------------------------
    # คลิป
    # ------------------------------------------------------------
    @property
    def frames_dir(self):
        return self.path / "frames"

    def can_add_clip(self):
        return len(self.clips) < MAX_CLIPS

    def copy_clip_in(self, source):
        """คัดลอกคลิปเข้าโปรเจกต์ (ชื่ออังกฤษ: VideoCapture ไม่ถูกกับ path ภาษาไทย + ไฟล์ต้นทางอาจถูกย้าย)"""
        source = Path(source)
        clip_id = 1 + max([c["id"] for c in self.clips], default=-1)
        dest = self.path / "clips" / f"clip_{clip_id}{source.suffix.lower()}"
        shutil.copy2(source, dest)
        return clip_id, dest

    def add_clip(self, clip_id, source_name, dest, result):
        self.clips.append(dict(id=clip_id, name=source_name, file=dest.name, **result["info"]))
        self.frames.extend(result["frames"])
        self.save()

    def remove_clip(self, clip_id):
        for frame in [f for f in self.frames if f["clip"] == clip_id]:
            image = self.frames_dir / frame["file"]
            for path in (image, label_path_for(image), no_object_path_for(image)):
                path.unlink(missing_ok=True)
        self.frames = [f for f in self.frames if f["clip"] != clip_id]
        for clip in [c for c in self.clips if c["id"] == clip_id]:
            (self.path / "clips" / clip["file"]).unlink(missing_ok=True)
        self.clips = [c for c in self.clips if c["id"] != clip_id]
        self.save()

    def clip_path(self, clip):
        return self.path / "clips" / clip["file"]

    def clip_frames(self, clip_id):
        return [f for f in self.frames if f["clip"] == clip_id]

    def next_index(self):
        return next_frame_index(self.frames_dir)

    # ------------------------------------------------------------
    # กรอบ
    # ------------------------------------------------------------
    def set_box(self, frame, box):
        frame["state"] = "drawn"
        frame["boxes"] = [list(box)]
        frame["correction"] = False

    def mark_none(self, frame):
        frame["state"] = "none"
        frame["boxes"] = []

    def todo_count(self):
        return sum(1 for f in self.frames if f["state"] == "unset")

    def labeled_count(self):
        return sum(1 for f in self.frames if f["state"] == "drawn")

    def next_todo(self, after=None):
        """รูปถัดไปที่ยังต้องทำ (วนกลับต้นได้)"""
        if not self.frames:
            return None
        start = 0 if after is None else self.frames.index(after) + 1
        for frame in self.frames[start:] + self.frames[:start]:
            if frame["state"] == "unset":
                return frame
        return None

    # ------------------------------------------------------------
    # แบ่งชุดและสร้าง dataset
    # ------------------------------------------------------------
    def split_plan(self):
        """คืน dict train/val/test -> รายการรูป: ต้นคลิปสอน ท้ายคลิปตรวจ/ทดสอบ แบ่งแยกกันทีละคลิป"""
        plan = dict(train=[], val=[], test=[])
        for clip in self.clips:
            frames = self.clip_frames(clip["id"])
            usable = [f for f in frames if f["state"] in READY_STATES and not f["correction"]]
            labeled = [f for f in usable if f["boxes"]]
            n = len(labeled)
            n_test = math.ceil(n * TEST_FRACTION) if n >= 10 else 0
            n_val = math.ceil(n * VAL_FRACTION) if n >= 7 else 0
            if n_test == 0 and n_val == 0:
                plan["train"].extend(usable)
                continue
            t_test = labeled[n - n_test]["t"] if n_test else math.inf
            t_val = labeled[n - n_test - n_val]["t"] if n_val else t_test
            for frame in labeled:
                t = frame["t"]
                if t >= t_test:
                    plan["test"].append(frame)
                elif t >= t_val:
                    if t <= t_test - SPLIT_GAP_SECONDS:
                        plan["val"].append(frame)
                elif t <= t_val - SPLIT_GAP_SECONDS:
                    plan["train"].append(frame)
            # รูปไม่มีของแบ่งตามลำดับของมันเอง: ถ้าใช้จุดตัดเดียวกับรูปมีของ ช่วงที่ของออกนอกจอนาน ๆ
            # จะตกไปอยู่ชุดตรวจทั้งหมด ชุดสอนเลยไม่มีตัวอย่าง "ไม่ใช่" เลย
            empty = [f for f in usable if not f["boxes"]]
            cut_val = len(empty) - round(len(empty) * (VAL_FRACTION + TEST_FRACTION))
            cut_test = len(empty) - round(len(empty) * TEST_FRACTION)
            plan["train"].extend(empty[:cut_val])
            plan["val"].extend(empty[cut_val:cut_test])
            plan["test"].extend(empty[cut_test:])
        plan["train"].extend(f for f in self.frames if f["correction"])
        self._borrow_if_empty(plan)
        return {name: self._cap_backgrounds(items) for name, items in plan.items()}

    @staticmethod
    def _borrow_if_empty(plan):
        # คลิปสั้นมากหลายคลิป: ยังต้องมีชุดตรวจอย่างน้อย 1 รูป (ultralytics ต้องการ) ยืมท้ายชุดสอนมา
        for name in ("val", "test"):
            if not any(f["boxes"] for f in plan[name]):
                labeled = [f for f in plan["train"] if f["boxes"] and not f["correction"]]
                if len(labeled) > 4:
                    plan[name].append(labeled[-1])
                    plan["train"].remove(labeled[-1])

    @staticmethod
    def _cap_backgrounds(frames):
        # รูปไม่มีของมากเกินไปทำให้ AI ไม่กล้าตอบ (ดู MAX_BACKGROUND_RATIO) แต่รูปที่เด็กแก้เองต้องเก็บไว้เสมอ
        labeled = [f for f in frames if f["boxes"]]
        corrected = [f for f in frames if not f["boxes"] and f["correction"]]
        others = [f for f in frames if not f["boxes"] and not f["correction"]]
        room = max(0, int(len(labeled) * MAX_BACKGROUND_RATIO) - len(corrected))
        if room < len(others):  # เกลี่ยตามเวลา ไม่เอาแค่ต้นคลิป
            step = len(others) / max(1, room)
            others = [others[int(i * step)] for i in range(room)]
        return labeled + corrected + others

    def write_labels(self):
        """เขียน .txt / .noobject ข้างรูป และลบของเก่าที่ไม่ใช้แล้ว"""
        for frame in self.frames:
            image = self.frames_dir / frame["file"]
            label, marker = label_path_for(image), no_object_path_for(image)
            label.unlink(missing_ok=True)
            marker.unlink(missing_ok=True)
            if frame["state"] in ("drawn", "correction") and frame["boxes"]:
                rows = []
                for x1, y1, x2, y2 in frame["boxes"]:
                    rows.append(dict(class_id=0, cx=(x1 + x2) / 2, cy=(y1 + y2) / 2, w=x2 - x1, h=y2 - y1))
                write_yolo_labels(label, rows)
            elif frame["state"] in ("none", "correction"):
                marker.write_text("checked: no object in this photo\n", encoding="utf-8")

    def build_dataset(self, dataset_dir, plan=None):
        """สร้างโฟลเดอร์ dataset ตาม split_plan (คืนค่าแบบเดียวกับ build_dataset_split ของ TrainAIThread)"""
        plan = plan or self.split_plan()
        labeled_total = sum(1 for split in plan.values() for f in split if f["boxes"])
        if labeled_total < MIN_LABELED:
            raise ValueError(f"มีรูปที่มีกรอบ {labeled_total} รูป ต้องมีอย่างน้อย {MIN_LABELED} รูป")
        self.write_labels()
        dataset_dir = Path(dataset_dir)
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir, ignore_errors=True)
        splits = [name for name in ("train", "val", "test") if plan[name]]
        for name in splits:
            (dataset_dir / "images" / name).mkdir(parents=True, exist_ok=True)
            (dataset_dir / "labels" / name).mkdir(parents=True, exist_ok=True)
            for frame in plan[name]:
                image = self.frames_dir / frame["file"]
                shutil.copy2(image, dataset_dir / "images" / name / image.name)
                dest = dataset_dir / "labels" / name / label_path_for(image).name
                if frame["boxes"]:
                    shutil.copy2(label_path_for(image), dest)
                else:
                    dest.write_text("", encoding="utf-8")
        # ชื่อภาษาไทยใส่เครื่องหมายคำพูด (json.dumps เป็น YAML ที่ถูกต้อง)
        yaml_text = (
            f"path: {dataset_dir.as_posix()}\ntrain: images/train\nval: images/val\n"
            + ("test: images/test\n" if "test" in splits else "")
            + f"names:\n  0: {json.dumps(self.name, ensure_ascii=False)}\n"
        )
        data_yaml = dataset_dir / "data.yaml"
        data_yaml.write_text(yaml_text, encoding="utf-8")
        backgrounds = [f["file"] for split in plan.values() for f in split if not f["boxes"]]
        return dict(
            train=len(plan["train"]), val=len(plan["val"]), test=len(plan["test"]),
            backgrounds=len(backgrounds), backgrounds_available=len(backgrounds), background_names=backgrounds,
            total_labeled=labeled_total, total_frames=len(self.frames), data_yaml=data_yaml,
        )

    # ------------------------------------------------------------
    # ทักผิด
    # ------------------------------------------------------------
    def add_correction(self, frame, keep_boxes):
        """เด็กบอกว่ากรอบที่ AI วาด "ไม่ใช่": ให้รูปนี้ไปอยู่ชุดสอน

        รูปจากคลิปในโปรเจกต์มีกรอบจริงอยู่แล้ว (พื้นที่นอกกรอบคือ "ไม่ใช่" อยู่แล้ว) จึงแค่ย้ายชุด
        ส่วนรูปจากคลิปอื่น กรอบที่ AI วาดแล้วเด็กไม่ได้แย้งถือว่าถูก ถ้าไม่มีเลยเป็นรูป "ไม่มีของ"
        """
        if frame["state"] not in READY_STATES:
            frame["state"] = "correction"
            frame["boxes"] = [list(b) for b in keep_boxes]
        frame["correction"] = True
        self.save()

    def add_external_correction(self, image, keep_boxes):
        """รูปจากคลิปทดสอบภายนอก: บันทึกเข้าโปรเจกต์"""
        from ai_detection_lab.image_io import imwrite_any

        name = f"frame_{self.next_index():06d}.jpg"
        if not imwrite_any(self.frames_dir / name, image):
            raise ValueError("บันทึกรูปไม่สำเร็จ")
        frame = dict(file=name, clip=-1, src=0, t=0.0, state="correction",
                     boxes=[list(b) for b in keep_boxes], correction=True)
        self.frames.append(frame)
        self.save()
        return frame

    # ------------------------------------------------------------
    # โมเดล
    # ------------------------------------------------------------
    def install_model(self, best_path, calibration, imgsz):
        """คัดลอกโมเดลเข้า MODEL_DIR ชื่อ teach_<วันที่>_<ชื่อของ>.pt แทนรุ่นก่อนของโปรเจกต์นี้
        สอนซ้ำจะเติม _v2, _v3 เพราะตัวตรวจจับที่เปิดอยู่จำชื่อไฟล์ ถ้าใช้ชื่อเดิมจะไม่โหลดรุ่นใหม่
        ถ้าเก็บทุกรุ่นไว้ โหมดอัตโนมัติจะรันทุกรุ่นพร้อมกัน (ช้าและกรอบซ้อน) จึงลบรุ่นเก่าของโปรเจกต์นี้ทิ้ง"""
        config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        filename = self._next_model_filename()
        shutil.copy2(best_path, config.MODEL_DIR / filename)
        registry_path, registry = _load_registry()
        if self.model:
            registry.pop(self.model, None)
            (config.MODEL_DIR / self.model).unlink(missing_ok=True)
        # ถ้าวัดความมั่นใจไม่ได้ ใช้ค่ากลาง: 0.8 ของ config เข้มเกินไปสำหรับชุดรูปเล็ก
        registry[filename] = dict(calibration or {"conf_thres": DEFAULT_CONF}, imgsz=imgsz)
        registry_path.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")
        self.model = filename
        self.save()
        return config.MODEL_DIR / filename

    def delete(self):
        """ลบโปรเจกต์ถาวร: คลิป รูป กรอบ และโมเดลของโปรเจกต์นี้ — ลบโฟลเดอร์ก่อน ถ้าไฟล์ถูกใช้อยู่จะยก OSError โดยโมเดลยังอยู่"""
        shutil.rmtree(self.path)
        if self.model:
            registry_path, registry = _load_registry()
            if registry.pop(self.model, None) is not None:
                registry_path.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")
            (config.MODEL_DIR / self.model).unlink(missing_ok=True)

    def _next_model_filename(self):
        date = datetime.datetime.now().strftime("%Y%m%d")
        safe = re.sub(r'[<>:"/\\|?*\s]+', "_", self.name).strip("._") or "model"
        while True:
            self.version += 1
            suffix = "" if self.version == 1 else f"_v{self.version}"
            filename = f"teach_{date}_{safe}{suffix}.pt"
            # ชื่อซ้ำได้เมื่อมีโปรเจกต์ชื่อเดียวกันในวันเดียวกัน ห้ามเขียนทับโมเดลของโปรเจกต์อื่น
            if not (config.MODEL_DIR / filename).exists():
                return filename

    def model_settings(self):
        """(ความมั่นใจ, ขนาดภาพ) ชุดเดียวกับที่ตัวตรวจจับใช้ตอนบินจริง คะแนนทดสอบจึงตรงกับของจริง"""
        try:
            registry = json.loads((config.MODEL_DIR / MODEL_CONF_REGISTRY_FILENAME).read_text(encoding="utf-8"))
            entry = registry[self.model]
            return float(entry["conf_thres"]), int(entry["imgsz"])
        except (OSError, ValueError, KeyError, TypeError):
            return DEFAULT_CONF, config.YOLO_IMGSZ
