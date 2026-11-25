"""ทดสอบส่วนที่ไม่มีหน้าจอ: ตัดคลิป, สถานะกรอบ, แปลงโปรเจกต์เก่า, แบ่งชุดตามเวลา, สร้าง dataset"""
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "vendor", Path(__file__).resolve().parent):
    sys.path.insert(0, str(p))

import make_sample_clip  # noqa: E402

work = Path(tempfile.mkdtemp(prefix="teach_core_"))
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from ai_detection_lab.teach import project as tp  # noqa: E402
from ai_detection_lab.teach.clips import extract_frames  # noqa: E402

tp.TRAINING_ROOT = work  # ไม่เขียนลงข้อมูลจริง


def visible(box):
    return box[2] > 0.02 and box[0] < 0.98


clip_src = work / "src clip.mp4"
truth = make_sample_clip.make_clip(clip_src, w=640, h=360)
project = tp.TeachProject.create("ลูกบอล")
assert project.path.name.startswith("teach_") and project.path.name.isascii(), project.path.name

t0 = time.time()
clip_id, dest = project.copy_clip_in(clip_src)
result = extract_frames(dest, project.frames_dir, project.next_index(), clip_id)
project.add_clip(clip_id, clip_src.name, dest, result)
clip = project.clips[0]
print(f"extract: {len(project.frames)} frames from {clip['duration']}s in {time.time()-t0:.1f}s, "
      f"warn={clip['warnings']} blur={clip['skipped_blur']} dup={clip['skipped_duplicate']}")
assert 15 <= len(project.frames) <= 30, "about 1 frame per second"

# สร้างหลายโปรเจกต์ในวินาทีเดียวกันต้องไม่ทับกัน
burst = [tp.TeachProject.create("ซ้ำ") for _ in range(3)]
assert len({p.path for p in burst}) == 3 and all(tp.TeachProject(p.path).name == "ซ้ำ" for p in burst)
for p in burst:
    shutil.rmtree(p.path)

# วาดกรอบรูปเดียวต้องไม่กระทบรูปอื่น: ไม่มีการเดากรอบให้
frames = project.frames
project.set_box(frames[3], truth[frames[3]["src"]])
assert frames[3]["state"] == "drawn" and all(f["state"] == "unset" for f in frames if f is not frames[3])
project.mark_none(frames[4])
assert frames[4]["state"] == "none" and frames[5]["state"] == "unset"
assert project.next_todo(frames[3]) is frames[5]

# ทำครบทุกรูป (รวมรูปที่ของออกนอกจอ)
for frame in frames:
    if frame["state"] == "unset":
        if visible(truth[frame["src"]]):
            project.set_box(frame, truth[frame["src"]])
        else:
            project.mark_none(frame)
assert project.todo_count() == 0 and project.next_todo() is None

# แบ่งชุด: เรียงตามเวลา ไม่ปนกัน และทุกชุดมีรูปที่มีกรอบ
plan = project.split_plan()
for name, items in plan.items():
    ts = [f["t"] for f in items]
    print(f"split {name}: {len(items)} frames, labeled={sum(1 for f in items if f['boxes'])}, "
          f"t={min(ts, default=0):.1f}-{max(ts, default=0):.1f}")
assert not (set(f["file"] for f in plan["train"]) & set(f["file"] for f in plan["test"]))
assert (max(f["t"] for f in plan["train"] if f["boxes"]) < min(f["t"] for f in plan["test"] if f["boxes"])), \
    "test must be after train"
assert sum(1 for f in plan["test"] if f["boxes"]) >= 1 and sum(1 for f in plan["val"] if f["boxes"]) >= 1
assert sum(1 for f in plan["train"] if not f["boxes"]) >= 1, "training needs no-object examples too"
train_labeled = sum(1 for f in plan["train"] if f["boxes"])
train_bg = sum(1 for f in plan["train"] if not f["boxes"])
assert train_bg <= train_labeled * 0.5 + 1, (train_bg, train_labeled)

# วาดไม่ครบก็ใช้ได้: รูปที่ยังไม่ทำต้องไม่เข้าชุดใดเลย
skipped = frames[10]
skipped["state"], skipped["boxes"] = "unset", []
assert skipped not in [f for items in project.split_plan().values() for f in items]
project.set_box(skipped, truth[skipped["src"]])

info = project.build_dataset(work / "dataset")
print("dataset:", {k: v for k, v in info.items() if k != "background_names"})
assert "ลูกบอล" in Path(info["data_yaml"]).read_text(encoding="utf-8")
image = project.frames_dir / frames[3]["file"]
assert image.with_suffix(".txt").stat().st_size > 0
assert (project.frames_dir / frames[4]["file"]).with_suffix(".noobject").exists()

# กรอบน้อยเกินไปต้องไม่เริ่มสอน
few = tp.TeachProject.create("น้อย")
few.clips = [dict(id=0)]
few.frames = [dict(file=f"{i}.jpg", clip=0, src=i, t=i / 2, state="drawn", boxes=[[0.1, 0.1, 0.2, 0.2]],
                   correction=False) for i in range(tp.MIN_LABELED - 1)]
try:
    few.build_dataset(work / "few")
    raise AssertionError("should refuse")
except ValueError as exc:
    print("refused:", exc)

# โปรเจกต์จากรุ่นที่มีตัวติดตาม: เก็บกรอบที่วาดเอง ทิ้งกรอบที่ไหลตามมาเอง
old = tp.TeachProject.create("รถสีส้ม")
meta = json.loads((old.path / tp.META_FILE).read_text(encoding="utf-8"))
meta["clips"] = [dict(id=0, name="รถ.mp4", file="clip_0.mp4", fps=30.0, duration=2.0)]
box = [0.1, 0.1, 0.5, 0.9]
meta["frames"] = [dict(file=f"frame_{i:06d}.jpg", clip=0, src=i, t=i / 2.0, state=state,
                       boxes=[box] if state in ("anchor", "tracked") else [], correction=False)
                  for i, state in enumerate(("anchor", "tracked", "lost", "none", "unset"))]
(old.path / tp.META_FILE).write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
upgraded = tp.TeachProject(old.path)
assert [f["state"] for f in upgraded.frames] == ["drawn", "unset", "unset", "none", "unset"]
assert upgraded.frames[0]["boxes"] == [box] and upgraded.frames[1]["boxes"] == []
assert upgraded.labeled_count() == 1 and upgraded.todo_count() == 3

# แก้ทักผิด: รูปชุดทดสอบย้ายไปชุดสอนและไม่อยู่ในชุดทดสอบอีก
wrong = plan["test"][0]
project.add_correction(wrong, [])
plan2 = project.split_plan()
assert wrong in plan2["train"] and wrong not in plan2["test"]
assert wrong["state"] in tp.READY_STATES, wrong["state"]
ext = project.add_external_correction(np.zeros((90, 160, 3), "uint8"), [])
project.write_labels()
assert (project.frames_dir / ext["file"]).with_suffix(".noobject").exists()

# เปิดโปรเจกต์ใหม่จากไฟล์ แล้วลบคลิป
again = tp.TeachProject(project.path)
assert len(again.frames) == len(project.frames) and again.name == "ลูกบอล"
assert project.path in [p.path for p in tp.list_projects()]
again.remove_clip(clip_id)
assert not again.clips and all(f["clip"] == -1 for f in again.frames)

shutil.rmtree(work, ignore_errors=True)
print("ALL OK")
