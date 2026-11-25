"""
clips.py
========
ตัดคลิปเป็นรูปสำหรับสอน AI

เลือกรูปที่คมที่สุดในทุก ๆ 1 วินาที (แทนการตัดตามเวลาเฉย ๆ ซึ่งมักได้รูปเบลอ) แล้วทิ้งรูปที่แทบไม่ต่างจากรูปก่อน
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ai_detection_lab.image_io import imwrite_any

VIDEO_SUFFIXES = (".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm")
MAX_CLIP_SECONDS = 60
MAX_CLIPS = 5
WINDOW_SECONDS = 1.0        # เก็บ 1 รูปต่อช่วงนี้
MAX_SAVE_SIDE = 1280        # คลิป 4K จากมือถือ ย่อก่อนเก็บ
DUPLICATE_DIFF = 1.5        # ต่างจากรูปก่อนน้อยกว่านี้ (0-255) = รูปซ้ำ
KEEP_AT_LEAST_SECONDS = 2.0  # แม้ภาพนิ่ง ก็เก็บอย่างน้อย 1 รูปทุกกี่วินาที
BLUR_FRACTION = 0.25        # คมน้อยกว่าสัดส่วนนี้ของค่ากลางในคลิป = เบลอ ทิ้ง
TOO_DARK = 55
TOO_BLURRY = 25.0
TOO_SHAKY = 28.0
TOO_SHORT_SECONDS = 3.0


def _sharpness(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def extract_frames(video_path, frames_dir, start_index, clip_index, on_progress=None, should_stop=None):
    """คืน dict(info ของคลิป, frames=รายการรูปที่บันทึกแล้ว) — ยก ValueError เมื่ออ่านคลิปไม่ได้"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("เปิดคลิปนี้ไม่ได้ ลองแปลงเป็นไฟล์ .mp4 แล้วอัปโหลดใหม่")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 1.0 or fps > 240.0:
        fps = 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    limit = int(MAX_CLIP_SECONDS * fps)
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    window = max(1, int(round(fps * WINDOW_SECONDS)))
    check_every = max(1, window // 4)  # ดูความคมประมาณ 4 รูปต่อวินาที
    picks, sharp_all, bright_all, diffs = [], [], [], []
    best = None
    prev_small = None
    src = 0
    size = None
    while src < limit:
        if should_stop is not None and should_stop():
            cap.release()
            raise InterruptedError
        if src % check_every != 0:
            if not cap.grab():
                break
            src += 1
            continue
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        size = frame.shape[1], frame.shape[0]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA)
        sharp = _sharpness(cv2.resize(gray, (320, max(1, 320 * gray.shape[0] // gray.shape[1]))))
        sharp_all.append(sharp)
        bright_all.append(float(small.mean()))
        if prev_small is not None:
            diffs.append(float(cv2.absdiff(small, prev_small).mean()))
        prev_small = small
        if best is None or sharp > best["sharp"]:
            best = dict(sharp=sharp, frame=frame, src=src, small=small)
        if (src + check_every) // window != src // window:  # จบช่วง 1 วินาที
            picks.append(best)
            best = None
        src += 1
        if on_progress is not None and src % 30 == 0:
            on_progress(min(src, limit), min(total, limit) if total > 0 else limit)
    if best is not None:
        picks.append(best)
    truncated = total > limit
    cap.release()
    if not picks:
        raise ValueError("อ่านภาพจากคลิปนี้ไม่ได้ ลองแปลงเป็นไฟล์ .mp4 แล้วอัปโหลดใหม่")

    median_sharp = float(np.median(sharp_all))
    saved, skipped_blur, skipped_dup = [], 0, 0
    last_small, last_t = None, -1e9
    index = start_index
    for pick in picks:
        t = pick["src"] / fps
        if pick["sharp"] < median_sharp * BLUR_FRACTION:
            skipped_blur += 1
            continue
        if last_small is not None and t - last_t < KEEP_AT_LEAST_SECONDS:
            if float(cv2.absdiff(pick["small"], last_small).mean()) < DUPLICATE_DIFF:
                skipped_dup += 1
                continue
        frame = pick["frame"]
        side = max(frame.shape[:2])
        if side > MAX_SAVE_SIDE:
            ratio = MAX_SAVE_SIDE / side
            frame = cv2.resize(frame, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
        name = f"frame_{index:06d}.jpg"
        if not imwrite_any(frames_dir / name, frame, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise ValueError("บันทึกรูปลงเครื่องไม่สำเร็จ พื้นที่ดิสก์อาจเต็ม")
        saved.append(dict(file=name, clip=clip_index, src=pick["src"], t=round(t, 3),
                          state="unset", boxes=[], correction=False))
        index += 1
        last_small, last_t = pick["small"], t

    duration = (min(total, limit) if total > 0 else src) / fps
    warnings = []
    if duration < TOO_SHORT_SECONDS:
        warnings.append("คลิปสั้นไปหน่อย ถ้าถ่ายให้ยาวขึ้น AI จะเก่งขึ้น")
    if float(np.median(bright_all)) < TOO_DARK:
        warnings.append("คลิปมืดไปหน่อย ใช้ได้ แต่ถ้าถ่ายในที่สว่างจะดีกว่า")
    if median_sharp < TOO_BLURRY:
        warnings.append("คลิปเบลอไปหน่อย ลองถือกล้องให้นิ่งขึ้นหรือถ่ายในที่สว่าง")
    elif diffs and float(np.median(diffs)) > TOO_SHAKY:
        warnings.append("คลิปสั่นไปหน่อย ลองขยับกล้องให้ช้าลง")
    if truncated:
        warnings.append(f"ใช้แค่ {MAX_CLIP_SECONDS} วินาทีแรกของคลิป")
    info = dict(fps=fps, duration=round(duration, 2), width=size[0], height=size[1],
                source_frames=min(total, limit) if total > 0 else src, warnings=warnings,
                skipped_blur=skipped_blur, skipped_duplicate=skipped_dup)
    return dict(info=info, frames=saved)
