"""สร้างคลิปทดสอบ: ลูกบอลลายเคลื่อนที่บนพื้นหลังมีลาย มีช่วงออกนอกจอ และบันทึกตำแหน่งจริงไว้เทียบ"""
import json
import math
import sys

import cv2
import numpy as np


def make_background(w, h, seed):
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 255, (h // 8, w // 8, 3), dtype=np.uint8)
    bg = cv2.GaussianBlur(cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC), (0, 0), 6)
    for _ in range(25):
        p = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        q = (p[0] + int(rng.integers(-120, 120)), p[1] + int(rng.integers(-120, 120)))
        cv2.rectangle(bg, p, q, tuple(int(v) for v in rng.integers(40, 220, 3)), -1)
    return cv2.GaussianBlur(bg, (0, 0), 1.2)


def draw_ball(img, cx, cy, r, angle):
    cv2.circle(img, (int(cx), int(cy)), int(r), (20, 110, 240), -1, cv2.LINE_AA)
    for k in range(3):
        a = angle + k * 2.094
        q = (int(cx + r * 0.6 * math.cos(a)), int(cy + r * 0.6 * math.sin(a)))
        cv2.circle(img, q, max(2, int(r * 0.22)), (250, 250, 250), -1, cv2.LINE_AA)
    cv2.circle(img, (int(cx), int(cy)), int(r), (10, 40, 90), 2, cv2.LINE_AA)


def make_clip(path, seconds=24, fps=30, w=1280, h=720, seed=1):
    rng = np.random.default_rng(seed)
    bg = make_background(w, h, seed)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    truth = []
    total = seconds * fps
    for i in range(total):
        t = i / fps
        # ช่วง 40-55% ของคลิป ลูกบอลบินออกนอกจอด้านขวาแล้วกลับเข้ามา
        cx = w * (0.5 + 0.3 * math.sin(t * 0.9))
        cy = h * (0.5 + 0.25 * math.sin(t * 1.3 + 1))
        p = i / total
        if 0.40 < p < 0.55:
            cx += (1 - abs(p - 0.475) / 0.075) * w * 1.1
        r = 55 + 25 * math.sin(t * 0.6)
        pan = (int(40 * math.sin(t * 0.4)), int(25 * math.cos(t * 0.5)))
        frame = np.roll(bg, pan, axis=(1, 0)).copy()
        draw_ball(frame, cx, cy, r, t * 1.5)
        frame = np.clip(frame.astype(np.int16) + rng.normal(0, 3, frame.shape), 0, 255).astype(np.uint8)
        writer.write(frame)
        truth.append([(cx - r) / w, (cy - r) / h, (cx + r) / w, (cy + r) / h])
    writer.release()
    return truth


if __name__ == "__main__":
    out = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    truth = make_clip(out, seed=seed)
    with open(out + ".json", "w", encoding="utf-8") as f:
        json.dump(truth, f)
    print("ok", out, len(truth))
