"""Draw labels on video frames, including Thai.

cv2.putText only has ASCII glyphs: a class named "ลูกบอล" comes out as
"??????". ASCII text keeps the fast cv2 path; anything else goes through
Pillow with a Thai-capable font that ships with Windows.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

_FONT_CANDIDATES = (
    r"C:\Windows\Fonts\leelawad.ttf",
    r"C:\Windows\Fonts\tahoma.ttf",
    "/System/Library/Fonts/Supplemental/Tahoma.ttf",
    "/System/Library/Fonts/Thonburi.ttc",
)
_fonts: dict = {}


def _font(size):
    if size not in _fonts:
        from PIL import ImageFont

        font = None
        for candidate in _FONT_CANDIDATES:
            if Path(candidate).is_file():
                try:
                    font = ImageFont.truetype(candidate, size)
                    break
                except OSError:
                    continue
        if font is None:
            print(f"⚠️ No Thai font found on {sys.platform}", file=sys.stderr)
            font = ImageFont.load_default()
        _fonts[size] = font
    return _fonts[size]


def put_text(img, text, org, scale=0.6, color=(255, 255, 255), thickness=2):
    """Same idea as cv2.putText (org = baseline-left, color = BGR), drawn in place."""
    if str(text).isascii():
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
        return img
    from PIL import Image, ImageDraw

    size = max(12, int(scale * 34))
    font = _font(size)
    draw_probe = ImageDraw.Draw(Image.new("L", (1, 1)))
    left, top, right, bottom = draw_probe.textbbox((0, 0), text, font=font)
    pad = 4
    tw, th = right - left + pad * 2, bottom - top + pad * 2
    x, y = int(org[0]), int(org[1]) - th + pad
    h, w = img.shape[:2]
    x, y = max(0, min(x, w - 1)), max(0, min(y, h - 1))
    x2, y2 = min(w, x + tw), min(h, y + th)
    if x2 <= x or y2 <= y:
        return img
    # Only convert the small area under the text, so a label costs little per frame.
    region = Image.fromarray(img[y:y2, x:x2][:, :, ::-1])
    ImageDraw.Draw(region).text((pad - left, pad - top), text, font=font, fill=(color[2], color[1], color[0]))
    img[y:y2, x:x2] = np.asarray(region)[:, :, ::-1]
    return img


def draw_boxes(frame, detections):
    """Boxes + "name 90%" labels for a list of detector detections; returns a new image."""
    img = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["xyxy"]]
        cv2.rectangle(img, (x1, y1), (x2, y2), (85, 214, 190), 2)
        label = f"{det.get('class_name', '?')} {float(det.get('confidence', 0)):.0%}"
        put_text(img, label, (x1, max(20, y1 - 6)), 0.6, (223, 252, 247), 2)
    return img
