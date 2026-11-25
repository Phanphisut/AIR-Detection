"""Unicode-safe image read/write.

On Windows, cv2.imread()/cv2.imwrite() cannot handle paths containing
non-ASCII characters - e.g. a Thai Windows user name (the data folder lives
under C:\\Users\\<name>\\AppData\\Local\\...) or a Thai project name. They don't
raise: imread returns None and imwrite returns False, so frames silently
never get saved. Encoding/decoding in memory and letting Python do the file
IO works with any path on every OS.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imread_any(path, flags: int = cv2.IMREAD_COLOR):
    """cv2.imread replacement that accepts any Unicode path. Returns None if
    the file is missing or not a readable image."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except (OSError, ValueError):
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_any(path, image, params=None) -> bool:
    """cv2.imwrite replacement that accepts any Unicode path. Returns True on
    success, False otherwise (same contract as cv2.imwrite)."""
    ext = Path(path).suffix or ".jpg"
    try:
        ok, buf = cv2.imencode(ext, image, params or [])
        if not ok:
            return False
        buf.tofile(str(path))
        return True
    except (OSError, cv2.error, ValueError):
        return False
