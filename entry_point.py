"""จุดเริ่มของ AIR Detection (ทั้งรันจากซอร์สด้วย Run.bat และตัว .exe ที่ PyInstaller สร้าง)

ส่วนตั้งค่าด้านบนมาจาก entry_point.py ของ AI Detection Lab v9 ซึ่งแก้ปัญหาจริงของตัว .exe ไว้แล้ว:
- โปรแกรมแบบไม่มีหน้าต่าง console: sys.stdout/stderr เป็น None -> แถบ progress ของ ultralytics เขียนแล้วพัง
  จึงต่อเข้า os.devnull แบบ utf-8 (Windows ภาษาไทยใช้ cp874 เข้ารหัสอีโมจิไม่ได้)
- freeze_support() ต้องมาก่อน import อื่น: การสอน AI สร้าง process ลูก ถ้าไม่มี ตัว .exe จะเปิดตัวเองซ้ำวนไป
"""
import multiprocessing
import os
import sys

if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8", errors="replace")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8", errors="replace")

# รันจากซอร์ส: ให้ import ai_detection_lab และ pyhulax (vendor/) ได้ (ตัว .exe รวมไว้ให้แล้ว)
if not getattr(sys, "frozen", False):
    _here = os.path.dirname(os.path.abspath(__file__))
    for _path in (_here, os.path.join(_here, "vendor")):
        if _path not in sys.path:
            sys.path.insert(0, _path)

if __name__ == "__main__":
    multiprocessing.freeze_support()

    # GPU Pack (โค้ดเดิมของ v9): คัดลอก PyTorch รุ่น CUDA ตอนแอปปิดอยู่ ก่อนมีอะไร import torch
    if len(sys.argv) >= 3 and sys.argv[1] == "--apply-gpupack":
        from ai_detection_lab.ai.gpupack import apply_main

        raise SystemExit(apply_main(sys.argv))

    if len(sys.argv) == 3 and sys.argv[1] == "--smoke-test":
        from ai_detection_lab.build_diagnostics import run_smoke_test

        raise SystemExit(run_smoke_test(sys.argv[2]))

    from ai_detection_lab.app_window import run

    raise SystemExit(run())
