import os
import sys
from pathlib import Path

# The built app cannot pip-install. Ultralytics reads this once, when it is
# first imported, and otherwise tries to "AutoUpdate" packages (CoreML export
# asks for another numpy) - which hangs the frozen app. config is the first
# module the app imports, so this is set before ultralytics ever loads.
os.environ.setdefault("YOLO_AUTOINSTALL", "False")

APP_VERSION = "9.0.0"   # shown in the Studio sidebar and Platform uploads

if getattr(sys, "frozen", False):
    # Running from the PyInstaller-frozen build (see AI_Detection_Lab.spec).
    # __file__ would resolve inside the bundle, not next to the distributed
    # .exe, so models/captures/training_sessions must be found relative to
    # the executable itself instead. Mirrors ui/code_editor.py's
    # project_root(), which already handles this the same way.
    #
    # macOS .app bundles lay this out differently from a Windows onedir
    # build: sys.executable sits alone in Contents/MacOS/, while every
    # bundled data file (the spec's `datas`, e.g. the seed "models" folder)
    # lands in Contents/Resources/ instead - the two are siblings, not the
    # same folder. Using sys.executable's parent directly (as on Windows,
    # where the exe and its data genuinely are siblings) would make
    # PROJECT_ROOT point at Contents/MacOS/, which never has a "models"
    # folder to seed from, silently leaving a fresh install with zero
    # default models.
    BASE_DIR = Path(sys.executable).resolve().parent
    if sys.platform == "darwin":
        BASE_DIR = BASE_DIR.parent / "Resources"
    elif sys.platform == "win32" and (BASE_DIR / "_internal").is_dir():
        # PyInstaller 6.x onedir builds (see AI_Detection_Lab_Windows.spec)
        # collect every `datas` entry - the seed "models" folder, app_icon.png,
        # the loose ai_detection_lab/ source copy - under a "_internal"
        # subfolder next to the exe, not directly beside it. Confirmed by
        # inspecting a real build: BASE_YOLO_WEIGHTS below silently resolved
        # to None (bundled_models search below found nothing either) because
        # this assumed the exe and its data were siblings, which stopped
        # being true once PyInstaller introduced the _internal/ layout - the
        # `.is_dir()` guard keeps this working if a future/older PyInstaller
        # ever goes back to flat onedir output.
        BASE_DIR = BASE_DIR / "_internal"
    PROJECT_ROOT = BASE_DIR
else:
    BASE_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = BASE_DIR.parent

# Running inside "AI Detection Lab.app/Contents/Resources/..." (the
# self-contained macOS app bundle - see AI Detection Lab.app)? The bundle
# can be dragged into /Applications, which a normal user can't write to,
# and even where it can be written the app bundle's own contents shouldn't
# be treated as a scratch folder. AI Data & Training needs a real per-user
# writable location instead, same as any other macOS app's user data -
# ~/Library/Application Support is the standard place for it.
_APP_BUNDLE_DIR = next((p for p in PROJECT_ROOT.parents if p.suffix == ".app"), None)

# The Windows onedir build has the same underlying problem even without an
# .app-style bundle: PROJECT_ROOT there is dist/AI Detection Lab/ (see
# AI_Detection_Lab_Windows.spec's COLLECT step), and every `pyinstaller`
# rebuild wipes and recreates that whole folder from scratch - confirmed by
# losing a real, already-labeled training session (32 labeled frames) to
# exactly this during a rebuild-to-fix-a-bug cycle. Anything under
# PROJECT_ROOT is disposable build output, not a safe home for a student's
# captured photos, labels or trained models, so frozen Windows builds get
# the same treatment as the macOS bundle: a real per-user writable folder
# outside the build output, using the standard per-user app-data location
# (%LOCALAPPDATA%, falling back to ~/AppData/Local if that env var is ever
# unset).
_EXTERNAL_DATA_DIR = None
if _APP_BUNDLE_DIR is not None:
    _EXTERNAL_DATA_DIR = Path.home() / "Library" / "Application Support" / "AIR Detection"
elif getattr(sys, "frozen", False) and sys.platform == "win32":
    import os

    _local_appdata = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    _EXTERNAL_DATA_DIR = Path(_local_appdata) / "AIR Detection"

_DATA_PARENT = _EXTERNAL_DATA_DIR if _EXTERNAL_DATA_DIR is not None else PROJECT_ROOT

# All AI data/output folders (trained models, captured evidence, training
# session logs) live grouped under one visible folder
# next to the app, instead of scattered loose at the project root - purely
# a folder-tidiness change requested by the user; nothing about what gets
# read/written or how changes, only where. ui/label_train_dialog.py's
# TRAINING_ROOT builds its paths off DATA_ROOT (not PROJECT_ROOT directly) for the same
# reason - keep all of them in sync if this ever moves again.
DATA_ROOT = _DATA_PARENT / "AI Data & Training"
MODEL_DIR = DATA_ROOT / "models"
CAPTURE_DIR = DATA_ROOT / "captures"
DRONE_PHOTOS_DIR = DATA_ROOT / "drone_photos"

# Where ui/code_editor.py's "Quick Parameters" tab saves its edits. It
# CANNOT write those edits back into this very file: the installer's
# default install location is Program Files ({autopf} in installer.iss),
# which a normal (non-admin) user cannot write to, so every "Save
# Parameters" click there silently/loudly failed with a permission error -
# same class of problem DATA_ROOT above already had to work around for
# captured photos/trained models, just hitting config.py itself this time.
# The fix is the same shape: park the user's edits in the writable
# per-user data folder instead, as a small JSON file of
# {CONST_NAME: value} overrides layered on top of the defaults below at
# import time (see _apply_user_overrides at the bottom of this file), so
# config.py itself never needs to be written to after install.
USER_OVERRIDES_PATH = _DATA_PARENT / "user_config_overrides.json"

# Base checkpoint that label_train_dialog.py fine-tunes from. Resolved to a
# path bundled inside the app/source tree (Resources/models/yolov8n.pt when
# frozen, the bare project-root copy when running from source) so training
# works offline on a fresh machine instead of ultralytics silently reaching
# out to the internet the first time a student clicks "Train". None if
# neither copy is present, in which case ultralytics falls back to its own
# download-on-demand behavior.
BASE_YOLO_WEIGHTS = next(
    (p for p in (PROJECT_ROOT / "models" / "yolov8n.pt", PROJECT_ROOT / "yolov8n.pt") if p.is_file()),
    None,
)



def webcam_capture_backend() -> int:
    """cv2.VideoCapture backend flag for opening the laptop's own webcam
    (Simulation Mode's camera stand-in, Webcam Test) on whichever OS
    this is running on.

    cv2.CAP_AVFOUNDATION only exists on macOS. Passed on Windows or Linux,
    VideoCapture opens without error but cap.isOpened() never becomes
    True and no frame ever arrives - it just hangs at "waiting for
    camera" forever with nothing in the logs to explain why. Every
    VideoCapture(index, ...) call for a live webcam should go through
    this instead of hardcoding a backend constant."""
    import cv2

    if sys.platform == "darwin":
        return cv2.CAP_AVFOUNDATION
    if sys.platform == "win32":
        return cv2.CAP_DSHOW
    return getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)


def ensure_default_models_seeded() -> None:
    """First run of the .app bundle / Windows onedir build only: copy the
    models shipped inside the app into the writable per-user models folder,
    so there's something to detect with out of the box. Never overwrites or
    re-copies once MODEL_DIR has at least one .pt file in it - after that,
    MODEL_DIR is the instructor/student's own folder to add or remove
    models from freely, without a later app update ever clobbering their
    changes. No-op when MODEL_DIR already lives next to PROJECT_ROOT
    (running from source, or a frozen build with no external data dir) -
    MODEL_DIR is already the live, bundled folder in that case."""
    if _EXTERNAL_DATA_DIR is None:
        return
    if MODEL_DIR.exists() and any(MODEL_DIR.glob("*.pt")):
        return
    bundled_models = PROJECT_ROOT / "models"
    if not bundled_models.is_dir():
        return
    import shutil

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for item in bundled_models.iterdir():
        dest = MODEL_DIR / item.name
        if item.is_file() and not dest.exists():
            shutil.copy2(item, dest)

DRONE_IP = None
# ที่อยู่ IP ของโดรน (Hula/HG-Fly F09-lite ในโหมด Access Point)
# None = ใช้ค่าเริ่มต้นของ pyhulax เอง (192.168.100.1)
# ใส่ค่าที่นี่เฉพาะกรณีโดรนถูกตั้งค่าให้ใช้ IP อื่น

DRONE_TEMP_WARN_C = 60
# แสดง ⚠️ ข้างอุณหภูมิโดรนเมื่อถึงค่านี้ (°C) — ค่าประมาณ ยังไม่มีเกณฑ์จากผู้ผลิต
# (ทดสอบจริง 2026-10-07: วางเปิดไว้ ~1 ชม. ขึ้นจาก 34 → 51°C)

LANGUAGE = "en"
# ภาษาที่ใช้แสดงผลในหน้าต่างต่าง ๆ ของโปรแกรม — "en" (English) หรือ "th" (ไทย)
# แก้ได้จาก Code Editor > Quick Parameters > ทั่วไป
# มีผลหลังปิดแล้วเปิดโปรแกรมใหม่เท่านั้น (เหมือนค่าอื่น ๆ ใน Quick Parameters)
# ทุกหน้าต่างที่สร้างขึ้นหลังจากนั้นจะใช้ภาษานี้ ผ่าน ai_detection_lab/i18n.py's tr()

# ระบบจะค้นหาโมเดล YOLO อัตโนมัติจากโฟลเดอร์ MODEL_DIR
# สามารถนำไฟล์ .pt ใส่ไว้ในโฟลเดอร์นี้ได้หลายไฟล์
# ไม่จำเป็นต้องกำหนดชื่อโมเดลในไฟล์ Config
# หากมีการเพิ่มไฟล์ .pt ใหม่ ระบบจะตรวจพบอัตโนมัติขณะ Lab กำลังทำงาน
YOLO_CONF_THRES = 0.8
YOLO_IMGSZ = 640
YOLO_NMS_IOU = 0.5
# กรอบที่ซ้อนกันเกินนี้ (0-1) ถือเป็นวัตถุชิ้นเดียว ให้เหลือกรอบเดียว
# (ค่าเดิมของ YOLO คือ 0.7 ทำให้วัตถุ 1 ชิ้นขึ้นหลายกรอบ และนับเกินจริง)
YOLO_DEVICE = None
# อุปกรณ์ที่ใช้ตรวจจับสด: None/"auto" = เลือกให้เอง (mps บน Mac ชิป Apple, cuda ถ้ามีการ์ด NVIDIA, ไม่งั้น cpu)
# หรือบังคับ "mps" / "cuda" / "cpu" — ถ้าเครื่องไม่มีอุปกรณ์ที่เลือก จะใช้ cpu แทนพร้อมแจ้งเตือน (ดู ai/devices.py)
TRAIN_DEVICE = "auto"
# อุปกรณ์ที่ใช้ตอน Train AI (เลือกได้ที่หน้า Train): "auto" / "mps" / "cuda" / "cpu"
# วัดบน Mac ชิป Apple: mps เทรนเร็วกว่า cpu ราว 2 เท่า (Ultralytics เลือกเองจะได้ cpu เสมอ)



SKIP_STOCK_MODEL_WHEN_CUSTOM_EXISTS = True
# True = เมื่อมีโมเดลที่นักเรียนเทรนเองอยู่ในโฟลเดอร์ models แล้ว จะไม่รัน yolov8n.pt
# (โมเดลตั้งต้น 80 คลาส เช่น person, chair) พร้อมกัน เพื่อไม่ให้มีกรอบที่ไม่เกี่ยวข้อง
# ปนในการนับ/ภาพหลักฐาน และไม่เปลือง CPU สองเท่า (ไฟล์ไม่ถูกลบ)
# False = รันทุกไฟล์ .pt ในโฟลเดอร์พร้อมกัน (พฤติกรรมเดิม)

TRAINING_PHOTO_INTERVAL_SEC = 1.0
# ปุ่ม "📸 Record Training Photos" บันทึกภาพจริงจากกล้องโดรน (ไม่มีกรอบ AI ทับ)
# ทุกกี่วินาที ภาพเก็บไว้ที่ AI Data & Training/drone_photos/ เพื่อนำไปเทรนต่อ
# ============================================================
# การตรวจสอบความเสถียรของการตรวจจับ
# ============================================================

STABILITY_WINDOW_FRAMES = 5
# ใช้ผลการตรวจจับย้อนหลังจำนวน 5 เฟรมล่าสุด
# เพื่อคำนวณความเสถียร (Stability) ของเป้าหมายแต่ละตัว

STABILITY_MIN_FRAMES = 5
# ต้องมีข้อมูลการตรวจจับอย่างน้อย 4 เฟรม
# ก่อนที่จะพิจารณาว่าเป้าหมายมีสถานะเป็น STABLE
# เพื่อป้องกันการตัดสินเร็วเกินไปในช่วงที่เพิ่งพบเป้าหมาย

STABILITY_MIN_RATIO = 0.8
# เป้าหมายต้องถูกตรวจพบในสัดส่วนอย่างน้อย 80%
# ของจำนวนเฟรมที่ใช้ตรวจสอบ
# ตัวอย่าง: 4 จาก 5 เฟรม = 0.8 จึงถือว่าเป็นการตรวจจับที่สม่ำเสมอ

POSITION_JUMP_MAX_PX = 80
# กำหนดระยะการกระโดดสูงสุดของ Bounding Box
# เป้าหมายเดียวกันต้องไม่เคลื่อนที่เกิน 80 พิกเซล
# ระหว่างเฟรมที่ตรวจพบต่อเนื่องกัน
# ใช้ช่วยป้องกันการนำวัตถุคนละตัวมารวมเป็นเป้าหมายเดียวกัน

CAPTURE_ON_STABLE = True
# ถ่ายภาพอีกครั้งเมื่อเป้าหมายเปลี่ยนสถานะเป็น STABLE
# นอกเหนือจากการถ่ายภาพครั้งแรกที่ตรวจพบเป้าหมาย


# ============================================================
# ระบบตรวจสอบแบตเตอรี่
# ============================================================

BATTERY_CRITICAL_PCT = 15
# ระดับแบตเตอรี่ขั้นต่ำที่ถือว่าอยู่ในสถานะวิกฤต (%)

BATTERY_WARN_PCT = 30
# v9.0.0: แบตเหลือเท่านี้ (%) หรือน้อยกว่า → แจ้งเตือน "แบตเตอรี่ใกล้หมด" + เสียงเตือน 3 วินาที

BATTERY_TAKEOFF_MIN_PCT = 20
# v9.0.0: แบตเหลือเท่านี้ (%) หรือน้อยกว่า → แจ้ง "แบตเตอรี่ต่ำเกินไปสำหรับการบิน"
# และไม่ยอมให้ขึ้นบิน (ทุกปุ่ม: บินสำรวจ, Flight Plan, บินด้วยคีย์บอร์ด)
# ถ้ากำลังบินอยู่จะเตือนให้ลงจอดทันที (ลงจอดอัตโนมัติยังเป็นที่ BATTERY_CRITICAL_PCT)

DRONE_LINK_LOST_SEC = 5.0
# v9.0.0: ไม่ได้รับข้อมูลใดๆ จากโดรนนานเกินกี่วินาที ถึงถือว่า "โดรนหลุดการเชื่อมต่อ"

ALERT_SOUND_ENABLED = True
# v9.0.0: เปิด/ปิดเสียงเตือน (ดังถี่ๆ 3 วินาทีแล้วหยุดเอง) ตอนเกิดการแจ้งเตือน

BATTERY_POLL_FAIL_DISCONNECT_COUNT = 3
# จำนวนครั้งสูงสุดที่สามารถอ่านค่าแบตเตอรี่ไม่สำเร็จ
# หากอ่านไม่สำเร็จติดต่อกันถึงจำนวนที่กำหนด
# ระบบจะถือว่าโดรนอาจขาดการเชื่อมต่อ


# ============================================================
# การตั้งค่ากล้องของโดรน
# ============================================================

CAMERA_AUTO_DOWN = True
# ให้ระบบปรับมุมกล้องลงโดยอัตโนมัติเมื่อเริ่มต้น

CAMERA_DOWN_DIRECTION = 1
# ทิศทางการหมุนกล้องลง
# 1 = ลง

CAMERA_DOWN_ANGLE = 90
# มุมกล้องที่ต้องการให้ก้มลง (องศา)

FRAME_SLEEP_SEC = 0.01
# เวลาหน่วงระหว่างการประมวลผลแต่ละเฟรม
# เมื่อได้รับภาพจากกล้องสำเร็จ

NO_FRAME_SLEEP_SEC = 0.03
# เวลาหน่วงเมื่อไม่สามารถรับภาพจากกล้องได้
# ช่วยลดการใช้ CPU ในขณะที่ยังไม่มีภาพ

FIRST_FRAME_TIMEOUT_SEC = 20.0
# เวลาสูงสุดที่ยอมให้ "รอเฟรมแรก" หลัง RTP stream เริ่มทำงาน
# ก่อนจะถือว่าล้มเหลวและแจ้งสถานะ แทนที่จะค้างรอเงียบๆ ตลอดไป
# (พบเจอเวลาทดสอบกับโดรนหลายลำต่อเนื่องกัน — session ก่อนหน้าที่ปิด
# ไม่สนิทอาจทำให้พอร์ต RTP ของ session ถัดไปไม่ได้รับแพ็กเก็ตเลย)


# ============================================================
# นโยบายการบันทึกภาพ
# ============================================================

AUTO_CAPTURE = True
# เปิดใช้งานระบบถ่ายภาพอัตโนมัติ

CAPTURE_ON_FIRST_DETECTION = True
# ถ่ายภาพทันทีเมื่อพบเป้าหมายเป็นครั้งแรก

CAPTURE_ON_NEW_MAXIMUM = True
# ถ่ายภาพเมื่อพบจำนวนเป้าหมายสูงสุดใหม่
# เช่น จากเดิมพบ 1 เป้าหมาย แล้วพบเพิ่มเป็น 2 เป้าหมาย

CAPTURE_COOLDOWN_SEC = 1.0
# ระยะเวลาขั้นต่ำระหว่างการถ่ายภาพแต่ละครั้ง (วินาที)
# ป้องกันการถ่ายภาพซ้ำถี่เกินไป

TRACK_IOU_MATCH_THRES = 0.3
# ค่า IoU ขั้นต่ำที่ถือว่ากล่องในเฟรมนี้คือ "วัตถุชิ้นเดิม" กับเฟรมก่อนหน้า
# (ดู ai/tracker.py) ต่ำเกินไป = วัตถุคนละชิ้นถูกมองเป็นชิ้นเดียวกัน
# สูงเกินไป = วัตถุเดิมได้ ID ใหม่ทุกครั้งที่ขยับเร็วๆ

TRACK_MAX_AGE_FRAMES = 15
# ถ้าวัตถุหายจากภาพเกินจำนวนเฟรมนี้ จะเลิกตามรอย ID เดิม
# หากกลับมาใหม่จะนับเป็นการพบครั้งใหม่ (ID ไม่ถูกใช้ซ้ำ)

UNIQUE_DETECTION_RADIUS_CM = 50
# ระยะห่างขั้นต่ำที่ใช้พิจารณาว่าเป็นเป้าหมายใหม่ (เซนติเมตร)
# หากตำแหน่งอยู่ใกล้กันภายในระยะนี้
# ระบบจะพิจารณาว่าอาจเป็นเป้าหมายเดิม


# ============================================================
# ขนาดพื้นที่สนามจริง
# ============================================================

# ขนาดสนามจริงที่ผู้สอนเตรียมไว้

FIELD_WIDTH_CM = 100
# ความกว้างของสนาม (เซนติเมตร)

FIELD_HEIGHT_CM = 100
# ความยาว/ความสูงของสนาม (เซนติเมตร)

GRID_SPACING_CM = 50
# ระยะห่างระหว่างแต่ละแนวของ Grid (เซนติเมตร)


# ============================================================
# การนำทางสำหรับการสำรวจพื้นที่
# ============================================================

INITIAL_HEADING = "EAST"
# ทิศทางเริ่มต้นของโดรน

SURVEY_ALTITUDE_CM = 100
# ความสูงที่ใช้ในการสำรวจพื้นที่ (เซนติเมตร)
# ต้องอยู่ในช่วง 100-3000 ซม. ตามข้อจำกัดของเฟิร์มแวร์โดรน (ค่าต่ำกว่า 100
# จะถูกปฏิเสธและทำให้ takeoff() timeout)

TAKEOFF_SETTLE_SEC = 2.0
# เวลาที่ให้โดรนทรงตัวหลังจาก Takeoff (วินาที)

SURVEY_ALTITUDE_SETTLE_SEC = 1.0
# เวลาที่ให้โดรนทรงตัวหลังจากปรับระดับความสูง
# ก่อนเริ่มเคลื่อนที่สำรวจ

TURN_YAW_SPEED = 15
# ความเร็วในการหมุนซ้าย/ขวา (Yaw)

TURN_SETTLE_SEC = 1.5
# เวลาที่ให้โดรนทรงตัวหลังจากหมุนเปลี่ยนทิศทาง


# ============================================================
# ระบบบินสำรวจแบบ Autonomous Grid
# ============================================================

AUTONOMOUS_FLIGHT_ENABLED = True
# เปิดใช้งานระบบบินอัตโนมัติแบบ Grid

TAKEOFF_ENABLED = True
# อนุญาตให้ระบบสั่ง Takeoff อัตโนมัติ


# ============================================================
# การตั้งค่าความปลอดภัยและการเคลื่อนที่ของ Grid
# ============================================================

# คำสั่งเคลื่อนที่ของ Hula จะถูกตีความว่าเป็น
# การเคลื่อนที่ตามระยะทางที่กำหนดในหน่วยเซนติเมตร

GRID_MOVE_STEP_CM = 25
# ระยะการเคลื่อนที่ต่อหนึ่งคำสั่ง (เซนติเมตร)

GRID_MOVE_INTERVAL_SEC = 0.35
# ระยะเวลาหน่วงระหว่างคำสั่งเคลื่อนที่แต่ละคำสั่ง (วินาที)

GRID_ROW_PAUSE_SEC = 1.0
# เวลาหยุดพักเมื่อจบแต่ละแถวของ Grid

GRID_WAYPOINT_PAUSE_SEC = 0.25
# เวลาหยุดพักเมื่อถึงแต่ละ Waypoint

AUTO_LAND_ON_COMPLETE = True
# ให้โดรนลงจอดอัตโนมัติเมื่อบินสำรวจครบทุก Waypoint


# ============================================================
# การยืนยันก่อนเริ่มบินอัตโนมัติ
# ============================================================

# กำหนดให้ผู้ใช้งานต้องยืนยันความพร้อมก่อน
# ระบบจึงจะอนุญาตให้โดรน Takeoff และเริ่มบินอัตโนมัติ

REQUIRE_PREFLIGHT_CONFIRMATION = True
# True  = ต้องยืนยัน Preflight ก่อนเริ่มบิน
# False = สามารถเริ่มบินอัตโนมัติได้ทันที


# ============================================================
# โหมดผู้พัฒนา (Developer Mode) — บังคับโดรนด้วยคีย์บอร์ดโดยตรง
# ============================================================
# ปิดไว้เป็นค่าเริ่มต้นโดยตั้งใจ: การบิน manual ด้วยคีย์บอร์ดข้ามระบบ
# ตรวจสอบความปลอดภัยแบบ Grid ไปเกือบทั้งหมด (ไม่มี geofence,
# ไม่มี waypoint ที่วางแผนไว้ล่วงหน้า) จึงต้องกดปุ่ม "🔧 Developer Mode"
# ในหน้าโปรแกรมเพื่อเปิดใช้งานปุ่ม "🕹️ Manual Flight" ก่อน ป้องกันนักเรียน
# กดเข้าโหมดนี้โดยไม่ตั้งใจ

MANUAL_FLIGHT_MOVE_STEP_CM = 30
# ระยะขยับต่อการกดปุ่มเคลื่อนที่ 1 ครั้ง (W/A/S/D หรือลูกศร)

MANUAL_FLIGHT_ALTITUDE_STEP_CM = 20
# ระยะขยับความสูงต่อการกดปุ่มขึ้น/ลง 1 ครั้ง (R/F)

MANUAL_FLIGHT_ROTATE_DEG = 15
# มุมหมุนต่อการกดปุ่มหมุน 1 ครั้ง (Q/E)


# ============================================================
# MISSIONS — Flight Plan (Python) / Keyboard Control
# ============================================================
FLIGHT_PLAN_MAX_MOVE_CM = 200
# ระยะสูงสุดต่อ 1 คำสั่งเคลื่อนที่ใน Flight Plan (เช่น drone.forward(500) จะถูกจำกัดเหลือ 200)

FLIGHT_PLAN_MAX_ROTATE_DEG = 360
# มุมหมุนสูงสุดต่อ 1 คำสั่ง

FLIGHT_PLAN_MAX_COMMANDS = 200
# จำนวนคำสั่งบินสูงสุดต่อการรัน 1 ครั้ง กันลูปไม่รู้จบทำให้โดรนบินไม่หยุด



# ============================================================
# โหมดจำลอง (Simulation Mode) — ทดสอบทั้งระบบโดยไม่ต้องมีโดรนจริง
# ============================================================
# เมื่อเปิดใช้ (ปุ่ม "🧪 Simulation Mode" ในหน้าโปรแกรม, ต้องเปิดก่อน
# เชื่อมต่อ/บิน) main_window.py จะสลับไปใช้ SimulatedDroneController แทน
# HulaDroneController จริง ทุกคำสั่งเชื่อมต่อ/บิน/ลงจอดจะสำเร็จทันที (มี
# หน่วงเวลาสั้นๆ ให้สมจริง) โดยใช้กล้องเว็บแคมของโน้ตบุ๊กแทนภาพจากโดรน
# เพื่อให้ยังทดสอบ YOLO detection ได้จากภาพจริง

SIMULATION_CAMERA_INDEX = 0
# Index กล้องเว็บแคมของโน้ตบุ๊กที่ใช้แทนภาพจากกล้องโดรนขณะอยู่ใน
# Simulation Mode

SIMULATION_BATTERY_DRAIN_SEC_PER_PCT = 2.0
# แบตเตอรี่จำลองจะลดลง 1% ทุกๆกี่วินาทีที่ "บิน" อยู่ (ใช้ทดสอบ
# BATTERY_CRITICAL_PCT / การลงจอดฉุกเฉินเมื่อแบตใกล้หมดได้โดยไม่ต้องรอ
# แบตจริงหมด) — ค่าเริ่มต้น 2 วินาที/1% = จากเต็มถึงระดับวิกฤต (15%) ใน
# ประมาณ 170 วินาที


# ============================================================
# ค่าที่ผู้ใช้แก้ไขเองผ่าน Code Editor > Quick Parameters
# ============================================================
# ต้องอยู่ท้ายไฟล์เสมอ: ทับค่า default ด้านบนทั้งหมดด้วยค่าที่ผู้ใช้บันทึกไว้
# ใน USER_OVERRIDES_PATH (ดูคำอธิบายที่นิยาม USER_OVERRIDES_PATH ด้านบน)
# ใช้ globals()[name] = value ตรงๆ แทนการ exec()/eval() ทั้งไฟล์ ปลอดภัยกว่า
# และเฉพาะชื่อค่าคงที่ที่มีอยู่แล้วในไฟล์นี้เท่านั้นที่จะถูกทับ ชื่อแปลกปลอม
# ใน JSON (เช่นไฟล์เก่าจาก schema รุ่นก่อน) จะถูกข้ามเงียบๆ ไม่ถูกสร้างเป็น
# ตัวแปรใหม่


def _apply_user_overrides() -> None:
    if not USER_OVERRIDES_PATH.is_file():
        return
    import json

    try:
        overrides = json.loads(USER_OVERRIDES_PATH.read_text(encoding="utf-8"))
    except Exception:
        return
    for name, value in overrides.items():
        if name in globals():
            globals()[name] = value


_apply_user_overrides()
