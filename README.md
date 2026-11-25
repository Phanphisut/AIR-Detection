# AIR Detection 1.1.0

แอปสอน AI และบินโดรน Hula 1 ลำพร้อม AI ตรวจจับ สำหรับเด็ก ชื่อหน้า ปุ่ม และหัวข้อเป็นภาษาอังกฤษ ส่วนคำอธิบายเป็นภาษาไทย
แยกออกมาจาก Swarm Detection Lab เหลือแค่ 2 หน้า (ไม่มีส่วน Swarm)

| หน้า | ทำอะไร |
|---|---|
| **Train AI** (หน้าแรก) | 4 ขั้น: Project → Upload clips → Draw boxes (1 รูป/วินาที) → Train and test (Quick 20 / Normal 40 / Detailed 60 รอบ) |
| **Fly & Detect** | 4 ขั้น: Connect (Real drone / Simulator / Webcam only) → Choose AI (ได้หลายตัว) → Flight mode → Fly |

ภาษา: ใน 2 หน้าหลัก ชื่อหน้า ขั้นตอน ปุ่ม หัวข้อ และชื่อการ์ดเป็นอังกฤษ คำอธิบาย คำแนะนำ และเนื้อความในกล่องข้อความเป็นไทย
หน้าต่างเดิมของ v9 ที่เด้งขึ้นมา (คีย์บอร์ด เขียนโค้ด รายงาน แจ้งเตือนแบต) เป็นอังกฤษทั้งหมด (`LANGUAGE = "en"` ใน config.py)
ชื่อของที่เด็กพิมพ์เป็นภาษาไทยยังใช้ได้ และแสดงบนกรอบได้ถูกต้อง

## ติดตั้งบนเครื่องใหม่ (รันจากซอร์ส)
1. โหลดโค้ด: `git clone https://github.com/Phanphisut/AIR-Detection.git` (หรือ Code → Download ZIP แล้วแตกไฟล์)
2. ดับเบิลคลิก **`setup.bat`** ครั้งเดียว (ไม่ต้องใช้สิทธิ์ admin, ต่อเน็ต, ใช้เวลาราว 5-15 นาที)
   - ติดตั้ง Python 3.13 ผ่าน winget ถ้ายังไม่มี
   - สร้าง venv ที่ `%USERPROFILE%\.venvs\air-detection` (นอกโฟลเดอร์โปรเจกต์ จะได้ไม่ซิงก์ขึ้น OneDrive)
   - เจอการ์ดจอ NVIDIA (`nvidia-smi`) จะลง PyTorch รุ่น CUDA 12.6 ให้เอง ไม่เจอก็ลงรุ่น CPU
     (บังคับรุ่น CPU: `set TORCH_CUDA=cpu` ก่อนรัน)
   - รันซ้ำได้ ใช้ซ่อม/อัปเดตแพ็กเกจ
3. เปิดโปรแกรม: ดับเบิลคลิก **`Run.bat`** (ถ้ายังไม่ได้ setup จะเรียก setup.bat ให้เอง)
   หรือเปิดโฟลเดอร์ใน **VS Code** แล้วกด **F5** (ต้องมีส่วนขยาย Python ของ Microsoft)
   - `AIR Detection (Pixel UI)` = หน้าตาพิกเซล, `AIR Detection (original UI)` = หน้าตาเดิม

ข้อมูล (โมเดล รูป โปรเจกต์) ตอนรันจากซอร์สเก็บที่ `AI Data & Training\` ในโฟลเดอร์นี้ (ไม่ขึ้น git)
เปิดพร้อม AI Detection Lab v9 หรือ Swarm Detection Lab ไม่ได้ (ใช้โดรนและกล้องพร้อมกันไม่ได้)
ปิดธีมพิกเซล: `AIR_PIXEL_UI=0` · ปิดพื้นหลังโดรนเคลื่อนไหว: `AIR_BACKDROP=0`

## Build เป็น .exe + ตัวติดตั้ง (แบบเดียวกับ AI Detection Lab v9 Build Kit)
1. `setup_windows.bat` ครั้งเดียว: ติดตั้ง Python 3.13, Visual Studio Build Tools (C++), Inno Setup 6
2. `build_windows.bat`: สร้าง `.venv` (PyTorch รุ่น CPU) → Nuitka คอมไพล์ `ai_detection_lab` เป็น .pyd
   → PyInstaller → ตัวตรวจเครื่อง → Inno Setup
3. ผลลัพธ์: `dist\AIR Detection\AIR Detection.exe` และ `dist_installer\AIR Detection Setup v1.1.0.exe`
4. ทดสอบตัวที่ build แล้ว: `"dist\AIR Detection\AIR Detection.exe" --smoke-test <โฟลเดอร์ผล>` ดูผลใน `smoke-test.json`

ตัวที่ติดตั้งแล้วเก็บข้อมูลที่ `%LOCALAPPDATA%\AIR Detection\` (แยกจาก v9) ตัวติดตั้งมี AppId ใหม่ ไม่ทับ v9

## ทดสอบจากซอร์ส
```
python tests/test_teach_core.py      ตัดคลิป วาดกรอบ แบ่งชุด
python tests/test_teach_projects.py  สร้าง เลือก ลบ ออกจากโปรเจกต์
python tests/test_teach_ui.py        ครบทุกขั้นพร้อมสอนจริง (นานหลายนาทีบน CPU)
python tests/test_fly.py             หน้า Fly & Detect กับโดรนจำลอง (เปิดเว็บแคมของเครื่อง)
```

## โครงสร้าง
```
entry_point.py               จุดเริ่ม (Run.bat และตัว .exe)
ai_detection_lab/app_window.py  หน้าต่างหลัก + แถบสลับ 2 หน้า
ai_detection_lab/teach/      หน้า Train AI
ai_detection_lab/fly/        หน้า Fly & Detect (ใช้ ui/mission_control.py เดิมเป็นเครื่องยนต์ ไม่แสดงบนจอ)
ai_detection_lab/            โค้ด AI Detection Lab v9 (คัดลอกจาก Swarm Detection Lab ซึ่งแก้ชื่อไทยบนภาพแล้ว)
vendor/pyhulax/              SDK โดรน
AIR_Detection_Windows.spec, build_windows.*, installer.iss, setup_windows.bat, build_assets/   ชุด build
```

## ข้อจำกัดที่รู้แล้ว
- **ไม่มีปุ่ม GPU Pack:** โค้ดติดตั้ง GPU Pack ของ v9 ยังอยู่ แต่ไม่มีปุ่มเข้าในหน้า Train AI (ปุ่มเดิมอยู่ในหน้า Label & Train ที่ตัดออกไป)
  ตัวติดตั้งเป็น PyTorch รุ่น CPU จึงสอนบนการ์ดจอไม่ได้ ถ้าต้องการ CUDA ทั้งก้อน: `set TORCH_CUDA=cu126` ก่อน `build_windows.bat`
- หน้าที่เด้งขึ้นจากเครื่องยนต์เดิม (บังคับด้วยคีย์บอร์ด, เขียนโค้ด, รายงานหลังบิน) ยังเป็นหน้าตาเดิม
- ข้อความของ GPU Pack (ไม่มีปุ่มเข้า) ยังเป็นภาษาไทย
