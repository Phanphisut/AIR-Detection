"""Explicit, hardware-free verification of the packaged application.

The normal launcher never enters this path. Invoke the built executable with
--smoke-test OUTPUT_DIRECTORY to render its UI and run bundled YOLO weights
against a synthetic image, without touching the operator's saved data.
"""
from pathlib import Path
import json
import os
import sys
import traceback


def run_smoke_test(output_directory):
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {"frozen": bool(getattr(sys, "frozen", False)), "hardware_used": False}
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ["YOLO_CONFIG_DIR"] = str(output / "ultralytics")
    os.environ["MPLCONFIGDIR"] = str(output / "matplotlib")
    os.environ["TORCH_HOME"] = str(output / "torch")
    try:
        from . import config
        config.DATA_ROOT = output / "test_data"
        config.MODEL_DIR = config.DATA_ROOT / "models"
        config.CAPTURE_DIR = config.DATA_ROOT / "captures"
        config.DRONE_PHOTOS_DIR = config.DATA_ROOT / "drone_photos"
        config.USER_OVERRIDES_PATH = output / "user_config_overrides.json"

        from PySide6 import QtCore, QtWidgets
        from .ui.mission_control import MissionControlWindow
        app = QtWidgets.QApplication([])
        window = MissionControlWindow()
        window.show()
        for _ in range(5):
            app.processEvents()
        report["window_title"] = window.windowTitle()
        report["ui_rendered"] = window.grab().save(str(output / "packaged-ui.png"))

        # 🗺️ Survey Route in 3D: walk the waypoint reports a real grid flight sends.
        from .grid.planner import generate_waypoint_objects
        from .ui.route_3d import SurveyRouteViews
        window.grid_btn.click()
        for _ in range(5):
            app.processEvents()
        views = window._grid_preview_widget
        route_points = generate_waypoint_objects()
        for wp in route_points:
            window._on_grid_progress_ui(wp.index, len(route_points), wp.x_cm, wp.y_cm, wp.heading, wp.turn)
        import time as _time
        _end = _time.monotonic() + 1.0
        while _time.monotonic() < _end:
            app.processEvents()
            _time.sleep(0.03)
        route_pose = views.view_3d.track.snapshot().pose if isinstance(views, SurveyRouteViews) else None
        report["route_3d_passed"] = (route_pose is not None and route_pose.flying
                                     and views.currentWidget() is views.view_3d
                                     and abs(route_pose.x - route_points[-1].x_cm) < 1)
        report["route_3d_rendered"] = window._grid_preview_dialog.grab().save(str(output / "packaged-route-3d.png"))
        window._grid_preview_dialog.close()
        app.processEvents()
        report["camera_off"] = window.camera is None
        report["drone_disconnected"] = not window.drone.connected
        report["takeoff_disabled"] = not window.auto_btn.isEnabled()
        report["qt_version"] = QtCore.qVersion()
        from .ui.label_train_dialog import LabelTrainDialog
        studio = LabelTrainDialog()
        studio.showNormal()
        studio.resize(1360, 900)
        studio.show()
        rendered = []
        for page, name in ((4, "projects"), (0, "datasets"), (2, "train"), (3, "models")):
            studio._navigate(page)
            for _ in range(5):
                app.processEvents()
            rendered.append(studio.grab().save(str(output / f"packaged-studio-{name}.png")))
        report["studio_pages_rendered"] = all(rendered)
        report["studio_training_guard"] = not studio.train_btn.isEnabled()
        studio.close()

        # 🐍 Flight Plan: ✔ Check Code + 🧊 3D simulation (virtual drone only).
        from .drone.simulated_controller import SimulatedDroneController
        from .missions.flight_sim import check_plan
        from .ui import mission_dialogs
        square = "drone.takeoff()\nfor i in range(4):\n    drone.forward(100)\n    drone.turn_right(90)\ndrone.land()\n"
        plan = check_plan(square)
        report["flight_check_passed"] = (plan.ok and plan.commands == 10
                                         and not check_plan("drone.takeoff()\ndrone.fowrad(5)\n").ok)
        mission_dialogs.DEFAULT_PLAN = output / "flight_plan_smoke.py"
        sim_drone = SimulatedDroneController()
        fp = mission_dialogs.FlightPlanDialog(sim_drone, lambda: [], lambda: None)
        fp.resize(1320, 860)
        fp.show()
        fp.editor.setPlainText(square)
        fp.speed_combo.setCurrentIndex(fp.speed_combo.count() - 1)
        fp.simulate_plan()
        import time
        deadline = time.monotonic() + 30
        while fp._runner is not None and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.03)
        pose = fp.track.snapshot().pose
        report["flight_sim_passed"] = (fp._runner is None and not pose.flying
                                       and abs(pose.x) < 1 and abs(pose.y) < 1 and not sim_drone.connected)
        for _ in range(5):
            app.processEvents()
        report["flight_sim_rendered"] = fp.grab().save(str(output / "packaged-flight-sim.png"))

        # 🎯 Mission Guide: every answer key earns ★★★, the window grades one.
        from .missions.challenges import MISSIONS, MISSIONS_BY_ID, grade
        report["missions_graded"] = all(grade(m, m.solution).stars == 3 for m in MISSIONS)
        fp.guide.select("rings")
        fp.editor.setPlainText(MISSIONS_BY_ID["rings"].solution)
        result = fp.grade_mission()
        for _ in range(5):
            app.processEvents()
        report["mission_guide_passed"] = (result is not None and result.stars == 3
                                          and len(fp.track.snapshot().world) == 4)
        fp.grab().save(str(output / "packaged-missions.png"))
        fp.close()
        window.close()
        app.processEvents()

        # AIR Detection: the main window and its two pages (teach AI, single drone + AI).
        from .app_window import AppWindow
        from .teach import project as tp
        tp.TRAINING_ROOT = config.DATA_ROOT / "training_sessions"
        tp.TRAINING_ROOT.mkdir(parents=True, exist_ok=True)
        main = AppWindow()
        main.resize(1400, 900)
        main.show()
        main.switch("fly")
        main.pages["teach"].ensure_loaded()  # หน้าแรกคือ Fly & Detect: สั่งโหลดหน้า Train AI ด้วย
        deadline = time.monotonic() + 120
        while (main.pages["fly"].page is None or main.pages["teach"].page is None) and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.03)
        fly_page = main.pages["fly"].page
        report["kid_pages_loaded"] = fly_page is not None and main.pages["teach"].page is not None
        report["kid_fly_engine_hidden"] = fly_page is not None and not fly_page.engine.isVisible()
        report["kid_pages_rendered"] = main.grab().save(str(output / "packaged-air-detection-fly.png"))
        main.switch("teach")
        app.processEvents()
        main.grab().save(str(output / "packaged-air-detection-teach.png"))
        main.close()
        app.processEvents()

        import numpy as np
        import torch
        from ultralytics import YOLO
        weights = config.PROJECT_ROOT / "models" / "yolov8n.pt"
        model = YOLO(str(weights))
        results = model.predict(np.zeros((320, 320, 3), dtype=np.uint8),
                                imgsz=320, device="cpu", verbose=False, save=False)
        report["torch_version"] = torch.__version__
        report["bundled_model"] = weights.is_file()
        report["inference_passed"] = len(results) == 1

        # MPS / CUDA / CPU: the device "auto" picks really runs, for detection
        # and for a 1-epoch training on synthetic TEST FIXTURE images.
        from .ai import devices
        auto_device, _ = devices.resolve("auto")
        report["device_auto"] = auto_device
        report["mps_available"] = devices.mps_available()
        dev_model = YOLO(str(weights))
        dev_model.predict(np.zeros((320, 320, 3), dtype=np.uint8), imgsz=320, device=auto_device,
                          verbose=False, save=False)
        report["device_inference_passed"] = str(dev_model.predictor.device).startswith(
            "mps" if auto_device == "mps" else "cuda" if auto_device.isdigit() else "cpu")
        import cv2 as _cv2
        from .ui.label_train_dialog import TrainAIThread
        frames = output / "device_train" / "frames"
        frames.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(1)
        for i in range(6):
            img = rng.integers(0, 90, (320, 320, 3), dtype=np.uint8)
            x, y = int(rng.integers(60, 260)), int(rng.integers(60, 260))
            _cv2.circle(img, (x, y), 30, (0, 140, 255), -1)
            _cv2.putText(img, "TEST FIXTURE", (5, 20), 0, .5, (255, 255, 255), 1)
            _cv2.imwrite(str(frames / f"f{i}.jpg"), img)
            (frames / f"f{i}.txt").write_text(f"0 {x / 320:.4f} {y / 320:.4f} {60 / 320:.4f} {60 / 320:.4f}\n")
        done = {}
        trainer = TrainAIThread(frames, ["cone"], output / "device_train" / "dataset", output / "device_train" / "runs",
                                "train", 1, on_progress=lambda *_: None, on_status=lambda *_: None,
                                on_error=lambda *_: None, on_finished=lambda p, c: done.update(best=p), device="auto")
        trainer.start()
        trainer.join(600)
        report["device_train_used"] = trainer.device_used
        report["device_train_passed"] = done.get("best") is not None and trainer.device_used == auto_device

        # Studio tools: offline exports, Platform SDK (v6.0.0: no SAM / auto-label).
        from .ui import studio_tools
        report["no_auto_label"] = not any(hasattr(studio, a) for a in ("_auto_label", "_run_sam", "sam_tool_btn", "auto_label_btn"))
        exports = output / "exports"
        report["exports"] = {}
        for fmt in ("pt",):
            try:
                report["exports"][fmt] = studio_tools.export_model(weights, fmt, exports).exists()
            except Exception as exc:
                report["exports"][fmt] = f"failed: {exc}"
        report["exports_passed"] = all(v is True for v in report["exports"].values())
        report["passed"] = all(report[key] for key in (
            "frozen", "ui_rendered", "camera_off", "drone_disconnected",
            "takeoff_disabled", "studio_pages_rendered", "studio_training_guard",
            "flight_check_passed", "flight_sim_passed", "flight_sim_rendered",
            "route_3d_passed", "route_3d_rendered",
            "missions_graded", "mission_guide_passed",
            "bundled_model", "inference_passed", "no_auto_label",
            "device_inference_passed", "device_train_passed",
            "exports_passed", "kid_pages_loaded", "kid_fly_engine_hidden", "kid_pages_rendered"))
    except BaseException:
        report["passed"] = False
        report["error"] = traceback.format_exc()
    (output / "smoke-test.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if report["passed"] else 1
