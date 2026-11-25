from __future__ import annotations

import json
import threading
from pathlib import Path

from .. import config
from ..text_overlay import draw_boxes

# Name of the per-model confidence registry written by label_train_dialog.py
# right after a model is auto-calibrated (see _calibrate_confidence() there
# for how the number is derived). Lives inside config.MODEL_DIR, next to the
# .pt files themselves, so it travels with the models folder.
CONF_REGISTRY_FILENAME = "model_confidence.json"
# Stock checkpoints shipped with the app (not trained by a student).
STOCK_MODEL_NAMES = {"yolov8n.pt", "yolov8n-cls.pt", "yolo11n.pt"}

# Which .pt files the AI uses - chosen in "🧠 Choose Models" and saved next
# to the models: {"mode": "auto" | "all" | "selected", "selected": [names]}
#   auto     - (default, no file) student-trained models only; the stock
#              yolov8n.pt only when there is nothing else
#   all      - every .pt in the folder ("All.pt")
#   selected - only the ticked files
MODEL_SELECTION_FILENAME = "active_models.json"
SELECTION_MODES = ("auto", "all", "selected")


def load_model_selection(model_dir) -> dict:
    """Never raises; a missing/broken file means the default "auto"."""
    path = Path(model_dir) / MODEL_SELECTION_FILENAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        mode = data.get("mode", "auto")
        if mode not in SELECTION_MODES:
            mode = "auto"
        selected = [str(n) for n in data.get("selected", []) if isinstance(n, str)]
        return {"mode": mode, "selected": selected}
    except FileNotFoundError:
        pass
    except Exception as exc:
        print(f"⚠️ Could not read {MODEL_SELECTION_FILENAME}: {exc} - using all trained models")
    return {"mode": "auto", "selected": []}


def save_model_selection(model_dir, mode: str, selected=None) -> None:
    if mode not in SELECTION_MODES:
        raise ValueError(f"unknown mode {mode!r}")
    path = Path(model_dir) / MODEL_SELECTION_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"mode": mode, "selected": sorted(set(selected or []))},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)  # atomic: a detector reading mid-save never sees half a file


def apply_model_selection(files, selection, warn: bool = True) -> list:
    """Pure function: which of `files` (Paths) the AI should run."""
    mode = selection.get("mode", "auto")
    if mode == "all":
        return list(files)
    if mode == "selected":
        wanted = set(selection.get("selected", []))
        chosen = [p for p in files if p.name in wanted]
        if chosen:
            return chosen
        if warn:
            print("⚠️ None of the chosen models are in the models folder any more - using all trained models")
    # auto
    if getattr(config, "SKIP_STOCK_MODEL_WHEN_CUSTOM_EXISTS", True):
        custom = [p for p in files if p.name.lower() not in STOCK_MODEL_NAMES]
        if custom:
            return custom
    return list(files)


class YOLODetector:
    """Loads every YOLO .pt model found in config.MODEL_DIR.

    Models are hot-discovered: when a new .pt file is copied into the models
    folder while the Lab is running, the next inference cycle discovers it
    and loads it automatically. Existing loaded models are kept in memory.

    Confidence threshold: each model uses its own calibrated threshold from
    model_confidence.json when one exists (written automatically after
    training - see label_train_dialog.py's _calibrate_confidence()), and
    falls back to the single global config.YOLO_CONF_THRES for any model
    that has no calibration on record (manually dropped-in models, or ones
    trained before this feature existed). One fixed number for every model
    was measured to be a poor fit - see the comment above MIN_AUTO_CONF in
    label_train_dialog.py for the real numbers that motivated this.
    """

    def __init__(self):
        self.model_dir = Path(config.MODEL_DIR)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.models: dict[str, YOLO] = {}
        self.model_paths: dict[str, Path] = {}
        self._known_files: set[Path] = set()
        self._lock = threading.RLock()
        self._device_for = object()   # config.YOLO_DEVICE the cache below was resolved for
        self.device = None            # what predict() really runs on ("mps" / "0" / "cpu")
        self.device_note = ""         # e.g. "no cuda here - using CPU" (shown once)
        # Swarm + AI: many cameras share one CPU, so every model runs at most
        # this size (None = no cap). Set by swarm_ai/detector.py.
        self.imgsz_cap = None
        self.refresh_models(force=True)

        if not self.models:
            raise FileNotFoundError(
                f"No YOLO .pt models found in: {self.model_dir}\n"
                "Put one or more trained .pt files into the models folder."
            )

    def _scan_files(self) -> list[Path]:
        files = sorted(
            (p for p in self.model_dir.glob("*.pt") if p.is_file()),
            key=lambda p: p.name.lower(),
        )
        # "🧠 Choose Models" selection. Re-read only when the file changes
        # (this runs on every analysed frame).
        sel_path = self.model_dir / MODEL_SELECTION_FILENAME
        try:
            stamp = sel_path.stat().st_mtime_ns
        except OSError:
            stamp = None
        changed = stamp != getattr(self, "_selection_stamp", "unset")
        if changed:
            self._selection_stamp = stamp
            self._selection = load_model_selection(self.model_dir)
        # Warn only when the choice changes - this runs every frame.
        return apply_model_selection(files, self._selection, warn=changed)

    def _load_conf_registry(self) -> dict:
        """Reads models/model_confidence.json fresh each call (it's a tiny
        file - a handful of small dicts - so this costs far less than a
        single frame of inference). Never raises: any problem reading or
        parsing it just means "no per-model overrides available right now",
        never a crash mid-detection."""
        registry_path = self.model_dir / CONF_REGISTRY_FILENAME
        try:
            data = json.loads(registry_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except FileNotFoundError:
            pass
        except Exception as exc:
            print(f"⚠️ Could not read {CONF_REGISTRY_FILENAME}: {exc}")
        return {}

    def refresh_models(self, force: bool = False) -> list[str]:
        """Discover and load newly added .pt files. Returns newly loaded names."""
        files = self._scan_files()
        file_set = set(files)
        with self._lock:
            if not force and file_set == self._known_files:
                return []
            self._known_files = file_set

            # Unload models that are no longer wanted (file removed, or the
            # stock model stepping aside for a newly trained one).
            wanted = {p.resolve().as_posix() for p in files}
            for key in [k for k in self.models if k not in wanted]:
                print(f"   ⏏️ Unloading {self.model_paths[key].name}")
                self.models.pop(key, None)
                self.model_paths.pop(key, None)

            registry = self._load_conf_registry()
            loaded_now: list[str] = []
            for path in files:
                key = path.resolve().as_posix()
                if key in self.models:
                    continue
                try:
                    print(f"🤖 Loading AI LAB YOLO model: {path.name}")
                    # Model selection reads the lightweight helpers above.
                    # Import the inference runtime only when loading weights.
                    from ultralytics import YOLO

                    model = YOLO(str(path))
                    # MODEL_DIR is seeded (see config.ensure_default_models_seeded)
                    # with stock checkpoints; a classify-task .pt (e.g. an older
                    # yolov8n-cls.pt left in the folder) is never something the
                    # live detector should run.
                    # Loading it here anyway was harmless for detections() (which
                    # already skips boxes-less results), but not for annotate():
                    # result.plot() on a classify result draws ultralytics' own
                    # top-5 ImageNet label list straight onto the frame -
                    # confirmed via a real screenshot showing "solar_dish 0.11",
                    # "space_shuttle 0.04" etc. overlaid during Test with a Video
                    # Clip. Any non-"detect" task model dropped into MODEL_DIR
                    # (by seeding, or by hand) would hit the same thing, so this
                    # skips by task rather than by filename.
                    if getattr(model, "task", None) != "detect":
                        print(
                            f"   ⏭️ Skipping {path.name}: task={getattr(model, 'task', None)!r}, "
                            "not a detection model (MODEL_DIR is for YOLO detect models only)"
                        )
                        continue
                    self.models[key] = model
                    self.model_paths[key] = path
                    print(f"   📚 {path.name} classes: {model.names}")
                    entry = registry.get(path.name)
                    if entry and isinstance(entry.get("conf_thres"), (int, float)):
                        print(
                            f"   🎯 {path.name} using auto-calibrated confidence "
                            f"{entry['conf_thres']:.2f} (global default is "
                            f"{config.YOLO_CONF_THRES:.2f})"
                        )
                    else:
                        print(
                            f"   🎯 {path.name} has no calibration on record - "
                            f"using the global default confidence ({config.YOLO_CONF_THRES:.2f})"
                        )
                    loaded_now.append(path.name)
                except Exception as exc:
                    print(f"❌ Failed to load model {path.name}: {exc}")

            return loaded_now

    def resolve_device(self) -> str:
        """config.YOLO_DEVICE -> a device this computer really has. Resolved
        once per setting (not per frame); ultralytics' own default would be
        cpu on every Mac and an "Invalid CUDA device" error on every frame
        if cuda was chosen without an NVIDIA card (see ai/devices.py)."""
        wanted = getattr(config, "YOLO_DEVICE", None)
        if wanted != self._device_for:
            from . import devices
            self.device, self.device_note = devices.resolve(wanted)
            self._device_for = wanted
            print(f"🖥️ AI detection on {devices.describe(self.device)}"
                  + (f" — {self.device_note}" if self.device_note else ""))
        return self.device

    def model_status(self) -> list[dict]:
        with self._lock:
            registry = self._load_conf_registry()
            rows = []
            for key, model in self.models.items():
                path = self.model_paths[key]
                entry = registry.get(path.name) or {}
                conf_thres = entry.get("conf_thres", config.YOLO_CONF_THRES)
                rows.append(
                    {
                        "name": path.name,
                        "path": str(path),
                        "classes": dict(model.names) if hasattr(model, "names") else {},
                        "conf_thres": conf_thres,
                        "conf_calibrated": entry.get("conf_thres") is not None,
                    }
                )
            return rows

    def predict(self, frame):
        new_models = self.refresh_models()
        if new_models:
            print(f"🆕 AUTO-DETECTED MODEL(S): {', '.join(new_models)}")

        registry = self._load_conf_registry()

        base_kwargs = {
            "source": frame,
            "imgsz": config.YOLO_IMGSZ,
            "iou": float(getattr(config, "YOLO_NMS_IOU", 0.5)),
            "verbose": False,
        }
        base_kwargs["device"] = self.resolve_device()

        with self._lock:
            items = list(self.models.items())

        results = []
        for key, model in items:
            path = self.model_paths[key]
            entry = registry.get(path.name)
            if entry and isinstance(entry.get("conf_thres"), (int, float)):
                conf = float(entry["conf_thres"])
            else:
                conf = config.YOLO_CONF_THRES
            kwargs = dict(base_kwargs, conf=conf)
            # A model trained small (the kid wizard's quick mode) misses objects
            # when run at a larger size: they look bigger than anything it saw.
            if entry and isinstance(entry.get("imgsz"), int):
                kwargs["imgsz"] = entry["imgsz"]
            if self.imgsz_cap:
                kwargs["imgsz"] = min(int(kwargs["imgsz"]), int(self.imgsz_cap))
            result = model.predict(**kwargs)[0]
            results.append((path.name, result))
        return results

    @staticmethod
    def annotate(results, frame=None):
        if not results:
            return frame
        annotated = frame
        for _, result in results:
            names = (result.names or {}).values() if isinstance(result.names, dict) else (result.names or [])
            if all(str(n).isascii() for n in names):
                annotated = result.plot(img=annotated)
            else:
                # ultralytics draws non-ASCII names by downloading a font on
                # every plot() - a stall per frame on an offline classroom
                # machine - so Thai names are drawn with a local font instead.
                annotated = draw_boxes(annotated if annotated is not None else result.orig_img,
                                       YOLODetector.detections([("", result)]))
        return annotated

    @staticmethod
    def detections(results):
        out: list[dict] = []
        for model_name, result in results:
            names = result.names or {}
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0]) if box.cls is not None else -1
                conf = float(box.conf[0]) if box.conf is not None else 0.0
                xyxy = [float(v) for v in box.xyxy[0].tolist()]
                if isinstance(names, dict):
                    name = names.get(cls_id, str(cls_id))
                else:
                    name = names[cls_id] if 0 <= cls_id < len(names) else str(cls_id)
                out.append(
                    {
                        "model_name": model_name,
                        "class_id": cls_id,
                        "class_name": name,
                        "confidence": conf,
                        "xyxy": xyxy,
                    }
                )
        return out

    @classmethod
    def count(cls, results) -> int:
        return len(cls.detections(results))
