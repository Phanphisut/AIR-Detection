from . import config
from .ui.main_window import run

if __name__ == "__main__":
    config.ensure_default_models_seeded()
    raise SystemExit(run())
