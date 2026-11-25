"""PyInstaller runtime hook: keep pyhulax's stdlib-`logging` recovery
working in a frozen build.

pyhulax/__init__.py does `import logging` near its own top, then later
does `from pyhulax.logging.command_logger import CommandLogger` (pyhulax
ships its own subpackage also named `logging`). Importing that submodule
for the FIRST time makes Python's import system set `logging` as an
attribute on the `pyhulax` package object (standard behaviour for any
`pkg.child` submodule import) - and since `pyhulax/__init__.py` is
executing WITH `sys.modules['pyhulax'].__dict__` as its own globals(),
that silently overwrites the plain `logging` name the file bound earlier
to the real stdlib module. This is a real bug in pyhulax itself, present
on every platform, not a PyInstaller artifact.

pyhulax's own __init__.py already defends against exactly this
(`if getattr(logging, "__name__", "") != "logging": ...` -> re-resolve via
`importlib.machinery.PathFinder.find_spec("logging", [sysconfig stdlib
path])`), and that recovery works fine in a normal venv, where the stdlib
exists as real loose .py files on disk. It only fails frozen, where there
is no such on-disk stdlib for PathFinder to search - `find_spec` returns
None and pyhulax raises ImportError instead of recovering.

Patching PathFinder.find_spec here to hand back the already-correct,
already-imported stdlib `logging` module's own spec for exactly this one
lookup lets pyhulax's existing recovery path succeed unchanged, without
touching sys.modules['logging'] itself (nothing else in the app is
affected - pyhulax only ever corrupts its own local name, never the
global one).
"""
import logging as _stdlib_logging
from importlib.machinery import PathFinder

_original_find_spec = PathFinder.find_spec.__func__


def _find_spec_with_logging_fallback(cls, name, path=None, target=None):
    if name == "logging" and path is not None:
        return _stdlib_logging.__spec__
    return _original_find_spec(cls, name, path, target)


PathFinder.find_spec = classmethod(_find_spec_with_logging_fallback)
