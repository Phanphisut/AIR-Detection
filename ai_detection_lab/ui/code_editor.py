"""
Code Editor popup for AI Detection Lab.

Opened from a button on the main window (does NOT sit inline as a tab in
the mission dashboard) - clicking "Code Editor" pops up a separate window
with two tabs:

    - Quick Parameters: friendly, labelled controls (spin boxes/checkboxes/
      dropdowns) for a curated set of the most commonly-tuned values in
      config.py, generated straight from that file so it never drifts out
      of sync. Saving rewrites only the changed value on each line, keeping
      every comment and the file's layout intact. Takes effect on the NEXT
      run of the program (values already latched into live objects such as
      HulaDroneController are not touched while the app is running). Any
      constant not listed here can still be edited directly as text via the
      Full Code Editor tab below (open ai_detection_lab/config.py).
    - Full Code Editor: a lightweight, VS-Code-ish editor - file explorer
      rooted at the project folder, a tabbed text editor with line numbers
      and Python syntax highlighting, and Run / Run in Terminal / Stop for
      executing the file that's open. No LSP, no autocomplete - this is
      for quick edits and re-runs, not full IDE work.

(The "test the AI with your laptop's webcam" feature lives on the main
Mission Control dashboard itself now - see main_window.py's WebcamAIThread
and the START/STOP WEBCAM TEST controls - not in this popup.)
"""
from __future__ import annotations

import ast
import json
import platform
import re
import sys
import subprocess
import threading
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from .. import config
from ..i18n import tr

# Windows systems set to a Thai display language can have "native digits"
# turned on in Regional Settings, which makes Qt's default locale format
# numbers with Thai numerals (๐-๙) instead of 0-9. Every numeric input in
# this module is explicitly pinned to this locale so the digits are always
# plain 0-9 regardless of the OS's regional/number settings.
EN_US_LOCALE = QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates)


# ---------------------------------------------------------------------------
# Project root discovery
# ---------------------------------------------------------------------------
def project_root() -> Path:
    """Best-effort root folder to show in the file explorer.

    In normal (non-frozen) operation this file lives at
    ``<root>/ai_detection_lab/ui/code_editor.py``, so the project root is
    two levels up. If this ever runs from a PyInstaller-frozen build,
    fall back to the folder containing the executable - except on macOS,
    where a loose, editable copy of ai_detection_lab/ (see
    AI_Detection_Lab.spec's `datas`) is shipped under Contents/Resources/
    instead, since sys.executable itself sits alone in Contents/MacOS/ on
    that platform (see config.py's matching PROJECT_ROOT fix).
    """
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        if sys.platform == "darwin":
            return exe_dir.parent / "Resources"
        if sys.platform == "win32" and (exe_dir / "_internal").is_dir():
            # PyInstaller 6.x onedir builds put the loose ai_detection_lab/
            # source copy (and every other `datas` entry) under _internal/,
            # not directly beside the exe - see config.py's matching
            # PROJECT_ROOT fix for how this was confirmed.
            return exe_dir / "_internal"
        return exe_dir
    try:
        return Path(__file__).resolve().parents[2]
    except IndexError:
        return Path.cwd()


RUNNABLE_SUFFIXES = {".py", ".bat", ".cmd"}


# ---------------------------------------------------------------------------
# Editor widget: QPlainTextEdit + line-number gutter
# (standard Qt "Code Editor" recipe, trimmed down)
# ---------------------------------------------------------------------------
class _LineNumberArea(QtWidgets.QWidget):
    def __init__(self, editor: "CodeTextEdit"):
        super().__init__(editor)
        self._editor = editor

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(self._editor.line_number_area_width(), 0)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        self._editor.line_number_area_paint_event(event)


class CodeTextEdit(QtWidgets.QPlainTextEdit):
    """A plain-text editor with line numbers and 4-space tabs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        font = QtGui.QFont("Cascadia Mono")
        if not font.exactMatch():
            font = QtGui.QFont("Consolas")
        font.setStyleHint(QtGui.QFont.Monospace)
        font.setPointSize(10)
        self.setFont(font)
        self.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        tab_stop = self.fontMetrics().horizontalAdvance(" ") * 4
        self.setTabStopDistance(tab_stop)

        self._gutter = _LineNumberArea(self)
        self.blockCountChanged.connect(self._update_gutter_width)
        self.updateRequest.connect(self._update_gutter)
        self.cursorPositionChanged.connect(self._highlight_current_line)
        self._update_gutter_width(0)
        self._highlight_current_line()

    # -- gutter plumbing -----------------------------------------------
    def line_number_area_width(self) -> int:
        digits = max(2, len(str(max(1, self.blockCount()))))
        return 12 + self.fontMetrics().horizontalAdvance("9") * digits

    def _update_gutter_width(self, _new_block_count: int) -> None:
        self.setViewportMargins(self.line_number_area_width(), 0, 0, 0)

    def _update_gutter(self, rect: QtCore.QRect, dy: int) -> None:
        if dy:
            self._gutter.scroll(0, dy)
        else:
            self._gutter.update(0, rect.y(), self._gutter.width(), rect.height())
        if rect.contains(self.viewport().rect()):
            self._update_gutter_width(0)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        cr = self.contentsRect()
        self._gutter.setGeometry(
            QtCore.QRect(cr.left(), cr.top(), self.line_number_area_width(), cr.height())
        )

    def _highlight_current_line(self) -> None:
        selections = []
        if not self.isReadOnly():
            sel = QtWidgets.QTextEdit.ExtraSelection()
            sel.format.setBackground(QtGui.QColor("#132030"))
            sel.format.setProperty(QtGui.QTextFormat.FullWidthSelection, True)
            sel.cursor = self.textCursor()
            sel.cursor.clearSelection()
            selections.append(sel)
        self.setExtraSelections(selections)

    def line_number_area_paint_event(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self._gutter)
        painter.fillRect(event.rect(), QtGui.QColor("#0b1118"))
        block = self.firstVisibleBlock()
        block_number = block.blockNumber()
        top = self.blockBoundingGeometry(block).translated(self.contentOffset()).top()
        bottom = top + self.blockBoundingRect(block).height()
        text_color = QtGui.QColor("#4c6379")
        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                painter.setPen(text_color)
                painter.drawText(
                    0,
                    int(top),
                    self._gutter.width() - 6,
                    self.fontMetrics().height(),
                    QtCore.Qt.AlignRight,
                    str(block_number + 1),
                )
            block = block.next()
            top = bottom
            bottom = top + self.blockBoundingRect(block).height()
            block_number += 1

    # -- convenience: Tab inserts 4 spaces, Enter keeps indentation -----
    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key_Tab and not self.textCursor().hasSelection():
            self.insertPlainText("    ")
            return
        if event.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
            cursor = self.textCursor()
            block_text = cursor.block().text()
            indent = block_text[: len(block_text) - len(block_text.lstrip(" "))]
            if block_text.rstrip().endswith(":"):
                indent += "    "
            super().keyPressEvent(event)
            self.insertPlainText(indent)
            return
        super().keyPressEvent(event)


# ---------------------------------------------------------------------------
# Minimal Python syntax highlighter (single-line rules; good enough for a
# lightweight in-app editor, not a full tokenizer)
# ---------------------------------------------------------------------------
class PythonHighlighter(QtGui.QSyntaxHighlighter):
    KEYWORDS = [
        "False", "None", "True", "and", "as", "assert", "async", "await",
        "break", "class", "continue", "def", "del", "elif", "else",
        "except", "finally", "for", "from", "global", "if", "import",
        "in", "is", "lambda", "nonlocal", "not", "or", "pass", "raise",
        "return", "try", "while", "with", "yield", "self",
    ]

    def __init__(self, document: QtGui.QTextDocument):
        super().__init__(document)

        def fmt(color: str, bold: bool = False) -> QtGui.QTextCharFormat:
            f = QtGui.QTextCharFormat()
            f.setForeground(QtGui.QColor(color))
            if bold:
                f.setFontWeight(QtGui.QFont.Bold)
            return f

        self._keyword_fmt = fmt("#569CD6", bold=True)
        self._string_fmt = fmt("#CE9178")
        self._comment_fmt = fmt("#6A9955")
        self._number_fmt = fmt("#B5CEA8")
        self._def_fmt = fmt("#DCDCAA", bold=True)
        self._decorator_fmt = fmt("#C586C0")

        self._rules: list[tuple[QtCore.QRegularExpression, QtGui.QTextCharFormat]] = []
        for kw in self.KEYWORDS:
            self._rules.append(
                (QtCore.QRegularExpression(r"\b%s\b" % kw), self._keyword_fmt)
            )
        self._rules.append((QtCore.QRegularExpression(r"\b[0-9]+(\.[0-9]+)?\b"), self._number_fmt))
        self._rules.append((QtCore.QRegularExpression(r"@\w+"), self._decorator_fmt))
        self._def_re = QtCore.QRegularExpression(r"\b(def|class)\s+(\w+)")
        self._string_re = QtCore.QRegularExpression(r"(\"[^\"\\]*(\\.[^\"\\]*)*\")|('[^'\\]*(\\.[^'\\]*)*')")
        self._comment_re = QtCore.QRegularExpression(r"#[^\n]*")

    def highlightBlock(self, text: str) -> None:
        for pattern, fmt_ in self._rules:
            it = pattern.globalMatch(text)
            while it.hasNext():
                m = it.next()
                self.setFormat(m.capturedStart(), m.capturedLength(), fmt_)

        it = self._def_re.globalMatch(text)
        while it.hasNext():
            m = it.next()
            self.setFormat(m.capturedStart(2), m.capturedLength(2), self._def_fmt)

        it = self._string_re.globalMatch(text)
        while it.hasNext():
            m = it.next()
            self.setFormat(m.capturedStart(), m.capturedLength(), self._string_fmt)

        it = self._comment_re.globalMatch(text)
        while it.hasNext():
            m = it.next()
            self.setFormat(m.capturedStart(), m.capturedLength(), self._comment_fmt)
            break  # only one comment possible per line, and it wins last


# ---------------------------------------------------------------------------
# Main panel: file tree + tabbed editors + run console
# ---------------------------------------------------------------------------
class _ScriptStopped(BaseException):
    """Raised inside an in-process script when ⏹ Stop is pressed."""


class _InProcessScript(QtCore.QObject):
    """Runs a .py file inside this app (used by the packaged .exe/.app).

    A packaged app has no separate python.exe: sys.executable IS the app, so
    the normal "python script.py" subprocess would just launch a second copy
    of the app. Instead the script runs here, on a worker thread, with the
    Python + libraries bundled in the app (cv2, numpy, ultralytics, ...).
    Only that thread's print() output goes to the console; ⏹ Stop interrupts
    the script at its next line.
    """

    output = QtCore.Signal(str)
    finished = QtCore.Signal(int)

    def __init__(self, path: Path, parent=None):
        super().__init__(parent)
        self.path = path
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True, name="AI-LAB-SCRIPT")
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _tracer(self, frame, event, arg):
        if self._stop.is_set():
            raise _ScriptStopped()
        return self._tracer if frame.f_code.co_filename == str(self.path) else None

    def _run(self):
        import runpy
        import traceback

        me = threading.get_ident()
        emit = self.output.emit

        class _Stream:
            def __init__(self, fallback):
                self.fallback = fallback

            def write(self, text):
                if threading.get_ident() == me:
                    emit(text)
                elif self.fallback is not None:
                    self.fallback.write(text)
                return len(text)

            def flush(self):
                pass

        old_out, old_err, old_argv = sys.stdout, sys.stderr, sys.argv
        sys.stdout, sys.stderr = _Stream(old_out), _Stream(old_err)
        added = str(self.path.parent)
        sys.path.insert(0, added)
        code = 0
        try:
            sys.argv = [str(self.path)]
            sys.settrace(self._tracer)
            runpy.run_path(str(self.path), run_name="__main__")
        except _ScriptStopped:
            emit("\n[stopped]\n")
            code = -1
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 0
        except BaseException:
            emit(traceback.format_exc())
            code = 1
        finally:
            sys.settrace(None)
            sys.stdout, sys.stderr, sys.argv = old_out, old_err, old_argv
            try:
                sys.path.remove(added)
            except ValueError:
                pass
            self.finished.emit(code)


class CodeEditorPanel(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # The built app shows the student's own work folder (Flight Plans,
        # scripts), not the app's internals - those are compiled (v6.0.0).
        if getattr(sys, "frozen", False):
            from .. import config
            config.DATA_ROOT.mkdir(parents=True, exist_ok=True)
            self._root = config.DATA_ROOT
        else:
            self._root = project_root()
        self._process: QtCore.QProcess | None = None
        self._open_paths: dict[int, Path | None] = {}  # tab index -> path (None = untitled)
        self._build_ui()

    # -- UI ---------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(8)

        # Toolbar
        toolbar = QtWidgets.QHBoxLayout()
        toolbar.setSpacing(6)

        self.new_btn = QtWidgets.QPushButton(tr("📄 New"))
        self.open_btn = QtWidgets.QPushButton(tr("📂 Open"))
        self.save_btn = QtWidgets.QPushButton(tr("💾 Save"))
        self.save_as_btn = QtWidgets.QPushButton(tr("💾 Save As"))
        self.run_btn = QtWidgets.QPushButton(tr("▶ Run"))
        self.run_btn.setObjectName("primaryButton")
        self.run_terminal_btn = QtWidgets.QPushButton(tr("🖥 Run in Terminal"))
        self.stop_btn = QtWidgets.QPushButton(tr("■ Stop"))
        self.stop_btn.setObjectName("dangerButton")
        self.stop_btn.setEnabled(False)

        for b in (
            self.new_btn, self.open_btn, self.save_btn, self.save_as_btn,
            self.run_btn, self.run_terminal_btn, self.stop_btn,
        ):
            toolbar.addWidget(b)

        toolbar.addStretch()
        self.current_file_label = QtWidgets.QLabel(tr("(no file open)"))
        self.current_file_label.setObjectName("muted")
        toolbar.addWidget(self.current_file_label)
        layout.addLayout(toolbar)

        # Body: [file tree | editor tabs / console]
        body_split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        body_split.setChildrenCollapsible(False)
        layout.addWidget(body_split, 1)

        # File tree
        self.fs_model = QtWidgets.QFileSystemModel()
        self.fs_model.setRootPath(str(self._root))
        self.tree = QtWidgets.QTreeView()
        self.tree.setModel(self.fs_model)
        self.tree.setRootIndex(self.fs_model.index(str(self._root)))
        self.tree.setMinimumWidth(220)
        self.tree.setMaximumWidth(340)
        self.tree.setHeaderHidden(False)
        for col in (1, 2, 3):
            self.tree.hideColumn(col)
        self.tree.doubleClicked.connect(self._on_tree_double_clicked)
        body_split.addWidget(self.tree)

        # Right side: editor tabs over an output console
        right_split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        right_split.setChildrenCollapsible(False)
        body_split.addWidget(right_split)
        body_split.setSizes([260, 900])

        self.editor_tabs = QtWidgets.QTabWidget()
        self.editor_tabs.setTabsClosable(True)
        self.editor_tabs.tabCloseRequested.connect(self._close_tab)
        self.editor_tabs.currentChanged.connect(self._on_current_tab_changed)
        right_split.addWidget(self.editor_tabs)

        console_wrap = QtWidgets.QWidget()
        console_layout = QtWidgets.QVBoxLayout(console_wrap)
        console_layout.setContentsMargins(0, 0, 0, 0)
        console_layout.setSpacing(4)
        console_label = QtWidgets.QLabel(tr("OUTPUT"))
        console_label.setObjectName("sectionTitle")
        console_layout.addWidget(console_label)
        self.console = QtWidgets.QPlainTextEdit()
        self.console.setObjectName("terminal")
        self.console.setReadOnly(True)
        console_layout.addWidget(self.console)
        right_split.addWidget(console_wrap)
        right_split.setSizes([560, 180])

        # Start with one empty, untitled tab so the panel is never blank.
        self._new_file()

        # Signals
        self.new_btn.clicked.connect(self._new_file)
        self.open_btn.clicked.connect(self._open_file_dialog)
        self.save_btn.clicked.connect(self._save_current)
        self.save_as_btn.clicked.connect(self._save_current_as)
        self.run_btn.clicked.connect(self._run_current)
        self.run_terminal_btn.clicked.connect(self._run_current_in_terminal)
        self.stop_btn.clicked.connect(self._stop_process)

        QtGui.QShortcut(QtGui.QKeySequence.Save, self, activated=self._save_current)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Shift+S"), self, activated=self._save_current_as)
        QtGui.QShortcut(QtGui.QKeySequence("F5"), self, activated=self._run_current)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+N"), self, activated=self._new_file)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+O"), self, activated=self._open_file_dialog)

    # -- tab / file helpers -------------------------------------------
    def _current_editor(self) -> CodeTextEdit | None:
        return self.editor_tabs.currentWidget()

    def _make_editor(self) -> CodeTextEdit:
        editor = CodeTextEdit()
        editor.textChanged.connect(self._mark_current_dirty)
        return editor

    def _new_file(self) -> None:
        editor = self._make_editor()
        PythonHighlighter(editor.document())
        index = self.editor_tabs.addTab(editor, "untitled.py")
        self._open_paths[id(editor)] = None
        self.editor_tabs.setCurrentIndex(index)
        editor.setFocus()

    def _open_path(self, path: Path) -> None:
        # If already open, just focus it.
        for i in range(self.editor_tabs.count()):
            w = self.editor_tabs.widget(i)
            if self._open_paths.get(id(w)) == path:
                self.editor_tabs.setCurrentIndex(i)
                return
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, tr("Open failed"), f"{tr('Could not open')}:\n{path}\n\n{exc}")
            return

        editor = self._make_editor()
        if path.suffix == ".py":
            PythonHighlighter(editor.document())
        editor.setPlainText(text)
        editor.document().setModified(False)
        index = self.editor_tabs.addTab(editor, path.name)
        self._open_paths[id(editor)] = path
        self.editor_tabs.setCurrentIndex(index)
        editor.setFocus()

    def _on_tree_double_clicked(self, index: QtCore.QModelIndex) -> None:
        path = Path(self.fs_model.filePath(index))
        if path.is_file():
            self._open_path(path)

    def _open_file_dialog(self) -> None:
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open File", str(self._root), "All Files (*.*)"
        )
        if file_name:
            self._open_path(Path(file_name))

    def _mark_current_dirty(self) -> None:
        editor = self.sender()
        idx = self.editor_tabs.indexOf(editor)
        if idx < 0:
            return
        title = self.editor_tabs.tabText(idx)
        if not title.startswith("● ") and editor.document().isModified():
            self.editor_tabs.setTabText(idx, "● " + title)

    def _on_current_tab_changed(self, index: int) -> None:
        editor = self.editor_tabs.widget(index)
        path = self._open_paths.get(id(editor)) if editor else None
        self.current_file_label.setText(str(path) if path else "(unsaved file)")

    def _save_current(self) -> None:
        editor = self._current_editor()
        if editor is None:
            return
        path = self._open_paths.get(id(editor))
        if path is None:
            self._save_current_as()
            return
        self._write_file(editor, path)

    def _save_current_as(self) -> None:
        editor = self._current_editor()
        if editor is None:
            return
        start_dir = str(self._open_paths.get(id(editor)) or self._root)
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save As", start_dir, "Python Files (*.py);;Batch Files (*.bat);;All Files (*.*)"
        )
        if not file_name:
            return
        path = Path(file_name)
        self._open_paths[id(editor)] = path
        self._write_file(editor, path)

    def _write_file(self, editor: CodeTextEdit, path: Path) -> None:
        try:
            path.write_text(editor.toPlainText(), encoding="utf-8")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, tr("Save failed"), f"{tr('Could not save')}:\n{path}\n\n{exc}")
            return
        editor.document().setModified(False)
        idx = self.editor_tabs.indexOf(editor)
        if idx >= 0:
            self.editor_tabs.setTabText(idx, path.name)
        self.current_file_label.setText(str(path))
        self._append_console(f"{tr('Saved')} {path}")

    def _close_tab(self, index: int) -> None:
        editor = self.editor_tabs.widget(index)
        if editor is not None and editor.document().isModified():
            reply = QtWidgets.QMessageBox.question(
                self, tr("Unsaved changes"),
                f"\"{self.editor_tabs.tabText(index).lstrip('● ')}\" " + tr("has unsaved changes. Close anyway?"),
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            )
            if reply != QtWidgets.QMessageBox.Yes:
                return
        if editor is not None:
            self._open_paths.pop(id(editor), None)
        self.editor_tabs.removeTab(index)
        if self.editor_tabs.count() == 0:
            self._new_file()

    # -- run ------------------------------------------------------------
    def _ensure_saved_for_run(self) -> Path | None:
        editor = self._current_editor()
        if editor is None:
            return None
        path = self._open_paths.get(id(editor))
        if path is None:
            QtWidgets.QMessageBox.information(
                self, tr("Save first"), tr("Save this file before running it.")
            )
            self._save_current_as()
            path = self._open_paths.get(id(editor))
            if path is None:
                return None
        if editor.document().isModified():
            self._write_file(editor, path)
        if path.suffix.lower() not in RUNNABLE_SUFFIXES:
            dont_know_how = tr("Don't know how to run")
            QtWidgets.QMessageBox.information(
                self, tr("Can't run this file type"),
                f"{dont_know_how} '{path.suffix}' {tr('files.')}\n{tr('Run works with')}: "
                + ", ".join(sorted(RUNNABLE_SUFFIXES)),
            )
            return None
        return path

    def _append_console(self, text: str) -> None:
        self.console.appendPlainText(text)
        self.console.verticalScrollBar().setValue(self.console.verticalScrollBar().maximum())

    def _run_current(self) -> None:
        if self._process is not None:
            QtWidgets.QMessageBox.information(self, tr("Already running"), tr("Stop the current run first."))
            return
        path = self._ensure_saved_for_run()
        if path is None:
            return

        if path.suffix.lower() == ".py" and getattr(sys, "frozen", False):
            self._run_in_process(path)
            return
        if path.suffix.lower() == ".py":
            program, args = sys.executable, ["-u", str(path)]
        else:  # .bat / .cmd
            program, args = "cmd.exe" if sys.platform.startswith("win") else "/bin/sh", (
                ["/c", str(path)] if sys.platform.startswith("win") else [str(path)]
            )

        self.console.clear()
        self._append_console(f"$ {program} {' '.join(args)}\n")

        proc = QtCore.QProcess(self)
        proc.setWorkingDirectory(str(path.parent))
        proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(lambda: self._on_process_output(proc))
        proc.finished.connect(self._on_process_finished)
        proc.errorOccurred.connect(self._on_process_error)
        proc.start(program, args)
        self._process = proc
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def _run_in_process(self, path: Path) -> None:
        self.console.clear()
        self._append_console(f"$ python {path.name}   ({tr('running inside the app')})\n")
        runner = _InProcessScript(path, self)
        runner.output.connect(self._on_inprocess_output)
        runner.finished.connect(self._on_process_finished)
        self._process = runner
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        runner.start()

    def _on_inprocess_output(self, text: str) -> None:
        self.console.moveCursor(QtGui.QTextCursor.End)
        self.console.insertPlainText(text)
        self.console.moveCursor(QtGui.QTextCursor.End)

    def _on_process_output(self, proc: QtCore.QProcess) -> None:
        data = bytes(proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        if data:
            self.console.moveCursor(QtGui.QTextCursor.End)
            self.console.insertPlainText(data)
            self.console.moveCursor(QtGui.QTextCursor.End)

    def _on_process_finished(self, exit_code: int, _status=None) -> None:
        self._append_console(f"\n[{tr('process exited with code')} {exit_code}]")
        self._process = None
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _on_process_error(self, _error) -> None:
        if self._process is not None:
            self._append_console(f"\n[{tr('failed to start')}: {self._process.errorString()}]")
        self._process = None
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _stop_process(self) -> None:
        if isinstance(self._process, _InProcessScript):
            self._process.stop()
        elif self._process is not None:
            self._process.kill()

    def _run_current_in_terminal(self) -> None:
        path = self._ensure_saved_for_run()
        if path is None:
            return
        if path.suffix.lower() == ".py" and getattr(sys, "frozen", False):
            QtWidgets.QMessageBox.information(
                self, tr("Run in Terminal"),
                tr("The installed app has no separate Python for a terminal - use ▶ Run instead."),
            )
            return
        try:
            if sys.platform.startswith("win"):
                if path.suffix.lower() == ".py":
                    cmd_line = f'"{sys.executable}" "{path}"'
                else:
                    cmd_line = f'"{path}"'
                subprocess.Popen(
                    f'start "AI Detection Lab" cmd /k {cmd_line}',
                    shell=True,
                    cwd=str(path.parent),
                )
            elif sys.platform == "darwin":
                subprocess.Popen(["open", "-a", "Terminal", str(path)], cwd=str(path.parent))
            else:
                subprocess.Popen(
                    ["x-terminal-emulator", "-e", sys.executable, str(path)],
                    cwd=str(path.parent),
                )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, tr("Could not open terminal"), str(exc))


# ---------------------------------------------------------------------------
# Quick Parameters: a friendly editor for config.py's tunable constants.
#
# Schema mirrors config.py's own section layout so the UI and the source
# file never feel like two different documents. Reading/writing goes
# through a small regex + ast.literal_eval pass, never exec()/eval() of
# the whole file, so a broken edit elsewhere in config.py can't run
# arbitrary code just because someone opened this panel.
# ---------------------------------------------------------------------------
CONFIG_RELATIVE_PATH = Path("ai_detection_lab") / "config.py"

# What "GPU" means for torch depends on the machine: Apple Silicon Macs
# expose their GPU as "mps" and have no CUDA at all, while NVIDIA cards on
# Windows/Linux are "cuda" and never "mps". Both are listed so the setting
# means the same thing on every machine the lab is copied to, each marked
# with whether THIS computer can actually use it - a GPU option that
# silently cannot work is worse than no option at all.
GPU_DEVICE_OPTIONS = [
    ("mps", "Mac ชิป Apple Silicon (M1/M2/M3/M4)"),
    ("cuda", "PC/Linux ที่มีการ์ดจอ NVIDIA"),
]

# Answered from the platform instead of by importing torch, which would
# cost this settings panel seconds of startup just to draw a dropdown.
_IS_APPLE_SILICON = sys.platform == "darwin" and platform.machine() == "arm64"


def _device_support_note(device: str) -> str:
    if device == "mps":
        return " ✓ เครื่องนี้ใช้ได้" if _IS_APPLE_SILICON else " ✗ ใช้ไม่ได้บนเครื่องนี้"
    if device == "cuda":
        return " ✗ ใช้ไม่ได้บนเครื่องนี้" if sys.platform == "darwin" else " (ต้องมีการ์ดจอ NVIDIA)"
    return ""

# Deliberately short. This used to mirror every single constant in
# config.py (37 of them) but that turned a "quick" panel into another wall
# of settings nobody actually touches session-to-session. This list keeps
# only the ones a student/instructor realistically re-tunes for a new
# field, a new trained model, or a safety margin - everything else in
# config.py is still one click away as plain text in the Full Code Editor
# tab (just open ai_detection_lab/config.py there).
#
# (section id, section title, [param dicts])
LANGUAGE_OPTIONS = [
    ("en", "English"),
    ("th", "ภาษาไทย (Thai)"),
]

PARAM_SCHEMA: list[tuple[str, str, list[dict]]] = [
    ("general", "ทั่วไป", [
        dict(name="LANGUAGE", kind="language",
             desc="ภาษาที่ใช้แสดงผลในหน้าต่างต่าง ๆ ของโปรแกรม — มีผลหลังรีสตาร์ทโปรแกรม "
                  "(ปิดแล้วเปิดใหม่) เท่านั้น หน้าต่างที่เปิดค้างอยู่จะไม่เปลี่ยนภาษาให้ทันที"),
    ]),
    ("ai", "AI Detection", [
        dict(name="YOLO_CONF_THRES", kind="float", minimum=0.0, maximum=1.0, step=0.01, decimals=2,
             desc="ค่าความเชื่อมั่นขั้นต่ำที่ YOLO ต้องมี จึงจะนับเป็นการตรวจพบ — ลดค่านี้ถ้าโมเดลตรวจจับยากเกินไป"),
        dict(name="YOLO_DEVICE", kind="device",
             desc="อุปกรณ์ที่ใช้ตรวจจับสด — Auto (แนะนำ) จะเลือก mps บน Mac ชิป Apple Silicon, "
                  "cuda ถ้ามีการ์ดจอ NVIDIA, ไม่งั้น cpu. วัดบน Mac ชิป Apple: mps ~9 ms/เฟรม "
                  "เร็วกว่า cpu (~43 ms/เฟรม) ราว 4.6 เท่า. ถ้าเลือกอุปกรณ์ที่เครื่องนี้ไม่มี "
                  "โปรแกรมจะใช้ cpu แทนและแจ้งในบันทึก (ไม่ error ทุกเฟรม). "
                  "อุปกรณ์ตอน Train เลือกแยกได้ที่หน้า Train"),
        dict(name="YOLO_IMGSZ", kind="int", minimum=32, maximum=1280, step=32,
             desc="ขนาดภาพที่ส่งเข้าโมเดล (พิกเซล) — ใหญ่ขึ้นช่วยให้จับวัตถุเล็กได้ดีขึ้นแต่ช้าลง "
                  "ต้องเป็นค่าที่หารด้วย 32 ลงตัว เช่น 640"),
        dict(name="STABILITY_MIN_RATIO", kind="float", minimum=0.0, maximum=1.0, step=0.01, decimals=2,
             desc="สัดส่วนเฟรมที่ต้องตรวจพบซ้ำๆ ก่อนถือว่าเป้าหมายเสถียร (STABLE) เช่น 0.8 = 80%"),
    ]),
    ("flight", "การตั้งค่าบิน", [
        dict(name="INITIAL_HEADING", kind="choice", options=["NORTH", "EAST", "SOUTH", "WEST"],
             desc="ทิศทางเริ่มต้นของโดรนก่อนเริ่มบินสำรวจ — ต้องตรงกับทิศที่วางโดรนจริงหน้าสนาม"),
        dict(name="SURVEY_ALTITUDE_CM", kind="int", minimum=0, maximum=1000, step=5,
             desc="ความสูงที่ใช้บินสำรวจพื้นที่ (ซม.)"),
        dict(name="CAMERA_DOWN_ANGLE", kind="int", minimum=0, maximum=180, step=1,
             desc="มุมกล้องที่ต้องการให้ก้มลง (องศา) ก่อนเริ่มบิน"),
    ]),
    ("field", "สนามและ Grid", [
        dict(name="FIELD_WIDTH_CM", kind="int", minimum=1, maximum=2000, step=10,
             desc="ความกว้างของสนามจริง (ซม.) — ตั้งใหม่ทุกครั้งที่เปลี่ยนสถานที่"),
        dict(name="FIELD_HEIGHT_CM", kind="int", minimum=1, maximum=2000, step=10,
             desc="ความยาว/ความสูงของสนามจริง (ซม.)"),
        dict(name="GRID_SPACING_CM", kind="int", minimum=1, maximum=500, step=5,
             desc="ระยะห่างระหว่างแต่ละแนวของ Grid (ซม.) — ยิ่งถี่ยิ่งครอบคลุมแต่ยิ่งบินนาน"),
        dict(name="GRID_MOVE_STEP_CM", kind="int", minimum=1, maximum=200, step=5,
             desc="ระยะการเคลื่อนที่ต่อหนึ่งคำสั่งขณะบิน Grid (ซม.)"),
    ]),
    ("safety", "ความปลอดภัย", [
        dict(name="BATTERY_CRITICAL_PCT", kind="int", minimum=0, maximum=100, step=1,
             desc="ระดับแบตเตอรี่ขั้นต่ำ (%) ที่ถือว่าอยู่ในสถานะวิกฤต"),
        dict(name="BATTERY_WARN_PCT", kind="int", minimum=0, maximum=100, step=1,
             desc="แบตเหลือเท่านี้ (%) แจ้งเตือนแบตใกล้หมด + เสียงเตือน"),
        dict(name="BATTERY_TAKEOFF_MIN_PCT", kind="int", minimum=0, maximum=100, step=1,
             desc="แบตเหลือเท่านี้ (%) หรือน้อยกว่า ไม่ให้ขึ้นบิน"),
        dict(name="DRONE_LINK_LOST_SEC", kind="float", minimum=1.0, maximum=60.0, step=0.5, decimals=1,
             desc="ไม่ได้รับสัญญาณจากโดรนกี่วินาที ถึงแจ้งว่าโดรนหลุดการเชื่อมต่อ"),
        dict(name="ALERT_SOUND_ENABLED", kind="bool",
             desc="เปิดเสียงเตือน 3 วินาทีตอนโดรนหลุด / แบตต่ำ"),
        dict(name="REQUIRE_PREFLIGHT_CONFIRMATION", kind="bool",
             desc="ต้องกดยืนยัน Preflight ก่อน ระบบจึงจะอนุญาตให้โดรน Takeoff"),
        dict(name="AUTO_LAND_ON_COMPLETE", kind="bool",
             desc="ให้โดรนลงจอดอัตโนมัติเมื่อบินสำรวจครบทุก Waypoint"),
    ]),
]


# v6.0.0: the built app no longer carries its source. Only a read-only text
# copy of config.py ships (under app_params/, see AI_Detection_Lab.spec) so
# Quick Parameters can list each setting's default; edits go to
# USER_OVERRIDES_PATH as before.
FROZEN_CONFIG_RELATIVE_PATH = Path("app_params") / "config.py"


def _config_path() -> Path:
    if getattr(sys, "frozen", False):
        return project_root() / FROZEN_CONFIG_RELATIVE_PATH
    return project_root() / CONFIG_RELATIVE_PATH


def _read_config_values(text: str) -> dict:
    """Parse every top-level ``NAME = <literal>`` assignment in config.py.

    Only literal values (True/False/None/numbers/quoted strings) parse;
    expressions like ``PROJECT_ROOT / "models"`` raise inside
    ast.literal_eval and are simply skipped - this panel only ever needs
    the plain tunable constants, never the derived Path objects.
    """
    values = {}
    for m in re.finditer(r"^([A-Z][A-Z0-9_]*)\s*=\s*(.+?)\s*(?:#.*)?$", text, re.MULTILINE):
        name, raw = m.group(1), m.group(2)
        try:
            values[name] = ast.literal_eval(raw)
        except Exception:
            continue
    return values


def _read_user_overrides(path: Path) -> dict:
    """Values saved by a previous ParametersPanel._save, layered on top of
    config.py's own defaults. See config.py's USER_OVERRIDES_PATH docstring
    for why these live in a separate per-user file instead of being written
    back into config.py itself (a normal install can't write there)."""
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


class ParametersPanel(QtWidgets.QWidget):
    """Friendly form over config.py's tunable constants."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._config_path = _config_path()
        self._controls: dict[str, tuple[str, QtWidgets.QWidget]] = {}  # name -> (kind, widget)
        self._build_ui()
        self._load()

    def _build_ui(self) -> None:
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(8)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        outer.addWidget(scroll, 1)

        content = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(content)
        content_layout.setContentsMargins(4, 4, 4, 4)
        content_layout.setSpacing(12)
        scroll.setWidget(content)

        for _section_id, title, params in PARAM_SCHEMA:
            box = QtWidgets.QGroupBox(title)
            grid = QtWidgets.QGridLayout(box)
            grid.setColumnStretch(0, 1)
            grid.setHorizontalSpacing(14)
            grid.setVerticalSpacing(4)

            row = 0
            for p in params:
                name_label = QtWidgets.QLabel(p["name"])
                name_label.setObjectName("paramName")
                widget = self._make_control(p)
                self._controls[p["name"]] = (p["kind"], widget)

                grid.addWidget(name_label, row, 0)
                grid.addWidget(widget, row, 1)
                row += 1

                desc = QtWidgets.QLabel(p.get("desc", ""))
                desc.setObjectName("paramDesc")
                desc.setWordWrap(True)
                grid.addWidget(desc, row, 0, 1, 2)
                row += 1

            content_layout.addWidget(box)

        content_layout.addStretch()

        # Bottom bar
        bottom = QtWidgets.QHBoxLayout()
        self.reload_btn = QtWidgets.QPushButton(tr("🔄 Reload"))
        self.save_btn = QtWidgets.QPushButton(tr("💾 Save Parameters"))
        self.save_btn.setObjectName("primaryButton")
        self.status_label = QtWidgets.QLabel("")
        self.status_label.setObjectName("muted")
        bottom.addWidget(self.reload_btn)
        bottom.addWidget(self.save_btn)
        bottom.addStretch()
        bottom.addWidget(self.status_label)
        outer.addLayout(bottom)

        note = QtWidgets.QLabel(
            "⚠ " + tr("Changes take effect on the next run of the program (close and reopen)")
        )
        note.setObjectName("paramDesc")
        outer.addWidget(note)

        self.reload_btn.clicked.connect(self._load)
        self.save_btn.clicked.connect(self._save)

    def _make_control(self, p: dict) -> QtWidgets.QWidget:
        kind = p["kind"]
        if kind == "bool":
            w = QtWidgets.QCheckBox()
            return w
        if kind == "int":
            w = QtWidgets.QSpinBox()
            w.setLocale(EN_US_LOCALE)  # force 0-9 digits even on a Thai-numeral Windows locale
            w.setRange(p.get("minimum", 0), p.get("maximum", 100))
            w.setSingleStep(p.get("step", 1))
            return w
        if kind == "float":
            w = QtWidgets.QDoubleSpinBox()
            w.setLocale(EN_US_LOCALE)  # force 0-9 digits even on a Thai-numeral Windows locale
            w.setDecimals(p.get("decimals", 2))
            w.setRange(p.get("minimum", 0.0), p.get("maximum", 1.0))
            w.setSingleStep(p.get("step", 0.1))
            return w
        if kind == "choice":
            w = QtWidgets.QComboBox()
            w.addItems(p.get("options", []))
            return w
        if kind == "language":
            w = QtWidgets.QComboBox()
            for code, label in LANGUAGE_OPTIONS:
                w.addItem(label, code)
            return w
        if kind == "device":
            w = QtWidgets.QComboBox()
            w.addItem("Auto (แนะนำ) — mps บน Mac ชิป Apple / cuda ถ้ามี NVIDIA / ไม่งั้น cpu", None)
            for device, machines in GPU_DEVICE_OPTIONS:
                w.addItem(f"GPU / {device} — {machines}{_device_support_note(device)}", device)
            w.addItem("CPU (cpu) — ใช้ได้ทุกเครื่อง แต่ช้ากว่า GPU", "cpu")
            return w
        raise ValueError(f"unknown parameter kind: {kind}")

    def _set_control_value(self, kind: str, widget: QtWidgets.QWidget, value) -> None:
        if kind == "bool":
            widget.setChecked(bool(value))
        elif kind in ("int", "float"):
            widget.setValue(value)
        elif kind == "choice":
            idx = widget.findText(str(value))
            widget.setCurrentIndex(max(0, idx))
        elif kind == "language":
            idx = widget.findData(value)
            widget.setCurrentIndex(max(0, idx))
        elif kind == "device":
            idx = widget.findData(value)
            if idx < 0:
                # A device this dropdown doesn't offer (e.g. "cuda:1" typed
                # straight into config.py). Keep it as a real choice rather
                # than falling back to Auto, which would quietly overwrite
                # the value the moment Save is pressed.
                widget.addItem(f"{value} — ตั้งไว้เองใน config.py", value)
                idx = widget.count() - 1
            widget.setCurrentIndex(idx)

    def _get_control_value(self, kind: str, widget: QtWidgets.QWidget):
        if kind == "bool":
            return widget.isChecked()
        if kind in ("int", "float"):
            return widget.value()
        if kind == "choice":
            return widget.currentText()
        if kind == "language":
            return widget.currentData()
        if kind == "device":
            return widget.currentData()
        raise ValueError(f"unknown parameter kind: {kind}")

    def _load(self) -> None:
        try:
            text = self._config_path.read_text(encoding="utf-8")
        except Exception as exc:
            self.status_label.setText(f"❌ {tr('Could not read config.py')}: {exc}")
            return
        values = _read_config_values(text)
        values.update(_read_user_overrides(config.USER_OVERRIDES_PATH))
        missing = []
        for _section_id, _title, params in PARAM_SCHEMA:
            for p in params:
                name = p["name"]
                if name not in values:
                    missing.append(name)
                    continue
                kind, widget = self._controls[name]
                self._set_control_value(kind, widget, values[name])
        if missing:
            self.status_label.setText(f"⚠ {tr('Loaded, but could not parse')}: {', '.join(missing)}")
        else:
            self.status_label.setText(f"{tr('Loaded from')} {self._config_path.name}")

    def _save(self) -> None:
        overrides = _read_user_overrides(config.USER_OVERRIDES_PATH)
        for _section_id, _title, params in PARAM_SCHEMA:
            for p in params:
                name = p["name"]
                kind, widget = self._controls[name]
                overrides[name] = self._get_control_value(kind, widget)
        try:
            config.USER_OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
            config.USER_OVERRIDES_PATH.write_text(
                json.dumps(overrides, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as exc:
            self.status_label.setText(f"❌ {tr('Save failed')}: {exc}")
            QtWidgets.QMessageBox.warning(self, tr("Save failed"), str(exc))
            return
        self.status_label.setText(f"✅ {tr('Saved — restart the program to apply')}")


# ---------------------------------------------------------------------------
# Self-contained dark theme for the popup, so it looks right regardless of
# whether Qt style-sheet inheritance from the main window applies to a
# separately-flagged top-level dialog.
# ---------------------------------------------------------------------------
from .panel_theme import DIALOG_STYLE as DIALOG_STYLESHEET


# ---------------------------------------------------------------------------
# The popup window itself
# ---------------------------------------------------------------------------
class CodeEditorDialog(QtWidgets.QDialog):
    """Non-modal popup: Quick Parameters + Full Code Editor, opened from a
    button on the main window instead of living inline in the dashboard."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("AI Detection Lab — Code Editor"))
        self.setWindowFlag(QtCore.Qt.Window, True)
        # A plain QDialog only gets a close button in its title bar on
        # Windows. Add minimize/maximize so the OS title bar itself can
        # also maximize/restore the popup, in addition to the in-dialog
        # Full Screen button below.
        self.setWindowFlag(QtCore.Qt.WindowMinMaxButtonsHint, True)
        self.resize(1300, 820)
        self.setMinimumSize(980, 620)
        self.setStyleSheet(DIALOG_STYLESHEET)
        self._is_maximized_view = False

        layout = QtWidgets.QVBoxLayout(self)
        from .panel_theme import prepare
        prepare(layout)

        header_row = QtWidgets.QHBoxLayout()
        title_col = QtWidgets.QVBoxLayout()
        title = QtWidgets.QLabel(tr("Mission workspace"))
        title.setObjectName("dlgTitle")
        subtitle = QtWidgets.QLabel(tr(
            "Tune mission parameters, or edit the lab's Python source and run it directly."
        ))
        subtitle.setObjectName("dlgSubtitle")
        title_col.addWidget(title)
        title_col.addWidget(subtitle)
        header_row.addLayout(title_col)
        header_row.addStretch()

        self.fullscreen_btn = QtWidgets.QPushButton(tr("⛶  Full Screen"))
        self.fullscreen_btn.setToolTip(tr("Expand this window to fill the screen"))
        self.fullscreen_btn.clicked.connect(self._toggle_fullscreen)
        header_row.addWidget(self.fullscreen_btn, 0, QtCore.Qt.AlignTop)
        layout.addLayout(header_row)

        tabs = QtWidgets.QTabWidget()
        self.params_panel = ParametersPanel()
        self.code_panel = CodeEditorPanel()
        tabs.addTab(self.params_panel, tr("Mission parameters"))
        tabs.addTab(self.code_panel, tr("Python editor"))
        layout.addWidget(tabs, 1)

    def _toggle_fullscreen(self):
        """One button that both enlarges the popup to fill the screen and
        restores it back to its normal size on the next click.

        This button is the single source of truth for the toggle state -
        it does not try to also detect maximize/restore done some other
        way (double-clicking the title bar, OS snap, etc.), because on
        some platforms the window-state-changed notification arrives
        asynchronously and would then race with this method and flip the
        label back at the wrong moment. Simple and reliable beats clever
        here.
        """
        if self._is_maximized_view:
            self.showNormal()
            self.fullscreen_btn.setText(tr("⛶  Full Screen"))
            self.fullscreen_btn.setToolTip(tr("Expand this window to fill the screen"))
        else:
            self.showMaximized()
            self.fullscreen_btn.setText(tr("🗗  Restore Size"))
            self.fullscreen_btn.setToolTip(tr("Restore this window to its normal size"))
        self._is_maximized_view = not self._is_maximized_view

