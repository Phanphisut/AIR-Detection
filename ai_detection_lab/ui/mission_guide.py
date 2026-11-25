"""🎯 Mission Guide panel for the 🐍 Flight Plan window.

Pick a level, read the story and goals, ask for hints one at a time, then
press 🏁 Grade (or 🧊 Simulate 3D). Grading itself lives in
missions/challenges.py and runs on the simulation only.
"""
from __future__ import annotations

import html

from PySide6 import QtCore, QtWidgets

from ..i18n import tr
from ..missions.challenges import LEVELS, MAX_STARS, MISSIONS, Progress, rank


def stars_text(n, out_of=3):
    return "★" * n + "☆" * (out_of - n)


class MissionGuidePanel(QtWidgets.QWidget):
    mission_changed = QtCore.Signal(object)     # Mission or None (free flight)
    starter_requested = QtCore.Signal()
    grade_requested = QtCore.Signal()

    def __init__(self, progress_path, parent=None):
        super().__init__(parent)
        self.progress = Progress(progress_path)
        self._hints_shown = 0
        self._last_grade = None

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 0)
        lay.setSpacing(8)

        top = QtWidgets.QHBoxLayout()
        self.rank_label = QtWidgets.QLabel()
        self.rank_label.setStyleSheet("font-weight:700; color:#f2c778;")
        top.addWidget(self.rank_label)
        top.addStretch()
        self.total_label = QtWidgets.QLabel()
        self.total_label.setStyleSheet("font-weight:700; color:#f2c778;")
        top.addWidget(self.total_label)
        lay.addLayout(top)

        self.combo = QtWidgets.QComboBox()
        self.combo.setMaxVisibleItems(16)
        lay.addWidget(self.combo)

        # -- the mission card (scrolls) ------------------------------------
        card = QtWidgets.QWidget()
        cl = QtWidgets.QVBoxLayout(card)
        cl.setContentsMargins(2, 2, 8, 2)
        cl.setSpacing(8)
        self.title = QtWidgets.QLabel()
        self.title.setStyleSheet("font-size:17px; font-weight:700; color:#f0f5fb;")
        self.title.setWordWrap(True)
        cl.addWidget(self.title)
        self.chips = QtWidgets.QLabel()
        self.chips.setTextFormat(QtCore.Qt.RichText)
        cl.addWidget(self.chips)
        self.story = QtWidgets.QLabel()
        self.story.setWordWrap(True)
        self.story.setStyleSheet("color:#b9c9d8; font-size:12.5px;")
        cl.addWidget(self.story)
        self.goals = QtWidgets.QLabel()
        self.goals.setWordWrap(True)
        self.goals.setTextFormat(QtCore.Qt.RichText)
        self.goals.setStyleSheet("background:#131d29; border:1px solid #29394b; border-radius:8px; padding:8px;")
        cl.addWidget(self.goals)
        self.hint_box = QtWidgets.QLabel()
        self.hint_box.setWordWrap(True)
        self.hint_box.setTextFormat(QtCore.Qt.RichText)
        self.hint_box.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.hint_box.setStyleSheet("background:#1d2a1f; border:1px solid #33553a; border-radius:8px; padding:8px; color:#cdeccf;")
        cl.addWidget(self.hint_box)
        self.result = QtWidgets.QLabel()
        self.result.setWordWrap(True)
        self.result.setTextFormat(QtCore.Qt.RichText)
        cl.addWidget(self.result)
        cl.addStretch()
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.scroll.setWidget(card)
        lay.addWidget(self.scroll, 1)

        row = QtWidgets.QGridLayout()
        row.setSpacing(6)
        self.starter_btn = QtWidgets.QPushButton(tr("📝 Starter code"))
        self.starter_btn.setToolTip(tr("Put the mission's starter code back in the editor."))
        self.hint_btn = QtWidgets.QPushButton()
        self.solution_btn = QtWidgets.QPushButton(tr("🔑 Answer"))
        self.solution_btn.setToolTip(tr("Shows one possible answer — available after all the hints."))
        self.grade_btn = QtWidgets.QPushButton(tr("🏁 Grade mission"))
        self.grade_btn.setObjectName("primaryButton")
        self.grade_btn.setToolTip(tr("Checks your code against the mission goals in the simulation — the drone does not move."))
        for i, b in enumerate((self.hint_btn, self.solution_btn, self.starter_btn, self.grade_btn)):
            row.addWidget(b, i // 2, i % 2)
        lay.addLayout(row)

        self.combo.currentIndexChanged.connect(self._on_combo)
        self.starter_btn.clicked.connect(self.starter_requested.emit)
        self.hint_btn.clicked.connect(self._next_hint)
        self.solution_btn.clicked.connect(self._show_solution)
        self.grade_btn.clicked.connect(self.grade_requested.emit)
        self._fill_combo()
        self._show(None)

    # -- data --------------------------------------------------------------
    def current(self):
        return self.combo.currentData()

    def select(self, mission_id):
        for i in range(self.combo.count()):
            m = self.combo.itemData(i)
            if (m.id if m else None) == mission_id:
                self.combo.setCurrentIndex(i)
                return

    def _fill_combo(self):
        keep = self.current()
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItem(tr("🆓 Free flight (no mission)"), None)
        for i, m in enumerate(MISSIONS, 1):
            got = self.progress.stars.get(m.id, 0)
            self.combo.addItem(f"{i}. {m.name}   {stars_text(got)}   · {LEVELS[m.level]}", m)
        if keep is not None:
            for i in range(self.combo.count()):
                if self.combo.itemData(i) is keep:
                    self.combo.setCurrentIndex(i)
        self.combo.blockSignals(False)
        total = self.progress.total
        self.rank_label.setText(rank(total))
        self.total_label.setText(f"⭐ {total}/{MAX_STARS}")

    def _on_combo(self):
        m = self.current()
        self._show(m)
        self.mission_changed.emit(m)

    # -- card --------------------------------------------------------------
    def _show(self, m):
        self._hints_shown = 0
        self._last_grade = None
        free = m is None
        for b in (self.starter_btn, self.hint_btn, self.solution_btn, self.grade_btn):
            b.setVisible(not free)
        self.result.clear()
        if free:
            self.title.setText(tr("🆓 Free flight"))
            self.chips.clear()
            self.story.setText(tr(
                "Write anything you like. Pick a mission above for a challenge with goals, "
                "objects in the 3D world, hints and stars ★★★."))
            done = sum(1 for m in MISSIONS if self.progress.stars.get(m.id, 0) > 0)
            self.goals.setText(
                f"<b>🎯 {len(MISSIONS)} " + tr("missions") + f"</b> — " + tr("passed") + f" {done}<br>"
                + "<br>".join(f"{stars_text(self.progress.stars.get(x.id, 0))}&nbsp; {html.escape(x.name)}"
                              for x in MISSIONS))
            self.hint_box.hide()
            return
        got = self.progress.stars.get(m.id, 0)
        self.title.setText(f"{m.name}   <span style='color:#f2c778'>{stars_text(got)}</span>")
        self.title.setTextFormat(QtCore.Qt.RichText)
        chip = ("<span style='background:#1c2a3a; color:{c}; border-radius:6px;'>&nbsp;{t}&nbsp;</span>")
        level_color = {1: "#7ee9cf", 2: "#f2c778", 3: "#ff9cab"}[m.level]
        self.chips.setText(chip.format(c=level_color, t=f"{'⭐' * m.level} {LEVELS[m.level]}") + "&nbsp; "
                           + chip.format(c="#9ec9ff", t="📘 " + html.escape(m.teaches)))
        self.story.setText(m.story)
        self._render_goals(None)
        self.hint_box.hide()
        self._update_hint_btn()

    def _render_goals(self, g):
        m = self.current()
        if m is None:
            return
        rows = [f"<b>🎯 {html.escape(m.goal)}</b>"]
        req = g.required if g else [(o.text, None, "") for o in m.objectives]
        bon = g.bonus if g else [(o.text, None, "") for o in m.bonus]

        def line(icon, text, detail):
            d = f" <span style='color:#8ea0b2'>— {html.escape(detail)}</span>" if detail else ""
            return f"{icon} {html.escape(text)}{d}"
        for text, ok, detail in req:
            rows.append(line("⬜" if ok is None else "✅" if ok else "❌", text, detail))
        for k, (text, ok, detail) in enumerate(bon):
            star = "⭐" * (k + 2)
            rows.append(line("⬜" if ok is None else "✅" if ok else "▫️", f"{star} {tr('bonus')}: {text}", detail))
        self.goals.setText("<br>".join(rows))

    def _update_hint_btn(self):
        m = self.current()
        if m is None:
            return
        n = len(m.hints)
        self.hint_btn.setText(tr("💡 Hint") + f" {min(self._hints_shown + 1, n)}/{n}")
        self.hint_btn.setEnabled(self._hints_shown < n)
        self.solution_btn.setEnabled(self._hints_shown >= n)

    def _hint_html(self, extra=""):
        m = self.current()
        parts = [f"💡 <b>{tr('Hint')} {i + 1}:</b> " + _code_html(h) for i, h in enumerate(m.hints[:self._hints_shown])]
        return "<br>".join(parts) + extra

    def _next_hint(self):
        m = self.current()
        if m is None or self._hints_shown >= len(m.hints):
            return
        self._hints_shown += 1
        self.hint_box.setText(self._hint_html())
        self.hint_box.show()
        self._update_hint_btn()

    def _show_solution(self):
        m = self.current()
        if m is None:
            return
        self.hint_box.setText(self._hint_html(
            f"<br><br>🔑 <b>{tr('One possible answer')}:</b>" + _code_html(m.solution, block=True)
            + f"<span style='color:#8ea0b2'>{tr('Try writing it yourself first — and try to beat it!')}</span>"))
        self.hint_box.show()

    # -- results -----------------------------------------------------------
    def show_grade(self, g):
        """Show a challenges.Grade; returns True if it was a new best."""
        m = self.current()
        if m is None:
            return False
        self._last_grade = g
        self._render_goals(g)
        new_best = self.progress.record(m.id, g.stars)
        if g.errors:
            self.result.setText("<span style='color:#ff9cab; font-weight:700'>🛠 " + tr("The code does not run yet")
                                + ":</span><br>" + "<br>".join(html.escape(e) for e in g.errors[:3]))
        elif g.passed:
            msg = {1: tr("Mission complete! Try the bonus goals for more stars."),
                   2: tr("Great flying! One more bonus for the full 3 stars."),
                   3: tr("PERFECT! You are a real drone programmer.")}[g.stars]
            best = f" <span style='color:#7ee9cf'>({tr('new best!')})</span>" if new_best else ""
            self.result.setText(f"<span style='font-size:20px; color:#f2c778'>{stars_text(g.stars)}</span>"
                                f"<br><b>{msg}</b>{best}")
        else:
            self.result.setText("<span style='color:#ffd166; font-weight:700'>🚧 " + tr("Not yet — check the ❌ goals above, "
                                "fix the code and try again.") + "</span>")
        self._fill_combo()
        got = self.progress.stars.get(m.id, 0)
        self.title.setText(f"{m.name}   <span style='color:#f2c778'>{stars_text(got)}</span>")
        QtCore.QTimer.singleShot(0, lambda: self.scroll.ensureWidgetVisible(self.result))
        return new_best


def _code_html(text, block=False):
    esc = html.escape(text).replace(" ", "&nbsp;").replace("\n", "<br>")
    if block or "\n" in text:
        return ("<div style='font-family:Menlo,Consolas,monospace; background:#0e1722; color:#d6e6f5; "
                f"padding:6px; margin:4px 0'>{esc}</div>")
    return esc
