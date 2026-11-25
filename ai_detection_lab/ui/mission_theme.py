"""Visual tokens for the independently versioned Mission Control interface."""

STYLE = """
QMainWindow, QDialog { background: #0c1119; }
QWidget { color: #e4eaf3; font-family: "SF Pro Text", "Leelawadee UI", "Thonburi"; font-size: 13px; }
QWidget#workspace, QWidget#body { background: #0c1119; }
QFrame#sidebar { background: #101721; border-right: 1px solid #23303f; }
QFrame#topbar { background: #101721; border-bottom: 1px solid #23303f; }
QFrame#panel, QFrame#metric, QFrame#logCard { background: #131d29; border: 1px solid #273545; border-radius: 12px; }
QFrame#videoCard { background: #131d29; border: 1px solid #2b3d4e; border-radius: 12px; }
QFrame#accentMetric { background: #142c31; border: 1px solid #285256; border-radius: 12px; }
QLabel { background: transparent; border: none; }
QLabel#brand { color: #f3f7fc; font-size: 17px; font-weight: 700; }
QLabel#wordmark { color: #58d9c5; font-size: 11px; font-weight: 600; }
QLabel#pageTitle { color: #f4f7fc; font-size: 25px; font-weight: 700; }
QLabel#sectionTitle { color: #edf3fa; font-size: 14px; font-weight: 600; }
QLabel#eyebrow { color: #7e95ab; font-size: 10px; font-weight: 600; }
QLabel#muted { color: #9aaabd; font-size: 12px; }
QLabel#valueBig, QLabel#valueAccent { color: #f0f7fd; font-size: 30px; font-weight: 600; }
QLabel#valueAccent { color: #6be0cc; }
QLabel#targetName { color: #e8f3fc; font-size: 16px; font-weight: 600; }
QLabel#badge { color: #b7c5d5; background: #1c2938; border: 1px solid #304154; border-radius: 12px; padding: 5px 10px; font-size: 11px; }
QLabel#badge[tone="live"] { color: #7ee9cf; background: #15392f; border-color: #295647; }
QLabel#badge[tone="warn"] { color: #f2c778; background: #362d1e; border-color: #54462d; }
QLabel#badge[tone="danger"] { color: #ffa7b2; background: #3c222b; border-color: #673443; }
QLabel#step { color: #9eb1c5; background: #202e3e; border-radius: 13px; font-weight: 600; }
QLabel#step[complete="true"] { color: #6fe0c7; background: #194136; }
QPushButton { background: #1c2a3a; color: #d9e5f0; border: 1px solid #33465b; border-radius: 7px; padding: 7px 11px; min-height: 20px; font-weight: 500; }
QPushButton:hover { background: #25384b; border-color: #58728b; }
QPushButton:pressed { background: #30465a; }
QPushButton:focus { border-color: #75e8d2; }
QPushButton:checked { color: #77e4d0; background: #183b38; border-color: #357366; }
QPushButton#primaryButton { color: #062920; background: #67ddc3; border-color: #67ddc3; font-weight: 700; }
QPushButton#primaryButton:hover { background: #93efd9; border-color: #93efd9; }
QPushButton#dangerButton { background: #3a202a; border-color: #82404c; color: #ffaab6; font-weight: 600; }
QPushButton#dangerButton:hover { background: #5a2936; border-color: #e27b8c; }
QPushButton:disabled, QPushButton#primaryButton:disabled, QPushButton#dangerButton:disabled { background: #18222f; color: #64768a; border-color: #293748; }
QPushButton#nav { background: transparent; border: 1px solid transparent; color: #a5b6c8; text-align: left; padding: 10px 12px; }
QPushButton#nav:hover { background: #1a2938; color: #f0f7fc; }
QPushButton#nav:checked { background: #1a3738; color: #77e4d0; border: 1px solid #2e514f; }
QPushButton#quiet { background: transparent; border-color: #2a3c4f; color: #b0c0d0; }
QPushButton#logToggle { background: transparent; border: none; text-align: left; color: #aabccc; padding: 0; }
QPlainTextEdit { background: #0e1722; color: #b0c4d6; border: 1px solid #2a3c50; border-radius: 7px; padding: 8px; selection-background-color: #286458; }
QPlainTextEdit#terminal { font-family: "Menlo", "Consolas"; font-size: 11px; border: none; }
QPlainTextEdit#modelList { font-size: 12px; padding: 4px; border: none; background: transparent; }
QScrollArea { border: none; background: transparent; }
QScrollArea > QWidget > QWidget { background: transparent; }
QScrollBar:vertical { background: transparent; width: 6px; margin: 0; }
QScrollBar::handle:vertical { background: #354a60; border-radius: 3px; min-height: 26px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
QToolTip { background: #243549; color: #e8f2ff; border: 1px solid #536c83; padding: 7px; }
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit { background: #101a27; color: #dce8f5; border: 1px solid #3b526c; border-radius: 5px; padding: 5px; }
QProgressBar { background: #243547; border: none; border-radius: 3px; max-height: 6px; }
QProgressBar::chunk { background: #67ddc3; border-radius: 3px; }
"""
