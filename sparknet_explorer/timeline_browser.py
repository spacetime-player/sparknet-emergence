"""
SparkNet Explorer - Timeline Browser

A Qt widget for browsing and loading saved timeline recordings.
Displays a list of available timelines with metadata (date, steps, run number).
Allows user to select and load a timeline for playback.
"""

import os
import json
from datetime import datetime
from typing import Optional, Callable, List
from dataclasses import dataclass

from pyqtgraph.Qt import QtCore, QtWidgets, QtGui


@dataclass
class TimelineInfo:
    """Metadata about a saved timeline."""
    filepath: str           # Full path without extension
    filename: str           # Just the filename
    run_number: int
    timestamp: datetime
    max_step: int
    num_checkpoints: int

    @property
    def display_name(self) -> str:
        return f"Run {self.run_number:02d}"

    @property
    def date_str(self) -> str:
        return self.timestamp.strftime("%Y-%m-%d %H:%M")

    @property
    def steps_str(self) -> str:
        return f"{self.max_step:,} steps"


def scan_timelines(timeline_dir: str) -> List[TimelineInfo]:
    """
    Scan directory for saved timeline files.

    Returns list of TimelineInfo sorted by run number (descending).
    """
    timelines = []

    if not os.path.exists(timeline_dir):
        return timelines

    for filename in os.listdir(timeline_dir):
        if filename.endswith('.json') and 'timeline' in filename:
            filepath = os.path.join(timeline_dir, filename)
            base_path = filepath[:-5]  # Remove .json

            # Check that arrays file also exists
            arrays_path = base_path + '_arrays.npz'
            if not os.path.exists(arrays_path):
                continue

            try:
                # Parse filename: Run_XX_timeline_YYYY-MM-DD_HH-MM-SS.json
                parts = filename.replace('.json', '').split('_')
                run_number = int(parts[1])

                # Extract timestamp from filename
                # Format: Run_XX_timeline_YYYY-MM-DD_HH-MM-SS
                timestamp_str = '_'.join(parts[3:])  # YYYY-MM-DD_HH-MM-SS
                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")

                # Load JSON metadata
                with open(filepath, 'r') as f:
                    data = json.load(f)

                max_step = data.get('max_step', 0)
                num_checkpoints = len(data.get('checkpoint_steps', []))

                timelines.append(TimelineInfo(
                    filepath=base_path,
                    filename=filename,
                    run_number=run_number,
                    timestamp=timestamp,
                    max_step=max_step,
                    num_checkpoints=num_checkpoints
                ))

            except (ValueError, KeyError, json.JSONDecodeError) as e:
                # Skip malformed files
                continue

    # Sort by run number descending (newest first)
    timelines.sort(key=lambda t: t.run_number, reverse=True)
    return timelines


class TimelineBrowserWidget(QtWidgets.QWidget):
    """
    Widget for browsing and selecting saved timelines.

    Displays:
    - List of saved timelines with metadata
    - Preview info for selected timeline
    - Load button to load selected timeline
    """

    # Signal emitted when user wants to load a timeline
    timeline_selected = QtCore.Signal(str)  # Emits filepath

    def __init__(self, timeline_dir: str = 'exploration_runs/timelines', parent=None):
        super().__init__(parent)
        self.timeline_dir = timeline_dir
        self.timelines: List[TimelineInfo] = []
        self.selected_timeline: Optional[TimelineInfo] = None

        self._setup_ui()
        self.refresh()

    def _setup_ui(self):
        """Create the browser UI."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Header
        header = QtWidgets.QLabel("SAVED TIMELINES")
        header.setStyleSheet("""
            QLabel {
                background-color: #2d2d2d;
                color: #4CAF50;
                font-weight: bold;
                font-size: 12pt;
                padding: 10px;
                border: 1px solid #4CAF50;
                border-radius: 4px;
            }
        """)
        header.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)

        # Timeline list
        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setStyleSheet("""
            QListWidget {
                background-color: #0d0d0d;
                color: white;
                border: 1px solid #333;
                font-family: Consolas, monospace;
                font-size: 10pt;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #222;
            }
            QListWidget::item:selected {
                background-color: #2d5016;
                border: 1px solid #4CAF50;
            }
            QListWidget::item:hover {
                background-color: #1a3010;
            }
        """)
        self.list_widget.itemSelectionChanged.connect(self._on_selection_changed)
        self.list_widget.itemDoubleClicked.connect(self._on_double_click)
        layout.addWidget(self.list_widget, stretch=2)

        # Info panel for selected timeline
        self.info_panel = QtWidgets.QTextEdit()
        self.info_panel.setReadOnly(True)
        self.info_panel.setMaximumHeight(120)
        self.info_panel.setStyleSheet("""
            QTextEdit {
                background-color: #0a0a0a;
                color: #aaa;
                border: 1px solid #333;
                font-family: Consolas, monospace;
                font-size: 9pt;
            }
        """)
        self.info_panel.setText("Select a timeline to view details")
        layout.addWidget(self.info_panel)

        # Button row
        button_row = QtWidgets.QHBoxLayout()
        button_row.setSpacing(10)

        self.refresh_btn = QtWidgets.QPushButton("Refresh")
        self.refresh_btn.setStyleSheet(self._button_style("#555", "#666"))
        self.refresh_btn.clicked.connect(self.refresh)
        button_row.addWidget(self.refresh_btn)

        button_row.addStretch()

        self.load_btn = QtWidgets.QPushButton("Load Timeline")
        self.load_btn.setStyleSheet(self._button_style("#4CAF50", "#66BB6A"))
        self.load_btn.setEnabled(False)
        self.load_btn.clicked.connect(self._on_load_clicked)
        button_row.addWidget(self.load_btn)

        layout.addLayout(button_row)

    def _button_style(self, bg_color: str, hover_color: str) -> str:
        """Generate button stylesheet."""
        return f"""
            QPushButton {{
                background-color: {bg_color};
                color: white;
                border: none;
                padding: 10px 20px;
                font-weight: bold;
                font-size: 10pt;
                border-radius: 4px;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
            }}
            QPushButton:pressed {{
                background-color: {bg_color};
            }}
            QPushButton:disabled {{
                background-color: #333;
                color: #666;
            }}
        """

    def refresh(self):
        """Scan for timelines and update the list."""
        self.timelines = scan_timelines(self.timeline_dir)
        self.list_widget.clear()

        if not self.timelines:
            item = QtWidgets.QListWidgetItem("No saved timelines found")
            item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsSelectable)
            item.setForeground(QtGui.QColor("#666"))
            self.list_widget.addItem(item)
            return

        for timeline in self.timelines:
            # Create rich display text
            display = f"{timeline.display_name}  |  {timeline.date_str}  |  {timeline.steps_str}"
            item = QtWidgets.QListWidgetItem(display)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, timeline)
            self.list_widget.addItem(item)

    def _on_selection_changed(self):
        """Handle timeline selection change."""
        items = self.list_widget.selectedItems()
        if not items:
            self.selected_timeline = None
            self.load_btn.setEnabled(False)
            self.info_panel.setText("Select a timeline to view details")
            return

        timeline = items[0].data(QtCore.Qt.ItemDataRole.UserRole)
        if not isinstance(timeline, TimelineInfo):
            return

        self.selected_timeline = timeline
        self.load_btn.setEnabled(True)

        # Update info panel
        info_text = f"""
Run #{timeline.run_number:02d}
Date: {timeline.date_str}
Total Steps: {timeline.max_step:,}
Checkpoints: {timeline.num_checkpoints}

File: {timeline.filename}
        """.strip()
        self.info_panel.setText(info_text)

    def _on_double_click(self, item):
        """Handle double-click to load timeline."""
        if self.selected_timeline:
            self._on_load_clicked()

    def _on_load_clicked(self):
        """Emit signal to load selected timeline."""
        if self.selected_timeline:
            self.timeline_selected.emit(self.selected_timeline.filepath)


class TimelineBrowserTab(QtWidgets.QWidget):
    """
    Full tab widget containing the timeline browser.

    Includes:
    - Browser widget
    - New simulation button
    - Status display
    """

    # Signals
    load_timeline_requested = QtCore.Signal(str)  # filepath
    new_simulation_requested = QtCore.Signal()

    def __init__(self, timeline_dir: str = 'exploration_runs/timelines', parent=None):
        super().__init__(parent)
        self.timeline_dir = timeline_dir
        self._setup_ui()

    def _setup_ui(self):
        """Create the tab layout."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)

        # New simulation section
        new_sim_frame = QtWidgets.QFrame()
        new_sim_frame.setStyleSheet("""
            QFrame {
                background-color: #1a2a1a;
                border: 1px solid #4CAF50;
                border-radius: 6px;
                padding: 10px;
            }
        """)
        new_sim_layout = QtWidgets.QVBoxLayout(new_sim_frame)

        new_sim_label = QtWidgets.QLabel("Start New Simulation")
        new_sim_label.setStyleSheet("""
            QLabel {
                color: #4CAF50;
                font-weight: bold;
                font-size: 11pt;
            }
        """)
        new_sim_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        new_sim_layout.addWidget(new_sim_label)

        new_sim_desc = QtWidgets.QLabel("Initialize a fresh exploration run with default parameters")
        new_sim_desc.setStyleSheet("color: #888; font-size: 9pt;")
        new_sim_desc.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        new_sim_layout.addWidget(new_sim_desc)

        self.new_sim_btn = QtWidgets.QPushButton("New Simulation")
        self.new_sim_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 12px 30px;
                font-weight: bold;
                font-size: 11pt;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #66BB6A;
            }
            QPushButton:pressed {
                background-color: #388E3C;
            }
        """)
        self.new_sim_btn.clicked.connect(lambda: self.new_simulation_requested.emit())
        new_sim_layout.addWidget(self.new_sim_btn)

        layout.addWidget(new_sim_frame)

        # Separator
        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        separator.setStyleSheet("background-color: #333;")
        layout.addWidget(separator)

        # Or label
        or_label = QtWidgets.QLabel("OR")
        or_label.setStyleSheet("color: #666; font-weight: bold;")
        or_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(or_label)

        # Browser widget
        self.browser = TimelineBrowserWidget(self.timeline_dir, self)
        self.browser.timeline_selected.connect(self._on_timeline_selected)
        layout.addWidget(self.browser, stretch=1)

    def _on_timeline_selected(self, filepath: str):
        """Forward timeline selection to parent."""
        self.load_timeline_requested.emit(filepath)

    def refresh(self):
        """Refresh the timeline list."""
        self.browser.refresh()

    def set_controls_enabled(self, enabled: bool):
        """Enable/disable controls based on app state."""
        self.new_sim_btn.setEnabled(enabled)
        self.browser.load_btn.setEnabled(enabled and self.browser.selected_timeline is not None)
