"""
SparkNet Explorer - Live Training Interface (PyQtGraph Version)

High-performance real-time visualization using PyQtGraph + OpenGL.
Supports per-step updates (update_freq=1) with smooth 60fps rendering.

Tabs:
- Browser: Load saved timelines or start new simulation
- Networks: Weight matrices + spring graphs
- Metrics: Live reward/novelty/curiosity plots
- Trajectory: Agent path + density heatmap
- State Space: PCA of state embeddings

App States:
- IDLE: App started, can browse timelines or start new simulation
- RUNNING: Simulation actively running (Play pressed)
- PAUSED: Simulation paused (Pause pressed)
- STOPPED: Simulation stopped, timeline mode active for this run

Timeline Mode:
- After training, scrub through history to view any step
- Parameter checkpoints saved every N steps
- Reconstructs visualizations on-demand from saved data
"""

import numpy as np
import torch
import json
import os
from collections import deque
from enum import Enum, auto
from typing import Optional, Callable
from sklearn.decomposition import PCA

# PyQtGraph imports
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui

# Configure PyQtGraph for performance
pg.setConfigOptions(antialias=False, useOpenGL=True, enableExperimental=True)


class AppState(Enum):
    """Application lifecycle states."""
    IDLE = auto()      # Can browse timelines or start new simulation
    RUNNING = auto()   # Simulation actively running
    PAUSED = auto()    # Simulation paused, can resume or stop
    STOPPED = auto()   # Simulation stopped, viewing timeline


class TrainingTimeline:
    """
    Stores training history for post-run timeline scrubbing.

    Saves:
    - Parameter checkpoints at regular intervals (compressed numpy)
    - Spring graph positions at each checkpoint (for accurate replay)
    - All metrics at every step (lightweight)
    - Positions and embeddings for visualization reconstruction
    """

    def __init__(self, checkpoint_interval=100, save_dir=None):
        """
        Args:
            checkpoint_interval: Save full parameters every N steps
            save_dir: Directory to save timeline data (None = memory only)
        """
        self.checkpoint_interval = checkpoint_interval
        self.save_dir = save_dir

        # Metrics history (every step - lightweight)
        self.metrics_history = {
            'step': [],
            'novelty': [],
            'curiosity': [],
            'total_reward': [],
            'exploration_rate': [],
            'position_x': [],
            'position_y': [],
        }

        # Embeddings (every 20 steps for PCA)
        self.embeddings = []
        self.embedding_steps = []

        # Parameter checkpoints (every checkpoint_interval steps)
        self.checkpoints = {}  # step -> {param_name: numpy_array}
        self.checkpoint_steps = []

        # Spring graph positions at EVERY step (for accurate timeline replay)
        # Lists parallel to metrics_history - one entry per step
        self.spring_action_positions = []    # List of numpy arrays
        self.spring_embedder_positions = []  # List of numpy arrays
        self.spring_curiosity_positions = [] # List of numpy arrays

        # Current step
        self.max_step = 0
        self.is_complete = False

    def record_step(self, step, metrics, embedding=None, spring_positions=None):
        """Record data for a single training step.

        Args:
            step: Current training step
            metrics: Dict with novelty, curiosity, total_reward, exploration_rate, position
            embedding: Optional state embedding for PCA
            spring_positions: Dict with 'action', 'embedder', 'curiosity' position arrays
        """
        self.max_step = max(self.max_step, step)

        # Always record metrics
        self.metrics_history['step'].append(step)
        self.metrics_history['novelty'].append(metrics.get('novelty', 0))
        self.metrics_history['curiosity'].append(metrics.get('curiosity', 0))
        self.metrics_history['total_reward'].append(metrics.get('total_reward', 0))
        self.metrics_history['exploration_rate'].append(metrics.get('exploration_rate', 0))

        pos = metrics.get('position', [0, 0])
        self.metrics_history['position_x'].append(float(pos[0]))
        self.metrics_history['position_y'].append(float(pos[1]))

        # Record spring positions at every step (deep copy to prevent reference issues)
        if spring_positions:
            action_pos = spring_positions.get('action')
            embedder_pos = spring_positions.get('embedder')
            curiosity_pos = spring_positions.get('curiosity')
            self.spring_action_positions.append(action_pos.copy() if action_pos is not None else None)
            self.spring_embedder_positions.append(embedder_pos.copy() if embedder_pos is not None else None)
            self.spring_curiosity_positions.append(curiosity_pos.copy() if curiosity_pos is not None else None)
        else:
            self.spring_action_positions.append(None)
            self.spring_embedder_positions.append(None)
            self.spring_curiosity_positions.append(None)

        # Record embedding every step for accurate timeline replay
        if embedding is not None:
            self.embeddings.append(embedding.copy())
            self.embedding_steps.append(step)

    def record_checkpoint(self, step, model):
        """Save parameter checkpoint from model (weights only).

        Args:
            step: Current training step
            model: The model to checkpoint

        Note: Spring positions are now saved at every step via record_step().
        """
        if step % self.checkpoint_interval != 0:
            return

        checkpoint = {}
        for name, param in model.named_parameters():
            # Only save key weights for visualization
            if any(key in name for key in ['hidden_layers.0.weight', 'hidden_layers.3.weight',
                                            'state_embedder.0.weight',
                                            'curiosity_module.forward_model.0.weight']):
                checkpoint[name] = param.detach().cpu().numpy().copy()

        self.checkpoints[step] = checkpoint
        self.checkpoint_steps.append(step)

    def mark_complete(self):
        """Mark training as complete, enabling timeline mode."""
        self.is_complete = True

    def get_metrics_at_step(self, target_step):
        """Get metrics up to a specific step."""
        if not self.metrics_history['step']:
            return None

        # Find index for target step
        steps = self.metrics_history['step']
        idx = 0
        for i, s in enumerate(steps):
            if s <= target_step:
                idx = i
            else:
                break

        return {
            'novelty': self.metrics_history['novelty'][:idx+1],
            'curiosity': self.metrics_history['curiosity'][:idx+1],
            'total_reward': self.metrics_history['total_reward'][:idx+1],
            'exploration_rate': self.metrics_history['exploration_rate'][:idx+1],
            'positions': np.column_stack([
                self.metrics_history['position_x'][:idx+1],
                self.metrics_history['position_y'][:idx+1]
            ]) if idx >= 0 else np.array([]).reshape(0, 2),
            'current_position': [
                self.metrics_history['position_x'][idx],
                self.metrics_history['position_y'][idx]
            ] if idx >= 0 else [0, 0]
        }

    def get_checkpoint_at_step(self, target_step):
        """Get nearest checkpoint at or before target step."""
        if not self.checkpoint_steps:
            return None, None

        # Find nearest checkpoint
        nearest = 0
        for cs in self.checkpoint_steps:
            if cs <= target_step:
                nearest = cs
            else:
                break

        return self.checkpoints.get(nearest, None), nearest

    def get_spring_positions_at_step(self, target_step):
        """Get spring positions at exact step (or nearest available).

        Returns the spring positions saved at the exact target_step.
        This enables accurate timeline replay showing spring graph evolution.
        """
        if not self.metrics_history['step']:
            return None

        # Find index for target step
        steps = self.metrics_history['step']
        idx = None
        for i, s in enumerate(steps):
            if s == target_step:
                idx = i
                break
            elif s > target_step:
                # Use previous index if we passed target
                idx = max(0, i - 1)
                break

        if idx is None:
            idx = len(steps) - 1  # Use last if target beyond recorded

        # Return spring positions at this index
        if idx < len(self.spring_action_positions):
            return {
                'action': self.spring_action_positions[idx],
                'embedder': self.spring_embedder_positions[idx],
                'curiosity': self.spring_curiosity_positions[idx],
            }
        return None

    def get_embeddings_at_step(self, target_step):
        """Get embeddings up to a specific step."""
        result = []
        for i, es in enumerate(self.embedding_steps):
            if es <= target_step:
                result.append(self.embeddings[i])
        return np.array(result) if result else None

    def save_to_disk(self, filepath):
        """Save timeline data to disk including per-step spring positions."""
        data = {
            'metrics': self.metrics_history,
            'embedding_steps': self.embedding_steps,
            'checkpoint_steps': self.checkpoint_steps,
            'max_step': self.max_step,
            'num_spring_positions': len(self.spring_action_positions),
        }

        # Save main data as JSON
        with open(filepath + '.json', 'w') as f:
            json.dump(data, f)

        # Collect all arrays to save
        arrays_to_save = {
            'embeddings': np.array(self.embeddings) if self.embeddings else np.array([])
        }

        # Add checkpoint weight arrays
        for step, ckpt in self.checkpoints.items():
            for name, arr in ckpt.items():
                arrays_to_save[f'ckpt_{step}_{name.replace(".", "_")}'] = arr

        # Save per-step spring positions as object arrays (handles None values)
        # Convert to list of arrays, using empty array for None
        def pack_positions(pos_list):
            """Pack position list into saveable format."""
            result = []
            for pos in pos_list:
                if pos is not None:
                    result.append(pos)
                else:
                    result.append(np.array([]))  # Empty array as sentinel for None
            return result

        arrays_to_save['spring_action'] = np.array(pack_positions(self.spring_action_positions), dtype=object)
        arrays_to_save['spring_embedder'] = np.array(pack_positions(self.spring_embedder_positions), dtype=object)
        arrays_to_save['spring_curiosity'] = np.array(pack_positions(self.spring_curiosity_positions), dtype=object)

        np.savez_compressed(filepath + '_arrays.npz', **arrays_to_save)

    def load_from_disk(self, filepath):
        """Load timeline data from disk including per-step spring positions."""
        with open(filepath + '.json', 'r') as f:
            data = json.load(f)

        self.metrics_history = data['metrics']
        self.embedding_steps = data['embedding_steps']
        self.checkpoint_steps = data['checkpoint_steps']
        self.max_step = data['max_step']
        self.is_complete = True

        # Load numpy arrays
        arrays = np.load(filepath + '_arrays.npz', allow_pickle=True)
        self.embeddings = list(arrays['embeddings']) if len(arrays['embeddings']) > 0 else []

        # Reconstruct checkpoints
        self.checkpoints = {}
        for key in arrays.files:
            if key.startswith('ckpt_'):
                parts = key.split('_', 2)
                step = int(parts[1])
                name = parts[2].replace('_', '.')
                if step not in self.checkpoints:
                    self.checkpoints[step] = {}
                self.checkpoints[step][name] = arrays[key]

        # Load per-step spring positions
        def unpack_positions(arr):
            """Unpack position array, converting empty arrays back to None."""
            result = []
            for pos in arr:
                if pos is not None and len(pos) > 0:
                    result.append(pos)
                else:
                    result.append(None)
            return result

        if 'spring_action' in arrays.files:
            self.spring_action_positions = unpack_positions(arrays['spring_action'])
            self.spring_embedder_positions = unpack_positions(arrays['spring_embedder'])
            self.spring_curiosity_positions = unpack_positions(arrays['spring_curiosity'])
        else:
            # Backwards compatibility: no per-step positions in old files
            num_steps = len(self.metrics_history['step'])
            self.spring_action_positions = [None] * num_steps
            self.spring_embedder_positions = [None] * num_steps
            self.spring_curiosity_positions = [None] * num_steps


def force_layout_fast(weights, pos=None, steps=3, lr=0.05):
    """Fast force-directed layout using vectorized numpy."""
    W = weights.detach().cpu().numpy() if torch.is_tensor(weights) else weights
    n = W.shape[0]

    if pos is None:
        pos = np.random.randn(n, 2) * 2
    else:
        pos = pos.copy()

    W_abs = np.abs(W)
    w_max = W_abs.max()
    if w_max > 0:
        W_norm = W_abs / w_max
    else:
        return pos

    if W.shape[0] != W.shape[1]:
        m = W.shape[1]
        total = max(n, m)
        W_sym = np.zeros((total, total))
        W_sym[:n, :m] = W_norm
        W_sym[:m, :n] = W_norm.T
        W_norm = W_sym
        if len(pos) < total:
            pos = np.vstack([pos, np.random.randn(total - len(pos), 2) * 2])
        n = total

    for _ in range(steps):
        diff = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(diff, axis=2) + 0.1
        rep = diff / (dist[:, :, None] ** 2) * 0.02
        att = -diff * W_norm[:, :, None] * 0.1
        force = (rep + att).sum(axis=1)
        pos += lr * force

        # Anchor center of mass to origin (removes drift, keeps structure)
        pos -= pos.mean(axis=0)

    return pos[:W.shape[0]]


class LiveInterfaceQt:
    """High-performance live training interface with tabbed views and timeline mode."""

    def __init__(self, model=None, update_freq=1, checkpoint_interval=100,
                 timeline_dir='exploration_runs/timelines', start_in_browser=False):
        """
        Initialize the interface.

        Args:
            model: SparkNetExplorer model (can be None if starting in browser mode)
            update_freq: Update visualization every N steps
            checkpoint_interval: Save checkpoints every N steps
            timeline_dir: Directory for saved timelines
            start_in_browser: If True, start in IDLE state showing browser
        """
        self.model = model
        self.update_freq = update_freq
        self.checkpoint_interval = checkpoint_interval
        self.timeline_dir = timeline_dir
        self.step = 0

        # Application state
        self.app_state = AppState.IDLE if start_in_browser else AppState.RUNNING

        # Graph positions for spring layout
        self.action_pos = None
        self.embedder_pos = None
        self.curiosity_pos = None

        # Full history buffers (keep ALL data)
        self.novelty_history = []
        self.curiosity_history = []
        self.reward_history = []
        self.exploration_history = []
        self.loss_history = []
        self.extrinsic_history = []
        self.intrinsic_history = []
        self.homeostatic_history = []
        self.boredom_history = []
        self.weight_coverage_history = []
        self.weight_penalty_history = []

        # Position history (keep ALL data)
        self.position_history = []

        # Display buffers - append-only with periodic decimation
        self.max_display_points = 2000
        self.display_novelty = []
        self.display_curiosity = []
        self.display_reward = []
        self.display_exploration = []
        self.display_steps = []
        self.display_positions = []
        self.display_pos_steps = []
        self.last_display_step = -1

        # State embedding buffer for PCA
        self.embedding_buffer = []
        self.embedding_maxlen = 1000
        self.pca = PCA(n_components=2)
        self.pca_fitted = False

        # Terminal buffer
        self.terminal_lines = deque(maxlen=25)

        # Timeline for post-run scrubbing
        self.timeline = TrainingTimeline(checkpoint_interval=checkpoint_interval)
        self.timeline_mode = False
        self.timeline_step = 0

        # State buffer for inactive tabs
        self.tab_state_buffer = {0: [], 1: [], 2: [], 3: [], 4: []}
        self.last_active_tab = 0
        self.pending_updates = {}

        # Callbacks for external control
        self._on_play_clicked: Optional[Callable] = None
        self._on_pause_clicked: Optional[Callable] = None
        self._on_stop_clicked: Optional[Callable] = None
        self._on_new_simulation: Optional[Callable] = None
        self._on_load_timeline: Optional[Callable[[str], None]] = None

        # Setup Qt
        self.app = pg.mkQApp("SparkNet Explorer")
        self._setup_window()
        self._setup_tabs()
        self._update_ui_for_state()

        self.win.show()
        self.app.processEvents()

    def set_callbacks(self,
                      on_play: Callable = None,
                      on_pause: Callable = None,
                      on_stop: Callable = None,
                      on_new_simulation: Callable = None,
                      on_load_timeline: Callable[[str], None] = None):
        """Set callbacks for control button actions."""
        self._on_play_clicked = on_play
        self._on_pause_clicked = on_pause
        self._on_stop_clicked = on_stop
        self._on_new_simulation = on_new_simulation
        self._on_load_timeline = on_load_timeline

    def set_model(self, model):
        """Set or update the model (for new simulation)."""
        self.model = model

    def set_state(self, new_state: AppState):
        """Update application state and refresh UI."""
        old_state = self.app_state
        self.app_state = new_state
        self._update_ui_for_state()
        self.update_state_label(new_state)
        self.log(f"State: {old_state.name} -> {new_state.name}")

    def _update_ui_for_state(self):
        """Update UI elements based on current app state."""
        state = self.app_state

        # Update window title
        state_titles = {
            AppState.IDLE: 'SparkNet Explorer',
            AppState.RUNNING: 'SparkNet Explorer - Running',
            AppState.PAUSED: 'SparkNet Explorer - Paused',
            AppState.STOPPED: 'SparkNet Explorer - Timeline Mode',
        }
        self.win.setWindowTitle(state_titles.get(state, 'SparkNet Explorer'))

        # Update control buttons
        if hasattr(self, 'play_btn'):
            self.play_btn.setEnabled(state in (AppState.IDLE, AppState.PAUSED))
            self.pause_btn.setEnabled(state == AppState.RUNNING)
            self.stop_btn.setEnabled(state in (AppState.RUNNING, AppState.PAUSED))

        # Update progress bar and step label visibility
        if hasattr(self, 'progress_bar'):
            show_progress = state in (AppState.RUNNING, AppState.PAUSED)
            self.progress_bar.setVisible(show_progress)
            self.step_label.setVisible(not show_progress)

        # Show/hide timeline controls
        if hasattr(self, 'timeline_panel'):
            self.timeline_panel.setVisible(state == AppState.STOPPED)

        # Update browser tab
        if hasattr(self, 'browser_tab'):
            self.browser_tab.set_controls_enabled(state == AppState.IDLE)

        # Switch to appropriate tab
        if state == AppState.IDLE and hasattr(self, 'tabs'):
            self.tabs.setCurrentIndex(0)  # Browser tab
        elif state in (AppState.RUNNING, AppState.PAUSED) and hasattr(self, 'tabs'):
            if self.tabs.currentIndex() == 0:  # If on browser, switch to Networks
                self.tabs.setCurrentIndex(1)

    def reset_for_new_run(self):
        """Reset all visualization state for a new simulation run."""
        self.step = 0

        # Reset spring positions
        self.action_pos = None
        self.embedder_pos = None
        self.curiosity_pos = None

        # Clear history buffers
        self.novelty_history.clear()
        self.curiosity_history.clear()
        self.reward_history.clear()
        self.exploration_history.clear()
        self.loss_history.clear()
        self.extrinsic_history.clear()
        self.intrinsic_history.clear()
        self.homeostatic_history.clear()
        self.boredom_history.clear()
        self.weight_coverage_history.clear()
        self.weight_penalty_history.clear()
        self.position_history.clear()

        # Clear display buffers
        self.display_novelty.clear()
        self.display_curiosity.clear()
        self.display_reward.clear()
        self.display_exploration.clear()
        self.display_steps.clear()
        self.display_positions.clear()
        self.display_pos_steps.clear()
        self.last_display_step = -1

        # Clear embedding buffer
        self.embedding_buffer.clear()
        self.pca_fitted = False

        # Clear terminal
        self.terminal_lines.clear()
        if hasattr(self, 'terminal'):
            self.terminal.clear()

        # Reset timeline
        self.timeline = TrainingTimeline(checkpoint_interval=self.checkpoint_interval)
        self.timeline_mode = False
        self.timeline_step = 0

        # Clear pending updates
        self.pending_updates.clear()

        # Reset progress bar
        if hasattr(self, 'progress_bar'):
            self.progress_bar.setValue(0)

        # Clear plot data
        if hasattr(self, 'novelty_curve'):
            self.novelty_curve.setData([], [])
            self.curiosity_curve.setData([], [])
            self.reward_curve.setData([], [])
            self.exploration_curve.setData([], [])

        if hasattr(self, 'traj_scatter'):
            self.traj_scatter.setData(pos=[])
            self.current_pos_marker.setData(pos=[])

        if hasattr(self, 'pca_scatter'):
            self.pca_scatter.setData(pos=[])

        self.log("Interface reset for new run")

    def _setup_window(self):
        """Create main window."""
        self.win = QtWidgets.QMainWindow()
        self.win.setWindowTitle('SparkNet Explorer - Live Training')
        self.win.resize(1600, 900)
        self.win.setStyleSheet("""
            QMainWindow { background-color: #1a1a1a; }
            QTabWidget::pane { border: 1px solid #333; background: #1a1a1a; }
            QTabBar::tab {
                background: #2d2d2d; color: white; padding: 8px 20px;
                border: 1px solid #333; margin-right: 2px;
            }
            QTabBar::tab:selected { background: #4CAF50; color: white; }
            QTabBar::tab:hover { background: #3d3d3d; }
            QLabel { color: white; }
            QTextEdit {
                background-color: #0a0a0a; color: #00ff00;
                font-family: Consolas, monospace; font-size: 9pt;
                border: 1px solid #333;
            }
        """)

    def _setup_tabs(self):
        """Create tabbed interface with control bar."""
        # Main layout: control bar on top, then tabs + terminal
        central = QtWidgets.QWidget()
        self.win.setCentralWidget(central)
        outer_layout = QtWidgets.QVBoxLayout(central)
        outer_layout.setContentsMargins(5, 5, 5, 5)
        outer_layout.setSpacing(5)

        # Control bar at top
        self._create_control_bar(outer_layout)

        # Content area: tabs on left, terminal on right
        content_layout = QtWidgets.QHBoxLayout()
        content_layout.setSpacing(5)

        # Tab widget (left side, 75% width)
        self.tabs = QtWidgets.QTabWidget()
        content_layout.addWidget(self.tabs, 3)

        # Create tabs - Browser first
        self._create_browser_tab()
        self._create_networks_tab()
        self._create_metrics_tab()
        self._create_trajectory_tab()
        self._create_statespace_tab()

        # Right panel: Terminal + Description
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(5)

        self._create_terminal(right_layout)
        self._create_description(right_layout)

        content_layout.addWidget(right_panel, 1)
        outer_layout.addLayout(content_layout, 1)

    def _create_control_bar(self, parent_layout):
        """Create the playback control bar with Play/Pause/Stop buttons."""
        control_frame = QtWidgets.QFrame()
        control_frame.setStyleSheet("""
            QFrame {
                background-color: #252525;
                border: 1px solid #333;
                border-radius: 6px;
            }
        """)
        control_layout = QtWidgets.QHBoxLayout(control_frame)
        control_layout.setContentsMargins(10, 8, 10, 8)
        control_layout.setSpacing(10)

        # Play button
        self.play_btn = QtWidgets.QPushButton("▶ Play")
        self.play_btn.setStyleSheet(self._control_button_style("#4CAF50", "#66BB6A"))
        self.play_btn.setFixedWidth(100)
        self.play_btn.clicked.connect(self._on_play)
        control_layout.addWidget(self.play_btn)

        # Pause button
        self.pause_btn = QtWidgets.QPushButton("⏸ Pause")
        self.pause_btn.setStyleSheet(self._control_button_style("#FF9800", "#FFB74D"))
        self.pause_btn.setFixedWidth(100)
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self._on_pause)
        control_layout.addWidget(self.pause_btn)

        # Stop button
        self.stop_btn = QtWidgets.QPushButton("⏹ Stop")
        self.stop_btn.setStyleSheet(self._control_button_style("#F44336", "#EF5350"))
        self.stop_btn.setFixedWidth(100)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop)
        control_layout.addWidget(self.stop_btn)

        # Separator
        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        separator.setStyleSheet("background-color: #444;")
        control_layout.addWidget(separator)

        # Progress bar
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #1a1a1a;
                border: 1px solid #333;
                border-radius: 4px;
                text-align: center;
                color: white;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
        """)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Step %v / %m")
        self.progress_bar.setVisible(False)
        control_layout.addWidget(self.progress_bar, 1)

        # Step counter (when progress bar hidden)
        self.step_label = QtWidgets.QLabel("Ready")
        self.step_label.setStyleSheet("""
            QLabel {
                color: #4CAF50;
                font-weight: bold;
                font-size: 11pt;
                padding: 0 20px;
            }
        """)
        control_layout.addWidget(self.step_label, 1)

        # State indicator
        self.state_label = QtWidgets.QLabel("IDLE")
        self.state_label.setStyleSheet("""
            QLabel {
                color: #888;
                font-weight: bold;
                font-size: 10pt;
                padding: 5px 15px;
                background-color: #1a1a1a;
                border: 1px solid #333;
                border-radius: 4px;
            }
        """)
        control_layout.addWidget(self.state_label)

        parent_layout.addWidget(control_frame)

    def _control_button_style(self, bg_color: str, hover_color: str) -> str:
        """Generate control button stylesheet."""
        return f"""
            QPushButton {{
                background-color: {bg_color};
                color: white;
                border: none;
                padding: 8px 15px;
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

    def _on_play(self):
        """Handle Play button click."""
        if self._on_play_clicked:
            self._on_play_clicked()

    def _on_pause(self):
        """Handle Pause button click."""
        if self._on_pause_clicked:
            self._on_pause_clicked()

    def _on_stop(self):
        """Handle Stop button click."""
        if self._on_stop_clicked:
            self._on_stop_clicked()

    def _create_browser_tab(self):
        """Create the timeline browser tab."""
        from sparknet_explorer.timeline_browser import TimelineBrowserTab

        self.browser_tab = TimelineBrowserTab(self.timeline_dir)
        self.browser_tab.new_simulation_requested.connect(self._on_new_simulation_requested)
        self.browser_tab.load_timeline_requested.connect(self._on_load_timeline_requested)
        self.tabs.addTab(self.browser_tab, "Browser")

    def _on_new_simulation_requested(self):
        """Handle new simulation request from browser."""
        if self._on_new_simulation:
            self._on_new_simulation()

    def _on_load_timeline_requested(self, filepath: str):
        """Handle timeline load request from browser."""
        if self._on_load_timeline:
            self._on_load_timeline(filepath)
        else:
            # Default behavior: load timeline directly
            self.load_timeline(filepath)
            self.enter_timeline_mode()

    def update_progress(self, current_step: int, total_steps: int):
        """Update the progress bar."""
        self.progress_bar.setMaximum(total_steps)
        self.progress_bar.setValue(current_step)
        self.progress_bar.setFormat(f"Step {current_step:,} / {total_steps:,}")

    def update_state_label(self, state: AppState):
        """Update the state indicator label."""
        state_colors = {
            AppState.IDLE: ("#888", "IDLE"),
            AppState.RUNNING: ("#4CAF50", "RUNNING"),
            AppState.PAUSED: ("#FF9800", "PAUSED"),
            AppState.STOPPED: ("#2196F3", "TIMELINE"),
        }
        color, text = state_colors.get(state, ("#888", "UNKNOWN"))
        self.state_label.setText(text)
        self.state_label.setStyleSheet(f"""
            QLabel {{
                color: {color};
                font-weight: bold;
                font-size: 10pt;
                padding: 5px 15px;
                background-color: #1a1a1a;
                border: 1px solid {color};
                border-radius: 4px;
            }}
        """)

    def _create_networks_tab(self):
        """Tab 1: Weight matrices and spring graphs."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QGridLayout(tab)
        layout.setSpacing(5)

        # Get RdBu_r colormap (Red-White-Blue diverging)
        self.weight_cmap = pg.colormap.get('RdBu_r', source='matplotlib')

        # Row 0: Weight matrices
        # Action Network
        self.action_widget = pg.GraphicsLayoutWidget()
        self.action_widget.setBackground('#1a1a1a')
        self.action_plot = self.action_widget.addPlot()
        self.action_plot.setTitle("ACTION NETWORK", color='#4CAF50', size='11pt')
        self.action_img = pg.ImageItem()
        self.action_img.setColorMap(self.weight_cmap)
        self.action_plot.addItem(self.action_img)
        self.action_plot.setLabel('bottom', 'Input', color='white')
        self.action_plot.setLabel('left', 'Output', color='white')
        layout.addWidget(self.action_widget, 0, 0)

        # State Embedder
        self.embedder_widget = pg.GraphicsLayoutWidget()
        self.embedder_widget.setBackground('#1a1a1a')
        self.embedder_plot = self.embedder_widget.addPlot()
        self.embedder_plot.setTitle("STATE EMBEDDER", color='#2196F3', size='11pt')
        self.embedder_img = pg.ImageItem()
        self.embedder_img.setColorMap(self.weight_cmap)
        self.embedder_plot.addItem(self.embedder_img)
        self.embedder_plot.setLabel('bottom', 'Input', color='white')
        self.embedder_plot.setLabel('left', 'Output', color='white')
        layout.addWidget(self.embedder_widget, 0, 1)

        # Curiosity Module
        self.curiosity_widget = pg.GraphicsLayoutWidget()
        self.curiosity_widget.setBackground('#1a1a1a')
        self.curiosity_plot = self.curiosity_widget.addPlot()
        self.curiosity_plot.setTitle("CURIOSITY MODULE", color='#FF9800', size='11pt')
        self.curiosity_img = pg.ImageItem()
        self.curiosity_img.setColorMap(self.weight_cmap)
        self.curiosity_plot.addItem(self.curiosity_img)
        self.curiosity_plot.setLabel('bottom', 'Input', color='white')
        self.curiosity_plot.setLabel('left', 'Output', color='white')
        layout.addWidget(self.curiosity_widget, 0, 2)

        # Row 1: Spring graphs
        self.action_graph_widget = pg.GraphicsLayoutWidget()
        self.action_graph_widget.setBackground('#0d0d0d')
        self.action_graph_plot = self.action_graph_widget.addPlot()
        self.action_graph_plot.setTitle("Action (spring)", color='#4CAF50', size='9pt')
        self.action_scatter = pg.ScatterPlotItem(size=5, pen=None, brush=pg.mkBrush('#4CAF50'))
        self.action_graph_plot.addItem(self.action_scatter)
        self.action_graph_plot.showGrid(x=True, y=True, alpha=0.1)
        layout.addWidget(self.action_graph_widget, 1, 0)

        self.embedder_graph_widget = pg.GraphicsLayoutWidget()
        self.embedder_graph_widget.setBackground('#0d0d0d')
        self.embedder_graph_plot = self.embedder_graph_widget.addPlot()
        self.embedder_graph_plot.setTitle("Embedder (spring)", color='#2196F3', size='9pt')
        self.embedder_scatter = pg.ScatterPlotItem(size=5, pen=None, brush=pg.mkBrush('#2196F3'))
        self.embedder_graph_plot.addItem(self.embedder_scatter)
        self.embedder_graph_plot.showGrid(x=True, y=True, alpha=0.1)
        layout.addWidget(self.embedder_graph_widget, 1, 1)

        self.curiosity_graph_widget = pg.GraphicsLayoutWidget()
        self.curiosity_graph_widget.setBackground('#0d0d0d')
        self.curiosity_graph_plot = self.curiosity_graph_widget.addPlot()
        self.curiosity_graph_plot.setTitle("Curiosity (spring)", color='#FF9800', size='9pt')
        self.curiosity_scatter = pg.ScatterPlotItem(size=5, pen=None, brush=pg.mkBrush('#FF9800'))
        self.curiosity_graph_plot.addItem(self.curiosity_scatter)
        self.curiosity_graph_plot.showGrid(x=True, y=True, alpha=0.1)
        layout.addWidget(self.curiosity_graph_widget, 1, 2)

        self.tabs.addTab(tab, "Networks")

    def _create_metrics_tab(self):
        """Tab 2: Live metrics plots."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QGridLayout(tab)
        layout.setSpacing(5)

        # Novelty plot
        self.novelty_widget = pg.GraphicsLayoutWidget()
        self.novelty_widget.setBackground('#1a1a1a')
        self.novelty_plot = self.novelty_widget.addPlot()
        self.novelty_plot.setTitle("Novelty", color='#9C27B0', size='11pt')
        self.novelty_plot.setLabel('bottom', 'Step', color='white')
        self.novelty_plot.setLabel('left', 'Value', color='white')
        self.novelty_plot.showGrid(x=True, y=True, alpha=0.2)
        self.novelty_curve = self.novelty_plot.plot(pen=pg.mkPen('#9C27B0', width=2))
        layout.addWidget(self.novelty_widget, 0, 0)

        # Curiosity plot
        self.curiosity_metric_widget = pg.GraphicsLayoutWidget()
        self.curiosity_metric_widget.setBackground('#1a1a1a')
        self.curiosity_metric_plot = self.curiosity_metric_widget.addPlot()
        self.curiosity_metric_plot.setTitle("Curiosity", color='#FF9800', size='11pt')
        self.curiosity_metric_plot.setLabel('bottom', 'Step', color='white')
        self.curiosity_metric_plot.setLabel('left', 'Value', color='white')
        self.curiosity_metric_plot.showGrid(x=True, y=True, alpha=0.2)
        self.curiosity_curve = self.curiosity_metric_plot.plot(pen=pg.mkPen('#FF9800', width=2))
        layout.addWidget(self.curiosity_metric_widget, 0, 1)

        # Total Reward plot
        self.reward_widget = pg.GraphicsLayoutWidget()
        self.reward_widget.setBackground('#1a1a1a')
        self.reward_plot = self.reward_widget.addPlot()
        self.reward_plot.setTitle("Total Reward", color='#4CAF50', size='11pt')
        self.reward_plot.setLabel('bottom', 'Step', color='white')
        self.reward_plot.setLabel('left', 'Value', color='white')
        self.reward_plot.showGrid(x=True, y=True, alpha=0.2)
        self.reward_curve = self.reward_plot.plot(pen=pg.mkPen('#4CAF50', width=2))
        layout.addWidget(self.reward_widget, 1, 0)

        # Exploration Rate plot
        self.exploration_widget = pg.GraphicsLayoutWidget()
        self.exploration_widget.setBackground('#1a1a1a')
        self.exploration_plot = self.exploration_widget.addPlot()
        self.exploration_plot.setTitle("Exploration Rate", color='#00BCD4', size='11pt')
        self.exploration_plot.setLabel('bottom', 'Step', color='white')
        self.exploration_plot.setLabel('left', 'Rate', color='white')
        self.exploration_plot.showGrid(x=True, y=True, alpha=0.2)
        self.exploration_curve = self.exploration_plot.plot(pen=pg.mkPen('#00BCD4', width=2))
        layout.addWidget(self.exploration_widget, 1, 1)

        self.tabs.addTab(tab, "Metrics")

    def _create_trajectory_tab(self):
        """Tab 3: Trajectory scatter + heatmap."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QGridLayout(tab)
        layout.setSpacing(5)

        # Trajectory scatter (colored by time)
        self.traj_widget = pg.GraphicsLayoutWidget()
        self.traj_widget.setBackground('#1a1a1a')
        self.traj_plot = self.traj_widget.addPlot()
        self.traj_plot.setTitle("Exploration Trajectory", color='white', size='11pt')
        self.traj_plot.setLabel('bottom', 'X Position', color='white')
        self.traj_plot.setLabel('left', 'Y Position', color='white')
        self.traj_plot.setXRange(-1.1, 1.1)
        self.traj_plot.setYRange(-1.1, 1.1)
        self.traj_plot.setAspectLocked(True)
        self.traj_plot.showGrid(x=True, y=True, alpha=0.2)

        # Boundary box
        boundary = pg.PlotDataItem(
            [-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1],
            pen=pg.mkPen('#FFEB3B', width=2, style=QtCore.Qt.PenStyle.DashLine)
        )
        self.traj_plot.addItem(boundary)

        # Trajectory scatter
        self.traj_scatter = pg.ScatterPlotItem(size=4, pen=None)
        self.traj_plot.addItem(self.traj_scatter)

        # Current position marker
        self.current_pos_marker = pg.ScatterPlotItem(
            size=15, pen=pg.mkPen('white', width=2),
            brush=pg.mkBrush('#F44336')
        )
        self.traj_plot.addItem(self.current_pos_marker)

        layout.addWidget(self.traj_widget, 0, 0)

        # Density heatmap
        self.heatmap_widget = pg.GraphicsLayoutWidget()
        self.heatmap_widget.setBackground('#1a1a1a')
        self.heatmap_plot = self.heatmap_widget.addPlot()
        self.heatmap_plot.setTitle("Visit Density", color='white', size='11pt')
        self.heatmap_plot.setLabel('bottom', 'X Position', color='white')
        self.heatmap_plot.setLabel('left', 'Y Position', color='white')
        self.heatmap_plot.setAspectLocked(True)

        self.heatmap_img = pg.ImageItem()
        self.heatmap_plot.addItem(self.heatmap_img)

        # Colormap for heatmap
        cmap = pg.colormap.get('inferno', source='matplotlib')
        self.heatmap_img.setColorMap(cmap)

        layout.addWidget(self.heatmap_widget, 0, 1)

        self.tabs.addTab(tab, "Trajectory")

    def _create_statespace_tab(self):
        """Tab 4: PCA of state embeddings."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QGridLayout(tab)
        layout.setSpacing(5)

        # PCA scatter plot
        self.pca_widget = pg.GraphicsLayoutWidget()
        self.pca_widget.setBackground('#1a1a1a')
        self.pca_plot = self.pca_widget.addPlot()
        self.pca_plot.setTitle("State Space (PCA)", color='white', size='11pt')
        self.pca_plot.setLabel('bottom', 'PC1', color='white')
        self.pca_plot.setLabel('left', 'PC2', color='white')
        self.pca_plot.showGrid(x=True, y=True, alpha=0.2)

        self.pca_scatter = pg.ScatterPlotItem(size=5, pen=None)
        self.pca_plot.addItem(self.pca_scatter)

        layout.addWidget(self.pca_widget, 0, 0)

        # Info panel
        info_widget = QtWidgets.QWidget()
        info_layout = QtWidgets.QVBoxLayout(info_widget)

        self.pca_info = QtWidgets.QTextEdit()
        self.pca_info.setReadOnly(True)
        self.pca_info.setStyleSheet("""
            background-color: #0a0a0a; color: white;
            font-family: Consolas, monospace; font-size: 10pt;
            border: 1px solid #333;
        """)
        self.pca_info.setText("""STATE SPACE ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PCA reduces 64D state embeddings
to 2D for visualization.

Clusters = behavioral modes
Gaps = unused representations

Collecting embeddings...""")
        info_layout.addWidget(self.pca_info)

        layout.addWidget(info_widget, 0, 1)
        layout.setColumnStretch(0, 2)
        layout.setColumnStretch(1, 1)

        self.tabs.addTab(tab, "State Space")

    def _create_terminal(self, layout):
        """Create terminal panel."""
        title = QtWidgets.QLabel("TERMINAL OUTPUT")
        title.setStyleSheet("""
            background-color: #2d2d2d; color: #4CAF50;
            font-weight: bold; font-size: 10pt; padding: 5px;
            border: 1px solid #4CAF50;
        """)
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        self.terminal = QtWidgets.QTextEdit()
        self.terminal.setReadOnly(True)
        self.terminal.setLineWrapMode(QtWidgets.QTextEdit.LineWrapMode.NoWrap)
        layout.addWidget(self.terminal, 2)

    def _create_description(self, layout):
        """Create description panel with timeline controls."""
        # Architecture info
        title = QtWidgets.QLabel("ARCHITECTURE")
        title.setStyleSheet("""
            background-color: #2d2d2d; color: #9C27B0;
            font-weight: bold; font-size: 10pt; padding: 5px;
            border: 1px solid #9C27B0;
        """)
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        desc = QtWidgets.QTextEdit()
        desc.setReadOnly(True)
        desc.setStyleSheet("""
            background-color: #0a0a0a; color: white;
            font-family: Consolas, monospace; font-size: 8pt;
            border: 1px solid #333;
        """)
        desc.setText("""ACTION NET: 2D→256→512→256→2D
EMBEDDER: 256D→128→64D
CURIOSITY: Predict + Infer
HOMEOSTASIS: 22 params

Flow: Pos→Embed→Act→Learn""")
        layout.addWidget(desc, 1)

        # Timeline controls (hidden until training complete)
        self.timeline_panel = QtWidgets.QWidget()
        timeline_layout = QtWidgets.QVBoxLayout(self.timeline_panel)
        timeline_layout.setContentsMargins(0, 5, 0, 0)
        timeline_layout.setSpacing(3)

        timeline_title = QtWidgets.QLabel("TIMELINE")
        timeline_title.setStyleSheet("""
            background-color: #2d2d2d; color: #FF9800;
            font-weight: bold; font-size: 10pt; padding: 5px;
            border: 1px solid #FF9800;
        """)
        timeline_title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        timeline_layout.addWidget(timeline_title)

        # Timeline slider
        self.timeline_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.timeline_slider.setMinimum(0)
        self.timeline_slider.setMaximum(100)
        self.timeline_slider.setValue(100)
        self.timeline_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #333;
                height: 8px;
                background: #1a1a1a;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #FF9800;
                border: 1px solid #FF9800;
                width: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }
            QSlider::sub-page:horizontal {
                background: #FF9800;
                border-radius: 4px;
            }
        """)
        self.timeline_slider.valueChanged.connect(self._on_timeline_changed)
        timeline_layout.addWidget(self.timeline_slider)

        # Step display
        self.timeline_label = QtWidgets.QLabel("Step: 0 / 0")
        self.timeline_label.setStyleSheet("color: #FF9800; font-size: 9pt;")
        self.timeline_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        timeline_layout.addWidget(self.timeline_label)

        # Play/Pause button
        self.play_button = QtWidgets.QPushButton("▶ Play")
        self.play_button.setStyleSheet("""
            QPushButton {
                background-color: #2d2d2d; color: #FF9800;
                border: 1px solid #FF9800; padding: 5px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #3d3d3d; }
            QPushButton:pressed { background-color: #FF9800; color: black; }
        """)
        self.play_button.clicked.connect(self._toggle_playback)
        timeline_layout.addWidget(self.play_button)

        # Playback timer
        self.playback_timer = QtCore.QTimer()
        self.playback_timer.timeout.connect(self._playback_tick)
        self.playback_speed = 50  # ms per step

        layout.addWidget(self.timeline_panel)
        self.timeline_panel.hide()  # Hidden until training complete

    def _get_action_weights(self):
        for name, param in self.model.named_parameters():
            if 'hidden_layers.3.weight' in name:
                return param.detach().cpu().numpy()
        for name, param in self.model.named_parameters():
            if 'hidden_layers.0.weight' in name:
                return param.detach().cpu().numpy()
        return np.zeros((10, 10))

    def _get_embedder_weights(self):
        for name, param in self.model.named_parameters():
            if 'state_embedder.0.weight' in name:
                return param.detach().cpu().numpy()
        return np.zeros((10, 10))

    def _get_curiosity_weights(self):
        for name, param in self.model.named_parameters():
            if 'curiosity_module.forward_model.0.weight' in name:
                return param.detach().cpu().numpy()
        return np.zeros((10, 10))

    def log(self, message):
        self.terminal_lines.append(message)

    def _decimate_display_buffers(self):
        """
        Perform 2:1 decimation on display buffers when they exceed max size.
        Averages pairs of points, halving the buffer size.
        This is called ONCE when buffer exceeds limit, not every frame.
        Old data stays visually stable after decimation.
        """
        if len(self.display_novelty) <= self.max_display_points:
            return

        # Decimate metrics buffers (average pairs)
        def decimate_list(data):
            result = []
            for i in range(0, len(data) - 1, 2):
                result.append((data[i] + data[i + 1]) / 2)
            # Handle odd length
            if len(data) % 2 == 1:
                result.append(data[-1])
            return result

        def decimate_steps(steps):
            # For steps, take the second of each pair (represents the range)
            result = []
            for i in range(0, len(steps) - 1, 2):
                result.append(steps[i + 1])
            if len(steps) % 2 == 1:
                result.append(steps[-1])
            return result

        self.display_novelty = decimate_list(self.display_novelty)
        self.display_curiosity = decimate_list(self.display_curiosity)
        self.display_reward = decimate_list(self.display_reward)
        self.display_exploration = decimate_list(self.display_exploration)
        self.display_steps = decimate_steps(self.display_steps)

    def _decimate_position_buffer(self):
        """Decimate position display buffer when it exceeds max size."""
        max_pos_points = 1000
        if len(self.display_positions) <= max_pos_points:
            return

        # For positions, take every other point (can't average 2D positions meaningfully)
        self.display_positions = self.display_positions[::2]
        self.display_pos_steps = self.display_pos_steps[::2]

    def _append_to_display(self, step, metrics):
        """
        Append new data point to display buffers.
        Called every step. Handles decimation when needed.
        """
        if step <= self.last_display_step:
            return  # Already added this step

        # Append new data
        self.display_novelty.append(metrics.get('novelty', 0))
        self.display_curiosity.append(metrics.get('curiosity', 0))
        self.display_reward.append(metrics.get('total_reward', 0))
        self.display_exploration.append(metrics.get('exploration_rate', 0))
        self.display_steps.append(step)

        pos = metrics.get('position', [0, 0])
        self.display_positions.append([float(pos[0]), float(pos[1])])
        self.display_pos_steps.append(step)

        self.last_display_step = step

        # Check if we need to decimate
        self._decimate_display_buffers()
        self._decimate_position_buffer()

    def update(self, step, metrics=None):
        """Update all visualizations and record to timeline."""
        if self.timeline_mode:
            return  # Don't update in timeline mode

        self.step = step

        if step % self.update_freq != 0:
            return

        # ALWAYS compute spring layouts every step (for accurate timeline recording)
        # This runs regardless of which tab is active
        self._update_spring_layouts(step)

        # Record to timeline (for post-run scrubbing) - MUST come after spring layout update
        if metrics:
            embedding = metrics.get('embedding', None)
            # Pass current spring positions for accurate timeline replay
            spring_positions = {
                'action': self.action_pos.copy() if self.action_pos is not None else None,
                'embedder': self.embedder_pos.copy() if self.embedder_pos is not None else None,
                'curiosity': self.curiosity_pos.copy() if self.curiosity_pos is not None else None,
            }
            self.timeline.record_step(step, metrics, embedding, spring_positions)
            self.timeline.record_checkpoint(step, self.model)

        # Get current tab index
        current_tab = self.tabs.currentIndex()

        # Always update history (keep ALL data) and display buffers
        if metrics:
            self.novelty_history.append(metrics.get('novelty', 0))
            self.curiosity_history.append(metrics.get('curiosity', 0))
            self.reward_history.append(metrics.get('total_reward', 0))
            self.exploration_history.append(metrics.get('exploration_rate', 0))
            self.loss_history.append(metrics.get('loss', 0))
            self.extrinsic_history.append(metrics.get('extrinsic', 0))
            self.intrinsic_history.append(metrics.get('intrinsic', 0))
            self.homeostatic_history.append(metrics.get('homeostatic', 0))
            self.boredom_history.append(metrics.get('boredom_penalty', 0))
            self.weight_coverage_history.append(metrics.get('weight_coverage', 0))
            self.weight_penalty_history.append(metrics.get('weight_exploration_penalty', 0))

            pos = metrics.get('position', [0, 0])
            self.position_history.append(pos)

            # Update display buffers (append-only with decimation)
            self._append_to_display(step, metrics)

            # Store embedding for PCA (every step for responsive visualization)
            embedding = metrics.get('embedding', None)
            if embedding is not None:
                self.embedding_buffer.append(embedding)
                if len(self.embedding_buffer) > self.embedding_maxlen:
                    self.embedding_buffer.pop(0)

            # Buffer updates for inactive tabs (indices 1-4 are visualization tabs)
            for tab_idx in range(1, 5):
                if tab_idx != current_tab:
                    if tab_idx not in self.pending_updates:
                        self.pending_updates[tab_idx] = []
                    self.pending_updates[tab_idx].append({'step': step, 'metrics': metrics.copy()})
                    # Keep only last 100 pending updates per tab
                    if len(self.pending_updates[tab_idx]) > 100:
                        self.pending_updates[tab_idx] = self.pending_updates[tab_idx][-100:]

        # Check if tab changed - apply pending updates
        if current_tab != self.last_active_tab:
            self._apply_pending_updates(current_tab)
            self.last_active_tab = current_tab

        # Update only active tab visuals for performance
        # Tab indices: 0=Browser, 1=Networks, 2=Metrics, 3=Trajectory, 4=State Space
        if current_tab == 1:  # Networks
            self._update_networks_tab_visuals()
        elif current_tab == 2:  # Metrics
            self._update_metrics_tab()
        elif current_tab == 3:  # Trajectory
            self._update_trajectory_tab(metrics)
        elif current_tab == 4:  # State Space
            self._update_statespace_tab(step)

        # Always update terminal
        if metrics:
            pos = metrics.get('position', [0, 0])
            nov = metrics.get('novelty', 0)
            cur = metrics.get('curiosity', 0)
            rew = metrics.get('total_reward', 0)
            exp = metrics.get('exploration_rate', 0)
            exploring = metrics.get('exploring', False)

            exp_tag = "[EXPLORING]" if exploring else ""
            line = f"[{step:5d}] pos:[{pos[0]:+.2f},{pos[1]:+.2f}] nov:{nov:.3f} cur:{cur:.4f} rew:{rew:+.3f} {exp_tag}"
            self.log(line)

            self.terminal.setText('\n'.join(self.terminal_lines))
            scrollbar = self.terminal.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

        self.app.processEvents()

    def _apply_pending_updates(self, tab_idx):
        """Apply buffered updates when switching to a tab."""
        if tab_idx not in self.pending_updates:
            return

        pending = self.pending_updates.get(tab_idx, [])
        if not pending:
            return

        # For metrics/trajectory tabs, the ring buffers already have the data
        # Just need to trigger a refresh
        # Tab indices: 0=Browser, 1=Networks, 2=Metrics, 3=Trajectory, 4=State Space
        if tab_idx == 1:  # Networks
            self._update_networks_tab_visuals()
        elif tab_idx == 2:  # Metrics
            self._update_metrics_tab()
        elif tab_idx == 3:  # Trajectory
            # Get latest metrics from pending
            if pending:
                self._update_trajectory_tab(pending[-1]['metrics'])
        elif tab_idx == 4:  # State Space
            self._update_statespace_tab(self.step, force=True)

        # Clear pending updates for this tab
        self.pending_updates[tab_idx] = []

    def _update_spring_layouts(self, step):
        """Update spring graph positions every step (regardless of active tab).

        This is called every step to ensure accurate timeline recording.
        The spring layout evolves incrementally - each step applies a small
        force-directed update to the previous positions.
        """
        action_w = self._get_action_weights()
        embedder_w = self._get_embedder_weights()
        curiosity_w = self._get_curiosity_weights()

        # Initialize positions randomly on first step (step 0)
        # This captures the initial chaotic state for timeline replay
        if self.action_pos is None:
            n_action = max(action_w.shape[0], action_w.shape[1])
            self.action_pos = np.random.randn(n_action, 2) * 2
        if self.embedder_pos is None:
            n_embedder = max(embedder_w.shape[0], embedder_w.shape[1])
            self.embedder_pos = np.random.randn(n_embedder, 2) * 2
        if self.curiosity_pos is None:
            n_curiosity = max(curiosity_w.shape[0], curiosity_w.shape[1])
            self.curiosity_pos = np.random.randn(n_curiosity, 2) * 2

        # Apply incremental force-directed layout update every step
        # Using steps=1 for small incremental updates that show gradual evolution
        self.action_pos = force_layout_fast(action_w, self.action_pos, steps=1)
        self.embedder_pos = force_layout_fast(embedder_w, self.embedder_pos, steps=1)
        self.curiosity_pos = force_layout_fast(curiosity_w, self.curiosity_pos, steps=1)

    def _update_networks_tab_visuals(self):
        """Update networks tab visuals (only when tab is active)."""
        action_w = self._get_action_weights()
        embedder_w = self._get_embedder_weights()
        curiosity_w = self._get_curiosity_weights()

        # Use symmetric range centered at 0 for proper RdBu colormap
        # Red = negative, White = zero, Blue = positive
        action_range = max(abs(action_w.min()), abs(action_w.max()), 0.01)
        self.action_img.setImage(action_w.T)
        self.action_img.setLevels([-action_range, action_range])

        embedder_range = max(abs(embedder_w.min()), abs(embedder_w.max()), 0.01)
        self.embedder_img.setImage(embedder_w.T)
        self.embedder_img.setLevels([-embedder_range, embedder_range])

        curiosity_range = max(abs(curiosity_w.min()), abs(curiosity_w.max()), 0.01)
        self.curiosity_img.setImage(curiosity_w.T)
        self.curiosity_img.setLevels([-curiosity_range, curiosity_range])

        # Update spring graph scatter plots with current positions
        if self.action_pos is not None:
            self.action_scatter.setData(pos=self.action_pos)
        if self.embedder_pos is not None:
            self.embedder_scatter.setData(pos=self.embedder_pos)
        if self.curiosity_pos is not None:
            self.curiosity_scatter.setData(pos=self.curiosity_pos)

    def _update_metrics_tab(self):
        """Update metrics tab using stable display buffers.

        Uses append-only buffers with periodic decimation.
        Old data stays frozen - only new data appends to end.
        This prevents the visual jitter from resampling every frame.
        """
        n = len(self.display_novelty)
        if n < 2:
            return

        # Use the stable display buffers (already decimated as needed)
        x = np.array(self.display_steps)
        novelty = np.array(self.display_novelty)
        curiosity = np.array(self.display_curiosity)
        reward = np.array(self.display_reward)
        exploration = np.array(self.display_exploration)

        self.novelty_curve.setData(x, novelty)
        self.curiosity_curve.setData(x, curiosity)
        self.reward_curve.setData(x, reward)
        self.exploration_curve.setData(x, exploration)

        # Don't auto-range every frame - let user zoom/pan freely
        # Only auto-range on first update
        if not hasattr(self, '_metrics_initialized'):
            self._metrics_initialized = True
            self.novelty_plot.enableAutoRange()
            self.curiosity_metric_plot.enableAutoRange()
            self.reward_plot.enableAutoRange()
            self.exploration_plot.enableAutoRange()

    def _update_trajectory_tab(self, metrics):
        """Update trajectory tab using stable display buffer.

        Uses append-only position buffer with periodic decimation.
        Old positions stay frozen - only new positions append.
        Heatmap still uses full history for accuracy.
        """
        if len(self.display_positions) < 2:
            return

        # Use stable display buffer for scatter
        display_pos = np.array(self.display_positions)
        display_steps = np.array(self.display_pos_steps)
        n = len(display_pos)

        # Color by step number (normalized to max step seen)
        max_step = max(display_steps.max(), 1)
        colors_norm = display_steps / max_step
        brushes = [pg.mkBrush(
            int(60 + 150 * (1 - c)),  # R
            int(180 * c),              # G
            int(200 * (1 - c) + 50)    # B
        ) for c in colors_norm]

        self.traj_scatter.setData(
            pos=display_pos,
            brush=brushes
        )

        # Update current position marker
        if metrics:
            pos = metrics.get('position', [0, 0])
            self.current_pos_marker.setData(pos=[pos])

        # Update heatmap (every 50 steps) - use FULL history for accuracy
        if self.step % 50 == 0 and len(self.position_history) > 10:
            positions = np.array(self.position_history)
            heatmap, xedges, yedges = np.histogram2d(
                positions[:, 0], positions[:, 1],
                bins=30, range=[[-1.1, 1.1], [-1.1, 1.1]]
            )
            self.heatmap_img.setImage(heatmap)
            self.heatmap_img.setRect(-1.1, -1.1, 2.2, 2.2)

    def _update_statespace_tab(self, step, force=False):
        """Update state space PCA tab every step."""

        if len(self.embedding_buffer) < 10:
            return

        try:
            embeddings = np.array(self.embedding_buffer)

            # Fit or transform PCA
            if not self.pca_fitted or step % 500 == 0:
                pca_result = self.pca.fit_transform(embeddings)
                self.pca_fitted = True
            else:
                pca_result = self.pca.transform(embeddings)

            # Color by time
            n = len(pca_result)
            colors_norm = np.linspace(0, 1, n)
            brushes = [pg.mkBrush(
                int(68 + 187 * c),   # viridis-like
                int(1 + 190 * c),
                int(84 + 83 * (1 - c))
            ) for c in colors_norm]

            self.pca_scatter.setData(
                pos=pca_result,
                brush=brushes
            )

            # Update info
            var1 = self.pca.explained_variance_ratio_[0] * 100
            var2 = self.pca.explained_variance_ratio_[1] * 100
            self.pca_info.setText(f"""STATE SPACE ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Embeddings: {len(embeddings)}
PC1 variance: {var1:.1f}%
PC2 variance: {var2:.1f}%
Total: {var1 + var2:.1f}%

Step: {step}

Clusters = behavioral modes
Gaps = unused representations""")

        except Exception as e:
            pass  # Silently handle PCA errors

    # ==================== TIMELINE MODE METHODS ====================

    def enter_timeline_mode(self):
        """Switch from live mode to timeline playback mode."""
        self.timeline.mark_complete()
        self.timeline_mode = True
        self.timeline_step = self.timeline.max_step

        # Show timeline controls
        self.timeline_panel.show()
        self.timeline_slider.setMaximum(self.timeline.max_step)
        self.timeline_slider.setValue(self.timeline.max_step)
        self._update_timeline_label()

        # Update window title
        self.win.setWindowTitle('SparkNet Explorer - Timeline Mode')

        # Log
        self.log("=" * 45)
        self.log("TIMELINE MODE ACTIVATED")
        self.log(f"Total steps: {self.timeline.max_step}")
        self.log(f"Checkpoints: {len(self.timeline.checkpoint_steps)}")
        self.log("Use slider to scrub through training history")
        self.log("=" * 45)

        self.terminal.setText('\n'.join(self.terminal_lines))
        self.app.processEvents()

    def _on_timeline_changed(self, value):
        """Handle timeline slider change."""
        if not self.timeline_mode:
            return

        self.timeline_step = value
        self._update_timeline_label()
        self._render_timeline_frame(value)

    def _update_timeline_label(self):
        """Update the step label display."""
        self.timeline_label.setText(f"Step: {self.timeline_step} / {self.timeline.max_step}")

    def _toggle_playback(self):
        """Toggle timeline playback."""
        if self.playback_timer.isActive():
            self.playback_timer.stop()
            self.play_button.setText("▶ Play")
        else:
            if self.timeline_step >= self.timeline.max_step:
                self.timeline_step = 0
                self.timeline_slider.setValue(0)
            self.playback_timer.start(self.playback_speed)
            self.play_button.setText("⏸ Pause")

    def _playback_tick(self):
        """Advance timeline by one step during playback."""
        self.timeline_step += 10  # Skip by 10 for smoother playback
        if self.timeline_step >= self.timeline.max_step:
            self.timeline_step = self.timeline.max_step
            self.playback_timer.stop()
            self.play_button.setText("▶ Play")

        self.timeline_slider.setValue(self.timeline_step)

    def _render_timeline_frame(self, target_step):
        """Render visualizations for a specific timeline step."""
        current_tab = self.tabs.currentIndex()

        # Get data at this step
        metrics_data = self.timeline.get_metrics_at_step(target_step)
        checkpoint, checkpoint_step = self.timeline.get_checkpoint_at_step(target_step)
        spring_pos = self.timeline.get_spring_positions_at_step(target_step)
        embeddings = self.timeline.get_embeddings_at_step(target_step)

        if not metrics_data:
            return

        # Render based on active tab
        # Tab indices: 0=Browser, 1=Networks, 2=Metrics, 3=Trajectory, 4=State Space
        if current_tab == 1:  # Networks
            self._render_timeline_networks(checkpoint, checkpoint_step, spring_pos)
        elif current_tab == 2:  # Metrics
            self._render_timeline_metrics(metrics_data)
        elif current_tab == 3:  # Trajectory
            self._render_timeline_trajectory(metrics_data)
        elif current_tab == 4:  # State Space
            self._render_timeline_statespace(embeddings)

        self.app.processEvents()

    def _render_timeline_networks(self, checkpoint, checkpoint_step, spring_pos):
        """Render networks tab from checkpoint/spring position data.

        Uses saved spring positions for accurate replay of how the spring graphs
        looked at that point in training. Spring positions are saved every step,
        while weight checkpoints are only saved every N steps.
        """
        # Spring positions are now saved every step - use them directly
        if spring_pos:
            if spring_pos.get('action') is not None:
                self.action_scatter.setData(pos=spring_pos['action'])
            if spring_pos.get('embedder') is not None:
                self.embedder_scatter.setData(pos=spring_pos['embedder'])
            if spring_pos.get('curiosity') is not None:
                self.curiosity_scatter.setData(pos=spring_pos['curiosity'])

        # Update weight matrix heatmaps from checkpoint (if available)
        if checkpoint:
            # Find weight matrices - prioritize hidden_layers.0 for action network
            action_w = None
            embedder_w = None
            curiosity_w = None

            for name, arr in checkpoint.items():
                # Match live behavior: prefer hidden_layers.3 for action (larger layer)
                if 'hidden_layers.3.weight' in name:
                    action_w = arr
                elif 'hidden_layers.0.weight' in name and action_w is None:
                    action_w = arr
                elif 'state_embedder.0.weight' in name:
                    embedder_w = arr
                elif 'curiosity_module.forward_model.0.weight' in name:
                    curiosity_w = arr

            # Update weight matrices
            if action_w is not None:
                action_range = max(abs(action_w.min()), abs(action_w.max()), 0.01)
                self.action_img.setImage(action_w.T)
                self.action_img.setLevels([-action_range, action_range])

            if embedder_w is not None:
                embedder_range = max(abs(embedder_w.min()), abs(embedder_w.max()), 0.01)
                self.embedder_img.setImage(embedder_w.T)
                self.embedder_img.setLevels([-embedder_range, embedder_range])

            if curiosity_w is not None:
                curiosity_range = max(abs(curiosity_w.min()), abs(curiosity_w.max()), 0.01)
                self.curiosity_img.setImage(curiosity_w.T)
                self.curiosity_img.setLevels([-curiosity_range, curiosity_range])

    def _render_timeline_metrics(self, metrics_data):
        """Render metrics tab from timeline data."""
        novelty = np.array(metrics_data['novelty'])
        curiosity = np.array(metrics_data['curiosity'])
        reward = np.array(metrics_data['total_reward'])
        exploration = np.array(metrics_data['exploration_rate'])

        if len(novelty) > 1:
            # Use real step numbers (0 to current timeline step)
            x = np.arange(len(novelty))
            self.novelty_curve.setData(x, novelty)
            self.curiosity_curve.setData(x, curiosity)
            self.reward_curve.setData(x, reward)
            self.exploration_curve.setData(x, exploration)

    def _render_timeline_trajectory(self, metrics_data):
        """Render trajectory tab from timeline data."""
        positions = metrics_data['positions']

        if len(positions) > 1:
            # Downsample for display
            if len(positions) > 500:
                indices = np.linspace(0, len(positions)-1, 500, dtype=int)
                display_pos = positions[indices]
                colors = indices
            else:
                display_pos = positions
                colors = np.arange(len(positions))

            colors_norm = colors / max(colors.max(), 1)
            brushes = [pg.mkBrush(
                int(60 + 150 * (1 - c)),
                int(180 * c),
                int(200 * (1 - c) + 50)
            ) for c in colors_norm]

            self.traj_scatter.setData(pos=display_pos, brush=brushes)

            # Current position marker
            current = metrics_data['current_position']
            self.current_pos_marker.setData(pos=[current])

            # Heatmap
            heatmap, _, _ = np.histogram2d(
                positions[:, 0], positions[:, 1],
                bins=30, range=[[-1.1, 1.1], [-1.1, 1.1]]
            )
            self.heatmap_img.setImage(heatmap)
            self.heatmap_img.setRect(-1.1, -1.1, 2.2, 2.2)

    def _render_timeline_statespace(self, embeddings):
        """Render state space tab from timeline embeddings."""
        if embeddings is None or len(embeddings) < 10:
            return

        try:
            pca_result = self.pca.fit_transform(embeddings)

            n = len(pca_result)
            colors_norm = np.linspace(0, 1, n)
            brushes = [pg.mkBrush(
                int(68 + 187 * c),
                int(1 + 190 * c),
                int(84 + 83 * (1 - c))
            ) for c in colors_norm]

            self.pca_scatter.setData(pos=pca_result, brush=brushes)

            var1 = self.pca.explained_variance_ratio_[0] * 100
            var2 = self.pca.explained_variance_ratio_[1] * 100
            self.pca_info.setText(f"""STATE SPACE ANALYSIS (Timeline)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Embeddings: {len(embeddings)}
PC1 variance: {var1:.1f}%
PC2 variance: {var2:.1f}%
Total: {var1 + var2:.1f}%

Timeline Step: {self.timeline_step}

Scrub slider to view history""")

        except Exception:
            pass

    def save_timeline(self, filepath):
        """Save timeline data to disk for later playback."""
        self.timeline.save_to_disk(filepath)
        self.log(f"Timeline saved to: {filepath}")

    def load_timeline(self, filepath):
        """Load timeline data from disk."""
        self.timeline.load_from_disk(filepath)
        self.enter_timeline_mode()
        self.log(f"Timeline loaded from: {filepath}")

    def get_timeline(self):
        """Get the timeline object for external saving."""
        return self.timeline

    def is_open(self):
        return self.win.isVisible()

    def close(self):
        self.win.close()


def create_live_interface_qt(model, update_freq=1, checkpoint_interval=100):
    """Factory function for PyQtGraph interface."""
    return LiveInterfaceQt(model, update_freq, checkpoint_interval)
