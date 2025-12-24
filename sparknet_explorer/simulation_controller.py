"""
SparkNet Explorer - Simulation Controller

Manages the training loop lifecycle independently from the UI.
Supports states: IDLE, RUNNING, PAUSED, STOPPED.

The controller emits signals that the UI subscribes to for updates.
This clean separation allows:
- UI to remain responsive during training
- Easy state management (play/pause/stop)
- Seamless transition to timeline mode after stopping
"""

from enum import Enum, auto
from typing import Callable, Optional, Dict, Any
import torch
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime


class SimulationState(Enum):
    """Simulation lifecycle states."""
    IDLE = auto()      # App started, no simulation running, can browse timelines
    RUNNING = auto()   # Simulation actively running
    PAUSED = auto()    # Simulation paused, can resume
    STOPPED = auto()   # Simulation stopped, timeline mode active for this run


@dataclass
class SimulationConfig:
    """Configuration for a simulation run."""
    num_steps: int = 20000
    checkpoint_interval: int = 100
    update_freq: int = 1
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Model hyperparameters
    input_dim: int = 2
    hidden_dims: tuple = (256, 512, 256)
    output_dim: int = 2
    state_embedding_dim: int = 64
    curiosity_weight: float = 0.8
    homeostasis_weight: float = 0.005
    novelty_weight: float = 0.7
    prediction_error_weight: float = 0.3

    # Exploration parameters
    initial_exploration_rate: float = 0.3
    exploration_decay: float = 0.9998

    # Environment parameters
    env_bounds: tuple = (-1, 1)
    env_dt: float = 0.1
    num_attractors: int = 4


@dataclass
class StepMetrics:
    """Metrics from a single training step."""
    step: int
    position: np.ndarray
    novelty: float
    curiosity: float
    total_reward: float
    exploration_rate: float
    exploring: bool
    embedding: Optional[np.ndarray] = None
    loss: float = 0.0

    # Additional detailed metrics
    extrinsic: float = 0.0
    intrinsic: float = 0.0
    homeostatic: float = 0.0
    boredom_penalty: float = 0.0


class SimulationController:
    """
    Controls the simulation lifecycle and training loop.

    Decoupled from UI - communicates via callbacks.
    Supports play/pause/stop operations.
    """

    def __init__(self):
        self._state = SimulationState.IDLE
        self._current_step = 0
        self._config: Optional[SimulationConfig] = None

        # Model and training components (created on start)
        self._model = None
        self._optimizer = None
        self._env = None
        self._position = None

        # Callbacks for UI updates
        self._on_state_change: Optional[Callable[[SimulationState], None]] = None
        self._on_step: Optional[Callable[[StepMetrics], None]] = None
        self._on_log: Optional[Callable[[str], None]] = None
        self._on_complete: Optional[Callable[[], None]] = None

        # Metrics history for this run
        self._metrics_history: Dict[str, list] = {}

        # Run metadata
        self._run_number: Optional[int] = None
        self._start_time: Optional[datetime] = None

    @property
    def state(self) -> SimulationState:
        return self._state

    @property
    def current_step(self) -> int:
        return self._current_step

    @property
    def config(self) -> Optional[SimulationConfig]:
        return self._config

    @property
    def model(self):
        return self._model

    @property
    def metrics_history(self) -> Dict[str, list]:
        return self._metrics_history

    @property
    def run_number(self) -> Optional[int]:
        return self._run_number

    def set_callbacks(self,
                      on_state_change: Callable[[SimulationState], None] = None,
                      on_step: Callable[[StepMetrics], None] = None,
                      on_log: Callable[[str], None] = None,
                      on_complete: Callable[[], None] = None):
        """Register callbacks for simulation events."""
        self._on_state_change = on_state_change
        self._on_step = on_step
        self._on_log = on_log
        self._on_complete = on_complete

    def _set_state(self, new_state: SimulationState):
        """Update state and notify listeners."""
        old_state = self._state
        self._state = new_state
        if self._on_state_change:
            self._on_state_change(new_state)
        self._log(f"State: {old_state.name} -> {new_state.name}")

    def _log(self, message: str):
        """Send log message to UI."""
        if self._on_log:
            self._on_log(message)

    def initialize(self, config: SimulationConfig, run_number: int):
        """
        Initialize a new simulation with given config.

        Creates model, optimizer, environment but doesn't start training.
        Call start() to begin.
        """
        from sparknet_explorer.sparknet_explorer import SparkNetExplorer

        if self._state not in (SimulationState.IDLE, SimulationState.STOPPED):
            raise RuntimeError(f"Cannot initialize in state {self._state.name}")

        self._config = config
        self._run_number = run_number
        self._current_step = 0
        self._start_time = datetime.now()

        # Reset metrics history
        self._metrics_history = {
            'extrinsic': [], 'intrinsic': [], 'novelty': [],
            'curiosity': [], 'homeostatic': [], 'boredom_penalty': [],
            'total_reward': [], 'loss': [], 'positions': [],
            'exploration_rate': []
        }

        # Create device
        device = torch.device(config.device)

        # Create model
        self._model = SparkNetExplorer(
            input_dim=config.input_dim,
            hidden_dims=list(config.hidden_dims),
            output_dim=config.output_dim,
            state_embedding_dim=config.state_embedding_dim,
            curiosity_weight=config.curiosity_weight,
            homeostasis_weight=config.homeostasis_weight,
            novelty_weight=config.novelty_weight,
            prediction_error_weight=config.prediction_error_weight,
            device=device
        )

        self._model.exploration_rate = config.initial_exploration_rate
        self._model.exploration_decay = config.exploration_decay

        # Create optimizer
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=1e-3)

        # Create environment
        self._env = self._create_environment(config)
        self._position = self._env.reset()

        self._log(f"Initialized Run #{run_number}")
        self._log(f"Device: {device} | Steps: {config.num_steps}")
        self._log(f"Checkpoints: every {config.checkpoint_interval} steps")

    def _create_environment(self, config: SimulationConfig):
        """Create the exploration environment."""
        # Import here to avoid circular imports
        from examples.live_exploration_qt import ContinuousExplorationEnvironment
        return ContinuousExplorationEnvironment(
            bounds=config.env_bounds,
            dt=config.env_dt,
            num_attractors=config.num_attractors
        )

    def start(self):
        """Start or resume the simulation."""
        if self._state == SimulationState.IDLE:
            if self._model is None:
                raise RuntimeError("Must call initialize() before start()")
            self._set_state(SimulationState.RUNNING)
            self._log("Simulation started")
        elif self._state == SimulationState.PAUSED:
            self._set_state(SimulationState.RUNNING)
            self._log("Simulation resumed")
        else:
            raise RuntimeError(f"Cannot start in state {self._state.name}")

    def pause(self):
        """Pause the simulation."""
        if self._state == SimulationState.RUNNING:
            self._set_state(SimulationState.PAUSED)
            self._log(f"Paused at step {self._current_step}")
        else:
            raise RuntimeError(f"Cannot pause in state {self._state.name}")

    def stop(self):
        """Stop the simulation permanently. Activates timeline mode."""
        if self._state in (SimulationState.RUNNING, SimulationState.PAUSED):
            self._set_state(SimulationState.STOPPED)
            self._log(f"Stopped at step {self._current_step}")
            if self._on_complete:
                self._on_complete()
        else:
            raise RuntimeError(f"Cannot stop in state {self._state.name}")

    def reset(self):
        """Reset to IDLE state for a new run."""
        self._state = SimulationState.IDLE
        self._current_step = 0
        self._model = None
        self._optimizer = None
        self._env = None
        self._position = None
        self._config = None
        self._metrics_history = {}
        self._run_number = None
        self._start_time = None

    def execute_step(self) -> Optional[StepMetrics]:
        """
        Execute a single training step.

        Returns StepMetrics if step was executed, None if not running.
        Call this in a loop from the UI's event loop.
        """
        if self._state != SimulationState.RUNNING:
            return None

        if self._current_step >= self._config.num_steps:
            self.stop()
            return None

        device = torch.device(self._config.device)

        # Forward pass
        current_pos = self._position.unsqueeze(0).to(device)
        action, exploring = self._model.exploration_step(current_pos)

        # Perturbation every 50 steps
        if self._current_step % 50 == 0 and self._current_step > 0:
            action = action + torch.randn_like(action) * 0.3

        # Environment step
        next_position, env_reward, _, _ = self._env.step(action.cpu())
        next_pos = next_position.unsqueeze(0).to(device)

        # Compute reward and train
        self._optimizer.zero_grad()
        total_reward, metrics = self._model.compute_total_reward(
            current_pos, next_pos, next_pos
        )

        loss = -total_reward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
        self._optimizer.step()

        # Update curiosity
        with torch.no_grad():
            cur_emb, _ = self._model.get_state_embedding(current_pos)
            nxt_emb, _ = self._model.get_state_embedding(next_pos)
        self._model.curiosity_module.update(cur_emb.detach(), action.detach(), nxt_emb.detach())

        # Get embedding for PCA visualization
        with torch.no_grad():
            embedding, _ = self._model.get_state_embedding(current_pos)
            embedding_np = embedding.squeeze().cpu().numpy()

        # Update position
        self._position = next_position.detach()

        # Track metrics history
        self._metrics_history['extrinsic'].append(metrics['extrinsic'])
        self._metrics_history['intrinsic'].append(metrics['intrinsic'])
        self._metrics_history['novelty'].append(metrics['novelty'])
        self._metrics_history['curiosity'].append(metrics['curiosity'])
        self._metrics_history['homeostatic'].append(metrics['homeostatic_penalty'])
        self._metrics_history['boredom_penalty'].append(metrics['boredom_penalty'])
        self._metrics_history['total_reward'].append(metrics['total_reward'])
        self._metrics_history['loss'].append(loss.item())
        self._metrics_history['positions'].append(self._position.numpy().copy())
        self._metrics_history['exploration_rate'].append(self._model.exploration_rate)

        # Create step metrics
        step_metrics = StepMetrics(
            step=self._current_step + 1,  # 1-indexed for display
            position=self._position.numpy().copy(),
            novelty=metrics['novelty'],
            curiosity=metrics['curiosity'],
            total_reward=metrics['total_reward'],
            exploration_rate=self._model.exploration_rate,
            exploring=exploring,
            embedding=embedding_np,
            loss=loss.item(),
            extrinsic=metrics['extrinsic'],
            intrinsic=metrics['intrinsic'],
            homeostatic=metrics['homeostatic_penalty'],
            boredom_penalty=metrics['boredom_penalty']
        )

        self._current_step += 1

        # Notify UI
        if self._on_step:
            self._on_step(step_metrics)

        # Console log every 1000
        if self._current_step % 1000 == 0:
            buffer = self._model.experience_buffer.get_coverage_stats()
            self._log(f"Step {self._current_step}/{self._config.num_steps} | "
                     f"Nov: {metrics['novelty']:.3f} | "
                     f"Cov: {buffer['coverage']*100:.0f}%")

        return step_metrics

    def get_progress(self) -> float:
        """Get training progress as fraction 0-1."""
        if self._config is None:
            return 0.0
        return self._current_step / self._config.num_steps

    def is_complete(self) -> bool:
        """Check if training reached target steps."""
        if self._config is None:
            return False
        return self._current_step >= self._config.num_steps
