"""
SparkNet Explorer - Main Application

Full-featured application with:
- Timeline browser for loading saved runs
- Play/Pause/Stop controls for simulation
- Automatic transition to timeline mode after stopping
- Real-time visualization during training

Usage:
    python examples/sparknet_app.py

The app starts in IDLE state showing the browser.
- Click "New Simulation" to initialize a fresh run
- Click "Play" to start training
- Click "Pause" to pause (resume with Play)
- Click "Stop" to stop and enter timeline mode
- Or load a saved timeline from the browser

"""

import torch
import numpy as np
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sparknet_explorer.sparknet_explorer import SparkNetExplorer
from sparknet_explorer.live_interface_qt import LiveInterfaceQt, AppState
from sparknet_explorer.simulation_controller import SimulationController, SimulationConfig, SimulationState
from sparknet_explorer.visualize import plot_training_metrics, plot_state_space_exploration, plot_trajectory


class ContinuousExplorationEnvironment:
    """Full 2D continuous exploration environment."""

    def __init__(self, bounds=(-1, 1), dt=0.1, num_attractors=4):
        self.bounds = bounds
        self.dt = dt
        self.position = torch.tensor([0.0, 0.0])
        self.trajectory = [self.position.clone().numpy()]
        self.num_attractors = num_attractors
        self.attractors = []
        self.attractor_lifetimes = []
        self.time_in_zone = {}
        self.step_count = 0
        self._init_attractors()

    def _init_attractors(self):
        for i in range(self.num_attractors):
            if i == 0:
                pos = torch.tensor([0.0, 0.0])
            else:
                angle = (i / self.num_attractors) * 2 * np.pi
                radius = 0.3
                pos = torch.tensor([radius * np.cos(angle), radius * np.sin(angle)])
            self.attractors.append(pos)
            self.attractor_lifetimes.append(0)
            self.time_in_zone[i] = 0

    def reset(self):
        self.position = torch.tensor([0.0, 0.0])
        self.trajectory = [self.position.clone().numpy()]
        self.time_in_zone = {i: 0 for i in range(self.num_attractors)}
        self.step_count = 0
        return self.position.clone()

    def step(self, action):
        self.position = self.position + action.squeeze() * self.dt
        self.step_count += 1

        boundary_penalty = 0.0
        margin = 0.3

        for i in range(2):
            dist_from_max = self.bounds[1] - self.position[i]
            dist_from_min = self.position[i] - self.bounds[0]
            if dist_from_max < margin:
                boundary_penalty += (margin - dist_from_max) ** 2
            if dist_from_min < margin:
                boundary_penalty += (margin - dist_from_min) ** 2

        self.position = torch.clamp(self.position, self.bounds[0], self.bounds[1])

        corner_distance = min(
            abs(self.position[0] - self.bounds[0]),
            abs(self.position[0] - self.bounds[1]),
            abs(self.position[1] - self.bounds[0]),
            abs(self.position[1] - self.bounds[1])
        )
        if corner_distance < 0.15:
            noise_strength = (0.15 - corner_distance) / 0.15
            boundary_noise = torch.randn(2) * 0.2 * noise_strength
            self.position = self.position + boundary_noise
            self.position = torch.clamp(self.position, self.bounds[0], self.bounds[1])

        attractor_reward = 0.0
        attractor_zone_radius = 0.25

        for i, attractor_pos in enumerate(self.attractors):
            dist_to_attractor = torch.norm(self.position - attractor_pos).item()
            if dist_to_attractor < attractor_zone_radius:
                self.time_in_zone[i] += 1
                satiation_factor = np.exp(-self.time_in_zone[i] / 50.0)
                zone_reward = (1.0 - dist_to_attractor / attractor_zone_radius) * satiation_factor
                attractor_reward += zone_reward
                if self.time_in_zone[i] > 100:
                    self.attractors[i] = torch.rand(2) * 1.2 - 0.6
                    self.time_in_zone[i] = 0
            else:
                self.time_in_zone[i] = max(0, self.time_in_zone[i] - 1)

            self.attractor_lifetimes[i] += 1
            if self.attractor_lifetimes[i] > 200:
                self.attractors[i] = torch.rand(2) * 1.2 - 0.6
                self.attractor_lifetimes[i] = 0

        self.trajectory.append(self.position.clone().numpy())
        reward = attractor_reward - boundary_penalty * 5.0

        info = {
            'position': self.position.clone(),
            'boundary_penalty': boundary_penalty,
            'attractor_reward': attractor_reward
        }
        return self.position.clone(), reward, False, info


def get_next_run_number(archive_dir='exploration_runs'):
    """Get next available run number."""
    import re
    if not os.path.exists(archive_dir):
        return 1
    max_run = 0
    pattern = re.compile(r'Run_(\d+)_')
    for item in os.listdir(archive_dir):
        match = pattern.match(item)
        if match:
            max_run = max(max_run, int(match.group(1)))
    log_dir = os.path.join(archive_dir, 'logs')
    if os.path.exists(log_dir):
        for item in os.listdir(log_dir):
            match = pattern.match(item)
            if match:
                max_run = max(max_run, int(match.group(1)))
    return max_run + 1


class SparkNetApp:
    """
    Main application class that coordinates UI and simulation.

    Manages the lifecycle:
    IDLE -> (New Simulation) -> IDLE with model ready
    IDLE -> (Play) -> RUNNING
    RUNNING -> (Pause) -> PAUSED
    PAUSED -> (Play) -> RUNNING
    RUNNING/PAUSED -> (Stop) -> STOPPED (timeline mode)
    STOPPED -> (via browser) -> IDLE
    """

    def __init__(self):
        self.archive_dir = 'exploration_runs'
        self.timeline_dir = os.path.join(self.archive_dir, 'timelines')
        self.log_dir = os.path.join(self.archive_dir, 'logs')

        # Ensure directories exist
        os.makedirs(self.timeline_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # Simulation state
        self.config = SimulationConfig(num_steps=20000)
        self.model = None
        self.optimizer = None
        self.env = None
        self.position = None
        self.run_number = None
        self.start_time = None
        self.current_step = 0

        # Create interface (starts in IDLE/browser mode)
        self.interface = LiveInterfaceQt(
            model=None,
            update_freq=1,
            checkpoint_interval=100,
            timeline_dir=self.timeline_dir,
            start_in_browser=True
        )

        # Set up callbacks
        self.interface.set_callbacks(
            on_play=self._on_play,
            on_pause=self._on_pause,
            on_stop=self._on_stop,
            on_new_simulation=self._on_new_simulation,
            on_load_timeline=self._on_load_timeline
        )

        # Simulation timer for running steps
        self.sim_timer = self.interface.app.instance()

    def _on_new_simulation(self):
        """Initialize a new simulation."""
        self.run_number = get_next_run_number(self.archive_dir)
        self.start_time = datetime.now()
        self.current_step = 0

        # Device
        device = torch.device(self.config.device)

        # Create model
        self.model = SparkNetExplorer(
            input_dim=self.config.input_dim,
            hidden_dims=list(self.config.hidden_dims),
            output_dim=self.config.output_dim,
            state_embedding_dim=self.config.state_embedding_dim,
            curiosity_weight=self.config.curiosity_weight,
            homeostasis_weight=self.config.homeostasis_weight,
            novelty_weight=self.config.novelty_weight,
            prediction_error_weight=self.config.prediction_error_weight,
            device=device
        )
        self.model.exploration_rate = self.config.initial_exploration_rate
        self.model.exploration_decay = self.config.exploration_decay

        # Create optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        # Create environment
        self.env = ContinuousExplorationEnvironment(
            bounds=self.config.env_bounds,
            dt=self.config.env_dt,
            num_attractors=self.config.num_attractors
        )
        self.position = self.env.reset()

        # Update interface
        self.interface.set_model(self.model)
        self.interface.reset_for_new_run()

        self.interface.log(f"Initialized Run #{self.run_number}")
        self.interface.log(f"Device: {device} | Steps: {self.config.num_steps}")
        self.interface.log("-" * 40)
        self.interface.log("Press Play to start training")

        # Update progress bar
        self.interface.update_progress(0, self.config.num_steps)

    def _on_play(self):
        """Start or resume simulation."""
        if self.model is None:
            # No simulation initialized, create one first
            self._on_new_simulation()

        self.interface.set_state(AppState.RUNNING)
        self.interface.log("Training started" if self.current_step == 0 else "Training resumed")

    def _on_pause(self):
        """Pause simulation."""
        self.interface.set_state(AppState.PAUSED)
        self.interface.log(f"Paused at step {self.current_step}")

    def _on_stop(self):
        """Stop simulation and enter timeline mode."""
        self.interface.set_state(AppState.STOPPED)
        self.interface.log(f"Stopped at step {self.current_step}")

        # Save timeline
        if self.run_number and self.start_time:
            timestamp = self.start_time.strftime("%Y-%m-%d_%H-%M-%S")
            timeline_path = os.path.join(
                self.timeline_dir,
                f'Run_{self.run_number:02d}_timeline_{timestamp}'
            )
            self.interface.save_timeline(timeline_path)
            self.interface.log(f"Timeline saved")

            # Save PNG visualizations
            if self.current_step > 100:
                self._save_visualizations(timestamp)

        # Enter timeline mode
        self.interface.enter_timeline_mode()
        self.interface.log("Timeline mode active - use slider to scrub")

        # Refresh browser for new timeline
        self.interface.browser_tab.refresh()

    def _save_visualizations(self, timestamp):
        """Save PNG visualizations at end of run."""
        run_prefix = f'Run_{self.run_number:02d}_'

        metrics_history = {
            'novelty': self.interface.novelty_history,
            'curiosity': self.interface.curiosity_history,
            'total_reward': self.interface.reward_history,
            'exploration_rate': self.interface.exploration_history,
            'positions': self.interface.position_history,
            'extrinsic': self.interface.extrinsic_history,
            'intrinsic': self.interface.intrinsic_history,
            'homeostatic': self.interface.homeostatic_history,
            'boredom_penalty': self.interface.boredom_history,
            'loss': self.interface.loss_history,
        }

        try:
            plot_training_metrics(
                metrics_history,
                save_path=os.path.join(self.archive_dir, f'{run_prefix}metrics_{timestamp}.png')
            )
            if self.model:
                plot_state_space_exploration(
                    self.model.experience_buffer,
                    save_path=os.path.join(self.archive_dir, f'{run_prefix}state_space_{timestamp}.png')
                )
            if self.interface.position_history:
                plot_trajectory(
                    np.array(self.interface.position_history),
                    save_path=os.path.join(self.archive_dir, f'{run_prefix}trajectory_{timestamp}.png')
                )
            self.interface.log(f"Visualizations saved to {self.archive_dir}/")
        except Exception as e:
            self.interface.log(f"Warning: Could not save visualizations: {e}")

    def _on_load_timeline(self, filepath: str):
        """Load a saved timeline for playback."""
        self.interface.load_timeline(filepath)
        self.interface.set_state(AppState.STOPPED)
        self.interface.enter_timeline_mode()

    def _execute_step(self):
        """Execute a single training step."""
        if self.interface.app_state != AppState.RUNNING:
            return

        if self.current_step >= self.config.num_steps:
            self._on_stop()
            return

        device = torch.device(self.config.device)

        # Forward pass
        current_pos = self.position.unsqueeze(0).to(device)
        action, exploring = self.model.exploration_step(current_pos)

        # Perturbation
        if self.current_step % 50 == 0 and self.current_step > 0:
            action = action + torch.randn_like(action) * 0.3

        # Environment step
        next_position, env_reward, _, _ = self.env.step(action.cpu())
        next_pos = next_position.unsqueeze(0).to(device)

        # Train
        self.optimizer.zero_grad()
        total_reward, metrics = self.model.compute_total_reward(current_pos, next_pos, next_pos)
        loss = -total_reward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update curiosity
        with torch.no_grad():
            cur_emb, _ = self.model.get_state_embedding(current_pos)
            nxt_emb, _ = self.model.get_state_embedding(next_pos)
        self.model.curiosity_module.update(cur_emb.detach(), action.detach(), nxt_emb.detach())

        # Get embedding
        with torch.no_grad():
            embedding, _ = self.model.get_state_embedding(current_pos)
            embedding_np = embedding.squeeze().cpu().numpy()

        self.position = next_position.detach()
        self.current_step += 1

        # Update interface
        self.interface.update(self.current_step, {
            'position': self.position.numpy(),
            'novelty': metrics['novelty'],
            'curiosity': metrics['curiosity'],
            'total_reward': metrics['total_reward'],
            'exploration_rate': self.model.exploration_rate,
            'exploring': exploring,
            'embedding': embedding_np,
            # Additional metrics for visualization
            'extrinsic': metrics['extrinsic'],
            'intrinsic': metrics['intrinsic'],
            'homeostatic': metrics['homeostatic_penalty'],
            'boredom_penalty': metrics['boredom_penalty'],
            'loss': loss.item(),
            # Weight exploration metrics
            'weight_coverage': metrics.get('weight_coverage', 0),
            'weight_exploration_penalty': metrics.get('weight_exploration_penalty', 0),
        })

        # Update progress
        self.interface.update_progress(self.current_step, self.config.num_steps)

    def run(self):
        """Main application loop."""
        print("=" * 60)
        print("SPARKNET EXPLORER")
        print("=" * 60)
        print("Starting application...")
        print("Close the window to exit.\n")

        try:
            while self.interface.is_open():
                # Execute step if running
                if self.interface.app_state == AppState.RUNNING:
                    self._execute_step()

                # Process Qt events
                self.interface.app.processEvents()

        except KeyboardInterrupt:
            print("\nInterrupted.")

        finally:
            print("\nApplication closed.")


def main():
    """Entry point."""
    app = SparkNetApp()
    app.run()


if __name__ == '__main__':
    main()
