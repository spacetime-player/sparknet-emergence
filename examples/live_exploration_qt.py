"""
Live Exploration Task - PyQtGraph Version

High-performance real-time visualization with per-step updates.
Uses OpenGL acceleration for smooth 60fps rendering.

Features:
- Live training visualization with per-step updates
- Timeline mode: After training, scrub through history like a video
- Parameter checkpoints saved every 100 steps
- PNG export only at end of run

Run: python examples/live_exploration_qt.py
"""

import torch
import numpy as np
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sparknet_explorer.sparknet_explorer import SparkNetExplorer
from sparknet_explorer.live_interface_qt import create_live_interface_qt
from sparknet_explorer.visualize import plot_training_metrics, plot_state_space_exploration, plot_trajectory


class TeeOutput:
    """Capture output to both terminal and file."""
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log = open(file_path, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


class ContinuousExplorationEnvironment:
    """
    Full 2D continuous exploration environment with triple-defense.

    Matches simple_exploration_task.py exactly for reproducible 4-cluster behavior.
    Triple defense: (1) Attractors at spawn, (2) Boundary noise, (3) Satiation + rotation
    """

    def __init__(self, bounds=(-1, 1), dt=0.1, num_attractors=4):
        self.bounds = bounds
        self.dt = dt
        self.position = torch.tensor([0.0, 0.0])
        self.trajectory = [self.position.clone().numpy()]

        # Dynamic attractors (reward zones that shift)
        self.num_attractors = num_attractors
        self.attractors = []
        self.attractor_lifetimes = []
        self.time_in_zone = {}
        self.step_count = 0
        self._init_attractors()

    def _init_attractors(self):
        """Initialize attractor positions - first at spawn, others nearby."""
        for i in range(self.num_attractors):
            if i == 0:
                # SOLUTION 1: First attractor AT spawn to intercept agent immediately
                pos = torch.tensor([0.0, 0.0])
            else:
                # Others in ring around spawn (radius 0.3)
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
        # Move based on action
        self.position = self.position + action.squeeze() * self.dt
        self.step_count += 1

        # 1. Boundary penalty (stronger push away from edges)
        boundary_penalty = 0.0
        margin = 0.3  # Larger safety margin

        for i in range(2):  # x and y
            dist_from_max = self.bounds[1] - self.position[i]
            dist_from_min = self.position[i] - self.bounds[0]

            if dist_from_max < margin:
                boundary_penalty += (margin - dist_from_max) ** 2
            if dist_from_min < margin:
                boundary_penalty += (margin - dist_from_min) ** 2

        # Clip to bounds (hard constraint)
        self.position = torch.clamp(self.position, self.bounds[0], self.bounds[1])

        # SOLUTION 2: Boundary noise - make corners unpredictable
        # When near boundaries, add random kicks to break predictability
        corner_distance = min(
            abs(self.position[0] - self.bounds[0]),
            abs(self.position[0] - self.bounds[1]),
            abs(self.position[1] - self.bounds[0]),
            abs(self.position[1] - self.bounds[1])
        )
        if corner_distance < 0.15:  # Very close to edge
            # Random kick away from boundary (breaks zero prediction error)
            noise_strength = (0.15 - corner_distance) / 0.15  # Stronger when closer
            boundary_noise = torch.randn(2) * 0.2 * noise_strength
            self.position = self.position + boundary_noise
            # Re-clip after noise
            self.position = torch.clamp(self.position, self.bounds[0], self.bounds[1])

        # 2. Attractor rewards (with satiation)
        attractor_reward = 0.0
        attractor_zone_radius = 0.25

        for i, attractor_pos in enumerate(self.attractors):
            dist_to_attractor = torch.norm(self.position - attractor_pos).item()

            if dist_to_attractor < attractor_zone_radius:
                # Inside attractor zone
                self.time_in_zone[i] += 1

                # Satiation: reward decreases with time spent in zone
                satiation_factor = np.exp(-self.time_in_zone[i] / 50.0)  # Decay over 50 steps
                zone_reward = (1.0 - dist_to_attractor / attractor_zone_radius) * satiation_factor
                attractor_reward += zone_reward

                # Rotate attractor if overstayed (force exploration)
                if self.time_in_zone[i] > 100:
                    self.attractors[i] = torch.rand(2) * 1.2 - 0.6
                    self.time_in_zone[i] = 0
            else:
                # Reset satiation when leaving zone
                self.time_in_zone[i] = max(0, self.time_in_zone[i] - 1)

            # Slowly drift attractors (every 200 steps)
            self.attractor_lifetimes[i] += 1
            if self.attractor_lifetimes[i] > 200:
                self.attractors[i] = torch.rand(2) * 1.2 - 0.6
                self.attractor_lifetimes[i] = 0

        # Track trajectory
        self.trajectory.append(self.position.clone().numpy())

        # Total reward = attractors - boundary penalty (5x multiplier)
        reward = attractor_reward - boundary_penalty * 5.0

        info = {
            'position': self.position.clone(),
            'boundary_penalty': boundary_penalty,
            'attractor_reward': attractor_reward
        }

        return self.position.clone(), reward, False, info


def get_next_run_number(archive_dir='exploration_runs'):
    """Scan archive directory and return next available run number."""
    import re
    if not os.path.exists(archive_dir):
        return 1

    max_run = 0
    pattern = re.compile(r'Run_(\d+)_')

    # Check all files and subdirectories
    for item in os.listdir(archive_dir):
        match = pattern.match(item)
        if match:
            run_num = int(match.group(1))
            max_run = max(max_run, run_num)

    # Also check logs subdirectory
    log_dir = os.path.join(archive_dir, 'logs')
    if os.path.exists(log_dir):
        for item in os.listdir(log_dir):
            match = pattern.match(item)
            if match:
                run_num = int(match.group(1))
                max_run = max(max_run, run_num)

    return max_run + 1


def run_live_exploration_qt(num_steps=20000, update_freq=1, run_number=None, checkpoint_interval=100):
    """
    Run exploration with PyQtGraph live visualization.

    Args:
        num_steps: Total training steps
        update_freq: Update viz every N steps (1 = smoothest)
        run_number: Run number for file naming (e.g., 10 -> "Run_10_...")
        checkpoint_interval: Save parameter checkpoints every N steps (for timeline)
    """
    # Setup directories
    archive_dir = 'exploration_runs'

    # Auto-increment run number if not specified
    if run_number is None:
        run_number = get_next_run_number(archive_dir)

    print("=" * 70)
    print(f"SPARKNET EXPLORER - LIVE TRAINING (PyQtGraph) - RUN #{run_number}")
    print("=" * 70)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_prefix = f"Run_{run_number:02d}_"
    log_dir = os.path.join(archive_dir, 'logs')
    timeline_dir = os.path.join(archive_dir, 'timelines')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(timeline_dir, exist_ok=True)

    # Setup log file capture
    log_file = os.path.join(log_dir, f'{run_prefix}log_{timestamp}.md')
    tee = TeeOutput(log_file)
    sys.stdout = tee

    # Environment
    env = ContinuousExplorationEnvironment()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Model
    model = SparkNetExplorer(
        input_dim=2,
        hidden_dims=[256, 512, 256],
        output_dim=2,
        state_embedding_dim=64,
        curiosity_weight=0.8,
        homeostasis_weight=0.005,
        novelty_weight=0.7,
        prediction_error_weight=0.3,
        device=device
    )

    model.exploration_rate = 0.3
    model.exploration_decay = 0.9998

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Create PyQtGraph interface with timeline support
    interface = create_live_interface_qt(model, update_freq=update_freq, checkpoint_interval=checkpoint_interval)
    interface.log(f"Training started: {timestamp}")
    interface.log(f"Device: {device} | Steps: {num_steps}")
    interface.log(f"Checkpoints: every {checkpoint_interval} steps")
    interface.log("-" * 45)

    # Init
    position = env.reset()

    metrics_history = {
        'extrinsic': [], 'intrinsic': [], 'novelty': [],
        'curiosity': [], 'homeostatic': [], 'boredom_penalty': [],
        'total_reward': [], 'loss': [], 'positions': [],
        'exploration_rate': []
    }

    print(f"Starting training with update_freq={update_freq}...")
    print("Close the window to stop.\n")

    try:
        for step in range(num_steps):
            # Check if window closed
            if not interface.is_open():
                print("\nWindow closed. Stopping.")
                break

            # Forward pass
            current_pos = position.unsqueeze(0).to(device)
            action, exploring = model.exploration_step(current_pos)

            # Perturbation every 50 steps
            if step % 50 == 0 and step > 0:
                action = action + torch.randn_like(action) * 0.3

            # Environment step
            next_position, env_reward, _, _ = env.step(action.cpu())
            next_pos = next_position.unsqueeze(0).to(device)

            # Compute reward and train
            optimizer.zero_grad()
            total_reward, metrics = model.compute_total_reward(
                current_pos, next_pos, next_pos
            )

            loss = -total_reward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Update curiosity
            with torch.no_grad():
                cur_emb, _ = model.get_state_embedding(current_pos)
                nxt_emb, _ = model.get_state_embedding(next_pos)
            model.curiosity_module.update(cur_emb.detach(), action.detach(), nxt_emb.detach())

            # Track metrics
            metrics_history['extrinsic'].append(metrics['extrinsic'])
            metrics_history['intrinsic'].append(metrics['intrinsic'])
            metrics_history['novelty'].append(metrics['novelty'])
            metrics_history['curiosity'].append(metrics['curiosity'])
            metrics_history['homeostatic'].append(metrics['homeostatic_penalty'])
            metrics_history['boredom_penalty'].append(metrics['boredom_penalty'])
            metrics_history['total_reward'].append(metrics['total_reward'])
            metrics_history['loss'].append(loss.item())
            metrics_history['positions'].append(position.numpy().copy())
            metrics_history['exploration_rate'].append(model.exploration_rate)

            position = next_position.detach()

            # Get embedding for PCA visualization
            with torch.no_grad():
                embedding, _ = model.get_state_embedding(current_pos)
                embedding_np = embedding.squeeze().cpu().numpy()

            # Update interface
            interface.update(step + 1, {
                'position': position.numpy(),
                'novelty': metrics['novelty'],
                'curiosity': metrics['curiosity'],
                'total_reward': metrics['total_reward'],
                'exploration_rate': model.exploration_rate,
                'exploring': exploring,
                'embedding': embedding_np,
            })

            # Console log every 1000
            if (step + 1) % 1000 == 0:
                buffer = model.experience_buffer.get_coverage_stats()
                print(f"Step {step+1}/{num_steps} | "
                      f"Nov: {metrics['novelty']:.3f} | "
                      f"Cov: {buffer['coverage']*100:.0f}%")

    except KeyboardInterrupt:
        print("\nInterrupted.")

    finally:
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE - Entering Timeline Mode")
        print("=" * 70)

        # Save PNG files (only at end of run)
        if len(metrics_history['positions']) > 100:
            plot_training_metrics(
                metrics_history,
                save_path=os.path.join(archive_dir, f'{run_prefix}metrics_{timestamp}.png')
            )
            plot_state_space_exploration(
                model.experience_buffer,
                save_path=os.path.join(archive_dir, f'{run_prefix}state_space_{timestamp}.png')
            )
            plot_trajectory(
                np.array(metrics_history['positions']),
                save_path=os.path.join(archive_dir, f'{run_prefix}trajectory_{timestamp}.png')
            )
            print(f"PNGs saved to {archive_dir}/{run_prefix}*_{timestamp}.png")

        # Save timeline data for later playback
        timeline_path = os.path.join(timeline_dir, f'{run_prefix}timeline_{timestamp}')
        interface.save_timeline(timeline_path)
        print(f"Timeline saved to {timeline_path}")

        # Enter timeline mode - window stays open for scrubbing
        interface.enter_timeline_mode()

        print("\nTimeline mode active. Use the slider to scrub through training history.")
        print("Close the window when done exploring.\n")

        # Keep window open for timeline exploration
        while interface.is_open():
            interface.app.processEvents()

        # Close log file and restore stdout
        print(f"\nLog saved to: {log_file}")
        tee.close()
        sys.stdout = tee.terminal

    return model, metrics_history


if __name__ == '__main__':
    # Run with per-step updates for smooth visualization
    # run_number auto-increments based on existing files in exploration_runs/
    # checkpoint_interval=100 saves network state every 100 steps for timeline
    model, history = run_live_exploration_qt(
        num_steps=20000,  # 20k steps
        update_freq=1,
        # run_number=None means auto-increment from last run
        checkpoint_interval=100
    )
