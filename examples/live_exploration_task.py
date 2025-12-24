"""
Live Exploration Task - SparkNet Explorer with Real-Time Visualization

Same as simple_exploration_task.py but with live interface showing:
- Weight matrices for all 3 networks
- Spring-physics graph layouts
- Terminal output with live stats
- Network architecture description

Run this to see the network learn in real-time.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sparknet_explorer.sparknet_explorer import SparkNetExplorer
from sparknet_explorer.live_interface import LiveInterface
from sparknet_explorer.visualize import plot_training_metrics, plot_state_space_exploration

plt.style.use('dark_background')


class ContinuousExplorationEnvironment:
    """
    Simple 2D continuous exploration environment.
    Agent starts at origin and can move in any direction.
    """

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
        """Initialize attractor positions - first at spawn, others nearby."""
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
        """Reset to origin."""
        self.position = torch.tensor([0.0, 0.0])
        self.trajectory = [self.position.clone().numpy()]
        self.time_in_zone = {i: 0 for i in range(self.num_attractors)}
        self.step_count = 0
        return self.position.clone()

    def step(self, action):
        """Take action and update position."""
        self.position = self.position + action.squeeze() * self.dt
        self.step_count += 1

        # Boundary penalty
        boundary_dist = torch.abs(self.position) - (self.bounds[1] - 0.1)
        boundary_penalty = -0.5 * torch.sum(torch.relu(boundary_dist))

        # SOLUTION 2: Add noise near boundaries
        if torch.abs(self.position).max() > 0.8:
            noise = torch.randn(2) * 0.1
            self.position = self.position + noise

        # Clip
        self.position = torch.clamp(
            self.position,
            self.bounds[0],
            self.bounds[1]
        )

        # Attractor reward
        attractor_reward = 0.0
        attractor_radius = 0.2
        satiation_limit = 50

        for i, attractor in enumerate(self.attractors):
            dist = torch.norm(self.position - attractor)
            if dist < attractor_radius:
                self.time_in_zone[i] += 1
                if self.time_in_zone[i] < satiation_limit:
                    reward_scale = 1.0 - (self.time_in_zone[i] / satiation_limit)
                    attractor_reward += 0.3 * reward_scale * (1 - dist / attractor_radius)
            else:
                self.time_in_zone[i] = max(0, self.time_in_zone[i] - 1)

        self.trajectory.append(self.position.clone().numpy())
        total_reward = boundary_penalty + attractor_reward

        return self.position.clone(), total_reward, False, {}


def run_live_exploration(num_steps=20000, update_freq=100):
    """
    Run exploration with live visualization interface.

    Args:
        num_steps: Total training steps
        update_freq: How often to update visualization
    """
    print("=" * 80)
    print("SPARKNET EXPLORER - LIVE TRAINING INTERFACE")
    print("=" * 80)

    # Setup logging
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = 'exploration_runs/logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'live_run_{timestamp}.md')

    # Environment
    env = ContinuousExplorationEnvironment()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Initialize network
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

    # Create live interface
    interface = LiveInterface(model, update_freq=update_freq)
    interface.log(f"Training started at {timestamp}")
    interface.log(f"Device: {device}")
    interface.log(f"Steps: {num_steps}")
    interface.log("-" * 40)

    # Start position
    position = env.reset()

    # Tracking
    metrics_history = {
        'extrinsic': [], 'intrinsic': [], 'novelty': [],
        'curiosity': [], 'homeostatic': [], 'boredom_penalty': [],
        'total_reward': [], 'loss': [], 'positions': [],
        'exploration_rate': []
    }

    print("Starting live training...")
    print("Close the visualization window to stop.\n")

    try:
        for step in range(num_steps):
            # Check if window closed
            if not plt.fignum_exists(interface.fig.number):
                print("\nVisualization window closed. Stopping training.")
                break

            # Current state
            current_pos = position.unsqueeze(0).to(device)

            # Decide action
            action, exploring = model.exploration_step(current_pos)

            # Periodic perturbation
            if step % 50 == 0 and step > 0:
                perturbation = torch.randn_like(action) * 0.3
                action = action + perturbation.to(device)

            # Take action
            next_position, env_reward, done, info = env.step(action.cpu())
            next_pos = next_position.unsqueeze(0).to(device)

            # Compute reward
            optimizer.zero_grad()
            total_reward, metrics = model.compute_total_reward(
                current_pos, next_pos, next_pos
            )

            # Backward
            loss = -total_reward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Update curiosity module
            with torch.no_grad():
                current_embedding, _ = model.get_state_embedding(current_pos)
                next_embedding, _ = model.get_state_embedding(next_pos)
            model.curiosity_module.update(
                current_embedding.detach(),
                action.detach(),
                next_embedding.detach()
            )

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

            # Update position
            position = next_position.detach()

            # Update live interface
            buffer_stats = model.experience_buffer.get_coverage_stats()
            health = model.homeostasis.get_health_report()

            interface.update(step + 1, {
                'position': position.numpy(),
                'novelty': metrics['novelty'],
                'curiosity': metrics['curiosity'],
                'total_reward': metrics['total_reward'],
                'exploration_rate': model.exploration_rate,
                'exploring': exploring,
                'diversity': buffer_stats['diversity'],
                'param_health': f"{health.get('healthy_params', '?')}/{health.get('total_params', '?')}"
            })

            # Console logging (less frequent)
            if (step + 1) % 1000 == 0:
                print(f"Step {step + 1}/{num_steps} - "
                      f"Novelty: {metrics['novelty']:.3f}, "
                      f"Curiosity: {metrics['curiosity']:.4f}, "
                      f"Coverage: {buffer_stats['coverage']*100:.1f}%")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    finally:
        # Save results
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)

        archive_dir = 'exploration_runs'
        os.makedirs(archive_dir, exist_ok=True)

        # Save visualizations
        if len(metrics_history['positions']) > 0:
            plot_training_metrics(
                metrics_history,
                save_path=os.path.join(archive_dir, f'live_metrics_{timestamp}.png')
            )
            plot_state_space_exploration(
                model.experience_buffer,
                save_path=os.path.join(archive_dir, f'live_state_space_{timestamp}.png')
            )
            print(f"Results saved to {archive_dir}/live_*_{timestamp}.png")

        # Close interface
        interface.close()

    return model, metrics_history


if __name__ == '__main__':
    model, history = run_live_exploration(num_steps=20000, update_freq=50)
