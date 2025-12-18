"""
Simple Exploration Task - SparkNet Explorer Demo

Goal: Network learns to explore a 2D continuous space and discover 'interesting' regions.

The network receives its (x, y) position and outputs a movement action.
Success is measured by:
1. How much of the space it explores (novelty)
2. How well it predicts where it will end up (curiosity)
3. How stable its parameters remain (homeostasis)

This demonstrates emergent exploratory behavior - the network learns to seek
novel states while maintaining internal stability.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sparknet_explorer.sparknet_explorer import SparkNetExplorer
from sparknet_explorer.visualize import plot_training_metrics, plot_state_space_exploration

plt.style.use('dark_background')


class ContinuousExplorationEnvironment:
    """
    Simple 2D continuous exploration environment.
    Agent starts at origin and can move in any direction.
    """

    def __init__(self, bounds=(-1, 1), dt=0.1):
        """
        Initialize environment.

        Args:
            bounds: Tuple of (min, max) for both x and y
            dt: Time step for movement
        """
        self.bounds = bounds
        self.dt = dt
        self.position = torch.tensor([0.0, 0.0])
        self.trajectory = [self.position.clone().numpy()]

    def reset(self):
        """Reset to origin."""
        self.position = torch.tensor([0.0, 0.0])
        self.trajectory = [self.position.clone().numpy()]
        return self.position.clone()

    def step(self, action):
        """
        Take action and update position.

        Args:
            action: 2D movement vector

        Returns:
            Tuple of (next_position, reward, done, info)
        """
        # Move based on action
        self.position = self.position + action.squeeze() * self.dt

        # Boundary penalty (quadratic gradient pushing away from edges)
        boundary_penalty = 0.0
        margin = 0.2  # Start penalty when within 0.2 of boundary

        for i in range(2):  # x and y
            # Distance from edges
            dist_from_max = self.bounds[1] - self.position[i]
            dist_from_min = self.position[i] - self.bounds[0]

            # Quadratic penalty if too close to boundary
            if dist_from_max < margin:
                boundary_penalty += (margin - dist_from_max) ** 2
            if dist_from_min < margin:
                boundary_penalty += (margin - dist_from_min) ** 2

        # Clip to bounds (hard constraint)
        self.position = torch.clamp(self.position,
                                     self.bounds[0], self.bounds[1])

        # Track trajectory
        self.trajectory.append(self.position.clone().numpy())

        # Reward: distance from origin MINUS boundary penalty
        reward = torch.norm(self.position).item() - boundary_penalty * 2.0

        done = False
        info = {
            'position': self.position.clone(),
            'boundary_penalty': boundary_penalty
        }

        return self.position.clone(), reward, done, info

    def get_trajectory(self):
        """Return trajectory as numpy array."""
        return np.array(self.trajectory)


def exploration_task(num_steps=5000, visualize_interval=500):
    """
    Run exploration task.

    Args:
        num_steps: Number of exploration steps
        visualize_interval: Show progress every N steps

    Returns:
        Trained model
    """
    print("\n" + "="*80)
    print("SPARKNET EXPLORER - CONTINUOUS EXPLORATION TASK")
    print("="*80)
    print("Goal: Explore 2D space driven by curiosity and novelty")
    print("="*80 + "\n")

    # Initialize environment
    env = ContinuousExplorationEnvironment(bounds=(-1, 1), dt=0.1)

    # Auto-detect device (will use CUDA if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Initialize network
    model = SparkNetExplorer(
        input_dim=2,  # 2D position
        hidden_dims=[128, 256, 128],
        output_dim=2,  # 2D action (movement)
        state_embedding_dim=32,
        curiosity_weight=0.8,  # Very high curiosity for exploration
        homeostasis_weight=0.005,  # Lower homeostasis for more freedom
        novelty_weight=0.7,  # Prioritize novelty
        prediction_error_weight=0.3,
        device=device
    )

    # Higher exploration rate, slower decay
    model.exploration_rate = 0.3  # Start with 30% random
    model.exploration_decay = 0.9998  # Slower decay

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Start position
    position = env.reset()

    # Tracking
    metrics_history = {
        'extrinsic': [],
        'intrinsic': [],
        'novelty': [],
        'curiosity': [],
        'homeostatic': [],
        'total_reward': [],
        'loss': [],
        'positions': []
    }

    print("Starting exploration...")
    print(f"Initial position: {position.numpy()}\n")

    for step in range(num_steps):
        # Current state - move to device
        current_pos = position.unsqueeze(0).to(device)

        # Decide action (explore or exploit)
        action, exploring = model.exploration_step(current_pos)

        # Add periodic perturbation to disturb equilibrium (every 50 steps)
        if step % 50 == 0 and step > 0:
            perturbation = torch.randn_like(action) * 0.3
            action = action + perturbation.to(device)

        # Take action in environment
        next_position, env_reward, done, info = env.step(action.cpu())

        # Next state - move to device
        next_pos = next_position.unsqueeze(0).to(device)

        # Compute total reward
        # Use next position as "target" - network learns to predict its movement
        optimizer.zero_grad()
        total_reward, metrics = model.compute_total_reward(
            current_pos,
            next_pos,  # Target is next position
            next_pos   # For curiosity computation
        )

        # Loss is negative total reward
        loss = -total_reward
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Update curiosity module (detach to avoid double backward)
        with torch.no_grad():
            current_embedding, _ = model.get_state_embedding(current_pos)
            next_embedding, _ = model.get_state_embedding(next_pos)
        model.curiosity_module.update(current_embedding.detach(), action.detach(), next_embedding.detach())

        # Track metrics
        metrics_history['extrinsic'].append(metrics['extrinsic'])
        metrics_history['intrinsic'].append(metrics['intrinsic'])
        metrics_history['novelty'].append(metrics['novelty'])
        metrics_history['curiosity'].append(metrics['curiosity'])
        metrics_history['homeostatic'].append(metrics['homeostatic_penalty'])
        metrics_history['total_reward'].append(metrics['total_reward'])
        metrics_history['loss'].append(loss.item())
        metrics_history['positions'].append(position.numpy().copy())

        # Update position for next step
        position = next_position.detach()

        # Logging
        if (step + 1) % visualize_interval == 0:
            print(f"\nStep {step + 1}/{num_steps}")
            print(f"  Position: [{position[0]:.4f}, {position[1]:.4f}]")
            print(f"  Exploring: {exploring}")
            print(f"  Novelty: {metrics['novelty']:.4f}")
            print(f"  Curiosity: {metrics['curiosity']:.4f}")
            print(f"  Total Reward: {metrics['total_reward']:.4f}")
            print(f"  Exploration Rate: {model.exploration_rate:.4f}")

            # Show coverage
            buffer_stats = model.experience_buffer.get_coverage_stats()
            print(f"  State Space Coverage: {buffer_stats['coverage']*100:.1f}% "
                  f"({buffer_stats['size']} states)")
            print(f"  Diversity: {buffer_stats['diversity']:.4f}")

            # Homeostasis check
            health = model.homeostasis.get_health_report()
            if health['status'] == 'monitoring':
                print(f"  Parameter Health: {health['healthy_params']}/{health['total_params']} healthy")

    print("\n" + "="*80)
    print("EXPLORATION COMPLETE")
    print("="*80 + "\n")

    # Final summary
    print("Final Statistics:")
    print(f"  Total steps: {num_steps}")
    print(f"  Final position: [{position[0]:.4f}, {position[1]:.4f}]")
    print(f"  States explored: {len(model.experience_buffer)}")

    summary = model.get_metrics_summary(window=100)
    print(f"\nLast 100 steps:")
    print(f"  Avg Novelty: {np.mean(metrics_history['novelty'][-100:]):.4f}")
    print(f"  Avg Curiosity: {np.mean(metrics_history['curiosity'][-100:]):.4f}")
    print(f"  Avg Total Reward: {summary['total_reward']['mean']:.4f}")

    # Visualize results
    print("\nGenerating visualizations...")

    # Create timestamped folder for this run
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    archive_dir = 'exploration_runs'
    os.makedirs(archive_dir, exist_ok=True)

    print(f"Saving timestamped versions to: {archive_dir}/")

    # Plot 1: Training metrics
    plot_training_metrics(metrics_history, save_path='exploration_metrics.png')
    plot_training_metrics(metrics_history, save_path=os.path.join(archive_dir, f'exploration_metrics_{timestamp}.png'))

    # Plot 2: State space exploration
    plot_state_space_exploration(model.experience_buffer,
                                  save_path='exploration_state_space.png')
    plot_state_space_exploration(model.experience_buffer,
                                  save_path=os.path.join(archive_dir, f'exploration_state_space_{timestamp}.png'))

    # Plot 3: Trajectory in 2D space
    plot_trajectory(np.array(metrics_history['positions']),
                    save_path='exploration_trajectory.png')
    plot_trajectory(np.array(metrics_history['positions']),
                    save_path=os.path.join(archive_dir, f'exploration_trajectory_{timestamp}.png'))

    print(f"\nVisualization complete!")
    print(f"Latest versions: exploration_*.png")
    print(f"Archived versions: {archive_dir}/exploration_*_{timestamp}.png")

    return model, metrics_history


def plot_trajectory(positions, save_path='exploration_trajectory.png'):
    """
    Plot the agent's trajectory through 2D space.

    Args:
        positions: Numpy array of shape (num_steps, 2)
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor('#2e2e2e')

    # Plot 1: Full trajectory colored by time
    ax = axes[0]
    ax.set_facecolor('#1e1e1e')

    scatter = ax.scatter(positions[:, 0], positions[:, 1],
                         c=range(len(positions)), cmap='viridis',
                         alpha=0.6, s=10, edgecolors='none')

    # Mark start and end
    ax.plot(positions[0, 0], positions[0, 1], 'go', markersize=15,
            label='Start', markeredgecolor='white', markeredgewidth=2)
    ax.plot(positions[-1, 0], positions[-1, 1], 'ro', markersize=15,
            label='End', markeredgecolor='white', markeredgewidth=2)

    ax.set_xlabel('X Position', color='white')
    ax.set_ylabel('Y Position', color='white')
    ax.set_title('Exploration Trajectory (colored by time)', color='white',
                 fontweight='bold', fontsize=14)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, alpha=0.2, color='white')
    ax.tick_params(colors='white')
    ax.legend(facecolor='#3e3e3e', edgecolor='white')

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Time Step', color='white')
    cbar.ax.tick_params(colors='white')

    # Plot 2: Density heatmap
    ax = axes[1]
    ax.set_facecolor('#1e1e1e')

    heatmap, xedges, yedges = np.histogram2d(positions[:, 0], positions[:, 1],
                                              bins=20, range=[[-1, 1], [-1, 1]])

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax.imshow(heatmap.T, extent=extent, origin='lower',
                   cmap='hot', interpolation='bilinear', aspect='auto')

    ax.set_xlabel('X Position', color='white')
    ax.set_ylabel('Y Position', color='white')
    ax.set_title('Exploration Density Heatmap', color='white',
                 fontweight='bold', fontsize=14)
    ax.tick_params(colors='white')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Visit Frequency', color='white')
    cbar.ax.tick_params(colors='white')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, facecolor='#2e2e2e')
    print(f"Trajectory plot saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    # Run exploration task
    model, history = exploration_task(num_steps=5000, visualize_interval=500)

    # Save model
    model.save('exploration_model.pt')
    print("\nModel saved to exploration_model.pt")

    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)
    print("\nKey Observations:")
    print("1. Network explores space driven by novelty (seeks unexplored regions)")
    print("2. Curiosity helps predict movement outcomes (reduces prediction error)")
    print("3. Homeostasis maintains parameter stability throughout learning")
    print("4. Emergent behavior: strategic exploration without explicit programming")
    print("="*80 + "\n")
