"""
Training Loop for SparkNet Explorer

Implements training with triple reward system:
- Extrinsic: Task performance
- Intrinsic: Curiosity + Novelty
- Homeostatic: Parameter stability

The network learns to maximize total reward, which creates emergent
exploratory behavior.
"""

import torch
import torch.optim as optim
import numpy as np
from .sparknet_explorer import SparkNetExplorer


def train_sparknet_explorer(
    model,
    train_data_fn,
    num_epochs=1000,
    batch_size=32,
    learning_rate=1e-3,
    log_interval=100,
    device=None
):
    """
    Training loop for SparkNet Explorer with curiosity-driven learning.

    Args:
        model: SparkNetExplorer instance
        train_data_fn: Function that returns (x, y, next_x) batches
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate for optimizer
        log_interval: Print metrics every N steps
        device: Torch device

    Returns:
        Tuple of (trained_model, metrics_history)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    metrics_history = {
        'loss': [],
        'extrinsic': [],
        'intrinsic': [],
        'novelty': [],
        'curiosity': [],
        'homeostatic': [],
        'total_reward': [],
        'exploration_rate': []
    }

    step = 0

    print("\n" + "="*80)
    print("SPARKNET EXPLORER TRAINING")
    print("="*80)
    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print("="*80 + "\n")

    for epoch in range(num_epochs):
        # Get training batch
        x, y, next_x = train_data_fn(batch_size)

        # Move to device
        x = x.to(device)
        y = y.to(device)
        if next_x is not None:
            next_x = next_x.to(device)

        optimizer.zero_grad()

        # Compute total reward (includes curiosity and homeostasis)
        total_reward, metrics = model.compute_total_reward(x, y, next_x)

        # Loss is negative reward (we want to maximize reward)
        loss = -total_reward

        # Backward pass
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Track metrics
        metrics_history['loss'].append(loss.item())
        metrics_history['extrinsic'].append(metrics['extrinsic'])
        metrics_history['intrinsic'].append(metrics['intrinsic'])
        metrics_history['novelty'].append(metrics['novelty'])
        metrics_history['curiosity'].append(metrics['curiosity'])
        metrics_history['homeostatic'].append(metrics['homeostatic_penalty'])
        metrics_history['total_reward'].append(metrics['total_reward'])
        metrics_history['exploration_rate'].append(metrics['exploration_rate'])

        step += 1

        # Logging
        if step % log_interval == 0:
            print(f"Epoch {epoch:4d} | Step {step:5d}")
            print(f"  Loss: {loss.item():8.4f}")
            print(f"  Rewards:")
            print(f"    Extrinsic:   {metrics['extrinsic']:8.4f}")
            print(f"    Intrinsic:   {metrics['intrinsic']:8.4f}")
            print(f"      Novelty:   {metrics['novelty']:8.4f}")
            print(f"      Curiosity: {metrics['curiosity']:8.4f}")
            print(f"    Total:       {metrics['total_reward']:8.4f}")
            print(f"  Homeostatic Penalty: {metrics['homeostatic_penalty']:8.4f}")
            print(f"  Violations: {metrics['num_violations']}")
            print(f"  Exploration Rate: {metrics['exploration_rate']:.4f}")

            # Homeostasis health check
            health = model.homeostasis.get_health_report()
            if health['status'] == 'adapting':
                print(f"  Homeostasis: Adapting ({health['progress']*100:.1f}% complete)")
            else:
                if health['violations']:
                    print(f"  Parameter Violations: {len(health['violations'])}/{health['total_params']}")
                    for v in health['violations'][:3]:  # Show first 3
                        print(f"    {v['param']}: drift={v['drift']:.6f} ({v['severity']})")

            # Experience buffer stats
            buffer_stats = model.experience_buffer.get_coverage_stats()
            print(f"  State Space Coverage: {buffer_stats['coverage']*100:.1f}% "
                  f"({buffer_stats['size']}/{model.experience_buffer.max_size})")
            print(f"  Diversity: {buffer_stats['diversity']:.4f}")

            print()

    print("\n" + "="*80)
    print("TRAINING COMPLETED")
    print("="*80)

    # Final summary
    summary = model.get_metrics_summary(window=100)
    print("\nFinal Metrics (last 100 steps):")
    print(f"  Extrinsic Reward: {summary['extrinsic_reward']['mean']:.4f} ± {summary['extrinsic_reward']['std']:.4f}")
    print(f"  Intrinsic Reward: {summary['intrinsic_reward']['mean']:.4f} ± {summary['intrinsic_reward']['std']:.4f}")
    print(f"  Total Reward: {summary['total_reward']['mean']:.4f} ± {summary['total_reward']['std']:.4f}")
    print(f"  Homeostatic Penalty: {summary['homeostatic_penalty']['mean']:.4f} ± {summary['homeostatic_penalty']['std']:.4f}")
    print(f"\nExploration Coverage: {summary['exploration']['coverage']*100:.1f}%")
    print(f"Exploration Diversity: {summary['exploration']['diversity']:.4f}")
    print(f"\nHomeostasis Health: {summary['homeostasis']['status']}")
    if summary['homeostasis']['status'] == 'monitoring':
        print(f"  Healthy Parameters: {summary['homeostasis']['healthy_params']}/{summary['homeostasis']['total_params']}")

    print("\n" + "="*80 + "\n")

    return model, metrics_history


def create_dummy_data_generator(input_dim, output_dim):
    """
    Create a simple data generator for testing.
    Replace this with your actual task data.

    Args:
        input_dim: Input dimension
        output_dim: Output dimension

    Returns:
        Function that generates (x, y, next_x) batches
    """
    def generate_batch(batch_size):
        x = torch.randn(batch_size, input_dim)
        y = torch.randn(batch_size, output_dim)  # Dummy target
        next_x = torch.randn(batch_size, input_dim)  # Next state
        return x, y, next_x

    return generate_batch


# Example usage
if __name__ == "__main__":
    # Initialize network
    model = SparkNetExplorer(
        input_dim=10,
        hidden_dims=[256, 512, 256],
        output_dim=10,
        state_embedding_dim=64,
        curiosity_weight=0.1,
        homeostasis_weight=0.01
    )

    # Create data generator
    data_fn = create_dummy_data_generator(input_dim=10, output_dim=10)

    # Train
    trained_model, metrics = train_sparknet_explorer(
        model,
        data_fn,
        num_epochs=1000,
        batch_size=32,
        learning_rate=1e-3,
        log_interval=100
    )

    # Save model
    trained_model.save('sparknet_explorer_model.pt')

    # Visualize results
    try:
        from .visualize import plot_training_metrics
        plot_training_metrics(metrics)
    except ImportError:
        print("Visualization module not found. Skipping plots.")
