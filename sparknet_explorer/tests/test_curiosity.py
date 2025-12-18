"""
Unit tests for curiosity-driven components.

Tests the core components:
- ExperienceBuffer (novelty detection)
- CuriosityModule (prediction error)
- HomeostasisMonitor (parameter health)
- SparkNetExplorer (full integration)
"""

import torch
import sys
import os

# Set UTF-8 encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from sparknet_explorer.modules.curiosity_module import CuriosityModule
from sparknet_explorer.core.experience_buffer import ExperienceBuffer
from sparknet_explorer.modules.homeostasis import HomeostasisMonitor
from sparknet_explorer.sparknet_explorer import SparkNetExplorer


def test_curiosity_module():
    """Test curiosity module forward and reward computation."""
    print("\n" + "="*60)
    print("Testing CuriosityModule...")
    print("="*60)

    module = CuriosityModule(state_dim=64, action_dim=10, hidden_dim=128)

    batch_size = 32
    state = torch.randn(batch_size, 64)
    action = torch.randn(batch_size, 10)
    next_state = torch.randn(batch_size, 64)

    # Test forward prediction
    predicted_next = module(state, action)
    assert predicted_next.shape == next_state.shape, \
        f"Shape mismatch: {predicted_next.shape} vs {next_state.shape}"
    print("✓ Forward prediction shape correct")

    # Test curiosity reward
    reward = module.compute_curiosity_reward(state, action, next_state)
    assert reward.shape == (batch_size,), f"Reward shape wrong: {reward.shape}"
    assert (reward >= 0).all(), "MSE-based curiosity should be non-negative"
    print(f"✓ Curiosity reward computed: mean={reward.mean():.4f}, std={reward.std():.4f}")

    # Test update
    losses = module.update(state, action, next_state)
    assert 'forward_loss' in losses, "Missing forward_loss"
    assert 'inverse_loss' in losses, "Missing inverse_loss"
    print(f"✓ Update successful: forward_loss={losses['forward_loss']:.4f}, "
          f"inverse_loss={losses['inverse_loss']:.4f}")

    # Test inverse model
    predicted_action = module.predict_action(state, next_state)
    assert predicted_action.shape == action.shape
    print("✓ Inverse model prediction shape correct")

    print("\n✓ CuriosityModule tests PASSED\n")


def test_experience_buffer():
    """Test experience buffer novelty computation."""
    print("="*60)
    print("Testing ExperienceBuffer...")
    print("="*60)

    buffer = ExperienceBuffer(max_size=100, embedding_dim=64)

    # Test adding states
    for i in range(10):
        state = torch.randn(64)
        buffer.add(state)

    assert len(buffer) == 10, f"Buffer size wrong: {len(buffer)}"
    print(f"✓ Buffer stores states correctly: {len(buffer)} states")

    # Test novelty computation
    new_state = torch.randn(64)
    novelty = buffer.compute_novelty(new_state, metric='cosine')
    assert 0 <= novelty <= 1, f"Cosine novelty out of range: {novelty}"
    print(f"✓ Novelty (cosine): {novelty:.4f}")

    # Test that similar state has lower novelty
    import numpy as np
    similar_state = torch.tensor(list(buffer.buffer)[0], dtype=torch.float32) + torch.randn(64) * 0.01
    low_novelty = buffer.compute_novelty(similar_state, metric='cosine')
    assert low_novelty < novelty, "Similar state should have lower novelty"
    print(f"✓ Similar state has lower novelty: {low_novelty:.4f} < {novelty:.4f}")

    # Test L2 metric
    l2_novelty = buffer.compute_novelty(new_state, metric='l2')
    assert l2_novelty >= 0, "L2 novelty should be non-negative"
    print(f"✓ Novelty (L2): {l2_novelty:.4f}")

    # Test coverage stats
    stats = buffer.get_coverage_stats()
    assert 'size' in stats and 'diversity' in stats and 'coverage' in stats
    print(f"✓ Coverage stats: size={stats['size']}, diversity={stats['diversity']:.4f}, "
          f"coverage={stats['coverage']*100:.1f}%")

    # Test batch adding
    batch_states = torch.randn(5, 64)
    buffer.add(batch_states)
    assert len(buffer) == 15, "Batch adding failed"
    print("✓ Batch adding works")

    # Test buffer overflow (circular)
    buffer_small = ExperienceBuffer(max_size=5, embedding_dim=64)
    for i in range(10):
        buffer_small.add(torch.randn(64))
    assert len(buffer_small) == 5, "Circular buffer not working"
    print("✓ Circular buffer overflow handling works")

    print("\n✓ ExperienceBuffer tests PASSED\n")


def test_homeostasis_monitor():
    """Test homeostasis monitor parameter tracking."""
    print("="*60)
    print("Testing HomeostasisMonitor...")
    print("="*60)

    # Create a simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 5)
    )

    monitor = HomeostasisMonitor(model, adaptation_period=10, margin=2.0)

    # Test initial state
    assert monitor.step_count == 0
    print("✓ Initial state correct")

    # Test adaptation period
    for step in range(10):
        monitor.update_desirable_ranges()

    assert monitor.step_count == 10
    print(f"✓ Adaptation period tracking works: {monitor.step_count} steps")

    # Finalize ranges
    monitor.update_desirable_ranges()
    assert len(monitor.desirable_ranges) > 0
    print(f"✓ Desirable ranges established: {len(monitor.desirable_ranges)} parameters")

    # Test penalty computation
    penalty, violations = monitor.compute_penalty()
    assert isinstance(penalty, torch.Tensor)
    assert violations >= 0
    print(f"✓ Penalty computation works: penalty={penalty.item():.6f}, violations={violations}")

    # Test health report
    health = monitor.get_health_report()
    assert health['status'] == 'monitoring'
    assert 'violations' in health
    assert 'healthy_params' in health
    print(f"✓ Health report: {health['healthy_params']} healthy, "
          f"{len(health['violations'])} violations")

    # Test with parameter drift
    with torch.no_grad():
        for param in model.parameters():
            param.data += torch.randn_like(param) * 10  # Large drift

    penalty_after, violations_after = monitor.compute_penalty()
    assert penalty_after > penalty, "Penalty should increase after drift"
    print(f"✓ Penalty increases with drift: {penalty.item():.6f} → {penalty_after.item():.6f}")

    print("\n✓ HomeostasisMonitor tests PASSED\n")


def test_sparknet_explorer():
    """Test full SparkNetExplorer integration."""
    print("="*60)
    print("Testing SparkNetExplorer (Integration)...")
    print("="*60)

    # Use CPU for testing to avoid device mismatch issues
    device = torch.device('cpu')
    model = SparkNetExplorer(
        input_dim=10,
        hidden_dims=[64, 128, 64],
        output_dim=10,
        state_embedding_dim=32,
        curiosity_weight=0.1,
        homeostasis_weight=0.01,
        device=device
    )

    batch_size = 16
    x = torch.randn(batch_size, 10, device=device)
    y_true = torch.randn(batch_size, 10, device=device)
    next_x = torch.randn(batch_size, 10, device=device)

    # Test forward pass
    output = model(x)
    assert output.shape == (batch_size, 10)
    print(f"✓ Forward pass: input {x.shape} → output {output.shape}")

    # Test forward with embedding
    output, embedding, hidden = model(x, return_embedding=True)
    assert embedding.shape == (batch_size, 32)
    print(f"✓ State embedding: {embedding.shape}")

    # Test total reward computation
    total_reward, metrics = model.compute_total_reward(x, y_true, next_x)
    assert isinstance(total_reward, torch.Tensor)
    assert 'extrinsic' in metrics
    assert 'intrinsic' in metrics
    assert 'homeostatic_penalty' in metrics
    assert 'novelty' in metrics
    assert 'curiosity' in metrics
    print(f"✓ Total reward computation:")
    print(f"  Extrinsic: {metrics['extrinsic']:.4f}")
    print(f"  Intrinsic: {metrics['intrinsic']:.4f}")
    print(f"  Homeostatic: {metrics['homeostatic_penalty']:.4f}")
    print(f"  Total: {metrics['total_reward']:.4f}")

    # Test exploration step
    action, is_exploring = model.exploration_step(x)
    assert action.shape == (batch_size, 10)
    print(f"✓ Exploration step: exploring={is_exploring}")

    # Test metrics tracking
    assert len(model.metrics['total_reward']) > 0
    print(f"✓ Metrics tracked: {len(model.metrics['total_reward'])} steps")

    # Test metrics summary
    summary = model.get_metrics_summary(window=10)
    assert 'extrinsic_reward' in summary
    assert 'exploration' in summary
    assert 'homeostasis' in summary
    print("✓ Metrics summary generated")

    # Test save/load
    save_path = 'test_model.pt'
    model.save(save_path)
    assert os.path.exists(save_path)
    print(f"✓ Model saved to {save_path}")

    # Create new model and load
    model2 = SparkNetExplorer(
        input_dim=10,
        hidden_dims=[64, 128, 64],
        output_dim=10,
        state_embedding_dim=32
    )
    model2.load(save_path)
    print("✓ Model loaded successfully")

    # Clean up
    os.remove(save_path)

    print("\n✓ SparkNetExplorer tests PASSED\n")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*80)
    print("SPARKNET EXPLORER - UNIT TESTS")
    print("="*80 + "\n")

    try:
        test_curiosity_module()
        test_experience_buffer()
        test_homeostasis_monitor()
        test_sparknet_explorer()

        print("="*80)
        print("ALL TESTS PASSED ✓")
        print("="*80 + "\n")
        return True

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}\n")
        return False
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
