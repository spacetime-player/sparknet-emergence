# SparkNet Explorer - Quick Start Guide

## 🚀 5-Minute Setup

### 1. Install Dependencies
```bash
pip install torch matplotlib numpy scikit-learn networkx
```

### 2. Run Tests (Verify Installation)
```bash
cd "c:\My_test\AI-model--1"
python sparknet_explorer/tests/test_curiosity.py
```

Expected output: `ALL TESTS PASSED ✓`

### 3. Run Demo
```bash
python examples/simple_exploration_task.py
```

This runs a 2D exploration task demonstrating curiosity-driven behavior.

---

## 💡 Basic Usage

### Minimal Example

```python
import torch
from sparknet_explorer import SparkNetExplorer

# 1. Create model
model = SparkNetExplorer(
    input_dim=10,
    hidden_dims=[128, 256, 128],
    output_dim=10,
    curiosity_weight=0.1,      # Exploration strength
    homeostasis_weight=0.01    # Stability strength
)

# 2. Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(1000):
    # Get data: (current_state, target, next_state)
    x = torch.randn(32, 10)
    y = torch.randn(32, 10)
    next_x = torch.randn(32, 10)

    # Compute total reward
    optimizer.zero_grad()
    total_reward, metrics = model.compute_total_reward(x, y, next_x)

    # Maximize reward (minimize negative reward)
    loss = -total_reward
    loss.backward()
    optimizer.step()

    # Monitor progress
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Total Reward = {metrics['total_reward']:.4f}")
        print(f"  Novelty: {metrics['novelty']:.4f}")
        print(f"  Curiosity: {metrics['curiosity']:.4f}")
```

### With Visualization

```python
from sparknet_explorer.train_explorer import train_sparknet_explorer
from sparknet_explorer.visualize import create_comprehensive_report

# Define data generator
def generate_batch(batch_size):
    x = torch.randn(batch_size, 10)
    y = torch.randn(batch_size, 10)
    next_x = torch.randn(batch_size, 10)
    return x, y, next_x

# Train
model, metrics = train_sparknet_explorer(
    model,
    generate_batch,
    num_epochs=1000,
    batch_size=32
)

# Visualize
create_comprehensive_report(model, metrics, output_dir='./results')
```

---

## 🎛️ Key Parameters

### Essential
- `input_dim`: Size of input features
- `output_dim`: Size of output/action space
- `hidden_dims`: List of hidden layer sizes, e.g., `[256, 512, 256]`

### Tuning Exploration
- `curiosity_weight` (0.01-1.0): Higher = more exploration
  - **Low task**: 0.05-0.1
  - **High exploration**: 0.3-0.5

- `novelty_weight` (0.0-1.0): Weight for novelty vs prediction error
  - Default: 0.5 (balanced)

### Tuning Stability
- `homeostasis_weight` (0.001-0.1): Higher = stronger stability
  - **Stable task**: 0.005
  - **Unstable training**: 0.05

---

## 📊 Understanding Metrics

### During Training
```python
metrics = {
    'extrinsic': -0.85,      # Task performance (higher = better)
    'intrinsic': 0.42,       # Curiosity + Novelty (higher = more exploration)
    'novelty': 0.65,         # How novel current state is
    'curiosity': 0.28,       # Prediction error (surprise)
    'homeostatic_penalty': 0.01,  # Parameter drift penalty
    'total_reward': -0.52    # Combined reward
}
```

### Good Signs
✓ Novelty decreases over time (space gets explored)
✓ Few homeostatic violations after adaptation
✓ Total reward increases
✓ Balance between extrinsic and intrinsic

### Warning Signs
⚠ Novelty stays high (not exploring efficiently)
⚠ Many homeostatic violations (unstable)
⚠ Total reward diverges (increase homeostasis_weight)

---

## 🔍 Debugging

### Problem: Network won't explore
**Symptoms**: Novelty always low, clustering in state space

**Solution**:
```python
model = SparkNetExplorer(
    ...,
    curiosity_weight=0.3,     # Increase (was 0.1)
    novelty_weight=0.7        # Favor novelty
)
model.exploration_rate = 0.2  # More random exploration
```

### Problem: Training unstable
**Symptoms**: Loss spikes, NaN values, parameter violations

**Solution**:
```python
model = SparkNetExplorer(
    ...,
    homeostasis_weight=0.05   # Increase (was 0.01)
)

# Also add gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Problem: Ignores the task
**Symptoms**: High novelty, poor task performance

**Solution**:
```python
model = SparkNetExplorer(
    ...,
    curiosity_weight=0.05,    # Decrease (was 0.1)
    novelty_weight=0.3        # Focus less on novelty
)
```

---

## 📈 Typical Training Progression

### Phase 1: Adaptation (Steps 0-1000)
- Homeostasis learning desirable parameter ranges
- No penalties yet
- High exploration

### Phase 2: Exploration (Steps 1000-3000)
- Novelty high, actively seeking new states
- Curiosity learning to predict
- Balance forming

### Phase 3: Exploitation (Steps 3000+)
- Novelty decreases (space explored)
- Focus shifts to task performance
- Stable parameters

---

## 🎨 Visualization Quick Reference

### Plot Training Metrics
```python
from sparknet_explorer.visualize import plot_training_metrics
plot_training_metrics(metrics_history)
```

### Plot State Space Coverage
```python
from sparknet_explorer.visualize import plot_state_space_exploration
plot_state_space_exploration(model.experience_buffer)
```

### Plot Homeostasis Health
```python
from sparknet_explorer.visualize import plot_homeostasis_health
plot_homeostasis_health(model)
```

### Generate Full Report
```python
from sparknet_explorer.visualize import create_comprehensive_report
create_comprehensive_report(model, metrics, output_dir='./results')
```

---

## 💾 Save/Load

### Save
```python
model.save('my_model.pt')
```

### Load
```python
model = SparkNetExplorer(input_dim=10, ...)
model.load('my_model.pt')
```

---

## 🎯 Quick Reference Table

| What | Code |
|------|------|
| **Create model** | `model = SparkNetExplorer(input_dim, hidden_dims, output_dim)` |
| **Compute reward** | `reward, metrics = model.compute_total_reward(x, y, next_x)` |
| **Training step** | `loss = -reward; loss.backward(); optimizer.step()` |
| **Explore/Exploit** | `action, exploring = model.exploration_step(x)` |
| **Get summary** | `summary = model.get_metrics_summary(window=100)` |
| **Save model** | `model.save('model.pt')` |
| **Load model** | `model.load('model.pt')` |

---

## 📚 Next Steps

1. **Run the demo**: `python examples/simple_exploration_task.py`
2. **Read the README**: `sparknet_explorer/README.md`
3. **Review the code**: Start with `sparknet_explorer/sparknet_explorer.py`
4. **Adapt to your task**: Modify data generator
5. **Experiment**: Tune hyperparameters

---

## 🆘 Help

### File Structure
```
sparknet_explorer/
├── sparknet_explorer.py     # Main model
├── train_explorer.py         # Training loop
├── visualize.py              # Plotting
├── core/experience_buffer.py # Novelty detection
├── modules/curiosity_module.py # Prediction error
└── modules/homeostasis.py    # Stability
```

### Documentation
- **Full docs**: `sparknet_explorer/README.md`
- **Summary**: `SPARKNET_EXPLORER_SUMMARY.md`
- **This guide**: `QUICK_START.md`

### Examples
- **Simple task**: `examples/simple_exploration_task.py`
- **Unit tests**: `sparknet_explorer/tests/test_curiosity.py`

---

## ⚡ Pro Tips

1. **Start simple**: Use default hyperparameters first
2. **Monitor metrics**: Watch novelty and violations
3. **Adapt period**: Let homeostasis adapt ~1000 steps
4. **Batch size**: 32-64 works well for most tasks
5. **Learning rate**: 1e-3 is a good starting point
6. **Visualize often**: Catch issues early

---

**Ready to explore? Start with the demo!**

```bash
python examples/simple_exploration_task.py
```
