# SparkNet Explorer

**Curiosity-Driven Neural Network with Intrinsic Motivation**

SparkNet Explorer extends the original SparkNet with curiosity-driven learning, creating artificial "why?" - a system that doesn't just respond but actively seeks to resolve uncertainty.

## Philosophy

The gap between known and unknown becomes the driving force. The network exhibits emergent exploratory behavior through incompleteness - it learns to seek what it doesn't understand, then resolves that uncertainty.

## Core Principles

### Triple Reward System

1. **EXTRINSIC**: Task-based performance (traditional loss)
2. **INTRINSIC**: Curiosity/novelty rewards (exploration bonus)
3. **HOMEOSTATIC**: Stability maintenance (viability constraints)

```
R_total = α * R_extrinsic + β * R_intrinsic - γ * P_homeostatic
```

### Curiosity as Prediction Error

High prediction error in novel states = high curiosity reward. The network learns to seek what it doesn't understand, then resolve that uncertainty.

### Weight Space Constraints

Define "desirable" and "undesirable" regions in parameter space. Like evolutionary fitness landscapes - some directions are viable, others lead to collapse.

### Emergence Through Exploration

By rewarding random exploration within constraints, complex behaviors emerge that weren't explicitly programmed.

## Quick Start

### Installation

```bash
pip install torch matplotlib numpy scikit-learn networkx
```

### Run Tests

```bash
python sparknet_explorer/tests/test_curiosity.py
```

### Run Example

```bash
python examples/simple_exploration_task.py
```

## Architecture

### Core Components

#### 1. ExperienceBuffer
Stores embeddings of visited states for novelty detection.

```python
from sparknet_explorer.core.experience_buffer import ExperienceBuffer

buffer = ExperienceBuffer(max_size=10000, embedding_dim=128)
buffer.add(state_embedding)
novelty = buffer.compute_novelty(new_state)
```

#### 2. CuriosityModule
Forward model predicting next state from current state and action.

```python
from sparknet_explorer.modules.curiosity_module import CuriosityModule

curiosity = CuriosityModule(state_dim=64, action_dim=10)
curiosity_reward = curiosity.compute_curiosity_reward(state, action, next_state)
```

#### 3. HomeostasisMonitor
Tracks parameter health and enforces stability.

```python
from sparknet_explorer.modules.homeostasis import HomeostasisMonitor

homeostasis = HomeostasisMonitor(model, adaptation_period=1000)
penalty, violations = homeostasis.compute_penalty()
```

#### 4. SparkNetExplorer
Main architecture combining all components.

```python
from sparknet_explorer import SparkNetExplorer

model = SparkNetExplorer(
    input_dim=10,
    hidden_dims=[256, 512, 256],
    output_dim=10,
    curiosity_weight=0.1,
    homeostasis_weight=0.01
)

total_reward, metrics = model.compute_total_reward(x, y, next_x)
```

## Usage Example

```python
import torch
from sparknet_explorer import SparkNetExplorer
from sparknet_explorer.train_explorer import train_sparknet_explorer

# Initialize model
model = SparkNetExplorer(
    input_dim=10,
    hidden_dims=[256, 512, 256],
    output_dim=10,
    state_embedding_dim=64,
    curiosity_weight=0.1,
    homeostasis_weight=0.01
)

# Define data generator
def generate_batch(batch_size):
    x = torch.randn(batch_size, 10)
    y = torch.randn(batch_size, 10)
    next_x = torch.randn(batch_size, 10)
    return x, y, next_x

# Train
trained_model, metrics = train_sparknet_explorer(
    model,
    generate_batch,
    num_epochs=1000,
    batch_size=32
)

# Visualize
from sparknet_explorer.visualize import create_comprehensive_report
create_comprehensive_report(trained_model, metrics)
```

## Key Hyperparameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `curiosity_weight` | 0.1 | 0.01-1.0 | Balance between task and exploration |
| `homeostasis_weight` | 0.01 | 0.001-0.1 | Strength of stability penalty |
| `novelty_weight` | 0.5 | 0.0-1.0 | Weight for novelty in intrinsic reward |
| `prediction_error_weight` | 0.5 | 0.0-1.0 | Weight for curiosity in intrinsic reward |
| `exploration_rate` | 0.1 | 0.05-0.5 | Initial random exploration probability |
| `experience_buffer_size` | 10000 | 1000-100000 | States to remember for novelty |
| `adaptation_period` | 1000 | 500-5000 | Steps before homeostasis activates |

## Expected Emergent Behaviors

### 1. Strategic Exploration
Network explores uncertain regions first, then focuses on exploitation.
- **Look for**: Novelty high early, decreasing as space is covered

### 2. Homeostatic Self-Regulation
Network maintains parameter health without explicit constraints.
- **Look for**: Few violations after adaptation, stable distributions

### 3. Curiosity-Task Balance
Network dynamically balances exploration and task performance.
- **Look for**: Intrinsic reward high when stuck, low when progressing

### 4. Efficient State Space Coverage
Network doesn't revisit same states unnecessarily.
- **Look for**: Uniform coverage in state space visualization

## Visualization

The suite includes comprehensive visualizations:

### Training Metrics
```python
from sparknet_explorer.visualize import plot_training_metrics
plot_training_metrics(metrics_history)
```

### State Space Exploration
```python
from sparknet_explorer.visualize import plot_state_space_exploration
plot_state_space_exploration(model.experience_buffer)
```

### Weight Heatmap
```python
from sparknet_explorer.visualize import visualize_weight_heatmap
visualize_weight_heatmap(model, layer_name='hidden_layers.0')
```

### Homeostasis Health
```python
from sparknet_explorer.visualize import plot_homeostasis_health
plot_homeostasis_health(model)
```

## Debugging Guide

### Issue: No Exploration
**Symptoms**: Novelty stays low, state clustering

**Fixes**:
- Increase `curiosity_weight`
- Increase `exploration_rate`
- Check experience buffer is storing distinct states

### Issue: Unstable Training
**Symptoms**: Loss explodes, NaN values

**Fixes**:
- Increase `homeostasis_weight`
- Reduce learning rate
- Add gradient clipping
- Check homeostasis ranges are reasonable

### Issue: Ignores Task
**Symptoms**: High novelty, poor task performance

**Fixes**:
- Decrease `curiosity_weight`
- Ensure extrinsic reward signal is clear
- Check if task is too difficult

## Mathematical Formulation

### Total Reward
```
R_total = α * R_extrinsic + β * R_intrinsic - γ * P_homeostatic

where:
  R_extrinsic = -L(prediction, target)
  R_intrinsic = w1*novelty + w2*prediction_error
  P_homeostatic = Σ max(0, |w| - threshold)²
```

### Novelty
```
novelty(s_t) = min_distance(s_t, experience_buffer)
```

### Prediction Error
```
prediction_error = ||f_forward(s_t, a_t) - s_{t+1}||²
```

### Homeostatic Penalty
```
P_homeostatic = Σ_layers Σ_weights max(0, |w| - threshold)²
```

## Project Structure

```
sparknet_explorer/
├── __init__.py
├── sparknet_explorer.py       # Main architecture
├── train_explorer.py           # Training loop
├── visualize.py                # Visualization suite
├── core/
│   ├── __init__.py
│   └── experience_buffer.py    # Novelty detection
├── modules/
│   ├── __init__.py
│   ├── curiosity_module.py     # Intrinsic motivation
│   └── homeostasis.py          # Parameter health
└── tests/
    ├── __init__.py
    └── test_curiosity.py       # Unit tests
```

## Examples

See `examples/simple_exploration_task.py` for a complete demonstration of:
- 2D continuous space exploration
- Emergent exploratory behavior
- Novelty-driven movement
- Homeostatic stability
- Comprehensive visualization

## Research Context

This implementation explores:
- **Intrinsic Motivation**: How curiosity drives learning
- **Homeostatic Regulation**: Self-organizing stability
- **Emergent Behavior**: Complex patterns from simple rules
- **Consciousness Prerequisites**: What architectural features enable "why?"

## Related Work

- **Curiosity-Driven Exploration** (Pathak et al., 2017)
- **Random Network Distillation** (Burda et al., 2018)
- **Homeostatic Plasticity** in neuroscience
- **Free Energy Principle** (Karl Friston)

## Citation

If you use SparkNet Explorer in your research:

```
@software{sparknet_explorer_2025,
  title={SparkNet Explorer: Curiosity-Driven Neural Networks},
  author={SparkNet Project},
  year={2025},
  url={https://github.com/spacetime-player/sparknet-emergence}
}
```

## License

Same as parent SparkNet project.

## Contributing

Contributions welcome! Areas of interest:
- Alternative curiosity metrics
- Different homeostatic mechanisms
- Novel exploration strategies
- Benchmark tasks

## Acknowledgments

Built on the SparkNet architecture, extending it with curiosity-driven learning principles from reinforcement learning and neuroscience.
