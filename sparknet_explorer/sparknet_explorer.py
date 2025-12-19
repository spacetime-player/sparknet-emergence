"""
SparkNet Explorer - Main Architecture

Neural network with intrinsic motivation through curiosity-driven learning.
Combines three reward signals:
1. Extrinsic: Task performance (traditional loss)
2. Intrinsic: Curiosity/novelty rewards (exploration bonus)
3. Homeostatic: Stability maintenance (viability constraints)

This creates emergent exploratory behavior - the network actively seeks
to resolve uncertainty while maintaining parameter health.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules.curiosity_module import CuriosityModule
from .modules.homeostasis import HomeostasisMonitor
from .core.experience_buffer import ExperienceBuffer


class SparkNetExplorer(nn.Module):
    """
    Neural network with intrinsic motivation through curiosity-driven learning.
    Combines extrinsic task rewards, intrinsic curiosity rewards, and homeostatic stability.
    """

    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        state_embedding_dim=64,
        curiosity_weight=0.1,
        homeostasis_weight=0.01,
        novelty_weight=0.5,
        prediction_error_weight=0.5,
        device=None
    ):
        """
        Initialize SparkNet Explorer.

        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer sizes (e.g., [256, 512, 256])
            output_dim: Output dimension
            state_embedding_dim: Size of state embedding for curiosity
            curiosity_weight: Weight for curiosity reward (β)
            homeostasis_weight: Weight for homeostatic penalty (γ)
            novelty_weight: Weight for novelty component of intrinsic reward
            prediction_error_weight: Weight for prediction error component
            device: Torch device (cuda/cpu)
        """
        super().__init__()

        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.state_embedding_dim = state_embedding_dim

        # Hyperparameters
        self.curiosity_weight = curiosity_weight
        self.homeostasis_weight = homeostasis_weight
        self.novelty_weight = novelty_weight
        self.prediction_error_weight = prediction_error_weight

        # Main network
        layers = []
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim

        # Store hidden layers separately for embedding extraction
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)

        # State embedding network (maps final hidden layer to embedding space)
        self.state_embedder = nn.Sequential(
            nn.Linear(hidden_dims[-1], 128),
            nn.ReLU(),
            nn.Linear(128, state_embedding_dim)
        )

        # Curiosity module
        self.curiosity_module = CuriosityModule(
            state_dim=state_embedding_dim,
            action_dim=output_dim,
            hidden_dim=256
        )

        # Experience buffer for novelty detection
        self.experience_buffer = ExperienceBuffer(
            max_size=10000,
            embedding_dim=state_embedding_dim
        )

        # Homeostasis monitor
        self.homeostasis = HomeostasisMonitor(
            self,
            adaptation_period=1000
        )

        # Exploration control
        self.exploration_rate = 0.1  # Start with 10% random exploration
        self.exploration_decay = 0.9995

        # Tracking metrics
        self.metrics = {
            'extrinsic_reward': [],
            'intrinsic_reward': [],
            'novelty_reward': [],
            'curiosity_reward': [],
            'homeostatic_penalty': [],
            'boredom_penalty': [],  # Track boredom penalty
            'total_reward': [],
            'exploration_rate': [],
            'stress_factor': [],  # Track homeostatic stress
            'adaptive_curiosity_weight': [],  # Track dynamic weight
        }

        self.to(self.device)

    def get_state_embedding(self, x):
        """
        Extract state embedding from hidden layer activations.

        Args:
            x: Input tensor

        Returns:
            Tuple of (state_embedding, final_hidden_activations)
        """
        # Forward through hidden layers
        hidden = self.hidden_layers(x)

        # Extract last hidden layer activation
        # (already have it from sequential forward)

        # Embed hidden state into embedding space
        state_embedding = self.state_embedder(hidden)

        return state_embedding, hidden

    def forward(self, x, return_embedding=False):
        """
        Forward pass through main network.

        Args:
            x: Input tensor
            return_embedding: Whether to return state embedding

        Returns:
            Output tensor, or (output, state_embedding, hidden) if return_embedding=True
        """
        hidden = self.hidden_layers(x)
        output = self.output_layer(hidden)

        if return_embedding:
            state_embedding = self.state_embedder(hidden)
            return output, state_embedding, hidden

        return output

    def compute_total_reward(self, x, y_true, next_x=None):
        """
        Compute total reward combining:
        1. Extrinsic (task performance)
        2. Intrinsic (curiosity + novelty)
        3. Homeostatic (stability penalty)

        Args:
            x: Current input
            y_true: Target output for task
            next_x: Next input state (optional, for curiosity computation)

        Returns:
            Tuple of (total_reward, metrics_dict)
        """
        # Forward pass with embeddings
        y_pred, state_embedding, hidden = self.forward(x, return_embedding=True)

        # 1. EXTRINSIC REWARD (negative loss)
        extrinsic_loss = F.mse_loss(y_pred, y_true)
        extrinsic_reward = -extrinsic_loss

        # 2. INTRINSIC REWARD (curiosity + novelty)

        # 2a. Novelty component
        novelty = self.experience_buffer.compute_novelty(state_embedding)
        novelty_tensor = torch.tensor(novelty, device=self.device, dtype=torch.float32)

        # Add state to experience buffer
        self.experience_buffer.add(state_embedding)

        # 2b. Curiosity component (prediction error)
        if next_x is not None:
            # Get next state embedding
            with torch.no_grad():
                _, next_state_embedding, _ = self.forward(next_x, return_embedding=True)

            # Compute curiosity reward
            curiosity_reward = self.curiosity_module.compute_curiosity_reward(
                state_embedding,
                y_pred,  # Use action/output as the "action"
                next_state_embedding
            ).mean()

            # Note: Curiosity module update should be done separately after backward()
            curiosity_losses = {'forward_loss': 0.0, 'inverse_loss': 0.0}
        else:
            # No next state available - use only novelty
            curiosity_reward = torch.tensor(0.0, device=self.device)
            curiosity_losses = {'forward_loss': 0.0, 'inverse_loss': 0.0}

        # Combined intrinsic reward
        intrinsic_reward = (
            self.novelty_weight * novelty_tensor +
            self.prediction_error_weight * curiosity_reward
        )

        # 3. HOMEOSTATIC PENALTY
        homeostatic_penalty, num_violations = self.homeostasis.compute_penalty()

        # Update homeostasis ranges if in adaptation period
        self.homeostasis.update_desirable_ranges()

        # ADAPTIVE CORRELATION: Stress drives exploration, exploration relieves stress
        # Normalize homeostatic penalty to [0, 1] range for scaling
        max_expected_penalty = 1.0  # Typical max observed penalty
        stress_factor = torch.clamp(homeostatic_penalty / max_expected_penalty, 0.0, 2.0)

        # When stressed → increase exploration drive to find relief
        # When healthy → reduce exploration urgency
        adaptive_curiosity_weight = self.curiosity_weight * (1.0 + stress_factor * 0.5)
        adaptive_novelty_weight = self.novelty_weight * (1.0 + stress_factor * 0.5)

        # Recalculate intrinsic reward with adaptive weights
        intrinsic_reward = (
            adaptive_novelty_weight * novelty_tensor +
            self.prediction_error_weight * curiosity_reward
        )

        # SOLUTION 3: Low-curiosity penalty (boredom penalty)
        # When curiosity (prediction error) is too low → agent is bored → penalize
        # This prevents settling into predictable states (like corners)
        boredom_threshold = 0.001  # Curiosity below this = bored
        curiosity_value = curiosity_reward.item() if isinstance(curiosity_reward, torch.Tensor) else curiosity_reward

        if curiosity_value < boredom_threshold:
            # Strong penalty for being too comfortable/predictable
            boredom_penalty = (boredom_threshold - curiosity_value) * 10.0
            boredom_penalty_tensor = torch.tensor(boredom_penalty, device=self.device, dtype=torch.float32)
        else:
            boredom_penalty_tensor = torch.tensor(0.0, device=self.device, dtype=torch.float32)

        # TOTAL REWARD
        total_reward = (
            extrinsic_reward +
            adaptive_curiosity_weight * intrinsic_reward -
            self.homeostasis_weight * homeostatic_penalty -
            boredom_penalty_tensor  # Subtract boredom penalty
        )

        # Track metrics
        self.metrics['extrinsic_reward'].append(extrinsic_reward.item())
        self.metrics['intrinsic_reward'].append(intrinsic_reward.item())
        self.metrics['novelty_reward'].append(novelty)
        self.metrics['curiosity_reward'].append(curiosity_reward.item() if isinstance(curiosity_reward, torch.Tensor) else curiosity_reward)
        self.metrics['homeostatic_penalty'].append(homeostatic_penalty.item())
        self.metrics['boredom_penalty'].append(boredom_penalty_tensor.item())
        self.metrics['total_reward'].append(total_reward.item())
        self.metrics['exploration_rate'].append(self.exploration_rate)
        self.metrics['stress_factor'].append(stress_factor.item())
        self.metrics['adaptive_curiosity_weight'].append(adaptive_curiosity_weight.item())

        metrics_dict = {
            'extrinsic': extrinsic_reward.item(),
            'intrinsic': intrinsic_reward.item(),
            'novelty': novelty,
            'curiosity': curiosity_reward.item() if isinstance(curiosity_reward, torch.Tensor) else curiosity_reward,
            'homeostatic_penalty': homeostatic_penalty.item(),
            'boredom_penalty': boredom_penalty_tensor.item(),
            'num_violations': num_violations,
            'total_reward': total_reward.item(),
            'exploration_rate': self.exploration_rate,
            **curiosity_losses
        }

        return total_reward, metrics_dict

    def exploration_step(self, x):
        """
        Decide whether to explore (random action) or exploit (network output).
        Gradually decrease exploration rate over time.

        Args:
            x: Input tensor

        Returns:
            Tuple of (action, is_exploring)
        """
        if torch.rand(1).item() < self.exploration_rate:
            # Explore: random action
            action = torch.randn(x.shape[0], self.output_dim, device=self.device)
            exploring = True
        else:
            # Exploit: use network
            with torch.no_grad():
                action = self.forward(x)
            exploring = False

        # Decay exploration rate
        self.exploration_rate *= self.exploration_decay

        return action, exploring

    def get_metrics_summary(self, window=100):
        """
        Get summary statistics of recent metrics.

        Args:
            window: Number of recent steps to analyze

        Returns:
            Dictionary with summary statistics
        """
        summary = {}

        for key in ['extrinsic_reward', 'intrinsic_reward', 'homeostatic_penalty', 'total_reward']:
            if self.metrics[key]:
                recent = self.metrics[key][-window:]
                summary[key] = {
                    'mean': sum(recent) / len(recent),
                    'std': torch.tensor(recent).std().item() if len(recent) > 1 else 0.0,
                    'latest': recent[-1]
                }

        # Add curiosity stats
        curiosity_stats = self.curiosity_module.get_prediction_stats()
        summary['curiosity'] = curiosity_stats

        # Add experience buffer stats
        buffer_stats = self.experience_buffer.get_coverage_stats()
        summary['exploration'] = buffer_stats

        # Add homeostasis health
        health = self.homeostasis.get_health_report()
        summary['homeostasis'] = health

        return summary

    def reset_metrics(self):
        """Clear all tracked metrics."""
        for key in self.metrics:
            if isinstance(self.metrics[key], list):
                self.metrics[key] = []

        self.curiosity_module.reset_stats()

    def save(self, path):
        """
        Save model state.

        Args:
            path: File path to save to
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'curiosity_state_dict': self.curiosity_module.state_dict(),
            'metrics': self.metrics,
            'exploration_rate': self.exploration_rate,
            'homeostasis_ranges': self.homeostasis.desirable_ranges,
        }, path)
        print(f"Model saved to {path}")

    def load(self, path):
        """
        Load model state.

        Args:
            path: File path to load from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.curiosity_module.load_state_dict(checkpoint['curiosity_state_dict'])
        self.metrics = checkpoint['metrics']
        self.exploration_rate = checkpoint['exploration_rate']
        self.homeostasis.desirable_ranges = checkpoint['homeostasis_ranges']
        print(f"Model loaded from {path}")
