"""
Weight Usage Explorer - Force Network to Use Unexplored Weights

Tracks how much each weight contributes to outputs and penalizes
overuse of popular weights while rewarding exploration of unused ones.

This prevents the network from settling into fixed patterns and forces
it to discover new computational paths.
"""

import torch
import torch.nn as nn
import numpy as np


class WeightUsageTracker:
    """
    Tracks weight usage across network layers and computes exploration penalty.

    For each weight matrix, tracks cumulative activation contribution.
    Penalizes weights used MORE than average (overused).
    Rewards weights used LESS than average (encourages exploration).
    """

    def __init__(self, decay=0.99, penalty_scale=0.01, exploration_bonus=0.005):
        """
        Args:
            decay: Exponential decay for usage history (0.99 = slow decay)
            penalty_scale: How much to penalize overused weights
            exploration_bonus: Bonus for using underused weights
        """
        self.decay = decay
        self.penalty_scale = penalty_scale
        self.exploration_bonus = exploration_bonus

        # Usage tracking: {layer_name: tensor of cumulative usage}
        self.usage = {}

        # Activation hooks storage
        self.hooks = []
        self.current_activations = {}

        # Metrics for visualization
        self.metrics_history = {
            'weight_coverage': [],      # % of weights actively used
            'overused_ratio': [],       # % of weights overused
            'underused_ratio': [],      # % of weights underused
            'usage_penalty': [],        # penalty value
            'exploration_bonus': [],    # bonus value
        }

    def register_hooks(self, model):
        """
        Register forward hooks on linear layers to capture activations.

        Args:
            model: nn.Module to track
        """
        self.clear_hooks()

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                hook = module.register_forward_hook(
                    self._make_hook(name)
                )
                self.hooks.append(hook)

    def _make_hook(self, layer_name):
        """Create a forward hook that captures activation magnitudes."""
        def hook(module, input, output):
            # Store absolute activation values
            # Shape: (batch, output_features)
            self.current_activations[layer_name] = output.detach().abs()
        return hook

    def clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def update_usage(self):
        """
        Update usage statistics from current activations.
        Call this after each forward pass.
        """
        for layer_name, activations in self.current_activations.items():
            # Average across batch dimension
            # Shape: (output_features,)
            avg_activation = activations.mean(dim=0)

            if layer_name not in self.usage:
                # Initialize usage tracker
                self.usage[layer_name] = torch.zeros_like(avg_activation)

            # Exponential moving average of usage
            self.usage[layer_name] = (
                self.decay * self.usage[layer_name] +
                (1 - self.decay) * avg_activation
            )

        # Clear current activations for next forward pass
        self.current_activations = {}

    def compute_penalty(self):
        """
        Compute penalty for weight usage imbalance.

        Returns:
            Tuple of (penalty_tensor, metrics_dict)
        """
        if not self.usage:
            return torch.tensor(0.0), {}

        total_penalty = 0.0
        total_bonus = 0.0
        total_weights = 0
        overused_count = 0
        underused_count = 0
        active_count = 0

        for layer_name, usage in self.usage.items():
            n = usage.numel()
            total_weights += n

            mean_usage = usage.mean()
            std_usage = usage.std() + 1e-8

            # Threshold for overused/underused (in standard deviations)
            overused_threshold = mean_usage + std_usage
            underused_threshold = mean_usage * 0.1  # Less than 10% of mean

            # Count categories
            overused_mask = usage > overused_threshold
            underused_mask = usage < underused_threshold
            active_mask = usage > underused_threshold

            overused_count += overused_mask.sum().item()
            underused_count += underused_mask.sum().item()
            active_count += active_mask.sum().item()

            # Penalty for overused weights (scaled by how much over)
            if overused_mask.any():
                overused_values = usage[overused_mask]
                excess = (overused_values - overused_threshold) / (std_usage + 1e-8)
                total_penalty += excess.sum().item() * self.penalty_scale

            # Bonus for using underused weights (encourage exploration)
            # This is a negative penalty (reward)
            if underused_mask.any():
                # Small bonus when underused weights get activated
                underused_activations = self.current_activations.get(layer_name)
                if underused_activations is not None:
                    # Check if any underused weights got activated this step
                    recent_activation = underused_activations.mean(dim=0)
                    newly_used = (recent_activation > 0.01) & underused_mask
                    total_bonus += newly_used.sum().item() * self.exploration_bonus

        # Compute coverage metric
        weight_coverage = active_count / max(total_weights, 1)
        overused_ratio = overused_count / max(total_weights, 1)
        underused_ratio = underused_count / max(total_weights, 1)

        # Net penalty (penalty minus bonus)
        net_penalty = total_penalty - total_bonus

        # Track metrics
        self.metrics_history['weight_coverage'].append(weight_coverage)
        self.metrics_history['overused_ratio'].append(overused_ratio)
        self.metrics_history['underused_ratio'].append(underused_ratio)
        self.metrics_history['usage_penalty'].append(total_penalty)
        self.metrics_history['exploration_bonus'].append(total_bonus)

        metrics = {
            'weight_coverage': weight_coverage,
            'overused_ratio': overused_ratio,
            'underused_ratio': underused_ratio,
            'usage_penalty': total_penalty,
            'exploration_bonus': total_bonus,
            'net_weight_penalty': net_penalty,
        }

        # Return as tensor for gradient flow (though typically used as penalty only)
        return torch.tensor(net_penalty, dtype=torch.float32), metrics

    def get_usage_heatmap(self, layer_name=None):
        """
        Get usage data for visualization.

        Args:
            layer_name: Specific layer or None for all

        Returns:
            Dict of {layer_name: usage_tensor} or single tensor
        """
        if layer_name:
            return self.usage.get(layer_name)
        return dict(self.usage)

    def get_summary(self, window=100):
        """Get summary statistics of recent metrics."""
        summary = {}
        for key, values in self.metrics_history.items():
            if values:
                recent = values[-window:]
                summary[key] = {
                    'mean': sum(recent) / len(recent),
                    'latest': recent[-1],
                    'trend': recent[-1] - recent[0] if len(recent) > 1 else 0
                }
        return summary

    def reset(self):
        """Reset all tracking data."""
        self.usage = {}
        self.current_activations = {}
        for key in self.metrics_history:
            self.metrics_history[key] = []
