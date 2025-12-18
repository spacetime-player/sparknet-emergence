"""
Homeostasis Monitor - Parameter Health Tracking

Tracks network parameter statistics and computes penalties for drift outside
"desirable" ranges. This creates a fitness landscape in weight space where
some regions are viable and others lead to instability.

The system learns which parameter configurations are healthy during an
adaptation period, then maintains those ranges through soft penalties.
"""

import torch
import torch.nn as nn
from collections import defaultdict
import numpy as np


class HomeostasisMonitor:
    """
    Tracks network parameter health and computes penalties for drift.
    Defines 'desirable' ranges based on initialization and early training.
    """

    def __init__(self, model, adaptation_period=1000, margin=2.0):
        """
        Initialize homeostasis monitor.

        Args:
            model: The neural network to monitor
            adaptation_period: Number of steps to observe before establishing ranges
            margin: Standard deviations from mean for desirable range
        """
        self.model = model
        self.adaptation_period = adaptation_period
        self.margin = margin
        self.step_count = 0

        # Store initial statistics
        self.initial_stats = self._compute_param_stats()

        # Will store desirable ranges after adaptation
        self.desirable_ranges = {}

        # History during adaptation period
        self.stats_history = defaultdict(list)

        # Track violations over time
        self.violation_history = []

    def _compute_param_stats(self):
        """
        Compute mean, std, and max for each parameter.

        Returns:
            Dictionary mapping parameter names to statistics
        """
        stats = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                stats[name] = {
                    'mean': param.data.mean().item(),
                    'std': param.data.std().item(),
                    'max': param.data.abs().max().item(),
                    'min': param.data.min().item()
                }
        return stats

    def update_desirable_ranges(self):
        """
        Define desirable ranges as [mean - margin*std, mean + margin*std]
        based on statistics collected during adaptation period.

        This should be called at every step during adaptation.
        """
        if self.step_count < self.adaptation_period:
            # Still collecting statistics
            current_stats = self._compute_param_stats()
            for name in current_stats:
                self.stats_history[name].append(current_stats[name])
            self.step_count += 1

        elif self.step_count == self.adaptation_period:
            # Finalize desirable ranges
            print(f"\n{'='*60}")
            print(f"Homeostasis: Establishing desirable parameter ranges")
            print(f"{'='*60}")

            for name in self.stats_history:
                stats_list = self.stats_history[name]

                # Average statistics over adaptation period
                avg_mean = sum(s['mean'] for s in stats_list) / len(stats_list)
                avg_std = sum(s['std'] for s in stats_list) / len(stats_list)

                # Define desirable range
                self.desirable_ranges[name] = {
                    'min': avg_mean - self.margin * avg_std,
                    'max': avg_mean + self.margin * avg_std,
                    'reference_mean': avg_mean,
                    'reference_std': avg_std
                }

                print(f"  {name}:")
                print(f"    Mean: {avg_mean:.6f}, Std: {avg_std:.6f}")
                print(f"    Desirable range: [{self.desirable_ranges[name]['min']:.6f}, "
                      f"{self.desirable_ranges[name]['max']:.6f}]")

            print(f"{'='*60}\n")
            self.step_count += 1

    def compute_penalty(self):
        """
        Compute homeostatic penalty for parameters outside desirable ranges.
        Uses soft quadratic penalty that increases with distance from range.

        Returns:
            Tuple of (total_penalty tensor, number of violated parameters)
        """
        if self.step_count <= self.adaptation_period:
            # No penalty during adaptation period
            return torch.tensor(0.0, device=next(self.model.parameters()).device), 0

        total_penalty = torch.tensor(0.0, device=next(self.model.parameters()).device)
        violation_count = 0
        violation_details = []

        for name, param in self.model.named_parameters():
            if name not in self.desirable_ranges:
                continue

            ranges = self.desirable_ranges[name]
            min_val = ranges['min']
            max_val = ranges['max']

            # Compute violations (how much parameters exceed bounds)
            below_min = torch.relu(min_val - param)
            above_max = torch.relu(param - max_val)

            # Quadratic penalty (smooth and differentiable)
            penalty = (below_min ** 2).sum() + (above_max ** 2).sum()
            total_penalty += penalty

            if penalty.item() > 0:
                violation_count += 1
                violation_details.append({
                    'param': name,
                    'penalty': penalty.item(),
                    'mean': param.data.mean().item(),
                    'range': (min_val, max_val)
                })

        # Track violations
        self.violation_history.append({
            'step': self.step_count,
            'total_penalty': total_penalty.item(),
            'num_violations': violation_count,
            'details': violation_details
        })

        return total_penalty, violation_count

    def get_health_report(self):
        """
        Generate comprehensive report on parameter health.

        Returns:
            Dictionary with health status
        """
        if self.step_count <= self.adaptation_period:
            return {
                'status': 'adapting',
                'progress': self.step_count / self.adaptation_period,
                'steps_remaining': self.adaptation_period - self.step_count
            }

        report = {
            'status': 'monitoring',
            'violations': [],
            'healthy_params': 0,
            'total_params': 0
        }

        current_stats = self._compute_param_stats()

        for name, param in self.model.named_parameters():
            if name not in self.desirable_ranges:
                continue

            report['total_params'] += 1
            ranges = self.desirable_ranges[name]
            stats = current_stats[name]

            # Check if mean is within range
            current_mean = stats['mean']
            in_range = ranges['min'] <= current_mean <= ranges['max']

            if not in_range:
                drift = current_mean - ranges['reference_mean']
                drift_in_stds = drift / (ranges['reference_std'] + 1e-8)

                report['violations'].append({
                    'param': name,
                    'current_mean': current_mean,
                    'desirable_range': (ranges['min'], ranges['max']),
                    'drift': drift,
                    'drift_std': drift_in_stds,
                    'severity': 'high' if abs(drift_in_stds) > 3 else 'moderate'
                })
            else:
                report['healthy_params'] += 1

        # Add summary statistics
        if self.violation_history:
            recent_violations = self.violation_history[-100:]
            report['recent_penalty_mean'] = np.mean([v['total_penalty'] for v in recent_violations])
            report['recent_violation_rate'] = np.mean([v['num_violations'] for v in recent_violations])

        return report

    def get_violation_trend(self, window=100):
        """
        Get trend of violations over time.

        Args:
            window: Number of recent steps to analyze

        Returns:
            Dictionary with trend information
        """
        if not self.violation_history:
            return {'trend': 'no_data'}

        recent = self.violation_history[-window:]

        penalties = [v['total_penalty'] for v in recent]
        counts = [v['num_violations'] for v in recent]

        return {
            'mean_penalty': np.mean(penalties),
            'std_penalty': np.std(penalties),
            'mean_violations': np.mean(counts),
            'max_violations': max(counts),
            'trend': 'increasing' if penalties[-1] > penalties[0] else 'decreasing'
        }

    def reset_adaptation(self):
        """Reset and restart adaptation period."""
        self.step_count = 0
        self.desirable_ranges = {}
        self.stats_history = defaultdict(list)
        self.violation_history = []
        self.initial_stats = self._compute_param_stats()
        print("Homeostasis monitor reset. Starting new adaptation period.")

    def is_healthy(self, tolerance=0.1):
        """
        Check if network is currently healthy.

        Args:
            tolerance: Maximum allowed violation rate (0-1)

        Returns:
            Boolean indicating health status
        """
        if self.step_count <= self.adaptation_period:
            return True  # Always healthy during adaptation

        report = self.get_health_report()

        if report['total_params'] == 0:
            return True

        violation_rate = len(report['violations']) / report['total_params']
        return violation_rate <= tolerance
