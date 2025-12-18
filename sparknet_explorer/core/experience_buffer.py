"""
Experience Buffer for Novelty Detection

Stores embeddings of visited states and computes novelty as minimum distance
to all stored states. High novelty = far from known states = exploration reward.
"""

import numpy as np
from collections import deque
import torch


class ExperienceBuffer:
    """Circular buffer storing state embeddings for novelty detection."""

    def __init__(self, max_size=10000, embedding_dim=128):
        """
        Initialize experience buffer.

        Args:
            max_size: Maximum number of states to store
            embedding_dim: Dimensionality of state embeddings
        """
        self.max_size = max_size
        self.embedding_dim = embedding_dim
        self.buffer = deque(maxlen=max_size)

    def add(self, state_embedding):
        """
        Add new state embedding to buffer.

        Args:
            state_embedding: Torch tensor of shape (embedding_dim,) or (batch, embedding_dim)
        """
        if state_embedding.dim() == 2:
            # Batch of embeddings - add each one
            for emb in state_embedding:
                self.buffer.append(emb.detach().cpu().numpy())
        else:
            self.buffer.append(state_embedding.detach().cpu().numpy())

    def compute_novelty(self, state_embedding, metric='cosine', percentile=10):
        """
        Compute novelty as minimum distance to all stored states.
        Higher = more novel = farther from known states.

        Args:
            state_embedding: Current state embedding tensor
            metric: Distance metric ('cosine' or 'l2')
            percentile: Use Nth percentile of distances instead of minimum
                       (more robust to outliers)

        Returns:
            Novelty score (float). Higher = more novel.
        """
        if len(self.buffer) == 0:
            return 1.0  # Maximum novelty for first state

        # Handle batch dimension
        if state_embedding.dim() == 2:
            current = state_embedding.detach().cpu().numpy()
        else:
            current = state_embedding.detach().cpu().numpy().reshape(1, -1)

        # Compute distances to all stored states
        distances = []

        if metric == 'cosine':
            # Cosine distance = 1 - cosine similarity
            for past in self.buffer:
                past = past.reshape(1, -1)
                similarity = np.dot(current, past.T) / (
                    np.linalg.norm(current, axis=1, keepdims=True) *
                    np.linalg.norm(past, axis=1, keepdims=True).T + 1e-8
                )
                distance = 1.0 - similarity
                distances.append(distance[0, 0])

        elif metric == 'l2':
            for past in self.buffer:
                past = past.reshape(1, -1)
                distance = np.linalg.norm(current - past, axis=1)
                distances.append(distance[0])

        else:
            raise ValueError(f"Unknown metric: {metric}")

        # Use percentile instead of minimum for robustness
        novelty = float(np.percentile(distances, percentile))

        # Normalize to [0, 1] for cosine distance
        if metric == 'cosine':
            novelty = np.clip(novelty, 0.0, 1.0)

        return novelty

    def get_coverage_stats(self):
        """
        Return statistics about state space coverage.

        Returns:
            Dictionary with coverage metrics
        """
        if len(self.buffer) < 2:
            return {
                'size': len(self.buffer),
                'diversity': 0.0,
                'coverage': len(self.buffer) / self.max_size
            }

        # Compute pairwise distances as measure of diversity
        states = np.array(list(self.buffer))

        # Sample for efficiency on large buffers
        sample_size = min(100, len(states))
        indices = np.random.choice(len(states), sample_size, replace=False)
        sampled_states = states[indices]

        distances = []
        for i in range(len(sampled_states)):
            for j in range(i+1, len(sampled_states)):
                dist = np.linalg.norm(sampled_states[i] - sampled_states[j])
                distances.append(dist)

        return {
            'size': len(self.buffer),
            'diversity': float(np.mean(distances)) if distances else 0.0,
            'std_diversity': float(np.std(distances)) if distances else 0.0,
            'coverage': len(self.buffer) / self.max_size,
            'is_full': len(self.buffer) == self.max_size
        }

    def clear(self):
        """Clear all stored experiences."""
        self.buffer.clear()

    def __len__(self):
        """Return number of stored states."""
        return len(self.buffer)

    def get_states_array(self):
        """
        Get all stored states as numpy array.

        Returns:
            Numpy array of shape (num_states, embedding_dim)
        """
        if len(self.buffer) == 0:
            return np.array([]).reshape(0, self.embedding_dim)
        return np.array(list(self.buffer))