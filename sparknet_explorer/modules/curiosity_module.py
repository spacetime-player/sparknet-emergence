"""
Curiosity Module - Intrinsic Motivation Through Prediction Error

Implements a forward model that predicts the next state given current state and action.
The prediction error serves as an intrinsic curiosity reward signal:
- High error = model is surprised = explore this more
- Low error = model understands this = focus elsewhere

Also includes an inverse model that predicts actions from state transitions,
which helps learn better state representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CuriosityModule(nn.Module):
    """
    Forward model that predicts next state given current state and action.
    Prediction error serves as intrinsic curiosity reward.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=256, learning_rate=1e-3):
        """
        Initialize curiosity module.

        Args:
            state_dim: Dimensionality of state embeddings
            action_dim: Dimensionality of action space
            hidden_dim: Hidden layer size
            learning_rate: Learning rate for curiosity model updates
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Forward model: predicts next state from (state, action)
        self.forward_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, state_dim)
        )

        # Inverse model: predicts action from (state, next_state)
        # This helps learn features that are predictive of actions
        self.inverse_model = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_dim)
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # Track prediction errors for analysis
        self.prediction_errors = []

    def forward(self, state, action):
        """
        Predict next state given current state and action.

        Args:
            state: State embedding tensor of shape (batch, state_dim)
            action: Action tensor of shape (batch, action_dim)

        Returns:
            Predicted next state embedding
        """
        x = torch.cat([state, action], dim=-1)
        return self.forward_model(x)

    def predict_action(self, state, next_state):
        """
        Predict action from state transition (inverse model).

        Args:
            state: Current state embedding
            next_state: Next state embedding

        Returns:
            Predicted action
        """
        x = torch.cat([state, next_state], dim=-1)
        return self.inverse_model(x)

    def compute_curiosity_reward(self, state, action, next_state):
        """
        Compute intrinsic reward as prediction error.
        High error = model is surprised = explore more.

        Args:
            state: Current state embedding (batch, state_dim)
            action: Action taken (batch, action_dim)
            next_state: Actual next state (batch, state_dim)

        Returns:
            Curiosity reward tensor of shape (batch,)
        """
        with torch.no_grad():
            predicted_next = self.forward(state, action)
            prediction_error = F.mse_loss(predicted_next, next_state, reduction='none')

            # Average across features to get scalar reward per sample
            curiosity_reward = prediction_error.mean(dim=-1)

            # Track for analysis
            self.prediction_errors.append(curiosity_reward.mean().item())

        return curiosity_reward

    def update(self, state, action, next_state):
        """
        Update forward and inverse models to reduce prediction error.

        Args:
            state: Current state embedding (batch, state_dim)
            action: Action taken (batch, action_dim)
            next_state: Actual next state (batch, state_dim)

        Returns:
            Dictionary with loss values
        """
        self.optimizer.zero_grad()

        # Forward model loss - predict next state
        predicted_next = self.forward(state, action)
        forward_loss = F.mse_loss(predicted_next, next_state)

        # Inverse model loss - predict action from state transition
        # This helps learn state representations that encode action-relevant info
        predicted_action = self.predict_action(state, next_state)
        inverse_loss = F.mse_loss(predicted_action, action)

        # Combined loss (forward model is primary objective)
        total_loss = forward_loss + 0.5 * inverse_loss

        total_loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        self.optimizer.step()

        return {
            'forward_loss': forward_loss.item(),
            'inverse_loss': inverse_loss.item(),
            'total_curiosity_loss': total_loss.item()
        }

    def compute_information_gain(self, state, action, next_state):
        """
        Compute information gain as reduction in prediction uncertainty.

        This measures how much the model's understanding improved.

        Args:
            state: Current state
            action: Action taken
            next_state: Actual next state

        Returns:
            Information gain estimate
        """
        with torch.no_grad():
            # Compute prediction error before
            pred_before = self.forward(state, action)
            error_before = F.mse_loss(pred_before, next_state, reduction='none').mean(dim=-1)

            # After one gradient step, error would be lower (in expectation)
            # We approximate this by looking at gradient magnitude
            # High gradient = high potential for learning = high info gain

            # Simple proxy: use prediction error as info gain potential
            info_gain = error_before

        return info_gain

    def get_prediction_stats(self):
        """
        Get statistics about recent predictions.

        Returns:
            Dictionary with statistics
        """
        if not self.prediction_errors:
            return {
                'mean_error': 0.0,
                'std_error': 0.0,
                'num_predictions': 0
            }

        recent_errors = self.prediction_errors[-100:]  # Last 100 predictions

        return {
            'mean_error': sum(recent_errors) / len(recent_errors),
            'std_error': torch.tensor(recent_errors).std().item() if len(recent_errors) > 1 else 0.0,
            'num_predictions': len(self.prediction_errors)
        }

    def reset_stats(self):
        """Clear prediction error history."""
        self.prediction_errors = []
