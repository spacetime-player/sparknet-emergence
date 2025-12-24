"""
SparkNet Explorer - Live Training Interface

Real-time visualization dashboard showing:
- Weight matrices for 3 main networks (Action, State Embedder, Curiosity)
- Spring-physics graph layouts (weight = spring strength)
- Live terminal output with training stats
- Network architecture description

Layout:
┌─────────────┬─────────────┬─────────────┬──────────────────────┐
│  ACTION NET │  EMBEDDER   │  CURIOSITY  │  TERMINAL OUTPUT     │
│  (weights)  │  (weights)  │  (weights)  │                      │
├─────────────┼─────────────┼─────────────┤                      │
│  ACTION NET │  EMBEDDER   │  CURIOSITY  │                      │
│  (spring)   │  (spring)   │  (spring)   ├──────────────────────┤
│             │             │             │  NETWORK DESCRIPTION │
└─────────────┴─────────────┴─────────────┴──────────────────────┘
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import numpy as np
import torch
import time

# Dark theme
plt.style.use('dark_background')


def force_layout(weights, pos=None, steps=10, lr=0.02):
    """
    Force-directed graph layout where weight = spring strength.
    Stronger weights pull nodes closer together.

    Args:
        weights: Weight matrix (n x m) or (n x n)
        pos: Initial positions (optional)
        steps: Number of physics iterations
        lr: Learning rate for position updates

    Returns:
        Node positions (n, 2)
    """
    W = weights.detach().cpu().numpy() if torch.is_tensor(weights) else weights
    n = W.shape[0]

    # Initialize positions
    if pos is None:
        pos = np.random.randn(n, 2) * 0.5
    else:
        pos = pos.copy()

    # Normalize weights
    W_abs = np.abs(W)
    if W_abs.max() > 0:
        W_norm = W_abs / W_abs.max()
    else:
        W_norm = W_abs

    # For non-square matrices, use transpose for bidirectional forces
    if W.shape[0] != W.shape[1]:
        # Create symmetric version for force calculation
        m = W.shape[1]
        total_nodes = max(n, m)
        W_sym = np.zeros((total_nodes, total_nodes))
        W_sym[:n, :m] = W_norm
        W_sym[:m, :n] = W_norm.T
        W_norm = W_sym

        # Expand positions if needed
        if len(pos) < total_nodes:
            extra = np.random.randn(total_nodes - len(pos), 2) * 0.5
            pos = np.vstack([pos, extra])
        n = total_nodes

    for _ in range(steps):
        # Pairwise differences
        diff = pos[:, None, :] - pos[None, :, :]  # (n, n, 2)
        dist = np.linalg.norm(diff, axis=2) + 1e-6  # (n, n)

        # Repulsion: all nodes repel each other (inverse square)
        repulsion = diff / (dist[:, :, None] ** 2) * 0.01

        # Attraction: weighted by connection strength (spring force)
        # F = -k * x, where k = weight strength
        attraction = -diff * W_norm[:, :, None] * 0.05

        # Total force
        force = (repulsion + attraction).sum(axis=1)

        # Update positions
        pos += lr * force

        # Keep bounded
        pos = np.clip(pos, -3, 3)

    return pos[:W.shape[0]]  # Return only relevant nodes


class LiveInterface:
    """
    Real-time training visualization interface for SparkNetExplorer.
    """

    def __init__(self, model, update_freq=100):
        """
        Initialize live interface.

        Args:
            model: SparkNetExplorer model
            update_freq: Update visualization every N steps
        """
        self.model = model
        self.update_freq = update_freq
        self.step = 0

        # Store graph positions for smooth animation
        self.action_pos = None
        self.embedder_pos = None
        self.curiosity_pos = None

        # Terminal log buffer
        self.terminal_lines = []
        self.max_terminal_lines = 20

        # Setup figure
        self._setup_figure()

    def _setup_figure(self):
        """Create the visualization layout."""
        self.fig = plt.figure(figsize=(18, 10))
        self.fig.patch.set_facecolor('#1a1a1a')

        # Grid: 2 rows, 4 columns
        # Left 3 cols: networks (2 rows each)
        # Right col: terminal (top) + description (bottom)

        gs = self.fig.add_gridspec(
            2, 4,
            width_ratios=[1, 1, 1, 1.2],
            height_ratios=[1, 1],
            hspace=0.25,
            wspace=0.2,
            left=0.05, right=0.98, top=0.92, bottom=0.05
        )

        # === ROW 1: Weight Matrices ===
        self.ax_action_w = self.fig.add_subplot(gs[0, 0])
        self.ax_embedder_w = self.fig.add_subplot(gs[0, 1])
        self.ax_curiosity_w = self.fig.add_subplot(gs[0, 2])

        # === ROW 2: Spring Graphs ===
        self.ax_action_g = self.fig.add_subplot(gs[1, 0])
        self.ax_embedder_g = self.fig.add_subplot(gs[1, 1])
        self.ax_curiosity_g = self.fig.add_subplot(gs[1, 2])

        # === RIGHT COLUMN: Terminal + Description ===
        self.ax_terminal = self.fig.add_subplot(gs[0, 3])
        self.ax_description = self.fig.add_subplot(gs[1, 3])

        # Initialize plots
        self._init_weight_plots()
        self._init_graph_plots()
        self._init_terminal()
        self._init_description()

        # Main title
        self.fig.suptitle(
            'SparkNet Explorer - Live Training Interface',
            color='white', fontsize=14, fontweight='bold'
        )

        plt.ion()
        plt.show(block=False)

    def _init_weight_plots(self):
        """Initialize weight matrix heatmaps."""
        # Get initial weights
        action_w = self._get_action_weights()
        embedder_w = self._get_embedder_weights()
        curiosity_w = self._get_curiosity_weights()

        # Action Network
        self.action_w_img = self.ax_action_w.imshow(
            action_w, cmap='RdBu_r', aspect='auto',
            interpolation='bilinear', vmin=-0.3, vmax=0.3
        )
        self.ax_action_w.set_title('ACTION NETWORK', color='#4CAF50', fontsize=11, fontweight='bold')
        self.ax_action_w.set_xlabel('Input', color='white', fontsize=8)
        self.ax_action_w.set_ylabel('Output', color='white', fontsize=8)
        self.ax_action_w.tick_params(colors='white', labelsize=7)
        cbar1 = self.fig.colorbar(self.action_w_img, ax=self.ax_action_w, fraction=0.046)
        cbar1.ax.tick_params(colors='white', labelsize=7)

        # State Embedder
        self.embedder_w_img = self.ax_embedder_w.imshow(
            embedder_w, cmap='RdBu_r', aspect='auto',
            interpolation='bilinear', vmin=-0.3, vmax=0.3
        )
        self.ax_embedder_w.set_title('STATE EMBEDDER', color='#2196F3', fontsize=11, fontweight='bold')
        self.ax_embedder_w.set_xlabel('Input', color='white', fontsize=8)
        self.ax_embedder_w.set_ylabel('Output', color='white', fontsize=8)
        self.ax_embedder_w.tick_params(colors='white', labelsize=7)
        cbar2 = self.fig.colorbar(self.embedder_w_img, ax=self.ax_embedder_w, fraction=0.046)
        cbar2.ax.tick_params(colors='white', labelsize=7)

        # Curiosity Module
        self.curiosity_w_img = self.ax_curiosity_w.imshow(
            curiosity_w, cmap='RdBu_r', aspect='auto',
            interpolation='bilinear', vmin=-0.3, vmax=0.3
        )
        self.ax_curiosity_w.set_title('CURIOSITY MODULE', color='#FF9800', fontsize=11, fontweight='bold')
        self.ax_curiosity_w.set_xlabel('Input', color='white', fontsize=8)
        self.ax_curiosity_w.set_ylabel('Output', color='white', fontsize=8)
        self.ax_curiosity_w.tick_params(colors='white', labelsize=7)
        cbar3 = self.fig.colorbar(self.curiosity_w_img, ax=self.ax_curiosity_w, fraction=0.046)
        cbar3.ax.tick_params(colors='white', labelsize=7)

    def _init_graph_plots(self):
        """Initialize spring-physics graph visualizations."""
        for ax, title, color in [
            (self.ax_action_g, 'Action Network (spring layout)', '#4CAF50'),
            (self.ax_embedder_g, 'State Embedder (spring layout)', '#2196F3'),
            (self.ax_curiosity_g, 'Curiosity Module (spring layout)', '#FF9800'),
        ]:
            ax.set_facecolor('#0d0d0d')
            ax.set_title(title, color=color, fontsize=9)
            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)
            ax.tick_params(colors='white', labelsize=7)
            ax.grid(True, alpha=0.1, color='white')

        # Create scatter plots
        self.action_scatter = self.ax_action_g.scatter([], [], s=8, c='#4CAF50', alpha=0.7)
        self.embedder_scatter = self.ax_embedder_g.scatter([], [], s=8, c='#2196F3', alpha=0.7)
        self.curiosity_scatter = self.ax_curiosity_g.scatter([], [], s=8, c='#FF9800', alpha=0.7)

    def _init_terminal(self):
        """Initialize terminal output panel."""
        self.ax_terminal.set_facecolor('#0a0a0a')
        self.ax_terminal.set_xlim(0, 1)
        self.ax_terminal.set_ylim(0, 1)
        self.ax_terminal.axis('off')

        # Title bar
        self.ax_terminal.add_patch(Rectangle(
            (0, 0.92), 1, 0.08,
            facecolor='#2d2d2d', edgecolor='#4CAF50', linewidth=1
        ))
        self.ax_terminal.text(
            0.5, 0.96, 'TERMINAL OUTPUT',
            ha='center', va='center', color='#4CAF50',
            fontsize=10, fontweight='bold', family='monospace'
        )

        # Terminal text area
        self.terminal_text = self.ax_terminal.text(
            0.02, 0.88, '',
            ha='left', va='top', color='#00ff00',
            fontsize=7, family='monospace',
            transform=self.ax_terminal.transAxes
        )

        # Border
        self.ax_terminal.add_patch(Rectangle(
            (0, 0), 1, 0.92,
            facecolor='none', edgecolor='#333333', linewidth=1
        ))

    def _init_description(self):
        """Initialize network description panel."""
        self.ax_description.set_facecolor('#0a0a0a')
        self.ax_description.set_xlim(0, 1)
        self.ax_description.set_ylim(0, 1)
        self.ax_description.axis('off')

        # Title bar
        self.ax_description.add_patch(Rectangle(
            (0, 0.92), 1, 0.08,
            facecolor='#2d2d2d', edgecolor='#9C27B0', linewidth=1
        ))
        self.ax_description.text(
            0.5, 0.96, 'NETWORK ARCHITECTURE',
            ha='center', va='center', color='#9C27B0',
            fontsize=10, fontweight='bold', family='monospace'
        )

        # Description text
        desc = self._get_network_description()
        self.ax_description.text(
            0.03, 0.88, desc,
            ha='left', va='top', color='white',
            fontsize=7, family='monospace',
            transform=self.ax_description.transAxes,
            linespacing=1.4
        )

        # Border
        self.ax_description.add_patch(Rectangle(
            (0, 0), 1, 0.92,
            facecolor='none', edgecolor='#333333', linewidth=1
        ))

    def _get_network_description(self):
        """Generate network architecture description."""
        # Get architecture info from model
        hidden_dims = []
        for name, param in self.model.named_parameters():
            if 'hidden_layers' in name and 'weight' in name:
                hidden_dims.append(param.shape[0])

        # Count unique hidden sizes (skip duplicates from bias)
        hidden_sizes = list(dict.fromkeys(hidden_dims))[:3]

        desc = f"""SPARKNET EXPLORER ARCHITECTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ACTION NETWORK (decides movement)
  Input:  {self.model.input_dim}D position
  Hidden: {' → '.join(map(str, hidden_sizes))}
  Output: {self.model.output_dim}D action
  Role:   "WHERE should I go?"

STATE EMBEDDER (compresses meaning)
  Input:  {hidden_sizes[-1] if hidden_sizes else '?'}D hidden state
  Hidden: 128 → {self.model.state_embedding_dim}D
  Role:   "What does this MEAN?"

CURIOSITY MODULE (measures surprise)
  Forward: Predicts next state
  Inverse: Infers action from change
  Role:   "Was this SURPRISING?"

HOMEOSTASIS (monitors stability)
  Tracks: {len(list(self.model.parameters()))} parameter groups
  Role:   "Is system HEALTHY?"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Data Flow: Position → Embed → Act
           → Predict → Learn → Repeat
"""
        return desc

    def _get_action_weights(self):
        """Get first layer weights from action network."""
        for name, param in self.model.named_parameters():
            if 'hidden_layers.0.weight' in name:
                W = param.detach().cpu().numpy()
                # Subsample if too large
                if W.shape[0] > 100:
                    W = W[::W.shape[0]//100, :]
                if W.shape[1] > 100:
                    W = W[:, ::W.shape[1]//100]
                return W
        return np.zeros((10, 10))

    def _get_embedder_weights(self):
        """Get weights from state embedder."""
        for name, param in self.model.named_parameters():
            if 'state_embedder.0.weight' in name:
                W = param.detach().cpu().numpy()
                if W.shape[0] > 100:
                    W = W[::W.shape[0]//100, :]
                if W.shape[1] > 100:
                    W = W[:, ::W.shape[1]//100]
                return W
        return np.zeros((10, 10))

    def _get_curiosity_weights(self):
        """Get weights from curiosity forward model."""
        for name, param in self.model.named_parameters():
            if 'curiosity_module.forward_model.0.weight' in name:
                W = param.detach().cpu().numpy()
                if W.shape[0] > 100:
                    W = W[::W.shape[0]//100, :]
                if W.shape[1] > 100:
                    W = W[:, ::W.shape[1]//100]
                return W
        return np.zeros((10, 10))

    def log(self, message):
        """Add message to terminal output."""
        self.terminal_lines.append(message)
        if len(self.terminal_lines) > self.max_terminal_lines:
            self.terminal_lines.pop(0)

    def update(self, step, metrics=None):
        """
        Update visualization with current training state.

        Args:
            step: Current training step
            metrics: Dict with current metrics (optional)
        """
        self.step = step

        # Only update at specified frequency
        if step % self.update_freq != 0:
            return

        # Update weight matrices
        self.action_w_img.set_data(self._get_action_weights())
        self.embedder_w_img.set_data(self._get_embedder_weights())
        self.curiosity_w_img.set_data(self._get_curiosity_weights())

        # Update spring graphs
        self._update_graphs()

        # Update terminal
        if metrics:
            self._update_terminal(step, metrics)

        # Refresh display
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def _update_graphs(self):
        """Update spring-physics graph layouts."""
        # Action network graph
        action_w = self._get_action_weights()
        self.action_pos = force_layout(action_w, self.action_pos, steps=5)
        self.action_scatter.set_offsets(self.action_pos)

        # Embedder graph
        embedder_w = self._get_embedder_weights()
        self.embedder_pos = force_layout(embedder_w, self.embedder_pos, steps=5)
        self.embedder_scatter.set_offsets(self.embedder_pos)

        # Curiosity graph
        curiosity_w = self._get_curiosity_weights()
        self.curiosity_pos = force_layout(curiosity_w, self.curiosity_pos, steps=5)
        self.curiosity_scatter.set_offsets(self.curiosity_pos)

    def _update_terminal(self, step, metrics):
        """Update terminal output with current stats."""
        # Format metrics
        pos = metrics.get('position', [0, 0])
        novelty = metrics.get('novelty', 0)
        curiosity = metrics.get('curiosity', 0)
        total_reward = metrics.get('total_reward', 0)
        exploration_rate = metrics.get('exploration_rate', 0)
        exploring = metrics.get('exploring', False)
        diversity = metrics.get('diversity', 0)
        health = metrics.get('param_health', '?/?')

        # Add log line
        line = f"[{step:5d}] pos:[{pos[0]:+.2f},{pos[1]:+.2f}] " \
               f"nov:{novelty:.3f} cur:{curiosity:.4f} " \
               f"rew:{total_reward:.3f} exp:{exploration_rate:.3f} " \
               f"{'[EXPLORING]' if exploring else ''}"
        self.log(line)

        # Update terminal text
        terminal_content = '\n'.join(self.terminal_lines)
        self.terminal_text.set_text(terminal_content)

    def close(self):
        """Close the interface."""
        plt.close(self.fig)


def create_live_interface(model, update_freq=100):
    """
    Factory function to create live interface.

    Args:
        model: SparkNetExplorer model
        update_freq: Update every N steps

    Returns:
        LiveInterface instance
    """
    return LiveInterface(model, update_freq)


# Standalone test
if __name__ == '__main__':
    print("Live Interface module loaded.")
    print("Use create_live_interface(model) to create visualization.")
