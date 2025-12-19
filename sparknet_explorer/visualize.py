"""
Visualization Suite for SparkNet Explorer

Provides comprehensive visualizations of:
- Reward decomposition
- Exploration metrics (novelty, state space coverage)
- Homeostatic health
- Network structure and weights
- Learning dynamics
"""

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import torch

# Set dark theme for consistency with main SparkNet
plt.style.use('dark_background')


def plot_training_metrics(metrics_history, save_path='sparknet_explorer_metrics.png'):
    """
    Plot comprehensive training metrics including reward decomposition
    and exploration statistics.

    Args:
        metrics_history: Dictionary of tracked metrics
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.patch.set_facecolor('#2e2e2e')

    # Plot 1: Reward decomposition
    ax = axes[0, 0]
    if 'extrinsic' in metrics_history:
        ax.plot(metrics_history['extrinsic'], label='Extrinsic', alpha=0.7, linewidth=1.5, color='#4CAF50')
    if 'intrinsic' in metrics_history:
        ax.plot(metrics_history['intrinsic'], label='Intrinsic', alpha=0.7, linewidth=1.5, color='#2196F3')
    if 'homeostatic' in metrics_history:
        ax.plot(metrics_history['homeostatic'], label='Homeostatic Penalty', alpha=0.7, linewidth=1.5, color='#F44336')
    if 'boredom_penalty' in metrics_history:
        ax.plot(metrics_history['boredom_penalty'], label='Boredom Penalty', alpha=0.7, linewidth=1.5, color='#FF6F00')
    ax.set_xlabel('Training Step', color='white')
    ax.set_ylabel('Reward/Penalty', color='white')
    ax.set_title('Reward Decomposition', color='white', fontweight='bold')
    ax.legend(facecolor='#3e3e3e', edgecolor='white', fontsize=9)
    ax.grid(True, alpha=0.2, color='white')
    ax.tick_params(colors='white')

    # Plot 2: Novelty over time
    ax = axes[0, 1]
    if 'novelty' in metrics_history:
        ax.plot(metrics_history['novelty'], alpha=0.7, color='#9C27B0', linewidth=1.5)
    ax.set_xlabel('Training Step', color='white')
    ax.set_ylabel('Novelty Score', color='white')
    ax.set_title('State Space Exploration (Novelty)', color='white', fontweight='bold')
    ax.grid(True, alpha=0.2, color='white')
    ax.tick_params(colors='white')

    # Plot 3: Curiosity (prediction error) over time
    ax = axes[0, 2]
    if 'curiosity' in metrics_history:
        ax.plot(metrics_history['curiosity'], alpha=0.7, color='#FF9800', linewidth=1.5)
    ax.set_xlabel('Training Step', color='white')
    ax.set_ylabel('Curiosity Reward', color='white')
    ax.set_title('Curiosity (Prediction Error)', color='white', fontweight='bold')
    ax.grid(True, alpha=0.2, color='white')
    ax.tick_params(colors='white')

    # Plot 4: Total loss
    ax = axes[1, 0]
    if 'loss' in metrics_history:
        ax.plot(metrics_history['loss'], alpha=0.7, color='#F44336', linewidth=1.5)
    ax.set_xlabel('Training Step', color='white')
    ax.set_ylabel('Loss', color='white')
    ax.set_title('Total Loss (Negative Total Reward)', color='white', fontweight='bold')
    ax.grid(True, alpha=0.2, color='white')
    ax.tick_params(colors='white')

    # Plot 5: Moving averages of rewards
    ax = axes[1, 1]
    window = min(50, len(metrics_history.get('extrinsic', [])) // 10)
    if window > 0 and 'extrinsic' in metrics_history:
        if len(metrics_history['extrinsic']) > window:
            extrinsic_ma = np.convolve(metrics_history['extrinsic'],
                                        np.ones(window)/window, mode='valid')
            intrinsic_ma = np.convolve(metrics_history['intrinsic'],
                                        np.ones(window)/window, mode='valid')
            total_ma = np.convolve(metrics_history['total_reward'],
                                   np.ones(window)/window, mode='valid')

            ax.plot(extrinsic_ma, label='Extrinsic', alpha=0.7, linewidth=2, color='#4CAF50')
            ax.plot(intrinsic_ma, label='Intrinsic', alpha=0.7, linewidth=2, color='#2196F3')
            ax.plot(total_ma, label='Total', alpha=0.7, linewidth=2, color='cyan')

    ax.set_xlabel('Training Step', color='white')
    ax.set_ylabel('Reward (Moving Average)', color='white')
    ax.set_title(f'Reward Trends ({window}-step MA)', color='white', fontweight='bold')
    ax.legend(facecolor='#3e3e3e', edgecolor='white')
    ax.grid(True, alpha=0.2, color='white')
    ax.tick_params(colors='white')

    # Plot 6: Exploration rate decay
    ax = axes[1, 2]
    if 'exploration_rate' in metrics_history and len(metrics_history['exploration_rate']) > 0:
        rates = metrics_history['exploration_rate']
        ax.plot(rates, alpha=0.7, color='#00BCD4', linewidth=1.5)
        # Auto-scale to actual data range with small padding
        min_rate = min(rates)
        max_rate = max(rates)
        padding = (max_rate - min_rate) * 0.1 if max_rate > min_rate else 0.1
        ax.set_ylim(max(0, min_rate - padding), min(1, max_rate + padding))
    ax.set_xlabel('Training Step', color='white')
    ax.set_ylabel('Exploration Rate', color='white')
    ax.set_title('Exploration Rate Decay', color='white', fontweight='bold')
    ax.grid(True, alpha=0.2, color='white')
    ax.tick_params(colors='white')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, facecolor='#2e2e2e')
    print(f"Training metrics saved to {save_path}")
    plt.close()  # Close instead of show to prevent opening


def visualize_weight_heatmap(model, layer_name='hidden_layers.0', save_path='weight_heatmap.png'):
    """
    Visualize weight matrix as heatmap with fluid-like coloring.

    Args:
        model: SparkNetExplorer model
        layer_name: Name of layer to visualize
        save_path: Path to save figure
    """
    # Get weights
    weights = None
    for name, param in model.named_parameters():
        if layer_name in name and 'weight' in name:
            weights = param.detach().cpu().numpy()
            break

    if weights is None:
        print(f"Layer {layer_name} not found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor('#2e2e2e')

    # Heatmap
    ax = axes[0]
    im = ax.imshow(weights, cmap='RdBu_r', aspect='auto', interpolation='bilinear')
    ax.set_title(f'Weight Matrix: {layer_name}', color='white', fontweight='bold')
    ax.set_xlabel('Input Dimension', color='white')
    ax.set_ylabel('Output Dimension', color='white')
    ax.tick_params(colors='white')
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(colors='white')
    cbar.set_label('Weight Value', color='white')

    # Weight distribution
    ax = axes[1]
    ax.hist(weights.flatten(), bins=50, alpha=0.7, edgecolor='white', color='#2196F3')
    ax.set_xlabel('Weight Value', color='white')
    ax.set_ylabel('Frequency', color='white')
    ax.set_title('Weight Distribution', color='white', fontweight='bold')
    ax.axvline(0, color='#F44336', linestyle='--', alpha=0.7, label='Zero', linewidth=2)
    ax.axvline(weights.mean(), color='#4CAF50', linestyle='--', alpha=0.7, label='Mean', linewidth=2)
    ax.legend(facecolor='#3e3e3e', edgecolor='white')
    ax.grid(True, alpha=0.2, color='white')
    ax.tick_params(colors='white')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, facecolor='#2e2e2e')
    print(f"Weight heatmap saved to {save_path}")
    plt.show()


def visualize_network_graph(model, threshold=0.1, save_path='network_graph.png'):
    """
    Visualize network as graph where edges are weighted connections.
    Uses 'fluid-like' visualization with edge thickness = weight magnitude.

    Args:
        model: SparkNetExplorer model
        threshold: Minimum weight magnitude to display
        save_path: Path to save figure
    """
    G = nx.DiGraph()

    # Build graph from first two layers (for visualization clarity)
    layer_count = 0
    max_layers = 2  # Limit for visualization

    for name, param in model.named_parameters():
        if 'weight' in name and 'hidden' in name and layer_count < max_layers:
            weights = param.detach().cpu().numpy()

            # Limit nodes for visualization
            n_out, n_in = weights.shape
            n_in = min(n_in, 20)  # Max 20 input nodes
            n_out = min(n_out, 20)  # Max 20 output nodes
            weights = weights[:n_out, :n_in]

            # Add nodes
            input_nodes = [f"L{layer_count}_I{i}" for i in range(n_in)]
            output_nodes = [f"L{layer_count+1}_O{i}" for i in range(n_out)]

            G.add_nodes_from(input_nodes, layer=layer_count)
            G.add_nodes_from(output_nodes, layer=layer_count+1)

            # Add edges above threshold
            for i in range(n_out):
                for j in range(n_in):
                    weight = weights[i, j]
                    if abs(weight) > threshold:
                        G.add_edge(input_nodes[j], output_nodes[i],
                                   weight=float(weight))

            layer_count += 1

    if len(G.nodes()) == 0:
        print("No nodes to visualize with current threshold")
        return

    # Layout
    pos = nx.multipartite_layout(G, subset_key='layer')

    # Draw
    fig = plt.figure(figsize=(15, 10))
    fig.patch.set_facecolor('#2e2e2e')
    ax = plt.gca()
    ax.set_facecolor('#1e1e1e')

    # Edges with weight-based thickness and color
    edges = G.edges()
    if len(edges) > 0:
        weights = [G[u][v]['weight'] for u, v in edges]
        abs_weights = [abs(w) for w in weights]

        # Normalize for visualization
        max_weight = max(abs_weights) if abs_weights else 1
        edge_widths = [3 * (w / max_weight) for w in abs_weights]
        edge_colors = ['#F44336' if w < 0 else '#2196F3' for w in weights]

        nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors,
                                alpha=0.6, arrows=True, arrowsize=10, ax=ax)

    # Nodes
    nx.draw_networkx_nodes(G, pos, node_size=200, node_color='#4CAF50',
                            alpha=0.9, edgecolors='white', linewidths=2, ax=ax)

    plt.title('SparkNet Explorer: Network Graph Visualization\n'
              '(Edge thickness = weight magnitude, Red = negative, Blue = positive)',
              color='white', fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#2e2e2e')
    print(f"Network graph saved to {save_path}")
    plt.show()


def plot_state_space_exploration(experience_buffer, save_path='state_space_exploration.png'):
    """
    Visualize state space coverage using PCA for dimensionality reduction.

    Args:
        experience_buffer: ExperienceBuffer instance
        save_path: Path to save figure
    """
    from sklearn.decomposition import PCA

    if len(experience_buffer) < 10:
        print("Not enough states to visualize (need at least 10)")
        return

    # Get all states
    states = experience_buffer.get_states_array()

    # Reduce to 2D using PCA
    pca = PCA(n_components=2)
    states_2d = pca.fit_transform(states)

    # Plot
    fig = plt.figure(figsize=(12, 9))
    fig.patch.set_facecolor('#2e2e2e')
    ax = plt.gca()
    ax.set_facecolor('#1e1e1e')

    # Color by time (order in buffer)
    scatter = ax.scatter(states_2d[:, 0], states_2d[:, 1],
                c=range(len(states_2d)), cmap='viridis',
                alpha=0.6, s=50, edgecolors='white', linewidths=0.5)

    cbar = plt.colorbar(scatter, label='Time Step')
    cbar.ax.tick_params(colors='white')
    cbar.set_label('Time Step', color='white')

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', color='white')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', color='white')
    ax.set_title('State Space Exploration Over Time\n(PCA projection of state embeddings)',
                 color='white', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.2, color='white')
    ax.tick_params(colors='white')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, facecolor='#2e2e2e')
    print(f"State space exploration saved to {save_path}")
    plt.close()


def plot_homeostasis_health(model, save_path='homeostasis_health.png'):
    """
    Visualize homeostatic health and parameter drift.

    Args:
        model: SparkNetExplorer model
        save_path: Path to save figure
    """
    health = model.homeostasis.get_health_report()

    if health['status'] == 'adapting':
        print("Homeostasis still in adaptation period - no health data yet")
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor('#2e2e2e')

    # Plot 1: Parameter health overview
    ax = axes[0]
    healthy = health['healthy_params']
    violations = len(health['violations'])
    total = health['total_params']

    colors = ['#4CAF50', '#F44336']
    sizes = [healthy, violations]
    labels = [f'Healthy ({healthy})', f'Violations ({violations})']

    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
           startangle=90, textprops={'color': 'white', 'fontsize': 12})
    ax.set_title(f'Parameter Health Overview ({total} total params)',
                 color='white', fontsize=14, fontweight='bold')

    # Plot 2: Violation severity
    ax = axes[1]
    if health['violations']:
        param_names = [v['param'].split('.')[-2] + '.' + v['param'].split('.')[-1]
                       for v in health['violations'][:10]]  # Top 10
        drifts = [abs(v['drift_std']) for v in health['violations'][:10]]
        colors_bar = ['#FF9800' if v['severity'] == 'moderate' else '#F44336'
                      for v in health['violations'][:10]]

        y_pos = np.arange(len(param_names))
        ax.barh(y_pos, drifts, color=colors_bar, edgecolor='white', linewidth=1)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(param_names, color='white', fontsize=9)
        ax.set_xlabel('Drift (standard deviations)', color='white')
        ax.set_title('Parameter Violations (drift from desirable range)',
                     color='white', fontsize=14, fontweight='bold')
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.2, color='white', axis='x')

        # Add severity legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#FF9800', label='Moderate'),
                           Patch(facecolor='#F44336', label='High')]
        ax.legend(handles=legend_elements, facecolor='#3e3e3e',
                  edgecolor='white', loc='lower right')
    else:
        ax.text(0.5, 0.5, 'No violations!\nAll parameters healthy',
                ha='center', va='center', color='#4CAF50',
                fontsize=16, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, facecolor='#2e2e2e')
    print(f"Homeostasis health saved to {save_path}")
    plt.show()


def create_comprehensive_report(model, metrics_history, output_dir='./'):
    """
    Create comprehensive visualization report with all plots.

    Args:
        model: SparkNetExplorer model
        metrics_history: Dictionary of tracked metrics
        output_dir: Directory to save plots
    """
    import os

    print("\nGenerating comprehensive visualization report...")

    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # Generate all plots
    plot_training_metrics(metrics_history,
                          save_path=os.path.join(output_dir, 'training_metrics.png'))

    visualize_weight_heatmap(model,
                             save_path=os.path.join(output_dir, 'weight_heatmap.png'))

    visualize_network_graph(model,
                            save_path=os.path.join(output_dir, 'network_graph.png'))

    plot_state_space_exploration(model.experience_buffer,
                                 save_path=os.path.join(output_dir, 'state_space.png'))

    plot_homeostasis_health(model,
                            save_path=os.path.join(output_dir, 'homeostasis_health.png'))

    print(f"\nVisualization report complete! Files saved to {output_dir}")
