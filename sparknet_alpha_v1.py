import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Set dark theme for plots
plt.style.use('dark_background')

# ===================== CONFIG =====================

N_NEURONS = 500
GRID_H, GRID_W = 20, 25
assert GRID_H * GRID_W == N_NEURONS

TOTAL_STEPS = 10000

NUM_SPARKS = 3

STATE_DECAY = 0.95
NOISE_STD = 0.05

LR_EDGE = 0.05          # strengthen used edges
LR_GLOBAL_DECAY = 0.001 # small decay everywhere

# Memory system (inspired by v26)
MEM_DECAY = 0.92        # memory field fading per step
MEM_DEPOSIT = 0.15      # how much memory sparks deposit
MEM_BIAS = 0.8          # how strongly memory influences movement
EXPLORE_CHANCE = 0.05   # random exploration probability

# Movement control
TEMPERATURE = 0.3       # softmax temperature for movement

# Graph layout physics
LAYOUT_FORCE_STEPS = 40
LAYOUT_LR = 0.01

# Spark properties
SPARK_ENERGY_DECAY = 0.98
SPARK_MIN_ENERGY = 0.05
SPARK_FORCE_STEPS = 5


# ===================== FORCE-LAYOUT GRAPH =====================

def force_layout(W, steps=50, lr=0.01):
    """Simple force-directed graph layout for visualization."""
    n = W.shape[0]
    pos = np.random.randn(n, 2)

    # Convert weights to numpy safely
    Wp = np.clip(W.detach().cpu().numpy(), 0, None)
    if Wp.max() > 0:
        Wp /= Wp.max()

    for _ in range(steps):
        diff = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(diff, axis=2) + 1e-6

        # Repulsion force
        rep = diff / (dist[:, :, None] ** 2) * (-0.001)
        # Attraction force based on weights
        att = diff * Wp[:, :, None] * 0.02

        force = rep + att
        pos += lr * force.sum(axis=1)

    return pos


# ===================== NETWORK =====================

class SparkNetAlpha(nn.Module):
    def __init__(self, n, k):
        super().__init__()
        self.n = n
        self.k = k

        # Core network weights
        self.W = nn.Parameter(0.1 * torch.randn(n, n))

        # Neuron states
        self.register_buffer("s", torch.zeros(n))

        # Memory field - key innovation from v26
        self.register_buffer("M", torch.zeros(n))

        # Spark system
        self.spark_pos = torch.randint(0, n, (k,))
        self.spark_energy = torch.ones(k)
        self.spark_age = torch.zeros(k, dtype=torch.long)

    def reset(self):
        self.s.zero_()
        self.M.zero_()
        self.spark_pos = torch.randint(0, self.n, (self.k,))
        self.spark_energy = torch.ones(self.k)
        self.spark_age = torch.zeros(self.k, dtype=torch.long)

    @torch.no_grad()
    def step(self):
        # ===== 1) DECAY SYSTEMS =====
        self.s *= STATE_DECAY           # neuron activations fade
        self.M *= MEM_DECAY             # memory field fades

        # ===== 2) BASIC NETWORK UPDATE =====
        noise = NOISE_STD * torch.randn_like(self.s)
        self.s = torch.sigmoid(self.W @ self.s + noise)

        # ===== 3) SPARK PRESENCE =====
        # Young sparks force their neurons to high activation
        for i in range(self.k):
            if self.spark_age[i] < SPARK_FORCE_STEPS:
                self.s[self.spark_pos[i]] = 1.0

        # ===== 4) SPARK MOVEMENT WITH MEMORY BIAS =====
        for i in range(self.k):
            prev = int(self.spark_pos[i].item())

            # Get outgoing weights from current position
            row = self.W[prev, :]
            base_weights = torch.relu(row) + 1e-6

            # KEY INNOVATION: Combine weights with memory bias
            # Higher memory at target locations makes them more attractive
            logits = base_weights / TEMPERATURE + MEM_BIAS * self.M

            # Sometimes ignore everything and explore randomly
            if torch.rand(1).item() < EXPLORE_CHANCE:
                next_pos = torch.randint(0, self.n, (1,)).item()
            else:
                probs = torch.softmax(logits, dim=0)
                next_pos = torch.multinomial(probs, 1).item()

            self.spark_pos[i] = next_pos

            # ===== 5) UPDATE LONG-TERM STRUCTURE =====
            # Strengthen the connection that was just used
            self.W.data[next_pos, prev] = (
                self.W.data[next_pos, prev] * (1 - LR_EDGE)
                + self.s[prev] * LR_EDGE
            )

            # ===== 6) UPDATE MEMORY FIELD =====
            # Deposit memory at new location - this influences future movement
            self.M[next_pos] += MEM_DEPOSIT

            # ===== 7) SPARK LIFECYCLE =====
            self.spark_energy[i] *= SPARK_ENERGY_DECAY
            self.s[next_pos] = self.spark_energy[i]
            self.spark_age[i] += 1

            # Respawn spark if energy too low
            if self.spark_energy[i] < SPARK_MIN_ENERGY:
                # Respawn with memory bias - hotspots more likely
                if self.M.sum() > 0:
                    mem_probs = torch.softmax(self.M * 2.0, dim=0)  # amplify memory differences
                    new_pos = torch.multinomial(mem_probs, 1).item()
                else:
                    new_pos = torch.randint(0, self.n, (1,)).item()
                
                self.spark_pos[i] = new_pos
                self.spark_energy[i] = 1.0
                self.spark_age[i] = 0

        # ===== 8) GLOBAL WEIGHT DECAY =====
        self.W.data *= (1 - LR_GLOBAL_DECAY)
        self.W.data.clamp_(-1, 1)

        return self.spark_pos.tolist()


# ===================== PLOTTING SETUP =====================

def setup_plots():
    # Create 2x2 subplot layout with dark background
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('#2e2e2e')  # Dark gray background
    fig.subplots_adjust(hspace=0.3, wspace=0.25)

    ax_s = axes[0, 0]    # Top left: Neuron activations
    ax_w = axes[0, 1]    # Top right: Weight matrix
    ax_m = axes[1, 0]    # Bottom left: Memory field
    ax_g = axes[1, 1]    # Bottom right: Graph layout

    # Neuron activations plot
    s_img = ax_s.imshow(np.zeros((GRID_H, GRID_W)), vmin=0, vmax=1, 
                        interpolation="nearest", cmap="viridis")
    ax_s.set_title("Neuron Activations", color='white', fontsize=12)
    ax_s.tick_params(colors='white')

    # Weight matrix plot
    w_img = ax_w.imshow(np.zeros((N_NEURONS, N_NEURONS)), cmap="bwr",
                        vmin=-0.3, vmax=0.3, interpolation="nearest")
    ax_w.set_title("Weight Matrix W", color='white', fontsize=12)
    ax_w.tick_params(colors='white')
    cbar_w = plt.colorbar(w_img, ax=ax_w)
    cbar_w.ax.tick_params(colors='white')

    # Memory field plot
    m_img = ax_m.imshow(np.zeros((GRID_H, GRID_W)), vmin=0, vmax=1, 
                        interpolation="nearest", cmap="plasma")
    ax_m.set_title("Memory Field M", color='white', fontsize=12)
    ax_m.tick_params(colors='white')
    cbar_m = plt.colorbar(m_img, ax=ax_m)
    cbar_m.ax.tick_params(colors='white')

    # Graph layout plot
    ax_g.set_title("2D Graph Layout (force physics)", color='white', fontsize=12)
    ax_g.set_xlim(-3, 3)
    ax_g.set_ylim(-3, 3)
    ax_g.tick_params(colors='white')
    ax_g.set_facecolor('#1e1e1e')  # Slightly darker background for graph
    g_scatter = ax_g.scatter([], [], s=6, c='cyan', alpha=0.7)

    return fig, s_img, w_img, m_img, ax_g, g_scatter


# ===================== MAIN LOOP =====================

def main():
    net = SparkNetAlpha(N_NEURONS, NUM_SPARKS)
    net.reset()

    fig, s_img, w_img, m_img, ax_g, g_scatter = setup_plots()

    for step in range(1, TOTAL_STEPS + 1):
        positions = net.step()

        # Update main title with white text
        fig.suptitle(f"Step {step}/{TOTAL_STEPS} — Sparks at {positions}",
                     fontsize=14, x=0.02, ha="left", color='white')

        # Update neuron activations
        s_img.set_data(net.s.view(GRID_H, GRID_W).cpu().numpy())
        
        # Update weight matrix
        w_img.set_data(net.W.detach().cpu().numpy())
        
        # Update memory field
        M_clamped = torch.clamp(net.M, min=0.0, max=1.0)
        m_img.set_data(M_clamped.view(GRID_H, GRID_W).cpu().numpy())

        # Update graph layout (every 40 steps to avoid too much computation)
        if step % 40 == 0:
            layout = force_layout(net.W, steps=LAYOUT_FORCE_STEPS, lr=LAYOUT_LR)
            ax_g.clear()
            ax_g.scatter(layout[:, 0], layout[:, 1], s=6, c='cyan', alpha=0.7)
            ax_g.set_title("2D Graph Layout (force physics)", color='white', fontsize=12)
            ax_g.set_xlim(-3, 3)
            ax_g.set_ylim(-3, 3)
            ax_g.tick_params(colors='white')
            ax_g.set_facecolor('#1e1e1e')

        plt.pause(0.001)

    plt.show()


if __name__ == "__main__":
    main()