import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

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
LR_MEMORY = 0.15        # short-term memory learning
MEM_DECAY = 0.92        # short-term memory fading

LAYOUT_FORCE_STEPS = 40  # iterations of physics layout
LAYOUT_LR = 0.01         # layout movement strength


# ===================== FORCE-LAYOUT GRAPH =====================

def force_layout(W, steps=50, lr=0.01):
    """Simple force-directed graph layout for 500 neurons."""
    n = W.shape[0]
    pos = np.random.randn(n, 2)

    # FIX — convert to numpy safely
    Wp = np.clip(W.detach().cpu().numpy(), 0, None)
    if Wp.max() > 0:
        Wp /= Wp.max()

    for _ in range(steps):
        diff = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(diff, axis=2) + 1e-6

        rep = diff / (dist[:, :, None] ** 2) * (-0.001)
        att = diff * Wp[:, :, None] * 0.02

        force = rep + att
        pos += lr * force.sum(axis=1)

    return pos


# ===================== NETWORK =====================

class SparkNet(nn.Module):
    def __init__(self, n, k):
        super().__init__()
        self.n = n
        self.k = k

        # weights
        self.W = nn.Parameter(0.1 * torch.randn(n, n))

        # neuron state
        self.register_buffer("s", torch.zeros(n))

        # short-term memory (same shape as neuron grid)
        self.register_buffer("M", torch.zeros(GRID_H, GRID_W))

        # sparks
        self.spark_pos = torch.randint(0, n, (k,))
        self.spark_energy = torch.ones(k)

    def reset(self):
        self.s.zero_()
        self.M.zero_()
        self.spark_pos = torch.randint(0, self.n, (self.k,))
        self.spark_energy = torch.ones(self.k)

    @torch.no_grad()
    def step(self):
        # neuron state decay
        self.s *= STATE_DECAY

        # basic network update
        noise = NOISE_STD * torch.randn_like(self.s)
        self.s = torch.sigmoid(self.W @ self.s + noise)

        # SPARKS UPDATE
        for i in range(self.k):

            prev = self.spark_pos[i]

            # movement probabilities
            row = self.W[prev, :]
            weights = torch.relu(row) + 1e-6
            probs = weights / weights.sum()

            next_pos = torch.multinomial(probs, 1).item()
            self.spark_pos[i] = next_pos

            # strengthen used connection
            self.W.data[next_pos, prev] = (
                self.W.data[next_pos, prev] * (1 - LR_EDGE)
                + self.s[prev] * LR_EDGE
            )

            # memory update on GRID
            gx = next_pos % GRID_W
            gy = next_pos // GRID_W
            self.M[gy, gx] = self.M[gy, gx] * MEM_DECAY + LR_MEMORY

            # spark energy decay
            self.spark_energy[i] *= 0.98
            self.s[next_pos] = 1.0

        # global decay
        self.W.data *= (1 - LR_GLOBAL_DECAY)
        self.W.data.clamp_(-1, 1)

        return self.spark_pos.tolist()


# ===================== PLOTTING =====================

def setup_plots():
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.subplots_adjust(hspace=0.3, wspace=0.25)

    ax_s = axes[0, 0]
    ax_w = axes[0, 1]
    ax_m = axes[1, 0]
    ax_g = axes[1, 1]

    s_img = ax_s.imshow(np.zeros((GRID_H, GRID_W)), vmin=0, vmax=1, interpolation="nearest")
    ax_s.set_title("Neuron Activations")

    w_img = ax_w.imshow(np.zeros((N_NEURONS, N_NEURONS)), cmap="bwr",
                        vmin=-0.3, vmax=0.3, interpolation="nearest")
    ax_w.set_title("Weight Matrix W")
    plt.colorbar(w_img, ax=ax_w)

    m_img = ax_m.imshow(np.zeros((GRID_H, GRID_W)), vmin=0, vmax=1, interpolation="nearest", cmap="viridis")
    ax_m.set_title("Memory Field M")

    g_scatter = ax_g.scatter([], [])
    ax_g.set_title("2D Graph Layout (force physics)")
    ax_g.set_xlim(-3, 3)
    ax_g.set_ylim(-3, 3)

    return fig, s_img, w_img, m_img, ax_g, g_scatter


# ===================== MAIN LOOP =====================

def main():
    net = SparkNet(N_NEURONS, NUM_SPARKS)
    net.reset()

    fig, s_img, w_img, m_img, ax_g, g_scatter = setup_plots()

    for step in range(1, TOTAL_STEPS + 1):

        positions = net.step()

        # update title
        fig.suptitle(f"Step {step}/{TOTAL_STEPS} — Sparks at {positions}",
                     fontsize=12, x=0.02, ha="left")

        # update plots
        s_img.set_data(net.s.view(GRID_H, GRID_W).cpu().numpy())
        w_img.set_data(net.W.detach().cpu().numpy())
        m_img.set_data(net.M.cpu().numpy())

        # update graph layout (every 40 steps)
        if step % 40 == 0:
            layout = force_layout(net.W, steps=LAYOUT_FORCE_STEPS, lr=LAYOUT_LR)
            ax_g.clear()
            ax_g.scatter(layout[:, 0], layout[:, 1], s=6)
            ax_g.set_title("2D Graph Layout (force physics)")
            ax_g.set_xlim(-3, 3)
            ax_g.set_ylim(-3, 3)

        plt.pause(0.001)

    plt.show()


if __name__ == "__main__":
    main()
