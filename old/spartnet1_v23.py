import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# ===================== CONFIG =====================
N_NEURONS = 500
GRID_H, GRID_W = 20, 25
assert GRID_H * GRID_W == N_NEURONS

TOTAL_STEPS = 10000

NOISE_STD = 0.05

NUM_SPARKS = 3               # 3 particles
SPARK_FORCE_STEPS = 5
SPARK_ENERGY_DECAY = 0.98
SPARK_MIN_ENERGY = 0.05

STATE_DECAY = 0.95

LR_EDGE = 0.05               # ONLY used edge strengthened
LR_GLOBAL_DECAY = 0.001      # all edges fade slightly

GRAPH_UPDATE_EVERY = 50      # recompute MDS every N steps (performance)


# ===================== NETWORK =====================
class MultiSpark(nn.Module):
    def __init__(self, n, k):
        super().__init__()
        self.n = n
        self.k = k

        self.W = nn.Parameter(0.1 * torch.randn(n, n))
        self.register_buffer("s", torch.zeros(n))

        self.spark_pos = torch.arange(k) % n      # starting positions
        self.spark_energy = torch.ones(k)
        self.spark_age = torch.zeros(k, dtype=torch.long)

    def reset(self):
        self.s = torch.zeros(self.n)
        self.spark_pos = torch.arange(self.k) % self.n
        self.spark_energy = torch.ones(self.k)
        self.spark_age = torch.zeros(self.k, dtype=torch.long)

    @torch.no_grad()
    def step(self):
        # ===== GLOBAL STATE DECAY =====
        self.s *= STATE_DECAY

        # ===== NORMAL NETWORK UPDATE =====
        noise = NOISE_STD * torch.randn_like(self.s)
        self.s = torch.sigmoid(self.W @ self.s + noise)

        # ===== APPLY SPARK PRESENCE =====
        for i in range(self.k):
            if self.spark_age[i] < SPARK_FORCE_STEPS:
                self.s[self.spark_pos[i]] = 1.0

        # ===== MOVE SPARKS (one hop) + EDGE UPDATE =====
        for i in range(self.k):
            prev_pos = int(self.spark_pos[i].item())

            # movement probabilities based on outgoing weights
            row = self.W[prev_pos, :]
            weights = torch.relu(row) + 1e-6
            probs = weights / weights.sum()

            next_pos = torch.multinomial(probs, 1).item()
            self.spark_pos[i] = next_pos

            # ===== EDGE-ONLY UPDATE =====
            # strengthen ONLY the used connection prev→next
            self.W.data[next_pos, prev_pos] = (
                self.W.data[next_pos, prev_pos] * (1 - LR_EDGE)
                + self.s[prev_pos] * LR_EDGE
            )

            # update spark energy
            self.spark_energy[i] *= SPARK_ENERGY_DECAY
            self.s[next_pos] = self.spark_energy[i]
            self.spark_age[i] += 1

            # respawn if dead
            if self.spark_energy[i] < SPARK_MIN_ENERGY:
                self.spark_pos[i] = i % self.n
                self.spark_energy[i] = 1.0
                self.spark_age[i] = 0

        # ===== GLOBAL WEIGHT DECAY =====
        self.W.data *= (1 - LR_GLOBAL_DECAY)
        self.W.data.clamp_(-2, 2)

        return self.spark_pos.tolist()


# ===================== MDS LAYOUT (W -> 2D coords) =====================
def mds_from_weights(W_np):
    """
    Classical MDS on symmetrized |W|:
    - symmetrize W
    - map larger weights -> smaller distances
    - run classical MDS to get 2D coordinates
    """
    # symmetrize and take abs
    W_sym = 0.5 * (W_np + W_np.T)
    W_pos = np.abs(W_sym)

    # normalize to [0, 1]
    w_min, w_max = W_pos.min(), W_pos.max()
    W_norm = (W_pos - w_min) / (w_max - w_min + 1e-8)

    # distance = 1 - normalized weight (stronger weight => smaller distance)
    D = 1.0 - W_norm
    np.fill_diagonal(D, 0.0)

    # classical MDS: B = -0.5 * J D^2 J
    n = D.shape[0]
    D2 = D ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J

    # eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]  # largest first
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # take top-2 components
    lam = np.maximum(eigvals[:2], 0)
    V = eigvecs[:, :2]
    coords = V * np.sqrt(lam + 1e-8)

    # small rescale for nicer plot
    coords /= (np.abs(coords).max() + 1e-8)
    return coords


# ===================== VISUAL SETUP =====================
def setup_plots():
    fig, (ax_s, ax_w, ax_g) = plt.subplots(1, 3, figsize=(18, 6))
    fig.subplots_adjust(top=0.8)

    # neuron activations
    s_img = ax_s.imshow(
        np.zeros((GRID_H, GRID_W)), vmin=0, vmax=1, interpolation="nearest"
    )
    ax_s.set_title("Neuron Activations (3 Sparks)")
    ax_s.set_xlabel("Neuron X index")
    ax_s.set_ylabel("Neuron Y index")

    # weight matrix
    w_img = ax_w.imshow(
        np.zeros((N_NEURONS, N_NEURONS)),
        cmap="bwr",
        vmin=-0.3,
        vmax=0.3,
        interpolation="nearest"
    )
    ax_w.set_title("Weight Matrix W")
    ax_w.set_xlabel("Pre-synaptic neuron index")
    ax_w.set_ylabel("Post-synaptic neuron index")
    plt.colorbar(w_img, ax=ax_w)

    # graph layout view
    ax_g.set_title("Graph Layout (MDS: strong weights = close)")
    ax_g.set_xticks([])
    ax_g.set_yticks([])

    initial_coords = np.random.randn(N_NEURONS, 2)
    graph_scatter = ax_g.scatter(
        initial_coords[:, 0], initial_coords[:, 1], s=8
    )

    return fig, s_img, w_img, graph_scatter


# ===================== MAIN =====================
def main():
    net = MultiSpark(N_NEURONS, NUM_SPARKS)
    net.reset()

    fig, s_img, w_img, graph_scatter = setup_plots()

    coords = np.random.randn(N_NEURONS, 2)  # placeholder

    for step in range(1, TOTAL_STEPS + 1):
        positions = net.step()

        fig.suptitle(
            f"Step {step}/{TOTAL_STEPS} — Sparks: {positions}",
            fontsize=12, x=0.02, ha='left'
        )

        # update heatmaps
        s_img.set_data(net.s.view(GRID_H, GRID_W).cpu().numpy())
        w_img.set_data(net.W.data.cpu().numpy())

        # update graph layout every N steps
        if step % GRAPH_UPDATE_EVERY == 0 or step == 1:
            W_np = net.W.data.cpu().numpy()
            coords = mds_from_weights(W_np)
            graph_scatter.set_offsets(coords)

        plt.pause(0.001)

    plt.show()


if __name__ == "__main__":
    main()
