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

LR_PATH = 0.02           # how much used edges are strengthened
LR_GLOBAL_DECAY = 0.001  # how fast unused edges fade
DECAY_RESIST_FACTOR = 0.1  # visited edges decay 10× slower

# spark parameters
SPARK_FORCE_STEPS = 5
SPARK_ENERGY_DECAY = 0.98
SPARK_MIN_ENERGY = 0.05
BRANCH_THRESHOLD = 0.6    # if weight is this high, spark may branch
BRANCH_PROB = 0.3         # chance of branching


# ===================== NETWORK =====================
class MultiSparkNetwork(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.W = nn.Parameter(0.1 * torch.randn(n, n))

        # neuron activations
        self.register_buffer("s", torch.zeros(n))

        # a list of spark objects: (pos, energy, age)
        self.sparks = []

    def reset(self):
        self.s = torch.zeros(self.n)
        # start with single spark at neuron 0
        self.sparks = [(0, 1.0, 0)]   # (position, energy, age)

    @torch.no_grad()
    def step(self):

        # ===== DECAY + STANDARD RNN UPDATE =====
        self.s *= 0.95
        noise = NOISE_STD * torch.randn_like(self.s)
        self.s = torch.sigmoid(self.W @ self.s + noise)

        used_edges = []  # will store (i,j) all edges that were traveled

        new_sparks = []  # sparks after this step

        for (pos, energy, age) in self.sparks:

            # ===== FORCE SPARK EARLY =====
            if age < SPARK_FORCE_STEPS:
                self.s[pos] = 1.0

            # ===== DETERMINE NEXT MOVES (branching allowed) =====
            row = self.W[pos, :]
            positive = torch.relu(row) + 1e-6
            probs = positive / positive.sum()

            # always choose one main next position
            main_next = torch.multinomial(probs, 1).item()
            moves = [main_next]

            # branching: if some weights are strong enough
            candidates = torch.where(positive > BRANCH_THRESHOLD)[0]
            for cand in candidates:
                if torch.rand(1).item() < BRANCH_PROB:
                    moves.append(int(cand))

            # ===== UPDATE ALL MOVES =====
            for nxt in moves:

                # record edge used
                used_edges.append((pos, nxt))

                # strengthen edge only
                self.W.data[pos, nxt] += LR_PATH * energy

                # create new spark
                new_energy = energy * SPARK_ENERGY_DECAY
                new_age = age + 1

                # if energy too low → do not keep spark
                if new_energy >= SPARK_MIN_ENERGY:
                    new_sparks.append((nxt, new_energy, new_age))

        self.sparks = new_sparks

        # ===== GLOBAL WEIGHT DECAY (slow on used edges) =====
        decay_matrix = torch.ones_like(self.W) * (1 - LR_GLOBAL_DECAY)

        # reduce decay on all used edges
        for (i, j) in used_edges:
            decay_matrix[i, j] = 1 - (LR_GLOBAL_DECAY * DECAY_RESIST_FACTOR)

        self.W.data *= decay_matrix
        self.W.data.clamp_(-2, 2)


# ===================== VISUALS =====================
def setup_plots():
    fig, (ax_s, ax_w) = plt.subplots(1, 2, figsize=(14, 6))
    fig.subplots_adjust(top=0.8)

    state_img = ax_s.imshow(
        np.zeros((GRID_H, GRID_W)), vmin=0, vmax=1, interpolation="nearest"
    )
    ax_s.set_title("Neuron Activations")
    ax_s.set_xlabel("Neuron X index")
    ax_s.set_ylabel("Neuron Y index")

    weight_img = ax_w.imshow(
        np.zeros((N_NEURONS, N_NEURONS)),
        cmap="bwr",
        vmin=-0.3,
        vmax=0.3,
        interpolation="nearest"
    )
    ax_w.set_title("Weight Matrix W")
    ax_w.set_xlabel("Pre-synaptic neuron index")
    ax_w.set_ylabel("Post-synaptic neuron index")
    plt.colorbar(weight_img, ax=ax_w)

    return fig, state_img, weight_img


# ===================== MAIN =====================
def main():
    net = MultiSparkNetwork(N_NEURONS)
    net.reset()

    fig, s_img, w_img = setup_plots()

    for step in range(1, TOTAL_STEPS + 1):
        net.step()

        # HUD
        fig.suptitle(
            f"Step {step}/{TOTAL_STEPS} — Sparks alive: {len(net.sparks)}",
            fontsize=12,
            x=0.02,
            ha='left'
        )

        # update visuals
        s_img.set_data(net.s.view(GRID_H, GRID_W).cpu().numpy())
        w_img.set_data(net.W.data.cpu().numpy())

        plt.pause(0.001)

    plt.show()


if __name__ == "__main__":
    main()
