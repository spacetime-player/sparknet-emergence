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

NUM_SPARKS = 3
SPARK_FORCE_STEPS = 5
SPARK_ENERGY_DECAY = 0.98
SPARK_MIN_ENERGY = 0.05

STATE_DECAY = 0.95

LR_EDGE = 0.05
LR_GLOBAL_DECAY = 0.001

K_NEIGH = 5                # ripple neighbors
RIPPLE_STRENGTH = 0.01

# movement pattern:
# 0 = deterministic, 1 = softmax, 2 = random
TEMPERATURE = 0.3

# graph physics
SPRING_STEPS = 2
ATTR_SCALE = 0.001
REP_SCALE = 0.0005
DT = 0.1
GRAPH_UPDATE_EVERY = 20


# ===================== NETWORK =====================
class MultiSpark(nn.Module):
    def __init__(self, n, k):
        super().__init__()
        self.n = n
        self.k = k

        self.W = nn.Parameter(0.1 * torch.randn(n, n))
        self.register_buffer("s", torch.zeros(n))

        self.spark_pos = torch.arange(k) % n
        self.spark_energy = torch.ones(k)
        self.spark_age = torch.zeros(k, dtype=torch.long)

    def reset(self):
        self.s = torch.zeros(self.n)
        self.spark_pos = torch.arange(self.k) % self.n
        self.spark_energy = torch.ones(self.k)
        self.spark_age = torch.zeros(self.k, dtype=torch.long)

    @torch.no_grad()
    def step(self, step_num):

        # decay activation
        self.s *= STATE_DECAY

        # normal network update
        noise = NOISE_STD * torch.randn_like(self.s)
        self.s = torch.sigmoid(self.W @ self.s + noise)

        # sparks force their position
        for i in range(self.k):
            if self.spark_age[i] < SPARK_FORCE_STEPS:
                self.s[self.spark_pos[i]] = 1.0

        # move sparks
        for i in range(self.k):

            prev = int(self.spark_pos[i].item())

            row = self.W[prev, :]
            weights = torch.relu(row) + 1e-6

            mode = step_num % 3

            # ========== MOVEMENT SYSTEM ==========
            if mode == 0:
                # deterministic (greedy)
                next_pos = torch.argmax(weights).item()

            elif mode == 1:
                # softmax controlled choice
                logits = weights / TEMPERATURE
                probs = logits / logits.sum()
                next_pos = torch.multinomial(probs, 1).item()

            else:
                # random exploratory
                next_pos = torch.randint(0, self.n, (1,)).item()

            self.spark_pos[i] = next_pos

            # ========== DIRECT EDGE UPDATE ==========
            self.W.data[next_pos, prev] = (
                self.W.data[next_pos, prev] * (1 - LR_EDGE)
                + self.s[prev] * LR_EDGE
            )

            # ========== RIPPLE UPDATE ==========
            row_vals = self.W.data[prev, :]
            top_idx = torch.topk(torch.relu(row_vals), K_NEIGH).indices

            for j in top_idx:
                self.W.data[prev, j] += RIPPLE_STRENGTH
                self.W.data[j, prev] += RIPPLE_STRENGTH * 0.5

            for j in top_idx:
                for k in top_idx:
                    self.W.data[j, k] += RIPPLE_STRENGTH * 0.3

            # update spark life
            self.spark_energy[i] *= SPARK_ENERGY_DECAY
            self.s[next_pos] = self.spark_energy[i]
            self.spark_age[i] += 1

            # respawn dead spark
            if self.spark_energy[i] < SPARK_MIN_ENERGY:
                self.spark_pos[i] = i % self.n
                self.spark_energy[i] = 1.0
                self.spark_age[i] = 0

        # global weight decay
        self.W.data *= (1 - LR_GLOBAL_DECAY)
        self.W.data.clamp_(-2, 2)

        return self.spark_pos.tolist()


# ===================== FORCE PHYSICS =====================
def spring_layout_step(coords, W):
    n = coords.shape[0]
    forces = np.zeros_like(coords)

    # repulsion between all nodes
    for i in range(n):
        diff = coords[i] - coords
        dist2 = np.sum(diff**2, axis=1) + 1e-6
        forces[i] += REP_SCALE * np.sum(diff / dist2[:, None], axis=0)

    # attraction by weight
    Wp = np.maximum(W, 0)
    rows, cols = np.nonzero(Wp > 0)

    for i, j in zip(rows, cols):
        diff = coords[j] - coords[i]
        forces[i] += ATTR_SCALE * Wp[i, j] * diff

    coords += DT * forces
    return coords


# ===================== VISUALS =====================
def setup_plots():
    fig, (ax_s, ax_w, ax_g) = plt.subplots(1, 3, figsize=(18, 6))
    fig.subplots_adjust(top=0.8)

    s_img = ax_s.imshow(
        np.zeros((GRID_H, GRID_W)), vmin=0, vmax=1, interpolation="nearest"
    )
    ax_s.set_title("Neuron Activations (3 Sparks)")

    w_img = ax_w.imshow(
        np.zeros((N_NEURONS, N_NEURONS)),
        cmap="bwr", vmin=-0.3, vmax=0.3, interpolation="nearest"
    )
    ax_w.set_title("Weight Matrix W")
    plt.colorbar(w_img, ax=ax_w)

    ax_g.set_title("2D Graph Layout (Force Physics)")
    ax_g.set_xticks([]); ax_g.set_yticks([])

    coords = np.random.randn(N_NEURONS, 2)
    coords /= coords.std()
    scatter = ax_g.scatter(coords[:, 0], coords[:, 1], s=8)

    return fig, s_img, w_img, scatter, coords


# ===================== MAIN =====================
def main():
    net = MultiSpark(N_NEURONS, NUM_SPARKS)
    net.reset()

    fig, s_img, w_img, scatter, coords = setup_plots()

    for step in range(1, TOTAL_STEPS + 1):

        positions = net.step(step)

        fig.suptitle(
            f"Step {step}/{TOTAL_STEPS} — Sparks: {positions}",
            fontsize=12, x=0.02, ha='left'
        )

        s_img.set_data(net.s.view(GRID_H, GRID_W).cpu().numpy())
        W_np = net.W.data.cpu().numpy()
        w_img.set_data(W_np)

        if step % GRAPH_UPDATE_EVERY == 0:
            for _ in range(SPRING_STEPS):
                coords = spring_layout_step(coords, W_np)
            scatter.set_offsets(coords)

        plt.pause(0.001)

    plt.show()


if __name__ == "__main__":
    main()
