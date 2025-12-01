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
LR_PATH = 0.02
LR_GLOBAL_DECAY = 0.001

# spark settings
SPARK_START = 0
SPARK_FORCE_STEPS = 5
SPARK_ENERGY_DECAY = 0.98
SPARK_MIN_ENERGY = 0.05

# neuron activation decay
STATE_DECAY = 0.95


# ===================== NETWORK =====================
class MovingSpark(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.W = nn.Parameter(0.1 * torch.randn(n, n))
        self.register_buffer("s", torch.zeros(n))

        self.spark_pos = SPARK_START
        self.spark_energy = 1.0
        self.spark_age = 0

    def reset(self):
        self.s = torch.zeros(self.n)
        self.spark_pos = SPARK_START
        self.spark_energy = 1.0
        self.spark_age = 0

    @torch.no_grad()
    def step(self):

        # ===== GLOBAL DECAY OF STATE =====
        self.s *= STATE_DECAY

        # ===== STANDARD STATE UPDATE =====
        noise = NOISE_STD * torch.randn_like(self.s)
        self.s = torch.sigmoid(self.W @ self.s + noise)

        # ===== SPARK INJECTION =====
        if self.spark_age < SPARK_FORCE_STEPS:
            self.s[self.spark_pos] = 1.0

        # ===== SPARK MOVEMENT =====
        row = self.W[self.spark_pos, :]
        weights = torch.relu(row) + 1e-6
        probs = weights / weights.sum()
        next_pos = torch.multinomial(probs, 1).item()

        # ===== PATH WRITE (Hebbian-like reinforcement) =====
        self.W.data[self.spark_pos, next_pos] += LR_PATH * self.spark_energy

        # ===== GLOBAL WEIGHT DECAY =====
        self.W.data *= (1 - LR_GLOBAL_DECAY)
        self.W.data.clamp_(-2, 2)

        # ===== UPDATE SPARK =====
        self.spark_pos = next_pos
        self.spark_energy *= SPARK_ENERGY_DECAY
        self.s[self.spark_pos] = self.spark_energy
        self.spark_age += 1

        # respawn
        if self.spark_energy < SPARK_MIN_ENERGY:
            self.spark_pos = SPARK_START
            self.spark_energy = 1.0
            self.spark_age = 0


# ===================== VISUAL SETUP =====================
def setup_plots():
    fig, (ax_s, ax_w) = plt.subplots(1, 2, figsize=(14, 6))
    fig.subplots_adjust(top=0.8)

    state_img = ax_s.imshow(
        np.zeros((GRID_H, GRID_W)), vmin=0, vmax=1, interpolation="nearest"
    )
    ax_s.set_title("Neuron Activations (Spark Position)")
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
    net = MovingSpark(N_NEURONS)
    net.reset()

    fig, s_img, w_img = setup_plots()

    for step in range(1, TOTAL_STEPS + 1):
        net.step()

        fig.suptitle(
            f"Step {step}/{TOTAL_STEPS} — Spark at {net.spark_pos}",
            fontsize=12,
            x=0.02, ha='left'
        )

        s_img.set_data(net.s.view(GRID_H, GRID_W).cpu().numpy())
        w_img.set_data(net.W.data.cpu().numpy())

        plt.pause(0.001)

    # After finishing all steps, show static final state
    plt.show()


if __name__ == "__main__":
    main()
