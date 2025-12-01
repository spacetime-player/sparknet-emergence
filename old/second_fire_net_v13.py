import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# ===================== CONFIG =====================
N_NEURONS = 500
GRID_H, GRID_W = 20, 25
assert GRID_H * GRID_W == N_NEURONS

TOTAL_STEPS = 100000

NOISE_STD = 0.05
LR = 0.01

# spark movement
SPARK_START = 0            # start neuron index
SPARK_FORCE_STEPS = 5      # spark stays strong at start
SPARK_ENERGY_DECAY = 0.98  # how spark fades
SPARK_MIN_ENERGY = 0.05    # dies when below this


# ===================== NETWORK =====================
class MovingSpark(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.W = nn.Parameter(0.1 * torch.randn(n, n))
        self.register_buffer("s", torch.zeros(n))

        # spark properties
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
        # ===== GLOBAL PARALLEL UPDATE =====
        noise = NOISE_STD * torch.randn_like(self.s)
        self.s = torch.sigmoid(self.W @ self.s + noise)

        # ===== SPARK INJECTION =====
        if self.spark_age < SPARK_FORCE_STEPS:
            self.s[self.spark_pos] = 1.0

        # ===== SPARK MOVEMENT =====
        row = self.W[self.spark_pos, :]       # weights from current neuron
        weights = torch.relu(row) + 1e-6      # only positive direction movement
        probs = weights / weights.sum()       # normalize to probabilities

        next_pos = torch.multinomial(probs, 1).item()

        # ===== WRITE PATH (Hebbian) =====
        self.W.data[next_pos, :] += LR * self.s
        self.W.data.clamp_(-2, 2)

        # ===== UPDATE SPARK =====
        self.spark_pos = next_pos
        self.spark_energy *= SPARK_ENERGY_DECAY
        self.s[self.spark_pos] = self.spark_energy
        self.spark_age += 1

        # spark dies + instantly respawns at start
        if self.spark_energy < SPARK_MIN_ENERGY:
            self.spark_pos = SPARK_START
            self.spark_energy = 1.0
            self.spark_age = 0


# ===================== VISUAL =====================
def setup_plots():
    fig, (ax_s, ax_w) = plt.subplots(1, 2, figsize=(12, 6))
    fig.subplots_adjust(top=0.78)

    state_img = ax_s.imshow(
        np.zeros((GRID_H, GRID_W)), vmin=0, vmax=1, interpolation="nearest"
    )
    ax_s.set_title("Neuron Activations (Spark Position)")
    ax_s.set_xticks([]); ax_s.set_yticks([])

    weight_img = ax_w.imshow(
        np.zeros((N_NEURONS, N_NEURONS)),
        cmap="bwr", vmin=-0.3, vmax=0.3, interpolation="nearest"
    )
    plt.colorbar(weight_img, ax=ax_w)
    ax_w.set_title("Weight Matrix W")
    ax_w.set_xticks([]); ax_w.set_yticks([])

    return fig, state_img, weight_img


# ===================== MAIN =====================
def main():
    net = MovingSpark(N_NEURONS)
    net.reset()

    fig, s_img, w_img = setup_plots()

    for step in range(1, TOTAL_STEPS + 1):
        net.step()

        fig.suptitle(f"Step {step}/{TOTAL_STEPS} — Spark at {net.spark_pos}", 
                     x=0.02, ha='left')

        # update visuals
        s_img.set_data(net.s.view(GRID_H, GRID_W).cpu().numpy())
        w_img.set_data(net.W.data.cpu().numpy())

        plt.pause(0.001)


if __name__ == "__main__":
    main()
    plt.show(block=False)
    plt.pause(0.1)
    plt.close("all")
