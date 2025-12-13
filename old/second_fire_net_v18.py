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
TEMP = 0.3               # softmax temperature
EXPLORE_CHANCE = 0.1     # 10% pure random move

REINFORCE_MUL = 1.5      # multiplicative trace
REINFORCE_ADD = 0.05     # additive bump
LR_DECAY = 0.001         # decay for unvisited edges only


# ===================== NETWORK =====================
class SparkNet(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.W = nn.Parameter(0.1 * torch.randn(n, n))

        # neuron state vector
        self.register_buffer("s", torch.zeros(n))

        # spark state
        self.spark_pos = 0

        # persistent memory (visited edges)
        self.register_buffer("visited", torch.zeros((n, n), dtype=torch.bool))

    def reset(self):
        self.s = torch.zeros(self.n)
        self.spark_pos = 0
        self.visited[:] = False

    @torch.no_grad()
    def step(self):

        # ===== STATE DECAY + RNN UPDATE =====
        self.s *= 0.95
        noise = NOISE_STD * torch.randn_like(self.s)
        self.s = torch.sigmoid(self.W @ self.s + noise)

        # ===== SPARK ALWAYS FULL INTENSITY =====
        self.s[self.spark_pos] = 1.0

        prev = self.spark_pos

        # ===== MOVEMENT DECISION =====
        row = self.W[prev, :]
        positive = torch.relu(row) + 1e-6

        # temperature softmax for smooth preference
        logits = positive / TEMP
        probs = logits / logits.sum()

        # exploration: occasionally pick random neuron
        if torch.rand(1).item() < EXPLORE_CHANCE:
            next_pos = torch.randint(0, self.n, (1,)).item()
        else:
            next_pos = torch.multinomial(probs, 1).item()

        # ===== REINFORCEMENT OF PATH =====
        self.visited[prev, next_pos] = True

        self.W.data[prev, next_pos] = (
            self.W.data[prev, next_pos] * REINFORCE_MUL + REINFORCE_ADD
        )
        self.W.data.clamp_(-2, 2)

        # ===== DECAY UNVISITED EDGES =====
        decay_mask = ~self.visited
        self.W.data[decay_mask] *= (1 - LR_DECAY)

        # ===== UPDATE SPARK POSITION =====
        self.spark_pos = next_pos


# ===================== VISUAL SETUP =====================
def setup_plots():
    fig, (ax_s, ax_w) = plt.subplots(1, 2, figsize=(14, 6))
    fig.subplots_adjust(top=0.8)

    state_img = ax_s.imshow(
        np.zeros((GRID_H, GRID_W)),
        vmin=0, vmax=1, interpolation="nearest"
    )
    ax_s.set_title("Neuron Activations (Spark Highlight)")
    ax_s.set_xlabel("Neuron X index")
    ax_s.set_ylabel("Neuron Y index")

    weight_img = ax_w.imshow(
        np.zeros((500, 500)),
        cmap="bwr", vmin=-0.3, vmax=0.3, interpolation="nearest"
    )
    ax_w.set_title("Weight Matrix W")
    ax_w.set_xlabel("Pre-synaptic neuron")
    ax_w.set_ylabel("Post-synaptic neuron")
    plt.colorbar(weight_img, ax=ax_w)

    return fig, state_img, weight_img


# ===================== MAIN LOOP =====================
def main():
    net = SparkNet(N_NEURONS)
    net.reset()

    fig, s_img, w_img = setup_plots()

    for step in range(1, TOTAL_STEPS + 1):
        net.step()

        fig.suptitle(
            f"Step {step}/{TOTAL_STEPS} — Spark at neuron {net.spark_pos}",
            fontsize=12, x=0.02, ha='left'
        )

        s_img.set_data(net.s.view(GRID_H, GRID_W).cpu().numpy())
        w_img.set_data(net.W.data.cpu().numpy())

        plt.pause(0.001)

    plt.show()


if __name__ == "__main__":
    main()
