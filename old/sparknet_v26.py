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

# long-term structure (weights)
LR_EDGE = 0.05          # only used edge strengthened
LR_GLOBAL_DECAY = 0.001 # all weights fade slightly

# short-term memory field M
M_DECAY = 0.95          # memory fade per step
M_DEPOSIT = 0.2         # how much a spark adds at its position
M_GAIN = 0.8            # how strongly M biases movement

# movement stochasticity
TEMP = 0.3              # softmax temperature
EXPLORE_CHANCE = 0.05   # 5% pure random jump


# ===================== NETWORK =====================
class SparkFieldNet(nn.Module):
    def __init__(self, n, k):
        super().__init__()
        self.n = n
        self.k = k

        # structure: recurrent weights
        self.W = nn.Parameter(0.1 * torch.randn(n, n))

        # state: neuron activations
        self.register_buffer("s", torch.zeros(n))

        # short-term memory field (not a weight matrix)
        self.register_buffer("M", torch.zeros(n))

        # sparks: positions, energy, age
        self.spark_pos = torch.arange(k) % n
        self.spark_energy = torch.ones(k)
        self.spark_age = torch.zeros(k, dtype=torch.long)

    def reset(self):
        self.s = torch.zeros(self.n)
        self.M = torch.zeros(self.n)
        self.spark_pos = torch.arange(self.k) % self.n
        self.spark_energy = torch.ones(self.k)
        self.spark_age = torch.zeros(self.k, dtype=torch.long)

    @torch.no_grad()
    def step(self):
        # ===== 1) DECAY FIELD & STATE =====
        self.M *= M_DECAY              # short-term memory fades
        self.s *= STATE_DECAY          # neuron activations fade a bit

        # ===== 2) NORMAL RECURRENT UPDATE =====
        noise = NOISE_STD * torch.randn_like(self.s)
        self.s = torch.sigmoid(self.W @ self.s + noise)

        # early-life sparks force their neuron to 1.0
        for i in range(self.k):
            if self.spark_age[i] < SPARK_FORCE_STEPS:
                self.s[self.spark_pos[i]] = 1.0

        # ===== 3) MOVE EACH SPARK (M influences choice) =====
        for i in range(self.k):
            prev = int(self.spark_pos[i].item())

            # outgoing weights from prev
            row = self.W[prev, :]
            base = torch.relu(row) + 1e-6

            # combine weights with memory field M
            # higher M[j] makes that neuron more attractive
            logits = base / TEMP + M_GAIN * self.M

            # exploration: sometimes ignore structure+M and jump anywhere
            if torch.rand(1).item() < EXPLORE_CHANCE:
                next_pos = torch.randint(0, self.n, (1,)).item()
            else:
                probs = torch.softmax(logits, dim=0)
                next_pos = torch.multinomial(probs, 1).item()

            self.spark_pos[i] = next_pos

            # ===== 4) UPDATE LONG-TERM STRUCTURE (W) ALONG USED EDGE =====
            self.W.data[next_pos, prev] = (
                self.W.data[next_pos, prev] * (1 - LR_EDGE)
                + self.s[prev] * LR_EDGE
            )

            # ===== 5) UPDATE SHORT-TERM MEMORY FIELD (M) =====
            # deposit at new position; this biases future moves
            self.M[next_pos] += M_DEPOSIT

            # ===== 6) SPARK ENERGY / LIFE =====
            self.spark_energy[i] *= SPARK_ENERGY_DECAY
            self.s[next_pos] = self.spark_energy[i]
            self.spark_age[i] += 1

            # respawn when energy too low
            if self.spark_energy[i] < SPARK_MIN_ENERGY:
                self.spark_pos[i] = i % self.n
                self.spark_energy[i] = 1.0
                self.spark_age[i] = 0

        # ===== 7) GLOBAL WEIGHT DECAY =====
        self.W.data *= (1 - LR_GLOBAL_DECAY)
        self.W.data.clamp_(-2, 2)

        return self.spark_pos.tolist()


# ===================== VISUALS =====================
def setup_plots():
    fig, (ax_s, ax_w, ax_m) = plt.subplots(1, 3, figsize=(18, 6))
    fig.subplots_adjust(top=0.8)

    # neuron activations (left)
    s_img = ax_s.imshow(
        np.zeros((GRID_H, GRID_W)), vmin=0, vmax=1, interpolation="nearest"
    )
    ax_s.set_title("Neuron Activations (Sparks)")
    ax_s.set_xlabel("Neuron X index")
    ax_s.set_ylabel("Neuron Y index")

    # weight matrix (middle)
    w_img = ax_w.imshow(
        np.zeros((N_NEURONS, N_NEURONS)),
        cmap="bwr", vmin=-0.3, vmax=0.3, interpolation="nearest"
    )
    ax_w.set_title("Weight Matrix W (Structure)")
    ax_w.set_xlabel("Pre-synaptic neuron index")
    ax_w.set_ylabel("Post-synaptic neuron index")
    plt.colorbar(w_img, ax=ax_w)

    # memory field (right)
    m_img = ax_m.imshow(
        np.zeros((GRID_H, GRID_W)),
        cmap="viridis", vmin=0, vmax=1, interpolation="nearest"
    )
    ax_m.set_title("Memory Field M (Short-term)")
    ax_m.set_xlabel("Neuron X index")
    ax_m.set_ylabel("Neuron Y index")

    return fig, s_img, w_img, m_img


# ===================== MAIN =====================
def main():
    net = SparkFieldNet(N_NEURONS, NUM_SPARKS)
    net.reset()

    fig, s_img, w_img, m_img = setup_plots()

    for step in range(1, TOTAL_STEPS + 1):
        positions = net.step()

        fig.suptitle(
            f"Step {step}/{TOTAL_STEPS} — Sparks at {positions}",
            fontsize=12, x=0.02, ha='left'
        )

        # activations
        s_img.set_data(net.s.view(GRID_H, GRID_W).cpu().numpy())

        # weights
        w_img.set_data(net.W.data.cpu().numpy())

        # memory field
        M_clamped = torch.clamp(net.M, min=0.0, max=1.0)
        m_img.set_data(M_clamped.view(GRID_H, GRID_W).cpu().numpy())

        plt.pause(0.001)

    plt.show()


if __name__ == "__main__":
    main()
