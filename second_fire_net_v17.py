import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# =====================================================
# CONFIG
# =====================================================
N_NEURONS = 500
GRID_H, GRID_W = 20, 25
assert GRID_H * GRID_W == N_NEURONS

TOTAL_STEPS = 10000

NOISE_STD = 0.05

LR_PATH = 0.02                # strengthening of used edges
LR_DECAY = 0.001              # base decay rate
DECAY_VISITED_FACTOR = 0.02   # visited edges decay 50× slower

# spark behaviour
SPARK_FORCE_STEPS = 5
SPARK_ENERGY_DECAY = 0.98
SPARK_MIN_ENERGY = 0.05
BRANCH_THRESHOLD = 0.6
BRANCH_PROB = 0.3


# =====================================================
# NETWORK
# =====================================================
class MultiSparkMemoryNet(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n

        # weights
        self.W = nn.Parameter(0.1 * torch.randn(n, n))

        # neuron activations
        self.register_buffer("s", torch.zeros(n))

        # sparks = (pos, energy, age)
        self.sparks = []

        # persistent visited mask
        self.register_buffer("visited", torch.zeros((n, n), dtype=torch.bool))

    def reset(self):
        self.s = torch.zeros(self.n)
        self.sparks = [(0, 1.0, 0)]
        self.visited[:] = False

    @torch.no_grad()
    def step(self):

        # =============== STATE UPDATE ===============
        self.s *= 0.95
        noise = NOISE_STD * torch.randn_like(self.s)
        self.s = torch.sigmoid(self.W @ self.s + noise)

        new_sparks = []
        used_edges = []

        # =============== PROCESS ALL SPARKS ===============
        for (pos, energy, age) in self.sparks:

            # early strong forced spike
            if age < SPARK_FORCE_STEPS:
                self.s[pos] = 1.0

            # choose moves
            row = self.W[pos, :]
            positive = torch.relu(row) + 1e-6
            probs = positive / positive.sum()

            # always one main move
            main_next = torch.multinomial(probs, 1).item()
            moves = [main_next]

            # branching
            strong = torch.where(positive > BRANCH_THRESHOLD)[0]
            for cand in strong:
                if torch.rand(1).item() < BRANCH_PROB:
                    moves.append(int(cand))

            # update sparks and record used edges
            for nxt in moves:

                used_edges.append((pos, nxt))

                # mark persistent memory
                self.visited[pos, nxt] = True

                # strengthen only used edge
                self.W.data[pos, nxt] += LR_PATH * energy

                # update spark energy
                new_energy = energy * SPARK_ENERGY_DECAY
                new_age = age + 1

                if new_energy >= SPARK_MIN_ENERGY:
                    new_sparks.append((nxt, new_energy, new_age))

        self.sparks = new_sparks

        # =============== DECAY MANAGEMENT ===============
        # base decay mask
        decay = torch.ones_like(self.W) * (1 - LR_DECAY)

        # visited edges decay much slower
        decay[self.visited] = (1 - LR_DECAY * DECAY_VISITED_FACTOR)

        # apply decay
        self.W.data *= decay
        self.W.data.clamp_(-2, 2)


# =====================================================
# VISUALS
# =====================================================
def setup_plots():
    fig, (ax_s, ax_w) = plt.subplots(1, 2, figsize=(14, 6))
    fig.subplots_adjust(top=0.8)

    state_img = ax_s.imshow(
        np.zeros((GRID_H, GRID_W)),
        vmin=0, vmax=1, interpolation="nearest"
    )
    ax_s.set_title("Neuron Activations")
    ax_s.set_xlabel("Neuron X index")
    ax_s.set_ylabel("Neuron Y index")

    weight_img = ax_w.imshow(
        np.zeros((N_NEURONS, N_NEURONS)),
        cmap="bwr", vmin=-0.3, vmax=0.3, interpolation="nearest"
    )
    ax_w.set_title("Weight Matrix W")
    ax_w.set_xlabel("Pre-synaptic neuron index")
    ax_w.set_ylabel("Post-synaptic neuron index")
    plt.colorbar(weight_img, ax=ax_w)

    return fig, state_img, weight_img


# =====================================================
# MAIN LOOP
# =====================================================
def main():
    net = MultiSparkMemoryNet(N_NEURONS)
    net.reset()

    fig, s_img, w_img = setup_plots()

    for step in range(1, TOTAL_STEPS + 1):
        net.step()

        # display info
        fig.suptitle(
            f"Step {step}/{TOTAL_STEPS} — Sparks alive: {len(net.sparks)}",
            fontsize=12, x=0.01, ha='left'
        )

        s_img.set_data(net.s.view(GRID_H, GRID_W).cpu().numpy())
        w_img.set_data(net.W.data.cpu().numpy())

        plt.pause(0.001)

    plt.show()


if __name__ == "__main__":
    main()
