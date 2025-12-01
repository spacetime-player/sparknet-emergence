import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# ===================== CONFIG =====================
N_NEURONS = 500
GRID_H, GRID_W = 20, 25
assert GRID_H * GRID_W == N_NEURONS

CYCLE_LENGTH = 200
TOTAL_CYCLES = 10000

FIRE_THRESHOLD = 0.8
ACTIVE_THRESHOLD = 0.2
SPARK_PROB = 2e-5
FORCED_LIFE = 5
LR = 0.01
NOISE_STD = 0.1


# ===================== NETWORK =====================
class SparkNet(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.W = nn.Parameter(0.1 * torch.randn(n, n))

        self.register_buffer("s", torch.zeros(n))
        self.register_buffer("alive", torch.zeros(n, dtype=torch.bool))
        self.register_buffer("age", -torch.ones(n, dtype=torch.long))
        self.register_buffer("history_sum", torch.zeros(n, n))

    def reset_state(self):
        self.s = torch.rand(self.n)
        self.alive.zero_()
        self.age.fill_(-1)
        self.history_sum.zero_()

    @torch.no_grad()
    def step(self):
        noise = NOISE_STD * torch.randn_like(self.s)
        raw = self.W @ self.s + noise
        s_new = torch.sigmoid(raw)

        fired = s_new > FIRE_THRESHOLD

        spawn = (torch.rand(self.n) < SPARK_PROB) & (~self.alive)
        if spawn.any():
            self.alive[spawn] = True
            self.age[spawn] = 0
            self.history_sum[spawn, :] = 0

        active = torch.where(self.alive)[0]
        if active.numel() > 0:
            ages = self.age[active]

            safe_idx = active[ages < FORCED_LIFE]
            if safe_idx.numel() > 0:
                s_new[safe_idx] = 1.0

            self.history_sum[active, :] += s_new.unsqueeze(0)

            new_ages = ages + 1
            self.age[active] = new_ages

            can_die = active[new_ages > FORCED_LIFE]
            if can_die.numel() > 0:
                dead_mask = s_new[can_die] < ACTIVE_THRESHOLD
                dead_idx = can_die[dead_mask]
                if dead_idx.numel() > 0:
                    self.W.data[dead_idx, :] += LR * self.history_sum[dead_idx, :]
                    self.W.data[dead_idx, :] = torch.clamp(
                        self.W.data[dead_idx, :], -2, 2
                    )
                    self.alive[dead_idx] = False
                    self.age[dead_idx] = -1
                    self.history_sum[dead_idx, :] = 0

        self.s.copy_(s_new)
        return int(spawn.sum()), int(fired.sum()), int(self.alive.sum())


# ===================== VISUALS =====================
def setup_plots():
    fig, (ax_s, ax_w) = plt.subplots(1, 2, figsize=(12, 6))

    fig.subplots_adjust(top=0.78)

    state_img = ax_s.imshow(
        np.zeros((GRID_H, GRID_W)), vmin=0, vmax=1, interpolation="nearest"
    )
    ax_s.set_xticks([])
    ax_s.set_yticks([])
    ax_s.set_title("Neuron Activations")
    ax_s.set_xlabel("Neuron X index (25)")
    ax_s.set_ylabel("Neuron Y index (20)")

    weight_img = ax_w.imshow(
        np.zeros((N_NEURONS, N_NEURONS)),
        cmap="bwr", vmin=-0.3, vmax=0.3, interpolation="nearest"
    )
    ax_w.set_xticks([])
    ax_w.set_yticks([])
    ax_w.set_title("Weight Matrix W")
    ax_w.set_xlabel("Pre-synaptic neuron index")
    ax_w.set_ylabel("Post-synaptic neuron index")

    plt.colorbar(weight_img, ax=ax_w)
    plt.tight_layout()
    return fig, state_img, weight_img


# HUD updater
def update_hud(fig, cycle, step, stats):
    new_sparks, fired, alive = stats
    fig.suptitle(
        f"Cycle: {cycle}/{TOTAL_CYCLES}    "
        f"Step: {step}/{CYCLE_LENGTH}    "
        f"New: {new_sparks}    Fired: {fired}    Alive: {alive}",
        fontsize=12,
        x=0.02, ha='left'
    )


# ===================== MAIN =====================
def main():
    net = SparkNet(N_NEURONS)
    fig, s_img, w_img = setup_plots()

    for cycle in range(1, TOTAL_CYCLES + 1):
        net.reset_state()

        for step in range(1, CYCLE_LENGTH + 1):
            stats = net.step()

            update_hud(fig, cycle, step, stats)

            # update images every step
            s_img.set_data(net.s.view(GRID_H, GRID_W).cpu().numpy())
            w_img.set_data(net.W.data.cpu().numpy())

            plt.pause(0.001)


if __name__ == "__main__":
    main()
    plt.show(block=False)
    plt.pause(0.1)
    plt.close("all")
