import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# ===================== CONFIG =====================
N_NEURONS = 500
GRID_H, GRID_W = 20, 25
assert GRID_H * GRID_W == N_NEURONS

CYCLE_LENGTH = 200        # steps per cycle
TOTAL_CYCLES = 10000      # 10k runs

FIRE_THRESHOLD = 0.8
ACTIVE_THRESHOLD = 0.2
SPARK_PROB = 2e-5         # very rare sparks
FORCED_LIFE = 5           # safe steps
LR = 0.01
NOISE_STD = 0.1

DEVICE = "cpu"


# ===================== NET =====================
class SparkNet(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.W = nn.Parameter(0.1 * torch.randn(n, n))
        self.register_buffer("s", torch.zeros(n))

        # spark state
        self.register_buffer("alive", torch.zeros(n, dtype=torch.bool))
        self.register_buffer("age", -torch.ones(n, dtype=torch.long))

        # path accumulator: sum of all states while spark is alive
        self.register_buffer("history_sum", torch.zeros(n, n))

    def reset_state(self):
        self.s = torch.rand(self.n)      # new random activations each cycle
        self.alive.zero_()
        self.age.fill_(-1)
        self.history_sum.zero_()

    @torch.no_grad()
    def step(self):
        # base dynamics
        noise = NOISE_STD * torch.randn_like(self.s)
        raw = self.W @ self.s + noise
        s_new = torch.sigmoid(raw)

        fired = s_new > FIRE_THRESHOLD

        # new sparks (from random chance, not firing)
        spawn = (torch.rand(self.n) < SPARK_PROB) & (~self.alive)
        if spawn.any():
            self.alive[spawn] = True
            self.age[spawn] = 0
            self.history_sum[spawn, :] = 0.0

        active = torch.where(self.alive)[0]
        if active.numel() > 0:
            ages = self.age[active]

            # forced life: keep at 1.0 for first FORCED_LIFE steps
            safe_idx = active[ages < FORCED_LIFE]
            if safe_idx.numel() > 0:
                s_new[safe_idx] = 1.0

            # accumulate path
            self.history_sum[active, :] += s_new.unsqueeze(0)

            # age++
            new_ages = ages + 1
            self.age[active] = new_ages

            # can die after safe period
            can_die = active[new_ages > FORCED_LIFE]
            if can_die.numel() > 0:
                dead_mask = s_new[can_die] < ACTIVE_THRESHOLD
                dead_idx = can_die[dead_mask]
                if dead_idx.numel() > 0:
                    # one-shot path rewrite at death (spark lived >= 6 steps)
                    self.W.data[dead_idx, :] += LR * self.history_sum[dead_idx, :]
                    self.W.data[dead_idx, :] = torch.clamp(
                        self.W.data[dead_idx, :], -2.0, 2.0
                    )
                    # clear spark state
                    self.alive[dead_idx] = False
                    self.age[dead_idx] = -1
                    self.history_sum[dead_idx, :] = 0.0

        self.s.copy_(s_new)
        return int(spawn.sum()), int(fired.sum()), int(self.alive.sum())


# ===================== VISUALS =====================
def setup_plots():
    fig, (ax_s, ax_w) = plt.subplots(1, 2, figsize=(12, 5))

    state_img = ax_s.imshow(
        np.zeros((GRID_H, GRID_W)), vmin=0, vmax=1, interpolation="nearest"
    )
    ax_s.set_xticks([]); ax_s.set_yticks([]); ax_s.set_title("Neuron activations")

    weight_img = ax_w.imshow(
        np.zeros((N_NEURONS, N_NEURONS)), cmap="bwr",
        vmin=-0.3, vmax=0.3, interpolation="nearest"
    )
    ax_w.set_xticks([]); ax_w.set_yticks([]); ax_w.set_title("Weight matrix W")
    plt.colorbar(weight_img, ax=ax_w)

    plt.tight_layout()
    return fig, state_img, weight_img


def update_plots(net, fig, s_img, w_img, cycle, step, stats):
    new_sparks, fired, alive = stats
    s_img.set_data(net.s.view(GRID_H, GRID_W).cpu().numpy())
    w_img.set_data(net.W.data.cpu().numpy())
    fig.suptitle(
        f"Cycle {cycle}/{TOTAL_CYCLES} — Step {step}/{CYCLE_LENGTH} — "
        f"new {new_sparks} — fired {fired} — alive sparks {alive}",
        fontsize=11,
    )
    plt.pause(0.001)


# ===================== MAIN =====================
def main():
    net = SparkNet(N_NEURONS)
    fig, s_img, w_img = setup_plots()

    for cycle in range(1, TOTAL_CYCLES + 1):
        net.reset_state()
        for step in range(1, CYCLE_LENGTH + 1):
            stats = net.step()
            if step % 200 == 0 or step == 1:
                update_plots(net, fig, s_img, w_img, cycle, step, stats)


if __name__ == "__main__":
    main()
    plt.show(block=False)
    plt.pause(0.1)
    plt.close("all")
