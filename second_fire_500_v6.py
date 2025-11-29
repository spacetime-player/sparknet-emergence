import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ============================================================
#                  MODEL WITH LOCAL STDP
# ============================================================

class Chain(nn.Module):
    def __init__(self, n=250):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n, n) * 0.30)   # stronger weights
        self.n = n

    def forward(self, s):
        x = self.W @ s
        x = torch.relu(x)
        x = torch.clamp(x, 0, 1)
        x = x * 0.95  # leak (prevents explosion)
        return x

    def local_stdp(self, pre, post, pre_mask, post_mask, eta=0.02, decay=0.0005):
        Hebb = torch.outer(post, pre)
        mask = torch.outer(post_mask, pre_mask)
        self.W.data += eta * Hebb * mask
        self.W.data -= decay * self.W.data
        self.W.data = torch.clamp(self.W.data, -1, 1)


# ============================================================
#                    EXPERIMENT SETTINGS
# ============================================================

model = Chain(250)
s = torch.zeros(model.n)

second_fire_p = 0.002
spark_threshold = 0.3      # easier propagation → more STDP
steps = 10000

spark_count = 0
total_sparks = 0

survival_counter = 0
forced_survival_steps = 2   # spark must survive 2 steps before real influence


# ============================================================
#                    VISUAL SETUP
# ============================================================

fig, (ax_a, ax_w) = plt.subplots(1, 2, figsize=(14, 6))
plt.tight_layout()

bars = ax_a.bar(range(model.n), s)
ax_a.set_ylim(0, 1)

heat = ax_w.imshow(model.W.detach().numpy(), cmap='seismic',
                   vmin=-0.3, vmax=0.3)
plt.colorbar(heat, ax=ax_w)


# ============================================================
#                    ANIMATION UPDATE
# ============================================================

step = 0

def update(frame):
    global s, step, spark_count, total_sparks, survival_counter
    step += 1

    pre = s.clone()

    # spontaneous spark
    spark_pre_mask = (torch.rand(model.n) < second_fire_p).float()

    if spark_pre_mask.sum() > 0:
        s = torch.clamp(s + spark_pre_mask, 0, 1)
        spark_count += 1
        total_sparks += spark_pre_mask.sum().item()
        survival_counter = 1   # spark starts

    pre_mask = spark_pre_mask.clone()

    # forward
    s = model(s)

    # strong activations
    post_mask = (s > spark_threshold).float()
    propagated = (post_mask.sum() > 0)

    # survival logic
    if propagated and survival_counter > 0:
        survival_counter += 1
    elif survival_counter > 0 and not propagated:
        survival_counter = 0

    # only allow STDP if spark survived more than forced steps
    if survival_counter > forced_survival_steps:
        model.local_stdp(pre, s, pre_mask, post_mask)

    # update visuals
    for i, b in enumerate(bars):
        b.set_height(float(s[i]))

    heat.set_data(model.W.detach().numpy())

    ax_a.set_title(
        f"Step {step}/{steps} — sparks {spark_count} — neurons {int(total_sparks)} — alive {survival_counter}"
    )

    if step >= steps:
        ani.event_source.stop()

    return bars, heat


ani = animation.FuncAnimation(fig, update, interval=40, cache_frame_data=False)
plt.show()
