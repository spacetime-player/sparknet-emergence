import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# ============================================================
#                MODEL WITH LOCALIZED STDP
# ============================================================

class Chain(nn.Module):
    def __init__(self, n=500):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n, n) * 0.05)
        self.n = n

    def forward(self, s):
        s = torch.sigmoid(self.W @ s)
        return torch.clamp(s, 0, 1)

    # ---- LOCAL STDP: only update connections involved in spark path ----
    def local_stdp(self, pre, post, pre_mask, post_mask, eta=0.01, decay=0.0001):

        # outer product (post × preᵀ)
        Hebb = torch.outer(post, pre)

        # mask where BOTH pre and post were spark-driven
        M = torch.outer(post_mask, pre_mask)  # shape (n,n), 1 only on spark paths

        # apply STDP **only on masked connections**
        self.W.data += eta * Hebb * M

        # decay slows runaway growth
        self.W.data -= decay * self.W.data

        # clip to safe range
        self.W.data = torch.clamp(self.W.data, -1, 1)



# ============================================================
#                EXPERIMENT PARAMETERS
# ============================================================

model = Chain(500)
s = torch.zeros(model.n)

second_fire_p = 0.002          # spontaneous sparks
spark_threshold = 0.8          # what counts as "spark-driven"
steps = 1000

spark_count = 0
total_sparks = 0


# ============================================================
#                MATPLOTLIB SETUP
# ============================================================

fig, (ax_a, ax_w) = plt.subplots(1, 2, figsize=(14, 6))
plt.tight_layout()

bars = ax_a.bar(range(model.n), s)
ax_a.set_ylim(0, 1)
ax_a.set_title("Neuron activity")

heat = ax_w.imshow(model.W.detach().numpy(), cmap='seismic', vmin=-0.2, vmax=0.2)
ax_w.set_title("Weights (local STDP)")
plt.colorbar(heat, ax=ax_w)


# ============================================================
#                UPDATE FUNCTION
# ============================================================

step = 0

def update(frame):
    global s, step, spark_count, total_sparks

    step += 1

    pre = s.clone()

    # spontaneous second fire
    spark_pre_mask = (torch.rand(model.n) < second_fire_p).float()

    if spark_pre_mask.sum() > 0:
        s = torch.clamp(s + spark_pre_mask, 0, 1)
        spark_count += 1
        total_sparks += spark_pre_mask.sum().item()

    # save mask BEFORE forward pass
    pre_mask = spark_pre_mask.clone()

    # forward activation
    s = model(s)

    # which neurons fired strongly BECAUSE of spark?
    post_mask = (s > spark_threshold).float()

    # apply LOCAL STDP only on connections touched by spark
    if pre_mask.sum() > 0 and post_mask.sum() > 0:
        model.local_stdp(pre, s, pre_mask, post_mask)

    # title
    ax_a.set_title(
        f"Step {step}/{steps} — sparks {spark_count} (neurons {int(total_sparks)})"
    )

    # update bars
    for i, b in enumerate(bars):
        b.set_height(float(s[i]))

    # update heatmap
    heat.set_data(model.W.detach().numpy())

    # stop condition
    if step >= steps:
        ani.event_source.stop()

    return bars, heat


ani = animation.FuncAnimation(fig, update, interval=40, cache_frame_data=False)
plt.show()
