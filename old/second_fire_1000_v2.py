import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ============================================================
#                MODEL WITH SIMPLE STDP LEARNING
# ============================================================

class Chain(nn.Module):
    def __init__(self, n=500):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n, n) * 0.05)   # weights
        self.n = n

    def forward(self, s):
        s = torch.sigmoid(self.W @ s)
        return torch.clamp(s, 0, 1)

    # ---- STDP: ΔW = eta * (post * preᵀ) - decay ----
    def stdp(self, pre, post, eta=0.001, decay=0.0001):
        Hebb = torch.outer(post, pre)
        self.W.data += eta * Hebb
        self.W.data -= decay * self.W.data


# ============================================================
#                EXPERIMENT PARAMETERS
# ============================================================

model = Chain(500)
s = torch.zeros(model.n)

second_fire_p = 0.002      # rare spontaneous spark
spark_threshold = 0.9      # anything above = spark detected
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
ax_w.set_title("Weights (STDP live)")
plt.colorbar(heat, ax=ax_w)


# ============================================================
#                UPDATE FUNCTION (ANIMATION)
# ============================================================

step = 0

def update(frame):
    global s, step, spark_count, total_sparks

    step += 1

    pre = s.clone()

    # spontaneous double fire
    spark_mask = (torch.rand(model.n) < second_fire_p).float()
    if spark_mask.sum() > 0:
        s = s + spark_mask
        s = torch.clamp(s, 0, 1)
        spark_count += 1
        total_sparks += spark_mask.sum().item()

    # forward update
    s = model(s)

    # STDP update (live learning)
    model.stdp(pre, s)

    # visualize sparks
    title = f"Step {step}/{steps} — sparks {spark_count} (total {int(total_sparks)})"
    ax_a.set_title(title)

    # update bar graph
    for i, b in enumerate(bars):
        b.set_height(float(s[i]))

    # update heatmap
    heat.set_data(model.W.detach().numpy())

    # stop
    if step >= steps:
        ani.event_source.stop()

    return bars, heat


ani = animation.FuncAnimation(fig, update, interval=40, cache_frame_data=False)

plt.show()
