import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from model import Chain
import threading
import sys

# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------
N = 500
P_REFIRE = 0.01
MAX_STEPS = 1000
# ----------------------------------------------------

# Model and state
model = Chain(N)
s = torch.zeros(N)
step_counter = 0
poke_flag = False
total_sparks = 0

# ----------------------------------------------------
# LISTEN FOR "p"
# ----------------------------------------------------
def listen_console():
    global poke_flag
    while True:
        if sys.stdin.readline().strip() == "p":
            poke_flag = True

threading.Thread(target=listen_console, daemon=True).start()

# ----------------------------------------------------
# FIGURE SETUP (LEFT = activation, RIGHT = heatmap)
# ----------------------------------------------------
fig, (ax_act, ax_w) = plt.subplots(1, 2, figsize=(14, 4))

# Activation bars
bars = ax_act.bar(range(N), s.numpy())
ax_act.set_ylim(0, 1)
ax_act.set_title("Neuron activity (press 'p' to poke)")

# Weight heatmap
im = ax_w.imshow(model.W.detach().numpy(), cmap="bwr", vmin=-0.3, vmax=0.3)
ax_w.set_title("Weight matrix")

# ----------------------------------------------------
# UPDATE LOOP
# ----------------------------------------------------
def update(frame):
    global s, poke_flag, step_counter, total_sparks

    # Stop updating after MAX_STEPS, but do NOT close window
    if step_counter >= MAX_STEPS:
        return bars

    # ----- 1. recurrent update -----
    s = torch.sigmoid(model.W @ s)

    # ----- 2. rare second-fire spark -----
    extra = (torch.rand_like(s) < (s * P_REFIRE)).float()
    sparks = int(extra.sum().item())
    total_sparks += sparks
    s = torch.clamp(s + extra, 0, 1)

    # ----- 3. manual poke -----
    if poke_flag:
        idx = torch.randint(0, N, (1,)).item()
        s[idx] = 1.0
        print(f"POKE neuron {idx}")
        poke_flag = False

    # ----- 4. update activation bars -----
    s_np = s.detach().numpy()
    for i, b in enumerate(bars):
        b.set_height(s_np[i])

    # ----- 5. update heatmap -----
    im.set_data(model.W.detach().numpy())

    # ----- 6. title -----
    ax_act.set_title(
        f"Neuron activity — step {step_counter}/{MAX_STEPS} — sparks {sparks} (total {total_sparks})"
    )

    step_counter += 1
    return bars

ani = animation.FuncAnimation(fig, update, interval=40)
plt.show()
