import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

# =====================================================
# -------------------- PARAMETERS ----------------------
# =====================================================
N = 500                    # neurons
STEPS = 100000             # total system updates
LR_REINFORCE = 0.05       # strengthen visited edges
DECAY = 0.0005            # global slow decay
SPARK_JUMP_NOISE = 0.001  # random exploration
GRID_W = 25               # 25×20 = 500 grid for visualization

# =====================================================
# ------------------- INITIAL STATE --------------------
# =====================================================
W = torch.randn(N, N) * 0.01
act = torch.zeros(N)

# Two independently moving sparks
spark_pos = [random.randint(0, N-1),
             random.randint(0, N-1)]

fig, (axA, axW) = plt.subplots(1, 2, figsize=(14, 6))

# =====================================================
# ----------------- VISUAL SETUP -----------------------
# =====================================================
img_act = axA.imshow(act.reshape(20, 25), cmap="viridis", vmin=0, vmax=1)
axA.set_title("Neuron Activations (2 Sparks)")
axA.set_xlabel("Neuron X index")
axA.set_ylabel("Neuron Y index")

img_w = axW.imshow(W, cmap="seismic", vmin=-0.3, vmax=0.3)
axW.set_title("Weight Matrix W")
axW.set_xlabel("Pre-synaptic neuron")
axW.set_ylabel("Post-synaptic neuron")
plt.colorbar(img_w, ax=axW)

# =====================================================
# ----------- FUNCTION: SPARK LOCAL MOVEMENT ----------
# =====================================================
def pick_next(n):
    """Move spark based mostly on weights from current neuron."""
    weights = W[n].clone()

    # soften weights into probabilities
    weights = torch.relu(weights) + SPARK_JUMP_NOISE
    probs = weights / weights.sum()

    # random weighted jump
    next_n = torch.multinomial(probs, 1).item()
    return next_n

# =====================================================
# -------------------- ANIMATION -----------------------
# =====================================================
step = 0
def update(frame):
    global act, W, spark_pos, step
    step += 1

    # baseline activation slowly fades
    act *= 0.97

    # process each spark independently
    for i in range(len(spark_pos)):
        pos = spark_pos[i]
        nxt = pick_next(pos)

        # reinforce specific used connection
        W[pos, nxt] += LR_REINFORCE

        # save new position
        spark_pos[i] = nxt

        # highlight spark in activation map
        act[nxt] = 1.0

    # global slow decay
    W *= (1.0 - DECAY)

    # clip weights to sane range
    W.clamp_(-0.3, 0.3)

    # update visuals
    img_act.set_data(act.reshape(20, 25))
    img_w.set_data(W)

    axA.set_title(f"Neuron Activations (Sparks at {spark_pos[0]}, {spark_pos[1]})")
    axW.set_title(f"Weight Matrix W  —  Step {step}/{STEPS}")

    if step >= STEPS:
        ani.event_source.stop()

    return img_act, img_w


ani = animation.FuncAnimation(fig, update, interval=30, blit=False)
plt.show()
