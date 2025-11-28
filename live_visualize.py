import torch
import matplotlib.pyplot as plt
import numpy as np
import msvcrt  # console key input on Windows
from model import Chain

N = 60                 # number of neurons (keep modest so W is visible)
model = Chain(N)
s = torch.zeros(N)

# structure: weight matrix from THIS model
W = model.W.detach().numpy()

plt.ion()
fig, (ax_act, ax_W) = plt.subplots(1, 2, figsize=(10, 4))

# --- left: live activations ---
bars = ax_act.bar(range(N), s.numpy())
ax_act.set_ylim(0, 1)
ax_act.set_xlabel("Neuron index")
ax_act.set_ylabel("Activation")
ax_act.set_title("Live neuron activity  (press 'p' to poke)")

# --- right: structure (weight matrix) ---
imW = ax_W.imshow(W, aspect="auto", cmap="bwr")
ax_W.set_title("Weight matrix (structure)")
ax_W.set_xlabel("From neuron")
ax_W.set_ylabel("To neuron")
fig.colorbar(imW, ax=ax_W, shrink=0.8, label="Weight value")

# --- main loop: runs until you close the window ---
while plt.fignum_exists(fig.number):
    # update dynamics
    s = torch.sigmoid(model.W @ s)

    # double firing (moderate)
    extra = (torch.rand_like(s) < (s * 0.15)).float()
    s = torch.clamp(s + extra, 0, 1)

    # small noise
    noise = (torch.rand_like(s) < 0.01).float()
    s = torch.clamp(s + noise, 0, 1)

    # slow decay
    s *= 0.98

    # console key: 'p' to poke a random neuron
    if msvcrt.kbhit():
        ch = msvcrt.getch()
        if ch in (b"p", b"P"):
            idx = torch.randint(0, N, (1,))
            s[idx] = 1.0
            print(f"POKE at neuron {idx.item()}")

    # update bars
    for i, b in enumerate(bars):
        b.set_height(float(s[i]))

    plt.pause(0.03)

plt.ioff()
plt.show()
