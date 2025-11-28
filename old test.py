import torch
import matplotlib.pyplot as plt
import numpy as np
import msvcrt              # lets us read keys from the console (for 'p')
from model import Chain    # our neuron network

# ---- basic setup ----
N = 60                     # how many neurons we have
model = Chain(N)           # make one Chain model with N neurons
s = torch.zeros(N)         # current activation of each neuron (all start at 0)

step = 0                   # cycle counter (how many updates we did)
eta = 0.01                 # learning rate for weight updates

# ---- create window and two plots ----
plt.ion()                  # interactive mode, so the plot can update
fig, (ax_act, ax_W) = plt.subplots(1, 2, figsize=(10, 4))

# left plot: bars showing activation of each neuron
bars = ax_act.bar(range(N), s.numpy())
ax_act.set_ylim(0, 1)
ax_act.set_xlabel("Neuron index")
ax_act.set_ylabel("Activation")
ax_act.set_title("Live neuron activity  (press 'p' to poke)")

# right plot: image of weight matrix (who connects to who)
imW = ax_W.imshow(model.W.detach().numpy(), aspect="auto", cmap="bwr")
ax_W.set_title("Weight matrix (structure)")
ax_W.set_xlabel("From neuron")
ax_W.set_ylabel("To neuron")
fig.colorbar(imW, ax=ax_W, shrink=0.8, label="Weight value")

# ---- main loop: runs until you close the window ----
while plt.fignum_exists(fig.number):
    step += 1  # increase cycle counter by 1

    # -------- neuron dynamics (how activity changes) --------
    # basic update: new activity = sigmoid(W * old_activity)
    s = torch.sigmoid(model.W @ s)

    # double-firing: active neurons have extra chance to spike again
    extra = (torch.rand_like(s) < (s * 0.15)).float()
    s = torch.clamp(s + extra, 0, 1)  # keep values between 0 and 1

    # background noise: random small spikes anywhere
    noise = (torch.rand_like(s) < 0.01).float()
    s = torch.clamp(s + noise, 0, 1)

    # decay: everything slowly calms down over time
    s *= 0.98

    # -------- simple online learning rule (weights change) --------
    # “neurons that fire together, wire together”
    with torch.no_grad():                 # do not track gradients here
        outer = torch.outer(s, s)         # pairwise product of activations
        # strengthen connections where both are active, and slowly decay weights
        model.W += eta * (outer - 0.01 * model.W)

    # -------- user input: press 'p' in console to poke --------
    if msvcrt.kbhit():                    # check if a key was pressed
        ch = msvcrt.getch()               # read the key
        if ch in (b"p", b"P"):            # if it was 'p' or 'P'
            idx = torch.randint(0, N, (1,))  # pick a random neuron index
            s[idx] = 1.0                     # set that neuron to full activation
            print(f"step {step}: POKE at neuron {idx.item()}")

    # -------- update left plot (activations) --------
    for i, b in enumerate(bars):          # go through each bar
        b.set_height(float(s[i]))         # set bar height to neuron activation

    # show step number in the title so we see the cycle counter
    ax_act.set_title(f"Live neuron activity  (step {step}, press 'p' to poke)")

    # -------- update right plot (weights) --------
    imW.set_data(model.W.detach().numpy())

    # redraw everything with a small pause (controls speed)
    plt.pause(0.03)

# when window is closed, turn off interactive mode and finish
plt.ioff()
plt.show()
