import torch
import matplotlib.pyplot as plt
import numpy as np
from model import Chain

# load trained model
model = Chain(20)
model.load_state_dict(torch.load("chain.pt"))

steps = 20
history = []

# run model and record each step
s = torch.zeros(20)
for i in range(steps):
    s = torch.sigmoid(model.W @ s)

    # extra spikes from your rule
    extra = (torch.rand_like(s) < (s * 0.02)).float()
    s = torch.clamp(s + extra, 0, 1)

    history.append(s.detach().numpy())  # <-- FIXED

history = np.array(history).T   # shape: (neurons, steps)

# plot heatmap
plt.imshow(history, cmap="hot", aspect="auto")
plt.colorbar(label="Activation level")
plt.xlabel("Time step")
plt.ylabel("Neuron index")
plt.title("Neuron Activity Over Time")
plt.show()
