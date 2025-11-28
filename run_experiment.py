import torch
from model import Chain

# load trained model
model = Chain(20)
model.load_state_dict(torch.load("chain.pt"))

# run experiment with randomness ON
out = model(steps=15, p=0.02, train=False)

print("Final activations:", out)
print("Active neurons:", (out > 0.5).nonzero(as_tuple=True)[0].tolist())
