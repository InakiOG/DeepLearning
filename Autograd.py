import numpy as np
import torch

x = torch.tensor(1.0)
y = torch.tensor(2.0)
w = torch.tensor(1.0, requires_grad=True)

# forward pass and compute the loss
y_hat = w * x
loss = (y_hat - y) ** 2
print(loss)

# backward pass
loss.backward()
print(w.grad)

# prevent gradient history
x = torch.randn(3, requires_grad=True)
print(x)
# option 1
x.requires_grad_(False)
print(x)
# option 2
y = x.detach()
print(x)
# option 3
with torch.no_grad():
    y = x + 2
    print(y)


# training example
weights = torch.ones(4, requires_grad=True)
for epoch in range(3):
    model_output = (weights * 3).sum()  # dummy computation
    model_output.backward()
    print(weights.grad)
    # weights.grad.zero_()

a = torch.ones(5)
print(a)
b = a.numpy()
print(type(b))
# Both point to the same memory location
a.add_(1)
print(a)
print(b)
# numpy to torch
a = np.ones(5)
print(a)
b = torch.from_numpy(a)
print(b)
# Both point to the same memory location
a += 1
print(a)
print(b)
# Calculate gradient
x = torch.ones(5, requires_grad=True)
print(x)
