import torch
import torch.nn as nn

X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

n_samples, n_feature = X.shape

input_size = n_feature
output_size = n_feature

print(n_samples, n_feature)  # 4 1


class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)


model = LinearRegression(input_size, output_size)

X_test = torch.tensor([5], dtype=torch.float32)
print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

# Training
learning_rate = 0.051
n_iters = 100
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_iters):
    y_pred = model(X)
    l = loss(Y, y_pred)
    l.backward()  # dl/dw
    optimizer.step()
    optimizer.zero_grad()

    [w, b] = model.parameters()
    print(f'epoch {epoch + 1}: w={w[0][0].item(): .3f}, loss={l: .8f}')

print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')
