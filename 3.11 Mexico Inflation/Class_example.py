import yfinance as yf
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# Load data
data = yf.download('INTC', start='2024-01-01', end='2025-04-07')
close_prices = data['Close'].values.reshape(-1, 1).astype(float)

# Normalize using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
close_prices = scaler.fit_transform(close_prices).flatten()

# Convert data into sequences


def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:i+seq_length])
        ys.append(data[i+seq_length])
    return np.array(xs), np.array(ys)


seq_length = 15  # Reduced for better stability
X, y = create_sequences(close_prices, seq_length)

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32).reshape(len(X), seq_length, 1)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# Print shapes for debugging
print(f"X shape: {X.shape}")  # Expected: (num_samples, seq_length, 1)
print(f"y shape: {y.shape}")  # Expected: (num_samples, 1)

# Create DataLoader for batching
batch_size = 16
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define LSTM Model with Xavier Initialization


class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)  # Xavier initialization
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Take last time step
        return out


# Model and training setup
input_size = 1
hidden_size = 128  # Increased for better feature extraction
output_size = 1
model = LSTMPredictor(input_size, hidden_size, output_size)

num_epochs = 3000
learning_rate = 0.0005  # Lower LR to prevent skipping minima
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=500, gamma=0.5)

# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    scheduler.step()  # Adjust learning rate

    # Print every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(
            f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(dataloader):.6f}')

# Make prediction
model.eval()
with torch.no_grad():
    inputs = X[-1].reshape(1, seq_length, 1)
    pred = model(inputs).item()
    pred = scaler.inverse_transform(np.array([[pred]]))[0, 0]  # Scale back
    print(f"\nPredicted next day price: {pred:.2f}")

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(np.arange(len(close_prices)), scaler.inverse_transform(
    close_prices.reshape(-1, 1)), label='Actual')
plt.axvline(x=len(close_prices) - 1, color='r',
            linestyle='--', label='Prediction Point')
plt.scatter(len(close_prices) - 1, pred, color='r',
            marker='o', s=100, label='Predicted')
plt.xlabel('Days')
plt.ylabel('Stock Price')
plt.legend()
plt.grid(True)
plt.title('Stock Prediction for INTC')
plt.show()
