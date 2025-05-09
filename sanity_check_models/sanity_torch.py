import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

# Convert data to float32 and normalize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
X_eval_scaled = scaler.transform(X_eval).astype(np.float32)
y_train_tensor = y_train.astype(np.float32)
y_eval_tensor = y_eval.astype(np.float32)

# Convert to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train_scaled)
X_eval_tensor = torch.from_numpy(X_eval_scaled)
y_train_tensor = torch.from_numpy(y_train_tensor)
y_eval_tensor = torch.from_numpy(y_eval_tensor)

# Create DataLoaders
train_ds = TensorDataset(X_train_tensor, y_train_tensor)
eval_ds = TensorDataset(X_eval_tensor, y_eval_tensor)
train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)
eval_loader = DataLoader(eval_ds, batch_size=1024)

# Define the MLP model
class ListeningTimeModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

model = ListeningTimeModel(X_train.shape[1])

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 20
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb).squeeze()
        loss = criterion(preds, yb.squeeze())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1:02d} | Train Loss: {running_loss / len(train_loader):.4f}")