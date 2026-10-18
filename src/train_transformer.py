import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --------------------------------------------------
# Paths and setup
# --------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# Load prepared sequences
data = np.load(DATA_DIR / "sequences.npz")

X_train = torch.tensor(data["X_train"], dtype=torch.float32)
y_train = torch.tensor(data["y_train"], dtype=torch.float32).unsqueeze(1)

X_val = torch.tensor(data["X_val"], dtype=torch.float32)
y_val = torch.tensor(data["y_val"], dtype=torch.float32).unsqueeze(1)

X_test = torch.tensor(data["X_test"], dtype=torch.float32)
y_test = data["y_test"]

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=128,
    shuffle=True
)

val_loader = DataLoader(
    TensorDataset(X_val, y_val),
    batch_size=128,
    shuffle=False
)

# --------------------------------------------------
# Transformer model
# --------------------------------------------------
class EnergyTransformer(nn.Module):
    def __init__(
        self,
        input_features=7,
        d_model=64,
        n_heads=4,
        num_layers=2,
        dropout=0.1
    ):
        super().__init__()

        self.input_projection = nn.Linear(input_features, d_model)

        # Learnable positional embeddings for the 24 time steps
        self.position_embedding = nn.Parameter(
            torch.zeros(1, 24, d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.regression_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        # x shape: (batch, 24, 7)
        x = self.input_projection(x)
        x = x + self.position_embedding

        # Self-attention across the observed 24-hour history
        x = self.encoder(x)

        # Use the final time step's representation to predict next hour
        x = x[:, -1, :]
        return self.regression_head(x)


model = EnergyTransformer().to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

# --------------------------------------------------
# Training with validation-based early stopping
# --------------------------------------------------
EPOCHS = 30
PATIENCE = 5
best_val_loss = float("inf")
epochs_without_improvement = 0

best_model_path = MODEL_DIR / "energy_transformer.pth"

for epoch in range(EPOCHS):
    model.train()
    train_losses = []

    for xb, yb in train_loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        optimizer.zero_grad()
        predictions = model(xb)
        loss = criterion(predictions, yb)

        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

    # Validation
    model.eval()
    val_losses = []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            predictions = model(xb)
            val_losses.append(criterion(predictions, yb).item())

    train_loss = float(np.mean(train_losses))
    val_loss = float(np.mean(val_losses))

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train MSE: {train_loss:.4f} | "
        f"Validation MSE: {val_loss:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), best_model_path)
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print("Early stopping.")
        break

# --------------------------------------------------
# Evaluate on the held-out test set
# --------------------------------------------------
model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
model.eval()

with torch.no_grad():
    predictions = model(X_test.to(DEVICE)).cpu().numpy().flatten()

actual = y_test

def report_metrics(name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    # Avoid division by zero when calculating MAPE
    nonzero = np.abs(actual) > 1e-8
    mape = (
        np.mean(np.abs((actual[nonzero] - predicted[nonzero])
                       / actual[nonzero])) * 100
        if np.any(nonzero)
        else float("nan")
    )

    print(f"\n{name}")
    print(f"MAE:  {mae:.4f} kW")
    print(f"RMSE: {rmse:.4f} kW")
    print(f"MAPE: {mape:.2f}%")

report_metrics("Transformer test performance", actual, predictions)

# --------------------------------------------------
# Persistence baseline: predict next hour as latest hour
# --------------------------------------------------
scaler = joblib.load(DATA_DIR / "feature_scaler.pkl")

# Feature 0 is Global_active_power.
# Invert standardization to recover the latest observed power.
latest_power = (
    X_test[:, -1, 0].numpy() * scaler.scale_[0]
    + scaler.mean_[0]
)

report_metrics("Persistence baseline (last hour)", actual, latest_power)

# Save test predictions for plotting / later analysis
np.savez(
    DATA_DIR / "test_predictions.npz",
    actual=actual,
    transformer_predictions=predictions,
    persistence_predictions=latest_power
)

print("\nSaved model to:", best_model_path)
print("Saved predictions to:", DATA_DIR / "test_predictions.npz")