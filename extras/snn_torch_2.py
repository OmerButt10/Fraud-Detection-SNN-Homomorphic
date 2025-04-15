import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# -------------------------
# Hyperparameters and Settings
# -------------------------
DATA_PATH = "hf://datasets/rohan-chandrashekar/Financial_Fraud_Detection/New_Dataset.csv"
SAMPLE_FRAC = 0.005 # Adjust as needed
TIME_STEPS = 2 # For this unrolled example, we assume TIME_STEPS=2 (as in your spike shape)
HIDDEN_SIZE = 32 # Hidden layer size for SNN
LEARNING_RATE = 1e-3
NUM_EPOCHS = 1
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
print(f"Using device: {device}")

# -------------------------
# Data Loading & Preprocessing
# -------------------------
print("Loading and preprocessing dataset...")
df = pd.read_csv(DATA_PATH)
print("Dataset loaded. Shape:", df.shape)

df_sample = df.sample(frac=SAMPLE_FRAC, random_state=42)
df_sample = df_sample.drop(columns=['nameOrig', 'nameDest'])
print("Dropped non-relevant columns.")

# Assume target column is 'isFraud'
y = df_sample['isFraud'].values
X = df_sample.drop(columns=['isFraud']).values
print("Separated features and target.")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Features standardized.")

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_scaled, y, test_size=0.3, stratify=y, random_state=42
)
print("Split data into train_val and test sets.")
print("Train_val shape:", X_train_val.shape, "Test shape:", X_test.shape)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=VALIDATION_SPLIT, stratify=y_train_val, random_state=42
)
print("Split train_val into train and validation sets.")
print("Train shape:", X_train.shape, "Validation shape:", X_val.shape)

# -------------------------
# Convert Data to Spike Trains (Rate Coding)
# -------------------------
def rate_code(values, max_rate=100, time_steps=TIME_STEPS):
    firing_rates = (values - values.min()) / (values.max() - values.min() + 1e-8) * max_rate
    spikes = np.random.rand(time_steps, len(values)) < (firing_rates / max_rate)
    return spikes.astype(np.float32)

print("Converting training data to spike trains...")
X_train_spikes = np.array([rate_code(row) for row in X_train])
print("Training data spike shape:", X_train_spikes.shape)
print("Converting validation data to spike trains...")
X_val_spikes = np.array([rate_code(row) for row in X_val])
print("Validation data spike shape:", X_val_spikes.shape)
print("Converting test data to spike trains...")
X_test_spikes = np.array([rate_code(row) for row in X_test])
print("Test data spike shape:", X_test_spikes.shape)

# -------------------------
# Prepare PyTorch Datasets and DataLoaders
# -------------------------
X_train_tensor = torch.tensor(X_train_spikes, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_val_tensor = torch.tensor(X_val_spikes, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test_spikes, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -------------------------
# TorchScript-Compatible Surrogate Spike Function
# -------------------------
@torch.jit.script
def surrogate_spike(mem: torch.Tensor, threshold: float, slope: float = 10.0) -> torch.Tensor:
    # Smooth approximation to a step function.
    return torch.sigmoid(slope * (mem - threshold))

# -------------------------
# Define the SNN Model for Fraud Detection (Manually Unrolled for TIME_STEPS=2)
# -------------------------
class FraudSNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, time_steps: int,
                 beta: float = 0.9, threshold: float = 1.0):
        """
        input_size: Number of features.
        hidden_size: Number of neurons in the hidden layer.
        time_steps: Number of time steps (assumed to be 2 for this unrolled version).
        """
        super().__init__()
        assert time_steps == 2, "This unrolled version requires time_steps==2."
        self.time_steps = time_steps

        # Fully connected layers.
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

        # Build custom LIF neurons using surrogate_spike.
        self.lif1 = self.build_lif(beta, threshold)
        self.lif2 = self.build_lif(beta, threshold)

        # Final sigmoid activation.
        self.sigmoid = nn.Sigmoid()

    def build_lif(self, beta, threshold):
        # Define a TorchScript-friendly LIF neuron using the surrogate spike.
        class LIFNeuronWithSTE(nn.Module):
            def __init__(self, beta, threshold):
                super().__init__()
                self.beta = beta
                self.threshold = threshold

            def forward(self, input_current: torch.Tensor, mem: torch.Tensor):
                mem = self.beta * mem + input_current
                spike = surrogate_spike(mem, self.threshold)
                mem = mem - spike * self.threshold
                return spike, mem
        return LIFNeuronWithSTE(beta, threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, 2, input_size) since time_steps==2
        batch_size = x.size(0)
        # Initialize membrane potentials.
        mem1 = torch.zeros(batch_size, self.fc1.out_features, device=x.device)
        mem2 = torch.zeros(batch_size, 1, device=x.device)
        
        # Time step 0.
        cur1_0 = self.fc1(x[:, 0, :])
        spk1_0, mem1 = self.lif1(cur1_0, mem1)
        cur2_0 = self.fc2(spk1_0)
        spk2_0, mem2 = self.lif2(cur2_0, mem2)
        
        # Time step 1.
        cur1_1 = self.fc1(x[:, 1, :])
        spk1_1, mem1 = self.lif1(cur1_1, mem1)
        cur2_1 = self.fc2(spk1_1)
        spk2_1, mem2 = self.lif2(cur2_1, mem2)
        
        spk_sum = spk2_0 + spk2_1
        spk_mean = spk_sum / 2.0
        out = self.sigmoid(spk_mean)
        return out

# -------------------------
# Training Function
# -------------------------
def train_model(model, train_loader, val_loader, num_epochs=10, lr=1e-3, device="cpu"):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                outputs = model(X_val_batch)
                loss = criterion(outputs, y_val_batch)
                val_loss += loss.item() * X_val_batch.size(0)
                predictions = (outputs > 0.5).float()
                correct += (predictions == y_val_batch).sum().item()
        val_loss /= len(val_loader.dataset)
        accuracy = correct / len(val_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {accuracy:.4f}")
    
    return model

# -------------------------
# Main Training Section
# -------------------------
if __name__ == "__main__":
    input_size = X_train_tensor.size(2) # Number of features (should be 11 in your case)
    hidden_size = HIDDEN_SIZE
    time_steps = TIME_STEPS # Must be 2 for this unrolled version.

    model = FraudSNN(input_size, hidden_size, time_steps, beta=0.9, threshold=1.0)
    print("Starting training...")
    model = train_model(model, train_loader, val_loader, num_epochs=NUM_EPOCHS, lr=LEARNING_RATE, device=device)

    # Evaluate on test set.
    model.eval()
    test_preds = []
    test_targets = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            test_preds.append(outputs.cpu())
            test_targets.append(y_batch.cpu())
    test_preds = torch.cat(test_preds)
    test_targets = torch.cat(test_targets)
    test_acc = (test_preds > 0.5).float().eq(test_targets).float().mean().item()
    print(f"Test Accuracy: {test_acc:.4f}")

    # --- Save the trained model as TorchScript ---
    scripted_model = torch.jit.script(model)
    scripted_model.save("trained_fraud_snn_model.pt")
    print("Trained TorchScript model saved as 'trained_fraud_snn_model.pt'")

    # --- Compile the TorchScript model to FHE using Concrete ML ---
    from concrete.ml.torch.compile import compile_torch_model

    # Use a small calibration dataset (first 2 training samples from rate-coded data).
    calibration_dataset = X_train_tensor[:2]
    compiled_model = compile_torch_model(
        scripted_model,
        calibration_dataset,
        n_bits=6,
        rounding_threshold_bits={"n_bits": 6, "method": "approximate"}
    )
    # compiled_model.save("fhe_trained_fraud_snn_model.zip")
    print("FHE compiled model saved as 'fhe_trained_fraud_snn_model.zip'")

    # --- Test the FHE compiled model ---
    x_test = X_test_tensor.numpy()
    y_fhe_pred = compiled_model.forward(x_test, fhe="simulate")
    print("FHE simulated inference output:")
    print(y_fhe_pred)
    gc.collect()