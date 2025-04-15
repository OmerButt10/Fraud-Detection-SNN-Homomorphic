import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from typing import Tuple
from concrete.ml.torch.compile import compile_torch_model


# Use CPU for FHE compilation
device = torch.device("cpu")
print(f"Using device: {device}")


# -------------------------
# Define a Leaky (LIF) Neuron with Surrogate Gradient
# -------------------------
class TSLeakySurrogate(nn.Module):
    def __init__(self, beta: float, slope: float = 10.0):
        """
        beta: decay factor
        slope: surrogate slope for sigmoid
        """
        super(TSLeakySurrogate, self).__init__()
        self.beta = beta
        self.threshold = 1.0  # fixed threshold
        self.slope = slope
        self.register_buffer("mem", torch.zeros(1))  # Ensure proper initialization

    def reset(self, x: torch.Tensor):
        self.mem = torch.zeros_like(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Update membrane potential
        self.mem = self.beta * self.mem + x
        # Surrogate spike generation: use a steep sigmoid as a differentiable approximation
        spk = torch.sigmoid(self.slope * (self.mem - self.threshold))
        # Binarize for reset purposes (non-differentiable reset is acceptable)
        spk = torch.sigmoid(self.slope * (self.mem - self.threshold))
        self.mem = self.mem * (1.0 - spk)
        return spk  # return soft spike output

# -------------------------
# Define the SNN Model Using the Custom TSLeakySurrogate Neurons
# -------------------------
class SNNNetTS(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, beta: float, time_steps: int):
        super(SNNNetTS, self).__init__()

        self.time_steps = time_steps

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = TSLeakySurrogate(beta)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.lif2 = TSLeakySurrogate(beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, TIME_STEPS, input_size)
        batch_size = x.size(0)
        # Preallocate output tensor: (batch, TIME_STEPS, output_size)
        out_tensor = torch.empty(batch_size, self.time_steps, self.fc2.out_features, device=x.device)
        # Reset internal states
        self.lif1.reset(torch.zeros(batch_size, self.fc1.out_features, device=x.device))
        self.lif2.reset(torch.zeros(batch_size, self.fc2.out_features, device=x.device))

        # Iterate over time steps
        for t in range(self.time_steps):  # Removed torch.jit.unroll
            cur = x[:, t, :]          # (batch, input_size)
            cur1 = self.fc1(cur)       # (batch, hidden_size)
            spk1 = self.lif1(cur1)     # (batch, hidden_size)
            cur2 = self.fc2(spk1)      # (batch, output_size)
            spk2 = self.lif2(cur2)     # (batch, output_size)
            out_tensor[:, t, :] = spk2
        return out_tensor

# -------------------------
# Define a Wrapper for Aggregating the Spike Trains
# -------------------------
class SNNClassifier(nn.Module):
    def __init__(self, snn_model: nn.Module):
        super().__init__()
        self.snn_model = snn_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spk_rec = self.snn_model(x)   # (batch, TIME_STEPS, output_size)
        spike_count = spk_rec.sum(dim=1)  # (batch, output_size)
        return spike_count