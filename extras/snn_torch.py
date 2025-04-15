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
# Training Version: Differentiable Surrogate Spike Function
# -------------------------
class SurrogateSpikeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold=1.0):
        ctx.threshold = threshold
        ctx.save_for_backward(input)
        return (input >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        surrogate_grad = ((input - ctx.threshold).abs() < 0.5).float()
        return grad_output * surrogate_grad, None

def surrogate_spike_fn(x, threshold=1.0):
    return SurrogateSpikeFunction.apply(x, threshold)

# -------------------------
# Define a LIF Layer for Training
# -------------------------
class LIFLayerTrain(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, threshold: float = 1.0, decay: float = 0.9):
        super().__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.threshold = threshold
        self.decay = decay

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # inputs shape: (batch, time_steps, input_size)
        batch_size, time_steps, _ = inputs.shape
        h = torch.zeros(batch_size, self.fc.out_features, device=inputs.device)
        spike_record = []
        for t in range(time_steps):
            x_t = inputs[:, t, :]
            h = self.decay * h + self.fc(x_t)
            spike = surrogate_spike_fn(h, self.threshold)
            h = h * (1.0 - spike)
            spike_record.append(spike)
        spikes = torch.stack(spike_record, dim=1)
        return spikes, h

# -------------------------
# Define the Training SNN Model (with two LIF layers)
# -------------------------
class SNNModelTrain(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, threshold: float = 1.0, decay: float = 0.9):
        super().__init__()
        self.lif1 = LIFLayerTrain(input_size, hidden_size, threshold, decay)
        self.lif2 = LIFLayerTrain(hidden_size, output_size, threshold, decay)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        spikes1, _ = self.lif1(inputs)
        spikes2, _ = self.lif2(spikes1)
        out = spikes2.sum(dim=1)  # Aggregate spikes over time
        return out


# -------------------------
# Inference Version: TorchScript-Compatible Spike Function
# -------------------------
@torch.jit.script
def spike_fn_ts(x: torch.Tensor, threshold: float = 1.0) -> torch.Tensor:
    return (x >= threshold).float()

# -------------------------
# Define a LIF Layer for Inference (Unrolled with Accumulation)
# -------------------------
class LIFLayerInfUnrolled(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, threshold: float = 1.0, decay: float = 0.9, time_steps: int = 5):
        """
        Unrolls the recurrence for a fixed number of time steps.
        Instead of stacking spike outputs, it accumulates (sums) them.
        """
        super().__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.threshold = threshold
        self.decay = decay
        self.time_steps = time_steps

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # inputs shape: (batch, time_steps, input_size)
        batch_size = inputs.shape[0]
        device = inputs.device
        h = torch.zeros(batch_size, self.fc.out_features, device=device)
        spike_sum = torch.zeros(batch_size, self.fc.out_features, device=device)
        for t in range(self.time_steps):
            h = self.decay * h + self.fc(inputs[:, t, :])
            s = spike_fn_ts(h, self.threshold)
            h = h * (1.0 - s)
            spike_sum = spike_sum + s
        return spike_sum, h

# -------------------------
# Define the Inference SNN Model
#
# For FHE compatibility, we use only one recurrent (unrolled) layer.
# We aggregate its spike outputs (via summation) to obtain a static representation,
# then apply a final linear layer.
# -------------------------
class SNNModelInf(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, threshold: float = 1.0, decay: float = 0.9, time_steps: int = 5):
        super().__init__()
        self.lif1 = LIFLayerInfUnrolled(input_size, hidden_size, threshold, decay, time_steps)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        spike_sum, _ = self.lif1(inputs)   # shape: (batch, hidden_size)
        out = self.fc2(spike_sum)           # shape: (batch, output_size)
        return (torch.sigmoid(out) >= 0.5).float()  # Converts to 0 or 1