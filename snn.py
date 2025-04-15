import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.optim as optim  # type: ignore


# -------------------------
# TorchScript-Compatible Surrogate Spike Function
# -------------------------
@torch.jit.script
def surrogate_spike(mem: torch.Tensor, threshold: float, slope: float = 10.0) -> torch.Tensor:
    # Smooth approximation to a step function.
    return torch.sigmoid(slope * (mem - threshold))


class FraudSNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        time_steps: int,
        beta: float = 0.9,
        threshold: float = 1.0,
    ):
        """
        input_size: Number of features.
        hidden_size: Number of neurons in the hidden layer.
        time_steps: Number of time steps (manually unrolled, fixed to 10).
        """
        super().__init__()
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
        batch_size = x.size(0)

        # Initialize membrane potentials to persist across time steps
        mem1 = torch.zeros(batch_size, self.fc1.out_features, device=x.device)
        mem2 = torch.zeros(batch_size, 1, device=x.device)

        spk_sum = torch.zeros(batch_size, 1, device=x.device)  # Persistent spike accumulation

        for t in range(self.time_steps):  # Loop through time steps
            cur1 = self.fc1(x[:, t, :])  # Compute input current
            spk1, mem1 = self.lif1(cur1, mem1)  # LIF neuron with persistent membrane

            cur2 = self.fc2(spk1)  
            spk2, mem2 = self.lif2(cur2, mem2)  # Second LIF neuron

            spk_sum += spk2  # Accumulate spikes over time

        spk_mean = spk_sum / self.time_steps  # Compute mean spike rate
        out = self.sigmoid(spk_mean)  # Final activation

        return out

# -------------------------
# Modified Training Function to Record Loss Curves
# -------------------------

