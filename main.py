import base64
import random
import time  # type: ignore
import json  # type: ignore

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import torch  # type: ignore
from torch.utils.data import TensorDataset, DataLoader  # type: ignore

from dataset import load_dataset, rate_code
from snn import FraudSNN
# from train import train_snn_model
from snn_hfe import (
    get_quantize_module,
    get_quantize_input,
    get_encrypted_input,
    get_encrypted_output,
    get_decrypted_output,
    get_dequantize_output,
    get_normal_output
    )

# -------------------------
# Load the configuration file
# -------------------------
with open('config.json', 'r') as file:
    Config = json.load(file)

print("\n")
for k, v in Config.items():
    print(k, ": ", v)

# -------------------------
# Use CPU for FHE compilation
# -------------------------
device = torch.device("cpu")

# -------------------------
# Load the dataset and prepare PyTorch Datasets and DataLoaders
# -------------------------
X_train, X_test, X_val, y_val, y_train, y_test = load_dataset(Config["dataset_path"], Config["dataset_frac"], Config["validation_split"])
# Convert each sample into a spike train. The output shape will be [samples, time_steps, features]
X_train_spikes = np.array([rate_code(row) for row in X_train])
X_val_spikes  = np.array([rate_code(row) for row in X_val])
X_test_spikes  = np.array([rate_code(row) for row in X_test])

# Prepare PyTorch Datasets and DataLoaders
X_train_tensor = torch.tensor(X_train_spikes, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_val_tensor = torch.tensor(X_val_spikes, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test_spikes, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# # -------------------------
# # Loading and training snn model and saving trained model as TorchScript.
# # -------------------------
input_size = X_train_tensor.size(2)  # Number of features (should be 11 in your case)
hidden_size = Config["hidden_size"]
time_steps = Config["time_steps"]  # Must be 2 for this unrolled version.

snn_classifier = FraudSNN(input_size, hidden_size, time_steps, beta=0.9, threshold=1.0)

# load the models if it exists
try:
    snn_classifier.load_state_dict(
        torch.load("saved_models/snn_model.pth", map_location=torch.device(device))
    )
    print("\nModel loaded from saved_models/snn_model.pth")
except FileNotFoundError:
    print(
        "\nModel not found at saved_models/snn_model.pth. Please train the model first."
    )

try:
    scripted_model = torch.jit.load(
        "saved_models/snn_scripted.pt", map_location=torch.device(device)
    )
    print("\nModel loaded from saved_models/snn_scripted.pt")
except FileNotFoundError:
    print(
        "\nModel not found at saved_models/snn_scripted.pt. Please train the model first."
    )

# set the model to evaluation mode
snn_classifier.eval()
scripted_model.eval()

# -------------------------
# Running test example thorugh the pipeline.
# -------------------------

random_index = random.randint(0, X_test_spikes.shape[0]-1)

# Get single input to run through the model.
start = time.time()
torch_input = torch.tensor(X_test_spikes[random_index:random_index+1, :, :], dtype=torch.float32)
end = time.time()
print(f"\n({end - start:.4f} sec(s)) Input shape:", torch_input.shape)

# Compile the model for FHE inference.
quantized_module, execution_time = get_quantize_module(snn_classifier, torch_input, method="approximate", bits=8)
print(f"\n({execution_time} sec(s)) Model COMPILED for FHE inference successfully.")

# Quantize the input data.
q_input, execution_time = get_quantize_input(quantized_module, torch_input)
print(f"\n({execution_time} sec(s)) Input QUANTIZED successfully.")
print(f"Quantized Input: {q_input}")

# Encrypt the input
q_input_enc, execution_time = get_encrypted_input(quantized_module, q_input)
# Getting homomorphic encryption of the input
input_buffer = q_input_enc.serialize()
encrypted_input_str = base64.b64encode(input_buffer).decode('utf-8')
print(f"\n({execution_time} sec(s)) Input ENCRYPTED successfully.")
print(f"Encrypted Input: {encrypted_input_str}")

# Execute the linear product in FHE
q_y_enc, execution_time = get_encrypted_output(quantized_module, q_input_enc)
# Getting homomorphic encryption of the output
output_buffer = q_y_enc.serialize()
encrypted_output_str = base64.b64encode(output_buffer).decode('utf-8')
print(f"\n({execution_time} sec(s)) FHE MODEL EXECUTED sucessfully.")
print(f"Encrypted Output: {encrypted_output_str}")

# Decrypt the result (integer)
q_y, execution_time = get_decrypted_output(quantized_module, q_y_enc)
print(f"\n({execution_time} sec(s)) Output DECRYPTED successfully.")
print(f"Decrypted Output: {q_y}")

# De-quantize and post-process the result
y0, execution_time = get_dequantize_output(quantized_module, q_y)
print(f"\n({execution_time} sec(s)) Output DE-QUANTIZED successfully .")
print("De-quantized Output:", y0)

# Run the model on the original data for comparison.
y_proba_fhe, execution_time = get_normal_output(snn_classifier, torch_input)
print(print(f"\n({execution_time} sec(s)) NORMAL MODEL EXECUTED sucessfully."))
print("\nNormal Output:", y_proba_fhe)
