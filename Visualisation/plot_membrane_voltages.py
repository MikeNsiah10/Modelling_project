import sys
import os

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import numpy as np
import torch  # Ensure torch is imported for device handling



from scripts.mnist_pipeline import download_and_preprocess_mnist
from Temporal_coding.ftts import ftts_encode
from Temporal_coding.phase_encode import phase_encode
from STDP.stdp_update import STDPState, STDPParameters, stdp_step_linear

from scripts.model_stdp import SNN, SNNState

# Determine if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the path for the plots directory in the main project directory
plots_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'plots')

# Create the plots directory if it doesn't exist
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)
    print(f'Created directory: {plots_dir}')
else:
    print(f'Directory already exists: {plots_dir}')

# Use the function to get data loaders
train_loader, test_loader = download_and_preprocess_mnist()
print(f'Number of batches in train_loader: {len(train_loader)}')

T = 20
example_data, example_target = next(iter(train_loader))
example_image = example_data[0].unsqueeze(0).to(device)  # Move example_image to the correct device

# Initialize the SNN model and move it to the device
example_snn = SNN(28 * 28, 100, 100, 10, record=True, dt=0.001).to(device)

# Encode the example image
ftts_input = ftts_encode(example_image, T).to(device)  # Ensure input is on the same device as model
phase_input = phase_encode(example_image, T).to(device)

# Get the readout voltages for both encoding methods
ftts_readout_voltages = example_snn(ftts_input)
phase_readout_voltages = example_snn(phase_input)

# Squeeze the batch dimension if needed and detach from the computation graph
ftts_voltages = ftts_readout_voltages.squeeze(1).detach().cpu().numpy()
phase_voltages = phase_readout_voltages.squeeze(1).detach().cpu().numpy()

filename = "membrane_voltages.png"

# Plot the voltages for both encoding methods
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(ftts_voltages)
plt.title("FTTS Encoding")
plt.ylabel("Voltage [a.u.]")
plt.xlabel("Time [ms]")

plt.subplot(1, 2, 2)
plt.plot(phase_voltages)
plt.title("Phase Encoding")
plt.ylabel("Voltage [a.u.]")
plt.xlabel("Time [ms]")

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, filename))
plt.close()
print(f'Saved plot as: {filename}')
