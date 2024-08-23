import sys
import os
# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import numpy as np
import torch  
import norse
from Temporal_coding.ftts import ftts_encode
from Temporal_coding.phase_encode import phase_encode
from scripts.spiking_model import SNN, SNNState
from scripts.mnist_pipeline import download_and_preprocess_mnist


# Determine if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load mnist
train_loader, test_loader = download_and_preprocess_mnist()


# Define the path for the plots directory in the main project directory
plots_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'plots')

# Create the plots directory if it doesn't exist
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)
    print(f'Created directory: {plots_dir}')
else:
    print(f'Directory already exists: {plots_dir}')



T = 20 #timsteps
data, target = next(iter(train_loader))
#take an example image
example_image = data[0].unsqueeze(0).to(device)  

# Initialize the SNN model a
model = SNN(28 * 28, 64, 128, 10, record=True, dt=0.001).to(device)

# Encode the example image
ftts_input = ftts_encode(example_image, T).to(device)  
phase_input = phase_encode(example_image, T).to(device)

# Get the readout voltages for both encoding methods
ftts_readout_voltages = model(ftts_input)
phase_readout_voltages = model(phase_input)

# Squeeze the batch dimension 
ftts_voltages = ftts_readout_voltages.squeeze(1).detach().cpu().numpy()
phase_voltages = phase_readout_voltages.squeeze(1).detach().cpu().numpy()

#filename to store the membrane voltages
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
