import sys
import os
# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import numpy as np
import torch  
from scripts.mnist_pipeline import download_and_preprocess_mnist
from Temporal_coding.ftts import ftts_encode
from Temporal_coding.phase_encode import phase_encode




# Determine if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load data
train_loader, test_loader = download_and_preprocess_mnist()

# Define the path for the plots directory in the main project directory
plots_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'plots')

# Create the plots directory if it doesn't exist
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)
    print(f'Created directory: {plots_dir}')
else:
    print(f'Directory already exists: {plots_dir}')




T = 20#timesteps
image, target = next(iter(train_loader))
example_image = image[0].unsqueeze(0).to(device)  



# Encode the example image
ftts_input = ftts_encode(example_image, T).to(device) 
phase_input = phase_encode(example_image, T).to(device)

filename="ftts_and_phase_spike_train"
# Visualize the encoded spikes for both methods
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(ftts_input.squeeze().cpu().numpy().reshape(20, -1), aspect='auto', cmap='binary')
plt.title("FTTS Encoded Spikes")
plt.ylabel("Time [ms]")
plt.xlabel("Neuron")

plt.subplot(1, 2, 2)
plt.imshow(phase_input.squeeze().cpu().numpy().reshape(20, -1), aspect='auto', cmap='binary')
plt.title("Phase Encoded Spikes")
plt.ylabel("Time [ms]")
plt.xlabel("Neuron")

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, filename))
plt.close()
print(f'Saved plot as: {filename}')