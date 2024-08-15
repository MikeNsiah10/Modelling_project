import sys
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import torch

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the function from mnist_pipeline
from scripts.mnist_pipeline import download_and_preprocess_mnist
from Temporal_coding.ftts import ftts_encode
from Temporal_coding.phase_encode import phase_encode
from STDP.stdp_update import STDPState, STDPParameters, stdp_step_linear
from scripts.model_stdp import SNN, SNNState

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the path for the plots directory in the main project directory
plots_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'plots')

# Create the plots directory if it doesn't exist
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Use the function to get data loaders
train_loader, test_loader = download_and_preprocess_mnist()
print(f'Number of batches in train_loader: {len(train_loader)}')

# Select a sample image from the MNIST dataset
example_data, example_target = next(iter(train_loader))
example_image = example_data[0].unsqueeze(0)  # Shape: (1, 1, 28, 28)

# Number of time steps
num_steps = 30

# Encode the image using both encoding methods
ftts_input = ftts_encode(example_image, num_steps).to(device)
phase_input = phase_encode(example_image, num_steps).to(device)

# Initialize the SNN for both encoding methods
example_snn_ftts = SNN(input_features=28*28, hidden_features1=64, hidden_features2=128, output_features=10, record=True).to(device)
example_snn_phase = SNN(input_features=28*28, hidden_features1=64, hidden_features2=128, output_features=10, record=True).to(device)

# Get the readout voltages for FTTS encoding
ftts_readout_voltages = example_snn_ftts(ftts_input)

# Get the readout voltages for Phase encoding
phase_readout_voltages = example_snn_phase(phase_input)

# Spiking activities for LIF layer 1
ftts_spiking_activity_lif1 = example_snn_ftts.recording.lif0.z
phase_spiking_activity_lif1 = example_snn_phase.recording.lif0.z

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_spiking_activity(spiking_activity, encoding_method, interval=500, filename="spiking_activity.gif"):
    """
    Creates an animation of spiking activity over time and saves it as a file.

    Parameters:
    - spiking_activity: Tensor of shape [num_steps, channels, neurons]
    - encoding_method: String describing the encoding method (e.g., 'FTTS', 'Phase')
    - interval: Delay between frames in milliseconds
    - filename: Filename to save the animation (with appropriate extension)
    """
    num_steps, channels, num_neurons = spiking_activity.shape

    # Initialize the figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(np.zeros((channels, num_neurons)), cmap='viridis', aspect='auto', vmin=0, vmax=1)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Spike Activity')

    # Add gridlines
    ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)

    # Customize ticks and labels
    ax.set_xticks(np.arange(0, num_neurons, step=10))
    ax.set_xticklabels(np.arange(0, num_neurons, step=10))
    ax.set_yticks([])  # Remove y ticks
    ax.set_yticklabels([])  # Remove y tick labels

    # Set axis labels and title
    ax.set_title(f'Neuron Spiking Activity Over Time ({encoding_method} Encoding)')
    ax.set_xlabel('Neuron Index')
    ax.set_ylabel('')

    def update(frame):
        data = spiking_activity[frame].cpu().detach().numpy()
        data = (data > 0).astype(int)  # Convert to binary (0 and 1)
        im.set_array(data)
        ax.set_title(f'Neuron Spiking Activity Over Time ({encoding_method} Encoding) - Time Step: {frame}')
        return [im]

    # Create the animation
    anim = FuncAnimation(fig, update, frames=num_steps, interval=interval, blit=False)

    # Save the animation
    anim.save(os.path.join(plots_dir, filename))
    plt.close()
    print(f'Saved plot as: {filename}')

# Example usage
# Save animations for both FTTS and Phase encoding
animate_spiking_activity(ftts_spiking_activity_lif1, encoding_method='FTTS', interval=500, filename="ftts_spiking_activity.gif")
animate_spiking_activity(phase_spiking_activity_lif1, encoding_method='Phase', interval=500, filename="phase_spiking_activity.gif")
