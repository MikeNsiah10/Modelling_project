import sys
import os
# Add the root directory to sys.path
#very important please dont change it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import numpy as np
import torch
from scripts.mnist_pipeline import download_and_preprocess_mnist
from Temporal_coding.ftts import ftts_encode
from Temporal_coding.phase_encode import phase_encode




# Define the path for the plots directory in the main project directory
plots_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'plots')

# Create the plots directory if it doesn't exist
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Get a batch of images and labels
train_loader, _ = download_and_preprocess_mnist()
data_iter = iter(train_loader)
images, labels = next(data_iter)

# Function to plot and save images
def visualize_encoding_comparison(images, num_images=5, max_time=30, filename='encoding_comparison.png'):
    """
    Visualize and compare First Time to Spike (FTTS) and Phase encoding methods for a set of images.
    This function creates a comprehensive visual comparison between FTTS and Phase encoding
    for neural networks.
    """
    ftts_spikes = ftts_encode(images, max_time)
    phase_spikes = phase_encode(images, max_time)

    # Create a new figure
    fig, axes = plt.subplots(num_images, 4, figsize=(20, 4 * num_images))

    for i in range(num_images):
        ax1, ax2, ax3, ax4 = axes[i]

        # Original image
        ax1.imshow(images[i, 0], cmap='gray')
        ax1.set_title(f"Original Image {i}")
        ax1.axis('off')

        # FTTS encoding
        ftts_spike_train = ftts_spikes[:, i, 0, :, :].numpy()
        ftts_first_spike_times = np.argmax(ftts_spike_train, axis=0)
        ax2.imshow(ftts_first_spike_times, cmap='viridis')
        ax2.set_title(f"FTTS Encoding for Image {i}")
        ax2.set_xlabel("Pixel Position")
        ax2.set_ylabel("Pixel Position")

        # Phase encoding - time-averaged spike activity
        phase_spike_train = phase_spikes[:, i, 0, :, :].numpy()
        phase_spike_avg = np.mean(phase_spike_train, axis=0)  # Average over time
        ax3.imshow(phase_spike_avg, cmap='viridis')
        ax3.set_title(f"Phase Encoding (Time-averaged) for Image {i}")
        ax3.set_xlabel("Pixel Position")
        ax3.set_ylabel("Pixel Position")

        # Spike trains for selected pixels
        selected_pixels = [(14, 14), (7, 7), (21, 21)]  # Example pixel positions
        for idx, (x, y) in enumerate(selected_pixels):
            ax4.plot(ftts_spike_train[:, y, x] + idx, label=f'FTTS ({x},{y})')
            ax4.plot(phase_spike_train[:, y, x] + idx + 0.1, label=f'Phase ({x},{y})')
        ax4.set_title("Spike Trains for Selected Pixels")
        ax4.set_xlabel("Time")
        ax4.set_ylabel("Pixel")
        ax4.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, filename))
    plt.close()
    print(f'Saved plot as: {filename}')

# Use the visualization function
visualize_encoding_comparison(images, num_images=5, filename='encoding_comparison.png')