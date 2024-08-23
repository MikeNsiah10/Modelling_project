import sys
import os
# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import numpy as np
from scripts.mnist_pipeline import download_and_preprocess_mnist


# Import the function from mnist_pipeline
from scripts.mnist_pipeline import download_and_preprocess_mnist

# load mnist data
train_loader, test_loader = download_and_preprocess_mnist()


# Define the path for the plots directory in the main project directory
plots_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'plots')

# Create the plots directory if it doesn't exist
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)
    print(f'Created directory: {plots_dir}')
else:
    print(f'Directory already exists: {plots_dir}')

# Load a batch from the train_loader
data_iter = iter(train_loader)
images, labels = next(data_iter)

# Function to plot and save images
def plot_and_save_images(images, labels, num_images=10, filename='mnist_samples.png'):
    print(f'Plotting and saving images to: {os.path.join(plots_dir, filename)}')
    fig, axes = plt.subplots(1, num_images, figsize=(15, 4))
    for i in range(num_images):
        ax = axes[i]
        ax.imshow(images[i].numpy().squeeze(), cmap='gray')
        ax.set_title(f'Label: {labels[i].item()}')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, filename))
    plt.close()
    print(f'Saved plot as: {filename}')

# Plot images from train_loader
plot_and_save_images(images, labels, num_images=10, filename='mnist_samples.png')
