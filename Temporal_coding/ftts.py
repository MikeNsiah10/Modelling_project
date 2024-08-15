import torch
import numpy as np
# FTTS Encoding Function
def ftts_encode(images, max_time=30):
    """
    Encode images using First Time to Spike (FTTS) encoding.

    Args:
        images (torch.Tensor): Batch of images with shape (batch_size, 1, height, width).
        max_time (int): Maximum spike time in milliseconds.

    Returns:
        torch.Tensor: Encoded spike times with shape (max_time, batch_size, 1, height, width).
    """
    # Normalize pixel values to [0, 1]
    images = (images - images.min()) / (images.max() - images.min())
    # Check the minimum and maximum values
    min_value = images.min().item()
    max_value = images.max().item()

    # Print the results
    #print("Minimum value in normalized images:", min_value)
    #print("Maximum value in normalized images:", max_value)
    #print("Shape of normalized images:", images.shape)


    # Invert pixel values to get spike times (higher intensity -> earlier spike)
    spike_times = (1 - images) * max_time

    # Create tensor to store spikes
    batch_size, _, height, width = images.shape
    spikes = torch.zeros((max_time, batch_size, 1, height, width))

    # Populate spikes tensor
    for t in range(max_time):
        spikes[t] = (spike_times <= t).float()

    return spikes