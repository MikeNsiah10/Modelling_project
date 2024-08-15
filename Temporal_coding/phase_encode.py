import torch
import numpy as np
#phase encoding function
def phase_encode(images, num_steps=30, freq=10):
    """
    Applies a phase encoding process to a batch of images.

    Args:
        images (torch.Tensor): Input tensor of images with shape (batch_size, channels, height, width).
        num_steps (int): Number of temporal steps for encoding.
        freq (int): Frequency of the sinusoidal signal in Hz.

    Returns:
        torch.Tensor: Tensor containing spike trains resulting from the phase encoding.
    """

    # Move the images tensor to the same device as the input tensor
    device = images.device

    # Extract dimensions of the input images tensor
    batch_size, _, height, width = images.shape

    # Normalize images to range [0, 1]
    images = (images - images.min()) / (images.max() - images.min())

    # Create a tensor of time steps
    t = torch.arange(0, num_steps, 1).float().to(device)

    # Reshape time steps to match the dimensions for broadcasting
    t = t.view(num_steps, 1, 1, 1, 1)

    # Compute the phase shift for each pixel based on the normalized image
    phase_shift = 2 * np.pi * images.unsqueeze(0)

    # Generate spike trains using a sinusoidal function with a specified frequency
    spike_trains = torch.sin(2 * np.pi * freq * t / 1000 + phase_shift) > 0

    # Convert the boolean spike trains to float tensor (1s and 0s)
    return spike_trains.float()
