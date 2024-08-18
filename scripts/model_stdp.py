import torch
import norse
import norse.torch as norse
from norse.torch import LICell, SequentialState, LIFParameters, LIFState, LIState
import torchvision
import numpy as np
from norse.torch.module.lif import LIFCell, LIFRecurrentCell
from typing import NamedTuple

from torchvision import datasets, transforms
from typing import Tuple
from norse.torch.functional.heaviside import heaviside
import torch.nn as nn

import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns

import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from STDP.stdp_update import STDPState, STDPParameters, stdp_step_linear


# Determine if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the SNNState and SNN classes
class SNNState(NamedTuple):
    """
    State of the Spiking Neural Network (SNN) during simulation.

    Attributes:
        lif0 (LIFState): State of the first LIF layer.
        lif1 (LIFState): State of the second LIF layer.
        readout (LIState): State of the readout layer.
    """
    lif0: LIFState
    lif1: LIFState
    readout: LIState

class SNN(nn.Module):
    """
    Spiking Neural Network (SNN) model with two LIF recurrent layers and a readout layer.

    Args:
        input_features (int): Number of input features.
        hidden_features1 (int): Number of features in the first hidden layer.
        hidden_features2 (int): Number of features in the second hidden layer.
        output_features (int): Number of output features.
        record (bool): Whether to record the states for analysis.
        dt (float): Time step for simulation.
    """
    def __init__(self, input_features, hidden_features1, hidden_features2, output_features, record=False, dt=0.001):
        super(SNN, self).__init__()

        # Initialize the first LIF recurrent cell layer
        self.l1 = LIFRecurrentCell(input_features, hidden_features1, p=LIFParameters(alpha=100, v_th=torch.tensor(0.1)), dt=dt)
        # Initialize the second LIF recurrent cell layer
        self.l2 = LIFRecurrentCell(hidden_features2, hidden_features2, p=LIFParameters(alpha=100, v_th=torch.tensor(0.1)), dt=dt)
        # Linear transformation from first hidden layer to second hidden layer
        self.fc_hidden = nn.Linear(hidden_features1, hidden_features2, bias=False).to(device)
        # Linear transformation from second hidden layer to output
        self.fc_out = nn.Linear(hidden_features2, output_features, bias=False).to(device)
        # Readout layer
        self.out = LICell(dt=dt).to(device)

        # Store layer sizes and configuration parameters
        self.input_features = input_features
        self.hidden_features1 = hidden_features1
        self.hidden_features2 = hidden_features2
        self.output_features = output_features
        self.record = record

        # Initialize STDP parameters and state
        self.stdp_params = STDPParameters()
        self.stdp_state = STDPState(t_pre=torch.zeros(hidden_features1, device=device), t_post=torch.zeros(hidden_features2, device=device))
        self.voltages = []

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor with shape (seq_length, batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output voltages of the readout layer.
        """
        # Move input tensor to the appropriate device (CPU/GPU)
        x = x.to(device)
        seq_length, batch_size, _, _, _ = x.shape
        s1 = s2 = so = None
        voltages = []

        # Initialize recording if required
        if self.record:
            self.recording = SNNState(
                LIFState(z=torch.zeros(seq_length, batch_size, self.hidden_features1, device=device), v=torch.zeros(seq_length, batch_size, self.hidden_features1, device=device), i=torch.zeros(seq_length, batch_size, self.hidden_features1, device=device)),
                LIFState(z=torch.zeros(seq_length, batch_size, self.hidden_features2, device=device), v=torch.zeros(seq_length, batch_size, self.hidden_features2, device=device), i=torch.zeros(seq_length, batch_size, self.hidden_features2, device=device)),
                LIState(v=torch.zeros(seq_length, batch_size, self.output_features, device=device), i=torch.zeros(seq_length, batch_size, self.output_features, device=device)),
            )

        # Iterate over time steps
        for ts in range(seq_length):
            # Flatten the input tensor for the first layer
            z = x[ts, :, :, :].view(-1, self.input_features)
            # Forward pass through the first LIF layer
            z, s1 = self.l1(z, s1)
            # Apply linear transformation and forward pass through the second LIF layer
            z = self.fc_hidden(z)
            z, s2 = self.l2(z, s2)
            # Apply linear transformation and forward pass through the readout layer
            z = self.fc_out(z)
            vo, so = self.out(z, so)

            # Record states
            if self.record:
                self.recording.lif0.z[ts, :] = s1[1]
                self.recording.lif0.v[ts, :] = s1[0]
                self.recording.lif0.i[ts, :] = s1[2]
                self.recording.lif1.z[ts, :] = s2[1]
                self.recording.lif1.v[ts, :] = s2[0]
                self.recording.lif1.i[ts, :] = s2[2]
                self.recording.readout.v[ts, :] = so.v
                self.recording.readout.i[ts, :] = so.i

            # Collect output voltages
            voltages += [vo]

            # Perform STDP step
            #pre synaptic spikes
            z_pre = s1[1]
            #post synapnatic spikes
            z_post = s2[1]
            self.fc_hidden.weight.data = self.fc_hidden.weight.data.to(device)
            #modify weights  of linear layer
            self.fc_hidden.weight.data, self.stdp_state = stdp_step_linear(z_pre, z_post, self.fc_hidden.weight.data, self.stdp_state, self.stdp_params)

        # Stack voltages into a tensor
        self.voltages = torch.stack(voltages)
        return self.voltages
