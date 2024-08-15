import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from scripts.mnist_pipeline import download_and_preprocess_mnist
from scripts.mnist_pipeline import download_and_preprocess_mnist
from scripts.train_eval_utils import train, test, add_noise, evaluate_model,train_and_evaluate #train_evaluate_encoding
import torch
from STDP.stdp_update import STDPState, STDPParameters, stdp_step_linear
from Temporal_coding.ftts import ftts_encode
from Temporal_coding.phase_encode import phase_encode
from scripts.model_stdp import SNN, SNNState
import matplotlib.pyplot as plt
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import os 
# Define the path for the plots directory in the main project directory
plots_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'plots')

# Create the plots directory if it doesn't exist
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)


# Define the function to plot results
def plot_results(results, encoding,filename='training_results.png'):
    plt.figure(figsize=(12, 10))

    for noise_level, metrics in results.items():
        train_losses = metrics['train_losses']
        test_losses = metrics['test_losses']
        train_accuracies = metrics['train_accuracies']
        test_accuracies = metrics['test_accuracies']

        plt.subplot(2, 2, 1)
        plt.plot(range(1, len(train_losses) + 1), train_losses, label=f'Training Loss (Noise {noise_level})')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title(f'Loss over Epochs ({encoding} encoding)')

        plt.subplot(2, 2, 2)
        plt.plot(range(1, len(test_losses) + 1), test_losses, label=f'Testing Loss (Noise {noise_level})')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title(f'Loss over Epochs ({encoding} encoding)')

        plt.subplot(2, 2, 3)
        plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label=f'Training Accuracy (Noise {noise_level})')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title(f'Accuracy over Epochs ({encoding} encoding)')

        plt.subplot(2, 2, 4)
        plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, label=f'Testing Accuracy (Noise {noise_level})')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title(f'Accuracy over Epochs ({encoding} encoding)')

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, filename))
    plt.close()
    print(f'Saved plot as: {filename}')
#
def train_evaluate_encoding(encoding):
    results = {}
    model = SNN(input_features=28*28, hidden_features1=128, hidden_features2=64, output_features=10, record=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for noise_level in noise_levels:
        result = train_and_evaluate(model, device, train_loader, test_loader, optimizer, num_epochs, encoding=encoding, num_steps=num_steps, noise_levels=[noise_level])
        results[noise_level] = result[noise_level]

    return results, model
# Define parameters
num_epochs = 4
num_steps = 20
noise_levels = [0.0, 0.1, 0.2]  # List of noise levels to test
batch_size = 64
learning_rate = 0.001


# Define the path for the plots directory
plots_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'plots')
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)
    print(f'Created directory: {plots_dir}')
else:
    print(f'Directory already exists: {plots_dir}')

# Load data
train_loader, test_loader = download_and_preprocess_mnist()

# Main execution
if __name__ == "__main__":
    print("Starting training and evaluation...")

    # Train and evaluate with FTTS encoding
    print("Training and evaluating with FTTS encoding...")
    ftts_results, ftts_model = train_evaluate_encoding('ftts')
    print("FTTS training complete, plotting results...")
    plot_results(ftts_results, encoding='FTTS', filename='ftts_training_results.png')
    print("FTTS plotting done.")

    # Train and evaluate with Phase encoding
    print("Training and evaluating with Phase encoding...")
    phase_results, phase_model = train_evaluate_encoding('phase')
    print("Phase training complete, plotting results...")
    plot_results(phase_results, encoding='Phase', filename='phase_training_results.png')
    print("Phase plotting done.")

    print("Training and evaluation complete.")