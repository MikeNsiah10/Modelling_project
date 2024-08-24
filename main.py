import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
import torch
from scripts.mnist_pipeline import download_and_preprocess_mnist
from scripts.train_eval_utils import train, test, evaluate_model,train_and_evaluate,train_evaluate_encoding
import torch
from Visualisation.plot_confusion_matrix import plot_confusion_matrix
from Visualisation.plot_results import plot_results
from Temporal_coding.ftts import ftts_encode
from Temporal_coding.phase_encode import phase_encode
from scripts.spiking_model import SNN, SNNState
import matplotlib.pyplot as plt
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define parameters
#if changes are made here make sure to change also in train_eval_utils
num_epochs = 8
num_steps = 20  

# Load MNIST data
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


    #plot confusion matrix of true and predicted labels of FTTS encoding
    true_labels_ftts, predicted_labels_ftts = evaluate_model(ftts_model, device, test_loader, num_steps, encoding='ftts') 
    print("plotting FTTS confusion matrix")
    plot_confusion_matrix(true_labels_ftts, predicted_labels_ftts, title="Confusion Matrix (FTTS Encoding)",filename="FTTS_Conf_matrix.png")
    print("Plotting of FTTS matrix done...........")

    #plot confusion matrix of true and predicted labels of Phase encoding
    true_labels_phase, predicted_labels_phase = evaluate_model(phase_model, device, test_loader, num_steps, encoding='phase') 
    print("ploting Phase onfusion matrix")
    plot_confusion_matrix(true_labels_phase, predicted_labels_phase, title="Confusion Matrix (Phase Encoding)",filename="Phase_Conf_matrix.png")
    print("Plotting Phase matrix done...........")
    print("Training and evaluation complete.check plots folder for results")