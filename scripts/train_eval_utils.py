import sys
import os
# Add the project root directory to sys.path to allow importing from parent directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tqdm import tqdm
import torch
from Temporal_coding.ftts import ftts_encode
from Temporal_coding.phase_encode import phase_encode
from scripts.spiking_model import SNN, SNNState
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scripts.mnist_pipeline import download_and_preprocess_mnist

# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MNIST data
train_loader, test_loader = download_and_preprocess_mnist()

# Define training parameters
num_epochs = 8
num_steps = 20  


#define training function
def train(model, device, train_loader, optimizer, epoch, encoding='ftts', num_steps=20):
    model.train()
    train_loss = 0
    correct = 0
    #epoch spike and synaptic operation
    epoch_spike_count = 0
    epoch_synaptic_operations = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Training Epoch {epoch}', leave=False)
    
    for batch_idx, (data, target) in progress_bar:
        if data.size(0) == train_loader.batch_size:  # Process only full batches
            data, target = data.to(device), target.to(device)
            #reset the spike count and synaptic operation
            model.reset()
            optimizer.zero_grad(set_to_none=True)

            # Encode the data based on the specified encoding method
            if encoding == 'ftts':
                encoded_data = ftts_encode(data, num_steps).to(device)
            elif encoding == 'phase':
                encoded_data = phase_encode(data, num_steps).to(device)
            else:
                raise ValueError("Invalid encoding type. Choose 'ftts' or 'phase'.")

            output = model(encoded_data)
            output = output.mean(0)  # Averaging over the time dimension

            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data.size(0)  # Multiply loss by batch size
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            epoch_spike_count += model.spike_count
            epoch_synaptic_operations += model.synaptic_operations

            progress_bar.set_postfix({'loss': loss.item()})  # Display loss for the current batch

    train_loss /= (len(train_loader.dataset) - (len(train_loader.dataset) % train_loader.batch_size))  # Adjust for dropped batch
    train_accuracy = 100. * correct / (len(train_loader.dataset) - (len(train_loader.dataset) % train_loader.batch_size))

    print(f'\nTrain set: Average loss: {train_loss:.4f}, Accuracy: {correct}/{(len(train_loader.dataset) - (len(train_loader.dataset) % train_loader.batch_size))} '
          f'({train_accuracy:.0f}%)\n')

    return train_loss, train_accuracy, epoch_spike_count, epoch_synaptic_operations

# Define the function to train and evaluate the model 
def train_and_evaluate(model, device, train_loader, test_loader, optimizer, num_epochs, encoding, num_steps):
    results = {}

    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    total_spikes = 0
    total_synaptic_operations = 0

    for epoch in range(1, num_epochs + 1):
        train_loss, train_accuracy, epoch_spike_count, epoch_synaptic_operations = train(
            model, device, train_loader, optimizer, epoch, encoding, num_steps
        )
        test_loss, test_accuracy = test(model, device, test_loader, epoch, encoding, num_steps)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        # Accumulate spikes and synaptic operations across all epochs
        total_spikes += epoch_spike_count
        total_synaptic_operations += epoch_synaptic_operations

    print(f'Total spikes: {total_spikes}, Total Synaptic operations: {total_synaptic_operations}')

    results = {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_losses': test_losses,
        'test_accuracies': test_accuracies,
        'total_spikes': total_spikes,
        'total_synaptic_operations': total_synaptic_operations
    }

    return results

# Define the testing function
def test(model, device, test_loader, epoch, encoding='ftts', num_steps=20):
    model.eval()  
    test_loss = 0
    correct = 0

    with torch.no_grad():  # Disable gradient calculation
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # Encode the data based on the specified encoding method
            if encoding == 'ftts':
                encoded_data = ftts_encode(data, num_steps).to(device)
            elif encoding == 'phase':
                encoded_data = phase_encode(data, num_steps).to(device)
            else:
                raise ValueError("Invalid encoding type. Choose 'ftts' or 'phase'.")

            output = model(encoded_data)
            output = output.mean(0)  
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # Accumulate loss
            pred = output.argmax(dim=1, keepdim=True)
            # get correct predictions
            correct += pred.eq(target.view_as(pred)).sum().item()  

    # Calculate average loss and accuracy
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    
    return test_loss, accuracy


# Define the function to evaluate the model
def evaluate_model(model, device, test_loader, num_steps, encoding='ftts'):
    #evaluate model
    model.eval() 
    true_labels = []
    predicted_labels = []
    # Disable gradient calculation
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            if data.size(0) == test_loader.batch_size:
                data, target = data.to(device), target.to(device)

                # Encode the data based on the specified encoding method
                if encoding == 'ftts':
                    encoded_data = ftts_encode(data, num_steps).to(device)
                elif encoding == 'phase':
                    encoded_data = phase_encode(data, num_steps).to(device)
                else:
                    raise ValueError("Invalid encoding type. Choose 'ftts' or 'phase'.")

                output = model(encoded_data)
                output = output.mean(0)  
                  # Get the index of the max log-probability
                pred = output.argmax(dim=1, keepdim=True)
                
                 # Store true labels
                true_labels.extend(target.cpu().numpy()) 
                # Store predicted labels
                predicted_labels.extend(pred.cpu().numpy())  
    
    return true_labels, predicted_labels

# Main function to train and evaluate the model with the specified encoding
def train_evaluate_encoding(encoding):
    # Initialize the SNN model
    model = SNN(input_features=28*28, hidden_features1=64, hidden_features2=128, output_features=10, record=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  

    # Train and evaluate the model
    results = train_and_evaluate(model, device, train_loader, test_loader, optimizer, num_epochs, encoding=encoding, num_steps=num_steps)

    return results, model
