from tqdm import tqdm
import torch
from Temporal_coding.ftts import ftts_encode
from Temporal_coding.phase_encode import phase_encode
from scripts.model_stdp import SNN, SNNState
import torch.nn.functional as F

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the function to add noise to the data
def add_noise(images, noise_level=0.1):
    noise = torch.randn_like(images) * noise_level
    noisy_images = images + noise
    return torch.clamp(noisy_images, 0, 1)

# Define the training function
def train(model, device, train_loader, optimizer, epoch, encoding='ftts', num_steps=20, noise_level=0.1):
    model.train()
    train_loss = 0
    correct = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Training Epoch {epoch}', leave=False)

    for batch_idx, (data, target) in progress_bar:
        if data.size(0) == train_loader.batch_size:  # Process only full batches
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            # Add noise to the data
            data = add_noise(data, noise_level)

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

            progress_bar.set_postfix({'loss': loss.item()})  # Display loss for the current batch

    train_loss /= (len(train_loader.dataset) - (len(train_loader.dataset) % train_loader.batch_size))  # Adjust for dropped batch
    train_accuracy = 100. * correct / (len(train_loader.dataset) - (len(train_loader.dataset) % train_loader.batch_size))

    print(f'\nTrain set: Average loss: {train_loss:.4f}, Accuracy: {correct}/{(len(train_loader.dataset) - (len(train_loader.dataset) % train_loader.batch_size))} '
          f'({train_accuracy:.0f}%)\n')

    return train_loss, train_accuracy


# Define the testing function
def test(model, device, test_loader, epoch, encoding='ftts', num_steps=20, noise_level=0.1):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # Add noise to the data
            data = add_noise(data, noise_level)

            # Encode the data based on the specified encoding method
            if encoding == 'ftts':
                encoded_data = ftts_encode(data, num_steps).to(device)
            elif encoding == 'phase':
                encoded_data = phase_encode(data, num_steps).to(device)
            else:
                raise ValueError("Invalid encoding type. Choose 'ftts' or 'phase'.")

            output = model(encoded_data)
            output = output.mean(0)  # Average over time steps
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    return test_loss, accuracy




# Define the function to evaluate the model
def evaluate_model(model, device, test_loader, num_steps, encoding='ftts', noise_level=0.1):
    model.eval()
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            if data.size(0) == test_loader.batch_size:
                data, target = data.to(device), target.to(device)

                # Add noise to the data
                data = add_noise(data, noise_level)

                if encoding == 'ftts':
                    encoded_data = ftts_encode(data, num_steps).to(device)
                elif encoding == 'phase':
                    encoded_data = phase_encode(data, num_steps).to(device)
                else:
                    raise ValueError("Invalid encoding type. Choose 'ftts' or 'phase'.")

                output = model(encoded_data)
                output = output.mean(0)
                pred = output.argmax(dim=1, keepdim=True)
                true_labels.extend(target.cpu().numpy())
                predicted_labels.extend(pred.cpu().numpy())

    return true_labels, predicted_labels


# Define the function to train and evaluate the model
def train_and_evaluate(model, device, train_loader, test_loader, optimizer, num_epochs, encoding, num_steps, noise_levels):
    results = {}
    for noise_level in noise_levels:
        print(f"Training with {encoding} encoding and noise level: {noise_level}")

        train_losses = []
        train_accuracies = []
        test_losses = []
        test_accuracies = []

        for epoch in range(1, num_epochs + 1):
            train_loss, train_accuracy = train(model, device, train_loader, optimizer, epoch, encoding, num_steps, noise_level)
            test_loss, test_accuracy = test(model, device, test_loader, epoch, encoding, num_steps, noise_level)

            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)

        results[noise_level] = {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'test_losses': test_losses,
            'test_accuracies': test_accuracies
        }

    return results


def train_evaluate_encoding(encoding):
    results = {}
    model = SNN(input_features=28*28, hidden_features1=128, hidden_features2=64, output_features=10, record=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for noise_level in noise_levels:
        result = train_and_evaluate(model, device, train_loader, test_loader, optimizer, num_epochs, encoding=encoding, num_steps=num_steps, noise_levels=[noise_level])
        results[noise_level] = result[noise_level]

    return results, model