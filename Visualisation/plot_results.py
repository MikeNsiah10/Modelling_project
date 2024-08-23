import os 
import sys
import matplotlib.pyplot as plt

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Define the path for the plots directory in the main project directory
plots_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'plots')

# Create the plots directory if it doesn't exist
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)
    print(f'Created directory: {plots_dir}')
else:
    print(f'Directory already exists: {plots_dir}')


# Define the function to plot results
def plot_results(results, encoding,filename="training_results.png"): 
    plt.figure(figsize=(12, 10))

    # Access metrics directly from the results dictionary
    train_losses = results['train_losses'] 
    test_losses = results['test_losses']
    train_accuracies = results['train_accuracies']
    test_accuracies = results['test_accuracies']

    # Subplot for Training Loss
    plt.subplot(2, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Training Loss over Epochs ({encoding} encoding)')

    # Subplot for Testing Loss
    plt.subplot(2, 2, 2)
    plt.plot(range(1, len(test_losses) + 1), test_losses, label='Testing Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Testing Loss over Epochs ({encoding} encoding)')

    # Subplot for Training Accuracy
    plt.subplot(2, 2, 3)
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Training Accuracy', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(f'Training Accuracy over Epochs ({encoding} encoding)')

    # Subplot for Testing Accuracy
    plt.subplot(2, 2, 4)
    plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, label='Testing Accuracy', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(f'Testing Accuracy over Epochs ({encoding} encoding)')

    # Adjust layout to prevent overlap
    plt.tight_layout() 
    plt.savefig(os.path.join(plots_dir, filename))
    plt.close()
    print(f'Saved plot as: {filename}')