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
    plt.figure(figsize=(12, 12))

    train_losses = results['train_losses']
    test_losses = results['test_losses']
    train_accuracies = results['train_accuracies']
    test_accuracies = results['test_accuracies']
    total_spikes = results['total_spikes']
    total_synaptic_operations = results['total_synaptic_operations']

    #subplots for train_losses
    plt.subplot(3, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Training Loss over Epochs ({encoding} encoding)')
    plt.xlim(1, len(train_losses))
    plt.xticks(range(1, len(train_losses) + 1))

     #subplots for test_losses
    plt.subplot(3, 2, 2)
    plt.plot(range(1, len(test_losses) + 1), test_losses, label='Testing Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Testing Loss over Epochs ({encoding} encoding)')
    plt.xlim(1, len(test_losses))
    plt.xticks(range(1, len(test_losses) + 1))

    #subplots for train_accuracies
    plt.subplot(3, 2, 3)
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Training Accuracy', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(f'Training Accuracy over Epochs ({encoding} encoding)')
    plt.xlim(1, len(train_accuracies))
    plt.xticks(range(1, len(train_accuracies) + 1))

     #subplots for test_losses
    plt.subplot(3, 2, 4)
    plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, label='Testing Accuracy', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(f'Testing Accuracy over Epochs ({encoding} encoding)')
    plt.xlim(1, len(test_accuracies))
    plt.xticks(range(1, len(test_accuracies) + 1))

     #subplots for spike count
    plt.subplot(3, 2, 5)
    if total_spikes != 0:
        plt.plot(len(train_losses), total_spikes, label='Total Spikes', color='purple', marker='o')
        plt.annotate(f'{total_spikes:,.0f}',
                     xy=(len(train_losses), total_spikes),
                     xytext=(len(train_losses), total_spikes * 1.1),
                     arrowprops=dict(facecolor='black', shrink=0.05),
                     fontsize=12)
    plt.xlabel('Epochs')
    plt.ylabel('Spike Count')
    plt.legend()
    plt.title(f'Total Spikes ({encoding} encoding)')
    plt.xlim(0, len(train_losses) + 1)
    plt.xticks(range(1, len(train_losses) + 1))

     #subplots for synaptic operations
    plt.subplot(3, 2, 6)
    if total_synaptic_operations != 0:
        plt.plot(len(train_losses), total_synaptic_operations, label='Total Synaptic Operations', color='brown', marker='o')
        plt.annotate(f'{total_synaptic_operations:,.0f}',
                     xy=(len(train_losses), total_synaptic_operations),
                     xytext=(len(train_losses), total_synaptic_operations * 1.1),
                     arrowprops=dict(facecolor='black', shrink=0.05),
                     fontsize=12)
    plt.xlabel('Epochs')
    plt.ylabel('Synaptic Operations')
    plt.legend()
    plt.title(f'Total Synaptic Operations ({encoding} encoding)')
    plt.xlim(0, len(train_losses) + 1)
    plt.xticks(range(1, len(train_losses) + 1))

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, filename))
    plt.close()
    print(f'Saved plot as: {filename}')