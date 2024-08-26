## Modelling of synaptic plasticity 

# Project Overview

This repository is designed for working with the MNIST dataset, implementing various machine learning models, and exploring temporal coding techniques with surrogate gradient. It includes scripts for data handling, training, evaluation, and visualization. The code leverages the Norse framework for these implementations.


## Repository Structure
```
/data                              # Contains downloaded MNIST datasets
/scripts                           # Scripts for data processing and model training
    ├── train_eval_utils.py        # Script  with different functions to train , test and  evaluate the model 
    ├── model.py                   # Model definition without STDP
    ├── spiking_model.py              # Model and state during simualtion definition
    └── mnist_pipeline.py          # Script to download MNIST datasets
/learning_algorithm
    ├─ superspike_algo.py          #scripts of the superspike gradient implementation 
/visualization                     # Functions for plotting and visualizations
    ├── plot_confusion_matrix.py     # Script for plotting the confusion matrix
    ├── plot_encoded_images.py     # Script for visualizing and comparing FTTS and Phase encoding methods for MNIST images
    ├── plot_membrane_voltages.py  # Script for plotting membrane voltages of SNN with FTTS and Phase encoding methods
    └── plot_results.py     # Script for plotting and saving the accuracies, losses, and energy consumptions of the different strategies
    └── plot_samples_images.py     # Script for plotting and saving sample images from the MNIST dataset
    ├── plot_spikes.py             # Script for visualizing and saving an animation of spiking activity over time
 ├── plot_spiking_activity.py             # Script for visualizing and saving FTTS and Phase encoded spike trains
/plots                             # Directory to store visual outputs 
/Temporal_coding                   # Functions related to 2 differents schemes of temporal coding
    ├── phase.py                   # Phase coding function
    └── ftts.py                    # Function for First time to spike 
requirements.txt                   # file with the different libraries used on the project
main.py                            # Main implementation file to run the project
```
## Requirements

To use this repository, follow these steps:

1. Clone the Repository:
   ```
      git clone https://github.com/MikeNsiah10/Modelling_project.git
   cd Modelling_project
   ```

2. Setting Up a Python Environment
It is recommended to use a virtual environment to manage the dependencies for this project. A virtual environment helps to isolate your project's dependencies from your global Python environment, avoiding potential conflicts.
```
    # Create a virtual environment in a directory named 'env'
   python3 -m venv env

    # Activate the virtual environment
    # On Windows
    env\Scripts\activate
    # On macOS/Linux
    source env/bin/activate
```
   
3. Install Dependencies:
   Make sure you have the necessary libraries installed. You can use pip to install them:
   ```
      pip install -r requirements.txt
   ```
   

4. Download the MNIST Dataset:
   Run the mnist_pipeline.py script to download the dataset:
   ```
      python scripts/mnist_pipeline.py
   ```

5. Plot and view encoded_images using the coding functions(ftts_encode and phase_encode) :
   If you want to look at the spikes generated using both ftts and phase function, run and check ftts_and_spike_train.png in plots folder:
   ```
         python Visualisation/plot_encoded_images.py
   ```

6. Plot and view memory voltages produced using the spiking model:
   To view membrane voltages of encoded sample image , run and check membranes_voltages.png in plots folder:
         python Visualisation/plot_membrane_voltages.py
   

6. Plot and view sample mnist datasets:
   To see some sampled mnist images, run and check mnist_samples.png in plots folder:
   ```
      python Visualisation/plot_sample_images.py
   ```
## Purpose
This code is implemented to make a comparative study between two different schemes of temporal coding: phase coding and time to first spike. The comparison is made examinating the accuracy, the loss and the energy efficiency(spikes generated during simulation of the model during training) for both stategies with the SuperSpike algorithm (surrogate gradient) from Norse.
