## Modelling of synaptic plasticity 

# Project Overview

This repository is designed for working with the MNIST dataset, implementing various machine learning models, and exploring temporal coding techniques. It includes scripts for data handling, training, evaluation, and visualization, as well as specific implementations for Spike-Timing-Dependent Plasticity (STDP). The code leverages the Norse framework for these implementations.

## Repository Structure

```
/data                              # Contains downloaded MNIST datasets
/scripts                           # Scripts for data processing and model training
    ├── train_eval_utils.py        # Script  with different functions to add     
                                    noise, train , test and  evaluate the model 
    ├── model.py                   # Model definition without STDP
    ├── model_sdtp.py              # Model definition with STDP
    └── mnist_pipeline.py          # Script to download MNIST datasets
/STDP
    ├── stdp_updates.py            # Contains the STDP classes for weight updates
/visualization                     # Functions for plotting and visualizations
    ├── plot_encoded_images.py     # Script for visualizing and comparing FTTS and Phase encoding methods for MNIST images
    ├── plot_membrane_voltages.py  # Script for plotting membrane voltages of SNN with FTTS and Phase encoding methods
    └── plot_samples_images.py     # Script for plotting and saving sample images from the MNIST dataset
    ├── plot_spikes.py             # Script for visualizing and saving FTTS and Phase encoded spike trains
/plots                             # Directory to store visual outputs
/Temporal_coding                   # Functions related to 2 differents schemes of temporal coding
    ├── phase.py                   # Phase coding functions
    └── ftts.py                    # Functions for Fast Fourier Transform
requirements.txt                   # file with the different libraries used on the project
main.py                            # Main implementation file to run the project
```

## Usage

To use this repository, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/MikeNsiah10/Modelling_project.git
   cd Modelling_project
   ```

2. **Setting Up a Python Environment**
It is recommended to use a virtual environment to manage the dependencies for this project. A virtual environment helps to isolate your project's dependencies from your global Python environment, avoiding potential conflicts.
 ```bash
   # Create a virtual environment in a directory named 'env'
   python3 -m venv env

    # Activate the virtual environment
    # On Windows
    env\Scripts\activate
    # On macOS/Linux
    source env/bin/activate

   ```
3. **Install Dependencies**:
   Make sure you have the necessary libraries installed. You can use pip to install them:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the MNIST Dataset**:
   Run the `mnist_pipeline.py` script to download the dataset:
   ```bash
   python scripts/mnist_pipeline.py
   ```

5. **Plot and view encoded_images using the coding functions(ftts_encode and phase_encode)** :
   If you want to look at the spikes generated using both ftts and phase function, run and check ftts_and_spike_train.png in plots folder:
   ```bash
      python Visualisation/plot_encoded_images.py
   ```

6. **Plot and view memory voltages produced using the spiking model**:
   To view membrane voltages of encoded sample image , run and check membranes_voltages.png in plots folder:
   ```bash
      python Visualisation/plot_membrane_voltages.py
   ```

6. **Plot and view sample mnist datasets**:
   To see some sampled mnist images, run and check mnist_samples.png in plots folder:
   ```bash
   python Visualisation/plot_sample_images.py
   ```

7. **Plot and visualise spiking activity of the neurons**:
   Visualise the spiking activity of the neurons by running and checking phase_spiking_activity.gif for phase encoding and ftts_spiking_activity.gif for ftts respectively in plots folder:
   ```bash
   python Visualisation/plot_spiking_activity.py
   ```
8. **Main execution**
finally train and evaluate the model on mnist datasets.run and check the results(ftts_training_results.png and phase_training_results.png) and confusion matrix(FFTS_Conf_matrix.png and Phase_Conf_matrix.png ) in plots folder
 ```bash
  python main.py

   ```

## Usage

This project includes various functionalities, including:

- **Model Training and Evaluation**: Implementing a spiking neural network (SNN) model and assessing its performance on the MNIST dataset, both with and without noise.

- **STDP Implementation**: Utilizing the STDP class to update weights based on spike timings, providing insights into temporal learning mechanisms.

- **Data Visualization**: Generating plots to visualize model performance  and dataset characteristics, comparing training and evaluation metrics . The visualizations also include data transformations applied to phase coding and TTFS inputs. The plots can be observed in the `plots` directory.

- **Temporal Coding Techniques**: Exploring phase coding and time to first spike (TTFS) functions in conjunction with superspike surrogate gradient learning which is implemented automically by the use of lifrecurrent layers in our spikig model, and analyzing how these techniques affect the performance of the model.
