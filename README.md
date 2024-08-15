## Modelling of synaptic plasticity 

# Project Overview

This repository is designed for working with the MNIST dataset, implementing various machine learning models, and exploring temporal coding techniques. It includes scripts for data handling, training, evaluation, and visualization, as well as specific implementations for Spike-Timing-Dependent Plasticity (STDP). The code leverages the Norse framework for these implementations.

## Repository Structure

```
/data                # Contains downloaded MNIST datasets
/scripts             # Scripts for data processing and model training
    ├── train_eval_utils.py       # Script  with different functions to add     
noise, train , test and  evaluate the model 
    ├── model.py       # Model definition without STDP
    ├── model_sdtp.py  # Model definition with STDP
    └── mnist_pipeline.py # Script to download MNIST datasets
/STDP                # Contains the STDP class for weight updates
/visualization       # Functions for plotting and visualizations
    ├── plot_functions.py # Functions for creating plots
    └── plots         # Directory to store visual outputs
/temporal_coding     # Functions related to 2 differents schemes of temporal coding
    ├── phase.py      # Phase coding functions
    └── ftts.py       # Functions for Fast Fourier Transform
main.py              # Main implementation file to run the project
```

## Usage

To use this repository, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/MikeNsiah10/Modelling_project.git
   cd Modelling_project
   ```

2. **Install Dependencies**:
   Make sure you have the necessary libraries installed. You can use pip to install them:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the MNIST Dataset**:
   Run the `mnist_pipeline.py` script to download the dataset:
   ```bash
   python scripts/mnist_pipeline.py
   ```

4. **Add Noise to the Dataset** (optional):
   If you want to add noise to the dataset, run:
   ```bash
   python scripts/add_noise.py
   ```

5. **Train the Model**:
   Use the training script to train your model:
   ```bash
   python scripts/train.py
   ```

6. **Test the Model**:
   After training, test the model using:
   ```bash
   python scripts/test.py
   ```

7. **Evaluate the Model**:
   Finally, evaluate the model performance with:
   ```bash
   python scripts/evaluate.py
   ```

## Usage

This project includes various functionalities, including:

- **Model Training and Evaluation**: Implementing a spiking neural network (SNN) model and assessing its performance on the MNIST dataset, both with and without noise.

- **STDP Implementation**: Utilizing the STDP class to update weights based on spike timings, providing insights into temporal learning mechanisms.

- **Data Visualization**: Generating plots to visualize model performance with STDP and dataset characteristics, comparing metrics under different conditions (with and without noise). The visualizations also include data transformations applied to phase coding and TTFS inputs, stored in the `/visualization/plots` directory.

- **Temporal Coding Techniques**: Exploring phase coding and time to first spike (TTFS) functions in conjunction with STDP, analyzing how these techniques affect the performance of the model.
