# Diffraction Pattern Modeling and CNN Classification

A machine learning pipeline for generating synthetic diffraction patterns and training convolutional neural networks for pattern classification and analysis.

## 🚀 Quick Start

### 1. Generate Diffraction Patterns

# Run single simulation
python diffraction_modell.py

# Run batch simulations
./run_simulations.sh

### 2. Train CNN Model
python cnn_model_Aug.py



📊 Complete Workflow

Phase 1: Data Generation

diffraction_modell.py - Core diffraction simulation engine

    Generates synthetic diffraction intensity patterns

    Customizable physics parameters (wavelength, satellite heigh, SNR etc.)

    Outputs diffraction images detected by ground satellite receiver

run_simulations.sh - Automated batch processing

    Runs multiple simulations with parameter variations

    Handles large-scale data generation efficiently

    Organizes output into structured directories

Phase 2: Machine Learning

cnn_model_Aug.py - Deep learning classification

    Convolutional Neural Network for pattern recognition

    Integrated data augmentation for robust training

    
