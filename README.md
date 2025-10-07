# Diffraction Pattern Modeling and CNN Classification




## 🚀 Quick Start

### 1. Generate Diffraction Patterns

#### Run single simulation
python diffraction_modell.py

example outputs:
<img width="264" height="267" alt="image" src="https://github.com/user-attachments/assets/30ac4ac3-8268-4737-849d-8e6b30444f9c" />
<img width="264" height="267" alt="image" src="https://github.com/user-attachments/assets/2c580811-ec98-476c-8d82-87917f8ee009" />




#### Run batch simulations
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

    
