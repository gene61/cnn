# Diffraction Pattern Modeling and CNN Classification

The system models satellite-to-ground communication links (represented by black lines) where flying objects (simulated as green slabs) pass through the transmission path, creating detectable diffraction patterns at the ground receiver.
<img width="1153" height="424" alt="image" src="https://github.com/user-attachments/assets/2cb21cfa-1f51-4a4c-b791-8d9ee98135fb" />
Simulated flying trajectories passing through transmission path:
<img width="815" height="413" alt="image" src="https://github.com/user-attachments/assets/55c8a673-bf57-4ec1-ab99-53f5ac210551" />



## 🚀 Quick Start

### 1. Generate Diffraction Patterns

#### Run single simulation
python diffraction_modell.py

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

    
