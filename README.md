# Diffraction Pattern Modeling and CNN Classification

The system models satellite-to-ground communication links (represented by black lines) where flying objects (simulated as green slabs) pass through the transmission path, creating detectable diffraction patterns at the ground receiver.
<img width="1153" height="424" alt="image" src="https://github.com/user-attachments/assets/2cb21cfa-1f51-4a4c-b791-8d9ee98135fb" />
Simulated flying trajectories passing through one transmission path (solid black line):
<img width="815" height="413" alt="image" src="https://github.com/user-attachments/assets/55c8a673-bf57-4ec1-ab99-53f5ac210551" />



## 🚀 Quick Start

### 1. Generate Diffraction Patterns

#### Run single simulation
python diffraction_modell.py

example outputs:
<img width="264" height="267" alt="image" src="https://github.com/user-attachments/assets/91f3feb5-de79-471e-b9c6-04e5c460c3f4" />
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

    
