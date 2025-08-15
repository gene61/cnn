import numpy as np
import matplotlib.pyplot as plt
import argparse
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting
import os
from datetime import datetime
FLYING_OBJECTS = {
    'Drone1': {                        
        'fligh_height': np.round(np.random.uniform(60, 130), 2),  # Random height within range (2 decimal places)
        'width': np.round(np.random.uniform(0.2, 0.22), 2),       # width of the drone (2 decimal places)
        'length': np.round(np.random.uniform(0.24, 0.26), 2),      # length of the drone (2 decimal places)
        'speed_range': (2, 20),         # speed range in m/s
        'radius_detection': 12,           # detection radius from link
        'time_range': 40,                # time range from -40s to 40s
        'num_data_points' : 5000       #  data points per unit time
    },
        'Drone2': {                         
        'fligh_height': np.round(np.random.uniform(60, 130), 2),  
        'width': np.round(np.random.uniform(0.1, 0.13), 2),    
        'length': np.round(np.random.uniform(0.14, 0.17), 2),    
        'speed_range': (2, 20),     
        'radius_detection': 7,           
        'time_range': 40,                 
        'num_data_points' : 5000      
    },
        'Drone3': {                         
        'fligh_height': np.round(np.random.uniform(60, 130), 2),  
        'width': np.round(np.random.uniform(0.32, 0.35), 2),       
        'length': np.round(np.random.uniform(0.37, 0.45), 2),    
        'speed_range': (2, 20),     
        'radius_detection': 17,          
        'time_range': 40,                 
        'num_data_points' : 5000      
    },
    'No_Drone': {                       
        'fligh_height': np.round(np.random.uniform(10, 130), 2),  
        'width': 0.0001,                 
        'length': 0.0001,               
        'speed_range': (0, 0.1),         
        'radius_detection': 100,        
        'time_range': 40,               
        'num_data_points' : 5000         
    }

}

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--object', type=str, required=True, 
                   choices=['Drone1', 'Drone2', 'Drone3', 'No_Drone'],
                   help='Type of flying object to simulate')
parser.add_argument('--alpha_0', type=float, required=True,
                   help='Initial alpha angle in radians when t=0s')
args = parser.parse_args()
# Select flying object type from command line
flying_object = args.object
obj_specs = FLYING_OBJECTS[flying_object]


# Constants
hSat = 550e3  # Satellite height in meters
rE = 6371e3  # Earth radius in meters
TSat = 5731.1  # Satellite period in seconds
c = 3e8  # Speed of light in m/s
freq = 11e9  # Frequency in Hz. around 11.075, 11.325,11.575GHz for Ku-Band Beacons Starlink.
wavelength = c/freq  # Wavelength in meters
A_mag = 1.0                  # Magnitude of complex amplitude |A|
A_phase = np.pi/4            # Phase angle θ (radians)
A = A_mag * np.exp(1j * A_phase)  # Complex amplitude A = |A|e^{iθ}
k = 2*np.pi/wavelength
h = obj_specs['fligh_height']  # Use average height of selected flying object
SNR_value = np.random.uniform(35, 40)
SNR_linear = 10 ** (SNR_value / 10)  # Convert dB to linear scale
speed_min, speed_max = obj_specs['speed_range'] #speed of flying object in m/s
Overall_speed = np.random.uniform(speed_min, speed_max)  # Random speed within range in m/s
D_flying = np.random.choice([-1, 1])  # Random direction of flying object
D_Sat = np.random.choice([-1, 1])  # Random direction of satellite
D_vertical = np.random.choice([-1, 1])  # flying object vertical movement direction
a_prime = obj_specs['width']  # Use width of selected flying object
b_prime = obj_specs['length']  # Use length of selected flying object
# antenna_aperture = 0.05 * 0.05  # m^2
x0 = np.random.uniform(-obj_specs['radius_detection'], obj_specs['radius_detection'])   # x coordinate of the flying object when t=0s            
''' determine flying object's speed randomly in x, y ,z direction'''
while True:
    Polar_angle = np.random.uniform(0, np.pi / 2)  # Polar angle (0 to pi/2)
    phi = np.random.uniform(0, np.pi / 2)    # Azimuthal angle (0 to pi/2)
    cot_theta = 1 / np.tan(Polar_angle)  # cot(theta) = cos(theta) / sin(theta)
    if 3*cot_theta < np.cos(phi) and 3*cot_theta < np.sin(phi):
        break  # Exit the loop if conditions are satisfied
speed_x = np.random.choice([-1, 1]) *Overall_speed * np.sin(Polar_angle) * np.cos(phi)       
speed_y = np.random.choice([-1, 1]) *Overall_speed  * np.sin(Polar_angle) * np.sin(phi)
speed_z = np.random.choice([-1, 1]) *Overall_speed  * np.cos(Polar_angle)     
print("speed_x, speed_y and speed_z", speed_x, speed_y,speed_z)
''''''

t_min =-1*obj_specs['time_range']
t_max=obj_specs['time_range']
t = np.linspace(t_min, t_max, 1600)

# Initial alpha from user input
alpha_0 = args.alpha_0
theta_0 = np.pi/2 - np.arctan2((hSat + rE)*np.sin(alpha_0), 
                               (hSat + rE)*np.cos(alpha_0) - rE)
theta_0_degree = np.degrees(theta_0)

# Calculate y_0 bounds safely
sqrt_val = (FLYING_OBJECTS[flying_object]['radius_detection'])**2 - x0**2
y_0_min = h/np.tan(theta_0) - np.sqrt(sqrt_val)  # Lower bound of y_0
y_0_max = h/np.tan(theta_0) + np.sqrt(sqrt_val)  # Upper bound of y_0
# Generate y_0 only if bounds are valid
y_0 = np.random.randint(y_0_min, y_0_max + 1)  # Value of l when t=0. refer to figure 5 of David's paper " passive positioning of flying object with microwave signals from LEO satellites:..2020"

# Initialize arrays
intensity = np.zeros_like(t)
theta_values = np.zeros_like(t)
y_values = np.zeros_like(t)
z_values = np.zeros_like(t)
x_values = np.zeros_like(t)
y_L_values = np.zeros_like(t)
distances=np.zeros_like(t)
# Calculate intensity 
for i, time in enumerate(t):

    # Calculate alpha as function of time
    alpha = alpha_0 + D_Sat*(2*np.pi/TSat) * time

    # Formula 15: Elevation angle calculation
    theta = np.pi/2 - np.arctan2((hSat + rE)*np.sin(alpha), 
                               ((hSat + rE)*np.cos(alpha) - rE))
    # Store theta value
    theta_values[i] = theta

    # Equation 16  of "passive positioning of flying object with microwave signals from LEO satellites:..2020"
    dSat = np.sqrt((hSat + rE)**2 - (rE*np.cos(theta))**2) - rE*np.sin(theta)
    
    # Calculate l based on speed and time
    l = y_0 + speed_y * time
    y_values[i] = l  # Store l value
    h=obj_specs['fligh_height']+speed_z*time 

    # Calculate z, Equation 17 of "passive positioning of flying object with microwave signals from LEO satellites:..2020"
    z = (h/np.sin(theta) + 
         (l - h/np.tan(theta))*np.cos(theta))
    z_values[i] = h   # Store z value
    # Calculate y_L. Equation 18 of "passive positioning of flying object with microwave signals from LEO satellites:..2020"
    y_L = (l - h/np.tan(theta))*np.sin(theta)
    y_L_values[i]=y_L
    # Slab dimensions in x and y directions
    a = a_prime
    b = b_prime 
    
    # Equation 7: Compute signal intensity
    eps = 1e-10  # Small value to prevent division by zero
    x = x0 + speed_x * time  # Update x position based on speed
    x_values[i] = x  # Store x value
    distances[i] = np.sqrt(x**2 + y_L**2)  # Calculate distance
    
    # Plot and save distance data when simulation completes
    # fig_dist = plt.figure(figsize=(1.68, 1.68), dpi=200)
    # ax_dist = fig_dist.add_subplot(111)
    # ax_dist.plot(t, distances, linewidth=1, color='black')

min_dist_idx = np.argmin(distances)
min_dist_time = t[min_dist_idx]
print(f"\nMinimum distance occurs at time: {min_dist_time:.2f}s  {distances[min_dist_idx]:.2f}m")

# Highlight the 1s before and after minimum distance
# ax_dist.axvspan(min_dist_time-1, min_dist_time+1, color='yellow', alpha=0.3)
# ax_dist.set_ylabel('')
# ax_dist.set_yticks([])
# ax_dist.set_xticks([])
# plt.tight_layout()
# fig_dist.savefig('distances.png', dpi=300, bbox_inches='tight')

# Analyze 1s before and after minimum distance
time_range = (min_dist_time - 1, min_dist_time + 1)
mask = (t >= time_range[0]) & (t <= time_range[1])
print(f"Distance values from {time_range[0]:.2f}s to {time_range[1]:.2f}s:")
# plt.close()

angle=np.arctan(y_L_values[min_dist_idx]/x_values[min_dist_idx])





'''calculate intensity for 2s around minimum distance'''
t = np.linspace(min_dist_time - 1, min_dist_time + 1, obj_specs['num_data_points'])
intensity = np.zeros_like(t)
theta_values = np.zeros_like(t)
y_values = np.zeros_like(t)
z_values = np.zeros_like(t)
x_values = np.zeros_like(t)
x1_values = np.zeros_like(t)
y1_values = np.zeros_like(t)
y_L_values = np.zeros_like(t)
distances = np.zeros_like(t)
# Calculate intensity 
for i, time in enumerate(t):
    # Calculate alpha as function of time
    alpha = alpha_0 + D_Sat*(2*np.pi/TSat) * time 
    
    # Formula 15: Elevation angle calculation
    theta = np.pi/2 - np.arctan2((hSat + rE)*np.sin(alpha), 
                               ((hSat + rE)*np.cos(alpha) - rE))
    theta_values[i] = theta

    # Equation 16: Calculate dSat
    dSat = np.sqrt((hSat + rE)**2 - (rE*np.cos(theta))**2) - rE*np.sin(theta)
    
    # Calculate l based on speed and time
    l = y_0 + speed_y * time                        #   update y values to incoporate aperture along y axis
    # y_values[i] = l                                 
    h = obj_specs['fligh_height'] + speed_z*time 
    # Equation 17: Calculate z
    z = (h/np.sin(theta) + (l - h/np.tan(theta))*np.cos(theta))
    # z_values[i] = h                                
    
    # Equation 18: Calculate y_L
    y_L = (l - h/np.tan(theta))*np.sin(theta)
    # y_L_values[i] = y_L
    
    # Slab dimensions
    a = a_prime
    b = b_prime

    # Update x position
    x = x0 + speed_x * time            #incoporate aperture along x axis
    # x_values[i] = x
    distances[i] = np.sqrt(x**2 + y_L**2)

    # Calculate and store x1/y1 coordinates
    x1=x*np.cos(angle) + y_L*np.sin(angle)
    y1=-x*np.sin(angle) + y_L*np.cos(angle)
    # x1_values[i] = x1
    # y1_values[i] = y1
    
    # Create grid for aperture integration (-0.1m to 0.1m around center)
    grid_points = np.linspace(-0.05, 0.05, 3)  
    I_z_sum = 0
    count = 0
    for dx in grid_points:
        for dy in grid_points:
            # Calculate coordinates for this grid point
            x1_grid = x1 + dx
            y1_grid = y1 + dy
            # Compute signal intensity at grid point. incoporate noise.
            beta = np.pi * a * x1_grid / (wavelength * z)
            gamma = np.pi * b * y1_grid / (wavelength * z)
            C = (a * b) / (wavelength * z) * np.sinc(beta / np.pi) * np.sinc(gamma / np.pi)
            phi = (k * (x1_grid**2 + y1_grid**2 + 2 * z**2)) / (2 * z)
            sigma_n = A_mag / np.sqrt(SNR_linear)
            n_r = np.random.normal(0, sigma_n/np.sqrt(2))
            n_i = np.random.normal(0, sigma_n/np.sqrt(2))
            n = n_r + 1j * n_i
            I_z_point = (
                np.abs(A)**2 * (1 + C**2 - 2 * C * np.sin(phi)) 
                + (n_r**2 + n_i**2)
                + 2 * np.real(A * (1 + 1j * C * np.exp(1j * phi)) * np.conj(n))
            )
            I_z_sum += I_z_point
            count += 1
    I_z = I_z_sum / count    # Calculate average intensity over aperture

    
    # Equation 8: Calculate measured intensity with free space path losss
    c_fspl = 12  # Free space path loss constant. Does not affect the relative intensity
    fspl = c_fspl * (dSat**2)
    intensity_value=I_z / fspl 
    if intensity_value <= 0:
        intensity_value = 1e-10  # Small positive value to avoid log10(0)
    intensity[i] = 10 * np.log10(intensity_value)





''' Plot intensity for 2s around minimum distance'''
fig_intensity = plt.figure(figsize=(2.56, 2.56), dpi=100)
ax_intensity = fig_intensity.add_subplot(111)
ax_intensity.plot(t, intensity, linewidth=1.5, color='k', linestyle='-', rasterized=True)

ax_intensity.set_ylabel('')
ax_intensity.set_yticks([])
ax_intensity.set_xticks([])
plt.tight_layout(pad=0)  # Remove extra padding

# Determine output folder (train: 70%, test: 20%, val: 10%)
rand_val = np.random.rand()
if rand_val < 0.7:
    output_dir = 'train_images'
elif rand_val < 0.9:
    output_dir = 'test_images'
else:
    output_dir = 'val_images'
# Create flying object subfolder if it doesn't exist
class_dir = os.path.join(output_dir, flying_object)
os.makedirs(class_dir, exist_ok=True)

# Save x=0 plot with timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_file_x = os.path.join(class_dir, f'{flying_object}_x0_{timestamp}.png')
fig_intensity.savefig(output_file_x, bbox_inches='tight', pad_inches=0)  # Remove dpi and extra padding
print(f"Intensity plot saved to {output_file_x}")
plt.close(fig_intensity)
