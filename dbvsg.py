import matplotlib.pyplot as plt
import numpy as np

# Constants
e = 1.60217662e-19  # Elementary charge in C

# Define finer ranges for electron and ion velocities with the same range
velocities_e = np.linspace(1e3, 1e7, 1000)  # from 1e5 to 1e7 m/s in 50 steps
velocities_i = np.linspace(1e3, 1e7, 1000)  # from 1e5 to 1e7 m/s in 50 steps

# Arrays to store the results
single_precision_errors_high_density_same_range = []


# Define ranges for densities and velocities
high_density = 1e25  # Assume high density for simplicity

# Function to compute current density in single precision
def compute_current_density_single(ne, ni, ve, vi):
    ve_single = np.float32(ve)
    vi_single = np.float32(vi)
    ne_single = np.float32(ne)
    ni_single = np.float32(ni)
    e = np.float32(1.60217662e-19)
    mp1= np.float32(ne_single * ve_single)
    mp2= np.float32(ni_single * vi_single)
    J_single = e * (mp1 - mp2)
    return J_single

# Function to compute current density in double precision
def compute_current_density_double(ne, ni, ve, vi):
    ve_db = np.float64(ve)
    vi_db = np.float64(vi)
    ne_db = np.float64(ne)
    ni_db = np.float64(ni)
    e = np.float64(1.60217662e-19)
    mp1= np.float64(ne_db * ve_db)
    mp2= np.float64(ni_db * vi_db)
    J_double = e * (mp1 - mp2)
    return J_double

# Compute current density for the selected high density and varying electron and ion velocities in the same range
for ve in velocities_e:
    for vi in velocities_i:
        ni = high_density  # Assume quasi-neutrality for simplicity
        ne = high_density
        J_single = compute_current_density_single(ne, ni, ve, vi)
        J_double = compute_current_density_double(ne, ni, ve, vi)
        error = abs((J_double - J_single)/(J_double+1))
        single_precision_errors_high_density_same_range.append((ve, vi, error))

# Convert results to a numpy array for easier manipulation and plotting
single_precision_errors_high_density_same_range = np.array(single_precision_errors_high_density_same_range)

# Reshape the errors for a 2D heatmap plot
ve_vals = single_precision_errors_high_density_same_range[:, 0].reshape(len(velocities_i), len(velocities_e))
vi_vals = single_precision_errors_high_density_same_range[:, 1].reshape(len(velocities_i), len(velocities_e))
errors = single_precision_errors_high_density_same_range[:, 2].reshape(len(velocities_i), len(velocities_e))

# Plotting
plt.figure(figsize=(12, 8))

# Create a 2D heatmap
plt.contourf(ve_vals, vi_vals, errors, levels=50, cmap='RdBu_r')
plt.colorbar(label='Relative Error')
plt.xlabel('Electron Velocity (m/s)')
plt.ylabel('Ion Velocity (m/s)')
plt.title(f'Relative Error in Single Precision at Fixed High Density = {high_density} m^-3')

plt.show()



