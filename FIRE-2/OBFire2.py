# Import necessary libraries
import numpy as np
import scipy.integrate as integrate
import scipy.special as special
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import curve_fit
import pandas as pd

# Define the time intervals (in years) for different mass loss phases
t_ii = np.array([1*10**6, 3.5*10**6, 100*10**6])

# Define the mass loss rate for OB-AGB stars as a function of time (t in years) and metallicity (Z)
def OB_AGB_mass_loss(t, Z):
    # Define conditions for different time intervals
    conditions = [
        (t >= 0) & (t < t_ii[0]),   # Early phase
        (t_ii[0] <= t) & (t < t_ii[1]),  # Intermediate phase
        (t_ii[1] <= t) & (t < t_ii[2]),  # Advanced phase
        (t_ii[2] <= t)  # Late phase
    ]

    # Corresponding mass loss rates for each time interval (in units of M_dot/M)
    choices = [
        4.763*(0.01 + Z/0.013),  # Early phase: constant mass loss rate
        4.763 * (0.01 + Z / 0.013) * np.power(t / 10**6, 1.45 + 0.8 * np.log(Z / 0.013)),  # Intermediate phase: mass loss scales with time
        29.4 * (t / (3.5*10**6)) ** -3.25 + 0.0042,  # Advanced phase: strong decline in mass loss
        0  # Late phase: no mass loss
    ]
    return np.select(conditions, choices)  # Select appropriate mass loss based on time intervals


# Load data for Supernova AGB models with Z=0.02
sn_agb_mar = pd.read_csv('AGB_SN_mari_Z_0.02csv')
t_mar = np.array(sn_agb_mar["t"])  # Extract the time data from the dataset

# Define a range of metallicities (for future use)
z = np.array([0.01, 0.05, 0.1, 1])

# Define the time array (logarithmic scale), then append imported data to it
t = np.linspace(1*10**5, 3.22044362*10**6, 100)  # Generate time points
t = np.append(t, t_mar)  # Append imported time data to generated time array

# Calculate the mass loss rate (AGBOB_loss_rate) for each time point t at Z = 0.02
AGBOB_loss_rate = np.array([OB_AGB_mass_loss(t_i, 0.02) for t_i in t]) / 10**9  # Scale to Gyrs^-1

# Plot the continuous mass loss rate as a function of time
plt.plot(t, AGBOB_loss_rate)
plt.xscale('log')  # Logarithmic scale for time
plt.yscale('log')  # Logarithmic scale for mass loss rate
plt.xlabel('Time (log scale)')
plt.ylabel('M_dot/M (Gyrs)^-1')
plt.title('Continuous Mass Loss')
plt.show()

# Calculate the cumulative mass loss over time using trapezoidal integration
comulative_OBAGB = np.zeros(len(t))
for i in range(len(t) - 1):
    comulative_OBAGB[i + 1] = (AGBOB_loss_rate[i + 1] + AGBOB_loss_rate[i]) * (t[i + 1] - t[i]) / 2 + comulative_OBAGB[i]

# Plot the cumulative mass loss over time
plt.plot(t, comulative_OBAGB)
plt.xscale('log')  # Logarithmic scale for time
plt.xlabel('Time (log scale)')
plt.ylabel('Cumulative Mass Loss (M_dot/M)')
plt.title('Cumulative Mass Loss')
plt.show()

# Save the results to a CSV file
df = pd.DataFrame({'t': t, 'AGB+OB': comulative_OBAGB, 'AGB+OB_rate': AGBOB_loss_rate})
df.to_csv('../COnfronto/wind_loss_AGB_OBfire2_onlyOB.csv', index=False, header=True)

# Define an updated mass loss function with an additional late-phase term
def OB_AGBBB_mass_loss(t, Z):
    conditions = [
        (t >= 0) & (t < t_ii[0]),   # Early phase
        (t_ii[0] <= t) & (t < t_ii[1]),  # Intermediate phase
        (t_ii[1] <= t) & (t < t_ii[2]),  # Advanced phase
        (t_ii[2] <= t)  # Late phase
    ]
    
    # Updated choices with non-zero mass loss in the late phase
    choices = [
        4.763 * (0.01 + Z / 0.013),  # Early phase
        4.763 * (0.01 + Z / 0.013) * np.power(t / 10**6, 1.45 + 0.8 * np.log(Z / 0.013)),  # Intermediate phase
        29.4 * (t / (3.5*10**6)) ** -3.25 + 0.0042,  # Advanced phase
        0.42 * ((t / (1000*10**6)) ** -1.1) / (19.81 - np.log(t/10**6))  # Late phase: declining mass loss
    ]
    return np.select(conditions, choices)

# Calculate the updated mass loss rate (AGBBOB_loss_rate) for Z = 0.013
AGBBOB_loss_rate = np.array([OB_AGBBB_mass_loss(t_i, 0.013) for t_i in t]) / 10**9  # Scale to Gyrs^-1

# Save the updated results to a CSV file
df = pd.DataFrame({'t': t, 'AGB+OB': AGBBOB_loss_rate, 'AGB+OB_rate': AGBOB_loss_rate})
df.to_csv('../COnfronto/wind_loss_AGB_OBfire2_onlyOB.csv', index=False, header=True)

# Plot the updated mass loss rate as a function of time
plt.plot(t, AGBBOB_loss_rate)
plt.xscale('log')  # Logarithmic scale for time
plt.yscale('log')  # Logarithmic scale for mass loss rate
plt.xlabel('Time (log scale)')
plt.ylabel('M_dot/M (Gyrs)^-1')
plt.title('Updated Continuous Mass Loss')
plt.show()




