import numpy as np
import scipy.integrate as integrate
import scipy.special as special
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import curve_fit
from scipy.integrate import quad, trapz, simps
from numpy import interp
import csv
import pandas as pd

# Enabling matplotlib for interactive plotting
get_ipython().run_line_magic('matplotlib', 'qt')


# Function to compute the Initial Mass Function (IMF)
def IMF(m):
    # Define conditions based on mass ranges
    conditions = [
        m <= 0.08,  # Low-mass stars
        (0.08 < m) & (m <= 0.5),  # Intermediate mass
        m > 0.5  # High-mass stars
    ]

    # Define IMF for the different mass ranges
    choices = [
        m * m**(-0.3),
        m * 0.08 * m**(-1.3),
        m * 0.04 * m**(-2.3)
    ]

    # Apply the conditions to compute IMF
    return np.select(conditions, choices)


# Constants
t_I = 37.53  # Characteristic time for SNe Ia (in Myrs)
psi_I = -1.1  # Power law exponent for SNe Ia

# Function to compute the SNe Ia rate based on time
def SnIa_rate(t):
    # Define time conditions before and after characteristic time t_I
    conditions = [
        (0 <= t) & (t < t_I),  # Before t_I
        (t_I <= t)  # After t_I
    ]

    # Define the SNe Ia rate for each condition
    choices = [
        0,  # No SNe Ia before t_I
        (5.3e-8 + 1.6e-5 * np.exp(-((t - 50.) / 10.)**2 / 2)) / 1e6  # After t_I
    ]
    
    # Apply the conditions to compute the rate
    return np.select(conditions, choices)


# Characteristic times and amplitude factors for SNe II
t_II = np.array([3.401, 10.37, 37.53])  # Times in Myrs
a_II = np.array([0.39, 0.51, 0.18])  # Amplitude factors in Gyr/M☉

# Power-law exponents between time intervals
psi_II = np.array([
    (np.log(a_II[1] / a_II[0])) / (np.log(t_II[1] / t_II[0])),
    (np.log(a_II[2] / a_II[1])) / (np.log(t_II[2] / t_II[1]))
])

# Function to compute the SNe II rate based on time
def SnII_rate(t):
    # Define time conditions for different phases of SNe II
    conditions = [
        (t >= 0) & (t < t_II[0]),  # Before first characteristic time
        (t_II[0] <= t) & (t < t_II[1]),  # Between t_II[0] and t_II[1]
        (t_II[1] <= t) & (t < t_II[2]),  # Between t_II[1] and t_II[2]
        (t_II[2] <= t)  # After t_II[2]
    ]

    # Define the SNe II rate for each condition
    choices = [
        0,  # No SNe II before first characteristic time
        5.408e-4 / 1e6,  # Between t_II[0] and t_II[1]
        2.516e-4 / 1e6,  # Between t_II[1] and t_II[2]
        0  # No SNe II after t_II[2]
    ]
    
    # Apply the conditions to compute the rate
    return np.select(conditions, choices)


# Function to compute the total supernova rate (SNe Ia + SNe II)
def total_Sn_rate(t):
    return SnII_rate(t) + SnIa_rate(t)


# Define timesteps and compute logarithmic time scale
tn = np.linspace(1, 10000, num=1000)
dt = np.log10(tn[-1] / tn[0]) / len(tn)
logt = np.log10(tn[0]) + np.arange(len(tn)) * dt 
t = 10.**logt  # Transforming back to linear scale

# Calculate supernova rates for each time step
rate = np.array([total_Sn_rate(t_i) for t_i in t]) 
rate_snI = np.array([SnIa_rate(t_i) for t_i in t])
rate_snII = np.array([SnII_rate(t_i) for t_i in t])

# Plotting the supernova rate over time
plt.figure(figsize=(7.7, 5.2))
plt.plot(t, rate)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('t (Myr)')
plt.ylabel('SNe rate (R/M☉/yr)')
plt.title('Supernova Rate')
plt.show()


### Cumulative supernova numbers ###
comulative_snI = np.zeros(len(t))
comulative_snII = np.zeros(len(t))
lograteSnI = np.zeros(len(t))
lograteSnII = np.zeros(len(t))

# Calculate logarithmic rates for SNe Ia and SNe II
for i in range(len(t)):
    lograteSnI[i] = SnIa_rate(t[i]) * t[i]
    lograteSnII[i] = SnII_rate(t[i]) * t[i]

# Compute the cumulative rates using trapezoidal integration
for i in range(len(t)):
    slice_ratesnI = lograteSnI[:i]
    slice_ratesnII = lograteSnII[:i]
    comulative_snI[i] = np.trapz(slice_ratesnI, dx=dt) * np.log(10.)
    comulative_snII[i] = np.trapz(slice_ratesnII, dx=dt) * np.log(10.)

# Plotting the cumulative supernova numbers
plt.figure(figsize=(7.7, 5.2))
plt.plot(t, comulative_snII * 1e6, label='SNe II')
plt.plot(t, comulative_snI * 1e6, label='SNe Ia')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('t (Myr)')
plt.ylabel('Cumulative SNe number (per initial M☉)')
plt.title('Cumulative Supernova Numbers')
plt.legend()
plt.xlim(1, 10**4)
plt.ylim(10**-5, 10**-1)
plt.show()


### Mass ejection over time ###
plt.figure(figsize=(7.7, 5.2))
plt.plot(t, 10.5 * rate_snII, label='SNe II Ejection')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('t (Myr)')
plt.ylabel('Ejected mass (M☉)')
plt.title('Mass Ejection over Time')
plt.show()


### Cumulative mass ejection ###
comulative_SNII = np.zeros(len(t))
comulative_SNI = np.zeros(len(t))
lograteSNII = np.zeros(len(t))
lograteSNI = np.zeros(len(t))

# Compute the mass ejection rate for SNe II and SNe Ia
for i in range(len(t)):
    lograteSNII[i] = 10.5 * SnII_rate(t[i]) * t[i]
lograteSNI = 1.4 * rate_snI * t

# Compute cumulative mass ejection using trapezoidal integration
for i in range(len(t)):
    slice_ratesnI = lograteSNI[:i]
    slice_ratesnII = lograteSNII[:i]
    comulative_SNI[i] = np.trapz(slice_ratesnI, dx=dt) * np.log(10.)
    comulative_SNII[i] = np.trapz(slice_ratesnII, dx=dt) * np.log(10.)

# Plotting cumulative mass ejection over time
plt.figure(figsize=(7.7, 5.2))
plt.plot(t, comulative_SNII * 1e6)
plt.xscale('log')
plt.xlabel('t (Myr)')
plt.ylabel('Cumulative Ejected Mass (M☉)')
plt.title('Cumulative Mass Ejection')
plt.show()


### Saving data to CSV ###
# Save cumulative supernova numbers to CSV
df = pd.DataFrame({'t': t, 'SNIa': comulative_SNI * 1e6, 'SNII': comulative_SNII * 1e6})
df.to_csv('sn_numberfire2.csv', index=False, header=True)

# Save supernova rate and mass ejection to CSV
dp = pd.DataFrame({'t': t, 'SNI': rate_snI * 1.4, 'SNII': 10.5 * rate_snII})
dp.to_csv('../COnfronto/sn_rate_ejfire2.csv', index=False, header=True)


### Element yields calculation ###



# Define the metallicity z and calculate the scaling factor N
z = 0.0127  # Metallicity
N = max(z / 0.0127, 1.65)  # Scaling factor to adjust for metallicity

# List of elements and their yields per supernova event
elements = ["He", "C", "N", "O", "Ne", "Mg", "Si", "S", "Ca", "Fe"]
yields = np.array([3.87, 0.133, 0.0479 * N, 1.17, 0.30, 0.0987, 0.0933, 0.0397, 0.00458, 0.0741])

# Dictionary to store ejected mass for each element
elements_ej = {}

# Loop over each element and compute cumulative ejected mass
for j in range(len(elements)):
    elements_ej[f"{elements[j]}"] = np.zeros(len(t))  # Initialize array for each element
    for i in range(len(t)):
        # Calculate the cumulative ejected mass for each time step and element
        elements_ej[f"{elements[j]}"][i] = (yields[j] / 10.5) * comulative_SNII[i]

# Plotting the ejected mass of each element over time
plt.figure(figsize=(7.7, 5.2))
for j in elements:
    plt.plot(t, elements_ej[f"{j}"] * 1e6, label=f"{j}")  # Plot each element

# Setting the plot to logarithmic scale for x-axis
plt.xscale('log')
plt.xlabel('t (Myr)')
plt.ylabel('Ejected mass (M☉)')
plt.title('Elemental Mass Ejection Over Time')
plt.legend(loc="upper right")  # Adding legend for element labels

# Customizing the plot axis and ticks
ax = plt.gca()
ax.yaxis.set_ticks_position('both')  # Show ticks on both sides
ax.tick_params(axis='y', which='both', direction='in')

# Display the plot
plt.show()


