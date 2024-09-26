#!/usr/bin/env python
# coding: utf-8

# In[58]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import h5py
import warnings
from scipy.integrate import quad
# Filter out RuntimeWarnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import numpy as np
import scipy.integrate as integrate
import scipy.special as special
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import curve_fit
import pandas as pd
import h5py
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import multiprocessing
from scipy import integrate
from scipy.interpolate import interp2d
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import RectBivariateSpline
from scipy.integrate import quad
import h5py

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsmath}",
  
    "font.serif": ["Computer Modern Roman"]
})


# In[59]:



def IMF(m):
    conditions = [
        m<=0.08, 
        (0.08<m) &(m<=0.5),
        m>0.5
        
    ]

    choices = [
        (1/0.17820888079162378)*m**(-0.3),
        (1/0.17820888079162378)*0.08*m**(-1.3),
        (1/0.17820888079162378)*0.04*m**(-2.3)
    ]

    return np.select(conditions, choices)

    
    


# In[60]:


imf=quad(IMF, 8, 100)
print(imf)


# In[61]:


masses1 = [0.8, 0.9,  1 ,   1.1,   1.2,   1.3,   1.4,
         1.5,   1.6,   1.7,   1.8,   1.9,   2, 2.1,  2.2 ]  ##only in mesa 

masses2=[2.3, 2.4, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7] ##I use this with both mesa and ov0.5 tracks

Z = np.array([0.0004, 0.004, 0.008, 0.02])

lifetime1 = {}
windloss1 = {}
starmass1 = {}
stars1 = {}
state1={}
timel_mesa = np.zeros((len(Z), len(masses1)))  # Initialize timel as a matrix


####ONLY FOR MESA TRACKS####
for j_index, j in enumerate(Z):
    lifetime1[j] = {}
    windloss1[j] = {}
    starmass1[j] = {}
    stars1[j] = {}
    state1[j]={}
    for i_index, i in enumerate(masses1):
        stars1[j][i] = pd.read_csv(f"param/Test/mesa_tracks/z_{j}/{i}/output_0.csv")
        lifetime1[j][i] = np.array(stars1[j][i]["Worldtime"])
        windloss1[j][i] = np.array(stars1[j][i]["dMdt"])
        starmass1[j][i] = np.array(stars1[j][i]["Mass"])
        state1[j][i]=np.array(stars1[j][i]["Phase"])
        # Calculate and store the last worldtime for each mass and metallicity
        timel_mesa[j_index][i_index] = np.array(stars1[j][i]["Worldtime"])[-1]
        

        
        
        
####FOR all TRACKS####
lifetime2 = {}
windloss2 = {}
starmass2 = {}
stars2 = {}
state2={}
        
        
model=["ov04", "ov05", "mesa"]
lifetime2_arrays={}
timel={}
windloss2_arrays={}
starmass2_arrays={}
state2_arrays={}
windloss={}
state={}
starmass={}
lifetime={}
timel_ov5={} 


for k in model:
    timel_ov5[k]={} 
    lifetime2[k] = {}
    windloss2[k] = {}
    starmass2[k] = {}
    stars2[k] = {}
    state2[k]={}
    
    for j_index, j in enumerate(Z):
        timel_ov5[k][j]={}
        lifetime2[k][j] = {}
        windloss2[k][j] = {}
        starmass2[k][j] = {}
        stars2[k][j] = {}
        state2[k][j]={}
        for i_index, i in enumerate(masses2):
            stars2[k][j][i] = pd.read_csv(f"param/Test/{k}_tracks/z_{j}/{i}/output_0.csv")
            lifetime2[k][j][i] = np.array(stars2[k][j][i]["Worldtime"])
            windloss2[k][j][i] = np.array(stars2[k][j][i]["dMdt"])
            starmass2[k][j][i] = np.array(stars2[k][j][i]["Mass"])
            state2[k][j][i]=np.array(stars2[k][j][i]["Phase"])
            # Calculate and store the last worldtime for each mass and metallicity
            timel_ov5[k][j][i] = np.array(stars2[k][j][i]["Worldtime"])[-1]

            
            
for k in model:
    masses=np.append(masses1, masses2)
    timel_ov5_combined = np.zeros((len(Z), len(masses2)))  # Initialize array for combining timel_ov5 values
    
    for j_index, j in enumerate(Z):
        # Assuming timel_ov5[k] is a dictionary with keys corresponding to Z
        timel_ov5_combined[j_index, :] = np.array([timel_ov5[k][j][i] for i in masses2])

    # Concatenate timel_mesa and timel_ov5_combined
    timel[k] = np.concatenate((timel_mesa, timel_ov5_combined), axis=1)
    lifetime1_arrays = np.array([[lifetime1[j][i] for i in masses1] for j in Z], dtype=object)
    lifetime2_arrays[k] = np.array([[lifetime2[k][j][i] for i in masses2] for j in Z], dtype=object)

    # Concatenate the arrays
    lifetime[k] = np.concatenate((lifetime1_arrays, lifetime2_arrays[k]), axis=1)

    windloss1_arrays = np.array([[windloss1[j][i] for i in masses1] for j in Z], dtype=object)
    starmass1_arrays = np.array([[starmass1[j][i] for i in masses1] for j in Z], dtype=object)
    state1_arrays = np.array([[state1[j][i] for i in masses1] for j in Z], dtype=object)

    # Extract arrays from windloss2, starmass2, and state2 dictionaries
    windloss2_arrays[k] = np.array([[windloss2[k][j][i] for i in masses2] for j in Z], dtype=object)
    starmass2_arrays[k] = np.array([[starmass2[k][j][i] for i in masses2] for j in Z], dtype=object)
    state2_arrays[k] = np.array([[state2[k][j][i] for i in masses2] for j in Z], dtype=object)

    # Concatenate the arrays
    windloss[k] = np.concatenate((windloss1_arrays, windloss2_arrays[k]), axis=1)
    starmass[k] = np.concatenate((starmass1_arrays, starmass2_arrays[k]), axis=1)
    state[k] = np.concatenate((state1_arrays, state2_arrays[k]), axis=1)



# In[62]:


np.shape(timel_mesa)
np.shape(starmass)
np.shape(state)


# In[ ]:





# In[63]:


#CREATE THE NEW LIFETIME FUNCTION
plt.figure(figsize=(7.7, 5.2))
lifetimefunc = {}
masstimefunc = {}


for c_index, c in enumerate(model):
    lifetimefunc[c_index] = {}
    masstimefunc[c_index] = {}
    for j_index, j in enumerate(Z):
        log_masses = np.log10(masses)
        log_timel = np.log10([timel[c][j_index][i] for i in range(len(masses))])

        lifetimefunc[c_index][j_index] = interp1d(log_masses, log_timel, kind="linear", fill_value="extrapolate")
        masstimefunc[c_index][j_index] = interp1d(log_timel, log_masses, kind="linear", fill_value="extrapolate")


plt.scatter(masses, timel["mesa"][2][:] * 10**6, label=f'Z = {Z[2]}')


plt.plot(masses,  10**lifetimefunc[0][2](np.log10(masses))*10**6, label=f'Z = {Z[2]}')


plt.xscale("log")
plt.yscale('log')
plt.xlabel('Mass M\_sol')
plt.ylabel('t [yr]')
plt.legend()
plt.show()


# In[64]:


dM = {}
dt = {}
winds = {}
totwinds = {}



for k in model:
    dM[k] = {}
    dt[k] = {}
    winds[k] = {}
    totwinds[k] = {}
    for j in range(len(Z)):
        dM[k][j] = {}
        dt[k][j] = {}
        winds[k][j] = {}
        totwinds[k][j] = {}

        for i in range(len(masses)):
            dM[k][j][i] = np.zeros(len(starmass[k][j][i]))
            dt[k][j][i] = np.zeros(len(lifetime[k][j][i]))
            winds[k][j][i] = np.zeros(len(lifetime[k][j][i]))
            totwinds[k][j][i] = 0

            for c in range(len(starmass[k][j][i]) - 1):
                dM[k][j][i][c + 1] = starmass[k][j][i][c + 1] - starmass[k][j][i][c]

            for c in range(len(lifetime[k][j][i]) - 1):
                dt[k][j][i][c + 1] = lifetime[k][j][i][c + 1] - lifetime[k][j][i][c]
                winds[k][j][i][c] = dt[k][j][i][c] * windloss[k][j][i][c]  # Corrected indexing
                totwinds[k][j][i] += winds[k][j][i][c + 1]


# In[ ]:





# In[65]:


mass_sn1 = {}
tot_winds1 = {}
rem_mass = {}
kik={}
totwinds_ms={}

for k in model:
    mass_sn1[k] = np.zeros((len(Z), len(masses)))
    rem_mass[k] = np.zeros((len(Z), len(masses)))
    totwinds_ms[k] = np.zeros((len(Z), len(masses)))
    kik[k] = np.zeros((len(Z), len(masses)))
    for j_index, j in enumerate(Z):
        for i_index, mass in enumerate(masses):
            for c in range(len(state[k][j_index][i_index]) - 1):
                if state[k][j_index][i_index][c] == "TerminalCoreHeBurning" and state[k][j_index][i_index][c + 1] == "ShellHeBurning":
                    mass_sn1[k][j_index][i_index] = starmass[k][j_index][i_index][c] - starmass[k][j_index][i_index][-1]
                    rem_mass[k][j_index][i_index] = starmass[k][j_index][i_index][-1]
                    totwinds_ms[k][j_index][i_index] = totwinds[k][j_index][i_index]
                    kik[k][j_index][i_index] = c


# In[69]:


plt.figure(figsize=(7.7, 5.2))  # Adjust figure size if needed

#for j_index, j in enumerate(Z):

for j_index in range(len(model)):
  
    plt.plot(masses, mass_sn1[f"{model[j_index]}"][3] / masses,  label=f'Track: {model[j_index]}')





plt.yscale('log')


plt.xticks(fontsize=17)
plt.yticks(fontsize=16)
plt.tick_params(axis='x', which='both', direction='in', pad=6)  # Adjust pad to move x tick labels lower
plt.tick_params(axis='y', which='both', direction='in')


plt.xlabel('M $[M_{\odot}]$', fontsize=18, labelpad=5)
plt.ylabel(r' Ejected mass efficency $f^S_{rec}(M, Z=0.02)$', fontsize=18,labelpad=7)
plt.tick_params(axis='both', direction='in', which='both')
ax = plt.gca()  # Get the current axes
ax.yaxis.set_ticks_position('both')  # Show ticks on both sides
ax.tick_params(axis='y', which='both', direction='in')
plt.legend(fontsize=13) 
plt.savefig(f'../../Images/SEVN/AGB_eff_sevn_sn.pdf', format='pdf')
plt.show()


# In[48]:



with h5py.File('../../CONFRONTOFINALE/AGB_Ioro_ov0.5_ms.hdf5', 'w') as f:
    # Create datasets
    dataset1 = f.create_dataset('Masses', data=masses, dtype='f')
    dataset2=f.create_dataset('Metallicities', data=Z, dtype='f')
    group = f.create_group('Yields')
    for k in model:
        group3 = group.create_group(f'{k}')
        for i in range(len(Z)):
            group2 = group3.create_group(f'Z_{Z[i]}')
            dataset = group2.create_dataset('Ejected_mass', data=mass_sn1[k][i][:])

with h5py.File('../../CONFRONTOFINALE/Lifetimes_ioroAGB_ov0.5.hdf5', 'w') as f:
    # Create datasets for masses and metallicities
    f.create_dataset('Masses', data=masses, dtype='f')
    f.create_dataset('Metallicities', data=Z, dtype='f')
    
    # Create group for models
    group = f.create_group('models')
    
    for k in model:
        group_model = group.create_group(k)
        
        # Create an empty matrix for lifetimes with shape (len(Z), len(masses))
        lifetimes_matrix = np.zeros((len(Z), len(masses)))
        
        for j_index, j in enumerate(Z):
            for i_index, i in enumerate(masses):
                lifetimes_matrix[j_index, i_index] = timel[k][j_index][i_index] * 10**6
        
        # Create dataset for lifetimes
        group_model.create_dataset('Lifetimes', data=lifetimes_matrix, dtype='f')
    


# In[ ]:





# In[11]:


dt = {}
winds = {}
totwinds = {}
dM = {}

# Calculate changes in mass (dM) and time intervals (dt) for each combination of mass and metallicity
for k in model:
    dM[k] = {}
    dt[k] = {}
    winds[k] = {}
    totwinds[k] = {}

    for j_idx in range(len(Z)):
        dM[k][j_idx] = {}
        dt[k][j_idx] = {}
        winds[k][j_idx] = {}
        totwinds[k][j_idx] = {}

        for i_idx in range(len(masses)):
            dM[k][j_idx][i_idx] = np.zeros(len(starmass[k][j_idx][i_idx]))
            dt[k][j_idx][i_idx] = np.zeros(len(lifetime[k][j_idx][i_idx]))
            winds[k][j_idx][i_idx] = np.zeros(len(lifetime[k][j_idx][i_idx]))
            totwinds[k][j_idx][i_idx] = 0

            for c in range(len(starmass[k][j_idx][i_idx]) - 1):
                dM[k][j_idx][i_idx][c + 1] = starmass[k][j_idx][i_idx][c + 1] - starmass[k][j_idx][i_idx][c]

            for c in range(len(lifetime[k][j_idx][i_idx]) - 1):
                dt[k][j_idx][i_idx][c + 1] = lifetime[k][j_idx][i_idx][c + 1] - lifetime[k][j_idx][i_idx][c]
                winds[k][j_idx][i_idx][c] = dt[k][j_idx][i_idx][c] * windloss[k][j_idx][i_idx][c]
                totwinds[k][j_idx][i_idx] += winds[k][j_idx][i_idx][c + 1]





sliced_lifetime = {}
sliced_winds = {}

sliced_lifetime2 = {}
sliced_winds2 = {}

for c, cc in enumerate(model):
    sliced_lifetime[c] = {}
    sliced_winds[c] = {}  
    sliced_lifetime2[c] = {}
    sliced_winds2[c] = {}

    for j_index, j in enumerate(Z):
        sliced_lifetime[c][j_index] = {}
        sliced_winds[c][j_index] = {}  
        sliced_lifetime2[c][j_index] = {}
        sliced_winds2[c][j_index] = {}

        for i_index, i in enumerate(masses):
            k = int(kik[cc][j_index][i_index])
            sliced_lifetime[c][j_index][i_index] = np.array(lifetime[cc][j_index][i_index][:k])
            sliced_winds[c][j_index][i_index] = np.array(winds[cc][j_index][i_index][:k])
        


# In[12]:


i = 1  # Specific mass

j=1
plt.plot(lifetime["ov04"][j][i][:] * 10**6, -winds["ov04"][j][i][:], label=f'Z = {j}')
#plt.plot(sliced_lifetime2[j][i] * 10**6, -sliced_winds2[j][i], label=f'Z = {j}')

plt.plot(sliced_lifetime[0][j][i][:] * 10**6, -sliced_winds[0][j][i][:], label=f'Z = {j}')

i=20
#plt.plot(sliced_lifetime[j][i] * 10**6, -sliced_winds[j][i], label=f'Z = {j}')
#plt.plot(lifetime[j][i] * 10**6, -winds[j][i], label=f'Z = {j}')
#plt.xticks(t)
plt.xscale("log")
plt.xscale("log")
plt.yscale('log')
plt.xlabel('Lifetime [yr]')
plt.ylabel('Change in Mass [M\_sun]')
plt.legend()
plt.title(f'Change in Mass vs. Lifetime for Mass {i}')
plt.show()


# In[13]:


t = np.logspace(  np.log10(10**9), (np.log10(100000)),3000)
t=np.flip(t)

t1=np.linspace(0, 100000, 100)
t=np.append(t1, t)


# In[14]:


wind = {}
wind_func={}
life={}
for c_index, c in enumerate(model):

    wind[c] = {}
    wind_func[c]={}
    for z_index, z in enumerate(Z):
        wind[c][z] = {}
        wind_func[c][z]={}
        for m_index, m in enumerate(masses):
            wind[c][z][m] = np.zeros(len(t))
            life = np.array(sliced_lifetime[c_index][z_index][m_index]) * 10**6
            wind_func[c][z][m]=()
            ##INTERPOLAZIONE SUL TEMPO   
            for i in range(len(life) - 1):
                for k in range(len(t) - 1):
                    
                    if life[i] <= t[k] <= life[i + 1]:
                        wind[c][z][m][k] = -(sliced_winds[c_index][z_index][m_index][i + 1] - sliced_winds[c_index][z_index][m_index][i]) / (life[i + 1] - life[i]) * (t[k] - life[i]) - sliced_winds[c_index][z_index][m_index][i]


# In[15]:



wind_array = np.zeros((len(model), len(Z), len(masses), len(t)))



for c_index, c in enumerate(model):
    for z_index, z in enumerate(Z):
        for m_index, m in enumerate(masses):
            for k, time in enumerate(t):
                wind_array[c_index, z_index, m_index, k] =np.array( wind[c][z][m][k])







# In[16]:


#plt.plot(sliced_lifetime[0.02][20]*10**6, -sliced_winds[0.02][20], label=f'Z = {j}')

for i in range(3):
    plt.plot(t, wind_array[i][3][1][:], label=f'Z = {Z[2]}')
#plt.plot(t, wind[j][20], label=f'Z = {j}')
plt.xscale("log")
plt.yscale('log')
plt.xlabel('mass')
plt.ylabel('Change in Mass [M\_sun]')
plt.xlim(1e7, 1e10)
plt.legend()
plt.title(f'Change in Mass vs. Lifetime for Mass {i}')
plt.show()


# In[17]:


for i in range(len(Z)):
    sli=np.array( wind_array[0, i, :, 2200])
#plt.plot(sliced_lifetime[j][i] * 10**6, -sliced_winds[j][i], label=f'Z = {j}')

    plt.plot(masses, sli, label=f'Z = {Z[i]}')
#plt.plot(masses, wind_array[2][300], label=f'Z = {j}')
#plt.xlim(10**7, 10**8)

#plt.xscale("log")
plt.xscale("log")
plt.yscale('log')
plt.xlabel('mass')
plt.ylabel('Change in Mass [M\_sun]')
plt.legend()
plt.title(f'Change in Mass vs. Lifetime for Mass {i}')
plt.show()


# In[55]:


mass=np.linspace(1, 7, 100)

intrp_mass={}
#this is the mass interpolation


for k, kk in enumerate(model):
    intrp_mass[k]={}
    for z in range(len(Z)):
            intrp_mass[k][z]={}
            for i in range(len(t)):
                    ej_mass=np.array(wind_array[k, z, :, i])
                    intrp_mass[k][z][i]=interp1d(np.log10(masses), np.log10(ej_mass), kind="linear")


# In[56]:


len_Z = len(Z)
len_t = len(t)
len_mass = len(mass)

# Create a 3D NumPy array filled with zeros
ej_wind_mass = np.zeros((len(model),len_Z, len_t, len_mass))


for c_index, c in enumerate(model):
    for z_index, z in enumerate(Z):
        k=0
        for i in range(len(t)):

            masst=10**masstimefunc[c_index][z_index](np.log10(t[i]/10**6))

            if 8<=masst<masses[-1-k]:
                print(masses[-1-k])
                print(masst)
                k+=1
            for m in range(len(mass)):
                if mass[m]<=masst:
                    ej_wind_mass[c_index, z_index, i, m] = 10**intrp_mass[c_index][z_index][i](np.log10(mass[m]))
                    if ej_wind_mass[c_index, z_index, i, m]==0:
                        ej_wind_mass[c_index, z_index, i, m] =ej_wind_mass[c_index, z_index, i, m-1]



        r=25*(z_index+1)
        print(f'{r}%')


# In[57]:


ej_winds = np.zeros((len(model), len(Z), len(t)))
imf=0


###intrgral to get the W function
for c in range(len(model)):
    for z in range(len(Z)):   
        for i in range(len(t)-1):

            ej_winds[c][z][i]=0
            for m in range(len(mass)-1):
                #if t[i]<10**lifetimefunc[z](np.log10(mass[m]))*10**6:
                    ej_winds[c][z][i]+= (ej_wind_mass[c, z, i, m+1]*IMF(mass[m+1])+ej_wind_mass[c, z, i, m]*IMF(mass[m]))*(mass[m+1]-mass[m])/(2*t[i+1]-t[i])


                


# In[58]:


colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
import matplotlib.lines as mlines
plt.figure(figsize=(7.7, 5.2))
for i in range(len(model)):
    plt.plot(t, ej_winds[i][2][:]/0.225, label=f'Track {model[i]}', color=colors[i % len(colors)])

for i in range(len(model)):
    plt.plot(t, ej_winds[i][0][:]/0.225, color=colors[i % len(colors)], linestyle="-.")


plt.xscale('log')
plt.yscale('log')
#plt.xlim(10**6, 10**8)

plt.xticks(fontsize=17)
plt.yticks(fontsize=16)
plt.tick_params(axis='x', which='both', direction='in', pad=6)  # Adjust pad to move x tick labels lower
plt.tick_params(axis='y', which='both', direction='in')


plt.xlabel('SSP age $[yr]$', fontsize=18, labelpad=5)
plt.ylabel(r'Mass ejection Rate [$yr^{-1}$]', fontsize=18,labelpad=7)
plt.tick_params(axis='both', direction='in', which='both')
ax = plt.gca()  # Get the current axes
ax.yaxis.set_ticks_position('both')  # Show ticks on both sides
ax.tick_params(axis='y', which='both', direction='in')
plt.legend(fontsize=13) 

main_legend = plt.legend(fontsize=13, loc='lower left')

# Create custom legend entries for line styles
solid_line = mlines.Line2D([], [], color='black', linestyle='-', label='z=0.02')
dotted_line = mlines.Line2D([], [], color='black', linestyle=':', label='z=0.0004')

# Add the custom legend for line styles to the plot
line_style_legend = plt.legend(handles=[solid_line, dotted_line], fontsize=13, loc='upper right')

# Add the main legend back to the plot
ax.add_artist(main_legend)


plt.savefig(f'../../Images/SEVN/winds_rateAGB_sevn_model.pdf', format='pdf')
plt.show()


# In[59]:


totwinds=np.zeros((len(model), len(Z),len(t)))

for c in range(len(model)):
    for z in range(len(Z)):
    
        for i in range( len(t)-1):
            totwinds[c][z][i+1]=(ej_winds[c][z][i+1]/0.225+ej_winds[c][z][i]/0.225)*(t[i+1]-t[i])/2+ totwinds[c][z][i]


# In[ ]:





# In[60]:


#plt.xlim(10**7, 10**8)
plt.figure(figsize=(7.7, 5.2))
for i in range(len(model)):
    plt.plot(t, totwinds[i][3], label=f'Track {model[i]}', color=colors[i % len(colors)])

for i in range(len(model)):
    plt.plot(t, totwinds[i][0], color=colors[i % len(colors)], linestyle="-.")


plt.xscale('log')
#plt.yscale('log')
#plt.xlim(10**6, 10**8)

plt.xticks(fontsize=17)
plt.yticks(fontsize=16)
plt.tick_params(axis='x', which='both', direction='in', pad=6)  # Adjust pad to move x tick labels lower
plt.tick_params(axis='y', which='both', direction='in')


plt.xlabel('SSP age $[yr]$', fontsize=18, labelpad=5)
plt.ylabel(r'Ejected mass fraction', fontsize=18,labelpad=7)
plt.tick_params(axis='both', direction='in', which='both')
ax = plt.gca()  # Get the current axes
ax.yaxis.set_ticks_position('both')  # Show ticks on both sides
ax.tick_params(axis='y', which='both', direction='in')
plt.legend(fontsize=13) 

main_legend = plt.legend(fontsize=13, loc='center left')

# Create custom legend entries for line styles
solid_line = mlines.Line2D([], [], color='black', linestyle='-', label='z=0.02')
dotted_line = mlines.Line2D([], [], color='black', linestyle=':', label='z=0.0004')

# Add the custom legend for line styles to the plot
line_style_legend = plt.legend(handles=[solid_line, dotted_line], fontsize=13, loc='upper left')

# Add the main legend back to the plot
ax.add_artist(main_legend)


plt.savefig(f'../../Images/SEVN/winds_totAGB_sevn_model.pdf', format='pdf')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[62]:


with h5py.File('../../MarVsSevn/Winds_IoroAGB_ov0.5_ms.hdf5', 'w') as f:
    # Create datasets
  
    dataset1 = f.create_dataset('Time', data=t, dtype='f')  # Create 'Time' dataset once
    dataset2=f.create_dataset('Metallicities', data=Z, dtype='f')
    group = f.create_group('Yields')
    for k in range(len(model)):
        group3 = group.create_group(f'{model[k]}')
        for i in range(len(Z)):
            group2 = group3.create_group(f'Z_{Z[i]}')
            dataset = group2.create_dataset('Ejected_mass', data=ej_winds[k][i]/0.20)


# In[ ]:





# In[63]:


Ejected_mass={}

elements=["H", "He", "C", "N", "O", "Ne","Mg", "Si", "Fe"]

metallicities=['Z_0.0004', 'Z_0.004', 'Z_0.008', 'Z_0.02']

metallicities2=['Z_0.004', 'Z_0.008', 'Z_0.019']

file_path = 'SNII_2.hdf5'  
for i in metallicities:
    with h5py.File(file_path, 'r') as file:
        stellar_masses = file['Masses'][:]
        Ejected_mass[f"{i}"]=file[f'Yields/{i}/Ejected_mass'][:]
    
mass_table2=np.array(stellar_masses)




YieldsSNII={}
file_path = 'SNII_2.hdf5'  
for i in metallicities:
    with h5py.File(file_path, 'r') as file:
        stellar_masses = file['Masses'][:]
        YieldsSNII[f"{i}"]=file[f'Yields/{i}/Yield'][:][:]


#yields = np.zeros((len(metallicities),len(elements), len(mass_table2)))
yieldsSNII = {}
for y in elements:
    yieldsSNII[f"{y}"] = np.zeros((len(metallicities), len(mass_table2)))
Y=0
for i in range(len(metallicities)):
    for y in range(len(elements)):
        for j in range(len(mass_table2)):
            yieldsSNII[f"{elements[y]}"][i][j] = YieldsSNII[f"{metallicities[i]}"][y][j]


YieldsAGB={}

file_path = 'AGB_Margio.hdf5'  
for i in metallicities2:
    with h5py.File(file_path, 'r') as file:
        stellar_masses = file['Masses'][:]
        YieldsAGB[f"{i}"]=file[f'Yields/{i}/Yield'][:][:]


yieldsAGB={}
for y in elements:
    yieldsAGB[f"{y}"] = np.zeros((len(metallicities2), len(stellar_masses)))

for i in range(len(metallicities2)):
    for y in range(len(elements)):
        for j in range(len(stellar_masses)):
            yieldsAGB[f"{elements[y]}"][i][j] = YieldsAGB[f"{metallicities2[i]}"][y][j]

metalej={}
file_path = 'SNII_2.hdf5'  
for i in metallicities:
    with h5py.File(file_path, 'r') as file:
        metalej[f"{i}"]=file[f'Yields/{i}/Total_Metals'][:]

        


metal_yieldSNII = np.empty((len(metallicities), len(mass_table2)))
for i in range(len(metallicities)):
    for j in range(len(mass_table2)):
        metal_yieldSNII[i][j]=np.array(metalej[f"{metallicities[i]}"][j])
        
metalej={}
file_path = 'AGB_Margio.hdf5'  
for i in metallicities2:
    with h5py.File(file_path, 'r') as file:
        metalej[f"{i}"]=file[f'Yields/{i}/Total_Metals'][:]

metal_yieldAGB = np.empty((len(metallicities2), len(stellar_masses)))
for i in range(len(metallicities2)):
    for j in range(len(stellar_masses)):
        metal_yieldAGB[i][j]=np.array(metalej[f"{metallicities2[i]}"][j])


# In[64]:


yieldsAGB_ioro={}
yieldsSN_ioro={}

M_ioro_AGB=1
M_mar_AGB=1

M_ioro_SN=1
M_mar_SN=1

for i in range(len(metallicities2)):
    for y in range(len(elements)):
        for j in range(len(stellar_masses)):
            yieldsAGB_ioro[f"{elements[y]}"][i][j]=yieldsAGB[f"{elements[y]}"][i][j]*M_ioro_AGB/M_mar_AGB


# In[ ]:


for y in elements:
    yieldsSN_ioro[f"{y}"] = np.zeros((len(metallicities), len(mass_table2)))

for i in range(len(metallicities)):
    for y in range(len(elements)):
        for j in range(len(mass_table2)):
            yieldsSN_ioro[f"{elements[y]}"][i][j] = yieldsSNII[f"{elements[y]}"][i][j]*M_ioro_SN/M_mar_SN


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




