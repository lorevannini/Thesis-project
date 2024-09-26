#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsmath}",
  
    "font.serif": ["Computer Modern Roman"]
})


# In[2]:


##IMF used 



def IMF(m):
    conditions = [
        m<=1,
        m>1
        
    ]

    choices = [
        
        0.852464*m**(-1)*np.exp(-(np.log10(m/0.079))**2/(2*0.69**2)),  
        0.237912*m**(-2.3)
    ]

    return np.select(conditions, choices)

    

    
    


# In[3]:





# In[4]:



##EXTRAPOLATE FROM CSV FILE THE DATA from SEVN output



##Mass used to sample the SSP from SEVN data 
masses = [8, 9, 12, 15, 20, 23, 24, 25, 26, 28, 30, 33, 36,  40, 50, 60,  70, 80, 90, 100, 120]
Z = np.array([0.0004, 0.004, 0.008, 0.02])


remType={}
lifetime = {}
windloss = {}
starmass = {}
stars = {}
state={}
timel = {}
remType = {}


#can choose to vizualize different formalism for SN or different stellar tracks
models=["rapid", "delayed", "compact", "deathmatrix", "deathmatrix_new"]
#models=["ov05", "ov04", "mesa"]

lifetime = {k: {} for k in models}
windloss = {k: {} for k in models}
starmass = {k: {} for k in models}
stars = {k: {} for k in models}
state = {k: {} for k in models}
timel = {k: {} for k in models}
remType = {k: {} for k in models}

# Reading data
for k in models:
    for j in Z:
        lifetime[k][j] = {}
        windloss[k][j] = {}
        starmass[k][j] = {}
        stars[k][j] = {}
        state[k][j] = {}
        timel[k][j] = {}
        remType[k][j] = {}

        for i in masses:
            # Assuming the file path structure and files are correct
            file_path = f"param/Test/sn_tracks_ov05/z_{j}/{i}/{k}/output_0.csv"
            data = pd.read_csv(file_path)
            
            stars[k][j][i] = data
            lifetime[k][j][i] = data["Worldtime"].values
            windloss[k][j][i] = data["dMdt"].values
            starmass[k][j][i] = data["Mass"].values
            state[k][j][i] = data["Phase"].values
            timel[k][j][i] = data["Worldtime"].values[-1]
            remType[k][j][i] = data["RemnantType"].values[-1]


# In[ ]:





# In[5]:


#CREATE THE NEW LIFETIME FUNCTION

lifetimefunc = {}
masstimefunc = {}


for c_index, c in enumerate(models):
    lifetimefunc[c_index] = {}
    masstimefunc[c_index] = {}
    for j_index, j in enumerate(Z):
        log_masses = np.log10(masses)
        log_timel = np.log10([timel[c][j][i] for i in masses])

        lifetimefunc[c_index][j_index] = interp1d(log_masses, log_timel, kind="linear", fill_value="extrapolate")
        masstimefunc[c_index][j_index] = interp1d(log_timel, log_masses, kind="linear", fill_value="extrapolate")


# In[ ]:





# In[6]:


dt = {}
winds = {}
totwinds = {}
dM = {}

# Calculate changes in mass (dM) and time intervals (dt) for each combination of mass and metallicity
for k in models:
    dM[k] = {}
    dt[k] = {}
    winds[k] = {}
    totwinds[k] = {}

    for j in Z:
        dM[k][j] = {}
        dt[k][j] = {}
        winds[k][j] = {}
        totwinds[k][j] = {}

        for i in masses:
            dM[k][j][i] = np.zeros(len(starmass[k][j][i]))
            dt[k][j][i] = np.zeros(len(lifetime[k][j][i]))
            winds[k][j][i] = np.zeros(len(lifetime[k][j][i]))
            totwinds[k][j][i] = 0

            for c in range(len(starmass[k][j][i]) - 1):
                dM[k][j][i][c + 1] = starmass[k][j][i][c + 1] - starmass[k][j][i][c]

            for c in range(len(lifetime[k][j][i]) - 1):
                dt[k][j][i][c + 1] = lifetime[k][j][i][c + 1] - lifetime[k][j][i][c]
                winds[k][j][i][c ] = dt[k][j][i][c ] * windloss[k][j][i][c]
                totwinds[k][j][i] += winds[k][j][i][c + 1]



# In[ ]:





# In[7]:


##plotting the mass ejection rate (dM/dt) of single stars

plt.figure(figsize=(7.7, 5.2))
i = 8  # Specific mass

for j in Z:
    plt.plot(lifetime["delayed"][j][i] * 10**6, -dM["delayed"][j][i], label=f'Z = {j}')

plt.xscale('log')
plt.yscale('log')


plt.xticks(fontsize=17)
plt.yticks(fontsize=16)
plt.tick_params(axis='x', which='both', direction='in', pad=6)  # Adjust pad to move x tick labels lower
plt.tick_params(axis='y', which='both', direction='in')


plt.xlabel('SSP age $[yr]$', fontsize=18, labelpad=5)
plt.ylabel(r'dM [$M_{\odot}$]', fontsize=18,labelpad=7)
plt.tick_params(axis='both', direction='in', which='both')
ax = plt.gca()  # Get the current axes
ax.yaxis.set_ticks_position('both')  # Show ticks on both sides
ax.tick_params(axis='y', which='both', direction='in')
plt.legend(fontsize=13) 
plt.savefig(f'../../Images/SEVN/Single_star_mett.pdf', format='pdf')
plt.show()


# In[ ]:





# In[8]:


##inizialing stellar paramters for all masses metallicity and model

mass_sn1 = np.zeros((len(models), len(Z), len(masses)))
tot_winds1 = np.zeros((len(models), len(Z), len(masses)))
rem_mass =np.zeros((len(models), len(Z), len(masses)))
kik = np.zeros((len(models), len(Z), len(masses)))
mass_sn12 = np.zeros((len(Z), len(masses)))
tot_winds12 = np.zeros((len(Z), len(masses)))
rem_mass2 = np.zeros((len(Z), len(masses)))


# In[9]:





# In[19]:



rem_type={}
totwinds_ms = {}

for k_index, k in enumerate(models):
    totwinds_ms[k]={}
    for j in Z:
        totwinds_ms[k][j] = {}

    for j_index, j in enumerate(Z):
        for i_index, mass in enumerate(masses):
            mass_str = mass
            totwinds_ms[k][j][i_index] = 0  # Initialize totwinds_ms[j] for each mass index

            for c in range(len(state[k][j][mass_str]) - 1):  # Iterate over the entire state list
                #separate the mainsequence with the postmainsequence phases
                if state[k][j][mass_str][c] != "TerminalMainSequence" and  state[k][j][mass_str][c] !="MainSequence":
                    totwinds_ms[k][j][i_index] += -winds[k][j][mass_str][c]
                if state[k][j][mass_str][c] == "TerminalMainSequence" and state[k][j][mass_str][c + 1] == "ShellHBurning":
                    mass_sn1[k_index][j_index][i_index] = starmass[k][j][mass_str][c] - starmass[k][j][mass_str][-1]
                    tot_winds1[k_index][j_index][i_index] = totwinds[k][j][mass_str]
                    rem_mass[k_index][j_index][i_index] = starmass[k][j][mass_str][-1]
                    kik[k_index][j_index][i_index]=c

            



                


# In[ ]:





# 

# In[20]:


j=0.02
kik = kik.astype(int)


# In[ ]:





# In[21]:


plt.figure(figsize=(7.7, 5.2))  


j_index=3

plt.plot(masses, mass_sn1[0][j_index] / masses, label=f'Model: {models[0]}')
plt.plot(masses, mass_sn1[1][j_index] / masses, label=f'Model: {models[1]}')
plt.plot(masses, mass_sn1[4][j_index] / masses, label=f'Model: deathmatrix')
plt.plot(masses, mass_sn1[2][j_index] / masses, label=f'Model: {models[2]}')
plt.xscale('log')
#plt.yscale('log')


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
plt.savefig(f'../../Images/SEVN/Sn_eff_sevn_model.pdf', format='pdf')
plt.show()


# In[22]:


with h5py.File('../../MarVsSevn/SNII_Ioro__ms.hdf5', 'w') as f:
    # Create datasets
    dataset1 = f.create_dataset('Masses', data=masses, dtype='f')
    dataset2=f.create_dataset('Metallicities', data=Z, dtype='f')
    group = f.create_group('Yields')
    for k in range(len(models)):
        group3 = group.create_group(f'{models[k]}')
        for i in range(len(Z)):
            group2 = group3.create_group(f'Z_{Z[i]}')
            dataset = group2.create_dataset('Ejected_mass', data=mass_sn1[k][i][:])

with h5py.File('../../MarVsSevn/Lifetimes_ioro_ov05.hdf5', 'w') as f:
    # Create datasets for masses and metallicities
    f.create_dataset('Masses', data=masses, dtype='f')
    f.create_dataset('Metallicities', data=Z, dtype='f')
    
    # Create group for models
    group = f.create_group('models')
    
    for k in models:
        group_model = group.create_group(k)
        
        # Create an empty matrix for lifetimes with shape (len(Z), len(masses))
        lifetimes_matrix = np.zeros((len(Z), len(masses)))
        
        for j_index, j in enumerate(Z):
            for i_index, i in enumerate(masses):
                lifetimes_matrix[j_index, i_index] = timel[k][j][i] * 10**6
        
        # Create dataset for lifetimes
        group_model.create_dataset('Lifetimes', data=lifetimes_matrix, dtype='f')

      


# In[23]:


#sliceing winds from end-life mass losses

sliced_lifetime = {}
sliced_winds = {}

sliced_lifetime2 = {}
sliced_winds2 = {}

for c, cc in enumerate(models):
    sliced_lifetime[c] = {}
    sliced_winds[c] = {}  
    sliced_lifetime2[c] = {}
    sliced_winds2[c] = {}

    for j_index, j in enumerate(Z):
        sliced_lifetime[c][j] = {}
        sliced_winds[c][j] = {}  
        sliced_lifetime2[c][j] = {}
        sliced_winds2[c][j] = {}



        for i_index, i in enumerate(masses):
            k=kik[c][j_index][i_index]
            sliced_lifetime[c][j][i] = np.array(lifetime[cc][j][i][:k])
            sliced_winds[c][j][i] = np.array(winds[cc][j][i][:k])


        
        


# In[16]:


i = 20  # Specific mass

j=0.02
plt.plot(lifetime["compact"][j][i][:] * 10**6, -winds["compact"][j][i][:], label=f'Z = {j}')
i=20

plt.xlim(7.5*10**6, 9.5*10**6)
plt.xscale("log")
plt.xscale("log")
plt.yscale('log')
plt.xlabel('Lifetime [yr]')
plt.ylabel('Change in Mass [M\_sun]')
plt.legend()
plt.title(f'Change in Mass vs. Lifetime for Mass {i}')
plt.show()


# In[27]:


#initialinzing a universal time

t = np.logspace(  np.log10(10**8), (np.log10(100000)),3000)
t=np.flip(t)

t1=np.linspace(0, 100000, 100)
t=np.append(t1, t)


# In[ ]:





# In[34]:


#interpolating time in order to get the dM/dt at the universal time

wind = {}
wind_func={}
life={}
for c_index, c in enumerate(models):

    wind[c] = {}
    wind_func[c]={}
    for z in Z:
        wind[c][z] = {}
        wind_func[c][z]={}
        for m in masses:
            wind[c][z][m] = np.zeros(len(t))
            life = np.array(sliced_lifetime[c_index][z][m]) * 10**6
            wind_func[c][z][m]=()
            ##INTERPOLAZIONE SUL TEMPO   
            for i in range(len(life) - 1):
                for k in range(len(t) - 1):
                    if life[i] <= t[k] <= life[i + 1]:
                        wind[c][z][m][k] = -(sliced_winds[c_index][z][m][i + 1] - sliced_winds[c_index][z][m][i]) / (life[i + 1] - life[i]) * (t[k] - life[i]) - sliced_winds[c_index][z][m][i]


        
wind_array = np.zeros((len(models), len(Z), len(masses), len(t)))

for c_index, c in enumerate(models):
    for z_index, z in enumerate(Z):
        for m_index, m in enumerate(masses):
            for k, time in enumerate(t):
                wind_array[c_index, z_index, m_index, k] =np.array( wind[c][z][m][k])

                
###wind_array is an array that tells at given metallicty for a given model at a given time how much is dM/dt at given time
                    


# In[35]:


#plotting the winds sliced at universal time
for i in range(3):
    plt.plot(t, wind_array[i][3][4][:], label=f'Z = {Z[2]}')
plt.xscale("log")
plt.yscale('log')
plt.xlabel('mass')
plt.ylabel('Change in Mass [M\_sun]')
plt.xlim(1e6, 1e7)
plt.legend()
plt.title(f'Change in Mass vs. Lifetime for Mass {i}')
plt.show()


# In[36]:



for i in range(3):
    sli=np.array( wind_array[i, 3, :, 2000])
    plt.plot(masses, sli, label=f'Z = {models[i]}')


#plt.xscale("log")
plt.xscale("log")
plt.yscale('log')
plt.xlabel('mass')
plt.ylabel('Change in Mass [M\_sun]')
plt.legend()
plt.title(f'Change in Mass vs. Lifetime for Mass {i}')
plt.show()


# In[33]:


#for each universal time we need to interpolate between masses, so that i can have the winds for each 
mass=np.linspace(8, 100, 100)

intrp_mass={}


#this is the mass interpolation function


for k, kk in enumerate(models):
    intrp_mass[k]={}
    for z in range(len(Z)):
            intrp_mass[k][z]={}
            for i in range(len(t)):
                    ej_mass=np.array(wind_array[k, z, :, i])
                    intrp_mass[k][z][i]=interp1d(np.log10(masses), np.log10(ej_mass), kind="linear")
    


# In[37]:


len_Z = len(Z)
len_t = len(t)
len_mass = len(mass)

# Create a 3D NumPy array filled with zeros
ej_wind_mass = np.zeros((len(models),len_Z, len_t, len_mass))

#create a np array containing all the winds at each time each metallicity for all models.  
for c_index, c in enumerate(models):
    for z_index, z in enumerate(Z):
        k=0
        for i in range(len(t)):
            #we carry the interpolation up to stars with stellar masses that are still alive at a given time
            masst=10**masstimefunc[c_index][z_index](np.log10(t[i]/10**6))
            
            #if stars die update the limit of interpolation
            if 8<=masst<masses[-1-k]:
                print(masses[-1-k])
                print(masst)
                k+=1
            for m in range(len(mass)):
                if mass[m]<=masst:
                    ej_wind_mass[c_index, z_index, i, m] = 10**intrp_mass[c_index][z_index][i](np.log10(mass[m]))
                    if ej_wind_mass[c_index, z_index, i, m]==0:
                        ej_wind_mass[c_index, z_index, i, m] =ej_wind_mass[c_index, z_index, i, m-1]


# In[61]:


#plotting the sliced winds for the universal time an the arry of masses initalized
 
for i in range(3):
    

    plt.plot(mass, ej_wind_mass[i, 1, 2, :], label=f't = {i}')
    
plt.yscale('log')
plt.xlabel('Mass [Solar]')
plt.ylabel('dM/dt')
plt.legend()
#plt.title(f'Change in Mass vs. Lifetime for Mass {i}')
plt.show()


# In[ ]:





# In[38]:


##calculating the ejection rate from winds by intagrating over the masses at a given time and weighting with the IMF

ej_winds = np.zeros((len(models), len(Z), len(t)))



###intrgral to get the W function
for c in range(len(models)):
    for z in range(len(Z)):   
        for i in range(len(t)-1):

            ej_winds[c][z][i]=0
            for m in range(len(mass)-1):
                
                    ej_winds[c][z][i]+= (ej_wind_mass[c, z, i, m+1]*IMF(mass[m+1])+ej_wind_mass[c, z, i, m]*IMF(mass[m]))*(mass[m+1]-mass[m])/(2*t[i+1]-t[i])


                


# In[ ]:





# In[41]:


colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
import matplotlib.lines as mlines


plt.figure(figsize=(7.7, 5.2))
for i in range(len(models)):
    plt.plot(t, ej_winds[i][2][:]/0.225, color=colors[i % len(colors)])

for i in range(len(models)):
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
dotted_line = mlines.Line2D([], [], color='black', linestyle='-.', label='z=0.0004')

# Add the custom legend for line styles to the plot
line_style_legend = plt.legend(handles=[solid_line, dotted_line], fontsize=13, loc='upper right')

# Add the main legend back to the plot
ax.add_artist(main_legend)


plt.savefig(f'../../Images/SEVN/winds_rate_sevn_model.pdf', format='pdf')
plt.show()


# In[ ]:





# In[42]:


#total cumulaive function of winds

totwinds=np.zeros((len(models), len(Z),len(t)))

for c in range(len(models)):
    for z in range(len(Z)):
    
        for i in range( len(t)-1):
            totwinds[c][z][i+1]=(ej_winds[c][z][i+1]/0.225+ej_winds[c][z][i]/0.225)*(t[i+1]-t[i])/2+ totwinds[c][z][i]


# In[44]:




#plt.xlim(10**7, 10**8)
plt.figure(figsize=(7.7, 5.2))
for i in range(len(models)):
    plt.plot(t, totwinds[i][3], color=colors[i % len(colors)])

for i in range(len(models)):
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


plt.savefig(f'../../Images/SEVN/winds_tot_sevn_model.pdf', format='pdf')
plt.show()


# In[49]:


##create the new tables 


with h5py.File('../../MarVsSevn/Winds_Ioro_mesa_ms.hdf5', 'w') as f:
    # Create datasets
  
    dataset1 = f.create_dataset('Time', data=t, dtype='f')  # Create 'Time' dataset once
    dataset2=f.create_dataset('Metallicities', data=Z, dtype='f')
    group = f.create_group('Yields')
    for k in range(len(models)):
        group3 = group.create_group(f'{models[k]}')
        for i in range(len(Z)):
            group2 = group3.create_group(f'Z_{Z[i]}')
            dataset = group2.create_dataset('Ejected_mass', data=ej_winds[k][i][:]/0.225)


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




