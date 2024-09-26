#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import csv
from scipy.integrate import quad
import matplotlib.cm as cm

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.serif": ["Computer Modern Roman"]
})





# In[ ]:





# In[2]:


###Function i use un the code#####


#Define an IMF (same used in the volkberger and marinacci)

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

def IMF_krp(m):
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


#setting log scales
def convert_to_logscale(tn):
    dt=dt=np.log10(tn[len(tn)-1]/tn[0])/(len(tn))
    logt=np.log10(tn[0])+np.arange(len(tn))*dt 
    t=10.**logt
    return t


# In[3]:





# In[4]:



#Importing the data from the Hdf5 file from SMUGGLE code
file_path = 'Lifetimes.hdf5'  
with h5py.File(file_path, 'r') as file:
    stellar_masses = file['Masses'][:]
    lifetimes_file=file['Lifetimes'][:]
    #metallicity=file['Metallicities'][:]

    
#making the nparray, select the 3 raw that correspond to the solar metallicity 0.02 
mass_table=np.array(stellar_masses)

#metallicities in the file        
mett=np.array([0.0004, 0.004, 0.008, 0.02])

#We get the lifetime in a matrix
lifetime1 = np.empty((len(mett), len(mass_table)))
for i in range (len(mett)):
    for j in range(len(mass_table)):
        lifetime1[i][j]=np.array(lifetimes_file[i][j])
        
lifetime_=np.transpose(lifetime1)



#getting the ejected mass of SNII 
Ejected_mass={}
metallicities=['Z_0.0004', 'Z_0.004', 'Z_0.008', 'Z_0.02']
file_path = 'SNII.hdf5'  
for i in metallicities:
    with h5py.File(file_path, 'r') as file:
        stellar_masses = file['Masses'][:]
        Ejected_mass[f"{i}"]=file[f'Yields/{i}/Ejected_mass'][:]
    
mass_table2=np.array(stellar_masses)
ejected_SNII_ioro = np.empty((len(metallicities), len(mass_table2)))
ejected_SNII_mar= np.empty((len(metallicities), len(mass_table2)))


for i in range(len(metallicities)):
    for j in range(len(mass_table2)):
        ejected_SNII_ioro[i][j]=np.array(Ejected_mass[f"{metallicities[i]}"][j])
        ejected_SNII_mar
        

        
Ejected_mass2={}
metallicities=['Z_0.0004', 'Z_0.004', 'Z_0.008', 'Z_0.02']
file_path = 'SNII_2.hdf5'  
for i in metallicities:
    with h5py.File(file_path, 'r') as file:
        stellar_masses2 = file['Masses'][:]
        Ejected_mass2[f"{i}"]=file[f'Yields/{i}/Ejected_mass'][:]
    
mass_table2=np.array(stellar_masses)
ejected_SNII_ioro2 = np.empty((len(metallicities), len(mass_table2)))
ejected_SNII_mar2= np.empty((len(metallicities), len(mass_table2)))


for i in range(len(metallicities)):
    for j in range(len(mass_table2)):
        ejected_SNII_ioro2[i][j]=np.array(Ejected_mass2[f"{metallicities[i]}"][j])
        ejected_SNII_mar
        



##Getting the ejected mass from AGB


##Getting the ejected mass from AGB

metallicities2=['Z_0.004', 'Z_0.008', 'Z_0.019']
Ejected_mass_AGB={}

#Importing the AGB file

file_path = 'AGB_Margio.hdf5'  
for i in metallicities2:
    with h5py.File(file_path, 'r') as file:
        AGB_mass1 = file['Masses'][:]
        Ejected_mass_AGB[f"{i}"]=file[f'Yields/{i}/Ejected_mass'][:]
        
#Converting in np array

AGB_mass=np.array(AGB_mass1)


AGB_ej=np.empty((len(metallicities2), len(AGB_mass)))



for j in range(len(metallicities2)):
    for i in range(len(AGB_mass)):
        AGB_ej[j][i]=np.array(Ejected_mass_AGB[f"{metallicities2[j]}"][i])
mettAGB=np.array([ 0.004, 0.008, 0.019])


# In[5]:


###This is the function for SnII###

#defining characteristic times
t_II=np.array([3.7, 7.0, 44.0]) #Mys

#defining amplitude factors
a_II=np.array([0.39, 0.51, 0.18]) #GyrM_sol

#defining power law exponts
psi_II=np.array([(np.log(a_II[1]/a_II[0]))/(np.log(t_II[1]/t_II[0])),
                 (np.log(a_II[2]/a_II[1]))/(np.log(t_II[2]/t_II[1]))])

#defining the function for SnII rate
def SnII_rate(t):
    conditions = [
        (t >= 0) & (t < t_II[0]),
        (t_II[0] <= t) & (t < t_II[1]),
        (t_II[1] <= t) & (t < t_II[2]),
        (t_II[2]<=t)
    ]

    choices = [
        0,
        a_II[0] * np.power((t/t_II[0]), psi_II[0])/10**9,
        a_II[1] * np.power(t / t_II[1], psi_II[1])/10**9,
        0
    ]

    return np.select(conditions, choices)



###This is the function for SnII Of Fire 2###

#defining characteristic times
t_II2=np.array([3.401, 10.37, 37.53]) #Mys

#defining amplitude factors
a_II2=np.array([0.39, 0.51, 0.18]) #GyrM_sol

#defining power law exponts
psi_II2=np.array([(np.log(a_II[1]/a_II[0]))/(np.log(t_II[1]/t_II[0])),
                 (np.log(a_II[2]/a_II[1]))/(np.log(t_II[2]/t_II[1]))])

#defining the function for SnII rate
def SnII_rate2(t):
    conditions = [
        (t >= 0) & (t < t_II2[0]),
        (t_II2[0] <= t) & (t < t_II2[1]),
        (t_II2[1] <= t) & (t < t_II2[2]),
        (t_II2[2]<=t)
    ]

    choices = [
        0,
        5.408*10**-4/10**6,
        2.516*10**-4/10**6,
        0
    ]

    return np.select(conditions, choices)

tn=np.linspace(1, 10000, num=1000)

#Tranforming the timesteps in log-spaced timesteps
dt=np.log10(tn[len(tn)-1]/tn[0])/(len(tn))
logt=np.log10(tn[0])+np.arange(len(tn))*dt 
t_hop=10.**logt

rate_snII=np.array([SnII_rate(t_i) for t_i in t_hop])



t_ii = np.array([1*10**6, 3.5*10**6, 100*10**6])

def OB_AGB_mass_loss(t, Z):
    conditions = [
        (t >= 0) & (t < t_ii[0]),
        (t_ii[0] <= t) & (t < t_ii[1]),
        (t_ii[1] <= t) & (t < t_ii[2]),
        (t_ii[2] <= t)
    ]

    choices = [
        4.763*(0.01+Z/0.013),
        4.763 * (0.01 + Z / 0.013) * np.power(t/10**6, 1.45 + 0.8 * np.log(Z / 0.013)),
        29.4 * (t / (3.5*10**6)) ** -3.25 + 0.0042,
        0
    ]
    return np.select(conditions, choices)


# In[6]:


#####SELECT Z
z=0.02
#####SELECT Z


# In[7]:


###This calculate the interpolation function  given a generic z


def Z_interpolation( Z, mass, lifetime, metallicities):
    a=np.shape(lifetime)
    for i in range(len(metallicities)-1):
        ###This check where metallicity lands
        if metallicities[i]<=Z and Z<metallicities[i+1]:
            z=i
        elif Z>=metallicities[-1]:
            z=i
        elif Z<metallicities[0]:
            z=0
    
    
    ###This calculate the values at the metallicity inizialize
    lifetime_z=np.zeros(a[0])
    for i in range(a[0]):
        lifetime_z[i]=lifetime[i][z]+(lifetime[i][z+1]-lifetime[i][z])/(metallicities[z+1]-metallicities[z])*(Z-metallicities[z])
    
    
    ###The function returns a function that is the interpolation in logspace of the value calculated before 
    lif_z = interp1d(np.log10(mass), np.log10(lifetime_z), kind="linear", fill_value='extrapolate')
    return lif_z
            


# In[8]:


###We define the lifetime-function that gives the time at given metallicity
lifetimefunc=Z_interpolation( z,  mass_table, lifetime_, mett)

#this is the mass range scaled in log that we use to set up the variable
masses=np.linspace(120, 0.8, num=1000)
masses=convert_to_logscale(masses)


#we store the lifetime in a given array that correspon to the mass
lifetime=np.zeros(len(masses))
for i in range(len(masses)):
        lifetime[i]=10**lifetimefunc(np.log10(masses[i])) 
    
    


# In[9]:


##Plot the lifetime function with the interpolated function in log space

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
save_plot="../Images/Hop"
plt.rc('axes', prop_cycle=(plt.cycler('color', CB_color_cycle)))

plt.figure(figsize=(7.7, 5.2))
plt.scatter(lifetime1[:][0], mass_table, label='Data with Z=0.008' )
plt.scatter(lifetime1[:][3], mass_table, label='Data with Z=0.02')
plt.plot(10**lifetimefunc(np.log10(masses)), masses, label='Interpolated $\mathcal{M}(t, Z=0.01)$', color=CB_color_cycle[2])


plt.xscale('log')
plt.yscale('log')


plt.xticks(fontsize=17)
plt.yticks(fontsize=16)
plt.tick_params(axis='x', which='both', direction='in', pad=6)  # Adjust pad to move x tick labels lower
plt.tick_params(axis='y', which='both', direction='in')


plt.xlabel('SSP age $[yr]$', fontsize=18, labelpad=5)
plt.ylabel(r'M [$M_{\odot} ]$', fontsize=18,labelpad=7)
plt.tick_params(axis='both', direction='in', which='both')
ax = plt.gca()  # Get the current axes
ax.yaxis.set_ticks_position('both')  # Show ticks on both sides
ax.tick_params(axis='y', which='both', direction='in')
plt.legend(fontsize=13) 
plt.savefig(f'../Images/Mar/lifetime_int.pdf', format='pdf')
plt.show()


# In[10]:


#this is the mass range scaled in log
masses=np.logspace(np.log10(120), np.log10(0.8), num=1000)



#timescale given the mass
lifetime=np.zeros(len(masses))
for i in range(len(masses)):
        lifetime[i]=10**lifetimefunc(np.log10(masses[i])) 


# In[11]:


#inizializing the rate and calculting it through trapezoids

rate_SNII=np.zeros(len(masses)) 

rate_SNII_K=np.zeros(len(masses)) 
for i in range(len(masses)-1):
        if 8<=masses[i]<=100:
            rate_SNII[i+1]=-(IMF(masses[i+1])+IMF(masses[i]))*(masses[i+1]-masses[i])/(2*(lifetime[i+1]-lifetime[i]))
            rate_SNII_K[i+1]=-(IMF_krp(masses[i+1])+IMF_krp(masses[i]))*(masses[i+1]-masses[i])/(2*(lifetime[i+1]-lifetime[i]))
            


# In[12]:


#plotting the rate of SNII explosion

plt.figure(figsize=(7.7, 5.2))
plt.plot(lifetime, rate_SNII, color='tomato',label='SNII rate in SMUGGLE with Chabrier IMF')
plt.plot(lifetime, rate_SNII_K, color=CB_color_cycle[0], label='SNII rate in SMUGGLE with Kroopa IMF' )
plt.plot(t_hop*1e6, rate_snII, label='SNII in Fire-3', color=CB_color_cycle[0], linestyle='dotted')
plt.xscale('log')
plt.yscale('log')
plt.xlim(10**6, 5*10**8)

plt.xticks(fontsize=17)
plt.yticks(fontsize=16)
plt.tick_params(axis='x', which='both', direction='in', pad=6)  # Adjust pad to move x tick labels lower
plt.tick_params(axis='y', which='both', direction='in')


plt.xlabel('SSP age $[yr]$', fontsize=18, labelpad=5)
plt.ylabel(r'SnII Rate $R/M_{*}$ [$M_{\odot}^{-1}$ $yr^{-1}$]', fontsize=18,labelpad=7)
plt.tick_params(axis='both', direction='in', which='both')
ax = plt.gca()  # Get the current axes
ax.yaxis.set_ticks_position('both')  # Show ticks on both sides
ax.tick_params(axis='y', which='both', direction='in')
plt.legend(fontsize=13) 
plt.savefig(f'../Images/Mar/sn_rate_mar.pdf', format='pdf')
plt.show()


# In[ ]:





# In[13]:


##THIS is a fuction to calculate the efficiency of the ejection given the metallicty

def ej_interpolation( Z, mass, ejection, metallicities):
    ejection_eff=np.transpose(ejection)
    a=np.shape(ejection_eff)
    for i in range(len(metallicities)-1):
        if metallicities[i]<=Z and Z<metallicities[i+1]:
            z=i
        elif Z>=metallicities[-1]:
            z=i
        elif Z<metallicities[0]:
            z=0
    
    
    ejection_eff_z=np.zeros(a[0])
    for i in range(a[0]):
        ejection_eff_z[i]=ejection_eff[i][z]+(ejection_eff[i][z+1]-ejection_eff[i][z])/(metallicities[z+1]-metallicities[z])*(Z-metallicities[z])

    ej_z = interp1d(np.log10(mass), np.log10(ejection_eff_z), kind="linear", fill_value='extrapolate')
    return ej_z


# In[14]:


###I calculate the ejection efficency before doing anything

ejection_eff= np.empty((len(metallicities), len(mass_table2)))
ejection_eff2= np.empty((len(metallicities), len(mass_table2)))
for i in range(len(mett)):
    for j in range(len(mass_table2)):
        ejection_eff[i][j]=ejected_SNII_ioro[i][j]/mass_table2[j]

        ejection_eff2[i][j]=ejected_SNII_ioro2[i][j]/mass_table2[j]

np.shape(ejection_eff)


# In[ ]:





# In[15]:


#interpolation of the ejetion efficincy function f_eff
ej_eff=ej_interpolation( z, mass_table2, ejection_eff, mett)
ej_eff2=ej_interpolation( z, mass_table2, ejection_eff2, mett)


# In[16]:


#plot the interpolated ejection efficency function

plt.figure(figsize=(7.7, 5.2))
plt.scatter(mass_table2, ejection_eff2[2][:], label='Data for Z=0.008')
plt.plot(masses, 10**ej_eff2(np.log10(masses)), label='Efficiency $f(M, Z=0.01)$', color='sandybrown')
plt.scatter(mass_table2, ejection_eff2[3][:],label='Data for Z=0.02')


plt.xscale('log')
plt.xlim(6, 120)


plt.xticks(fontsize=17)
plt.yticks(fontsize=16)
plt.tick_params(axis='x', which='both', direction='in', pad=6)  # Adjust pad to move x tick labels lower
plt.tick_params(axis='y', which='both', direction='in')


plt.xlabel('M $[M_{\odot}]$', fontsize=18, labelpad=5)
plt.ylabel(r'Ejected mass efficency $f_{rec}(M, Z)$', fontsize=18,labelpad=7)
plt.tick_params(axis='both', direction='in', which='both')
ax = plt.gca()  # Get the current axes
ax.yaxis.set_ticks_position('both')  # Show ticks on both sides
ax.tick_params(axis='y', which='both', direction='in')
plt.legend(fontsize=13) 
plt.savefig(f'../Images/Mar/sn_eff_marnoW.pdf', format='pdf')
plt.show()


# In[17]:


#initialize and calculate the ejection mass rate 
mass=masses
ejection_SNII=np.zeros(len(masses))
ejection_SNII2=np.zeros(len(masses))
    

for i in range(len(mass) - 1):
        if 8 <= mass[i] <= 100 and i + 1 < len(mass):
             ejection_SNII[i]= -(10**ej_eff(np.log10(masses[i + 1]))*masses[i+1]* IMF(masses[i + 1]) + 10**ej_eff(np.log10(masses[i]))* IMF(masses[i])*masses[i]) * (masses[i + 1] - masses[i]) / (2 * (lifetime[i + 1] - lifetime[i]))
             ejection_SNII2[i]= -(10**ej_eff2(np.log10(masses[i + 1]))*masses[i+1]* IMF(masses[i + 1]) + 10**ej_eff2(np.log10(masses[i]))* IMF(masses[i])*masses[i]) * (masses[i + 1] - masses[i]) / (2 * (lifetime[i + 1] - lifetime[i]))
            
            
        else:
            ejection_SNII[i] = 0
            ejection_SNII2[i]=0


# In[18]:



plt.figure(figsize=(7.7, 5.2))

plt.plot(lifetime,  ejection_SNII, label='SNII+Winds', color='sandybrown')
plt.plot(lifetime,  ejection_SNII2, label='SNII', color=CB_color_cycle[2] )

plt.xscale('log')
plt.yscale('log')
plt.xlim(10**6, 10**9)

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
plt.savefig(f'../Images/Mar/mass_sn_rate_mar.pdf', format='pdf')
plt.show()


# In[19]:


#calculate the cumulative mass ejection from SNII

comulative_snII=np.zeros(len(masses))
comulative_snII2=np.zeros(len(masses))

for i in range(len(masses)-1):
        comulative_snII[i+1]=(ejection_SNII[i+1]+ejection_SNII[i])*(lifetime[i+1]-lifetime[i])/2+comulative_snII[i]
        comulative_snII2[i+1]=(ejection_SNII2[i+1]+ejection_SNII2[i])*(lifetime[i+1]-lifetime[i])/2+comulative_snII2[i]
 
ej_int=0
imf_int=0
for i in range(len(masses)-1):
    if 8<=masses[i]<=100:
        ej_int+=((10**ej_eff(np.log10(masses[i + 1]))*masses[i+1]* IMF(masses[i + 1]) + 10**ej_eff(np.log10(masses[i]))* IMF(masses[i])*masses[i]) * (masses[i + 1] - masses[i])/2)
        imf_int+=((IMF(masses[i+1])+IMF(masses[i]))*(masses[i+1]-masses[i])/2)


# In[20]:


ej_int/imf_int


# In[ ]:





# In[21]:


plt.plot(lifetime, comulative_snII2)
#plt.plot(t, comulative_snII['Z_0.02'], label='Mass interpolation')
#plt.plot(t, comulative_snII_q, label='Quadratic interpolation')
plt.xscale('log')
#plt.yscale('log')
plt.xlabel('t $[yr]$', fontsize=14)
plt.ylabel('$[M/M_{\odot}]$', fontsize=14)
plt.title(r'\textbf{Total mass ejected by a SnII in a stellar population of a solar mass', fontsize=19)
##plt.legend()
ax = plt.gca()  # Get the current axes
ax.yaxis.set_ticks_position('both')  # Show ticks on both sides
ax.tick_params(axis='y', which='both', direction='in')
plt.show()


# In[ ]:


##do the same procedure for AGB stars


# In[22]:


#calculate the ejection efficency for points in table

ejection_eff_AGB= np.zeros((len(mettAGB), len(AGB_mass)))
for i in range(len(mettAGB)):
    for j in range(len(AGB_mass)):
        ejection_eff_AGB[i][j]=AGB_ej[i][j]/AGB_mass[j]

 


# In[23]:


#interpolate the efficiency function for the AGB 

ej_eff_AGB=ej_interpolation( z, AGB_mass, ejection_eff_AGB, mettAGB)


# In[24]:



plt.figure(figsize=(7.7, 5.2))

plt.scatter(AGB_mass, AGB_ej[1][:]/AGB_mass, label='Data for Z=0.008')
plt.plot(masses, 10**ej_eff_AGB(np.log10(masses)), color=CB_color_cycle[7] ,label='Efficiency $f(M, Z=0.01)$ Margio' )
plt.scatter(AGB_mass, AGB_ej[2][:]/AGB_mass, label='Data for Z=0.02')
plt.xscale('log')
plt.xlim(1.2,5.5)


plt.xticks(fontsize=17)
plt.yticks(fontsize=16)
plt.tick_params(axis='x', which='both', direction='in', pad=6)  # Adjust pad to move x tick labels lower
plt.tick_params(axis='y', which='both', direction='in')


plt.xlabel('M $[M_{\odot}]$', fontsize=18, labelpad=5)
plt.ylabel(r'Ejected mass efficency $f_{rec}(M, Z)$', fontsize=18,labelpad=7)
plt.tick_params(axis='both', direction='in', which='both')
ax = plt.gca()  # Get the current axes
ax.yaxis.set_ticks_position('both')  # Show ticks on both sides
ax.tick_params(axis='y', which='both', direction='in')
plt.legend(fontsize=13) 
plt.savefig(f'../Images/Mar/AGB_eff_marMrg.pdf', format='pdf')
plt.show()


# In[25]:


#calculate the mass ejection rate for AGB

AGB_ejec = np.zeros(len(masses))
for i in range(len(masses) - 1):
        if  0.6<=masses[i] <= 5 and i + 1 < len(masses):
            value = -(10**ej_eff_AGB( np.log10(masses[i + 1]))*masses[i+1] * IMF(masses[i + 1]) + 10**ej_eff_AGB( np.log10(masses[i]))*masses[i] * IMF(masses[i])) * (masses[i + 1] - masses[i]) / (2 * (lifetime[i + 1] -lifetime[i]))
            AGB_ejec[i] = value
        elif 5 < masses[i] <8:
            value = -(10**ej_eff_AGB( np.log10(5))*masses[i+1] * IMF(masses[i + 1]) + 10**ej_eff_AGB( np.log10(5))*masses[i] * IMF(masses[i])) * (masses[i + 1] - masses[i]) / (2 * (lifetime[i + 1] - lifetime[i]))
            AGB_ejec[i] = value
        
        else:
            AGB_ejec[i] = 0
            


# In[26]:





# In[27]:


file="../COnfronto/AGB_SN_mar_Z_"+str(z)+"_noWcsv"

df = pd.DataFrame({"t":lifetime, 'AGB':AGB_ejec , 'SNII': ejection_SNII })

df.to_csv(file, index=False, header=True)


# In[28]:


#add winds from FIRE-2 hokins et al. 2017

#define t from 10^5 yr
t=np.linspace(1*10**5, lifetime[0], 100 )

#addig 100 elements previusly define to the already calculated lifetime 
tt=np.append(t, lifetime)
ejection_SNII_=0

OBW_fire2=OB_AGB_mass_loss(tt, z)/1e9

AGB_ejec_=np.append(np.zeros(100), AGB_ejec)
ejection_SNII_=np.append(np.zeros(100), ejection_SNII)
ejection_SNII2_=np.append(np.zeros(100), ejection_SNII2)


# In[29]:


#plotting all the rates together

plt.figure(figsize=(7.7, 5.2)) 

plt.plot(tt, AGB_ejec_, label='AGB', color=CB_color_cycle[7])
plt.plot(tt, ejection_SNII_, label='SNII+Winds', color='sandybrown')
plt.plot(tt, OBW_fire2, label='SNII+Winds Fire2', color=CB_color_cycle[2])


plt.xscale('log')
plt.yscale('log')
plt.xlim(5*10**5, 10**10)

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
plt.savefig(f'../Images/Mar/mass_rate_mar.pdf', format='pdf')
plt.show()


# In[30]:





# In[31]:


#calculate the cumulative mass ejection for the agb and SNII

comulative_AGB=np.zeros(1000) 
for i in range(1000-1):
        if 0.8 <= masses[i] <= 8:
            value = (AGB_ejec[i+1] + AGB_ejec[i]) * (lifetime[i+1] -lifetime[i]) / 2 + comulative_AGB[i]
            comulative_AGB[i+1] = value
        else:
            comulative_AGB[i+1] = comulative_AGB[i] 

            
comulative_snII=np.zeros(1000)


for i in range(1000-1):
        comulative_snII[i+1]=(ejection_SNII2[i+1]+ejection_SNII2[i])*(lifetime[i+1]-lifetime[i])/2+comulative_snII[i]
        


# 

# In[32]:


#add the fist 100 elemnts of time to the rates that we inizialized for the winds
comulativesnII=0
comulativeAGB=0

comulativesnII=np.append(np.zeros(100), comulative_snII)
comulativeAGB=np.append(np.zeros(100), comulative_AGB)

comulative2=np.append(np.zeros(100), comulative_snII)

totW=np.zeros(len(tt))

#calculate the total winds ejection
for i in range(len(tt) - 1):
            value = (OBW_fire2[i+1] + OBW_fire2[i]) * (tt[i+1] - tt[i]) / 2 + totW[i]
            totW[i+1] = value


# In[33]:


#plotting the total cumulative functions

plt.figure(figsize=(7.7, 5.2))

plt.plot(tt, totW+comulativesnII+comulativeAGB, label='AGB + SNII + Fire-2 Winds', color=CB_color_cycle[7])
plt.plot(tt, totW+comulativesnII, label='SNII + Winds Fire-2' ,color=CB_color_cycle[0] )
plt.plot(tt, totW, label='Winds Fire-2' ,color=CB_color_cycle[2] )


plt.xscale('log')
plt.xlim(10**5, 1e10)

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
plt.savefig(f'../Images/Mar/mass_tot_mar.pdf', format='pdf')
plt.show()
print(totW[-1])
print(totW[-1]+comulativesnII[-1]+comulativeAGB[-1])


# In[34]:


file="../CONFRONTOFINALE/MAR_Z_"+str(z)+".csv"

dp = pd.DataFrame({"t":tt, 'AGB':AGB_ejec_ , 'SNII':ejection_SNII_, 'Winds':OBW_fire2, "tot_AGB":comulativeAGB, "tot_SNII":comulativesnII, "tot_winds":totW })

dp.to_csv(file, index=False, header=True)





#define the inital ratio of abundance of elemnts

Z_sol=0.02
###/* initial gas abundances */
H_in=0.75      #INITIAL_ABUNDANCE_HYDROGEN HYDROGEN_MASSFRAC
He_in=1.-H_in  # (1. - GFM_INITIAL_ABUNDANCE_HYDROGEN)
C_in=0
N_in=0
O_in=0
Ne_in=0
Mg_in=0
Si_in=0
Fe_in=0


#/* solar gas abundances (from Asplund+ 2009) */
H=H_in+(0.7388-H_in)/Z_sol*z#define GFM_SOLAR_ABUNDANCE_HYDROGEN 0.7388
He=He_in+(1.-0.7388-Z_sol-He_in)/Z_sol*z#
C=(0.0024)/Z_sol*z
N=0.0007/Z_sol*z
O=0.0057/Z_sol*z
Ne=0.0012/Z_sol*z
Mg=0.0007/Z_sol*z
Si=0.0007/Z_sol*z
Fe=0.0013/Z_sol*z
#define GFM_SOLAR_ABUNDANCE_IRON 0.0013
#define GFM_SOLAR_ABUNDANCE_OTHER 0


#define GFM_SOLAR_METALLICITY 0.0127

#GFM_INITIAL_ABUNDANCE_HYDROGEN + zsolar * (GFM_SOLAR_ABUNDANCE_HYDROGEN
#- GFM_INITIAL_ABUNDANCE_HYDROGEN);
abundance = np.array([H, He, C, N, O, Ne, Mg, Si, Fe])


# In[35]:


elements=["H", "He", "C", "N", "O", "Ne","Mg", "Si", "Fe"]


# In[36]:


#define a dictionary for every yield and getting them from file for SNII

YieldsSNII={}
file_path = 'SNII_2.hdf5'  
for i in metallicities:
    with h5py.File(file_path, 'r') as file:
        stellar_masses = file['Masses'][:]
        YieldsSNII[f"{i}"]=file[f'Yields/{i}/Yield'][:][:]



yieldsSNII = {}
for y in elements:
    yieldsSNII[f"{y}"] = np.zeros((len(metallicities), len(mass_table2)))
Y=0
for i in range(len(metallicities)):
    for y in range(len(elements)):
        for j in range(len(mass_table2)):
            yieldsSNII[f"{elements[y]}"][i][j] = YieldsSNII[f"{metallicities[i]}"][y][j]


            
            
#same for AGB
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


            
#define the total metal ejection for SNII            
metalej={}
file_path = 'SNII_2.hdf5'  
for i in metallicities:
    with h5py.File(file_path, 'r') as file:
        metalej[f"{i}"]=file[f'Yields/{i}/Total_Metals'][:]

        
metal_yieldSNII = np.empty((len(metallicities), len(mass_table2)))
for i in range(len(metallicities)):
    for j in range(len(mass_table2)):
        metal_yieldSNII[i][j]=np.array(metalej[f"{metallicities[i]}"][j])
        

        
        
#same procedure for AGB
metalej={}
file_path = 'AGB_Margio.hdf5'  
for i in metallicities2:
    with h5py.File(file_path, 'r') as file:
        metalej[f"{i}"]=file[f'Yields/{i}/Total_Metals'][:]

metal_yieldAGB = np.empty((len(metallicities2), len(stellar_masses)))
for i in range(len(metallicities2)):
    for j in range(len(stellar_masses)):
        metal_yieldAGB[i][j]=np.array(metalej[f"{metallicities2[i]}"][j])


# In[37]:


#function used to inteprolate between meteallycity the yields

def Z_yield( Z, mass, lifetime, metallicities):
    a=np.shape(lifetime)
    for i in range(len(metallicities)-1):
        if metallicities[i]<=Z and Z<metallicities[i+1]:
            z=i
        elif Z>=metallicities[-1]:
            z=i
        elif Z<metallicities[0]:
            z=0
    
    
    
    lifetime_z=np.zeros(a[0])
    for i in range(a[0]):
        lifetime_z[i]=lifetime[i][z]+(lifetime[i][z+1]-lifetime[i][z])/(metallicities[z+1]-metallicities[z])*(Z-metallicities[z])
    
    lif_z = interp1d(np.log10(mass), lifetime_z, kind="linear", fill_value='extrapolate')
    return lif_z


# In[52]:


#interpolated function of the yields
yieldSNII_Z={}

for y in elements:
    yield_=yieldsSNII[f"{y}"]
    yield_=np.transpose(yield_)
    yieldSNII_Z[f"{y}"]=Z_yield( z, mass_table2, yield_, mett)
    
     


# In[ ]:



    


# In[ ]:





# In[53]:


X="Fe"

plt.plot(masses, yieldSNII_Z[str(X)](np.log10(masses)), label='Metallicity: '+str(z))
plt.scatter(mass_table2, yieldsSNII[str(X)][2][:])
plt.scatter(mass_table2, yieldsSNII[str(X)][3][:])
#plt.plot(lifetime, comulative_snII+comulative_AGB, label='AGB+SNII, Z=0.02')

#plt.plot(lifetime[82][:], comulative_SNII[82][:]+comulative_AGB[82][:], label='Time interpolation')
#plt.plot(lifetime[99][:], comulative_SNII[99][:]+comulative_AGB[99][:], label='Time interpolation')
#plt.plot(lifetime[0][:], comulative_SNII[0], label='Time interpolation')
#plt.plot(t, comulative_snII['Z_0.02'], label='Mass interpolation')
#plt.plot(t, comulative_snII_q, label='Quadratic interpolation')
#plt.xscale('log')
#plt.yscale('log')
plt.xlabel('M $[M_{\odot}]$', fontsize=14)
plt.ylabel('Yield for "' +str(X)+'" $[M_{\odot}]$', fontsize=14)
plt.title(r'\textbf{Yield for '+str(X)+' at Z='+str(z), fontsize=19)
plt.legend()
ax = plt.gca()  # Get the current axes
ax.yaxis.set_ticks_position('both')  # Show ticks on both sides
ax.tick_params(axis='y', which='both', direction='in')
plt.show()


# In[54]:


#plotting the yields
def plot_yieldSNII_Z(yieldSNII_Z):
    # Determine the grid size (3x3)
    n_plots = len(yieldSNII_Z)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols
    i = 0
    
    # Create the figure and GridSpec object
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 15))
    
    viridis = cm.get_cmap('viridis', len(elements))
    
    # Flatten the axis array if it's not already flat
    axs = axs.flat if isinstance(axs, np.ndarray) else [axs]
    
    # Plot each dataset
    for (y, func), ax in zip(yieldSNII_Z.items(), axs):
        # Get the color from Viridis colormap
        color = viridis(i)
        
        # Plot the interpolated data
        x_vals = np.log10(mass_table2)
        y_vals = func(x_vals)
        ax.plot(10**x_vals, y_vals, label='Yield for'+y+' at Z=0.01', color=color)
        
        # Add scatter plots
        ax.scatter(mass_table2, yieldsSNII[str(y)][2][:])
        ax.scatter(mass_table2, yieldsSNII[str(y)][3][:])
        
        # Add a legend with the element name
        ax.legend(fontsize=15)
        
        # Ensure ticks are enabled for all subplots
        ax.tick_params(axis='both', which='major', labelsize=18, direction='in')
        ax.tick_params(axis='both', which='minor', direction='in')
        
        # Add labels to x and y axes
        ax.set_xlabel('M $[M_{\odot}]$', fontsize=18)
        ax.set_ylabel(r'Yield $y_{\text{' + y + '}}(M, Z)$ [$M_{\odot}$]', fontsize=16)

        i += 1
    
    # Adjust layout to ensure labels fit and increase distance between plots
    plt.tight_layout(pad=3)
    plt.subplots_adjust(wspace=0.25, hspace=0.2)

    # Save or show the figure
    plt.savefig(f'../Images/Mar/totYeilds_mar.pdf', format='pdf')
    # Save the figure if needed
    plt.show()

plot_yieldSNII_Z(yieldSNII_Z)


# In[42]:


#ejection rate for single element

ej_SNII_elements={}
    
elements=["H", "He", "C", "N", "O", "Ne","Mg", "Si", "Fe"]
value=0
for x in range(len(elements)):
    ej_SNII_elements[f"{elements[x]}"] =np.zeros(len(masses)) 
    for i in range(len(masses) - 1):
        if 8 <= mass[i] <= 100 and i + 1 < len(masses):
             ej_SNII_elements[f"{elements[x]}"][i]= -((10**ej_eff2(np.log10(masses[i + 1]))*masses[i+1]*abundance[x]+yieldSNII_Z[f"{elements[x]}"](np.log10(masses[i+1])))* IMF(masses[i + 1]) + (10**ej_eff2(np.log10(masses[i]))*masses[i]*abundance[x]+yieldSNII_Z[f"{elements[x]}"](np.log10(masses[i])))* IMF(masses[i])) * (masses[i + 1] - masses[i]) / (2 * (lifetime[i + 1] - lifetime[i]))
            
            
        else:
            ej_SNII_elements[f"{elements[x]}"][i] = 0


# In[43]:


#plot of the fractional yield

totej=np.zeros(len(lifetime))
for i in elements:
    totej=ej_SNII_elements[f"{i}"]+totej

np.seterr(divide='ignore', invalid='ignore')
plt.plot(lifetime,  ej_SNII_elements["O"]/ejection_SNII, label='Element: O')
plt.plot(lifetime,  ej_SNII_elements["Fe"]/ejection_SNII, label='Element: Fe')
plt.plot(lifetime,  ej_SNII_elements["Mg"]/ejection_SNII, label='Element: Mg')
#plt.plot(lifetime,  1-totej/ejection_SNII, label='Element: Missing part')
#plt.ploMgt(lifetime,  ejection_SNII, label='Metallicity:' +str(z))
#plt.scatter(lifetime[i], -(ej_eff(np.log10(masses[i + 1])) * IMF(masses[i + 1]) * masses[i + 1] + ej_eff(np.log10(masses[i])) * IMF(masses[i]) * masses[i]) * (masses[i + 1] - masses[i]) / (2 * (lifetime[i + 1] - lifetime[i])))
#plt.scatter(lifetime[500], -(10**ej_eff(np.log10(masses[500 + 1])) * IMF(masses[500 + 1]) * masses[500 + 1] + 10**ej_eff(np.log10(masses[500])) * IMF(masses[i]) * masses[500]) * (masses[500 + 1] - masses[500]) / (2 * (lifetime[500 + 1] - lifetime[500])))
#plt.plot(t, ejection_SNII_q, label='Quadratic interpolation')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('t $[yr]$', fontsize=14)
#plt.ylabel('Ejection rate fraction', fontsize=14)
plt.title(r'\textbf{Ejection rate fraction}', fontsize=19)
plt.legend()
plt.xlim(10**6, 10**9)
ax = plt.gca()  # Get the current axes
ax.yaxis.set_ticks_position('both')  # Show ticks on both sides
ax.tick_params(axis='y', which='both', direction='in')
plt.show()


# In[44]:




for i in elements:
    ej_SNII_elements[f"{i}"]=ej_SNII_elements[f"{i}"]/ejection_SNII


with open("../COnfronto/elemnt_ej_noW_ior.csv", mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)

    # Write header (element names)
    writer.writerow(elements)

    # Write data columns
    for i in range(len(ej_SNII_elements["H"])):
        writer.writerow([str(ej_SNII_elements[element][i]) for element in elements])
        
for i in elements:
    ej_SNII_elements[f"{i}"]=ej_SNII_elements[f"{i}"]*ejection_SNII


# In[45]:


#same procedure for AGB yields

yieldAGB_Z={}

for y in elements:
    yield_=yieldsAGB[f"{y}"]
    yield_=np.transpose(yield_)
    yieldAGB_Z[f"{y}"]=Z_yield( z, stellar_masses, yield_, mettAGB)
    
    
    


# In[46]:


X="C"

def plot_yieldSNII_Z(yieldSNII_Z):
    # Determine the grid size (3x3)
    n_plots = len(yieldSNII_Z)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols
    i = 0
    
    # Create the figure and GridSpec object
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 15))
    
    viridis = cm.get_cmap('viridis', len(elements))
    
    # Flatten the axis array if it's not already flat
    axs = axs.flat if isinstance(axs, np.ndarray) else [axs]
    
    # Plot each dataset
    for (y, func), ax in zip(yieldAGB_Z.items(), axs):
        # Get the color from Viridis colormap
        color = viridis(i)
        
        # Plot the interpolated data
        x_vals = np.log10(stellar_masses)
        y_vals = func(x_vals)
        ax.plot(10**x_vals, y_vals, label='Yield for '+y+' at Z=0.01', color=color)
        
        # Add scatter plots
        ax.scatter(stellar_masses, yieldsAGB[str(y)][1][:])
        ax.scatter(stellar_masses, yieldsAGB[str(y)][2][:])
        
        # Add a legend with the element name
        ax.legend(fontsize=15)
        
        # Ensure ticks are enabled for all subplots
        ax.tick_params(axis='both', which='major', labelsize=18, direction='in')
        ax.tick_params(axis='both', which='minor', direction='in')
        
        # Add labels to x and y axes
        ax.set_xlabel('M $[M_{\odot}]$', fontsize=18)
        ax.set_ylabel(r'Yield $y_{\text{' + y + '}}(M, Z)$ [$M_{\odot}$]', fontsize=16)

        i += 1
    
    # Adjust layout to ensure labels fit and increase distance between plots
    plt.tight_layout(pad=3)
    plt.subplots_adjust(wspace=0.25, hspace=0.2)

    # Save or show the figure
    plt.savefig(f'../Images/Mar/totYeilds_marAGB.pdf', format='pdf')
    # Save the figure if needed
    plt.show()

plot_yieldSNII_Z(yieldAGB_Z)


# In[47]:


ej_AGB_elements={}
    #for i in range(len()):
value=0
elements=["H", "He", "C", "N", "O", "Ne","Mg", "Si", "Fe"]

for x in range(len(elements)):
    ej_AGB_elements[f"{elements[x]}"]=np.zeros(len(masses))
    for i in range(len(masses) - 1):
        if 0.85<=masses[i] <= 8 and i + 1 < len(masses):
            ej_AGB_elements[f"{elements[x]}"][i] = -((10**ej_eff_AGB( np.log10(masses[i + 1]))*masses[i+1]*abundance[x]+yieldAGB_Z[f"{elements[x]}"](np.log10(masses[i+1])) )* IMF(masses[i + 1]) + (10**ej_eff_AGB( np.log10(masses[i]))*masses[i]*abundance[x]+yieldAGB_Z[f"{elements[x]}"](np.log10(masses[i]))) * IMF(masses[i])) * (masses[i + 1] - masses[i]) / (2 * (lifetime[i + 1] - lifetime[i]))
            
        elif 5 < masses[i] < 8:
            ej_AGB_elements[f"{elements[x]}"][i] = -((10**ej_eff_AGB(np.log10(5))*masses[i+1]*abundance[x]+yieldAGB_Z[f"{elements[x]}"](np.log10(5))) * IMF(masses[i + 1]) + (10**ej_eff_AGB( np.log10(5))*masses[i]*abundance[x]+yieldAGB_Z[f"{elements[x]}"](np.log10(5))) * IMF(masses[i])) * (masses[i + 1] - masses[i]) / (2 * (lifetime[i + 1] - lifetime[i]))
            
        
        else:
            ej_AGB_elements[f"{elements[x]}"][i] = 0
            


# In[ ]:





# In[48]:


totej=np.zeros(len(lifetime))
for i in elements:
    totej=ej_SNII_elements[f"{i}"]+totej

np.seterr(divide='ignore', invalid='ignore')
plt.plot(lifetime,  ej_SNII_elements["O"]/ejection_SNII, label='Element: O', color='blue')
plt.plot(lifetime,  ej_SNII_elements["C"]/ejection_SNII, label='Element: C', color='red')
plt.plot(lifetime,  ej_SNII_elements["N"]/ejection_SNII, label='Element: N', color='green')
plt.plot(lifetime,  ej_AGB_elements["O"]/AGB_ejec, color='blue')
plt.plot(lifetime,  ej_AGB_elements["C"]/AGB_ejec, color='red')
plt.plot(lifetime,  ej_AGB_elements["N"]/AGB_ejec, color='green')
#plt.plot(lifetime,  1-totej/ejection_SNII, label='Element: Missing part')
#plt.ploMgt(lifetime,  ejection_SNII, label='Metallicity:' +str(z))
#plt.scatter(lifetime[i], -(ej_eff(np.log10(masses[i + 1])) * IMF(masses[i + 1]) * masses[i + 1] + ej_eff(np.log10(masses[i])) * IMF(masses[i]) * masses[i]) * (masses[i + 1] - masses[i]) / (2 * (lifetime[i + 1] - lifetime[i])))
#plt.scatter(lifetime[500], -(10**ej_eff(np.log10(masses[500 + 1])) * IMF(masses[500 + 1]) * masses[500 + 1] + 10**ej_eff(np.log10(masses[500])) * IMF(masses[i]) * masses[500]) * (masses[500 + 1] - masses[500]) / (2 * (lifetime[500 + 1] - lifetime[500])))
#plt.plot(t, ejection_SNII_q, label='Quadratic interpolation')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('t [yr]', fontsize=14)
plt.ylabel('')
plt.title(r'\textbf{Ejected mass fraction rate for both AGB and SnII}', fontsize=19 )
plt.legend()
plt.xlim(10**6, 10**9)
ax = plt.gca()  # Get the current axes
ax.yaxis.set_ticks_position('both')  # Show ticks on both sides
ax.tick_params(axis='y', which='both', direction='in')
plt.show()


# In[49]:


for i in elements:
    ej_AGB_elements[f"{i}"]=ej_AGB_elements[f"{i}"]/AGB_ejec


with open("../COnfronto/elemntAGB_ej_noW_ior.csv", mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)

    # Write header (element names)
    writer.writerow(elements)

    # Write data columns
    for i in range(len(ej_SNII_elements["H"])):
        writer.writerow([str(ej_AGB_elements[element][i]) for element in elements])
        
for i in elements:
    ej_AGB_elements[f"{i}"]=ej_AGB_elements[f"{i}"]*AGB_ejec


# In[50]:


#getting the interpolated function for the total metal yield ofboth AGB and SNII


yieldmetalSNII=Z_yield( z, mass_table2, (np.transpose(metal_yieldSNII)), mett)  
yieldmetalAGB=Z_yield( z, stellar_masses, (np.transpose(metal_yieldAGB)), mettAGB) #ej_interpolation( z, stellar_masses, metal_yieldAGB, mettAGB)  


# In[126]:





# In[127]:


plt.figure(figsize=(7.7, 5.2)) 
plt.plot(masses[:650], yieldmetalSNII(np.log10(masses[:650])), label='Metal yield at Z='+str(z), color=CB_color_cycle[2])
plt.scatter(mass_table2, metal_yieldSNII[2][:], label='Data for Z=0.008',  color=CB_color_cycle[0])
plt.scatter(mass_table2, metal_yieldSNII[3][:],  label='Data for Z=0.02', color=CB_color_cycle[1])
plt.plot(masses[650:], yieldmetalAGB(np.log10(masses[650:])), color=CB_color_cycle[2])
plt.scatter(AGB_mass, metal_yieldAGB[1][:], color=CB_color_cycle[0])
plt.scatter(AGB_mass, metal_yieldAGB[2][:], color=CB_color_cycle[1])
plt.xscale('log')
#plt.yscale('log')
#plt.xlim(1.4, 10**2)

plt.xticks(fontsize=17)
plt.yticks(fontsize=16)
plt.tick_params(axis='x', which='both', direction='in', pad=6)  # Adjust pad to move x tick labels lower
plt.tick_params(axis='y', which='both', direction='in')


plt.xlabel('M $[M_{\odot}]$', fontsize=18, labelpad=5)
plt.ylabel(r'SnII metal yield [$M_{\odot}$]', fontsize=18,labelpad=7)
plt.tick_params(axis='both', direction='in', which='both')
ax = plt.gca()  # Get the current axes
ax.yaxis.set_ticks_position('both')  # Show ticks on both sides
ax.tick_params(axis='y', which='both', direction='in')
plt.legend(fontsize=13) 
plt.savefig(f'../Images/Mar/yield_met.pdf', format='pdf')
plt.show()


# In[128]:


#calculating the metals ejection rate of both AGB and SNII

ejection_metalSNII = np.zeros(len(masses))
ejection_metalAGB=np.zeros(len(masses))


for i in range(len(masses) - 1):
    if 8 <= mass[i] <= 100 and i + 1 < len(masses):
        ejection_metalSNII[i] = -((10**ej_eff2(np.log10(masses[i + 1])) * masses[i + 1] * z + yieldmetalSNII(np.log10(masses[i + 1]))) * IMF(masses[i + 1]) + (10**ej_eff2(np.log10(masses[i])) * masses[i] * z + yieldmetalSNII(np.log10(masses[i]))) * IMF(masses[i])) * (masses[i + 1] - masses[i]) / (2 * (lifetime[i + 1] - lifetime[i]))
    else:
        ejection_metalSNII[i] = 0
        
for i in range(len(masses) - 1):
    if 0.1<=masses[i] <=5 and i + 1 < len(masses):
        ejection_metalAGB[i] = -((10**ej_eff_AGB(np.log10(masses[i + 1]))*masses[i+1]* z + yieldmetalAGB(np.log10(masses[i + 1]))) * IMF(masses[i + 1]) + (10**ej_eff_AGB(np.log10(masses[i]))*masses[i] * z + yieldmetalAGB(np.log10(masses[i]))) * IMF(masses[i])) * (masses[i + 1] - masses[i]) / (2 * (lifetime[i + 1] - lifetime[i]))
        
    elif 5 < masses[i] < 8:
            ejection_metalAGB[i] = -((10**ej_eff_AGB(np.log10(5))*masses[i+1]* z + yieldmetalAGB(np.log10(5))) * IMF(masses[i + 1]) + (10**ej_eff_AGB(np.log10(5))*masses[i] * z + yieldmetalAGB(np.log10(5))) * IMF(masses[i])) * (masses[i + 1] - masses[i]) / (2 * (lifetime[i + 1] - lifetime[i]))
    else:
        ejection_metalAGB[i] = 0


# In[129]:





# In[130]:


np.seterr(divide='ignore', invalid='ignore')
plt.figure(figsize=(7.7, 5.2))


plt.plot(lifetime, ejection_metalSNII, label='SNII')
plt.plot(lifetime, ejection_metalAGB, label='AGB')

plt.xlim(2*1e6, 1e10)
plt.xscale('log')
plt.yscale('log')


plt.xticks(fontsize=17)
plt.yticks(fontsize=16)
plt.tick_params(axis='x', which='both', direction='in', pad=6)  # Adjust pad to move x tick labels lower
plt.tick_params(axis='y', which='both', direction='in')


plt.xlabel('SSP age $[yr]$', fontsize=18, labelpad=5)
plt.ylabel(r'Metal ejection rate [$yr^{-1}$]', fontsize=18,labelpad=7)
plt.tick_params(axis='both', direction='in', which='both')
ax = plt.gca()  # Get the current axes
ax.yaxis.set_ticks_position('both')  # Show ticks on both sides
ax.tick_params(axis='y', which='both', direction='in')
plt.legend(fontsize=13) 
plt.savefig(f'../Images/Mar/yield_met_rate.pdf', format='pdf')
plt.show()


# In[131]:


file="../COnfronto/tot_Z_"+str(z)+"_ior.csv"

dp = pd.DataFrame({"t":lifetime, 'EJ_Z_SNII':ejection_metalSNII, 'EJ_Z_AGB': ejection_metalAGB})

dp.to_csv(file, index=False, header=True)


# In[132]:


#getting the toal mass ejection rate by summing the ejection rate of single elemnts

totSNII=np.zeros(len(masses))
totAGB=np.zeros(len(masses))

for i in elements:
    totAGB+=ej_AGB_elements[f"{i}"]

for i in elements:
    totSNII+=ej_SNII_elements[f"{i}"]
    


# In[ ]:





# In[ ]:





# In[108]:


#total cumulative metal mass ejection

cum_metalsSN=np.zeros(len(lifetime))
cum_metalsAGB=np.zeros(len(lifetime))

for i in range(len(lifetime)-1):
        cum_metalsSN[i+1]=(ejection_metalSNII[i+1]+ejection_metalSNII[i])*(lifetime[i+1]-lifetime[i])/2+cum_metalsSN[i]
        cum_metalsAGB[i+1]=(ejection_metalAGB[i+1]+ejection_metalAGB[i])*(lifetime[i+1]-lifetime[i])/2+cum_metalsAGB[i]
        


# In[134]:


#cumulative single element mass ejection for SN

comulative_elementSN={} 

for j in elements:
    comulative_elementSN[f"{j}"] = np.zeros(len(masses))
    for i in range(len(masses) - 1):
        if 8 <= masses[i] <= 100 and not np.isnan(comulative_elementSN[f"{j}"][i]):
            comulative_elementSN[f"{j}"][i+1] = (
                (ej_SNII_elements[f'{j}'][i+1] + ej_SNII_elements[f'{j}'][i]) * (lifetime[i+1] - lifetime[i]) / 2 
                + comulative_elementSN[f'{j}'][i]
            )
        else:
            comulative_elementSN[f'{j}'][i] = comulative_elementSN[f'{j}'][i-1]
            comulative_elementSN[f'{j}'][i+1] = comulative_elementSN[f'{j}'][i]

totejSN = np.zeros(len(masses))          
for i in elements:
    totejSN += comulative_elementSN[f"{i}"]


# In[135]:


#cumulative single element mass ejection for AGB

comulative_elementAGB={} 

for j in elements:
    comulative_elementAGB[f"{j}"] = np.zeros(len(masses))
    for i in range(len(masses) - 1):
        if 0.85 <= masses[i] <=8 and not np.isnan(comulative_elementAGB[f"{j}"][i]):
            comulative_elementAGB[f"{j}"][i+1] = (ej_AGB_elements[f'{j}'][i+1] + ej_AGB_elements[f'{j}'][i]) * (lifetime[i+1] - lifetime[i]) / 2 + comulative_elementAGB[f'{j}'][i]
  
        else:
            comulative_elementAGB[f'{j}'][i] = comulative_elementAGB[f'{j}'][i-1]
            comulative_elementAGB[f'{j}'][i+1] = comulative_elementAGB[f'{j}'][i]

totejAGB = np.zeros(len(masses))          
for i in elements:
    totejAGB += comulative_elementAGB[f"{i}"]


# 

# In[136]:


#plot the cumulative

plt.figure(figsize=(7.7, 5.2))
plt.plot(lifetime, cum_metalsSN+cum_metalsAGB, label='AGB', color='C1') 
plt.plot(lifetime, cum_metalsSN, label='SNII' , color='C0')


plt.xlim(2*1e6, 1e10)
plt.xscale('log')



plt.xticks(fontsize=17)
plt.yticks(fontsize=16)
plt.tick_params(axis='x', which='both', direction='in', pad=6)  # Adjust pad to move x tick labels lower
plt.tick_params(axis='y', which='both', direction='in')


plt.xlabel('SSP age $[yr]$', fontsize=18, labelpad=5)
plt.ylabel(r'Metal ejection fraction', fontsize=18,labelpad=7)
plt.tick_params(axis='both', direction='in', which='both')
ax = plt.gca()  # Get the current axes
ax.yaxis.set_ticks_position('both')  # Show ticks on both sides
ax.tick_params(axis='y', which='both', direction='in')
plt.legend(fontsize=13) 
plt.savefig(f'../Images/Mar/yield_met_tot.pdf', format='pdf')
plt.show()
print(cum_metalsSN[-1]+cum_metalsAGB[-1])


# In[137]:


#plot the cumulative for each element

def plot_yieldSNII_Z(yieldSNII_Z):
    # Determine the grid size (3x3)
    n_plots = len(yieldSNII_Z)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols
    i = 0
    
    # Create the figure and GridSpec object
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(21, 12))
    
    viridis = cm.get_cmap('viridis', len(elements))
    viridis2 = cm.get_cmap('viridis', len(elements))
    # Flatten the axis array if it's not already flat
    axs = axs.flat if isinstance(axs, np.ndarray) else [axs]
    
    # Plot each dataset
    for (y, func), ax in zip(yieldSNII_Z.items(), axs):
        # Get the color from Viridis colormap
        color = viridis(i)
        color2=viridis2(len(elements)-i)
        # Plot the interpolated data
        
        ax.plot(tt, np.append(np.zeros(100), comulative_elementSN[str(y)]+comulative_elementAGB[str(y)])+totW*abundance[i], label='AGB', color=color2)
        ax.plot(tt, np.append(np.zeros(100), comulative_elementSN[str(y)])+totW*abundance[i], label='SNII', color=color)
        ax.plot(tt, totW*abundance[i], label='Winds', color="forestgreen")
        
        # Add scatter plots
      
        
        # Add a legend with the element name
        ax.legend(fontsize=18)
        
        # Ensure ticks are enabled for all subplots
        ax.tick_params(axis='both', which='major', labelsize=17, direction='in')
        ax.tick_params(axis='both', which='minor', direction='in')
        
        # Add labels to x and y axes
        ax.set_xlabel('SSP age $[yr]$', fontsize=23)
        ax.set_ylabel(r'Ejection fraction of '+y, fontsize=23)
        ax.set_xscale('log')
        ax.set_xlim(1e5, 1e10)
        i += 1
    
    # Adjust layout to ensure labels fit and increase distance between plots
    plt.tight_layout(pad=3)
    plt.subplots_adjust(wspace=0.27, hspace=0.3)

    # Save or show the figure
    plt.savefig(f'../Images/Mar/totYeilds_mar_all.pdf', format='pdf')
    # Save the figure if needed
    plt.show()

plot_yieldSNII_Z(yieldSNII_Z)


# In[229]:


metals=["C", "N", "O", "Ne","Mg", "Si", "Fe"]

totSN=np.zeros(len(tt))
totAGB=np.zeros(len(tt))
totWM=np.zeros(len(tt))
for i, element in enumerate(metals):
        totSN+=np.append(np.zeros(100),comulative_elementSN[element])
        totAGB+=np.append(np.zeros(100), comulative_elementAGB[element])
        totWM+=totW*abundance[i+2]


# In[233]:


plt.figure(figsize=(7.7, 5.2))


plt.plot(tt, totWM+totSN+totAGB, label='AGB + SNII + Fire-2 Winds', color=CB_color_cycle[7])
plt.plot(tt, totWM+totSN, label='SNII + Winds Fire-2' ,color=CB_color_cycle[0] )
plt.plot(tt, totWM, label='Winds Fire-2' ,color=CB_color_cycle[2] )





plt.xscale('log')

plt.xlim(10**5, 1e10)

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
plt.savefig(f'../Images/Mar/yield_met_tot.pdf', format='pdf')
plt.show()
print(totW[-1])
print(totW[-1]+comulativesnII[-1]+comulativeAGB[-1])


# In[145]:


cum={}
for i, element in enumerate(elements):
    cum[f"{element}"]=np.append(np.zeros(100), comulative_elementSN[element]+comulative_elementAGB[element])+totW*abundance[i]


# In[147]:


df = pd.DataFrame(cum)

# Write the DataFrame to a CSV file
df.to_csv('../CONFRONTOFINALE/tot_elemnt_ej_mar.csv', index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




