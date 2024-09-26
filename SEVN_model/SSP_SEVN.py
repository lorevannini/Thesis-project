#!/usr/bin/env python
# coding: utf-8

# In[112]:


import warnings
import csv
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
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"]
})

seconds=3.156*1e+7
grams=1.98892*1e33


# In[113]:



###Function i use un the code#####


#Define an IMF (same used in the volkberger and marinacci)


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
    



#setting log scales
def convert_to_logscale(tn):
    dt=dt=np.log10(tn[len(tn)-1]/tn[0])/(len(tn))
    logt=np.log10(tn[0])+np.arange(len(tn))*dt 
    t=10.**logt
    return t



###This calculate the interpolation function  given a generic z


def Z_interpolation( Z, mass, lifetime, metallicities):
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
    
    lif_z = interp1d(np.log10(mass), np.log10(lifetime_z), kind="linear", fill_value='extrapolate')
    return lif_z
        
    
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
    
    ej_z = interp1d(np.log10(mass), np.log10(ejection_eff_z), kind="linear", fill_value="extrapolate")
    return ej_z


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


# In[114]:


######SELECT THE TRACK THE SN FORMALISM AND THE METALLICITY#######
track=["ov05"]
sn_type=['delayed']
z=[0.02]
######SELECT THE TRACK THE SN FORMALISM AND THE METALLICITY#######


# In[115]:


####WE USE THIS TO IMPORT THE LIFETIME

for k in track:
    for j in sn_type:
        #Importing the data from the Hdf5 file from SMUGGLE code
        file_path = f'Lifetimes_ioro_{k}.hdf5'  
        with h5py.File(file_path, 'r') as file:
            stellar_masses = file['Masses'][:]
            lifetimes_file=file[f'/models/{j}/Lifetimes'][:]
            #metallicity=file['Metallicities'][:]

file_path = 'Lifetimes_ioroAGB_ov0.5.hdf5'  


for j in track:
    with h5py.File(file_path, 'r') as file:
        stellar_massesAGB = file['Masses'][:]
        lifetimes_fileAGB=file[f'/models/{j}/Lifetimes'][:] 
    #making the nparray, select the 3 raw that correspond to the solar metallicity 0.02 

mass_tableAGB=np.array(stellar_massesAGB)
mass_table=np.array(stellar_masses)
mett=np.array([0.0004, 0.004, 0.008, 0.02])
#metallicities in the file        


#We get the lifetime in a matrix
lifetime2 = np.empty((len(mett), len(mass_table)))
lifetime1=np.empty((len(mett), len(mass_tableAGB)))



for i in range (len(mett)):
    for j in range(len(mass_table)):
        lifetime2[i][j]=np.array(lifetimes_file[i][j])
    for j in range(len(mass_tableAGB)):
        lifetime1[i][j]=np.array(lifetimes_fileAGB[i][j])
        
lifetime = np.concatenate((lifetime1, lifetime2), axis=1)
lifetime_=np.transpose(lifetime)



# In[116]:


#getting the ejected mass from files

Ejected_mass_ioro={}
Ejected_mass_mar={}
AGBEjected_mass_ioro={}
winds_mass_ioro={}
AGBwinds_mass_ioro={}
metallicities=['Z_0.0004', 'Z_0.004', 'Z_0.008', 'Z_0.02']


sn_type=["delayed"]
for j in sn_type:
    file_path1 = f'SNII_Ioro__ms.hdf5'
    for i in metallicities:
        with h5py.File(file_path1, 'r') as file:
            stellar_masses_io = file['Masses'][:]
            Ejected_mass_ioro[f"{i}"]=file[f'Yields/{j}/{i}/Ejected_mass'][:]
    
   
file_path1 = f'AGB_Ioro_ov0.5_ms.hdf5'
for i in metallicities:
    with h5py.File(file_path1, 'r') as file:
        AGBstellar_masses_io = file['Masses'][:]
        AGBEjected_mass_ioro[f"{i}"]=file[f'Yields/ov05/{i}/Ejected_mass'][:]


    



# In[117]:


#getting the winds from file


for i in metallicities:
        with h5py.File(f'Winds_Ioro_ov0.5_ms.hdf5', 'r') as file:
            timesnov = np.array(file['Time'][:])
            winds_mass_ioro[f"{i}"]=file[f'Yields/rapid/{i}/Ejected_mass'][:]
for i in metallicities:
        with h5py.File(f'Winds_IoroAGB_ov0.5_ms.hdf5', 'r') as file:
            timeAGBov =  np.array(file['Time'][:])
            AGBwinds_mass_ioro[f"{i}"]=file[f'Yields/ov05/{i}/Ejected_mass'][:]


    

   


# In[118]:


#calculate the ejection rate from files data

ejected_SNII_ioroov = np.empty((len(metallicities), len(mass_table)))
ejected_AGB_ioroov = np.empty((len(metallicities), len(mass_tableAGB)))

ejected_SNII_mar= np.empty((len(metallicities), len(mass_table)))
winds_iorosnov = np.zeros((len(metallicities), len(timesnov)))
AGBwinds_ioroov = np.zeros((len(metallicities), len(timeAGBov)))

for i in range(len(metallicities)):
    for j in range(len(mass_table)):
        ejected_SNII_ioroov[i][j]=np.array(Ejected_mass_ioro[f"{metallicities[i]}"][j])
    
    for j in range(len(mass_tableAGB)):
        ejected_AGB_ioroov[i][j]=np.array(AGBEjected_mass_ioro[f"{metallicities[i]}"][j])
    
    

    for j in range(len(timesnov)):
        winds_iorosnov[i][j]=np.array(winds_mass_ioro[f"{metallicities[i]}"][j])
    
    for j in range(len(timeAGBov)):    
        AGBwinds_ioroov[i][j] = np.array(AGBwinds_mass_ioro[f"{metallicities[i]}"][j])


# In[119]:


##concatenate the times and the masses of AGB and SN tables

mass_tab = np.concatenate((mass_tableAGB, mass_table))
time=np.concatenate((timesnov, timeAGBov))
winds_ioroov=np.concatenate((winds_iorosnov, AGBwinds_ioroov))


# In[120]:


lifetimefunc1=Z_interpolation( z,  mass_tab, lifetime_, mett)
lifetimefunc2=Z_interpolation( z,  mass_tab, lifetime_, mett)
winds_sn=Z_interpolation( z,  timesnov, np.transpose(winds_iorosnov), mett)
winds_AGB=Z_interpolation( z,  timeAGBov, np.transpose(AGBwinds_ioroov), mett)



#this is the mass range scaled in log
masses=np.linspace(100, 0.5, num=1000)
masses=convert_to_logscale(masses)


#timescale given the mass
lifetime1=np.zeros(len(masses))
lifetime2=np.zeros(len(masses))
for i in range(len(masses)):
        lifetime1[i]=10**lifetimefunc1(np.log10(masses[i]))
        lifetime2[i]=10**lifetimefunc2(np.log10(masses[i]))
        


# In[121]:


#plotting the winds from SN and AGB

plt.plot(timesnov,  winds_iorosnov[3][:], label='Parssec ov0.5')
plt.plot(timeAGBov, AGBwinds_ioroov[3][:])

plt.xscale('log')
plt.yscale('log')
plt.xlabel('t [yrs]', fontsize=14)
plt.ylabel('$[M/M_{\odot}$ $yr^{-1}]$', fontsize=14)
plt.title(r'\textbf{Mass enjection rate}', fontsize=19)
plt.legend()
#plt.xlim(10**6, 10**9)
ax = plt.gca()  # Get the current axes
ax.yaxis.set_ticks_position('both')  # Show ticks on both sides
ax.tick_params(axis='y', which='both', direction='in')
plt.show()


# In[122]:


#rate of AGB and SN from files

rate_SNII=np.zeros(len(masses)) 
rate_SNIIov=np.zeros(len(masses)) 
rate_AGB=np.zeros(len(masses))
rate_AGB2=np.zeros(len(masses))
for i in range(len(masses)-1):
        #if 8<=masses[i]<=100:
        #    rate_SNII[i+1]=-(IMF(masses[i+1])+IMF(masses[i]))*(masses[i+1]-masses[i])/(2*(lifetime[i+1]-lifetime[i]))
        if 0.8<=masses[i]<8:      
            rate_AGB[i+1]=-(IMF(masses[i+1])+IMF(masses[i]))*(masses[i+1]-masses[i])/(2*(lifetime1[i+1]-lifetime1[i]))
            rate_AGB2[i+1]=-(IMF(masses[i+1])+IMF(masses[i]))*(masses[i+1]-masses[i])/(2*(lifetime2[i+1]-lifetime2[i]))


# In[123]:


plt.figure(figsize=(7.7, 5.2))

plt.plot(lifetime2,  rate_AGB2 , label='AGB Z=0.02')
plt.plot(lifetime1,  rate_AGB1 , label='AGB Z=0.0004',linestyle="dotted")


plt.xscale("log")
plt.yscale("log")


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
plt.savefig(f'../Images/SEVN/rate_iorio.pdf', format='pdf')
plt.show()


# In[124]:


with open('rate'+str(z)+'.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Lifetime', 'Rate_SNII', 'Rate_AGB'])
    for lt, sn, agb in zip(lifetime, rate_SNII, rate_AGB):
        writer.writerow([lt, sn, agb])


# In[125]:


###I calculate the ejection efficency before doing anything
ejection_eff_ioro= np.empty((len(metallicities), len(mass_table)))
ejection_eff_ioroAGB=np.empty((len(metallicities), len(mass_tableAGB)))
ejection_eff_ioroov= np.empty((len(metallicities), len(mass_table)))
ejection_eff_ioroAGBov=np.empty((len(metallicities), len(mass_tableAGB)))

for i in range(len(mett)):
    for j in range(len(mass_table)):
        
        ejection_eff_ioroov[i][j]=ejected_SNII_ioroov[i][j]/mass_table[j]
    
        
    for j in range(len(mass_tableAGB)):
        
        ejection_eff_ioroAGBov[i][j]=ejected_AGB_ioroov[i][j]/mass_tableAGB[j]
        
ejection_eff_ioroov = np.concatenate((ejection_eff_ioroAGBov, ejection_eff_ioroov), axis=1)
ej_eff_ioroov=ej_interpolation(z, mass_tab, ejection_eff_ioroov, mett)


# In[126]:




plt.scatter(mass_tab, ejection_eff_ioroov[3][:],label='Z=0.02')
plt.plot(masses, 10**ej_eff_ioroov(np.log10(masses)), label='ov0.5')


plt.xlim(10, 120)
plt.xlabel('$M_{\odot}$', fontsize=14)
plt.ylabel('Ejected mass efficiency', fontsize=14)
plt.title(r'\textbf{Ejection efficiency interpolated at Z='f'{z}', fontsize=19)
#plt.xlim(10**-1, 10)
ax = plt.gca()  # Get the current axes
ax.yaxis.set_ticks_position('both')  # Show ticks on both sides
ax.tick_params(axis='y', which='both', direction='in')
plt.legend()
plt.show()


# In[145]:


t=0
#plotting all the rates of new models

ejection_SNII_ioro1=np.zeros(len(lifetime1))
ejection_SNII_ioroov=np.zeros(len(lifetime1))
ejection_SNII_mar=np.zeros(len(lifetime1))
ejection_AGB_ioro=np.zeros(len(lifetime1))
ejection_AGB_ioroov=np.zeros(len(lifetime1))

for i in range(len(masses) - 1):

        if 8 <= masses[i] < 100:
            ejection_SNII_ioroov[i] = -(10**ej_eff_ioroov(np.log10(masses[i + 1]))*masses[i+1]*IMF(masses[i + 1]) + 10**ej_eff_ioroov(np.log10(masses[i]))*IMF(masses[i])*masses[i]) * (masses[i + 1] - masses[i]) / (2 * (lifetime1[i + 1] - lifetime1[i]))
        
            
            
        elif 0.85<=masses[i]<8:
            ejection_AGB_ioroov[i] = -(10**ej_eff_ioroov(np.log10(masses[i + 1]))*masses[i+1]*IMF(masses[i + 1]) + 10**ej_eff_ioroov(np.log10(masses[i]))*IMF(masses[i])*masses[i]) * (masses[i + 1] - masses[i]) / (2 * (lifetime1[i + 1] - lifetime1[i]))
        
            

t=np.append(np.linspace(1e4, lifetime1[0], 100), lifetime1 )           


windsSNov=np.zeros(len(t))
windsAGBov=np.zeros(len(t))
windsov=np.zeros(len(t))

for i in range(len(t)):
    windsSNov[i]=10**winds_sn(np.log10(t[i]))
    windsAGBov[i]=10**winds_AGB(np.log10(t[i]))
    


windsSNov=np.nan_to_num(windsSNov, nan=0.)
windsAGBov=np.nan_to_num(windsAGBov, nan=0.)    
windsov=windsSNov+windsAGBov

zoro=(np.linspace(0, lifetime[-1],100))
zoro=np.zeros(len(zoro))


ejection_SNII_ioroov=np.append(zoro, ejection_SNII_ioroov)
ejection_AGB_ioroov=np.append(zoro, ejection_AGB_ioroov)


# In[146]:



plt.plot(t,  ejection_SNII_ioroov , label='ov0.5')
plt.plot(t,  ejection_AGB_ioroov , label='ov0.5')
plt.plot(t,  windsov , label='ov0.5')
#plt.plot(lifetime,  ejection_SNII_mar, label='Mar')
#plt.plot(lifetime, winds)

#plt.scatter(lifetime[i], -(ej_eff(np.log10(masses[i + 1])) * IMF(masses[i + 1]) * masses[i + 1] + ej_eff(np.log10(masses[i])) * IMF(masses[i]) * masses[i]) * (masses[i + 1] - masses[i]) / (2 * (lifetime[i + 1] - lifetime[i])))
#plt.scatter(lifetime[500], -(10**ej_eff(np.log10(masses[500 + 1])) * IMF(masses[500 + 1]) * masses[500 + 1] + 10**ej_eff(np.log10(masses[500])) * IMF(masses[i]) * masses[500]) * (masses[500 + 1] - masses[500]) / (2 * (lifetime[500 + 1] - lifetime[500])))
#plt.plot(t, ejection_SNII_q, label='Quadratic interpolation')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('t [yrs]', fontsize=14)
plt.ylabel('$[M/M_{\odot}$ $yr^{-1}]$', fontsize=14)
plt.title(r'\textbf{SNII rate}', fontsize=19)
plt.legend()
#plt.xlim(10**6, 10**9)
ax = plt.gca()  # Get the current axes
ax.yaxis.set_ticks_position('both')  # Show ticks on both sides
ax.tick_params(axis='y', which='both', direction='in')
plt.show()


# In[147]:


###visualizing the points to sample the winds


plt.plot(t, windsov, label='ov0.5')



plt.scatter(t[0], windsov[0])
plt.scatter(t[1], windsov[1])
plt.scatter(t[10], windsov[10])
plt.scatter(t[42], windsov[42])
plt.scatter(t[75], windsov[75])
plt.scatter(t[115], windsov[115])
plt.scatter(t[370], windsov[370])
plt.scatter(t[468], windsov[468])
plt.scatter(t[490], windsov[490])
plt.scatter(t[660], windsov[660])
plt.scatter(t[800], windsov[800])
#plt.scatter(lifetime[i], -(ej_eff(np.log10(masses[i + 1])) * IMF(masses[i + 1]) * masses[i + 1] + ej_eff(np.log10(masses[i])) * IMF(masses[i]) * masses[i]) * (masses[i + 1] - masses[i]) / (2 * (lifetime[i + 1] - lifetime[i])))
#plt.scatter(lifetime[500], -(10**ej_eff(np.log10(masses[500 + 1])) * IMF(masses[500 + 1]) * masses[500 + 1] + 10**ej_eff(np.log10(masses[500])) * IMF(masses[i]) * masses[500]) * (masses[500 + 1] - masses[500]) / (2 * (lifetime[500 + 1] - lifetime[500])))
#plt.plot(t, ejection_SNII_q, label='Quadratic interpolation')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('t [yrs]', fontsize=14)
plt.ylabel('$[M/M_{\odot}$ $yr^{-1}]$', fontsize=14)
plt.title(r'\textbf{Mass enjection rate}', fontsize=19)
plt.legend()
#plt.xlim(10**7, 10**9)
ax = plt.gca()  # Get the current axes
ax.yaxis.set_ticks_position('both')  # Show ticks on both sides
ax.tick_params(axis='y', which='both', direction='in')
plt.show()


# In[148]:


###visualizing the points to sample the winds

t_index=np.array([0, 1, 10, 38,40, 50, 80, 115, 370, 430,680, 800])

plt.figure(figsize=(7.7, 5.2))
winds_interp=interp1d( np.log10(t[t_index]), np.log10(windsov[t_index]), kind='linear', fill_value='extrapolate')
winds_int=np.zeros(len(t))
for i in range(len(t)):
    if t[i]<=1e9:
        winds_int[i]=10**winds_interp(np.log10(t[i]))

k=0

for i in t_index:
    if k==0 :
        
        plt.scatter(t[t_index],windsov[t_index], label="Sampled points", color="firebrick")
        k=+1
    else:
        plt.scatter(t[t_index],windsov[t_index], color="firebrick")
       
plt.plot(t, windsov, label='Winds rate ov0.5')    
plt.plot(t, winds_int, label='Intrpolated winds ov0.5')

plt.xscale('log')
plt.yscale('log')
plt.xlim(10**4, 10**10)
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
plt.savefig(f'../Images/sevn/int_winds.pdf', format='pdf')
plt.show()


# In[149]:


#cumulative winds ejection from SEVN


comulative_snII_ioroov=np.zeros(len(t))

comulative_winds_ioroov=np.zeros(len(t))
windsov=np.nan_to_num(windsov, nan=0.0)

comulative_AGB_ioroov=np.zeros(len(t))

comulative_winds_ioroov=np.zeros(len(t))
comulative_winds_ioroov_intp=np.zeros(len(t))
windsov=np.nan_to_num(windsov, nan=0.0)
comulative_AGB_ioro=np.zeros(len(t))


for i in range(len(t)-1):
        
        comulative_snII_ioroov[i+1]=(ejection_SNII_ioroov[i+1]+ejection_SNII_ioroov[i])*(t[i+1]-t[i])/2+comulative_snII_ioroov[i]
        
for i in range(len(t)-1):
        
        comulative_AGB_ioroov[i+1]=(ejection_AGB_ioroov[i+1]+ejection_AGB_ioroov[i])*(t[i+1]-t[i])/2+comulative_AGB_ioroov[i]
        
        
for i in range(len(t)-1):        
    
    comulative_winds_ioroov[i+1]=(windsov[i+1]+windsov[i])*(t[i+1]-t[i])/2+comulative_winds_ioroov[i]
    comulative_winds_ioroov_intp[i+1]=(winds_int[i+1]+winds_int[i])*(t[i+1]-t[i])/2+comulative_winds_ioroov_intp[i]


# In[150]:


#total ejection cumulative of masses

plt.plot(t, comulative_snII_ioroov)
plt.plot(t, comulative_AGB_ioroov)

plt.plot(t, comulative_AGB_ioroov+comulative_snII_ioroov+comulative_winds_ioroov)


plt.xlim(1e6, 1e10)
plt.xscale('log')
plt.xlabel('t $[yr]$', fontsize=14)
plt.ylabel('$[M/M_{\odot}]$', fontsize=14)
plt.title(r'\textbf{Total mass ejected by a SnII in a stellar population of a solar mass', fontsize=19)
ax = plt.gca()  # Get the current axes
ax.yaxis.set_ticks_position('both')  # Show ticks on both sides
ax.tick_params(axis='y', which='both', direction='in')
plt.show()


# In[152]:


#comparison between the sampled winds and non-sampled

plt.figure(figsize=(7.7, 5.2))
plt.plot(t, comulative_winds_ioroov, label="Numerical function")
plt.plot(t, comulative_winds_ioroov_intp, label="Interpolated function")

plt.xscale('log')

plt.xlim(10**4, 10**10)
plt.yticks(fontsize=16)
plt.tick_params(axis='x', which='both', direction='in', pad=6)  # Adjust pad to move x tick labels lower
plt.tick_params(axis='y', which='both', direction='in')


plt.xlabel('SSP age $[yr]$', fontsize=18, labelpad=5)
plt.ylabel(r'Total mass return fraction', fontsize=18,labelpad=7)
plt.tick_params(axis='both', direction='in', which='both')
ax = plt.gca()  # Get the current axes
ax.yaxis.set_ticks_position('both')  # Show ticks on both sides
ax.tick_params(axis='y', which='both', direction='in')
plt.legend(fontsize=13) 
plt.savefig(f'../Images/sevn/int_winds_tot.pdf', format='pdf')
plt.show()


# In[154]:



# Step 1: Multiply by 1000
z_shifted = int(z * 10000)
# Step 2: Convert to string and zero-fill to 4 digits
z_str = str(z_shifted).zfill(4)




file="../CONFRONTOFINALE/IORIO_Z_"+str(z_str)+"_"+str(sn_type[0])+"_"+str(track[0])+".csv"

df = pd.DataFrame({"t":t, 'winds':windsov, 'SNII': ejection_SNII_ioroov, 'AGB':ejection_AGB_ioroov, 'tot_SNII':comulative_snII_ioroov, 'tot_winds':comulative_winds_ioroov,  'tot_AGB':comulative_AGB_ioroov })

df.to_csv(file, index=False, header=True)



# In[163]:


#CALCULATUBG THE YIELDS

#We use the yields from SMUGGLE so we load them


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


for i in range(len(metallicities)):
    for y in range(len(elements)):
        for j in range(len(mass_table2)):
            yieldsSNII[f"{elements[y]}"][i][j] = YieldsSNII[f"{metallicities[i]}"][y][j]


YieldsAGB={}

file_path = 'AGB_Margio.hdf5'  
for i in metallicities2:
    with h5py.File(file_path, 'r') as file:
        stellar_masses1 = file['Masses'][:]
        YieldsAGB[f"{i}"]=file[f'Yields/{i}/Yield'][:][:]

mass_tableAGB2=np.array(stellar_masses1)

yieldsAGB={}
for y in elements:
    yieldsAGB[f"{y}"] = np.zeros((len(metallicities2), len(mass_tableAGB2)))

for i in range(len(metallicities2)):
    for y in range(len(elements)):
        for j in range(len(mass_tableAGB2)):
            yieldsAGB[f"{elements[y]}"][i][j] = YieldsAGB[f"{metallicities2[i]}"][y][j]

metalej={}
file_path = 'SNII_2.hdf5'  
for i in metallicities:
    with h5py.File(file_path, 'r') as file:
        metalej[f"{i}"]=file[f'Yields/{i}/Total_Metals'][:]

        


metal_yieldSNII = np.empty((len(metallicities), len(mass_table2)))
for i in range(len(metallicities)):
    for j in range(len(mass_table2)):
        metal_yieldSNII[i][j]=np.array(metalej[f"{metallicities[i]}"][j])*(0.45085746705712315/0.49414264142818165)
        
metalej={}
file_path = 'AGB_Margio.hdf5'  
for i in metallicities2:
    with h5py.File(file_path, 'r') as file:
        metalej[f"{i}"]=file[f'Yields/{i}/Total_Metals'][:]

metal_yieldAGB = np.empty((len(metallicities2), len(mass_tableAGB2)))
for i in range(len(metallicities2)):
    for j in range(len(mass_tableAGB2)):
        metal_yieldAGB[i][j]=np.array(metalej[f"{metallicities2[i]}"][j])*(0.45085746705712315/0.49414264142818165)


# In[164]:




yieldsAGB_ioro={}
yieldsSN_ioro={}



for y in elements:
    yieldsAGB_ioro[f"{y}"] = np.zeros((len(metallicities), len(mass_tableAGB2)))


for i in range(len(metallicities2)):
    for y in range(len(elements)):
        for j in range(len(mass_tableAGB2)):
            yieldsAGB_ioro[f"{elements[y]}"][i][j]=yieldsAGB[f"{elements[y]}"][i][j]*(0.45085746705712315/0.49414264142818165)


# In[165]:


##redefinition of yield by normalizing the yields for SN

for y in elements:
    yieldsSN_ioro[f"{y}"] = np.zeros((len(metallicities), len(mass_table2)))

for i in range(len(metallicities)):
    for y in range(len(elements)):
        for j in range(len(mass_table2)):
            yieldsSN_ioro[f"{elements[y]}"][i][j] = yieldsSNII[f"{elements[y]}"][i][j]*(0.45085746705712315/0.49414264142818165)


# In[166]:


#Same for AGB  


# In[167]:


mettAGB=np.array([ 0.004, 0.008, 0.019])


# In[168]:


yieldmetalSNII=Z_yield( z, mass_table2, np.transpose(metal_yieldSNII), mett)  
yieldmetalAGB=Z_yield( z, mass_tableAGB2, np.transpose(metal_yieldAGB), mettAGB)  


# In[169]:


yield_in_tab_snII = np.zeros((len(mett), len(mass_table)))
yield_in_tab_AGB = np.zeros((len(mettAGB), len(mass_tableAGB)))


yield_elemts_snII= np.zeros((len(mett), len(elements), len(mass_table)))
yield_elemts_AGB= np.zeros(( len(mettAGB), len(elements),len(mass_tableAGB)))


for z in range(len(mett)):
    yield_funcSN=Z_yield( mett[z], mass_table2, np.transpose(metal_yieldSNII), mett)  
    for m in range(len(mass_table)):
        yield_in_tab_snII[z][m]=yield_funcSN(np.log10(mass_table[m]))
    
    
for z in range(len(mettAGB)):    
    yield_funcAGB=Z_yield( mettAGB[z], mass_tableAGB2,np.transpose( metal_yieldAGB), mettAGB) 
    for m in range(len(mass_tableAGB)):
        yield_in_tab_AGB[z][m]=yield_funcAGB(np.log10(mass_tableAGB[m]))
 

for y in range(len(elements)):
    yield_=yieldsSN_ioro[f"{elements[y]}"]
    yield_=np.transpose(yield_)
    
    for z in range(len(mett)):
        yieldSNII_Z=Z_yield( mett[z], mass_table2, yield_, mett)
        for m in range(len(mass_table)):
            yield_elemts_snII[z][y][m]=yieldSNII_Z(np.log10(mass_table[m]))
            

            
            
for y in range(len(elements)):
    yield_=yieldsAGB_ioro[f"{elements[y]}"]
    yield_=np.transpose(yield_)
    for z in range(len(mettAGB)):
        yieldAGB_Z=Z_yield( mettAGB[z], mass_tableAGB2, yield_, mettAGB)
        for m in range(len(mass_tableAGB)):
            yield_elemts_AGB[z][y][m]=yieldAGB_Z(np.log10(mass_tableAGB[m]))
        


# In[175]:


elemnts=["Hydrogen", "Helium", "Carbon", "Nitrogen", "Oxygen", "Neon", "Magnesium",
         "Silicon", "Iron"]


with h5py.File("../SN_calculations/SNII_sevn.hdf5", "w") as f:
    f.create_dataset("Masses", data=mass_table)
    f.create_dataset("Metallicities", data=mett)
    
    # Store number of masses, metallicities, and species
    f.create_dataset("Number_of_masses", data=len(mass_table))
    f.create_dataset("Number_of_metallicities", data=len(mett))
    f.create_dataset("Number_of_species", data=len(elemnts))
    
    # Store reference information
    f.create_dataset("Reference", data="Iorio et al 2023")
    
    # Store element names as variable-length strings
    elemnts_bytes = [name.encode('utf-8') for name in elemnts]
    dt = h5py.string_dtype(encoding='ascii')  # Variable-length string datatype
    f.create_dataset("Species_name", data=elemnts_bytes, dtype=dt)
    
    elemnts_bytes = [name.encode('utf-8') for name in metallicities]
    dt = h5py.string_dtype(encoding='ascii') 
    f.create_dataset("Yields_names", data=elemnts_bytes)
    
    group1 = f.create_group("Yields")
    
    
    for z in range(len(metallicities)):
        
       
        
        
        group=group1.create_group(f"{metallicities[z]}")
        group.create_dataset("Ejected_mass", data=ejected_SNII_ioroov[z][:])
        group.create_dataset("Total_Metals", data=yield_in_tab_snII[z][:])
        
        group.create_dataset("Yield", data=yield_elemts_snII[:][z][:])


# In[176]:





with h5py.File("../SN_calculations/AGB_sevn.hdf5", "w") as f:
    f.create_dataset("Masses", data=mass_tableAGB)
    f.create_dataset("Metallicities", data=mettAGB)
    
    # Store number of masses, metallicities, and species
    f.create_dataset("Number_of_masses", data=len(mass_tableAGB))
    f.create_dataset("Number_of_metallicities", data=len(mettAGB))
    f.create_dataset("Number_of_species", data=len(elemnts))
    
    # Store reference information
    f.create_dataset("Reference", data="Iorio et al 2023")
    
    # Store element names as variable-length strings
    elemnts_bytes = [name.encode('utf-8') for name in elemnts]
    dt = h5py.string_dtype(encoding='ascii')  # Variable-length string datatype
    f.create_dataset("Species_name", data=elemnts_bytes, dtype=dt)
    
    elemnts_bytes = [name.encode('utf-8') for name in metallicities2]
    dt = h5py.string_dtype(encoding='ascii') 
    f.create_dataset("Yields_names", data=elemnts_bytes)
    
    group1 = f.create_group("Yields")
    
    
    for z in range(len(metallicities2)):
        
        
        group=group1.create_group(f"{metallicities2[z]}")
        group.create_dataset("Ejected_mass", data=ejected_AGB_ioroov[z][:])
        group.create_dataset("Total_Metals", data=yield_in_tab_AGB[z][:])
        group.create_dataset("Yield", data=yield_elemts_AGB[:][z][:])


# In[177]:


base_string = "winds_tab_z"

# Define the vector mm with the numbers
mm = ["00004", "0004", "0008", "002"]

with h5py.File("winds_sevn.hdf5", "w") as f:
    f.create_dataset("Times", data=t[t_index])
    f.create_dataset("Metallicities", data=mett)
    
    # Store number of masses, metallicities, and species
    f.create_dataset("Number_of_masses", data=len(t_index))
    f.create_dataset("Number_of_metallicities", data=len(mett))
    
    
    # Store reference information
    f.create_dataset("Reference", data="Iorio et al 2023")
    
    # Store element names as variable-length strings

    
    group1 = f.create_group("Winds")
    
    
    for z in range(len(metallicities)):
        
        result = base_string + mm[z]
        cic=globals()[result]
        group=group1.create_group(f"{metallicities[z]}")
        group.create_dataset("winds", data=cic)
        


# In[178]:


z=0.02
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
print(H+He+C+O+N+Ne+Mg+Si+Fe)

yieldSNII_Z={}
z=0.02
for y in elements:
    yield_=yieldsSN_ioro[f"{y}"]
    yield_=np.transpose(yield_)
    yieldSNII_Z[f"{y}"]=Z_yield( z, mass_table2, yield_, mett)

yieldAGB_Z={}

for y in elements:
    yield_=yieldsAGB[f"{y}"]
    yield_=np.transpose(yield_)
    yieldAGB_Z[f"{y}"]=Z_yield( z, mass_tableAGB2, yield_, mettAGB)
    
yield_=metal_yieldSNII
yield_=np.transpose(yield_)
yieldMetal_Z_SNII=Z_yield( z, mass_table2, yield_, mett)
    
yield_=metal_yieldAGB
yield_=np.transpose(yield_)
yieldMetal_Z_AGB=Z_yield( z, mass_tableAGB, yield_, mettAGB)
    
    
           
        
    
     


# In[ ]:





# In[179]:


X="Fe"

plt.plot(masses, yieldSNII_Z[str(X)](np.log10(masses)), label='Metallicity: '+str(z))
plt.scatter(mass_table2, yieldsSN_ioro[str(X)][2][:])
plt.scatter(mass_table2, yieldsSN_ioro[str(X)][3][:])
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


# In[180]:


X="O"

plt.plot(masses, yieldAGB_Z[str(X)](np.log10(masses)), label='Metallicity: '+str(z))
plt.scatter(mass_tableAGB2, yieldsAGB_ioro[str(X)][1][:])
plt.scatter(mass_tableAGB2, yieldsAGB_ioro[str(X)][2][:])
#plt.plot(lifetime, comulative_snII+comulative_AGB, label='AGB+SNII, Z=0.02')

#plt.plot(lifetime[82][:], comulative_SNII[82][:]+comulative_AGB[82][:], label='Time interpolation')
#plt.plot(lifetime[99][:], comulative_SNII[99][:]+comulative_AGB[99][:], label='Time interpolation')
#plt.plot(lifetime[0][:], comulative_SNII[0], label='Time interpolation')
#plt.plot(t, comulative_snII['Z_0.02'], label='Mass interpolation')
#plt.plot(t, comulative_snII_q, label='Quadratic interpolation')
#plt.xscale('log')
plt.xlim(0.8, 8)
plt.xlabel('M $[M_{\odot}]$', fontsize=14)
plt.ylabel('Yield for "' +str(X)+'" $[M_{\odot}]$', fontsize=14)
plt.title(r'\textbf{Yield for '+str(X)+' at Z='+str(z), fontsize=19)
plt.legend()
ax = plt.gca()  # Get the current axes
ax.yaxis.set_ticks_position('both')  # Show ticks on both sides
ax.tick_params(axis='y', which='both', direction='in')
plt.show()


# In[184]:


ej_AGB_elements={}
    #for i in range(len()):
value=0
elements=["H", "He", "C", "N", "O", "Ne","Mg", "Si", "Fe"]

for x in range(len(elements)):
    ej_AGB_elements[f"{elements[x]}"]=np.zeros(len(masses))
    for i in range(len(masses) - 1):
        if 0.85<=masses[i] <= 8 and i + 1 < len(masses):
            ej_AGB_elements[f"{elements[x]}"][i] = -((10**ej_eff_ioroov( np.log10(masses[i + 1]))*masses[i+1]*abundance[x]+yieldAGB_Z[f"{elements[x]}"](np.log10(masses[i+1])) )* IMF(masses[i + 1]) + (10**ej_eff_ioroov( np.log10(masses[i]))*masses[i]*abundance[x]+yieldAGB_Z[f"{elements[x]}"](np.log10(masses[i]))) * IMF(masses[i])) * (masses[i + 1] - masses[i]) / (2 * (lifetime1[i + 1] - lifetime1[i]))
            
        elif 5 < masses[i] < 8:
            ej_AGB_elements[f"{elements[x]}"][i] = -((10**ej_eff_ioroov(np.log10(5))*masses[i+1]*abundance[x]+yieldAGB_Z[f"{elements[x]}"](np.log10(5))) * IMF(masses[i + 1]) + (10**ej_eff_ioroov( np.log10(5))*masses[i]*abundance[x]+yieldAGB_Z[f"{elements[x]}"](np.log10(5))) * IMF(masses[i])) * (masses[i + 1] - masses[i]) / (2 * (lifetime1[i + 1] - lifetime1[i]))
            
        
        else:
            ej_AGB_elements[f"{elements[x]}"][i] = 0

ej_SNII_elements={}
    #for i in range(len()):
elements=["H", "He", "C", "N", "O", "Ne","Mg", "Si", "Fe"]
value=0
for x in range(len(elements)):
    ej_SNII_elements[f"{elements[x]}"] =np.zeros(len(masses)) 
    for i in range(len(masses) - 1):
        if 8 <= masses[i] <= 100 and i + 1 < len(masses):
             ej_SNII_elements[f"{elements[x]}"][i]= -((10**ej_eff_ioroov(np.log10(masses[i + 1]))*masses[i+1]*abundance[x]+yieldSNII_Z[f"{elements[x]}"](np.log10(masses[i+1])))* IMF(masses[i + 1]) + (10**ej_eff_ioroov(np.log10(masses[i]))*masses[i]*abundance[x]+yieldSNII_Z[f"{elements[x]}"](np.log10(masses[i])))* IMF(masses[i])) * (masses[i + 1] - masses[i]) / (2 * (lifetime1[i + 1] - lifetime1[i]))
            
            
        else:
            ej_SNII_elements[f"{elements[x]}"][i] = 0


            
ej_SNII_metals=np.zeros(len(masses))          
for i in range(len(masses) - 1):
        if 8 <= masses[i] <= 100 and i + 1 < len(masses):
             ej_SNII_metals[i]= -((10**ej_eff_ioroov(np.log10(masses[i + 1]))*masses[i+1]*z+yieldMetal_Z_SNII(np.log10(masses[i+1])))* IMF(masses[i + 1]) + (10**ej_eff_ioroov(np.log10(masses[i]))*masses[i]*z+yieldMetal_Z_SNII(np.log10(masses[i])))* IMF(masses[i])) * (masses[i + 1] - masses[i]) / (2 * (lifetime1[i + 1] - lifetime1[i]))

                
ej_AGB_metals=np.zeros(len(masses))          
for i in range(len(masses) - 1):
        if 0.85 <= masses[i] <= 5 and i + 1 < len(masses):
             ej_AGB_metals[i]= -((10**ej_eff_ioroov(np.log10(masses[i + 1]))*masses[i+1]*z+yieldMetal_Z_AGB(np.log10(masses[i+1])))* IMF(masses[i + 1]) + (10**ej_eff_ioroov(np.log10(masses[i]))*masses[i]*z+yieldMetal_Z_AGB(np.log10(masses[i])))* IMF(masses[i])) * (masses[i + 1] - masses[i]) / (2 * (lifetime1[i + 1] - lifetime1[i]))
        elif 5 < masses[i] < 8:
            ej_AGB_metals[i]= -((10**ej_eff_ioroov(np.log10(5))*masses[i+1]*z+yieldMetal_Z_AGB(np.log10(5))) * IMF(masses[i + 1]) + (10**ej_eff_ioroov( np.log10(5))*masses[i]*z+yieldMetal_Z_AGB(np.log10(5))) * IMF(masses[i])) * (masses[i + 1] - masses[i]) / (2 * (lifetime1[i + 1] - lifetime1[i]))
            


# In[186]:



plt.plot(lifetime1, ejection_AGB_ioroov[100:]-ej_AGB_elements["H"]-ej_AGB_elements["He"], label='AGB metals from single yield')

plt.plot(lifetime1, ejection_SNII_ioroov[100:]-ej_SNII_elements["H"]-ej_SNII_elements["He"], label='SN metals from single yield')
plt.plot(lifetime1,  ej_SNII_metals, label='SN metals from single yield')
plt.plot(lifetime1,  ej_AGB_metals, label='SN metals from single yield')
#plt.plot(t, rate_SNII_q, label='Quadratic interpolation')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Time [yrs]')
plt.ylabel('Mass ejection rate ')
plt.title('Supernovae rate with time interpolation')
plt.legend()
ax = plt.gca()  # Get the current axes
ax.yaxis.set_ticks_position('both')  # Show ticks on both sides
ax.tick_params(axis='y', which='both', direction='in')
#plt.xlim(10**6, 4*10**10)
#plt.ylim(10**-11, 10**-7)
plt.show()


# In[188]:


comulative_elementSN={} 

for j in elements:
    comulative_elementSN[f"{j}"] = np.zeros(len(masses))
    for i in range(len(masses) - 1):
        if 8 <= masses[i] <= 100 and not np.isnan(comulative_elementSN[f"{j}"][i]):
            comulative_elementSN[f"{j}"][i+1] = (
                (ej_SNII_elements[f'{j}'][i+1] + ej_SNII_elements[f'{j}'][i]) * (lifetime1[i+1] - lifetime1[i]) / 2 
                + comulative_elementSN[f'{j}'][i]
            )
        else:
            comulative_elementSN[f'{j}'][i] = comulative_elementSN[f'{j}'][i-1]
            comulative_elementSN[f'{j}'][i+1] = comulative_elementSN[f'{j}'][i]
    
comulative_elementAGB={} 

for j in elements:
    comulative_elementAGB[f"{j}"] = np.zeros(len(masses))
    for i in range(len(masses) - 1):
        if 0.85 <= masses[i] <=8 and not np.isnan(comulative_elementAGB[f"{j}"][i]):
            comulative_elementAGB[f"{j}"][i+1] = (ej_AGB_elements[f'{j}'][i+1] + ej_AGB_elements[f'{j}'][i]) * (lifetime1[i+1] - lifetime1[i]) / 2 + comulative_elementAGB[f'{j}'][i]
  
        else:
            comulative_elementAGB[f'{j}'][i] = comulative_elementAGB[f'{j}'][i-1]
            comulative_elementAGB[f'{j}'][i+1] = comulative_elementAGB[f'{j}'][i]

cum_metals_SNII=np.zeros(len(masses))
cum_metals_AGB=np.zeros(len(masses))

for i in range(len(masses) - 1):
    if 0.85 <= masses[i] <=8 and not np.isnan(comulative_elementAGB[f"{j}"][i]):
            cum_metals_AGB[i+1] = (ej_AGB_metals[i+1] + ej_AGB_metals[i]) * (lifetime1[i+1] - lifetime1[i]) / 2 + cum_metals_AGB[i]
    if 8 <= masses[i] <= 100 and not np.isnan(comulative_elementSN[f"{j}"][i]):
            cum_metals_SNII[i+1] = (
                (ej_SNII_metals[i+1] + ej_SNII_metals[i]) * (lifetime1[i+1] - lifetime1[i]) / 2 
                + cum_metals_SNII[i]
            )
    else:
            cum_metals_SNII[i] = cum_metals_SNII[i-1]
            cum_metals_SNII[i+1] = cum_metals_SNII[i]

cum_metals_SNII=np.append(np.zeros(100), cum_metals_SNII)
cum_metals_AGB=np.append(np.zeros(100), cum_metals_AGB)


# In[189]:


import matplotlib.cm as cm 
totshit=np.zeros(len(t))



def plot_yieldSNII_Z(yieldSNII_Z):
    totshit = np.zeros(len(t))
    # Determine the grid size (3x3)
    n_plots = len(yieldSNII_Z)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols
    i = 0
    
    # Create the figure and GridSpec object
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(21, 18))
    
    viridis = cm.get_cmap('viridis', len(elements))
    viridis2 = cm.get_cmap('viridis', len(elements))
    # Flatten the axis array if it's not already flat
    axs = axs.flat if isinstance(axs, np.ndarray) else [axs]
    
    # Plot each dataset
    for (y, func), ax in zip(yieldSNII_Z.items(), axs):
        # Get the color from Viridis colormap
        color = viridis(i)
        color2 = viridis2(len(elements) - i - 1)
        
        # Plot the interpolated data
        ax.plot(t, np.append(np.zeros(100), comulative_elementSN[str(y)] + comulative_elementAGB[str(y)]) + comulative_winds_ioroov * abundance[i], label='AGB', color=color2)
        ax.plot(t, np.append(np.zeros(100), comulative_elementSN[str(y)]) + comulative_winds_ioroov * abundance[i], label='SNII', color=color)
        ax.plot(t, comulative_winds_ioroov * abundance[i], label='Winds', color="forestgreen")
        
        # Add a legend with the element name
        ax.legend(fontsize=18)
        
        # Ensure ticks are enabled for all subplots
        ax.tick_params(axis='both', which='major', labelsize=17, direction='in')
        ax.tick_params(axis='both', which='minor', direction='in')
        
        # Add labels to x and y axes
        ax.set_xlabel('SSP age $[yr]$', fontsize=23)
        ax.set_ylabel(r'Ejection fraction of ' + y, fontsize=23)
        ax.set_xscale('log')
        ax.set_xlim(1e5, 1e10)

        totshit = np.append(np.zeros(100), comulative_elementSN[str(y)] + comulative_elementAGB[str(y)]) + comulative_winds_ioroov * abundance[i] + totshit
        i += 1

    # Adjust layout to ensure labels fit and increase distance between plots
    plt.tight_layout(pad=3)
    plt.subplots_adjust(wspace=0.27, hspace=0.3)

    # Save or show the figure
    plt.savefig(f'../Images/Mar/totYeilds_mar_all.pdf', format='pdf')
    plt.show()
    return totshit

cacca=plot_yieldSNII_Z(yieldSNII_Z)


# In[190]:


plt.plot(t, cum_metals_SNII+cum_metals_AGB)
plt.plot(t, cacca)
#plt.plot(lifetime, comulative_snII_mar)
plt.xlim(1e6, 1e10)
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


# In[191]:


cum={}
for i, element in enumerate(elements):
    cum[f"{element}"]=np.append(np.zeros(100), comulative_elementSN[element]+comulative_elementAGB[element])+comulative_winds_ioroov*abundance[i]


# In[192]:


df = pd.DataFrame(cum)
df.insert(0, 'time', t) 
# Write the DataFrame to a CSV file
df.to_csv('../CONFRONTOFINALE/tot_elemnt_ej_ior.csv', index=False)


# In[193]:


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
        
        ax.plot(t, cum[str(y)])
        ax.plot(t, np.append(np.zeros(100), comulative_elementSN[str(y)] + comulative_elementAGB[str(y)]) + comulative_winds_ioroov * abundance[i], label='AGB', color=color2)
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


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




