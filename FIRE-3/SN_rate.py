#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import scipy.integrate as integrate
import scipy.special as special
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import curve_fit
from scipy.integrate import quad
from scipy.integrate import trapz, simps
from numpy import interp
import csv
import pandas as pd

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
save_plot="../Images/Hop"
plt.rc('axes', prop_cycle=(plt.cycler('color', CB_color_cycle)))


# In[3]:


#Point of this script is to reproduce the plot of the Sn rate of 3.3 of Hopkins et al. 2022


# In[4]:


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


# In[5]:


###DEFINING THE SNIa RATE###

#defining characteristic times
t_I= 44.0 #Mys

#defining amplitude factors
a_I=0.0083 #GyrM_sol

#defining power law exponts
psi_I=-1.1

#defining the function for SnII rate
def SnIa_rate(t):
    conditions = [
        (0 <=t)  & (t< (t_I)),
        (t_I) <= (t)
    ]

    choices = [
        0,
        a_I*np.power(t/t_I, psi_I)/10**9,
    ]

    return np.select(conditions, choices)



##SNIa RATE FIRE2##
t_I2 = 37.53  # Myrs

# defining amplitude factors

# defining power law exponents
psi_I2 = -1.1

# defining the function for SnIa rate
def SnIa_rate2(t):
    conditions = [
        (0 <= t) & (t < t_I2),
        (t_I2 <= t)
    ]

    choices = [
        0,
        (5.3 * 10**-8. + 1.6 * 10**-5. * np.exp(-((t - 50.) / 10.)**2. / 2))/10**6,
    ]
    return np.select(conditions, choices)



# In[6]:


#defining the total sn rate


def total_Sn_rate(t):
    return SnII_rate(t) + SnIa_rate(t)



#defining timesteps
tn=np.linspace(1, 10000, num=1000)

#Tranforming the timesteps in log-spaced timesteps
dt=np.log10(tn[len(tn)-1]/tn[0])/(len(tn))
logt=np.log10(tn[0])+np.arange(len(tn))*dt 
t=10.**logt



#assignig a vector to the value of the rates functions
rate=np.array([total_Sn_rate(t_i) for t_i in t]) 
rate_snI=np.array([SnIa_rate(t_i) for t_i in t])
rate_snII=np.array([SnII_rate(t_i) for t_i in t])
rate_snII2=np.array([SnII_rate2(t_i) for t_i in t])
rate_snI2=np.array([SnIa_rate2(t_i) for t_i in t])

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"]
})




#plotting the rate
#plt.style.use('tableau-colorblind10')
plt.figure(figsize=(7.7, 5.2))
plt.plot(t*1e6, rate_snII, label='SNII Fire-3', color=CB_color_cycle[0])
plt.plot(t*1e6, rate_snII2, label='SNII Fire-2', color=CB_color_cycle[0], linestyle='dotted')
plt.plot(t*1e6, rate_snI, label='SNIa Fire-3', color=CB_color_cycle[1])
plt.plot(t*1e6, rate_snI2, label='SNIa Fire-2', color=CB_color_cycle[1], linestyle='dotted')


plt.xscale('log')
plt.yscale('log')

#plt.title(r'\textbf{Total SNe rate}', fontsize=19)
plt.xticks(fontsize=17)
plt.yticks(fontsize=16)
plt.tick_params(axis='x', which='both', direction='in', pad=6)  # Adjust pad to move x tick labels lower
plt.tick_params(axis='y', which='both', direction='in')

plt.xlabel('SSP age $[yr]$', fontsize=18, labelpad=5)
plt.ylabel(r'SNe rate $R/M_*$ [ $yr$ $M_{\odot} ]^-1$', fontsize=18,labelpad=7)
plt.tick_params(axis='both', direction='in', which='both')
ax = plt.gca()  # Get the current axes
ax.yaxis.set_ticks_position('both')  # Show ticks on both sides
ax.tick_params(axis='y', which='both', direction='in')
plt.legend(fontsize=13) 
plt.savefig(f'{save_plot}/SN_rate.pdf', format='pdf')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[7]:


#i calculate analytically the integral of the rates

def SnII_total(t):
    conditions = [
        (t >= 0) & (t < t_II[0]),
        (t_II[0] <= t) & (t < t_II[1]),
        (t_II[1] <= t) & (t < t_II[2]),
        (t_II[2]<=t)
    ]

    choices = [
        0,
        (a_II[0]/(psi_II[0]+1)) * np.power((t/t_II[0]), psi_II[0]+1),
        (a_II[1]/(psi_II[1]+1)) * np.power(t / t_II[1], psi_II[1]+1)+(a_II[0]/(psi_II[0]+1)) * np.power((t_II[1]/t_II[0]), psi_II[0]+1),
        (a_II[1]/(psi_II[1]+1)) * np.power(t_II[2] / t_II[1], psi_II[1]+1)+(a_II[0]/(psi_II[0]+1)) * np.power((t_II[1]/t_II[0]), psi_II[0]+1)
    ]

    return np.select(conditions, choices)

def SnIa_total(t):
    conditions = [
        (t >= 0) & (t < (t_I)),
        ((t_I) <= t)
    ]

    choices = [
        0,
        (a_I/(psi_I+1))*np.power(t/t_I, psi_I+1)
    ]

    return np.select(conditions, choices)


# In[8]:


len(t)


# In[9]:


###we want to plot the comulative Sn number###


#cacultaing the integral function with the timestep in Gyrs
comulative_rate=np.zeros(len(t))
comulative_snI=np.zeros(len(t))
comulative_snII=np.zeros(len(t))
comulative_snI_slice=np.zeros(len(t))
lograteSnI=np.zeros(len(t))
lograteSnII=np.zeros(len(t))
comulative_snI_s=np.zeros(len(t))
comulative_snII_s=np.zeros(len(t))


for i in range(len(t)):
    lograteSnI[i]=SnIa_rate(t[i])*t[i]
    lograteSnII[i]=SnII_rate(t[i])*t[i]



for i in range(len(t)):
        slice_ratesnI=lograteSnI[:i]
        slice_ratesnII=lograteSnII[:i]
        comulative_snI[i]=np.trapz(slice_ratesnI, dx=dt)*np.log(10.)
        comulative_snII[i]=np.trapz(slice_ratesnII, dx=dt)*np.log(10.)
        
    
#cacultaing the integral function with the timestep in Gyrs
comulative_rate2=np.zeros(len(t))

comulative_snI2=np.zeros(len(t))
comulative_snII2=np.zeros(len(t))
comulative_snI_slice2=np.zeros(len(t))
lograteSnI2=np.zeros(len(t))
lograteSnII2=np.zeros(len(t))
comulative_snI_s2=np.zeros(len(t))
comulative_snII_s2=np.zeros(len(t))


for i in range(len(t)):
    lograteSnI2[i]=SnIa_rate2(t[i])*t[i]
    lograteSnII2[i]=SnII_rate2(t[i])*t[i]



for i in range(len(t)):
        slice_ratesnI2=lograteSnI2[:i]
        slice_ratesnII2=lograteSnII2[:i]
        comulative_snI2[i]=np.trapz(slice_ratesnI2, dx=dt)*np.log(10.)
        comulative_snII2[i]=np.trapz(slice_ratesnII2, dx=dt)*np.log(10.)


# In[10]:


#plottig the number of Sne
plt.figure(figsize=(7.7, 5.2))
plt.plot(t*1e6, comulative_snII*10**6, label='SNII Fire-3',  color=CB_color_cycle[0])
plt.plot(t*1e6, comulative_snII2*10**6, label='SNII Fire-2',  color=CB_color_cycle[0], linestyle='dotted')
plt.plot(t*1e6, comulative_snI*10**6, label='SNIa Fire-3', color=CB_color_cycle[1])
plt.plot(t*1e6, comulative_snI2*10**6, label='SNIa Fire-2', color=CB_color_cycle[1], linestyle='dotted')

plt.xscale('log')
plt.yscale('log')


plt.xticks(fontsize=17)
plt.yticks(fontsize=16)
plt.tick_params(axis='x', which='both', direction='in', pad=6)  # Adjust pad to move x tick labels lower
plt.tick_params(axis='y', which='both', direction='in')
plt.xlim(1e6, 1e10)
plt.ylim(5*1e-5, 2*1e-2)

plt.xlabel('SSP age $[yr]$', fontsize=18, labelpad=5)
plt.ylabel(r'SNe number/$M_{*}$ [$M_{\odot} ]^-1$', fontsize=18,labelpad=7)
plt.tick_params(axis='both', direction='in', which='both')
ax = plt.gca()  # Get the current axes
ax.yaxis.set_ticks_position('both')  # Show ticks on both sides
ax.tick_params(axis='y', which='both', direction='in')
plt.legend(fontsize=13) 
plt.savefig(f'{save_plot}/SN_number.pdf', format='pdf')
plt.show()


# In[ ]:





# In[11]:


#We add the parametrization for the mass of the ejecta in fuction of the time


# In[12]:


#defining the function of mass returned per single event given the time
def mass_ejecta(t):
    conditions=[
        (t<=6.5),
        (6.5<t)
    ]
    choices=[
        10*np.power(t/6.5, -2.22 ),
        10*np.power(t/6.5, -0.267)
    ]
    return np.select(conditions, choices)
ejecta=np.array([mass_ejecta(t_i) for t_i in t]) 


ejecta2=np.zeros(len(t))
for i in range(len(t)):
    if t[i]<=t_II2[2]:
        ejecta2[i]=10.4


#Plotting the graph
plt.figure(figsize=(7.7, 5.2))
plt.plot(t*1e6, ejecta, label="Fire-3", color=CB_color_cycle[0])
plt.plot(t*1e6, ejecta2, label="Fire-2", color=CB_color_cycle[0], linestyle='dotted')



plt.xscale('log')
plt.yscale('log')


plt.xticks(fontsize=17)
plt.yticks(fontsize=16)
plt.tick_params(axis='x', which='both', direction='in', pad=6)  # Adjust pad to move x tick labels lower
plt.tick_params(axis='y', which='both', direction='in')
plt.xlim(1e6, 1e10)
#plt.ylim(5*1e-5, 2*1e-2)

plt.xlabel('SSP age $[yr]$', fontsize=18, labelpad=5)
plt.ylabel(r'$M_{ej}$ $[M_{\odot} ]$', fontsize=18,labelpad=7)
plt.tick_params(axis='both', direction='in', which='both')
ax = plt.gca()  # Get the current axes
ax.yaxis.set_ticks_position('both')  # Show ticks on both sides
ax.tick_params(axis='y', which='both', direction='in')
plt.legend(fontsize=13) 
plt.savefig(f'{save_plot}/SN_mass_ej.pdf', format='pdf')
plt.show()


# In[13]:


#We plot the mass ejction rate by simply multiply the rate with the 



plt.figure(figsize=(7.7, 5.2))
plt.plot(t*1e6, ejecta*rate_snII, label="Fire-3", color=CB_color_cycle[0])
plt.plot(t*1e6, ejecta2*rate_snII2, label="Fire-2", color=CB_color_cycle[0], linestyle='dotted')
plt.xscale('log')
plt.yscale('log')


plt.xticks(fontsize=17)
plt.yticks(fontsize=16)
plt.tick_params(axis='x', which='both', direction='in', pad=6)  # Adjust pad to move x tick labels lower
plt.tick_params(axis='y', which='both', direction='in')
plt.xlim(1e6, 1e9)
#plt.ylim(5*1e-5, 2*1e-2)

plt.xlabel('SSP age $[yr]$', fontsize=18, labelpad=5)
plt.ylabel(r'Mass ejection rate/$M_{*}$ $[M_{\odot}$ $yr^{-1}$]', fontsize=18,labelpad=7)
plt.tick_params(axis='both', direction='in', which='both')
ax = plt.gca()  # Get the current axes
ax.yaxis.set_ticks_position('both')  # Show ticks on both sides
ax.tick_params(axis='y', which='both', direction='in')
plt.legend(fontsize=13) 
plt.savefig(f'{save_plot}/SN_mass_ej_rate.pdf', format='pdf')
plt.show()


# In[14]:


#we calculate the cumulative mass ejection from both SNII and SNIa using the logaritmimc integral


comulative_SNII=np.zeros(len(t))
comulative_SNI=np.zeros(len(t))


lograteSNII=np.zeros(len(t))
lograteSNI=np.zeros(len(t))


for i in range (len(t)):
    lograteSNII[i]=mass_ejecta(t[i])*SnII_rate(t[i])*t[i]


lograteSNI=1.4*rate_snI*t

for i in range(len(t)):
        slice_ratesnI=lograteSNI[:i]
        slice_ratesnII=lograteSNII[:i]
        comulative_SNI[i]=np.trapz(slice_ratesnI, dx=dt)*np.log(10.)
        comulative_SNII[i]=np.trapz(slice_ratesnII, dx=dt)*np.log(10.)

        
        

comulative_SNII2=np.zeros(len(t))
comulative_SNI2=np.zeros(len(t))


lograteSNII2=np.zeros(len(t))
lograteSNI2=np.zeros(len(t))


for i in range (len(t)):
    lograteSNII2[i]=10.5*SnII_rate2(t[i])*t[i]


lograteSNI2=1.4*rate_snI2*t

for i in range(len(t)):
        slice_ratesnI2=lograteSNI2[:i]
        slice_ratesnII2=lograteSNII2[:i]
        comulative_SNI2[i]=np.trapz(slice_ratesnI2, dx=dt)*np.log(10.)
        comulative_SNII2[i]=np.trapz(slice_ratesnII2, dx=dt)*np.log(10.)


# In[ ]:





# In[15]:


##Plotting the total mass ejection

plt.figure(figsize=(7.7, 5.2))
plt.plot(t, comulative_SNII * 10**6, label="SnII Fire3", color=CB_color_cycle[0])
plt.plot(t, comulative_snI * 10**6 * 1.4 , label="SnIa Fire3", color=CB_color_cycle[1])
plt.plot(t, comulative_SNII2 * 10**6 , label="SnII Fire2", color=CB_color_cycle[0], linestyle='dotted')
plt.plot(t, comulative_SNI2 * 10**6 * 1.4 , label="SnIa Fire2", color=CB_color_cycle[1], linestyle='dotted')
plt.xscale('log')



plt.xticks(fontsize=17)
plt.yticks(fontsize=16)
plt.tick_params(axis='x', which='both', direction='in', pad=6)  # Adjust pad to move x tick labels lower
plt.tick_params(axis='y', which='both', direction='in')
plt.xlim(1e0, 1e4)


plt.xlabel('SSP age $[Myr]$', fontsize=18, labelpad=5)
plt.ylabel(r'Total Ejected Mass/$M_{*}$', fontsize=18,labelpad=7)
plt.tick_params(axis='both', direction='in', which='both')
ax = plt.gca()  # Get the current axes
ax.yaxis.set_ticks_position('both')  # Show ticks on both sides
ax.tick_params(axis='y', which='both', direction='in')
plt.legend(fontsize=13) 
plt.savefig(f'{save_plot}/SN_mass_tot.pdf', format='pdf')
plt.show()


# In[16]:


df = pd.DataFrame({'t': t, 'SNII_tot': comulative_SNII*10**6, 'SNII':ejecta*rate_snII })

df.to_csv('../CONFRONTOFINALE/SNII_HOP.csv', index=False, header=True)


dp=pd.DataFrame({'t': t, 'SNI':rate_snI*1.4,'SNII': ejecta*rate_snII })

dp.to_csv('../COnfronto/sn_rate_ej.csv', index=False, header=True)


# In[17]:


#######YIELDS FROM SNe ############


# In[18]:


#Yield fraction for solar metallicity






# In[19]:


##We reproduce the yields and the mass ejected for single element for SNIa

elements=['He', 'C', 'N', 'O', 'Ne', 'Mg', 'Si', 'S', 'Ca', 'Fe']
yields=np.array([ 0, 1.76e-2, 2.10e-6, 7.36e-2, 2.02e-3, 6.21e-3, 0.146, 7.62e-2, 1.29e-2, 0.558]) 




def SNIa_yield(t, X, elements, yields, t_I):
    conditions = [
        (0 <= t) & (t < t_I),
        t_I <= t
    ]

    if X in elements:
        i = elements.index(X)
        choices = np.select(conditions, [0, yields[i]])
        print(yields[i])
    else:
        choices = np.zeros_like(t)

    return choices
            
        


# In[ ]:





# In[20]:


plt.figure(figsize=(7.7, 5.2))
plt.plot(t,  SNIa_yield(t, 'Fe' , elements, yields, t_I), label='Fe')
plt.plot(t,  SNIa_yield(t, 'O' , elements, yields, t_I), label='=')
plt.plot(t,  SNIa_yield(t, 'Mg' , elements, yields, t_I), label='Mg')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('t')
plt.ylabel('Ejected mass')
plt.title('Mass ejection over time')
ax = plt.gca()  # Get the current axes
ax.yaxis.set_ticks_position('both')  # Show ticks on both sides
ax.tick_params(axis='y', which='both', direction='in')
plt.legend()
plt.show()


# In[21]:


###We loaad the coefficent used to model the fra ctional yields in SNII

column_types = {'a1': float, 'a2': float, 'a3': float, 'a4': float, 'a5': float}
sn_yields_par = pd.read_csv('yieldscoeff.csv', dtype=column_types )


# In[22]:


sn_yields_par = pd.read_csv('yieldscoeff.csv')

# Extract individual columns
columns = ['a1', 'a2', 'a3', 'a4', 'a5']  # Replace with actual column names
vectors = [np.array(sn_yields_par[col]) for col in columns]

# Create a matrix with a1, a2, a3, a4, a5 as columns
a = np.column_stack(vectors)
t_y=np.array([3.7, 8, 18, 30, 44])


# In[ ]:





# In[23]:


##We reproduce the yields and the mass ejected for single element for SNII


def SNII_yield(t, t_y, a, X, elements):
    if X in elements:
        i = elements.index(X)  # No need to subtract 1
        conditions = [
            (t >= 0) & (t < t_y[0]),
            (t_y[0] <= t) & (t < t_y[1]),
            (t_y[1] <= t) & (t < t_y[2]),
            (t_y[2] <= t) & (t < t_y[3]),
            (t_y[3] <= t) & (t < t_y[4]),
            (t_y[4] <= t)
        ]
        choices = [
            0,
            a[i][0] * np.power((t / t_y[0]), np.log(a[i][1] / a[i][0]) / np.log(t_y[1] / t_y[0])),
            a[i][1] * np.power((t / t_y[1]), np.log(a[i][2] / a[i][1]) / np.log(t_y[2] / t_y[1])),
            a[i][2] * np.power((t / t_y[2]), np.log(a[i][3] / a[i][2]) / np.log(t_y[3] / t_y[2])),
            a[i][3] * np.power((t / t_y[3]), np.log(a[i][4] / a[i][3]) / np.log(t_y[4] / t_y[3])),
            0
        ]

        return np.select(conditions, choices)
    else:
        # Handle the case where X is not in elements
        return 0  # You may want to return some default value or handle this case differently


# In[ ]:


###plotting the fractional yields as a function of time


# In[24]:




from matplotlib.lines import Line2D
plt.figure(figsize=(7.7, 5.2))
plt.plot(t*1e6,  SNII_yield(t, t_y, a, 'Fe' , elements), label='Fe', color='silver')
plt.plot(t*1e6,  SNII_yield(t, t_y, a, 'O' , elements), label='O', color='royalblue')
plt.plot(t*1e6,  SNII_yield(t, t_y, a, 'Mg' , elements), label='Mg', color='salmon' )
plt.plot(t*1e6,  SNIa_yield(t, 'Fe' , elements, yields, t_I), color='silver', linestyle='-.')
plt.plot(t*1e6, SNIa_yield(t, 'O' , elements, yields, t_I), color='royalblue', linestyle='-.')
plt.plot(t*1e6,  SNIa_yield(t, 'Mg' , elements, yields, t_I), color='salmon', linestyle='-.')
plt.xscale('log')
plt.yscale('log')


plt.xticks(fontsize=17)
plt.yticks(fontsize=16)
plt.tick_params(axis='x', which='both', direction='in', pad=6)  # Adjust pad to move x tick labels lower
plt.tick_params(axis='y', which='both', direction='in')
plt.xlim(1e6, 1e10)
#plt.ylim(5*1e-5, 2*1e-2)

plt.xlabel('SSP age $[yr]$', fontsize=18, labelpad=5)
plt.ylabel(r'Yield fraction', fontsize=18,labelpad=7)
plt.tick_params(axis='both', direction='in', which='both')
ax = plt.gca()  # Get the current axes
ax.yaxis.set_ticks_position('both')  # Show ticks on both sides
ax.tick_params(axis='y', which='both', direction='in')

legend_elements = [
    Line2D([0], [0], color='black', linestyle='-', label='SNII'),
    Line2D([0], [0], color='black', linestyle='-.', label='SNIa')
]

# Adding the legends to the plot
first_legend = ax.legend(handles=legend_elements, loc='upper left', fontsize=13)
ax.add_artist(first_legend)
ax.legend(fontsize=13, loc='lower right')
plt.savefig(f'{save_plot}/yields_rate.pdf', format='pdf')
plt.show()


# In[25]:


#assign a dictionary to save in a file

elements=[ 'He', 'C', 'N', 'O', 'Ne', 'Mg', 'Si', 'S', 'Ca', 'Fe']
ej_SNII_elements={}
for i in elements:
    ej_SNII_elements[f"{i}"]=np.zeros(len(t))
    ej_SNII_elements[f"{i}"]=SNII_yield(t, t_y, a, i , elements)


# In[ ]:


#save in file


# In[26]:


with open("../COnfronto/elemnt_ej_fire3.csv", mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)

    # Write header (element names)
    writer.writerow(elements)

    # Write data columns
    for i in range(len(ej_SNII_elements["He"])):
        writer.writerow([str(ej_SNII_elements[element][i]) for element in elements])


# In[27]:


#cumulative ejection for each element

cum_elements_rate={}
for i in elements:
    cum_elements_rate[f"{i}"]=np.zeros(len(t))
    cum_elements_rate[f"{i}"]=ejecta*rate_snII*ej_SNII_elements[f"{i}"]

cum_elements={}
for i in elements:
    cum_elements[f"{i}"]=np.zeros(len(t))
    for tt in range(len(t)-1):
        cum_elements[f"{i}"][tt+1]= 1e6*(cum_elements_rate[f"{i}"][tt+1]+ cum_elements_rate[f"{i}"][tt])*(t[tt+1]-t[tt])/2+cum_elements[f"{i}"][tt]
    


# In[28]:


#plot of cumulative of each element

plt.figure(figsize=(7.7, 5.2))
plt.plot(t*1e6,  cum_elements["He"], label='He', color='silver')
plt.xscale('log')
#plt.yscale('log')


plt.xticks(fontsize=17)
plt.yticks(fontsize=16)
plt.tick_params(axis='x', which='both', direction='in', pad=6)  # Adjust pad to move x tick labels lower
plt.tick_params(axis='y', which='both', direction='in')
plt.xlim(1e6, 1e10)
#plt.ylim(5*1e-5, 2*1e-2)

plt.xlabel('SSP age $[yr]$', fontsize=18, labelpad=5)
plt.ylabel(r'Yield fraction', fontsize=18,labelpad=7)
plt.tick_params(axis='both', direction='in', which='both')
ax = plt.gca()  # Get the current axes
ax.yaxis.set_ticks_position('both')  # Show ticks on both sides
ax.tick_params(axis='y', which='both', direction='in')

legend_elements = [
    Line2D([0], [0], color='black', linestyle='-', label='SNII'),
    Line2D([0], [0], color='black', linestyle='-.', label='SNIa')
]

# Adding the legends to the plot
first_legend = ax.legend(handles=legend_elements, loc='upper left', fontsize=13)
ax.add_artist(first_legend)
ax.legend(fontsize=13, loc='lower right')
plt.savefig(f'{save_plot}/yields_rate.pdf', format='pdf')
plt.show()


# In[29]:


for i in elements:
    cum_elements[f"{i}"]=np.append(np.zeros(100), cum_elements[f"{i}"])

df = pd.DataFrame(cum_elements)


# Write the DataFrame to a CSV file
df.to_csv('../CONFRONTOFINALE/tot_elemnt_ej_SNHop.csv', index=False)


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




