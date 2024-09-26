#!/usr/bin/env python
# coding: utf-8

# In[27]:



import numpy as np
import scipy.integrate as integrate
import scipy.special as special
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import curve_fit
import pandas as pd


# In[28]:


#Point of the script is to reproduce the plot for the Mass loss of O/B stars in chap 3.3 of Hopkins et al. 2022

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"]
})

save_plot="../Images/Hop"

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
save_plot="../Images/Hop"
plt.rc('axes', prop_cycle=(plt.cycler('color', CB_color_cycle)))


# In[ ]:





# In[29]:


#creating intervall 
tn=np.linspace(1, 10000, num=1000)  
dt=np.log10(tn[len(tn)-1]/tn[0])/(len(tn))
logt=np.log10(tn[0])+np.arange(len(tn))*dt 
t=10.**logt

#define metalicity range
z=np.array([0.01, 0.05, 0.1, 1])

#define the vector for the mass loss rate
#OB_loss_rate=np.array([OB_mass_loss(t_i, 1) for t_i in t]) 


# In[ ]:





# In[30]:



plt.show()


# In[ ]:





# In[31]:


#Also adding the plot fot the velocity of the winds


# In[ ]:





# In[32]:


##Adding the AGB phases parametrization


# In[33]:



#defining characteristic times
t_ii = np.array([1.7, 4.0, 20, 800])#Myr


#define amplitude coefficents 
def aa_1(z):
    return 3*np.power(z, 0.87)
def aa_2(z):
    return 20*np.power(z,0.45)
def aa_3(z):
    return 0.6*z
aa=np.array([aa_1, aa_2, aa_3, 0.11, 0.01])#M_dot/M in Gyr


a=np.array([0, 0, 0, 0.01, 0.01])
t_i=[0, 0, 0, 1000]


#define power law exponents
def ppsi_1(t ,z):
    return np.log((aa[1](z)/aa[0](z)))/np.log((t_ii[1]/t_ii[0]))
def ppsi_2(t ,z):
    return np.log((aa[2](z)/aa[1](z)))/np.log((t_ii[2]/t_ii[1]))
ppsi=np.array([ppsi_1, ppsi_2, -3.1])


#define the actual function 
def OB_AGB_mass_loss(t, z):
    conditions = [
        (t >= 0)&(t < t_ii[0]),
        (t_ii[0] <= t) & (t < t_ii[1]),
        (t_ii[1] <= t) & (t < t_ii[2]),
        (t_ii[2] <= t) 
    ]

    choices = [
        aa[0](z)+aa[3]*np.power((t_ii[3]/t), 1.6)*(np.exp(-(np.power((t_ii[3]/t),6)))+np.power((aa[4]**-1+np.power((t_ii[3]/t),2)),-1)),#+a[3] * np.power((1 + np.power(t / t_i[3], 1.1)) * (1 + a[4] * np.power(t / t_i[3], -1)), -1),

        aa[0](z) * np.power(t / t_ii[0], ppsi[0](t, z)) + aa[3] * np.power((t_ii[3] / t), 1.6) * (np.exp(-(np.power((t_ii[3] / t), 6)))+ np.power((aa[4]**-1 + np.power((t_ii[3] / t), 2)), -1)),# + a[3] * np.power((1 + np.power(t / t_i[3], 1.1)) * (1 + a[4] * np.power(t / t_i[3], -1)), -1),

        aa[1](z)*np.power(t / t_ii[1], ppsi[1](t, z))+aa[3]*np.power((t_ii[3]/t), 1.6)*(np.exp(-(np.power((t_ii[3]/t),6)))+np.power((aa[4]**-1+np.power((t_ii[3]/t),2)),-1)),#+a[3] * np.power((1 + np.power(t / t_i[3], 1.1)) * (1 + a[4] * np.power(t / t_i[3], -1)), -1),

        aa[2](z)*np.power(t / t_ii[2], ppsi[2])+aa[3]*np.power((t_ii[3]/t), 1.6)*(np.exp(-(np.power((t_ii[3]/t),6)))+np.power((aa[4]**-1+np.power((t_ii[3]/t),2)),-1)),
    ]

    return np.select(conditions, choices)



t_ii2 = np.array([1*10**6, 3.5*10**6, 100*10**6])

def OB_AGB_mass_loss2(t, Z):
    conditions = [
        (t >= 0) & (t < t_ii2[0]),
        (t_ii2[0] <= t) & (t < t_ii2[1]),
        (t_ii2[1] <= t) & (t < t_ii2[2]),
        (t_ii2[2] <= t)
    ]

    choices = [
        4.763*(0.01+Z/0.013),
        4.763 * (0.01 + Z / 0.013) * np.power(t/10**6, 1.45 + 0.8 * np.log(Z / 0.013)),
        29.4 * (t / (3.5*10**6)) ** -3.25 + 0.0042,
        0.42 * ((t / (1000*1e6)) ** -1.1) / (19.81 - np.log(t/1e6)),
    ]
    return np.select(conditions, choices)


# In[34]:


#Assigning a vector to the rate 
AGBOB_loss_rate=np.array([OB_AGB_mass_loss(t_i, 1) for t_i in t]) 
AGBOB_loss_rate2=np.array([OB_AGB_mass_loss2(t_i*1e6, 0.02) for t_i in t]) 

AGBOB_loss_rate3=np.array([OB_AGB_mass_loss(t_i, 0.0004/0.02) for t_i in t]) 
AGBOB_loss_rate4=np.array([OB_AGB_mass_loss2(t_i*1e6, 0.0004) for t_i in t]) 


# In[35]:


#plot for the both ob and AGB stars mass loss     


plt.figure(figsize=(7.7, 5.2))
plt.plot(t*1e6, OB_AGB_mass_loss(t, 1)/1e9, label='Winds Fire-3 Z=0.02', color=CB_color_cycle[2])
plt.plot(t*1e6, OB_AGB_mass_loss2(t*10**6, 0.02)/1e9, label='Winds Fire-2 Z=0.02', color=CB_color_cycle[2], linestyle='-.')
plt.plot(t*1e6, OB_AGB_mass_loss(t, 0.0004/0.02)/1e9, label='Winds Fire-3 Z=0.0004', color=CB_color_cycle[5])
plt.plot(t*1e6, OB_AGB_mass_loss2(t*10**6, 0.0004)/1e9, label='Winds Fire-2 Z=0.0004', color=CB_color_cycle[5], linestyle='-.')

plt.yscale('log')
plt.xscale('log')

plt.xticks(fontsize=17)
plt.yticks(fontsize=16)
plt.tick_params(axis='x', which='both', direction='in', pad=6)  # Adjust pad to move x tick labels lower
plt.tick_params(axis='y', which='both', direction='in')
plt.xlim(1e6, 1e10)
#plt.ylim(5*1e-5, 2*1e-2)

plt.xlabel('SSP age $[yr]$', fontsize=18, labelpad=5)
plt.ylabel(r'Continuous mass loss rate $\dot{M}_w/M_*$ [$yr^{-1}$]', fontsize=18,labelpad=7)
plt.tick_params(axis='both', direction='in', which='both')
ax = plt.gca()  # Get the current axes
ax.yaxis.set_ticks_position('both')  # Show ticks on both sides
ax.tick_params(axis='y', which='both', direction='in')
plt.legend(fontsize=13) 
plt.savefig(f'{save_plot}/Winds_rate.pdf', format='pdf')
plt.show()


# In[36]:


#defining the mass loss
#comulative_snI=np.array([SnIa_total(t_i) for t_i in t])
#comulative_snII=np.array([SnII_total(t_i) for t_i in t])
comulative_OBAGB=np.zeros(len(t))
comulative_OBAGB3=np.zeros(len(t))
comulative_OBAGB4=np.zeros(len(t))
#rate_snI=np.array([SnIa_lograte(t_i) for t_i in t])
comulative_OB_slice=np.zeros(len(t))
lograteOB=np.zeros(len(t))



for i in range(len(t)):
    lograteOB[i]=OB_AGB_mass_loss(t[i], 1)*t[i]



for i in range(len(t)):
        slice_rateOB=lograteOB[:i]
        comulative_OBAGB[i]=np.trapz(slice_rateOB, dx=dt)*np.log(10.)
        

comulative_OBAGB2=np.zeros(len(t))

for i in range(len(t)-1):
    comulative_OBAGB2[i+1]=(AGBOB_loss_rate2[i+1]+AGBOB_loss_rate2[i])*(t[i+1]-t[i])/2+comulative_OBAGB2[i]
    comulative_OBAGB3[i+1]=(AGBOB_loss_rate3[i+1]+AGBOB_loss_rate3[i])*(t[i+1]-t[i])/2+comulative_OBAGB3[i]
    comulative_OBAGB4[i+1]=(AGBOB_loss_rate4[i+1]+AGBOB_loss_rate4[i])*(t[i+1]-t[i])/2+comulative_OBAGB4[i]

    
    
df = pd.DataFrame({'t': t, 'AGB+OB': comulative_OBAGB/10**3, 'AGB+OB_rate': AGBOB_loss_rate, 'AGB+OB_z':comulative_OBAGB3/10**3 })
df.to_csv('../confrontofinale/wind_loss_AGB_OB.csv', index=False, header=True)


dp=pd.DataFrame({'t': t, 'AGB+OB': AGBOB_loss_rate/10**9})
dp.to_csv('../COnfronto/wind_rate_AGB_OB00004.csv', index=False, header=True)



# In[37]:


plt.figure(figsize=(7.7, 5.2))

plt.plot(t*1e6, comulative_OBAGB/10**3,  label='Winds Fire-3 Z=0.02', color=CB_color_cycle[2])
plt.plot(t*1e6, comulative_OBAGB2/10**3, label='Winds Fire-3 Z=0.02', color=CB_color_cycle[2], linestyle='-.')
plt.plot(t*1e6, comulative_OBAGB3/10**3, label='Winds Fire-3 Z=0.0004', color=CB_color_cycle[5])
plt.plot(t*1e6, comulative_OBAGB4/10**3, label='Winds Fire-3 Z=0.0004', color=CB_color_cycle[5], linestyle='dotted')
plt.xscale('log')
plt.xscale('log') 

plt.xticks(fontsize=17)
plt.yticks(fontsize=16)
plt.tick_params(axis='x', which='both', direction='in', pad=6)  # Adjust pad to move x tick labels lower
plt.tick_params(axis='y', which='both', direction='in')
plt.xlim(1e6, 1e10)
#plt.ylim(5*1e-5, 2*1e-2)

plt.xlabel('SSP age $[Myr]$', fontsize=18, labelpad=5)
plt.ylabel(r'Tatal mass ejected', fontsize=18,labelpad=7)
plt.tick_params(axis='both', direction='in', which='both')
ax = plt.gca()  # Get the current axes
ax.yaxis.set_ticks_position('both')  # Show ticks on both sides
ax.tick_params(axis='y', which='both', direction='in')
plt.legend(fontsize=13) 
plt.savefig(f'{save_plot}/Winds_tot.pdf', format='pdf')
plt.show()


# In[38]:


##if X is a species we have that the mass injection of that species is actually X_inje=f_x*M_dot 
f_C0=2.53e-3
f_O0=6.13e-3
f_N0=7.41e-4
f_H0=0.73
f_He0=0.2703
z_CNO=1



t_HHe = np.array([0.0028, 0.01, 2.3, 3.0, 100])*10**3
a_HHe = np.array([0.4*min((z_CNO + 0.001)**0.6 , 2), 0.08, 0.07, 0.042, 0.042])

psi_HHe0 = 3


a_CNO = np.array([0.2*min(z_CNO**2 + 10**-4 , 0.9), 0.68*min(( z_CNO + 0.001)**0.1 , 0.9), 0.4, 0.23, 0.065, 0.065])
t_CNO = np.array([0.001, 0.0028, 0.05, 1.9, 14, 100])*10**3
psi_CNO0 =3.5

psi_HC0=3
t_HC = np.array([0.005, 0.04, 10, 100])*10**3
a_HC= np.array([10**-6 , 0.001, 0.005, 0.005])




def yields_HHe(t, z_CNO):
    conditions=[
        t<=t_HHe[0],
        t_HHe[0]<t<=t_HHe[1],
        t_HHe[1]<t<=t_HHe[2],
        t_HHe[2]<t<=t_HHe[3],
        t_HHe[3]<t<=t_HHe[4],       
    ]
    choices =[
        a_HHe[0]*(t/t_HHe[0])**psi_HHe0,
        a_HHe[0]*(t/t_HHe[0])**(np.log( a_HHe[1]/a_HHe[0])/np.log(t_HHe[1] /t_HHe[0] )),
        a_HHe[1]*(t/t_HHe[1])**(np.log( a_HHe[2]/a_HHe[1])/np.log(t_HHe[2] /t_HHe[1] )),
        a_HHe[2]*(t/t_HHe[2])**(np.log( a_HHe[3]/a_HHe[2])/np.log(t_HHe[3] /t_HHe[2] )),
        a_HHe[3]*(t/t_HHe[3])**(np.log( a_HHe[4]/a_HHe[3])/np.log(t_HHe[4] /t_HHe[3] )),
        
    ]
    return np.select(conditions, choices)
    
def yields_CNO(t, z_CNO):
    conditions=[
        t<=t_CNO[0],
        t_CNO[0]<t<=t_CNO[1],
        t_CNO[1]<t<=t_CNO[2],
        t_CNO[2]<t<=t_CNO[3],
        t_CNO[3]<t<=t_CNO[4],       
    ]
    choices =[
        a_CNO[0]*(t/t_CNO[0])**psi_CNO0,
        a_CNO[0]*(t/t_CNO[0])**(np.log( a_CNO[1]/a_CNO[0])/np.log(t_CNO[1] /t_CNO[0] )),
        a_CNO[1]*(t/t_CNO[1])**(np.log( a_CNO[2]/a_CNO[1])/np.log(t_CNO[2] /t_CNO[1] )),
        a_CNO[2]*(t/t_CNO[2])**(np.log( a_CNO[3]/a_CNO[2])/np.log(t_CNO[3] /t_CNO[2] )),
        a_CNO[3]*(t/t_CNO[3])**(np.log( a_CNO[4]/a_CNO[3])/np.log(t_CNO[4] /t_CNO[3] )),

    ]
    return np.select(conditions, choices)

    
def yields_HC(t, z_CNO):
    conditions=[
        t<=t_HC[0],
        t_HC[0]<t<=t_HC[1],
        t_HC[1]<t<=t_HC[2],
        t_HC[2]<t<=t_HC[3],
            
    ]
    choices =[
        a_HC[0]*(t/t_HC[0])**psi_HC0,
        a_HC[0]*(t/t_HC[0])**(np.log( a_HC[1]/a_HC[0])/np.log(t_HC[1] /t_HC[0] )),
        a_HC[1]*(t/t_HC[1])**(np.log( a_HC[2]/a_HC[1])/np.log(t_HC[2] /t_HC[1] )),
        a_HC[2]*(t/t_HC[2])**(np.log( a_HC[3]/a_HC[2])/np.log(t_HC[3] /t_HC[2] )),
        
        
    ]
    return np.select(conditions, choices)

y_HHe=np.array([yields_HHe(t_i, z_CNO) for t_i in t]) 
y_CNO=np.array([yields_CNO(t_i, z_CNO) for t_i in t]) 
y_HC=np.array([yields_HC(t_i, z_CNO) for t_i in t]) 
    
    





# In[39]:


x_OC=f_O0/f_C0
y_HeC=y_HC
y_CN = np.minimum(1, 0.5 * y_CNO * (1 + x_OC))
y_ON=y_CNO+(y_CNO-y_CN)*x_OC**-1

#y_CN ok, y_CNO ok, 



#fraction of helium
f_He = f_He0 * (1 - y_HeC) + y_HHe * f_H0

#franction of N
f_N = f_N0 + y_CN * f_C0 + y_ON * f_O0

#fraction of C
f_C = f_C0 * (1 - y_CN) + y_HeC * f_He0 + y_HC * f_H0 * (1 - y_HHe)

#oxygen
f_O = f_O0 * (1 - y_ON)


# In[40]:


plt.figure(figsize=(7.7, 5.2))
plt.plot(t, f_He, label=r'f$_{He}$', color='coral')
plt.plot(t, f_N, label=r'f$_{N}$', color='royalblue')
plt.plot(t, f_C, label=r'f$_{C}$', color='dimgray')
plt.plot(t, f_O, label=r'f$_{O}$', color='mediumseagreen')

# Set logarithmic scales for both axes
plt.yscale('log')
plt.xscale('log')

# Adjust the appearance of ticks and labels
plt.xticks(fontsize=17)
plt.yticks(fontsize=16)
plt.tick_params(axis='x', which='both', direction='in', pad=6)
plt.tick_params(axis='y', which='both', direction='in')
plt.ylim(1e-4, 1)

# Set the axis labels with appropriate padding
plt.xlabel('SSP age $[Myr]$', fontsize=18, labelpad=5)
plt.ylabel('Fractional yields  ', fontsize=18, labelpad=7)
plt.tick_params(axis='both', direction='in', which='both')

# Show ticks on both sides of the y-axis
ax = plt.gca()  # Get the current axes
ax.yaxis.set_ticks_position('both')
ax.tick_params(axis='y', which='both', direction='in')

# Add legend with appropriate font size
plt.legend(fontsize=13)

# Save the plot as a PDF file
save_plot = '.'  # Change this to your desired save directory
plt.savefig(f'../Images/Hop/ywinds_rate.pdf', format='pdf')

# Display the plot
plt.show()


# In[41]:


dp=pd.DataFrame({'t': t, 'He': f_He, 'O': f_O, 'C': f_C, 'N': f_N })
dp.to_csv('../COnfronto/ejwind_yield_fire3.csv', index=False, header=True)


# In[42]:


#define the ejection rate of each traced element

He_ej=f_He*AGBOB_loss_rate/10**9
C_ej=f_C*AGBOB_loss_rate/10**9
N_ej=f_N*AGBOB_loss_rate/10**9
O_ej=f_O*AGBOB_loss_rate/10**9
H_ej=AGBOB_loss_rate/10**9-He_ej-C_ej-N_ej-O_ej


# In[43]:


#plot each element ejection
plt.plot(t, He_ej, label='He')
plt.plot(t, C_ej, label='C')
plt.plot(t, N_ej, label='N')
plt.plot(t, O_ej, label='O')
plt.plot(t, H_ej, label='H')
plt.plot(t, AGBOB_loss_rate/10**9)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('t $[Myr]$', fontsize=14)
plt.ylabel('$[M_{\odot}yr^{-1}]$', fontsize=14)
plt.title(r'\textbf{OB mass loss rate}', fontsize=19)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend()
plt.show()


# In[63]:


#Logaritmic integral of the rate to get the total element ejection

lograteHe=np.zeros(len(t))      
lograteC=np.zeros(len(t))       
lograteN=np.zeros(len(t))
lograteO=np.zeros(len(t))
lograteH=np.zeros(len(t))


comulative_H=np.zeros(len(t))  
comulative_He=np.zeros(len(t))  
comulative_C=np.zeros(len(t))  
comulative_N=np.zeros(len(t))  
comulative_O=np.zeros(len(t))  

for i in range(len(t)):
    lograteHe[i]=He_ej[i]*t[i]
    lograteH[i]=H_ej[i]*t[i]
    lograteC[i]=C_ej[i]*t[i]
    lograteO[i]=O_ej[i]*t[i]
    lograteN[i]=N_ej[i]*t[i]



for i in range(len(t)):
        slice_He=lograteHe[:i]
        slice_H=lograteH[:i]
        slice_C=lograteC[:i]
        slice_N=lograteN[:i]
        slice_O=lograteO[:i]
    
        comulative_He[i]=np.trapz(slice_He, dx=dt)*np.log(10.)
        comulative_H[i]=np.trapz(slice_H, dx=dt)*np.log(10.)
        comulative_C[i]=np.trapz(slice_C, dx=dt)*np.log(10.)
        comulative_N[i]=np.trapz(slice_N, dx=dt)*np.log(10.)
        comulative_O[i]=np.trapz(slice_O, dx=dt)*np.log(10.)
        
        
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

elements=["Ne", "Mg", "Si", "Fe"]
#define GFM_SOLAR_METALLICITY 0.0127
abundance = np.array([ Ne, Mg, Si, Fe])
#GFM_INITIAL_ABUNDANCE_HYDROGEN + zsolar * (GFM_SOLAR_ABUNDANCE_HYDROGEN
#- GFM_INITIAL_ABUNDANCE_HYDROGEN);


cum_elements={}
for i, ii in enumerate(elements):
        c=abundance[i]
        print(c)
        cum_elements["He"]=comulative_He*10**6
        cum_elements["H"]=comulative_H*10**6
        cum_elements["C"]=comulative_C*10**6
        cum_elements["N"]=comulative_N*10**6
        cum_elements["O"]=comulative_O*10**6
        cum_elements[f"{ii}"]=comulative_OBAGB/10**3*c
        


# In[70]:


plt.plot(t, comulative_OBAGB/10**3)
plt.plot(t,  cum_elements["He"])



plt.xscale('log')
#plt.yscale('log')
plt.xlabel('Time (log scale)', fontsize=14)
plt.ylabel('$[M_{\odot}]$', fontsize=14)
plt.title(r'\textbf{OB total mass loss per unit solar mass}', fontsize=19)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

plt.show()


# In[71]:


dp=pd.DataFrame({'t': t, 'He': comulative_He*10**6, 'O': comulative_O*10**6, 'C': comulative_C*10**6, 'N': comulative_N*10**6 })
dp.to_csv('../CONFRONTOFINALE/totejwind_yield_fire3.csv', index=False, header=True)


# In[73]:


df = pd.DataFrame(cum_elements)

# Write the DataFrame to a CSV file
df.to_csv('../CONFRONTOFINALE/tot_elemnt_ej_WHop.csv', index=False)


# In[60]:





# In[ ]:




