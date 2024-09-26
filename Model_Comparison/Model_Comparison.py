#!/usr/bin/env python
# coding: utf-8

# In[57]:


import warnings

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
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.serif": ["Computer Modern Roman"]
})

seconds=3.156*1e+7
grams=1.98892*1e33


# In[58]:





SN_fire2 = pd.read_csv('sn_rate_ejfire2.csv')
OBAGB_fire2= pd.read_csv('wind_loss_AGB_OBfire2_onlyOB.csv')
OB_fire2= pd.read_csv('wind_loss_AGB_OBfire2_onlyOB.csv')

t_fire2=np.array(SN_fire2["t"])
SnII_fire2=np.array(SN_fire2["SNII"])
SnI_fire2=np.array(SN_fire2["SNI"])

t_fire2OB=np.array(OB_fire2['t'])
OBAGBW_fire2=np.array(OBAGB_fire2['AGB+OB_rate'])
OBW_fire2=np.array(OB_fire2['AGB+OB_rate'])
SnII_fire2=np.append(np.zeros(100), SnII_fire2)



#FIRE 2 COMULATIVE
tot_SN_fire2 = pd.read_csv('sn_numberfire2.csv')
OBAGB_fire2= pd.read_csv('wind_loss_AGB_OBfire2_onlyOB.csv')
OB_fire2= pd.read_csv('wind_loss_AGB_OBfire2_onlyOB.csv')

t_fire2=np.array(tot_SN_fire2["t"])
tot_SnII_fire2=np.array(tot_SN_fire2["SNII"])
tot_SnI_fire2=np.array(tot_SN_fire2["SNIa"])

t_fire2OB=np.array(OB_fire2['t'])
tot_OBAGBW_fire2=np.array(OBAGB_fire2['AGB+OB'])
tot_OBW_fire2=np.array(OB_fire2['AGB+OB_rate'])


# In[59]:


#FIRE 3 shit
SN_fire3 = pd.read_csv('sn_rate_ej.csv')
OBAGB_fire3= pd.read_csv('wind_rate_AGB_OB1.csv')
OBAGB_fire3_z= pd.read_csv('wind_rate_AGB_OB00004.csv')


t_fire3=np.array(SN_fire3["t"])*10**6
SNII_fire3=np.array(SN_fire3["SNII"])
SNI_fire3=np.array(SN_fire3["SNI"])

t_fire3w=np.array(OBAGB_fire3["t"])*10**6
OBAGBW_fire3=np.array(OBAGB_fire3["AGB+OB"])
OBAGBW_fire3_z=np.array(OBAGB_fire3_z["AGB+OB"])

SNII_fire3=np.append(np.zeros(100),SNII_fire3)
OBAGBW_fire3=np.append( np.ones(100)*OBAGBW_fire3[0],OBAGBW_fire3 )
OBAGBW_fire3_z=np.append( np.ones(100)*OBAGBW_fire3_z[0],OBAGBW_fire3_z )
tt = np.linspace(1e5, t_fire3[0], 100)

# Append tt to the beginning of t_fire3
t_fire3 = np.append(tt, t_fire3)



#FIRE 3 COMULATIVE
tot_SN_fire3 = pd.read_csv('sn_number.csv')
tot_OBAGB_fire3= pd.read_csv('wind_loss_AGB_OB.csv')

t_fire3w=np.array(SN_fire3["t"])*10**6
tot_SNII_fire3=np.array(tot_SN_fire3["SNII"])
tot_SNII_fire3=np.append( np.zeros(100),tot_SNII_fire3)


tot_SNI_fire3=np.array(tot_SN_fire3["SNIa"])

t_fire3w=np.array(tot_OBAGB_fire3['t'])*10**6
tot_OBAGBW_fire3=np.array(tot_OBAGB_fire3['AGB+OB'])
tot_OBAGBW_fire3=np.append( np.zeros(100),tot_OBAGBW_fire3)

tot_OBAGBW_fire3_z=np.array(tot_OBAGB_fire3['AGB+OB_z'])
tot_OBAGBW_fire3_z=np.append( np.zeros(100),tot_OBAGBW_fire3_z)

                           
len(t_fire3)


# In[60]:


mar_z002 = pd.read_csv('MAR_Z_0.02.csv')


t_z002=np.array(mar_z002 ["t"])
snII_mar_z002=np.array(mar_z002["SNII"])
agb_mar_z002=np.array(mar_z002["AGB"])
winds_mar_z002=np.array(mar_z002["Winds"])

totsnII_mar_z002=np.array(mar_z002["tot_SNII"])
totagb_mar_z002=np.array(mar_z002["tot_AGB"])
totwinds_mar_z002=np.array(mar_z002["tot_winds"])


mar_z00004 = pd.read_csv('MAR_Z_0.0004.csv')
t_z00004=np.array(mar_z00004 ["t"])
snII_mar_z00004=np.array(mar_z00004["SNII"])
agb_mar_z00004=np.array(mar_z00004["AGB"])
winds_mar_z00004=np.array(mar_z00004["Winds"])

totsnII_mar_z00004=np.array(mar_z00004["tot_SNII"])
totagb_mar_z00004=np.array(mar_z00004["tot_AGB"])
totwinds_mar_z00004=np.array(mar_z00004["tot_winds"])


# In[61]:




for track in ["ov05", "ov04", "mesa"]:
    for sn in [ "rapid", "compact", "delayed", "deathmatrix"]:
        for z in ["02", "0004"]:
            # Dynamically construct the file name
            filename = f'IORIO_Z_{z}_{sn}_{track}.csv'
            
            # Read the CSV file
            df = pd.read_csv(filename)
            
            # Create dynamic variable names
            t_var = f't_{track}_{sn}_z{z}'
            snII_var = f'snII_{track}_{sn}_z{z}'
            agb_var = f'agb_{track}_{sn}_z{z}'
            winds_var = f'winds_{track}_{sn}_z{z}'
            totsnII_var = f'totsnII_{track}_{sn}_z{z}'
            totagb_var = f'totagb_{track}_{sn}_z{z}'
            totwinds_var = f'totwinds_{track}_{sn}_z{z}'
            
            # Use the exec function to dynamically create variables
            exec(f"{t_var} = np.array(df['t'])")
            exec(f"{snII_var} = np.array(df['SNII'])")
            exec(f"{agb_var} = np.array(df['AGB'])")
            exec(f"{winds_var} = np.array(df['winds'])")
            exec(f"{totsnII_var} = np.array(df['tot_SNII'])")
            exec(f"{totagb_var} = np.array(df['tot_AGB'])")
            exec(f"{totwinds_var} = np.array(df['tot_winds'])")


# In[62]:


plt.figure(figsize=(7.7, 5.2))


color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.plot(t_z002, snII_mar_z002+agb_mar_z002+winds_mar_z002, label='Portinari et al.(1998) Z=0.02')   
plt.plot(t_fire3, SNII_fire3+OBAGBW_fire3, label='Hopkins et al.(2022) Z=0.02')   
k=0
for track in ["ov05"]:
    for sn in ["delayed"]:
        for z in ["02"]:
            t_var = eval(f't_{track}_{sn}_z{z}')
            snII_var = eval(f'snII_{track}_{sn}_z{z}')
            agb_var = eval(f'agb_{track}_{sn}_z{z}')
            winds_var = eval(f'winds_{track}_{sn}_z{z}')
            label_sn = sn.replace("_new", "")
            plt.plot(t_var, winds_var+snII_var+agb_var, label=f'SEVN ov0.5 delayed Z=0.{z}')
            
    

plt.xscale('log')
plt.yscale('log')
plt.xlim(2e5, 10**10)
#plt.ylim(1e-10, 5*1e-8)
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
plt.savefig(f'../Images/DISC/mass_rate_z02.pdf', format='pdf')
plt.show()


# In[63]:


OBAGBW_fire3[0]


# In[64]:


plt.figure(figsize=(7.7, 5.2))

linestyle=["-",".",".", "--"]
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.plot(t_z00004, snII_mar_z00004+agb_mar_z00004+winds_mar_z00004, label='Portinari et al.(1998) Z=0.0004')   ##FIRE3
plt.plot(t_fire3, SNII_fire3+OBAGBW_fire3_z, label='Hopkins et al.(2022) Z=0.0004')   ##FIRE3

for track in ["ov05"]:
    for sn in ["delayed"]:
        k=0
        for z in ["0004"]:
            t_var = eval(f't_{track}_{sn}_z{z}')
            snII_var = eval(f'snII_{track}_{sn}_z{z}')
            agb_var = eval(f'agb_{track}_{sn}_z{z}')
            winds_var = eval(f'winds_{track}_{sn}_z{z}')
            label_sn = sn.replace("_new", "")
            plt.plot(t_var, snII_var + agb_var+winds_var, label=f'SEVN ov0.5 delayed Z=0.{z}', linestyle=linestyle[k])
            k=+3
    

plt.xscale('log')
plt.yscale('log')
plt.xlim(2e5, 10**10)

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
plt.savefig(f'../Images/disc/mass_rate_met.pdf', format='pdf')
plt.show()


# In[65]:


plt.figure(figsize=(7.7, 5.2))

linestyle=["-",".",".", "--"]
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.plot(t_z002, totsnII_mar_z002+totagb_mar_z002+totwinds_mar_z002, label='Portinari et al.(1998) Z=0.02')   
plt.plot(t_fire3, tot_SNII_fire3+tot_OBAGBW_fire3, label='Hopkins et al.(2022) Z=0.02')   
for track in [ "ov05"]:
    for sn in ["delayed"]:
        k=0
        for z in [ "02"]:
            t_var = eval(f't_{track}_{sn}_z{z}')
            snII_var = eval(f'totsnII_{track}_{sn}_z{z}')
            agb_var = eval(f'totagb_{track}_{sn}_z{z}')
            winds_var = eval(f'totwinds_{track}_{sn}_z{z}')
            label_sn = sn.replace("_new", "")
            plt.plot(t_var, winds_var+ snII_var+agb_var, label=f'SEVN ov0.5 delayed Z=0.{z}', linestyle=linestyle[k])
            k=+3
    

plt.xscale('log')
#plt.yscale('log')
plt.xlim(10**5, 1e10)

plt.xticks(fontsize=17)
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
plt.savefig(f'../Images/disc/mass_return_all_winds.pdf', format='pdf')
plt.show()

print(totsnII_mar_z002[-1]+totagb_mar_z002[-1]+totwinds_mar_z002[-1])
print(winds_var[-1]+ snII_var[-1]+agb_var[-1])


# In[66]:


plt.figure(figsize=(7.7, 5.2))


plt.plot(t_z00004, totsnII_mar_z00004+totagb_mar_z00004+totwinds_mar_z00004, label='Portinari et al.(1998) Z=0.0004')   
plt.plot(t_fire3, tot_SNII_fire3+tot_OBAGBW_fire3_z, label='Hopkins et al.(2022) Z=0.0004')   
for track in ["ov05"]:
    for sn in [ "delayed"]:
        for z in ["0004"]:
            t_var = eval(f't_{track}_{sn}_z{z}')
            snII_var = eval(f'totsnII_{track}_{sn}_z{z}')
            agb_var = eval(f'totagb_{track}_{sn}_z{z}')
            winds_var = eval(f'totwinds_{track}_{sn}_z{z}')
            plt.plot(t_var, snII_var + agb_var + winds_var, label=f'Rate IORIO {track} {sn} Z=0.{z}')
    
    

plt.xscale('log')
#plt.yscale('log')
plt.xlim(1e5, 1e10)

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
plt.savefig(f'../Images/DISC/mass_all.pdf', format='pdf')
plt.show()


# In[106]:


df = pd.read_csv('tot_elemnt_ej_mar.csv')

# Initialize an empty dictionary
element_mar = {}

# List of elements (columns) to extract
elements = ["H", "He", "C", "N", "O", "Ne", "Mg", "Si", "Fe"]

# Iterate over each element and add to the dictionary
for element in elements:
    element_mar[element] = df[element].tolist()

    
df = pd.read_csv('tot_elemnt_ej_ior.csv')

# Initialize an empty dictionary
element_ior = {}

# List of elements (columns) to extract
elements = ["H", "He", "C", "N", "O", "Ne", "Mg", "Si", "Fe"]

# Iterate over each element and add to the dictionary
for element in elements:
    element_ior[element] = df[element].tolist()
t_ior=df["time"].tolist()


df = pd.read_csv('tot_elemnt_ej_SNHop.csv')
element_hop = {}

# List of elements (columns) to extract
elements = [ "He", "C", "N", "O", "Ne", "Mg", "Si", "Fe"]

# Iterate over each element and add to the dictionary
for element in elements:
    element_hop[element] = df[element].tolist()


df = pd.read_csv('tot_elemnt_ej_WHop.csv')
element_hopW= {}

# List of elements (columns) to extract
elements = [ "He", "C", "N", "O", "Ne", "Mg", "Si", "Fe"]

# Iterate over each element and add to the dictionary
for element in elements:
    print(element)
    element_hopW[element] = df[element].tolist()
    print(element_hopW[element][-1])
    element_hopW[element]=np.append(np.zeros(100), element_hopW[element])

element_hop["H"]=tot_OBAGBW_fire3+tot_SNII_fire3   
for element in elements:
    element_hop["H"]=element_hop["H"]-element_hopW[element]-element_hop[element]


# In[134]:


import matplotlib.cm as cm

def plot_yieldSNII_Z( element_mar, t_z002):
    # Determine the grid size (3x3)
    n_plots = len( element_mar)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    # Create the figure and GridSpec object
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(21, 15))
    
    viridis = cm.get_cmap('plasma', len(element_mar))
    viridis2 = cm.get_cmap('cividis', len(element_mar))
    
    # Flatten the axis array if it's not already flat
    axs = axs.flat if isinstance(axs, np.ndarray) else [axs]
    
    # Plot each dataset
    for i, (y, func) in enumerate(element_mar.items()):
        ax = axs[i]
        # Get the color from Viridis colormap
        color = viridis(i)
        color2 = viridis2(len(element_mar) - i - 1)
        
        # Plot the interpolated data
        ax.plot(t_var , element_mar[str(y)], label='mar', color="black")
        ax.plot(t_ior, element_ior[str(y)], label='SEVN', color=color)
        if str(y)!="H":
            ax.plot(t_fire3, element_hopW[str(y)]+element_hop[str(y)], label='Hopkins et al. (2022)', color=color2)
        else:
            ax.plot(t_fire3, element_hop["H"], label='Hopkins et al. (2022)', color=color2)
        
        ax.text(0.95, 0.05, f'\\textbf{{ {y} }}', transform=ax.transAxes,
                fontsize=25, verticalalignment='bottom', horizontalalignment='right')
        
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels, fontsize=13, loc='upper left')
        
        
        # Ensure ticks are enabled for all subplots
        ax.tick_params(axis='both', which='major', labelsize=23, direction='in', pad=10)
        ax.tick_params(axis='both', which='minor', direction='in', pad=10)
        
        # Add labels to x and y axes
        ax.set_xlabel('SSP age $[yr]$', fontsize=25)
        ax.set_xscale('log')
        ax.set_xlim(1e5, 1e10)
    
    # Remove any empty subplots
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])
    
    # Adjust layout to ensure labels fit and increase distance between plots
    plt.tight_layout(pad=3)
    plt.subplots_adjust(wspace=0.22, hspace=0.35)
    plt.savefig(f'../Images/DISC/Yeilds_confronto.pdf', format='pdf')
    # Display the plot
    plt.show()


plot_yieldSNII_Z(element_mar, t_z002)


# In[136]:


metals=["C", "N", "O", "Ne", "Mg", "Si", "Fe"]


totHop=np.zeros(len(t_fire3))
totIor=np.zeros(len(t_ior))
totMar=np.zeros(len(t_var))

for element in metals:
    totHop+=element_hopW[element]+element_hop[element]
    totIor+=element_ior[element]
    totMar+=element_mar[element]


# In[141]:


plt.figure(figsize=(7.7, 5.2))
plt.plot(t_fire3,  totHop, label='Hopkins et al. (2022)')
plt.plot(t_ior,  totIor, label='SEVN')
plt.plot(t_var,  totMar, label='Portinari et al. (1998)')


plt.xscale('log')



plt.xticks(fontsize=17)
plt.yticks(fontsize=16)
plt.tick_params(axis='x', which='both', direction='in', pad=6)  # Adjust pad to move x tick labels lower
plt.tick_params(axis='y', which='both', direction='in')
plt.xlim(1e5, 1e10)
#plt.ylim(5*1e-5, 2*1e-2)

plt.xlabel('SSP age $[yr]$', fontsize=18, labelpad=5)
plt.ylabel('Metal ejection fraction', fontsize=18,labelpad=7)
plt.tick_params(axis='both', direction='in', which='both')
ax = plt.gca()  # Get the current axes
ax.yaxis.set_ticks_position('both')  # Show ticks on both sides
ax.tick_params(axis='y', which='both', direction='in')
plt.legend(fontsize=13) 
plt.savefig(f'../Images/Disc/tot_metals.pdf', format='pdf')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




