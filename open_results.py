# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 11:54:55 2020
@author: James Cotter

Opens the results of the lambda_vs_radius.py program and plots them.
"""

#%%
#imports
import pickle
import matplotlib.pyplot as plt
from scipy.stats import expon
from scipy.stats import powerlaw
import numpy as np

#load data
pick = open("7_10_r2.0_annealing.dat","rb")
data = pickle.load(pick)

#network paramters used
radius = 2
realizations = 2

#%% Plotting

#------------------ EDR Networks -----------------------------------

o_dist1 = data[0][1]
f_dist1 = data[0][2]

original_len = sum(o_dist1)/realizations
final_len = sum(f_dist1)/realizations

#fit
r = expon.fit(o_dist1)
rx = np.linspace(0,max(o_dist1),100)
rp = expon.pdf(rx,*r)
l1_fit = 1/r[1]

#plot it
fig1,(ax1,ax2) = plt.subplots(2)
n1,bins1,patches1 = ax1.hist(o_dist1,bins = 100, density = True, alpha = 0)
ax1.set(xlabel = "Edge Length",ylabel = "log(counts)")
ax1.set_title('Original Length Distribution length = %.2f' %(sum(o_dist1)/realizations))
ax1.set_yscale('log')
ax1.scatter(bins1[:-1]+0.5*(bins1[1:]-bins1[:-1]),n1,marker = 'x',c = 'red',s = 40,alpha = 1)
#ax1.plot(rx,rp,linewidth = 2)
ax1.set_ylim(ymin = 10e-6, ymax = 10e1)
ax1.legend()

n1,bins1,patches1 = ax2.hist(f_dist1,bins = 100, density = True, alpha = 0)
ax2.set(xlabel = "Edge Length",ylabel = "log(counts)")
ax2.set_title('Annealed Length Distribution, length = %.2f' %(sum(f_dist1)/realizations))
ax2.set_yscale('log')
ax2.scatter(bins1[:-1]+0.5*(bins1[1:]-bins1[:-1]),n1,marker = 'x',c = 'red',s = 40,alpha = 1)
ax2.set_ylim(ymin = 10e-6, ymax = 10e1)
fig1.tight_layout()

#plt.savefig('7_7_r%.1f_plots_EDR.png' %radius)

#plot trajectory
plt.figure(2)
mins = []
for i in range(realizations):
    trajectory = data[0][3][i]
    x = np.arange(0,len(trajectory))
    mins.append(trajectory[-1])
    plt.plot(x,trajectory)
    
plt.xlabel("Swaps")
plt.ylabel("Total length")
plt.title("Trajectory of Search for Min Total Length, Average Min = %.1f" %np.mean(mins))



#------------------------ Power Law Networks ----------------------------------
o_dist2 = data[1][1]
f_dist2 = data[1][2]

#original and final length
original_len = sum(o_dist2)/realizations
final_len = sum(f_dist2)/realizations

#fitting
r = powerlaw.fit(o_dist2)
rx = np.linspace(0,max(o_dist2),100)
rp = powerlaw.pdf(rx,*r)
l1_fit = 1/r[1]

fig3,(ax1,ax2) = plt.subplots(2)
n2,bins2,patches2 = ax1.hist(o_dist2,bins = 100, density = True, alpha = 0)
ax1.set(xlabel = "log(Edge Length)",ylabel = "log(counts)")
ax1.set_title('Original Length Distribution length = %.2f' %(sum(o_dist2)/realizations))
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.scatter(bins2[:-1]+0.5*(bins2[1:]-bins2[:-1]),n2,marker = 'x',c = 'red',s = 40,alpha = 1)
ax1.set_ylim(ymin = 10e-5)
#ax1.plot(rx,rp,linewidth = 2)
ax1.legend()

n2,bins2,patches2 = ax2.hist(f_dist2,bins = 100, density = True, alpha = 0)
ax2.set(xlabel = "Edge Length",ylabel = "log(counts)")
ax2.set_title('Annealed Length Distribution, length = %.2f' %(sum(f_dist2)/realizations))
ax2.set_yscale('log')
ax2.set_xscale('log')
ax2.scatter(bins2[:-1]+0.5*(bins2[1:]-bins2[:-1]),n2,marker = 'x',c = 'red',s = 40,alpha = 1)
ax2.set_ylim(ymin = 10e-5)
fig3.tight_layout()
#plt.savefig('7_7_r%.1f_plots_powerlaw.png' %radius)


#plot trajectory
plt.figure(4)
mins = []
for i in range(realizations):
    trajectory = data[1][3][i]
    x = np.arange(0,len(trajectory))
    mins.append(trajectory[-1])
    plt.plot(x,trajectory)
    
plt.xlabel("Swaps")
plt.ylabel("Total length")
plt.title("Trajectory of Search for Min Total Length, Average Min = %.1f" %np.mean(mins))


