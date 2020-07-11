# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 09:51:31 2020
@author: James Cotter

This program is motivated by the following question: if we start with a fixed 
lambda value EDR network and grow the radius of the circle, is there a point at
which we can no longer improve the total length? Furthermore, is this unique
for EDR networks or does it apply to other length distributions such as a power
law?

In order to answer this, I generate networks with both EDR and power law length
distributions and plot the initial/final length distributions after simulated
annealing for increasing values of r.
"""

#modules to generate EDR and power law length distribution networks
import make_edr_graph
import make_powerlaw_graph

#module for annealing
import annealing_util

#general
import pickle

#%%
def anneal_variable_r(N,l,a,r,distribution):
    """
    Anneals either a power law or EDR network to minimize the total length of
    network embedded on circle of radius r.
    
    Inputs:
        N: int, number of nodes
        l: float, lambda value for EDR
        a: float, exponent for power law
        r: float, radius of circle
        distribution: string, choice between EDR and POWER_LAW
    
    Outputs:
        globalMin: float, minimum found by annealing
        original_dist: array, original length distribution
        final_dist: array, final length distribution
        trajectory: array, trajectory of lengths during annealing
    """
    
    #make node positions
    x,y = make_edr_graph.make_nodes_variable_r(N,r)
    
    #make the distance matrix of distance between all pairs of points
    dist = annealing_util.distances(x,y)
    
    if distribution == "EDR":
                
        #generate adjacency matrix
        adjacency = make_edr_graph.get_adj_variable_r(N,l,r)
        
        #anneal it
        minimum,og_dist,final_dist,trajectory = annealing_util.anneal(N,dist,adjacency)
    
    elif distribution == "POWER_LAW":
        
        #generate adjacency matrix
        adjacency = make_powerlaw_graph.get_adj_pl(N,a,r)
        
        #anneal it
        minimum,og_dist,final_dist,trajectory = annealing_util.anneal(N,dist,adjacency)
        
    else:
        print("Not a valid distribution choice")
    
    return minimum,og_dist,final_dist,trajectory


def multiple_trials(N,l,a,r,distribution,trials):
    """
    Performs multiple trials of the anneal_variable_r function.
    """
    
    #counter on trials
    k = 0
    
    #to store results
    minima = []
    original_distribution = []
    final_distribution = []
    trajectories = []
    
    while k < trials:
        m,o_dist,f_dist,traj = anneal_variable_r(N,l,a,r,distribution)
        minima.append(m)
        original_distribution.extend(o_dist)
        final_distribution.extend(f_dist)
        trajectories.append(traj)
        k+=1
    
    return minima,original_distribution,final_distribution,trajectories

#%%
if __name__ == "__main__":

    #set parameters    
    N = 100
    l = 1
    a = 0.8
    
    #radius value to use
    r = 2
    
    #number of trials
    tr = 2
    
    #compute results (very easy to parallelize, do so for CRC script)
    m1,o_dist1,f_dist1,traj1 = multiple_trials(N,l,a,r,"EDR",tr)
    m2,o_dist2,f_dist2,traj2 = multiple_trials(N,l,a,r,"POWER_LAW",tr)
  
    #store and save
    data = []
    data.append([m1,o_dist1,f_dist1,traj1])
    data.append([m2,o_dist2,f_dist2,traj2])
    
    with open('7_10_r%.1f_annealing.dat' %r,'wb') as f:
        pickle.dump(data,f)

