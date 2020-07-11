# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 21:55:39 2020
@author: James Cotter

This is the module for all the annealing related functions needed. In the
original version I simply imported what I needed from the various things I 
have written over the past two years, but for the purposes of putting it on the
box it is good to consolidate.
"""

import numpy as np
import matplotlib.pyplot as plt

def distances(x,y):
    """Calculates the distance between every combination of N points 
    and enters them in a numpy array. The matrix should be symmetric with 
    trace = 0.
    Input 
        N: int, number of points
    Output 
        dist: array, distance between points i,j.
    """
    
    dx = x[:,None] - x
    dy = y[:,None] - y
    
    dist = np.sqrt(dx**2 + dy**2)
     
    #print(dist)
    return dist

def plot(x,y,N,adj):
    """Plots the network, good for sanity checks.
    Inputs
        N: int, the number of points on the circle
        l: float, lambda parameter
    """
    
    connection = adj
    x_edges = []
    y_edges = []
    
    for i in range(N):
        for j in range(N):
            if connection[i,j] == 1:
                x_edges.append(x[i])
                y_edges.append(y[i])
                x_edges.append(x[j])
                y_edges.append(y[j])
                x_edges.append(np.nan)
                y_edges.append(np.nan)
    
    plt.plot(x_edges,y_edges,'r')
    plt.scatter(x,y)
    
def neuron_length_distribution(dist,adjacency,N):
    """Returns the distribution of edge lengths of the neuron level network.
    Inputs:
        N: int, number of neurons
        dist: array, neuron distance matrix of the network
        adjacency: array neuron adjacency matrix
    Outputs:
        lengths_list: array, list of average lengths
    """
    
    lengths_list = []
    
    edges = np.nonzero(adjacency)
    edge_array = np.stack((edges[0],edges[1]))       
    r = 0
    
    while r < len(edges[0]):
        x,y = edge_array[:,r]
        lengths_list.append(dist[x][y])
        r+=1

        #print(k)
    return lengths_list

def total_length2(dist,adjacency):
    """Returns the total length by summing the elementwise multiplication of the 
    distance matrix and the adjacency matrix (i.e. the weighted adjacency matrix).
    Possibly faster than the other method.
    Inputs:
        dist: arr, distance matrix
        adjacency: arr, adjacency matrix
    Returns:
        length: float, total length
    """
    
    return np.sum(np.multiply(dist,adjacency))


def swaps(dist,N):
    """Performs a random swap of two rows and columns of the distance matrix
    which swaps the location of nodes without changing the adjacency.
    Inputs:
        dist: arr, NxN matrix of distance between all pairs of nodes
        N: int, number of points
    Outputs:
        dist: arr, swapped array
    """
    np.random.seed(None)
    rand1 = np.random.randint(N)
    rand2 = np.random.randint(N)
    
    #swap rows
    dist[[rand1,rand2]] = dist[[rand2,rand1]]
    
    #swap columns
    dist[:,[rand1,rand2]] = dist[:,[rand2,rand1]]
    
    return dist

#Temperature Parameters
T0 = 15
Tf = 0.1
Tstep = 1.2
    
def epochLength(T,N):
    """
    Epoch length was determined by measuring the number of swaps needed to reach
    saturation for a typical 1000 node network.
    """
    return ((np.log(T) - np.log(T0)) / (np.log(Tf) - np.log(T0))*2*N + 2*N).astype(np.int)

def anneal(N,dist,adj):
    """
    Performs simulated annealing to reduce total edge length by iteratively
    swapping the positions of nodes.
    Inputs:
        N: int, number of nodes
        dist: arr, NxN matrix of distance between all pairs of nodes
        adj: arr, adjacency matrix of the network
    Outputs:
        globalMinF: float, global min found
        original_dist: arr, original distribution of lengths
        final_dist: arr, distribution of lengths after swaps
        trajectory: arr, trajectory of total length through each swap.
    """
    
    #distributions
    original_dist = neuron_length_distribution(dist,adj,N)
    final_dist = neuron_length_distribution(dist,adj,N)

    #Evaluate the function
    edge_length = total_length2(dist,adj)

    #Set it as global minima
    globalMinF = edge_length
    min_dist = dist
    print('Original edge length: %f' %(globalMinF))
    
    # compute the temperatures and the epochs for each temperature
    tSteps = np.round(-np.log(Tf / T0) / np.log(Tstep)) #If Tstep > 1
    temps = np.geomspace(T0, Tf, tSteps)
    lengths = epochLength(temps,N)
    
    # allocate arrays for recording the trajectory
    totalSteps = np.sum(lengths)
    trajectory = np.zeros(totalSteps)
    
    step = 0 # counter
    
    # iterate over temperatures    
    for j, T in enumerate(temps):
        
        # steps of an epoch
        for i in range(lengths[j]):
            # propose a random step
            dist_copy = np.copy(dist)
            new_dist = swaps(dist_copy,N)
            
            # evaluate the function
            new_el = total_length2(new_dist,adj)
            
            a = np.exp((edge_length - new_el) / T)
            r = np.random.rand()
                        
            # Metropolis: accept or reject
            if ((new_el < edge_length) | (r < a)):
                
                # accept
                edge_length = new_el
                dist = new_dist
                
                #check if we have new global minimum
                if new_el < globalMinF:
                    globalMinF = new_el
                    #print('***********DECREASE*********')
                    min_dist = dist
                    final_dist = neuron_length_distribution(min_dist,adj,N)

            # record trajectory
            trajectory[step] = edge_length
            
            # count the steps, show progress
            step +=1
            if (step % 100*N == 0):
                print('T = %f' % T)
                        
    return globalMinF,original_dist,final_dist,trajectory
