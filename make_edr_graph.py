# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 21:44:49 2020
@author: James Cotter

Generates EDR networks using Ferenc's method of computing the binning to avoid
errors. 
"""

import numpy as np
import matplotlib.pyplot as plt


def make_nodes_variable_r(N,r):
    """Get coordinates of nodes equidistant on circle of radius r"""
    angle = (np.arange(N)/N)*np.pi*2
    x = r*np.cos(angle)
    y = r*np.sin(angle)
    return x,y

def make_distances(x, y):
    """compute the 2-norm distances between adjacent nodes (rotationally symmetric, one to all)"""
    D = np.sqrt((x[0]-x)**2 + (y[0]-y)**2)
    return D

def get_edges(D,lam):
    """Sample the EDR to get edges of the graph, one outgoing edge per node.
    Input D: One to all distances
    Output: list of edge OFFSETS (for each node, how many steps to take along the circle to find where the edge should points to),
    and the corresponding lengths of those edges."""

    # IDEA: the distance values will define bins, bin boundaries are placed exactly in between the distance values.
    # These bins have variable width!!
    # Picking a bin between a and b should have the probability of Integrate[a to b][exp(-lam*x)dx]
    # The integral of the exponential has an explicit formula.

    N = D.shape[0] # number of nodes

    # set of UNIQUE distances
    dd = set(D)
    d = np.array(sorted(dd))
    
    # For each distance value, find out which links fall into the corresponding bin.
    # Some will have 1 edge, some will have 2.
    # Analytically, each bin should have 2, except the longest distance when N is odd.
    # BUT Numerically the roundoff errors mess this up! So we need to compute binning of edges explicitly.
    idx = []
    for dist in d:
        idx.append(np.nonzero(D == dist)[0])
    
    # compute the interior bin boundaries to unique distances
    bounds = (d[1:] + d[:-1])/2
    
    # edges: mirror the bin width
    left = d[0] - (d[1]-d[0])/2
    right = d[-1] + (d[-1]-d[-2])/2
    
    # add the edges to the interior bounds
    bounds = np.hstack((left, bounds, right))
    
    # integrate EDR between these bounds to get each bin probability
    prob = (np.exp(-lam * bounds[:-1]) - np.exp(-lam * bounds[1:])) / lam
    
    prob[0] = 0 # zero distance not allowed
    
    # compute normalized cumulative sum for inverse sampling
    p = np.cumsum(prob)
    p /= p[-1]
    
    # inverse sampling (vectorized)
    r = np.random.rand(N)
    e = np.searchsorted(p, r) # random samples of DISTANCE index
    
    distances = d[e] # edge lengths
    
    # pick the edges (random choice of which edge to choose from each selected bin)
    L = np.array([len(i) for i in idx])  # length of each bin, 1 or 2 
    pick = np.floor(np.random.rand(L.shape[0]) * L).astype(np.int)
    
    
    neighbor_offsets = np.array([idx[i][pick[i]] for i in e])
    
    return neighbor_offsets, distances

def get_adj_variable_r(N,l,r):
    """
    Gets the adjacency matrix using the get_edges function.
    Inputs:
        N: int, number of nodes
        l: int, lambda value
        r: int, radius
    Outpus:
        adj: array, adjacency matrix
    """
    
    x,y = make_nodes_variable_r(N,r)
    D = make_distances(x,y)
    offsets,lengths = get_edges(D,l)
    
    edge_list = [(i, (i+offsets[i]) % N) for i in range(N)]
    adj = np.zeros((N,N))
    
    for i in range(N):
        adj[i][edge_list[i][1]] += 1
    
    return adj
    

if __name__ == "__main__":
    
    # Make an EDR graph
    
    N = 10000 # number of nodes
    r = 2    # radius
    l = 1.5  # lambda for the exponential
    x, y = make_nodes_variable_r(N,r)
    D = make_distances(x, y)
    
    offsets, lengths = get_edges(D,l)
    
    # this is the EDR graph, given as an edge list
    edge_list = [(i, (i+offsets[i]) % N) for i in range(N)]
    
    #edge length distribution: average over many graphs
    all_lengths = []
    for i in range(20):
        _, lengths = get_edges(D,l)
        all_lengths.extend(lengths)
        
    plt.figure(1)
    plt.clf()
    plt.hist(all_lengths, bins=np.linspace(0, 2*r, 50))
    plt.yscale('log')
    
