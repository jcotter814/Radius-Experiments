# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 10:56:59 2020
@author: James Cotter

Making a power law distribution of length P_ij = (d_ij)^(-lambda) of edges using 
Ferenc's method for computing bin sizes.

When the bin widths are integrated to get probabilities, it will give a warning
for negative power laws because it will divide by 0 or take a root of a negative
number. This is ok because it is always for the first bin which is set to 0 
anyway because self loops are not permitted.
"""

import numpy as np
import matplotlib.pyplot as plt

def make_distances(x, y):
    """compute the 2-norm distances between adjacent nodes (rotationally symmetric, one to all)"""
    D = np.sqrt((x[0]-x)**2 + (y[0]-y)**2)
    return D

def make_nodes_variable_r(N,r):
    """Get coordinates of nodes equidistant on circle of radius r"""
    angle = (np.arange(N)/N)*np.pi*2
    x = r*np.cos(angle)
    y = r*np.sin(angle)
    
    return x,y

def get_edges_pl(D,lam):
    """Sample power law distribution to get edges of the graph, one outgoing edge per node.
    Input D: One to all distances
    Output: list of edge OFFSETS (for each node, how many steps to take along the circle to find where the edge should points to),
    and the corresponding lengths of those edges."""

    # IDEA: the distance values will define bins, bin boundaries are placed exactly in between the distance values.
    # These bins have variable width!!
    # Picking a bin between a and b should have the probability of Integrate[a to b][exp(x^(-lambda)dx]
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
    
    # integrate power law distribution between these bounds to get each bin probability
    prob = (np.power(bounds[:-1],(-lam+1))-np.power(bounds[1:],(-lam+1)))/(-lam+1)
    
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

def get_adj_pl(N,a,r):
    """ 
    Gets the adjacency matrix for an N node network with power law exponent
    a on a circle of radius r.
    
    """
    
    x,y = make_nodes_variable_r(N,r)

    D = make_distances(x,y)
    offsets,lengths = get_edges_pl(D,a)
    
    edge_list = [(i, (i+offsets[i]) % N) for i in range(N)]
    adj = np.zeros((N,N))
    
    for i in range(N):
        adj[i][edge_list[i][1]] += 1
    
    return adj
    
if __name__ == "__main__":
    
    #Make power law graph
    N = 5000
    r = 1
    l = 1.1  # lambda for the power law
    x, y = make_nodes_variable_r(N,r)
    D = make_distances(x, y)
    
    offsets, lengths = get_edges_pl(D,l)
    
    # power law graph, given as an edge list
    edge_list = [(i, (i+offsets[i]) % N) for i in range(N)]
    
    # edge length distribution: average over many graphs
    all_lengths = []
    for i in range(20):
        _, lengths = get_edges_pl(D,l)
        all_lengths.extend(lengths)
    
    plt.figure(1)
    plt.clf()
    plt.hist(all_lengths, bins=np.linspace(0, 2*r, 50),density = True)
    plt.yscale('log')
    plt.xscale('log')

