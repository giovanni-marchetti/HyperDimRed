import torch
from torch.linalg import vector_norm
import numpy as np
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import dijkstra
import scipy
EPS = 0.00001


def dist_matrix(data, dist_fun):
    #Builds a distance matrix given a vectorized distance function
    #data: (B, D)

    preliminary =  dist_fun(data.unsqueeze(0), data.unsqueeze(1))
    result = preliminary + torch.eye(data.shape[0])
    return result.float()


def Euclidean(x, y):
    # x, y : (..., D)

    # return ((x - y)**2).sum(-1)
    return vector_norm(x-y, dim=-1)


def Poincare(x, y): 
    # x, y : (..., D)

    euc_dist = ((x - y)**2).sum(-1)
    norm_x = torch.clamp(1 - (x**2).sum(-1), min=EPS)
    norm_y = torch.clamp(1 - (y**2).sum(-1), min=EPS)
    x = torch.clamp(1 + 2 * torch.div(euc_dist, (norm_x * norm_y )), min=1 + EPS)
    dist = torch.acosh(x)
    # print('dist', dist.mean().item())
    return dist

def hamming_distance(x, y):
    return (x.astype(np.int32) ^ y.astype(np.int32)).sum()


def geo_distance(data,n_neighbors=3):
    data_nn_matrix = kneighbors_graph(data, n_neighbors, mode='distance', include_self=False)
    data_nn_matrix = data_nn_matrix.toarray()
    data_dist_matrix = dijkstra(data_nn_matrix)
    data_dist_matrix = torch.FloatTensor(data_dist_matrix)
    data_dist_matrix = torch.where(data_dist_matrix == torch.inf, 1000 * torch.ones_like(data_dist_matrix),data_dist_matrix)
    return data_dist_matrix

def nngraph_distance(data,n_neighbors=3,metric='minkowski'):
    data_nn_matrix = kneighbors_graph(data, n_neighbors, mode='distance', include_self=False,metric=metric)
    data_nn_matrix = data_nn_matrix.toarray()

    return data_nn_matrix

def hamming_distance_matrix(data):
    data_nn_matrix = scipy.spatial.distance.cdist(data,data, metric='hamming')*data.shape[-1]

    return data_nn_matrix



