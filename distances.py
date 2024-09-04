import torch

def dist_matrix(data, dist_fun):
    #data: (B, D)
    return dist_fun(data.unsqueeze(0), data.unsqueeze(1))

def Euclidean(x, y):
    # x, y : (..., D)
    return ((x - y)**2).sum(-1)