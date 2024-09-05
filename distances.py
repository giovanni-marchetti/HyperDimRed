import torch
    

def dist_matrix(data, dist_fun):
    #Builds a distance matrix given a vectorized distance function
    #data: (B, D)

    return dist_fun(data.unsqueeze(0), data.unsqueeze(1))

def Euclidean(x, y):
    # x, y : (..., D)

    return ((x - y)**2).sum(-1)

def Poincare(x, y):
    # x, y : (..., D)

    euc_dist = ((x - y)**2).sum(-1)
    norm_x = (x**2).sum(-1)
    norm_y = (y**2).sum(-1)
    dist = torch.acosh(1 + 2 * euc_dist / (( 1 - norm_x )*( 1 - norm_y )))**2
    return dist

