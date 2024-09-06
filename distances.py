import torch
from torch.linalg import vector_norm
    
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




# if __name__ == '__main__':
#     x = torch.randn( (10,), requires_grad=True)
#     y = torch.randn( (10,), requires_grad=True )
    
#     xp = torch.div(x,  vector_norm(x))

#     yp = torch.div(y, vector_norm(y))

#     dist = Poincare(xp, yp).mean()    
#     dist.backward()
#     print(x.grad)

