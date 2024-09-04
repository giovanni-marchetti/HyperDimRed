import torch 
from distances import * 


class Embedder():
    def __init__(self, data_size, latent_dim, latent_dist_fun):
        self.embeddings = torch.randn((data_size, latent_dim), requires_grad=True)
        self.latent_dist_fun = latent_dist_fun

    def loss_fun(self):
        return None
 
class MDS(Embedder):
    def loss_fun(self, data_dist_matrix):
        latent_dist_matrix = dist_matrix(self.embeddings, self.latent_dist_fun)
        return ((data_dist_matrix - latent_dist_matrix)**2).mean()