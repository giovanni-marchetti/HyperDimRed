import torch 
from distances import * 



class Embedder():
    #An abstract class for embedding methods. 
    def __init__(self, data_size, latent_dim, latent_dist_fun=Euclidean, distr='gaussian'):
        if distr == 'gaussian':
            self.embeddings = torch.randn((data_size, latent_dim), requires_grad=True)
        elif distr == 'disc':
            samples = torch.randn((data_size, latent_dim))
            self.embeddings = torch.div(samples, torch.norm(samples, dim=-1, keepdim=True) + 0.001).detach()
            self.embeddings.requires_grad = True
            self.embeddings.retain_grad = True
        self.latent_dist_fun = latent_dist_fun

    def loss_fun(self):
        pass

   
 
class MDS(Embedder):
    def loss_fun(self, data_dist_matrix):
        latent_dist_matrix = dist_matrix(self.embeddings, self.latent_dist_fun)
        return ((data_dist_matrix - latent_dist_matrix)**2).mean()
    
# class Contrastive(Embedder):
#     def __init__(self, data_size, latent_dim, latent_dist_fun, temperature=1.):
#         super(self).__init__(data_size, latent_dim) 
#         self.temperature = temperature 

#     def loss_fun(self, data_dist_mask):
#         latent_dist_matrix = dist_matrix(self.embeddings, self.latent_dist_fun) * data_dist_mask

#         positive = 
#         negative = 