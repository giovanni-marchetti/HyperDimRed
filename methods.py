import torch 
from torch.linalg import vector_norm
from distances import * 

EPS = 0.00001

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Embedder():
    #An abstract class for embedding methods. 
    
    def __init__(self, data_size, latent_dim, latent_dist_fun=Euclidean, distr='gaussian'):
        if distr == 'gaussian':
            self.embeddings = torch.randn((data_size, latent_dim), requires_grad=True)
        self.latent_dist_fun = latent_dist_fun
        self.data_size = data_size

    def normalize(self):
        with torch.no_grad():
            norms = vector_norm(self.embeddings, dim=-1).unsqueeze(-1)  
            self.embeddings  = torch.where(norms < 1, self.embeddings , self.embeddings / (norms + EPS))
            self.embeddings.requires_grad = True
            self.embeddings.retain_grad()
            
    def loss_fun(self):
        pass


class MDS(Embedder):
    def loss_fun(self, data_dist_matrix,idx):
        latent_dist_matrix = dist_matrix(self.embeddings[idx], self.latent_dist_fun)
        return ((data_dist_matrix - latent_dist_matrix)**2).mean()

    
def isomap_kernel(data_dist_matrix): #input should be a distance matrix D
    N = data_dist_matrix.shape[0] # number of points considered
    I = torch.eye(N).to(device)
    A = torch.ones(N,N).to(device)
    return -0.5*torch.matmul(torch.matmul(I - (1/N) * A, torch.matmul(data_dist_matrix, data_dist_matrix)), (I - (1 / N) * A))
    
class Isomap(Embedder):
    def loss_fun(self, data_dist_matrix, idx):
        latent_dist_matrix = dist_matrix(self.embeddings[idx], self.latent_dist_fun)
        isomap_term = isomap_kernel(data_dist_matrix)-isomap_kernel(latent_dist_matrix)
        ##loss = torch.norm(isomap_term, p='fro')/self.data_size
        loss = torch.einsum("ij, ij ->", isomap_term, isomap_term)/self.data_size      
        return loss

class Contrastive(Embedder):
    def loss_fun(self, data_binary_dist_matrix,idx, temperature=1.):
        latent_dist_matrix = dist_matrix(self.embeddings[idx], self.latent_dist_fun)
        print(data_binary_dist_matrix.shape,"shapeee")
        
        positive_pairs = data_binary_dist_matrix[idx,idx] == 1
        print("pos1",positive_pairs.shape)
        pos_loss = latent_dist_matrix[positive_pairs].sum()/temperature
        print("pos2",pos_loss.shape)
        negative_pairs = data_binary_dist_matrix == 0
        # negative_pairs = negative_pairs[idx,idx]
        print("neg1",data_binary_dist_matrix[idx,idx])
        neg_loss = torch.logsumexp(-latent_dist_matrix[negative_pairs]/temperature, dim=0)
        print("neg2",neg_loss.shape)
        loss = pos_loss + neg_loss
        
        return loss
