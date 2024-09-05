import torch 
from torch.linalg import vector_norm

from distances import *
from methods import * 
from optimizers import * 

latent_dim = 2
lr = 0.001
num_epochs = 10
normalize = True

data = torch.randn(10, 10)
data_dist_matrix = dist_matrix(data, Euclidean)

 
# torch.manual_seed(42)       
# torch.cuda.empty_cache()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print('Using device: ', device)

    


model = MDS(data.shape[0], latent_dim, Poincare)
optimizer = StandardOptim(model, lr=lr)

if __name__ == "__main__":
    for i in range(num_epochs):
        if normalize:
            model.normalize()

        print('norms', vector_norm(model.embeddings, dim=-1).mean().item(), vector_norm(model.embeddings, dim=-1).max().item())

        optimizer.zero_grad()
        loss = model.loss_fun(data_dist_matrix)
        loss.backward()
        print('grads', vector_norm(model.embeddings.grad, dim=-1).max().item())
        optimizer.step()
        

        if i % 100 == 0:
            print(f'Epoch {i}, loss: {loss:.3f}')

