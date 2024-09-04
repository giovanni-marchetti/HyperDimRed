import torch 
from torch.optim import Adam

from distances import *
from methods import * 


latent_dim = 10
lr = 0.001
num_epochs = 1000

data = torch.randn(100, 100)
data_dist_matrix = dist_matrix(data, Euclidean)


# torch.manual_seed(42)       
# torch.cuda.empty_cache()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print('Using device: ', device)



model = MDS(data.shape[0], latent_dim, Euclidean)
optimizer = Adam([model.embeddings], lr=lr)

if __name__ == "__main__":
    for i in range(num_epochs):

        optimizer.zero_grad()
        loss = model.loss_fun(data_dist_matrix)
        loss.backward()
        optimizer.step()
        print(f'Epoch {i}, loss: {loss:.3f}')
