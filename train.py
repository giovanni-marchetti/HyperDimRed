import torch 
from torch.linalg import vector_norm
from scipy.sparse.csgraph import dijkstra
from constants import *
from distances import *
from methods import * 
from optimizers import * 
from visulization import *
from OdorDataset import OdorMonoDataset
import numpy as np

latent_dim = 2
lr = 0.001
num_epochs = 1000
normalize = False

geodesic = False
min_dist = 1.

dataset_name='sagar'
model_name = 'molformer'


def select_descriptors(dataset_name):
    if dataset_name=='sagar':
        return sagar_descriptors
    elif dataset_name=='keller':
        return keller_descriptors
    else:
        return None


if dataset_name == 'random':
    data = torch.randn(10, 10)

else:
    base_dir = '../../../T5 EVO/alignment_olfaction_datasets/curated_datasets/'
    input_embeddings = f'embeddings/{model_name}/{dataset_name}_{model_name}_embeddings_13_Apr17.csv'
    dataset = OdorMonoDataset(base_dir, input_embeddings, transform=None, grand_avg=True, descriptors=select_descriptors(dataset_name))
    data = dataset.embeddings
data_dist_matrix = dist_matrix(data, Euclidean)

#IsoMap-style geodesic distance for data
if geodesic:
    truncated_matrix = torch.where(data_dist_matrix < min_dist, data_dist_matrix, torch.inf)
    data_dist_matrix = dijkstra(truncated_matrix.detach().cpu().numpy())
    data_dist_matrix = torch.FloatTensor(data_dist_matrix)
    data_dist_matrix = torch.where(data_dist_matrix == torch.inf, 1000 * torch.ones_like(data_dist_matrix), data_dist_matrix)


# torch.manual_seed(42)       
# torch.cuda.empty_cache()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print('Using device: ', device)

    


model = MDS(data.shape[0], latent_dim, Euclidean)
optimizer = StandardOptim(model, lr=lr)

if __name__ == "__main__":
    for i in range(num_epochs):
        if normalize:
            model.normalize()

        # print('norms', vector_norm(model.embeddings, dim=-1).mean().item(), vector_norm(model.embeddings, dim=-1).max().item())

        optimizer.zero_grad()
        loss = model.loss_fun(data_dist_matrix)
        loss.backward()
        # print('grads', vector_norm(model.embeddings.grad, dim=-1).mean().item())
        optimizer.step()
        

        if i % 10 == 0:
            print(f'Epoch {i}, loss: {loss:.3f}')

    scatterplot_2d(model.embeddings.detach().cpu().numpy(), labels=None, title='Poincare Embeddings')



