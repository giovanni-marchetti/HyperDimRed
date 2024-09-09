
from torch.utils.data import DataLoader
import random
from constants import *
from methods import *
from optimizers import * 
from utils.visulization import *
from OdorDataset import OdorMonoDataset
from utils.helpers import *
latent_dim = 2
lr = 0.1
num_epochs = 100000
normalize = False
geodesic = False
min_dist = 1.
distance_method = None




############ Synthetic binary tree ############

import numpy as np
from distances import hamming_distance



# #Load the binary tree data
# depth = 11
# binary_tree = get_binary_data(depth) # creates a binary tree dataset consisting of (2**(depth)-1) vectors of dimension (2**(depth)-1)


## binary_tree is a dataset of binary sequences.
## The root of the tree is the node 0: binary_tree[0]
## groundtruth distance from node i to the root of the tree (i.e. shortest path distance from node i to the root): hamming_distance(binary_tree[0], binary_tree[i])  
## For visualizations, one can color a node by its groundtruth distance to the tree.

##############################################


dataset_name='gslf'
model_name = 'molformer'
batch_size =10

def set_seeds(seed):

    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)



# set_seeds(2025)
base_dir = '../../../T5 EVO/alignment_olfaction_datasets/curated_datasets/'
input_embeddings = f'embeddings/{model_name}/{dataset_name}_{model_name}_embeddings_13_Apr17.csv'

dataset = OdorMonoDataset(base_dir, None, transform=None, grand_avg=False, descriptors=select_descriptors(dataset_name))
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,drop_last=True)


    # del dataset






#IsoMap-style geodesic distance for data


# torch.manual_seed(42)
# torch.cuda.empty_cache()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print('Using device: ', device)

    


#model = MDS(data.shape[0], latent_dim, Poincare)
# print(len(dataset))
# model = Isomap(len(dataset), latent_dim, Euclidean)
# model = Contrastive(len(dataset), latent_dim, Euclidean)


model = Contrastive(len(dataset), latent_dim, Poincare)

#optimizer = StandardOptim(model, lr=lr)
optimizer = PoincareOptim(model, lr=lr)

if __name__ == "__main__":
    for i in range(num_epochs):
        total_loss=0

        for idx, batch in data_loader:
            if normalize:
                model.normalize()

            if distance_method == 'graph':
                data_nn_matrix = graph_distance(batch)
                data_dist_matrix = (data_nn_matrix > 0).astype(int)
            elif distance_method == 'geo':
                data_dist_matrix = geo_distance(batch)
            else:
                data_dist_matrix = dist_matrix(batch, Euclidean)
            # if geodesic:
            #     data_dist_matrix = geo_distance(batch)
            # else:
            #     data_dist_matrix = dist_matrix(batch, Euclidean)


            #binary matrix


            optimizer.zero_grad()
            loss = model.loss_fun(data_dist_matrix,idx)
            loss.backward()
            optimizer.step(idx)
            total_loss += loss.item()



            # print('norms', vector_norm(model.embeddings, dim=-1).mean().item(), vector_norm(model.embeddings, dim=-1).max().item())


        if i % 10 == 0:
            print(f'Epoch {i}, loss: {total_loss/len(data_loader):.3f}')

    scatterplot_2d(model.embeddings.detach().cpu().numpy(), labels=None, title='Poincare Embeddings')



    # size1 = (0.3, 0.28)
    # size2 = (0.6, 1)
    # size3 = (1, 0.35)
    #
    # plt.rcParams["font.size"] = 35
    # df_gslf_mols = prepare_goodscentleffignwell_mols(base_dir)
    # pom_frame(np.asarray(model.embeddings.detach().cpu().numpy().values.tolist()),
    #           np.asarray(df_gslf_mols.y.values.tolist()), "/kaggle/working/", gs_lf_tasks, "molformer", size1, size2,
    #           size3)


