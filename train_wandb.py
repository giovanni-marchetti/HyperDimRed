import argparse

import torch
import wandb

from OdorDataset import OdorMonoDataset
from utils.helpers import *
from methods import *
from optimizers import *
from torch.utils.data import DataLoader
from utils.visualization import *
import uuid
from utils.helpers import set_seeds
import scipy
from distances import (
    distance_matrix,
    euclidean_distance,
    poincare_distance,
    knn_geodesic_distance_matrix,
    knn_graph_weighted_adjacency_matrix,
    hamming_distance_matrix
)

def hasone(node_index, dim_index):
    bin_i, bin_j = np.binary_repr(node_index), np.binary_repr(dim_index)
    length = len(bin_j)
    return (bin_i[:length] == bin_j) * 1


def get_tree_data(depth, dtype=np.float32):
    n = 2 ** depth - 1
    x = np.fromfunction(lambda i, j: np.vectorize(hasone)(i + 1, j + 1),
                        (n, n), dtype=np.int32).astype(dtype)
    # print(x.shape)
    return x


sweep_configuration = {
"method": "grid",
"parameters": {
"representation_name": {"values": ['molformer']},
"batch_size": {"values": [4983]},
"num_epochs": {"values": [1000,10000]},
"latent_dim": {"values": [2]},
"lr": {"values": [0.1, 0.01, 0.001, 0.05]},
"base_dir": {"values": ['../../../T5 EVO/alignment_olfaction_datasets/curated_datasets/']},
"dataset_name": {"values": ['gslf']},
"seed": {"values": [2024]},

# "depth": {"values": [1]},
"temperature": {"values": [1,0.9,0.95,0.8,0.85]},
"normalize": {"values": [True]},
"optimizer": {"values": ['poincare']},
"model_name": {"values": [ 'contrastive']},
"latent_dist_fun": {"values": ['poincare']},
"distance_method": {"values": [ 'geo']},
"distr": {"values": [ 'hypergaussian']},
# "random_string": {"values": [uuid.uuid4().hex]}
}
}
# 3: Start the sweep

def main():
# if __name__ == "__main__":
    # add arguments


    # wandb.init(project='hyperbolic_smell')
    # wandb.config = args
    with wandb.init(config=None):
        args = wandb.config
        # args.random_string = uuid.uuid4().hex
        args.random_string = wandb.run.id
        dataset_name = args.dataset_name
        representation_name = args.representation_name
        num_epochs = args.num_epochs
        normalize = args.normalize
        # geodesic = args.geodesic
        # min_dist = args.min_dist
        latent_dim = args.latent_dim
        lr = args.lr
        seed = args.seed
        base_dir = args.base_dir
        optimizer = args.optimizer
        model_name = args.model_name
        latent_dist_fun = args.latent_dist_fun
        distr = args.distr
        distance_method = args.distance_method
        # n_samples = args.n_samples
        # dim = args.dim

        temperature = args.temperature
        batch_size = args.batch_size
        set_seeds(seed)




        set_seeds(seed)

        if dataset_name == 'tree':
            embeddings = get_tree_data(depth)
            ## binary_tree is a dataset of binary sequences.
            ## The root of the tree is the node 0: binary_tree[0]
            ## groundtruth distance from node i to the root of the tree (i.e. shortest path distance from node i to the root): hamming_distance(binary_tree[0], binary_tree[i])
            ## For visualizations, one can color a node by its groundtruth distance to the tree.
        elif dataset_name == 'random':
            embeddings = torch.randn(n_samples, dim)
        else:
            input_embeddings = f'embeddings/{representation_name}/{dataset_name}_{representation_name}_embeddings_13_Apr17.csv'
            embeddings, labels = read_embeddings(base_dir, select_descriptors(dataset_name), input_embeddings,
                                                 grand_avg=True if dataset_name == 'keller' else False)
            print(embeddings.shape)
        dataset = OdorMonoDataset(embeddings, transform=None)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        if latent_dist_fun != 'euclidean' and latent_dist_fun != 'poincare':
            raise ValueError('Latent distance function not recognized')
        if model_name == 'isomap':
            model = Isomap(len(dataset), latent_dim,
                           euclidean_distance if latent_dist_fun == 'euclidean' else poincare_distance)
        elif model_name == 'mds':
            model = MDS(len(dataset), latent_dim,
                        euclidean_distance if latent_dist_fun == 'euclidean' else poincare_distance)
        elif model_name == 'contrastive':
            model = Contrastive(len(dataset), latent_dim,
                                euclidean_distance if latent_dist_fun == 'euclidean' else poincare_distance,
                                distr=distr)
        else:
            raise ValueError('Model not recognized')

        if optimizer == 'standard':
            optimizer = StandardOptim(model, lr=lr)
        elif optimizer == 'poincare':
            optimizer = PoincareOptim(model, lr=lr)
        else:
            raise ValueError('Optimizer not recognized')


        losses = []
        # wandb.watch(model)
        for i in range(num_epochs):
            total_loss=0


            for idx, batch in data_loader:
                if normalize:
                    model.normalize()
                if distance_method == 'graph':
                    data_nn_matrix = knn_graph_weighted_adjacency_matrix(batch, n_neighbors=3, metric='minkowski')
                    data_dist_matrix = (data_nn_matrix > 0).astype(int)
                    data_dist_matrix = torch.tensor(data_dist_matrix)
                elif distance_method == 'geo':
                    data_dist_matrix = knn_geodesic_distance_matrix(batch)
                    if model_name == 'contrastive':
                        data_binary_dist_matrix = (data_dist_matrix <= 1.01).to(torch.int)
                elif distance_method == 'hamming':
                    data_dist_matrix = hamming_distance_matrix(batch)
                    if model_name == 'contrastive':
                        data_binary_dist_matrix = (data_dist_matrix <= 1.01).astype(int)
                        data_binary_dist_matrix = torch.tensor(data_binary_dist_matrix)
                    data_dist_matrix = torch.tensor(data_dist_matrix)
                else:
                    data_dist_matrix = distance_matrix(batch, euclidean_distance)

                #     data_dist_matrix = dist_matrix(batch, Euclidean)


                #binary matrix

                optimizer.zero_grad()
                loss = model.loss_fun(data_dist_matrix, idx, data_binary_dist_matrix, temperature)
                loss.backward()
                optimizer.step(idx)
                total_loss += loss.item()

            print(f'Epoch {i}, loss: {total_loss / len(data_loader):.3f}')
            losses.append(total_loss/len(data_loader))
            wandb.log({'loss': total_loss/len(data_loader)})
                # print('norms', vector_norm(model.embeddings, dim=-1).mean().item(), vector_norm(model.embeddings, dim=-1).max().item())
            # if i % 10 == 0:


        # laten_embeddings_norm= torch.norm(model.embeddings, dim=-1).cpu().detach().numpy()
        # e=scipy.spatial.distance.cdist(data_loader.dataset.embeddings, data_loader.dataset.embeddings, metric='hamming')*data_loader.dataset.embeddings.shape[-1]
        # scatterplot_2d(losses, model.embeddings.detach().cpu().numpy(), laten_embeddings_norm, args=args, data_dist_matrix=e)
        scatterplot_2d(losses, model.embeddings,dataset.embeddings, args=args)

sweep_id = wandb.sweep(sweep=sweep_configuration, project="hyperbolic_smell")

wandb.agent(sweep_id, function=main, count=40)