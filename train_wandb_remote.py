import argparse

# import torch
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

os.environ["WANDB_API_KEY"] = "128ba237ed5b081911f0564e6c8d34eb864f78cf"
os.environ['WANDB_START_METHOD'] = 'thread'
parser = argparse.ArgumentParser(description='HyperDimRed')


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


# 3: Start the sweep

def main():


    with wandb.init(config=None):






        args = wandb.config

        if torch.cuda.is_available():
            args.device = torch.device('cuda')
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            print('Using GPU')
        else:
            args.device = torch.device('cpu')
            args.gpu_index = -1
            print('Using CPU')


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

        if dataset_name == 'tree':
            embeddings = get_tree_data(depth)
            ## binary_tree is a dataset of binary sequences.
            ## The root of the tree is the node 0: binary_tree[0]
            ## groundtruth distance from node i to the root of the tree (i.e. shortest path distance from node i to the root): hamming_distance(binary_tree[0], binary_tree[i])
            ## For visualizations, one can color a node by its groundtruth distance to the tree.
        elif dataset_name == 'random':
            embeddings = torch.randn(n_samples, dim)
        else:
            print("x1")
            input_embeddings = f'alignment_olfaction_datasets/curated_datasets/embeddings/{representation_name}/{dataset_name}_{representation_name}_embeddings_13_Apr17.csv'
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
        model = model.to(args.device)

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
                batch = batch.to(args.device)
                if normalize:
                    model.normalize()
                if distance_method == 'graph':
                    data_nn_matrix = knn_graph_weighted_adjacency_matrix(batch, n_neighbors=3, metric='minkowski')
                    data_dist_matrix = (data_nn_matrix > 0).astype(int)
                    data_dist_matrix = torch.tensor(data_dist_matrix)
                elif distance_method == 'geo':
                    data_dist_matrix = knn_geodesic_distance_matrix(batch.detach().cpu().numpy())
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
                optimizer.zero_grad()
                loss = model.loss_fun(data_dist_matrix, idx, data_binary_dist_matrix, temperature)
                loss.backward()
                optimizer.step(idx)
                total_loss += loss.item()

            print(f'Epoch {i}, loss: {total_loss / len(data_loader):.3f}')
            losses.append(total_loss/len(data_loader))
            wandb.log({'loss': total_loss/len(data_loader)})

        scatterplot_2d( model.embeddings.detach().cpu().numpy(),dataset.embeddings.detach().cpu().numpy(), args=args,losses=losses,losses_neg=model.losses_neg if model_name=='contrastive' else [],losses_pos=model.losses_pos if model_name=='contrastive' else [])
if __name__ == "__main__":
    main()
# sweep_id = wandb.sweep(sweep=sweep_configuration, project="hyperbolic_smell")

# wandb.agent(sweep_id, function=main, count=40)