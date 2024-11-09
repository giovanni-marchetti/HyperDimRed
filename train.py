import argparse

import numpy as np
import torch
# import wandb

from OdorDataset import OdorMonoDataset
from utils.helpers import *
from methods import *
from optimizers import *
from torch.utils.data import DataLoader
from utils.visualization import *
import uuid
from utils.helpers import set_seeds
from sklearn.neighbors import kneighbors_graph
import scipy
from distances import (
    distance_matrix,
    euclidean_distance,
    poincare_distance,
    knn_geodesic_distance_matrix,
    knn_graph_weighted_adjacency_matrix,
    # hamming_distance_matrix
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
    return x, x


### If using Jupyter Notebook:###
# import sys
# if 'ipykernel_launcher' in sys.argv[0]:
#     sys.argv = sys.argv[:1]
###





if __name__ == "__main__":

    parser = argparse.ArgumentParser('Hyperbolic Smell')
    parser.add_argument('--representation_name', type=str, default='molformer')
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--num_epochs', type=int, default=100)
    # parser.add_argument('--min_dist', type=float, default=1.)
    parser.add_argument('--latent_dim', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.001)
    # parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--base_dir', type=str,
                        default='./data/')

    parser.add_argument('--dataset_name', type=str, default='gslf' , choices={"gslf"," ravia","keller","sagar"})  # tree for synthetic, gslf for real
    parser.add_argument('--normalize', type=bool, default=True)  # only for Hyperbolic embeddings
    parser.add_argument('--optimizer', type=str, default='poincare', choices=['standard', 'poincare'])
    parser.add_argument('--model_name', type=str, default='contrastive', choices=['isomap', 'mds', 'contrastive'])
    parser.add_argument('--latent_dist_fun', type=str, default='poincare', choices=['euclidean', 'poincare'])
    parser.add_argument('--distr', type=str, default='hypergaussian', choices=['gaussian', 'hypergaussian'])
    parser.add_argument('--distance_method', type=str, default='euclidean',
                        choices=['geo', 'graph', 'hamming', 'euclidean','similarity'])
    parser.add_argument('--n_samples', type=int, default=4000)
    parser.add_argument('--dim', type=int, default=768)
    parser.add_argument('--depth', type=int, default=5)  # Changed from bool to int
    parser.add_argument('--temperature', type=float, default=100)  # 10
    parser.add_argument('--n_neighbors', type=int, default=10)
    # args = argparse.Namespace()
    args = parser.parse_args()

    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print('Using GPU')
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1
        print('Using CPU')

    args.random_string = uuid.uuid4().hex
    dataset_name = args.dataset_name
    representation_name = args.representation_name
    num_epochs = args.num_epochs
    normalize = args.normalize
    latent_dim = args.latent_dim
    lr = args.lr
    seed = args.seed
    base_dir = args.base_dir
    optimizer = args.optimizer
    model_name = args.model_name
    latent_dist_fun = args.latent_dist_fun
    distr = args.distr
    distance_method = args.distance_method
    n_neighbors = args.n_neighbors
    temperature = args.temperature

    ### Overwrite the batchsize ###
    depth = args.depth
    # args.batch_size = 2 ** args.depth - 1  # to get full batch
    batch_size = args.batch_size
    set_seeds(seed)

    if distance_method == 'similarity' and dataset_name not in ['ravia']:
        raise ValueError('Similarity distance method can only be used with Ravia dataset')

    if dataset_name == 'tree':
        embeddings, labels = get_tree_data(depth)
        labels = torch.tensor(labels)
        ## binary_tree is a dataset of binary sequences.
        ## The root of the tree is the node 0: binary_tree[0]
        ## groundtruth distance from node i to the root of the tree (i.e. shortest path distance from node i to the root): hamming_distance(binary_tree[0], binary_tree[i])
        ## For visualizations, one can color a node by its groundtruth distance to the tree.
    elif dataset_name == 'random':
        #todo do we need this?
        embeddings = torch.randn(n_samples, dim)
    elif dataset_name in ['gslf', 'keller' , 'sagar']:
        input_embeddings = f'embeddings/{representation_name}/{dataset_name}_{representation_name}_embeddings_13_Apr17.csv'
        embeddings, labels,subjects,CIDs = read_embeddings(base_dir, select_descriptors(dataset_name), input_embeddings,
                                             grand_avg=False)


    elif dataset_name in ['ravia']:
        data_dist_matrix, intensity_labels = prepare_ravia_or_snitz(
            dataset=f'embeddings/{representation_name}/curated_ravia2020_behavior_similarity_intensity.csv',
            base_path=base_dir)
        #define dummy embeddings and labels
        embeddings = torch.randn(data_dist_matrix.shape[0], 10)
        labels = torch.randn(data_dist_matrix.shape[0], 10)
        CIDs = None
        subjects = np.zeros(data_dist_matrix.shape[0], dtype=int)
        data_dist_matrix = torch.tensor(data_dist_matrix)
        batch_size= 28
    else:
        raise ValueError('Dataset not recognized')

    dataset = OdorMonoDataset(embeddings, labels, transform=None)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


    #select a subset of subjects if needed
    # embeddings, labels, CIDs, subjects = select_subjects(subjects, embeddings, labels, CIDs,subjects.unique(),subject_id=None,n_subject=3)
    # embeddings, labels, CIDs, subjects = select_subjects(subjects, embeddings, labels, CIDs,subjects.unique(),subject_id=3,n_subject=None)

    #select a subset of molecules based on the labels if needed
    embeddings, labels, CIDs, subjects = select_molecules(subjects, embeddings, labels, CIDs,selected_labels=["bergamot", "grapefruit", "lemon", "orange", "black currant", "raspberry", "strawberry", "banana", "coconut", "pineapple","tropical","berry", "citrus","fruity"] )


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
                            euclidean_distance if latent_dist_fun == 'euclidean' else poincare_distance, distr=distr)
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
        total_loss = 0
        for idx, batch, label in data_loader:
            batch = batch.to(args.device)
            label = label.to(args.device)
            if normalize:
                model.normalize()
            else:
                model.normalize(normalization=False)
            if distance_method == 'graph':
                data_nn_matrix = knn_graph_weighted_adjacency_matrix(batch, n_neighbors=3, metric='minkowski')
                data_dist_matrix = (data_nn_matrix > 0).astype(int)
                data_dist_matrix = torch.tensor(data_dist_matrix)
            elif distance_method == 'geo':
                data_dist_matrix = knn_geodesic_distance_matrix(batch)
                if model_name == 'contrastive':
                    data_binary_dist_matrix = (data_dist_matrix <= 1.01).to(torch.int)
            elif distance_method == 'hamming':
                #todo: check and fix this
                data_dist_matrix = hamming_distance_matrix(batch)
                if model_name == 'contrastive':
                    data_binary_dist_matrix = (data_dist_matrix <= 1.01).astype(int)
                    data_binary_dist_matrix = torch.tensor(data_binary_dist_matrix)
                data_dist_matrix = torch.tensor(data_dist_matrix)
            elif distance_method == 'euclidean':
                data_dist_matrix = scipy.spatial.distance.cdist(embeddings, embeddings, metric='euclidean')
                data_dist_matrix = torch.tensor(data_dist_matrix)
                if model_name == 'contrastive':
                    data_binary_dist_matrix = kneighbors_graph(data_dist_matrix, n_neighbors=n_neighbors,
                                                               mode='connectivity', include_self=False).toarray()
            elif distance_method == 'similarity':
                if model_name == 'contrastive':
                    data_binary_dist_matrix = kneighbors_graph(data_dist_matrix, n_neighbors=n_neighbors,
                                                             mode='connectivity', include_self=False).toarray()
                idx = list(range(data_dist_matrix.shape[0]))
                ent_array = np.zeros(data_dist_matrix.shape[0])
                for i, arr in enumerate(intensity_labels):
                    arr = np.array(arr)
                    entropy = arr / arr.sum()
                    c = -(entropy * np.log(entropy)).sum(-1)
                    ent_array[i] = c


            else:
                data_dist_matrix = distance_matrix(batch, euclidean_distance)

            optimizer.zero_grad()


            #todo: how to make this case general so it works when model_name is isomap or mds (data_binary_dist_matrix?)
            loss = model.loss_fun(data_dist_matrix, idx, data_binary_dist_matrix, temperature)
            loss.backward()
            optimizer.step(idx)
            total_loss += loss.item()


        print(f'Epoch {i}, loss: {total_loss / len(data_loader):.3f}')
        losses.append(total_loss / len(data_loader))

        if i % 10 == 0:  # 1000
            # save_embeddings(i, args, model.embeddings.detach().cpu().numpy(), losses=losses,
            #                 losses_neg=model.losses_neg if model_name == 'contrastive' else [],
            #                 losses_pos=model.losses_pos if model_name == 'contrastive' else [])
            # scatterplot_2d(i, model.embeddings.detach().cpu().numpy(), dataset.labels.detach().cpu().numpy(), CIDs,labels, subjects=subjects,
            scatterplot_2d(i, model.embeddings.detach().cpu().numpy(),
                                          ent_array, CIDs, labels, subjects=subjects,
                           color_by='color', shape_by='none',
                           save=True, args=args,
                           losses=losses, losses_neg=model.losses_neg if model_name == 'contrastive' else [],
                           losses_pos=model.losses_pos if model_name == 'contrastive' else [])

