import argparse
from OdorDataset import OdorMonoDataset
from utils.helpers import *
from methods import *
from optimizers import *
from torch.utils.data import DataLoader


if __name__ == "__main__":
    # add arguments
    parser = argparse.ArgumentParser('Hyperbolic Smell')
    parser.add_argument('--dataset_name', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='molformer')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--num_epochs', type=int, default=100000)
    # parser.add_argument('--geodesic', type=bool, default=False)
    parser.add_argument('--min_dist', type=float, default=1.)
    parser.add_argument('--latent_dim', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--base_dir', type=str,
                        default='../../../T5 EVO/alignment_olfaction_datasets/curated_datasets/')
    parser.add_argument('--normalize', type=bool, default=False)
    parser.add_argument('--optimizer', type=str, default='standard', choices=['standard', 'poincare'])
    parser.add_argument('--model', type=str, default='contrastive', choices=['isomap', 'mds', 'contrastive'])
    parser.add_argument('--latent_dist_fun', type=str, default='euclidean', choices=['euclidean', 'poincare'])
    parser.add_argument('--distance_method', type=str, default='graph', choices=['geo', 'graph'])

    args = parser.parse_args()
    dataset_name = args.dataset_name
    model_name = args.model_name
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    normalize = args.normalize
    # geodesic = args.geodesic
    min_dist = args.min_dist
    latent_dim = args.latent_dim
    lr = args.lr
    seed = args.seed
    base_dir = args.base_dir
    optimizer = args.optimizer
    model = args.model
    latent_dist_fun = args.latent_dist_fun
    distance_method = args.distance_method

    if dataset_name is not None:
        input_embeddings = f'embeddings/{model_name}/{dataset_name}_{model_name}_embeddings_13_Apr17.csv'
    else:
        input_embeddings = None
    dataset = OdorMonoDataset(base_dir, input_embeddings, transform=None, grand_avg=False,
                                  descriptors=select_descriptors(dataset_name))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    if latent_dist_fun != 'euclidean' and latent_dist_fun != 'poincare':
        raise ValueError('Latent distance function not recognized')
    if model == 'isomap':
        model = Isomap(len(dataset), latent_dim, Euclidean if latent_dist_fun == 'euclidean' else Poincare)
    elif model == 'mds':
        model = MDS(len(dataset), latent_dim, Euclidean if latent_dist_fun == 'euclidean' else Poincare)
    elif model == 'contrastive':
        model = Contrastive(len(dataset), latent_dim, Euclidean if latent_dist_fun == 'euclidean' else Poincare)
    else:
        raise ValueError('Model not recognized')

    if optimizer == 'standard':
        optimizer = StandardOptim(model, lr=lr)
    elif optimizer == 'poincare':
        optimizer = PoincareOptim(model, lr=lr)
    else:
        raise ValueError('Optimizer not recognized')



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