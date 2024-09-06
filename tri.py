from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import dijkstra
from torch.utils.data import DataLoader
import random
from constants import *
from methods import *
from optimizers import *
from utils.visulization import *
from OdorDataset import OdorMonoDataset
from utils.helpers import *
import argparse


# latent_dim = 2
# lr = 0.1
# num_epochs = 100000
# normalize = False
# geodesic = False
# min_dist = 1.
# dataset_name='gslf'
# model_name = 'molformer'
# batch_size =10

def set_seeds(seed):
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def select_descriptors(dataset_name):
    if dataset_name == 'sagar':
        return sagar_descriptors
    elif dataset_name == 'keller':
        return keller_descriptors
    else:
        return None

    # del dataset


def geo_distance(data):
    #     truncated_matrix = torch.where(data_dist_matrix < min_dist, data_dist_matrix, torch.inf)
    #     data_dist_matrix = dijkstra(truncated_matrix.detach().cpu().numpy())
    #     data_dist_matrix = torch.FloatTensor(data_dist_matrix)
    #     data_dist_matrix = torch.where(data_dist_matrix == torch.inf, 1000 * torch.ones_like(data_dist_matrix), data_dist_matrix)

    data_nn_matrix = kneighbors_graph(data, 3, mode='distance', include_self=False)
    data_nn_matrix = data_nn_matrix.toarray()
    data_dist_matrix = dijkstra(data_nn_matrix)
    data_dist_matrix = torch.FloatTensor(data_dist_matrix)
    data_dist_matrix = torch.where(data_dist_matrix == torch.inf, 1000 * torch.ones_like(data_dist_matrix),
                                   data_dist_matrix)
    return data_dist_matrix


# IsoMap-style geodesic distance for data


# torch.manual_seed(42)
# torch.cuda.empty_cache()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print('Using device: ', device)


# model = MDS(data.shape[0], latent_dim, Poincare)
# print(len(dataset))


if __name__ == "__main__":
    # add arguments
    parser = argparse.ArgumentParser('Hyperbolic Smell')
    parser.add_argument('--dataset_name', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='molformer')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--num_epochs', type=int, default=100000)
    parser.add_argument('--normalize', type=bool, default=False)
    parser.add_argument('--geodesic', type=bool, default=False)
    parser.add_argument('--min_dist', type=float, default=1.)
    parser.add_argument('--latent_dim', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--base_dir', type=str,
                        default='../../../T5 EVO/alignment_olfaction_datasets/curated_datasets/')
    parser.add_argument('--optimizer', type=str, default='standard', choices=['standard', 'poincare'])
    parser.add_argument('--model', type=str, default='isomap', choices=['isomap', 'mds', 'contrastive'])
    parser.add_argument('--latent_dist_fun', type=str, default='euclidean', choices=['euclidean', 'poincare'])

    args = parser.parse_args()
    dataset_name = args.dataset_name
    model_name = args.model_name
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    normalize = args.normalize
    geodesic = args.geodesic
    min_dist = args.min_dist
    latent_dim = args.latent_dim
    lr = args.lr
    seed = args.seed
    base_dir = args.base_dir
    optimizer = args.optimizer
    model = args.model
    latent_dist_fun = args.latent_dist_fun

    if dataset_name is not None:
        input_embeddings = f'embeddings/{model_name}/{dataset_name}_{model_name}_embeddings_13_Apr17.csv'

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

    if dataset_name == 'gslf':
        dataset = OdorMonoDataset(base_dir, input_embeddings, transform=None, grand_avg=False,
                                  descriptors=select_descriptors(dataset_name))

    # set_seeds(2025)

    #
    # model = Isomap(len(dataset), latent_dim, Euclidean)

    # optimizer = StandardOptim(model, lr=lr)
    if optimizer == 'standard':
        optimizer = StandardOptim(model, lr=lr)
    elif optimizer == 'poincare':
        optimizer = PoincareOptim(model, lr=lr)
    else:
        raise ValueError('Optimizer not recognized')

    for i in range(num_epochs):
        total_loss = 0
        if normalize:
            model.normalize()
        for idx, batch in data_loader:

            if geodesic:
                data_dist_matrix = geo_distance(batch)
            else:
                data_dist_matrix = dist_matrix(batch, Euclidean)

            optimizer.zero_grad()
            loss = model.loss_fun(data_dist_matrix, idx)
            loss.backward()
            optimizer.step(idx)
            total_loss += loss.item()

            # print('norms', vector_norm(model.embeddings, dim=-1).mean().item(), vector_norm(model.embeddings, dim=-1).max().item())

        if i % 10 == 0:
            print(f'Epoch {i}, loss: {total_loss / len(data_loader):.3f}')

    scatterplot_2d(model.embeddings.detach().cpu().numpy(), labels=None, title='Poincare Embeddings')

    size1 = (0.3, 0.28)
    size2 = (0.6, 1)
    size3 = (1, 0.35)

    plt.rcParams["font.size"] = 35
    df_gslf_mols = prepare_goodscentleffignwell_mols(base_dir)
    pom_frame(np.asarray(model.embeddings.detach().cpu().numpy().values.tolist()),
              np.asarray(df_gslf_mols.y.values.tolist()), "/kaggle/working/", gs_lf_tasks, "molformer", size1, size2,
              size3)


