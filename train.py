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
latent_dim = 2
lr = 0.01
num_epochs = 100000
normalize = False
geodesic = False
min_dist = 1.



dataset_name='random'
model_name = 'molformer'
batch_size = 10

def set_seeds(seed):

    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
def select_descriptors(dataset_name):
    if dataset_name=='sagar':
        return sagar_descriptors
    elif dataset_name=='keller':
        return keller_descriptors
    else:
        return None


set_seeds(2025)
base_dir = '../../../T5 EVO/alignment_olfaction_datasets/curated_datasets/'
input_embeddings = f'embeddings/{model_name}/{dataset_name}_{model_name}_embeddings_13_Apr17.csv'

dataset = OdorMonoDataset(base_dir, None, transform=None, grand_avg=False, descriptors=select_descriptors(dataset_name))
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,drop_last=False)


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
    data_dist_matrix = torch.where(data_dist_matrix == torch.inf, 1000 * torch.ones_like(data_dist_matrix),data_dist_matrix)
    return data_dist_matrix


#IsoMap-style geodesic distance for data


# torch.manual_seed(42)
# torch.cuda.empty_cache()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print('Using device: ', device)

    


#model = MDS(data.shape[0], latent_dim, Poincare)
# print(len(dataset))
model = Isomap(len(dataset), latent_dim, Euclidean)

#optimizer = StandardOptim(model, lr=lr)
optimizer = StandardOptim(model, lr=lr)

if __name__ == "__main__":
    for i in range(num_epochs):
        if normalize:
            model.normalize()
        for idx, batch in data_loader:

            if geodesic:
                data_dist_matrix = geo_distance(batch)
            else:
                data_dist_matrix = dist_matrix(batch, Euclidean)
            optimizer.zero_grad()
            loss = model.loss_fun(data_dist_matrix,idx)
            loss.backward()
            optimizer.step(idx)



            # print('norms', vector_norm(model.embeddings, dim=-1).mean().item(), vector_norm(model.embeddings, dim=-1).max().item())


        if i % 10 == 0:
            print(f'Epoch {i}, loss: {loss:.3f}')

    scatterplot_2d(model.embeddings.detach().cpu().numpy(), labels=None, title='Poincare Embeddings')



    size1 = (0.3, 0.28)
    size2 = (0.6, 1)
    size3 = (1, 0.35)

    plt.rcParams["font.size"] = 35
    df_gslf_mols = prepare_goodscentleffignwell_mols(base_dir)
    pom_frame(np.asarray(model.embeddings.detach().cpu().numpy().values.tolist()),
              np.asarray(df_gslf_mols.y.values.tolist()), "/kaggle/working/", gs_lf_tasks, "molformer", size1, size2,
              size3)


