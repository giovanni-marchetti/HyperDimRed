#create a custom pytorch dataset for the odor dataset
from torch.utils.data import Dataset
from sklearn.metrics.pairwise import euclidean_distances


from utils.helpers import *
class OdorMonoDataset(Dataset):
    def __init__(self, base_dir, embeddings_perception_csv, transform=None, grand_avg=False, avg_per_subject=False, descriptors=None):
        self.base_dir = base_dir
        self.ds = pd.read_csv(base_dir + embeddings_perception_csv)
        self.ds = prepare_dataset(self.ds)
        if grand_avg:
            self.ds = grand_average(self.ds, descriptors)
        elif avg_per_subject:
            self.ds = average_per_subject(self.ds, descriptors)
        else:
            pass
        self.labels = self.ds['y']
        self.embeddings = self.ds['embeddings']
        self.embeddings =torch.from_numpy(np.array(self.embeddings.tolist()))



        self.transform = transform

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        sample = self.embeddings[idx]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, label





class OdorMixDataset(Dataset):
    def __init__(self,base_dir, perception_csv, embeddings_csv, transform=None, grand_average=False, average_per_subject=False, descriptors=None):
        df_mols, df_mols_embeddings, df_mols_embeddings_zscored = prepare_similarity_mols_mix_on_representations(base_dir, perception_csv,
                                                       embeddings_csv,
                                                       mixing_type='average', sep=';',
                                                       start='0', end='255')

        self.base_dir = base_dir
        if grand_average:
            self.ds = self.grand_average(self.ds, descriptors)
        elif average_per_subject:
            self.ds = self.average_per_subject(self.ds, descriptors)
        else:
            pass
        self.labels = self.ds['y']
        self.data = df_mols['embeddings']
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, label





