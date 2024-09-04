#create a custom pytorch dataset for the odor dataset
import torch
from torch.utils.data import Dataset
import pandas as pd
import ast
class OdorMonoDataset(Dataset):
    def __init__(self,base_dir, csv_file, transform=None,grand_average=False,average_per_subject=False,descriptors=None):
        self.base_dir = base_dir
        self.ds = pd.read_csv(base_dir+csv_file)
        self.ds = self.prepare_dataset(self.ds)
        if grand_average:
            self.ds = self.grand_average(self.ds, descriptors)
        elif average_per_subject:
            self.ds = self.average_per_subject(self.ds, descriptors)
        else:
            pass
        self.labels = self.ds['y']
        self.data = self.ds['embeddings']
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, label

    def prepare_dataset(self,ds):
        ds['y'] = ds['y'].apply(ast.literal_eval)
        ds['embeddings'] = ds['embeddings'].apply(ast.literal_eval)
        return ds

    def grand_average(self,df, descriptors):
        df_groupbyCID = df.groupby('CID')[descriptors].mean().reset_index()

        df_groupbyCID['y'] = df_groupbyCID.loc[:, '0.1':descriptors[-1]].values.tolist()
        df_embeddings = df.drop_duplicates(subset=['CID'])
        df_embeddings = df_embeddings[['CID', 'embeddings']]
        df_groupbyCID = pd.merge(df_groupbyCID, df_embeddings, on='CID', how='left')
        return df_groupbyCID

    def average_per_subject(self, df, descriptors):
        df_groupbyCID = df.groupby(['CID', 'subject'])[descriptors].mean().reset_index()
        df_groupbyCID['y'] = df_groupbyCID.loc[:, '0.1':descriptors[-1]].values.tolist()
        df_embeddings = df.drop_duplicates(subset=['CID'])
        df_embeddings = df_embeddings[['CID', 'embeddings']]
        df_groupbyCID = pd.merge(df_groupbyCID, df_embeddings, on='CID', how='left')
        return df_groupbyCID

