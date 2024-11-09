import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import pandas as pd
import ast
import numpy as np
from constants import *
import random
def set_seeds(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def prepare_dataset(ds):
    if 'y' in ds.columns:
        ds['y'] = ds['y'].apply(ast.literal_eval)
    ds['embeddings'] = ds['embeddings'].apply(ast.literal_eval)
    # ds['CID'] = ds['CID'].apply(ast.literal_eval)
    # if 'subject' in ds.columns:
    #     ds['subject'] = ds['subject'].apply(ast.literal_eval)
    return ds


def grand_average(df, descriptors):
    df_groupbyCID = df.groupby('CID')[descriptors].mean().reset_index()

    df_groupbyCID['y'] = df_groupbyCID.loc[:, '0.1':descriptors[-1]].values.tolist()
    df_embeddings = df.drop_duplicates(subset=['CID'])
    df_embeddings = df_embeddings[['CID', 'embeddings']]
    df_groupbyCID = pd.merge(df_groupbyCID, df_embeddings, on='CID', how='left')
    return df_groupbyCID


def average_per_subject( df, descriptors):
    df_groupbyCID = df.groupby(['CID', 'subject'])[descriptors].mean().reset_index()
    df_groupbyCID['y'] = df_groupbyCID.loc[:, '0.1':descriptors[-1]].values.tolist()
    df_embeddings = df.drop_duplicates(subset=['CID'])
    df_embeddings = df_embeddings[['CID', 'embeddings']]
    df_groupbyCID = pd.merge(df_groupbyCID, df_embeddings, on='CID', how='left')
    return df_groupbyCID



def prepare_similarity_mols_mix_on_representations(base_dir,perceptual_similarity_csv, embeddings_csv,
                                                   mixing_type='sum', sep=';',
                                                   start='0', end='255'):

    df, df_similarity_mean, df_similarity_mean_pivoted = prepare_similarity_ds(base_dir,
        perceptual_similarity_csv)

    df_mols = create_pairs(df_similarity_mean)

    df_embeddings = pd.read_csv(base_dir+embeddings_csv)[['embeddings', 'CID']]
    df_embeddings['embeddings'] = df_embeddings['embeddings'].apply(lambda x: np.array(eval(x)))

    if mixing_type == 'sum':

        df_mols['embeddings'] = df_mols['CID'].apply(
            lambda x: sum_embeddings(list(map(int, x.split(sep))), df_embeddings))
    elif mixing_type == 'average':
        df_mols['embeddings'] = df_mols['CID'].apply(
            lambda x: average_embeddings(list(map(int, x.split(sep))), df_embeddings))

    df_mols_embeddings_original = [
        torch.from_numpy(np.asarray(df_mols['Stimulus Embedding Sum'].values.tolist()))]

    df_ravia_mols_embeddings, df_ravia_mols_embeddings_zscored =prepare_mols_helper_mixture(
        df_mols_embeddings_original, df_mols, start, end)

    return df_mols, df_ravia_mols_embeddings, df_ravia_mols_embeddings_zscored



def prepare_similarity_ds(base_dir, perceptual_similarity_csv):
    """
    Prepare the similarity dataset for the alignment task
    :param base_path:   (str) path to the base directory where the datasets are stored
    :return:         (tuple) a tuple containing the original  dataset, the mean  dataset, and the pivoted mean  dataset
    """

    input_file = base_dir + perceptual_similarity_csv
    df_original = pd.read_csv(input_file)
    df = df_original.copy()

    features = ['CID Stimulus 1', 'CID Stimulus 2', 'Stimulus 1-IsomericSMILES', 'Stimulus 2-IsomericSMILES',
                'Stimulus 1-nonStereoSMILES', 'Stimulus 2-nonStereoSMILES', 'RatedSimilarity']
    agg_functions = {}
    features_all = features
    df = df.reindex(columns=features_all)

    agg_functions['RatedSimilarity'] = 'mean'

    df = df[features_all]
    df_copy = df.copy()
    df_copy = df_copy.rename(columns={'Stimulus 1-IsomericSMILES': 'Stimulus 2-IsomericSMILES',
                                      'Stimulus 2-IsomericSMILES': 'Stimulus 1-IsomericSMILES',
                                      'CID Stimulus 1': 'CID Stimulus 2',
                                      'CID Stimulus 2': 'CID Stimulus 1',
                                      'Stimulus 1-nonStereoSMILES': 'Stimulus 2-nonStereoSMILES',
                                      'Stimulus 2-nonStereoSMILES': 'Stimulus 1-nonStereoSMILES'})
    df_copy['RatedSimilarity'] = np.nan
    df_concatenated = pd.concat([df, df_copy], ignore_index=True, axis=0).reset_index(drop=True)
    df = df_concatenated.drop_duplicates(
        ['CID Stimulus 1', 'CID Stimulus 2', 'Stimulus 1-IsomericSMILES', 'Stimulus 2-IsomericSMILES',
         'Stimulus 1-nonStereoSMILES', 'Stimulus 2-nonStereoSMILES'])

    df_ravia_mean = df.groupby(
        ['CID Stimulus 1', 'CID Stimulus 2', 'Stimulus 1-IsomericSMILES', 'Stimulus 2-IsomericSMILES',
         'Stimulus 1-nonStereoSMILES', 'Stimulus 2-nonStereoSMILES']).agg(agg_functions).reset_index()

    df_mean_pivoted = df_ravia_mean.pivot(index='CID Stimulus 1', columns='CID Stimulus 2',
                                          values='RatedSimilarity')

    df_mean_pivoted = df_mean_pivoted.reindex(sorted(df_mean_pivoted.columns), axis=1)
    df_mean_pivoted = df_mean_pivoted.sort_index(ascending=True)

    return df_original, df_ravia_mean, df_mean_pivoted


def create_pairs( df_similarity_mean):
    df_mean_mols1 = df_similarity_mean[
        ['Stimulus 1-IsomericSMILES', 'Stimulus 1-nonStereoSMILES',
         'CID Stimulus 1']].drop_duplicates().reset_index(
        drop=True)
    df_mean_mols2 = df_similarity_mean[
        ['Stimulus 2-IsomericSMILES', 'Stimulus 2-nonStereoSMILES',
         'CID Stimulus 2']].drop_duplicates().reset_index(
        drop=True).rename(columns={'Stimulus 2-nonStereoSMILES': 'Stimulus 1-nonStereoSMILES',
                                   'Stimulus 2-IsomericSMILES': 'Stimulus 1-IsomericSMILES',
                                   'CID Stimulus 2': 'CID Stimulus 1'})
    df_mols = pd.concat([df_mean_mols1, df_mean_mols2], ignore_index=True, axis=0).reset_index(
        drop=True)
    df_mols = df_mols.drop_duplicates().reset_index(drop=True)
    df_mols = df_mols.rename(
        columns={'Stimulus 1-IsomericSMILES': 'IsomericSMILES', 'Stimulus 1-nonStereoSMILES': 'nonStereoSMILES',
                 'CID Stimulus 1': 'CID'})
    return df_mols


def postproce_molembeddings( embeddings, index):
    molecules_embeddings_penultimate = torch.cat(embeddings)
    columns_size = int(molecules_embeddings_penultimate.size()[1])
    if index.ndim > 1:
        molecules_embeddings_penultimate = torch.cat(
            (torch.from_numpy(index.to_numpy()), molecules_embeddings_penultimate), dim=1)
        df_molecules_embeddings = pd.DataFrame(molecules_embeddings_penultimate,
                                               columns=['CID', 'subject'] + [str(i) for i in range(columns_size)])
        df_molecules_embeddings = df_molecules_embeddings.set_index(['CID', 'subject'])
        df_molecules_embeddings['Combined'] = df_molecules_embeddings.loc[:,
                                              '0':str(columns_size - 1)].values.tolist()
    else:
        df_molecules_embeddings = pd.DataFrame(molecules_embeddings_penultimate,
                                               columns=[str(i) for i in range(columns_size)])
        df_molecules_embeddings['CID'] = index
        df_molecules_embeddings = df_molecules_embeddings.set_index(['CID'])
        df_molecules_embeddings['Combined'] = df_molecules_embeddings.loc[:,
                                              '0':str(columns_size - 1)].values.tolist()
    df_molecules_embeddings = df_molecules_embeddings.reset_index()
    return df_molecules_embeddings


def prepare_mols_helper_mixture( df_mols_embeddings_original, df_mols, start, end, mol_type="nonStereoSMILES",
                                index="CID"):
    df_mols_embeddings = postproce_molembeddings(df_mols_embeddings_original, df_mols[index])

    # z-score embeddings
    df_mols_embeddings_zscored = df_mols_embeddings.copy()
    scaled_features = StandardScaler().fit_transform(df_mols_embeddings_zscored.loc[:, start:end].values.tolist())
    df_mols_embeddings_zscored.loc[:, start:end] = pd.DataFrame(scaled_features,
                                                                index=df_mols_embeddings_zscored.index,
                                                                columns=[str(i) for i in range(int(end) + 1)])
    df_mols_embeddings_zscored['Combined'] = df_mols_embeddings_zscored.loc[:, start:end].values.tolist()

    return  df_mols_embeddings, df_mols_embeddings_zscored,


def sum_embeddings(cid_list, df_embeddings):
    embedding_sum = np.zeros(len(df_embeddings.iloc[0]['embeddings']))
    for cid in cid_list:
        if cid in df_embeddings['CID'].values:
            embedding_sum += df_embeddings.loc[df_embeddings['CID'] == cid, 'embeddings'].values[0]
    return embedding_sum


def average_embeddings(cid_list, df_embeddings):
    embedding_sum = np.zeros(len(df_embeddings.iloc[0]['embeddings']))
    n_cid = 0
    for cid in cid_list:
        if cid in df_embeddings['CID'].values:
            embedding_sum += df_embeddings.loc[df_embeddings['CID'] == cid, 'embeddings'].values[0]
            n_cid += 1
    return embedding_sum / n_cid


def prepare_goodscentleffignwell_mols(base_dir):
    goodscentleffignwell_input_file = base_dir+'mols_datasets/curated_GS_LF_merged_4983.csv' # or new downloaded file path
    df_goodscentleffignwell=pd.read_csv(goodscentleffignwell_input_file)
    df_goodscentleffignwell.index.names = ['CID']
    df_goodscentleffignwell= df_goodscentleffignwell.reset_index()
    df_goodscentleffignwell['y'] = df_goodscentleffignwell.loc[:,'alcoholic':'woody'].values.tolist()
    return df_goodscentleffignwell

def select_descriptors(dataset_name):
    if dataset_name=='sagar':
        return sagar_descriptors
    elif dataset_name=='keller':
        return keller_descriptors
    elif dataset_name=='gslf':
        return gs_lf_tasks
    else:
        return None

def read_embeddings(base_dir, descriptors, embeddings_perception_csv, grand_avg,avg_per_subject=False):

    ds = pd.read_csv(base_dir + embeddings_perception_csv)
    ds = prepare_dataset(ds)
    if avg_per_subject:
        ds = average_per_subject(ds, descriptors)
    elif grand_avg:
        ds = grand_average(ds, descriptors)
    else:
        pass
    embeddings = ds['embeddings']
    if 'y' in ds.columns:
        labels = ds['y']
    else:
        labels = torch.from_numpy(np.ones(len(embeddings.tolist()),dtype=int))


    if 'subject' in ds.columns:
        subjects = ds['subject']


        subjects = torch.from_numpy(np.array(subjects.tolist()))
    else:

        subjects = torch.from_numpy(np.ones(len(labels),dtype=int))
    CIDs = ds['CID']
    embeddings = torch.from_numpy(np.array(embeddings.tolist()))
    labels = torch.from_numpy(np.array(labels.tolist()))

    CIDs = torch.from_numpy(np.array(CIDs.tolist()))
    return embeddings,labels,subjects,CIDs


def prepare_ravia_or_snitz(dataset, base_path='/local_storage/datasets/farzaneh/alignment_olfaction_datasets/'):
    # generate docstrings for this function with a brief description of the function and the parameters and return values
    """
    Prepare the similarity dataset for the alignment task
    :param base_path:   (str) path to the base directory where the datasets are stored
    :return:         (tuple) a tuple containing the original  dataset, the mean  dataset, and the pivoted mean  dataset
    """

    def make_symmetric(matrix):
        return matrix.combine_first(matrix.T)

    input_file = base_path + dataset
    df_ravia_original = pd.read_csv(input_file)
    df_ravia = df_ravia_original.copy()

    features = ['CID Stimulus 1', 'CID Stimulus 2', 'Stimulus 1-IsomericSMILES', 'Stimulus 2-IsomericSMILES',
                'Stimulus 1-nonStereoSMILES', 'Stimulus 2-nonStereoSMILES', 'RatedSimilarity', 'Intensity 1',
                'Intensity 2']
    agg_functions = {}
    features_all = features
    df_ravia = df_ravia.reindex(columns=features_all)

    agg_functions['RatedSimilarity'] = 'mean'

    df_ravia = df_ravia[features_all]
    df_ravia_copy = df_ravia.copy()
    df_ravia_copy = df_ravia_copy.rename(columns={'Stimulus 1-IsomericSMILES': 'Stimulus 2-IsomericSMILES',
                                                  'Stimulus 2-IsomericSMILES': 'Stimulus 1-IsomericSMILES',
                                                  'CID Stimulus 1': 'CID Stimulus 2',
                                                  'CID Stimulus 2': 'CID Stimulus 1',
                                                  'Stimulus 1-nonStereoSMILES': 'Stimulus 2-nonStereoSMILES',
                                                  'Stimulus 2-nonStereoSMILES': 'Stimulus 1-nonStereoSMILES',
                                                  'Intensity 1': 'Intensity 2',
                                                  'Intensity 2': 'Intensity 1'
                                                  })
    df_ravia_copy['RatedSimilarity'] = np.nan
    df_ravia_concatenated = pd.concat([df_ravia, df_ravia_copy], ignore_index=True, axis=0).reset_index(drop=True)
    df_ravia = df_ravia_concatenated.drop_duplicates(
        ['CID Stimulus 1', 'CID Stimulus 2', 'Stimulus 1-IsomericSMILES', 'Stimulus 2-IsomericSMILES',
         'Stimulus 1-nonStereoSMILES', 'Stimulus 2-nonStereoSMILES', 'Intensity 1', 'Intensity 2'])

    df_ravia_mean = df_ravia.groupby(
        ['CID Stimulus 1', 'CID Stimulus 2', 'Stimulus 1-IsomericSMILES', 'Stimulus 2-IsomericSMILES',
         'Stimulus 1-nonStereoSMILES', 'Stimulus 2-nonStereoSMILES', 'Intensity 1', 'Intensity 2']).agg(
        agg_functions).reset_index()

    df_ravia_mean_pivoted = df_ravia_mean.pivot(index='CID Stimulus 1', columns='CID Stimulus 2',
                                                values='RatedSimilarity')

    df_ravia_mean_pivoted = df_ravia_mean_pivoted.reindex(sorted(df_ravia_mean_pivoted.columns), axis=1)
    df_ravia_mean_pivoted = df_ravia_mean_pivoted.sort_index(ascending=True)
    df_ravia_mean_pivoted = make_symmetric(df_ravia_mean_pivoted)

    # fill all nans with -1
    df_ravia_mean_pivoted = df_ravia_mean_pivoted.fillna(1000)

    # create a table with the intensity and CID columns for each stimulus
    df_ravia = df_ravia.reindex(columns=['CID Stimulus 1', 'CID Stimulus 2', 'Intensity 1', 'Intensity 2'])
    df_ravia_intensity = df_ravia[['CID Stimulus 1', 'CID Stimulus 2', 'Intensity 1', 'Intensity 2']]

    # stack CID Stimulus 1 and CID Stimulus 2 columns as a CID column and  Intensity 1 and Intensity 2 columns as an Intensity column
    df_ravia_intensity = pd.concat([df_ravia_intensity[['CID Stimulus 1', 'Intensity 1']].rename(
        columns={'CID Stimulus 1': 'CID', 'Intensity 1': 'Intensity'}),
                                    df_ravia_intensity[['CID Stimulus 2', 'Intensity 2']].rename(
                                        columns={'CID Stimulus 2': 'CID', 'Intensity 2': 'Intensity'})],
                                   ignore_index=True)
    df_ravia_intensity = df_ravia_intensity.drop_duplicates()
    df_ravia_intensity['Intensity'] = df_ravia_intensity['Intensity'].apply(lambda x: x.replace(';', ','))

    df_ravia_intensity['Intensity'] = df_ravia_intensity['Intensity'].apply(ast.literal_eval)
    df_ravia_intensity = df_ravia_intensity['Intensity'].to_numpy()

    return  df_ravia_mean_pivoted.to_numpy(), df_ravia_intensity


def prepare_ravia_similarity_mols_mix_on_representations(input_file_embeddings, df_ravia_similarity_mean,
                                                         modeldeepchem_gslf=None, mixing_type='sum', sep=';', start='0',
                                                         end='255'):
    df_ravia_mols = create_pairs(df_ravia_similarity_mean)

    df_embeddigs = pd.read_csv(input_file_embeddings)[['embeddings', 'CID']]
    df_embeddigs['embeddings'] = df_embeddigs['embeddings'].apply(lambda x: np.array(eval(x)))

    if mixing_type == 'sum':

        df_ravia_mols['Stimulus Embedding Sum'] = df_ravia_mols['CID'].apply(
            lambda x: sum_embeddings(list(map(int, x.split(sep))), df_embeddigs))
    elif mixing_type == 'average':
        df_ravia_mols['Stimulus Embedding Sum'] = df_ravia_mols['CID'].apply(
            lambda x: average_embeddings(list(map(int, x.split(sep))), df_embeddigs))

    df_mols_embeddings_original = [
        torch.from_numpy(np.asarray(df_ravia_mols['Stimulus Embedding Sum'].values.tolist()))]

    df_ravia_mols_embeddings_original, df_ravia_mols_embeddings, df_ravia_mols_embeddings_zscored = prepare_mols_helper_mixture(
        df_mols_embeddings_original, df_ravia_mols, start, end, modeldeepchem_gslf)



    return df_ravia_mols, df_ravia_mols_embeddings_original, df_ravia_mols_embeddings, df_ravia_mols_embeddings_zscored


def select_subjects(subjects, embeddings, labels, CIDs, subjects_ids, subject_id=None, n_subject=None):
    if subject_id is not None and n_subject is not None:
        raise ValueError('You can only provide either subject_id or n_subject')
    elif subject_id is not None:
        if type(subject_id) == int:

            embeddings = embeddings[subjects == subject_id]
            labels = labels[subjects == subject_id]
            CIDs = CIDs[subjects == subject_id]
            subjects = subjects[subjects == subject_id]
        else:
            embeddings = embeddings[np.isin(subjects, subject_id)]
            labels = labels[np.isin(subjects, subject_id)]
            CIDs = CIDs[np.isin(subjects, subject_id)]
            subjects = subjects[np.isin(subjects, subject_id)]

    elif n_subject is not None:
        # randomize subjects_id and select n_subjects
        subjects_ids = np.random.permutation(subjects_ids)
        n_subjects = subjects_ids[:n_subject]
        embeddings = embeddings[np.isin(subjects, n_subjects)]
        labels = labels[np.isin(subjects, n_subjects)]
        CIDs = CIDs[np.isin(subjects, n_subjects)]
        subjects = subjects[np.isin(subjects, n_subjects)]
    else:
        raise ValueError('Please provide either subject_id or n_subject')

    return embeddings, labels, CIDs, subjects


def select_molecules(subjects, embeddings, labels, CIDs, selected_labels):



    #convert torch tensors to numpy arrays
    # labels = labels.numpy()
    arrs = []
    # for ar in labels:
    #     arrs.append(np.asarray(ar))
    # labels = np.asarray(arrs)

    indices = [gs_lf_tasks.index(label) for label in selected_labels]
    filtered_labels = labels[torch.any(labels[:, indices], axis=1)]
    filtered_subjects = subjects[torch.any(labels[:, indices], axis=1)]
    filtered_embeddings = embeddings[torch.any(labels[:, indices], axis=1)]
    filtered_CIDs = CIDs[torch.any(labels[:, indices], axis=1)]

    return filtered_embeddings, filtered_labels, filtered_CIDs, filtered_subjects