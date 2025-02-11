import argparse
import numpy as np
import torch
# import wandb
from SyntheticTreeDataset import *
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
from torch.optim import Adam

from distances import (
    distance_matrix,
    euclidean_distance,
    poincare_distance,
    knn_geodesic_distance_matrix,
    knn_graph_weighted_adjacency_matrix,
    # hamming_distance_matrix
)

from sklearn.manifold import TSNE

from sklearn.decomposition import PCA


### If using Jupyter Notebook:###
# import sys
# if 'ipykernel_launcher' in sys.argv[0]:
#     sys.argv = sys.argv[:1]
###


# def select_subjects(subjects, embeddings, labels, CIDs,subjects_ids,subject_id=None,n_subject=None):
#     if subject_id is not None and n_subject is not None:
#         raise ValueError('You can only provide either subject_id or n_subject')
#     elif subject_id is not None:
#         if type(subject_id) == int:


#             embeddings = embeddings[subjects == subject_id]
#             labels = labels[subjects == subject_id]
#             CIDs = CIDs[subjects == subject_id]
#             subjects = subjects[subjects == subject_id]
#         else:
#             embeddings = embeddings[np.isin(subjects, subject_id)]
#             labels = labels[np.isin(subjects, subject_id)]
#             CIDs = CIDs[np.isin(subjects, subject_id)]
#             subjects = subjects[np.isin(subjects, subject_id)]

#     elif n_subject is not None:
#         # randomize subjects_id and select n_subjects
#         subjects_ids = np.random.permutation(subjects_ids)
#         print('subjects_ids', subjects_ids)
#         n_subjects = subjects_ids[:n_subject]
#         embeddings = embeddings[np.isin(subjects, n_subjects)]
#         labels = labels[np.isin(subjects, n_subjects)]
#         CIDs = CIDs[np.isin(subjects, n_subjects)]
#         subjects = subjects[np.isin(subjects, n_subjects)]
#     else:
#         raise ValueError('Please provide either subject_id or n_subject')




if __name__ == "__main__":

    parser = argparse.ArgumentParser('Hyperbolic Smell')
    parser.add_argument('--data_type', type=str, default='labels' , choices={"representation","labels"}) #label or batch
    parser.add_argument('--representation_name', type=str, default='pom', choices={"molformer","pom"})
    parser.add_argument('--batch_size', type=int, default=195) #195
    parser.add_argument('--num_epochs', type=int, default=5001) #100
    # parser.add_argument('--min_dist', type=float, default=1.)
    parser.add_argument('--latent_dim', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.0001)
    # parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--base_dir', type=str,
                        default='./data/')
    parser.add_argument('--dataset_name', type=str, default='gslf' , choices={"gslf","ravia","keller","sagar","sagarfmri"})  # tree for synthetic, gslf for real
    parser.add_argument('--normalize', type=bool, default=True) #* # only for Hyperbolic embeddings
    parser.add_argument('--optimizer_type', type=str, default='AdamOptim', choices=['standard', 'poincare', 'Adam', 'AdamOptim','sgd']) #*
    parser.add_argument('--model_name', type=str, default='mds', choices=['isomap', 'mds', 'contrastive'])
    parser.add_argument('--latent_dist_fun', type=str, default='poincare', choices=['euclidean', 'poincare']) #*
    parser.add_argument('--distr', type=str, default='hypergaussian', choices=['gaussian', 'hypergaussian']) #*
    parser.add_argument('--distance_method', type=str, default='euclidean',
                        choices=['geo', 'graph', 'hamming', 'euclidean','similarity']) #'euclidean' for sagar/keller, 'similarity' for ravia
    parser.add_argument('--n_samples', type=int, default=4000)
    parser.add_argument('--dim', type=int, default=768)
    parser.add_argument('--depth', type=int, default=5)  # Changed from bool to int
    parser.add_argument('--temperature', type=float, default=0.1)  # 0.1 #100
    parser.add_argument('--n_neighbors', type=int, default=20) # 20 #10
    parser.add_argument('--epsilon', type=float, default=10.0) #
    parser.add_argument('--roi', type=str, default=None,choices=["OFC", "PirF","PirT","AMY",None]) #
    parser.add_argument('--subject', type=float, default=None,choices=[1,2,3,None]) #
    parser.add_argument('--filter_dragon', type=bool, default=False) #for chemical data
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
    data_type = args.data_type
    dataset_name = args.dataset_name
    representation_name = args.representation_name
    num_epochs = args.num_epochs
    normalize = args.normalize
    latent_dim = args.latent_dim
    lr = args.lr
    seed = args.seed
    base_dir = args.base_dir
    optimizer_type = args.optimizer_type
    model_name = args.model_name
    latent_dist_fun = args.latent_dist_fun
    distr = args.distr
    distance_method = args.distance_method
    n_neighbors = args.n_neighbors
    epsilon = args.epsilon
    temperature = args.temperature
    subject = args.subject
    roi = args.roi
    ### Overwrite the batchsize ###
    depth = args.depth
    # args.batch_size = 2 ** args.depth - 1  # to get full batch
    batch_size = args.batch_size
    filter_dragon = args.filter_dragon
#    set_seeds(seed)

    if distance_method == 'similarity' and dataset_name not in ['ravia']:
        raise ValueError('Similarity distance method can only be used with Ravia dataset')

    if dataset_name == 'tree':
        embeddings, labels = get_tree_data(depth)
        labels = torch.tensor(labels)
        embeddings = torch.tensor(embeddings)
        ## binary_tree is a dataset of binary sequences.
        ## The root of the tree is the node 0: binary_tree[0]
        ## groundtruth distance from node i to the root of the tree (i.e. shortest path distance from node i to the root): hamming_distance(binary_tree[0], binary_tree[i])
        ## For visualizations, one can color a node by its groundtruth distance to the tree.
    elif dataset_name == 'random':
        #todo do we need this?
        embeddings = torch.randn(n_samples, dim)
    elif dataset_name in ['gslf', 'keller' , 'sagar']: ### If multiple subjects, to average among them put grand_avg=True. If individual subjects then put grand_avg=False and below use select_subjects function
        input_embeddings = f'embeddings/{representation_name}/{dataset_name}_{representation_name}_embeddings_13_Apr17.csv'
        embeddings, labels,subjects,CIDs = read_embeddings(base_dir, select_descriptors(dataset_name), input_embeddings,
                                             grand_avg=True if (dataset_name == 'keller' or (dataset_name == 'saagar' and subject==None)) else False)
        # embeddings, labels,subjects,CIDs = read_embeddings(base_dir, select_descriptors(dataset_name), input_embeddings,
        #                                      grand_avg=True if dataset_name == 'keller' or dataset_name=='sagar' else False) #grand_avg averages among subjects so put false for analyzing each subject individually
        # embeddings, labels,subjects,CIDs = read_embeddings(base_dir, select_descriptors(dataset_name), input_embeddings,
        #                                      grand_avg=True if dataset_name=='sagar' else False)

        if filter_dragon:
            embeddings, labels, subjects, CIDs, embeddings_chemical=read_dragon_features(embeddings, labels, subjects, CIDs)
            args.representation_name = 'chemical'
            embeddings =  embeddings_chemical
        #embeddings = 100000 * torch.randn(4983, 20)
        
        #To perform PCA or t-SNE on MolFormer or POM enbeddings:
        # X_embedded = TSNE(n_components=2, learning_rate='auto',
        #          init='random', perplexity=300).fit_transform(embeddings)
        #X_embedded = PCA(n_components=2).fit_transform(embeddings)

        #Embed labels:
        # X_embedded = TSNE(n_components=2, learning_rate='auto',
        #          init='random', perplexity=1000).fit_transform(labels)
        #X_embedded = PCA(n_components=2).fit_transform(labels)


        # X_embedded = PCA(n_components=20).fit_transform(embeddings)
        # embeddings = torch.tensor(X_embedded, dtype=torch.float32)  # Convert to a PyTorch tensor

        # print('embeddings after PCA', embeddings.shape)

         
        if subject != None:
            # # Individual subects use a line below, otherwise comment it. To select subjects for example for sagar:
            # #embeddings, labels, CIDs, subjects = select_subjects(subjects, embeddings, labels, CIDs,subjects.unique(),subject_id=[1,2,3],n_subject=None) # ,subject_id=None,n_subject=3)
            embeddings, labels, CIDs, subjects = select_subjects(subjects, embeddings, labels, CIDs,subjects.unique(),subject_id=subject, n_subject=None)

        print('labels', labels.shape)
        #print(labels)
        print('embeddings', embeddings.shape)

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
    elif dataset_name in ['sagarfmri']:
        input_embeddings = f'embeddings/{representation_name}/sagar_{representation_name}_embeddings_13_Apr17.csv'
        embeddings, labels, subjects, CIDs,rois = read_fmri_sagar(base_dir, select_descriptors(dataset_name),
                                                             input_embeddings,subject_id=subject,selected_roi=roi)
        args.representation_name = 'fmri'



        #how to read for one specific subject and one specific area of the brain
        # subject =1 #can be 1,2,3
        # roi = 'APC' # can be
        #(n_stim, n_voxels, n_timecomponents) = data[subject - 1][roi].shape


        print('labels', labels.shape)
        #print(labels)
        print('embeddings', embeddings.shape)


    else:
        raise ValueError('Dataset not recognized')
    #keep the maximum in the third dimension of data


    dataset = OdorMonoDataset(embeddings, labels, transform=None)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

############
    # # To visualize PCA and t-SNE applied to MolFormer or POM embeddings:

    # c = torch.norm(embeddings, dim=-1)

    # #c = torch.norm(dataset.labels.detach(), dim=-1)
    # c = c/c.max()

    # # entropy = softmax(dataset.labels.detach().cpu().numpy(), -1)
    # # c = -(entropy * np.log(entropy)).sum(-1)

    # #plt.figure(figsize=(10, 6))
    # plt.scatter(X_embedded[:,0], X_embedded[:,1], s = 10,c = c, cmap = 'plasma')
    # #plt.scatter(X_embedded[:,0], X_embedded[:,1], c = c)
    # # print(X_embedded.shape)
    # # print(embeddings.shape)
    # #plt.savefig('keller_labels_PCA_norm.png')
    # #plt.savefig('keller_labels_tsne_perplexity1000_norm.png')
    # plt.savefig('gaussian_tsne_perplexity5000_norm.png')
    # plt.show()
#################




    #select a subset of subjects if needed
    # embeddings, labels, CIDs, subjects = select_subjects(subjects, embeddings, labels, CIDs,subjects.unique(),subject_id=None,n_subject=3)
    # embeddings, labels, CIDs, subjects = select_subjects(subjects, embeddings, labels, CIDs,subjects.unique(),subject_id=3,n_subject=None)



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

    if optimizer_type == 'standard':
        optimizer = StandardOptim(model, lr=lr)
    elif optimizer_type == 'sgd':
        optimizer = torch.optim.SGD([model.embeddings], lr=lr)
    elif optimizer_type == 'poincare':
        optimizer = PoincareOptim(model, lr=lr)
    elif optimizer_type == 'Adam':
        optimizer = Adam([model.embeddings], lr=lr)
    elif optimizer_type=='AdamOptim':
        optimizer = AdamOptim(model, lr=lr)
        # optimizer = Adam(model.embeddings, lr= lr)
    else:
        raise ValueError('Optimizer not recognized')

    losses = []
    # wandb.watch(model)

    correlation_coefficients = []

    if dataset_name == 'ravia':
        ent_array = np.zeros(data_dist_matrix.shape[0])
        for j, arr in enumerate(intensity_labels):
            arr = np.array(arr)
            entropy = arr / arr.sum()
            c = -(entropy * np.log(entropy)).sum(-1)
            ent_array[j] = c


    for i in range(num_epochs):
        total_loss = 0
        for idx, batch, label in data_loader:
            batch = batch.to(args.device)
            label = label.to(args.device)
            #print(batch.shape)
            if normalize:
                model.normalize()
            else:
                model.normalize(normalization=False)
            if distance_method == 'graph' :
                # data_nn_matrix = knn_graph_weighted_adjacency_matrix(batch, n_neighbors=3, metric='minkowski')
                # data_dist_matrix = (data_nn_matrix > 0).astype(int)
                # data_dist_matrix = torch.tensor(data_dist_matrix)

                #epsilon = 30.0
                if data_type == 'representation':
                    data_dist_matrix = scipy.spatial.distance.cdist(batch.detach().numpy(), batch.detach().numpy(), metric='minkowski', p=2)
                elif data_type == 'labels':
                    data_dist_matrix = scipy.spatial.distance.cdist(label.detach().numpy(), label.detach().numpy(), metric='minkowski', p=2)

                data_dist_matrix = torch.tensor(data_dist_matrix, dtype=torch.float32)

                # data_dist_matrix = 0.8*(data_dist_matrix / data_dist_matrix.max())

                ###%%%%%%%%%#########
                data_dist_matrix = 5.0*(data_dist_matrix / data_dist_matrix.max()) # 5.0 good for individual subjects


                # histo = data_dist_matrix.flatten()
                # plt.hist(histo, bins=100)
                # plt.show()

                if model_name == 'contrastive':
                    data_binary_dist_matrix = (data_dist_matrix < epsilon).int()

                # print('data_dist_matrix mean', data_dist_matrix.mean())
                # print('data_dist_matrix min', data_dist_matrix.min())
                # print('data_dist_matrix max', data_dist_matrix.max())

            elif distance_method == 'geo':
                if data_type == 'representation':
                    data_dist_matrix = knn_geodesic_distance_matrix(batch)
                elif data_type == 'labels':
                    data_dist_matrix = knn_geodesic_distance_matrix(label)

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
                #data_dist_matrix = scipy.spatial.distance.cdist(batch, batch, metric='euclidean')
                #data_dist_matrix = scipy.spatial.distance.cdist(label, label, metric='euclidean')
                
                # # Counting the numbers of zeros in descriptors
                # zero_counts = np.sum(labels.numpy() == 0, axis=0)  # Convert to NumPy array if it's a tensor               
                # plt.figure(figsize=(10, 6))
                # plt.bar(range(len(zero_counts)), zero_counts, color='blue', alpha=0.7)
                # plt.xlabel('Column Index')
                # plt.ylabel('Count of Zeros')
                # plt.title('Count of Zeros in Each Column of Labels')
                # plt.xticks(range(len(zero_counts)))  # Set x-ticks to be the column indices
                # plt.grid(axis='y')
                # plt.show()

                # import pdb; pdb.set_trace() # to enter debugging mode

                # # Plotting the histogram
                # histo = data_dist_matrix.flatten()
                # plt.hist(histo, bins=100)
                # plt.show()
                

                if data_type == 'representation':
                    data_dist_matrix = scipy.spatial.distance.cdist(batch, batch, metric='minkowski', p=2) #metric='euclidean'
                elif data_type == 'labels':
                    data_dist_matrix = scipy.spatial.distance.cdist(label, label, metric='minkowski', p=2)

                #data_dist_matrix = scipy.spatial.distance.cdist(embeddings, embeddings, metric='euclidean')

                data_dist_matrix = torch.tensor(data_dist_matrix)

                # data_dist_matrix = data_dist_matrix / data_dist_matrix.max()
                # data_dist_matrix = 0.8*(data_dist_matrix / data_dist_matrix.max())

                ######%%%%%%%%%%%%%%%##########
                ###data_dist_matrix = 5.0*(data_dist_matrix / data_dist_matrix.max()) # 5.0 good for Sagar individual subjects


                if model_name == 'contrastive':
                    data_binary_dist_matrix = kneighbors_graph(data_dist_matrix, n_neighbors=n_neighbors,
                                                               mode='connectivity', include_self=False).toarray()


            
            elif distance_method == 'similarity':
                if model_name == 'contrastive':
                    data_binary_dist_matrix = kneighbors_graph(data_dist_matrix, n_neighbors=n_neighbors,
                                                             mode='connectivity', include_self=False).toarray()
                idx = list(range(data_dist_matrix.shape[0]))


            else:
                data_dist_matrix = distance_matrix(batch, euclidean_distance)

            optimizer.zero_grad()


            if model_name == 'contrastive':
                loss = model.loss_fun(data_dist_matrix, idx, data_binary_dist_matrix, temperature) #for contrastive
            elif model_name == 'mds':
                loss = model.loss_fun(data_dist_matrix, idx)
            loss.backward()
            # print('grad', all(p.grad is None for p in model.embeddings))
            model_embeddings_old = model.embeddings.clone()
            if optimizer_type == 'Adam' or optimizer_type == 'sgd':
                optimizer.step()
            else:
                optimizer.step(idx)

            total_loss += loss.item()

        if dataset_name == 'tree':
            print(f'Epoch {i}, loss: {total_loss:.3f}')
            losses.append(total_loss)         
        else:
            # print(f'Epoch {i}, loss: {total_loss / len(data_loader):.3f}')
            # losses.append(total_loss / len(data_loader))
            print(f'Epoch {i}, loss: {total_loss :.3f}')
            losses.append(total_loss)

        if i % 10 == 0:  # 1000
            # save_embeddings(i, args, model.embeddings.detach().cpu().numpy(), losses=losses,
            #                 losses_neg=model.losses_neg if model_name == 'contrastive' else [],
            #                 losses_pos=model.losses_pos if model_name == 'contrastive' else [])
            # scatterplot_2d(i, model.embeddings.detach().cpu().numpy(), dataset.labels.detach().cpu().numpy(), CIDs,labels, subjects=subjects,
            
            if dataset_name in ['ravia', 'snitz']:
                scatterplot_2d(i, model.embeddings.detach().cpu().numpy(),
                                            ent_array, CIDs, labels, subjects=subjects,
                            color_by='color', shape_by='none',
                            save=True, args=args,
                            hyperbolic_boundary = normalize)

                plot_losses(i, args=args, save=True, losses=losses, losses_neg=model.losses_neg if model_name == 'contrastive' else None,
                            losses_pos=model.losses_pos if model_name == 'contrastive' else None)
                c = ent_array


            elif dataset_name in ['sagar', 'sagarfmri']: # For sagar for example, with color_by='entropy'
                scatterplot_2d(i, model.embeddings.detach().cpu().numpy(),
                                            dataset.labels.detach(), CIDs, labels, subjects=subjects,
                            color_by='entropy', shape_by='none',
                            save=True, args=args,
                            hyperbolic_boundary = normalize)

                # plot_losses(i, args=args, save=True, losses=losses, losses_neg=model.losses_neg if model_name == 'contrastive' else None,
                #             losses_pos=model.losses_pos if model_name == 'contrastive' else None)

                # For sagar for example, print correlation coefficient between entropy and hyperbolic radius:
                entropy = softmax(dataset.labels.detach().cpu().numpy(), -1)
                c = -(entropy * np.log(entropy)).sum(-1)


            elif dataset_name in ['keller']: #['gslf','keller'] # For gslf and keller for example, with color_by='input_norm'
                scatterplot_2d(i, model.embeddings.detach().cpu().numpy(),
                                            dataset.labels.detach(), CIDs, labels, subjects=subjects, #for gaussian, 3rd argument can be embeddings instead of dataset.labels.detach()
                            color_by='input_norm', shape_by='none',
                            save=True, args=args,
                            hyperbolic_boundary = normalize)
                c = torch.norm(dataset.labels.detach(), dim=-1)

            elif dataset_name == 'gslf':

                # scatterplot_2d(i, model.embeddings.detach().cpu().numpy(),
                #                             dataset.labels.detach(), CIDs, labels, subjects=subjects,
                #             color_by='input_norm', shape_by='none',
                #             save=True, args=args,
                #             losses=losses, losses_neg=model.losses_neg if model_name == 'contrastive' else [],
                #             losses_pos=model.losses_pos if model_name == 'contrastive' else [],
                #             hyperbolic_boundary = normalize)
                # c = torch.norm(dataset.labels.detach(), dim=-1)

                scatterplot_2d(i, model.embeddings.detach().cpu().numpy(),
                                            dataset.labels.detach(), CIDs, labels, subjects=subjects,
                            color_by='input_norm', shape_by='none',
                            save=True, args=args,
                            hyperbolic_boundary = normalize)
                plot_losses(i, args=args, save=True, losses=losses, losses_neg=model.losses_neg if model_name == 'contrastive' else None,
                            losses_pos=model.losses_pos if model_name == 'contrastive' else None)
                c = torch.norm(dataset.labels.detach(), dim=-1)


                # To visualize some classes
                # colors = {
                #     # "fruity": "#32CD32",  # Lime Green (top-level fruity category)
                #     # "citrus": "#FFA500",  # Orange (subcategory citrus)
                #     # "berry": "#8A2BE2",  # Blue Violet (subcategory berry)
                #     # "tropical": "#FFD700",  # Gold (subcategory tropical)
                #     "specific": {  # Specific labels mapped to their colors
                #         # "bergamot": "#FFD580",
                #         # "grapefruit": "#FF6347",
                #         # "lemon": "#FFF44F",
                #         "orange": "#FFA500",
                #         # "black currant": "#4B0082",
                #         # "raspberry": "#E30B5D",
                #         "strawberry": "#FF4500",
                #         "banana": "#FFFACD",
                #         "coconut": "#FFE4C4",
                #         # "pineapple": "#FFD700",
                #     },
                # }
                # color_codes= generate_colors_for_labels(labels, colors)
                # # select a subset of molecules based on the labels if needed
                # selected_labels,selected_subjects,selected_embeddings,selected_CIDs,selected_model_embeddings,selected_colors  = select_molecules( labels,
                #                                                      [
                #                                                          # "bergamot", "grapefruit",
                #                                                          #               "lemon",
                #                                                       "orange",
                #                                                                        # "black currant", "raspberry",
                #                                                                        "strawberry",
                #                                                       "banana",
                #                                                                        "coconut",
                #                                                       # "pineapple",
                #                                                                        # "tropical",
                #                                                       # "berry", "citrus",
                #                                                       #                  "fruity"
                #                                                       ],subjects, embeddings,CIDs,model.embeddings,color_codes)
                # #


                # scatterplot_2d(i, selected_model_embeddings.detach().cpu().numpy(),
                #                             selected_colors, selected_CIDs, selected_labels, subjects=selected_subjects,
                #             color_by='color', shape_by='none',
                #             save=True, args=args,
                #             hyperbolic_boundary = normalize)
                c = torch.norm(dataset.labels.detach(), dim=-1)
                # plot_losses(i, args=args, save=True, losses=losses, losses_neg=model.losses_neg if model_name == 'contrastive' else None,
                #             losses_pos=model.losses_pos if model_name == 'contrastive' else None)







            if dataset_name != 'tree':
                radius = poincare_distance(model.embeddings.detach().cpu(), torch.zeros((1, 2)))
                corr = np.corrcoef(radius, c)[0, 1]  # Get the correlation coefficient
                correlation_coefficients.append(corr)  # Store the correlation coefficient
                print(correlation_coefficients)
        if i % 50 == 0:
            save_embeddings_npy(model.embeddings, args, i)



    # # After the training loop, plot the correlation coefficient vs. epochs
    # plt.figure(figsize=(10, 6))
    # plt.plot(range(correlation_coefficients), correlation_coefficients, marker='o', color='blue', linestyle='-', linewidth=2)
    # plt.xlabel('Epoch', fontsize=14)
    # plt.ylabel('Correlation Coefficient', fontsize=14)
    # plt.title('Correlation Coefficient vs. Number of Epochs', fontsize=16)
    # plt.grid(True)
    # plt.tight_layout()
    # # Save the correlation coefficient plot
    # plt.savefig('figs2/correlation_coefficient_vs_epochs.png')
    # plt.close()


            # scatterplot_2d_loss(i, model.embeddings.detach(), save=True, args=args,
            #                losses=losses, losses_neg=model.losses_neg if model_name == 'contrastive' else [],
            #                losses_pos=model.losses_pos if model_name == 'contrastive' else [])




