# scattorplot of 2d visualizations
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import manifold
from scipy.stats import gaussian_kde
from scipy.special import softmax
from sklearn.decomposition import PCA
from matplotlib.patches import Patch
import numpy as np
import scipy
import torch
import os
import matplotlib
from distances import poincare_distance
plt.rcParams["font.size"] = 45
plt.rcParams['agg.path.chunksize'] = 10000

from constants import *


def plot_losses(i, input_embeddings, args=None, save=False, losses=[],
                losses_pos=[], losses_neg=[]):
    color_map = 'plasma'
    # latent_embeddings_norm = torch.norm(latent_embeddings, dim=-1)
    data_dist_matrix = scipy.spatial.distance.cdist(input_embeddings, input_embeddings, metric='hamming') * \
                       input_embeddings.shape[-1]
    fig, ax = plt.subplots(2, 1, figsize=(30, 90), sharey=False)
    ax[1].plot(np.arange(len(losses)), losses, label='total')

    ax[1].plot(np.arange(len(losses_pos)), losses_pos, label='positive')
    ax[2].plot(np.arange(len(losses_neg)), losses_neg, label='negative')

    fig.subplots_adjust(hspace=0.3)
    # showing the legend
    ax[1].legend()
    ax[2].legend()
    plt.title(
        f'dataset_name={args.dataset_name}, lr = {args.lr}, latent_dim = {args.latent_dim}, epochs = {args.num_epochs}, \n batch_size = {args.batch_size}, normalize = {args.normalize}, distance_method = {args.distance_method},\n  model = {args.model_name}, optimizer = {args.optimizer}, latent_dist_fun = {args.latent_dist_fun} \n temperature = {args.temperature}, depth = {args.depth}')

    # create a folder if it does not exist
    if save:
        if not os.path.exists(
                f"figs2/{args.dataset_name}/{args.latent_dist_fun}/{args.normalize}/{args.model_name}/{args.optimizer}/{args.lr}/{args.temperature}/"):
            os.makedirs(
                f"figs2/{args.dataset_name}/{args.latent_dist_fun}/{args.normalize}/{args.model_name}/{args.optimizer}/{args.lr}/{args.temperature}/")
        plt.savefig(
            f"figs2/{args.dataset_name}/{args.latent_dist_fun}/{args.normalize}/{args.model_name}/{args.optimizer}/{args.lr}/{args.temperature}/{args.random_string}_{i}_{args.seed}_{args.dataset_name}_losses.png")
        plt.close()
    else:
        plt.show()
def scatterplot_2d(i, latent_embeddings, input_embeddings, CIDs, labels, subjects=None, color_by='entropy', shape_by='none', args=None, save=False, plot_edges=False, losses=[],
                   losses_pos=[], losses_neg=[], hyperbolic_boundary=True):
    color_map = 'plasma'
    #
    # data_dist_matrix = scipy.spatial.distance.cdist(input_embeddings, input_embeddings, metric='hamming') * \
    #                    input_embeddings.shape[-1]
    fig, ax = plt.subplots(1, 1, figsize=(30, 30), sharey=False)
    markers = ["o", "s", "D", "P", "X", "v", ">", "<", "^", "d", "p", "*", "h", "H", "+", "x", "|", "_"]
    if shape_by=='subject':
        groups = subjects.unique()
        selected_groups = subjects
    elif shape_by=='descriptor':
        groups = np.arange(labels.shape[1])
        selected_groups = [np.argmax(label[2:]) + 2 for label in labels]
    elif shape_by=='none':
        groups = None
        markers = ["o"]*subjects.max()

    else:
        raise ValueError('shape_by not recognized')

    if color_by=='input_norm':
        c = torch.norm(input_embeddings, dim=-1)
    elif color_by=='entropy':
        entropy = softmax(input_embeddings, -1)
        c = -(entropy * np.log(entropy)).sum(-1)
    elif color_by=='cid':
        unique_vals = np.unique(CIDs)
        mapping = {val: idx for idx, val in enumerate(unique_vals)}
        c = np.vectorize(mapping.get)(CIDs)
    elif color_by=='distance':
        #todo
        c = torch.norm(latent_embeddings, dim=-1)
    elif color_by=='none':
        c =np.ones(latent_embeddings.shape[0])
    elif color_by=='color':
        c = input_embeddings



    else:
        raise ValueError('color_by not recognized')


    radius = np.sqrt(np.sum(np.square(latent_embeddings), axis=1))
    radius = poincare_distance(torch.from_numpy(latent_embeddings), torch.zeros((1, 2)))
    corr = np.corrcoef(radius, c)
    #print('corr', corr)

    # Plotting the correlation
    plt.figure(figsize=(10, 6))
    plt.scatter(radius, c, color='#6a0dad', alpha=0.7, s=50)  # Scatter plot # edgecolor='k'
    #
    # # Calculate the line of best fit
    slope, intercept = np.polyfit(radius, c, 1)  # Linear regression
    line = slope * radius + intercept  # Calculate the y values for the line
    #
    # # Plot the regression line
    plt.plot(radius, line, color='#ffbf00', linewidth=2)  # Add the line to the plot
    #
    plt.xlabel('Hyperbolic radius', fontsize=30)
    # plt.ylabel('Entropy', fontsize=30)
    plt.ylabel('Color', fontsize=30)

   # plt.title('Correlation between Radius and Entropy', fontsize=16)  # Added title
    plt.grid(True)
    plt.legend()  # Show legend
    plt.tight_layout()  # Adjust layout for better spacing

    # plt.xticks([])
    # plt.yticks([])
    #
    # #plt.axis('off')
    #
    # # Save the figure with 'corr' in the filename
    plt.savefig(f'figs2/{i}_corr.png')  # Changed filename to include 'corr'
    plt.close()  # Close the plot to avoid display if running in a script




    if plot_edges:
        colors = sns.color_palette("plasma", data_dist_matrix.shape[0])
        for i in range(data_dist_matrix.shape[0]):
            for j in range(i + 1, data_dist_matrix.shape[1]):
                if data_dist_matrix[i, j] <= 1.01:
                    ax.plot([latent_embeddings[i, 0], latent_embeddings[j, 0]],
                               [latent_embeddings[i, 1], latent_embeddings[j, 1]], color=colors[i], linewidth=5.)

    ax.axis('equal')
    ax.axis('off')
    if args.normalize == True:
        ax.set_ylim(-1.09, 1.09)
        ax.set_xlim(-1.09, 1.09)


    ## ax[0].scatter(latent_embeddings[:, 0], latent_embeddings[:, 1], c=np.linalg.norm(input_embeddings, axis=-1), cmap=color_map, s=300, zorder=10   )
    
    # for subject in subjects.unique():
    #     if subject==3:
    #         idx = subjects == subject
    #         ax.scatter(latent_embeddings[idx, 0], latent_embeddings[idx, 1], c=c[idx], cmap="plasma" , s=300, zorder=10,marker=markers[subject-1])

    if groups is None:
        ax.scatter(latent_embeddings[:, 0], latent_embeddings[:, 1], c=c, cmap=color_map, s=300, zorder=10)
    else:
        for group in groups:
            idx = selected_groups == group
            ax.scatter(latent_embeddings[idx, 0], latent_embeddings[idx, 1], c=c[idx], cmap=color_map , s=300, zorder=10,marker=markers[group-1],vmin = c.min(), vmax = c.max())



    if hyperbolic_boundary:
        circle = plt.Circle((0, 0), 1., color='gray', fill=False, linewidth=10)
        ax.add_patch(circle)


    # create a folder if it does not exist
    if save:
        if not os.path.exists(
                f"figs2/{args.dataset_name}/{args.lr}/{args.temperature}/{args.n_neighbors}/"):
            os.makedirs(
                f"figs2/{args.dataset_name}/{args.lr}/{args.temperature}/{args.n_neighbors}/")
        plt.savefig(
            f"figs2/{args.dataset_name}/{args.lr}/{args.temperature}/{args.n_neighbors}/{args.random_string}_{i}_{args.seed}_{args.dataset_name}_embeddings.png")
        plt.close()
    else:
        plt.show()



def save_embeddings(i, args, latent_embeddings, losses=[], losses_pos=[], losses_neg=[]):
    if not os.path.exists(
            f"results/{args.dataset_name}/{args.lr}/{args.temperature}/{args.n_neighbors}/"):
        os.makedirs(
            f"results/{args.dataset_name}/{args.lr}/{args.temperature}/{args.n_neighbors}/")

    np.save(
        f"results/{args.dataset_name}/{args.lr}/{args.temperature}/{args.n_neighbors}/{args.random_string}_{i}_{args.seed}_{args.dataset_name}_embeddings.npy",
        latent_embeddings)
    np.save(
        f"results/{args.dataset_name}/{args.lr}/{args.temperature}/{args.n_neighbors}/{args.random_string}_{i}_{args.seed}_{args.dataset_name}_losses.npy",
        losses)
    np.save(
        f"results/{args.dataset_name}/{args.lr}/{args.temperature}/{args.n_neighbors}/{args.random_string}_{i}_{args.seed}_{args.dataset_name}_lossespos.npy",
        losses_pos)
    np.save(
        f"results/{args.dataset_name}/{args.lr}/{args.temperature}/{args.n_neighbors}/{args.random_string}_{i}_{args.seed}_{args.dataset_name}_lossesneg.npy",
        losses_neg)


def scatterplot_2d_gslf(latent_embeddings, input_embeddings, labels, color_map='viridis', args=None, losses=[],
                        losses_pos=[], losses_neg=[]):
    type1 = {'floral': '#F3F1F7', 'muguet': '#FAD7E6', 'lavender': '#8883BE', 'jasmin': '#BD81B7'}
    type2 = {'meaty': '#F5EBE8', 'savory': '#FBB360', 'beefy': '#7B382A', 'roasted': '#F7A69E'}
    type3 = {'ethereal': '#F2F6EC', 'cognac': '#BCE2D2', 'fermented': '#79944F', 'alcoholic': '#C2DA8F'}
    types = [type1, type2, type3]

    data_dist_matrix = scipy.spatial.distance.cdist(input_embeddings, input_embeddings, metric='hamming') * \
                       input_embeddings.shape[-1]
    latent_embeddings = latent_embeddings.detach().cpu().numpy()
    fig, ax = plt.subplots(3, 1, figsize=(10, 30), sharey=False)

    # make n different colors
    # colors = sns.color_palette("hsv", data_dist_matrix.shape[0])

    # for i in range(data_dist_matrix.shape[0]):
    #     for j in range(i+1, data_dist_matrix.shape[1]):
    #         if data_dist_matrix[i,j] <= 1.01:
    #             ax[0].plot([latent_embeddings[i,0], latent_embeddings[j,0]], [latent_embeddings[i,1], latent_embeddings[j,1]], color=colors[i], linewidth=0.5)

    labels = labels.detach().cpu().numpy()
    for type in types:
        for key in type.keys():
            # where in gs_lf_tasks is the key
            key_idx = gs_lf_tasks.index(key)
            # choose the indices of the labels that are 1 in the key_idx
            embedding_idx = np.where(labels[:, key_idx] == 1)[0]
            # idx = labels.index(key)
            ax[0].scatter(latent_embeddings[embedding_idx, 0], latent_embeddings[embedding_idx, 1], c=type[key],
                          label=key)

    ax[1].plot(np.arange(len(losses)), losses, label='total')

    ax[1].plot(np.arange(len(losses_pos)), losses_pos, label='positive')
    ax[2].plot(np.arange(len(losses_neg)), losses_neg, label='negative')
    ax[0].set_ylim(-1.09, 1.09)
    ax[0].set_xlim(-1.09, 1.09)

    fig.subplots_adjust(hspace=0.3)
    # showing the legend
    ax[1].legend()
    ax[2].legend()
    plt.title(
        f'dataset_name={args.dataset_name}, lr = {args.lr}, latent_dim = {args.latent_dim}, epochs = {args.num_epochs}, \n batch_size = {args.batch_size}, normalize = {args.normalize}, distance_method = {args.distance_method},\n  model = {args.model_name}, optimizer = {args.optimizer}, latent_dist_fun = {args.latent_dist_fun} \n temperature = {args.temperature}, depth = {args.depth}')

    # create a folder if it does not exist
    if not os.path.exists(
            f"figs2/{args.depth}/{args.latent_dist_fun}/{args.normalize}/{args.model_name}/{args.optimizer}/{args.lr}/{args.temperature}/"):
        os.makedirs(
            f"figs2/{args.depth}/{args.latent_dist_fun}/{args.normalize}/{args.model_name}/{args.optimizer}/{args.lr}/{args.temperature}/")
    plt.savefig(
        f"figs2/{args.depth}/{args.latent_dist_fun}/{args.normalize}/{args.model_name}/{args.optimizer}/{args.lr}/{args.temperature}/{args.random_string}_{args.num_epochs}_{args.seed}_{args.dataset_name}.png")

    plt.close()


#


def pom_frame(pom_embeds, y, dir, required_desc, title, size1, size2, size3, reduction_method=None, perplexity=None):
    sns.set_style("ticks")
    sns.despine()

    # pom_embeds = model.predict_embedding(dataset)
    # y_preds = model.predict(dataset)
    # required_desc = list(dataset.tasks)
    type1 = {'floral': '#F3F1F7', 'subs': {'muguet': '#FAD7E6', 'lavender': '#8883BE', 'jasmin': '#BD81B7'}}
    type2 = {'meaty': '#F5EBE8', 'subs': {'savory': '#FBB360', 'beefy': '#7B382A', 'roasted': '#F7A69E'}}
    type3 = {'ethereal': '#F2F6EC', 'subs': {'cognac': '#BCE2D2', 'fermented': '#79944F', 'alcoholic': '#C2DA8F'}}

    # Assuming you have your features in the 'features' array
    if reduction_method == 'PCA':
        pca = PCA(n_components=2,
                  iterated_power=10)  # You can choose the number of components you want (e.g., 2 for 2D visualization)
        reduced_features = pca.fit_transform(pom_embeds)  # try different variations
        variance_explained = pca.explained_variance_ratio_
        variance_pc1 = variance_explained[0]
        variance_pc2 = variance_explained[1]
        print(variance_pc1, variance_pc2)

    elif reduction_method == 'tsne':
        tsne = manifold.TSNE(
            n_components=2,
            init="random",
            random_state=0,
            perplexity=perplexity,

        )
        reduced_features = tsne.fit_transform(pom_embeds)
    elif reduction_method == 'UMAP':
        reduced_features = umap.UMAP(n_components=2, n_neighbors=perplexity, min_dist=0.0,
                                     metric='euclidean').fit_transform(X=pom_embeds)
    elif reduction_method is None:
        reduced_features = pom_embeds
    else:
        raise ValueError('Invalid reduction method')

    # if is_preds:
    #     y = np.where(y_preds>threshold, 1.0, 0.0) # try quartile range (or rank)
    # else:
    #     y = dataset.y

    # Generate grid points to evaluate the KDE on (try kernel convolution)
    x_grid, y_grid = np.meshgrid(np.linspace(reduced_features[:, 0].min(), reduced_features[:, 0].max(), 500),
                                 np.linspace(reduced_features[:, 1].min(), reduced_features[:, 1].max(), 500))
    grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()])
    print(reduced_features[:, 0].min(), reduced_features[:, 0].max(), reduced_features[:, 1].min(),
          reduced_features[:, 1].max())

    def get_kde_values(label):
        plot_idx = required_desc.index(label)
        # print(y[:, plot_idx])
        label_indices = np.where(y[:, plot_idx] == 1)[0]
        kde_label = gaussian_kde(reduced_features[label_indices].T)
        kde_values_label = kde_label(grid_points)
        kde_values_label = kde_values_label.reshape(x_grid.shape)
        return kde_values_label

    def plot_contours(type_dictionary, bbox_to_anchor):
        main_label = list(type_dictionary.keys())[0]
        plt.contourf(x_grid, y_grid, get_kde_values(main_label), levels=1,
                     colors=['#00000000', type_dictionary[main_label], type_dictionary[main_label]])
        axes = plt.gca()  # Getting the current axis

        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False)
        legend_elements = []
        for label, color in type_dictionary['subs'].items():
            plt.contour(x_grid, y_grid, get_kde_values(label), levels=1, colors=color, linewidths=2)
            legend_elements.append(Patch(facecolor=color, label=label))
        legend = plt.legend(handles=legend_elements, title=main_label, bbox_to_anchor=bbox_to_anchor, prop={'size': 30})
        legend.get_frame().set_facecolor(type_dictionary[main_label])
        plt.gca().add_artist(legend)

    fig = plt.figure(figsize=(15, 15), dpi=700)
    # ax.spines[['right', 'top']].set_visible(False)
    # plt.title('KDE Density Estimation with Contours in Reduced Space')
    # plt.xlabel(f'Principal Component 1 ({round(variance_pc1*100, ndigits=2)}%)')
    # plt.ylabel(f'Principal Component 2 ({round(variance_pc2*100, ndigits=2)}%)')
    plt.xlabel('Principal Component 1', fontsize=35)
    plt.ylabel('Principal Component 2', fontsize=35)
    plot_contours(type_dictionary=type1, bbox_to_anchor=size1)
    plot_contours(type_dictionary=type2, bbox_to_anchor=size2)
    plot_contours(type_dictionary=type3, bbox_to_anchor=size3)
    # plt.colorbar(label='Density')
    # plt.show()
    # png_file = os.path.join(dir, 'pom_frame.png')
    # plt.savefig(png_file)
    plt.savefig("figs/islands/realign_islands_" + title + "_" + reduction_method + "_" + str(perplexity) + ".svg")
    plt.savefig("figs/islands/realign_islands_" + title + "_" + reduction_method + "_" + str(perplexity) + ".pdf")
    plt.savefig("figs/islands/realign_islands_" + title + "_" + reduction_method + "_" + str(perplexity) + ".jpg")

    # plt.show()
    # plt.close()