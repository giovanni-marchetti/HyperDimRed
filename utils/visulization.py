#scattorplot of 2d visualizations
import matplotlib.pyplot as plt




def scatterplot_2d(embeddings, labels, title, color_map='viridis'):


    plt.figure(figsize=(10, 10))
    plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap=color_map)
    plt.title(title)
    plt.show()

    return embeddings