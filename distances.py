import torch
from torch.linalg import vector_norm
import numpy as np
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import dijkstra
import scipy
from typing import Callable, Union

EPS = 1e-5

def distance_matrix(data: torch.Tensor, distance_func: Callable) -> torch.Tensor:
    """Builds a distance matrix given a vectorized distance function."""
    preliminary = distance_func(data.unsqueeze(0), data.unsqueeze(1))
    result = preliminary + torch.eye(data.shape[0], device=data.device)
    return result.float()

def euclidean_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute Euclidean distance between x and y."""
    return vector_norm(x - y, dim=-1)

def poincare_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute Poincare distance between x and y."""
    euc_dist = ((x - y)**2).sum(-1)
#     norm_x = 1 - (x**2).sum(-1)
#     norm_y = 1 - (y**2).sum(-1)
    norm_x = torch.clamp(1 - (x**2).sum(-1), min=EPS)
    norm_y = torch.clamp(1 - (y**2).sum(-1), min=EPS)
    x = torch.clamp(1 + 2 * torch.div(euc_dist, (norm_x * norm_y)), min=1 + EPS)
    return torch.acosh(x)

def hamming_distance(x: np.ndarray, y: np.ndarray) -> int:
    """Compute Hamming distance between x and y."""
    return (x.astype(np.int32) ^ y.astype(np.int32)).sum()

def knn_geodesic_distance_matrix(data: np.ndarray, n_neighbors: int = 3) -> torch.Tensor:
    """Compute geodesic distance matrix using k-nearest neighbors."""
    data_nn_matrix = kneighbors_graph(data, n_neighbors, mode='connectivity', include_self=False)
    data_dist_matrix = data_nn_matrix.toarray()
    # data_dist_matrix = dijkstra(data_nn_matrix)

    data_dist_matrix = torch.FloatTensor(data_dist_matrix)
    # data_dist_matrix = torch.where(data_dist_matrix == torch.inf, 1000 * torch.ones_like(data_dist_matrix), data_dist_matrix)
    return data_dist_matrix

def knn_graph_weighted_adjacency_matrix(data: np.ndarray, n_neighbors: int = 3, metric: str = 'minkowski') -> np.ndarray:
    """Compute k-nearest neighbors graph weighted adjacency matrix."""
    data_nn_matrix = kneighbors_graph(data, n_neighbors, mode='distance', include_self=False, metric=metric)
    return data_nn_matrix.toarray()

# def hamming_distance_matrix(data: np.ndarray) -> np.ndarray:
#     """Compute Hamming distance matrix for binary data."""
#     return scipy.spatial.distance.cdist(data, data, metric='hamming') * data.shape[-1]
#
