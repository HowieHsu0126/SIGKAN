from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import itertools
import logging
import os

import igraph
import numpy as np
import sklearn
import torch
from sklearn import preprocessing
from torch.utils.data import Dataset, Sampler
from utils import load_w2v_feature

logger = logging.getLogger(__name__)


class ChunkSampler(Sampler):
    """
    A Sampler that yields a specified number of samples starting from a given offset.

    Args:
        num_samples (int): The number of samples to draw.
        start (int): The starting index to sample from.
    """

    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        # Yield a sequential range of sample indices
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples


class InfluenceDataSet(Dataset):
    """
    A PyTorch Dataset for handling influence networks, including loading graphs, 
    vertex features, labels, and embeddings.

    Args:
        file_dir (str): Directory containing the dataset files.
        embedding_dim (int): Dimensionality of node embeddings.
        seed (int): Random seed for shuffling data.
        shuffle (bool): Whether to shuffle the dataset.
        model (str): The model type ('gat', 'pscn', 'gcn') to determine graph preprocessing.
    """

    def __init__(self, file_dir, embedding_dim, seed, shuffle, model):
        # Load and preprocess graph adjacency matrices
        self.graphs = np.load(os.path.join(
            file_dir, "adjacency_matrix.npy")).astype(np.float32)

        # Add self-loops and binarize the graph if necessary
        identity = np.identity(self.graphs.shape[1])
        self.graphs += identity
        self.graphs[self.graphs != 0] = 1.0

        if model == "gat" or model == "pscn":
            # GAT/PSCN expect binary adjacency matrices
            self.graphs = self.graphs.astype(np.uint8)
        elif model == "gcn":
            # Normalize graphs for GCN: D^{-1/2}AD^{-1/2}
            self._normalize_graphs()
        else:
            raise NotImplementedError(f"Model '{model}' not supported.")
        logger.info("Graphs loaded!")

        # Load other dataset components: influence features, labels, vertex IDs, and vertex features
        self.influence_features = np.load(os.path.join(
            file_dir, "influence_feature.npy")).astype(np.float32)
        self.labels = np.load(os.path.join(file_dir, "label.npy"))
        self.vertices = np.load(os.path.join(file_dir, "vertex_id.npy"))

        if shuffle:
            self._shuffle_data(seed)

        self.vertex_features = torch.FloatTensor(preprocessing.scale(
            np.load(os.path.join(file_dir, "vertex_feature.npy"))))
        logger.info("Vertex features loaded!")

        # Load pre-trained node embeddings
        embedding_path = os.path.join(
            file_dir, f"deepwalk.emb_{embedding_dim}")
        self.embedding = torch.FloatTensor(
            load_w2v_feature(embedding_path, np.max(self.vertices)))
        logger.info(f"{embedding_dim}-dim embeddings loaded!")

        self.N = self.graphs.shape[0]  # Total number of ego networks
        logger.info(
            f"{self.N} ego networks loaded, each with size {self.graphs.shape[1]}")

        # Calculate class weights for imbalanced data
        self.class_weight = self._compute_class_weight()

    def _normalize_graphs(self):
        """Normalize the adjacency matrices for GCN."""
        for i, graph in enumerate(self.graphs):
            d_root_inv = 1. / np.sqrt(np.sum(graph, axis=1))
            graph = (graph.T * d_root_inv).T * d_root_inv
            self.graphs[i] = graph

    def _shuffle_data(self, seed):
        """Shuffle the dataset components (graphs, features, labels, and vertices)."""
        self.graphs, self.influence_features, self.labels, self.vertices = sklearn.utils.shuffle(
            self.graphs, self.influence_features, self.labels, self.vertices, random_state=seed
        )
        logger.info("Data shuffled!")

    def _compute_class_weight(self):
        """Compute class weights to handle imbalanced data."""
        n_classes = self.get_num_class()
        class_weight = self.N / (n_classes * np.bincount(self.labels))
        return torch.FloatTensor(class_weight)

    def get_embedding(self):
        """Return the pre-trained node embeddings."""
        return self.embedding

    def get_vertex_features(self):
        """Return the vertex features."""
        return self.vertex_features

    def get_feature_dimension(self):
        """Return the dimensionality of the influence features."""
        return self.influence_features.shape[-1]

    def get_num_class(self):
        """Return the number of unique classes in the dataset."""
        return np.unique(self.labels).shape[0]

    def get_class_weight(self):
        """Return the class weights."""
        return self.class_weight

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        """Return a single data point from the dataset."""
        return self.graphs[idx], self.influence_features[idx], self.labels[idx], self.vertices[idx]


class PatchySanDataSet(InfluenceDataSet):
    """
    A subclass of InfluenceDataSet designed for Patchy-SAN models, which requires
    generating receptive fields based on BFS traversal.

    Args:
        file_dir (str): Directory containing the dataset files.
        embedding_dim (int): Dimensionality of node embeddings.
        seed (int): Random seed for shuffling data.
        shuffle (bool): Whether to shuffle the dataset.
        model (str): The model type (should be 'pscn' for this dataset).
        sequence_size (int): Size of the BFS sequence to generate.
        stride (int): Stride for the sliding window (unused in this case).
        neighbor_size (int): Size of the neighborhood to consider for each node.
    """

    def __init__(self, file_dir, embedding_dim, seed, shuffle, model, sequence_size=8, stride=1, neighbor_size=8):
        assert model == "pscn", "PatchySanDataSet only supports 'pscn' model."
        super().__init__(file_dir, embedding_dim, seed, shuffle, model)

        logger.info("Generating receptive fields...")
        self.receptive_fields = self._generate_receptive_fields(
            sequence_size, neighbor_size)
        logger.info("Receptive fields generated!")

    def _generate_receptive_fields(self, sequence_size, neighbor_size):
        """Generate receptive fields for each graph using BFS traversal."""
        n_vertices = self.graphs.shape[1]
        receptive_fields = []

        for i in range(self.graphs.shape[0]):
            adj = self.graphs[i]
            edges = list(zip(*np.where(adj)))
            g = igraph.Graph(edges=edges, directed=False)
            g.simplify()
            assert g.vcount() == n_vertices, "Vertex count mismatch."

            sequence = self.get_bfs_order(
                g, n_vertices - 1, sequence_size, self.influence_features[i])
            neighborhoods = np.full(
                (sequence_size, neighbor_size), -1, dtype=np.int32)

            for j, v in enumerate(sequence):
                if v < 0:
                    break
                shortest = list(itertools.islice(
                    g.bfsiter(int(v), mode='ALL'), neighbor_size))
                for k, vtx in enumerate(shortest):
                    neighborhoods[j][k] = vtx.index

            neighborhoods = neighborhoods.reshape(
                sequence_size * neighbor_size)
            receptive_fields.append(neighborhoods)

        return np.array(receptive_fields, dtype=np.int32)

    def get_bfs_order(self, g, v, size, key):
        """
        Generate a BFS order for the graph and sort each layer by the given key.

        Args:
            g (igraph.Graph): The graph to perform BFS on.
            v (int): The starting vertex for BFS.
            size (int): The maximum size of the BFS sequence.
            key (array-like): Features used to sort nodes in each BFS layer.

        Returns:
            list: A BFS-ordered list of nodes, limited by 'size'.
        """
        order, indices, _ = g.bfs(v, mode="ALL")
        for j, start in enumerate(indices[:-1]):
            if start >= size:
                break
            end = indices[j + 1]
            order[start:end] = sorted(
                order[start:end], key=lambda x: key[x][0], reverse=True)

        return order[:size]

    def __getitem__(self, idx):
        """Return a single data point from the dataset, including receptive fields."""
        return self.receptive_fields[idx], self.influence_features[idx], self.labels[idx], self.vertices[idx]
