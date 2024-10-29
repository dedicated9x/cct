# -*- coding: utf-8 -*-
"""
A graph representation of reactions for the "GAT" (Graph Attention Network)
"""
import os
import json
from multiprocessing import Pool

from scipy import sparse
from tqdm import tqdm

from src_.data.dataset import Dataset
from src_.featurization.reaction_featurizer import ReactionFeaturizer
from src_.featurization.utils import *


def _get_adj_path(feat_dir: str) -> str:
    return os.path.join(feat_dir, 'adj.npz')


def _get_nodes_path(feat_dir: str) -> str:
    return os.path.join(feat_dir, 'nodes.npz')


def _get_n_nodes_path(feat_dir: str) -> str:
    return os.path.join(feat_dir, 'n_nodes.npz')


def gat_featurize_parallel(params: Tuple) -> Tuple:
    """
    A pickable function to be executed in parallel for converting reaction products/substrates SMILES strings
    into graphs represented as sparse matrices
    :param params: tuple of parameters need for featurizing a chunk of data
    :return: a tuple of (vector with graph sizes, matrix with node data, matrix with edge data)
    """
    thread_num, data_len, data_idx, data_x, max_n_nodes = params

    n_reactions = len(data_x['substrates'])
    k = 15000  # to save RAM, save matrices every k reactions

    n_nodes = np.zeros(n_reactions, dtype=int)
    nodes_mat = sparse.csr_matrix(([], ([], [])), shape=(data_len, max_n_nodes))
    adj_mat = sparse.csr_matrix(([], ([], [])), shape=(data_len, max_n_nodes ** 2))

    adj_vals = [], [], []  # vals, rows, cols
    nodes_vals = [], [], []  # vals, rows, cols

    for i, (ind, sub_smi, prod_smi) in tqdm(enumerate(zip(data_idx, data_x['substrates'], data_x['product'])),
                                            desc='Thread {}: converting SMILES to graphs...'.format(thread_num),
                                            total=n_reactions):
        sub_mol = safe_mol_from_smiles(sub_smi)
        prod_mol = safe_mol_from_smiles(prod_smi)
        graph = GatReactionGraph((sub_mol, prod_mol))

        n_nodes[i] = len(graph.nodes)

        for j, node in enumerate(graph.nodes):
            # the shape of node data is (data_len, max_n_nodes)
            nodes_vals[0].append(node)
            nodes_vals[1].append(ind)
            nodes_vals[2].append(j)

        for val, row, col in zip(graph.adj_vals, graph.adj_rows, graph.adj_cols):
            # scipy sparse only supports 2D matrices
            # we store adjacency matrix in a flattened format
            # where the shape of adjacency data is (data_len, max_n_nodes^2)
            adj_vals[0].append(val)
            adj_vals[1].append(ind)
            adj_vals[2].append(row * max_n_nodes + col)

        if (i > 0 and i % k == 0) or i == n_reactions - 1:
            # we save lists of vals/rows/cols to sparse matrices every K reactions to save RAM during featurization
            nodes_mat += sparse.csr_matrix((nodes_vals[0], (nodes_vals[1], nodes_vals[2])),
                                           shape=(data_len, max_n_nodes))
            adj_mat += sparse.csr_matrix((adj_vals[0], (adj_vals[1], adj_vals[2])),
                                         shape=(data_len, max_n_nodes ** 2))
            adj_vals = [], [], []
            nodes_vals = [], [], []

    return n_nodes, nodes_mat, adj_mat


class GatGraphFeaturizer(ReactionFeaturizer):
    """
    Converts reaction as SMILES to a "GAT graph" representation.
    The representation that we used is inspired by https://arxiv.org/abs/1710.10903
    with the exception that we use both bond and atom features for attention in GAT layers
    """

    def __init__(self, n_jobs: int = 1, k: int = 8, key: str = 'gat_graph', global_max_nodes: int = 1024):
        """
        :param n_jobs: number of threads to use for featurizing a dataset
        :param k: a multiplier to use so that the final number of featurization chunks is (n_jobs * k)
        :param key: key of the featurizer
        :param global_max_nodes: maximum number of possible nodes in a graph for any reaction
        """
        super(GatGraphFeaturizer, self).__init__()
        self.n_jobs = n_jobs
        self.k = k
        self.key = key
        self.global_max_nodes = global_max_nodes

    def dir(self, feat_dir: str) -> str:
        return os.path.join(feat_dir, self.key)

    def has_finished(self, feat_dir: dir) -> bool:
        this_feat_dir = self.dir(feat_dir)
        return all(os.path.exists(get_path(this_feat_dir))
                   for get_path in (_get_adj_path, _get_nodes_path, _get_n_nodes_path))

    def _get_max_number_of_nodes(self, dataset: Dataset):
        """
        :param dataset: a Dataset to featurize
        :return: maximum number of nodes in a reaction graph for the featurized dataset
        """
        if 'max_n_nodes' in dataset.meta_info:
            # if the dataset has information about the maximum number of nodes, we use it
            max_n_nodes = dataset.meta_info['max_n_nodes']
            max_n_nodes = min(max_n_nodes, self.global_max_nodes)
        else:
            # if the dataset does not have information about the maximum number of nodes, we use some "global" maximum
            max_n_nodes = self.global_max_nodes

        return max_n_nodes

    def _get_chunk_ids(self, data_len: int) -> List[np.ndarray]:
        # we use k times as many chunks as we have threads to avoid delays by longer running threads
        n_chunks = self.n_jobs * self.k if self.n_jobs > 1 else 1
        chunk_size = int(data_len / n_chunks)
        chunk_ends = [chunk_size * i for i in range(n_chunks + 1)]
        chunk_ends[-1] = data_len
        return [np.arange(chunk_ends[i], chunk_ends[i + 1]) for i in range(len(chunk_ends) - 1)]

    def featurize_dataset(self, dataset: Dataset):
        data = dataset.load_x()
        for required_field in ['product', 'substrates']:
            if required_field not in data:
                raise NotImplementedError()

        feat_dir = self.dir(dataset.feat_dir)
        if not os.path.exists(feat_dir):
            os.makedirs(feat_dir)

        max_n_nodes = self._get_max_number_of_nodes(dataset)
        print("Max. number of nodes: {}".format(max_n_nodes))

        # split data into "chunks" that are featurized in separate threads
        data_len = len(data['substrates'])
        chunk_idx = self._get_chunk_ids(data_len)
        parallel_args = []
        for i, idx in enumerate(chunk_idx):
            new_x = dict((k, x.values[idx]) for k, x in data.items())
            parallel_args.append((i, data_len, idx, new_x, max_n_nodes))

        # if there is only one job, do not use thread pool
        if self.n_jobs == 1:
            chunk_results = [gat_featurize_parallel(parallel_args[0])]
        else:
            pool = Pool(self.n_jobs)
            chunk_results = pool.map(gat_featurize_parallel, parallel_args)

        # vector of number of nodes for each reaction
        # we need it to determine zero-padding for each reaction
        n_nodes = np.zeros(data_len, dtype=int)

        # sparse matrices for node features and edge features
        # scipy.sparse matrices can only have 2D shape, so we use (data_len, max_n_nodes^2) for storing edge information
        nodes_mat = sparse.csr_matrix(([], ([], [])), shape=(data_len, max_n_nodes))
        adj_mat = sparse.csr_matrix(([], ([], [])), shape=(data_len, max_n_nodes ** 2))

        for idx, (chunk_n_nodes, chunk_nodes, chunk_adj) in tqdm(zip(chunk_idx, chunk_results),
                                                                 desc='merging reactions from chunks',
                                                                 total=len(chunk_idx)):
            n_nodes[idx] = chunk_n_nodes
            nodes_mat += chunk_nodes
            adj_mat += chunk_adj

        print("Saving featurized data")
        np.savez(_get_n_nodes_path(feat_dir), n=n_nodes)
        sparse.save_npz(_get_nodes_path(feat_dir), nodes_mat)
        sparse.save_npz(_get_adj_path(feat_dir), adj_mat)

        print("Saving featurization metadata")
        meta_info = {
            'description': 'Graph representation of molecules with discrete node and edge features',
            'features': ['atom', 'bond'],
            'features_dim': [sum(NODE_OH_DIM), sum(EDGE_OH_DIM)],
            'features_type': ['atom', 'bond'],
            'max_n_nodes': max_n_nodes,
            'format': 'sparse'
        }
        meta_path = self.meta_info_path(dataset.feat_dir)
        with open(meta_path, 'w') as fp:
            json.dump(meta_info, fp, indent=2)

    def featurize_batch(self, metadata_dir: str, batch: dict) -> dict:
        if not all(k in batch for k in ['product', 'substrates']):
            raise ValueError()
        batch_len = len(batch['product'])

        n_nodes = np.zeros(batch_len, dtype=int)
        nodes = []
        adj = []

        for i, (sub_smi, prod_smi) in enumerate(zip(batch['substrates'], batch['product'])):
            sub_mol = safe_mol_from_smiles(sub_smi)
            prod_mol = safe_mol_from_smiles(prod_smi)
            graph = GatReactionGraph((sub_mol, prod_mol), ravel=False)

            n_nodes[i] = len(graph.nodes)
            nodes.append(graph.nodes)
            adj.append((graph.adj_vals, graph.adj_rows, graph.adj_cols))

        max_n_nodes = max(n_nodes)
        nodes_arr = np.zeros((batch_len, max_n_nodes, 2 * len(NODE_OH_DIM)), dtype=int)
        adj_arr = np.zeros((batch_len, max_n_nodes, max_n_nodes, 2 * len(EDGE_OH_DIM)), dtype=int)

        for i, (a, n) in enumerate(zip(adj, nodes)):
            nodes_arr[i, :n_nodes[i]] = n
            for val, row, col in zip(*a):
                adj_arr[i, row, col] = val

        return {
            'n_nodes': n_nodes,
            'atom': nodes_arr,
            'bond': adj_arr
        }

    def load(self, feat_dir: str) -> dict:
        this_feat_dir = self.dir(feat_dir)
        return {
            'n_nodes': np.load(_get_n_nodes_path(this_feat_dir))['n'],
            'atom': sparse.load_npz(_get_nodes_path(this_feat_dir)),
            'bond': sparse.load_npz(_get_adj_path(this_feat_dir))
        }

    def unpack(self, data: dict) -> dict:
        """
        Converts loaded featurized data (numpy/sparse format) to tensors ready to input into a models
        :param data: dictionary with featurized data (as numpy/sparse matrices)
        :return: dictionary with data as tensors
        """
        new_data = dict((k, v) for k, v in data.items())
        n_nodes = data['n_nodes']  # number of nodes in each graph in batch
        batch_max_nodes = max(int(np.max(n_nodes)), 1)

        # trim data to the minimum required size
        nodes = new_data['atom'][:, :batch_max_nodes]
        if hasattr(nodes, 'toarray'):
            nodes = nodes.toarray()
        nodes = nodes.astype(int)

        edges = new_data['bond']
        if hasattr(edges, 'toarray'):
            edges = edges.toarray()

        if len(nodes.shape) == 2:
            max_n = int(np.sqrt(edges.shape[-1]))
            edges = edges.reshape(edges.shape[0], max_n, max_n)
        edges = edges[:, :batch_max_nodes, :batch_max_nodes]
        edges = edges.astype(int)

        # unravel discrete features
        if len(nodes.shape) == 2:
            nodes = GatReactionGraph.unravel_atom_features(nodes)
            nodes = nodes.transpose(1, 2, 0)

            edges = GatReactionGraph.unravel_bond_features(edges)
            edges = edges.transpose(1, 2, 3, 0)

        new_data['atom'] = to_torch_tensor(nodes)
        new_data['bond'] = to_torch_tensor(edges)

        return new_data


if __name__ == "__main__":
    from src_.data.conditions_prediction_dataset import ConditionsPredictionToyTask

    dataset = ConditionsPredictionToyTask()
    featurizer = GatGraphFeaturizer(n_jobs=1)
    data_x = featurizer.load(dataset.feat_dir)
    batch_dict = {col: data_x[col][0: 0 + 100] for col in data_x.keys()}
    X = featurizer.unpack(batch_dict)

    # X2_all = featurizer.unpack(data_x)
    # X2 = {col: X2_all[col][0: 0 + 100, ...] for col in X2_all.keys()}

    a = 2
    print(f"There are {X['atom'].shape[0]} examples in the batch")
    print(f"The maximum number of atoms is {X['atom'].shape[1]} and each atom has {X['atom'].shape[2]} features.")
