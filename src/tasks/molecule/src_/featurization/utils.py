# -*- coding: utf-8 -*-
"""
Utility functions for featurization
"""
from typing import List
from typing import Tuple

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.rdchem import Bond, Atom
from rdkit.Chem.rdchem import Mol

# all possible values of atom features found on our training dataset
ATOM_PROPS = {
    'is_supernode': [0, 1],
    'atomic_num': [5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53],
    'formal_charge': [-1, 0, 1],
    'chiral_tag': [0, 1, 2],
    'is_aromatic': [0, 1],
    'degree': [0, 1, 2, 3, 4, 6],
    'explicit_valence': [0, 1, 2, 3, 4, 5, 6],
    'in_ring': [0, 1],
    'num_hs': [0, 1, 2, 3]
}

# all possible values of bond features found on our training dataset
BOND_PROPS = {
    'bond_type': ['self', 'supernode', 1, 2, 3, 12],
    'bond_stereo': [0, 2, 3],
    'bond_dir': [0, 3, 4],
    'in_ring': [0, 1],
}

# helper dictionary from feature values to their one-hot indices
ATOM_PROP2OH = dict((k, (dict((ap, i + 1) for i, ap in enumerate(vals)))) for k, vals in ATOM_PROPS.items())
BOND_PROP2OH = dict((k, (dict((ap, i + 1) for i, ap in enumerate(vals)))) for k, vals in BOND_PROPS.items())

# ordered lists of all reaction features used in the GAT model
ORDERED_ATOM_OH_KEYS = ['is_supernode', 'atomic_num', 'formal_charge', 'chiral_tag',
                        'is_aromatic', 'degree', 'explicit_valence', 'in_ring', 'num_hs']
ORDERED_BOND_OH_KEYS = ['bond_type', 'bond_stereo', 'bond_dir', 'in_ring']

# define some pre-computed atom and bond features for special cases such as 'supernode' or 'self' bond
SUPERNODE_ATOM_FEATURES = np.zeros(len(ORDERED_ATOM_OH_KEYS), dtype=int)
SELF_BOND_FEATURES = np.zeros(len(ORDERED_BOND_OH_KEYS), dtype=int)
SUPERNODE_BOND_FEATURES = np.zeros(len(ORDERED_BOND_OH_KEYS), dtype=int)
NODE_OH_DIM = [len(ATOM_PROPS[feat_key]) + 1 for feat_key in ORDERED_ATOM_OH_KEYS]
EDGE_OH_DIM = [len(BOND_PROPS[feat_key]) + 1 for feat_key in ORDERED_BOND_OH_KEYS]

for i, feat_key in enumerate(ORDERED_ATOM_OH_KEYS):
    if feat_key == 'is_supernode':
        SUPERNODE_ATOM_FEATURES[i] = ATOM_PROP2OH['is_supernode'][1]

for i, feat_key in enumerate(ORDERED_BOND_OH_KEYS):
    if feat_key == 'bond_type':
        SELF_BOND_FEATURES[i] = BOND_PROP2OH['bond_type']['self']
        SUPERNODE_BOND_FEATURES[i] = BOND_PROP2OH['bond_type']['supernode']


def to_torch_tensor(arr, long: bool = False) -> torch.Tensor:
    if not isinstance(arr, np.ndarray):  # sparse matrix
        arr = arr.toarray()
    # noinspection PyUnresolvedReferences
    ten = torch.from_numpy(arr)
    if long:
        ten = ten.long()
    else:
        ten = ten.float()

    if torch.cuda.is_available():
        pass
        # TODO wrocic z tym do normalnosci
        return ten.cuda()
    return ten


def try_get_bond_feature(bond: Bond, feat_key: str) -> int:
    try:
        if feat_key == 'bond_type':
            return int(bond.GetBondType())
        elif feat_key == 'bond_stereo':
            return int(bond.GetStereo())
        elif feat_key == 'bond_dir':
            return int(bond.GetBondDir())
        elif feat_key == 'in_ring':
            return int(bond.IsInRing())
        else:
            raise KeyError(f"Unknown bond feature: {feat_key}")
    except RuntimeError as e:
        print(f'Runtime error why try_get_bond_feature {feat_key}: {str(e)}')
        return 0


def try_get_atom_feature(atom: Atom, feat_key: str) -> int:
    try:
        if feat_key == 'is_supernode':  # this feature is 'manually' set elsewhere
            return 0
        elif feat_key == 'atomic_num':
            return atom.GetAtomicNum()
        elif feat_key == 'chiral_tag':
            return int(atom.GetChiralTag())
        elif feat_key == 'formal_charge':
            return atom.GetFormalCharge()
        elif feat_key == 'degree':
            try:
                return atom.GetDegree()
            except RuntimeError:  # may be "degree not defined for atoms not associated with molecules"
                return 0
        elif feat_key == 'is_aromatic':
            return int(atom.GetIsAromatic())
        elif feat_key == 'explicit_valence':
            return atom.GetExplicitValence()
        elif feat_key == 'in_ring':
            return int(atom.IsInRing())
        elif feat_key == 'num_heavy_neighbors':
            return len(atom.GetNeighbors())
        elif feat_key == 'num_hs':
            return int(atom.GetTotalNumHs())
        else:
            raise KeyError(f"Unknown atom feature: {feat_key}")
    except RuntimeError as e:
        print(f'Runtime error why try_get_atom_feature {feat_key}: {str(e)}')
        return 0


def get_atom_features(atom: Atom) -> List[int]:
    result = []
    for feat_key in ORDERED_ATOM_OH_KEYS:
        val = try_get_atom_feature(atom, feat_key)
        if val not in ATOM_PROP2OH[feat_key]:
            raise ValueError(f'{val} not found in {feat_key}')
        result.append(ATOM_PROP2OH[feat_key][val])
    return result


def get_bond_features(bond: Bond) -> List[int]:
    result = []
    for feat_key in ORDERED_BOND_OH_KEYS:
        val = try_get_bond_feature(bond, feat_key)
        if val not in BOND_PROP2OH[feat_key]:
            raise ValueError(f'{val} not found in {feat_key}')
        result.append(BOND_PROP2OH[feat_key][val])
    return result


def safe_mol_from_smiles(smiles) -> Chem.rdchem.Mol:
    # rdkit returns None or raises an exception if parsing SMILES was unsuccessful
    try:
        mol = Chem.MolFromSmiles(smiles)
    except RuntimeError:
        mol = None

    # make empty mols if parsing SMILES was unsuccessful
    if mol is None:
        mol = Chem.MolFromSmiles('')
    return mol


class GatReactionGraph(object):
    """
    Graph representation a single reaction in the "GAT graph" reaction representation
    """

    def __init__(self, reaction_mols: Tuple[Mol, Mol], supernode: bool = True, ravel: bool = True):
        """
        :param reaction_mols: rdkit Mols for reaction: (substrates, product). Molecules must be correctly mapped.
        :param supernode: whether to add a special 'supernode' connected to all nodes by a special bond
        :param ravel: whether to ravel features
        """

        nodes = {}  # atom map number -> (raveled) atom features
        edges = {}  # (atom map number1, atom map number2) -> (raveled) bond features

        if supernode:
            # first node will always be the 'supernode', other nodes will represent unique atoms in the reaction
            nodes[-1] = np.array([SUPERNODE_ATOM_FEATURES] * len(reaction_mols))
            # add 'self' bond for the 'supernode'
            if (-1, -1) not in edges:
                edges[(-1, -1)] = np.zeros((len(reaction_mols), len(EDGE_OH_DIM)), dtype=int)

        for mol_i, mol in enumerate(reaction_mols):
            if supernode:
                edges[(-1, -1)][mol_i] = SELF_BOND_FEATURES

            for i, a in enumerate(mol.GetAtoms()):
                atom_map = a.GetAtomMapNum()

                atom_features = get_atom_features(a)
                if atom_map not in nodes:
                    nodes[atom_map] = np.zeros((len(reaction_mols), len(NODE_OH_DIM)), dtype=int)
                nodes[atom_map][mol_i] = atom_features

                # add 'self' bond
                if (atom_map, atom_map) not in edges:
                    edges[(atom_map, atom_map)] = np.zeros((len(reaction_mols), len(EDGE_OH_DIM)), dtype=int)
                edges[(atom_map, atom_map)][mol_i] = SELF_BOND_FEATURES
                if supernode:
                    # add 'supernode' bond
                    if (-1, atom_map) not in edges:
                        edges[(-1, atom_map)] = np.zeros((len(reaction_mols), len(EDGE_OH_DIM)), dtype=int)
                    edges[(-1, atom_map)][mol_i] = SUPERNODE_BOND_FEATURES

            for bond in mol.GetBonds():
                a1 = bond.GetBeginAtom().GetAtomMapNum()
                a2 = bond.GetEndAtom().GetAtomMapNum()

                if a2 < a1:
                    a1, a2 = a2, a1

                bond_features = get_bond_features(bond)

                if (a1, a2) not in edges:
                    edges[(a1, a2)] = np.zeros((len(reaction_mols), len(EDGE_OH_DIM)), dtype=int)
                edges[(a1, a2)][mol_i] = bond_features

        # in final matrix representation, each position correspond to an atom mapping number in the reaction graph
        all_maps = sorted(set(nodes.keys()))
        # helper dictionary from atom map number to its index in node vector or row & column index in edge matrices
        all_map2i = dict((m, i) for i, m in enumerate(all_maps))

        # for easier building of sparse matrices, we represent edge matrix as lists of row indices/column indices/values
        self.adj_rows = []
        self.adj_cols = []
        self.adj_vals = []

        # create final vector of node features
        if ravel:
            # we 'ravel' node features, that is convert a 1D node feature vector to a single raveled number
            # this is necessary if we want to store them in sparse matrices which have only 2 dimensions
            self.nodes = np.zeros(len(all_maps), dtype=np.int64)
            for atom_map in nodes.keys():
                raveled_node = GatReactionGraph.ravel_atom_features(nodes[atom_map])
                self.nodes[all_map2i[atom_map]] = raveled_node
        else:
            # if we do not ravel nodes, they are represented as 1D feature vectors
            self.nodes = np.zeros((len(all_maps), 2 * len(NODE_OH_DIM)), dtype=np.int64)
            for atom_map in nodes.keys():
                self.nodes[all_map2i[atom_map]] = nodes[atom_map].flatten()

        for bond_atom_maps in edges.keys():
            if ravel:
                # we 'ravel' edge features, that is convert a 1D edge feature vector to a single raveled number
                # this is necessary if we want to store them in sparse matrices which have only 2 dimensions
                raveled_edge = GatReactionGraph.ravel_bond_features(edges[bond_atom_maps])
            else:
                # if we do not ravel edges, they are represented as 1D feature vectors
                raveled_edge = edges[bond_atom_maps].flatten()

            i1, i2 = all_map2i[bond_atom_maps[0]], all_map2i[bond_atom_maps[1]]
            self.adj_rows.append(i1)
            self.adj_cols.append(i2)
            self.adj_vals.append(raveled_edge)

            if i1 != i2:
                self.adj_rows.append(i2)
                self.adj_cols.append(i1)
                self.adj_vals.append(raveled_edge)

    @staticmethod
    def ravel_atom_features(atom_features: np.array) -> int:
        """
        Turns an OH representation of bond features to a single number
        :param atom_features: array of bond features for a reaction
        :return: a single number representing features ot the atom
        """
        oh_dim = NODE_OH_DIM * atom_features.shape[0]
        raveled = np.ravel_multi_index(atom_features.flatten(), oh_dim)
        return int(raveled)

    @staticmethod
    def unravel_atom_features(atom_features: np.array, n_mols: int = 2) -> np.array:
        """
        Reverses 'unravel_atom_features'
        :param atom_features: bond features encoded as a single number
        :param n_mols: number of molecules that were encoded in the number (for a reaction: 2)
        :return: array of OH numbers representing features of the atom
        """
        oh_dim = NODE_OH_DIM * n_mols
        return np.array(np.unravel_index(atom_features, oh_dim))

    @staticmethod
    def ravel_bond_features(bond_features: np.array) -> int:
        """
        Turns an OH representation of bond features to a single number
        :param bond_features: array of bond features for a reaction
        :return: a single number representing features ot the bond
        """
        oh_dim = EDGE_OH_DIM * bond_features.shape[0]
        raveled = np.ravel_multi_index(bond_features.flatten(), oh_dim)
        return int(raveled)

    @staticmethod
    def unravel_bond_features(bond_features: np.array, n_mols: int = 2) -> np.array:
        """
        Reverses 'ravel_bond_features'
        :param bond_features: bond features encoded as a single number
        :param n_mols: number of molecules that were encoded in the number (for a reaction: 2)
        :return: array of OH numbers representing features of the bond
        """
        oh_dim = EDGE_OH_DIM * n_mols
        return np.array(np.unravel_index(bond_features, oh_dim))
