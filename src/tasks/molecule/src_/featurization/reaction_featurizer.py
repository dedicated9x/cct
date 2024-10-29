import json
import os
from abc import ABCMeta

from src_.data.dataset import Dataset
from src_.featurization.utils import to_torch_tensor


class ReactionFeaturizer(metaclass=ABCMeta):
    """
    Base class for classes that preprocess reactions (for instance by converting to fingerprint or graphs)
    """

    def __init__(self):
        super(ReactionFeaturizer, self).__init__()

    def load(self, feat_dir: str) -> dict:
        """
        :param feat_dir: base directory for featurized data of a dataset
        :return: dictionary with loaded featurized data
        """
        raise NotImplementedError("Abstract method")

    def dir(self, feat_dir: str) -> str:
        """
        :param feat_dir: base directory for featurized data of a dataset
        :return: path to a directory where featurizer stores data
        """
        raise NotImplementedError("Abstract method")

    def meta_info_path(self, feat_dir) -> str:
        return os.path.join(self.dir(feat_dir), 'info.json')

    def read_meta_info(self, feat_dir) -> dict:
        """
        :param feat_dir: base directory for featurized data of a dataset
        :return: dictionary with loaded featurized data
        """
        path = self.meta_info_path(feat_dir)
        if os.path.exists(path):
            with open(path) as f:
                info = json.load(f)
        else:
            info = {}
        return info

    def featurize_dataset(self, dataset: Dataset):
        """
        Featurizes a dataset of reactions represented as SMILES. Saves featurized dataset.
        Can be me more optimal to use than a multiple usage of 'featurize_reaction' method.
        :param dataset: Dataset to featurize
        """
        raise NotImplementedError("Abstract method")

    def has_finished(self, feat_dir: dir) -> bool:
        """
        :param feat_dir: base directory for featurized data for a dataset
        :return: whether the featurization is already computed and saved
        """
        raise NotImplementedError("Abstract method")

    def featurize_batch(self, metadata_dir: str, batch: dict) -> dict:
        """
        Featurizes a single batch of mapped smiles into list of features
        :param metadata_dir: path to a directory with metadata about the featurizer (e. g. vocabularies)
        :param batch: dictionary of raw reaction inputs needed for this featurization
        :return batch ready to input to a models as a dictionary with numpy/sparse features
        """
        raise NotImplementedError("Abstract method")

    # noinspection PyMethodMayBeStatic
    def unpack(self, data: dict) -> dict:
        """
        Converts loaded featurized data (numpy/sparse format) to tensors ready to input into a models
        :param data: dictionary with featurized data (as numpy/sparse matrices)
        :return: dictionary with data as tensors
        """
        return dict((k, to_torch_tensor(v)) for k, v in data.items())