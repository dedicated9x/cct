# -*- coding: utf-8 -*-
"""
Dataset used in the project
"""
import os
from typing import Optional

from src_.data.dataset import Dataset


class ConditionsPredictionToyTask(Dataset):
    """
    Dataset consisting of two types of reactions: otrho-lithiation and electrophilic aromatic substitution.
    This dataset is merged with the main dataset. Original class column is replaced with mechanism column.
    """

    def __init__(self,
                 raw_file_name: str = 'conditions-experiment.csv',
                 overwrite_datum_type: bool = False,
                 overwrite_class: bool = True,
                 verify_data_integrity: bool = True,
                 mask_out_products: bool = False,
                 key: Optional[str] = None):
        self._key = key
        super().__init__()
        self.raw_file_name = raw_file_name
        self.overwrite_datum_type = overwrite_datum_type
        self.overwrite_class = overwrite_class
        self.verify_data_integrity = verify_data_integrity
        self.mask_out_products = mask_out_products

    @property
    def key(self) -> str:
        return self._key or 'conditions_prediction'

    @property
    def raw_data_path(self) -> str:
        return os.path.join(self.feat_dir, self.raw_file_name)

    @property
    def meta_info(self) -> dict:
        return {'max_n_nodes': 256,  # hard-coded so it is large enough (in data the maximum is 206)
                'max_smi_len': 1000}


if __name__ == "__main__":
    import numpy as np

    dataset = ConditionsPredictionToyTask()
    X = dataset.load_x()
    metadata = dataset.load_metadata()
    print(f"Found {X.shape[0]} rows in the dataset.")
    print(f"Proportion of ortho-lithiation reactions: {np.mean(metadata['ortho_lithiation'])}")
