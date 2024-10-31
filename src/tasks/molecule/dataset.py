
import torch
from pathlib import Path
import omegaconf
import torch.utils.data
import numpy as np

from src_.data.conditions_prediction_dataset import ConditionsPredictionToyTask
from src_.featurization.gat_featurizer import GatGraphFeaturizer



class OrthoLithiationDataset(torch.utils.data.Dataset):
    def __init__(self, config: omegaconf.DictConfig, split: str):
        super(OrthoLithiationDataset, self).__init__()

        dataset = ConditionsPredictionToyTask()
        featurizer = GatGraphFeaturizer(n_jobs=1)
        data_x = featurizer.load(dataset.feat_dir)

        # Load dataframe containing targets
        metadata = dataset.load_metadata()

        # Limiting size of dataset (optional). Useful for development purposes.
        if config.dataset.limit_size is not None:
            data_x = {k: v[:config.dataset.limit_size, ...] for k, v in data_x.items()}
            metadata = metadata.iloc[:config.dataset.limit_size]
        else:
            pass

        ## Train/val split
        # Choosing ratio
        train_val_ratio = 0.8

        # Sampling indices
        all_idxs = np.arange(len(data_x['n_nodes']))
        train_idxs = np.random.choice(all_idxs, size=int(train_val_ratio * len(all_idxs)), replace=False)
        train_idxs = np.sort(train_idxs)
        val_idxs = np.setdiff1d(all_idxs, train_idxs)

        if split == "train":
            chosen_idxs = train_idxs
        elif split == "val":
            chosen_idxs = val_idxs
        else:
            raise NotImplementedError

        # Indexing
        metadata = metadata.iloc[chosen_idxs]
        data_x = {
            "n_nodes": data_x['n_nodes'][chosen_idxs],
            "atom": data_x['atom'][chosen_idxs, :],
            "bond": data_x['bond'][chosen_idxs, :],
        }


        self.X_all = featurizer.unpack(data_x)
        self.meta_data = metadata

        self.split = split

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        sample = {
            "n_nodes": self.X_all['n_nodes'][idx],
            "atom": self.X_all['atom'][idx],
            "bond": self.X_all['bond'][idx],
            "y": torch.tensor(self.meta_data.iloc[idx]['ortho_lithiation'], dtype=torch.float32)
        }

        return sample


if __name__ == '__main__':
    import hydra

    # Updated _display_dataset function call to pass both y and y_orig
    @hydra.main(version_base="1.2", config_path="conf", config_name="00_base")
    def _display_dataset(config: omegaconf.DictConfig) -> None:
        config.paths.root = str(Path(__file__).parents[3])
        # config.dataset.limit_size = 500
        ds = OrthoLithiationDataset(
            config=config,
            split="train"
        )

        for idx in range(len(ds)):
            sample = ds[idx]

    _display_dataset()

"""
/home/admin2/Documents/repos/cct/src/tasks/molecule/data/conditions_prediction/feat/conditions-experiment.csv
"""