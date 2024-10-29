
import torch
from pathlib import Path
import omegaconf
import torch.utils.data

from src_.data.conditions_prediction_dataset import ConditionsPredictionToyTask
from src_.featurization.gat_featurizer import GatGraphFeaturizer



class OrthoLithiationDataset(torch.utils.data.Dataset):
    def __init__(self, config: omegaconf.DictConfig, split: str):
        super(OrthoLithiationDataset, self).__init__()

        dataset = ConditionsPredictionToyTask()
        featurizer = GatGraphFeaturizer(n_jobs=1)
        data_x = featurizer.load(dataset.feat_dir)

        self.X_all = featurizer.unpack(data_x)
        self.meta_data = dataset.load_metadata()

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
        ds = OrthoLithiationDataset(
            config=config,
            split="train"
        )

        for idx in range(len(ds)):
            sample = ds[idx]

    _display_dataset()