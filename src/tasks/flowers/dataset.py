from pathlib import Path
import PIL.Image
import pandas as pd
import scipy.io
import omegaconf
import torch
import torch.utils.data
import timm.data.transforms_factory


def get_transform(config: omegaconf.DictConfig, split: str):
    """
    Note that these transforms play 2 roles:
    - preprocessing,
    - augmentation (in the case of training)
    """
    assert split in ["train", "val", "test"]

    pp_params = config.dataset.preprocessing
    aug_params = config.dataset.augmentations.get(split)

    _dict = lambda x: omegaconf.OmegaConf.to_container(x, resolve=True)

    transform = timm.data.transforms_factory.create_transform(
        **_dict(pp_params),
        **_dict(aug_params)
    )

    return transform


class FlowersDataset(torch.utils.data.Dataset):
    def __init__(self, config: omegaconf.DictConfig, split: str):
        super(FlowersDataset, self).__init__()
        assert split in ["train", "test", "val"]

        path_data = Path(config.paths.data)

        self.path_images = path_data / "17flowers" / "jpg"
        list_filenames = (self.path_images / "files.txt").read_text().split("\n")
        self.df = pd.DataFrame().assign(filename=list_filenames)
        self.df["label"] = self.df["filename"].apply(lambda x: divmod(int(x[6:-4]) - 1, 80)[0])

        mask_idxs = scipy.io.loadmat(path_data / "datasplits.mat")[self.get_mat_key(split)]
        mask_idxs = mask_idxs - 1   #Indices in .mat file they start from 1, not from 0.
        mask_dense = np.zeros(self.df.shape[0])
        mask_dense[mask_idxs] = 1
        self.df = self.df[mask_dense.astype(bool)].reset_index(drop=True)
        self.transform = get_transform(config, split)

    def get_mat_key(self, role):
        return {
            "train": "trn1",
            "val": "val1",
            "test": "tst1",
        }[role]

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        x = PIL.Image.open(self.path_images / row["filename"])
        x = self.transform(x)

        return {
            "x": x,
            "y": torch.tensor(row["label"]).to(torch.float32),
        }

if __name__ == '__main__':
    import hydra
    import numpy as np
    import matplotlib.pyplot as plt
    from src.common.utils.printing import pprint_sample
    from src.common.utils.plotting import plt_show_fixed

    # TODO ustal seeda
    # TODO dodaj oryginlny obrazek
    # TODO zrob to w petli.

    def visualize_sample(sample):
        # Convert the tensor `x` to a numpy array and transpose to (H, W, C) for visualization
        image_tensor = sample['x']
        image = image_tensor.permute(1, 2, 0).numpy()

        # Normalize the image back to the original pixel values
        # TODO znajdz, skad on to zabral `norm_constants`
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

        # Convert the label tensor `y` to a scalar
        label = sample['y'].item()

        # Plot the image and the label
        plt.imshow(image)
        plt.title(f"Label: {label}")
        plt.axis('off')  # Hide axis
        plt_show_fixed(plt)
        # plt.show()

    # TODO da sie chyba uproscic `config_path`
    @hydra.main(version_base="1.2", config_path="../../../src/tasks/flowers/conf", config_name="base")
    def _display_sample(config: omegaconf.DictConfig) -> None:
        config.paths.root = str(Path(__file__).parents[3])
        ds = FlowersDataset(
            config=config,
            split="val"
        )
        sample = next(iter(ds))
        pprint_sample(sample)
        visualize_sample(sample)

    _display_sample()
