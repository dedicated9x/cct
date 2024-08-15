from pathlib import Path
import PIL.Image
import pandas as pd
import numpy as np
import scipy.io
import omegaconf
import torch
import torch.utils.data
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import timm.data.transforms_factory

def get_transform(config: omegaconf.DictConfig, split: str):
    """
    Note that these transforms play 2 roles:
    - preprocessing,
    - augmentation (in the case of training)
    """
    assert split in ["train", "val", "test"]
    #
    # preprocessing_params = config.dataset.preprocessing
    # aug_params = config.dataset.augmentations.get(split)
    #
    # _dict = lambda x: omegaconf.OmegaConf.to_container(x, resolve=True)
    #
    # transform = timm.data.transforms_factory.create_transform(
    #     **_dict(preprocessing_params),
    #     **_dict(aug_params)
    # )

    # Define an identity transformation
    transform = transforms.Compose([
        # This transform simply returns the input as-is
        transforms.Lambda(lambda x: x)
    ])

    return transform


class ImagesDataset(torch.utils.data.Dataset):
    def __init__(self, config: omegaconf.DictConfig, split: str):
        super(ImagesDataset, self).__init__()
        assert split in ["train", "test", "val"]

        self.path_images = Path(config.paths.data) / "gsn1" / "data"
        self.path_csv = self.path_images
        self.aug = config.dataset.augmentations

        df_all = pd.read_csv(self.path_csv / "labels.csv")
        df_train, df_temp = train_test_split(
            df_all,
            train_size=config.dataset.split.train_size,
            random_state=config.dataset.split.seed
        )
        if split == "train":
            self.df = df_train
        else:
            relative_val_size = \
                config.dataset.split.val_size / (config.dataset.split.val_size + config.dataset.split.test_size)

            df_val, df_test = train_test_split(
                df_temp,
                test_size=relative_val_size,
                random_state=config.dataset.split.seed
            )

            if split == "val":
                self.df = df_val
            else:
                self.df = df_test

        self.transform = get_transform(config, split)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x_orig = PIL.Image.open(self.path_images / row["name"])

        # TODO trzeba zrobic transforma, aby zrobilo sie
        x = self.transform(x_orig)

        return {
            "x": transforms.ToTensor()(x),
            "y": torch.tensor(
                [row[col_name] for col_name in ['squares', 'circles', 'up', 'right', 'down', 'left']]
            ).to(torch.int32),
            "filename": row['name']
        }


if __name__ == '__main__':
    import hydra

    import matplotlib.pyplot as plt
    import matplotlib; matplotlib.use('TkAgg')
    from src.common.utils.printing import pprint_sample
    import torch


    def plot_tensor_as_image(tensor, labels, filename):
        # Ensure the tensor is on the CPU and convert to numpy
        tensor = tensor.cpu().numpy()

        # Take the first three channels and stack them along the last axis to form an RGB image
        rgb_tensor = tensor[:3, :, :].transpose(1, 2, 0)

        # Clip the values to be between 0 and 1, if necessary
        rgb_tensor = rgb_tensor.clip(0, 1)

        # Plot the RGB image
        plt.imshow(rgb_tensor)
        plt.axis('off')  # Turn off axis labels

        # Display the filename and labels above the image
        title = f"Filename: {filename}\nLabels: {labels.tolist()}"
        plt.title(title, fontsize=12)
        plt.show()

    @hydra.main(version_base="1.2", config_path="../../../src/tasks/gsn1/conf", config_name="base")
    def _display_dataset(config: omegaconf.DictConfig) -> None:
        config.paths.root = str(Path(__file__).parents[3])
        ds = ImagesDataset(
            config=config,
            split="val"
        )
        for idx in range(len(ds)):
            sample = ds[idx]
            pprint_sample(sample)
            plot_tensor_as_image(sample['x'], sample['y'], sample['filename'])
            break

    _display_dataset()

# TODO znajdz miejsce na augmentacje