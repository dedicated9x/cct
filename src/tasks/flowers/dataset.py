from pathlib import Path
import PIL.Image
import pandas as pd
import numpy as np
import scipy.io
import omegaconf
import torch
import torch.utils.data
import timm.data.transforms_factory
from torchvision import transforms  # Add this import



def get_transform(config: omegaconf.DictConfig, split: str):
    """
    Note that these transforms play 2 roles:
    - preprocessing,
    - augmentation (in the case of training)
    """
    assert split in ["train", "val", "test"]

    preprocessing_params = config.dataset.preprocessing
    aug_params = config.dataset.augmentations.get(split)

    _dict = lambda x: omegaconf.OmegaConf.to_container(x, resolve=True)

    transform = timm.data.transforms_factory.create_transform(
        **_dict(preprocessing_params),
        **_dict(aug_params)
    )

    return transform


class FlowersDataset(torch.utils.data.Dataset):
    def __init__(self, config: omegaconf.DictConfig, split: str):
        super(FlowersDataset, self).__init__()
        assert split in ["train", "test", "val"]

        path_data = Path(config.paths.data) / "flowers"

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

        image = PIL.Image.open(self.path_images / row["filename"])
        x = self.transform(image)

        sample = {
            "x": x,
            "y": torch.tensor(row["label"]).to(torch.float32),
        }

        debug = True
        if debug:
            x_orig = transforms.ToTensor()(image)
            sample['x_orig'] = x_orig

        return sample

if __name__ == '__main__':
    import hydra
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib; matplotlib.use('TkAgg')
    from src.common.utils.printing import pprint_sample

    # TODO ustal seeda
    # TODO zrob to w petli.
    # TODO (wywal augmentacje z kolorami, bo chyba to poprawi)

    def visualize_sample(sample, normalization_params: tuple):
        # Convert the tensors `x` and `x_orig` to numpy arrays and transpose to (H, W, C) for visualization
        image_tensor = sample['x']
        image_orig_tensor = sample['x_orig']

        image = image_tensor.permute(1, 2, 0).numpy()
        image_orig = image_orig_tensor.permute(1, 2, 0).numpy()

        # Normalize the transformed image `x` back to the original pixel values
        mean = np.array(normalization_params[0])
        std = np.array(normalization_params[1])
        image = std * image + mean
        image = np.clip(image, 0, 1)

        # Convert the label tensor `y` to a scalar
        label = sample['y'].item()

        # Plot the original image and the transformed image side by side
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # Plot the transformed image `x`
        axs[0].imshow(image)
        axs[0].set_title(f"Transformed Image (Label: {label})")
        axs[0].axis('off')  # Hide axis

        # Plot the original image `x_orig`
        axs[1].imshow(image_orig)
        axs[1].set_title("Original Image")
        axs[1].axis('off')  # Hide axis

        plt.show()

    @hydra.main(version_base="1.2", config_path="conf", config_name="base")
    def _display_dataset(config: omegaconf.DictConfig) -> None:
        config.paths.root = str(Path(__file__).parents[3])
        ds = FlowersDataset(
            config=config,
            split="train"
        )
        # for idx in range(len(ds)):
        sample = ds[0]
        pprint_sample(sample)
        visualize_sample(
            sample,
            normalization_params=(config.dataset.preprocessing.mean, config.dataset.preprocessing.std)
        )

    _display_dataset()
