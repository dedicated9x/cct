from pathlib import Path
import PIL.Image
import pandas as pd
import omegaconf
import torch.utils.data
from sklearn.model_selection import train_test_split
from torchvision import transforms

class Augmentation:
    def __init__(self, transform_x, transform_y):
        self.transform_x = transform_x
        self.transform_y = transform_y

def get_transform(config: omegaconf.DictConfig, split: str):
    assert split in ["train", "val", "test"]

    # rotation_degrees = [90, 180, 270]
    # rotation_augs = [transforms.RandomRotation(degrees=(d, d)) for d in rotation_degrees]
    # flip_augs = [transforms.RandomVerticalFlip(p=1.0), transforms.RandomHorizontalFlip(p=1.0)]
    #
    # transform = transforms.Compose([
    #     # transforms.RandomRotation(degrees=(90, 90))
    #     transforms.RandomHorizontalFlip(p=1.0)
    # ])

    _transform = transforms.RandomRotation(degrees=(90, 90))

    return _transform


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
        image = PIL.Image.open(self.path_images / row["name"])

        # x_orig = self.transform(image)
        # x = transforms.ToTensor()(x_orig)

        x_orig = transforms.ToTensor()(image)
        x = self.transform(x_orig)

        y = [row[col_name] for col_name in ['squares', 'circles', 'up', 'right', 'down', 'left']]
        y = torch.tensor(y).to(torch.int32)

        sample = {
            "x": x,
            "y": y,
            "filename": row['name']
        }

        debug = True
        if debug:
            sample["x_orig"] = x_orig

        return sample


if __name__ == '__main__':
    import hydra

    import matplotlib.pyplot as plt
    import matplotlib; matplotlib.use('TkAgg')
    from src.common.utils.printing import pprint_sample
    import torch

    # TODO rozwiaz problem 4-ego kanalu na wczesniejszym etapie
    # TODO sprawdzic, w jakiej libce sa zaimplementowane te augmentacje z timm.
    def plot_tensor_as_image(tensor, orig_tensor, labels, filename):
        # Ensure both tensors are on the CPU and convert to numpy
        tensor = tensor.cpu().numpy()
        orig_tensor = orig_tensor.cpu().numpy()

        # Take the first three channels and stack them along the last axis to form RGB images
        rgb_tensor = tensor[:3, :, :].transpose(1, 2, 0)
        rgb_orig_tensor = orig_tensor[:3, :, :].transpose(1, 2, 0)

        # Clip the values to be between 0 and 1, if necessary
        rgb_tensor = rgb_tensor.clip(0, 1)
        rgb_orig_tensor = rgb_orig_tensor.clip(0, 1)

        # Create a subplot with 1 row and 2 columns to show both images side by side
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # Plot the transformed image
        ax[0].imshow(rgb_tensor)
        ax[0].axis('off')  # Turn off axis labels
        ax[0].set_title('Transformed Image', fontsize=12)

        # Plot the original image
        ax[1].imshow(rgb_orig_tensor)
        ax[1].axis('off')  # Turn off axis labels
        ax[1].set_title('Original Image', fontsize=12)

        # Set a super title for the figure with filename and labels
        title = f"Filename: {filename}\nLabels: {labels.tolist()}"
        fig.suptitle(title, fontsize=14)

        # Display the plot and wait for a key or mouse button press
        plt.show(block=False)
        plt.waitforbuttonpress()  # Wait until a key or mouse click is pressed
        plt.close()  # Close the current figure after keypress/mouse click

    @hydra.main(version_base="1.2", config_path="conf", config_name="base")
    def _display_dataset(config: omegaconf.DictConfig) -> None:
        config.paths.root = str(Path(__file__).parents[3])
        ds = ImagesDataset(
            config=config,
            split="val"
        )
        for idx in range(len(ds)):
            sample = ds[idx]
            pprint_sample(sample)
            plot_tensor_as_image(sample['x'], sample['x_orig'], sample['y'], sample['filename'])
            # break

    _display_dataset()

# TODO znajdz miejsce na augmentacje