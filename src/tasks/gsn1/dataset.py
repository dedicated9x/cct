from pathlib import Path
import PIL.Image
import pandas as pd
import omegaconf
import torch
import torch.utils.data
from sklearn.model_selection import train_test_split
from torchvision import transforms


class Augmentation:
    def __init__(self, transform_x, transform_y_scheme):
        self.transform_x = transform_x
        self.transform_y_scheme = transform_y_scheme

    def _transform_y(self, y):
        base_scheme = ['squares', 'circles', 'up', 'right', 'down', 'left']

        new2old = {k: v for k, v in zip(self.transform_y_scheme, base_scheme)}

        dict_counts_orig = {k: v for k, v in zip(base_scheme, y)}
        dict_counts = {k: dict_counts_orig[new2old[k]] for k in base_scheme}

        y = list(dict_counts.values())
        return y

    def __call__(self, x, y):
        x = self.transform_x(x)
        y = self._transform_y(y)
        return x, y


def get_aug(name):
    assert name in ["rotation_90", "rotation_180", "rotation_270", "hflip", "vflip"]

    dict_augs = {
        "rotation_90": Augmentation(
            transforms.RandomRotation(degrees=(90, 90)),
            ['squares', 'circles', 'left', 'up', 'right', 'down']
        ),
        "rotation_180": Augmentation(
            transforms.RandomRotation(degrees=(180, 180)),
            ['squares', 'circles', 'down', 'left', 'up', 'right']
        ),
        "rotation_270": Augmentation(
            transforms.RandomRotation(degrees=(270, 270)),
            ['squares', 'circles', 'right', 'down', 'left', 'up']
        ),
        "hflip": Augmentation(
            transforms.RandomHorizontalFlip(p=1.0),
            ['squares', 'circles', 'up', 'left', 'down', 'right']
        ),
        "vflip": Augmentation(
            transforms.RandomVerticalFlip(p=1.0),
            ['squares', 'circles', 'down', 'right', 'up', 'left']
        )
    }
    return dict_augs[name]

def encode_counts(counts):
    counts = counts.tolist()

    idxs = []
    for index, value in enumerate(counts):
        if value != 0:
            idxs.append(index)

    left_idx, right_idx = idxs
    left_value = counts[left_idx]

    encoding = 15 * (left_value - 1) + int((left_idx * (11 - left_idx)) / 2 + (right_idx - left_idx))
    return encoding

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

        self.visualization_mode = config.dataset.visualization_mode

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = PIL.Image.open(self.path_images / row["name"])
        x_orig = transforms.ToTensor()(image)
        y_counts_orig = [row[col_name] for col_name in ['squares', 'circles', 'up', 'right', 'down', 'left']]

        x, y_counts = x_orig, y_counts_orig

        # Augmentations
        if torch.rand(1).item() < self.aug.prob_rotation:
            aug_name = ["rotation_90", "rotation_180", "rotation_270"][torch.randint(0, 3, (1,)).item()]
            x, y_counts = get_aug(aug_name)(x, y_counts)
        else:
            pass
        if torch.rand(1).item() < self.aug.prob_mirroring:
            aug_name = ["hflip", "vflip"][torch.randint(0, 2, (1,)).item()]
            x, y_counts = get_aug(aug_name)(x, y_counts)
        else:
            pass

        # Channels 0,1,2 are equal, channel 3 has no information.
        x = x[0].unsqueeze(0)
        y_counts = torch.tensor(y_counts).to(torch.int32)

        y_counts_encoded = encode_counts(y_counts)

        y_shapes = (y_counts >= 1).int()

        sample = {
            "x": x,
            "y_shapes": y_shapes,
            "y_counts": y_counts,
            "y_counts_encoded": torch.tensor(y_counts_encoded),
            "filename": row['name']
        }

        if self.visualization_mode:
            # See
            x_orig = x_orig[0].unsqueeze(0)
            y_counts_orig = torch.tensor(y_counts_orig).to(torch.int32)

            sample["x_orig"] = x_orig
            sample["y_counts_orig"] = y_counts_orig

        return sample


if __name__ == '__main__':
    import hydra

    import matplotlib.pyplot as plt
    import matplotlib; matplotlib.use('TkAgg')
    from src.common.utils.printing import pprint_sample
    import torch

    def plot_tensor_as_image(tensor, orig_tensor, labels, orig_labels, filename):
        tensor = tensor.repeat(3, 1, 1)
        orig_tensor = orig_tensor.repeat(3, 1, 1)

        # Ensure both tensors are on the CPU and convert to numpy
        tensor = tensor.cpu().numpy()
        orig_tensor = orig_tensor.cpu().numpy()

        # Take the channels and stack them along the last axis to form RGB images
        rgb_tensor = tensor.transpose(1, 2, 0)
        rgb_orig_tensor = orig_tensor.transpose(1, 2, 0)

        # Clip the values to be between 0 and 1, if necessary
        rgb_tensor = rgb_tensor.clip(0, 1)
        rgb_orig_tensor = rgb_orig_tensor.clip(0, 1)

        # Create a subplot with 2 rows and 2 columns to show both images side by side
        fig, ax = plt.subplots(2, 2, figsize=(12, 12))

        # Plot the transformed image with its label
        ax[0, 0].imshow(rgb_tensor)
        ax[0, 0].axis('off')  # Turn off axis labels
        ax[0, 0].set_title('Transformed Image', fontsize=12)

        # Plot the original image with its label
        ax[0, 1].imshow(rgb_orig_tensor)
        ax[0, 1].axis('off')  # Turn off axis labels
        ax[0, 1].set_title('Original Image', fontsize=12)

        # Display labels for the transformed and original
        ax[1, 0].text(0.5, 0.5, f"Transformed Labels: {labels.tolist()}", ha='center', va='center', fontsize=12)
        ax[1, 0].axis('off')  # Hide axis for labels text
        ax[1, 1].text(0.5, 0.5, f"Original Labels: {orig_labels.tolist()}", ha='center', va='center', fontsize=12)
        ax[1, 1].axis('off')  # Hide axis for labels text

        # Set a super title for the figure with filename
        fig.suptitle(f"Filename: {filename}", fontsize=14)

        # Display the plot and wait for a key or mouse button press
        plt.show(block=False)
        plt.waitforbuttonpress()  # Wait until a key or mouse click is pressed
        plt.close()  # Close the current figure after keypress/mouse click


    # Updated _display_dataset function call to pass both y and y_orig
    @hydra.main(version_base="1.2", config_path="conf", config_name="00_shapes_base")
    def _display_dataset(config: omegaconf.DictConfig) -> None:
        config.paths.root = str(Path(__file__).parents[3])
        config.dataset.visualization_mode = True
        ds = ImagesDataset(
            config=config,
            split="val"
        )

        assert ds[0]['x'].shape == torch.Size([1, 28, 28])

        for idx in range(len(ds)):
            sample = ds[idx]
            pprint_sample(sample)
            plot_tensor_as_image(sample['x'], sample['x_orig'], sample['y_counts'], sample['y_counts_orig'], sample['filename'])
            # break


    _display_dataset()