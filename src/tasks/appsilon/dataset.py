from pathlib import Path
import PIL.Image
import scipy.io
import pandas as pd
import omegaconf
import torch.utils.data
from torchvision import transforms
from torchvision.transforms import RandAugment

class AppsilonDataset(torch.utils.data.Dataset):
    def __init__(self, config: omegaconf.DictConfig, split: str):
        super(AppsilonDataset, self).__init__()
        assert split in ["train", "test", "val"]

        path_images_dir = Path(config.paths.data) / "appsilon/17flowers/jpg"
        list_paths = sorted(list(path_images_dir.glob('*.jpg')))

        # Convert to normal strings
        list_paths = [str(e) for e in list_paths]

        df_all = pd.DataFrame({'path': list_paths})

        def _get_label(x):
            sample_idx = int(x.split("/")[-1].split("_")[1].split(".")[0].lstrip("0"))
            label = int((sample_idx - 1) / 80)
            return label

        df_all['label'] = df_all['path'].apply(_get_label)

        path_splits = Path(config.paths.data) / "appsilon/datasplits.mat"
        mat_data = scipy.io.loadmat(path_splits)
        split_key = {
            "train": "trn",
            "val": "val",
            "test": "tst"
        }[split] + str(config.dataset.split_idx)
        list_idxs = mat_data[split_key][0]

        # Indices should start from0, not from 1
        list_idxs = [idx - 1 for idx in list_idxs]

        self.df = df_all.iloc[list_idxs]

        self.base_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        train_transform = transforms.Compose([
            RandAugment(num_ops=2, magnitude=9),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        if split == "train":
            self.transform = train_transform
        else:
            self.transform = self.base_transform

        self.visualization_mode = config.dataset.visualization_mode


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = PIL.Image.open(row['path']).convert('RGB')

        sample = {}
        if self.visualization_mode:
            image_original = self.base_transform(image)
            sample['image_original'] = image_original

        image = self.transform(image)

        sample = {
            **sample,
            "image": image,
            "label": row['label'],
            "filename": Path(row['path']).name
        }

        return sample


if __name__ == '__main__':
    import hydra
    import matplotlib.pyplot as plt
    import matplotlib; matplotlib.use('TkAgg')
    from src.common.utils.printing import pprint_sample


    def plot_image(image, image_original, label):
        # Convert the tensor to a numpy array and transpose to (H, W, C)
        image = image.permute(1, 2, 0).numpy()
        image_original = image_original.permute(1, 2, 0).numpy()

        # Unnormalize the image (reverse the normalization)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image = image * std + mean
        image_original = image_original * std + mean

        # Clip any values outside [0, 1] range
        image = image.clip(0, 1)
        image_original = image_original.clip(0, 1)

        # Create a figure and axis
        fig, ax = plt.subplots(1, 2)

        # Plot the image on the axis
        ax[0].imshow(image)
        ax[0].set_title("augmented")
        ax[0].axis('off')  # Hide the axes

        ax[1].imshow(image_original)
        ax[1].set_title("original")
        ax[1].axis('off')  # Hide the axes

        fig.suptitle(f"Label: {label}")

        # Display the plot
        plt.show()

    # Updated _display_dataset function call to pass both y and y_orig
    @hydra.main(version_base="1.2", config_path="conf", config_name="00_base")
    def _display_dataset(config: omegaconf.DictConfig) -> None:
        config.paths.root = str(Path(__file__).parents[3])
        config.dataset.visualization_mode = True
        ds = AppsilonDataset(
            config=config,
            split="train"
        )

        for idx in range(len(ds)):
            sample = ds[idx]
            pprint_sample(sample)

            plot_image(sample['image'], sample['image_original'], sample['label'])
            plt.waitforbuttonpress()  # Wait until a key or mouse click is pressed
            plt.close()  # Close the current figure after keypress/mouse click


    _display_dataset()