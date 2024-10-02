from typing import Optional

import numpy as np
import torch.utils.data
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('TkAgg')

from src.tasks.gsn2.structures import crop_insignificant_values, get_random_canvas, get_mnist_data

class ImagesDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, size: Optional[int] = None):
        super(ImagesDataset, self).__init__()
        assert split in ["train", "val"]
        assert (split == "val") or (split == "train" and size is not None)

        (mnist_x_train, mnist_y_train), (mnist_x_test, mnist_y_test) = get_mnist_data()

        self.TRAIN_DIGITS = [
            crop_insignificant_values(digit) / 255.0
            for digit_index, digit in enumerate(mnist_x_train[:10000])
        ]
        self.TRAIN_CLASSES = mnist_y_train[:10000]

        self.TEST_DIGITS = [
            crop_insignificant_values(digit) / 255.0
            for digit_index, digit in enumerate(mnist_x_test[:1000])
        ]
        self.TEST_CLASSES = mnist_y_test[:1000]

        if split == "val":
            np.random.seed(42)
            self.list_samples = [
                get_random_canvas(
                    digits=self.TEST_DIGITS,
                    classes=self.TEST_CLASSES,
                )
                for _ in range(256)
            ]
        else: #  split == "train"
            self.size = size

        self.split = split

    def __len__(self):
        if self.split == "val":
            size = len(self.list_samples)
        else:  # self.split == "train"
            size = self.size
        return size

    def __getitem__(self, idx):
        if self.split == "val":
            mnist_canvas = self.list_samples[idx]
        else:   # self.split == "train"
            mnist_canvas = get_random_canvas(self.TRAIN_DIGITS,  self.TRAIN_CLASSES)
        return mnist_canvas

if __name__ == '__main__':
    # ds = ImagesDataset(split="train", size=100)
    ds = ImagesDataset(split="val", size=100)
    for i in range(len(ds)):
        mnist_canvas = ds[i]

        fig, ax = plt.subplots()
        mnist_canvas.plot_on_ax(ax)

        plt.show()

        plt.waitforbuttonpress()  # Wait until a key or mouse click is pressed
        plt.close()  # Close the current figure after keypress/mouse click
