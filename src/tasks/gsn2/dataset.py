from typing import Optional, List

import numpy as np
import torch.utils.data
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('TkAgg')

from src.tasks.gsn2.structures import MnistBox
from src.tasks.gsn2.anchor_set import AnchorSet
from src.tasks.gsn2.structures import crop_insignificant_values, get_random_canvas, get_mnist_data
from src.tasks.gsn2.target_decoder import TargetDecoder

class ImagesDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            split: str,
            size: Optional[int] = None,
            iou_threshold: float = 0.5,
            anchors: List[MnistBox] = None
    ):
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

        self.decoder = TargetDecoder()

        self.split = split
        self.iou_threshold = iou_threshold
        self.anchors = anchors

    def __len__(self):
        if self.split == "val":
            size = len(self.list_samples)
        else:  # self.split == "train"
            size = self.size
        return size

    def get_canvas(self, idx):
        if self.split == "val":
            mnist_canvas = self.list_samples[idx]
        else:   # self.split == "train"
            mnist_canvas = get_random_canvas(self.TRAIN_DIGITS,  self.TRAIN_CLASSES)
        return mnist_canvas

    def __getitem__(self, idx):
        canvas = self.get_canvas(idx)
        target = self.decoder.get_targets(canvas, self.anchors, self.iou_threshold)

        sample = target.as_dict_of_tensors()

        sample['canvas'] = canvas.get_torch_tensor().squeeze(0)
        return sample

if __name__ == '__main__':
    # ds = ImagesDataset(split="train", size=100)
    ds = ImagesDataset(split="val", size=100)
    for i in range(len(ds)):
        mnist_canvas = ds.get_canvas(i)

        fig, ax = plt.subplots()
        mnist_canvas.plot_on_ax(ax)

        plt.show()

        plt.waitforbuttonpress()  # Wait until a key or mouse click is pressed
        plt.close()  # Close the current figure after keypress/mouse click
