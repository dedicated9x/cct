from tkinter.font import names
from typing import List
from typing import Optional
from typing import Tuple

import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('TkAgg')
from matplotlib.mathtext import math_to_image

import matplotlib.patches as patches
import numpy as np
import torch
import pickle
import torch.utils.data

# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MnistBox:
    def __init__(
            self,
            x_min: int,
            y_min: int,
            x_max: int,
            y_max: int,
            class_nb: Optional[int] = None,
    ):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.class_nb = class_nb

    @property
    def x_diff(self):
        return self.x_max - self.x_min

    @property
    def y_diff(self):
        return self.y_max - self.y_min

    def __repr__(self):
        return f'Mnist Box: x_min = {self.x_min},' + \
            f' x_max = {self.x_max}, y_min = {self.y_min},' + \
            f' y_max = {self.y_max}. Class = {self.class_nb}'

    def plot_on_ax(self, ax, color: Optional[str] = 'r'):
        ax.add_patch(
            patches.Rectangle(
                (self.y_min, self.x_min),
                self.y_diff,
                self.x_diff,
                linewidth=1,
                edgecolor=color,
                facecolor='none',
            )
        )
        ax.text(
            self.y_min,
            self.x_min,
            str(self.class_nb),
            bbox={"facecolor": color, "alpha": 0.4},
            clip_box=ax.clipbox,
            clip_on=True,
        )

    @property
    def area(self):
        return max((self.x_max - self.x_min), 0) * max((self.y_max - self.y_min), 0)

    def iou_with(self, other_box: "MnistBox"):
        aux_box = MnistBox(
            x_min=max(self.x_min, other_box.x_min),
            x_max=min(self.x_max, other_box.x_max),
            y_min=max(self.y_min, other_box.y_min),
            y_max=min(self.y_max, other_box.y_max),
        )
        return aux_box.area / (self.area + other_box.area - aux_box.area)

class MnistCanvas:
    def __init__(
            self,
            image: np.ndarray,
            boxes: List[MnistBox],
    ):
        self.image = image
        self.boxes = boxes

    def add_digit(
            self,
            digit: np.ndarray,
            class_nb: int,
            x_min: int,
            y_min: int,
            iou_threshold=0.1,
    ) -> bool:
        """
        Add a digit to an image if it does not overlap with existing boxes
        above iou_threshold.
        """
        image_x, image_y = digit.shape
        if x_min >= self.image.shape[0] and y_min >= self.image.shape[1]:
            raise ValueError('Wrong initial corner box')
        new_box_x_min = x_min
        new_box_y_min = y_min
        new_box_x_max = min(x_min + image_x, self.image.shape[0])
        new_box_y_max = min(y_min + image_y, self.image.shape[1])
        new_box = MnistBox(
            x_min=new_box_x_min,
            x_max=new_box_x_max,
            y_min=new_box_y_min,
            y_max=new_box_y_max,
            class_nb=class_nb,
        )
        old_background = self.image[
                         new_box_x_min:new_box_x_max,
                         new_box_y_min:new_box_y_max
                         ]
        for box in self.boxes:
            if new_box.iou_with(box) > iou_threshold:
                return False
        self.image[
        new_box_x_min:new_box_x_max,
        new_box_y_min:new_box_y_max
        ] = np.maximum(old_background, digit)
        self.boxes.append(
            new_box
        )
        return True

    def get_torch_tensor(self, device) -> torch.Tensor:
        np_image = self.image.astype('float32')
        np_image = np_image.reshape(
            (1, 1, self.image.shape[0], self.image.shape[1])
        )
        return torch.from_numpy(np_image).to(device)

    @classmethod
    def get_empty_of_size(cls, size: Tuple[int, int]):
        return cls(
            image=np.zeros(size),
            boxes=[],
        )

    def plot(self, boxes: Optional[List[MnistBox]] = None):
        fig, ax = plt.subplots()
        ax.imshow(self.image)
        boxes = boxes or self.boxes
        for box in boxes:
            box.plot_on_ax(ax)
        plt.show()

def get_mnist_data():
    with open('../../../data/gsn2/mnist_data.pkl', 'rb') as f:
        mnist_data = pickle.load(f)
        return mnist_data

def crop_insignificant_values(digit:np.ndarray, threshold=0.1):
    bool_digit = digit > threshold
    x_range = bool_digit.max(axis=0)
    y_range = bool_digit.max(axis=1)
    start_x = (x_range.cumsum() == 0).sum()
    end_x = (x_range[::-1].cumsum() == 0).sum()
    start_y = (y_range.cumsum() == 0).sum()
    end_y = (y_range[::-1].cumsum() == 0).sum()
    return digit[start_y:-end_y - 1, start_x:-end_x - 1]

def get_random_canvas(
    digits: List[np.ndarray] = None,
    classes: List[int] = None,
    nb_of_digits: Optional[int] = None,
    ):
    nb_of_digits = nb_of_digits if nb_of_digits is not None else np.random.randint(low=3, high=6 + 1)

    new_canvas = MnistCanvas.get_empty_of_size(size=(128, 128))
    attempts_done = 0
    while attempts_done < nb_of_digits:
        current_digit_index = np.random.randint(len(digits))
        current_digit = digits[current_digit_index]
        random_x_min = np.random.randint(0, 128 - current_digit.shape[0] - 3)
        random_y_min = np.random.randint(0, 128 - current_digit.shape[1] - 3)
        if new_canvas.add_digit(
            digit=current_digit,
            x_min=random_x_min,
            y_min=random_y_min,
            class_nb=classes[current_digit_index],
        ):
            attempts_done += 1
    return new_canvas

class ImagesDataset(torch.utils.data.Dataset):
    def __init__(self, split: str):
        super(ImagesDataset, self).__init__()
        (mnist_x_train, mnist_y_train), (mnist_x_test, mnist_y_test) = get_mnist_data()
        for x in mnist_x_train, mnist_y_train, mnist_x_test, mnist_y_test:
            print(type(x), x.shape)

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

        self.split = split

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        if self.split == "train":
            mnist_canvas = get_random_canvas(self.TRAIN_DIGITS,  self.TRAIN_CLASSES)
        else:
            raise NotImplementedError
        return mnist_canvas

if __name__ == '__main__':
    ds = ImagesDataset(split="train")
    mnist_canvas = ds[0]
    mnist_canvas.plot()