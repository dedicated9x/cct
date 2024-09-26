import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('TkAgg')
from src.tasks.gsn2.dataset import ImagesDataset, get_mnist_data, crop_insignificant_values

import numpy as np

ds = ImagesDataset(split="train")
xs = [ds.TRAIN_DIGITS[i].shape[0] for i in range(10000)]
ys = [ds.TRAIN_DIGITS[i].shape[1] for i in range(10000)]
show_limit = 5000

fig, ax = plt.subplots(1, 1)

# Create scatter plot
ax.scatter(xs[:show_limit], ys[:show_limit])
# ax.scatter(ys[:show_limit], xs[:show_limit])

# Add titles and labels
ax.set_xlabel('x')
ax.set_ylabel('y')

# Show the plot
plt.show()