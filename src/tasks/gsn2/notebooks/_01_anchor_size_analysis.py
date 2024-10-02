import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('TkAgg')
import pandas as pd
import numpy as np
from src.tasks.gsn2.dataset import ImagesDataset


def plot_sizes_with_counts():
    # Assuming xs and ys are already defined
    ds = ImagesDataset(split="train", size=10000)
    xs = [ds.TRAIN_DIGITS[i].shape[0] for i in range(10000)]
    ys = [ds.TRAIN_DIGITS[i].shape[1] for i in range(10000)]

    # Prepare the result as a series of tuples
    result = np.column_stack((xs, ys))
    series_of_tuples = pd.Series([tuple(row) for row in result])

    # Get value counts of each tuple
    value_counts = series_of_tuples.value_counts()

    fig, ax = plt.subplots()

    # Plot each unique (x, y) pair and annotate with its count
    for (x, y), count in value_counts.items():
        ax.scatter(x, y)
        ax.annotate(str(count), (x, y), textcoords="offset points", xytext=(0, 5), ha='center')

    # Add titles and labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Show the plot
    plt.show()


plot_sizes_with_counts()
