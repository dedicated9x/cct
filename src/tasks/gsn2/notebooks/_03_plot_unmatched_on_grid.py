import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('TkAgg')

from src.tasks.gsn2.anchor_set import AnchorSet
from src.tasks.gsn2.target_decoder import RandomMnistBoxSet


def plot_unmatched_on_grid(box_set):
    # Define a function to calculate the anchor size as a tuple
    def _calculate_size(item):
        return (item.x_max - item.x_min, item.y_max - item.y_min)

    df = pd.DataFrame({'MnistBox': box_set.list_nonmatched})
    df['Size'] = df['MnistBox'].apply(_calculate_size)
    df = df[df['Size'].apply(lambda x: x[0] == 19)]
    df['Center'] = df['MnistBox'].apply(lambda x: x.center)

    xs = df['Center'].apply(lambda x: x[0]).tolist()
    ys = df['Center'].apply(lambda x: x[1]).tolist()

    fig, ax = plt.subplots(1, 1)
    ax.imshow(np.zeros((128, 128)))
    ax.scatter(ys, xs, s=3)

    box_set.anchor_set.grid.plot_on_ax(ax, color='red')
    plt.show()

if __name__ == '__main__':
    anchor_sizes = [
        (19, 19),
        (19, 15),
        (19, 13),
        (19, 11),
        (19, 5),
    ]
    box_set = RandomMnistBoxSet(1000)
    anchor_set = AnchorSet(anchor_sizes, k_grid=3)
    box_set.match_with_anchorset(anchor_set, iou_threshold=0.5)

    plot_unmatched_on_grid(box_set)
    # box_set.plot_next_unmatched(anchor_set)
