from src.tasks.gsn2.anchor_set import AnchorSet
from src.tasks.gsn2.notebooks._03_plot_unmatched_on_grid import match_with_anchorset, get_random_mnistboxes

ANCHOR_SIZES_GRIDS = {
    "all": [
        (19, 19),
        (19, 18),
        (19, 17),
        (19, 16),
        (19, 15),
        (19, 14),
        (19, 13),
        (19, 12),
        (19, 11),
        (19, 10),
        (19, 9),
        (19, 8),
        (19, 7),
        (19, 6),
        (19, 5),
        (19, 4),
        (19, 3),
        (19, 2),
        (19, 1),
        (11, 19),
        (12, 19),
        (13, 19),
        (14, 19),
        (15, 19),
        (16, 19),
        (17, 19),
        (18, 19),
        (9, 19),
        (18, 18),
        (18, 15),
    ],
    "kmeans": [
        (19, 19),
        (19, 15),
        (19, 13),
        (19, 11),
        (19, 5),
    ]
}


if __name__ == '__main__':
    n_samples = 1000

    list_boxes = get_random_mnistboxes(n_boxes=1000)

    for anchor_sizes_grid in ["kmeans", "all"]:
        for k in [2, 3, 4]:
    # for anchor_sizes_grid in ["kmeans"]:
    #     for k in [3]:

            anchor_sizes = ANCHOR_SIZES_GRIDS[anchor_sizes_grid]
            anchor_set = AnchorSet(anchor_sizes, k_grid=k)

            list_nonmatched = match_with_anchorset(list_boxes, anchor_set.list_mnistboxes, iou_threshold=0.5)

            cover_ratio = 1 - len(list_nonmatched) / n_samples
            print(anchor_sizes_grid, k, cover_ratio)

"""
kmeans 2 0.99
kmeans 3 0.878
kmeans 4 0.257
all 2 0.997
all 3 0.931
all 4 0.30300000000000005
"""