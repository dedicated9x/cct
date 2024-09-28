
from src.tasks.gsn2.dataset import ImagesDataset


def generate_batch_mnistboxes(n_boxes):
    ds = ImagesDataset(split="train")

    list_boxes = []
    i = 0
    while len(list_boxes) < n_boxes:
        mnist_canvas = ds[i]
        list_boxes += mnist_canvas.boxes
        i += 1
    list_boxes = list_boxes[:n_boxes]
    return list_boxes

def generate_batch_from_cluster(n, selected_anchor_size, anchor_sizes):
    pass




list_boxes = generate_batch_mnistboxes(n_boxes=100)