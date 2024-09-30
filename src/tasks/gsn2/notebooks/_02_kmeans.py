# TODO napisz to samemu

import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('TkAgg')
from src.tasks.gsn2.dataset import ImagesDataset, get_mnist_data, crop_insignificant_values

import numpy as np
import numpy as np
import matplotlib.pyplot as plt



def iou(box, clusters):
    """
    Oblicza IoU między jednym boxem a klastrami.
    box: numpy array o kształcie (2,) reprezentujący szerokość i wysokość
    clusters: numpy array o kształcie (k, 2) reprezentujący szerokości i wysokości klastrów
    Zwraca: numpy array o kształcie (k,) z IoU między boxem a każdym klastrem
    """
    x = np.minimum(box[0], clusters[:, 0])
    y = np.minimum(box[1], clusters[:, 1])
    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]
    union = box_area + cluster_area - intersection
    iou = intersection / union
    return iou


def kmeans(boxes, k, dist=np.median, max_iter=300):
    """
    Wykonuje klasteryzację k-średnich z metryką opartą na IoU.
    boxes: numpy array o kształcie (n_samples, 2)
    k: liczba klastrów
    dist: funkcja do obliczania nowego centrum klastra (domyślnie np.median)
    max_iter: maksymalna liczba iteracji
    Zwraca: clusters o kształcie (k, 2) oraz przypisania do klastrów
    """
    n_samples = boxes.shape[0]
    distances = np.empty((n_samples, k))
    last_clusters = np.zeros((n_samples,))
    np.random.seed(42)
    # Inicjalizacja klastrów przez losowy wybór k boxów
    clusters = boxes[np.random.choice(n_samples, k, replace=False)]

    for iteration in range(max_iter):
        # Przypisywanie boxów do najbliższego klastra
        for i in range(n_samples):
            distances[i] = 1 - iou(boxes[i], clusters)
        nearest_clusters = np.argmin(distances, axis=1)
        if (last_clusters == nearest_clusters).all():
            break
        # Aktualizacja klastrów
        for cluster in range(k):
            if len(boxes[nearest_clusters == cluster]) == 0:
                continue
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)
        last_clusters = nearest_clusters
    return clusters, nearest_clusters

if __name__ == '__main__':
    ds = ImagesDataset(split="train")
    xs = [ds.TRAIN_DIGITS[i].shape[0] for i in range(10000)]
    ys = [ds.TRAIN_DIGITS[i].shape[1] for i in range(10000)]

    # Konwersja xs i ys do tablic numpy
    xs = np.array(xs)
    ys = np.array(ys)

    # Łączenie xs i ys w jedną tablicę o kształcie (n_samples, 2)
    boxes = np.stack((xs, ys), axis=1)

    # Ustawienie k, liczby klastrów
    k = 5  # Możesz dostosować tę wartość

    # Wykonanie klasteryzacji k-średnich
    clusters, nearest_clusters = kmeans(boxes, k)

    # Wykres punktowy klastrów
    plt.figure(figsize=(10, 6))
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']  # Dodaj więcej kolorów, jeśli k > 10
    for i in range(k):
        cluster_boxes = boxes[nearest_clusters == i]
        plt.scatter(cluster_boxes[:, 0], cluster_boxes[:, 1], c=colors[i % len(colors)], label=f'Klaster {i}')
        plt.scatter(clusters[i][0], clusters[i][1], c='black', marker='x')
        plt.annotate(f'[{clusters[i][0]:.1f}, {clusters[i][1]:.1f}]',
                     (clusters[i][0], clusters[i][1]),
                     textcoords="offset points", xytext=(0, 10), ha='center', color='black')
    plt.xlabel('Szerokość')
    plt.ylabel('Wysokość')
    plt.title('Klasteryzacja Anchor Boxes')
    plt.legend()
    plt.grid(True)
    plt.show()


