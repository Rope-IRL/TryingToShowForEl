import numpy as np
import matplotlib.pyplot as plt
from forEl import get_clusters
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

def visualize(max_r):
    data = make_blobs(n_samples=100, n_features=2, centers=3)
    random_points = data[0]

    points, labels = get_clusters(random_points, max_r)
    plt.scatter(random_points[:, 0], random_points[:, 1], c = labels)
    plt.scatter(points[:, 0], points[:, 1], c = range(len(points)), marker = "*", s = 200)
    plt.show()

def main(r):
    # random_points = np.random.randint(0, 100, (100, 2))
    data = make_blobs(n_samples=100, n_features=2, centers=3)
    random_points = data[0]

    points, labels = get_clusters(random_points, r)

    max_acc = adjusted_rand_score(data[1], labels)
    return max_acc


if __name__ == "__main__":
    max_acc = 0
    cnt = 0
    for i in range(1, 500, 1):
        t = main(float(i / 100))
        if t > max_acc:
            max_acc = t
            cnt = i
    print(f"Max acc is {max_acc} i is {cnt}")
    visualize(cnt)

