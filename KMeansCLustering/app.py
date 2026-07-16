from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from kmeans import kmeans

centroids = [[-5, -5], [5, 5],[0, 0]]
cluster_std = [1.0, 1.0,1.0]

X, y = make_blobs(
    n_samples=100,
    centers=centroids,
    cluster_std=cluster_std,
    random_state=0
)

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap="viridis")
plt.title("Original Data")
plt.show()

km = kmeans(n_clusters=2, max_iters=100)
y_means = km.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_means, s=50, cmap="viridis")
plt.scatter(
    km.centroids[:, 0],
    km.centroids[:, 1],
    s=200,
    c="red",
    marker="X",
    label="Centroids"
)
plt.scatter(
    km.centroids[:, 0],
    km.centroids[:, 1],
    s=200,
    c="red",
    marker="X",
    label="Centroids"
)
plt.legend()
plt.title("K-Means Clustering")
plt.show()