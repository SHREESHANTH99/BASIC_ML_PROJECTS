import random
import numpy as np


class kmeans:
    def __init__(self, n_clusters=2, max_iters=100):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.centroids = None

    def fit_predict(self, X):
        # Initialize centroids randomly
        random_index = random.sample(range(X.shape[0]), self.n_clusters)
        self.centroids = X[random_index]

        for _ in range(self.max_iters):
            # Assign each point to the nearest centroid
            cluster_group = self.assign_clusters(X)

            # Store old centroids
            old_centroids = self.centroids.copy()

            # Compute new centroids
            self.centroids = self.move_centroids(X, cluster_group)

            # Stop if centroids do not change
            if np.allclose(old_centroids, self.centroids):
                break

        return cluster_group

    def assign_clusters(self, X):
        cluster_group = []

        for row in X:
            distances = []

            for centroid in self.centroids:
                distance = np.sqrt(np.dot(row - centroid, row - centroid))
                distances.append(distance)

            cluster_group.append(np.argmin(distances))

        return np.array(cluster_group)

    def move_centroids(self, X, cluster_group):
        new_centroids = []

        for cluster in range(self.n_clusters):
            cluster_points = X[cluster_group == cluster]

            # Handle empty cluster
            if len(cluster_points) == 0:
                new_centroids.append(self.centroids[cluster])
            else:
                new_centroids.append(np.mean(cluster_points, axis=0))

        return np.array(new_centroids)