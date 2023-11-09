import numpy as np

def kmeans_basic(X, k, init_centroids=null, max_iters=100):
    # Randomly initialize k centroids
    if init_centroids == null:
        centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    else:
        centroids = init_centroids

    for i in range(max_iters):
        # Assign each data point to the nearest centroid
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        # Update centroids by taking the mean of data points in each cluster
        new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(k)])

        # Check for convergence
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return centroids, labels

# Example usage:
if __name__ == "__main__":
    # Generate random multi-dimensional data points
    np.random.seed(0)
    X = np.random.rand(100, 2)  # 100 data points in 2D space

    k = 3  # Number of clusters
    centroids, labels = kmeans(X, k)

    print("Final centroids:")
    print(centroids)
    print("Cluster labels for each data point:")
    print(labels)
