import numpy as np

def kmeans_cp(X, k,  max_iters=100, converg_percentage=1):
    # Randomly initialize k centroids
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    change_percentage = 0

    for i in range(max_iters):
        # Assign each data point to the nearest centroid
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        # Update centroids by taking the mean of data points in each cluster
        new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(k)])

        new_labels = np.argmin(distances, axis=1)        

        # percentage change
        change_percentage = np.mean(labels == new_labels)
        if change_percentage < converg_percentage:
            break

        # Check for convergence
        if np.mean(centroids != new_centroids):
            break

        centroids = new_centroids

    return centroids, labels, change_percentage


if __name__ == "__main__":
    import context

    import utils.clusters as disp

    m = 5  # Number of dimensions
    n = 900  # Number of data points
    X = np.random.rand(n, m)
    #X[m-1] = X[0]**2 + X[1]**2

    # Define the number of clusters (change this value as needed)
    k = 4
    def anno(changep):
        return f'change percentage = {changep:.2f}%'
    disp.cluster_iter_anim(X, k, kmeans_cp, 1, [0,4], anno)