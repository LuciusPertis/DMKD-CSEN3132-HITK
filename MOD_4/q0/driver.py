import numpy as np

def kmeans_cp(X, k,  max_iters=100, centroids=[], converg_percentage=1):
    if len(centroids) == 0:
        centroids = X[np.random.choice(X.shape[0], k, replace=False)]

    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    labels = np.argmin(distances, axis=1)
    _labels = labels
    cp = 0

    for i in range(max_iters):

        # Update centroids by taking the mean of data points in each cluster
        new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(k)])

        distances = np.linalg.norm(X[:, np.newaxis] - new_centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        cp = np.count_nonzero(labels != _labels)/len(labels)*100

        # Check for convergence
        if np.all(centroids == new_centroids): break

        centroids = new_centroids
        _labels = labels

    return centroids, labels, [cp]


if __name__ == "__main__":
    import context

    import utils.clusters as disp

    m = 3
    n = 1200
    X = np.random.rand(n, m)
    #X[m-1] = X[0]**2 + X[1]**2

    # Define the number of clusters (change this value as needed)
    k = 4
    def anno(ext):
        return f'change percentage = {ext[0]:.5f}%'

    disp.cluster_iter_anim(X, k, kmeans_cp, 1, [0,1,2], anno)