import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D visualization

# Generate random data points with m dimensions
np.random.seed(0)
m = 5  # Number of dimensions
n = 100  # Number of data points
X = np.random.rand(n, m)

# Define the number of clusters (change this value as needed)
k = 4

# K-Means function (as previously defined)
def kmeans(X, k, max_iters=100):
    # Randomly initialize k centroids
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]

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


while True:

    # User's choice of dimensions to display (2D or 3D)
    #user_dimensions = [4, 3]  # You can change these indices as needed
    user_dimensions = [ int(x) for x in input("user_dimensions[space-separated] = ").split(' ') ]


    # Randomly choose dimensions for projection
    all_dimensions = list(range(m))
    projected_dimensions = [d for d in all_dimensions if d not in user_dimensions]
    random_projection = np.random.rand(n, len(projected_dimensions))

    # Combine user-selected dimensions and random projection
    projected_X = np.hstack((X[:, user_dimensions], random_projection))

    # Run K-Means clustering in the projected space
    centroids, labels = kmeans(projected_X, k)

    # Visualize the results in 2D or 3D
    if len(user_dimensions) == 2:
        # 2D visualization
        plt.figure(figsize=(8, 6))
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        for i in range(k):
            cluster_points = projected_X[labels == i]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i + 1}', c=colors[i])
        plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, c=colors[0:k], label='Centroids')
        plt.xlabel(f'Dimension {user_dimensions[0]}')
        plt.ylabel(f'Dimension {user_dimensions[1]}')
        plt.title('2D K-Means Clustering')
        plt.legend()
        plt.grid(True)
        plt.show()
    elif len(user_dimensions) == 3:
        # 3D visualization
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        for i in range(k):
            cluster_points = projected_X[labels == i]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], label=f'Cluster {i + 1}', c=colors[i])
        ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], marker='X', s=200, c=colors[0:k], label='Centroids')
        ax.set_xlabel(f'Dimension {user_dimensions[0]}')
        ax.set_ylabel(f'Dimension {user_dimensions[1]}')
        ax.set_zlabel(f'Dimension {user_dimensions[2]}')
        ax.set_title('3D K-Means Clustering')
        ax.legend()
        plt.show()
