import numpy as np
import matplotlib.pyplot as plt

# Generate random data points
np.random.seed(0)
X = np.random.rand(100, 2)  # 100 data points in 2D space

# Define the number of clusters
k = 3

# K-Means function (as previously defined)
def kmeans(X, k, max_iters=100):
    # ... (previous K-Means code)

# Run K-Means clustering
centroids, labels = kmeans(X, k)

# Visualize the results
plt.figure(figsize=(8, 6))

# Define a list of colors for the clusters
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

# Plot data points with different colors for each cluster
for i in range(k):
    cluster_points = X[labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i + 1}', c=colors[i])

# Plot centroids with 'X' marker and larger size
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, c='black', label='Centroids')

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('K-Means Clustering with Cluster Colors')
plt.legend()
plt.grid(True)

plt.show()
