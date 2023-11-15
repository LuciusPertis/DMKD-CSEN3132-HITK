import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

# Generate random data points for clusters
np.random.seed(0)
points = np.random.rand(10, 2)

# Create Voronoi diagram
vor = Voronoi(points)

# Plot the Voronoi diagram
voronoi_plot_2d(vor, show_vertices=False, show_points=True, line_colors='black', line_width=2, line_alpha=0.6, point_size=10)
plt.scatter(points[:, 0], points[:, 1], c='red', marker='o', s=50, label='Cluster Centers')
plt.legend()
plt.title('2D Voronoi Diagram for Cluster Visualization')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Generate random data points for clusters in 3D
np.random.seed(0)
points_3d = np.random.rand(10, 3)

# Create Voronoi diagram
vor = Voronoi(points_3d)

# Plot the 3D Voronoi diagram
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the Voronoi regions as polygons
for region in vor.regions:
    if not -1 in region and len(region) > 0:
        polygon = Poly3DCollection([vor.vertices[region]])
        ax.add_collection3d(polygon, zs=0, zdir='z')

# Plot the cluster centers
ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='red', marker='o', s=50, label='Cluster Centers')

ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.legend()
plt.title('3D Voronoi Diagram for Cluster Visualization')
plt.show()