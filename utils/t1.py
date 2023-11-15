import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Data for plotting
x = np.linspace(0, 2 * np.pi, 100)
y1 = np.sin(x)
z1 = np.cos(x)
z2 = x  # Additional data for the 3D plot

# Create a figure with multiple subplots
fig, axs = plt.subplots(2, 1, figsize=(8, 10))  # 2 rows, 1 column

# Plot on the first subplot
axs[0].plot(x, y1)
axs[0].set_title('Plot 1: Sin(x)')

# Plot on the second subplot as a 3D plot
axs[1] = fig.add_subplot(212, projection='3d')  # 2 rows, 1 column, subplot 2 (3D)
axs[1].plot(x, z1, y1, label='Cos(x) vs Sin(x)')
axs[1].set_title('Plot 2: 3D Plot')
axs[1].set_xlabel('X-axis')
axs[1].set_ylabel('Z-axis')
axs[1].set_zlabel('Y-axis')
axs[1].legend()

plt.tight_layout()  # Adjust layout for better spacing
plt.show()
