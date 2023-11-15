import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation

# K-Means function (as previously defined)
def kmeans(X, k, max_iters=100, centroids = [], rnd_seed=None):
    if rnd_seed:    np.random.seed(rnd_seed)
    if len(centroids) == 0:
        centroids = X[np.random.choice(X.shape[0], k, replace=False)]

    for i in range(max_iters):
        # Assign each data point to the nearest centroid
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        # Update centroids by taking the mean of data points in each cluster
        new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(k)])

        # Check for convergence
        if np.all(centroids == new_centroids): break

        centroids = new_centroids

    return centroids, labels, []

def cluster_iter_anim(datapoints, nlabels, kmeans_upd_func, iters_per_anim = 1, disp_dimen=[0,1], getannotation=lambda x,: ""):
    """ will animate 2d or 3d plots scatter plot
        for datapoints using different colors arr-
        ording to "labels" (returned by kmeans_upd_func)

        kmeans_upd_func : prototype should be
            ()( datapoints, nlabels, 
                max_iters=, centroids= ) -> centroids, labels, ext:[]

        disp_dimen : the dimension you want to be
            displayed, can be 2 or 3 only, also
            dimensions are 0-indexed
    """
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    ud = disp_dimen[0:3]
    opt_s = ((20, 10),(10, 5),(5, 1))[(min(2, len(datapoints)//500))][(len(ud))%2]

    
    fig = plt.figure(figsize=(8, 6))
    if len(ud) == 2:    ax = fig.add_subplot()
    else:               ax = fig.add_subplot(111, projection='3d')
    
    centroids, labels, ext = kmeans_upd_func(datapoints, nlabels, iters_per_anim)

    scs = []
    # Visualize the results in 2D or 3D
    if len(ud) == 2:
        for i in range(nlabels):
            scs.append(
                ax.scatter(   datapoints[labels == i][:, ud[0]], datapoints[labels==i][:, ud[1]],
                                label=f'Cluster {i + 1}', c=colors[i], s=opt_s))
        scs.append(
            ax.scatter(centroids[:, ud[0]], centroids[:, ud[1]], marker='X', s=200, c=colors[0:nlabels], label='Centroids')   )
          
        pass 

    elif len(ud) == 3:
        for i in range(nlabels):
            scs.append(
                ax.scatter(   datapoints[labels == i][:, ud[0]], datapoints[labels==i][:, ud[1]], 
                                datapoints[labels == i][:, ud[2]], label=f'Cluster {i + 1}', c=colors[i], s=opt_s))
        scs.append(
            ax.scatter(centroids[:, ud[0]], centroids[:, ud[1]], centroids[:, ud[2]], marker='X', s=100, c=colors[0:nlabels], label='Centroids'))
        

    # Update function for the animation
    def update(frame):
        nonlocal centroids, labels, scs

        centroids, labels, ext = kmeans_upd_func(datapoints, nlabels, iters_per_anim, centroids)

        if len(ud) == 2:
            for i in range(len(scs)-1):
                data = np.hstack([datapoints[labels == i][:, ud[0]][:, np.newaxis], datapoints[labels==i][:, ud[1]][:, np.newaxis]])
                scs[i].set_offsets(data)
            data = np.hstack([centroids[:, ud[0]][:, np.newaxis], centroids[:, ud[1]][:, np.newaxis]])
            scs[-1].set_offsets(data)

        elif len(ud) == 3:
            for i in range(len(scs)-1):
                scs[i]._offsets3d=(datapoints[labels == i][:, ud[0]], datapoints[labels==i][:, ud[1]], 
                                    datapoints[labels == i][:, ud[2]])
            scs[-1]._offsets3d=(centroids[:, ud[0]], centroids[:, ud[1]], centroids[:, ud[2]])
            # Set the view angles for rotation
            ax.view_init(azim=30 + frame)

        plt.title(f'Clustering (frame={frame})\n{getannotation(ext)}')
        return tuple(scs)

    
    plt.legend()
    plt.grid(True)

    interval, num_frames = 1000, 1000
    anim = FuncAnimation(fig, update, frames=range(num_frames), blit=False, repeat=False, interval=interval)

    plt.show()

def cluster_anim_multi_func(datapoints, nlabels, kmeans_upd_funcs:[], iters_per_anim = 1, disp_dimen=[0,1], getannos=[lambda x,: ""]):
    """ creates multi plt.subplot objects for
        2d or 3d plots scatter plot
        for datapoints using different colors arr-
        ording to "labels" (returned by kmeans_upd_func)

        see func: cluster_iter_anim
    """

    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    ud = disp_dimen[0:3]
    opt_s = ((20, 10), (10, 5), (5, 1))[(min(2, len(datapoints)//500))][(len(ud)) % 2]

    fig = plt.figure(figsize=(8, 6))
    subplots = []
    anims = []

    # for initial centroids
    rnd_seed = np.random.randint(1, 10e5)

    for i, kmeans_upd_func in enumerate(kmeans_upd_funcs):
        anno = getannos[i]

        if len(ud) == 2:    ax = fig.add_subplot(2, 2, i + 1)
        else:               ax = fig.add_subplot(2, 2, i + 1, projection='3d')

        centroids, labels, ext = kmeans_upd_func(datapoints, nlabels, iters_per_anim, [], rnd_seed)

        scs = []

        # Visualize the results in 2D or 3D
        if len(ud) == 2:
            for j in range(nlabels):
                scs.append(ax.scatter(datapoints[labels == j][:, ud[0]], datapoints[labels == j][:, ud[1]],
                                        label=f'Cluster {j + 1}', c=colors[j], s=opt_s))
            scs.append(ax.scatter(centroids[:, ud[0]], centroids[:, ud[1]], marker='X', s=200, c=colors[0:nlabels],
                                        label='Centroids'))

        elif len(ud) == 3:
            for j in range(nlabels):
                scs.append(ax.scatter(datapoints[labels == j][:, ud[0]], datapoints[labels == j][:, ud[1]],
                                        datapoints[labels == j][:, ud[2]], label=f'Cluster {j + 1}', c=colors[j], s=opt_s))
            scs.append(ax.scatter(centroids[:, ud[0]], centroids[:, ud[1]], centroids[:, ud[2]], marker='X', s=100,
                                        c=colors[0:nlabels], label='Centroids'))

        plt.legend()
        plt.grid(True)

        subplots.append(ax)

        # Update function for the animation
        def update(frame):
            nonlocal centroids, labels, scs

            centroids, labels, ext = kmeans_upd_func(datapoints, nlabels, iters_per_anim, centroids)

            if len(ud) == 2:
                for j in range(len(scs) - 1):
                    data = np.hstack(
                        [datapoints[labels == j][:, ud[0]][:, np.newaxis], datapoints[labels == j][:, ud[1]][:, np.newaxis]])
                    scs[j].set_offsets(data)
                data = np.hstack([centroids[:, ud[0]][:, np.newaxis], centroids[:, ud[1]][:, np.newaxis]])
                scs[-1].set_offsets(data)

            elif len(ud) == 3:
                for j in range(len(scs) - 1):
                    scs[j]._offsets3d = (
                    datapoints[labels == j][:, ud[0]], datapoints[labels == j][:, ud[1]],
                    datapoints[labels == j][:, ud[2]])
                scs[-1]._offsets3d = (centroids[:, ud[0]], centroids[:, ud[1]], centroids[:, ud[2]])
                # Set the view angles for rotation
                ax.view_init(azim=30 + frame)

            ax.set_title(f'Clustering (frame={frame})\n{anno(ext)}')
            return tuple(scs)

        interval, num_frames = 1000, 1000
        anim = FuncAnimation(fig, update, frames=range(num_frames), blit=False, repeat=False, interval=interval)

        anims.append(anim)

    plt.show()
    return subplots, anims

if __name__ =="__main__":
    m = 5  # Number of dimensions
    n = 900  # Number of data points
    X = np.random.rand(n, m)
    X[m-1] = X[0]**2 + X[1]**2

    # Define the number of clusters (change this value as needed)
    k = 4
    #anno = lambda x:"create my asss"
    cluster_iter_anim(X, k, kmeans, 1, [0,4]) #), anno)

