import numpy as np

def L1_dist(datapoints, centroids):
    return np.sum(np.abs(X[:, np.newaxis] - centroids), axis=2)

def CosineSim_dist(datapoints, centroids):
    return np.dot(  (datapoints  / np.linalg.norm(datapoints,    axis=1))[:, np.newaxis],
                    centroids   / np.linalg.norm(centroids, axis=1))

def L2_dist(datapoints, centroids):
    return np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)

def CosineSim_dist1(X, new_centroids):
    # Cosine similarity
    def cosine_similarity(u, v):
        dot_product = np.dot(u, v)
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        similarity = dot_product / (norm_u * norm_v)
        return similarity

    cosine_similarities = np.zeros((X.shape[0], new_centroids.shape[0]))

    for i in range(X.shape[0]):
        for j in range(new_centroids.shape[0]):
            cosine_similarities[i, j] = cosine_similarity(X[i], new_centroids[j])
    
    return cosine_similarities


def kmeans_distm(dist_func):

    def kmeans_cp(X, k,  max_iters=100, centroids=[], rnd_seed=None):
        if rnd_seed:    np.random.seed(rnd_seed)
        if len(centroids) == 0:
            centroids = X[np.random.choice(X.shape[0], k, replace=False)]

        distances = dist_func(X, centroids)
        labels = np.argmin(distances, axis=1)
        _labels = labels
        cp = 0

        for i in range(max_iters):
            new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(k)])

            # choose your own distance metric 
            distances = dist_func(X, centroids)
            labels = np.argmin(distances, axis=1)
            cp = np.count_nonzero(labels != _labels)

            # Check for convergence
            if np.all(centroids == new_centroids): break

            centroids = new_centroids
            _labels = labels

        return centroids, labels, [cp]
    
    return kmeans_cp


if __name__ == "__main__":
    import context

    import utils.clusters as disp

    m = 3
    n = 300
    X = np.random.rand(n, m)

    #init_centroids = X[np.random.choice(X.shape[0], k, replace=False)]

    k = 4
    def anno(ext):
        return f'change percentage = {ext[0]}%'

   # disp.cluster_iter_anim(X, k, kmeans_cp, 1, [0,1,2], anno)

    subplots, anims = disp.cluster_anim_multi_func(X, k, 
                        [kmeans_distm(L1_dist), kmeans_distm(L2_dist), kmeans_distm(CosineSim_dist1)],
                        1, [0,1], [anno]*4)
    #plt.show()