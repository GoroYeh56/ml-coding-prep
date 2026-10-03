"""Implement K means Clustering"""

import numpy as np

class Kmeans():
    def __init__(self, K=3, tolerance=1e-3, max_iter=1000, random_state=None):
        self.K = K
        self.tolerance=tolerance # centroid covergence threshold
        self.max_iter = max_iter
        self.random_state = random_state # random number generator seed
        self.centroids = None

    def compute_square_dists(self, X, Z):
        """ X:(N,D) Z:(K,D) 
        Return: dists_sq (N,K)
        """
        # Compute pairwise distance between points to each cluster (N,K)
        # Use "squared distance" to avoid sqrt for faster exec. time
        # || X - Z||^2 = |X|^2 + |Z|^2 - 2|X||Z| (Use maximum to avoid negative)
        X_sq = np.sum(X**2, axis=-1, keepdims=True) # (N,)
        Z_sq = np.sum(Z**2, axis=-1, keepdims=True) # (K,)
        dists_sq = np.maximum(0, X_sq + Z_sq.T - 2*X@Z.T) # (N, K)        
        return dists_sq

    def fit(self, X:np.array):
        """Fix K clusters from input array X (N,D)"""
        N,_ = X.shape # N points
        rng = np.random.default_rng(self.random_state)

        # Pick K unique data randomly as the initial centroids.
        rand_idx = rng.choice(N, size=self.K, replace=False)
        self.centroids = X[rand_idx]

        # Iterate at most self.max_iter
        for it in range(self.max_iter):

            # Compute squared distance is FASTER than sqrt().
            sq_dists = self.compute_square_dists(X, self.centroids)
            # Find the closest cluster for each point
            cluster_idx = np.argmin(sq_dists,axis=-1) # (N,)

            # Update new cluster centroids
            new_clusters = np.zeros_like(self.centroids) # (K,D)
            new_cluster_sizes = np.zeros(self.K) # (K,)

            # Assign points to clusters
            # new_clusters[cluster_idx[i]] = X[i]
            np.add.at(new_clusters, cluster_idx, X)            
            np.add.at(new_cluster_sizes, cluster_idx, 1)

            # Handle Empty Cluster
            empty_clusters = (new_cluster_sizes==0)
            new_cluster_sizes[empty_clusters] = 1 # Avoid division by zero.
            new_centroids = new_clusters / new_cluster_sizes[:,None] # (K,D)

            if np.any(empty_clusters):
                num_empty_clusters = np.sum(empty_clusters)
                rand_idx = rng.choice(N, size=num_empty_clusters, replace=False)
                new_centroids[empty_clusters] = X[rand_idx]

            # Check for convergence (Use the max norm of centroid shift)
            shift = np.max(np.linalg.norm(new_centroids-self.centroids, axis=-1))
            if np.abs(shift) < self.tolerance:
                break

            # Update cluster centroids
            self.centroids = new_centroids

        # Return
        print(f"After fit(): cluster centroids: {self.centroids}")

    def infer(self, Xtest:np.array)->np.array:
        """Infer the cluster indices from test input Xtest
            Args: Xtest: np.array of shape (N,D)
            Return: cluster indices (N,)
        """
        if self.centroids is None:
            raise RuntimeError("Must fit() before calling infer()! Centroids are None.")

        sq_dists = self.compute_square_dists(Xtest, self.centroids)
        return np.argmin(sq_dists, axis=-1) # (N)



# Test
if __name__ == "__main__":
    N = 100
    D = 2
    K = 3
    X = np.random.rand(N,D) # Random from uniform distribution [0,1]
    kmeans = Kmeans(K)

    Xtest = np.random.rand(3,D)
    # Test infer before fit raise runtime error
    # kmeans.infer(Xtest)

    kmeans.fit(X)
    print(f"Input test data: {Xtest}")
    print(f"Fit clusters: {kmeans.infer(Xtest)}") # Expect shape: (3,)

    # Visualize points and clusters
    from matplotlib import pyplot as plt
    fig = plt.figure()
    plt.scatter(X[:,0], X[:,1], label="Input data", c="black")
    plt.scatter(kmeans.centroids[:,0], kmeans.centroids[:,1], label="Fitted Centroids", c="red")
    plt.scatter(Xtest[:,0], Xtest[:,1], c="blue", label="Test data")
    plt.legend()
    plt.show()