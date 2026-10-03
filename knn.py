import numpy as np

class KNNClassifier:
    def __init__(self, k: int = 5, metric: str = "euclidean"):
        self.k = k
        self.metric = metric
        self.X_train = None
        self.y_train = None
        self.classes = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train = X
        self.y_train = y
        self.classes = np.unique(y)
        return self

    def _compute_dists(self, X: np.ndarray) -> np.ndarray:
        """Computes distance matrix between test (N, D) and train (Ntrain, D) -> (N, Ntrain)"""
        if self.metric == "euclidean":
            X_sq = np.sum(X**2, axis=-1, keepdims=True)                   # (N, 1)
            train_sq = np.sum(self.X_train**2, axis=-1, keepdims=True).T   # (1, Ntrain)
            dists_sq = np.maximum(0.0, X_sq + train_sq - 2.0 * (X @ self.X_train.T))
            return np.sqrt(dists_sq)
        
        elif self.metric == "cosine":
            # Pre-normalize vectors: Cosine Dist = 1 - dot(X_norm, Y_norm)
            X_norm = X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-10)
            train_norm = self.X_train / (np.linalg.norm(self.X_train, axis=-1, keepdims=True) + 1e-10)
            return 1.0 - (X_norm @ train_norm.T)
        
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

    def _get_knn_labels(self, X: np.ndarray) -> np.ndarray:
        """Finds top-k neighbor labels for test set (N, D) -> (N, K)"""
        if self.X_train is None:
            raise RuntimeError("Call fit() before predicting!")
        
        dists = self._compute_dists(X)  # (N, Ntrain) pariwise distance between each sample and all Ntrain data
        # O(N) partial selection to get top-k smallest distance indices
        knn_idx = np.argpartition(dists, kth=self.k - 1, axis=1)[:, :self.k]
        return self.y_train[knn_idx]    # (N, K)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """ X: (N, 1) N test samples (each sample is a scalar: x-axis value)
            Broadcast (N, 1, K) == (1, C, 1) -> Boolean mask of shape (N, C, K)
            e.g. N=1, C=4, K=3, knn_labels=[[1,0,1]] --- Broadcast --> [[1,0,1], [1,0,1], [1,0,1,  [1,0,1]]
            For each row (C dimension): (assume C=4, total 4 classes)
            class=0: [[F, T, F]]                            [1]
            class=1:  [T, F, T].     --> sum each row       [2]. --> argmax: class=1
            class=2:  [F, F,F]      to get class votes      [0]
            class=3:  [F, F, F]                             [0]      
        """
        knn_labels = self._get_knn_labels(X)  # (N, K)        
        matches = (knn_labels[:, None, :] == self.classes[None, :, None])

        # Sum matches along K axis to get vote counts per class
        class_counts = matches.sum(axis=-1) # (N, C)

        # Assign class index to each sample in X with max votes
        return self.classes[np.argmax(class_counts, axis=-1)]  # (N,)


    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        N=1, C=3, K=3
        labels: [[0, 1]] --> [[0, 1, 0]            
                            [0, 1, 2] 
                            [1, 1, 1]]
        match_class_mask: (N, C, K) 
        class 0               [[T, F, T]            2/3= [0.67] probability of being class 0: 0.67          
        class 1               [F, T, F]  --> mean:  1/3= [0.33] probability of being class 2: 0.33
        class 2               [F, F, F]]            0/3= [0].   probability of being class 2: 0
        """
        knn_labels = self._get_knn_labels(X)  # (N, K)
        # Broadcast (N, K) against (1, C, 1): each sample, each class, whether the kth vote matches the class.
        match_class_mask = knn_labels[:,None,:] == self.classes[None, :, None] # (N, C, K)
        probs = np.mean(match_class_mask, axis=-1) # Take mean over K votes as the probability.
        return probs  # (N, C)


# ----- Test Verification -----
if __name__ == "__main__":
    np.random.seed(42)
    X_tr, y_tr = np.random.randn(100, 10), np.random.randint(0, 3, size=100)
    X_te = np.random.randn(10, 10)

    knn = KNNClassifier(k=5, metric="cosine").fit(X_tr, y_tr)
    preds = knn.predict(X_te)
    probs = knn.predict_proba(X_te)

    assert preds.shape == (10,)
    assert probs.shape == (10, 3)
    assert np.allclose(probs.sum(axis=1), 1.0)
    print("All tests passed cleanly!")