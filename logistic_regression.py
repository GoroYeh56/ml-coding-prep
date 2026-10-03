import numpy as np

def sigmoid(z):
    # Numerically stable sigmoid (prevents overflow in exp(-z) for large negative z)
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))

def logistic_regression_gd(X, y, lr=0.1, epochs=1000):
    N, D = X.shape
    w = np.zeros(D)
    b = 0.0
    
    for epoch in range(epochs):
        # 1. Forward Pass
        logits = X @ w + b
        y_pred = sigmoid(logits)

        # 2. Compute Loss (BCE) for monitoring
        eps = 1e-15
        y_clipped = np.clip(y_pred, eps, 1 - eps)
        loss = -np.mean(y * np.log(y_clipped) + (1 - y) * np.log(1 - y_clipped))

        # 3. Gradients (dL/dz = y_pred - y)
        dz = (1 / N) * y_pred - y
        dw =  (X.T @ dz)
        db =  np.sum(dz)
        
        # 4. Parameter Updates
        w -= lr * dw
        b -= lr * db

        if epoch % 200 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | w: {np.round(w, 3)} | b: {b:.3f}")
            
    return w, b

# ----- Corrected Test Case -----
if __name__ == "__main__":
    np.random.seed(42)
    N, D = 4, 3
    X = np.random.randn(N, D)
    y = np.array([0, 1, 0, 1], dtype=np.float64)  # Binary targets (0 or 1)

    w, b = logistic_regression_gd(X, y, lr=0.5, epochs=1000)