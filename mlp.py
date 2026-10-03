import numpy as np

class MLP:
    def __init__(self, D, M):
        self.W1 = np.random.randn(D, M) * 0.1
        self.b1 = np.zeros(M)

        self.W2 = np.random.randn(M, M) * 0.1
        self.b2 = np.zeros(M)

    def forward(self, X):
        self.X = X

        self.Z1 = X @ self.W1 + self.b1
        self.A1 = np.maximum(self.Z1, 0)
        self.Y_pred = self.A1 @ self.W2 + self.b2

        return self.Y_pred

    def backward(self, Y_true):
        N, M = Y_true.shape

        dYpred = (2.0 / (N * M)) * (self.Y_pred - Y_true)

        # Linear 2
        dA1 = dYpred @ self.W2.T
        self.dW2 = self.A1.T @ dYpred
        self.db2 = np.sum(dYpred, axis=0)

        # ReLU
        dZ1 = dA1 * (self.Z1 > 0)

        # Linear 1
        self.dW1 = self.X.T @ dZ1
        self.db1 = np.sum(dZ1, axis=0)

        dX = dZ1 @ self.W1.T

        return dX

np.random.seed(42)

N, D, M = 3, 2, 4

X = np.random.randn(N, D)
Y_true = np.random.randn(N, M)

model = MLP(D, M)

# Forward
Y_pred = model.forward(X)

print("Forward:")
print("X       :", X.shape)
print("W1      :", model.W1.shape)
print("b1      :", model.b1.shape)
print("Z1      :", model.Z1.shape)
print("A1      :", model.A1.shape)
print("W2      :", model.W2.shape)
print("b2      :", model.b2.shape)
print("Y_pred  :", Y_pred.shape)

# Loss
loss = np.mean((Y_pred - Y_true) ** 2)
print("Loss    :", loss)
print("Loss shape:", np.shape(loss))

# Backward
dX = model.backward(Y_true)

print("\nBackward:")
print("dYpred :", (model.Y_pred - Y_true).shape)
print("dW1    :", model.dW1.shape)
print("db1    :", model.db1.shape)
print("dW2    :", model.dW2.shape)
print("db2    :", model.db2.shape)
print("dX     :", dX.shape)


def check_gradient(model, X, Y, param, grad, idx, eps=1e-5):
    original = param[idx]

    param[idx] = original + eps
    loss_plus = np.mean((model.forward(X) - Y) ** 2)

    param[idx] = original - eps
    loss_minus = np.mean((model.forward(X) - Y) ** 2)

    param[idx] = original

    numerical = (loss_plus - loss_minus) / (2 * eps)
    analytical = grad[idx]

    print("numerical :", numerical)
    print("analytical:", analytical)
    print("abs error :", abs(numerical - analytical))

np.random.seed(42)

N, D, M = 3, 2, 4

X = np.random.randn(N, D)
Y = np.random.randn(N, M)

model = MLP(D, M)

model.forward(X)
model.backward(Y)

check_gradient(model, X, Y, model.W1, model.dW1, (0, 0))
# check_gradient(model, X, Y, model.W2, model.dW2, (0, 0))
# check_gradient(model, X, Y, model.b1, model.db1, (0,))
# check_gradient(model, X, Y, model.b2, model.db2, (0,))