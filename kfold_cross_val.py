import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X = np.random.randn(100, 5)
y = np.random.randint(0, 2, size=100)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
model = LogisticRegression()
scores = []

for train_idx, val_idx in kf.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    scores.append(accuracy_score(y_val, preds))

print(f"Mean Accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}")