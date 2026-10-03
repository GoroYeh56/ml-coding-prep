import numpy as np
from convolution_vectorized_forward import get_im2col_indices, vectorized_convolution

def col2im_indices(dXcol, X, kh, kw, pad=0, stride=1):
    """
    Converts dXcol back to dX using scatter-add (np.add.at) to aggregate
    gradients from overlapping receptive fields.
    dXcol: (Cin * kh * kw, N * Hout * Wout))
    """
    N, Cin, H, W = X.shape
    H_pad, W_pad = H + 2 * pad, W + 2 * pad
    dX_pad = np.zeros((N, Cin, H_pad, W_pad), dtype=dXcol.dtype)

    k, i, j, Hout, Wout = get_im2col_indices(X, kh, kw, pad, stride)

    # Reshape dXcol (Cin*kh*kw, N*Hout*Wout) -> (N, Cin*kh*kw, Hout*Wout)
    dXcol_reshaped = dXcol.reshape(Cin * kh * kw, N, Hout * Wout).transpose(1, 0, 2)

    # CRITICAL: Use np.add.at instead of standard indexing to accumulate overlapping gradients!
    np.add.at(dX_pad, (slice(None), k, i, j), dXcol_reshaped)

    if pad == 0:
        return dX_pad
    return dX_pad[:, :, pad:-pad, pad:-pad] # Only return the "Original X shape"


def vectorized_convolution_backward(dY, X, K, Xcol, pad=0, stride=1):
    """
    dY: Gradient of loss w.r.t output Y -> (N, Cout, Hout, Wout)
    Xcol: Saved from forward pass -> (Cin * kh * kw, N * Hout * Wout)
    
    Returns:
        dX: Gradient w.r.t input X -> (N, Cin, H, W)
        dK: Gradient w.r.t filter weights K -> (Cout, Cin, kh, kw)
    """
    N, Cin, H, W = X.shape
    Cout, _, kh, kw = K.shape

    # 1. Reverse final transpose & reshape of forward pass
    # (N, Cout, Hout, Wout) -> (Cout, N * Hout * Wout)
    dY_mat = dY.transpose(1, 0, 2, 3).reshape(Cout, -1)

    # 2. Reconstruct Kmat from K
    Kmat = K.reshape(Cout, -1) # (Cout, Cin * kh * kw)

    # 3. Matrix Multiplication Gradients
    # Y = Kmat @ Xcol  =>  dKmat = dL/dY @ dY/dKmat = dY_mat @ Xcol.T (Cout, N*Hout*Wout) x (N*Hout*Wout, Cin*kh*kw) = (Cout, Cin, kh, kw)
    # dXcol = Kmat.T @ dY_mat (Cin * kh * kw, Cout) @ (Cout, N*Hout*Wout) = (Cin*kh*kw, N*Hout*Wout)
    dKmat = dY_mat @ Xcol.T                      # (Cout, Cin * kh * kw)
    dXcol = Kmat.T @ dY_mat                      # (Cin * kh * kw, N * Hout * Wout)

    # 4. Reshape dKmat back to 4D Filter Gradient
    dK = dKmat.reshape(Cout, Cin, kh, kw)

    # 5. Transform dXcol back to 4D Input Gradient via col2im (N, Cin, H, W)
    dX = col2im_indices(dXcol, X, kh, kw, pad, stride)

    return dX, dK


# ----- Verification Test (Finite Difference Numerical Gradient Check) -----
if __name__ == "__main__":
    np.random.seed(42)
    N, Cin, H, W = 2, 2, 4, 4
    Cout, kh, kw = 2, 3, 3
    pad, stride = 1, 1

    X = np.random.randn(N, Cin, H, W)
    K = np.random.randn(Cout, Cin, kh, kw)
    dY = np.random.randn(N, Cout, 4, 4)

    # Forward
    k, i, j, Hout, Wout = get_im2col_indices(X, kh, kw, pad, stride)
    X_pad = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    Xcol = X_pad[:, k, i, j].transpose(1, 0, 2).reshape(Cin * kh * kw, -1)

    # Backward
    dX, dK = vectorized_convolution_backward(dY, X, K, Xcol, pad=pad, stride=stride)

    # Numerical gradient check for dX[0, 0, 0, 0]
    eps = 1e-5
    X_plus = X.copy(); X_plus[0, 0, 0, 0] += eps
    X_minus = X.copy(); X_minus[0, 0, 0, 0] -= eps

    out_plus = vectorized_convolution(X_plus, K, pad=pad, stride=stride)
    out_minus = vectorized_convolution(X_minus, K, pad=pad, stride=stride)
    
    num_grad_X0 = np.sum((out_plus - out_minus) * dY) / (2 * eps)
    
    print(f"Analytical dX[0,0,0,0]: {dX[0, 0, 0, 0]:.6f}")
    print(f"Numerical  dX[0,0,0,0]: {num_grad_X0:.6f}")
    assert np.isclose(dX[0, 0, 0, 0], num_grad_X0, atol=1e-4)
    print("Gradient Check Passed!")