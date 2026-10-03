"""
The optimal way to implement a vectorized 2D Convolution forward and backward pass in NumPy/Python
is by using the im2col (image-to-column) and col2im transformations.

im2col Get indices:
k: which input channel to use
i: which input row to use (in this channel)
j: which input column to use (in this channel)
So each Xcol[:, entry] = X[:,k,i,j] (First dimension: N(batch size))

Example:
Assume X=(2,4,4), K=(2,2), pad=0, stride=1. Hout=3, Wout=3.
X = [1 2 3 4       K = [w1 w2
     5 6 7 8            w3 w4]
    9 10 11 12
    13 14 15 16]

|   -->  relative to image (sliding window start position).  
v             i = [ 0 0 0 1 1 1 2 2 2              j = [ 0 1 2 0 1 2 0 1 2    
relative            0 0 0 1 1 1 2 2 2                    1 2 3 1 2 3 1 2 3 
within              1 1 1 2 2 2 3 3 3                    0 1 2 0 1 2 0 1 2
kernel              1 1 1 2 2 2 3 3 3 ]                  1 2 3 1 2 3 1 2 3]

k 矩陣 (which channel in)      i 矩陣 (which row)      j 矩陣 (which col)
  Shape: (Cin * kh * kw, 1)    Shape: (Cin * kh * kw, Hout * Wout)    Shape: (Cin * kh * kw, Hout * Wout)

     ┌─────────┐            ┌─────────┐            ┌─────────┐
     │  Ch 0   │            │ i_00... │            │ j_00... │
     │  Ch 0   │            │ i_01... │            │ j_01... │ ──► (y, x) 在原圖上的
     ├─────────┤      +     ├─────────┤     +      ├─────────┤     絕對座標，隨窗口變動
     │  Ch 1   │            │ i_10... │            │ j_10... │
     │  Ch 1   │            │ i_11... │            │ j_11... │
     └─────────┘            └─────────┘            └─────────┘
          │                      │                      │
          └──────────────────────┴──────────────────────┘
                                 │ (Broadcasting 擴展組合) (Cin * kh * kw, Hout * Wout)
                                 ▼
                     X_pad[:, k, i, j]  ──► 一秒抽完所有窗口！

"""

import numpy as np

def get_im2col_indices(X, kh, kw, pad=0, stride=1):
    """
    Return:
        k: (Cin * kw * kh,)
        i: (Cin * kw * kh, Hout * Wout)
        j: (Cin * kw * kh, Hout * Wout)
        Hout: scalar
        Wout: scalar
    """
    N, Cin, H, W = X.shape
    Hout = (H+2*pad-kh)//stride +1 # Key: use integer division(floor)//
    Wout = (W+2*pad-kw)//stride +1
    print(f"Hout {Hout}, Wout {Wout}")
    """
    k: repeat Cin for every kernel cells (kh * kw)
    i: i0 (first col) + j0 (first row) (Broadcast to final shape)
          (Cin * kh * kw, 1).  (1, Hout * Wout)
    j: j0 (first col) + j1 (first row)

    i0,j0: indices within the kernel
    i1,j1: indices of the sliding window top-left w.r.t input X.

    Example: X=(1,1,4,4), K=(1,1,2,2)
    i = 0 0 0 1 1 1 2 2 2   j = 0 1 2 0 1 2 0 1 2
        0 0 0 1 1 1 2 2 2       1 2 3 1 2 3 1 2 3
        1 1 1 2 2 2 3 3 3       0 1 2 0 1 2 0 1 2 
        1 1 1 2 2 2 3 3 3       1 2 3 1 2 3 1 2 3 
    kh=2, kw=2
    Hout=3, Wout=3    
    """
    k = np.repeat(np.arange(Cin), kh * kw).reshape(-1,1) # for Broadcasting

    # First column: Cin * kh * kw
    i0 = np.repeat(np.arange(kh), kw)   # (kh * kw,)
    i0 = np.tile(i0, Cin)               # (Cin * kh * kw,)
    j0 = np.tile(np.arange(kw), kh*Cin) # (Cin * kh * kw,)

    # First row: Hout * Wout
    i1 = stride * np.repeat(np.arange(Hout), Wout) # (Hout x Wout,)
    j1 = stride * np.tile(np.arange(Wout), Hout)   # (Hout x Wout,)

    # Broadcast!
    i = i0.reshape(-1, 1) + i1.reshape(1,-1)
    j = j0.reshape(-1, 1) + j1.reshape(1,-1)

    return k.astype(int), i.astype(int), j.astype(int), Hout, Wout

def vectorized_convolution(X, K, pad=0, stride=1):
    """ Vectorized Convolution via GEMM(Genarl Matrix Multiplication)"""
    N, Cin, H, W = X.shape
    Cout, _, kh, kw = K.shape   

    # Get indices from unpadded X.
    k, i, j, Hout, Wout = get_im2col_indices(X, kh, kw, pad, stride)

    # Pad X along H and W axes
    X_pad = np.pad(X, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant')

    # Prepare Xcol via Advanced Indexing
    Xcol = X_pad[:,k,i,j] # (N, Cin * kh * kw, Hout * Wout)
    # Permute and reshape to the correct shape
    Xcol = Xcol.transpose(1, 0, 2).reshape(Cin*kh*kw, -1) # (Cin*kh*kw, N*Hout*Wout)

    Kmat = K.reshape(Cout, -1) # (Cout, Cin * kh * kw)

    Y = Kmat @ Xcol # (Cout, N * Hout * Wout)
    Y = Y.reshape(Cout, N, Hout, Wout).transpose(1,0,2,3)
    return Y

# ----- Test ------
N, Cin, H, W = 1, 1, 4, 4
Cout, kh, kw = 1, 2, 2
padding = 0
X = np.random.rand(N, Cin, H, W)
K = np.random.rand(Cout, Cin, kh, kw)
out = vectorized_convolution(X, K, pad=padding)

Hout, Wout = 3, 3
assert out.shape == (N, Cout, Hout, Wout), f"Error: expect output shape to be {N},{Cout},{Hout},{Wout} while actual output shape: {out.shape}"
print(f"output: {out}")