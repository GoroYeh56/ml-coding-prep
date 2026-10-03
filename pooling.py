import numpy as np

# Top K Layer
class TopKLayer:
    """
    Given input as a 1 dimensional tensor, keep the top k largest elements and mask out the rest.
    Example: 
    input = [1,2,3,8,0]
    if k = 2, output = [0, 0, 3, 8, 0]
    if k = 3, output = [0, 2, 3, 8, 0]
    """
    def __init__(self, k=5):
        self.k = k
        self.mask = None

    def forward(self, tensor):
        value = sorted(tensor)[-self.k]
        self.mask = tensor >= np.array([value] * len(tensor))
        return tensor * self.mask

    def backward(self, gradient):
        if self.mask is None:
            return None
        return gradient * self.mask
# Tested:
layer = TopKLayer(k=3)
input_tensor = np.array([1,2,3,8,0])
output = layer.forward(input_tensor)
print(output) # [0 2 3 8 0]
gradient = np.array([100,200,300,800,10])
output_grad = layer.backward(gradient)
print(output_grad) # [  0 200 300 800   0]

# Max Pooling Layer
import numpy as np

class MaxPooling2D:
   def __init__(self, k=2, stride=2):
       self.k, self.stride = k, stride

   def forward(self, x):
       self.x = x
       B, C, H, W = x.shape
       Hout = (H - self.size) // self.stride + 1
       Wout = (W - self.size) // self.stride + 1
       out = np.zeros((B, C, Hout, Wout))

       for i in range(Hout):
           for j in range(Wout):
               h_start, w_start = i * self.stride, j * self.stride
               patch = x[:, :, h_start:h_start+self.k, w_start:w_start+self.k]
               out[:, :, i, j] = np.max(patch, axis=(2, 3))
       return out

   def backward(self, dout):
       dx = np.zeros_like(self.x)
       _, _, Hout, Wout = dout.shape
       for i in range(Hout):
           for j in range(Wout):
               h_start, w_start = i * self.stride, j * self.stride
               patch = self.x[:, :, h_start:h_start+self.k, w_start:w_start+self.k]
               # Route gradient only to the max element using a boolean mask
               mask = (patch == np.max(patch, axis=(2, 3), keepdims=True))
               dx[:, :, h_start:h_start+self.k, w_start:w_start+self.k] += mask * dout[:, :, i:i+1, j:j+1]
       return dx




# Average Pooling Layer

class AveragePooling2D:
    def __init__(self, size=2, stride=2):
        self.size, self.stride = size, stride

    def forward(self, x):
        self.x = x
        B, C, H, W = x.shape
        out_H = (H - self.size) // self.stride + 1
        out_W = (W - self.size) // self.stride + 1
        out = np.zeros((B, C, out_H, out_W))

        for i in range(out_H):
            for j in range(out_W):
                h_start, w_start = i * self.stride, j * self.stride
                patch = x[:, :, h_start:h_start+self.size, w_start:w_start+self.size]
                out[:, :, i, j] = np.mean(patch, axis=(2, 3))
        return out

    def backward(self, dout):
        dx = np.zeros_like(self.x)
        _, _, out_H, out_W = dout.shape
        area = self.size * self.size
        for i in range(out_H):
            for j in range(out_W):
                h_start, w_start = i * self.stride, j * self.stride
                # Distribute the gradient equally across the patch window
                dx[:, :, h_start:h_start+self.size, w_start:w_start+self.size] += dout[:, :, i:i+1, j:j+1] / area
        return dx

