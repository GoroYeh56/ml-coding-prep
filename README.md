# Machine Learning Coding Prep (`ml-coding-prep`)

A collection of clean, framework-agnostic (PyTorch/NumPy) implementations of core Machine Learning, Deep Learning, and Reinforcement Learning concepts designed for coding interviews and technical preparation.

---

## 📌 Topic Overview & Recommended Study Order

| # | Topic | Core Concepts / Key Features |
|---|---|---|
| **1** | **K-Means Clustering** | Iterative cluster centroid updates, distance metric optimization, convergence criteria. |
| **2** | **Multi-Head Self-Attention (MHSA)** | Query-Key-Value ($Q, K, V$) projections, scaled dot-product attention, multi-head splitting and merging, attention masking. |
| **3** | **Transformer Components** | Layer Normalization (`LayerNorm`), residual connections, Transformer Encoder block integration. |
| **4** | **Transformer Decoder** | Causal/masked self-attention, cross-attention over encoder outputs, auto-regressive decoding structure. |
| **5** | **Vectorized Convolution** | Efficient 2D Convolution using General Matrix Multiplication (`im2col` / `col2im` GEMM paradigm). |
| **6** | **Pooling Layers** | Max Pooling, Average Pooling, and Top-$k$ Pooling custom implementations. |
| **7** | **Multi-Layer Perceptron (MLP)** | Fully connected feedforward layers, activation functions, linear transformations. |
| **8** | **Variational Autoencoder (VAE)** | Latent variable models, encoder-decoder structure, Gaussian reparameterization trick ($z = \mu + \sigma \odot \epsilon$). |
| **9** | **K-Nearest Neighbors (KNN)** | Vectorized distance calculation ($L_2$/Euclidean distance), $k$-nearest neighbor search, majority vote / regression aggregation. |
| **10** | **Non-Maximum Suppression (NMS)** | Object detection post-processing, Intersection over Union (IoU) calculation, confidence score filtering, bounding box suppression. |
| **11** | **Loss Functions** | Numerical stability: Log-sum-exp trick, Stable Cross-Entropy (CE), InfoNCE loss (contrastive learning), and Focal Loss (class imbalance). |
| **12** | **Tabular Q-Learning** | Model-free Reinforcement Learning, Q-table updates via the Bellman equation, $\epsilon$-greedy exploration policy. |

---

## 🛠️ Key Implementations Summary

### 1. K-Means
- **Objective:** Cluster unlabelled multi-dimensional data into $K$ distinct groups.
- **Key Mechanics:**
  - Center initialization.
  - Vectorized pairwise Euclidean distance computation using matrix operations:
  $$\Vert{}A - B\Vert{}^2 = A^2 + B^2 - 2AB^T$$
  - Vectorized centroid assignment using ` np.add.at(new_clusters, cluster_idx, X)`.
  - Update step computing the mean vector of each cluster.
  - Handle empty clusters.

### 2. Multi-Head Self-Attention (MHSA)
- **Objective:** Compute dependencies between tokens across multiple representation subspaces.
- **Key Formula:**
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### 3. Transformer & Encoder
- **Objective:** Deep sequence modeling architecture.
- **Components:**
  - **LayerNorm:** $\text{LN}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$
  - **Encoder Block:** Residual connection around MHSA and Feed-Forward Networks (FFN).

### 4. Transformer Decoder
- **Objective:** Sequential auto-regressive decoding.
- **Components:**
  - Masked self-attention (prevents attending to future tokens).
  - Encoder-Decoder cross-attention.

### 5. Vectorized Convolution (`im2col`)
- **Objective:** Fast implementation of 2D convolutions without explicit nested Python loops.
- **Technique:** Rearranging sliding image patches into matrix columns (`im2col`) to leverage high-performance General Matrix Multiplication (GEMM).

### 6. Pooling Layers
- **Types Covered:**
  - **Max Pooling:** Extract maximum features within sliding receptive fields.
  - **Average Pooling:** Smooth feature representations across spatial regions.
  - **Top-$k$ Pooling:** Extract top $k$ activated values per region.

### 7. Multi-Layer Perceptron (MLP)
- **Objective:** Standard multi-layer dense neural network.
- **Details:** Weight initialization, linear transformations $W x + b$, non-linear activations (ReLU/GELU).

### 8. Variational Autoencoder (VAE)
- **Objective:** Generative latent variable modeling.
- **Reparameterization Trick:** Enables backpropagation through stochastic nodes by sampling noise $\epsilon \sim \mathcal{N}(0, I)$ outside the computational graph:
  $$z = \mu + \sigma \odot \epsilon$$

### 9. K-Nearest Neighbors (KNN)
- **Objective:** Non-parametric classification/regression.
- **Details:** Vectorized pairwise distance computation using matrix operations:
  $$\Vert{}A - B\Vert{}^2 = A^2 + B^2 - 2AB^T$$

### 10. Non-Maximum Suppression (NMS)
- **Objective:** Remove redundant bounding boxes in object detection.
- **Process:**
  1. Sort boxes by confidence score.
  2. Select highest confidence box and suppress boxes exceeding an IoU threshold.
  3. Iteratively repeat for remaining boxes.

### 11. Loss Functions
- **Numerical Stability Tricks:**
  - **Log-Sum-Exp Trick:** Prevents numerical overflow/underflow when computing Softmax/Cross-Entropy.
  - **InfoNCE Loss:** $L_{InfoNCE} = -\log \frac{\exp(q \cdot k_+ / \tau)}{\sum_i \exp(q \cdot k_i / \tau)}$
  - **Focal Loss:** Addresses hard vs. easy samples and extreme class imbalance:
    $$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

### 12. Q-Learning
- **Objective:** Model-free tabular reinforcement learning.
- **Bellman Equation Update:**
  $$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- NumPy

### Installation
Clone the repository and install necessary packages:

```bash
git clone [https://github.com/GoroYeh56/ml-coding-prep.git](https://github.com/GoroYeh56/ml-coding-prep.git)
cd ml-coding-prep
pip install -r requirements.txt