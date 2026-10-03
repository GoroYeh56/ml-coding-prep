# 1. LogSumExp loss
# Cross-Entropy Loss: log-sum-exp
import torch

def log_sum_exp(logits, dim=-1):
    """ Calculates the denominator of the log-softmax.
        stable_logits = xi - xM
        Return xM + log(sum_i(e^(xi - xM)))
        where xM:= the max logit value max(xi)

        logits: (B, C) Batch size, number of classes.
        Return: (B, ) if dim is -1.
    """
    # 1. Find the max value along the specified dimension. 
    # keepdim=True ensures the shape matches the original tensor for broadcasting.
    # torch.max returns (values, indices)
    max_logits, _ = torch.max(logits, dim=dim, keepdims=True) # (B,1)
    print(f"max_logits.shape: {max_logits.shape}")

    # 2. Subtract the max for numerical stability (this prevents overflow)
    stable_logits = logits - max_logits # (B, C)

    # 3. Exponentiate, sum, take the log, and add the max back outside
    return max_logits.reshape(-1) + torch.log(torch.sum(torch.exp(stable_logits), dim=dim)) # (B,)

def stable_cross_entropy_loss(logits, targets):
    """
        logits: (B, C) 
        targets:(B,)   target indices for each batch
        xc: The target logits of each batch. (B,)
        xM: The max logits of each batch. (B,)
        Loss = -log(P_stable_soft_max) = -xc + xM+ log(sum_i(e^(xi-xM))) 
            = [xM+ log(sum_i(e^(xi-xM))] - xc
            = log_sum_exp - xc  
        Return: CE loss (B,)
    """
    B, C = logits.shape
    target_logits = logits[torch.arange(B), targets] # (B,)
    return log_sum_exp(logits) - target_logits

def normal_softmax(logits):
    return torch.exp(logits) / torch.sum(torch.exp(logits), dim=-1, keepdims=True)

def normal_CE_loss(logits, targets):
    """ -log(Pc) where c:= target index
        Pc = exp(xc) / sum_k(exp(xk))

        logits: (B, C)
        targets: (B,) the target indices
        Return: CE loss for each sample in the batch (B,)
    """
    probs = normal_softmax(logits) # (B, C)
    B, C = probs.shape
    target_probs = probs[torch.arange(B), targets]
    return -torch.log(target_probs)

# --- Example Usage ---
# A tensor with very large values that would normally cause e^1000 to overflow to 'inf'
x = torch.tensor([1000.0, 1001.0, 1002.0])
batch_x = torch.tensor([[1000.0, 1001.0, 7002.0],    # --> predicts class 2 
                        [10000.0, 40001.0, 4002.0]]) # --> predicts class 1
targets = torch.tensor([1, 1]) # Assume both examples have Ground-truth 1
print(f"Normal_CE_loss: {normal_CE_loss(batch_x, targets)}")
print(f"Stable Cross entropy loss: {stable_cross_entropy_loss(batch_x, targets)}")

# Min-of-N loss: Encourage Winner Take All!
def min_of_n_loss(pred_trajs, gt_traj):
    """
    pred_trajs: (B, N, T, D)
    gt_traj: (B, T, D)
        B: Batch Size
        N: Number of Candidates
        T: Trajectory Horizon (Numnber of waypoints)
        D: Waypoint dimension (x, y, yaw, speed)
    Return: Average of loss over the batch (1,) scalar
    """
    gt_expanded = gt_traj.unsqueeze(1) # (B, 1, T, D) for broadcasting when computing dists matrix.
    
    # L2 distance across T and D
    dists = torch.sum((pred_trajs - gt_expanded)**2, dim=(2, 3)) # (B, N) Distance between each sample to the GT.
    best_idx = torch.argmin(dists, dim=1) # (B, ) The best candidates from each batch sample.
    
    B = pred_trajs.shape[0]
    best_preds = pred_trajs[torch.arange(B), best_idx] # Select the best candidates (B, T, D)
    
    return torch.mean((best_preds - gt_traj)**2)


# Smooth L1 loss
def smooth_l1_loss(pred, target, beta=1.0):
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 
                       0.5 * diff**2 / beta,
                       diff - 0.5 * beta)
    return torch.mean(loss)


def giou_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    preds, targets: [N, 4] formatted as (x1, y1, x2, y2)
    """
    # Intersection coordinates
    x1_i = torch.max(preds[:, 0], targets[:, 0])
    y1_i = torch.max(preds[:, 1], targets[:, 1])
    x2_i = torch.min(preds[:, 2], targets[:, 2])
    y2_i = torch.min(preds[:, 3], targets[:, 3])

    intersection = (x2_i - x1_i).clamp(min=0) * (y2_i - y1_i).clamp(min=0)

    # Individual Areas
    area_p = (preds[:, 2] - preds[:, 0]) * (preds[:, 3] - preds[:, 1])
    area_t = (targets[:, 2] - targets[:, 0]) * (targets[:, 3] - targets[:, 1])
    union = area_p + area_t - intersection + 1e-7

    iou = intersection / union

    # Smallest Enclosing Box (Convex Hull)
    x1_c = torch.min(preds[:, 0], targets[:, 0])
    y1_c = torch.min(preds[:, 1], targets[:, 1])
    x2_c = torch.max(preds[:, 2], targets[:, 2])
    y2_c = torch.max(preds[:, 3], targets[:, 3])

    area_c = (x2_c - x1_c) * (y2_c - y1_c) + 1e-7

    giou = iou - ((area_c - union) / area_c)
    return (1.0 - giou).mean()

# diou loss
# ciou loss: L_diou + Aspect Ratio



def smooth_l1_loss(preds: torch.Tensor, targets: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    diff = torch.abs(preds - targets)
    loss = torch.where(
        diff < beta,
        0.5 * (diff ** 2) / beta,
        diff - 0.5 * beta
    )
    return loss.mean()

def chamfer_distance(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    # p1: [B, N, 3], p2: [B, M, 3]
    # $\|a - b\|^2 = \|a\|^2 + \|b\|^2 - 2 \langle a, b \rangle$
    p1_sq = torch.sum(p1 ** 2, dim=-1, keepdim=True)  # [B, N, 1]
    p2_sq = torch.sum(p2 ** 2, dim=-1, keepdim=True)  # [B, M, 1]
    
    dist_matrix = p1_sq + p2_sq.transpose(1, 2) - 2.0 * torch.matmul(p1, p2.transpose(1, 2))
    dist_matrix = torch.clamp(dist_matrix, min=0.0)

    # Minimum distance from p1 to p2, and p2 to p1
    min_dist_p1 = torch.min(dist_matrix, dim=2)[0]  # [B, N]
    min_dist_p2 = torch.min(dist_matrix, dim=1)[0]  # [B, M]

    return (min_dist_p1.mean(dim=1) + min_dist_p2.mean(dim=1)).mean()

import numpy as np
from scipy.optimize import linear_sum_assignment

@torch.no_grad()
def hungarian_matching(pred_logits: torch.Tensor, pred_boxes: torch.Tensor, 
                       gt_labels: torch.Tensor, gt_boxes: torch.Tensor, 
                       cost_class: float = 1.0, cost_bbox: float = 1.0):
    """
    pred_logits: [N_queries, C], pred_boxes: [N_queries, 4]
    gt_labels: [N_targets], gt_boxes: [N_targets, 4]
    """
    # 1. Classification cost (Softmax probability for target class)
    out_prob = pred_logits.softmax(dim=-1)
    cost_cls = -out_prob[:, gt_labels]  # [N_queries, N_targets]

    # 2. L1 Bounding box cost
    cost_box = torch.cdist(pred_boxes, gt_boxes, p=1)  # [N_queries, N_targets]

    # Total cost matrix
    C = cost_class * cost_cls + cost_bbox * cost_box
    C = C.cpu().numpy()

    # Bipartite matching via Hungarian Algorithm
    row_ind, col_ind = linear_sum_assignment(C)
    return torch.as_tensor(row_ind, dtype=torch.int64), torch.as_tensor(col_ind, dtype=torch.int64)