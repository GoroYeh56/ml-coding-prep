import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Binary Cross-Entropy Loss (From Logits for numerical stability)
def bce_loss_with_logits(y_pred_logits: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(y_pred_logits, y_true)

# 2. Categorical Cross-Entropy Loss
def categorical_cross_entropy(y_pred_logits: torch.Tensor, y_true_target_indices: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(y_pred_logits, y_true_target_indices)

# 3. Mean Squared Error (MSE)
def mse_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return torch.mean((y_pred - y_true) ** 2)

# 4. Mean Absolute Error (MAE)
def mae_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(y_pred - y_true))

# 5. Huber Loss (Smooth L1)
def huber_loss(y_pred: torch.Tensor, y_true: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    return F.huber_loss(y_pred, y_true, delta=delta)

# 6. Hinge Loss (Targets must be -1 or 1)
def hinge_loss(y_pred_logits: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.clamp(1.0 - y_true * y_pred_logits, min=0.0))

# 7. KL Divergence Loss
def kl_divergence(p_prob: torch.Tensor, q_log_prob: torch.Tensor) -> torch.Tensor:
    # F.kl_div expects log probabilities for Q
    return F.kl_div(q_log_prob, p_prob, reduction='batchmean')

# 8. Cosine Similarity Loss
def cosine_similarity_loss(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return torch.mean(1.0 - F.cosine_similarity(u, v, dim=-1))

# 9. InfoNCE Loss
def info_nce_loss(query: torch.Tensor, keys: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    # query: (B, D), keys: (B, D) where key[i] is positive for query[i]
    query = F.normalize(query, dim=-1)
    keys = F.normalize(keys, dim=-1)
    
    # Compute similarity matrix: (B, B)
    logits = torch.matmul(query, keys.T) / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, labels)


# 10–13. Bounding Box Losses (IoU, GIoU, DIoU, CIoU)
# Boxes format: (N, 4) in [x1, y1, x2, y2]
def box_iou_variants(boxes1: torch.Tensor, boxes2: torch.Tensor, mode: str = 'ciou') -> torch.Tensor:
    from torchvision.ops import complete_box_iou_loss, distance_box_iou_loss, generalized_box_iou_loss
    
    if mode == 'iou':
        # Standard IoU Loss = 1 - IoU
        x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
        y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
        x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
        y2 = torch.min(boxes1[:, 3], boxes2[:, 3])
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = area1 + area2 - intersection
        iou = intersection / (union + 1e-7)
        return torch.mean(1.0 - iou)
    
    elif mode == 'giou':
        return torch.mean(generalized_box_iou_loss(boxes1, boxes2))
    elif mode == 'diou':
        return torch.mean(distance_box_iou_loss(boxes1, boxes2))
    elif mode == 'ciou':
        return torch.mean(complete_box_iou_loss(boxes1, boxes2))

# 14. DDPM Noise Prediction Loss
def ddpm_loss(model: nn.Module, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, alpha_bar_t: torch.Tensor) -> torch.Tensor:
    # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
    x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1.0 - alpha_bar_t) * noise
    predicted_noise = model(x_t, t)
    return F.mse_loss(predicted_noise, noise)

# 15. Flow Matching Velocity Loss
def flow_matching_loss(model: nn.Module, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor, cond: torch.Tensor = None) -> torch.Tensor:
    # Linear interpolation trajectory: x_t = (1 - t) * x_0 + t * x_1
    # Target velocity vector field: v_target = x_1 - x_0
    t_expanded = t.view(-1, 1, 1, 1) if x_0.ndim == 4 else t.view(-1, 1)
    x_t = (1.0 - t_expanded) * x_0 + t_expanded * x_1
    v_target = x_1 - x_0
    
    predicted_velocity = model(x_t, t, cond)
    return F.mse_loss(predicted_velocity, v_target)