import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (N, C) unnormalized predictions
        targets: (N,) target class indices
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none') # (N,)
        p_t = torch.exp(-ce_loss)                                    # (N,) probability of correct class
        focal_loss = self.alpha * ((1.0 - p_t) ** self.gamma) * ce_loss
        return focal_loss.mean()

# Verification
logits = torch.tensor([[2.0, 0.5], [-1.0, 3.0]], requires_grad=True) # 2 samples, 2 classes
targets = torch.tensor([0, 1])
loss = FocalLoss(alpha=0.25, gamma=2.0)(logits, targets)
loss.backward()
print("Focal Loss:", loss.item())
