import torch

def pairwise_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Computes pairwise IoU between N boxes and M boxes.
    
    Args:
        boxes1: (N, 4) in [x1, y1, x2, y2]
        boxes2: (M, 4) in [x1, y1, x2, y2]
        
    Returns:
        IoU matrix of shape (N, M)
    """
    # 1. Intersection Coordinates via Broadcast: (N, 1, 2) vs (1, M, 2) -> (N, M, 2)
    inter_tl = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])  # (N, M, 2)
    inter_br = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])  # (N, M, 2)

    # 2. Intersection Width & Height
    inter_wh = torch.clamp(inter_br - inter_tl, min=0.0)                # (N, M, 2)
    inter_area = inter_wh[:, :, 0] * inter_wh[:, :, 1]                  # (N, M)

    # 3. Individual Box Areas
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1]) # (N,)
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1]) # (M,)

    # 4. Pairwise Union via Broadcast: (N, 1) + (1, M) - (N, M) -> (N, M)
    union = area1[:, None] + area2[None, :] - inter_area               # (N, M)

    # 5. IoU Matrix
    return inter_area / (union + 1e-7)                                  # (N, M)

if __name__ == "__main__":
    # N = 2 predicted boxes
    boxes1 = torch.tensor([
        [0.0, 0.0, 10.0, 10.0],   # Area = 100
        [10.0, 10.0, 20.0, 20.0]  # Area = 100
    ], requires_grad=True)

    # M = 3 target boxes
    boxes2 = torch.tensor([
        [0.0, 0.0, 10.0, 10.0],   # Box 0: Perfect match with boxes1[0]
        [0.0, 0.0, 10.0, 20.0],   # Box 1: 50% overlap with boxes1[0]
        [20.0, 20.0, 30.0, 30.0]  # Box 2: Zero overlap with both
    ])

    # Compute (2, 3) IoU matrix
    iou_matrix = pairwise_box_iou(boxes1, boxes2)
    
    print("Computed IoU Matrix Shape:", iou_matrix.shape)
    print("IoU Matrix Values:\n", iou_matrix)

    # Check shapes
    assert iou_matrix.shape == (2, 3), f"Expected shape (2, 3), got {iou_matrix.shape}"
    
    # Check key expected values
    assert torch.isclose(iou_matrix[0, 0], torch.tensor(1.0)), "boxes1[0] vs boxes2[0] should be 1.0"
    assert torch.isclose(iou_matrix[0, 1], torch.tensor(0.5)), "boxes1[0] vs boxes2[1] should be 0.5"
    assert torch.isclose(iou_matrix[0, 2], torch.tensor(0.0)), "boxes1[0] vs boxes2[2] should be 0.0"

    # Test backprop through the (N, M) matrix
    loss = (1.0 - iou_matrix).mean()
    loss.backward()
    assert boxes1.grad is not None, "Autograd failed!"
    print("Pairwise IoU Test Passed Successfully!")