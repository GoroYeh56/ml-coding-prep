import numpy as np

def compute_iou_1vsN(best_box: np.ndarray, boxes: np.ndarray)-> np.ndarray:
    """ Return the array of IoU between best_box to all other box in boxes
    best_box: (4) [x1, y1, x2, y2]
    boxes: (N,4) N boxes, each [x1, y1, x2, y2]
    Return:
        iou: (N,)
    """

    # Find intersection coordinates
    inter_topleft = np.maximum(best_box[:2], boxes[:,:2])    # (N, 2) 
    inter_bottomright = np.minimum(best_box[2:], boxes[:,2:])

    # Compute intersection area
    inter_wh = np.maximum(0.0, inter_bottomright - inter_topleft) # (N, 2)
    inter_area = inter_wh[:,0] * inter_wh[:,1]   # (N,)

    # Compute individual box area
    best_box_area = (best_box[2]-best_box[0]) * (best_box[3] - best_box[1])
    boxes_area = (boxes[:,2] - boxes[:,0]) * (boxes[:,3] - boxes[:,1])  # (N,)

    # Union
    unions = best_box_area + boxes_area - inter_area # (N,)

    # Iou = intersection / union+eps (avoid division by 0)
    ious = inter_area / (unions + 1e-9) # (N,)
    return ious

def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """
    Greedy Non-Maximum Suppression (NMS).
    
    Args:
        boxes: (N, 4) containing [x1, y1, x2, y2]
        scores: (N,) confidence scores
        iou_threshold: float (e.g. 0.5)
    
    Returns: 
        List of retained box indices
    """
    # Sort box indices by confidence scores descendingly
    order = scores.argsort()[::-1] # largest score first
    keeps = [] # Final bboxes to be kept.

    # Iteratively remove non-overlapping boxes
    while order.size>=1:

        best_idx = order[0]
        keeps.append(best_idx)
        if order.size==1:
            break

        # Compute IoU of best_idx box against all remaining candidates
        best_box = boxes[best_idx]
        remaining_boxes = boxes[order[1:]]
        ious = compute_iou_1vsN(best_box, remaining_boxes)

        # Remove all boxes that overlaps with the current best_box.
        # Only keep boxes do NOT overlap too much (via masking)
        keepmask = ious < iou_threshold
        order = order[1:][keepmask] # Start from 1: exclude current best_box

    return keeps

# ----- Verification Test -----
if __name__ == "__main__":
    boxes = np.array([
        [10, 10, 50, 50],  # Box 0 (High overlap with Box 1)
        [12, 12, 52, 52],  # Box 1
        [100, 100, 150, 150],  # Box 2 (Separate region)
    ], dtype=np.float32)

    scores = np.array([0.9, 0.85, 0.75])

    keep_indices = nms(boxes, scores, iou_threshold=0.5)
    print("Kept Box Indices:", keep_indices)
    assert keep_indices == [0, 2], f"Expected [0, 2], got {keep_indices}"
    print("NMS Test Passed!")



def box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """ Return the IoU between boxes1 and boxes2 (N,M) 
        Compute the IoU score using vectorization
    """
    # boxes format: [x1, y1, x2, y2] Top-left, Bottom-right
    # boxes1: (N, 4), boxes2: (M, 4)

    # Compute intersection top-left & bottom right coordinates
    # Top-left: take np.minimum
    # Bottom-right: take np.maximum
    inter_top_left = np.maximum(boxes1[:,None,:2], boxes2[:,:2])
    inter_bottom_right = np.minimum(boxes1[:,None,2:], boxes2[:,2:]) # (N, M, 2)
    inter_wh = np.maximum(0, inter_bottom_right - inter_top_left) # (N, M, 2)
    inter_area = inter_wh[:,:,0] * inter_wh[:,:,1] # (N, M) areas

    boxes1_area = (boxes1[:,2] - boxes1[:,0]) * (boxes1[:,3] - boxes1[:,1]) # (N,)
    boxes2_area = (boxes2[:,2] - boxes2[:,0]) * (boxes2[:,3] - boxes2[:,1]) # (M,)

    union = boxes1_area[:,None] + boxes2_area[None,:] - inter_area  # (N, M)
    return inter_area / (union+1e-9)


