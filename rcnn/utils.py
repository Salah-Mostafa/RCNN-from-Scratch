import numpy as np
import pandas as pd
import cv2, os, PIL, matplotlib.pyplot as plt


def iou(a, b):
    # a is a group of boxes (proposals regions) with shape (N , 4)
    # b is a group of boxes (ground-truth bboxes) with shape (M , 4)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    x_diff = np.clip(x2-x1, 0, None)  # np.maximum(x2-x1, 0)
    y_diff = np.clip(y2-y1, 0, None)
    
    intersection = x_diff * y_diff
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    
    union = area_a[:, None] + area_b[None, :] - intersection + 1e-9  # 1e-9: small value to avoid dividing by zero 
    """area_a will be (N, 1, 4) and area_b will be (1, M, 4) and the result becomes (N, M)"""
    if np.any(union == 0):
        raise ValueError("Union area is zero, cannot compute IoU.")
    
    return intersection / union



def nms(boxes, scores, thr=0.3):
    # boxes: (N, 4) array of boxes
    # scores: (N,) array of scores  (confidence score of each predicted bbox)
    # thr: IoU threshold for suppression
    idxs_order = np.argsort(scores)[::-1] # sort indices by scores in descending order
    keep = []
    while len(idxs_order): # still processing while len(idx_order) not equal to 0 (idsx_order not empty) 
        i = idxs_order[0]  # index of the box with the highest score
        keep.append(int(i))     # add this index to the keep list
        if len(idxs_order) == 1: break
        ious = iou(boxes[i:i+1], boxes[idxs_order[1:]])[0]
        idxs_order = idxs_order[1:][ious < thr] # keep only boxes with IoU less than threshold
    return keep