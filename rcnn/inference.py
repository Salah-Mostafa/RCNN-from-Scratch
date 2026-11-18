import cv2
import numpy as np

from rcnn.selective_search import SelectiveSearch
from rcnn.box_ops import crop_warp
from rcnn.extractor import extract_torch
from rcnn.utils import nms


def inference(image_path, svc, regressors, confidence_threshold=0.4, nms_threshold=0.1):

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Image not found.")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    proposals = SelectiveSearch(img, max_proposals=500, mode="quality")

    if len(proposals) == 0:
        return img, []

    crops = [crop_warp(img, box) for box in proposals]

    features = extract_torch(crops)

    # Same normalization used during training
    features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-10)

    scores = svc.predict_proba(features)

    detections = []

    for class_id, regressor in regressors.items():

        class_columns = np.where(svc.classes_ == class_id)[0]

        if len(class_columns) == 0:
            continue

        class_scores = scores[:, class_columns[0]]

        selected = np.where(class_scores >= confidence_threshold)[0]

        if len(selected) == 0:
            continue

        selected_features = features[selected]
        selected_boxes = np.array(proposals)[selected]
        selected_scores = class_scores[selected]

        # Predict bounding box adjustments
        deltas = regressor.predict(selected_features)

        predicted_boxes = apply_regression(selected_boxes, deltas, img.shape)

        # Remove invalid boxes
        valid = (
            (predicted_boxes[:, 2] > predicted_boxes[:, 0]) &
            (predicted_boxes[:, 3] > predicted_boxes[:, 1])
        )

        predicted_boxes = predicted_boxes[valid]
        selected_scores = selected_scores[valid]

        if len(predicted_boxes) == 0:
            continue

        keep = nms(predicted_boxes, selected_scores, nms_threshold)

        for idx in keep:
            detections.append({
                "box": predicted_boxes[idx],
                "class_id": class_id,
                "score": selected_scores[idx]
            })

    return img, detections


def apply_regression(boxes, deltas, img_shape):

    boxes = np.array(boxes, dtype=float)

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]

    centers_x = boxes[:, 0] + widths / 2
    centers_y = boxes[:, 1] + heights / 2

    pred_x = deltas[:, 0] * widths + centers_x
    pred_y = deltas[:, 1] * heights + centers_y

    pred_w = np.exp(deltas[:, 2]) * widths
    pred_h = np.exp(deltas[:, 3]) * heights

    x1 = pred_x - pred_w / 2
    y1 = pred_y - pred_h / 2
    x2 = pred_x + pred_w / 2
    y2 = pred_y + pred_h / 2

    h, w = img_shape[:2]

    predicted_boxes = np.column_stack([x1, y1, x2, y2])

    predicted_boxes[:, [0, 2]] = np.clip(predicted_boxes[:, [0, 2]], 0, w - 1)
    predicted_boxes[:, [1, 3]] = np.clip(predicted_boxes[:, [1, 3]], 0, h - 1)

    return predicted_boxes
