from rcnn.box_ops import apply_targets
from rcnn.utils import iou, nms
from sklearn.metrics import classification_report, accuracy_score, recall_score
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings("ignore")


def evaluate_classifier(model, X, y):
    y_pred = model.predict(X)
    print("\nPredicted Label Distribution:")
    print(Counter(y_pred))

    print(f"Accuracy: {accuracy_score(y, y_pred):.4f}")
    print(f"Macro Recall: {recall_score(y, y_pred, average='macro', zero_division=0):.4f}")

    print("\nClassification Report:")
    print(classification_report(y, y_pred, zero_division=0))


def evaluate_regressor(regressors, X, y, proposals, matched_gt):
    positive = y > 0

    X = X[positive]
    y = y[positive]
    proposals = proposals[positive]
    matched_gt = matched_gt[positive]

    original_ious = []
    refined_ious = []

    for class_id, regressor in regressors.items():
        mask = y == class_id

        if not mask.any():
            continue

        class_features = X[mask]
        class_props = proposals[mask]
        class_gt = matched_gt[mask]

        predicted_offsets = regressor.predict(class_features)

        refined_boxes = apply_targets(
            class_props,
            predicted_offsets
        )

        before_iou = iou(class_props, class_gt).diagonal()
        after_iou = iou(refined_boxes, class_gt).diagonal()

        original_ious.extend(before_iou)
        refined_ious.extend(after_iou)

        print(f"\nClass {class_id}:")
        print(f"Proposals mean IoU: {before_iou.mean():.4f}")
        print(f"Refined boxes mean IoU: {after_iou.mean():.4f}")


    if not original_ious:
        print("No positive samples available for regression evaluation.")
        return

    print("\nOverall Regression Evaluation:")
    print(f"Mean IoU before regression: {np.mean(original_ious):.4f}")
    print(f"Mean IoU after regression:  {np.mean(refined_ious):.4f}")
    print(f"Mean IoU improvement:       {np.mean(refined_ious) - np.mean(original_ious):.4f}")







"""--- Full Object Detector Evaluation ---"""
def calculate_ap(tp, fp, num_gt):
    if num_gt == 0:
        return 0.0

    tp = np.cumsum(tp)
    fp = np.cumsum(fp)

    recall = tp / num_gt
    precision = tp / (tp + fp + 1e-9)

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    idx = np.where(mrec[1:] != mrec[:-1])[0]

    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])

    return ap




def evaluate_detector(svc, regressors, detection_data, iou_threshold=0.5, 
                      nms_threshold=0.3, confidence_threshold=0.0):
    all_detections = []
    all_ground_truth = []

    for image_idx, sample in enumerate(detection_data):

        features = sample["features"]
        proposals = sample["proposals"]
        gt_boxes = sample["gt_boxes"]
        gt_labels = sample["gt_labels"]

        image_height, image_width = sample["image_shape"]

        all_ground_truth.append({
            "boxes": gt_boxes,
            "labels": gt_labels
        })

        if len(features) == 0:
            continue

        scores = svc.predict_proba(features)
        if image_idx == (len(detection_data) - 1):
            print("\nSVM Classes:", svc.classes_)
            print("Scores shape:", scores.shape)
            
            """--- Return Min, Max, and Mean of each class which SVC was trained ---"""
            for col, class_id in enumerate(svc.classes_):
                print(
                    f"Class {class_id:02d}: "
                    f"Min={scores[:, col].min():.4f}, "
                    f"Max={scores[:, col].max():.4f}, "
                    f"Mean={scores[:, col].mean():.4f}"
                )
            print("\n")
        for class_id, regressor in regressors.items():

            class_columns = np.where(svc.classes_ == class_id)[0]

            if len(class_columns) == 0:
                continue

            class_scores = scores[:, class_columns[0]]
            selected = np.where(class_scores >= confidence_threshold)[0]

            if len(selected) == 0:
                continue

            class_features = features[selected]
            class_proposals = proposals[selected]
            selected_scores = class_scores[selected]

            offsets = regressor.predict(class_features)

            refined_boxes = apply_targets(class_proposals, offsets)

            # Clip boxes to image boundaries
            refined_boxes[:, [0, 2]] = np.clip(refined_boxes[:, [0, 2]], 0, image_width)
            refined_boxes[:, [1, 3]] = np.clip(refined_boxes[:, [1, 3]], 0, image_height)

            # Remove invalid boxes
            valid = (
                (refined_boxes[:, 2] > refined_boxes[:, 0]) & (refined_boxes[:, 3] > refined_boxes[:, 1])
                )

            refined_boxes = refined_boxes[valid]
            selected_scores = selected_scores[valid]

            if len(refined_boxes) == 0:
                continue

            # Class-wise NMS
            keep = nms(refined_boxes, selected_scores, thr=nms_threshold)

            for idx in keep:
                all_detections.append({
                    "image_idx": image_idx,
                    "class_id": class_id,
                    "box": refined_boxes[idx],
                    "score": selected_scores[idx]
                })

    # Classes present in the evaluated Ground Truth
    classes = np.unique(np.concatenate([sample["labels"] for sample in all_ground_truth]))

    ap_per_class = {}
    for class_id in classes:

        num_gt = sum(np.sum(sample["labels"] == class_id) for sample in all_ground_truth)

        detections = [det for det in all_detections if det["class_id"] == class_id]

        detections.sort(key=lambda det: det["score"], reverse=True)

        matched_gt = {
            i: np.zeros(len(sample["labels"]), dtype=bool)
            for i, sample in enumerate(all_ground_truth)
        }

        tp = np.zeros(len(detections))
        fp = np.zeros(len(detections))

        for det_idx, det in enumerate(detections):
            image_idx = det["image_idx"]
            gt = all_ground_truth[image_idx]

            gt_mask = gt["labels"] == class_id
            class_gt_boxes = gt["boxes"][gt_mask]

            original_gt_indices = np.where(gt_mask)[0]

            if len(class_gt_boxes) == 0:
                fp[det_idx] = 1
                continue

            overlaps = iou(det["box"].reshape(1, 4), class_gt_boxes)[0]

            best_idx = np.argmax(overlaps)
            best_iou = overlaps[best_idx]

            original_idx = original_gt_indices[best_idx]

            if (
                best_iou >= iou_threshold and
                not matched_gt[image_idx][original_idx]
            ):
                tp[det_idx] = 1
                matched_gt[image_idx][original_idx] = True
            else:
                fp[det_idx] = 1

        ap = calculate_ap(tp, fp, num_gt)
        ap_per_class[class_id] = ap

        print(
            f"Class {class_id:02d}: "
            f"AP = {ap:.4f} | "
            f"GT = {num_gt} | "
            f"Detections = {len(detections)}"
        )

    mean_ap = np.mean(list(ap_per_class.values()))

    print("\nObject Detection Evaluation:")
    print(f"mAP@{iou_threshold}: {mean_ap:.4f}")

    return mean_ap, ap_per_class

