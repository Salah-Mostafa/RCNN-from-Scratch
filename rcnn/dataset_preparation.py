import cv2, numpy as np
from rcnn.selective_search import SelectiveSearch
from rcnn.utils import iou, nms
from rcnn.box_ops import crop_warp, box_to_targets
from sklearn.svm import SVC
from sklearn.linear_model import Ridge


def l2norm(x): 
    return x/(np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
    
    
def build_train_set(dataset, extract_fn, pos_iou=0.5, neg_iou=0.3, neg_ratio=3, max_images=None, return_metadata=False):
    """
    dataset is list of tuples, each tuple has three values ==> (images, gt_boxes, gt_labels)
    dataset ==> [(imgs, gt_boxes, gt_labels), ....]  >> gt_boxes (N, 4), gt_labels (N,)
    Each image has a list of boxes and list of labels if this image has more than one object.
    If image contain 3 objects, gt_boxes shape [3, 4] and gt_labels shape [3]
    Each bbox of dataset has 4 values [x, y, x+w, y+h]
    """ 
    x, y, T = [], [], []
    all_gt, all_props = [], []
    
    used_imgs = 0
    if not max_images:
        used_imgs = len(dataset)
    else: used_imgs = max_images
    
    for i in range(used_imgs):
        img_path, gt, lab = dataset[i]
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        
        gt, lab = np.asarray(gt, float), np.asarray(lab)
        props = np.vstack([SelectiveSearch(img), gt])
        ious = iou(props, gt)
        best = ious.argmax(axis=1)    # Indices of best ious for each propsal with ground truth bbox
        biou = ious.max(axis=1)       # Values of best ious  ...
        pos = np.where(biou >= pos_iou)[0]   # Return the indices of positive proposals
        neg = np.where(biou < neg_iou)[0]    # Return the indices of negative proposals
        """If 1000 neg, 20 pos the next instruction just returns 60 neg.."""
        neg = np.random.choice(neg, min(len(neg), neg_ratio*max(len(pos), 1)), replace=False)
        idxs = np.concatenate([pos, neg])
        
        all_props.append(props[idxs])
        all_gt.append(gt[best[idxs]])
        
        features = extract_fn([crop_warp(img, props[i]) for i in idxs])
        x.append(features)
        labels = np.concatenate([lab[best[pos]], np.zeros(len(neg), int)])
        y.append(labels)
        t = np.zeros((len(idxs), 4)); t[:len(pos)] = box_to_targets(props[pos], gt[best[pos]])
        T.append(t)
        
    X = l2norm(np.vstack(x))
    y = np.concatenate(y)
    T = np.vstack(T)
    if return_metadata:
            metadate = {
                "proposals": np.vstack(all_props),
                "matched_gt": np.vstack(all_gt)
            }
            return X, y, T, metadate
    
    return X, y, T







# For mAP:
def prepare_detection_data(dataset, extract_fn, max_images=5, max_proposals=2000, feature_batch_size=64):
    """
    Prepare all region proposals for full object detector evaluation.

    Returns a list containing one dictionary per image:
        features:   (N, 4096)
        proposals:  (N, 4)
        gt_boxes:   (M, 4)
        gt_labels:  (M,)
    """

    detection_data = []
    num_images = min(len(dataset), max_images) if max_images else len(dataset)
    
    for i in range(num_images):

        img_path, gt_boxes, gt_labels = dataset[i]

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        gt_boxes = np.asarray(gt_boxes, dtype=float)
        gt_labels = np.asarray(gt_labels)

        # Generate proposals without using Ground Truth
        proposals = SelectiveSearch(img, max_proposals=max_proposals)

        # Extract features in smaller batches to reduce RAM usage
        feature_parts = []
        for start in range(0, len(proposals), feature_batch_size):
            batch_props = proposals[start:start + feature_batch_size]

            crops = [crop_warp(img, box) for box in batch_props]
            features = extract_fn(crops)
            feature_parts.append(features)

        if feature_parts:
            features = np.vstack(feature_parts)
            features = l2norm(features)
        else:
            features = np.empty((0, 4096), dtype=np.float32)

        detection_data.append({
            "image_path": img_path,
            "image_shape": img.shape[:2],
            "features": features,
            "proposals": proposals,
            "gt_boxes": gt_boxes,
            "gt_labels": gt_labels
        })

        print(
            f"Image {i + 1:02d}/{num_images} | "
            f"Proposals: {len(proposals)} | "
            f"Features: {features.shape}"
            )
    return detection_data





def train_heads(X, y, T):
    # One classifier for all classes + background (20 main class label, and background with class 0)
    svm = SVC(C=0.01, kernel='rbf', probability=True, decision_function_shape='ovr', max_iter=10000).fit(X, y)
    pos = y > 0   # return boolean mask of positive samples
    
    # One bounding-box regressor for each object class
    regressors = {}
    for class_id in range(1, 21):  # For each class_id from 1 to 20 (VOC classes)
        mask = (y == class_id)
        
        if mask.sum() == 0:
            continue
        ridge = Ridge(alpha=1.0).fit(X[mask], T[mask])
        regressors[class_id] = ridge
    return svm, regressors