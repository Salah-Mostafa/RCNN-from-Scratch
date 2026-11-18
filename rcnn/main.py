from rcnn.voc_dataset import train_valid_sets
from rcnn.train import train
from rcnn.dataset_preparation import build_train_set, prepare_detection_data
from rcnn.extractor import extract_torch
from rcnn.evaluate import evaluate_classifier, evaluate_regressor, evaluate_detector
from collections import Counter
from rcnn.model_saver import save_model

def main() -> None:
    # 1. Training
    train_obj, valid_obj = train_valid_sets()
    svc, regressors, X_train, y_train, T_train = train(train_obj, max_images=200)
    print("\nTraining Label Distribution:")
    print(Counter(y_train))
    
    
    # 2. Individual component evaluation
    X_val, y_val, T_val, metadata = build_train_set(valid_obj, extract_torch, max_images=200, return_metadata=True)
    evaluate_classifier(svc, X_val, y_val)
    evaluate_regressor(regressors, X_val, y_val, metadata["proposals"], metadata["matched_gt"])
    
    
    # 3. Full detection data preparation
    print("\n" + "---"*35)
    detection_data = prepare_detection_data(valid_obj, extract_torch, max_images=200, max_proposals=500)
    sample = detection_data[0]
    print("Features shape:", sample["features"].shape)
    print("Proposals shape:", sample["proposals"].shape)
    print("GT boxes shape:", sample["gt_boxes"].shape)
    print("GT labels shape:", sample["gt_labels"].shape)
    print("---"*35 + "\n")
    
    
    # 4. Full detector evaluation
    mean_ap, ap_per_class = evaluate_detector(
        svc, regressors, detection_data, iou_threshold=0.5, nms_threshold=0.3)
    
    save_model(svc, regressors)
    
if __name__ == "__main__":
    main() 
    
    
# To run this script:   python -m rcnn.main