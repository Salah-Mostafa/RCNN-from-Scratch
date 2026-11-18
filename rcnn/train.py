from rcnn.voc_dataset import train_valid_sets
from rcnn.dataset_preparation import build_train_set, train_heads
from rcnn.extractor import extract_torch


def train(train_obj, max_images):
    
    X, y, T = build_train_set(train_obj, extract_torch, max_images=max_images)
    svc, regressors = train_heads(X, y, T)
    
    # SVC and Regressors became a trained models ==> we are save them here..
    return svc, regressors, X, y, T