from rcnn.model_loader import load_model
from rcnn.inference import inference
from rcnn.visualization import visualize


svc, regressors = load_model()

image_path = "test_images/test1.jpg"

img, detections = inference(
    image_path,
    svc,
    regressors
)

print("Number of detections:", len(detections))

visualize(img, detections)
