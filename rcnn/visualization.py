import cv2


classes = {
    1: "aeroplane",
    2: "bicycle",
    3: "bird",
    4: "boat",
    5: "bottle",
    6: "bus",
    7: "car",
    8: "cat",
    9: "chair",
    10: "cow",
    11: "diningtable",
    12: "dog",
    13: "horse",
    14: "motorbike",
    15: "person",
    16: "pottedplant",
    17: "sheep",
    18: "sofa",
    19: "train",
    20: "tvmonitor"
}


def visualize(img, detections, save_path="output.jpg"):

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for detection in detections:

        x1, y1, x2, y2 = map(int, detection["box"])

        class_id = detection["class_id"]
        score = detection["score"]

        class_name = classes[class_id]

        label = f"{class_name}: {score:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

        cv2.putText(
            img,
            label,
            (x1, max(y1, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    cv2.imwrite(save_path, img)

    print(f"Result saved at: {save_path}")

    cv2.imshow("R-CNN Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
