import os, cv2, glob as gb
from shutil import ExecError
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from pathlib import Path

from rcnn.selective_search import SelectiveSearch
from rcnn.utils import iou


#======================================================================================================
def SelectImage():
    try:
        voc_path = Path(r"C:\Users\Salah\Downloads\VOCtrainval_03-May-2010\VOCdevkit\VOC2010")
        print("\nVOC folder exists:", os.path.isdir(voc_path))
        print("JPEGImages exists:", os.path.isdir(os.path.join(voc_path, "JPEGImages")))

        #images = gb.glob(os.path.join(voc_path, "JPEGImages", "*.jpg"))
        images = sorted((voc_path / "JPEGImages").glob("*.jpg"))
        print("Number of images:", len(images))

        if images:
            img_path = images[135]
            img = cv2.imread(img_path)   # BGR
            
    except ExecError as e:
        print("Invalid load image path")
        print(f"Exception: {str(e)}")



    # Read XML annotation
    xml_path = voc_path / "Annotations" / f"{img_path.stem}.xml"
    root = ET.parse(xml_path).getroot()
    gt_boxes = []
    gt_labels = []
    for obj in root.findall("object"):
        class_name = obj.findtext("name")
        bbox = obj.find("bndbox")
        x1 = int(bbox.findtext("xmin"))
        y1 = int(bbox.findtext("ymin"))
        x2 = int(bbox.findtext("xmax"))
        y2 = int(bbox.findtext("ymax"))
        gt_boxes.append([x1, y1, x2, y2])
        gt_labels.append(class_name)
        
    gt_boxes = np.asarray(gt_boxes, dtype=np.float32)
    print("Image:", img_path.name)
    print("Image shape:", img.shape)
    print("Ground Truth Boxes:", gt_boxes.shape)
    print("Classes:", gt_labels)
    print("=="*30)
    
    return img, gt_boxes, gt_labels
#======================================================================================================


#======================================================================================================
def Display(img, gt_boxes, gt_labels, props=None):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #visualization = img.copy()

    for box, label in zip(gt_boxes, gt_labels):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if props is not None:
        for box in props:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)

    plt.figure(figsize=(10, 7))
    plt.imshow(img)
    plt.title("PASCAL VOC 2010 - Ground Truth")
    plt.axis("off")
    plt.show()
#======================================================================================================


#======================================================================================================
def PropRegion(img, gt_boxes):
    proposals = SelectiveSearch(img, max_proposals=500)

    # Include GT boxes to guarantee positive examples
    props = np.vstack([proposals, gt_boxes])

    ious = iou(props, gt_boxes)
    best = ious.argmax(axis=1)
    biou = ious.max(axis=1)

    pos = np.where(biou >= 0.5)[0]
    neg = np.where(biou < 0.3)[0]

    print("Total proposals:", len(props))
    print("Positive proposals:", len(pos))
    print("Negative proposals:", len(neg))
    return props[pos]
#======================================================================================================

if __name__ == "__main__":
    img, gt_boxes, gt_labels = SelectImage()
    Display(img, gt_boxes, gt_labels, PropRegion(img, gt_boxes))
    
    
# run script:  python -m examples.voc_single_image_demo