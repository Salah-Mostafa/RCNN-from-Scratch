import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset



voc_path = r"C:\Users\Salah\Downloads\VOCtrainval_03-May-2010\VOCdevkit\VOC2010"
class VOCDataset(Dataset):
    def __init__(self, voc_path=voc_path, split='train'):
        self.voc_path = Path(voc_path)
        self.split = split
        self.imgs = []
        self.boxes = []
        self.labels = []
        
        self.VOC_CLASSES = [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]

        self.CLASS_TO_IDX = {
            name: idx + 1
            for idx, name in enumerate(self.VOC_CLASSES)
        }

        self.IDX_TO_CLASS = {
            idx: name
            for name, idx in self.CLASS_TO_IDX.items()
        }
        
        self._load_data()

    def _load_data(self):
        img_dir = self.voc_path / 'JPEGImages'
        ann_dir = self.voc_path / 'Annotations'
        split_file = self.voc_path / 'ImageSets' / 'Main' / f'{self.split}.txt'

        with open(split_file) as f:
            img_ids = [line.strip() for line in f.readlines()]

        for img_id in img_ids:
            img_path = img_dir / f'{img_id}.jpg'
            ann_path = ann_dir / f'{img_id}.xml'

            if not img_path.exists() or not ann_path.exists():
                continue

            self.imgs.append(img_path)

            # Load annotations
            tree = ET.parse(ann_path)
            root = tree.getroot()
            boxes = []
            labels = []
            for obj in root.findall('object'):
                label = obj.find('name').text
                bndbox = obj.find('bndbox')
                box = [
                    int(bndbox.find('xmin').text) - 1,
                    int(bndbox.find('ymin').text) - 1,
                    int(bndbox.find('xmax').text),
                    int(bndbox.find('ymax').text)
                ]
                boxes.append(box)
                labels.append(self.CLASS_TO_IDX[label])

            self.boxes.append(np.array(boxes))
            self.labels.append(np.array(labels))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.imgs[idx], self.boxes[idx], self.labels[idx]
    
    
    
#=========================== Collect Function ========================
def train_valid_sets():
    train_obj = VOCDataset(voc_path=voc_path, split='train')
    valid_obj = VOCDataset(voc_path=voc_path, split='val')  
    return train_obj, valid_obj