import numpy as np, matplotlib.pyplot as plt, cv2, os, PIL, PIL.ImageDraw
from rcnn.box_ops import box_to_targets, apply_targets

h, w = 750, 500
img = np.zeros((h, w, 3))

# Ground-truth bounding box:
box_h, box_w = 500, 200
x, y = np.random.randint(0, w - box_w +1), np.random.randint(0, h - box_h +1)
g_box = np.array([[x, y, x+box_w, y+box_h]])

# Create proposal region manually:
p_h, p_w = np.random.randint(0, h+1), np.random.randint(0, w+1)
p_x, p_y = np.random.randint(0, w - p_w +1), np.random.randint(0, h - p_h +1) 
p_box = np.array([[p_x, p_y, p_x+p_w, p_y+p_h]])
#==========================================================================================


targets = box_to_targets(p_box, g_box)
updated_box = apply_targets(p_box, targets)

# Original bbox:
cv2.rectangle(img, (x, y), (x+box_w, y+box_h), (255, 255, 255), thickness=10)
# Random proposal (bbox) before updating:
cv2.rectangle(img, (p_x, p_y), (p_x+p_w, p_y+p_h), (0, 0, 255), thickness=4)
# Updated bbox:
x1, y1, x2, y2 = np.rint(updated_box.flatten()).astype(int)
cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), thickness=4)
plt.imshow(img)
plt.show()