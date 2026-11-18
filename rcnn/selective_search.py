import numpy as np
import os, cv2, glob as gb, matplotlib.pyplot as plt, PIL
from cv2 import ximgproc


# Modified version of the Selective Search function by Salah:
def SelectiveSearch(img, max_proposals=2000, mode='fast'):
    # To accelerate the processing, we can enable OpenCV's optimization features
    cv2.setUseOptimized(True)
    
    # Create selective search object
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    
    # Set the base image for selective search
    bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ss.setBaseImage(bgr_img)
    
    # Switch to fast mode for faster processing 
    if mode == 'quality':
        ss.switchToSelectiveSearchQuality()
    else:
        ss.switchToSelectiveSearchFast()
    
    # Run selective search to get region proposals
    rects = ss.process()
    
    # Limit the number of proposals to max_proposals
    return np.array([[x, y, x+w, y+h] for x, y, w, h in rects[:max_proposals]])