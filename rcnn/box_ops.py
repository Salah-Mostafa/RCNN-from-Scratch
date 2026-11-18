import numpy as np, cv2

def crop_warp(img, box, size=224):
    """Crop each proposal from original image and resize them to (224, 224)"""
    x1, y1, x2, y2 = [int(v) for v in box]
    x2, y2 = max(x2, x1+1), max(y2, y1+1)
    return cv2.resize(img[y1:y2, x1:x2], (size, size))

  
def box_to_targets(p, g):       # p: proposals, g: groung-truth (gt)
    """Calc. different between four coordinates of p , gt"""
    pw, ph = p[:, 2]-p[:, 0], p[:, 3]-p[:, 1]
    px, py = p[:, 0]+(pw/2), p[:, 1]+(ph/2)
    
    gw, gh = g[:, 2]-g[:, 0], g[:, 3]-g[:, 1]
    gx, gy = g[:, 0]+(gw/2), g[:, 1]+(gh/2)
    
    # Calculate targets (differences):
    tx = (gx-px)/pw ; ty = (gy-py)/ph ; tw = np.log(gw/pw) ; th = np.log(gh/ph)
    return np.stack([tx, ty, tw, th], axis=1)   # np.stack(.., axis=1) to make stack array in 2D as a vertical stacking


def apply_targets(p, t):  # t: output array of box_to_targets()
    pw, ph = p[:, 2]-p[:, 0], p[:, 3]-p[:, 1] ; px, py = p[:, 0]+(pw/2), p[:, 1]+(ph/2)
    
    tx, ty, tw, th = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
    cx, cy = px+(tx * pw), py+(ty * ph) 
    w, h = pw * np.exp(tw), ph * np.exp(th)
    
    # New coordinates of proposal region (boundary box) after updated:
    x1, y1, x2, y2 = cx-(w/2), cy-(h/2), cx+(w/2), cy+(h/2)
    return np.stack([x1, y1, x2, y2], 1) 
