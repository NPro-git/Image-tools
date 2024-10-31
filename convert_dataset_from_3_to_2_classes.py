
import os
import cv2
import numpy as np
import time

dir = '/home/jinling/Documents/data/full-seg-datasets/models_performance/fingers_v4.18_alpha_p0_ep01_natural_angles/train/masks'

files = [file for file in os.listdir(dir) if '.png' in file]

for idx, file in enumerate(files):
    path = os.path.join(dir, file)
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = (mask==2).astype(np.uint8)
    cv2.imwrite(path, mask)
    print('%d/%d: '%(idx+1, len(files)), file)
    time.sleep(0.01)
