
import os
import cv2
import numpy as np

dir = '/home/jinling/Documents/data/Buckets/EP01/10b-capture-backup/ALLLEDs/EP1_ValidationScans/imgs_segs/examples'
save_dir = '/home/jinling/Documents/data/Buckets/EP01/10b-capture-backup/ALLLEDs/EP1_ValidationScans/imgs_segs/examples/threes'

files = [file for file in os.listdir(dir) if '.png' in file]

for file in files:
    path = os.path.join(dir, file)
    img_seg = cv2.imread(path)
    img_h, img_w = img_seg.shape[:2]
    img_w = img_w//2
    img, seg = img_seg[:,:img_w], img_seg[:,img_w:]
    third = np.minimum(img.astype(np.uint32) + seg//3, 255).astype(np.uint8)
    threes = np.concatenate((img_seg,third), axis=1)
    save_path = os.path.join(save_dir, file)
    cv2.imwrite(save_path, threes)
    #cv2.imshow('threes', threes)
    #cv2.waitKey(0)
    print(file)

print('Completed. ')
