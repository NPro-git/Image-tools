
import os
import cv2
import numpy as np

img_dir = '/home/jinling/Documents/data/full-seg-datasets/fingers_p0_v4.8_skin_tones_bgr/images'
mask_dir = '/home/jinling/Documents/data/full-seg-datasets/fingers_p0_v4.8_skin_tones_bgr/masks'
out_dir = '/home/jinling/Documents/data/full-seg-datasets/fingers_p0_v4.8_skin_tones_bgr/outs'

img_files = [file for file in os.listdir(img_dir) if '.png' in file]

for img_idx, img_file in enumerate(img_files):
    img_path = os.path.join(img_dir, img_file)
    mask_path = os.path.join(mask_dir, img_file)
    save_path = os.path.join(out_dir, img_file)
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path)[:,:,0]
    mask_ = np.zeros(img.shape, dtype='uint8')
    mask_[np.where(mask==1)] = np.array([0,255,0], dtype=np.uint8)
    mask_[np.where(mask == 2)] = np.array([0, 0, 255], dtype=np.uint8)
    img_mask = np.concatenate((img, mask_), axis=1)
    cv2.imwrite(save_path, img_mask)
    print('%06d/%06d'%(img_idx, len(img_files)), img_file)

print('Completed. ')
