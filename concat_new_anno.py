
import os
import cv2
import numpy as np

img_dir_old = '/home/jinling/Documents/data/full-seg-datasets/fingers_p0_v4.8_skin_tones_bgr/images'
mask_dir_old = '/home/jinling/Documents/data/full-seg-datasets/fingers_p0_v4.8_skin_tones_bgr/masks'
out_dir_old = '/home/jinling/Documents/data/full-seg-datasets/fingers_p0_v4.8_skin_tones_bgr/outs'

img_dir_new = '/home/jinling/Documents/data/full-seg-datasets/fingers_skin_tones_bgr_p0_v4.8.240227a/images'
mask_dir_new = '/home/jinling/Documents/data/full-seg-datasets/fingers_skin_tones_bgr_p0_v4.8.240227a/masks'
out_dir_new = '/home/jinling/Documents/data/full-seg-datasets/fingers_skin_tones_bgr_p0_v4.8.240227a/outs'

img_files_old = [file for file in os.listdir(img_dir_old) if '.png' in file]
img_files_new = [file for file in os.listdir(img_dir_new) if '.png' in file and file not in img_files_old]

for img_idx, img_file in enumerate(img_files_new):
    img_path = os.path.join(img_dir_new, img_file)
    mask_path = os.path.join(mask_dir_new, img_file)
    save_path = os.path.join(out_dir_new, img_file)
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path)[:,:,0]
    mask_ = np.zeros(img.shape, dtype='uint8')
    mask_[np.where(mask==1)] = np.array([0,255,0], dtype=np.uint8)
    mask_[np.where(mask == 2)] = np.array([0, 0, 255], dtype=np.uint8)
    img_mask = np.concatenate((img, mask_), axis=1)
    cv2.imwrite(save_path, img_mask)
    print('%06d/%06d'%(img_idx, len(img_files_new)), img_file)

print('Completed. ')


