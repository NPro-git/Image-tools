
import os
import cv2
import numpy as np

img_dir = '/home/jinling/Documents/data/Buckets/EP01/10b-capture-backup/ALLLEDs/EP1_ValidationScans/ALLs/'
seg_dir = '/home/jinling/Documents/data/Buckets/EP01/10b-capture-backup/ALLLEDs/EP1_ValidationScans/segs/'
save_dir = '/home/jinling/Documents/data/Buckets/EP01/10b-capture-backup/ALLLEDs/EP1_ValidationScans/imgs_segs'

fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']

img_files = [file for file in os.listdir(img_dir) if '.jpg' in file]
img_h, img_w = 1944, 2592
overd_r = 0.2

for img_idx, img_file in enumerate(img_files):
    img_path = os.path.join(img_dir, img_file)
    for finger in fingers:
        seg_name = img_file.replace('.jpg', '_'+finger+'.png')
        seg_path = os.path.join(seg_dir, seg_name)
        if not os.path.exists(seg_path):
            continue
        seg = cv2.imread(seg_path)
        if seg.shape[:2] != (img_h, img_w):
            continue
        img = cv2.imread(img_path)
        seg_prj = seg.max(axis=-1)
        seg_prj_h = seg_prj.max(axis=0)
        seg_prj_v = seg_prj.max(axis=1)
        seg_l, seg_r = np.argmax(seg_prj_h), len(seg_prj_h)-1-np.argmax(seg_prj_h[::-1])
        seg_t, seg_b = np.argmax(seg_prj_v), len(seg_prj_v)-1-np.argmax(seg_prj_v[::-1])
        seg_w, seg_h = seg_r-seg_l+1, seg_b-seg_t+1
        crp_l, crp_r = max(seg_l-int(seg_w*overd_r+0.5),0), min(seg_r+int(seg_w*overd_r+0.5),img_w-1)
        crp_t, crp_b = max(seg_t-int(seg_h*overd_r+0.5),0), min(seg_b+int(seg_h*overd_r+0.5),img_h-1)
        crp_img = img[crp_t:crp_b+1,crp_l:crp_r+1]
        crp_seg = seg[crp_t:crp_b+1,crp_l:crp_r+1]
        crp_img_seg = np.concatenate((crp_img, crp_seg), axis=1)
        save_path = os.path.join(save_dir, seg_name)
        cv2.imwrite(save_path, crp_img_seg)
        print(seg_name)
        #cv2.imshow('crp', crp_img_seg)
        #cv2.waitKey(0)
    print('%d/%d: '%(img_idx+1, len(img_files)), img_file)
