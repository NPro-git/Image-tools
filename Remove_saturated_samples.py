
import os
import numpy as np
import cv2

frm_dir = '/home/jinling/Documents/data/full-seg-datasets/fingers_p0_v4.12_natural_polish_ALLs_HDRs/natural_1'
to_dir = '/home/jinling/Documents/data/full-seg-datasets/fingers_p0_v4.13_natural_polish_ALLs_NoSaturate/natural_1'
saturation_dir = '/home/jinling/Documents/data/full-seg-datasets/fingers_p0_v4.13_natural_polish_ALLs_NoSaturate/saturation'
img_frm_dir = os.path.join(frm_dir, 'alls')

files = [file for file in os.listdir(img_frm_dir) if '.png' in file]

nail_color, finger_color = [0,0,255], [0,255,0]
kernel = np.ones((10,10), np.uint8)
sature_th = 230
ratio_th = 0.1

for file in files:
    img_path = os.path.join(img_frm_dir, file)
    img = cv2.imread(img_path)
    mask_path = os.path.join(frm_dir, 'masks', file)
    mask = cv2.imread(mask_path)
    #mask[np.where(np.all(mask==np.array([2,2,2], dtype=np.uint8)))] = nail_color
    mask[np.where((mask == [2, 2, 2]).all(axis=-1))] = nail_color
    mask[np.where((mask == [1,1,1]).all(axis=-1))] = finger_color
    nail_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
    nail_mask[np.where((mask == nail_color).all(axis=-1))] = 255
    nail_dilate_mask = cv2.dilate(nail_mask, kernel, iterations=1)
    circle = np.zeros(mask.shape[:2], dtype=np.uint8)
    circle[np.where((nail_dilate_mask>128)*(nail_mask<=128))] = 255
    circle_img = img.copy()
    circle_img[np.where(circle<128)] = 0
    sature_img = np.zeros(mask.shape[:2], dtype=np.uint8)
    circle_points = np.where(circle>128)
    sature_points = np.where((circle_img>240).all(axis=-1))
    sature_img[circle_points] = 128
    sature_img[sature_points] = 255
    sature_img = np.concatenate([sature_img.reshape(sature_img.shape+(1,))]*3, axis=-1)
    show_img = np.concatenate((img, circle_img, sature_img), axis=1)
    ratio = len(sature_points[0])/(1+len(circle_points[0]))
    if ratio < ratio_th:
        command = 'cp ' + img_path + ' ' + os.path.join(to_dir, 'alls')
        print(command)
        #os.system(command)
        command = 'cp ' + mask_path + ' ' + os.path.join(to_dir, 'masks')
        print(command)
        #os.system(command)
    else:
        command = 'cp ' + img_path + ' ' + os.path.join(saturation_dir, 'alls')
        print(command)
        #os.system(command)
        command = 'cp ' + mask_path + ' ' + os.path.join(saturation_dir, 'masks')
        print(command)
        #os.system(command)

    print(ratio)
    cv2.imshow('show_img', show_img)
    cv2.waitKey(0)
    print(file)





