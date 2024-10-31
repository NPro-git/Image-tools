
import os
import cv2
import numpy as np

frm_dir = '/home/jinling/Documents/data/full-seg-datasets/models_performance/fingers_v4.18_alpha_p0_ep01_natural_angles_fullskin/test'
to_dir = '/home/jinling/Documents/data/full-seg-datasets/models_performance/fingers_v4.18_alpha_p0_ep01_natural_angles_CloseCrop/test'

image_files = [file for file in os.listdir(os.path.join(frm_dir, 'images')) if '.png' in file]
#image_files = sorted(image_files)
img_cx, img_cy = 256, 256
img_h, img_w = 512, 512

def get_overd(vv, std_overds=[[124, 46], [284, 65]]):
    (x1, y1), (x2, y2) = std_overds
    k = (y2-y1)/(x2-x1)
    b = y1 - k*x1
    ww = int(k*vv + b +0.5)
    return ww

for img_idx, img_file in enumerate(image_files):
    img = cv2.imread(os.path.join(frm_dir, 'images', img_file))
    seg = cv2.imread(os.path.join(frm_dir, 'masks', img_file))
    nail_seg = ((seg>1)*255).astype(np.uint8)
    contours, hierarchy = cv2.findContours(nail_seg[:,:,0], cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = [contour.reshape((-1, 2)) for contour in contours]
    nail_contour = contours[0]
    min_dist = img_cx**2+img_cy**2
    for contour in contours:
        contour_cx, contour_cy = np.average(contour, axis=0)
        dist = (contour_cx-img_cx)**2 + (contour_cy-img_cy)**2
        if dist < min_dist:
            min_dist = dist
            nail_contour = contour
    obj_l, obj_t = np.min(nail_contour, axis=0)
    obj_r, obj_b = np.max(nail_contour, axis=0)
    obj_h, obj_w = obj_b-obj_t+1, obj_r-obj_l+1
    overd_h, overd_w = get_overd(obj_h), get_overd(obj_w)
    crp_t, crp_b = max(0, obj_t-overd_h), min(img_h-1, obj_b+overd_h)
    crp_l, crp_r = max(0, obj_l-overd_w), min(img_w-1, obj_r+overd_w)
    crp_img = np.zeros(img.shape, np.uint8)
    crp_seg = np.zeros(img.shape, np.uint8)
    crp_img[crp_t:crp_b+1,crp_l:crp_r+1] = img[crp_t:crp_b+1,crp_l:crp_r+1]
    crp_seg[crp_t:crp_b + 1, crp_l:crp_r + 1] = nail_seg[crp_t:crp_b + 1, crp_l:crp_r + 1]
    crp_seg_invisible = (crp_seg[:,:,0]>128).astype(np.uint8)
    img_save_path = os.path.join(to_dir, 'images', img_file)
    mask_save_path = os.path.join(to_dir, 'masks', img_file)
    cv2.imwrite(img_save_path, crp_img)
    cv2.imwrite(mask_save_path, crp_seg_invisible)

    '''
    show_t = np.concatenate((img, seg*127), axis=1)
    show_b = np.concatenate((crp_img, crp_seg), axis=1)
    show = np.concatenate((show_t, show_b), axis=0)
    cv2.imshow('show', show)
    cv2.waitKey(0)
    '''
    print('%d/%d: '%(img_idx+1, len(image_files)), img_file)
