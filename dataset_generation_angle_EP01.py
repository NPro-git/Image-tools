

import os
import json
import random
import numpy as np
import cv2

dirs = [    '/home/jinling/Documents/data/Buckets/EP01/10b-capture-backup/ALLLEDs/hands_EP01_20240605-0709_hdr'
]

save_dir =  '/home/jinling/Documents/data/full-seg-datasets/models_performance/EPx_hdr'

finger_names = ['thumb', 'index', 'middle', 'ring', 'pinkie', 'palm']
fingers_hand_colors = [[255, 0, 0], [0, 255, 0], [255, 255, 0], \
                        [255, 0, 255], [0, 255, 255], [64, 255, 128]]
thumb_crp_h, thumb_crp_w = 640, 640
regular_crp_h, regular_crp_w = 512, 512
crp_h, crp_w = 512, 512
nail_color = (2, 2, 2)
finger_color = (1, 1, 1)

img_json_pairs = []
for dir in dirs:
    json_files = [file for file in os.listdir(dir) if '.json' in file]
    for json_file in json_files:
        img_file = json_file.removesuffix('.json')+'.jpg'
        img_path = os.path.join(dir, img_file)
        if os.path.exists(img_path):
            json_path = os.path.join(dir, json_file)
            img_json_pairs.append([img_path, json_path])

def get_orientation(shape, fingers_mask, json_data):
    nail_mask = np.zeros(fingers_mask.shape, dtype=np.uint8)
    points = [[int(p[0] + 0.5), int(p[1] + 0.5)] for p in shape['points']]
    cv2.fillPoly(nail_mask, [np.array(points, dtype=int)], [255,255,255])
    nail_mask = nail_mask[:,:,0]
    nail_finger_mask = np.zeros(fingers_mask.shape[:2], dtype=np.uint8)
    over_nail_r_max = -1
    nail_finger_idx = 0
    for f_idx in range(len(finger_names)-1):
        f_color = fingers_hand_colors[f_idx]
        finger_mask = np.zeros(nail_mask.shape[:2], dtype=np.uint8)
        finger_mask[np.where((fingers_mask[:,:,0]==f_color[0])*(fingers_mask[:,:,1]==f_color[1])*\
                             (fingers_mask[:,:,2]==f_color[2]))] = 255
        nail_area = len(np.where(nail_mask>128)[0])
        finger_nail_overlap_area = len(np.where((nail_mask>128)*(finger_mask>128))[0])
        over_nail_r = finger_nail_overlap_area/nail_area
        if over_nail_r_max < over_nail_r:
            over_nail_r_max = over_nail_r
            nail_finger_mask = finger_mask
            nail_finger_idx = f_idx
    cover_r_max = -1
    start_x, start_y, end_x, end_y = 0, 0, 0, 0
    for shape_idx in range(len(json_data['shapes'])):
        label = json_data['shapes'][shape_idx]['label']
        if label != 'orientation':
            continue
        ort_mask = np.zeros(fingers_mask.shape, dtype=np.uint8)
        points = json_data['shapes'][shape_idx]['points']
        points = [[int(p[0] + 0.5), int(p[1] + 0.5)] for p in points]
        points = np.array(points, dtype=int)
        cv2.polylines(ort_mask, [points], isClosed=False, color=255, thickness=1)
        ort_mask = ort_mask[:,:,0]
        ort_num = len(np.where(ort_mask > 128)[0])
        overlap_num = len(np.where((nail_finger_mask > 128) * (ort_mask > 128))[0])
        cover_r = overlap_num / ort_num
        if cover_r_max < cover_r:
            cover_r_max = cover_r
            (start_x, start_y), (end_x, end_y) = points
    k = 200 if start_x==end_x and start_y!=end_y else (start_y-end_y)/(start_x-end_x)
    angle = int(np.arctan(k)*180/np.pi+180+0.5)%180
    return nail_finger_idx, angle

for img_path, json_path in img_json_pairs:
    img = cv2.imread(img_path)
    img_name = img_path.split('/')[-1]
    img_h, img_w = img.shape[:2]
    with open(json_path, encoding='gbk') as fd:
        json_data = json.load(fd)
    mask = np.zeros(img.shape, dtype=np.uint8)
    fingers_mask = mask.copy()
    for idx in range(len(json_data['shapes'])):
        label = json_data['shapes'][idx]['label']
        if label not in finger_names:  # draw fingers first
            continue
        points = [[int(p[0] + 0.5), int(p[1] + 0.5)] for p in json_data['shapes'][idx]['points']]
        cv2.fillPoly(mask, [np.array(points, dtype=int)], finger_color)
        cv2.fillPoly(fingers_mask, [np.array(points, dtype=int)], \
                     fingers_hand_colors[finger_names.index(label)])
    for idx in range(len(json_data['shapes'])):
        label = json_data['shapes'][idx]['label']
        if label != 'nail':
            continue
        points = [[int(p[0] + 0.5), int(p[1] + 0.5)] for p in json_data['shapes'][idx]['points']]
        cv2.fillPoly(mask, [np.array(points, dtype=int)], nail_color)
    for idx in range(len(json_data['shapes'])):
        label = json_data['shapes'][idx]['label']
        if label != 'nail':
            continue
        nail_finger_idx, angle = get_orientation(json_data['shapes'][idx], fingers_mask, json_data)
        is_thumb_bot = 'EP01_' in img_name and ('bottomright' in img_name or 'bottomleft' in img_name) and 'thumb' in label
        crp_h, crp_w = (thumb_crp_h, thumb_crp_w) if is_thumb_bot else (regular_crp_h, regular_crp_w)
        points = json_data['shapes'][idx]['points']
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        nail_l, nail_r, nail_t, nail_b = min(xs), max(xs), min(ys), max(ys)
        nail_cx, nail_cy = int((nail_l + nail_r) / 2 + 0.5), int((nail_t + nail_b) / 2 + 0.5)
        crp_img = np.zeros((crp_h, crp_w, 3), dtype=np.uint8)
        crp_mask = crp_img.copy()
        org_crp_t, org_crp_l = nail_cy - crp_h // 2, nail_cx - crp_w // 2
        org_crp_b, org_crp_r = org_crp_t + crp_h - 1, org_crp_l + crp_w - 1
        org_crp_t, org_crp_b = max(org_crp_t, 0), min(org_crp_b, img_h - 1)
        org_crp_l, org_crp_r = max(org_crp_l, 0), min(org_crp_r, img_w - 1)
        dst_crp_t, dst_crp_b = crp_h // 2 + org_crp_t - nail_cy, crp_h // 2 + org_crp_b - nail_cy
        dst_crp_l, dst_crp_r = crp_w // 2 + org_crp_l - nail_cx, crp_w // 2 + org_crp_r - nail_cx
        crp_img[dst_crp_t:dst_crp_b + 1, dst_crp_l:dst_crp_r + 1] = img[org_crp_t:org_crp_b + 1,
                                                                    org_crp_l:org_crp_r + 1]
        crp_mask[dst_crp_t:dst_crp_b + 1, dst_crp_l:dst_crp_r + 1] = mask[org_crp_t:org_crp_b + 1,
                                                                     org_crp_l:org_crp_r + 1]
        if is_thumb_bot: # resizing cropped image and mask
            crp_img = cv2.resize(crp_img, (regular_crp_w, regular_crp_h))
            skin_mask = np.zeros(crp_mask.shape, np.uint8)
            nail_mask = np.zeros(crp_mask.shape, np.uint8)
            skin_mask[np.where(crp_mask==finger_color[0])] = 255
            nail_mask[np.where(crp_mask==nail_color[0])] = 255
            skin_mask = cv2.resize(skin_mask, (regular_crp_w, regular_crp_h))
            nail_mask = cv2.resize(nail_mask, (regular_crp_w, regular_crp_h))
            crp_mask = np.zeros((regular_crp_h, regular_crp_w, 3), np.uint8)
            crp_mask[np.where(skin_mask>128)] = finger_color[0]
            crp_mask[np.where(nail_mask>128)] = nail_color[0]

        if 'topleft' in img_name:
            None
        elif 'left' in img_name:
            crp_img = np.transpose(crp_img, (1,0,2))[:,::-1]
            crp_mask = np.transpose(crp_mask, (1, 0, 2))[:, ::-1]
            angle = (angle+90)%180
        else:
            crp_img = np.transpose(crp_img, (1,0,2))[::-1]
            crp_mask = np.transpose(crp_mask, (1, 0, 2))[::-1]
            angle = (angle-90)%180

        which_nail = finger_names[nail_finger_idx]
        save_name = img_name.removesuffix('.jpg')+'_' + which_nail+'_%03d'%angle+'.png'
        cv2.imwrite(os.path.join(save_dir, 'images', save_name), crp_img)
        cv2.imwrite(os.path.join(save_dir, 'masks', save_name), crp_mask[:,:,0])
        '''
        crp_mask_show = crp_mask*120
        crp_img_mask_show = np.concatenate((crp_img, crp_mask_show), axis=1)
        cv2.imshow('crp', crp_img_mask_show)
        cv2.waitKey(0)
        '''
    '''
    img_h, img_w = img.shape[:2]
    img_show = cv2.resize(img, (img_w//5, img_h//5))
    mask_show = cv2.resize(mask, (img_w//5, img_h//5))
    mask_show *= 120
    img_mask_show = np.concatenate((img_show, mask_show), axis=1)
    cv2.imshow('show', img_mask_show)
    cv2.waitKey(0)
    '''




    print(img_path, '--', json_path)

print('Completed. ')

