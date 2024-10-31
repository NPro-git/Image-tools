

import os
import json
import random
import numpy as np
import cv2

dirs = [
'/home/jinling/Documents/data/Buckets/EP01/10b-capture-backup/ALLLEDs/annotations/good_samples/picked_large_annotation_natural'
    ]

save_dir = '/home/jinling/Documents/data/Buckets/EP01/10b-capture-backup/ALLLEDs/annotations/good_samples/nail_samples'

finger_palm_names = ['thumb', 'index', 'middle', 'ring', 'pinky', 'palm']
finger_colors = [(vv, vv, vv) for vv in [40, 80, 120, 160, 200]]
huge_crp_h, huge_crp_w = 640, 640
regular_crp_h, regular_crp_w = 512, 512
crp_h, crp_w = 512, 512
nail_color = (2, 2, 2)
hand_color = (1, 1, 1)

img_json_pairs = []
for dir in dirs:
    json_files = [file for file in os.listdir(dir) if '.json' in file]
    for json_file in json_files:
        img_file = json_file.removesuffix('.json')+'.jpg'
        img_path = os.path.join(dir, img_file)
        if os.path.exists(img_path):
            json_path = os.path.join(dir, json_file)
            img_json_pairs.append([img_path, json_path])

for pair_idx, (img_path, json_path) in enumerate(img_json_pairs):
    img = cv2.imread(img_path)
    img_name = img_path.split('/')[-1]
    img_h, img_w = img.shape[:2]
    with open(json_path, encoding='gbk') as fd:
        json_data = json.load(fd)
    hand_mask = np.zeros(img.shape, dtype=np.uint8)
    fingers_mask = np.zeros(img.shape, dtype=np.uint8)
    for idx in range(len(json_data['shapes'])):
        label = json_data['shapes'][idx]['label']
        if label not in finger_palm_names:  # draw fingers first
            continue
        points = [[int(p[0] + 0.5), int(p[1] + 0.5)] for p in json_data['shapes'][idx]['points']]
        cv2.fillPoly(hand_mask, [np.array(points, dtype=int)], hand_color)
        finger_palm_idx = finger_palm_names.index(label)
        if finger_palm_idx >= 5:
            continue
        cv2.fillPoly(fingers_mask, [np.array(points, dtype=int)], finger_colors[finger_palm_idx])
    for idx in range(len(json_data['shapes'])):
        label = json_data['shapes'][idx]['label']
        if label != 'nail':
            continue
        points = [[int(p[0] + 0.5), int(p[1] + 0.5)] for p in json_data['shapes'][idx]['points']]
        cv2.fillPoly(hand_mask, [np.array(points, dtype=int)], nail_color)
    for idx in range(len(json_data['shapes'])):
        label = json_data['shapes'][idx]['label']
        if label != 'nail':
            continue
        cur_nail_mask = np.zeros(img.shape, dtype=np.uint8)
        points = [[int(p[0] + 0.5), int(p[1] + 0.5)] for p in json_data['shapes'][idx]['points']]
        cv2.fillPoly(cur_nail_mask, [np.array(points, dtype=int)], (255, 255, 255))
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        nail_l, nail_r, nail_t, nail_b = min(xs), max(xs), min(ys), max(ys)
        nail_h, nail_w = nail_b-nail_t, nail_r-nail_l
        is_huge_nail = nail_h > regular_crp_h*0.85 or nail_w > regular_crp_w*0.85
        crp_h, crp_w = (huge_crp_h, huge_crp_w) if is_huge_nail else (regular_crp_h, regular_crp_w)
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
        crp_mask[dst_crp_t:dst_crp_b + 1, dst_crp_l:dst_crp_r + 1] = hand_mask[org_crp_t:org_crp_b + 1,
                                                                     org_crp_l:org_crp_r + 1]
        if is_huge_nail: # resizing cropped image and mask
            crp_img = cv2.resize(crp_img, (regular_crp_w, regular_crp_h))
            crp_skin_mask = np.zeros(crp_mask.shape, np.uint8)
            crp_nail_mask = np.zeros(crp_mask.shape, np.uint8)
            crp_skin_mask[np.where(crp_mask==hand_color[0])] = 255
            crp_nail_mask[np.where(crp_mask==nail_color[0])] = 255
            skin_mask = cv2.resize(crp_skin_mask, (regular_crp_w, regular_crp_h))
            crp_nail_mask = cv2.resize(crp_nail_mask, (regular_crp_w, regular_crp_h))
            crp_skin_mask = cv2.resize(crp_skin_mask, (regular_crp_w, regular_crp_h))
            crp_mask = np.zeros((regular_crp_h, regular_crp_w, 3), np.uint8)
            crp_mask[np.where(crp_skin_mask>128)] = hand_color[0]
            crp_mask[np.where(crp_nail_mask>128)] = nail_color[0]

        if 'topleft' in img_name:
            None
        elif 'left' in img_name:
            crp_img = np.transpose(crp_img, (1,0,2))[:,::-1]
            crp_mask = np.transpose(crp_mask, (1, 0, 2))[:, ::-1]
        else:
            crp_img = np.transpose(crp_img, (1,0,2))[::-1]
            crp_mask = np.transpose(crp_mask, (1, 0, 2))[::-1]

        which_nail_idx = 0
        max_overlap = 0
        for fg_idx, finger in enumerate(finger_palm_names[:-1]):
            overlap_points = np.where((cur_nail_mask>0)*(fingers_mask==finger_colors[fg_idx]))
            if max_overlap < len(overlap_points[0]):
                max_overlap = len(overlap_points[0])
                which_nail_idx = fg_idx
        which_nail = finger_palm_names[which_nail_idx]
        save_name = img_name.removesuffix('.jpg')+'_' + which_nail+'.png'
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




    print('%d/%d: '%(pair_idx+1, len(img_json_pairs)), img_path)

print('Completed. ')

