
import os
import cv2
import numpy as np
import json

dir = '/home/jinling/Documents/data/Buckets/EP01/10b-capture-backup/ALLLEDs/hands_10b_20240605/bottomright'

json_files = [file for file in os.listdir(dir) if '.json' in file]

save_dir = '/home/jinling/Documents/data/picked_hands/tmp'

fingers_hand_colors = [[255, 0, 0], [0, 255, 0], [255, 255, 0], \
                        [255, 0, 255], [0, 255, 255], [128, 128, 255]]
nail_color = [0, 0, 255]
classes = ['thumb', 'index', 'middle', 'ring', 'pinkie', 'palm']

for json_file in json_files:
    print(json_file)
    json_path = os.path.join(dir, json_file)
    img_path = json_path.removesuffix('.json')+'.jpg'
    img = cv2.imread(img_path)
    img_h, img_w = img.shape[:2]
    mask = np.zeros(img.shape, dtype=np.uint8)
    strap_mark = np.zeros(img.shape, dtype=np.uint8)
    with open(json_path, encoding='gbk') as fd:
        json_data = json.load(fd)
    shapes = []  # eleminate the ignored areas
    for i in range(len(json_data['shapes'])):
        points = np.array(json_data['shapes'][i]['points'])
        points = [[int(p[0] + 0.5), int(p[1] + 0.5)] for p in points]
        label = json_data['shapes'][i]['label']
        if label == 'strap' or label == 'jewlery':
            cv2.fillPoly(strap_mark, [np.array(points, dtype=int)], (255, 255, 255))
            continue
        if label == 'ignore':
            cv2.fillPoly(mask, [np.array(points, dtype=int)], (255, 255, 255))
            continue
        if label == 'nail':
            continue
        if '_N' in label:
            continue
        cls_idx = classes.index(label)
        color = fingers_hand_colors[cls_idx]
        cv2.fillPoly(mask, [np.array(points, dtype=int)], color)
    for i in range(len(json_data['shapes'])):
        points = np.array(json_data['shapes'][i]['points'])
        points = [[int(p[0] + 0.5), int(p[1] + 0.5)] for p in points]
        label = json_data['shapes'][i]['label']
        if '_N' not in label:
            continue
        color = nail_color
        cv2.fillPoly(mask, [np.array(points, dtype=int)], color)
    mask[np.where(strap_mark==255)] = 0
    img = cv2.resize(img, (img_w//5, img_h//5))
    mask = cv2.resize(mask, (img_w // 5, img_h // 5))
    img_mask = np.concatenate((img, mask), axis=1)
    cv2.imwrite(os.path.join(save_dir, img_path.split('/')[-1]), img_mask)
    cv2.imshow('img_mask', img_mask)
    cv2.waitKey(10)

