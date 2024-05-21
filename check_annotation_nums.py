
import os
import cv2
import numpy as np
import json

dir = '/home/jinling/Documents/data/picked_hands/json_files'

json_files = [file for file in os.listdir(dir) if '.json' in file]

save_dir = '/home/jinling/Documents/data/picked_hands/tmp'

fingers_hand_colors = [[255, 0, 0], [0, 255, 0], [255, 255, 0], \
                        [255, 0, 255], [0, 255, 255], [128, 128, 255]]
nail_color = [0, 0, 255]
classes = ['thumb', 'index', 'middle', 'ring', 'pinky', 'hand']

for json_file in json_files:
    json_path = os.path.join(dir, json_file)
    img_path = json_path.removesuffix('.json')+'.png'
    img = cv2.imread(img_path)
    img_h, img_w = img.shape[:2]
    mask = np.zeros(img.shape, dtype=np.uint8)
    strap_mark = np.zeros(img.shape, dtype=np.uint8)
    with open(json_path, encoding='gbk') as fd:
        json_data = json.load(fd)
    shapes = []  # eleminate the ignored areas
    finger_hand_num = 0
    for i in range(len(json_data['shapes'])):
        points = np.array(json_data['shapes'][i]['points'])
        points = [[int(p[0] + 0.5), int(p[1] + 0.5)] for p in points]
        label = json_data['shapes'][i]['label']
        if label == 'strap':
            cv2.fillPoly(strap_mark, [np.array(points, dtype=int)], (255, 255, 255))
            continue
        if label == 'ignore':
            cv2.fillPoly(mask, [np.array(points, dtype=int)], (255, 255, 255))
            continue
        if label == 'nail':
            continue
        cls_idx = classes.index(label)
        color = fingers_hand_colors[cls_idx]
        cv2.fillPoly(mask, [np.array(points, dtype=int)], color)
        finger_hand_num += 1
    mask[np.where(strap_mark==255)] = 0
    img = cv2.resize(img, (img_w//5, img_h//5))
    mask = cv2.resize(mask, (img_w // 5, img_h // 5))
    img_mask = np.concatenate((img, mask), axis=1)
    #cv2.imwrite(os.path.join(save_dir, img_path.split('/')[-1]), img_mask)
    cv2.imshow('img_mask', img_mask)
    cv2.waitKey(10)
    if finger_hand_num != 6:
        print('Not the right number. ')
    print(json_file)
