
import os
import time
import json

import cv2
import numpy as np

image_dirs = ['/home/jinling/Documents/data/Buckets/deep-learning-datasets-10b/fingers-segmentation/p0_batches/images/devebec/batch_p0_04_09AUG_devebec',
'/home/jinling/Documents/data/Buckets/deep-learning-datasets-10b/fingers-segmentation/p0_batches/images/devebec/batch_p0_05_02NOV2023_devebec',
'/home/jinling/Documents/data/Buckets/deep-learning-datasets-10b/fingers-segmentation/p0_batches/images/devebec/batch_p0_05_24AUG_devebec',
'/home/jinling/Documents/data/Buckets/deep-learning-datasets-10b/fingers-segmentation/p0_batches/images/devebec/batch_p0_06_31OCT2023_devebec',
'/home/jinling/Documents/data/Buckets/deep-learning-datasets-10b/fingers-segmentation/p0_batches/images/devebec/batch_p0_09_30OCT2023_devebec',
              ]

json_dirs = ['/home/jinling/Documents/data/Buckets/deep-learning-datasets-10b/fingers-segmentation/p0_batches/annotations/devebec/batch_p0_04_09AUG_devebec',
            '/home/jinling/Documents/data/Buckets/deep-learning-datasets-10b/fingers-segmentation/p0_batches/annotations/devebec/batch_p0_05_02NOV2023_devebec/Lableme Json',
'/home/jinling/Documents/data/Buckets/deep-learning-datasets-10b/fingers-segmentation/p0_batches/annotations/devebec/batch_p0_05_24AUG_devebec',
            '/home/jinling/Documents/data/Buckets/deep-learning-datasets-10b/fingers-segmentation/p0_batches/annotations/devebec/batch_p0_06_31OCT2023_devebec/LabelMe_2/',
'/home/jinling/Documents/data/Buckets/deep-learning-datasets-10b/fingers-segmentation/p0_batches/annotations/devebec/batch_p0_06_31OCT2023_devebec/Lableme_1/',
'/home/jinling/Documents/data/Buckets/deep-learning-datasets-10b/fingers-segmentation/p0_batches/annotations/devebec/batch_p0_06_31OCT2023_devebec/LableMe/',
'/home/jinling/Documents/data/Buckets/deep-learning-datasets-10b/fingers-segmentation/p0_batches/annotations/devebec/batch_p0_06_31OCT2023_devebec/LableMe_3/',
'/home/jinling/Documents/data/Buckets/deep-learning-datasets-10b/fingers-segmentation/p0_batches/annotations/devebec/batch_p0_09_30OCT2023_devebec/Annotation_batch_p0_09_30OCT2023_devebec',
             '/home/jinling/Documents/data/Buckets/deep-learning-datasets-10b/fingers-segmentation/p0_batches/annotations/devebec/output_batch20231130171300_49151/output_batch20231130171300_49151/LableMe/'
                 ]

with_hand_dir = '/home/jinling/Documents/data/nail_classification/hand_imgs/real_hands'
hand_imgs_set = set([file.removesuffix('.png') for file in os.listdir(with_hand_dir) if '.png' in file])
std_json_path = '/home/jinling/Documents/data/picked_hands/json_files/NAIVE_A0_01-07-2022-09-56-33_All_left_.json'
std_json_data = json.load(open(std_json_path, encoding='gbk'))

save_dir = '/home/jinling/Documents/data/extracted_nails_jsons'

json_path_dict = dict()
for json_dir in json_dirs:
    json_names = [file.removesuffix('.json') for file in os.listdir(json_dir) if '.json' in file]
    for json_name in json_names:
        if json_name not in hand_imgs_set:
            continue
        json_path = os.path.join(json_dir, json_name+'.json')
        json_path_dict[json_name] = json_path

crp_ovrd_r = 1.0
for img_dir in image_dirs:
    img_names = [file.removesuffix('.png') for file in os.listdir(img_dir) if '.png' in file]
    for img_name in img_names:
        if img_name not in json_path_dict.keys():
            continue
        img_path = os.path.join(img_dir, img_name+'.png')
        json_path = json_path_dict[img_name]
        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]
        json_data = json.load(open(json_path, encoding='gbk'))
        img_nail_idx = 0
        for idx in range(len(json_data['shapes'])):
            if json_data['shapes'][idx]['label'] != 'N':
                continue
            points = json_data['shapes'][idx]['points']
            xs, ys = [p[0] for p in points], [p[1] for p in points]
            nail_l, nail_r, nail_t, nail_b = min(xs), max(xs), min(ys), max(ys)
            nail_h, nail_w = nail_b-nail_t, nail_r-nail_l
            crp_t, crp_b = nail_t-nail_h*crp_ovrd_r, nail_b+nail_h*crp_ovrd_r
            crp_l, crp_r = nail_l-nail_w*crp_ovrd_r, nail_r+nail_w*crp_ovrd_r
            crp_t, crp_b, crp_l, crp_r = [int(xx+0.5) for xx in [crp_t, crp_b, crp_l, crp_r]]
            crp_t, crp_b, crp_l, crp_r = max(crp_t, 0), min(crp_b, img_h-1), max(crp_l, 0), min(crp_r, img_w-1)
            crp_img = img[crp_t:crp_b+1, crp_l:crp_r+1]
            img_save_name = img_name+'%d.png'%img_nail_idx
            json_save_name = img_name+'%d.json'%img_nail_idx
            img_nail_idx += 1
            crp_json_data = std_json_data.copy()
            crp_json_data['imagePath'] = img_save_name
            crp_json_data['imageData'] = None
            crp_json_data['imageHeight'] = crp_img.shape[0]
            crp_json_data['imageWidth'] = crp_img.shape[1]
            crp_json_data['shapes'] = []
            shape = dict()
            shape['label'] = 'nail'
            crp_points = [[p[0]-crp_l,p[1]-crp_t] for p in points]
            shape['points'] = crp_points
            shape['group_id'] = None
            shape['description'] = ''
            shape['shape_type'] = 'polygon'
            shape['flags'] = {}
            shape['mask'] = None
            crp_json_data['shapes'].append(shape)
            img_save_path = os.path.join(save_dir, img_save_name)
            json_save_path = os.path.join(save_dir, json_save_name)
            cv2.imwrite(img_save_path, crp_img)
            with open(json_save_path, 'w') as save_json:
                json.dump(crp_json_data, save_json)
        print(img_path)



