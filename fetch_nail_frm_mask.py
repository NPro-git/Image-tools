
import os
import json
import numpy as np
import cv2
from contour import getContours

json_dir = '/home/jinling/Documents/data/picked_hands/edited_json_files_tobedeleted/'
mask_dir = '/home/jinling/Documents/data/Buckets/deep-learning-datasets-10b/fingers-segmentation/alpha_devebec/DEVEBEC/masks/'
json_files = [file for file in os.listdir(json_dir) if '.json' in file]
mask_files = [file for file in os.listdir(mask_dir) if '_all_objects.png' in file]

mask_dict = {}
for mask_file in mask_files:
    name = mask_file.removesuffix('_all_objects.png')
    mask_dict[name] = os.path.join(mask_dir, mask_file)

json_mask_pairs = []
for json_file in json_files:
    name = json_file.removesuffix('.json')
    if name not in mask_dict.keys():
        continue
    json_path = os.path.join(json_dir, json_file)
    mask_path = mask_dict[name]
    json_mask_pairs.append([json_path, mask_path])

for pair in json_mask_pairs:
    json_path, mask_path = pair
    json_data = json.load(open(json_path, encoding='gbk'))
    mask = cv2.imread(mask_path)
    #mask = cv2.resize(mask, (500, 400))
    nail_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
    nail_mask[np.where((mask[:,:,0]<128)*(mask[:,:,1]>128)*(mask[:,:,2]>128))] = 255
    contours = getContours(nail_mask, 128)
    for contour in contours:
        shape = dict()
        shape['label'] = 'nail'
        shape['points'] = [contour[idx] for idx in range(0, len(contour), 25)]
        shape['group_id'] = None
        shape['description'] = ''
        shape['shape_type'] = 'polygon'
        shape['flags'] = {}
        shape['mask'] = None
        json_data['shapes'].append(shape)
    json.dump(json_data, open(json_path, 'w', encoding='gbk'))
    #cv2.imshow('mask', nail_mask)
    #cv2.waitKey(0)
    print(json_path)