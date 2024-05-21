
import os
import json
import cv2
import numpy as np

#src_dir = '/home/jinling/Documents/data/picked_hands/json_files/'
#dst_dir = '/home/jinling/Documents/data/picked_hands/json_files'
src_dir = '/home/jinling/Documents/data/picked_hands/edited_json_files_tobedeleted/'
dst_dir = '/home/jinling/Documents/data/picked_hands/edited_json_files_tobedeleted/'

json_files = [file for file in os.listdir(src_dir) if '.json' in file]
json_files = sorted(json_files)
for json_file in json_files:
    img_path = os.path.join(src_dir, json_file.removesuffix('.json')+'.png')
    img = cv2.imread(img_path)
    json_data = json.load(open(os.path.join(src_dir, json_file), encoding='gbk'))
    json_data['imageData'] = None
    #if 'imageData' in json_data.keys():
    #    del json_data['imageData']
    shapes = []
    for shape_idx in range(len(json_data['shapes'])):
        label = json_data['shapes'][shape_idx]['label']
        if label == 'nail':
            mask = np.zeros(img.shape, dtype=np.uint8)
            points = np.array(json_data['shapes'][shape_idx]['points'])
            points = [[int(p[0] + 0.5), int(p[1] + 0.5)] for p in points]
            cv2.fillPoly(mask, [np.array(points, dtype=int)], (255, 255, 255))
            area = len(np.where(mask>128)[0])
            if area < 5000:
                continue
        shapes.append(json_data['shapes'][shape_idx])
    json_data['shapes'] = shapes
    json.dump(json_data, open(os.path.join(dst_dir, json_file), 'w', encoding='gbk'))
    print(json_file)


