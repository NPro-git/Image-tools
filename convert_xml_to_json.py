

# Already got json files for hands, but nails are not annotated
# generate the nail polygons from xmls

import os
import cv2
import numpy as np
import xmltodict
import json

std_json_path = '/home/jinling/Documents/data/picked_hands/json_files/NAIVE_A0_01-07-2022-09-56-33_All_left_.json'
xml_dirs = ['/home/jinling/Documents/data/Buckets/deep-learning-datasets-10b/fingers-segmentation/p0_batches/annotations/devebec/batch_p0_04_09AUG_devebec',
'/home/jinling/Documents/data/Buckets/deep-learning-datasets-10b/fingers-segmentation/p0_batches/annotations/devebec/batch_p0_05_24AUG_devebec',
'/home/jinling/Documents/data/Buckets/deep-learning-datasets-10b/fingers-segmentation/p0_batches/annotations/devebec/batch_p0_09_30OCT2023_devebec/Annotation_batch_p0_09_30OCT2023_devebec',
            ]

std_json_data = json.load(open(std_json_path, encoding='gbk'))

for xml_dir in xml_dirs:
    xml_files = [file for file in os.listdir(xml_dir) if '.xml' in file]
    for xml_file in xml_files:
        xml_path = os.path.join(xml_dir, xml_file)
        fd = open(xml_path, "r")
        xml_content = fd.read()
        fd.close()
        xml_data = xmltodict.parse(xml_content)
        json_data = std_json_data.copy()
        json_data['imagePath'] = xml_data['annotation']['filename']
        json_data['imageData'] = None
        json_data['imageHeight'] = xml_data['annotation']['size']['height']
        json_data['imageWidth'] = xml_data['annotation']['size']['width']
        json_data['shapes'] = []
        if 'object' not in xml_data['annotation'].keys():
            continue

        for obj in xml_data['annotation']['object']:
            shape = dict()
            shape['label'] = obj['name']
            xys = list(obj['polygon'].values())
            points = [[float(xys[idx*2]), float(xys[idx*2+1])] for idx in range(len(xys)//2)]
            shape['points'] = points
            shape['group_id'] = None
            shape['description'] = ''
            shape['shape_type'] = 'polygon'
            shape['flags'] = {}
            shape['mask'] = None
            json_data['shapes'].append(shape)
        json.dump(json_data,open(os.path.join(xml_dir, xml_file.removesuffix('.xml') + '.json'), 'w', encoding='gbk'))
        print(xml_path)












