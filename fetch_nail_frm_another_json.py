
import os
import json

json_dir = '/home/jinling/Documents/data/picked_hands/thick_strap'
frm_json_dirs = ['/home/jinling/Documents/data/Buckets/deep-learning-datasets-10b/fingers-segmentation/p0_batches/annotations/devebec/batch_p0_04_09AUG_devebec',
            '/home/jinling/Documents/data/Buckets/deep-learning-datasets-10b/fingers-segmentation/p0_batches/annotations/devebec/batch_p0_05_02NOV2023_devebec/Lableme Json',
'/home/jinling/Documents/data/Buckets/deep-learning-datasets-10b/fingers-segmentation/p0_batches/annotations/devebec/batch_p0_05_24AUG_devebec',
            '/home/jinling/Documents/data/Buckets/deep-learning-datasets-10b/fingers-segmentation/p0_batches/annotations/devebec/batch_p0_06_31OCT2023_devebec/LabelMe_2/',
'/home/jinling/Documents/data/Buckets/deep-learning-datasets-10b/fingers-segmentation/p0_batches/annotations/devebec/batch_p0_06_31OCT2023_devebec/Lableme_1/',
'/home/jinling/Documents/data/Buckets/deep-learning-datasets-10b/fingers-segmentation/p0_batches/annotations/devebec/batch_p0_06_31OCT2023_devebec/LableMe/',
'/home/jinling/Documents/data/Buckets/deep-learning-datasets-10b/fingers-segmentation/p0_batches/annotations/devebec/batch_p0_06_31OCT2023_devebec/LableMe_3/',
'/home/jinling/Documents/data/Buckets/deep-learning-datasets-10b/fingers-segmentation/p0_batches/annotations/devebec/batch_p0_09_30OCT2023_devebec/Annotation_batch_p0_09_30OCT2023_devebec',
                 ]

json_files = [file for file in os.listdir(json_dir) if '.json' in file]
frm_dict = {}
for frm_json_dir in frm_json_dirs:
    for file in os.listdir(frm_json_dir):
        if '.json' not in file:
            continue
        name = file.split('/')[-1].removesuffix('.json')
        frm_dict[name] = os.path.join(frm_json_dir, file)

for json_file in json_files:
    name = json_file.split('/')[-1].removesuffix('.json')
    if name not in frm_dict.keys():
        print(name, 'frm json file not found. ')
        continue
    frm_path = frm_dict[name]
    json_path = os.path.join(json_dir, json_file)
    json_data = json.load(open(json_path, encoding='gbk'))
    #json_nail_num = len(['N' for idx in range(len(json_data['shapes'])) if json_data['shapes'][idx]['label']=='N'])
    json_nail_num = 0
    for idx in range(len(json_data['shapes'])):
        if json_data['shapes'][idx]['label'] == 'N':
            json_nail_num += 1
            json_data['shapes'][idx]['label'] = 'nail'
        if json_data['shapes'][idx]['label'] == 'nail':
            json_nail_num += 1
    if json_nail_num > 0:
        continue
    frm_data = json.load(open(frm_path, encoding='gbk'))
    for idx in range(len(frm_data['shapes'])):
        label = frm_data['shapes'][idx]['label']
        if label != 'N':
            continue
        shape = frm_data['shapes'][idx]
        shape['label'] = 'nail'
        json_data['shapes'].append(shape)
    json.dump(json_data, open(json_path, 'w', encoding='gbk'))
    print(json_path)
#p0-06_12-07-2023-15-19-52_All_right_