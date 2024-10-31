
import os
import json

json_dirs = [
    '/home/jinling/Documents/data/Buckets/EP01/10b-capture-backup/ALLLEDs/annotations/imgs_annotations_for_autoroi/p0_images',
             ]

for json_dir in json_dirs:
    json_files = [file for file in os.listdir(json_dir) if '.json' in file]


    for json_file in json_files:
        name = json_file.split('/')[-1].removesuffix('.json')
        json_path = os.path.join(json_dir, json_file)
        json_data = json.load(open(json_path, encoding='gbk'))
        for idx in range(len(json_data['shapes'])):
            if 'hand' in json_data['shapes'][idx]['label']:
                json_data['shapes'][idx]['label'] = 'palm'
            if 'pinkie' in json_data['shapes'][idx]['label']:
                json_data['shapes'][idx]['label'] = 'pinky'
        json.dump(json_data, open(json_path, 'w', encoding='gbk'))
        print(json_path)
