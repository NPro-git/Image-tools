
import os
import json

json_dir = '/home/jinling/Documents/data/picked_hands/json_files'

json_files = [file for file in os.listdir(json_dir) if '.json' in file]


for json_file in json_files:
    name = json_file.split('/')[-1].removesuffix('.json')
    json_path = os.path.join(json_dir, json_file)
    json_data = json.load(open(json_path, encoding='gbk'))
    for idx in range(len(json_data['shapes'])):
        if json_data['shapes'][idx]['label'] == 'N':
            json_data['shapes'][idx]['label'] = 'nail'
    json.dump(json_data, open(json_path, 'w', encoding='gbk'))
    print(json_path)
