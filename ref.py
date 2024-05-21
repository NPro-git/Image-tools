
import os
import json

dir = '/home/jinling/Documents/data/picked_hands/edited_json_files/'
save_dir = '/home/jinling/Documents/data/picked_hands/debec_jsons'

json_files = [file for file in os.listdir(dir) if '.json' in file and 'NAIVE' in file]

for json_file in json_files:
    json_path = os.path.join(dir, json_file)
    with open(json_path, encoding='gbk') as f:
        json_data = json.load(f)
    save_json_file = 'DEBEVEC_A' + json_file.removeprefix('NAIVE_A')
    save_path = os.path.join(save_dir, save_json_file)
    json_data['imagePath'] = save_json_file.removesuffix('.json')+'.png'
    with open(save_path, 'w', encoding='gbk') as f:
        json.dump(json_data, f)
    print(json_file)
