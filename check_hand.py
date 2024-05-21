
import os
import json

json_dir = '/home/jinling/Documents/data/picked_hands/json_files'

json_files = [file for file in os.listdir(json_dir) if '.json' in file]

for json_file in json_files:
    name = json_file.split('/')[-1].removesuffix('.json')
    json_path = os.path.join(json_dir, json_file)
    json_data = json.load(open(json_path, encoding='gbk'))
    json_hand_num = len(['hand' for idx in range(len(json_data['shapes']))\
                         if json_data['shapes'][idx]['label']=='hand'])
    if json_hand_num > 0:
        print(json_path)
    else:
        print(json_path + '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    print(json_path)
#p0-06_12-07-2023-15-19-52_All_right_

