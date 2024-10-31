
import os
import json
import time

dir = '/home/jinling/Documents/data/Buckets/EP01/10b-capture-backup/ALLLEDs/hands_EP01_20240605_hdr'

files = [file for file in os.listdir(dir) if '.json' in file]
for file in files:
    with open(os.path.join(dir, file), encoding='gbk') as fd:
        json_data = json.load(fd)
    imgPath = json_data['imagePath']
    json_data['imagePath'] = imgPath.replace('ALLLEDs', 'DEBEVEC')
    with open(os.path.join(dir, file), 'w', encoding='gbk') as fd:
        json.dump(json_data, fd)
    new_name = file.replace('ALLLEDs', 'DEBEVEC')
    org_path = os.path.join(dir, file)
    new_path = os.path.join(dir, new_name)
    command = 'mv ' + org_path + ' ' + new_path
    print(command)
    os.system(command)
    print(file)
'''
files = [file for file in os.listdir(dir) if '.jpg' in file]
for file in files:
    org_path = os.path.join(dir, file)
    new_name = file.removesuffix('.png')+'ALLLEDs.png'
    new_path = os.path.join(dir, new_name)
    command = 'mv ' + org_path + ' ' + new_path
    print(command)
    os.system(command)
    print(file)
'''
print('Completed. ')
