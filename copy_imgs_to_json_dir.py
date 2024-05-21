
import os
import cv2
import json

json_dir = '/home/jinling/Documents/data/picked_hands/edited_json_files_tobedeleted'
img_dirs = ['/home/jinling/Documents/data/Buckets/deep-learning-datasets-10b/fingers-segmentation/alpha_devebec/DEVEBEC/images/']

json_files = [file for file in os.listdir(json_dir) if '.json' in file]
img_dict = dict()
for img_dir in img_dirs:
    for file in os.listdir(img_dir):
        if '.png' not in file:
            continue
        name = file.removesuffix('.png')
        img_dict[name] = os.path.join(img_dir, file)

for json_file in json_files:
    name = json_file.removesuffix('.json')
    if name not in img_dict.keys():
        continue
    json_path = os.path.join(json_dir, json_file)
    img_path = img_dict[name]
    save_path = os.path.join(json_dir, name+'.png')
    if os.path.exists(save_path):
        continue
    command = 'cp ' + img_path + ' ' + save_path
    os.system(command)
    #img = cv2.imread(img_path)
    #cv2.imwrite(save_path, img)
    print(command)
