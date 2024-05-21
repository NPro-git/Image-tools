
# Remove the image data from labelme json files. 

import os
import json

src_dir = '/home/jinling/Documents/data/picked_hands/edited_json_files_tobedeleted/'
dst_dir = '/home/jinling/Documents/data/picked_hands/edited_json_files_tobedeleted'
#src_dir = '/home/jinling/Documents/data/Buckets/deep-learning-datasets-10b/fingers-segmentation/raw_naive_hdr_only/images'
#dst_dir = '/home/jinling/Documents/data/Buckets/deep-learning-datasets-10b/fingers-segmentation/raw_naive_hdr_only/images'

json_files = [file for file in os.listdir(src_dir) if '.json' in file]
for json_file in json_files:
    json_data = json.load(open(os.path.join(src_dir, json_file), encoding='gbk'))
    json_data['imageData'] = None
    #if 'imageData' in json_data.keys():
    #    del json_data['imageData']
    json.dump(json_data, open(os.path.join(dst_dir, json_file), 'w', encoding='gbk'))
    print(json_file)


