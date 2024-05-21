
import os

json_dir = '/home/jinling/Documents/data/picked_hands/json_files'
img_dir = '/home/jinling/Documents/data/Buckets/deep-learning-datasets-10b/fingers-segmentation/raw_naive_hdr_only'
dst_dir = '/home/jinling/Documents/data/picked_hands/raw_naive_hdr_only'

json_files = [file for file in os.listdir(json_dir) if '.json' in file]
for json_file in json_files:
    img_file = json_file.strip('.json')+'.png'
    img_path = os.path.join(img_dir, img_file)
    if not os.path.exists(img_path):
        continue
    if os.path.exists(os.path.join(dst_dir, img_file)):
        continue
    command = 'cp '+ img_path+ ' ' + dst_dir
    os.system(command)
    print(json_file)

print('Completed. ')
