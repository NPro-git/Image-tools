

import os
import json

base_dir = '/home/jinling/Documents/data/Buckets/EP01/10b-capture-backup/ALLLEDs/RyanCaptureBefore20240709'

dirs = []

for dir_l1 in os.listdir(base_dir):
    if not os.path.isdir(os.path.join(base_dir, dir_l1)):
        continue
    for dir_l2 in os.listdir(os.path.join(base_dir, dir_l1)):
        if not os.path.isdir(os.path.join(base_dir, dir_l1, dir_l2)):
            continue
        for dir_l3 in os.listdir(os.path.join(base_dir, dir_l1, dir_l2)):
            dir = os.path.join(base_dir, dir_l1, dir_l2, dir_l3)
            if not os.path.isdir(dir):
                continue
            dirs.append(dir)
dirs = sorted(dirs)

for dir_idx, dir in enumerate(dirs):
    id = dir.split('/')[-1]
    try:
        rig, date, tm = id.split('_')
    except:
        print('here. ')
    uri = os.path.join('s3://10b-capture-backup/', rig, date, tm)
    uri_exists = os.system('aws s3 ls '+uri)
    print(uri_exists)
    continue
    if uri_exists == 0: #exists
        continue
    ref_cmd = 'aws s3api put-object --bucket 10b-capture-backup --key 10000000041d8f42/kk/ww/ss/ --content-length 0'
    cmd = 'aws s3api put-object --bucket 10b-capture-backup --key '+os.path.join(rig,date,tm)+'/ --content-length 0'
    #os.system(cmd)
    cmd = 'aws s3 cp ' + dir + ' ' + uri + ' --recursive'
    #os.system(cmd)

    print('%d/%d: '%(dir_idx+1, len(dirs)), uri_exists)

print('Completed. ')
