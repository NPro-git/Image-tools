
import os

import cv2
import numpy as np
from collections import defaultdict

base_dir = '/home/jinling/Documents/data/Buckets/EP01/10b-capture-backup/ALLLEDs/'

dirs = [os.path.join(base_dir, file) for file in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, file))]

id_dict = defaultdict(list)
for dir in dirs:
    files = [file for file in os.listdir(dir) if '.jpg' in file]
    for file in files:
        date, tm, rig, name, view = file.split('_')[1:6]
        id = date+'_'+tm+'_'+rig+'_'+view
        id_dict[id].append(os.path.join(dir, file))

for key in id_dict.keys():
    if len(id_dict[key]) < 2:
        continue
    for path in id_dict[key]:
        print(path, end='')
        if 'XXX' in path:
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            cmd = 'rm ' + path
            #os.system(cmd)
        else:
            print(' ')



    '''
    show = np.concatenate(imgs, axis=1)
    show = cv2.resize(show, (500*len(imgs), 400))
    cv2.imshow('show', show)
    cv2.waitKey(0)
    '''
print('Completed. ')



