
import os

ALL_dir = '/home/jinling/Documents/data/Buckets/EP01/10b-capture-backup/ALLLEDs/EP1_ValidationScans/ALLs'
save_dir = '/home/jinling/Documents/data/Buckets/EP01/10b-capture-backup/ALLLEDs/EP1_ValidationScans/segs'

fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']

img_files = [file for file in os.listdir(ALL_dir) if '.jpg' in file]

for img_idx, img_file in enumerate(img_files):
    date, tm, rig, tester, view, chi = img_file.split('_')[1:-1]
    rig_date_tm = rig + '_' + date + '_' + tm
    for finger in fingers:
        #'s3://10b-output-backup/10000000edd61851_2024-10-04_16-41-52/middle/segmentation/left_seg.png'
        seg_uri = os.path.join('s3://10b-output-backup', rig_date_tm, finger, 'segmentation', view+'_seg.png')
        save_name = img_file.replace('.jpg', '_'+finger+'.png')
        save_path = os.path.join(save_dir, save_name)
        cmd = 'aws s3 cp ' + seg_uri + ' ' + save_path
        os.system(cmd)
        print(cmd)



    print('%d/%d: '%(img_idx+1, len(img_files)), img_file)

print('Completed. ')
