
import os

restraints_dir = '/home/jinling/Documents/data/full-seg-datasets/models_performance/EP1_restraints/'
restraints_list = os.path.join(restraints_dir, 'restraints_list.txt')
save_dir = os.path.join(restraints_dir, 'ALLs')

fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
views = ['topleft', 'left', 'right', 'bottomleft', 'bottomright']

with open(restraints_list, 'r') as f:
    lines = f.readlines()
key_seg = 'prefix='
captures = [line[line.find(key_seg)+len(key_seg):].split('/')[0] for line in lines if key_seg in line]

for idx, capture in enumerate(captures):
    rig, date, tm = capture.split('_')
    for view in views:
        uri = os.path.join('s3://10b-output-backup/', capture, capture, view, 'image_All.jpg')
        save_name = 'EP1_' + date + '_' + tm + '_' + rig + '_XXX_' + view + '_handL_ALLLEDs.jpg'
        save_path = os.path.join(save_dir, save_name)
        cmd = 'aws s3 cp ' + uri + ' ' + save_path
        print(cmd)
        os.system(cmd)
    print('%d/%d: '%(idx+1, len(captures)), capture)


print('Completed. ')
