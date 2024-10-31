
import os
import time
from collections import defaultdict

img_dir = '/home/jinling/Documents/data/full-seg-datasets/models_performance/EP01/images'
mask_dir = '/home/jinling/Documents/data/full-seg-datasets/models_performance/EP01/masks'

save_dir = '/home/jinling/Documents/data/full-seg-datasets/models_performance/fingers_v4.18_alpha_p0_ep01_natural'
train_img_dir = os.path.join(save_dir, 'train', 'images')
train_mask_dir = os.path.join(save_dir, 'train', 'masks')
test_img_dir = os.path.join(save_dir, 'test', 'images')
test_mask_dir = os.path.join(save_dir, 'test', 'masks')

files = [file for file in os.listdir(img_dir) if '.png' in file]

hand_fingers_dict = defaultdict(list)

for file in files:
    hand = file[:file.rfind('_')]
    hand = hand.replace('_left_', '_').replace('_right_', '_').\
        replace('_bottomleft_', '_').replace('_bottomright_', '_').\
        replace('_topleft_', '_')
    hand_fingers_dict[hand].append(file)

hands = list(hand_fingers_dict.keys())
train_hands_num = int(len(hands)*0.7)
train_hands = hands[:train_hands_num]
test_hands = hands[train_hands_num:]

train_nails, test_nails = [], []
for hand in train_hands:
    train_nails.extend(hand_fingers_dict[hand])
for hand in test_hands:
    test_nails.extend(hand_fingers_dict[hand])

for nail in train_nails:
    frm_path = os.path.join(img_dir, nail)
    to_path = os.path.join(train_img_dir, nail)
    command = 'cp ' + frm_path + ' ' + to_path
    print(command)
    os.system(command)
    time.sleep(0.05)
    frm_path = os.path.join(mask_dir, nail)
    to_path = os.path.join(train_mask_dir, nail)
    command = 'cp ' + frm_path + ' ' + to_path
    print(command)
    os.system(command)
    time.sleep(0.05)

for nail in test_nails:
    frm_path = os.path.join(img_dir, nail)
    to_path = os.path.join(test_img_dir, nail)
    command = 'cp ' + frm_path + ' ' + to_path
    print(command)
    os.system(command)
    time.sleep(0.05)
    frm_path = os.path.join(mask_dir, nail)
    to_path = os.path.join(test_mask_dir, nail)
    command = 'cp ' + frm_path + ' ' + to_path
    print(command)
    os.system(command)
    time.sleep(0.05)

print('Completed. ')
