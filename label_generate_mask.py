
import os
import numpy as np
import cv2
import json
import pickle

dir = 'C:/FILEs/data/fingernails/'
save_dir = 'C:/FILEs/data/nails/'

json_files = [file for file in os.listdir(dir) if '.json' in file]

over_r = 0.15

for json_file in json_files:
    png_file = json_file.strip('.json')+'.png'
    json_path = os.path.join(dir, json_file)
    png_path = os.path.join(dir, png_file)
    json_data = json.load(open(json_path, encoding='gbk'))
    img = cv2.imread(png_path)
    img_h, img_w = img.shape[:2]
    for shape_idx in range(len(json_data['shapes'])):
        points = np.array(json_data['shapes'][shape_idx]['points'])
        points = [[int(p[0] + 0.5), int(p[1] + 0.5)] for p in points]
        lx, rx = min(p[0] for p in points), max(p[0] for p in points)
        ty, by = min(p[1] for p in points), max(p[1] for p in points)
        over_dx, over_dy = int((rx-lx)*over_r), int((by-ty)*over_r)
        lx, rx = max(0, lx-over_dx), min(img_w-1, rx+over_dx)
        ty, by = max(0, ty-over_dy), min(img_h-1, by+over_dy)
        points = [[p[0]-lx, p[1]-ty] for p in points]
        cropped_img = img[ty:by+1, lx:rx+1]
        '''
        for coor_x, coor_y in points:
            cropped_img[coor_y, coor_x] = np.array([255, 0, 0], dtype=np.uint8)
        '''
        mask = np.zeros(cropped_img.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(points, dtype=int)], (255, 255, 255))

        save_name = json_file.strip('.json')+'_%02d'%shape_idx
        save_img_name = save_name+'.png'
        save_img_path = os.path.join(save_dir, save_img_name)
        '''
        save_pkl_name = save_name+'.pkl'
        contour = getContours(mask[::,::,0], th=128)[0]
        save_pkl_path = os.path.join(save_dir, save_pkl_name)
        file = open(save_pkl_path, 'wb')
        pickle.dump(contour, file)
        pickle.dump(points, file)
        file.close()
        '''

        cropped_img_mask = np.concatenate((cropped_img, mask), axis=1)
        cv2.imwrite(save_img_path, cropped_img)

        cv2.imshow('crop_mask', cropped_img_mask)
        cv2.waitKey(200)


        print('here. ')



print('here. ')



