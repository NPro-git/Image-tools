
import os
import cv2
import numpy as np

img_h, img_w = 512, 512
dir = '/home/jinling/Documents/data/full-seg-datasets/models_performance/segs_beit1.3_on_data4.11/'

img_files = [file for file in os.listdir(dir) if '.png' in file]

def get_contour(bin_img):
    contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = [[coor[0] for coor in contour] for contour in contours]
    c_contour = contours[0]
    c_dist = img_h
    for idx, contour in enumerate(contours):
        xs, ys = [coor[0] for coor in contour], [coor[1] for coor in contour]
        cx, cy = sum(xs)/len(xs), sum(ys)/len(ys)
        dist = np.sqrt((cx-img_w/2)**2 + (cy-img_h/2)**2)
        if dist < c_dist:
            c_dist = dist
            c_contour = contours[idx]
    return c_contour

tgt_color = [0, 0, 255]
seg_color = [0, 255, 0]
i_color = [0, 0, 255]
o_color = [0, 255, 0]
for img_file in img_files:
    path = os.path.join(dir, img_file)
    performance_img = cv2.imread(path)
    img = performance_img[:img_h,:img_w]
    seg = performance_img[img_h:,:img_w,2]
    tgt = performance_img[img_h:,img_w:,2]
    contour_img = np.zeros(img.shape, dtype=np.uint8)
    tgt_contour = get_contour(tgt)
    xs, ys = [coor[0] for coor in tgt_contour], [coor[1] for coor in tgt_contour]
    cx, cy = sum(xs) // len(xs), sum(ys) // len(ys)
    max_delta_dist = 0
    max_tx, max_ty = tgt_contour[0]
    max_sx, max_sy = cx, cy
    for tx, ty in tgt_contour:
        sx, sy = cx, cy
        if abs(ty-cy) < abs(tx-cx):
            kk = (ty-cy)/(tx-cx)
            bb = ty - kk * tx
            if tx > cx:
                for xx in range(cx, img_w):
                    yy = int(kk*xx + bb+0.5)
                    if seg[yy,xx] > 128:
                       sx, sy = xx, yy
                    else:
                        break
            else:
                for xx in range(cx, -1, -1):
                    yy = int(kk*xx + bb +0.5)
                    if seg[yy,xx] > 128:
                        sx, sy = xx, yy
                    else:
                        break
        else:
            div_k = (tx-cx)/(ty-cy)
            ba = tx - div_k*ty
            if ty > cy:
                for yy in range(cy, img_h):
                    xx = int(div_k*yy+ba+0.5)
                    if seg[yy,xx] > 128:
                        sx, sy = xx, yy
                    else:
                        break
            else:
                for yy in range(cy, -1, -1):
                    xx = int(div_k*yy+ba+0.5)
                    if seg[yy,xx] > 128:
                        sx, sy = xx, yy
                    else:
                        break
        contour_img[ty,tx] = tgt_color
        contour_img[sy,sx] = seg_color
        tgt_dist = np.sqrt((tx-cx)**2+(ty-cy)**2)
        seg_dist = np.sqrt((sx-cx)**2+(sy-cy)**2)
        delta_dist = np.abs(tgt_dist - seg_dist)
        if max_delta_dist < delta_dist:
            max_delta_dist = delta_dist
            max_tx, max_ty, max_sx, max_sy = tx, ty, sx, sy
    #out of loop
    t_dist = np.sqrt((tx-cx)**2+(ty-cy)**2)
    s_dist = np.sqrt((sx-cx)**2+(sy-cy)**2)
    if s_dist > 0.1:
        if t_dist > s_dist:
            ix, iy = max_sx, max_sy
            ox, oy = max_tx, max_ty
        else:
            ix, iy = max_tx, max_ty
            ox, oy = max_sx, max_sy
        if abs(ix-cx) > abs(iy-cy):
            kk = (iy-cy)/(ix-cx)
            bb = cy - kk*cx
            min_x, max_x = min(ix, cx), max(ix, cx)
            for xx in range(min_x, max_x+1):
                yy = int(kk*xx + bb+0.5)
                contour_img[yy,xx] = i_color
            min_x, max_x = min(ix, ox), max(ix, ox)
            for xx in range(min_x, max_x+1):
                yy = int(kk*xx + bb+0.5)
                contour_img[yy,xx] = o_color
        else:
            div_k = (ix-cx)/(iy-cy)
            ba = cx - div_k*cy
            min_y, max_y = min(iy, cy), max(iy, cy)
            for yy in range(min_y, max_y+1):
                xx = int(div_k*yy+ba+0.5)
                contour_img[yy,xx] = i_color
            min_y, max_y = min(iy, oy), max(iy, oy)
            for yy in range(min_y, max_y+1):
                xx = int(div_k*yy+ba+0.5)
                contour_img[yy,xx] = o_color




    cv2.imshow('tgt_contour', np.concatenate((img, contour_img), axis=1))
    cv2.waitKey(0)




    delta_dist = np.abs(s_dist - t_dist)
    print('delta: ', delta_dist)



