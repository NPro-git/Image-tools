
import os
import cv2
import numpy as np

beit10_dir = '/home/jinling/Documents/data/full-seg-datasets/models_performance/segs_beit1.0_on_data4.11'
beit13_dir = '/home/jinling/Documents/data/full-seg-datasets/models_performance/segs_beit1.3_on_data4.11'
swin_dir = '/home/jinling/Documents/data/full-seg-datasets/models_performance/swin'
compare_dir = '/home/jinling/Documents/data/full-seg-datasets/models_performance/compare'

files = [file for file in os.listdir(beit10_dir) if '.png' in file]
img_h, img_w = 512, 512

#nail_color = np.array([[[0, 0, 255]]], dtype=np.uint8)
nail_color = np.array([0, 0, 255], dtype=np.uint8)
fg_color = np.array([0, 255, 0], dtype=np.uint8)
bg_color = np.array([0, 0, 0], dtype=np.uint8)

#https://www.jeremyjordan.me/evaluating-image-segmentation-models/
#accuracy = (TP+TN)/(TP+TN+FP+FN)
def get_iou_acc_dice(seg, anno, color):
    anno_points = np.where(np.all(anno==color, axis=-1))
    seg_points = np.where(np.all(seg==color, axis=-1))
    seg_mask = np.zeros(seg.shape[:2], dtype=np.uint8)
    anno_mask = np.zeros(anno.shape[:2], dtype=np.uint8)
    seg_mask[seg_points] = 255
    anno_mask[anno_points] = 255
    I = np.where((anno_mask>128)*(seg_mask>128))
    U = np.where(np.maximum(anno_mask, seg_mask)>128)
    TP = np.where((anno_mask>128)*(seg_mask>128))
    TN = np.where((anno_mask<=128)*(seg_mask<=128))
    FP = np.where((seg_mask>128)*(anno_mask<=128))
    FN = np.where((seg_mask<=128)*(anno_mask>128))
    '''
    seg_h, seg_w = seg.shape[:2]
    shows = np.zeros((seg_h, seg_w*4), dtype=np.uint8)
    shows[:,:seg_w][anno_points] = 255
    shows[:, seg_w:seg_w*2][seg_points] = 255
    shows[:, seg_w*2:seg_w*3][I_points] = 255
    shows[:, seg_w*3:][U_points] = 255
    cv2.imshow('shows', shows)
    cv2.waitKey(0)
    '''
    iou = len(I[0])/len(U[0])
    acc = (len(TP[0])+len(TN[0]))/(len(TP[0])+len(TN[0])+len(FP[0])+len(FN[0]))
    dice = 2*len(I[0])/(len(anno_points[0])+len(seg_points[0]))

    return iou, acc, dice

iou_nail_13_ave, acc_nail_13_ave, dice_nail_13_ave, iou_fg_13_ave, acc_fg_13_ave, dice_fg_13_ave, iou_bg_13_ave, acc_bg_13_ave, dice_bg_13_ave = 0, 0, 0, 0, 0, 0, 0, 0, 0
iou_nail_swin_ave, acc_nail_swin_ave, dice_nail_swin_ave, iou_fg_swin_ave, acc_fg_swin_ave, dice_fg_swin_ave, iou_bg_swin_ave, acc_bg_swin_ave, dice_bg_swin_ave = 0, 0, 0, 0, 0, 0, 0, 0, 0
iou_nail_10_ave, acc_nail_10_ave, dice_nail_10_ave, iou_fg_10_ave, acc_fg_10_ave, dice_fg_10_ave, iou_bg_10_ave, acc_bg_10_ave, dice_bg_10_ave = 0, 0, 0, 0, 0, 0, 0, 0, 0

for file_idx, file in enumerate(files):
    beit10_path = os.path.join(beit10_dir, file)
    beit13_path = os.path.join(beit13_dir, file)
    swin_path = os.path.join(swin_dir, file.removesuffix('.png')+'_result.png')
    img_10 = cv2.imread(beit10_path)
    img_13 = cv2.imread(beit13_path)
    img_swin = cv2.imread(swin_path)

    img = img_10[:img_h,:img_w]
    anno = img_10[img_h:,img_w:]
    seg_13 = img_13[img_h:,:img_w]
    seg_10 = img_10[img_h:,:img_w]
    seg_swin = img_swin

    print('%d/%d:'%(file_idx, len(files)), file)
    iou_bg_13, acc_bg_13, dice_bg_13 = get_iou_acc_dice(seg_13, anno, bg_color)
    iou_fg_13, acc_fg_13, dice_fg_13 = get_iou_acc_dice(seg_13, anno, fg_color)
    iou_nail_13, acc_nail_13, dice_nail_13 = get_iou_acc_dice(seg_13, anno, nail_color)
    print('beit1.3 -- %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f'%\
          (iou_bg_13, acc_bg_13, dice_bg_13, iou_fg_13, acc_fg_13, \
           dice_fg_13, iou_nail_13, acc_nail_13, dice_nail_13))
    iou_nail_13_ave += iou_nail_13
    acc_nail_13_ave += acc_nail_13
    dice_nail_13_ave += dice_nail_13
    iou_fg_13_ave += iou_fg_13
    acc_fg_13_ave += acc_fg_13
    dice_fg_13_ave += dice_fg_13
    iou_bg_13_ave += iou_bg_13
    acc_bg_13_ave += acc_bg_13
    dice_bg_13_ave += dice_bg_13

    iou_bg_swin, acc_bg_swin, dice_bg_swin = get_iou_acc_dice(seg_swin, anno, bg_color)
    iou_fg_swin, acc_fg_swin, dice_fg_swin = get_iou_acc_dice(seg_swin, anno, fg_color)
    iou_nail_swin, acc_nail_swin, dice_nail_swin = get_iou_acc_dice(seg_swin, anno, nail_color)
    print('swin    -- %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f'%\
          (iou_bg_swin, acc_bg_swin, dice_bg_swin, iou_fg_swin, acc_fg_swin, \
           dice_fg_swin, iou_nail_swin, acc_nail_swin, dice_nail_swin))
    iou_nail_swin_ave += iou_nail_swin
    acc_nail_swin_ave += acc_nail_swin
    dice_nail_swin_ave += dice_nail_swin
    iou_fg_swin_ave += iou_fg_swin
    acc_fg_swin_ave += acc_fg_swin
    dice_fg_swin_ave += dice_fg_swin
    iou_bg_swin_ave += iou_bg_swin
    acc_bg_swin_ave += acc_bg_swin
    dice_bg_swin_ave += dice_bg_swin

    iou_bg_10, acc_bg_10, dice_bg_10 = get_iou_acc_dice(seg_10, anno, bg_color)
    iou_fg_10, acc_fg_10, dice_fg_10 = get_iou_acc_dice(seg_10, anno, fg_color)
    iou_nail_10, acc_nail_10, dice_nail_10 = get_iou_acc_dice(seg_10, anno, nail_color)
    print('beit1.0 -- %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f'%\
          (iou_bg_10, acc_bg_10, dice_bg_10, iou_fg_10, acc_fg_10, \
           dice_fg_10, iou_nail_10, acc_nail_10, dice_nail_10))
    iou_nail_10_ave += iou_nail_10
    acc_nail_10_ave += acc_nail_10
    dice_nail_10_ave += dice_nail_10
    iou_fg_10_ave += iou_fg_10
    acc_fg_10_ave += acc_fg_10
    dice_fg_10_ave += dice_fg_10
    iou_bg_10_ave += iou_bg_10
    acc_bg_10_ave += acc_bg_10
    dice_bg_10_ave += dice_bg_10



    show_img1 = np.concatenate((img, anno, anno), axis=1)
    show_img2 = np.concatenate((seg_13, seg_swin, seg_10), axis=1)
    show_img = np.concatenate((show_img1, show_img2), axis=0)
    #cv2.imwrite(os.path.join(compare_dir, file), show_img)
    #cv2.imshow('show', show_img)
    #cv2.waitKey(0)


iou_nail_13_ave, acc_nail_13_ave, dice_nail_13_ave, iou_fg_13_ave, acc_fg_13_ave, dice_fg_13_ave, iou_bg_13_ave, acc_bg_13_ave, dice_bg_13_ave =\
[xx/len(files) for xx in [iou_nail_13_ave, acc_nail_13_ave, dice_nail_13_ave, iou_fg_13_ave, acc_fg_13_ave, dice_fg_13_ave, iou_bg_13_ave, acc_bg_13_ave, dice_bg_13_ave]]
iou_nail_swin_ave, acc_nail_swin_ave, dice_nail_swin_ave, iou_fg_swin_ave, acc_fg_swin_ave, dice_fg_swin_ave, iou_bg_swin_ave, acc_bg_swin_ave, dice_bg_swin_ave =\
[xx/len(files) for xx in [iou_nail_swin_ave, acc_nail_swin_ave, dice_nail_swin_ave, iou_fg_swin_ave, acc_fg_swin_ave, dice_fg_swin_ave, iou_bg_swin_ave, acc_bg_swin_ave, dice_bg_swin_ave]]
iou_nail_10_ave, acc_nail_10_ave, dice_nail_10_ave, iou_fg_10_ave, acc_fg_10_ave, dice_fg_10_ave, iou_bg_10_ave, acc_bg_10_ave, dice_bg_10_ave =\
[xx/len(files) for xx in [iou_nail_10_ave, acc_nail_10_ave, dice_nail_10_ave, iou_fg_10_ave, acc_fg_10_ave, dice_fg_10_ave, iou_bg_10_ave, acc_bg_10_ave, dice_bg_10_ave]]

print('Average performance: ')

print('beit1.3 -- %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f' % \
      (iou_bg_13_ave, acc_bg_13_ave, dice_bg_13_ave, iou_fg_13_ave, acc_fg_13_ave, \
       dice_fg_13_ave, iou_nail_13_ave, acc_nail_13_ave, dice_nail_13_ave))

print('swin    -- %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f' % \
      (iou_bg_swin_ave, acc_bg_swin_ave, dice_bg_swin_ave, iou_fg_swin_ave, acc_fg_swin_ave, \
       dice_fg_swin_ave, iou_nail_swin_ave, acc_nail_swin_ave, dice_nail_swin_ave))

print('beit1.0 -- %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f' % \
      (iou_bg_10_ave, acc_bg_10_ave, dice_bg_10_ave, iou_fg_10_ave, acc_fg_10_ave, \
       dice_fg_10_ave, iou_nail_10_ave, acc_nail_10_ave, dice_nail_10_ave))

'''
beit1.3 -- [0.96242, 0.97981, 0.97988], [0.92548, 0.97550, 0.95874], [0.92262, 0.99416, 0.95633]
swin    -- [0.93370, 0.96230, 0.96331], [0.87672, 0.95545, 0.92754], [0.86881, 0.98989, 0.91292]
beit1.0 -- [0.90691, 0.94754, 0.94680], [0.84265, 0.94181, 0.90614], [0.85699, 0.98930, 0.90661]
'''
