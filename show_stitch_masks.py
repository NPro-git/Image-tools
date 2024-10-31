import os
import cv2
import numpy as np
from mmseg.apis import init_model, inference_model

dir = '/home/ubuntu/dataset/train_set/hands_EPxx_20240920'
save_dir = '/home/ubuntu/dataset/tmp/train_set_stitch_masks'
rsz_h, rsz_w = 512, 512

model = dict()

model['top_autoroi'] = init_model(config='models/swin/swin_roi_top_v1.1/20241020_143355/vis_data/config.py',
                                  checkpoint='models/swin/swin_roi_top_v1.1/iter_40000.pth')
model['lr_autoroi'] = init_model(config='models/swin/swin_roi_lr_v1.1/20241020_100819/vis_data/config.py',
                                 checkpoint='models/swin/swin_roi_lr_v1.1/iter_34000.pth')
model['bot_autoroi'] = init_model(config='models/swin/swin_roi_bot_v1.1/20241020_035619/vis_data/config.py',
                                  checkpoint='models/swin/swin_roi_bot_v1.1/iter_58000.pth')
model['fullseg'] = init_model(config='models/swin/swin_fullseg_addEP1/20241019_182702/vis_data/config.py',
                              checkpoint='models/swin/swin_fullseg_addEP1/iter_100000.pth')


def get_patch_around_point(point, image, crp_h, crp_w):
    cx, cy = point
    img_h, img_w = image.shape[:2]
    patch_t, patch_l = cy - crp_h // 2, cx - crp_w // 2
    patch_b, patch_r = patch_t + crp_h - 1, patch_l + crp_w - 1
    img_eff_t, img_eff_b = max(0, patch_t), min(patch_b, img_h - 1)
    img_eff_l, img_eff_r = max(0, patch_l), min(patch_r, img_w - 1)
    crp_eff_t, crp_eff_b = crp_h // 2 - (cy - img_eff_t), crp_h // 2 + (img_eff_b - cy)
    crp_eff_l, crp_eff_r = crp_w // 2 - (cx - img_eff_l), crp_w // 2 + (img_eff_r - cx)
    crp_img = np.zeros((crp_h, crp_w, 3), np.uint8)
    crp_img[crp_eff_t:crp_eff_b + 1, crp_eff_l:crp_eff_r + 1] = image[img_eff_t:img_eff_b + 1, img_eff_l:img_eff_r + 1]
    return crp_img, (img_eff_t, img_eff_b, img_eff_l, img_eff_r), (crp_eff_t, crp_eff_b, crp_eff_l, crp_eff_r)


img_files = [file for file in os.listdir(dir) if '.jpg' in file]
for img_idx, img_file in enumerate(img_files):
    view, chi = img_file.split('_')[-3:-1]
    img_path = os.path.join(dir, img_file)
    img = cv2.imread(img_path)
    img_h, img_w = img.shape[:2]
    uni_img = img.copy()
    if view == 'topleft':
        None
    elif view == 'left' or view == 'bottomleft':
        uni_img = np.transpose(uni_img, (1, 0, 2))[:, ::-1]
    else:
        uni_img = np.transpose(uni_img, (1, 0, 2))[::-1]
    if 'handR' == chi:
        uni_img = uni_img[:, ::-1]
    rsz_img = cv2.resize(uni_img, (rsz_w, rsz_h))
    roi_model = model['top_autoroi'] if view == 'topleft' \
        else model['lr_autoroi'] if view == 'left' or view == 'right' \
        else model['bot_autoroi']
    autoroi_pred = inference_model(roi_model, rsz_img)
    autoroi_pred = autoroi_pred.pred_sem_seg.data.detach().cpu().numpy()[0]
    autoroi_mask = (autoroi_pred * 127).astype(np.uint8)
    if 'handR' == chi:
        autoroi_mask = autoroi_mask[:, ::-1]
    if view == 'topleft':
        None
    elif view == 'left' or view == 'bottomleft':
        autoroi_mask = np.transpose(autoroi_mask[:, ::-1], (1, 0))
    else:
        autoroi_mask = np.transpose(autoroi_mask[::-1], (1, 0))
    autoroi_mask = cv2.resize(autoroi_mask, (img_w, img_h))

    nail_mask = np.zeros(autoroi_mask.shape, dtype=np.uint8)
    nail_mask[np.where(autoroi_mask > 200)] = 255
    contours, _ = cv2.findContours(nail_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    result_mask = np.zeros(autoroi_mask.shape, dtype=np.uint8)
    result_mask[np.where(autoroi_mask > 64)] = 128
    size = 512
    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < 2000:
            continue

        M = cv2.moments(cnt)

        x, y, w, h = cv2.boundingRect(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        # print(cx, cy)

        roi_img, (img_eff_t, img_eff_b, img_eff_l, img_eff_r), \
            (crp_eff_t, crp_eff_b, crp_eff_l, crp_eff_r) = \
            get_patch_around_point((cx, cy), img.copy(), size, size)

        if 'topleft' in img_file:
            None
        elif 'left' in img_file:
            roi_img = np.transpose(roi_img, (1, 0, 2))[:, ::-1]
        else:
            roi_img = np.transpose(roi_img, (1, 0, 2))[::-1]

        roi_pred = inference_model(model['fullseg'], roi_img)
        roi_pred = roi_pred.pred_sem_seg.data.detach().cpu().numpy()[0]

        if 'topleft' in img_file:
            None
        elif 'left' in img_file:
            roi_pred = np.transpose(roi_pred[:, ::-1], (1, 0))
        else:
            roi_pred = np.transpose(roi_pred[::-1], (1, 0))

        # x1, x2, y1, y2 = roi_coords

        # lx, rx = int(cx - (patch_size / 2)), int(cx + (patch_size / 2))
        # ty, by = int(cy - (patch_size / 2)), int(cy + (patch_size / 2))
        # roi_h, roi_w = roi_pred.shape[:2]
        # roi_t, roi_b = y1-int(cy-(roi_h/2)), y2-int(cy-(roi_h/2))
        # roi_l, roi_r = x1-int(cx-(roi_w/2)), x2-int(cx-(roi_w/2))
        # result_mask[y1:y2, x1:x2] = roi_pred[roi_t:roi_b,roi_l:roi_r]
        roi_pred = ((roi_pred > 1) * 255).astype(np.uint8)
        contours, _ = cv2.findContours(roi_pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [contour.reshape((-1, 2)) for contour in contours]
        min_dist = 512 * 512
        nail_t, nail_b, nail_l, nail_r = 0, size - 1, 0, size - 1
        for contour in contours:
            obj_l, obj_t = np.min(contour, axis=0)
            obj_r, obj_b = np.max(contour, axis=0)
            cx, cy = (obj_l + obj_r) / 2, (obj_t + obj_b) / 2
            dist = (cx - rsz_w // 2) ** 2 + (cy - rsz_h // 2) ** 2
            if min_dist > dist:
                min_dist = dist
                nail_t, nail_b, nail_l, nail_r = obj_t, obj_b, obj_l, obj_r
        nail_bbx = np.zeros(roi_pred.shape, dtype=np.uint8)
        nail_bbx[nail_t:nail_b + 1, nail_l:nail_r + 1] = 255
        roi_pred[np.where(nail_bbx == 0)] = 0
        result_mask[img_eff_t:img_eff_b + 1, img_eff_l:img_eff_r + 1] \
            [np.where(roi_pred[crp_eff_t:crp_eff_b + 1, crp_eff_l:crp_eff_r + 1] > 128)] = 255

    save_path = os.path.join(save_dir, img_file.replace('.jpg', '.png'))
    cv2.imwrite(save_path, result_mask.astype(np.uint8))
    print('%d/%d: ' % (img_idx + 1, len(img_files)), img_path)
