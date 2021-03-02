"""
Author: xzshi20
Data: 2020.07.13
Aims: construct standard input and output for competition
"""
import os
import numpy as np
from deeplab_weights_xception_65 import Deeplabv3
from PIL import Image
from xml.dom.minidom import Document
from scipy import ndimage
from skimage import measure
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
eps = pow(2, -52)
EPS = 1e-6


def create_test_filename(img_dir):
    # img_filename = []
    names = []
    path_list = os.listdir(img_dir)
    path_list.sort()
    try:
        len(path_list) % 4 == 0
    except:
        print('Input path files is not complete. we need hh,hv,vh,vv.')
    for i in range(len(path_list) // 4):
        char_name = path_list[i * 4].split('_')
        names.append(char_name[0])
    # for filename in path_list:
    #     new_path = os.path.join(img_dir, filename)
    #     img_filename.append(new_path)

    return names


def prepro(img, rate=3., threshold=0):
    rate = rate
    black = (img <= (threshold + EPS)) * 1.
    img_sum = img * (img > (threshold + EPS))
    img_mean = img_sum.sum() / (img.shape[0] * img.shape[1] - black.sum() + EPS)
    img[img > (rate * img_mean + EPS)] = rate * img_mean
    img = (img / (np.max(img) + EPS)) * 255.0

    return img


def black_seg(img):
    label_0 = (img[:, :, 0] == 0) & (img[:, :, 1] == 0) & (img[:, :, 2] == 0)
    label_0 = label_0 * 1

    return label_0


def load_and_preprocess_image_change(hh_file, hv_file, vh_file, vv_file):
    img_hh = Image.open(hh_file)
    img_hv = Image.open(hv_file)
    img_vh = Image.open(vh_file)
    img_vv = Image.open(vv_file)

    img_hh = prepro(np.asarray(img_hh, dtype=float))
    img_m = (np.asarray(img_hv, dtype=float) + np.asarray(img_vh, dtype=float)) / 2
    img_m = prepro(img_m)
    img_vv = prepro(np.asarray(img_vv, dtype=float))

    img_shape = img_hh.shape
    img_all = np.zeros((img_shape[0], img_shape[1], 3))

    img_all[:, :, 0] = img_hh
    img_all[:, :, 1] = img_m
    img_all[:, :, 2] = img_vv

    img_all = np.around(img_all)
    img_all = img_all.astype('uint8')

    image_256 = []
    img_black = []
    if (img_all.shape[0] // 256 == 0) and (img_all.shape[1] // 256 == 0):
        pass
        # row_pad = 0
        # col_pad = 0
    else:
        if img_all.shape[0] >= 256:
            pad_h = img_all.shape[0] - (img_all.shape[0] // 256) * 256
        else:
            pad_h = 256 - img_all.shape[0]
        if img_all.shape[1] >= 256:
            pad_w = img_all.shape[1] - (img_all.shape[1] // 256) * 256
        else:
            pad_w = 256 - img_all.shape[1]
        # row_pad = pad_h // 2
        # col_pad = pad_w // 2
        # img_all = np.pad(img_all, ((row_pad, pad_h - row_pad), (col_pad, pad_w - col_pad), (0, 0)), 'constant')
        img_all = np.pad(img_all, ((0, pad_h), (0, pad_w), (0, 0)), 'constant')
    for i in range(img_all.shape[0] // 256):
        row_start = i * 256
        row_end = row_start + 256
        for j in range(img_all.shape[1] // 256):
            col_start = j * 256
            col_end = col_start + 256
            img_256 = img_all[row_start:row_end, col_start:col_end, :]
            black_256 = black_seg(img_256)
            image_256.append(img_256)
            img_black.append(black_256)

    return image_256, img_black, img_shape, img_all.shape


def load_and_preprocess_image_change_256(hh_file, hv_file, vh_file, vv_file):
    img_hh = Image.open(hh_file)
    img_hv = Image.open(hv_file)
    img_vh = Image.open(vh_file)
    img_vv = Image.open(vv_file)

    img_hh = np.asarray(img_hh, dtype=float)
    img_m = (np.asarray(img_hv, dtype=float) + np.asarray(img_vh, dtype=float)) / 2
    img_vv = np.asarray(img_vv, dtype=float)

    img_shape = img_hh.shape
    img_all = np.zeros((img_shape[0], img_shape[1], 3))

    img_all[:, :, 0] = img_hh
    img_all[:, :, 1] = img_m
    img_all[:, :, 2] = img_vv

    image_256 = []
    img_black = []
    if (img_all.shape[0] // 256 == 0) and (img_all.shape[1] // 256 == 0):
        pass
        # row_pad = 0
        # col_pad = 0
    else:
        if img_all.shape[0] >= 256:
            pad_h = img_all.shape[0] - (img_all.shape[0] // 256) * 256
        else:
            pad_h = 256 - img_all.shape[0]
        if img_all.shape[1] >= 256:
            pad_w = img_all.shape[1] - (img_all.shape[1] // 256) * 256
        else:
            pad_w = 256 - img_all.shape[1]
        # row_pad = pad_h // 2
        # col_pad = pad_w // 2
        # img_all = np.pad(img_all, ((row_pad, pad_h - row_pad), (col_pad, pad_w - col_pad), (0, 0)), 'constant')
        img_all = np.pad(img_all, ((0, pad_h), (0, pad_w), (0, 0)), 'constant')
    for i in range(img_all.shape[0] // 256):
        row_start = i * 256
        row_end = row_start + 256
        for j in range(img_all.shape[1] // 256):
            col_start = j * 256
            col_end = col_start + 256
            img_256 = img_all[row_start:row_end, col_start:col_end, :]
            for n in range(3):
                img_256[..., n] = prepro(img_256[..., n])
            black_256 = black_seg(img_256)
            image_256.append(img_256)
            img_black.append(black_256)

    return image_256, img_black, img_shape, img_all.shape


def load_and_preprocess_image_change_256_mean(hh_file, hv_file, vh_file, vv_file):
    img_hh = Image.open(hh_file)
    img_hv = Image.open(hv_file)
    img_vh = Image.open(vh_file)
    img_vv = Image.open(vv_file)

    img_hh = np.asarray(img_hh, dtype=float)
    img_hv = np.asarray(img_hv, dtype=float)
    img_vh = np.asarray(img_vh, dtype=float)
    img_vv = np.asarray(img_vv, dtype=float)
    img_shape = img_hh.shape

    image_256 = []
    img_black = []
    if (img_hh.shape[0] // 256 == 0) and (img_hh.shape[1] // 256 == 0):
        pass
        # row_pad = 0
        # col_pad = 0
    else:
        if img_hh.shape[0] >= 256:
            pad_h = img_hh.shape[0] - (img_hh.shape[0] // 256) * 256
        else:
            pad_h = 256 - img_hh.shape[0]
        if img_hh.shape[1] >= 256:
            pad_w = img_hh.shape[1] - (img_hh.shape[1] // 256) * 256
        else:
            pad_w = 256 - img_hh.shape[1]
        # row_pad = pad_h // 2
        # col_pad = pad_w // 2
        # img_all = np.pad(img_all, ((row_pad, pad_h - row_pad), (col_pad, pad_w - col_pad), (0, 0)), 'constant')
        img_hh = np.pad(img_hh, ((0, pad_h), (0, pad_w)), 'constant')
        img_hv = np.pad(img_hv, ((0, pad_h), (0, pad_w)), 'constant')
        img_vh = np.pad(img_vh, ((0, pad_h), (0, pad_w)), 'constant')
        img_vv = np.pad(img_vv, ((0, pad_h), (0, pad_w)), 'constant')

    for i in range(img_hh.shape[0] // 256):
        row_start = i * 256
        row_end = row_start + 256
        for j in range(img_hh.shape[1] // 256):
            col_start = j * 256
            col_end = col_start + 256
            img_256_hh = img_hh[row_start:row_end, col_start:col_end]
            img_256_hv = img_hv[row_start:row_end, col_start:col_end]
            img_256_vh = img_vh[row_start:row_end, col_start:col_end]
            img_256_vv = img_vv[row_start:row_end, col_start:col_end]
            img_256_hh = prepro(img_256_hh)
            img_256_hv = prepro(img_256_hv)
            img_256_vh = prepro(img_256_vh)
            img_256_vv = prepro(img_256_vv)
            img_all = np.zeros((256, 256, 3))
            img_all[:, :, 0] = img_256_hh
            img_all[:, :, 1] = (img_256_hv + img_256_vh) / 2
            img_all[:, :, 2] = img_256_vv
            black_256 = black_seg(img_all)
            image_256.append(img_all)
            img_black.append(black_256)

    return image_256, img_black, img_shape, img_hh.shape


def prepro_m(img, m, rate=3):
    rate = rate
    img[img > (rate * m + EPS)] = rate * m
    img = (img / (np.max(img) + EPS)) * 255.0

    return img


def load_and_preprocess_image_change_m(hh_file, hv_file, vh_file, vv_file, m_hh, m_hv, m_vh, m_vv):
    img_hh = Image.open(hh_file)
    img_hv = Image.open(hv_file)
    img_vh = Image.open(vh_file)
    img_vv = Image.open(vv_file)

    img_hh = prepro_m(np.array(img_hh), m_hh)
    img_hv = prepro_m(np.array(img_hv), m_hv)
    img_vh = prepro_m(np.array(img_vh), m_vh)
    img_vv = prepro_m(np.array(img_vv), m_vv)

    img_shape = img_hh.shape
    img_all = np.zeros((img_shape[0], img_shape[1], 3))

    img_all[:, :, 0] = img_hh
    img_all[:, :, 1] = (img_hv + img_vh) / 2
    img_all[:, :, 2] = img_vv

    img_all = np.around(img_all)
    img_all = img_all.astype('uint8')

    image_256 = []
    img_black = []
    if (img_all.shape[0] // 256 == 0) and (img_all.shape[1] // 256 == 0):
        pass
        # row_pad = 0
        # col_pad = 0
    else:
        if img_all.shape[0] >= 256:
            pad_h = img_all.shape[0] - (img_all.shape[0] // 256) * 256
        else:
            pad_h = 256 - img_all.shape[0]
        if img_all.shape[1] >= 256:
            pad_w = img_all.shape[1] - (img_all.shape[1] // 256) * 256
        else:
            pad_w = 256 - img_all.shape[1]
        # row_pad = pad_h // 2
        # col_pad = pad_w // 2
        # img_all = np.pad(img_all, ((row_pad, pad_h - row_pad), (col_pad, pad_w - col_pad), (0, 0)), 'constant')
        img_all = np.pad(img_all, ((0, pad_h), (0, pad_w), (0, 0)), 'constant')
    for i in range(img_all.shape[0] // 256):
        row_start = i * 256
        row_end = row_start + 256
        for j in range(img_all.shape[1] // 256):
            col_start = j * 256
            col_end = col_start + 256
            img_256 = img_all[row_start:row_end, col_start:col_end, :]
            black_256 = black_seg(img_256)
            image_256.append(img_256)
            img_black.append(black_256)

    return image_256, img_black, img_shape, img_all.shape


def union_256_to_origin(pred_256, img_black, pad_shape):
    shape_pred = pred_256.shape
    pred_np = np.zeros((shape_pred[0], shape_pred[1], shape_pred[2]))
    for n in range(shape_pred[0]):
        pred_1 = pred_256[n, ...]
        lab = np.argmax(pred_1, axis=2)

        black = img_black[n]
        black_pred = lab == 0
        other = lab == 1
        vegetation = lab == 2
        building = lab == 3
        industry = lab == 4
        water = lab == 5
        land = lab == 6

        black_li = black_pred * 1 - black * 1
        if np.sum(black_li) <= 0:
            other = other * 1 - (other & black) * 1
            vegetation = vegetation * 1 - (vegetation & black) * 1
            building = building * 1 - (building & black) * 1
            industry = industry * 1 - (industry & black) * 1
            water = water * 1 - (water & black) * 1
            land = land * 1 - (land & black) * 1
        else:
            sum_other = np.sum(other * 1)
            sum_vegetation = np.sum(vegetation * 1)
            sum_building = np.sum(building * 1)
            sum_industry = np.sum(industry * 1)
            sum_water = np.sum(water * 1)
            sum_land = np.sum(land * 1)
            index = np.argmax([sum_other, sum_vegetation, sum_building, sum_industry, sum_water, sum_land])
            if index == 0:
                other = other * 1 + black_li - (other & black) * 1
                vegetation = vegetation * 1 - (vegetation & black) * 1
                building = building * 1 - (building & black) * 1
                industry = industry * 1 - (industry & black) * 1
                water = water * 1 - (water & black) * 1
                land = land * 1 - (land & black) * 1
            elif index == 1:
                other = other * 1 - (other & black) * 1
                vegetation = vegetation * 1 + black_li - (vegetation & black) * 1
                building = building * 1 - (building & black) * 1
                industry = industry * 1 - (industry & black) * 1
                water = water * 1 - (water & black) * 1
                land = land * 1 - (land & black) * 1
            elif index == 2:
                other = other * 1 - (other & black) * 1
                vegetation = vegetation * 1 - (vegetation & black) * 1
                building = building * 1 + black_li - (building & black) * 1
                industry = industry * 1 - (industry & black) * 1
                water = water * 1 - (water & black) * 1
                land = land * 1 - (land & black) * 1
            elif index == 3:
                other = other * 1 - (other & black) * 1
                vegetation = vegetation * 1 - (vegetation & black) * 1
                building = building * 1 - (building & black) * 1
                industry = industry * 1 + black_li - (industry & black) * 1
                water = water * 1 - (water & black) * 1
                land = land * 1 - (land & black) * 1
            elif index == 4:
                other = other * 1 - (other & black) * 1
                vegetation = vegetation * 1 - (vegetation & black) * 1
                building = building * 1 - (building & black) * 1
                industry = industry * 1 - (industry & black) * 1
                water = water * 1 + black_li - (water & black) * 1
                land = land * 1 - (land & black) * 1
            elif index == 6:
                other = other * 1 - (other & black) * 1
                vegetation = vegetation * 1 - (vegetation & black) * 1
                building = building * 1 - (building & black) * 1
                industry = industry * 1 - (industry & black) * 1
                water = water * 1 - (water & black) * 1
                land = land * 1 + black_li - (land & black) * 1

        lab_num = other * 1 + vegetation * 2 + building * 3 + industry * 4 + water * 5 + land * 6
        pred_np[n, ...] = lab_num

    img_all = np.zeros((pad_shape[0], pad_shape[1]))
    for k in range(pad_shape[0] // 256):
        row_start = k * 256
        row_end = row_start + 256
        for nl in range(pad_shape[1] // 256):
            col_start = nl * 256
            col_end = col_start + 256
            img_all[row_start:row_end, col_start:col_end] = pred_np[k * (pad_shape[1] // 256) + nl]

    return img_all


"""
% black -- (0, 0, 0) -- 1
% other -- (255, 255, 255) -- 2
% vegetation -- (0, 255, 0) -- 3
% building -- (255, 255, 0) -- 4
% industry -- (0, 0, 255) -- 5
% water -- (0, 255, 255) -- 6
% land -- (255, 0, 0) -- 7
"""


def union_holes_land(land, building, industry, vegetation, land_filled, building_filled, industry_filled,
                     vegetation_filled, threshold=800):
    holes_delete_all1 = np.zeros(land.shape)
    holes_delete_all2 = np.zeros(land.shape)
    holes_delete_all3 = np.zeros(land.shape)

    holes_bl = (land_filled == 1) & (building_filled == 1)
    holes_il = (land_filled == 1) & (industry_filled == 1)
    holes_vl = (land_filled == 1) & (vegetation_filled == 1)

    holes_land1 = (land == 1) & holes_bl
    holes_land2 = (land == 1) & holes_il
    holes_land3 = (land == 1) & holes_vl

    label1, num1 = measure.label(holes_land1, connectivity=2, background=0, return_num=True)
    label2, num2 = measure.label(holes_land2, connectivity=2, background=0, return_num=True)
    label3, num3 = measure.label(holes_land3, connectivity=2, background=0, return_num=True)

    for i in range(num1):
        mask_num = label1 == (i + 1)
        if np.sum(mask_num * 1) <= threshold:
            holes_delete_all1 = holes_delete_all1 + mask_num * 1
    building_bl = building_filled - holes_bl * 1 + holes_delete_all1
    building = building + holes_delete_all1

    for i in range(num2):
        mask_num = label2 == (i + 1)
        if np.sum(mask_num * 1) <= threshold:
            holes_delete_all2 = holes_delete_all2 + mask_num * 1
    industry_il = industry_filled - holes_il * 1 + holes_delete_all2
    industry = industry + holes_delete_all2

    for i in range(num3):
        mask_num = label3 == (i + 1)
        if np.sum(mask_num * 1) <= threshold:
            holes_delete_all3 = holes_delete_all3 + mask_num * 1
    vegetation_vl = vegetation_filled - holes_vl * 1 + holes_delete_all3
    vegetation = vegetation + holes_delete_all3

    holes_delete_all = holes_delete_all1 + holes_delete_all2 + holes_delete_all3
    land_all = land_filled - holes_delete_all

    return land_all, building_bl, industry_il, vegetation_vl, building, industry, vegetation


def union_building(building, industry, vegetation, building_filled, industry_filled, vegetation_filled, threshold1=2400,
                   threshold2=2800, threshold3=1400):
    holes_delete_all_b1 = np.zeros(building.shape)
    holes_delete_all_b2 = np.zeros(building.shape)
    holes_delete_all_i1 = np.zeros(building.shape)
    holes_delete_all_i2 = np.zeros(building.shape)
    holes_delete_all_v1 = np.zeros(building.shape)
    holes_delete_all_v2 = np.zeros(building.shape)
    holes_bi = (building_filled == 1) & (industry_filled == 1)
    holes_bv = (building_filled == 1) & (vegetation_filled == 1)
    holes_iv = (industry_filled == 1) & (vegetation_filled == 1)

    holes_b1 = (building == 1) & holes_bi
    label_b1, num_b1 = measure.label(holes_b1, connectivity=2, background=0, return_num=True)
    for i in range(num_b1):
        mask_num = label_b1 == (i + 1)
        if np.sum(mask_num * 1) <= threshold1:
            holes_delete_all_b1 = holes_delete_all_b1 + mask_num * 1
    industry_all = industry + holes_delete_all_b1

    holes_b2 = (building == 1) & holes_bv
    label_b2, num_b2 = measure.label(holes_b2, connectivity=2, background=0, return_num=True)
    for i in range(num_b2):
        mask_num = label_b2 == (i + 1)
        if np.sum(mask_num * 1) <= threshold1:
            holes_delete_all_b2 = holes_delete_all_b2 + mask_num * 1
    vegetation_all = vegetation + holes_delete_all_b2

    holes_delete_all_b = holes_delete_all_b1 + holes_delete_all_b2
    building_all = building - holes_delete_all_b  # 删除连通域后，需赋值

    ################
    holes_i1 = (industry == 1) & holes_bi
    label_i1, num_i1 = measure.label(holes_i1, connectivity=2, background=0, return_num=True)
    for i in range(num_i1):
        mask_num = label_i1 == (i + 1)
        if np.sum(mask_num * 1) <= threshold2:
            holes_delete_all_i1 = holes_delete_all_i1 + mask_num * 1
    building_all = building_all + holes_delete_all_i1

    holes_i2 = (industry == 1) & holes_iv
    label_i2, num_i2 = measure.label(holes_i2, connectivity=2, background=0, return_num=True)
    for i in range(num_i2):
        mask_num = label_i2 == (i + 1)
        if np.sum(mask_num * 1) <= threshold2:
            holes_delete_all_i2 = holes_delete_all_i2 + mask_num * 1
    vegetation_all = vegetation_all + holes_delete_all_i2

    holes_delete_all_i = holes_delete_all_i1 + holes_delete_all_i2
    industry_all = industry_all - holes_delete_all_i

    ###################
    holes_v1 = (vegetation == 1) & holes_bv
    label_v1, num_v1 = measure.label(holes_v1, connectivity=2, background=0, return_num=True)
    for i in range(num_v1):
        mask_num = label_v1 == (i + 1)
        if np.sum(mask_num * 1) <= threshold3:
            holes_delete_all_v1 = holes_delete_all_v1 + mask_num * 1
    building_all = building_all + holes_delete_all_v1

    holes_v2 = (vegetation == 1) & holes_iv
    label_v2, num_v2 = measure.label(holes_v2, connectivity=2, background=0, return_num=True)
    for i in range(num_v2):
        mask_num = label_v2 == (i + 1)
        if np.sum(mask_num * 1) <= threshold3:
            holes_delete_all_v2 = holes_delete_all_v2 + mask_num * 1
    industry_all = industry_all + holes_delete_all_v2

    holes_delete_all_v = holes_delete_all_v1 + holes_delete_all_v2
    vegetation_all = vegetation_all - holes_delete_all_v

    return building_all, industry_all, vegetation_all


def mask_to_rgb_black(lab, img_shape):
    [height, width] = lab.shape
    rgb = np.zeros((height, width, 3))

    other = lab == 1
    vegetation = lab == 2
    building = lab == 3
    industry = lab == 4
    water = lab == 5
    land = lab == 6
    rgb[..., 0] = other * 255 + building * 255 + land * 255
    rgb[..., 1] = other * 255 + vegetation * 255 + building * 255 + water * 255
    rgb[..., 2] = other * 255 + industry * 255 + water * 255
    rgb = rgb.astype('uint8')

    rgb_origin = rgb[0:img_shape[0], 0:img_shape[1]]

    return rgb_origin


def mask_to_rgb_black_imfill_whole(lab, img_shape):
    [height, width] = lab.shape
    rgb = np.zeros((height, width, 3))

    other = lab == 1
    vegetation = lab == 2
    building = lab == 3
    industry = lab == 4
    water = lab == 5
    land = lab == 6

    water_filled = ndimage.binary_fill_holes(water).astype(int)
    land_filled = ndimage.binary_fill_holes(land).astype(int)
    other_filled = ndimage.binary_fill_holes(other).astype(int)
    industry_filled = ndimage.binary_fill_holes(industry).astype(int)
    building_filled = ndimage.binary_fill_holes(building).astype(int)
    vegetation_filled = ndimage.binary_fill_holes(vegetation).astype(int)

    water_all = water_filled
    land_all, building_bl, industry_il, vegetation_vl, building, industry, vegetation = union_holes_land(land, building,
                                                                                                         industry,
                                                                                                         vegetation,
                                                                                                         land_filled,
                                                                                                         building_filled,
                                                                                                         industry_filled,
                                                                                                         vegetation_filled,
                                                                                                         threshold=1800)
    other_all, building_bo, industry_io, vegetation_vo, building, industry, vegetation = union_holes_land(other,
                                                                                                          building,
                                                                                                          industry,
                                                                                                          vegetation,
                                                                                                          other_filled,
                                                                                                          building_bl,
                                                                                                          industry_il,
                                                                                                          vegetation_vl,
                                                                                                          threshold=1600)

    building_all, industry_all, vegetation_all = union_building(building, industry, vegetation, building_bo,
                                                                industry_io, vegetation_vo,
                                                                threshold1=3400, threshold2=4800, threshold3=2400)

    land_all = land_all - ((land_all == 1) & (water_all == 1)) * 1
    other_all = other_all - ((other_all == 1) & (water_all == 1)) * 1 - ((other_all == 1) & (land_all == 1)) * 1

    building_all = building_all - ((building_all == 1) & (water_all == 1)) * 1 - (
            (building_all == 1) & (land_all == 1)) * 1 - ((building_all == 1) & (other_all == 1)) * 1
    industry_all = industry_all - ((industry_all == 1) & (water_all == 1)) * 1 - (
            (industry_all == 1) & (land_all == 1)) * 1 - ((industry_all == 1) & (other_all == 1)) * 1
    vegetation_all = vegetation_all - ((vegetation_all == 1) & (water_all == 1)) * 1 - (
            (vegetation_all == 1) & (land_all == 1)) * 1 - ((vegetation_all == 1) & (other_all == 1)) * 1

    rgb[..., 0] = other_all * 255 + building_all * 255 + land_all * 255
    rgb[..., 1] = other_all * 255 + vegetation_all * 255 + building_all * 255 + water_all * 255
    rgb[..., 2] = other_all * 255 + industry_all * 255 + water_all * 255
    rgb = rgb.astype('uint8')

    rgb_origin = rgb[0:img_shape[0], 0:img_shape[1]]

    return rgb_origin


def writeInfoToXml(tif_name, png_name):
    # 创建dom文档
    doc = Document()
    # 创建根节点
    annotation = doc.createElement('annotation')
    # 根节点插入dom树
    doc.appendChild(annotation)
    # 依次将orderDict中的每一组元素提取出来，创建对应节点并插入dom树
    # source
    source = doc.createElement('source')
    annotation.appendChild(source)

    filename_list = [tif_name + '_HH.tif', tif_name + '_HV.tif', tif_name + '_VH.tif', tif_name + '_VV.tif']
    for name in filename_list:
        filename = doc.createElement('filename')
        customer_text = doc.createTextNode(name)
        filename.appendChild(customer_text)
        source.appendChild(filename)

    origin = doc.createElement('origin')
    phone_text = doc.createTextNode('GF2/GF3')
    origin.appendChild(phone_text)
    source.appendChild(origin)

    # research
    research = doc.createElement('research')
    annotation.appendChild(research)

    list_char = ['version', 'provider', 'author', 'pluginname', 'pluginclass', 'time']
    name_char = ['4.0', 'Fudan university', 'Feng Wang et al.', 'segmentation', 'segmentation', '2020-07-2020-11']
    for i, name in enumerate(list_char):
        filename = doc.createElement(name)
        customer_text = doc.createTextNode(name_char[i])
        filename.appendChild(customer_text)
        research.appendChild(filename)

    # segmentation
    seg = doc.createElement('segmentation')
    annotation.appendChild(seg)

    filename = doc.createElement('resultfile')
    customer_text = doc.createTextNode(png_name)
    filename.appendChild(customer_text)
    seg.appendChild(filename)

    return doc


def main(input_dir_A, input_dir_B, output_dir_A, output_dir_B):
    classes = 7

    if not os.path.exists(output_dir_A):
        os.makedirs(output_dir_A)
    if not os.path.exists(output_dir_B):
        os.makedirs(output_dir_B)

    model = Deeplabv3(input_shape=(None, None, 3), classes=classes, OS=16, middle_repeat=8)
    weights_file = '/workspace/code/deeplabv3plus_15-0.97966--0.74749_t0.hdf5'
    model.load_weights(weights_file)

    input_dir_list = [input_dir_A, input_dir_B]
    output_dir_list = [output_dir_A, output_dir_B]
    for n, input_dir in enumerate(input_dir_list):
        names = create_test_filename(input_dir)  # list
        try:
            len(names) > 0
        except:
            print('No file in {}'.format(input_dir))
        for seq in names:
            hh_file = os.path.join(input_dir, seq + '_HH.tiff')
            hv_file = os.path.join(input_dir, seq + '_HV.tiff')
            vh_file = os.path.join(input_dir, seq + '_VH.tiff')
            vv_file = os.path.join(input_dir, seq + '_VV.tiff')
            image_256, img_black, img_shape, pad_shape = load_and_preprocess_image_change_256_mean(hh_file, hv_file, vh_file, vv_file)
            # img_mean_hh = 146.6043408597247
            # img_mean_hv = 199.8496095425074
            # img_mean_vh = 197.1202233864348
            # img_mean_vv = 153.82159331536104
            # image_256, img_black, img_shape, pad_shape = load_and_preprocess_image_change_m(hh_file, hv_file, vh_file, vv_file, img_mean_hh, img_mean_hv, img_mean_vh, img_mean_vv)

            image_256_all = np.array(image_256)    # [N, 256, 256, 3]
            image_256_all = image_256_all / 127.5 - 1.
            sha = image_256_all.shape
            image_256_all = np.reshape(image_256_all, (sha[0], sha[1], sha[2], sha[3]))
            image_256_all_lr = np.flip(image_256_all, 1)
            image_256_all_ud = np.flip(image_256_all, 0)

            pred_256_1 = model.predict(image_256_all, batch_size=1, verbose=1)
            pred_256_lr = model.predict(image_256_all_lr, batch_size=1, verbose=1)
            pred_256_ud = model.predict(image_256_all_ud, batch_size=1, verbose=1)
            pred_256 = pred_256_1 + np.flip(pred_256_lr, 1) + np.flip(pred_256_ud, 0)

            image_all = union_256_to_origin(pred_256, img_black, pad_shape)

            # img_origin = image_all
            rgb_origin = mask_to_rgb_black(image_all, img_shape)
            # rgb_origin = mask_to_rgb_black_imfill_whole(img_all, img_shape)
            rgb_origin = Image.fromarray(rgb_origin)
            new_name = output_dir_list[n] + '/' + seq + '_gt.png'
            doc = writeInfoToXml(seq + '_HH.tiff', seq + '_gt.png')
            new_xml = output_dir_list[n] + '/' + seq + '.xml'
            with open(new_xml, 'w', encoding='utf-8') as f:
                doc.writexml(f, addindent='\t', newl='\n', encoding='utf-8')
            rgb_origin.save(new_name)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='xingtubei')
    args.add_argument("input_dir_A", type=str, default='/input_path/test_A')
    args.add_argument("input_dir_B", type=str, default='/input_path/test_B')
    args.add_argument("output_dir_A", type=str, default='/output_path/test_A')
    args.add_argument("output_dir_B", type=str, default='/output_path/test_B')
    args = args.parse_args()
    main(args.input_dir_A, args.input_dir_B, args.output_dir_A, args.output_dir_B)
