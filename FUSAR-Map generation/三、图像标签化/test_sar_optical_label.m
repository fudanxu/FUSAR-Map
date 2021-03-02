clear;
clc;

img_str = 'GF3_MDJ_UFS_002760_E120.0_N31.5_20170217_L1A_DH_L10002191778';
num_folder = '04';
% img_path = strcat('H:\cities\', num_folder, '\', img_str, '\', img_str, '_光学\', img_str, '_卫图.tif');
num_str = '26';
local_str = '_礼嘉镇';
% local_str2 = '_大港区';

% opt_box = strcat('F:\T_shixianzheng\datasets\', num_str, '--', img_str, '\', 'box_optical_image', local_str, '.png');
% img_box_path = strcat('F:\T_shixianzheng\datasets\', num_str, '--', img_str, '\', num_str, '边界.shp');
% sar_path = strcat('H:\cities\', num_folder, '\', img_str, '\', img_str, '_光学\', img_str, '_registered.tif');
water_name = strcat('F:\T_shixianzheng\datasets\', num_str, '--', img_str, '\elec_waterlabel_', num_str, '_logical_registered.png');
road_name = strcat('F:\T_shixianzheng\datasets\', num_str, '--', img_str, '\elec_roadlabel_', num_str, '_logical_registered.png');
water = imread(water_name);
road = imread(road_name);

build_name1 = strcat('F:\T_shixianzheng\datasets\', num_str, '--', img_str, '\', 'polygon_registered_', local_str, '.mat');
build1 = load(build_name1);
build1 = build1.polygon_registered;
build = build1 == 1;
% figure, imshow(build1*255)

% build_name2 = strcat('F:\T_shixianzheng\datasets\', num_str, '--', img_str, '\', 'polygon_registered_', local_str2, '.mat');
% build2 = load(build_name2);
% build2 = build2.polygon_registered;
% build2 = build2 == 1;

% build = build1 | build2;
% figure, imshow(build*255)

build = build - (build & water);
% figure, imshow(build * 255)

road = road - (build & road);

% lab = img(:, :, 1) <= 15 & img(:, :, 2) >= 10 & img(:, :, 2) <= 70 & img(:, :, 3) <= 60;

lab(:, :, 1) = build;
lab(:, :, 2) = road;
lab(:, :, 3) = water;
clear build1
% clear build2
clear water
clear build
clear road
lab = uint8(lab);
img_path = strcat('F:\T_shixianzheng\datasets\', num_str, '--', img_str, '\', img_str, '_label.mat');

save(img_path, 'lab');

% figure, imshow(lab*255)
% lab = load(img_path);
% lab = lab.lab;
% imwrite(lab, img_path)


% [row_min, row_max, col_min, col_max] = box_get(img_path, img_box_path, shp_box_path);
% sar = imread(sar_path);
% sar_box = sar(row_min:row_max, col_min:col_max);

% opt = imread(opt_box);
% opt_r = opt(:, :, 1);
% opt_g = opt(:, :, 2);
% opt_b = opt(:, :, 3);
% opt_lab = opt_r <= 20 & opt_g <= 50 & opt_g >= 30 & opt_b >= 45 & opt_b <= 65;
% figure, imshow(opt)
% figure, imshow(opt_r)
% figure, imhist(opt_r)
% figure, imshow(opt_lab*255)
