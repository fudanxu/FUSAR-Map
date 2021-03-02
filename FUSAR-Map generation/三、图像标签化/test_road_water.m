clear;
clc;
% elec_path = strcat('H:\cities\', num_folder, '\', img_str, '\', img_str, '电子地图_after.mat');
img_str = 'GF3_MYN_UFS_999992_E116.0_N29.7_20160815_L1A_DH_L10002010515';
num_folder = '01';
seq = '7';
% 
% elec_path = strcat('H:\cities\', num_folder, '\', img_str, '\', img_str, '_光学\', img_str, '_电子地图.tif');
% elec_img = imread(elec_path);

elec_path = strcat('H:\cities\', num_folder, '\', img_str, '\', img_str, '_光学\', img_str, '_卫图_registered.mat');
% % elec_img = imread(elec_path);
elec_img = load(elec_path);
% elec_img = elec_img.elecMap_registered;
elec_img = elec_img.aerial_registered;

figure, imshow(elec_img)

label = (elec_img(:, :, 1) == 170 & elec_img(:, :, 2) == 218 & elec_img(:, :, 3) == 255);
% figure, imshow(label * 255)
label_road = (elec_img(:, :, 1) == 255 & elec_img(:, :, 2) == 255 & elec_img(:, :, 3) == 255);
% figure, imshow(label_road * 255)

water_name = strcat('F:\T_shixianzheng\datasets\', seq, '--', img_str, '\elec_waterlabel_', seq, '_logical.png');
road_name = strcat('F:\T_shixianzheng\datasets\', seq, '--', img_str, '\elec_roadlabel_', seq, '_logical.png');

% water_name = strcat('F:\T_shixianzheng\datasets\', seq, '--', img_str, '\elec_waterlabel_', seq, '_logical_registered.png');
% road_name = strcat('F:\T_shixianzheng\datasets\', seq, '--', img_str, '\elec_roadlabel_', seq, '_logical_registered.png');

imwrite(label, water_name);
imwrite(label_road, road_name);


