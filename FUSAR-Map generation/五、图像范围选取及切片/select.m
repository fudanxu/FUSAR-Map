clear;
clc;

name_str = 'GF3_MYN_UFS_999992_E116.0_N29.7_20160815_L1A_DH_L10002010515';
lab_path = strcat(name_str, '_label_best.mat');
lab = load(lab_path);
lab = lab.lab;

opt_path = strcat(name_str, '_Œ¿Õº_registered.mat');
opt = load(opt_path);
% opt = imread(opt_path);
opt = opt.aerial_registered;

sar_path = strcat(name_str, '_registered.tif');
sar = imread(sar_path);

dem_path = strcat('data_registered_uint16.tif');
dem = imread(dem_path);

% figure, imshow(lab)
row = 2566:1.424e+4;
col = 1.091e+4:1.844e+4;
lab_select = lab(row, col, :);
% figure, imshow(lab_select)
opt_select = opt(row, col, :);
sar_select = sar(row, col, :);
dem_select = dem(row, col, :);

figure, imshow(sar_select);
figure, imshow(opt_select);
figure, imshow(lab_select);
figure, imshow(uint8(dem_select));


imwrite(sar_select, 'sar_select.tif');
imwrite(opt_select, 'opt_select.tif');
imwrite(lab_select, 'lab_select.tif');
imwrite(dem_select, 'dem_selcet.tif');
