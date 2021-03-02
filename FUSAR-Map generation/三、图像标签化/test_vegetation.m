clear;
clc;

%% 路径字符
img_str = 'GF3_MDJ_UFS_002760_E120.0_N31.5_20170217_L1A_DH_L10002191778';
num_folder = '04';
num_str = '26';
img_path = strcat('H:\cities\', num_folder, '\', img_str, '\', img_str, '_光学\', img_str, '_卫图.tif');
imfo = imfinfo(img_path);
info_path = strcat('F:\T_shixianzheng\datasets\', num_str, '--', img_str, '\', 'opt_info.mat');
save(info_path, 'imfo');
% local_str = '_九江市';
%% 读入光学遥感
img_path = strcat('H:\cities\', num_folder, '\', img_str, '\', img_str, '_光学\', img_str, '_卫图_registered.mat');
img = load(img_path);
img = img.aerial_registered;
figure, imshow(img)
% img_path = strcat('H:\cities\', num_folder, '\', img_str, '\', img_str, '_光学\', img_str, '_卫图_registered.tif');
% img = imread(img_path);
% figure, imshow(img)

% check
% veg = img(6847:7963, 2.265e+4:2.363e+4, :);
% figure, imshow(veg)
% figure, subplot(3, 1, 1);
% imhist(veg(:, :, 1))
% axis([0 100 0 inf])
% subplot(3, 1, 2);
% imhist(veg(:, :, 2))
% axis([0 100 0 inf])
% subplot(3, 1, 3);
% imhist(veg(:, :, 3))
% axis([0 100 0 inf])
%% 读入3个label
lab_path = strcat('F:\T_shixianzheng\datasets\', num_str, '--', img_str, '\', img_str, '_label.mat');
lab_3 = load(lab_path);
lab_3 = uint8(lab_3.lab);

figure, imshow(lab_3 * 255)
% veg_1 = img(2.556e+4:2.791e+4, 1.436e+4:1.647e+4, :);
% veg_2 = img(1.314e+4:1.449e+4, 1.279e+4:1.4e+4, :);
% veg_3 = img(1.014e+4:1.042e+4, 1.402e+4:1.427e+4, :);
% veg_4 = img(8358:8601, 1.807e+4:1.829e+4, :);
% veg_5 = img(1.263e+4:1.337e+4, 2.607e+4:2.675e+4, :);
% veg_6 = img(9621:1.112e+4, 1.989e+4:2.156e+4, :);

% figure, imshow(img)
% figure, subplot(3, 1, 1);
% imhist(veg_6(:, :, 1))
% axis([0 100 0 inf])
% subplot(3, 1, 2);
% imhist(veg_6(:, :, 2))
% axis([0 100 0 inf])
% subplot(3, 1, 3);
% imhist(veg_6(:, :, 3))
% axis([0 100 0 inf])
%% 生成veg label
% lab_veg2 = img(:, :, 1) <= 15 & img(:, :, 2) >= 10 & img(:, :, 2) <= 70 & img(:, :, 3) <= 60 | (img(:, :, 1) >=23 & img(:, :, 1) <= 70 & ...
%     img(:, :, 2) >= 50 & img(:, :, 2) <= 90 & img(:, :, 3) >= 52 & img(:, :, 3) <= 92);
% figure, imshow(lab_veg2)

lab_veg = img(:, :, 1) <= 15 & img(:, :, 2) >= 10 & img(:, :, 2) <= 70 & img(:, :, 3) <= 60;
figure, imshow(lab_veg * 255)

% lab_veg = img(:, :, 1) >= 20 & img(:, :, 1) <= 80 & img(:, :, 2) >= 40 & img(:, :, 2) <= 100 & img(:, :, 3) <= 90 & img(:, :, 3) >= 30;
% figure, imshow(lab_veg * 255)

lab_veg = img(:, :, 1) >= 20 & img(:, :, 1) <= 100 & img(:, :, 2) >= 30 & img(:, :, 2) <= 100 & img(:, :, 3) <= 100 & img(:, :, 3) >= 40;
figure, imshow(lab_veg * 255)

% lab_veg = img(:, :, 1) <= 15 & img(:, :, 2) >= 10 & img(:, :, 2) <= 70 & img(:, :, 3) <= 60 | (img(:, :, 1) <= 60 & img(:, :, 1) >= 20 & ...
%     img(:, :, 2) <= 55 & img(:, :, 2) >= 15 & img(:, :, 3) >= 5 & img(:, :, 3) <= 45);
% figure, imshow(lab_veg * 255)
% 
% lab_veg = img(:, :, 1) <= 15 & img(:, :, 2) >= 10 & img(:, :, 2) <= 70 & img(:, :, 3) <= 60 | (img(:, :, 1) <= 55 & img(:, :, 1) >= 5 & ...
%     img(:, :, 2) <= 65 & img(:, :, 2) >= 20 & img(:, :, 3) >= 5 & img(:, :, 3) <= 45);
% figure, imshow(lab_veg * 255)

% k = max((lab_3 / 255), [], 3);
% figure, imshow(k * 255)

%% 滤波 采用投票滤波
lab_get2 = votefilt2Pro(lab_veg, 2);
lab_get2 = votefilt2Pro(lab_get2, 1);

% figure, imshow(lab_get2 * 255)
%% 去重复
lab_get2 = (uint8(lab_get2) - uint8(max(lab_3, [], 3))) > 0.1;
% figure, imshow(lab_get2 * 255)

veg_path = strcat('F:\T_shixianzheng\datasets\', num_str, '--', img_str, '\', img_str, '_label_veg.mat');
save(veg_path, 'lab_get2');

%% 色调
% building red：[255, 0, 0]
% veg green:[0, 255, 0]
% water blue:[0, 0, 255]
% road yellow:[255, 255, 0]
clear img
clear lab_veg

lab = repmat(lab_3(:, :, 2), 1, 1, 2); % road
lab(:, :, 1) = lab(:, :, 1) + lab_3(:, :, 1); % build
lab(:, :, 2) = lab(:, :, 2) + uint8(lab_get2); % veg
lab(:, :, 3) = lab_3(:, :, 3); % water

lab = uint8(lab * 255);

% figure, imshow(lab_3 * 255)
figure, imshow(lab)

lab_path_best = strcat('F:\T_shixianzheng\datasets\', num_str, '--', img_str, '\', img_str, '_label_best.mat');
save(lab_path_best, 'lab')

