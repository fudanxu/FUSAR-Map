clear;
clc;
str = 'GF3_MYN_UFS_999992_E116.0_N29.7_20160815_L1A_DH_L10002010515';
numfolder = '7';
local = '九江市';
origin = '无偏移九江市卫图';
img_path = strcat('F:\T_shixianzheng\datasets\', numfolder, '--', str, '\', str, '_', local, '\', origin, '\', origin, '_Level_17.tif');
shp_box_path = strcat('F:\T_shixianzheng\datasets\', numfolder, '--', str, '\', str, '_', local, '\', '边界.shp');
shp_path = strcat('F:\T_shixianzheng\datasets\', numfolder, '--', str, '\', str, '_', local, '\', str, '_', local, '.shp');

lon_change = 0; 
lat_change = 0; 
polygon = shape_polygon_change(shp_path, img_path, shp_box_path, lon_change, lat_change);

img = imread(img_path);
figure, imshow(img)
figure, imshow(polygon * 255)

b = labeloverlay(img, polygon);
figure, imshow(b)

polygon_name = strcat('F:\T_shixianzheng\datasets\', numfolder, '--', str, '\', str, '_', local, '\', origin, '\', 'polygon.png');
labelover_name = strcat('F:\T_shixianzheng\datasets\', numfolder, '--', str, '\', str, '_', local, '\', origin, '\', 'labelover.png');

imwrite(polygon, polygon_name);
imwrite(b, labelover_name);
