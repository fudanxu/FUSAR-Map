clear;
clc;
%% 输入变量
% xls_path = 'F:\矢量建筑图\图像优先级2.0_3m.xlsx';
% xls_number = 7;
str = 'GF3_MYN_UFS_999992_E116.0_N29.7_20160815_L1A_DH_L10002010515';
num_str = '7';
shp_path = strcat('F:\T_shixianzheng\datasets\', num_str, '--', str, '\', num_str, '边界.shp');
shp = shaperead(shp_path);
box = shp.BoundingBox;
% xls_num = xls_number + 1;
% xlrange = strcat('Q', string(xls_num), ':', 'X', string(xls_num));
% sheet = 1;
% lon_lat = xlsread(xls_path, sheet, xlrange);
% lon_offset = [-0.02, 0.02, -0.02, 0.02];
% lat_offset = [0.02, 0.02, -0.02, -0.02];
% lat_get = lon_lat(:, [1, 3, 5, 7]) + lat_offset;
% lon_get = lon_lat(:, [2, 4, 6, 8]) + lon_offset;
min_lon = box(1, 1);
max_lon = box(2, 1);
min_lat = box(1, 2);
max_lat = box(2, 2);

opt_info = load('opt_info.mat');
info = opt_info.imfo;
[data, Ra] = geotiffread('7.tif');
lons = load('7lons.mat');
lons = lons.lons;
lats = load('7lats.mat');
lats = lats.lats;

lats = sort(lats, 'descend');

%% 确定图像范围
res = 1e-4;
row1 = round((max(lats) - max_lat) / res + 1);
row2 = round((max(lats) - min_lat) / res + 1);

col1 = round((min_lon - min(lons)) / res + 1);
col2 = round((max_lon - min(lons)) / res + 1);

same_data = data(row1:row2, col1:col2);
figure, imshow(uint8(same_data));

% dem_same = zeros(info.Height, info.Width);
% res = 1e-4;
% lon_step = (max_lon - min_lon) / (info.Width - 1);
% lat_step = (max_lat - min_lat) / (info.Height - 1);
% for i = 1:info.Height
%     lat = min_lat + lat_step * i;
%     row = round((lat - min(lats)) / res + 1);
%     for j = 1:info.Width
%         lon_get = min_lon + lon_step * j;
%         col = round((lon_get - min(lons)) / res + 1);
%         dem_same(i, j) = data(row, col);
%     end
% end

% lons2 = min_lon:lon_step:max_lon;
% lats2 = min_lat:lat_step:max_lat;
output_geotiff_path = 'dem_same_low7.tif';
% save('data_same.tif', 'dem_same');

% dem_same_16 = uint16(dem_same);
% save('data_same_16.jpg', 'dem_same_16');
% figure, imshow(uint8(dem_same_16));

lats2 = lats(row1:row2);
lons2 = lons(col1:col2);

%% 保存
R = gcr2ll(lons2,lats2);
geotiffwrite(output_geotiff_path,same_data,R);


%% 插值再保存
F = griddedInterpolant(double(same_data));
[sx,sy] = size(same_data);
rr = sx / (info.Height - 1);
rc = sy / (info.Width - 1);
xq = (0:rr:sx)';
yq = (0:rc:sy)';
vq = uint16(F({xq,yq}));
figure, imshow(uint8(vq))

ra = (max(lats2) - min(lats2)) / (info.Height - 1);
lats3 = lats2(1):-ra:lats2(sx);
ro = (max(lons2) - min(lons2)) / (info.Width - 1);
lons3 = lons2(1):ro:lons2(sy);

geotiff_path = 'dem_same_high7.tif';

save('data.mat', 'vq');
save('lons3.mat', 'lons3');
save('lats3.mat', 'lats3');

% imwrite(vq, 'data_img.tif');

R2 = gcr2ll(lons3,lats3);

geotiffwrite(geotiff_path,vq,R2);

geotiffwrite(geotiff_path,uint8(vq),R2);
geotiffwrite(geotiff_path, vq, R2, 'TiffType', 'bigtiff');

[M, N] = geotiffread('dem_same_high7.tif');
