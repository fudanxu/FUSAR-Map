clear;
clc;
%% information
%% 1
% seq = '01';
% location = 'JiuJiang';
% num = 7;
% NP = 1024;
% num_r = '7';
% row = 2566:1.424e+4;
% col = 1.091e+4:1.844e+4;
% source_name = 'GF3_MYN_UFS_999992_E116.0_N29.7_20160815_L1A_DH_L10002010515';
%% 2
% seq = '03';
% location = 'XiaMen';
% num = 18;
% NP = 1024;
% num_r = '18';
% row = 8996:3e+4;
% col = 2467:1.398e+4;
% source_name = 'GF3_MYN_UFS_002616_E118.2_N24.5_20170207_L1A_DH_L10002172522';
%% 3
% seq = '03';
% location = 'WuHan';
% num = 21;
% NP = 1024;
% num_r = '21';
% row = 1.029e+4:2.947e+4;
% col = 7493:1.841e+4;
% source_name = 'GF3_SAY_UFS_000605_E114.5_N30.5_20160920_L1A_DH_L10001982357';
%% 4
% seq = '05';
% location = 'TaiZhou';
% num = 33;
% num_r = '33';
% NP = 1024;
% row = 2.35e+4:3.937e+4;
% col = 1229:1.207e+4;
% source_name = 'GF3_MYN_UFS_002883_E121.5_N28.7_20170225_L1A_DH_L10002207051';
%% 5
% seq = '08';
% location = 'WenZhou';
% num = 64;
% num_r = '64';
% NP = 1024;
% row = 2.074e+4:3.412e+4;
% col = 6000:1.686e+4;
% source_name = 'GF3_MDJ_UFS_002760_E120.8_N28.0_20170217_L1A_DH_L10002191765';
%% 6
seq = '09';
location = 'TaiZhou';
num = 74;
num_r = '74';
NP = 1024;
row = 936:9355;
col = 8854:1.315e+4;
source_name = 'GF3_MYN_UFS_002883_E121.4_N28.5_20170225_L1A_DH_L10002207052';
%% 7
% seq = '10';
% location = 'NingBo';
% num = 83;
% num_r = '83';
% NP = 1024;
% row = 6667:2.075e+4;
% col = 5347:1.989e+4;
% source_name = 'GF3_SAY_UFS_002710_E121.9_N29.4_20170213_L1A_DH_L10002185830';
%% 8
% seq = '12';
% location = 'ZhangJiaKou';
% num = 100;
% num_r = '100';
% NP = 1024;
% row = 3903:1.153e+4;
% col = 1.903e+4:2.2e+4;
% source_name = 'GF3_MYN_UFS_002616_E114.8_N40.7_20170207_L1A_DH_L10002172578';
%% 8(useless)
% seq = '10';
% location = 'Tianjin';
% num = 88;
% num_r = '88';
% NP = 1024;
% row = 60:6874;
% col = 1.591e+4:3.4e+4;
% source_name = 'GF3_SAY_UFS_002868_E117.5_N38.8_20170224_L1A_DH_L10002205789';
%% source select
xlsx_path = 'D:\D_shixianzheng\图像优先级2.0_3m.xlsx';
lab_path = strcat('D:\D_shixianzheng\', seq, '-', num_r, '\', num_r, 'lab_select.tif');
sar_path = strcat('D:\D_shixianzheng\', seq, '-', num_r, '\', num_r, 'sar_select.tif');
opt_path = strcat('D:\D_shixianzheng\', seq, '-', num_r, '\', num_r, 'opt_select.tif');
dem_path = strcat('D:\D_shixianzheng\', seq, '-', num_r, '\', num_r, 'dem_selcet.tif');

xlsx = xlsread(xlsx_path);
xlsx = xlsx(num, :);
%% lat-lon:left-top, right-top, left-down, right-down
lon_lat = xlsx(17:24);
lat_get = lon_lat([1, 3, 5, 7]);
lon_get = lon_lat([2, 4, 6, 8]);
ortho_boundingbox = [min(lon_get), min(lat_get); max(lon_get), max(lat_get)];
ortho_name = strcat('D:\D_shixianzheng\', seq, '-', num_r, '\', source_name, '_registered.tif');
ortho_info = imfinfo(char(ortho_name));
%% each pixel lon and lat
% lon 
lon_step = (ortho_boundingbox(2, 1) - ortho_boundingbox(1, 1)) / (ortho_info.Width - 1);
% lat
lat_step = (ortho_boundingbox(2, 2) - ortho_boundingbox(1, 2)) / (ortho_info.Height - 1);

%% select tif image lon-lat
min_lat = max(lat_get) - max(row) * lat_step;
max_lat = max(lat_get) - min(row) * lat_step;
min_lon = min(lon_get) + min(col) * lon_step;
max_lon = min(lon_get) + max(col) * lon_step;

%% slice 1024 * 1024
lab_select_dir = strcat('D:\D_shixianzheng\', seq, '-', num_r, '\', num_r, 'Datasets\', 'lab_select');
if ~exist(lab_select_dir, 'dir')
    mkdir(lab_select_dir);
end
sar_select_dir = strcat('D:\D_shixianzheng\', seq, '-', num_r, '\', num_r, 'Datasets\', 'sar_select');
if ~exist(sar_select_dir, 'dir')
    mkdir(sar_select_dir);
end
opt_select_dir = strcat('D:\D_shixianzheng\', seq, '-', num_r, '\', num_r, 'Datasets\', 'opt_select');
if ~exist(opt_select_dir, 'dir')
    mkdir(opt_select_dir);
end
dem_select_dir = strcat('D:\D_shixianzheng\', seq, '-', num_r, '\', num_r, 'Datasets\', 'dem_select');
if ~exist(dem_select_dir, 'dir')
    mkdir(dem_select_dir);
end

lab_img = imread(lab_path);
sar_img = imread(sar_path);
opt_img = imread(opt_path);
dem_img = imread(dem_path);
[Row, Col, Ch] = size(lab_img);
row_num = floor(Row / NP);
col_num = floor(Col / NP);
%% .mat file
% 1- 1高分3号来源 2景中心俯仰角 3近端俯仰角 4远端俯仰角 5宽分辨率 6高分辨率 7标称分辨率 8光学图像分辨率 9DEM分辨率
% 10BoundingBox[min_lon, min_lat; max_lon, max_lat]
% 1- xlsx source_name, 7, 25, 26, 15, 16, 9, 1.02, 30, [boundingbox]
slice_info = {'文件名', '高分3号来源', '景中心俯仰角(度/°)', '近端俯仰角(度/°)', '远端俯仰角（度/°）', '宽分辨率（米/m）', '高分辨率（米/m）', ...
    '标称分辨率（米/m）', '光学图像分辨率（米/m）', 'DEM分辨率（米/m）', 'BoundingBox_lon_lat[min_lon, min_lat; max_lon, max_lat]', 'Boundingbox_row_col[min_row, min_col; max_row, max_col]'};
%% slice choice
select = 6;
switch select
    case 1
        K = 0;
        % row_begin = Row - row_num * NP;
        col_begin = Col - col_num * NP;
        for i = 1:row_num
            row_region = (NP * (i - 1) + 1) : NP * i;
            min_lat_get = max_lat - max(row_region) * lat_step;
            max_lat_get = max_lat - min(row_region) * lat_step;
            for j = 1:col_num
                K = K + 1;
                col_region = (col_begin + NP * (j - 1) + 1) : (col_begin + NP * j);
                boundingbox_row_col = [row(1) + row_region(1), col(1) + col_region(1); row(1) + row_region(end), col(1) + col_region(end)];
                min_lon_get = min_lon + min(col_region) * lon_step;
                max_lon_get = min_lon + max(col_region) * lon_step;
                l_select = lab_img(row_region, col_region, :);
                s_select = sar_img(row_region, col_region, :);
                o_select = opt_img(row_region, col_region, :);
                d_select = dem_img(row_region, col_region, :);
                boundingbox_get = [min_lon_get, min_lat_get; max_lon_get, max_lat_get];
                % same_name = strcat(location, '_', sprintf('%03d.tif', K));
                same_name = strcat(location, '_', sprintf('%02d', select), '_', sprintf('%02d', 1), '_', sprintf('%02d', i), '_', sprintf('%02d.tif', j));
                info_get = {same_name, source_name, xlsx(7), xlsx(25), xlsx(26), xlsx(15), xlsx(16), xlsx(9), 1.02, 30, boundingbox_get, boundingbox_row_col};
                slice_info = [slice_info; info_get];
                l_path = strcat(lab_select_dir, '\', same_name);
                s_path = strcat(sar_select_dir, '\', same_name);
                o_path = strcat(opt_select_dir, '\', same_name);
                d_path = strcat(dem_select_dir, '\', same_name);
                imwrite(l_select, l_path);
                imwrite(s_select, s_path);
                imwrite(o_select, o_path);
                imwrite(d_select, d_path);
            end
        end
        info_path = strcat('D:\D_shixianzheng\', seq, '-', num_r, '\', num_r, 'Datasets\', 'slice_info_1.mat');
        save(info_path, 'slice_info');
    case 2
        K = 0;
        row_begin = Row - row_num * NP;
        % col_begin = Col - col_num * NP;
        for i = 6:row_num
            row_region = (row_begin + NP * (i - 1) + 1) : (row_begin + NP * i);
            min_lat_get = max_lat - max(row_region) * lat_step;
            max_lat_get = max_lat - min(row_region) * lat_step;
            for j = 3:col_num
                K = K + 1;
                col_region = (NP * (j - 1) + 1) : NP * j;
                boundingbox_row_col = [row(1) + row_region(1), col(1) + col_region(1); row(1) + row_region(end), col(1) + col_region(end)];
                min_lon_get = min_lon + min(col_region) * lon_step;
                max_lon_get = min_lon + max(col_region) * lon_step;
                l_select = lab_img(row_region, col_region, :);
                s_select = sar_img(row_region, col_region, :);
                o_select = opt_img(row_region, col_region, :);
                d_select = dem_img(row_region, col_region, :);
                boundingbox_get = [min_lon_get, min_lat_get; max_lon_get, max_lat_get];
                % same_name = strcat(location, '_', sprintf('%03d.tif', K));
                same_name = strcat(location, '_', sprintf('%02d', select), '_', sprintf('%02d', 1), '_', sprintf('%02d', (i - 5)), '_', sprintf('%02d.tif', (j - 2)));
                info_get = {same_name, source_name, xlsx(7), xlsx(25), xlsx(26), xlsx(15), xlsx(16), xlsx(9), 1.02, 30, boundingbox_get, boundingbox_row_col};
                slice_info = [slice_info; info_get];
                l_path = strcat(lab_select_dir, '\', same_name);
                s_path = strcat(sar_select_dir, '\', same_name);
                o_path = strcat(opt_select_dir, '\', same_name);
                d_path = strcat(dem_select_dir, '\', same_name);
                imwrite(l_select, l_path);
                imwrite(s_select, s_path);
                imwrite(o_select, o_path);
                imwrite(d_select, d_path);
            end
        end
        info_path = strcat('D:\D_shixianzheng\', seq, '-', num_r, '\', num_r, 'Datasets\', 'slice_info_2.mat');
        save(info_path, 'slice_info');
    case 3
        K = 0;
        row_begin = Row - row_num * NP;
        col_begin = Col - col_num * NP;
        for i = 1:(row_num - 8)
            row_region = (row_begin + NP * (i - 1) + 1) : (row_begin + NP * i);
            min_lat_get = max_lat - max(row_region) * lat_step;
            max_lat_get = max_lat - min(row_region) * lat_step;
            for j = 2:col_num
                K = K + 1;
                col_region = (col_begin + NP * (j - 1) + 1) : (col_begin + NP * j);
                boundingbox_row_col = [row(1) + row_region(1), col(1) + col_region(1); row(1) + row_region(end), col(1) + col_region(end)];
                min_lon_get = min_lon + min(col_region) * lon_step;
                max_lon_get = min_lon + max(col_region) * lon_step;
                l_select = lab_img(row_region, col_region, :);
                s_select = sar_img(row_region, col_region, :);
                o_select = opt_img(row_region, col_region, :);
                d_select = dem_img(row_region, col_region, :);
                boundingbox_get = [min_lon_get, min_lat_get; max_lon_get, max_lat_get];
                % same_name = strcat(location, '_', sprintf('%03d.tif', K));
                same_name = strcat(location, '_', sprintf('%02d', select), '_', sprintf('%02d', 1), '_', sprintf('%02d', i), '_', sprintf('%02d.tif', (j - 1)));
                info_get = {same_name, source_name, xlsx(7), xlsx(25), xlsx(26), xlsx(15), xlsx(16), xlsx(9), 1.02, 30, boundingbox_get, boundingbox_row_col};
                slice_info = [slice_info; info_get];
                l_path = strcat(lab_select_dir, '\', same_name);
                s_path = strcat(sar_select_dir, '\', same_name);
                o_path = strcat(opt_select_dir, '\', same_name);
                d_path = strcat(dem_select_dir, '\', same_name);
                imwrite(l_select, l_path);
                imwrite(s_select, s_path);
                imwrite(o_select, o_path);
                imwrite(d_select, d_path);
            end
        end
        for i = (row_num - 7):row_num
            row_region = (row_begin + NP * (i - 1) + 1) : (row_begin + NP * i);
            min_lat_get = max_lat - max(row_region) * lat_step;
            max_lat_get = max_lat - min(row_region) * lat_step;
            for j = 1:col_num
                K = K + 1;
                col_region = (NP * (j - 1) + 1) : (NP * j);
                boundingbox_row_col = [row(1) + row_region(1), col(1) + col_region(1); row(1) + row_region(end), col(1) + col_region(end)];
                min_lon_get = min_lon + min(col_region) * lon_step;
                max_lon_get = min_lon + max(col_region) * lon_step;
                l_select = lab_img(row_region, col_region, :);
                s_select = sar_img(row_region, col_region, :);
                o_select = opt_img(row_region, col_region, :);
                d_select = dem_img(row_region, col_region, :);
                boundingbox_get = [min_lon_get, min_lat_get; max_lon_get, max_lat_get];
                % same_name = strcat(location, '_', sprintf('%03d.tif', K));
                same_name = strcat(location, '_', sprintf('%02d', select), '_', sprintf('%02d', 2), '_', sprintf('%02d', (i - (row_num - 8))), '_', sprintf('%02d.tif', j));
                info_get = {same_name, source_name, xlsx(7), xlsx(25), xlsx(26), xlsx(15), xlsx(16), xlsx(9), 1.02, 30, boundingbox_get, boundingbox_row_col};
                slice_info = [slice_info; info_get];
                l_path = strcat(lab_select_dir, '\', same_name);
                s_path = strcat(sar_select_dir, '\', same_name);
                o_path = strcat(opt_select_dir, '\', same_name);
                d_path = strcat(dem_select_dir, '\', same_name);
                imwrite(l_select, l_path);
                imwrite(s_select, s_path);
                imwrite(o_select, o_path);
                imwrite(d_select, d_path);
            end
        end
        info_path = strcat('D:\D_shixianzheng\', seq, '-', num_r, '\', num_r, 'Datasets\', 'slice_info_3.mat');
        save(info_path, 'slice_info');
    case 4
        K = 0;
        % row_begin = Row - row_num * NP + 1;
        col_begin = Col - col_num * NP;
        for i = 2:(row_num - 8)
            row_region = (NP * (i - 1) + 1) : NP * i;
            min_lat_get = max_lat - max(row_region) * lat_step;
            max_lat_get = max_lat - min(row_region) * lat_step;
            for j = 5:col_num
                K = K + 1;
                col_region = (col_begin + NP * (j - 1) + 1) : (col_begin + NP * j);
                boundingbox_row_col = [row(1) + row_region(1), col(1) + col_region(1); row(1) + row_region(end), col(1) + col_region(end)];
                min_lon_get = min_lon + min(col_region) * lon_step;
                max_lon_get = min_lon + max(col_region) * lon_step;
                l_select = lab_img(row_region, col_region, :);
                s_select = sar_img(row_region, col_region, :);
                o_select = opt_img(row_region, col_region, :);
                d_select = dem_img(row_region, col_region, :);
                boundingbox_get = [min_lon_get, min_lat_get; max_lon_get, max_lat_get];
                % same_name = strcat(location, '_', sprintf('%03d.tif', K));
                same_name = strcat(location, '_', sprintf('%02d', select), '_', sprintf('%02d', 1), '_', sprintf('%02d', (i - 1)), '_', sprintf('%02d.tif', (j - 4)));
                info_get = {same_name, source_name, xlsx(7), xlsx(25), xlsx(26), xlsx(15), xlsx(16), xlsx(9), 1.02, 30, boundingbox_get, boundingbox_row_col};
                slice_info = [slice_info; info_get];
                l_path = strcat(lab_select_dir, '\', same_name);
                s_path = strcat(sar_select_dir, '\', same_name);
                o_path = strcat(opt_select_dir, '\', same_name);
                d_path = strcat(dem_select_dir, '\', same_name);
                imwrite(l_select, l_path);
                imwrite(s_select, s_path);
                imwrite(o_select, o_path);
                imwrite(d_select, d_path);
            end
        end
        info_path = strcat('D:\D_shixianzheng\', seq, '-', num_r, '\', num_r, 'Datasets\', 'slice_info_4.mat');
        save(info_path, 'slice_info');
    case 5
        K = 0;
        row_begin = Row - row_num * NP;
        col_begin = Col - col_num * NP;
        for i = 1:8
            row_region = (NP * (i - 1) + 1) : (NP * i);
            min_lat_get = max_lat - max(row_region) * lat_step;
            max_lat_get = max_lat - min(row_region) * lat_step;
            for j = 1:col_num
                K = K + 1;
                col_region = (NP * (j - 1) + 1) : NP * j;
                boundingbox_row_col = [row(1) + row_region(1), col(1) + col_region(1); row(1) + row_region(end), col(1) + col_region(end)];
                min_lon_get = min_lon + min(col_region) * lon_step;
                max_lon_get = min_lon + max(col_region) * lon_step;
                l_select = lab_img(row_region, col_region, :);
                s_select = sar_img(row_region, col_region, :);
                o_select = opt_img(row_region, col_region, :);
                d_select = dem_img(row_region, col_region, :);
                boundingbox_get = [min_lon_get, min_lat_get; max_lon_get, max_lat_get];
                % same_name = strcat(location, '_', sprintf('%03d.tif', K));
                same_name = strcat(location, '_', sprintf('%02d', select), '_', sprintf('%02d', 1), '_', sprintf('%02d', i), '_', sprintf('%02d.tif', j));
                info_get = {same_name, source_name, xlsx(7), xlsx(25), xlsx(26), xlsx(15), xlsx(16), xlsx(9), 1.02, 30, boundingbox_get, boundingbox_row_col};
                slice_info = [slice_info; info_get];
                l_path = strcat(lab_select_dir, '\', same_name);
                s_path = strcat(sar_select_dir, '\', same_name);
                o_path = strcat(opt_select_dir, '\', same_name);
                d_path = strcat(dem_select_dir, '\', same_name);
                imwrite(l_select, l_path);
                imwrite(s_select, s_path);
                imwrite(o_select, o_path);
                imwrite(d_select, d_path);
            end
        end
        for i = 9:row_num
            row_region = (NP * (i - 1) + 1) : (NP * i);
            min_lat_get = max_lat - max(row_region) * lat_step;
            max_lat_get = max_lat - min(row_region) * lat_step;
            for j = 2:col_num
                K = K + 1;
                col_region = (col_begin + NP * (j - 1) + 1) : (col_begin + NP * j);
                boundingbox_row_col = [row(1) + row_region(1), col(1) + col_region(1); row(1) + row_region(end), col(1) + col_region(end)];
                min_lon_get = min_lon + min(col_region) * lon_step;
                max_lon_get = min_lon + max(col_region) * lon_step;
                l_select = lab_img(row_region, col_region, :);
                s_select = sar_img(row_region, col_region, :);
                o_select = opt_img(row_region, col_region, :);
                d_select = dem_img(row_region, col_region, :);
                boundingbox_get = [min_lon_get, min_lat_get; max_lon_get, max_lat_get];
                % same_name = strcat(location, '_', sprintf('%03d.tif', K));
                same_name = strcat(location, '_', sprintf('%02d', select), '_', sprintf('%02d', 2), '_', sprintf('%02d', (i - 8)), '_', sprintf('%02d.tif', (j - 1)));
                info_get = {same_name, source_name, xlsx(7), xlsx(25), xlsx(26), xlsx(15), xlsx(16), xlsx(9), 1.02, 30, boundingbox_get, boundingbox_row_col};
                slice_info = [slice_info; info_get];
                l_path = strcat(lab_select_dir, '\', same_name);
                s_path = strcat(sar_select_dir, '\', same_name);
                o_path = strcat(opt_select_dir, '\', same_name);
                d_path = strcat(dem_select_dir, '\', same_name);
                imwrite(l_select, l_path);
                imwrite(s_select, s_path);
                imwrite(o_select, o_path);
                imwrite(d_select, d_path);
            end
        end
        info_path = strcat('D:\D_shixianzheng\', seq, '-', num_r, '\', num_r, 'Datasets\', 'slice_info_5.mat');
        save(info_path, 'slice_info');
    case 6
        K = 0;
        row_begin = Row - row_num * NP;
        col_begin = Col - col_num * NP;
        for i = 3:5
            row_region = (NP * (i - 1) + 1) : NP * i;
            min_lat_get = max_lat - max(row_region) * lat_step;
            max_lat_get = max_lat - min(row_region) * lat_step;
            for j = 1:(col_num - 1)
                K = K + 1;
                col_region = (NP * (j - 1) + 1) : NP * j;
                boundingbox_row_col = [row(1) + row_region(1), col(1) + col_region(1); row(1) + row_region(end), col(1) + col_region(end)];
                min_lon_get = min_lon + min(col_region) * lon_step;
                max_lon_get = min_lon + max(col_region) * lon_step;
                l_select = lab_img(row_region, col_region, :);
                s_select = sar_img(row_region, col_region, :);
                o_select = opt_img(row_region, col_region, :);
                d_select = dem_img(row_region, col_region, :);
                boundingbox_get = [min_lon_get, min_lat_get; max_lon_get, max_lat_get];
                % same_name = strcat(location, '_', sprintf('%03d.tif', K));
                same_name = strcat(location, '_', sprintf('%02d', select), '_', sprintf('%02d', 1), '_', sprintf('%02d', (i - 2)), '_', sprintf('%02d.tif', j));
                info_get = {same_name, source_name, xlsx(7), xlsx(25), xlsx(26), xlsx(15), xlsx(16), xlsx(9), 1.02, 30, boundingbox_get, boundingbox_row_col};
                slice_info = [slice_info; info_get];
                l_path = strcat(lab_select_dir, '\', same_name);
                s_path = strcat(sar_select_dir, '\', same_name);
                o_path = strcat(opt_select_dir, '\', same_name);
                d_path = strcat(dem_select_dir, '\', same_name);
                imwrite(l_select, l_path);
                imwrite(s_select, s_path);
                imwrite(o_select, o_path);
                imwrite(d_select, d_path);
            end
        end
        for i = 1:2
            row_region = (row_begin + NP * (i - 1) + 1) : (row_begin + NP * i);
            min_lat_get = max_lat - max(row_region) * lat_step;
            max_lat_get = max_lat - min(row_region) * lat_step;
            for j = (col_num - 1):col_num
                K = K + 1;
                col_region = (NP * (j - 1) + 1) : NP * j;
                boundingbox_row_col = [row(1) + row_region(1), col(1) + col_region(1); row(1) + row_region(end), col(1) + col_region(end)];
                min_lon_get = min_lon + min(col_region) * lon_step;
                max_lon_get = min_lon + max(col_region) * lon_step;
                l_select = lab_img(row_region, col_region, :);
                s_select = sar_img(row_region, col_region, :);
                o_select = opt_img(row_region, col_region, :);
                d_select = dem_img(row_region, col_region, :);
                boundingbox_get = [min_lon_get, min_lat_get; max_lon_get, max_lat_get];
                % same_name = strcat(location, '_', sprintf('%03d.tif', K));
                same_name = strcat(location, '_', sprintf('%02d', select), '_', sprintf('%02d', 2), '_', sprintf('%02d', i), '_', sprintf('%02d.tif', (j - (col_num - 2))));
                info_get = {same_name, source_name, xlsx(7), xlsx(25), xlsx(26), xlsx(15), xlsx(16), xlsx(9), 1.02, 30, boundingbox_get, boundingbox_row_col};
                slice_info = [slice_info; info_get];
                l_path = strcat(lab_select_dir, '\', same_name);
                s_path = strcat(sar_select_dir, '\', same_name);
                o_path = strcat(opt_select_dir, '\', same_name);
                d_path = strcat(dem_select_dir, '\', same_name);
                imwrite(l_select, l_path);
                imwrite(s_select, s_path);
                imwrite(o_select, o_path);
                imwrite(d_select, d_path);
            end
        end
        for i = 7:row_num
            row_region = (NP * (i - 1) + 1) : NP * i;
            min_lat_get = max_lat - max(row_region) * lat_step;
            max_lat_get = max_lat - min(row_region) * lat_step;
            for j = 1:(col_num - 1)
                K = K + 1;
                col_region = (NP * (j - 1) + 1) : NP * j;
                boundingbox_row_col = [row(1) + row_region(1), col(1) + col_region(1); row(1) + row_region(end), col(1) + col_region(end)];
                min_lon_get = min_lon + min(col_region) * lon_step;
                max_lon_get = min_lon + max(col_region) * lon_step;
                l_select = lab_img(row_region, col_region, :);
                s_select = sar_img(row_region, col_region, :);
                o_select = opt_img(row_region, col_region, :);
                d_select = dem_img(row_region, col_region, :);
                boundingbox_get = [min_lon_get, min_lat_get; max_lon_get, max_lat_get];
                % same_name = strcat(location, '_', sprintf('%03d.tif', K));
                same_name = strcat(location, '_', sprintf('%02d', select), '_', sprintf('%02d', 3), '_', sprintf('%02d', (i - 6)), '_', sprintf('%02d.tif', j));
                info_get = {same_name, source_name, xlsx(7), xlsx(25), xlsx(26), xlsx(15), xlsx(16), xlsx(9), 1.02, 30, boundingbox_get, boundingbox_row_col};
                slice_info = [slice_info; info_get];
                l_path = strcat(lab_select_dir, '\', same_name);
                s_path = strcat(sar_select_dir, '\', same_name);
                o_path = strcat(opt_select_dir, '\', same_name);
                d_path = strcat(dem_select_dir, '\', same_name);
                imwrite(l_select, l_path);
                imwrite(s_select, s_path);
                imwrite(o_select, o_path);
                imwrite(d_select, d_path);
            end
        end
        info_path = strcat('D:\D_shixianzheng\', seq, '-', num_r, '\', num_r, 'Datasets\', 'slice_info_6.mat');
        save(info_path, 'slice_info');
    case 7
        K = 0;
        row_begin = Row - row_num * NP;
        col_begin = Col - col_num * NP;
        for i = 7:(row_num - 1)
            row_region = (NP * (i - 1) + 1) : (NP * i);
            min_lat_get = max_lat - max(row_region) * lat_step;
            max_lat_get = max_lat - min(row_region) * lat_step;
            for j = 9:col_num
                K = K + 1;
                col_region = (col_begin + NP * (j - 1) + 1) : (col_begin + NP * j);
                boundingbox_row_col = [row(1) + row_region(1), col(1) + col_region(1); row(1) + row_region(end), col(1) + col_region(end)];
                min_lon_get = min_lon + min(col_region) * lon_step;
                max_lon_get = min_lon + max(col_region) * lon_step;
                l_select = lab_img(row_region, col_region, :);
                s_select = sar_img(row_region, col_region, :);
                o_select = opt_img(row_region, col_region, :);
                d_select = dem_img(row_region, col_region, :);
                boundingbox_get = [min_lon_get, min_lat_get; max_lon_get, max_lat_get];
                % same_name = strcat(location, '_', sprintf('%03d.tif', K));
                same_name = strcat(location, '_', sprintf('%02d', select), '_', sprintf('%02d', 1), '_', sprintf('%02d', (i - 6)), '_', sprintf('%02d.tif', (j - 8)));
                info_get = {same_name, source_name, xlsx(7), xlsx(25), xlsx(26), xlsx(15), xlsx(16), xlsx(9), 1.02, 30, boundingbox_get, boundingbox_row_col};
                slice_info = [slice_info; info_get];
                l_path = strcat(lab_select_dir, '\', same_name);
                s_path = strcat(sar_select_dir, '\', same_name);
                o_path = strcat(opt_select_dir, '\', same_name);
                d_path = strcat(dem_select_dir, '\', same_name);
                imwrite(l_select, l_path);
                imwrite(s_select, s_path);
                imwrite(o_select, o_path);
                imwrite(d_select, d_path);
            end
        end
        info_path = strcat('D:\D_shixianzheng\', seq, '-', num_r, '\', num_r, 'Datasets\', 'slice_info_7.mat');
        save(info_path, 'slice_info');
%     case 8
%         K = 0;
%         % row_begin = Row - row_num * NP + 1;
%         col_begin = Col - col_num * NP;
%         for i = 3:(row_num - 1)
%             row_region = (NP * (i - 1) + 1) : NP * i;
%             min_lat_get = max_lat - max(row_region) * lat_step;
%             max_lat_get = max_lat - min(row_region) * lat_step;
%             if i == 4
%                 for j = 3:(col_num - 11)
%                     K = K + 1;
%                     col_region = (col_begin + NP * (j - 1) + 1) : (col_begin + NP * j);
%                     min_lon_get = min_lon + min(col_region) * lon_step;
%                     max_lon_get = min_lon + max(col_region) * lon_step;
%                     l_select = lab_img(row_region, col_region, :);
%                     s_select = sar_img(row_region, col_region, :);
%                     o_select = opt_img(row_region, col_region, :);
%                     d_select = dem_img(row_region, col_region, :);
%                     boundingbox_get = [min_lon_get, min_lat_get; max_lon_get, max_lat_get];
%                     same_name = strcat(location, '_', sprintf('%03d.tif', K));
%                     info_get = {same_name, source_name, xlsx(7), xlsx(25), xlsx(26), xlsx(15), xlsx(16), xlsx(9), 1.02, 30, boundingbox_get};
%                     slice_info = [slice_info; info_get];
%                     l_path = strcat(lab_select_dir, '\', same_name);
%                     s_path = strcat(sar_select_dir, '\', same_name);
%                     o_path = strcat(opt_select_dir, '\', same_name);
%                     d_path = strcat(dem_select_dir, '\', same_name);
%                     imwrite(l_select, l_path);
%                     imwrite(s_select, s_path);
%                     imwrite(o_select, o_path);
%                     imwrite(d_select, d_path);
%                 end
%             end
%             if i == 5
%                 for j = 3:(col_num - 6)
%                     K = K + 1;
%                     col_region = (col_begin + NP * (j - 1) + 1) : (col_begin + NP * j);
%                     min_lon_get = min_lon + min(col_region) * lon_step;
%                     max_lon_get = min_lon + max(col_region) * lon_step;
%                     l_select = lab_img(row_region, col_region, :);
%                     s_select = sar_img(row_region, col_region, :);
%                     o_select = opt_img(row_region, col_region, :);
%                     d_select = dem_img(row_region, col_region, :);
%                     boundingbox_get = [min_lon_get, min_lat_get; max_lon_get, max_lat_get];
%                     same_name = strcat(location, '_', sprintf('%03d.tif', K));
%                     info_get = {same_name, source_name, xlsx(7), xlsx(25), xlsx(26), xlsx(15), xlsx(16), xlsx(9), 1.02, 30, boundingbox_get};
%                     slice_info = [slice_info; info_get];
%                     l_path = strcat(lab_select_dir, '\', same_name);
%                     s_path = strcat(sar_select_dir, '\', same_name);
%                     o_path = strcat(opt_select_dir, '\', same_name);
%                     d_path = strcat(dem_select_dir, '\', same_name);
%                     imwrite(l_select, l_path);
%                     imwrite(s_select, s_path);
%                     imwrite(o_select, o_path);
%                     imwrite(d_select, d_path);
%                 end
%             end
%         end
%         info_path = strcat('D:\D_shixianzheng\', seq, '-', num_r, '\', 'slice_info_8.mat');
%         save(info_path, 'slice_info');
    case 8
        K = 0;
        row_begin = Row - row_num * NP;
        col_begin = round((Col - col_num * NP) / 2);
        % col_begin = Col - col_num * NP;
        for i = 1:(row_num - 3)
            row_region = (NP * (i - 1) + 1) : NP * i;
            min_lat_get = max_lat - max(row_region) * lat_step;
            max_lat_get = max_lat - min(row_region) * lat_step;
            for j = 1:col_num
                K = K + 1;
                col_region = (col_begin + NP * (j - 1) + 1) : (col_begin + NP * j);
                boundingbox_row_col = [row(1) + row_region(1), col(1) + col_region(1); row(1) + row_region(end), col(1) + col_region(end)];
                min_lon_get = min_lon + min(col_region) * lon_step;
                max_lon_get = min_lon + max(col_region) * lon_step;
                l_select = lab_img(row_region, col_region, :);
                s_select = sar_img(row_region, col_region, :);
                o_select = opt_img(row_region, col_region, :);
                d_select = dem_img(row_region, col_region, :);
                boundingbox_get = [min_lon_get, min_lat_get; max_lon_get, max_lat_get];
                % same_name = strcat(location, '_', sprintf('%03d.tif', K));
                same_name = strcat(location, '_', sprintf('%02d', select), '_', sprintf('%02d', 1), '_', sprintf('%02d', i), '_', sprintf('%02d.tif', j));
                info_get = {same_name, source_name, xlsx(7), xlsx(25), xlsx(26), xlsx(15), xlsx(16), xlsx(9), 1.02, 30, boundingbox_get, boundingbox_row_col};
                slice_info = [slice_info; info_get];
                l_path = strcat(lab_select_dir, '\', same_name);
                s_path = strcat(sar_select_dir, '\', same_name);
                o_path = strcat(opt_select_dir, '\', same_name);
                d_path = strcat(dem_select_dir, '\', same_name);
                imwrite(l_select, l_path);
                imwrite(s_select, s_path);
                imwrite(o_select, o_path);
                imwrite(d_select, d_path);
            end
        end
        for i = (row_num - 1):row_num
            row_region = (NP * (i - 1) + 1) : (NP * i);
            min_lat_get = max_lat - max(row_region) * lat_step;
            max_lat_get = max_lat - min(row_region) * lat_step;
            for j = 1:col_num
                K = K + 1;
                col_region = (col_begin + NP * (j - 1) + 1) : (col_begin + NP * j);
                boundingbox_row_col = [row(1) + row_region(1), col(1) + col_region(1); row(1) + row_region(end), col(1) + col_region(end)];
                min_lon_get = min_lon + min(col_region) * lon_step;
                max_lon_get = min_lon + max(col_region) * lon_step;
                l_select = lab_img(row_region, col_region, :);
                s_select = sar_img(row_region, col_region, :);
                o_select = opt_img(row_region, col_region, :);
                d_select = dem_img(row_region, col_region, :);
                boundingbox_get = [min_lon_get, min_lat_get; max_lon_get, max_lat_get];
                % same_name = strcat(location, '_', sprintf('%03d.tif', K));
                same_name = strcat(location, '_', sprintf('%02d', select), '_', sprintf('%02d', 2), '_', sprintf('%02d', (i - (row_num - 2))), '_', sprintf('%02d.tif', j));
                info_get = {same_name, source_name, xlsx(7), xlsx(25), xlsx(26), xlsx(15), xlsx(16), xlsx(9), 1.02, 30, boundingbox_get, boundingbox_row_col};
                slice_info = [slice_info; info_get];
                l_path = strcat(lab_select_dir, '\', same_name);
                s_path = strcat(sar_select_dir, '\', same_name);
                o_path = strcat(opt_select_dir, '\', same_name);
                d_path = strcat(dem_select_dir, '\', same_name);
                imwrite(l_select, l_path);
                imwrite(s_select, s_path);
                imwrite(o_select, o_path);
                imwrite(d_select, d_path);
            end
        end
        info_path = strcat('D:\D_shixianzheng\', seq, '-', num_r, '\', num_r, 'Datasets\', 'slice_info_8.mat');
        save(info_path, 'slice_info');
end



    
