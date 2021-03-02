clear;
clc;

sar_dir = fullfile('D:\D_shixianzheng', 'Datasets', 'sar_1024');
opt_dir = fullfile('D:\D_shixianzheng', 'Datasets', 'opt_1024');
lab_dir = fullfile('D:\D_shixianzheng', 'Datasets', 'lab_1024');
dem_dir = fullfile('D:\D_shixianzheng', 'Datasets', 'dem_1024');
if ~exist(sar_dir, 'dir')
    mkdir(sar_dir);
end
if ~exist(opt_dir, 'dir')
    mkdir(opt_dir);
end
if ~exist(lab_dir, 'dir')
    mkdir(lab_dir);
end
if ~exist(dem_dir, 'dir')
    mkdir(dem_dir);
end
datasets_info = {'文件名统称(location_图片序号_范围序号_行序号_列序号_序号)', '高分3号来源', '景中心俯仰角(度/°)', '近端俯仰角(度/°)', '远端俯仰角（度/°）', '宽分辨率（米/m）', ...
    '高分辨率（米/m）', '标称分辨率（米/m）', '光学图像分辨率（米/m）', 'DEM分辨率（米/m）', ...
    'BoundingBox[min_lon, min_lat; max_lon, max_lat]', 'Boundingbox_row_col[min_row, min_col; max_row, max_col]'};
seq = [01, 03, 03, 05, 08, 09, 10, 12]; %  10,
num = [7, 18, 21, 33, 64, 74, 83, 100]; % , 88
filepath = {'D:\D_shixianzheng\01-7\', 'D:\D_shixianzheng\03-18\', 'D:\D_shixianzheng\03-21\', 'D:\D_shixianzheng\05-33\', ...
    'D:\D_shixianzheng\08-64\', 'D:\D_shixianzheng\09-74\', 'D:\D_shixianzheng\10-83\', 'D:\D_shixianzheng\12-100\'}; % , 'D:\D_shixianzheng\10-88\'
Num = 0;
for i = 1:length(seq)
    info_path = strcat(filepath{i}, char(string(num(i))), 'Datasets\', 'slice_info_', char(string(i)), '.mat');
    sar_a_dir = strcat(filepath{i}, char(string(num(i))), 'Datasets\', 'sar_select');
    opt_a_dir = strcat(filepath{i}, char(string(num(i))), 'Datasets\', 'opt_select');
    lab_a_dir = strcat(filepath{i}, char(string(num(i))), 'Datasets\', 'lab_select');
    dem_a_dir = strcat(filepath{i}, char(string(num(i))), 'Datasets\', 'dem_select');
    info = load(info_path);
    info = info.slice_info;
    [M, N] = size(info);
    for j = 2:M
        Num = Num + 1;
        % normal_name = sprintf('%_03d.tif', Num);
        sar_name = sprintf('SAR_%03d.tif', Num);
        opt_name = sprintf('Optical_%03d.tif', Num);
        lab_name = sprintf('Label_%03d.tif', Num);
        dem_name = sprintf('DEM_%03d.tif', Num);
        filename = info{j, 1};
        filename1 = strsplit(info{j, 1}, '.');
        filename1 = filename1{1};
        
        sar_new_name = strcat(filename1, '_', sar_name);
        info_get = [{sar_new_name}, info(j, 2:N)];
        datasets_info = [datasets_info; info_get];
        sar_a_path = fullfile(sar_a_dir, filename);
        % sar_path = fullfile(sar_dir, sar_name);
        sar_path = fullfile(sar_dir, sar_new_name);
        f = imread(sar_a_path);
        imwrite(f, sar_path)
        % copyfile sar_a_path sar_path
        
        opt_new_name = strcat(filename1, '_', opt_name);
        opt_a_path = fullfile(opt_a_dir, filename);
        opt_path = fullfile(opt_dir, opt_new_name);
        f = imread(opt_a_path);
        imwrite(f, opt_path)
        % copyfile opt_a_path opt_path
        
        lab_new_name = strcat(filename1, '_', lab_name);
        lab_a_path = fullfile(lab_a_dir, filename);
        lab_path = fullfile(lab_dir, lab_new_name);
        f = imread(lab_a_path);
        imwrite(f, lab_path)
        % copyfile lab_a_path lab_path
        
        dem_new_name = strcat(filename1, '_', dem_name);
        dem_a_path = fullfile(dem_a_dir, filename);
        dem_path = fullfile(dem_dir, dem_new_name);
        f = imread(dem_a_path);
        imwrite(f, dem_path)
        % copyfile dem_a_path dem_path
    end
end
save('D:\D_shixianzheng\Datasets\datasets_info.mat', 'datasets_info');
% copyfile not success
