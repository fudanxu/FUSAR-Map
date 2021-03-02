% 需要修改
root_path = 'H:/cities/05/';

% 不用修改
files = dir(root_path);
numFiles = size(files, 1) - 2;
for i = 1:numFiles
    name = files(i+2).name;
    if ~exist(strcat(root_path, name, '/', 'points_', name, '.mat'))
        continue;
    else
        ortho_path = strcat(root_path, name, '/', name, '_geocode.tif');
        aerial_path = strcat(root_path, name, '/', name, '_光学/', name, '_卫图.tif');
        ortho = imread(ortho_path);

        % 对SAR图像预处理，并保存
        [h, w] = size(ortho);
        mean_ortho = 1000 * sum(ortho(:)) / (h * w - size(find(ortho == 0), 1));
        ortho(ortho <= 0) = 0;
        ortho(ortho >= mean_ortho) = 255;
        ortho = ortho / mean_ortho * 255;
        ortho(ortho >= 1) = 1;
        ortho = ortho * 255;

        % ortho(ortho == 0) = 1;
        % ortho = log10(ortho);
        % mean_ortho = 100 * mean(ortho(:));
        % ortho(ortho >= mean_ortho) = 255;
        % ortho = ortho./ mean_ortho * 255;

        % hyper = stretchlim(ortho);
        % ortho2 = imadjust(ortho, hyper);
        filename3 = strcat(root_path, name, '/', name, '_光学/', name, '_registered.tif');
        imwrite(uint8(ortho), filename3);
        clear ortho

        % 对卫图进行配准
        ortho_info = imfinfo(ortho_path);
        aerial = imread(aerial_path);
        ortho_ref = imref2d([ortho_info.Height, ortho_info.Width]);
        clear ortho_info
        load(strcat(root_path, name, '/', 'points_', name, '.mat'));
        t_concord = fitgeotrans(movingPoints, fixedPoints,'projective');
        aerial_registered = imwarp(aerial, t_concord, 'OutputView', ortho_ref);
        clear aerial
        filename = strcat(root_path, name, '/', name, '_光学/', name, '_卫图_registered.tif');
        bits = ortho_info.Height * ortho_info.Width * 3;
        if bits >= (2^32 - 1)
            imwrite(uint8(aerial_registered(:, :, 1)), filename);
            save(strcat(root_path, name, '/', name, '_光学/', name, '_卫图_registered.mat'), 'aerial_registered');
        else
            imwrite(uint8(aerial_registered), filename);
        end
        clear aerial_registered


        % 对电子地图进行配准
        elecMap_path = strcat(root_path, name, '/', name, '_光学/', name, '_电子地图.tif');            %电子地图路径
        elecMap = imread(elecMap_path);
        elecMap_registered = imwarp(elecMap, t_concord, 'OutputView', ortho_ref);
        clear elecMap
        filename2 = strcat(root_path, name, '/', name, '_光学/', name, '_电子地图_registered.tif');
        if bits >= (2^32 - 1)
            save(strcat(root_path, name, '/', name, '_光学/', name, '_电子地图_registered.mat'), 'elecMap_registered');
        else
            imwrite(uint8(elecMap_registered), filename2);
        end
        clear elecMap_registered
    end
end
