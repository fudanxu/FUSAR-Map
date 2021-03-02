function [filt_out]=votefilt2Pro(img, S)
    %the size of filter kernel is 2*S+1
    % img : 输入的聚类好的地图
    % S : 滑动窗口大小
    % num : 地图的类别数量
    % filt_out : 聚类结果
    % [rows, cols] = size(img);
    % voted_img = zeros(rows, cols, 'int8');
    kernel = ones(2*S+1, 2*S+1, 'int8');
    voted_img = filter2(kernel, img, 'same');
    % [~, filt_out] = max(voted_img, [], 3);
    filt_out = (voted_img - 4) >= 0;
end