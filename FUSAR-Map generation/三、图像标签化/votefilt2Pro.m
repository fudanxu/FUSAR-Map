function [filt_out]=votefilt2Pro(img, S)
    %the size of filter kernel is 2*S+1
    % img : ����ľ���õĵ�ͼ
    % S : �������ڴ�С
    % num : ��ͼ���������
    % filt_out : ������
    % [rows, cols] = size(img);
    % voted_img = zeros(rows, cols, 'int8');
    kernel = ones(2*S+1, 2*S+1, 'int8');
    voted_img = filter2(kernel, img, 'same');
    % [~, filt_out] = max(voted_img, [], 3);
    filt_out = (voted_img - 4) >= 0;
end