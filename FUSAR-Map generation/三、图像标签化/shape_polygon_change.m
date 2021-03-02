function polygon = shape_polygon_change(shapeFilePath, orthoImagePath, boundingImgPath, lon_change, lat_change)
% make shape file to polygon
% get polyshape
% take a offset
lon_offset = lon_change; % 0.005096;
lat_offset = lat_change; % -0.002532;
box_offset = [lon_offset, lat_offset; lon_offset, lat_offset];

ortho_img_box = shaperead(boundingImgPath);
ortho_boundingbox = ortho_img_box.BoundingBox;
ortho_info = imfinfo(orthoImagePath);
% lon 
lon_step = (ortho_boundingbox(2, 1) - ortho_boundingbox(1, 1)) / (ortho_info.Width - 1); % -2
% lat
lat_step = (ortho_boundingbox(2, 2) - ortho_boundingbox(1, 2)) / (ortho_info.Height - 1); % -2

% 确定在该区域中的shp文件数量
shapeFiles = shaperead(shapeFilePath);
numShapeFiles = size(shapeFiles, 1);
inBox_index = [];
parfor i = 1:numShapeFiles
    tmp = shapeFiles(i).BoundingBox + box_offset;
    if sum(tmp(1,:) >= ortho_boundingbox(1,:)) == 2 && sum(tmp(2,:) <= ortho_boundingbox(2,:)) == 2
        inBox_index = [inBox_index i];
    end
end

numInBox = size(inBox_index, 2);
polygon = zeros(ortho_info.Height, ortho_info.Width);

if numInBox ~= 0
    % 不使用tform校正
    for i = 1:numInBox
        % Floor = shapeFiles(inBox_index(i)).Floor;
        tmp = shapeFiles(inBox_index(i)).BoundingBox + box_offset;
        % outputShapeFiles(i) = shapeFiles(inBox_index(i));

%         % 真实边框外的最小边界框
%         tmp_pixelCoord = [(tmp(1,1) - ortho_img_boundingbox(1,1)) / lon_step, (tmp(1,2) - ortho_img_boundingbox(1,2)) / lat_step;
%                 (tmp(2,1) - ortho_img_boundingbox(1,1)) / lon_step, (tmp(2,2) - ortho_img_boundingbox(1,2)) / lat_step];
        tmp_pixelCoord = [(tmp(1, 1) - ortho_boundingbox(1, 1)) / lon_step, (ortho_boundingbox(2, 2) - tmp(2, 2)) / lat_step;
            (tmp(2,1) - ortho_boundingbox(1,1)) / lon_step, (ortho_boundingbox(2, 2) - tmp(1, 2)) / lat_step];

        % 真实边框
        % shapeFiles(i).X最后一个元素为NAN
        % shape文件中X代表经度，Y代表纬度
%         realBox_X = ((shapeFiles(inBox_index(i)).X + X_change) - ortho_img_boundingbox(1,1))./ lon_step;
%         realBox_Y = ((shapeFiles(inBox_index(i)).Y + Y_change) - ortho_img_boundingbox(1,2))./ lat_step;
        realBox_X = ((shapeFiles(inBox_index(i)).X + lon_offset) - ortho_boundingbox(1, 1)) ./ lon_step; % col
        realBox_Y = (ortho_boundingbox(2, 2) - (shapeFiles(inBox_index(i)).Y + lat_offset)) ./ lat_step; % row
        
        pixelCoord = [floor(tmp_pixelCoord(1, 1)), floor(tmp_pixelCoord(1, 2));
            ceil(tmp_pixelCoord(2, 1)), ceil(tmp_pixelCoord(2, 2))]; % 放大范围
        
        tmp_pixelCoord_X = pixelCoord(1,1):pixelCoord(2,1);
        tmp_pixelCoord_Y = pixelCoord(1,2):pixelCoord(2,2);
        real_X = round(realBox_X);
        real_Y = round(realBox_Y);
        
        tmp_pixelCoord_ySpace = reshape(repmat( tmp_pixelCoord_Y, [size( tmp_pixelCoord_X, 2), 1]),...
            [1, length( tmp_pixelCoord_X)*length( tmp_pixelCoord_Y)]); % row
        tmp_pixelCoord_xSpace = reshape(repmat(tmp_pixelCoord_X', [size(tmp_pixelCoord_Y, 2), 1]),...
            [1, length( tmp_pixelCoord_X)*length( tmp_pixelCoord_Y)]); % col
        tmp_pixelCoord_logical = inpolygon(tmp_pixelCoord_ySpace, tmp_pixelCoord_xSpace, real_Y, real_X);
        matrix_get = reshape( tmp_pixelCoord_logical, [length(tmp_pixelCoord_X), length(tmp_pixelCoord_Y)])';
        [rows, cols] = find(matrix_get);
        rows = rows + pixelCoord(1, 2); % - 1;
        cols = cols + pixelCoord(1, 1); % - 1;
        index = sub2ind(size(polygon), rows, cols);
        polygon(index) = 1;

%         outputShapeFiles(i).BoundingBox = tmp_pixelCoord;
%         outputShapeFiles(i).X = realBox_X;
%         outputShapeFiles(i).Y = realBox_Y;
    end
else
    polygon = [];
end