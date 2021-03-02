function [lons,lats] = gcr2ll(Ra,Rb)
%
%   conversion between geographicCellsReference and lat/lon axes
%

if isa(Ra, 'map.rasterref.GeographicCellsReference')
    if strcmp(Ra.RasterInterpretation,'cells') ...
       && strcmp(Ra.ColumnsStartFrom,'north') ...
       && strcmp(Ra.RowsStartFrom,'west') ...
       && strcmp(Ra.CoordinateSystemType, 'geographic') ...
       && strcmp(Ra.AngleUnit, 'degree')
        lons = Ra.LongitudeLimits(1)+(-0.5+(1:Ra.RasterSize(2)))*Ra.CellExtentInLongitude;
        lats = Ra.LatitudeLimits(2)-(-0.5+(1:Ra.RasterSize(1)))*Ra.CellExtentInLatitude;
    else
        error('some default elements does not match.');
    end
else
    slons = Ra;
    slats = Rb;
    
    dlon = median(diff(slons));
    dlat = median(diff(slats));
    
    if dlat<0 && dlon>0
        dlat = -dlat;
        R = georefcells(min(slats)-dlat/2+[0,length(slats)*dlat],...
            min(slons)-dlon/2+[0,length(slons)*dlon],...
            [length(slats),length(slons)],...
            'RowsStartFrom','west',...
            'ColumnsStartFrom','north');
        lons = R;
        lats = R;
    else
        error('incorrect lat/lon increasing direction.');
    end
end