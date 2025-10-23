% --- Configuration ---
clear; clc; close all;

% This script requires the MATLAB Mapping Toolbox.

% Map origin from your data generation script
mapOriginLat = 40.466198;
mapOriginLon = 33.898610;
mapSizeMeters = 4000.0;

% Earth constants for coordinate conversion
earthRadius = 6378137.0;
degPerMeterLat = 1 / ((pi / 180) * earthRadius);
degPerMeterLon = 1 / ((pi / 180) * earthRadius * cosd(mapOriginLat));

% --- Main Script ---

fprintf('--- Fetching Elevation Data via MATLAB Mapping Toolbox from USGS ---\n');

% 1. Define the geographic limits (bounding box) for the area of interest
latlim(1) = mapOriginLat;
latlim(2) = mapOriginLat + (mapSizeMeters * degPerMeterLat);
lonlim(1) = mapOriginLon;
lonlim(2) = mapOriginLon + (mapSizeMeters * degPerMeterLon);

% 2. Access the USGS National Map Web Coverage Service (WCS) for 3DEP data
% This is a more direct and robust method than the previous WMS approach.
serverURL = 'https://elevation.nationalmap.gov/arcgis/services/3DEPElevation/ImageServer/WCSServer?';
try
    fprintf('Querying USGS 3DEP server...\n');
    
    % Use readgeoraster to download the data directly for the specified region
    [Z, R] = readgeoraster(serverURL, 'Latlim', latlim, 'Lonlim', lonlim, 'OutputType', 'double');
    
    % 3. Visualize the elevation map
    fprintf('Generating elevation heatmap...\n');
    figure('Position', [100, 100, 800, 700]);
    
    % Create a map axes
    worldmap(latlim, lonlim);
    
    % Use mapshow (or geoshow) to display the raster
    mapshow(Z, R, 'DisplayType', 'surface');
    
    % Add contours for better visualization of terrain shape
    contourm(Z, R, 'LevelStep', 20, 'LineColor', 'black');
    
    % Adjust colormap and colorbar
    colormap('terrain');
    c = colorbar;
    ylabel(c, 'Elevation (m)', 'FontSize', 14);
    
    title('Real-World Terrain Elevation Map (from MATLAB/USGS)', 'FontSize', 18);
    
    fprintf('--- Script finished successfully. ---\n');

catch ME
    fprintf(2, 'An error occurred: %s\n', ME.message);
    fprintf(2, 'This could be due to several reasons:\n');
    fprintf(2, '- No internet connection or a firewall is blocking the request.\n');
    fprintf(2, '- The USGS server might be temporarily unavailable or has changed its URL.\n');
    fprintf(2, '- The MATLAB Mapping Toolbox is not installed or licensed.\n');
end 