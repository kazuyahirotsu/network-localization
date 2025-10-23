% --- Configuration ---
clear; clc; close all;

% This script requires the MATLAB Mapping Toolbox.

% --- Parameters ---
% Select the instance you want to visualize
instanceToLoad = 1;

% Path to the data file.
% This assumes you are running the script from the 'mid-term/multilateration/' directory.
matlabDataPath = '../../matlab/data/64beacons_100instances/';
fileName = sprintf('data_instance_%d.mat', instanceToLoad);
fullPath = fullfile(matlabDataPath, fileName);

% Node configuration
numAnchors = 16;
% --- End Parameters ---


% --- Main Script ---
fprintf('--- Displaying Node Locations on Satellite Map ---\n');

% 1. Load the data file
try
    fprintf('Loading data from: %s\n', fullPath);
    data = load(fullPath);
    nodeLatitudes = data.nodeLatitudes;
    nodeLongitudes = data.nodeLongitudes;
catch ME
    fprintf(2, 'Error loading data file: %s\n', ME.message);
    fprintf(2, 'Please ensure the path is correct and the file exists.\n');
    return;
end

% 2. Separate anchor and unknown node coordinates
anchorLats = nodeLatitudes(1:numAnchors);
anchorLons = nodeLongitudes(1:numAnchors);
unknownLats = nodeLatitudes(numAnchors+1:end);
unknownLons = nodeLongitudes(numAnchors+1:end);

% 3. Create the geographic plot
try
    fprintf('Generating satellite map...\n');
    figure('Position', [100, 100, 900, 800]);
    
    % Create a geographic axes
    gx = geoaxes;
    
    % Plot unknown nodes
    geoplot(gx, unknownLats, unknownLons, 'bo', 'MarkerSize', 8, 'DisplayName', 'Unknown Nodes');
    hold on;
    
    % Plot anchor nodes
    geoplot(gx, anchorLats, anchorLons, 'g^', 'MarkerSize', 10, 'MarkerFaceColor', 'g', 'DisplayName', 'Anchor Nodes');
    
    % Set the background to the satellite basemap
    geobasemap(gx, 'satellite');
    
    % Add title and legend
    title(gx, sprintf('Satellite View of Node Locations for Instance %d', instanceToLoad), 'FontSize', 16);
    legend('show', 'FontSize', 12);
    
    fprintf('--- Script finished successfully. ---\n');

catch ME
    fprintf(2, 'An error occurred during plotting: %s\n', ME.message);
    fprintf(2, 'This is likely because the MATLAB Mapping Toolbox is not installed or licensed.\n');
end 