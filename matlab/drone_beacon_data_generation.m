% Drone-Beacon Signal Simulation for Localization Research
% This script simulates RSSI measurements between a drone and fixed beacons
% Purpose: Study how beacon location uncertainty affects drone localization accuracy

clear; clc;

%% ===================== Configuration =====================

% Set fixed random seed for reproducibility
rng(42, 'twister');

% Grid sampling parameters for drone positions
droneGridSize = 20;  % 20x20 = 400 drone positions
droneAltitude = 100; % meters

% Map origin (bottom-left corner)
mapOriginLat = 40.466198;  % Latitude
mapOriginLon = 33.898610;  % Longitude

% Map and tile sizes in meters
mapSizeMeters = 4000;    % 4 km
tileSizeMeters = 1000;   % 1 km
numTilesPerAxis = mapSizeMeters / tileSizeMeters;

% Earth's radius in meters
earthRadius = 6378137;  % Mean radius

% Conversion factors
metersPerDegreeLat = (pi / 180) * earthRadius;
metersPerDegreeLon = (pi / 180) * earthRadius * cosd(mapOriginLat);

% Degrees per meter
degPerMeterLat = 1 / metersPerDegreeLat;
degPerMeterLon = 1 / metersPerDegreeLon;

% LoRa transmission parameters
frequency = 915e6;        % Frequency in Hz
transmitterPower = 0.020; % 20 mW (13 dBm)
beaconHeight = 1.0;       % meters

%% ===================== Beacon Placement =====================
% Using same algorithm as sample: 16 anchors (one per tile) + 48 random nodes = 64 total

numAnchors = numTilesPerAxis^2;  % 16 anchors in grid tiles
numUnknowns = 48;                % 48 additional random beacons
numBeacons = numAnchors + numUnknowns;  % 64 total

fprintf('Generating %d beacon positions...\n', numBeacons);

% Initialize arrays for beacon positions
beaconLatitudes = zeros(numBeacons, 1);
beaconLongitudes = zeros(numBeacons, 1);
beaconXMeters = zeros(numBeacons, 1);
beaconYMeters = zeros(numBeacons, 1);

% Place anchors randomly within each tile (16 beacons)
beaconIndex = 1;
for i = 0:(numTilesPerAxis - 1)
    for j = 0:(numTilesPerAxis - 1)
        % Tile boundaries in meters
        xMin = i * tileSizeMeters;
        yMin = j * tileSizeMeters;

        % Random position within the tile (in meters from origin)
        xPosMeters = xMin + rand() * tileSizeMeters;
        yPosMeters = yMin + rand() * tileSizeMeters;

        % Store meters
        beaconXMeters(beaconIndex) = xPosMeters;
        beaconYMeters(beaconIndex) = yPosMeters;

        % Convert meters to degrees
        deltaLat = yPosMeters * degPerMeterLat;
        deltaLon = xPosMeters * degPerMeterLon;

        % Beacon position in degrees
        beaconLatitudes(beaconIndex) = mapOriginLat + deltaLat;
        beaconLongitudes(beaconIndex) = mapOriginLon + deltaLon;
        beaconIndex = beaconIndex + 1;
    end
end

% Place additional random beacons (48 beacons)
unknownXPosMeters = rand(numUnknowns, 1) * mapSizeMeters;
unknownYPosMeters = rand(numUnknowns, 1) * mapSizeMeters;

% Store meters
beaconXMeters(numAnchors+1:end) = unknownXPosMeters;
beaconYMeters(numAnchors+1:end) = unknownYPosMeters;

% Convert meters to degrees
deltaLatUnknown = unknownYPosMeters * degPerMeterLat;
deltaLonUnknown = unknownXPosMeters * degPerMeterLon;

beaconLatitudes(numAnchors+1:end) = mapOriginLat + deltaLatUnknown;
beaconLongitudes(numAnchors+1:end) = mapOriginLon + deltaLonUnknown;

fprintf('  - %d tile-based beacons placed\n', numAnchors);
fprintf('  - %d random beacons placed\n', numUnknowns);

%% ===================== Drone Grid Positions =====================

fprintf('Generating %dx%d = %d drone positions at %.0fm altitude...\n', ...
    droneGridSize, droneGridSize, droneGridSize^2, droneAltitude);

numDronePositions = droneGridSize^2;

% Create grid of drone positions (avoiding exact edges)
margin = mapSizeMeters * 0.05;  % 5% margin from edges
gridSpacing = (mapSizeMeters - 2*margin) / (droneGridSize - 1);

[droneGridX, droneGridY] = meshgrid(...
    margin : gridSpacing : (mapSizeMeters - margin), ...
    margin : gridSpacing : (mapSizeMeters - margin));

droneXMeters = droneGridX(:);
droneYMeters = droneGridY(:);

% Convert to lat/lon
droneLatitudes = mapOriginLat + droneYMeters * degPerMeterLat;
droneLongitudes = mapOriginLon + droneXMeters * degPerMeterLon;

%% ===================== Pre-compute True Distances (Vectorized) =====================

fprintf('Pre-computing true 3D distances (vectorized)...\n');

% Vectorized distance calculation - much faster than nested loops!
% Reshape for broadcasting: drone positions (Nx1) vs beacon positions (1xM)
dx_matrix = droneXMeters - beaconXMeters';      % NxM matrix
dy_matrix = droneYMeters - beaconYMeters';      % NxM matrix
dz = droneAltitude - beaconHeight;              % scalar

true_distances = sqrt(dx_matrix.^2 + dy_matrix.^2 + dz^2);
fprintf('  ✓ Distance matrix computed: %dx%d\n', size(true_distances, 1), size(true_distances, 2));

%% ===================== Create Site Objects =====================

fprintf('Creating transmitter/receiver site objects...\n');

% Propagation model
prop_model = propagationModel("longley-rice");

% Create beacon transmitter sites as an ARRAY (not cell) for vectorized sigstrength
beaconTxSites = txsite('Latitude', beaconLatitudes, ...
                        'Longitude', beaconLongitudes, ...
                        'TransmitterFrequency', frequency, ...
                        'TransmitterPower', transmitterPower, ...
                        'AntennaHeight', beaconHeight);

fprintf('  ✓ Created %d beacon transmitter sites\n', numBeacons);

%% ===================== Signal Simulation (Optimized) =====================

fprintf('Simulating RSSI measurements for %d drone positions x %d beacons...\n', ...
    numDronePositions, numBeacons);
fprintf('Total measurements to simulate: %d\n', numDronePositions * numBeacons);
fprintf('Using vectorized sigstrength (1 call per drone position)...\n');

% Initialize output matrix
rssi_matrix = zeros(numDronePositions, numBeacons);

tic;

% Progress tracking settings
printInterval = max(1, floor(numDronePositions / 20));  % Print ~20 updates

fprintf('\n');
fprintf('  [                                                  ] 0%%\n');

% Check if Parallel Computing Toolbox is available
hasParallel = license('test', 'Distrib_Computing_Toolbox');

if hasParallel
    % Use parallel processing for even faster execution
    fprintf('  Using parallel processing (parfor)...\n');
    
    % Pre-allocate for parfor
    rssi_temp = zeros(numDronePositions, numBeacons);
    
    % Create a parallel pool if not already running
    pool = gcp('nocreate');
    if isempty(pool)
        parpool('local');
    end
    
    parfor d = 1:numDronePositions
        % Create drone receiver site at current position
        droneRxSite = rxsite('Latitude', droneLatitudes(d), ...
                              'Longitude', droneLongitudes(d), ...
                              'AntennaHeight', droneAltitude, ...
                              'ReceiverSensitivity', -135);
        
        % Vectorized sigstrength: pass ALL beacon sites at once!
        % Returns array of RSSI values for all beacons
        rssi_temp(d, :) = sigstrength(droneRxSite, beaconTxSites, prop_model);
    end
    
    rssi_matrix = rssi_temp;
    
else
    % Sequential processing with vectorized sigstrength
    fprintf('  Using sequential processing (no parallel toolbox)...\n');
    
    for d = 1:numDronePositions
        % Create drone receiver site at current position
        droneRxSite = rxsite('Latitude', droneLatitudes(d), ...
                              'Longitude', droneLongitudes(d), ...
                              'AntennaHeight', droneAltitude, ...
                              'ReceiverSensitivity', -135);
        
        % Vectorized sigstrength: pass ALL beacon sites at once!
        % Returns array of RSSI values for all beacons (64x faster than loop!)
        rssi_matrix(d, :) = sigstrength(droneRxSite, beaconTxSites, prop_model);
        
        % Progress update with ETA
        if mod(d, printInterval) == 0 || d == numDronePositions
            elapsed = toc;
            pct = 100 * d / numDronePositions;
            
            if d > 0
                timePerPos = elapsed / d;
                remaining = timePerPos * (numDronePositions - d);
                
                % Create progress bar
                barWidth = 50;
                filledWidth = round(barWidth * d / numDronePositions);
                bar = [repmat('=', 1, filledWidth), repmat(' ', 1, barWidth - filledWidth)];
                
                % Format time
                if remaining < 60
                    etaStr = sprintf('%.0fs', remaining);
                elseif remaining < 3600
                    etaStr = sprintf('%.1fmin', remaining/60);
                else
                    etaStr = sprintf('%.1fh', remaining/3600);
                end
                
                % Print progress
                fprintf('\r  [%s] %.1f%% | %d/%d | Elapsed: %.1fs | ETA: %s     ', ...
                    bar, pct, d, numDronePositions, elapsed, etaStr);
            end
        end
    end
end

elapsed = toc;
fprintf('\n\n');
fprintf('✓ Simulation complete!\n');
fprintf('  Total time: %.2f seconds (%.3f sec/position)\n', elapsed, elapsed/numDronePositions);
fprintf('  Average: %.1f measurements/second\n', (numDronePositions * numBeacons) / elapsed);

%% ===================== Save Data =====================

outputFile = 'data/drone_beacon_simulation_data.mat';
fprintf('\nSaving data to %s...\n', outputFile);

% Simulation metadata
metadata = struct();
metadata.description = 'Drone-Beacon RSSI simulation for localization research';
metadata.randomSeed = 42;
metadata.mapOriginLat = mapOriginLat;
metadata.mapOriginLon = mapOriginLon;
metadata.mapSizeMeters = mapSizeMeters;
metadata.droneAltitude = droneAltitude;
metadata.beaconHeight = beaconHeight;
metadata.frequency = frequency;
metadata.transmitterPower = transmitterPower;
metadata.propagationModel = 'longley-rice';
metadata.numBeacons = numBeacons;
metadata.numDronePositions = numDronePositions;
metadata.droneGridSize = droneGridSize;
metadata.degPerMeterLat = degPerMeterLat;
metadata.degPerMeterLon = degPerMeterLon;

save(outputFile, ...
    'beaconLatitudes', 'beaconLongitudes', ...
    'beaconXMeters', 'beaconYMeters', ...
    'droneLatitudes', 'droneLongitudes', ...
    'droneXMeters', 'droneYMeters', ...
    'droneAltitude', 'beaconHeight', ...
    'rssi_matrix', 'true_distances', ...
    'metadata', '-v7.3');

fprintf('Data saved successfully!\n');

%% ===================== Visualization =====================

fprintf('Generating visualization...\n');

figure('Position', [100, 100, 1200, 500]);

% Plot 1: Beacon and drone positions
subplot(1, 2, 1);
scatter(beaconXMeters(1:numAnchors)/1000, beaconYMeters(1:numAnchors)/1000, ...
    80, 'g^', 'filled', 'DisplayName', 'Tile Beacons');
hold on;
scatter(beaconXMeters(numAnchors+1:end)/1000, beaconYMeters(numAnchors+1:end)/1000, ...
    60, 'b^', 'filled', 'DisplayName', 'Random Beacons');
scatter(droneXMeters/1000, droneYMeters/1000, ...
    20, 'r.', 'DisplayName', 'Drone Positions');
xlabel('X (km)');
ylabel('Y (km)');
title(sprintf('Beacon and Drone Positions\n(%d beacons, %d drone positions)', numBeacons, numDronePositions));
legend('Location', 'best');
axis equal;
xlim([0, mapSizeMeters/1000]);
ylim([0, mapSizeMeters/1000]);
grid on;
hold off;

% Plot 2: RSSI heatmap for center drone position
subplot(1, 2, 2);
centerDroneIdx = round(numDronePositions / 2);
rssi_values_center = rssi_matrix(centerDroneIdx, :);

scatter(beaconXMeters/1000, beaconYMeters/1000, ...
    100, rssi_values_center, 'filled');
hold on;
plot(droneXMeters(centerDroneIdx)/1000, droneYMeters(centerDroneIdx)/1000, ...
    'rp', 'MarkerSize', 15, 'MarkerFaceColor', 'r');
colormap(jet);
colorbar;
xlabel('X (km)');
ylabel('Y (km)');
title(sprintf('RSSI from Beacons to Center Drone\n(Drone at [%.2f, %.2f] km, %.0fm altitude)', ...
    droneXMeters(centerDroneIdx)/1000, droneYMeters(centerDroneIdx)/1000, droneAltitude));
axis equal;
xlim([0, mapSizeMeters/1000]);
ylim([0, mapSizeMeters/1000]);
grid on;
hold off;

saveas(gcf, 'drone_beacon_visualization.png');
fprintf('Visualization saved to drone_beacon_visualization.png\n');

%% ===================== Summary Statistics =====================

fprintf('\n========== Summary ==========\n');
fprintf('Map size: %.0f x %.0f meters\n', mapSizeMeters, mapSizeMeters);
fprintf('Number of beacons: %d (16 tile-based + 48 random)\n', numBeacons);
fprintf('Number of drone positions: %d (%dx%d grid)\n', numDronePositions, droneGridSize, droneGridSize);
fprintf('Drone altitude: %.0f m\n', droneAltitude);
fprintf('Beacon height: %.1f m\n', beaconHeight);
fprintf('RSSI range: [%.2f, %.2f] dBm\n', min(rssi_matrix(:)), max(rssi_matrix(:)));
fprintf('Distance range: [%.2f, %.2f] m\n', min(true_distances(:)), max(true_distances(:)));
fprintf('Output file: %s\n', outputFile);
fprintf('==============================\n');

